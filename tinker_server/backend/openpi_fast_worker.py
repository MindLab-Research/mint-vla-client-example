from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .openpi_fast_runtime import OPENPI_FAST_WORKER_PROTOCOL_VERSION


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenPIFastRuntimeInitOverrides:
    weights_path: str | None = None
    random_init: bool = False

    @classmethod
    def from_env(cls) -> "OpenPIFastRuntimeInitOverrides":
        weights_path = (os.environ.get("MINT_OPENPI_FAST_WEIGHTS_PATH") or "").strip() or None
        random_init = (os.environ.get("MINT_OPENPI_FAST_RANDOM_INIT") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if weights_path and random_init:
            raise ValueError(
                "MINT_OPENPI_FAST_WEIGHTS_PATH and MINT_OPENPI_FAST_RANDOM_INIT are mutually exclusive"
            )
        if weights_path is not None:
            weights_path = str(Path(weights_path).resolve())
        return cls(weights_path=weights_path, random_init=random_init)


def _reply(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _float_scalar(value: Any) -> float:
    return float(np.asarray(value).item())


def _int_scalar(value: Any) -> int:
    return int(np.asarray(value).item())


def _decode_image(encoded: dict[str, Any]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for OpenPI FAST image decoding") from exc

    raw = base64.b64decode(encoded["data"])
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _compute_importance_sampling_stats(
    *,
    current_logprobs: np.ndarray,
    old_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
) -> dict[str, float | int]:
    current = np.asarray(current_logprobs, dtype=np.float32).reshape(-1)
    old = np.asarray(old_logprobs, dtype=np.float32).reshape(-1)
    adv = np.asarray(advantages, dtype=np.float32).reshape(-1)
    mask = np.asarray(loss_mask, dtype=np.bool_).reshape(-1)

    if not (current.shape == old.shape == adv.shape == mask.shape):
        raise ValueError("importance_sampling inputs must share the same length")

    token_count = int(mask.sum())
    if token_count == 0:
        raise ValueError("importance_sampling requires at least one masked token")

    mask_f = mask.astype(np.float32)
    log_ratio = np.clip(current - old, a_min=-20.0, a_max=20.0)
    ratio = np.exp(log_ratio)
    loss = -float(np.sum(ratio * adv * mask_f))
    ratio_mean = float(np.sum(ratio * mask_f) / token_count)
    return {
        "loss": loss,
        "ratio_mean": ratio_mean,
        "token_count": token_count,
    }


def _compute_ppo_stats(
    *,
    current_logprobs: np.ndarray,
    old_logprobs: np.ndarray,
    advantages: np.ndarray,
    loss_mask: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> dict[str, float | int]:
    current = np.asarray(current_logprobs, dtype=np.float32).reshape(-1)
    old = np.asarray(old_logprobs, dtype=np.float32).reshape(-1)
    adv = np.asarray(advantages, dtype=np.float32).reshape(-1)
    mask = np.asarray(loss_mask, dtype=np.bool_).reshape(-1)

    if not (current.shape == old.shape == adv.shape == mask.shape):
        raise ValueError("ppo inputs must share the same length")

    token_count = int(mask.sum())
    if token_count == 0:
        raise ValueError("ppo requires at least one masked token")

    mask_f = mask.astype(np.float32)
    log_ratio = np.clip(current - old, a_min=-20.0, a_max=20.0)
    ratio = np.exp(log_ratio)
    clipped_ratio = np.clip(ratio, a_min=clip_low, a_max=clip_high)
    unclipped = -ratio * adv
    clipped = -clipped_ratio * adv
    loss = float(np.sum(np.maximum(unclipped, clipped) * mask_f))
    ratio_mean = float(np.sum(ratio * mask_f) / token_count)
    clipfrac_mean = float(np.sum(((ratio < clip_low) | (ratio > clip_high)).astype(np.float32) * mask_f) / token_count)
    return {
        "loss": loss,
        "ratio_mean": ratio_mean,
        "clipfrac_mean": clipfrac_mean,
        "token_count": token_count,
    }


class _StaticDataLoader:
    def __init__(self, data_config: Any) -> None:
        self._data_config = data_config

    def data_config(self) -> Any:
        return self._data_config


class OpenPIFastWorkerSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import optax

        import openpi.models.model as openpi_model
        import openpi.models.pi0_fast as openpi_pi0_fast
        import openpi.shared.array_typing as array_typing
        import openpi.shared.nnx_utils as nnx_utils
        import openpi.training.checkpoints as checkpoints
        import openpi.training.config as config_mod
        import openpi.training.optimizer as optimizer_mod
        import openpi.training.sharding as sharding_mod
        import openpi.training.utils as training_utils
        import openpi.training.weight_loaders as weight_loaders
        import orbax.checkpoint as ocp
        from flax import traverse_util

        self._nnx = nnx
        self._jax = jax
        self._jnp = jnp
        self._optax = optax
        self._openpi_model = openpi_model
        self._openpi_pi0_fast = openpi_pi0_fast
        self._array_typing = array_typing
        self._nnx_utils = nnx_utils
        self._checkpoints = checkpoints
        self._config_mod = config_mod
        self._optimizer_mod = optimizer_mod
        self._sharding_mod = sharding_mod
        self._training_utils = training_utils
        self._weight_loaders = weight_loaders
        self._ocp = ocp
        self._traverse_util = traverse_util

        self._payload = payload
        self._model_id = str(payload["model_id"])
        self._action_dim = int(payload["action_dim"])
        self._action_horizon = int(payload["action_horizon"])
        self._config_name = str(payload["config_name"])
        self._learning_rate = float(payload.get("learning_rate") or 0.0)

        config = config_mod.get_config(self._config_name)
        checkpoint_base_dir = os.environ.get(
            "MINT_OPENPI_FAST_CHECKPOINT_BASE_DIR",
            str(Path("./checkpoints").resolve()),
        )
        assets_base_dir = os.environ.get(
            "MINT_OPENPI_FAST_ASSETS_BASE_DIR",
            str(Path("./assets").resolve()),
        )
        self._config = dataclasses.replace(
            config,
            exp_name=self._model_id,
            project_name="mint-openpi",
            checkpoint_base_dir=checkpoint_base_dir,
            assets_base_dir=assets_base_dir,
            wandb_enabled=False,
            overwrite=False,
            resume=False,
        )
        overrides = OpenPIFastRuntimeInitOverrides.from_env()
        if overrides.weights_path is not None:
            logger.info("OpenPI FAST worker using explicit weights path: %s", overrides.weights_path)
            self._config = dataclasses.replace(
                self._config,
                weight_loader=weight_loaders.CheckpointWeightLoader(overrides.weights_path),
            )
        elif overrides.random_init:
            logger.info("OpenPI FAST worker using explicit random-init mode")
            self._config = dataclasses.replace(
                self._config,
                weight_loader=weight_loaders.NoOpWeightLoader(),
            )
        self._data_loader = _StaticDataLoader(
            self._config.data.create(self._config.assets_dirs, self._config.model)
        )

        self._mesh = self._sharding_mod.make_mesh(self._config.fsdp_devices)
        self._state = self._init_train_state()
        self._rng = jax.random.key(self._config.seed)
        self._pending_grads = None
        self._closed = False

    def _load_weights_and_validate(self, loader: Any, params_shape: Any) -> Any:
        loaded_params = loader.load(params_shape)
        self._array_typing.check_pytree_equality(
            expected=params_shape,
            got=loaded_params,
            check_shapes=True,
            check_dtypes=True,
        )
        return self._traverse_util.unflatten_dict(
            {
                key: value
                for key, value in self._traverse_util.flatten_dict(loaded_params).items()
                if not isinstance(value, self._jax.ShapeDtypeStruct)
            }
        )

    def _init_train_state(self) -> Any:
        config = self._config
        jax = self._jax
        nnx = self._nnx

        tx = self._optimizer_mod.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

        def init(rng: Any, partial_params: Any | None = None) -> Any:
            rng, model_rng = jax.random.split(rng)
            model = config.model.create(model_rng)
            if partial_params is not None:
                graphdef, state = nnx.split(model)
                state.replace_by_pure_dict(partial_params)
                model = nnx.merge(graphdef, state)

            params = nnx.state(model)
            params = self._nnx_utils.state_map(
                params,
                config.freeze_filter,
                lambda p: p.replace(p.value.astype(self._jnp.bfloat16)),
            )
            return self._training_utils.TrainState(
                step=0,
                params=params,
                model_def=nnx.graphdef(model),
                tx=tx,
                opt_state=tx.init(params.filter(config.trainable_filter)),
                ema_decay=config.ema_decay,
                ema_params=None if config.ema_decay is None else params,
            )

        init_rng = jax.random.key(config.seed)
        train_state_shape = jax.eval_shape(init, init_rng)
        state_sharding = self._sharding_mod.fsdp_sharding(train_state_shape, self._mesh, log=True)
        partial_params = self._load_weights_and_validate(
            config.weight_loader,
            train_state_shape.params.to_pure_dict(),
        )
        replicated_sharding = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
        train_state = jax.jit(
            init,
            donate_argnums=(1,),
            in_shardings=replicated_sharding,
            out_shardings=state_sharding,
        )(init_rng, partial_params)
        return jax.block_until_ready(train_state)

    def _optimizer_tx(self, *, learning_rate: float, beta1: float, beta2: float, eps: float):
        optimizer_cfg = self._config.optimizer
        try:
            optimizer_cfg = dataclasses.replace(optimizer_cfg, b1=beta1, b2=beta2, eps=eps)
        except TypeError:
            pass
        lr = self._optax.constant_schedule(learning_rate)
        return optimizer_cfg.create(lr, weight_decay_mask=None)

    def _observation_from_payload(self, item: dict[str, Any]):
        images = {
            key: np.expand_dims(_decode_image(value), axis=0)
            for key, value in dict(item["image_bytes"]).items()
        }
        image_mask = {
            key: np.asarray([bool(value)], dtype=np.bool_)
            for key, value in dict(item["image_mask"]).items()
        }

        observation_dict = {
            "image": images,
            "image_mask": image_mask,
            "state": np.asarray([item["state"]], dtype=np.float32),
            "tokenized_prompt": np.asarray([item["tokenized_prompt"]], dtype=np.int32),
            "tokenized_prompt_mask": np.asarray([item["tokenized_prompt_mask"]], dtype=np.bool_),
            "token_ar_mask": np.asarray([item["token_ar_mask"]], dtype=np.int32),
            "token_loss_mask": np.asarray([item["token_loss_mask"]], dtype=np.bool_),
        }
        observation = self._openpi_model.Observation.from_dict(observation_dict)
        actions = np.zeros((1, self._action_horizon, self._action_dim), dtype=np.float32)
        return observation, actions

    def _grad_and_param_norm(self, model: Any, grads: Any) -> tuple[float, float]:
        nnx = self._nnx

        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(self._nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        grad_norm = _float_scalar(self._optax.global_norm(grads))
        param_norm = _float_scalar(self._optax.global_norm(kernel_params))
        return grad_norm, param_norm

    def _compute_grads(self, observation: Any, actions: Any) -> tuple[Any, float, float, float]:
        nnx = self._nnx
        jax = self._jax

        model = nnx.merge(self._state.model_def, self._state.params)
        model.train()
        self._rng, step_rng = jax.random.split(self._rng)

        def loss_fn(model_obj: Any, rng: Any, obs: Any, act: Any):
            chunked_loss = model_obj.compute_loss(rng, obs, act, train=True)
            return self._jnp.mean(chunked_loss)

        diff_state = nnx.DiffState(0, self._config.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, step_rng, observation, actions)
        grad_norm, param_norm = self._grad_and_param_norm(model, grads)
        loss_value = _float_scalar(jax.device_get(loss))
        return grads, loss_value, grad_norm, param_norm

    def _compute_target_logprobs(self, model_obj: Any, rng: Any, observation: Any) -> Any:
        observation = self._openpi_model.preprocess_observation(
            rng,
            observation,
            train=True,
            image_keys=list(observation.images.keys()),
        )
        input_token_embeddings, input_mask, ar_mask = model_obj.embed_inputs(observation)
        attn_mask = self._openpi_pi0_fast.make_attn_mask(input_mask, ar_mask)
        target_tokens = observation.tokenized_prompt[:, 1:]
        pre_logits, _, _ = model_obj.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )
        logits, _ = model_obj.PaliGemma.llm(
            pre_logits=pre_logits[:, -target_tokens.shape[1] :],
        )
        logp = self._jax.nn.log_softmax(logits, axis=-1)
        return self._jnp.take_along_axis(logp, target_tokens[..., None], axis=-1).squeeze(-1)

    def _importance_sampling_inputs(
        self,
        item: dict[str, Any],
        *,
        target_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        loss_mask = np.asarray(item["token_loss_mask"], dtype=np.bool_).reshape(-1)[1:]
        old_logprobs = np.asarray(item["old_logprobs"], dtype=np.float32).reshape(-1)
        advantages = np.asarray(item["advantages"], dtype=np.float32).reshape(-1)

        if loss_mask.shape[0] != target_len:
            raise ValueError("importance_sampling token_loss_mask must align with target tokens")
        if old_logprobs.shape != advantages.shape:
            raise ValueError("importance_sampling old_logprobs and advantages must share one length")
        if int(loss_mask.sum()) != old_logprobs.shape[0]:
            raise ValueError("importance_sampling suffix inputs must match masked token count")

        padded_old_logprobs = np.zeros(target_len, dtype=np.float32)
        padded_advantages = np.zeros(target_len, dtype=np.float32)
        padded_old_logprobs[loss_mask] = old_logprobs
        padded_advantages[loss_mask] = advantages
        return padded_old_logprobs, padded_advantages, loss_mask

    def _compute_importance_sampling_grads(
        self,
        observation: Any,
        actions: Any,
        item: dict[str, Any],
    ) -> tuple[Any, float, float, float, float, float, list[float]]:
        nnx = self._nnx
        jax = self._jax

        target_len = len(list(item["tokenized_prompt"])) - 1
        old_logprobs, advantages, loss_mask = self._importance_sampling_inputs(
            item,
            target_len=target_len,
        )

        old_logprobs_t = self._jnp.asarray(old_logprobs[None, :], dtype=self._jnp.float32)
        advantages_t = self._jnp.asarray(advantages[None, :], dtype=self._jnp.float32)
        loss_mask_t = self._jnp.asarray(loss_mask.astype(np.float32)[None, :], dtype=self._jnp.float32)

        model = nnx.merge(self._state.model_def, self._state.params)
        model.train()
        self._rng, step_rng = jax.random.split(self._rng)

        def loss_fn(model_obj: Any, rng: Any, obs: Any, act: Any):
            del act
            current_logprobs = self._compute_target_logprobs(model_obj, rng, obs)
            log_ratio = self._jnp.clip(current_logprobs - old_logprobs_t, a_min=-20.0, a_max=20.0)
            ratio = self._jnp.exp(log_ratio)
            loss = -self._jnp.sum(ratio * advantages_t * loss_mask_t)
            return loss, current_logprobs

        diff_state = nnx.DiffState(0, self._config.trainable_filter)
        (loss, current_logprobs), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model, step_rng, observation, actions)
        grad_norm, param_norm = self._grad_and_param_norm(model, grads)

        current_logprobs_np = np.asarray(jax.device_get(current_logprobs), dtype=np.float32).reshape(-1)
        stats = _compute_importance_sampling_stats(
            current_logprobs=current_logprobs_np,
            old_logprobs=old_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
        )
        _ = loss
        return (
            grads,
            float(stats["loss"]),
            grad_norm,
            param_norm,
            float(stats["ratio_mean"]),
            float(stats["token_count"]),
            current_logprobs_np[loss_mask].tolist(),
        )

    def _compute_ppo_grads(
        self,
        observation: Any,
        actions: Any,
        item: dict[str, Any],
        loss_fn_config: dict[str, Any] | None,
    ) -> tuple[Any, float, float, float, float, float, float, list[float]]:
        nnx = self._nnx
        jax = self._jax

        target_len = len(list(item["tokenized_prompt"])) - 1
        old_logprobs, advantages, loss_mask = self._importance_sampling_inputs(
            item,
            target_len=target_len,
        )

        cfg = dict(loss_fn_config or {})
        epsilon = float(cfg.get("epsilon", 0.2))
        clip_low = float(cfg.get("clip_low", 1.0 - epsilon))
        clip_high = float(cfg.get("clip_high", 1.0 + epsilon))
        if clip_low > clip_high:
            raise ValueError("ppo clip_low must be <= clip_high")

        old_logprobs_t = self._jnp.asarray(old_logprobs[None, :], dtype=self._jnp.float32)
        advantages_t = self._jnp.asarray(advantages[None, :], dtype=self._jnp.float32)
        loss_mask_t = self._jnp.asarray(loss_mask.astype(np.float32)[None, :], dtype=self._jnp.float32)

        model = nnx.merge(self._state.model_def, self._state.params)
        model.train()
        self._rng, step_rng = jax.random.split(self._rng)

        def loss_fn(model_obj: Any, rng: Any, obs: Any, act: Any):
            del act
            current_logprobs = self._compute_target_logprobs(model_obj, rng, obs)
            log_ratio = self._jnp.clip(current_logprobs - old_logprobs_t, a_min=-20.0, a_max=20.0)
            ratio = self._jnp.exp(log_ratio)
            clipped_ratio = self._jnp.clip(ratio, a_min=clip_low, a_max=clip_high)
            unclipped = -ratio * advantages_t
            clipped = -clipped_ratio * advantages_t
            loss = self._jnp.sum(self._jnp.maximum(unclipped, clipped) * loss_mask_t)
            return loss, current_logprobs

        diff_state = nnx.DiffState(0, self._config.trainable_filter)
        (loss, current_logprobs), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model, step_rng, observation, actions)
        grad_norm, param_norm = self._grad_and_param_norm(model, grads)

        current_logprobs_np = np.asarray(jax.device_get(current_logprobs), dtype=np.float32).reshape(-1)
        stats = _compute_ppo_stats(
            current_logprobs=current_logprobs_np,
            old_logprobs=old_logprobs,
            advantages=advantages,
            loss_mask=loss_mask,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        _ = loss
        return (
            grads,
            float(stats["loss"]),
            grad_norm,
            param_norm,
            float(stats["ratio_mean"]),
            float(stats["clipfrac_mean"]),
            float(stats["token_count"]),
            current_logprobs_np[loss_mask].tolist(),
        )

    def create_session(self) -> dict[str, Any]:
        return {"backend": "openpi_fast", "config_name": self._config_name}

    def forward_backward(self, payload: dict[str, Any]) -> dict[str, Any]:
        loss_fn = str(payload.get("loss_fn") or "")
        if loss_fn not in {"cross_entropy", "importance_sampling", "ppo"}:
            raise ValueError(
                "OpenPI FAST ST-03 only supports cross_entropy, importance_sampling, "
                f"and ppo, got {loss_fn!r}"
            )

        batch = list(payload.get("batch") or [])
        if not batch:
            raise ValueError("OpenPI FAST forward_backward requires a non-empty batch")

        loss_fn_config = dict(payload.get("loss_fn_config") or {})
        total_loss = 0.0
        total_tokens = 0.0
        total_grad_norm = 0.0
        total_param_norm = 0.0
        total_ratio = 0.0
        total_clipfrac = 0.0
        num_rl_items = 0
        loss_fn_outputs: list[dict[str, Any]] = []
        pending_grads = self._pending_grads

        for item in batch:
            observation, actions = self._observation_from_payload(item)
            if loss_fn == "cross_entropy":
                grads, loss_value, grad_norm, param_norm = self._compute_grads(observation, actions)
                token_count = float(sum(bool(x) for x in item["token_loss_mask"]))
                loss_fn_outputs.append(
                    {
                        "loss": {
                            "data": [loss_value],
                            "shape": [1],
                            "dtype": "float32",
                        }
                    }
                )
            elif loss_fn == "importance_sampling":
                (
                    grads,
                    loss_value,
                    grad_norm,
                    param_norm,
                    ratio_mean,
                    token_count,
                    new_logprobs,
                ) = self._compute_importance_sampling_grads(observation, actions, item)
                total_ratio += ratio_mean
                num_rl_items += 1
                loss_fn_outputs.append(
                    {
                        "loss": {
                            "data": [loss_value],
                            "shape": [1],
                            "dtype": "float32",
                        },
                        "logprobs": {
                            "data": new_logprobs,
                            "shape": [len(new_logprobs)],
                            "dtype": "float32",
                        },
                    }
                )
            else:
                (
                    grads,
                    loss_value,
                    grad_norm,
                    param_norm,
                    ratio_mean,
                    clipfrac_mean,
                    token_count,
                    new_logprobs,
                ) = self._compute_ppo_grads(observation, actions, item, loss_fn_config)
                total_ratio += ratio_mean
                total_clipfrac += clipfrac_mean
                num_rl_items += 1
                loss_fn_outputs.append(
                    {
                        "loss": {
                            "data": [loss_value],
                            "shape": [1],
                            "dtype": "float32",
                        },
                        "logprobs": {
                            "data": new_logprobs,
                            "shape": [len(new_logprobs)],
                            "dtype": "float32",
                        },
                    }
                )
            pending_grads = (
                grads
                if pending_grads is None
                else self._jax.tree.map(lambda a, b: a + b, pending_grads, grads)
            )
            total_loss += loss_value
            total_tokens += token_count
            total_grad_norm += grad_norm
            total_param_norm += param_norm

        self._pending_grads = pending_grads

        denom = max(total_tokens, 1.0)
        batch_size = float(len(batch))
        metrics = {
            "loss:mean": total_loss / denom,
            "num_samples:sum": batch_size,
            "num_tokens:sum": total_tokens,
            "grad_norm:mean": total_grad_norm / batch_size,
            "param_norm:mean": total_param_norm / batch_size,
        }
        if num_rl_items > 0:
            metrics["ratio:mean"] = total_ratio / num_rl_items
        if loss_fn == "ppo" and num_rl_items > 0:
            metrics["clipfrac:mean"] = total_clipfrac / num_rl_items
        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
        }

    def optim_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._pending_grads is None:
            raise ValueError("OpenPI FAST optim_step requires a preceding forward_backward")

        learning_rate = float(payload["learning_rate"])
        beta1 = float(payload["beta1"])
        beta2 = float(payload["beta2"])
        eps = float(payload["eps"])

        tx = self._optimizer_tx(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
        params = self._state.params.filter(self._config.trainable_filter)
        updates, new_opt_state = tx.update(self._pending_grads, self._state.opt_state, params)
        new_params = self._optax.apply_updates(params, updates)

        model = self._nnx.merge(self._state.model_def, self._state.params)
        self._nnx.update(model, new_params)
        full_params = self._nnx.state(model)
        new_state = dataclasses.replace(
            self._state,
            step=self._state.step + 1,
            params=full_params,
            opt_state=new_opt_state,
            tx=tx,
        )
        if self._state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=self._jax.tree.map(
                    lambda old, new: self._state.ema_decay * old + (1 - self._state.ema_decay) * new,
                    self._state.ema_params,
                    full_params,
                ),
            )
        self._state = new_state
        self._pending_grads = None
        self._learning_rate = learning_rate

        return {
            "metrics": {
                "step": float(_int_scalar(self._state.step)),
                "learning_rate": learning_rate,
            }
        }

    def save_weights(self, payload: dict[str, Any]) -> dict[str, Any]:
        save_path = str(Path(payload["save_path"]).resolve())
        manager, _ = self._checkpoints.initialize_checkpoint_dir(
            save_path,
            keep_period=None,
            overwrite=True,
            resume=False,
        )
        try:
            self._checkpoints.save_state(
                manager,
                self._state,
                self._data_loader,
                _int_scalar(self._state.step),
            )
            manager.wait_until_finished()
        finally:
            close = getattr(manager, "close", None)
            if callable(close):
                close()
        return {"path": save_path, "current_step": _int_scalar(self._state.step)}

    def load_weights(self, payload: dict[str, Any]) -> dict[str, Any]:
        load_path = str(Path(payload["load_path"]).resolve())
        manager, resuming = self._checkpoints.initialize_checkpoint_dir(
            load_path,
            keep_period=None,
            overwrite=False,
            resume=True,
        )
        if not resuming:
            close = getattr(manager, "close", None)
            if callable(close):
                close()
            raise FileNotFoundError(f"OpenPI FAST checkpoint has no saved steps: {load_path}")
        try:
            self._state = self._checkpoints.restore_state(
                manager,
                self._state,
                self._data_loader,
            )
        finally:
            close = getattr(manager, "close", None)
            if callable(close):
                close()
        return {
            "current_step": _int_scalar(self._state.step),
            "learning_rate": self._learning_rate,
        }

    def shutdown(self) -> dict[str, Any]:
        self._closed = True
        self._pending_grads = None
        return {"stopped": True}


def _dispatch(session: OpenPIFastWorkerSession | None, op: str, payload: dict[str, Any]) -> tuple[Any, bool]:
    if session is None:
        raise RuntimeError("OpenPI FAST worker session is not initialized")

    if op == "forward_backward":
        return session.forward_backward(payload), False
    if op == "optim_step":
        return session.optim_step(payload), False
    if op == "save_weights":
        return session.save_weights(payload), False
    if op == "load_weights":
        return session.load_weights(payload), False
    if op == "shutdown":
        return session.shutdown(), True

    raise ValueError(f"Unknown OpenPI FAST worker op: {op!r}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _reply({"event": "ready", "protocol_version": OPENPI_FAST_WORKER_PROTOCOL_VERSION})

    session: OpenPIFastWorkerSession | None = None
    for line in sys.stdin:
        request_id: str | None = None
        should_stop = False
        try:
            request = json.loads(line)
            request_id = str(request["id"])
            op = str(request["op"])
            payload = dict(request.get("payload") or {})
            if op == "create_session":
                if session is not None:
                    raise ValueError("OpenPI FAST worker session is already initialized")
                session = OpenPIFastWorkerSession(payload)
                response = session.create_session()
            else:
                response, should_stop = _dispatch(session, op, payload)
            _reply({"id": request_id, "ok": True, "payload": response})
        except Exception as exc:
            logger.exception("OpenPI FAST worker request failed")
            _reply(
                {
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
        if should_stop:
            break


if __name__ == "__main__":
    main()
