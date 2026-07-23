from __future__ import annotations

import base64
import contextlib
import dataclasses
import io
import json
import logging
import structlog

import os
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mint_server.backend.openpi.openpi_fast_action_runtime import find_openpi_policy_checkpoint_dir
from mint_server.backend.openpi.openpi_fast_runtime import OPENPI_FAST_WORKER_PROTOCOL_VERSION
from mint_server.backend.openpi.openpi_session_state import OpenPISessionStateManager
from mint_server.backend.openpi.pi05_profiles import (
    get_pi05_profile,
    validate_profile_manifest,
    write_profile_manifest,
)


logger = structlog.get_logger(__name__)
_PROTOCOL_STDOUT = None


def _install_protocol_stdout_redirect() -> None:
    global _PROTOCOL_STDOUT
    if _PROTOCOL_STDOUT is not None:
        return
    protocol_fd = os.dup(sys.stdout.fileno())
    _PROTOCOL_STDOUT = os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8", closefd=True)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def _reply(message: dict[str, Any]) -> None:
    stream = _PROTOCOL_STDOUT or sys.stdout
    stream.write(json.dumps(message) + "\n")
    stream.flush()


@dataclass(frozen=True)
class OpenPIPi05RuntimeInitOverrides:
    weights_path: str | None = None
    random_init: bool = False

    @classmethod
    def from_env(cls) -> "OpenPIPi05RuntimeInitOverrides":
        weights_path = (os.environ.get("MINT_OPENPI_PI05_WEIGHTS_PATH") or "").strip() or None
        random_init = (os.environ.get("MINT_OPENPI_PI05_RANDOM_INIT") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if weights_path and random_init:
            raise ValueError(
                "MINT_OPENPI_PI05_WEIGHTS_PATH and MINT_OPENPI_PI05_RANDOM_INIT are mutually exclusive"
            )
        if weights_path is not None:
            weights_path = str(Path(weights_path).resolve())
        return cls(weights_path=weights_path, random_init=random_init)


def _float_scalar(value: Any) -> float:
    return float(np.asarray(value).item())


def _int_scalar(value: Any) -> int:
    return int(np.asarray(value).item())


def _compute_importance_sampling_stats(
    *,
    current_logprobs: np.ndarray,
    old_logprobs: np.ndarray,
    advantages: np.ndarray,
) -> dict[str, float | int]:
    current = np.asarray(current_logprobs, dtype=np.float32).reshape(-1)
    old = np.asarray(old_logprobs, dtype=np.float32).reshape(-1)
    adv = np.asarray(advantages, dtype=np.float32).reshape(-1)

    if not (current.shape == old.shape == adv.shape):
        raise ValueError("OpenPI pi0.5 importance_sampling inputs must share the same shape")
    action_count = int(current.size)
    if action_count == 0:
        raise ValueError("OpenPI pi0.5 importance_sampling requires at least one action logprob")

    log_ratio = np.clip(current - old, a_min=-20.0, a_max=20.0)
    ratio = np.exp(log_ratio)
    loss = -float(np.sum(ratio * adv))
    return {
        "loss": loss,
        "ratio_mean": float(np.mean(ratio)),
        "action_count": action_count,
    }


def _compute_ppo_stats(
    *,
    current_logprobs: np.ndarray,
    old_logprobs: np.ndarray,
    advantages: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> dict[str, float | int]:
    current = np.asarray(current_logprobs, dtype=np.float32).reshape(-1)
    old = np.asarray(old_logprobs, dtype=np.float32).reshape(-1)
    adv = np.asarray(advantages, dtype=np.float32).reshape(-1)

    if not (current.shape == old.shape == adv.shape):
        raise ValueError("OpenPI pi0.5 ppo inputs must share the same shape")
    action_count = int(current.size)
    if action_count == 0:
        raise ValueError("OpenPI pi0.5 ppo requires at least one action logprob")

    log_ratio = np.clip(current - old, a_min=-20.0, a_max=20.0)
    ratio = np.exp(log_ratio)
    clipped_ratio = np.clip(ratio, a_min=clip_low, a_max=clip_high)
    unclipped = -ratio * adv
    clipped = -clipped_ratio * adv
    return {
        "loss": float(np.sum(np.maximum(unclipped, clipped))),
        "ratio_mean": float(np.mean(ratio)),
        "clipfrac_mean": float(np.mean((ratio < clip_low) | (ratio > clip_high))),
        "action_count": action_count,
    }


def _normal_logprob(sample: Any, mean: Any, std: Any, jnp: Any) -> Any:
    mask = std == 0
    std_safe = jnp.where(mask, jnp.ones_like(std), std)
    log_prob = -jnp.log(std_safe) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.square((sample - mean) / std_safe)
    return jnp.where(mask, jnp.zeros_like(log_prob), log_prob)


def _decode_image(encoded: dict[str, Any]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for OpenPI pi0.5 image decoding") from exc

    raw = base64.b64decode(encoded["data"])
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class _StaticDataLoader:
    def __init__(self, data_config: Any) -> None:
        self._data_config = data_config

    def data_config(self) -> Any:
        return self._data_config


class OpenPIPi05WorkerSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import optax

        import openpi.models.model as openpi_model  # type: ignore[reportMissingImports]
        import openpi.models.pi0 as pi0_model  # type: ignore[reportMissingImports]
        import openpi.models.pi0_config as pi0_config  # type: ignore[reportMissingImports]
        import openpi.shared.array_typing as array_typing  # type: ignore[reportMissingImports]
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
        self._pi0_model = pi0_model
        self._pi0_config = pi0_config
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

        # orbax-checkpoint compat: openpi pins orbax==0.11.13 (subscriptable
        # metadata), the runtime ships 0.11.40 (StepMetadata). Patch
        # restore_params so the vendored upstream src/ stays unmodified.
        from mint_server.backend.openpi.openpi_orbax_compat import (
            install_restore_params_compat,
        )

        install_restore_params_compat(openpi_model)

        self._payload = payload
        self._model_id = str(payload["model_id"])
        self._action_dim = int(payload["action_dim"])
        self._action_horizon = int(payload["action_horizon"])
        self._config_name = str(payload["config_name"])
        self._learning_rate = float(payload.get("learning_rate") or 0.0)
        self._max_token_len = int(payload["max_token_len"])
        profile_manifest = payload.get("profile")
        self._profile = None
        if profile_manifest is not None:
            self._profile = get_pi05_profile(str(profile_manifest.get("profile_id") or ""))
            if profile_manifest != self._profile.checkpoint_manifest():
                raise ValueError("OpenPI pi0.5 training payload profile manifest does not match its profile ID")
            if (self._action_dim, self._action_horizon, self._max_token_len) != (
                self._profile.action_dim,
                self._profile.action_horizon,
                self._profile.max_token_len,
            ):
                raise ValueError("OpenPI pi0.5 training payload dimensions disagree with profile manifest")

        config = config_mod.get_config(self._config_name)
        checkpoint_base_dir = os.environ.get(
            "MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR",
            str(Path("./checkpoints").resolve()),
        )
        assets_base_dir = os.environ.get(
            "MINT_OPENPI_PI05_ASSETS_BASE_DIR",
            str(Path("./assets").resolve()),
        )

        model_cfg = pi0_config.Pi0Config(
            **(
                self._profile.pi0_config_kwargs()
                if self._profile is not None
                else {
                    "pi05": True,
                    "action_dim": self._action_dim,
                    "action_horizon": self._action_horizon,
                    "max_token_len": self._max_token_len,
                    "discrete_state_input": False,
                    "paligemma_variant": "gemma_2b_lora",
                }
            )
        )
        if self._profile is None:
            freeze_filter = nnx.Not(nnx_utils.PathRegex(".*lora.*"))
        else:
            trainable = nnx.Any(
                nnx_utils.PathRegex(".*lora.*"),
                nnx_utils.PathRegex(".*action_in_proj.*"),
                nnx_utils.PathRegex(".*time_mlp_in.*"),
                nnx_utils.PathRegex(".*time_mlp_out.*"),
                nnx_utils.PathRegex(".*action_out_proj.*"),
            )
            freeze_filter = nnx.Not(trainable)

        self._config = dataclasses.replace(
            config,
            exp_name=self._model_id,
            project_name="mint-openpi",
            checkpoint_base_dir=checkpoint_base_dir,
            assets_base_dir=assets_base_dir,
            wandb_enabled=False,
            overwrite=False,
            resume=False,
            model=model_cfg,
            freeze_filter=freeze_filter,
            ema_decay=None,
        )
        overrides = OpenPIPi05RuntimeInitOverrides.from_env()
        self._seed_assets_dir: Path | None = None
        if overrides.weights_path is not None:
            logger.info("openpi_pi0_5_worker_using_explicit_weights_path___s")
            self._config = dataclasses.replace(
                self._config,
                weight_loader=weight_loaders.CheckpointWeightLoader(overrides.weights_path),
            )
            seed_path = Path(overrides.weights_path).resolve()
            seed_root = seed_path.parent if seed_path.name == "params" else seed_path
            candidate_assets = seed_root / "assets"
            if candidate_assets.is_dir():
                self._seed_assets_dir = candidate_assets
        elif overrides.random_init:
            logger.info("OpenPI pi0.5 worker using explicit random-init mode")
            self._config = dataclasses.replace(
                self._config,
                weight_loader=weight_loaders.NoOpWeightLoader(),
            )

        self._data_loader = _StaticDataLoader(
            self._config.data.create(self._config.assets_dirs, self._config.model)
        )
        self._session_state_manager = OpenPISessionStateManager(
            Path(checkpoint_base_dir) / "_mint_session_state"
        )

        self._mesh = self._sharding_mod.make_mesh(self._config.fsdp_devices)
        self._state = self._init_train_state()
        self._rng = jax.random.key(self._config.seed)
        self._pending_grads = None
        self._closed = False

        # Data-parallel batch sharding (Plan.md "多卡并行训练" 方案A). fsdp_devices=1
        # (the current default) makes the mesh's DATA_AXIS span every visible device
        # as a pure batch-parallel axis -- see openpi.training.sharding.make_mesh.
        # `_flow_matching_grad_fn` is jit-compiled lazily on first use and cached
        # per (graphdef, batch_size); the cache key check is cheap relative to a
        # training step, so no need to invalidate it eagerly.
        self._data_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec(self._sharding_mod.DATA_AXIS)
        )
        self._replicated_sharding = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
        self._flow_matching_grad_fn = None
        self._flow_matching_grad_fn_batch_size: tuple[int, bool] | None = None

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

    @staticmethod
    def _trainable_shape_breakdown(params: Any) -> tuple[int, list[str]]:
        total = 0
        breakdown: list[str] = []
        for path, variable in params.flat_state().items():
            value = getattr(variable, "value", variable)
            shape = tuple(int(dim) for dim in value.shape)
            count = int(np.prod(shape, dtype=np.int64))
            total += count
            breakdown.append(f"{'/'.join(str(part) for part in path)}: shape={shape} count={count}")
        return total, breakdown

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
        trainable_shape = train_state_shape.params.filter(config.trainable_filter)
        self._trainable_parameter_count, breakdown = self._trainable_shape_breakdown(trainable_shape)
        if self._profile is not None and self._trainable_parameter_count != self._profile.expected_trainable_count:
            details = "\n".join(breakdown)
            raise ValueError(
                "OpenPI pi0.5 profile trainable-count mismatch: "
                f"expected={self._profile.expected_trainable_count}, actual={self._trainable_parameter_count}. "
                f"Trainable path/shape breakdown:\n{details}"
            )
        logger.info(
            "openpi_pi05_trainable_parameter_count profile=%s count=%s",
            None if self._profile is None else self._profile.profile_id,
            self._trainable_parameter_count,
        )
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

    def _padded_prompt(self, tokens: Any, mask: Any) -> dict[str, Any]:
        # pi0.5 tokenizes prompts to a FIXED max_token_len (padding tokens masked
        # off). Callers may send a variable-length prompt (trimmed to real tokens);
        # if we fed that straight to JAX, every distinct length would trigger a
        # fresh XLA compile whose executable is never freed — VRAM climbs until
        # RESOURCE_EXHAUSTED. Pad/truncate to max_token_len so the traced shape is
        # constant and the step is compiled exactly once.
        jnp = self._jnp
        n = int(self._max_token_len)
        tok = [int(t) for t in tokens][:n]
        msk = [bool(m) for m in mask][:n]
        if len(tok) < n:
            pad = n - len(tok)
            tok = tok + [0] * pad
            msk = msk + [False] * pad
        return {
            "tokenized_prompt": jnp.asarray([tok], dtype=jnp.int32),
            "tokenized_prompt_mask": jnp.asarray([msk], dtype=jnp.bool_),
        }

    def _observation_from_payload(self, item: dict[str, Any]):
        jnp = self._jnp
        images = {
            key: jnp.asarray(np.expand_dims(_decode_image(value), axis=0), dtype=jnp.uint8)
            for key, value in dict(item["image_bytes"]).items()
        }
        image_mask = {
            key: jnp.asarray([bool(value)], dtype=jnp.bool_)
            for key, value in dict(item["image_mask"]).items()
        }

        observation = self._openpi_model.Observation.from_dict(
            {
                "image": images,
                "image_mask": image_mask,
                "state": jnp.asarray([item["state"]], dtype=jnp.float32),
                **self._padded_prompt(item["tokenized_prompt"], item["tokenized_prompt_mask"]),
            }
        )
        actions = jnp.asarray(item["actions"], dtype=jnp.float32)[None, ...]
        return observation, actions

    def _padded_prompt_arrays(self, tokens: Any, mask: Any) -> tuple[list[int], list[bool]]:
        """Same padding as `_padded_prompt`, but returns plain lists (for stacking
        into a batch with np.stack), not a batch=1 jnp-wrapped dict."""
        n = int(self._max_token_len)
        tok = [int(t) for t in tokens][:n]
        msk = [bool(m) for m in mask][:n]
        if len(tok) < n:
            pad = n - len(tok)
            tok = tok + [0] * pad
            msk = msk + [False] * pad
        return tok, msk

    def _stack_flow_matching_batch(self, batch: list[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
        """Vectorize a list of per-item flow_matching payloads into one batched
        observation dict (plain numpy, not yet placed on any device) + actions
        array, instead of building `batch_size` separate batch=1 arrays.

        This is the data-parallel counterpart to `_observation_from_payload`: it
        is what gets `jax.device_put` onto `self._data_sharding` before entering
        the jitted grad function, so XLA's SPMD partitioner actually splits the
        batch dimension across the mesh's DATA_AXIS instead of every micro-batch
        landing on the same default device (see Plan.md 方案A).
        """
        camera_names = list(batch[0]["image_bytes"].keys())
        images = {
            cam: np.stack([_decode_image(item["image_bytes"][cam]) for item in batch], axis=0).astype(np.uint8)
            for cam in camera_names
        }
        image_mask = {
            cam: np.asarray([bool(item["image_mask"][cam]) for item in batch], dtype=bool) for cam in camera_names
        }
        tokens_masks = [
            self._padded_prompt_arrays(item["tokenized_prompt"], item["tokenized_prompt_mask"]) for item in batch
        ]
        tokenized_prompt = np.asarray([tm[0] for tm in tokens_masks], dtype=np.int32)
        tokenized_prompt_mask = np.asarray([tm[1] for tm in tokens_masks], dtype=bool)
        state = np.asarray([item["state"] for item in batch], dtype=np.float32)
        actions = np.asarray([item["actions"] for item in batch], dtype=np.float32)

        obs_dict = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
        return obs_dict, actions

    def _get_flow_matching_grad_fn(self, batch_size: int, *, use_data_sharding: bool):
        """Lazily build (and cache) the jitted flow_matching grad function for a
        given (batch_size, use_data_sharding). XLA needs a fixed traced shape per
        distinct batch_size, so the cache key includes it -- this mirrors the
        existing `_padded_prompt` fixed-shape rationale (recompiling per distinct
        shape is a slow, VRAM-leaking path; caching by shape avoids that).

        `use_data_sharding=False` degrades to replicated input placement (no
        cross-device batch split) when `batch_size` isn't a multiple of
        `jax.device_count()` -- JAX requires the sharded dimension's size to be
        evenly divisible by the number of devices it's split across (see
        ExperimentLog_MultiGPU.md Step 2: batch_size=2 on 4 devices hit exactly
        this with `ValueError: ... global size ... should be divisible by 4 ...
        but it is equal to 2`). Still jitted either way -- the jit-only speedup
        (~278x per Step 1) is independent of whether the batch is sharded across
        devices, only the last ~2x (4 vs 1 GPU) needs the divisibility to hold.

        Returns a callable `(rng, params, obs_dict, actions) -> (mean_loss, per_example_loss, grads)`.
        `mean_loss` is what feeds the optimizer (openpi upstream semantics: MEAN
        over the whole batch, NOT the sum-of-per-item-means the old per-item loop
        used -- see ExperimentLog_MultiGPU.md Step 2 for why this was chosen over
        preserving the old SUM scale).
        """
        cache_key = (batch_size, use_data_sharding)
        if self._flow_matching_grad_fn is not None and self._flow_matching_grad_fn_batch_size == cache_key:
            return self._flow_matching_grad_fn

        nnx = self._nnx
        jax = self._jax
        jnp = self._jnp
        diff_state = nnx.DiffState(0, self._config.trainable_filter)
        graphdef = self._state.model_def

        # `nnx.DiffState`-filtered value_and_grad needs to operate on a merged
        # Module (matching the existing single-item `_compute_grads`'s pattern),
        # not the raw `params` State -- merging *inside* loss_fn and passing raw
        # params as argnums=0 hits a pytree-structure mismatch inside nnx's
        # custom-node bookkeeping (confirmed by hitting exactly that error here;
        # verified fix mirrors scripts/wip/openpi_multi_gpu_repro.py's working
        # merge-outside-loss_fn pattern).
        def loss_fn(model_obj: Any, rng: Any, obs_dict_: Any, actions_: Any):
            observation = self._openpi_model.Observation.from_dict(obs_dict_)
            per_example_loss = model_obj.compute_loss(rng, observation, actions_, train=True)  # [B, action_horizon]
            return jnp.mean(per_example_loss), jnp.mean(per_example_loss, axis=-1)

        def grad_step(rng: Any, params: Any, obs_dict_: Any, actions_: Any):
            model_obj = nnx.merge(graphdef, params)
            model_obj.train()
            (mean_loss, per_example_loss), grads = nnx.value_and_grad(
                loss_fn, argnums=diff_state, has_aux=True
            )(model_obj, rng, obs_dict_, actions_)
            return mean_loss, per_example_loss, grads

        replicated_sharding = self._replicated_sharding
        input_sharding = self._data_sharding if use_data_sharding else replicated_sharding
        jitted = jax.jit(
            grad_step,
            in_shardings=(replicated_sharding, replicated_sharding, input_sharding, input_sharding),
        )
        self._flow_matching_grad_fn = jitted
        self._flow_matching_grad_fn_batch_size = cache_key
        return jitted

    def _compute_flow_matching_grads_batched(
        self, batch: list[dict[str, Any]]
    ) -> tuple[Any, list[float], float, float]:
        """Data-parallel replacement for looping `_compute_grads` once per item.

        Returns (grads, per_example_loss, mean_grad_norm, mean_param_norm) so the
        caller (`forward_backward`) can keep its existing metrics/loss_fn_outputs
        shape (one entry per input item) without needing a true per-item forward
        pass -- `compute_loss` already returns a per-example loss for free
        (see openpi.models.pi0.Pi0.compute_loss's `[*b, ah]` return shape), we
        only reduce it once for the aux output instead of running it batch_size
        times.
        """
        jax = self._jax
        sharding_mod = self._sharding_mod

        batch_size = len(batch)
        device_count = jax.device_count()
        use_data_sharding = batch_size % device_count == 0
        input_sharding = self._data_sharding if use_data_sharding else self._replicated_sharding

        obs_dict, actions = self._stack_flow_matching_batch(batch)
        sharded_obs = jax.tree.map(lambda a: jax.device_put(a, input_sharding), obs_dict)
        sharded_actions = jax.device_put(actions, input_sharding)

        grad_fn = self._get_flow_matching_grad_fn(batch_size, use_data_sharding=use_data_sharding)
        self._rng, step_rng = jax.random.split(self._rng)
        with sharding_mod.set_mesh(self._mesh):
            mean_loss, per_example_loss, grads = grad_fn(
                step_rng, self._state.params, sharded_obs, sharded_actions
            )

        model = self._nnx.merge(self._state.model_def, self._state.params)
        grad_norm, param_norm = self._grad_and_param_norm(model, grads)
        per_example_loss_list = [float(v) for v in np.asarray(jax.device_get(per_example_loss))]
        return grads, per_example_loss_list, grad_norm, param_norm

    def _rl_observation_from_payload(self, item: dict[str, Any]):
        observation, _ = self._observation_from_payload({**item, "actions": [[0.0] * self._action_dim] * self._action_horizon})
        chains = self._jnp.asarray(item["chains"], dtype=self._jnp.float32)[None, ...]
        old_logprobs = np.asarray(item["old_logprobs"], dtype=np.float32).reshape(
            self._action_horizon,
            int(item["source_action_dim"]),
        )
        advantages = np.asarray(item["advantages"], dtype=np.float32).reshape(
            self._action_horizon,
            int(item["source_action_dim"]),
        )
        return observation, chains, old_logprobs, advantages

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

    def _compute_velocity(
        self,
        model_obj: Any,
        *,
        observation: Any,
        x_t: Any,
        t: Any,
        prefix_tokens: Any,
        prefix_mask: Any,
        kv_cache: Any,
    ) -> Any:
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model_obj.embed_suffix(
            observation,
            x_t,
            self._jnp.broadcast_to(t, batch_size),
        )
        suffix_attn_mask = self._pi0_model.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = self._jnp.repeat(prefix_mask[:, None, :], suffix_tokens.shape[1], axis=1)
        full_attn_mask = self._jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = self._jnp.sum(prefix_mask, axis=-1)[:, None] + self._jnp.cumsum(suffix_mask, axis=-1) - 1

        prefix_out, suffix_out = model_obj.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )[0]
        if prefix_out is not None:
            raise RuntimeError("OpenPI pi0.5 suffix-only flow pass unexpectedly returned prefix output")
        return model_obj.action_out_proj(suffix_out[:, -self._action_horizon :])

    def _compute_flow_transition_logprobs(
        self,
        model_obj: Any,
        rng: Any,
        observation: Any,
        chains: Any,
        *,
        denoise_inds: list[int],
        loss_fn_config: dict[str, Any],
        source_action_dim: int,
    ) -> Any:
        jnp = self._jnp
        observation = self._openpi_model.preprocess_observation(rng, observation, train=False)
        num_transitions = int(chains.shape[1]) - 1
        num_steps = int(loss_fn_config.get("num_steps", num_transitions))
        if num_steps != num_transitions:
            raise ValueError(
                f"OpenPI pi0.5 RL num_steps={num_steps} must match chains transitions={num_transitions}"
            )

        noise_method = str(loss_fn_config.get("noise_method", "flow_sde"))
        if noise_method not in {"flow_sde", "flow_noise", "flow_ode"}:
            raise ValueError("OpenPI pi0.5 RL noise_method must be flow_sde, flow_noise, or flow_ode")

        joint_logprob = bool(loss_fn_config.get("joint_logprob", False))
        noise_level = float(loss_fn_config.get("noise_level", 0.5))
        noise_std = float(loss_fn_config.get("noise_std", loss_fn_config.get("flow_noise_std", 0.1)))
        if noise_level < 0.0:
            raise ValueError("OpenPI pi0.5 RL noise_level must be non-negative")
        if noise_std < 0.0:
            raise ValueError("OpenPI pi0.5 RL noise_std must be non-negative")

        prefix_tokens, prefix_mask, prefix_ar_mask = model_obj.embed_prefix(observation)
        prefix_attn_mask = self._pi0_model.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model_obj.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        timesteps = jnp.linspace(1.0, 0.0, num_steps + 1, dtype=chains.dtype)
        if len(denoise_inds) == 1:
            selected = [int(denoise_inds[0])]
        else:
            selected = [int(value) for value in denoise_inds]
        if any(idx < 0 or idx >= num_steps for idx in selected):
            raise ValueError("OpenPI pi0.5 RL denoise index out of range")

        logprobs = []
        if joint_logprob:
            logprobs.append(
                _normal_logprob(chains[:, 0], jnp.zeros_like(chains[:, 0]), jnp.ones_like(chains[:, 0]), jnp)[
                    :, :, :source_action_dim
                ]
            )
        for idx in selected:
            x_t = chains[:, idx]
            x_next = chains[:, idx + 1]
            t_input = timesteps[idx]
            delta = timesteps[idx] - timesteps[idx + 1]
            v_t = self._compute_velocity(
                model_obj,
                observation=observation,
                x_t=x_t,
                t=t_input,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
            )
            x0_pred = x_t - v_t * t_input
            x1_pred = x_t + v_t * (1.0 - t_input)

            if noise_method == "flow_ode":
                x0_weight = 1.0 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = jnp.zeros_like(x_t)
            elif noise_method == "flow_sde":
                denom_timesteps = jnp.where(timesteps == 1.0, timesteps[1], timesteps)
                sigma_ratio = timesteps / (1.0 - denom_timesteps)
                sigmas = noise_level * jnp.sqrt(sigma_ratio)[:-1]
                sigma_i = sigmas[idx]
                x0_weight = 1.0 - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2.0 * t_input)
                x_t_std = jnp.ones_like(x_t) * jnp.sqrt(delta) * sigma_i
            else:
                x0_weight = 1.0 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = jnp.ones_like(x_t) * noise_std

            x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
            logprobs.append(_normal_logprob(x_next, x_t_mean, x_t_std, jnp)[:, :, :source_action_dim])

        stacked = jnp.stack(logprobs, axis=1)
        return jnp.mean(stacked, axis=1)

    def _compute_importance_sampling_grads(
        self,
        observation: Any,
        chains: Any,
        old_logprobs: np.ndarray,
        advantages: np.ndarray,
        item: dict[str, Any],
        loss_fn_config: dict[str, Any] | None,
    ) -> tuple[Any, float, float, float, float, float, list[float]]:
        nnx = self._nnx
        jax = self._jax
        cfg = dict(loss_fn_config or {})
        source_action_dim = int(item["source_action_dim"])
        denoise_inds = [int(value) for value in item["denoise_inds"]]

        old_logprobs_t = self._jnp.asarray(old_logprobs[None, ...], dtype=self._jnp.float32)
        advantages_t = self._jnp.asarray(advantages[None, ...], dtype=self._jnp.float32)

        model = nnx.merge(self._state.model_def, self._state.params)
        model.train()
        self._rng, step_rng = jax.random.split(self._rng)

        def loss_fn(model_obj: Any, rng: Any, obs: Any):
            current_logprobs = self._compute_flow_transition_logprobs(
                model_obj,
                rng,
                obs,
                chains,
                denoise_inds=denoise_inds,
                loss_fn_config=cfg,
                source_action_dim=source_action_dim,
            )
            log_ratio = self._jnp.clip(current_logprobs - old_logprobs_t, a_min=-20.0, a_max=20.0)
            ratio = self._jnp.exp(log_ratio)
            loss = -self._jnp.sum(ratio * advantages_t)
            return loss, current_logprobs

        diff_state = nnx.DiffState(0, self._config.trainable_filter)
        (loss, current_logprobs), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model, step_rng, observation)
        grad_norm, param_norm = self._grad_and_param_norm(model, grads)

        current_logprobs_np = np.asarray(jax.device_get(current_logprobs), dtype=np.float32).reshape(old_logprobs.shape)
        stats = _compute_importance_sampling_stats(
            current_logprobs=current_logprobs_np,
            old_logprobs=old_logprobs,
            advantages=advantages,
        )
        _ = loss
        return (
            grads,
            float(stats["loss"]),
            grad_norm,
            param_norm,
            float(stats["ratio_mean"]),
            float(stats["action_count"]),
            current_logprobs_np.reshape(-1).tolist(),
        )

    def _compute_ppo_grads(
        self,
        observation: Any,
        chains: Any,
        old_logprobs: np.ndarray,
        advantages: np.ndarray,
        item: dict[str, Any],
        loss_fn_config: dict[str, Any] | None,
    ) -> tuple[Any, float, float, float, float, float, float, list[float]]:
        nnx = self._nnx
        jax = self._jax
        cfg = dict(loss_fn_config or {})
        source_action_dim = int(item["source_action_dim"])
        denoise_inds = [int(value) for value in item["denoise_inds"]]

        epsilon = float(cfg.get("epsilon", 0.2))
        clip_low = float(cfg.get("clip_low", 1.0 - epsilon))
        clip_high = float(cfg.get("clip_high", 1.0 + epsilon))
        if clip_low > clip_high:
            raise ValueError("ppo clip_low must be <= clip_high")

        old_logprobs_t = self._jnp.asarray(old_logprobs[None, ...], dtype=self._jnp.float32)
        advantages_t = self._jnp.asarray(advantages[None, ...], dtype=self._jnp.float32)

        model = nnx.merge(self._state.model_def, self._state.params)
        model.train()
        self._rng, step_rng = jax.random.split(self._rng)

        def loss_fn(model_obj: Any, rng: Any, obs: Any):
            current_logprobs = self._compute_flow_transition_logprobs(
                model_obj,
                rng,
                obs,
                chains,
                denoise_inds=denoise_inds,
                loss_fn_config=cfg,
                source_action_dim=source_action_dim,
            )
            log_ratio = self._jnp.clip(current_logprobs - old_logprobs_t, a_min=-20.0, a_max=20.0)
            ratio = self._jnp.exp(log_ratio)
            clipped_ratio = self._jnp.clip(ratio, a_min=clip_low, a_max=clip_high)
            unclipped = -ratio * advantages_t
            clipped = -clipped_ratio * advantages_t
            loss = self._jnp.sum(self._jnp.maximum(unclipped, clipped))
            return loss, current_logprobs

        diff_state = nnx.DiffState(0, self._config.trainable_filter)
        (loss, current_logprobs), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model, step_rng, observation)
        grad_norm, param_norm = self._grad_and_param_norm(model, grads)

        current_logprobs_np = np.asarray(jax.device_get(current_logprobs), dtype=np.float32).reshape(old_logprobs.shape)
        stats = _compute_ppo_stats(
            current_logprobs=current_logprobs_np,
            old_logprobs=old_logprobs,
            advantages=advantages,
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
            float(stats["action_count"]),
            current_logprobs_np.reshape(-1).tolist(),
        )

    def create_session(self) -> dict[str, Any]:
        return {"backend": "openpi_pi05", "config_name": self._config_name}

    def forward_backward(self, payload: dict[str, Any]) -> dict[str, Any]:
        loss_fn = str(payload.get("loss_fn") or "")
        if loss_fn not in {"flow_matching", "importance_sampling", "ppo"}:
            raise ValueError(
                "OpenPI pi0.5 only supports flow_matching, importance_sampling, "
                f"and ppo, got {loss_fn!r}"
            )

        batch = list(payload.get("batch") or [])
        if not batch:
            raise ValueError("OpenPI pi0.5 forward_backward requires a non-empty batch")

        loss_fn_config = dict(payload.get("loss_fn_config") or {})

        if loss_fn == "flow_matching":
            # Data-parallel path (Plan.md 方案A): one batched, jitted, sharded
            # grad call instead of a Python for-loop of batch_size single-item
            # calls -- see _compute_flow_matching_grads_batched's docstring and
            # ExperimentLog_MultiGPU.md Step 2 for why this replaced the old loop.
            grads, per_example_loss, grad_norm, param_norm = self._compute_flow_matching_grads_batched(batch)
            self._pending_grads = (
                grads if self._pending_grads is None else self._jax.tree.map(lambda a, b: a + b, self._pending_grads, grads)
            )
            loss_fn_outputs = [
                {"loss": {"data": [loss_value], "shape": [1], "dtype": "float32"}}
                for loss_value in per_example_loss
            ]
            batch_size = float(len(batch))
            metrics = {
                # MEAN over the whole batch (openpi upstream semantics), not the
                # old SUM-of-per-item-means -- see ExperimentLog_MultiGPU.md.
                "loss:mean": sum(per_example_loss) / max(batch_size, 1.0),
                "num_samples:sum": batch_size,
                "num_tokens:sum": batch_size * float(self._action_horizon),
                "grad_norm:mean": grad_norm,
                "param_norm:mean": param_norm,
            }
            return {
                "loss_fn_output_type": f"{loss_fn}_loss",
                "loss_fn_outputs": loss_fn_outputs,
                "metrics": metrics,
            }

        # RL paths (importance_sampling / ppo) are unchanged by the multi-GPU
        # data-parallel work -- Plan.md scoped that to flow_matching only, since
        # this skill's driver never exercises the RL loss_fns. Still per-item
        # Python loop, no jit/sharding here.
        total_loss = 0.0
        total_units = 0.0
        total_grad_norm = 0.0
        total_param_norm = 0.0
        total_ratio = 0.0
        total_clipfrac = 0.0
        num_rl_items = 0
        loss_fn_outputs: list[dict[str, Any]] = []
        pending_grads = self._pending_grads

        for item in batch:
            observation, chains, old_logprobs, advantages = self._rl_observation_from_payload(item)
            if loss_fn == "importance_sampling":
                (
                    grads,
                    loss_value,
                    grad_norm,
                    param_norm,
                    ratio_mean,
                    unit_count,
                    new_logprobs,
                ) = self._compute_importance_sampling_grads(
                    observation,
                    chains,
                    old_logprobs,
                    advantages,
                    item,
                    loss_fn_config,
                )
                total_ratio += ratio_mean
                num_rl_items += 1
            else:
                (
                    grads,
                    loss_value,
                    grad_norm,
                    param_norm,
                    ratio_mean,
                    clipfrac_mean,
                    unit_count,
                    new_logprobs,
                ) = self._compute_ppo_grads(
                    observation,
                    chains,
                    old_logprobs,
                    advantages,
                    item,
                    loss_fn_config,
                )
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
                        "shape": [self._action_horizon, int(item["source_action_dim"])],
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
            total_units += unit_count
            total_grad_norm += grad_norm
            total_param_norm += param_norm

        self._pending_grads = pending_grads

        batch_size = float(len(batch))
        denom = max(total_units, 1.0)
        metrics = {
            "loss:mean": total_loss / denom,
            "num_samples:sum": batch_size,
            "num_tokens:sum": total_units,
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
            raise ValueError("OpenPI pi0.5 optim_step requires a preceding forward_backward")

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
        self._state = dataclasses.replace(
            self._state,
            step=self._state.step + 1,
            params=full_params,
            opt_state=new_opt_state,
            tx=tx,
        )
        self._pending_grads = None
        self._learning_rate = learning_rate

        return {
            "metrics": {
                "step": float(_int_scalar(self._state.step)),
                "learning_rate": learning_rate,
            }
        }

    def _session_state_signature(self) -> dict[str, Any]:
        return {
            "config_name": self._config_name,
            "action_dim": self._action_dim,
            "action_horizon": self._action_horizon,
            "max_token_len": self._max_token_len,
            "profile": None if self._profile is None else self._profile.checkpoint_manifest(),
            "trainable_parameter_count": self._trainable_parameter_count,
        }

    def _write_profile_manifest(self, root: Path) -> None:
        if self._profile is not None:
            write_profile_manifest(root, self._profile)

    def _validate_profile_manifest(self, root: Path) -> None:
        if self._profile is not None:
            validate_profile_manifest(root, self._profile)

    def _save_train_state_checkpoint(self, path: Path, state: Any) -> None:
        checkpoint_path_obj = Path(path).resolve()
        checkpoint_path_obj.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_path_obj)
        manager, _ = self._checkpoints.initialize_checkpoint_dir(
            checkpoint_path,
            keep_period=None,
            overwrite=True,
            resume=False,
        )
        try:
            self._write_profile_manifest(checkpoint_path_obj)
            checkpoint_step = max(1, _int_scalar(state.step))
            self._checkpoints.save_state(
                manager,
                state,
                self._data_loader,
                checkpoint_step,
            )
            manager.wait_until_finished()
        finally:
            close = getattr(manager, "close", None)
            if callable(close):
                close()

    def _save_checkpoint_assets(self, directory: Path) -> None:
        data_config = self._data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            self._checkpoints._normalize.save(directory / data_config.asset_id, norm_stats)
            return

        seed_assets_dir = getattr(self, "_seed_assets_dir", None)
        if seed_assets_dir is not None and Path(seed_assets_dir).is_dir():
            shutil.copytree(Path(seed_assets_dir), directory, dirs_exist_ok=True)
            return

        raise FileNotFoundError(
            "OpenPI pi0.5 checkpoint export missing norm_stats and seed assets directory"
        )

    def _save_sampler_checkpoint(self, path: Path, state: Any) -> None:
        checkpoint_path_obj = Path(path).resolve()
        checkpoint_path_obj.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(checkpoint_path_obj)
        manager, _ = self._checkpoints.initialize_checkpoint_dir(
            checkpoint_path,
            keep_period=None,
            overwrite=True,
            resume=False,
        )
        try:
            self._write_profile_manifest(checkpoint_path_obj)
            checkpoint_step = max(1, _int_scalar(state.step))
            # Sampler exports must reflect the current policy params, not a lagging EMA shadow.
            params = state.params

            manager.save(
                checkpoint_step,
                items={
                    "assets": (
                        self._save_checkpoint_assets
                        if callable(getattr(self, "_save_checkpoint_assets", None))
                        else lambda directory: OpenPIPi05WorkerSession._save_checkpoint_assets(self, directory)
                    ),
                    "params": {"params": params},
                },
            )
            manager.wait_until_finished()
            source_dir = find_openpi_policy_checkpoint_dir(checkpoint_path_obj)
            self._write_profile_manifest(source_dir)
        finally:
            close = getattr(manager, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _sampler_export_complete(export_dir: Path, profile: Any | None = None) -> bool:
        if not ((export_dir / "params").is_dir() and (export_dir / "assets").is_dir()):
            return False
        if profile is not None:
            try:
                validate_profile_manifest(export_dir, profile)
            except (FileNotFoundError, ValueError):
                return False
        return True

    def _save_sampler_export(self, export_path: Path, state: Any) -> Path:
        export_dir = Path(export_path).resolve()
        if export_dir.exists():
            if OpenPIPi05WorkerSession._sampler_export_complete(export_dir, self._profile):
                return export_dir
            shutil.rmtree(export_dir)
        temp_dir = export_dir.parent / f".openpi_pi05_sampler_export_{export_dir.name}_{uuid.uuid4().hex}"
        stage_dir = export_dir.parent / f".openpi_pi05_sampler_export_stage_{export_dir.name}_{uuid.uuid4().hex}"
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(stage_dir, ignore_errors=True)
        try:
            self._save_sampler_checkpoint(temp_dir, state)
            source_dir = find_openpi_policy_checkpoint_dir(temp_dir)
            params_dir = source_dir / "params"
            assets_dir = source_dir / "assets"
            if not params_dir.is_dir():
                raise FileNotFoundError(f"OpenPI pi0.5 sampler export missing params dir: {params_dir}")
            if not assets_dir.is_dir():
                raise FileNotFoundError(f"OpenPI pi0.5 sampler export missing assets dir: {assets_dir}")
            self._validate_profile_manifest(source_dir)
            stage_dir.mkdir(parents=True, exist_ok=False)
            shutil.copytree(params_dir, stage_dir / "params")
            shutil.copytree(assets_dir, stage_dir / "assets")
            self._write_profile_manifest(stage_dir)
            stage_dir.rename(export_dir)
            return export_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            shutil.rmtree(stage_dir, ignore_errors=True)

    def _load_train_state_checkpoint(self, path: Path) -> Any:
        checkpoint_path_obj = Path(path).resolve()
        self._validate_profile_manifest(checkpoint_path_obj)
        checkpoint_path = str(checkpoint_path_obj)
        manager, resuming = self._checkpoints.initialize_checkpoint_dir(
            checkpoint_path,
            keep_period=None,
            overwrite=False,
            resume=True,
        )
        if not resuming:
            close = getattr(manager, "close", None)
            if callable(close):
                close()
            raise FileNotFoundError(f"OpenPI pi0.5 checkpoint has no saved steps: {checkpoint_path}")
        try:
            return self._checkpoints.restore_state(
                manager,
                self._state,
                self._data_loader,
            )
        finally:
            close = getattr(manager, "close", None)
            if callable(close):
                close()

    def _session_state_tree(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": self._state.step,
            "params": self._state.params.filter(self._config.trainable_filter),
            "opt_state": self._state.opt_state,
        }
        if self._state.ema_params is not None:
            payload["ema_params"] = self._state.ema_params.filter(self._config.trainable_filter)
        return payload

    def _initialize_session_checkpoint_dir(
        self,
        path: Path,
        *,
        overwrite: bool,
        resume: bool,
    ) -> tuple[Any, bool]:
        checkpoint_path = Path(path).resolve()
        resuming = False
        if checkpoint_path.exists():
            if overwrite:
                shutil.rmtree(checkpoint_path)
                checkpoint_path.mkdir(parents=True, exist_ok=True)
            elif resume:
                resuming = True
            else:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_path} already exists. Use overwrite or resume."
                )
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        manager = self._ocp.CheckpointManager(
            str(checkpoint_path),
            item_handlers={"session_state": self._ocp.PyTreeCheckpointHandler()},
            options=self._ocp.CheckpointManagerOptions(
                max_to_keep=1,
                keep_period=None,
                create=False,
                async_options=self._ocp.AsyncOptions(timeout_secs=7200),
            ),
        )
        if resuming and tuple(manager.all_steps()) in [(), (0,)]:
            resuming = False
        return manager, resuming

    def _save_session_train_state_checkpoint(self, path: Path, state: dict[str, Any]) -> None:
        manager, _ = self._initialize_session_checkpoint_dir(path, overwrite=True, resume=False)
        try:
            self._write_profile_manifest(Path(path).resolve())
            checkpoint_step = max(1, _int_scalar(state["step"]))
            manager.save(checkpoint_step, items={"session_state": state})
            manager.wait_until_finished()
        finally:
            close = getattr(manager, "close", None)
            if callable(close):
                close()

    def _load_session_train_state_checkpoint(self, path: Path) -> Any:
        self._validate_profile_manifest(Path(path).resolve())
        manager, resuming = self._initialize_session_checkpoint_dir(path, overwrite=False, resume=True)
        if not resuming:
            close = getattr(manager, "close", None)
            if callable(close):
                close()
            raise FileNotFoundError(f"OpenPI pi0.5 session checkpoint has no saved steps: {path}")
        try:
            restored = manager.restore(
                None,
                items={"session_state": self._session_state_tree()},
            )["session_state"]
        finally:
            close = getattr(manager, "close", None)
            if callable(close):
                close()

        params = self._state.params
        params.replace_by_pure_dict(restored["params"].to_pure_dict())
        ema_params = self._state.ema_params
        restored_ema_params = restored.get("ema_params")
        if ema_params is None and restored_ema_params is not None:
            raise ValueError("OpenPI pi0.5 session checkpoint unexpectedly contains ema_params")
        if ema_params is not None and restored_ema_params is None:
            raise ValueError("OpenPI pi0.5 session checkpoint is missing ema_params")
        if ema_params is not None and restored_ema_params is not None:
            ema_params.replace_by_pure_dict(restored_ema_params.to_pure_dict())
        return dataclasses.replace(
            self._state,
            step=restored["step"],
            params=params,
            opt_state=restored["opt_state"],
            ema_params=ema_params,
        )

    def save_weights(self, payload: dict[str, Any]) -> dict[str, Any]:
        save_path = str(Path(payload["save_path"]).resolve())
        self._save_train_state_checkpoint(Path(save_path), self._state)
        return {"path": save_path, "current_step": _int_scalar(self._state.step)}

    def save_sampler_weights(self, payload: dict[str, Any]) -> dict[str, Any]:
        export_path = payload.get("export_path")
        if export_path:
            export_dir = self._save_sampler_export(Path(str(export_path)), self._state)
            return {"path": str(export_dir), "current_step": _int_scalar(self._state.step)}
        save_path = str(Path(payload["save_path"]).resolve())
        self._save_sampler_checkpoint(Path(save_path), self._state)
        return {"path": save_path, "current_step": _int_scalar(self._state.step)}

    def load_weights(self, payload: dict[str, Any]) -> dict[str, Any]:
        load_path = str(Path(payload["load_path"]).resolve())
        self._state = self._load_train_state_checkpoint(Path(load_path))
        return {
            "current_step": _int_scalar(self._state.step),
            "learning_rate": self._learning_rate,
        }

    def save_session_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        path = self._session_state_manager.save_state(
            session_id,
            worker_module=__name__,
            runtime_signature=self._session_state_signature(),
            state=self._session_state_tree(),
            rng=self._rng,
            pending_grads=self._pending_grads,
            learning_rate=self._learning_rate,
            current_step=_int_scalar(self._state.step),
            save_train_state_fn=self._save_session_train_state_checkpoint,
        )
        return {
            "path": str(path),
            "current_step": _int_scalar(self._state.step),
            "learning_rate": self._learning_rate,
        }

    def load_session_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        restored = self._session_state_manager.load_state(
            session_id,
            expected_worker_module=__name__,
            expected_runtime_signature=self._session_state_signature(),
            load_train_state_fn=self._load_session_train_state_checkpoint,
        )
        self._state = restored["state"]
        self._rng = restored["rng"]
        self._pending_grads = restored["pending_grads"]
        self._learning_rate = restored["learning_rate"]
        return {
            "current_step": restored["current_step"],
            "learning_rate": self._learning_rate,
        }

    def shutdown(self) -> dict[str, Any]:
        self._closed = True
        self._pending_grads = None
        return {"stopped": True}


def _dispatch(
    session: OpenPIPi05WorkerSession | None,
    op: str,
    payload: dict[str, Any],
) -> tuple[Any, bool]:
    if session is None:
        raise RuntimeError("OpenPI pi0.5 worker session is not initialized")

    if op == "forward_backward":
        return session.forward_backward(payload), False
    if op == "optim_step":
        return session.optim_step(payload), False
    if op == "save_weights":
        return session.save_weights(payload), False
    if op == "save_sampler_weights":
        return session.save_sampler_weights(payload), False
    if op == "load_weights":
        return session.load_weights(payload), False
    if op == "save_session_state":
        return session.save_session_state(payload), False
    if op == "load_session_state":
        return session.load_session_state(payload), False
    if op == "shutdown":
        return session.shutdown(), True

    raise ValueError(f"Unknown OpenPI pi0.5 worker op: {op!r}")


def _dispatch_with_protocol_stdout(
    session: OpenPIPi05WorkerSession | None,
    op: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        response, should_stop = _dispatch(session, op, payload)
    extra_stdout = capture.getvalue().strip()
    if extra_stdout:
        logger.warning("suppressed_non_protocol_stdout_from_openpi_worker___s")
    return response, should_stop


def main() -> None:
    _install_protocol_stdout_redirect()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _reply({"event": "ready", "protocol_version": OPENPI_FAST_WORKER_PROTOCOL_VERSION})

    session: OpenPIPi05WorkerSession | None = None
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
                    raise ValueError("OpenPI pi0.5 worker session is already initialized")
                session = OpenPIPi05WorkerSession(payload)
                response = session.create_session()
            else:
                response, should_stop = _dispatch_with_protocol_stdout(session, op, payload)
            _reply({"id": request_id, "ok": True, "payload": response})
        except Exception as exc:
            logger.exception("OpenPI pi0.5 worker request failed")
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
