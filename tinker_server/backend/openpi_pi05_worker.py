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

        import openpi.models.model as openpi_model
        import openpi.models.pi0_config as pi0_config
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

        self._payload = payload
        self._model_id = str(payload["model_id"])
        self._action_dim = int(payload["action_dim"])
        self._action_horizon = int(payload["action_horizon"])
        self._config_name = str(payload["config_name"])
        self._learning_rate = float(payload.get("learning_rate") or 0.0)
        self._max_token_len = int(payload["max_token_len"])

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
            pi05=True,
            action_dim=self._action_dim,
            action_horizon=self._action_horizon,
            max_token_len=self._max_token_len,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
        )
        freeze_filter = nnx.Not(nnx_utils.PathRegex(".*lora.*"))

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
        if overrides.weights_path is not None:
            logger.info("OpenPI pi0.5 worker using explicit weights path: %s", overrides.weights_path)
            self._config = dataclasses.replace(
                self._config,
                weight_loader=weight_loaders.CheckpointWeightLoader(overrides.weights_path),
            )
        elif overrides.random_init:
            logger.info("OpenPI pi0.5 worker using explicit random-init mode")
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
                "tokenized_prompt": jnp.asarray([item["tokenized_prompt"]], dtype=jnp.int32),
                "tokenized_prompt_mask": jnp.asarray([item["tokenized_prompt_mask"]], dtype=jnp.bool_),
            }
        )
        actions = jnp.asarray(item["actions"], dtype=jnp.float32)[None, ...]
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

    def create_session(self) -> dict[str, Any]:
        return {"backend": "openpi_pi05", "config_name": self._config_name}

    def forward_backward(self, payload: dict[str, Any]) -> dict[str, Any]:
        loss_fn = str(payload.get("loss_fn") or "")
        if loss_fn != "flow_matching":
            raise ValueError(f"OpenPI pi0.5 only supports flow_matching, got {loss_fn!r}")

        batch = list(payload.get("batch") or [])
        if not batch:
            raise ValueError("OpenPI pi0.5 forward_backward requires a non-empty batch")

        total_loss = 0.0
        total_grad_norm = 0.0
        total_param_norm = 0.0
        loss_fn_outputs: list[dict[str, Any]] = []
        pending_grads = self._pending_grads

        for item in batch:
            observation, actions = self._observation_from_payload(item)
            grads, loss_value, grad_norm, param_norm = self._compute_grads(observation, actions)
            loss_fn_outputs.append(
                {
                    "loss": {
                        "data": [loss_value],
                        "shape": [1],
                        "dtype": "float32",
                    }
                }
            )
            pending_grads = (
                grads
                if pending_grads is None
                else self._jax.tree.map(lambda a, b: a + b, pending_grads, grads)
            )
            total_loss += loss_value
            total_grad_norm += grad_norm
            total_param_norm += param_norm

        self._pending_grads = pending_grads

        batch_size = float(len(batch))
        metrics = {
            "loss:mean": total_loss / batch_size,
            "num_samples:sum": batch_size,
            "grad_norm:mean": total_grad_norm / batch_size,
            "param_norm:mean": total_param_norm / batch_size,
        }
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
            raise FileNotFoundError(f"OpenPI pi0.5 checkpoint has no saved steps: {load_path}")
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
    if op == "load_weights":
        return session.load_weights(payload), False
    if op == "shutdown":
        return session.shutdown(), True

    raise ValueError(f"Unknown OpenPI pi0.5 worker op: {op!r}")


def main() -> None:
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
                response, should_stop = _dispatch(session, op, payload)
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
