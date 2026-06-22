from __future__ import annotations

import base64
import gc
import io
import json
import logging
import structlog
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from mint_server.backend.openpi.openpi_fast_action_runtime import find_openpi_policy_checkpoint_dir
from mint_server.backend.openpi.openpi_fast_runtime import OPENPI_FAST_WORKER_PROTOCOL_VERSION
from mint_server.backend.openpi.openpi_session_state import OpenPISessionStateManager


logger = structlog.get_logger(__name__)


def _reply(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _decode_image(encoded: dict[str, Any]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for OpenPI pi0.5 action image decoding") from exc

    raw = base64.b64decode(encoded["data"])
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _tensor_to_numpy(tensor: dict[str, Any], *, dtype: np.dtype) -> np.ndarray:
    shape = list(tensor["shape"])
    arr = np.asarray(tensor["data"], dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def _tensor_flag(tensor: Any) -> bool:
    if tensor is None:
        return False
    if isinstance(tensor, dict) and "data" in tensor:
        return bool(np.asarray(tensor["data"]).reshape(-1)[0])
    return bool(tensor)


def _pad(values: list[float], target_dim: int) -> np.ndarray:
    if len(values) > target_dim:
        raise ValueError(f"state length {len(values)} exceeds action_dim {target_dim}")
    return np.asarray(values + ([0.0] * (target_dim - len(values))), dtype=np.float32)


class OpenPIPi05ActionSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        import jax
        import jax.numpy as jnp

        import openpi.models.model as openpi_model  # type: ignore[reportMissingImports]
        import openpi.models.pi0 as pi0_model  # type: ignore[reportMissingImports]
        import openpi.models.pi0_config as pi0_config  # type: ignore[reportMissingImports]

        self._jax = jax
        self._jnp = jnp
        self._openpi_model = openpi_model
        self._pi0_model = pi0_model
        self._pi0_config = pi0_config

        self._action_session_id = str(payload["action_session_id"])
        self._base_model = str(payload["base_model"])
        self._action_dim = int(payload["action_dim"])
        self._action_horizon = int(payload["action_horizon"])
        self._max_token_len = int(payload["max_token_len"])
        self._camera_layout = tuple(str(name) for name in payload["camera_layout"])
        self._checkpoint_dir = find_openpi_policy_checkpoint_dir(payload["checkpoint_path"])

        self._model_cfg = pi0_config.Pi0Config(
            pi05=True,
            action_dim=self._action_dim,
            action_horizon=self._action_horizon,
            max_token_len=self._max_token_len,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
        )
        self._load_checkpoint_dir(self._checkpoint_dir)
        state_root = str(os.environ.get("MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT") or "").strip()
        if not state_root:
            raise RuntimeError("OpenPI pi0.5 action inference requires MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT")
        self._session_state_manager = OpenPISessionStateManager(state_root)
        self._rng = jax.random.key(0)

    def _load_checkpoint_dir(self, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = Path(checkpoint_dir).resolve()
        self._model = self._model_cfg.load(
            self._openpi_model.restore_params(self._checkpoint_dir / "params", dtype=self._jnp.bfloat16)
        )

    def _session_state_signature(self) -> dict[str, Any]:
        return {
            "base_model": self._base_model,
            "action_dim": self._action_dim,
            "action_horizon": self._action_horizon,
            "max_token_len": self._max_token_len,
            "camera_layout": list(self._camera_layout),
            "pi05": True,
        }

    def _save_session_payload(self, path: Path, state: dict[str, Any]) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _load_session_payload(self, path: Path) -> dict[str, Any]:
        return json.loads((path / "state.json").read_text(encoding="utf-8"))

    def save_session_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        path = self._session_state_manager.save_state(
            session_id,
            worker_module=__name__,
            runtime_signature=self._session_state_signature(),
            state={"checkpoint_path": str(self._checkpoint_dir)},
            rng=self._rng,
            pending_grads=None,
            learning_rate=0.0,
            current_step=0,
            save_train_state_fn=self._save_session_payload,
        )
        return {"path": str(path)}

    def load_session_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        restored = self._session_state_manager.load_state(
            session_id,
            expected_worker_module=__name__,
            expected_runtime_signature=self._session_state_signature(),
            load_train_state_fn=self._load_session_payload,
        )
        self._load_checkpoint_dir(Path(restored["state"]["checkpoint_path"]))
        self._rng = restored["rng"]
        return {"path": str(self._session_state_manager.get_session_path(session_id))}

    def _observation_from_payload(self, payload: dict[str, Any]):
        jnp = self._jnp
        chunks = list(payload["observation"]["chunks"])
        image_chunks = [chunk for chunk in chunks if chunk["type"] == "image"]
        text_chunks = [chunk for chunk in chunks if chunk["type"] == "encoded_text"]
        if len(image_chunks) != len(self._camera_layout):
            raise ValueError(
                f"Expected {len(self._camera_layout)} image chunks, got {len(image_chunks)}"
            )
        if len(text_chunks) != 1:
            raise ValueError("OpenPI pi0.5 action inference expects exactly one encoded_text chunk")

        images = {
            name: jnp.asarray(_decode_image(chunk)[None, ...], dtype=jnp.uint8)
            for name, chunk in zip(self._camera_layout, image_chunks, strict=True)
        }
        image_mask = {name: jnp.asarray([True], dtype=jnp.bool_) for name in self._camera_layout}

        state = _tensor_to_numpy(payload["extra_inputs"]["state"], dtype=np.float32)
        if state.ndim != 1:
            state = state.reshape(-1)
        state = jnp.asarray(
            _pad([float(x) for x in state.tolist()], self._action_dim)[None, ...],
            dtype=jnp.float32,
        )

        prompt_tokens = jnp.asarray(text_chunks[0]["tokens"], dtype=jnp.int32)[None, ...]
        prompt_mask = jnp.ones_like(prompt_tokens, dtype=jnp.bool_)

        return self._openpi_model.Observation.from_dict(
            {
                "image": images,
                "image_mask": image_mask,
                "state": state,
                "tokenized_prompt": prompt_tokens,
                "tokenized_prompt_mask": prompt_mask,
            }
        )

    def _normal_logprob(self, sample: Any, mean: Any, std: Any) -> Any:
        jnp = self._jnp
        mask = std == 0
        std_safe = jnp.where(mask, jnp.ones_like(std), std)
        log_prob = -jnp.log(std_safe) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.square((sample - mean) / std_safe)
        return jnp.where(mask, jnp.zeros_like(log_prob), log_prob)

    def _compute_velocity(
        self,
        *,
        observation: Any,
        x_t: Any,
        t: Any,
        prefix_tokens: Any,
        prefix_mask: Any,
        kv_cache: Any,
    ) -> Any:
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._model.embed_suffix(
            observation,
            x_t,
            self._jnp.broadcast_to(t, batch_size),
        )
        suffix_attn_mask = self._pi0_model.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = self._jnp.repeat(prefix_mask[:, None, :], suffix_tokens.shape[1], axis=1)
        full_attn_mask = self._jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = self._jnp.sum(prefix_mask, axis=-1)[:, None] + self._jnp.cumsum(suffix_mask, axis=-1) - 1

        prefix_out, suffix_out = self._model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )[0]
        if prefix_out is not None:
            raise RuntimeError("OpenPI pi0.5 suffix-only flow pass unexpectedly returned prefix output")
        return self._model.action_out_proj(suffix_out[:, -self._action_horizon :])

    def _flow_step_mean_std(
        self,
        *,
        observation: Any,
        x_t: Any,
        idx: int,
        timesteps: Any,
        prefix_tokens: Any,
        prefix_mask: Any,
        kv_cache: Any,
        noise_method: str,
        noise_level: float,
        noise_std: float,
    ) -> tuple[Any, Any]:
        jnp = self._jnp
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        v_t = self._compute_velocity(
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
        elif noise_method == "flow_noise":
            x0_weight = 1.0 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = jnp.ones_like(x_t) * noise_std
        else:
            raise ValueError("OpenPI pi0.5 rollout trace noise_method must be flow_sde, flow_noise, or flow_ode")

        return x0_pred * x0_weight + x1_pred * x1_weight, x_t_std

    def _sample_actions_with_trace(
        self,
        rng: Any,
        observation: Any,
        *,
        trace_config: dict[str, Any],
    ) -> dict[str, Any]:
        jax = self._jax
        jnp = self._jnp
        num_steps = int(trace_config.get("num_steps", 10))
        if num_steps <= 0:
            raise ValueError("OpenPI pi0.5 rollout trace num_steps must be positive")
        noise_method = str(trace_config.get("noise_method", "flow_sde"))
        if noise_method not in {"flow_sde", "flow_noise", "flow_ode"}:
            raise ValueError("OpenPI pi0.5 rollout trace noise_method must be flow_sde, flow_noise, or flow_ode")
        joint_logprob = bool(trace_config.get("joint_logprob", False))
        ignore_last = bool(trace_config.get("ignore_last", False))
        noise_level = float(trace_config.get("noise_level", 0.5))
        noise_std = float(trace_config.get("noise_std", trace_config.get("flow_noise_std", 0.1)))
        if noise_level < 0.0:
            raise ValueError("OpenPI pi0.5 rollout trace noise_level must be non-negative")
        if noise_std < 0.0:
            raise ValueError("OpenPI pi0.5 rollout trace noise_std must be non-negative")

        rng, preprocess_rng, noise_rng, denoise_rng = jax.random.split(rng, 4)
        observation = self._openpi_model.preprocess_observation(preprocess_rng, observation, train=False)
        batch_size = observation.state.shape[0]
        x_t = jax.random.normal(noise_rng, (batch_size, self._action_horizon, self._action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self._model.embed_prefix(observation)
        prefix_attn_mask = self._pi0_model.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self._model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        timesteps = jnp.linspace(1.0, 0.0, num_steps + 1, dtype=x_t.dtype)
        if joint_logprob:
            sample_denoise_inds = list(range(num_steps))
            output_denoise_inds = sample_denoise_inds
        else:
            max_index = num_steps - 2 if ignore_last and num_steps > 1 else num_steps - 1
            selected = int(jax.random.randint(denoise_rng, (), 0, max_index + 1))
            sample_denoise_inds = [selected] * num_steps
            output_denoise_inds = [selected]

        chains = [x_t]
        logprobs = []
        if joint_logprob:
            logprobs.append(self._normal_logprob(x_t, jnp.zeros_like(x_t), jnp.ones_like(x_t)))

        for idx in range(num_steps):
            sample_method = noise_method if idx == sample_denoise_inds[idx] else "flow_ode"
            x_t_mean, x_t_std = self._flow_step_mean_std(
                observation=observation,
                x_t=x_t,
                idx=idx,
                timesteps=timesteps,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
                noise_method=sample_method,
                noise_level=noise_level,
                noise_std=noise_std,
            )
            rng, step_noise_rng = jax.random.split(rng)
            x_t = x_t_mean + jax.random.normal(step_noise_rng, x_t.shape) * x_t_std
            chains.append(x_t)
            if joint_logprob or idx == sample_denoise_inds[idx]:
                logprobs.append(self._normal_logprob(x_t, x_t_mean, x_t_std))

        chains_t = jnp.stack(chains, axis=1)
        logprobs_t = jnp.stack(logprobs, axis=1)[:, :, :, :7]
        if joint_logprob:
            prev_logprobs = jnp.mean(logprobs_t, axis=1)
        else:
            prev_logprobs = logprobs_t[:, 0]
        return {
            "actions": x_t,
            "chains": chains_t,
            "denoise_inds": jnp.asarray(output_denoise_inds, dtype=jnp.int32),
            "prev_logprobs": prev_logprobs,
            "rng": rng,
        }

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        if temperature != 0.0:
            raise ValueError(
                "OpenPI pi0.5 action inference does not support temperature-based exploration"
            )
        extra_inputs = dict(payload.get("extra_inputs") or {})
        return_rollout_trace = bool(
            payload.get("return_rollout_trace")
            or _tensor_flag(extra_inputs.get("return_rollout_trace"))
        )
        trace_config = dict(payload.get("rollout_trace_config") or extra_inputs.get("rollout_trace_config") or {})
        observation = self._observation_from_payload(payload)
        self._rng, rng = self._jax.random.split(self._rng)
        started = time.monotonic()
        if return_rollout_trace:
            trace = self._sample_actions_with_trace(rng, observation, trace_config=trace_config)
            self._rng = trace["rng"]
            raw_actions = trace["actions"][0]
        else:
            raw_actions = self._model.sample_actions(rng, observation)[0]
        actions = np.asarray(raw_actions, dtype=np.float32)[:, :7]
        infer_ms = (time.monotonic() - started) * 1000.0
        result = {
            "actions": {
                "data": actions.reshape(-1).tolist(),
                "shape": list(actions.shape),
                "dtype": "float32",
            },
            "policy_timing": {"infer_ms": infer_ms, "temperature": temperature},
        }
        if return_rollout_trace:
            chains = np.asarray(trace["chains"][0], dtype=np.float32)
            denoise_inds = np.asarray(trace["denoise_inds"], dtype=np.int32)
            prev_logprobs = np.asarray(trace["prev_logprobs"][0], dtype=np.float32)
            result["rollout_trace"] = {
                "chains": {
                    "data": chains.reshape(-1).tolist(),
                    "shape": list(chains.shape),
                    "dtype": "float32",
                },
                "denoise_inds": {
                    "data": denoise_inds.reshape(-1).tolist(),
                    "shape": list(denoise_inds.shape),
                    "dtype": "int32",
                },
                "logprobs": {
                    "data": prev_logprobs.reshape(-1).tolist(),
                    "shape": list(prev_logprobs.shape),
                    "dtype": "float32",
                },
            }
        return result

    def shutdown(self) -> dict[str, Any]:
        jax_mod = getattr(self, "_jax", None)
        for attr in ("_model", "_config", "_rng"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass
        try:
            if jax_mod is not None:
                jax_mod.clear_caches()
        except Exception:
            pass
        gc.collect()
        return {"stopped": True}


def _dispatch(
    session: OpenPIPi05ActionSession | None,
    op: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], OpenPIPi05ActionSession | None]:
    if op == "create_session":
        if session is not None:
            session.shutdown()
        return {"ready": True}, OpenPIPi05ActionSession(payload)
    if session is None:
        raise RuntimeError("OpenPI pi0.5 action session is not initialized")
    if op == "act":
        return session.act(payload), session
    if op == "save_session_state":
        return session.save_session_state(payload), session
    if op == "load_session_state":
        return session.load_session_state(payload), session
    if op == "shutdown":
        return session.shutdown(), None
    raise ValueError(f"Unknown OpenPI pi0.5 action worker op: {op}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _reply({"event": "ready", "protocol_version": OPENPI_FAST_WORKER_PROTOCOL_VERSION})

    session: OpenPIPi05ActionSession | None = None
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        request_id = request["id"]
        try:
            payload, session = _dispatch(session, request["op"], request.get("payload", {}))
            _reply({"id": request_id, "ok": True, "payload": payload})
            if request["op"] == "shutdown":
                break
        except Exception as e:
            logger.exception("OpenPI pi0.5 action worker request failed")
            _reply(
                {
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                }
            )


if __name__ == "__main__":
    main()
