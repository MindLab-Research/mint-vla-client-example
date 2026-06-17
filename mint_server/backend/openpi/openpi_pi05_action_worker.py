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


def _pad(values: list[float], target_dim: int) -> np.ndarray:
    if len(values) > target_dim:
        raise ValueError(f"state length {len(values)} exceeds action_dim {target_dim}")
    return np.asarray(values + ([0.0] * (target_dim - len(values))), dtype=np.float32)


class OpenPIPi05ActionSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        import jax
        import jax.numpy as jnp

        import openpi.models.model as openpi_model
        import openpi.models.pi0_config as pi0_config

        self._jax = jax
        self._jnp = jnp
        self._openpi_model = openpi_model
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

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        if temperature != 0.0:
            raise ValueError(
                "OpenPI pi0.5 action inference does not support temperature-based exploration"
            )
        observation = self._observation_from_payload(payload)
        self._rng, rng = self._jax.random.split(self._rng)
        started = time.monotonic()
        actions = np.asarray(self._model.sample_actions(rng, observation)[0], dtype=np.float32)
        actions = actions[:, :7]
        infer_ms = (time.monotonic() - started) * 1000.0
        return {
            "actions": {
                "data": actions.reshape(-1).tolist(),
                "shape": list(actions.shape),
                "dtype": "float32",
            },
            "policy_timing": {"infer_ms": infer_ms, "temperature": temperature},
        }

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
