from __future__ import annotations

import base64
import io
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .openpi_fast_action_runtime import find_openpi_policy_checkpoint_dir
from .openpi_fast_runtime import OPENPI_FAST_WORKER_PROTOCOL_VERSION


logger = logging.getLogger(__name__)


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

        model_cfg = pi0_config.Pi0Config(
            pi05=True,
            action_dim=self._action_dim,
            action_horizon=self._action_horizon,
            max_token_len=self._max_token_len,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
        )
        self._model = model_cfg.load(
            openpi_model.restore_params(self._checkpoint_dir / "params", dtype=jnp.bfloat16)
        )
        self._rng = jax.random.key(0)

    def _observation_from_payload(self, payload: dict[str, Any]):
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
            name: _decode_image(chunk)[None, ...]
            for name, chunk in zip(self._camera_layout, image_chunks, strict=True)
        }
        image_mask = {name: np.asarray([True], dtype=np.bool_) for name in self._camera_layout}

        state = _tensor_to_numpy(payload["extra_inputs"]["state"], dtype=np.float32)
        if state.ndim != 1:
            state = state.reshape(-1)
        state = _pad([float(x) for x in state.tolist()], self._action_dim)[None, ...]

        prompt_tokens = np.asarray(text_chunks[0]["tokens"], dtype=np.int32)[None, ...]
        prompt_mask = np.ones_like(prompt_tokens, dtype=np.bool_)

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
            "policy_timing": {"infer_ms": infer_ms},
        }

    def shutdown(self) -> dict[str, Any]:
        return {"stopped": True}


def _dispatch(
    session: OpenPIPi05ActionSession | None,
    op: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], OpenPIPi05ActionSession | None]:
    if op == "create_session":
        return {"ready": True}, OpenPIPi05ActionSession(payload)
    if session is None:
        raise RuntimeError("OpenPI pi0.5 action session is not initialized")
    if op == "act":
        return session.act(payload), session
    if op == "shutdown":
        return session.shutdown(), session
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

