from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .openpi_fast_action_runtime import find_openpi_policy_checkpoint_dir
from .openpi_fast_runtime import OPENPI_FAST_WORKER_PROTOCOL_VERSION


logger = logging.getLogger(__name__)
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


def _decode_image(encoded: dict[str, Any]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for OpenPI FAST action image decoding") from exc

    raw = base64.b64decode(encoded["data"])
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _tensor_to_numpy(tensor: dict[str, Any], *, dtype: np.dtype) -> np.ndarray:
    shape = list(tensor["shape"])
    arr = np.asarray(tensor["data"], dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def _resolve_fast_tokenizer_path(default_path: str) -> str:
    override = (os.environ.get("MINT_OPENPI_FAST_TOKENIZER_PATH") or "").strip()
    if override:
        return override

    hf_home = Path((os.environ.get("HF_HOME") or "").strip() or "/vePFS-Mindverse/share/huggingface")
    repo_root = hf_home / "hub" / "models--physical-intelligence--fast"
    refs_main = repo_root / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_dir = repo_root / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)

    snapshots_dir = repo_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])

    return default_path


class OpenPIFastActionSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        import jax
        import jax.numpy as jnp

        import openpi.models.model as openpi_model
        import openpi.models.tokenizer as tokenizer_mod
        import openpi.training.config as config_mod

        self._jax = jax
        self._jnp = jnp
        self._openpi_model = openpi_model
        self._tokenizer_mod = tokenizer_mod
        self._config_mod = config_mod

        self._action_session_id = str(payload["action_session_id"])
        self._base_model = str(payload["base_model"])
        self._config_name = str(payload["config_name"])
        self._action_dim = int(payload["action_dim"])
        self._action_horizon = int(payload["action_horizon"])
        self._camera_layout = tuple(str(name) for name in payload["camera_layout"])
        self._checkpoint_dir = find_openpi_policy_checkpoint_dir(payload["checkpoint_path"])

        config = config_mod.get_config(self._config_name)
        self._model = config.model.load(
            openpi_model.restore_params(self._checkpoint_dir / "params", dtype=jnp.bfloat16)
        )
        tokenizer_cls = (
            tokenizer_mod.FASTTokenizer
            if config.model.fast_model_tokenizer is None
            else config.model.fast_model_tokenizer
        )
        tokenizer_kwargs = {} if config.model.fast_model_tokenizer_kwargs is None else dict(config.model.fast_model_tokenizer_kwargs)
        if tokenizer_cls is tokenizer_mod.FASTTokenizer:
            tokenizer_kwargs.setdefault(
                "fast_tokenizer_path",
                _resolve_fast_tokenizer_path(tokenizer_kwargs.get("fast_tokenizer_path") or "physical-intelligence/fast"),
            )
        self._tokenizer = tokenizer_cls(config.model.max_token_len, **tokenizer_kwargs)
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
            raise ValueError("OpenPI FAST action inference expects exactly one encoded_text chunk")

        images = {
            name: _decode_image(chunk)[None, ...]
            for name, chunk in zip(self._camera_layout, image_chunks, strict=True)
        }
        image_mask = {name: np.asarray([True], dtype=np.bool_) for name in self._camera_layout}

        state = _tensor_to_numpy(payload["extra_inputs"]["state"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, ...]
        prompt_tokens = np.asarray(text_chunks[0]["tokens"], dtype=np.int32)[None, ...]
        prompt_mask = np.ones_like(prompt_tokens, dtype=np.bool_)
        token_ar_mask = np.zeros_like(prompt_tokens, dtype=np.int32)

        return self._openpi_model.Observation.from_dict(
            {
                "image": images,
                "image_mask": image_mask,
                "state": state,
                "tokenized_prompt": prompt_tokens,
                "tokenized_prompt_mask": prompt_mask,
                "token_ar_mask": token_ar_mask,
            }
        )

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._observation_from_payload(payload)
        self._rng, rng = self._jax.random.split(self._rng)
        started = time.monotonic()
        action_tokens = np.asarray(self._model.sample_actions(rng, observation)[0], dtype=np.int32)
        actions = np.asarray(
            self._tokenizer.extract_actions(
                action_tokens,
                action_horizon=self._action_horizon,
                action_dim=self._action_dim,
            ),
            dtype=np.float32,
        )
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


def _dispatch(session: OpenPIFastActionSession | None, op: str, payload: dict[str, Any]) -> tuple[dict[str, Any], OpenPIFastActionSession | None]:
    if op == "create_session":
        return {"ready": True}, OpenPIFastActionSession(payload)
    if session is None:
        raise RuntimeError("OpenPI FAST action session is not initialized")
    if op == "act":
        return session.act(payload), session
    if op == "shutdown":
        return session.shutdown(), session
    raise ValueError(f"Unknown OpenPI FAST action worker op: {op}")


def _dispatch_with_protocol_stdout(
    session: OpenPIFastActionSession | None, op: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], OpenPIFastActionSession | None]:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        response, next_session = _dispatch(session, op, payload)
    extra_stdout = capture.getvalue().strip()
    if extra_stdout:
        logger.warning("Suppressed non-protocol stdout from OpenPI FAST action worker: %s", extra_stdout)
    return response, next_session


def main() -> None:
    _install_protocol_stdout_redirect()
    logging.basicConfig(level=logging.INFO)
    _reply({"event": "ready", "protocol_version": OPENPI_FAST_WORKER_PROTOCOL_VERSION})

    session: OpenPIFastActionSession | None = None
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        request_id = request["id"]
        try:
            payload, session = _dispatch_with_protocol_stdout(session, request["op"], request.get("payload", {}))
            _reply({"id": request_id, "ok": True, "payload": payload})
            if request["op"] == "shutdown":
                break
        except Exception as e:
            logger.exception("OpenPI FAST action worker request failed")
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
