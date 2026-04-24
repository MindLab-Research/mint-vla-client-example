from __future__ import annotations

import base64
import contextlib
import gc
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
from .openpi_session_state import OpenPISessionStateManager


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
        phase = "imports"
        try:
            import jax
            import jax.numpy as jnp
            from scipy.fft import idct

            import openpi.models.model as openpi_model
            import openpi.models.tokenizer as tokenizer_mod
            import openpi.training.config as config_mod

            self._jax = jax
            self._jnp = jnp
            self._scipy_idct = idct
            self._openpi_model = openpi_model
            self._tokenizer_mod = tokenizer_mod
            self._config_mod = config_mod

            phase = "payload"
            self._action_session_id = str(payload["action_session_id"])
            self._base_model = str(payload["base_model"])
            self._config_name = str(payload["config_name"])
            self._action_dim = int(payload["action_dim"])
            self._action_horizon = int(payload["action_horizon"])
            self._action_token_budget = int(payload["action_token_budget"])
            if self._action_token_budget <= 0:
                raise ValueError(f"OpenPI FAST action inference requires positive action_token_budget, got {self._action_token_budget}")
            self._camera_layout = tuple(str(name) for name in payload["camera_layout"])
            self._checkpoint_dir = find_openpi_policy_checkpoint_dir(payload["checkpoint_path"])

            phase = "config"
            self._config = config_mod.get_config(self._config_name)
            phase = "checkpoint_load"
            self._load_checkpoint_dir(self._checkpoint_dir)
            phase = "tokenizer"
            tokenizer_cls = (
                tokenizer_mod.FASTTokenizer
                if self._config.model.fast_model_tokenizer is None
                else self._config.model.fast_model_tokenizer
            )
            tokenizer_kwargs = (
                {} if self._config.model.fast_model_tokenizer_kwargs is None else dict(self._config.model.fast_model_tokenizer_kwargs)
            )
            if tokenizer_cls is tokenizer_mod.FASTTokenizer:
                tokenizer_kwargs.setdefault(
                    "fast_tokenizer_path",
                    _resolve_fast_tokenizer_path(tokenizer_kwargs.get("fast_tokenizer_path") or "physical-intelligence/fast"),
                )
            self._tokenizer = tokenizer_cls(self._config.model.max_token_len, **tokenizer_kwargs)
            phase = "session_state"
            state_root = str(os.environ.get("MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT") or "").strip()
            if not state_root:
                raise RuntimeError("OpenPI FAST action inference requires MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT")
            self._session_state_manager = OpenPISessionStateManager(state_root)
            self._rng = jax.random.key(0)
            self._sample_counter = 0
        except Exception as exc:
            raise RuntimeError(
                f"OpenPI FAST action session init failed at phase={phase} "
                f"for checkpoint={payload.get('checkpoint_path')!r}"
            ) from exc

    def _load_checkpoint_dir(self, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = Path(checkpoint_dir).resolve()
        self._model = self._config.model.load(
            self._openpi_model.restore_params(self._checkpoint_dir / "params", dtype=self._jnp.bfloat16)
        )

    def _session_state_signature(self) -> dict[str, Any]:
        return {
            "base_model": self._base_model,
            "config_name": self._config_name,
            "action_dim": self._action_dim,
            "action_horizon": self._action_horizon,
            "action_token_budget": self._action_token_budget,
            "camera_layout": list(self._camera_layout),
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
            worker_module="tinker_server.backend.openpi_fast_action_worker",
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
            expected_worker_module="tinker_server.backend.openpi_fast_action_worker",
            expected_runtime_signature=self._session_state_signature(),
            load_train_state_fn=self._load_session_payload,
        )
        self._load_checkpoint_dir(Path(restored["state"]["checkpoint_path"]))
        self._rng = restored["rng"]
        return {"path": str(self._session_state_manager.get_session_path(session_id))}

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

    def _decode_sampled_tokens(self, action_tokens: np.ndarray) -> str:
        paligemma_tokenizer = getattr(self._tokenizer, "_paligemma_tokenizer", None)
        if paligemma_tokenizer is None:
            raise RuntimeError("OpenPI FAST action tokenizer is missing the PaliGemma decoder")
        return str(paligemma_tokenizer.decode(action_tokens.tolist()))

    def _extract_actions_strict(self, action_tokens: np.ndarray) -> np.ndarray:
        decoded_tokens = self._decode_sampled_tokens(action_tokens)
        if "Action: " not in decoded_tokens:
            snippet = decoded_tokens[:200].replace("\n", "\\n")
            raise RuntimeError(
                "OpenPI FAST sampler output is malformed: missing 'Action: ' prefix "
                f"(decoded={snippet!r}, token_prefix={action_tokens[:32].tolist()!r})"
            )

        paligemma_tokenizer = getattr(self._tokenizer, "_paligemma_tokenizer", None)
        fast_tokenizer = getattr(self._tokenizer, "_fast_tokenizer", None)
        if paligemma_tokenizer is None or fast_tokenizer is None:
            raise RuntimeError("OpenPI FAST action tokenizer is missing strict decode components")
        bpe_tokenizer = getattr(fast_tokenizer, "bpe_tokenizer", None)
        min_token = getattr(fast_tokenizer, "min_token", None)
        scale = getattr(fast_tokenizer, "scale", None)
        if bpe_tokenizer is None or min_token is None or scale is None:
            raise RuntimeError("OpenPI FAST action tokenizer cannot decode actions strictly")

        suffix_text = decoded_tokens.split("Action: ", 1)[1]
        has_pipe = "|" in suffix_text
        action_text = suffix_text.split("|", 1)[0].strip()
        raw_action_tokens = np.asarray(
            paligemma_tokenizer.encode(action_text),
            dtype=np.int32,
        )
        fast_action_tokens = np.asarray(
            self._tokenizer._act_tokens_to_paligemma_tokens(raw_action_tokens),
            dtype=np.int32,
        )
        decoded_action_text = str(bpe_tokenizer.decode(fast_action_tokens.tolist()))
        decoded_dct_coeff = np.asarray(list(map(ord, decoded_action_text)), dtype=np.float32) + float(min_token)
        try:
            decoded_dct_coeff = decoded_dct_coeff.reshape(-1, self._action_dim)
        except ValueError as exc:
            raise RuntimeError(
                "OpenPI FAST decoded action token count is not divisible by the configured action_dim "
                f"(count={decoded_dct_coeff.size}, action_dim={self._action_dim}, "
                f"sampled_token_count={action_tokens.size}, raw_action_token_count={raw_action_tokens.size}, "
                f"has_pipe={has_pipe}, action_text_prefix={action_text[:120]!r})"
            ) from exc

        expected_shape = (self._action_horizon, self._action_dim)
        if decoded_dct_coeff.shape != expected_shape:
            raise RuntimeError(
                "OpenPI FAST decoded action token matrix has the wrong shape "
                f"(got={decoded_dct_coeff.shape}, expected={expected_shape}, "
                f"sampled_token_count={action_tokens.size}, raw_action_token_count={raw_action_tokens.size}, "
                f"has_pipe={has_pipe}, action_text_prefix={action_text[:120]!r})"
            )
        return np.asarray(
            self._scipy_idct(decoded_dct_coeff / float(scale), axis=0, norm="ortho"),
            dtype=np.float32,
        )

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._observation_from_payload(payload)
        rng_before = [
            int(x)
            for x in np.asarray(self._jax.random.key_data(self._rng), dtype=np.uint32).reshape(-1).tolist()
        ]
        self._rng, rng = self._jax.random.split(self._rng)
        rng_after = [
            int(x)
            for x in np.asarray(self._jax.random.key_data(self._rng), dtype=np.uint32).reshape(-1).tolist()
        ]
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        if temperature < 0.0:
            raise ValueError(f"OpenPI FAST action inference temperature must be non-negative, got {temperature}")
        started = time.monotonic()
        action_tokens = np.asarray(
            self._model.sample_actions(
                rng,
                observation,
                max_decoding_steps=self._action_token_budget,
                temperature=temperature,
            )[0],
            dtype=np.int32,
        )
        debug_tokens_enabled = os.environ.get("MINT_OPENPI_FAST_DEBUG_TOKENS", "").strip() == "1"
        decoded_tokens = self._decode_sampled_tokens(action_tokens) if debug_tokens_enabled else None
        actions = self._extract_actions_strict(action_tokens)
        infer_ms = (time.monotonic() - started) * 1000.0
        self._sample_counter += 1
        result = {
            "actions": {
                "data": actions.reshape(-1).tolist(),
                "shape": list(actions.shape),
                "dtype": "float32",
            },
            "policy_timing": {
                "infer_ms": infer_ms,
                "temperature": temperature,
                "sample_counter": self._sample_counter,
                "rng_before": rng_before,
                "rng_after": rng_after,
            },
        }
        if decoded_tokens is not None:
            result["debug_sampled_tokens"] = action_tokens.reshape(-1).tolist()
            result["debug_decoded_tokens_prefix"] = decoded_tokens[:200]
        return result

    def shutdown(self) -> dict[str, Any]:
        jax_mod = getattr(self, "_jax", None)
        for attr in ("_model", "_tokenizer", "_config", "_rng"):
            with contextlib.suppress(Exception):
                setattr(self, attr, None)
        with contextlib.suppress(Exception):
            if jax_mod is not None:
                jax_mod.clear_caches()
        gc.collect()
        return {"stopped": True}


def _dispatch(session: OpenPIFastActionSession | None, op: str, payload: dict[str, Any]) -> tuple[dict[str, Any], OpenPIFastActionSession | None]:
    if op == "create_session":
        if session is not None:
            session.shutdown()
        return {"ready": True}, OpenPIFastActionSession(payload)
    if session is None:
        raise RuntimeError("OpenPI FAST action session is not initialized")
    if op == "act":
        return session.act(payload), session
    if op == "save_session_state":
        return session.save_session_state(payload), session
    if op == "load_session_state":
        return session.load_session_state(payload), session
    if op == "shutdown":
        return session.shutdown(), None
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
