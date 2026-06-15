from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable


_METADATA_FILE = "metadata.json"
_RNG_FILE = "rng.pkl"
_PENDING_GRADS_FILE = "pending_grads.pkl"
_TRAIN_STATE_DIR = "train_state"


def _normalize_runtime_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_runtime_signature(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_runtime_signature(item) for item in value]
    return value


class OpenPISessionStateManager:
    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def get_session_path(self, session_id: str) -> Path:
        return self._root_dir / session_id

    def session_exists(self, session_id: str) -> bool:
        return (self.get_session_path(session_id) / _METADATA_FILE).exists()

    def save_state(
        self,
        session_id: str,
        *,
        worker_module: str,
        runtime_signature: dict[str, Any],
        state: Any,
        rng: Any,
        pending_grads: Any,
        learning_rate: float,
        current_step: int,
        save_train_state_fn: Callable[[Path, Any], None],
    ) -> Path:
        session_path = self.get_session_path(session_id)
        session_path.mkdir(parents=True, exist_ok=True)

        save_train_state_fn(session_path / _TRAIN_STATE_DIR, state)

        with (session_path / _RNG_FILE).open("wb") as handle:
            pickle.dump(rng, handle)

        pending_grads_path = session_path / _PENDING_GRADS_FILE
        if pending_grads is None:
            if pending_grads_path.exists():
                pending_grads_path.unlink()
        else:
            with pending_grads_path.open("wb") as handle:
                pickle.dump(pending_grads, handle)

        metadata = {
            "worker_module": worker_module,
            "runtime_signature": _normalize_runtime_signature(runtime_signature),
            "learning_rate": float(learning_rate),
            "current_step": int(current_step),
            "has_pending_grads": pending_grads is not None,
        }
        with (session_path / _METADATA_FILE).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return session_path

    def load_state(
        self,
        session_id: str,
        *,
        expected_worker_module: str,
        expected_runtime_signature: dict[str, Any],
        load_train_state_fn: Callable[[Path], Any],
    ) -> dict[str, Any]:
        session_path = self.get_session_path(session_id)
        metadata_path = session_path / _METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"OpenPI session state missing for {session_id}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if metadata.get("worker_module") != expected_worker_module:
            raise ValueError(
                "OpenPI session state worker_module mismatch: "
                f"expected {expected_worker_module!r}, got {metadata.get('worker_module')!r}"
            )

        normalized_signature = _normalize_runtime_signature(expected_runtime_signature)
        if metadata.get("runtime_signature") != normalized_signature:
            raise ValueError(
                "OpenPI session state runtime_signature mismatch: "
                f"expected {normalized_signature!r}, got {metadata.get('runtime_signature')!r}"
            )

        state = load_train_state_fn(session_path / _TRAIN_STATE_DIR)
        with (session_path / _RNG_FILE).open("rb") as handle:
            rng = pickle.load(handle)

        pending_grads_path = session_path / _PENDING_GRADS_FILE
        pending_grads = None
        if pending_grads_path.exists():
            with pending_grads_path.open("rb") as handle:
                pending_grads = pickle.load(handle)

        return {
            "state": state,
            "rng": rng,
            "pending_grads": pending_grads,
            "learning_rate": float(metadata["learning_rate"]),
            "current_step": int(metadata["current_step"]),
            "metadata": metadata,
        }
