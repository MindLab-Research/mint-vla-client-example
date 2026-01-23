"""Detached Ray store for training session metadata.

This supports recovery after API server restarts:
- Training workers are detached Ray actors (or pools) that can survive process death.
- The API process loses in-memory TrainingSessionManager state on restart.
- Persist minimal session metadata in a detached Ray actor so we can restore routing.
"""

from __future__ import annotations

import os
from typing import Any


def _ray_namespace() -> str:
    return os.environ.get("MINT_RAY_NAMESPACE", "tinker")


def _actor_name() -> str:
    return os.environ.get("MINT_TRAINING_SESSION_STORE_ACTOR_NAME", "tinker_training_session_store")


def _get_or_create_actor():
    import ray

    name = _actor_name()
    namespace = _ray_namespace()
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        pass

    @ray.remote
    class _TrainingSessionStoreActor:
        def __init__(self) -> None:
            self._sessions: dict[str, dict[str, Any]] = {}

        def upsert(self, model_id: str, info: dict[str, Any]) -> None:
            merged = dict(info)
            if "current_step" not in merged:
                merged["current_step"] = int(self._sessions.get(model_id, {}).get("current_step", 0))
            self._sessions[model_id] = merged

        def get(self, model_id: str) -> dict[str, Any] | None:
            return self._sessions.get(model_id)

        def delete(self, model_id: str) -> None:
            self._sessions.pop(model_id, None)

        def bump_step(self, model_id: str) -> int:
            s = self._sessions.get(model_id)
            if s is None:
                return 0
            s["current_step"] = int(s.get("current_step", 0)) + 1
            return int(s["current_step"])

        def list(self) -> list[dict[str, Any]]:
            return list(self._sessions.values())

    try:
        return _TrainingSessionStoreActor.options(
            name=name,
            namespace=namespace,
            lifetime="detached",
        ).remote()
    except Exception:
        return ray.get_actor(name, namespace=namespace)


def upsert_training_session(info: dict[str, Any]) -> None:
    import ray

    if not ray.is_initialized():
        return
    actor = _get_or_create_actor()
    payload = dict(info)
    payload.setdefault("current_step", 0)
    ray.get(actor.upsert.remote(str(payload.get("model_id", "")), payload))


def delete_training_session(model_id: str) -> None:
    import ray

    if not ray.is_initialized():
        return
    actor = _get_or_create_actor()
    ray.get(actor.delete.remote(model_id))


def get_training_session_info(model_id: str) -> dict[str, Any] | None:
    import ray

    if not ray.is_initialized():
        return None
    actor = _get_or_create_actor()
    return ray.get(actor.get.remote(model_id))
