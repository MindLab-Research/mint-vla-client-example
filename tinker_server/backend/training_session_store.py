"""Detached Ray store for training session metadata.

This supports recovery after API server restarts:
- Training workers are detached Ray actors (or pools) that can survive process death.
- The API process loses in-memory TrainingSessionManager state on restart.
- Persist minimal session metadata in a detached Ray actor so we can restore routing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ..config import otel_env_vars


logger = logging.getLogger(__name__)


def _ray_namespace() -> str:
    env_ns = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


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

    @ray.remote(num_cpus=0)
    class _TrainingSessionStoreActor:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
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

        def set_step(self, model_id: str, step: int) -> int:
            s = self._sessions.get(model_id)
            if s is None:
                return int(step)
            s["current_step"] = max(int(s.get("current_step", 0)), int(step))
            return int(s["current_step"])

        def set_last_activity(self, model_id: str, last_activity: float) -> float | None:
            s = self._sessions.get(model_id)
            if s is None:
                return None
            s["last_activity"] = float(last_activity)
            return float(s["last_activity"])

        def list(self) -> list[dict[str, Any]]:
            return list(self._sessions.values())

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    try:
        if "node:__internal_head__" in ray.cluster_resources():
            options["resources"] = {"node:__internal_head__": 0.001}
    except Exception:
        pass
    actor_otel_env = otel_env_vars()
    from ..config import PFS_PYTHONPATH, actor_runtime_env
    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=actor_otel_env,
    )

    try:
        return _TrainingSessionStoreActor.options(
            **options
        ).remote()
    except Exception:
        return ray.get_actor(name, namespace=namespace)


def upsert_training_session(info: dict[str, Any]) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Training session store write skipped: Ray not initialized")
        return
    payload = dict(info)
    payload.setdefault("current_step", 0)
    payload.setdefault("last_activity", time.time())
    try:
        actor = _get_or_create_actor()
        actor.upsert.remote(str(payload.get("model_id", "")), payload)
    except Exception as e:
        logger.warning("Training session store write failed: upsert: %s", e)


def delete_training_session(model_id: str) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Training session store write skipped: Ray not initialized")
        return
    try:
        actor = _get_or_create_actor()
        actor.delete.remote(model_id)
    except Exception as e:
        logger.warning("Training session store write failed: delete: %s", e)


def set_training_session_last_activity(model_id: str, last_activity: float) -> None:
    import ray

    if not ray.is_initialized():
        return
    try:
        actor = _get_or_create_actor()
        actor.set_last_activity.remote(model_id, float(last_activity))
    except Exception as e:
        logger.debug("Training session store write failed: last_activity: %s", e)


def get_training_session_info(model_id: str) -> dict[str, Any] | None:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.get.remote(model_id))


def bump_training_session_step(model_id: str) -> int:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return int(ray.get(actor.bump_step.remote(model_id)))


def set_training_session_step(model_id: str, step: int) -> int:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return int(ray.get(actor.set_step.remote(model_id, int(step))))


def set_training_session_step_best_effort(model_id: str, step: int) -> None:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    actor.set_step.remote(model_id, int(step))


def bump_training_session_step_best_effort(model_id: str) -> None:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    actor.bump_step.remote(model_id)


def list_training_sessions() -> list[dict[str, Any]]:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.list.remote())
