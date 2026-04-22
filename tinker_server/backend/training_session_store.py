"""Detached Ray store for training session metadata.

This supports recovery after API server restarts:
- Training workers are detached Ray actors (or pools) that can survive process death.
- The API process loses in-memory TrainingSessionManager state on restart.
- Persist minimal session metadata in a detached Ray actor so we can restore routing.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from typing import Any

from ..config import otel_env_vars
from ..ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator


logger = logging.getLogger(__name__)
_ACTOR_HANDLE = None


def _reset_cached_actor_handle() -> None:
    global _ACTOR_HANDLE
    _ACTOR_HANDLE = None


_register_ray_reconnect_invalidator(_reset_cached_actor_handle)


async def _await_ray_ref(ref: Any) -> Any:
    if hasattr(ref, "__await__"):
        return await ref

    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, asyncio.Future):
            return await fut
        if isinstance(fut, concurrent.futures.Future):
            return await asyncio.wrap_future(fut)
        if hasattr(fut, "__await__"):
            return await fut

    raise TypeError(f"Ray ref is not awaitable: {type(ref)}")


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

    global _ACTOR_HANDLE
    name = _actor_name()
    namespace = _ray_namespace()
    try:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except ValueError:
        pass

    @ray.remote(num_cpus=0)
    class _TrainingSessionStoreActor:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._sessions: dict[str, dict[str, Any]] = {}

        def upsert(self, model_id: str, info: dict[str, Any]) -> None:
            current = dict(self._sessions.get(model_id, {}))
            incoming = dict(info)
            incoming_version = max(1, int(incoming.get("metadata_version") or 1))
            current_version = max(1, int(current.get("metadata_version") or 1))
            if incoming_version < current_version:
                incoming_last_activity = float(incoming.get("last_activity", 0.0) or 0.0)
                current_last_activity = float(current.get("last_activity", 0.0) or 0.0)
                current["last_activity"] = max(current_last_activity, incoming_last_activity)
                try:
                    incoming_step = int(incoming.get("current_step", 0))
                    current_step = int(current.get("current_step", 0))
                    current["current_step"] = max(current_step, incoming_step)
                except Exception:
                    pass
                self._sessions[model_id] = current
                return
            merged = dict(current)
            merged.update(incoming)
            merged.setdefault("current_step", int(current.get("current_step", 0)))
            merged.setdefault("last_activity", time.time())
            merged["metadata_version"] = incoming_version
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
    actor_otel_env = otel_env_vars()
    from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources
    apply_detached_actor_resources(options, ray)
    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=actor_otel_env,
    )

    try:
        created = _TrainingSessionStoreActor.options(
            **options
        ).remote()
        try:
            ray.get(created.list.remote())
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


def _get_cached_actor_for_async_request_path():
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    if _ACTOR_HANDLE is None:
        raise RuntimeError("Training session store actor is not ready on this API server")
    return _ACTOR_HANDLE


async def _reacquire_actor_for_async_request_path():
    import ray

    global _ACTOR_HANDLE

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")

    try:
        _ACTOR_HANDLE = await asyncio.to_thread(
            ray.get_actor,
            _actor_name(),
            namespace=_ray_namespace(),
        )
    except ValueError as e:
        _ACTOR_HANDLE = None
        raise RuntimeError("Training session store actor is not ready on this API server") from e
    return _ACTOR_HANDLE


async def _get_actor_for_async_request_path():
    if _ACTOR_HANDLE is not None:
        return _ACTOR_HANDLE
    return await _reacquire_actor_for_async_request_path()


async def _call_actor_for_async_request_path(remote_call):
    import ray

    global _ACTOR_HANDLE

    actor = await _get_actor_for_async_request_path()
    try:
        return await _await_ray_ref(remote_call(actor))
    except (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError):
        _ACTOR_HANDLE = None
        actor = await _reacquire_actor_for_async_request_path()
        return await _await_ray_ref(remote_call(actor))


def ensure_ready() -> None:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    out = ray.get(actor.list.remote())
    if not isinstance(out, list):
        raise TypeError(f"Training session store returned non-list: {type(out)}")


def _require_write_actor(op: str):
    import ray

    if not ray.is_initialized():
        raise RuntimeError(f"Training session store write failed: {op}: Ray not initialized")
    try:
        return _get_or_create_actor()
    except Exception as e:
        raise RuntimeError(f"Training session store write failed: {op}: {e}") from e


def upsert_training_session(info: dict[str, Any]) -> None:
    payload = dict(info)
    payload.setdefault("current_step", 0)
    payload.setdefault("last_activity", time.time())
    payload["metadata_version"] = max(1, int(payload.get("metadata_version") or 1))
    actor = _require_write_actor("upsert")
    actor.upsert.remote(str(payload.get("model_id", "")), payload)


async def async_upsert_training_session(info: dict[str, Any]) -> None:
    payload = dict(info)
    payload.setdefault("current_step", 0)
    payload.setdefault("last_activity", time.time())
    payload["metadata_version"] = max(1, int(payload.get("metadata_version") or 1))
    actor = await asyncio.to_thread(_get_or_create_actor)
    await _await_ray_ref(actor.upsert.remote(str(payload.get("model_id", "")), payload))


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


async def async_get_training_session_info(model_id: str) -> dict[str, Any] | None:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.get.remote(model_id)
    )
    if out is None:
        return None
    if not isinstance(out, dict):
        raise TypeError(f"Training session store returned non-dict: {type(out)}")
    return out


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


async def async_list_training_sessions() -> list[dict[str, Any]]:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.list.remote()
    )
    if not isinstance(out, list):
        raise TypeError(f"Training session store returned non-list: {type(out)}")
    return [dict(item) for item in out if isinstance(item, dict)]
