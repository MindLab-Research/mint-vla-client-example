"""Detached Ray store for sampling session metadata.

This supports recovery after API server restarts:
- vLLM actors are detached Ray actors and can survive process death.
- The API process loses in-memory SessionManager state on restart.
- Persist minimal sampling-session metadata in a detached Ray actor so
  routing and LoRA bookkeeping can be restored.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from typing import Any

from ..config import otel_env_vars


logger = logging.getLogger(__name__)
_ACTOR_HANDLE = None


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
    return os.environ.get("MINT_SAMPLING_SESSION_STORE_ACTOR_NAME", "tinker_sampling_session_store")


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
    class _SamplingSessionStoreActor:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._sessions: dict[str, dict[str, Any]] = {}

        def upsert(self, session_id: str, info: dict[str, Any]) -> None:
            current = dict(self._sessions.get(session_id, {}))
            incoming = dict(info)
            incoming_version = max(1, int(incoming.get("metadata_version") or 1))
            current_version = max(1, int(current.get("metadata_version") or 1))
            if incoming_version < current_version:
                # Ignore stale metadata writes but keep monotonic last_activity.
                incoming_last_activity = float(incoming.get("last_activity", 0.0) or 0.0)
                current["last_activity"] = max(float(current.get("last_activity", 0.0) or 0.0), incoming_last_activity)
                self._sessions[session_id] = current
                return
            current.update(incoming)
            current.setdefault("session_id", session_id)
            current.setdefault("last_activity", time.time())
            current.setdefault("lora_loaded", False)
            current.setdefault("uses_base_model", False)
            current.setdefault("inflight_requests", 0)
            current["metadata_version"] = incoming_version
            self._sessions[session_id] = current

        def get(self, session_id: str) -> dict[str, Any] | None:
            return self._sessions.get(session_id)

        def delete(self, session_id: str) -> None:
            self._sessions.pop(session_id, None)

        def set_last_activity(self, session_id: str, last_activity: float) -> float | None:
            s = self._sessions.get(session_id)
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
        created = _SamplingSessionStoreActor.options(**options).remote()
        try:
            ray.get(created.list.remote())
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
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
        raise RuntimeError("Sampling session store actor is not ready on this API server") from e
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
        raise TypeError(f"Sampling session store returned non-list: {type(out)}")


def upsert_sampling_session(info: dict[str, Any]) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Sampling session store write skipped: Ray not initialized")
        return
    payload = dict(info)
    session_id = str(payload.get("session_id", ""))
    if not session_id:
        logger.warning("Sampling session store write skipped: missing session_id")
        return
    payload.setdefault("last_activity", time.time())
    payload["metadata_version"] = max(1, int(payload.get("metadata_version") or 1))
    try:
        actor = _get_or_create_actor()
        actor.upsert.remote(session_id, payload)
    except Exception as e:
        logger.warning("Sampling session store write failed: upsert: %s", e)


def delete_sampling_session(session_id: str) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Sampling session store write skipped: Ray not initialized")
        return
    try:
        actor = _get_or_create_actor()
        actor.delete.remote(session_id)
    except Exception as e:
        logger.warning("Sampling session store write failed: delete: %s", e)


def set_sampling_session_last_activity(session_id: str, last_activity: float) -> None:
    import ray

    if not ray.is_initialized():
        return
    try:
        actor = _get_or_create_actor()
        actor.set_last_activity.remote(session_id, float(last_activity))
    except Exception as e:
        logger.debug("Sampling session store write failed: last_activity: %s", e)


async def async_set_sampling_session_last_activity(session_id: str, last_activity: float) -> float | None:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.set_last_activity.remote(session_id, float(last_activity))
    )
    if out is None:
        return None
    return float(out)


def get_sampling_session_info(session_id: str) -> dict[str, Any] | None:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.get.remote(session_id))


async def async_get_sampling_session_info(session_id: str) -> dict[str, Any] | None:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.get.remote(session_id)
    )
    if out is None:
        return None
    if not isinstance(out, dict):
        raise TypeError(f"Sampling session store returned non-dict: {type(out)}")
    return out


def list_sampling_sessions() -> list[dict[str, Any]]:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.list.remote())


async def async_list_sampling_sessions() -> list[dict[str, Any]]:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.list.remote()
    )
    if not isinstance(out, list):
        raise TypeError(f"Sampling session store returned non-list: {type(out)}")
    return [dict(item) for item in out if isinstance(item, dict)]
