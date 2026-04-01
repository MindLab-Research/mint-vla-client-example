"""Detached Ray store for session and sampler indices.

Persists minimal metadata for REST endpoints that need to enumerate or fetch
sessions and samplers across API server restarts.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
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
    return os.environ.get("MINT_SESSION_INDEX_ACTOR_NAME", "tinker_session_index_store")


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
    class _SessionIndexStore:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._sessions: dict[str, dict[str, Any]] = {}
            self._samplers: dict[str, dict[str, Any]] = {}

        def upsert_session(self, session_id: str, info: dict[str, Any]) -> None:
            current = dict(self._sessions.get(session_id, {}))
            current.update(info)
            current.setdefault("session_id", session_id)
            current.setdefault("training_run_ids", list(current.get("training_run_ids") or []))
            current.setdefault("sampler_ids", list(current.get("sampler_ids") or []))
            current.setdefault("heartbeat_sampler_ids", list(current.get("heartbeat_sampler_ids") or []))
            self._sessions[session_id] = current

        def add_training_run(
            self,
            session_id: str,
            training_run_id: str,
            user_id: str | None,
            created_at: str | None,
        ) -> None:
            current = dict(self._sessions.get(session_id, {}))
            runs = list(current.get("training_run_ids") or [])
            if training_run_id not in runs:
                runs.append(training_run_id)
            current["session_id"] = session_id
            current["training_run_ids"] = runs
            if user_id is not None:
                current.setdefault("user_id", user_id)
            if created_at is not None:
                current.setdefault("created_at", created_at)
            self._sessions[session_id] = current

        def add_sampler(
            self,
            session_id: str,
            sampler_id: str,
            user_id: str | None,
            created_at: str | None,
        ) -> None:
            current = dict(self._sessions.get(session_id, {}))
            samplers = list(current.get("sampler_ids") or [])
            if sampler_id not in samplers:
                samplers.append(sampler_id)
            current["session_id"] = session_id
            current["sampler_ids"] = samplers
            current.setdefault("heartbeat_sampler_ids", list(current.get("heartbeat_sampler_ids") or []))
            if user_id is not None:
                current.setdefault("user_id", user_id)
            if created_at is not None:
                current.setdefault("created_at", created_at)
            self._sessions[session_id] = current

        def add_heartbeat_sampler(
            self,
            session_id: str,
            sampler_id: str,
            user_id: str | None,
            created_at: str | None,
        ) -> None:
            current = dict(self._sessions.get(session_id, {}))
            samplers = list(current.get("sampler_ids") or [])
            if sampler_id not in samplers:
                samplers.append(sampler_id)
            heartbeat_samplers = list(current.get("heartbeat_sampler_ids") or [])
            if sampler_id not in heartbeat_samplers:
                heartbeat_samplers.append(sampler_id)
            current["session_id"] = session_id
            current["sampler_ids"] = samplers
            current["heartbeat_sampler_ids"] = heartbeat_samplers
            if user_id is not None:
                current.setdefault("user_id", user_id)
            if created_at is not None:
                current.setdefault("created_at", created_at)
            self._sessions[session_id] = current

        def get_session(self, session_id: str) -> dict[str, Any] | None:
            return self._sessions.get(session_id)

        def list_sessions(self) -> list[dict[str, Any]]:
            return list(self._sessions.values())

        def upsert_sampler(self, sampler_id: str, info: dict[str, Any]) -> None:
            current = dict(self._samplers.get(sampler_id, {}))
            current.update(info)
            current.setdefault("sampler_id", sampler_id)
            self._samplers[sampler_id] = current

        def get_sampler(self, sampler_id: str) -> dict[str, Any] | None:
            return self._samplers.get(sampler_id)

        def list_samplers(self) -> list[dict[str, Any]]:
            return list(self._samplers.values())

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
        _ACTOR_HANDLE = _SessionIndexStore.options(
            **options
        ).remote()
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


def _get_cached_actor_for_async_request_path():
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    if _ACTOR_HANDLE is None:
        raise RuntimeError("Session index store actor is not ready on this API server")
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
        raise RuntimeError("Session index store actor is not ready on this API server") from e
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
    out = ray.get(actor.list_sessions.remote())
    if not isinstance(out, list):
        raise TypeError(f"Session index store returned non-list: {type(out)}")


def upsert_session_index(info: dict[str, Any]) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Session index store write skipped: Ray not initialized")
        return
    session_id = str(info.get("session_id") or "")
    if not session_id:
        return
    try:
        actor = _get_or_create_actor()
        actor.upsert_session.remote(session_id, dict(info))
    except Exception as e:
        logger.warning("Session index store write failed: upsert_session: %s", e)


def add_training_run_to_session(
    session_id: str,
    training_run_id: str,
    *,
    user_id: str | None = None,
    created_at: str | None = None,
) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Session index store write skipped: Ray not initialized")
        return
    if not session_id or not training_run_id:
        return
    try:
        actor = _get_or_create_actor()
        actor.add_training_run.remote(session_id, training_run_id, user_id, created_at)
    except Exception as e:
        logger.warning("Session index store write failed: add_training_run: %s", e)


def add_sampler_to_session(
    session_id: str,
    sampler_id: str,
    *,
    user_id: str | None = None,
    created_at: str | None = None,
) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Session index store write skipped: Ray not initialized")
        return
    if not session_id or not sampler_id:
        return
    try:
        actor = _get_or_create_actor()
        actor.add_sampler.remote(session_id, sampler_id, user_id, created_at)
    except Exception as e:
        logger.warning("Session index store write failed: add_sampler: %s", e)


def add_heartbeat_sampler_to_session(
    session_id: str,
    sampler_id: str,
    *,
    user_id: str | None = None,
    created_at: str | None = None,
) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Session index store write skipped: Ray not initialized")
        return
    if not session_id or not sampler_id:
        return
    try:
        actor = _get_or_create_actor()
        try:
            actor.add_heartbeat_sampler.remote(session_id, sampler_id, user_id, created_at)
            return
        except AttributeError:
            logger.warning(
                "Session index store actor missing add_heartbeat_sampler; using compatibility upsert"
            )

        actor.add_sampler.remote(session_id, sampler_id, user_id, created_at)
        current = ray.get(actor.get_session.remote(session_id))
        heartbeat_samplers = []
        if isinstance(current, dict):
            heartbeat_samplers = list(current.get("heartbeat_sampler_ids") or [])
        if sampler_id not in heartbeat_samplers:
            heartbeat_samplers.append(sampler_id)
        actor.upsert_session.remote(
            session_id,
            {
                "session_id": session_id,
                "heartbeat_sampler_ids": heartbeat_samplers,
            },
        )
    except Exception as e:
        logger.warning("Session index store write failed: add_heartbeat_sampler: %s", e)


def get_session_index(session_id: str) -> dict[str, Any] | None:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.get_session.remote(session_id))


async def async_get_session_index(session_id: str) -> dict[str, Any] | None:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.get_session.remote(session_id)
    )
    if out is None:
        return None
    if not isinstance(out, dict):
        raise TypeError(f"Session index store returned non-dict: {type(out)}")
    return out


def list_session_index() -> list[dict[str, Any]]:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.list_sessions.remote())


async def async_list_session_index() -> list[dict[str, Any]]:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.list_sessions.remote()
    )
    if not isinstance(out, list):
        raise TypeError(f"Session index store returned non-list: {type(out)}")
    return [dict(item) for item in out if isinstance(item, dict)]


def upsert_sampler_index(info: dict[str, Any]) -> None:
    import ray

    if not ray.is_initialized():
        logger.warning("Session index store write skipped: Ray not initialized")
        return
    sampler_id = str(info.get("sampler_id") or "")
    if not sampler_id:
        return
    try:
        actor = _get_or_create_actor()
        actor.upsert_sampler.remote(sampler_id, dict(info))
    except Exception as e:
        logger.warning("Session index store write failed: upsert_sampler: %s", e)


def get_sampler_index(sampler_id: str) -> dict[str, Any] | None:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.get_sampler.remote(sampler_id))


async def async_get_sampler_index(sampler_id: str) -> dict[str, Any] | None:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.get_sampler.remote(sampler_id)
    )
    if out is None:
        return None
    if not isinstance(out, dict):
        raise TypeError(f"Sampler index store returned non-dict: {type(out)}")
    return out


def list_sampler_index() -> list[dict[str, Any]]:
    import ray

    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    actor = _get_or_create_actor()
    return ray.get(actor.list_samplers.remote())


async def async_list_sampler_index() -> list[dict[str, Any]]:
    out = await _call_actor_for_async_request_path(
        lambda actor: actor.list_samplers.remote()
    )
    if not isinstance(out, list):
        raise TypeError(f"Sampler index store returned non-list: {type(out)}")
    return [dict(item) for item in out if isinstance(item, dict)]
