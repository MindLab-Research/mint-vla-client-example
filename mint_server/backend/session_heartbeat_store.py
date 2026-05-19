from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
from typing import Any

from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, otel_env_vars
from ..runtime_env import env_nonempty

_ACTOR_HANDLE = None

def _reset_cached_actor_handle() -> None:
    global _ACTOR_HANDLE
    _ACTOR_HANDLE = None

from ..ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator
_register_ray_reconnect_invalidator(_reset_cached_actor_handle)


def _ray_namespace() -> str:
    env_ns = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _actor_name() -> str:
    return os.environ.get("MINT_SESSION_HEARTBEAT_ACTOR_NAME", "tinker_session_heartbeat_store")


def _awaitable(ref: Any) -> Any:
    if hasattr(ref, "__await__"):
        return ref
    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, asyncio.Future):
            return fut
        if isinstance(fut, concurrent.futures.Future):
            return asyncio.wrap_future(fut)
        if hasattr(fut, "__await__"):
            return fut
    raise TypeError(f"Ray ref is not awaitable: {type(ref)}")


def _get_or_create_actor():
    import ray

    from ..ray_utils import force_reconnect_ray, is_wrong_cluster_error

    global _ACTOR_HANDLE
    name = _actor_name()
    namespace = _ray_namespace()

    def _get_existing_actor() -> Any | None:
        try:
            return ray.get_actor(name, namespace=namespace)
        except ValueError:
            return None

    try:
        _ACTOR_HANDLE = _get_existing_actor()
        if _ACTOR_HANDLE is not None:
            return _ACTOR_HANDLE
    except Exception as e:
        if not is_wrong_cluster_error(e):
            raise
        force_reconnect_ray(namespace=namespace)
        _ACTOR_HANDLE = _get_existing_actor()
        if _ACTOR_HANDLE is not None:
            return _ACTOR_HANDLE

    @ray.remote(num_cpus=0)
    class _RaySessionHeartbeatStore:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._last_seen: dict[str, float] = {}
            self._max_age_s = float(os.environ.get("MINT_SESSION_HEARTBEAT_MAX_AGE_S", str(7 * 86400)))
            self._prune_every = max(1, int(os.environ.get("MINT_SESSION_HEARTBEAT_PRUNE_EVERY", "256")))
            self._updates_since_prune = 0

        def update(self, session_id: str, now: float | None = None) -> None:
            if not session_id:
                return
            ts = time.time() if now is None else float(now)
            self._last_seen[str(session_id)] = ts
            self._updates_since_prune += 1
            if self._updates_since_prune >= self._prune_every:
                self._prune_locked(now=ts, max_age_s=self._max_age_s)
                self._updates_since_prune = 0

        def last_seen(self, session_id: str) -> float | None:
            return self._last_seen.get(str(session_id))

        def delete(self, session_id: str) -> bool:
            return self._last_seen.pop(str(session_id), None) is not None

        def size(self) -> int:
            return len(self._last_seen)

        def is_stale(self, session_id: str, ttl_s: float) -> bool:
            if ttl_s <= 0 or not session_id:
                return False
            last = self._last_seen.get(str(session_id))
            if last is None:
                return False
            return (time.time() - last) > float(ttl_s)

        def prune(self, max_age_s: float) -> int:
            if max_age_s <= 0:
                return 0
            return self._prune_locked(now=time.time(), max_age_s=float(max_age_s))

        def _prune_locked(self, *, now: float, max_age_s: float) -> int:
            to_delete = [sid for sid, ts in self._last_seen.items() if (now - ts) > max_age_s]
            for sid in to_delete:
                del self._last_seen[sid]
            return len(to_delete)

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
        "max_restarts": -1,
    }
    apply_detached_actor_resources(options, ray)
    options["runtime_env"] = actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=otel_env_vars())

    def _create_actor() -> Any:
        created = _RaySessionHeartbeatStore.options(**options).remote()
        try:
            ray.get(created.size.remote())
            return created
        except Exception as e:
            existing = _get_existing_actor()
            if existing is not None:
                return existing
            raise e

    try:
        _ACTOR_HANDLE = _create_actor()
        return _ACTOR_HANDLE
    except Exception as e:
        if is_wrong_cluster_error(e):
            force_reconnect_ray(namespace=namespace)
            _ACTOR_HANDLE = _get_existing_actor()
            if _ACTOR_HANDLE is None:
                _ACTOR_HANDLE = _create_actor()
            return _ACTOR_HANDLE
        _ACTOR_HANDLE = _get_existing_actor()
        if _ACTOR_HANDLE is None:
            raise
        return _ACTOR_HANDLE


class SessionHeartbeatStore:
    def __init__(self) -> None:
        self._ray_actor = None

    def _get_actor(self):
        import ray

        global _ACTOR_HANDLE
        if self._ray_actor is not None:
            return self._ray_actor
        if _ACTOR_HANDLE is not None:
            self._ray_actor = _ACTOR_HANDLE
            return self._ray_actor
        if not ray.is_initialized():
            raise RuntimeError("Ray not initialized")
        self._ray_actor = _get_or_create_actor()
        return self._ray_actor

    def ensure_ready(self) -> None:
        self._get_actor()

    def update(self, session_id: str, now: float | None = None) -> None:
        actor = self._get_actor()
        import ray

        ray.get(actor.update.remote(session_id=session_id, now=now))

    async def async_update(self, session_id: str, now: float | None = None) -> None:
        actor = self._get_actor()
        await _awaitable(actor.update.remote(session_id=session_id, now=now))

    def last_seen(self, session_id: str) -> float | None:
        actor = self._get_actor()
        import ray

        return ray.get(actor.last_seen.remote(session_id=session_id))

    def delete(self, session_id: str) -> bool:
        actor = self._get_actor()
        import ray

        return bool(ray.get(actor.delete.remote(session_id=session_id)))

    def size(self) -> int:
        actor = self._get_actor()
        import ray

        return int(ray.get(actor.size.remote()))

    async def async_size(self) -> int:
        actor = self._get_actor()
        return int(await _awaitable(actor.size.remote()))

    def is_stale(self, session_id: str, ttl_s: float) -> bool:
        actor = self._get_actor()
        import ray

        return bool(ray.get(actor.is_stale.remote(session_id=session_id, ttl_s=float(ttl_s))))

    async def async_is_stale(self, session_id: str, ttl_s: float) -> bool:
        actor = self._get_actor()
        return bool(await _awaitable(actor.is_stale.remote(session_id=session_id, ttl_s=float(ttl_s))))

    def prune(self, max_age_s: float) -> int:
        actor = self._get_actor()
        import ray

        return int(ray.get(actor.prune.remote(max_age_s=float(max_age_s))))


session_heartbeat_store = SessionHeartbeatStore()
