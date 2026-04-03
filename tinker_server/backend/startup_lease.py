"""Detached Ray startup lease for leader-only API worker responsibilities."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import otel_env_vars, preferred_control_plane_resources, preferred_control_plane_resources

logger = logging.getLogger(__name__)
_ACTOR_HANDLE = None
_PROCESS_OWNER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


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
    return os.environ.get("MINT_STARTUP_LEASE_ACTOR_NAME", "tinker_startup_lease_store")


def _lease_ttl_s() -> float:
    return float(os.environ.get("MINT_STARTUP_LEASE_TTL_S", "120"))


def _lease_poll_s() -> float:
    return max(1.0, _lease_ttl_s() / 3.0)


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
    class _StartupLeaseStore:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._leases: dict[str, dict[str, Any]] = {}

        def try_acquire(self, role: str, owner_id: str, ttl_s: float) -> dict[str, Any]:
            now = time.time()
            current = dict(self._leases.get(role, {}))
            expires_at = float(current.get("expires_at") or 0.0)
            current_owner = str(current.get("owner_id") or "")
            if not current_owner or expires_at <= now or current_owner == owner_id:
                expires_at = now + float(ttl_s)
                self._leases[role] = {
                    "owner_id": owner_id,
                    "expires_at": expires_at,
                }
                return {"owner": True, "owner_id": owner_id, "expires_at": expires_at}
            return {"owner": False, "owner_id": current_owner, "expires_at": expires_at}

        def heartbeat(self, role: str, owner_id: str, ttl_s: float) -> bool:
            now = time.time()
            current = self._leases.get(role)
            if current is None:
                return False
            if str(current.get("owner_id") or "") != owner_id:
                return False
            current["expires_at"] = now + float(ttl_s)
            return True

        def release(self, role: str, owner_id: str) -> bool:
            current = self._leases.get(role)
            if current is None:
                return False
            if str(current.get("owner_id") or "") != owner_id:
                return False
            self._leases.pop(role, None)
            return True

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    try:
        resources = preferred_control_plane_resources(
            ray.cluster_resources(),
            env_var="MINT_STARTUP_LEASE_PINNED_NODE_IP",
        )
        if resources is not None:
            options["resources"] = resources
    except Exception:
        pass
    actor_otel_env = otel_env_vars()
    from ..config import PFS_PYTHONPATH, actor_runtime_env

    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=actor_otel_env,
    )

    try:
        created = _StartupLeaseStore.options(**options).remote()
        try:
            ray.get(created.try_acquire.remote("__bootstrap__", "__bootstrap__", 0.0))
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


async def _get_actor():
    import ray

    global _ACTOR_HANDLE
    if _ACTOR_HANDLE is not None:
        return _ACTOR_HANDLE
    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")
    _ACTOR_HANDLE = await asyncio.to_thread(_get_or_create_actor)
    return _ACTOR_HANDLE


@dataclass
class StartupLease:
    role: str
    owner_id: str
    ttl_s: float
    is_owner: bool
    local_only: bool = False

    async def heartbeat(self) -> bool:
        if not self.is_owner or self.local_only:
            return self.is_owner
        try:
            actor = await _get_actor()
            return bool(await _await_ray_ref(actor.heartbeat.remote(self.role, self.owner_id, self.ttl_s)))
        except Exception as e:
            logger.warning("startup lease heartbeat failed role=%s owner_id=%s: %s", self.role, self.owner_id, e)
            return False

    async def release(self) -> bool:
        if not self.is_owner or self.local_only:
            return False
        try:
            actor = await _get_actor()
            return bool(await _await_ray_ref(actor.release.remote(self.role, self.owner_id)))
        except Exception as e:
            logger.warning("startup lease release failed role=%s owner_id=%s: %s", self.role, self.owner_id, e)
            return False

    async def heartbeat_loop(self) -> None:
        if not self.is_owner or self.local_only:
            return
        while True:
            await asyncio.sleep(_lease_poll_s())
            ok = await self.heartbeat()
            if not ok:
                logger.error("startup lease lost role=%s owner_id=%s", self.role, self.owner_id)
                return


async def acquire_startup_lease(role: str, *, ttl_s: float | None = None) -> StartupLease:
    ttl = float(ttl_s if ttl_s is not None else _lease_ttl_s())
    try:
        import ray

        if not ray.is_initialized():
            raise RuntimeError("Ray not initialized")
        actor = await _get_actor()
        out = await _await_ray_ref(actor.try_acquire.remote(role, _PROCESS_OWNER_ID, ttl))
        return StartupLease(
            role=role,
            owner_id=_PROCESS_OWNER_ID,
            ttl_s=ttl,
            is_owner=bool(isinstance(out, dict) and out.get("owner")),
            local_only=False,
        )
    except Exception as e:
        logger.warning(
            "startup lease unavailable, failing closed to follower mode for role=%s: %s",
            role,
            e,
        )
        return StartupLease(
            role=role,
            owner_id=_PROCESS_OWNER_ID,
            ttl_s=ttl,
            is_owner=False,
            local_only=False,
        )
