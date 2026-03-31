from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import socket
import time
import uuid
from typing import Any

from ..config import PFS_PYTHONPATH, actor_runtime_env, otel_env_vars
from ..server_info import _git_sha

logger = logging.getLogger(__name__)
_ACTOR_HANDLE = None
_PROCESS_INSTANCE_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _actor_name() -> str:
    return os.environ.get("MINT_QUEUE_SUPERVISOR_ACTOR_NAME", "tinker_queue_supervisor")


def _ray_namespace() -> str:
    env_ns = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _lease_ttl_s() -> float:
    return float(os.environ.get("MINT_QUEUE_SUPERVISOR_TTL_S", "30"))


def _await_poll_s() -> float:
    return max(1.0, _lease_ttl_s() / 3.0)


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

    @ray.remote(num_cpus=0, max_concurrency=128)
    class _QueueSupervisorActor:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._code_identity = _git_sha()
            self._generation_id = 0
            self._owner_id: str | None = None
            self._expires_at = 0.0
            self._state = "inactive"
            self._started_at = time.time()
            self._last_claim_at = None
            self._last_heartbeat_at = None
            self._last_reconcile_at = None
            self._stale_reconciled = 0
            self._fenced_workers = 0

        def _lease_active(self) -> bool:
            return bool(self._owner_id) and float(self._expires_at) > time.time()

        def claim_generation(self, *, owner_id: str, ttl_s: float) -> dict[str, Any]:
            now = time.time()
            requested_owner = str(owner_id)
            if self._lease_active() and self._owner_id != requested_owner:
                return self.snapshot()
            if self._lease_active() and self._owner_id == requested_owner:
                self._expires_at = now + float(ttl_s)
                self._last_claim_at = now
                return self.snapshot()
            if self._owner_id != requested_owner:
                self._generation_id += 1
            self._owner_id = requested_owner
            self._expires_at = now + float(ttl_s)
            self._state = "starting"
            self._last_claim_at = now
            return self.snapshot()

        def heartbeat(self, *, owner_id: str, generation_id: int, ttl_s: float) -> bool:
            if str(owner_id) != str(self._owner_id):
                return False
            if int(generation_id) != int(self._generation_id):
                return False
            now = time.time()
            self._expires_at = now + float(ttl_s)
            self._last_heartbeat_at = now
            return True

        def begin_reconcile(self, *, owner_id: str, generation_id: int) -> bool:
            if str(owner_id) != str(self._owner_id):
                return False
            if int(generation_id) != int(self._generation_id):
                return False
            self._state = "reconciling"
            self._last_reconcile_at = time.time()
            return True

        def finish_reconcile(self, *, owner_id: str, generation_id: int, stale_reconciled: int) -> bool:
            if str(owner_id) != str(self._owner_id):
                return False
            if int(generation_id) != int(self._generation_id):
                return False
            self._state = "active"
            self._last_reconcile_at = time.time()
            self._stale_reconciled += int(stale_reconciled)
            return True

        def record_fenced_worker(self, *, generation_id: int) -> None:
            if int(generation_id) == int(self._generation_id):
                return
            self._fenced_workers += 1

        def is_generation_current(self, *, owner_id: str, generation_id: int) -> bool:
            if not self._lease_active():
                return False
            return str(owner_id) == str(self._owner_id) and int(generation_id) == int(self._generation_id)

        def snapshot(self) -> dict[str, Any]:
            return {
                "actor_name": _actor_name(),
                "namespace": _ray_namespace(),
                "code_identity": self._code_identity,
                "generation_id": int(self._generation_id),
                "owner_id": self._owner_id,
                "expires_at": float(self._expires_at),
                "state": self._state,
                "started_at": float(self._started_at),
                "last_claim_at": self._last_claim_at,
                "last_heartbeat_at": self._last_heartbeat_at,
                "last_reconcile_at": self._last_reconcile_at,
                "stale_reconciled": int(self._stale_reconciled),
                "fenced_workers": int(self._fenced_workers),
            }

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
    options["runtime_env"] = actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=otel_env_vars())

    try:
        created = _QueueSupervisorActor.options(**options).remote()
        try:
            ray.get(created.snapshot.remote())
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


class QueueSupervisor:
    def __init__(self) -> None:
        self._ray_actor = None
        self._owner_id = _PROCESS_INSTANCE_ID

    def _get_ray_actor(self):
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

    async def async_claim_generation(self, *, timeout_s: float = 15.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        return await asyncio.wait_for(
            _await_ray_ref(actor.claim_generation.remote(owner_id=self._owner_id, ttl_s=_lease_ttl_s())),
            timeout=float(timeout_s),
        )

    async def async_heartbeat(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        actor = self._get_ray_actor()
        return bool(
            await asyncio.wait_for(
                _await_ray_ref(
                    actor.heartbeat.remote(owner_id=self._owner_id, generation_id=int(generation_id), ttl_s=_lease_ttl_s())
                ),
                timeout=float(timeout_s),
            )
        )

    async def async_begin_reconcile(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        actor = self._get_ray_actor()
        return bool(
            await asyncio.wait_for(
                _await_ray_ref(actor.begin_reconcile.remote(owner_id=self._owner_id, generation_id=int(generation_id))),
                timeout=float(timeout_s),
            )
        )

    async def async_finish_reconcile(self, *, generation_id: int, stale_reconciled: int, timeout_s: float = 10.0) -> bool:
        actor = self._get_ray_actor()
        return bool(
            await asyncio.wait_for(
                _await_ray_ref(
                    actor.finish_reconcile.remote(
                        owner_id=self._owner_id,
                        generation_id=int(generation_id),
                        stale_reconciled=int(stale_reconciled),
                    )
                ),
                timeout=float(timeout_s),
            )
        )

    async def async_is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        actor = self._get_ray_actor()
        return bool(
            await asyncio.wait_for(
                _await_ray_ref(actor.is_generation_current.remote(owner_id=self._owner_id, generation_id=int(generation_id))),
                timeout=float(timeout_s),
            )
        )

    async def async_record_fenced_worker(self, *, generation_id: int, timeout_s: float = 10.0) -> None:
        actor = self._get_ray_actor()
        await asyncio.wait_for(
            _await_ray_ref(actor.record_fenced_worker.remote(generation_id=int(generation_id))),
            timeout=float(timeout_s),
        )

    async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        return await asyncio.wait_for(_await_ray_ref(actor.snapshot.remote()), timeout=float(timeout_s))

    def is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        import ray

        actor = self._get_ray_actor()
        return bool(
            ray.get(
                actor.is_generation_current.remote(owner_id=self._owner_id, generation_id=int(generation_id)),
                timeout=float(timeout_s),
            )
        )

    def snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        import ray

        actor = self._get_ray_actor()
        out = ray.get(actor.snapshot.remote(), timeout=float(timeout_s))
        if not isinstance(out, dict):
            raise TypeError(f"QueueSupervisor.snapshot returned non-dict: {type(out)}")
        return out

    def owner_id(self) -> str:
        return self._owner_id

    def poll_s(self) -> float:
        return _await_poll_s()


queue_supervisor = QueueSupervisor()
