from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import socket
import time
import uuid
from typing import Any

from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, otel_env_vars
from ..ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator
from ..server_info import _git_sha
from .async_ray_control import _await_with_ray_get_timeout, sync_get_ray_ref

logger = logging.getLogger(__name__)
CURRENT_CODE_IDENTITY = os.environ.get("MINT_GIT_SHA") or _git_sha()
RUNTIME_CONTRACT_DIGEST_ENV = "MINT_QUEUE_SUPERVISOR_RUNTIME_CONTRACT_DIGEST"
_ACTOR_HANDLE = None


def _runtime_contract_payload() -> dict[str, Any]:
    return {
        "actor_name": _actor_name(),
        "namespace": _ray_namespace(),
        "code_identity": CURRENT_CODE_IDENTITY,
        "lease_ttl_s": _lease_ttl_s(),
    }


def _runtime_contract_digest() -> str:
    payload = _runtime_contract_payload()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _reset_cached_actor_handle() -> None:
    global _ACTOR_HANDLE
    _ACTOR_HANDLE = None


_register_ray_reconnect_invalidator(_reset_cached_actor_handle)
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


def _kill_named_actor(actor: Any) -> None:
    from . import ray_kill

    ray_kill.kill(
        actor,
        reason="queue_supervisor_runtime_contract_mismatch",
        actor_name=_actor_name(),
        namespace=_ray_namespace(),
        no_restart=True,
        verify_absent=True,
    )


def _await_ray_ref_sync(ref: Any, *, timeout_s: float | None = None) -> Any:
    return sync_get_ray_ref(ref, timeout_s=timeout_s)


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
            self._code_identity = CURRENT_CODE_IDENTITY
            self._runtime_contract_digest = os.environ.get(RUNTIME_CONTRACT_DIGEST_ENV) or _runtime_contract_digest()
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
            if self._owner_id == requested_owner and int(self._generation_id) > 0:
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
                "runtime_contract_digest": self._runtime_contract_digest,
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
    apply_detached_actor_resources(options, ray)
    extra_env = otel_env_vars()
    if CURRENT_CODE_IDENTITY:
        extra_env["MINT_GIT_SHA"] = str(CURRENT_CODE_IDENTITY)
    extra_env[RUNTIME_CONTRACT_DIGEST_ENV] = _runtime_contract_digest()
    options["runtime_env"] = actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=extra_env)

    try:
        created = _QueueSupervisorActor.options(**options).remote()
        try:
            _await_ray_ref_sync(created.snapshot.remote(), timeout_s=15.0)
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
        self._runtime_contract_verified = False
        self._owner_id = _PROCESS_INSTANCE_ID

    def _reset_cached_actor(self) -> None:
        global _ACTOR_HANDLE
        _ACTOR_HANDLE = None
        self._ray_actor = None
        self._runtime_contract_verified = False

    @staticmethod
    def _runtime_contract_matches(snapshot: dict[str, Any]) -> bool:
        if snapshot.get("code_identity") != CURRENT_CODE_IDENTITY:
            return False
        return snapshot.get("runtime_contract_digest") == _runtime_contract_digest()

    async def _ensure_runtime_contract_async(self, snapshot: dict[str, Any]) -> None:
        if self._runtime_contract_verified and self._runtime_contract_matches(snapshot):
            return
        if self._runtime_contract_matches(snapshot):
            self._runtime_contract_verified = True
            return
        actor = self._get_ray_actor()
        self._reset_cached_actor()
        await asyncio.to_thread(_kill_named_actor, actor)
        actor = self._get_ray_actor()
        refreshed = await _await_with_ray_get_timeout(_await_ray_ref(actor.snapshot.remote()), timeout_s=15.0)
        if not isinstance(refreshed, dict) or not self._runtime_contract_matches(refreshed):
            raise RuntimeError(
                "queue supervisor runtime contract mismatch after recreate: "
                f"snapshot={refreshed!r}"
            )
        self._runtime_contract_verified = True

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

    async def _get_verified_actor(self) -> Any:
        actor = self._get_ray_actor()
        snapshot = await _await_with_ray_get_timeout(_await_ray_ref(actor.snapshot.remote()), timeout_s=15.0)
        if not isinstance(snapshot, dict):
            raise TypeError(f"QueueSupervisor.snapshot returned non-dict: {type(snapshot)}")
        await self._ensure_runtime_contract_async(snapshot)
        return self._get_ray_actor()

    async def async_claim_generation(self, *, timeout_s: float = 15.0) -> dict[str, Any]:
        actor = await self._get_verified_actor()
        return await _await_with_ray_get_timeout(
            _await_ray_ref(actor.claim_generation.remote(owner_id=self._owner_id, ttl_s=_lease_ttl_s())),
            timeout_s=float(timeout_s),
        )

    async def async_heartbeat(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        actor = await self._get_verified_actor()
        return bool(
            await _await_with_ray_get_timeout(
                _await_ray_ref(
                    actor.heartbeat.remote(owner_id=self._owner_id, generation_id=int(generation_id), ttl_s=_lease_ttl_s())
                ),
                timeout_s=float(timeout_s),
            )
        )

    async def async_begin_reconcile(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        actor = await self._get_verified_actor()
        return bool(
            await _await_with_ray_get_timeout(
                _await_ray_ref(actor.begin_reconcile.remote(owner_id=self._owner_id, generation_id=int(generation_id))),
                timeout_s=float(timeout_s),
            )
        )

    async def async_finish_reconcile(self, *, generation_id: int, stale_reconciled: int, timeout_s: float = 10.0) -> bool:
        actor = await self._get_verified_actor()
        return bool(
            await _await_with_ray_get_timeout(
                _await_ray_ref(
                    actor.finish_reconcile.remote(
                        owner_id=self._owner_id,
                        generation_id=int(generation_id),
                        stale_reconciled=int(stale_reconciled),
                    )
                ),
                timeout_s=float(timeout_s),
            )
        )

    async def async_is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        actor = await self._get_verified_actor()
        return bool(
            await _await_with_ray_get_timeout(
                _await_ray_ref(actor.is_generation_current.remote(owner_id=self._owner_id, generation_id=int(generation_id))),
                timeout_s=float(timeout_s),
            )
        )

    async def async_record_fenced_worker(self, *, generation_id: int, timeout_s: float = 10.0) -> None:
        actor = await self._get_verified_actor()
        await _await_with_ray_get_timeout(
            _await_ray_ref(actor.record_fenced_worker.remote(generation_id=int(generation_id))),
            timeout_s=float(timeout_s),
        )

    async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = await self._get_verified_actor()
        out = await _await_with_ray_get_timeout(
            _await_ray_ref(actor.snapshot.remote()),
            timeout_s=float(timeout_s),
        )
        if not isinstance(out, dict):
            raise TypeError(f"QueueSupervisor.snapshot returned non-dict: {type(out)}")
        return out

    def is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
        actor = self._get_ray_actor()
        return bool(
            _await_ray_ref_sync(
                actor.is_generation_current.remote(owner_id=self._owner_id, generation_id=int(generation_id)),
                timeout_s=float(timeout_s),
            )
        )

    def snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        out = _await_ray_ref_sync(actor.snapshot.remote(), timeout_s=float(timeout_s))
        if not isinstance(out, dict):
            raise TypeError(f"QueueSupervisor.snapshot returned non-dict: {type(out)}")
        return out

    def owner_id(self) -> str:
        return self._owner_id

    def poll_s(self) -> float:
        return _await_poll_s()


queue_supervisor = QueueSupervisor()
