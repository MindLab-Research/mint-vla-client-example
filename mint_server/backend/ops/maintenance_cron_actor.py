from __future__ import annotations

import asyncio
import concurrent.futures
import structlog
import os
import time
import uuid
from typing import Any

from mint_server.config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, otel_env_vars, TIER_CPU
from mint_server.ray.runtime_env import env_nonempty
from mint_server.checkpoints.checkpoints import (
    get_checkpoint_mirror_poll_s,
    get_checkpoint_reap_interval_s,
    process_pending_checkpoint_mirrors,
    reap_runtime_checkpoints,
)
from mint_server.ray.ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator
from mint_server.server_info import _git_sha
from mint_server.backend.ray_cluster.async_ray_control import _await_with_ray_get_timeout, sync_get_ray_ref

CURRENT_CODE_IDENTITY = os.environ.get("MINT_GIT_SHA") or _git_sha()

logger = structlog.get_logger(__name__)
_ACTOR_HANDLE = None


def _looks_like_dead_actor_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc)
    return (
        name in {"ActorDiedError", "ActorUnavailableError", "RayActorError"}
        or "actor died" in text.lower()
        or "actor is dead" in text.lower()
        or "actor is temporarily unavailable" in text.lower()
        or "Failed to look up actor" in text
    )

def _reset_cached_actor_handle() -> None:
    global _ACTOR_HANDLE
    _ACTOR_HANDLE = None


_register_ray_reconnect_invalidator(_reset_cached_actor_handle)

_LOOP_FUTURE_REAPER = "future_reaper"
_LOOP_CHECKPOINT_REAPER = "checkpoint_reaper"
_LOOP_CHECKPOINT_MIRROR = "checkpoint_mirror"
_LOOP_TRAINING_CLEANUP = "training_cleanup"
_LOOP_SAMPLING_CLEANUP = "sampling_cleanup"
_LOOP_BILLING_OUTBOX = "billing_outbox"


def _actor_name() -> str:
    return os.environ.get("MINT_MAINTENANCE_CRON_ACTOR_NAME", "mint_maintenance_cron")


def _ray_namespace() -> str:
    env_ns = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from mint_server.config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _future_reap_interval_s() -> float:
    return float(os.environ.get("MINT_MAINTENANCE_REAP_INTERVAL_S", "5.0"))


def _billing_outbox_interval_s() -> float:
    return float(os.environ.get("MINT_BILLING_OUTBOX_FLUSH_INTERVAL_S", "5.0"))


def run_future_reaper_once() -> dict[str, Any]:
    from mint_server.backend.stores.task_state_store import task_futures

    reaped = asyncio.run(task_futures.async_reap())
    return {
        "expired": list(reaped.get("expired", [])),
        "timed_out": list(reaped.get("timed_out", [])),
        "payload_evicted": list(reaped.get("payload_evicted", [])),
        "staged_payload_gc_deleted": list(reaped.get("staged_payload_gc_deleted", [])),
        "tombstones_deleted": list(reaped.get("tombstones_deleted", [])),
        "payload_evict_errors": list(reaped.get("payload_evict_errors", [])),
        "staged_payload_gc_errors": list(reaped.get("staged_payload_gc_errors", [])),
    }


def run_checkpoint_reaper_once() -> dict[str, Any]:
    return reap_runtime_checkpoints()


def run_checkpoint_mirror_once() -> dict[str, Any]:
    return process_pending_checkpoint_mirrors()


def run_training_cleanup_once() -> dict[str, Any]:
    stale_after_s = float(os.environ.get("MINT_TRAINING_HEARTBEAT_STALE_S", "300"))
    if stale_after_s <= 0:
        return {"cleaned": []}

    from mint_server.backend.training.training_cleanup_executor import cleanup_stale_training_sessions_once_impl

    import asyncio

    cleaned = asyncio.run(cleanup_stale_training_sessions_once_impl(stale_after_s=stale_after_s))
    return {"cleaned": list(cleaned)}


def run_sampling_cleanup_once() -> dict[str, Any]:
    from mint_server.backend.inference.sampling_cleanup_executor import cleanup_stale_sampling_sessions_once_impl

    import asyncio

    cleaned = asyncio.run(cleanup_stale_sampling_sessions_once_impl())
    return {"cleaned": list(cleaned)}


async def async_run_billing_outbox_once() -> dict[str, Any]:
    from mint_server.backend.stores.task_state_store import task_futures

    limit = int(os.environ.get("MINT_BILLING_OUTBOX_FLUSH_BATCH_SIZE", "100"))
    lease_ttl_s = float(os.environ.get("MINT_BILLING_OUTBOX_CLAIM_TTL_S", "60"))
    return await task_futures.async_flush_billing_outbox(
        limit=limit,
        lease_ttl_s=lease_ttl_s,
    )


def run_billing_outbox_once() -> dict[str, Any]:
    return asyncio.run(async_run_billing_outbox_once())


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
    import mint_server.backend.ray_cluster.ray_kill as ray_kill

    ray_kill.kill(
        actor,
        reason="maintenance_cron_actor_code_mismatch",
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

    @ray.remote(num_cpus=0, max_concurrency=64)
    class _MaintenanceCronActor:
        def __init__(self) -> None:
            from mint_server.observability.logging_context import init_actor_observability

            init_actor_observability()
            self._epoch_id = uuid.uuid4().hex
            self._started_at = time.time()
            self._code_identity = CURRENT_CODE_IDENTITY
            self._tasks: dict[str, asyncio.Task] = {}
            self._loop_state: dict[str, dict[str, Any]] = {}
            self._loop_specs: dict[str, dict[str, Any]] = {
                _LOOP_FUTURE_REAPER: {
                    "interval_s": _future_reap_interval_s(),
                    "run_immediately": False,
                    "runner": run_future_reaper_once,
                },
                _LOOP_CHECKPOINT_REAPER: {
                    "interval_s": float(get_checkpoint_reap_interval_s()),
                    "run_immediately": False,
                    "runner": run_checkpoint_reaper_once,
                },
                _LOOP_CHECKPOINT_MIRROR: {
                    "interval_s": float(get_checkpoint_mirror_poll_s()),
                    "run_immediately": True,
                    "runner": run_checkpoint_mirror_once,
                },
                _LOOP_TRAINING_CLEANUP: {
                    "interval_s": _future_reap_interval_s(),
                    "run_immediately": False,
                    "runner": run_training_cleanup_once,
                },
                _LOOP_SAMPLING_CLEANUP: {
                    "interval_s": _future_reap_interval_s(),
                    "run_immediately": False,
                    "runner": run_sampling_cleanup_once,
                },
                _LOOP_BILLING_OUTBOX: {
                    "interval_s": _billing_outbox_interval_s(),
                    "run_immediately": False,
                    "runner": async_run_billing_outbox_once,
                    "async_runner": True,
                },
            }
            for name, spec in self._loop_specs.items():
                self._loop_state[name] = {
                    "enabled": True,
                    "running": False,
                    "interval_s": float(spec["interval_s"]),
                    "last_tick_at": None,
                    "last_success_at": None,
                    "last_error_at": None,
                    "last_error": None,
                    "last_error_type": None,
                    "success_count": 0,
                    "error_count": 0,
                    "last_result": None,
                }

        async def _run_loop_once(self, loop_name: str) -> dict[str, Any]:
            state = self._loop_state[loop_name]
            runner = self._loop_specs[loop_name]["runner"]
            state["running"] = True
            state["last_tick_at"] = time.time()
            try:
                if bool(self._loop_specs[loop_name].get("async_runner")):
                    result = await runner()
                else:
                    result = await asyncio.to_thread(runner)
                state["last_success_at"] = time.time()
                state["success_count"] = int(state["success_count"]) + 1
                state["last_result"] = result
                state["last_error"] = None
                state["last_error_type"] = None
                return result if isinstance(result, dict) else {"result": result}
            except Exception as e:
                state["last_error_at"] = time.time()
                state["last_error"] = f"{type(e).__name__}: {e}"
                state["last_error_type"] = type(e).__name__
                state["error_count"] = int(state["error_count"]) + 1
                logger.exception(
                    "maintenance_cron_actor loop failed loop=%s error_type=%s error=%s",
                    loop_name,
                    type(e).__name__,
                    str(e),
                )
                return {
                    "error": state["last_error"],
                    "error_type": state["last_error_type"],
                }
            finally:
                state["running"] = False

        async def _loop_task(self, loop_name: str) -> None:
            spec = self._loop_specs[loop_name]
            interval_s = float(spec["interval_s"])
            run_immediately = bool(spec.get("run_immediately"))
            if run_immediately:
                await self._run_loop_once(loop_name)
            while True:
                await asyncio.sleep(interval_s)
                if not bool(self._loop_state[loop_name].get("enabled", True)):
                    continue
                await self._run_loop_once(loop_name)

        async def ensure_started(self) -> dict[str, Any]:
            for loop_name in self._loop_specs:
                task = self._tasks.get(loop_name)
                if task is None or task.done():
                    self._tasks[loop_name] = asyncio.create_task(self._loop_task(loop_name))
            return self.health_snapshot()

        def health_snapshot(self) -> dict[str, Any]:
            return {
                "actor_name": _actor_name(),
                "namespace": _ray_namespace(),
                "epoch_id": self._epoch_id,
                "started_at": self._started_at,
                "code_identity": self._code_identity,
                "loops": {name: dict(state) for name, state in self._loop_state.items()},
            }

        async def async_health_snapshot(self) -> dict[str, Any]:
            return self.health_snapshot()

        async def run_once(self, loop_name: str) -> dict[str, Any]:
            if loop_name not in self._loop_specs:
                raise KeyError(loop_name)
            return await self._run_loop_once(loop_name)

        async def shutdown(self) -> bool:
            for name, task in list(self._tasks.items()):
                task.cancel()
                self._tasks.pop(name, None)
            return True

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    apply_detached_actor_resources(options, ray)
    extra_env = otel_env_vars()
    if CURRENT_CODE_IDENTITY:
        extra_env["MINT_GIT_SHA"] = str(CURRENT_CODE_IDENTITY)
    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=extra_env,
        include_ray_attach_hints=False,
        tier=TIER_CPU,
    )

    try:
        created = _MaintenanceCronActor.options(**options).remote()
        try:
            _await_ray_ref_sync(created.health_snapshot.remote(), timeout_s=15.0)
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


class MaintenanceCronActor:
    def __init__(self) -> None:
        self._ray_actor = None

    def _reset_cached_actor(self) -> None:
        global _ACTOR_HANDLE
        _ACTOR_HANDLE = None
        self._ray_actor = None

    def _get_ray_actor(self, *, create_if_missing: bool = True):
        import ray

        global _ACTOR_HANDLE
        if self._ray_actor is not None:
            return self._ray_actor
        if _ACTOR_HANDLE is not None:
            self._ray_actor = _ACTOR_HANDLE
            return self._ray_actor
        if not ray.is_initialized():
            raise RuntimeError("Ray not initialized")
        if not create_if_missing:
            name = _actor_name()
            namespace = _ray_namespace()
            try:
                self._ray_actor = ray.get_actor(name, namespace=namespace)
            except Exception as e:
                raise RuntimeError(
                    f"MaintenanceCronActor unavailable actor_name={name!r} namespace={namespace!r}"
                ) from e
            _ACTOR_HANDLE = self._ray_actor
            return self._ray_actor
        self._ray_actor = _get_or_create_actor()
        return self._ray_actor

    def _ensure_code_identity_sync(self, snapshot: dict[str, Any]) -> None:
        actor_code_identity = snapshot.get("code_identity")
        if actor_code_identity == CURRENT_CODE_IDENTITY:
            return
        actor = self._get_ray_actor()
        self._reset_cached_actor()
        _kill_named_actor(actor)
        actor = self._get_ray_actor()

        refreshed = _await_ray_ref_sync(actor.health_snapshot.remote(), timeout_s=15.0)
        if refreshed.get("code_identity") != CURRENT_CODE_IDENTITY:
            raise RuntimeError(
                "maintenance cron actor code identity mismatch after recreate: "
                f"expected={CURRENT_CODE_IDENTITY!r} actual={refreshed.get('code_identity')!r}"
            )

    async def _ensure_code_identity_async(self, snapshot: dict[str, Any]) -> None:
        actor_code_identity = snapshot.get("code_identity")
        if actor_code_identity == CURRENT_CODE_IDENTITY:
            return
        actor = self._get_ray_actor()
        self._reset_cached_actor()
        await asyncio.to_thread(_kill_named_actor, actor)
        actor = self._get_ray_actor()
        refreshed = await _await_with_ray_get_timeout(
            _await_ray_ref(actor.async_health_snapshot.remote()),
            timeout_s=15.0,
        )
        if refreshed.get("code_identity") != CURRENT_CODE_IDENTITY:
            raise RuntimeError(
                "maintenance cron actor code identity mismatch after recreate: "
                f"expected={CURRENT_CODE_IDENTITY!r} actual={refreshed.get('code_identity')!r}"
            )

    async def async_ensure_started(self, *, timeout_s: float = 15.0) -> dict[str, Any]:
        try:
            actor = self._get_ray_actor()
            snapshot = await _await_with_ray_get_timeout(
                _await_ray_ref(actor.async_health_snapshot.remote()),
                timeout_s=float(timeout_s),
            )
        except Exception as e:
            if not _looks_like_dead_actor_error(e):
                raise
            logger.warning(
                "maintenance cron actor handle stale; recreating actor_name=%r namespace=%r error_type=%s error=%s",
                _actor_name(),
                _ray_namespace(),
                type(e).__name__,
                e,
            )
            self._reset_cached_actor()
            actor = self._get_ray_actor()
            snapshot = await _await_with_ray_get_timeout(
                _await_ray_ref(actor.async_health_snapshot.remote()),
                timeout_s=float(timeout_s),
            )
        await self._ensure_code_identity_async(snapshot)
        actor = self._get_ray_actor()
        return await _await_with_ray_get_timeout(
            _await_ray_ref(actor.ensure_started.remote()),
            timeout_s=float(timeout_s),
        )

    def ensure_started(self, *, timeout_s: float = 15.0) -> dict[str, Any]:
        try:
            actor = self._get_ray_actor()
            snapshot = _await_ray_ref_sync(actor.health_snapshot.remote(), timeout_s=float(timeout_s))
        except Exception as e:
            if not _looks_like_dead_actor_error(e):
                raise
            logger.warning(
                "maintenance cron actor handle stale; recreating actor_name=%r namespace=%r error_type=%s error=%s",
                _actor_name(),
                _ray_namespace(),
                type(e).__name__,
                e,
            )
            self._reset_cached_actor()
            actor = self._get_ray_actor()
            snapshot = _await_ray_ref_sync(actor.health_snapshot.remote(), timeout_s=float(timeout_s))
        self._ensure_code_identity_sync(snapshot)
        actor = self._get_ray_actor()
        return _await_ray_ref_sync(actor.ensure_started.remote(), timeout_s=float(timeout_s))

    async def async_health_snapshot(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = True,
    ) -> dict[str, Any]:
        try:
            actor = self._get_ray_actor(create_if_missing=create_if_missing)
            snapshot = await _await_with_ray_get_timeout(
                _await_ray_ref(actor.async_health_snapshot.remote()),
                timeout_s=float(timeout_s),
            )
        except Exception as e:
            if not _looks_like_dead_actor_error(e):
                raise
            logger.warning(
                "maintenance cron actor health handle stale; recreating actor_name=%r namespace=%r error_type=%s error=%s",
                _actor_name(),
                _ray_namespace(),
                type(e).__name__,
                e,
            )
            self._reset_cached_actor()
            actor = self._get_ray_actor(create_if_missing=create_if_missing)
            snapshot = await _await_with_ray_get_timeout(
                _await_ray_ref(actor.async_health_snapshot.remote()),
                timeout_s=float(timeout_s),
            )
        if not create_if_missing:
            return snapshot
        await self._ensure_code_identity_async(snapshot)
        actor = self._get_ray_actor(create_if_missing=create_if_missing)
        return await _await_with_ray_get_timeout(
            _await_ray_ref(actor.async_health_snapshot.remote()),
            timeout_s=float(timeout_s),
        )

    def health_snapshot(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = True,
    ) -> dict[str, Any]:
        try:
            actor = self._get_ray_actor(create_if_missing=create_if_missing)
            snapshot = _await_ray_ref_sync(actor.health_snapshot.remote(), timeout_s=float(timeout_s))
        except Exception as e:
            if not _looks_like_dead_actor_error(e):
                raise
            logger.warning(
                "maintenance cron actor health handle stale; recreating actor_name=%r namespace=%r error_type=%s error=%s",
                _actor_name(),
                _ray_namespace(),
                type(e).__name__,
                e,
            )
            self._reset_cached_actor()
            actor = self._get_ray_actor(create_if_missing=create_if_missing)
            snapshot = _await_ray_ref_sync(actor.health_snapshot.remote(), timeout_s=float(timeout_s))
        if not create_if_missing:
            return snapshot
        self._ensure_code_identity_sync(snapshot)
        actor = self._get_ray_actor(create_if_missing=create_if_missing)
        return _await_ray_ref_sync(actor.health_snapshot.remote(), timeout_s=float(timeout_s))

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        snapshot = await self.async_health_snapshot(timeout_s=timeout_s, create_if_missing=False)
        return {
            "ok": True,
            "actor_name": snapshot.get("actor_name"),
            "namespace": snapshot.get("namespace"),
            "epoch_id": snapshot.get("epoch_id"),
        }

    async def async_run_once(self, loop_name: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        return await _await_with_ray_get_timeout(
            _await_ray_ref(actor.run_once.remote(str(loop_name))),
            timeout_s=float(timeout_s),
        )


maintenance_cron_actor = MaintenanceCronActor()
