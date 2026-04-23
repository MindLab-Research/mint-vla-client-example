from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
import uuid
from typing import Any

from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, otel_env_vars
from ..checkpoints import (
    get_checkpoint_mirror_poll_s,
    get_checkpoint_reap_interval_s,
    process_pending_checkpoint_mirrors,
    reap_runtime_checkpoints,
)
from ..server_info import _git_sha

CURRENT_CODE_IDENTITY = os.environ.get("MINT_GIT_SHA") or _git_sha()

logger = logging.getLogger(__name__)
_ACTOR_HANDLE = None

def _reset_cached_actor_handle() -> None:
    global _ACTOR_HANDLE
    _ACTOR_HANDLE = None

from ..ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator
_register_ray_reconnect_invalidator(_reset_cached_actor_handle)

_LOOP_FUTURE_REAPER = "future_reaper"
_LOOP_CHECKPOINT_REAPER = "checkpoint_reaper"
_LOOP_CHECKPOINT_MIRROR = "checkpoint_mirror"
_LOOP_ACTOR_RECONCILIATION = "actor_reconciliation"
_LOOP_TRAINING_CLEANUP = "training_cleanup"
_LOOP_SAMPLING_CLEANUP = "sampling_cleanup"


def _actor_name() -> str:
    return os.environ.get("MINT_OWNER_RUNTIME_SUPERVISOR_ACTOR_NAME", "tinker_owner_runtime_supervisor")


def _ray_namespace() -> str:
    env_ns = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _future_reap_interval_s() -> float:
    from ..config import config as server_config

    return float(server_config.api_work_queue_reap_interval_s)


def run_future_reaper_once() -> dict[str, Any]:
    from .capacity_manager import capacity_manager
    from .future_store import future_store

    asyncio.run(future_store.async_ensure_started())
    reaped = asyncio.run(future_store.async_reap())
    released: list[str] = []
    for rid in list(reaped.get("expired", [])) + list(reaped.get("timed_out", [])):
        asyncio.run(capacity_manager.async_release_all(str(rid)))
        released.append(str(rid))
    return {
        "expired": list(reaped.get("expired", [])),
        "timed_out": list(reaped.get("timed_out", [])),
        "released": released,
    }


def run_checkpoint_reaper_once() -> dict[str, Any]:
    return reap_runtime_checkpoints()


def run_checkpoint_mirror_once() -> dict[str, Any]:
    return process_pending_checkpoint_mirrors()


def _actor_reconcile_interval_s() -> float:
    return float(os.environ.get("MINT_ACTOR_RECONCILE_INTERVAL_S", "60"))


def run_actor_reconciliation_once() -> dict[str, Any]:
    from .actor_reconciliation import cleanup_stale_actors_once

    import asyncio

    return asyncio.run(cleanup_stale_actors_once())


def run_training_cleanup_once() -> dict[str, Any]:
    from .training_cleanup_executor import training_cleanup_executor

    import asyncio

    cleaned = asyncio.run(training_cleanup_executor.async_cleanup_stale_sessions_once())
    return {"cleaned": list(cleaned)}


def run_sampling_cleanup_once() -> dict[str, Any]:
    from .sampling_cleanup_executor import sampling_cleanup_executor

    import asyncio

    cleaned = asyncio.run(sampling_cleanup_executor.async_cleanup_stale_sessions_once())
    return {"cleaned": list(cleaned)}


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
        reason="owner_runtime_supervisor_code_mismatch",
        actor_name=_actor_name(),
        namespace=_ray_namespace(),
        no_restart=True,
        verify_absent=True,
    )


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
    class _OwnerRuntimeSupervisorActor:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

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
                _LOOP_ACTOR_RECONCILIATION: {
                    "interval_s": _actor_reconcile_interval_s(),
                    "run_immediately": False,
                    "runner": run_actor_reconciliation_once,
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
                result = await asyncio.to_thread(runner)
                state["last_success_at"] = time.time()
                state["success_count"] = int(state["success_count"]) + 1
                state["last_result"] = result
                state["last_error"] = None
                return result if isinstance(result, dict) else {"result": result}
            except Exception as e:
                state["last_error_at"] = time.time()
                state["last_error"] = f"{type(e).__name__}: {e}"
                state["error_count"] = int(state["error_count"]) + 1
                logger.exception("owner_runtime_supervisor loop failed loop=%s", loop_name)
                return {"error": state["last_error"]}
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
    )

    try:
        created = _OwnerRuntimeSupervisorActor.options(**options).remote()
        _ACTOR_HANDLE = created
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


class OwnerRuntimeSupervisor:
    def __init__(self) -> None:
        self._ray_actor = None

    def _reset_cached_actor(self) -> None:
        global _ACTOR_HANDLE
        _ACTOR_HANDLE = None
        self._ray_actor = None

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

    def _ensure_code_identity_sync(self, snapshot: dict[str, Any]) -> None:
        actor_code_identity = snapshot.get("code_identity")
        if actor_code_identity == CURRENT_CODE_IDENTITY:
            return
        actor = self._get_ray_actor()
        self._reset_cached_actor()
        _kill_named_actor(actor)
        actor = self._get_ray_actor()
        import ray

        refreshed = ray.get(actor.health_snapshot.remote(), timeout=15.0)
        if refreshed.get("code_identity") != CURRENT_CODE_IDENTITY:
            raise RuntimeError(
                "owner runtime supervisor code identity mismatch after recreate: "
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
        refreshed = await asyncio.wait_for(_await_ray_ref(actor.async_health_snapshot.remote()), timeout=15.0)
        if refreshed.get("code_identity") != CURRENT_CODE_IDENTITY:
            raise RuntimeError(
                "owner runtime supervisor code identity mismatch after recreate: "
                f"expected={CURRENT_CODE_IDENTITY!r} actual={refreshed.get('code_identity')!r}"
            )

    async def async_ensure_started(self, *, timeout_s: float = 15.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        snapshot = await asyncio.wait_for(_await_ray_ref(actor.async_health_snapshot.remote()), timeout=float(timeout_s))
        await self._ensure_code_identity_async(snapshot)
        actor = self._get_ray_actor()
        return await asyncio.wait_for(_await_ray_ref(actor.ensure_started.remote()), timeout=float(timeout_s))

    def ensure_started(self, *, timeout_s: float = 15.0) -> dict[str, Any]:
        import ray

        actor = self._get_ray_actor()
        snapshot = ray.get(actor.health_snapshot.remote(), timeout=float(timeout_s))
        self._ensure_code_identity_sync(snapshot)
        actor = self._get_ray_actor()
        return ray.get(actor.ensure_started.remote(), timeout=float(timeout_s))

    async def async_health_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        snapshot = await asyncio.wait_for(_await_ray_ref(actor.async_health_snapshot.remote()), timeout=float(timeout_s))
        await self._ensure_code_identity_async(snapshot)
        actor = self._get_ray_actor()
        return await asyncio.wait_for(_await_ray_ref(actor.async_health_snapshot.remote()), timeout=float(timeout_s))

    def health_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        import ray

        actor = self._get_ray_actor()
        snapshot = ray.get(actor.health_snapshot.remote(), timeout=float(timeout_s))
        self._ensure_code_identity_sync(snapshot)
        actor = self._get_ray_actor()
        return ray.get(actor.health_snapshot.remote(), timeout=float(timeout_s))

    async def async_run_once(self, loop_name: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        return await asyncio.wait_for(_await_ray_ref(actor.run_once.remote(str(loop_name))), timeout=float(timeout_s))


owner_runtime_supervisor = OwnerRuntimeSupervisor()
