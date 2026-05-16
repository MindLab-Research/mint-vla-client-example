from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

from tinker_server import app as app_module


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _install_fake_ray(monkeypatch) -> None:
    try:
        import ray  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    ray_module = types.ModuleType("ray")
    ray_exceptions = types.ModuleType("ray.exceptions")
    ray_util_module = types.ModuleType("ray.util")
    ray_sched_module = types.ModuleType("ray.util.scheduling_strategies")
    ray_private_module = types.ModuleType("ray._private")
    ray_private_state_module = types.ModuleType("ray._private.state")

    class _RayActorError(Exception):
        pass

    class _GetTimeoutError(Exception):
        pass

    class _NodeAffinitySchedulingStrategy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    ray_exceptions.RayActorError = _RayActorError
    ray_exceptions.GetTimeoutError = _GetTimeoutError
    ray_sched_module.NodeAffinitySchedulingStrategy = _NodeAffinitySchedulingStrategy

    ray_private_state_module.available_resources_per_node = lambda: {}
    ray_private_state_module.actors = lambda *_args, **_kwargs: {}
    ray_private_module.state = ray_private_state_module

    ray_util_module.list_named_actors = lambda *args, **kwargs: []
    ray_util_module.get_placement_group = lambda *args, **kwargs: None
    ray_util_module.remove_placement_group = lambda *args, **kwargs: None
    ray_util_module.placement_group_table = lambda *args, **kwargs: {}

    ray_module.actor = SimpleNamespace(ActorHandle=object)
    ray_module.exceptions = ray_exceptions
    ray_module.util = ray_util_module
    ray_module._private = ray_private_module
    ray_module.init = lambda *args, **kwargs: None
    ray_module.is_initialized = lambda: False
    ray_module.get_actor = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("actor not found"))
    ray_module.get = lambda *args, **kwargs: None
    ray_module.nodes = lambda: []
    ray_module.kill = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.exceptions", ray_exceptions)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", ray_sched_module)
    monkeypatch.setitem(sys.modules, "ray._private", ray_private_module)
    monkeypatch.setitem(sys.modules, "ray._private.state", ray_private_state_module)


class _StubFutureStore:
    def __init__(self, *, fail_async_ensure_ready: bool = False):
        self.fail_async_ensure_ready = bool(fail_async_ensure_ready)
        self.async_ensure_started_calls = 0
        self.async_ensure_ready_calls = 0

    def ensure_ready(self, **_kwargs) -> None:
        return None

    async def async_ensure_started(self) -> None:
        self.async_ensure_started_calls += 1
        return None

    async def async_ensure_ready(self, **_kwargs) -> None:
        self.async_ensure_ready_calls += 1
        if self.fail_async_ensure_ready:
            raise RuntimeError("future store stats probe failed")
        return None


class _StubModelWorkScheduler:
    def __init__(self, *, fail_stats: str | None = None):
        self.fail_stats = fail_stats
        self.stats_calls: list[dict[str, float]] = []

    async def stats(self, *, timeout_s: float = 10.0) -> dict:
        self.stats_calls.append({"timeout_s": float(timeout_s)})
        if self.fail_stats is not None:
            raise RuntimeError(self.fail_stats)
        return {
            "depth": 0,
            "backlog_depth": 0,
            "backlog_depth_by_domain": {},
            "replica_queues": {},
            "leases": [],
            "counters": {},
        }


class _StubInitRayCalls(list):
    def __call__(self, *args, **kwargs):
        self.append({"args": args, "kwargs": kwargs})
        return None


class _StubUsageStore:
    async def health_check(self) -> bool:
        return True


class _StubCapacityManager:
    def ensure_ready(self) -> None:
        return None

    async def async_ensure_ready(self, *, timeout_s: float = 10.0):
        return {"capacity": 1, "inflight": 0, "timeout_s": float(timeout_s)}


class _StubSessionManager:
    def __init__(self, **_kwargs):
        self.multi_model_manager = None
        self.start_cleanup_task_calls = 0

    async def start_cleanup_task(self) -> None:
        self.start_cleanup_task_calls += 1
        return None

    def set_multi_model_manager(self, manager) -> None:
        self.multi_model_manager = manager

    def restore_sampling_session(self, _info) -> bool:
        return True

    async def shutdown_all(self) -> None:
        return None


class _StubTrainingManager:
    def __init__(self, *_args, **_kwargs):
        return None

    async def start_cleanup_task(self, _engine) -> None:
        return None

    async def shutdown_all(self, _engine) -> None:
        return None


class _StubTrainingEngine:
    async def initialize(self) -> None:
        return None


class _StubApiWorkQueue:
    def __init__(
        self,
        *,
        fail_async_ensure_ready: bool = False,
        fail_wait_until_execution_ready: bool = False,
    ):
        self.fail_async_ensure_ready = bool(fail_async_ensure_ready)
        self.fail_wait_until_execution_ready = bool(fail_wait_until_execution_ready)
        self.started_workers = 0
        self.async_ensure_started_calls = 0
        self.async_ensure_ready_calls = 0
        self.wait_until_execution_ready_calls: list[float] = []

    def ensure_ready(self) -> None:
        return None

    async def async_ensure_started(self) -> None:
        self.async_ensure_started_calls += 1
        return None

    async def async_ensure_ready(self, *, timeout_s: float = 10.0):
        self.async_ensure_ready_calls += 1
        if self.fail_async_ensure_ready:
            raise RuntimeError("api work queue stats probe failed")
        return {"depth": 0, "enqueued": 0, "dequeued": 0, "timeout_s": float(timeout_s)}
    def set_executor(self, _op: str, _executor) -> None:
        return None

    async def start_workers(self, num_workers: int) -> None:
        self.started_workers += int(num_workers)

    async def wait_until_execution_ready(self, *, timeout_s: float = 120.0) -> bool:
        self.wait_until_execution_ready_calls.append(float(timeout_s))
        if self.fail_wait_until_execution_ready:
            raise RuntimeError("local queue runtime not ready")
        return True

    async def shutdown(self) -> None:
        return None


class _StubOwnerRuntimeSupervisor:
    def __init__(self):
        self.started = 0
        self.run_once_calls: list[str] = []

    async def async_ensure_started(self, *, timeout_s: float = 15.0):
        self.started += 1
        return {
            "actor_name": "tinker_owner_runtime_supervisor",
            "epoch_id": "epoch-1",
            "timeout_s": float(timeout_s),
        }

    async def async_health_snapshot(self, *, timeout_s: float = 10.0):
        return {
            "actor_name": "tinker_owner_runtime_supervisor",
            "epoch_id": "epoch-1",
            "timeout_s": float(timeout_s),
        }

    async def async_run_once(self, loop_name: str, *, timeout_s: float = 30.0):
        self.run_once_calls.append(str(loop_name))
        return {"ok": True, "loop_name": str(loop_name), "timeout_s": float(timeout_s)}


class _StubQueueExecutionRuntime:
    def __init__(self, *, fail_async_ensure_started: str | None = None):
        self.fail_async_ensure_started = fail_async_ensure_started
        self.ensure_started_calls: list[dict[str, float | int]] = []

    async def async_ensure_started(self, *, num_workers: int, timeout_s: float = 120.0):
        self.ensure_started_calls.append(
            {
                "num_workers": int(num_workers),
                "timeout_s": float(timeout_s),
            }
        )
        if self.fail_async_ensure_started is not None:
            raise RuntimeError(self.fail_async_ensure_started)
        return {
            "actor_name": "tinker_queue_execution_runtime",
            "desired_workers": int(num_workers),
            "timeout_s": float(timeout_s),
        }


class _StubModelActorSupervisor:
    def __init__(self, *, desired_total: int = 0):
        self.desired_total = int(desired_total)
        self.reconcile_calls = 0

    def snapshot(self):
        return {"desired_total": self.desired_total}

    async def reconcile_once(self):
        self.reconcile_calls += 1
        return {"ok": True}


class _TrackedMultiModelManager:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    async def shutdown_all(self) -> None:
        self.shutdown_calls += 1


async def _noop_async(*_args, **_kwargs) -> None:
    return None


class _StubStartupLease:
    def __init__(self, *, is_owner: bool, local_only: bool = False):
        self.role = "test-startup-owner"
        self.owner_id = "owner-1"
        self.ttl_s = 60.0
        self.is_owner = bool(is_owner)
        self.local_only = bool(local_only)
        self.released = False
        self.heartbeat_started = False

    async def heartbeat_loop(self) -> None:
        self.heartbeat_started = True
        await asyncio.Future()

    async def release(self) -> bool:
        self.released = True
        return True


def _install_lifespan_stubs(
    monkeypatch,
    queue: _StubApiWorkQueue,
    owner_runtime: _StubOwnerRuntimeSupervisor,
    queue_execution_runtime: _StubQueueExecutionRuntime,
    init_ray_calls: _StubInitRayCalls | None = None,
    task_state_futures: _StubFutureStore | None = None,
    model_actor_supervisor: _StubModelActorSupervisor | None = None,
    model_work_scheduler: _StubModelWorkScheduler | None = None,
) -> None:
    monkeypatch.setattr(app_module, "_cleanup_stale_actors", _noop_async)
    monkeypatch.setattr(app_module, "_restore_sampling_sessions", _noop_async)
    monkeypatch.setattr(app_module, "SessionManager", _StubSessionManager)
    monkeypatch.setattr(app_module.config, "enable_multi_lora", False)
    monkeypatch.setattr(app_module.config, "api_work_queue_num_workers", 1)

    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")
    config_actor_module = importlib.import_module("tinker_server.backend.config_actor")
    task_state_store_module = importlib.import_module("tinker_server.backend.task_state_store")
    capacity_manager_module = importlib.import_module("tinker_server.backend.capacity_manager")
    gateway_session_store_module = importlib.import_module("tinker_server.backend.gateway_session_store")
    sampling_session_store_module = importlib.import_module("tinker_server.backend.sampling_session_store")
    session_heartbeat_store_module = importlib.import_module("tinker_server.backend.session_heartbeat_store")
    session_index_store_module = importlib.import_module("tinker_server.backend.session_index_store")
    training_session_manager_module = importlib.import_module("tinker_server.backend.training_session_manager")
    training_session_store_module = importlib.import_module("tinker_server.backend.training_session_store")
    future_replay_module = importlib.import_module("tinker_server.backend.future_replay")
    dense_session_state_module = importlib.import_module("tinker_server.backend.dense_session_state")
    checkpoints_module = importlib.import_module("tinker_server.checkpoints")
    gateway_module = importlib.import_module("tinker_server.gateway")
    usage_store_module = importlib.import_module("tinker_server.usage_store")
    owner_runtime_module = importlib.import_module("tinker_server.backend.owner_runtime_supervisor")
    queue_execution_runtime_module = importlib.import_module("tinker_server.backend.queue_execution_runtime")
    model_actor_supervisor_module = importlib.import_module("tinker_server.backend.model_actor_supervisor")
    model_work_scheduler_module = importlib.import_module("tinker_server.backend.model_work_scheduler")

    verl_training_module = types.ModuleType("tinker_server.backend.verl_training")
    verl_training_module.VerlTrainingEngine = _StubTrainingEngine
    monkeypatch.setitem(sys.modules, "tinker_server.backend.verl_training", verl_training_module)

    if init_ray_calls is not None:
        monkeypatch.setattr(app_module, "init_ray", init_ray_calls)
    else:
        monkeypatch.setattr(app_module, "init_ray", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_work_queue_module, "api_work_queue", queue)
    monkeypatch.setattr(config_actor_module, "async_ensure_started", _noop_async)
    monkeypatch.setattr(owner_runtime_module, "owner_runtime_supervisor", owner_runtime)
    monkeypatch.setattr(queue_execution_runtime_module, "queue_execution_runtime", queue_execution_runtime)
    monkeypatch.setattr(
        model_actor_supervisor_module,
        "model_actor_supervisor",
        model_actor_supervisor or _StubModelActorSupervisor(),
    )
    monkeypatch.setattr(
        model_work_scheduler_module,
        "model_work_scheduler",
        model_work_scheduler or _StubModelWorkScheduler(),
    )
    monkeypatch.setattr(capacity_manager_module, "capacity_manager", _StubCapacityManager())
    monkeypatch.setattr(task_state_store_module, "task_state_futures", task_state_futures or _StubFutureStore())
    monkeypatch.setattr(gateway_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(sampling_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(session_heartbeat_store_module, "session_heartbeat_store", SimpleNamespace(ensure_ready=lambda: None, async_size=lambda: 0))
    monkeypatch.setattr(session_index_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(training_session_manager_module, "TrainingSessionManager", _StubTrainingManager)
    monkeypatch.setattr(training_session_store_module, "ensure_ready", lambda: None)
    async def _async_list_training_sessions():
        return []

    monkeypatch.setattr(training_session_store_module, "async_list_training_sessions", _async_list_training_sessions)
    monkeypatch.setattr(future_replay_module, "ensure_future_replay_sweeper", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dense_session_state_module,
        "cleanup_legacy_dense_session_state_once",
        lambda *args, **kwargs: {"migrated": [], "deleted": [], "skipped": [], "errors": []},
    )
    monkeypatch.setattr(checkpoints_module, "get_checkpoint_reap_interval_s", lambda: 3600.0)
    monkeypatch.setattr(checkpoints_module, "get_checkpoint_mirror_poll_s", lambda: 3600.0)
    monkeypatch.setattr(checkpoints_module, "reap_runtime_checkpoints", lambda: {})
    monkeypatch.setattr(
        checkpoints_module,
        "process_pending_checkpoint_mirrors",
        lambda: {"mirrored": [], "failed": []},
    )
    async def _get_usage_store():
        return _StubUsageStore()

    monkeypatch.setattr(gateway_module, "close_http_clients", _noop_async)
    monkeypatch.setattr(usage_store_module, "get_usage_store", _get_usage_store)
    monkeypatch.setattr(usage_store_module, "close_usage_store", _noop_async)


def test_lifespan_surfaces_model_work_scheduler_start_failure(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    model_work_scheduler = _StubModelWorkScheduler(fail_stats="scheduler unavailable")
    _install_lifespan_stubs(
        monkeypatch,
        queue,
        owner_runtime,
        queue_execution_runtime,
        model_work_scheduler=model_work_scheduler,
    )
    lease = _StubStartupLease(is_owner=True)

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="scheduler unavailable"):
            async with app_module.lifespan(app_module.app):
                raise AssertionError("lifespan should not yield on scheduler startup failure")

    asyncio.run(_run())
    assert queue.started_workers == 0
    assert owner_runtime.started == 1
    assert lease.released is True
    assert app_module.service.session_manager is None
    assert queue_execution_runtime.ensure_started_calls == []
    assert model_work_scheduler.stats_calls == [{"timeout_s": 10.0}]


def test_lifespan_starts_model_work_scheduler_without_queue_runtime(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    model_work_scheduler = _StubModelWorkScheduler()
    _install_lifespan_stubs(
        monkeypatch,
        queue,
        owner_runtime,
        queue_execution_runtime,
        model_work_scheduler=model_work_scheduler,
    )
    lease = _StubStartupLease(is_owner=True)

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(_run())
    assert queue.started_workers == 0
    assert owner_runtime.started == 1
    assert lease.released is True
    assert app_module.service.session_manager is None
    assert queue_execution_runtime.ensure_started_calls == []
    assert queue.started_workers == 0
    assert model_work_scheduler.stats_calls == [{"timeout_s": 10.0}]


def test_lifespan_reconciles_model_actor_supervisor_when_desired(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    model_actor_supervisor = _StubModelActorSupervisor(desired_total=1)
    _install_lifespan_stubs(
        monkeypatch,
        queue,
        owner_runtime,
        queue_execution_runtime,
        model_actor_supervisor=model_actor_supervisor,
    )
    lease = _StubStartupLease(is_owner=True)

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(_run())
    assert model_actor_supervisor.reconcile_calls == 1


@pytest.mark.anyio
async def test_initialize_execution_runtime_runs_prewarm_after_bindings(monkeypatch) -> None:
    runtime_module = importlib.import_module("tinker_server.backend.queue_execution_runtime")
    calls: list[tuple[object, object]] = []
    train_engine = object()
    multi_model_manager = object()

    async def _fake_initialize_execution_bindings():
        return {
            "inference_manager": object(),
            "train_manager": object(),
            "train_engine": train_engine,
            "multi_model_manager": multi_model_manager,
        }

    async def _fake_prewarm(train_engine_arg, multi_model_manager_arg) -> None:
        calls.append((train_engine_arg, multi_model_manager_arg))

    monkeypatch.setattr(runtime_module, "_initialize_execution_bindings", _fake_initialize_execution_bindings)
    monkeypatch.setattr(
        "tinker_server.backend.persistent_prewarm.prewarm_persistent_models",
        _fake_prewarm,
    )

    bindings = await runtime_module._initialize_execution_runtime(prewarm=True)

    assert bindings["train_engine"] is train_engine
    assert bindings["multi_model_manager"] is multi_model_manager
    assert calls == [(train_engine, multi_model_manager)]


def test_lifespan_follower_skips_leader_only_startup(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    init_ray_calls = _StubInitRayCalls()
    _install_lifespan_stubs(monkeypatch, queue, owner_runtime, queue_execution_runtime, init_ray_calls)

    calls: list[str] = []
    lease = _StubStartupLease(is_owner=False)

    async def _count_cleanup(*_args, **_kwargs) -> None:
        calls.append("cleanup")

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(app_module, "_cleanup_stale_actors", _count_cleanup)
    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(_run())

    assert calls == []
    assert queue.started_workers == 0
    assert owner_runtime.started == 1
    assert lease.heartbeat_started is False
    assert lease.released is True
    assert app_module.service.session_manager is None
    assert queue_execution_runtime.ensure_started_calls == []
    assert len(init_ray_calls) == 1
    assert init_ray_calls[0]["kwargs"]["namespace"] == "tinker"


def test_lifespan_owner_runtime_local_only_uses_async_cleanup_helper(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    _install_lifespan_stubs(monkeypatch, queue, owner_runtime, queue_execution_runtime)
    lease = _StubStartupLease(is_owner=True)
    cleanup_calls: list[str] = []

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    async def _count_cleanup(*_args, **_kwargs) -> None:
        cleanup_calls.append("cleanup")

    monkeypatch.setenv("MINT_OWNER_RUNTIME_SUPERVISOR_LOCAL_ONLY", "1")
    monkeypatch.setattr(app_module, "_cleanup_stale_actors", _count_cleanup)
    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(_run())

    assert cleanup_calls == ["cleanup"]
    assert owner_runtime.started == 0
    assert lease.heartbeat_started is True
    assert lease.released is True
    assert queue_execution_runtime.ensure_started_calls == []


def test_lifespan_init_ray_when_head_address_path_configured(monkeypatch, tmp_path: Path) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    init_ray_calls = _StubInitRayCalls()
    _install_lifespan_stubs(monkeypatch, queue, owner_runtime, queue_execution_runtime, init_ray_calls)

    lease = _StubStartupLease(is_owner=False)
    head_address = tmp_path / "ray-head.txt"
    head_address.write_text("192.168.90.10\n", encoding="utf-8")
    monkeypatch.setenv("MINT_RAY_HEAD_ADDRESS_PATH", str(head_address))
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_CLIENT_ADDRESS", raising=False)

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(_run())

    assert len(init_ray_calls) == 1
    assert init_ray_calls[0]["kwargs"]["namespace"] == "tinker"


def test_lifespan_uses_started_probe_for_task_state_futures(monkeypatch) -> None:
    queue = _StubApiWorkQueue(fail_async_ensure_ready=True)
    task_state_futures = _StubFutureStore(fail_async_ensure_ready=True)
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    _install_lifespan_stubs(
        monkeypatch,
        queue,
        owner_runtime,
        queue_execution_runtime,
        task_state_futures=task_state_futures,
    )
    lease = _StubStartupLease(is_owner=True)

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(_run())

    assert task_state_futures.async_ensure_started_calls == 1
    assert task_state_futures.async_ensure_ready_calls == 0
    assert queue.async_ensure_started_calls == 0
    assert queue.async_ensure_ready_calls == 0
    assert queue_execution_runtime.ensure_started_calls == []
    assert lease.released is True


@pytest.mark.anyio
async def test_acquire_startup_lease_fails_closed_to_follower(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    startup_lease_module = importlib.import_module("tinker_server.backend.startup_lease")
    fake_ray = importlib.import_module("ray")

    monkeypatch.setattr(fake_ray, "is_initialized", lambda: True)

    async def _fail_get_actor():
        raise RuntimeError("lease backend unavailable")

    monkeypatch.setattr(startup_lease_module, "_get_actor", _fail_get_actor)

    lease = await startup_lease_module.acquire_startup_lease("test-role")

    assert lease.role == "test-role"
    assert lease.is_owner is False
    assert lease.local_only is False


def test_startup_lease_creation_prefers_explicit_pinned_node(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    startup_lease_module = importlib.import_module("tinker_server.backend.startup_lease")
    config_module = importlib.import_module("tinker_server.config")
    fake_ray = importlib.import_module("ray")
    created_options = {}

    class _FakeRemoteCall:
        def remote(self, *_args, **_kwargs):
            return object()

    class _FakeActorHandle:
        def __init__(self) -> None:
            self.try_acquire = _FakeRemoteCall()

    class _FakeRemoteBuilder:
        def remote(self):
            return _FakeActorHandle()

    class _FakeRemoteClass:
        @staticmethod
        def options(**kwargs):
            created_options.update(kwargs)
            return _FakeRemoteBuilder()

    def _fake_remote(**kwargs):
        assert kwargs == {"num_cpus": 0}

        def _decorator(_cls):
            return _FakeRemoteClass

        return _decorator

    monkeypatch.setattr(fake_ray, "remote", _fake_remote, raising=False)
    monkeypatch.setattr(
        fake_ray,
        "get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("actor not found")),
        raising=False,
    )
    monkeypatch.setattr(
        fake_ray,
        "cluster_resources",
        lambda: {"node:__internal_head__": 1.0},
        raising=False,
    )
    monkeypatch.setattr(fake_ray, "get", lambda *_args, **_kwargs: {"owner": True}, raising=False)
    monkeypatch.setattr(config_module, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(config_module, "PFS_TINKER_PATH", "/tmp/tinker-root")
    monkeypatch.setattr(config_module, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setattr(config_module, "PFS_PYTHONPATH", "/tmp/runtime-root/site-packages:/tmp/tinker-root")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.setenv("MINT_STARTUP_LEASE_PINNED_NODE_IP", "192.168.38.176")
    monkeypatch.setattr(startup_lease_module, "_ACTOR_HANDLE", None)

    startup_lease_module._get_or_create_actor()

    assert created_options["resources"] == {"node:192.168.38.176": 0.001}


def test_startup_lease_creation_defaults_to_head_pin(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    startup_lease_module = importlib.import_module("tinker_server.backend.startup_lease")
    config_module = importlib.import_module("tinker_server.config")
    fake_ray = importlib.import_module("ray")
    created_options = {}

    class _FakeRemoteCall:
        def remote(self, *_args, **_kwargs):
            return object()

    class _FakeActorHandle:
        def __init__(self) -> None:
            self.try_acquire = _FakeRemoteCall()

    class _FakeRemoteBuilder:
        def remote(self):
            return _FakeActorHandle()

    class _FakeRemoteClass:
        @staticmethod
        def options(**kwargs):
            created_options.update(kwargs)
            return _FakeRemoteBuilder()

    def _fake_remote(**kwargs):
        assert kwargs == {"num_cpus": 0}

        def _decorator(_cls):
            return _FakeRemoteClass

        return _decorator

    monkeypatch.setattr(fake_ray, "remote", _fake_remote, raising=False)
    monkeypatch.setattr(
        fake_ray,
        "get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("actor not found")),
        raising=False,
    )
    monkeypatch.setattr(
        fake_ray,
        "cluster_resources",
        lambda: {"node:__internal_head__": 1.0},
        raising=False,
    )
    monkeypatch.setattr(fake_ray, "get", lambda *_args, **_kwargs: {"owner": True}, raising=False)
    monkeypatch.setattr(config_module, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(config_module, "PFS_TINKER_PATH", "/tmp/tinker-root")
    monkeypatch.setattr(config_module, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setattr(config_module, "PFS_PYTHONPATH", "/tmp/runtime-root/site-packages:/tmp/tinker-root")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")
    monkeypatch.delenv("MINT_STARTUP_LEASE_PINNED_NODE_IP", raising=False)
    monkeypatch.setattr(startup_lease_module, "_ACTOR_HANDLE", None)

    startup_lease_module._get_or_create_actor()

    assert created_options["resources"] == {"node:__internal_head__": 0.001}


def test_lifespan_keeps_training_route_globals_unbound_in_stateless_api(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    _install_lifespan_stubs(monkeypatch, queue, owner_runtime, queue_execution_runtime)
    lease = _StubStartupLease(is_owner=True)

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    from tinker_server.routes import mint as mint_routes
    from tinker_server.routes import training as training_routes
    from tinker_server.routes import weights as weights_routes

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            assert training_routes.training_manager is None
            assert mint_routes.training_manager is None
            assert weights_routes.training_manager is None

    asyncio.run(_run())

    assert queue.started_workers == 0
    assert owner_runtime.started == 1
    assert lease.released is True
    assert queue_execution_runtime.ensure_started_calls == []


def test_lifespan_skips_tokenizer_preload_for_multi_worker_startup(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()
    _install_lifespan_stubs(monkeypatch, queue, owner_runtime, queue_execution_runtime)
    lease = _StubStartupLease(is_owner=True)
    preload_calls = []

    async def _acquire_startup_lease(*_args, **_kwargs):
        return lease

    monkeypatch.setenv("MINT_UVICORN_WORKERS", "8")
    monkeypatch.setattr(app_module.openai_compat, "preload_supported_tokenizers", lambda: preload_calls.append(True) or {})
    monkeypatch.setattr(
        "tinker_server.backend.startup_lease.acquire_startup_lease",
        _acquire_startup_lease,
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            return None

    asyncio.run(_run())

    assert preload_calls == []
    assert queue_execution_runtime.ensure_started_calls == []


@pytest.mark.anyio
async def test_prewarm_raises_when_training_prewarm_unavailable_in_stateless_api(monkeypatch) -> None:
    from tinker_server.backend import persistent_prewarm

    monkeypatch.setattr(
        persistent_prewarm.config,
        "prewarm_persistent_models_csv",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    monkeypatch.setattr(persistent_prewarm.config, "prewarm_enable_training", True)
    monkeypatch.setattr(persistent_prewarm.config, "prewarm_enable_inference", False)

    with pytest.raises(
        RuntimeError,
        match="persistent prewarm training configured but unavailable in the execution runtime",
    ):
        await persistent_prewarm.prewarm_persistent_models(None, SimpleNamespace())


@pytest.mark.anyio
async def test_prewarm_raises_when_inference_prewarm_fails(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")
    from tinker_server.backend import persistent_prewarm

    monkeypatch.setattr(
        resource_pool_module,
        "get_resource_pool",
        lambda: SimpleNamespace(
            set_protected=lambda *_args, **_kwargs: True,
            mark_ready=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        persistent_prewarm.config,
        "prewarm_persistent_models_csv",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    monkeypatch.setattr(persistent_prewarm.config, "prewarm_train_lora_rank", 16)
    monkeypatch.setattr(persistent_prewarm.config, "prewarm_train_lr", 5e-5)
    monkeypatch.setattr(persistent_prewarm.config, "prewarm_megatron_ready_timeout_s", 1.0)
    monkeypatch.setattr(persistent_prewarm.config, "prewarm_enable_training", False)
    monkeypatch.setattr(persistent_prewarm.config, "prewarm_enable_inference", True)

    class _FailingManager:
        async def get_engine(self, _model_name: str):
            raise RuntimeError("pinned worker full")

    with pytest.raises(RuntimeError, match="pinned worker full"):
        await persistent_prewarm.prewarm_persistent_models(SimpleNamespace(), _FailingManager())


@pytest.mark.anyio
async def test_cleanup_stale_actors_registers_openpi_shared_actor(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    from tinker_server.backend import multi_lora_engine

    ray = sys.modules["ray"]
    actor_name = "openpi_shared_runtime_deadbeef"
    actor = SimpleNamespace(
        __ray_ready__=SimpleNamespace(remote=lambda: "ready-ref"),
        describe=SimpleNamespace(remote=lambda: "describe-ref"),
    )
    registered: dict[str, object] = {}

    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.get_actor = lambda name, namespace=None: actor  # type: ignore[attr-defined]
    ray.util.list_named_actors = lambda all_namespaces=True: [  # type: ignore[attr-defined]
        {"name": actor_name, "namespace": multi_lora_engine.PERSISTENT_NAMESPACE}
    ]

    def _fake_ray_get(ref, *args, **kwargs):
        _ = args, kwargs
        if ref == "ready-ref":
            return None
        if ref == "describe-ref":
            return {
                "pool_key": {
                    "base_model": "openpi/pi0-fast-libero-low-mem-finetune",
                    "worker_module": "tinker_server.backend.openpi_fast_worker",
                },
                "actor_id": "actor-123",
                "node_id": "node-456",
                "node_ip": "192.168.0.8",
                "pid": 999,
                "cuda_visible_devices": "0",
                "current_session_id": "session-a",
            }
        raise AssertionError(f"unexpected ray.get ref: {ref!r}")

    ray.get = _fake_ray_get  # type: ignore[attr-defined]

    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")

    class _FakePool:
        def register(self, **kwargs):
            registered["register"] = kwargs

        def mark_ready(self, actor_name):
            registered["mark_ready"] = actor_name

    monkeypatch.setattr(resource_pool_module, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(app_module.config, "skip_actor_cleanup", False)

    await app_module._cleanup_stale_actors()

    register = registered["register"]
    assert register["actor_name"] == actor_name
    assert register["actor_type"].value == "openpi"
    assert register["num_gpus"] == 1
    assert register["base_model"] == "openpi/pi0-fast-libero-low-mem-finetune"
    assert register["session_id"] == "session-a"
    assert register["node_id"] == "node-456"
    assert register["metadata"]["pool_key"]["worker_module"] == "tinker_server.backend.openpi_fast_worker"
    assert register["metadata"]["actor_id"] == "actor-123"
    assert register["metadata"]["node_ip"] == "192.168.0.8"
    assert register["metadata"]["pid"] == 999
    assert register["metadata"]["cuda_visible_devices"] == "0"
    assert registered["mark_ready"] == actor_name


@pytest.mark.anyio
async def test_get_engine_skips_capacity_check_when_named_actor_exists(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    mle = importlib.import_module("tinker_server.backend.multi_lora_engine")

    actor_handle = SimpleNamespace(
        __ray_ready__=SimpleNamespace(remote=lambda: "ready-ref"),
        is_engine_ready=SimpleNamespace(remote=lambda: "engine-ready-ref"),
    )

    if not hasattr(mle, "ray"):
        mle.ray = SimpleNamespace(exceptions=SimpleNamespace(RayActorError=RuntimeError))

    monkeypatch.setattr(mle.ray, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(mle.ray, "get_actor", lambda *args, **kwargs: actor_handle, raising=False)

    async def _async_get_ray_ref(ref, *, timeout_s=None):
        _ = timeout_s
        return True if ref == "engine-ready-ref" else None

    monkeypatch.setattr(mle, "async_get_ray_ref", _async_get_ray_ref, raising=False)
    monkeypatch.setattr(
        mle,
        "parse_model_gpu_placement",
        lambda **_kwargs: SimpleNamespace(
            total_gpus=1,
            slices=(SimpleNamespace(replica=0, worker_index=1, gpu_count=1, node_ip="192.168.38.4"),),
        ),
        raising=False,
    )

    def _fail_capacity_check(**_kwargs):
        raise AssertionError("capacity check should be skipped when a named actor already exists")

    monkeypatch.setattr(mle, "assert_node_ip_capacity", _fail_capacity_check, raising=False)

    async def _fake_initialize(self) -> None:
        self._initialized = True
        self.server = actor_handle

    monkeypatch.setattr(mle.MultiLoRAInferenceEngine, "initialize", _fake_initialize)

    manager = mle.MultiModelInferenceManager()
    engine = await manager.get_engine("Qwen/Qwen3-0.6B")

    assert engine.server is actor_handle


@pytest.mark.anyio
async def test_get_engine_checks_capacity_when_named_actor_probe_fails(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    mle = importlib.import_module("tinker_server.backend.multi_lora_engine")

    actor_handle = SimpleNamespace(
        __ray_ready__=SimpleNamespace(remote=lambda: "ready-ref"),
        is_engine_ready=SimpleNamespace(remote=lambda: "engine-ready-ref"),
    )

    if not hasattr(mle, "ray"):
        mle.ray = SimpleNamespace(exceptions=SimpleNamespace(RayActorError=RuntimeError))

    monkeypatch.setattr(mle.ray, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(mle.ray, "get_actor", lambda *args, **kwargs: actor_handle, raising=False)

    async def _stale_async_get_ray_ref(ref, *, timeout_s=None):
        _ = timeout_s
        raise mle.ray.exceptions.RayActorError(f"stale actor during probe: {ref}")

    monkeypatch.setattr(mle, "async_get_ray_ref", _stale_async_get_ray_ref, raising=False)
    monkeypatch.setattr(
        mle,
        "parse_model_gpu_placement",
        lambda **_kwargs: SimpleNamespace(
            total_gpus=1,
            slices=(SimpleNamespace(replica=0, worker_index=1, gpu_count=1, node_ip="192.168.38.4"),),
        ),
        raising=False,
    )

    def _capacity_check(**_kwargs):
        raise RuntimeError("capacity check ran")

    monkeypatch.setattr(mle, "assert_node_ip_capacity", _capacity_check, raising=False)

    manager = mle.MultiModelInferenceManager()

    with pytest.raises(RuntimeError, match="capacity check ran"):
        await manager.get_engine("Qwen/Qwen3-0.6B")


@pytest.mark.anyio
async def test_issue_489_cleanup_stale_actors_uses_relaxed_ready_timeout(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    import importlib

    ray = importlib.import_module("ray")
    actor_reconciliation_module = importlib.import_module("tinker_server.backend.actor_reconciliation")
    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")
    multi_lora_engine_module = importlib.import_module("tinker_server.backend.multi_lora_engine")

    timeout_calls: list[float] = []
    register_calls: list[dict] = []
    mark_ready_calls: list[str] = []

    actor_handle = SimpleNamespace(
        __ray_ready__=SimpleNamespace(remote=lambda: "ready-ref"),
    )

    monkeypatch.delenv("MINT_STARTUP_RECONCILE_READY_TIMEOUT_S", raising=False)
    monkeypatch.setattr(app_module.config, "skip_actor_cleanup", False)
    monkeypatch.setattr(actor_reconciliation_module, "init_ray", lambda **_kwargs: None)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray.util, "list_named_actors", lambda **_kwargs: [
        {"name": "megatron_qwen3_30b_a3b_instruct_2507", "namespace": multi_lora_engine_module.PERSISTENT_NAMESPACE}
    ])
    monkeypatch.setattr(ray, "get_actor", lambda *_args, **_kwargs: actor_handle)

    def fake_get(_ref, timeout=None, **_kwargs):
        timeout_calls.append(timeout)
        raise ray.exceptions.GetTimeoutError("busy")

    monkeypatch.setattr(ray, "get", fake_get)
    monkeypatch.setattr(ray.util, "get_placement_group", lambda _name: object())
    monkeypatch.setattr(ray.util, "placement_group_table", lambda _pg: {"bundles": {"0": {"GPU": 4}}})

    class _FakePool:
        def register(self, **kwargs):
            register_calls.append(kwargs)

        def mark_ready(self, actor_name):
            mark_ready_calls.append(actor_name)

        def unregister(self, _name):
            return None

    monkeypatch.setattr(resource_pool_module, "get_resource_pool", lambda: _FakePool())

    await app_module._cleanup_stale_actors()

    assert timeout_calls == [5.0]
    assert len(register_calls) == 1
    assert register_calls[0]["metadata"] == {"startup_reconcile": "__ray_ready__timeout"}
    assert mark_ready_calls == []
