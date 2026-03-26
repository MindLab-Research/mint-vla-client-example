from __future__ import annotations

import asyncio
import importlib
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
    def ensure_ready(self) -> None:
        return None


class _StubSessionManager:
    def __init__(self, **_kwargs):
        self.multi_model_manager = None

    async def start_cleanup_task(self) -> None:
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
    def __init__(self):
        self.started_workers = 0

    def set_executor(self, _op: str, _executor) -> None:
        return None

    async def start_workers(self, num_workers: int) -> None:
        self.started_workers += int(num_workers)

    async def shutdown(self) -> None:
        return None


async def _noop_async(*_args, **_kwargs) -> None:
    return None


def _install_lifespan_stubs(monkeypatch, queue: _StubApiWorkQueue) -> None:
    monkeypatch.setattr(app_module, "_cleanup_stale_actors", _noop_async)
    monkeypatch.setattr(app_module, "_restore_sampling_sessions", _noop_async)
    monkeypatch.setattr(app_module, "SessionManager", _StubSessionManager)
    monkeypatch.setattr(app_module.config, "enable_multi_lora", False)
    monkeypatch.setattr(app_module.config, "api_work_queue_num_workers", 1)

    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")
    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    gateway_session_store_module = importlib.import_module("tinker_server.backend.gateway_session_store")
    sampling_session_store_module = importlib.import_module("tinker_server.backend.sampling_session_store")
    session_index_store_module = importlib.import_module("tinker_server.backend.session_index_store")
    training_session_manager_module = importlib.import_module("tinker_server.backend.training_session_manager")
    training_session_store_module = importlib.import_module("tinker_server.backend.training_session_store")
    checkpoints_module = importlib.import_module("tinker_server.checkpoints")
    gateway_module = importlib.import_module("tinker_server.gateway")
    usage_store_module = importlib.import_module("tinker_server.usage_store")

    verl_training_module = types.ModuleType("tinker_server.backend.verl_training")
    verl_training_module.VerlTrainingEngine = _StubTrainingEngine
    monkeypatch.setitem(sys.modules, "tinker_server.backend.verl_training", verl_training_module)

    monkeypatch.setattr(api_work_queue_module, "api_work_queue", queue)
    monkeypatch.setattr(future_store_module, "future_store", _StubFutureStore())
    monkeypatch.setattr(gateway_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(sampling_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(session_index_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(training_session_manager_module, "TrainingSessionManager", _StubTrainingManager)
    monkeypatch.setattr(training_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(checkpoints_module, "get_checkpoint_reap_interval_s", lambda: 3600.0)
    monkeypatch.setattr(checkpoints_module, "get_checkpoint_mirror_poll_s", lambda: 3600.0)
    monkeypatch.setattr(checkpoints_module, "reap_runtime_checkpoints", lambda: {})
    monkeypatch.setattr(
        checkpoints_module,
        "process_pending_checkpoint_mirrors",
        lambda: {"mirrored": [], "failed": []},
    )
    monkeypatch.setattr(gateway_module, "close_http_clients", _noop_async)
    monkeypatch.setattr(usage_store_module, "close_usage_store", _noop_async)


def test_lifespan_waits_for_prewarm_and_fails_startup(monkeypatch) -> None:
    queue = _StubApiWorkQueue()
    _install_lifespan_stubs(monkeypatch, queue)

    async def _fail_prewarm(*_args, **_kwargs) -> None:
        raise RuntimeError("prewarm failed: pinned worker full")

    monkeypatch.setattr(app_module, "_prewarm_persistent_models", _fail_prewarm)

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="prewarm failed: pinned worker full"):
            async with app_module.lifespan(app_module.app):
                raise AssertionError("lifespan should not yield on prewarm failure")

    asyncio.run(_run())
    assert queue.started_workers == 0


@pytest.mark.anyio
async def test_prewarm_raises_when_inference_prewarm_fails(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    resource_pool_module = importlib.import_module("tinker_server.backend.resource_pool")

    monkeypatch.setattr(
        resource_pool_module,
        "get_resource_pool",
        lambda: SimpleNamespace(
            set_protected=lambda *_args, **_kwargs: True,
            mark_ready=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(app_module.config, "prewarm_persistent_models_csv", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    monkeypatch.setattr(app_module.config, "prewarm_train_lora_rank", 16)
    monkeypatch.setattr(app_module.config, "prewarm_train_lr", 5e-5)
    monkeypatch.setattr(app_module.config, "prewarm_megatron_ready_timeout_s", 1.0)
    monkeypatch.setattr(app_module.config, "prewarm_enable_training", False)
    monkeypatch.setattr(app_module.config, "prewarm_enable_inference", True)

    class _FailingManager:
        async def get_engine(self, _model_name: str):
            raise RuntimeError("pinned worker full")

    with pytest.raises(RuntimeError, match="pinned worker full"):
        await app_module._prewarm_persistent_models(SimpleNamespace(), _FailingManager())


@pytest.mark.anyio
async def test_get_engine_skips_capacity_check_when_named_actor_exists(monkeypatch) -> None:
    _install_fake_ray(monkeypatch)
    from tinker_server.backend import multi_lora_engine as mle

    actor_handle = SimpleNamespace(
        __ray_ready__=SimpleNamespace(remote=lambda: "ready-ref"),
        is_engine_ready=SimpleNamespace(remote=lambda: "engine-ready-ref"),
    )

    monkeypatch.setattr(mle.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(mle.ray, "get_actor", lambda *args, **kwargs: actor_handle)
    monkeypatch.setattr(
        mle.ray,
        "get",
        lambda ref, *args, **kwargs: True if ref == "engine-ready-ref" else None,
    )
    monkeypatch.setattr(mle, "parse_model_single_node_ip", lambda **_kwargs: "192.168.38.4")
    monkeypatch.setattr(mle, "parse_model_node_ip_list", lambda **_kwargs: ["192.168.38.4"])

    def _fail_capacity_check(**_kwargs):
        raise AssertionError("capacity check should be skipped when a named actor already exists")

    monkeypatch.setattr(mle, "assert_node_ip_capacity", _fail_capacity_check)

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
    from tinker_server.backend import multi_lora_engine as mle

    actor_handle = SimpleNamespace(
        __ray_ready__=SimpleNamespace(remote=lambda: "ready-ref"),
        is_engine_ready=SimpleNamespace(remote=lambda: "engine-ready-ref"),
    )

    monkeypatch.setattr(mle.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(mle.ray, "get_actor", lambda *args, **kwargs: actor_handle)

    def _stale_actor_get(ref, *args, **kwargs):
        raise mle.ray.exceptions.RayActorError(f"stale actor during probe: {ref}")

    monkeypatch.setattr(mle.ray, "get", _stale_actor_get)
    monkeypatch.setattr(mle, "parse_model_single_node_ip", lambda **_kwargs: "192.168.38.4")
    monkeypatch.setattr(mle, "parse_model_node_ip_list", lambda **_kwargs: ["192.168.38.4"])

    def _capacity_check(**_kwargs):
        raise RuntimeError("capacity check ran")

    monkeypatch.setattr(mle, "assert_node_ip_capacity", _capacity_check)

    manager = mle.MultiModelInferenceManager()

    with pytest.raises(RuntimeError, match="capacity check ran"):
        await manager.get_engine("Qwen/Qwen3-0.6B")
