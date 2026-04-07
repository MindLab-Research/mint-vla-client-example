import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

from tinker_server import app as app_module
from tinker_server.models.types import ModelInput, SampleRequest, SamplingParams


class _StubFutureStore:
    def ensure_ready(self) -> None:
        return None

    async def async_ensure_started(self) -> None:
        return None

    async def async_ensure_ready(self) -> None:
        return None


class _StubCapacityManager:
    def ensure_ready(self) -> None:
        return None

    async def async_ensure_ready(self) -> None:
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
        self._executors: dict[str, object] = {}

    def ensure_ready(self) -> None:
        return None

    async def async_ensure_started(self) -> None:
        return None

    async def async_ensure_ready(self) -> None:
        return None

    def set_executor(self, op: str, executor) -> None:
        self._executors[str(op)] = executor

    async def start_workers(self, num_workers: int) -> None:
        self.started_workers = int(num_workers)

    async def shutdown(self) -> None:
        return None


class _StubOwnerRuntimeSupervisor:
    async def async_ensure_started(self, *, timeout_s: float = 15.0):
        return {
            "actor_name": "tinker_owner_runtime_supervisor",
            "epoch_id": "epoch-1",
            "timeout_s": float(timeout_s),
        }

    async def async_health_snapshot(self, *, timeout_s: float = 10.0):
        return {
            "actor_name": "tinker_owner_runtime_supervisor",
            "epoch_id": "epoch-1",
            "code_identity": app_module._git_sha(),
            "loops": {},
            "timeout_s": float(timeout_s),
        }

    async def async_run_once(self, loop_name: str, *, timeout_s: float = 30.0):
        return {"ok": True, "loop_name": str(loop_name), "timeout_s": float(timeout_s)}


class _StubQueueExecutionRuntime:
    async def async_ensure_started(self, *, num_workers: int, timeout_s: float = 120.0):
        from tinker_server.backend.api_work_queue import api_work_queue
        from tinker_server.backend.api_work_queue_dispatch import register_api_work_queue_executors

        register_api_work_queue_executors(api_work_queue)
        return {
            "actor_name": "tinker_queue_execution_runtime",
            "desired_workers": int(num_workers),
            "timeout_s": float(timeout_s),
        }


def test_sampling_queue_executor_forwards_gateway_auth(monkeypatch):
    captured: dict[str, object] = {}
    queue = _StubApiWorkQueue()
    owner_runtime = _StubOwnerRuntimeSupervisor()
    queue_execution_runtime = _StubQueueExecutionRuntime()

    async def _noop_async(*_args, **_kwargs) -> None:
        return None

    async def _capture_do_sample(request_id, req, user_id, gateway_auth=None) -> None:
        captured["request_id"] = request_id
        captured["sampling_session_id"] = req.sampling_session_id
        captured["user_id"] = user_id
        captured["gateway_auth"] = gateway_auth

    monkeypatch.setattr(app_module, "_cleanup_stale_actors", _noop_async)
    monkeypatch.setattr(app_module, "_prewarm_persistent_models", _noop_async)
    monkeypatch.setattr(app_module, "_restore_sampling_sessions", _noop_async)
    monkeypatch.setattr(app_module, "SessionManager", _StubSessionManager)
    monkeypatch.setattr(app_module, "_should_preload_openai_tokenizers", lambda: False)
    monkeypatch.setattr(app_module.sampling, "_do_sample", _capture_do_sample)
    monkeypatch.setattr(app_module.config, "enable_multi_lora", False)
    monkeypatch.setattr(app_module.config, "api_work_queue_num_workers", 1)

    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")
    capacity_manager_module = importlib.import_module("tinker_server.backend.capacity_manager")
    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    gateway_session_store_module = importlib.import_module("tinker_server.backend.gateway_session_store")
    sampling_session_store_module = importlib.import_module("tinker_server.backend.sampling_session_store")
    session_heartbeat_store_module = importlib.import_module("tinker_server.backend.session_heartbeat_store")
    session_index_store_module = importlib.import_module("tinker_server.backend.session_index_store")
    training_session_manager_module = importlib.import_module("tinker_server.backend.training_session_manager")
    training_session_store_module = importlib.import_module("tinker_server.backend.training_session_store")
    owner_runtime_module = importlib.import_module("tinker_server.backend.owner_runtime_supervisor")
    queue_execution_runtime_module = importlib.import_module("tinker_server.backend.queue_execution_runtime")
    checkpoints_module = importlib.import_module("tinker_server.checkpoints")
    gateway_module = importlib.import_module("tinker_server.gateway")
    usage_store_module = importlib.import_module("tinker_server.usage_store")
    verl_training_module = types.ModuleType("tinker_server.backend.verl_training")
    verl_training_module.VerlTrainingEngine = _StubTrainingEngine
    monkeypatch.setitem(sys.modules, "tinker_server.backend.verl_training", verl_training_module)

    monkeypatch.setattr(api_work_queue_module, "api_work_queue", queue)
    monkeypatch.setattr(capacity_manager_module, "capacity_manager", _StubCapacityManager())
    monkeypatch.setattr(future_store_module, "future_store", _StubFutureStore())
    monkeypatch.setattr(owner_runtime_module, "owner_runtime_supervisor", owner_runtime)
    monkeypatch.setattr(queue_execution_runtime_module, "queue_execution_runtime", queue_execution_runtime)
    monkeypatch.setattr(gateway_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(sampling_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(session_heartbeat_store_module, "session_heartbeat_store", SimpleNamespace(ensure_ready=lambda: None, async_size=lambda: 0))
    monkeypatch.setattr(session_index_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(training_session_manager_module, "TrainingSessionManager", _StubTrainingManager)
    monkeypatch.setattr(training_session_store_module, "ensure_ready", lambda: None)
    monkeypatch.setattr(training_session_store_module, "list_training_sessions", lambda: [])
    monkeypatch.setattr(verl_training_module, "VerlTrainingEngine", _StubTrainingEngine)
    monkeypatch.setattr(checkpoints_module, "get_checkpoint_reap_interval_s", lambda: 3600.0)
    monkeypatch.setattr(checkpoints_module, "get_checkpoint_mirror_poll_s", lambda: 3600.0)
    monkeypatch.setattr(checkpoints_module, "reap_runtime_checkpoints", lambda: {})
    monkeypatch.setattr(checkpoints_module, "process_pending_checkpoint_mirrors", lambda: {"mirrored": [], "failed": []})
    monkeypatch.setattr(gateway_module, "close_http_clients", _noop_async)
    monkeypatch.setattr(usage_store_module, "close_usage_store", _noop_async)

    request = SampleRequest(
        sampling_session_id="sess-test",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )
    item = SimpleNamespace(
        request_id="req-test",
        op="sampling.asample",
        request_json=request.model_dump_json().encode("utf-8"),
        user_id="user-test",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        throttle_principal="apikey:bbbbbbbbbbbbbbbbbbbbbbbb",
        webhook_url=None,
        extra={
            "gateway_auth": {
                "user_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
                "user_role": "user",
                "account_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
                "apikey_id": "bbbbbbbbbbbbbbbbbbbbbbbb",
                "request_id": "req-billing-test",
            }
        },
    )

    async def _run() -> None:
        async with app_module.lifespan(app_module.app):
            executor = queue._executors["sampling.asample"]
            await executor(item)

    asyncio.run(_run())

    assert captured == {
        "request_id": "req-test",
        "sampling_session_id": "sess-test",
        "user_id": "user-test",
        "gateway_auth": {
            "user_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "user_role": "user",
            "account_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
            "apikey_id": "bbbbbbbbbbbbbbbbbbbbbbbb",
            "request_id": "req-billing-test",
        },
    }
