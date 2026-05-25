import asyncio
import inspect
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("ray")

from mint_server.backend import dense_session_state as dense_state_module
import mint_server.backend.model_actor_inventory as model_actor_inventory_module
from mint_server.backend.model_actor_supervisor import ActorType, ModelActorSupervisor
from mint_server.backend.training_session_manager import TrainingSession
from mint_server.backend.verl_training import TrainingWorker, VerlTrainingEngine
from mint_server.config import config as server_config


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        result = self._fn(*args, **kwargs)

        class _DoneRef:
            def future(self):
                fut = asyncio.get_running_loop().create_future()
                fut.set_result(result)
                return fut

        return _DoneRef()


class _DeletingWorker:
    def __init__(self, delete_fn):
        self._delete_fn = delete_fn
        self.delete_calls: list[str] = []
        self.delete_session = _RemoteMethod(self._delete_session)
        self.shutdown = _RemoteMethod(lambda: None)

    def _delete_session(self, session_id: str, **_kwargs):
        self.delete_calls.append(session_id)
        return self._delete_fn(session_id)


def _write_dense_session_dir(root: Path, session_id: str, *, age_s: float | None = None) -> Path:
    import os

    session_dir = root / f"{session_id}_checkpoint"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (session_dir / "optimizer.pt").write_bytes(b"optimizer")
    if age_s is not None:
        ts = time.time() - float(age_s)
        for path in (session_dir, session_dir / "adapter_model.safetensors", session_dir / "optimizer.pt"):
            os.utime(path, (ts, ts))
    return session_dir


def test_issue_413_training_worker_signature_accepts_session_state_root() -> None:
    modified_class = TrainingWorker.__ray_metadata__.modified_class
    sig = inspect.signature(modified_class.__init__)

    assert "session_state_root" in sig.parameters


def test_issue_413_shutdown_session_reclaims_dense_state_for_shared_actor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dense_root = tmp_path / "runtime" / "dense_session_state"
    monkeypatch.setattr(server_config, "training_dense_session_state_root", str(dense_root))

    import mint_server.config as config_module

    runtime_env_root = (tmp_path / "runtime_env").resolve()
    runtime_env_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "PFS_RUNTIME_ENV_ROOT", str(runtime_env_root))
    monkeypatch.setattr(config_module, "PFS_PYTHONPATH", str((tmp_path / "runtime_py").resolve()))
    monkeypatch.setattr(model_actor_inventory_module.ray, "is_initialized", lambda: False)
    pool = ModelActorSupervisor()
    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.backend.verl_training as verl_training

    monkeypatch.setattr(supervisor_module, "model_actor_supervisor", pool)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: pool)
    monkeypatch.setattr(verl_training, "get_model_actor_supervisor", lambda: pool, raising=False)
    pool.clear(kill_actors=False)
    actor_name = f"mint_dense_test_{uuid.uuid4().hex}"
    model_id = f"model_{uuid.uuid4().hex}"
    other_model_id = f"model_{uuid.uuid4().hex}"

    pool.unregister(actor_name)
    entry = pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        base_model="/tmp/fake_model_path",
        session_id=model_id,
    )
    pool.mark_ready(actor_name)

    session_dir = _write_dense_session_dir(dense_root, model_id)
    worker = _DeletingWorker(lambda session_id: dense_state_module.delete_dense_session_state(session_id))
    engine = VerlTrainingEngine()
    engine._model_actor_supervisor_actor_names[model_id] = actor_name
    engine._model_actor_supervisor_actor_names[other_model_id] = actor_name
    engine._workers[model_id] = worker
    engine._workers[other_model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session-413",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        backend="peft",
    )

    import mint_server.backend.verl_training as verl_training

    monkeypatch.setattr(verl_training.ray, "get", lambda value, timeout=None: value)

    asyncio.run(engine.delete_session(session))

    assert worker.delete_calls == [model_id]
    assert not session_dir.exists()
    assert model_id not in engine._model_actor_supervisor_actor_names
    assert other_model_id in engine._model_actor_supervisor_actor_names
    assert entry.current_session == other_model_id

    pool.unregister(actor_name)


def test_issue_413_otel_metrics_include_dense_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.logging_context as logging_context

    callbacks: dict[str, object] = {}

    class _FakeMeter:
        def create_counter(self, *_args, **_kwargs):
            return None

        def create_histogram(self, *_args, **_kwargs):
            return None

        def create_observable_gauge(self, name, **kwargs):
            callbacks[name] = kwargs["callbacks"][0]

    class _Observation:
        def __init__(self, value, attributes):
            self.value = value
            self.attributes = attributes

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "_OTEL_INITIALIZED", False)
    monkeypatch.setattr(logging_context, "_API_PROCESS_OBSERVABLES_REGISTERED", False)
    monkeypatch.setattr(
        "mint_server.backend.dense_session_state.collect_dense_session_state_stats",
        lambda: {
            "dense_session_state_bytes": 1234,
            "dense_session_state_dirs": 5,
            "dense_session_state_oldest_age_s": 67.5,
        },
    )
    monkeypatch.setattr("mint_server.backend.session_heartbeat_store.session_heartbeat_store.size", lambda: 0)
    monkeypatch.setattr("mint_server.routes.sampling._lora_load_lock_count_sync", lambda: 0)
    monkeypatch.setattr("mint_server.routes.service.session_manager", None)

    logging_context._register_api_process_observable_metrics(_FakeMeter(), _Observation)

    assert callbacks["mint_dense_session_state_bytes"](None)[0].value == 1234.0
    assert callbacks["mint_dense_session_state_dirs"](None)[0].value == 5.0
    assert callbacks["mint_dense_session_state_oldest_age_s"](None)[0].value == 67.5


def test_issue_413_configure_logging_does_not_register_api_process_gauges_for_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentelemetry.metrics as otel_metrics

    import mint_server.logging_context as logging_context

    created: list[str] = []

    class _FakeMeter:
        def create_counter(self, *_args, **_kwargs):
            return None

        def create_histogram(self, *_args, **_kwargs):
            return None

        def create_observable_gauge(self, name, **_kwargs):
            created.append(name)

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel.example:4317")
    monkeypatch.setattr(otel_metrics, "get_meter", lambda _name: _FakeMeter())
    monkeypatch.setattr(logging_context, "_OTEL_INITIALIZED", False)
    monkeypatch.setattr(logging_context, "_OTEL_ENABLED", False)
    monkeypatch.setattr(logging_context, "_API_PROCESS_OBSERVABLES_REGISTERED", False)

    logging_context._configure_opentelemetry(__import__("logging").getLogger("test"))

    assert "mint_public_healthz_cache_age_seconds" in created
    assert "mint_dense_session_state_bytes" not in created
    assert "mint_driver_sampling_sessions_total" not in created
