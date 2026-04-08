import asyncio
import inspect
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("ray")

from tinker_server.backend import dense_session_state as dense_state_module
from tinker_server.backend.resource_pool import ActorType, get_resource_pool
from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.backend.verl_training import SessionStateManager, TrainingWorker, VerlTrainingEngine
from tinker_server.config import config as server_config
from tinker_server.routes import internal as internal_routes


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _DeletingWorker:
    def __init__(self, delete_fn):
        self._delete_fn = delete_fn
        self.delete_calls: list[str] = []
        self.delete_session = _RemoteMethod(self._delete_session)
        self.shutdown = _RemoteMethod(lambda: None)

    def _delete_session(self, session_id: str):
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


def test_issue_413_session_state_manager_migrates_legacy_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dense_root = tmp_path / "runtime" / "dense_session_state"
    legacy_root = tmp_path / "legacy_dense_state"
    monkeypatch.setattr(server_config, "training_dense_session_state_root", str(dense_root))
    monkeypatch.setattr(dense_state_module, "DEFAULT_LEGACY_DENSE_SESSION_STATE_ROOT", str(legacy_root))

    legacy_dir = _write_dense_session_dir(legacy_root, "model-413")
    manager = SessionStateManager(base_path=str(dense_root))

    migrated = dense_state_module.maybe_migrate_legacy_dense_session_state("model-413", root=str(dense_root))
    assert migrated == str((dense_root / "model-413_checkpoint").resolve())
    assert manager.session_exists("model-413") is True
    assert not legacy_dir.exists()
    assert (dense_root / "model-413_checkpoint" / "adapter_model.safetensors").exists()

    assert manager.delete_session("model-413") is True
    assert not (dense_root / "model-413_checkpoint").exists()


def test_issue_413_startup_cleanup_migrates_active_and_reaps_stale_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dense_root = tmp_path / "runtime" / "dense_session_state"
    legacy_root = tmp_path / "legacy_dense_state"
    monkeypatch.setattr(server_config, "training_dense_session_state_root", str(dense_root))
    monkeypatch.setattr(dense_state_module, "DEFAULT_LEGACY_DENSE_SESSION_STATE_ROOT", str(legacy_root))

    active_dir = _write_dense_session_dir(legacy_root, "model-active", age_s=600)
    stale_dir = _write_dense_session_dir(legacy_root, "model-stale", age_s=600)
    fresh_dir = _write_dense_session_dir(legacy_root, "model-fresh", age_s=10)

    result = dense_state_module.cleanup_legacy_dense_session_state_once(
        active_session_ids={"model-active"},
        stale_after_s=60.0,
        root=str(dense_root),
    )

    assert result["migrated"] == ["model-active"]
    assert result["deleted"] == ["model-stale"]
    assert result["skipped"] == ["model-fresh"]
    assert not active_dir.exists()
    assert not stale_dir.exists()
    assert fresh_dir.exists()
    assert (dense_root / "model-active_checkpoint" / "adapter_model.safetensors").exists()


def test_issue_413_shutdown_session_reclaims_dense_state_for_shared_actor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dense_root = tmp_path / "runtime" / "dense_session_state"
    monkeypatch.setattr(server_config, "training_dense_session_state_root", str(dense_root))

    import tinker_server.config as config_module

    runtime_env_root = (tmp_path / "runtime_env").resolve()
    runtime_env_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "PFS_RUNTIME_ENV_ROOT", str(runtime_env_root))
    monkeypatch.setattr(config_module, "PFS_PYTHONPATH", str((tmp_path / "runtime_py").resolve()))

    pool = get_resource_pool()
    actor_name = f"peft_trainer_test_{uuid.uuid4().hex}_maxr64"
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
    engine._resource_pool_actor_names[model_id] = actor_name
    engine._resource_pool_actor_names[other_model_id] = actor_name
    engine._workers[model_id] = worker
    engine._workers[other_model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session-413",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        backend="peft",
    )

    import tinker_server.backend.verl_training as verl_training

    monkeypatch.setattr(verl_training.ray, "get", lambda value, timeout=None: value)

    asyncio.run(engine.shutdown_session(session))

    assert worker.delete_calls == [model_id]
    assert not session_dir.exists()
    assert model_id not in engine._resource_pool_actor_names
    assert other_model_id in engine._resource_pool_actor_names
    assert entry.current_session == other_model_id

    pool.unregister(actor_name)


@pytest.mark.anyio
async def test_issue_413_internal_metrics_include_dense_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_admission_stats() -> dict:
        return {
            "driver_state": {
                "dense_session_state_bytes": 1234,
                "dense_session_state_dirs": 5,
                "dense_session_state_oldest_age_s": 67.5,
            }
        }

    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats)

    response = await internal_routes.metrics()
    payload = bytes(response.body).decode("utf-8")

    assert "mint_dense_session_state_bytes 1234" in payload
    assert "mint_dense_session_state_dirs 5" in payload
    assert "mint_dense_session_state_oldest_age_s 67.5" in payload
