import asyncio
import concurrent.futures
from types import SimpleNamespace

import pytest

from mint_server.backend.training.training_session_manager import TrainingSession, TrainingSessionManager
from mint_server.backend.training.verl.verl_training import VerlTrainingEngine


def _completed_ref(value):
    future = concurrent.futures.Future()
    future.set_result(value)
    return SimpleNamespace(future=lambda: future)


class _StubRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _StubSharedWorker:
    def __init__(self):
        self.delete_calls: list[tuple[str, dict]] = []
        self.shutdown_calls = 0
        self.delete_session = _StubRemoteMethod(self._delete_session)
        self.shutdown = _StubRemoteMethod(self._shutdown)

    def _delete_session(self, session_id: str, **kwargs):
        self.delete_calls.append((session_id, dict(kwargs)))
        return _completed_ref({"status": "ok", "session_id": session_id, "deleted": True})

    def _shutdown(self):
        self.shutdown_calls += 1
        return _completed_ref(None)


def _make_session(model_id: str, *, backend: str) -> TrainingSession:
    return TrainingSession(
        model_id=model_id,
        session_id=f"sess-{model_id}",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        backend=backend,
    )


def test_issue_381_delete_session_cleans_shared_megatron_state(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    actor_name = "shared-megatron-actor"
    model_id = "model-a"
    sibling_model_id = "model-b"
    session = _make_session(model_id, backend="megatron")
    worker = _StubSharedWorker()
    killed: list[dict] = []
    set_session_calls: list[tuple[str, str | None]] = []

    engine._workers[model_id] = worker
    engine._workers[sibling_model_id] = worker
    engine._model_actor_supervisor_actor_names[model_id] = actor_name
    engine._model_actor_supervisor_actor_names[sibling_model_id] = actor_name
    engine._actor_loaded_sessions[actor_name] = model_id
    engine._actor_volatile_sessions[actor_name] = {model_id}

    import mint_server.backend.training.verl.verl_training as verl_training

    monkeypatch.setattr(
        "mint_server.backend.actors.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(
            is_protected=lambda name: False,
            set_session=lambda name, session_id: set_session_calls.append((name, session_id)),
        ),
    )
    monkeypatch.setattr(verl_training.ray_kill, "kill", lambda *args, **kwargs: killed.append(dict(kwargs)))

    asyncio.run(engine.delete_session(session))

    assert worker.delete_calls == [(model_id, {"traceparent": None})]
    assert worker.shutdown_calls == 0
    assert killed == []
    assert set_session_calls == [(actor_name, sibling_model_id)]
    assert model_id not in engine._workers
    assert model_id not in engine._model_actor_supervisor_actor_names
    assert sibling_model_id in engine._workers
    assert sibling_model_id in engine._model_actor_supervisor_actor_names
    assert engine._actor_loaded_sessions == {}
    assert engine._actor_volatile_sessions == {}
    assert session.is_active is False


def test_issue_381_delete_session_cleans_shared_dense_state(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    actor_name = "shared-dense-actor"
    model_id = "dense-a"
    sibling_model_id = "dense-b"
    session = _make_session(model_id, backend="peft")
    worker = _StubSharedWorker()
    killed: list[dict] = []
    set_session_calls: list[tuple[str, str | None]] = []
    cleared_dense_sessions: list[str] = []

    engine._workers[model_id] = worker
    engine._workers[sibling_model_id] = worker
    engine._model_actor_supervisor_actor_names[model_id] = actor_name
    engine._model_actor_supervisor_actor_names[sibling_model_id] = actor_name

    import mint_server.backend.training.verl.verl_training as verl_training

    monkeypatch.setattr(
        "mint_server.backend.actors.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(
            is_protected=lambda name: False,
            set_session=lambda name, session_id: set_session_calls.append((name, session_id)),
        ),
    )
    monkeypatch.setattr(
        "mint_server.backend.training.dense.dense_trainer.clear_dense_trainer_session",
        lambda target_model_id: cleared_dense_sessions.append(target_model_id),
    )
    monkeypatch.setattr(verl_training.ray_kill, "kill", lambda *args, **kwargs: killed.append(dict(kwargs)))

    asyncio.run(engine.delete_session(session))

    assert worker.delete_calls == [(model_id, {"traceparent": None})]
    assert worker.shutdown_calls == 0
    assert killed == []
    assert cleared_dense_sessions == [model_id]
    assert set_session_calls == [(actor_name, sibling_model_id)]
    assert model_id not in engine._workers
    assert sibling_model_id in engine._workers
    assert session.is_active is False


@pytest.mark.anyio
async def test_issue_381_idle_cleanup_uses_engine_delete_session(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = TrainingSessionManager(inactivity_timeout=1.0)
    session = manager.create_session(
        model_id="idle-model",
        session_id="idle-session",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )
    session.last_activity = 0.0
    session.is_active = True

    engine_delete_calls: list[str] = []
    deleted_store_sessions: list[str] = []
    cleared_sessions: list[str] = []

    class _StubEngine:
        async def delete_session(self, target_session):
            engine_delete_calls.append(target_session.model_id)
            target_session.is_active = False

    manager._engine = _StubEngine()
    monkeypatch.setattr(
        "mint_server.backend.stores.training_session_store.delete_training_session",
        lambda model_id: deleted_store_sessions.append(model_id),
    )
    monkeypatch.setattr(
        "mint_server.backend.actors.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(clear_session=lambda model_id: cleared_sessions.append(model_id)),
    )

    await manager._cleanup_session("idle-model")

    assert engine_delete_calls == ["idle-model"]
    assert manager.get_session("idle-model") is None
    assert deleted_store_sessions == ["idle-model"]
    assert cleared_sessions == ["idle-model"]
