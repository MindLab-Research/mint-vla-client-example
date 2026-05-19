import importlib
import sys
from types import SimpleNamespace

import pytest

from mint_server.backend import session_heartbeat_store as heartbeat_store_module
from mint_server.backend import training_session_store as training_store_module
from mint_server.routes import training as training_routes

task_state_store_module = importlib.import_module("mint_server.backend.task_state_store")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _StubTrainingStoreActor:
    def __init__(self):
        self.bump_calls = []
        self.set_calls = []

    def _bump_step(self, model_id: str) -> int:
        self.bump_calls.append(model_id)
        return 7

    def _set_step(self, model_id: str, step: int) -> int:
        self.set_calls.append((model_id, step))
        return step


class _StubHeartbeatStore:
    def __init__(self, stale_session_ids):
        self._stale = set(stale_session_ids)
        self.calls = []

    def is_stale(self, session_id: str, ttl_s: float) -> bool:
        self.calls.append((session_id, ttl_s))
        return session_id in self._stale


class _StubSession:
    def __init__(self, model_id: str, session_id: str):
        self.model_id = model_id
        self.session_id = session_id
        self.current_step = 0
        self.is_active = True


class _StubTrainingManager:
    def __init__(self, sessions):
        self._sessions = dict(sessions)
        self.deleted = []

    def get_session(self, model_id: str):
        return self._sessions.get(model_id)

    def delete_session(self, model_id: str):
        self.deleted.append(model_id)
        self._sessions.pop(model_id, None)
        return True


class _StubTrainingEngine:
    def __init__(self):
        self.unbind_calls = []
        self.delete_calls = []
        self._workers = {}
        self._model_actor_supervisor_actor_names = {}

    async def unbind_session(self, session):
        self.unbind_calls.append(session.model_id)
        session.is_active = False

    async def delete_session(self, session):
        self.delete_calls.append(session.model_id)
        worker = self._workers.get(session.model_id)
        delete_session = getattr(worker, "delete_session", None) if worker is not None else None
        if delete_session is not None:
            delete_session.remote(session.model_id)
        self._workers.pop(session.model_id, None)
        self._model_actor_supervisor_actor_names.pop(session.model_id, None)
        session.is_active = False

    async def shutdown_session(self, session):
        await self.delete_session(session)


class _StubRemoteDelete:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _StubSharedWorker:
    def __init__(self):
        self.delete_calls = []
        self.delete_session = _StubRemoteDelete(self._delete_session)

    def _delete_session(self, model_id: str) -> bool:
        self.delete_calls.append(model_id)
        return True


@pytest.mark.anyio
async def test_issue_368_cleanup_stale_training_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_session = _StubSession("model-stale", "sess-stale")
    live_session = _StubSession("model-live", "sess-live")
    manager = _StubTrainingManager({
        stale_session.model_id: stale_session,
        live_session.model_id: live_session,
    })
    engine = _StubTrainingEngine()
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []
    cleared_model_ids = []
    failed_future_calls = []

    monkeypatch.setattr(training_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "training_engine", engine)

    async def _async_fail_training_requests_for_model(model_id: str, error: str) -> list[str]:
        failed_future_calls.append((model_id, error))
        return ["req-stale"]

    monkeypatch.setattr(
        training_routes.task_futures,
        "async_fail_training_requests_for_model",
        _async_fail_training_requests_for_model,
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        "mint_server.backend.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(clear_session=lambda model_id: cleared_model_ids.append(model_id)),
    )

    deleted = await training_routes._best_effort_delete_training_session(
        stale_session.model_id,
        reason="stale heartbeat (> 123.0s)",
        allow_actor_shutdown=True,
    )

    assert deleted is True
    assert engine.delete_calls == ["model-stale"]
    assert manager.deleted == ["model-stale"]
    assert deleted_model_ids == ["model-stale"]
    assert cleared_model_ids == ["model-stale"]
    assert failed_future_calls == [
        ("model-stale", "Training session terminated due to stale heartbeat (> 123.0s)")
    ]


@pytest.mark.anyio
async def test_issue_368_cleanup_can_restore_session_before_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    restored = _StubSession("model-restore", "sess-restore")
    manager = _StubTrainingManager({})
    engine = _StubTrainingEngine()
    heartbeat_store = _StubHeartbeatStore({"sess-restore"})
    deleted_model_ids = []

    monkeypatch.setattr(training_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "training_engine", engine)

    async def _async_fail_training_requests_for_model(model_id: str, error: str) -> list[str]:
        _ = (model_id, error)
        return []

    monkeypatch.setattr(
        training_routes.task_futures,
        "async_fail_training_requests_for_model",
        _async_fail_training_requests_for_model,
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)

    async def _restore_training_session(_model_id: str):
        return restored

    monkeypatch.setattr(training_routes, "_restore_training_session", _restore_training_session)
    monkeypatch.setattr(
        "mint_server.backend.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(clear_session=lambda model_id: None),
    )

    deleted = await training_routes._best_effort_delete_training_session(
        restored.model_id,
        reason="stale heartbeat (> 60.0s)",
        allow_actor_shutdown=True,
    )

    assert deleted is True
    assert engine.delete_calls == ["model-restore"]
    assert deleted_model_ids == ["model-restore"]


@pytest.mark.anyio
async def test_issue_368_cleanup_aborts_if_future_fail_path_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    stale_session = _StubSession("model-stale", "sess-stale")
    manager = _StubTrainingManager({stale_session.model_id: stale_session})
    engine = _StubTrainingEngine()
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []

    monkeypatch.setattr(training_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "training_engine", engine)

    async def _async_fail_training_requests_for_model(model_id: str, error: str) -> list[str]:
        _ = (model_id, error)
        raise RuntimeError("future-store-down")

    monkeypatch.setattr(
        training_routes.task_futures,
        "async_fail_training_requests_for_model",
        _async_fail_training_requests_for_model,
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        "mint_server.backend.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(clear_session=lambda model_id: None),
    )

    deleted = await training_routes._best_effort_delete_training_session(
        stale_session.model_id,
        reason="stale heartbeat (> 60.0s)",
        allow_actor_shutdown=True,
    )

    assert deleted is False
    assert engine.delete_calls == []
    assert manager.deleted == []
    assert deleted_model_ids == []


@pytest.mark.anyio
async def test_issue_368_cleanup_skips_shared_actor_shutdown_after_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _StubSession("model-stale", "sess-stale")
    shared_worker = _StubSharedWorker()
    manager = _StubTrainingManager({})
    engine = _StubTrainingEngine()
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []

    engine._workers[stale.model_id] = shared_worker
    engine._model_actor_supervisor_actor_names[stale.model_id] = "shared-actor"

    monkeypatch.setattr(training_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "training_engine", engine)

    async def _async_fail_training_requests_for_model(model_id: str, error: str) -> list[str]:
        _ = (model_id, error)
        return []

    monkeypatch.setattr(
        training_routes.task_futures,
        "async_fail_training_requests_for_model",
        _async_fail_training_requests_for_model,
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)

    async def _restore_training_session(_model_id: str):
        return stale

    async def _async_get_ray_ref(value, timeout_s=None):
        _ = timeout_s
        return value

    monkeypatch.setattr(training_routes, "_restore_training_session", _restore_training_session)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value, timeout=None: value))
    monkeypatch.setattr(training_routes, "async_get_ray_ref", _async_get_ray_ref)
    monkeypatch.setattr(
        "mint_server.backend.model_actor_supervisor.get_model_actor_supervisor",
        lambda: SimpleNamespace(clear_session=lambda model_id: None),
    )

    deleted = await training_routes._best_effort_delete_training_session(
        stale.model_id,
        reason="stale heartbeat (> 60.0s)",
        allow_actor_shutdown=False,
    )

    assert deleted is True
    assert engine.delete_calls == []
    assert shared_worker.delete_calls == ["model-stale"]
    assert deleted_model_ids == ["model-stale"]
    assert stale.model_id not in engine._workers
    assert stale.model_id not in engine._model_actor_supervisor_actor_names


def test_issue_368_sync_training_session_step_bumps_when_result_has_no_step(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _StubTrainingStoreActor()
    monkeypatch.setattr(training_store_module, "bump_training_session_step_best_effort", actor._bump_step)
    monkeypatch.setattr(training_store_module, "set_training_session_step_best_effort", actor._set_step)

    result = task_state_store_module._sync_training_session_step(
        {"op": "training.optim_step", "model_id": "model-a"},
        {"metrics": {"loss": 1.0}},
    )

    assert actor.bump_calls == ["model-a"]
    assert actor.set_calls == []
    assert "step" not in result["metrics"]


def test_issue_368_sync_training_session_step_uses_reported_step_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _StubTrainingStoreActor()
    monkeypatch.setattr(training_store_module, "bump_training_session_step_best_effort", actor._bump_step)
    monkeypatch.setattr(training_store_module, "set_training_session_step_best_effort", actor._set_step)

    result = task_state_store_module._sync_training_session_step(
        {"op": "training.train_step", "model_id": "model-b"},
        {"metrics": {"step": 11, "loss": 0.5}},
    )

    assert actor.bump_calls == []
    assert actor.set_calls == [("model-b", 11)]
    assert result["metrics"]["step"] == 11
