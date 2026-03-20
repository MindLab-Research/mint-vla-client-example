import importlib
import sys
from types import SimpleNamespace

import pytest

future_store_module = importlib.import_module("tinker_server.backend.future_store")
from tinker_server.backend import session_heartbeat_store as heartbeat_store_module
from tinker_server.backend import training_session_store as training_store_module
from tinker_server.routes import training as training_routes


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
        self.shutdown_calls = []
        self._workers = {}
        self._resource_pool_actor_names = {}

    async def shutdown_session(self, session):
        self.shutdown_calls.append(session.model_id)
        session.is_active = False


class _StubRemoteDelete:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _StubRemoteMethod:
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


class _StubFutureStoreActor:
    def __init__(self, failed_request_ids):
        self.calls = []
        self._failed_request_ids = list(failed_request_ids)
        self.fail_training_requests_for_model = _StubRemoteMethod(self._fail_training_requests_for_model)

    def _fail_training_requests_for_model(self, *, model_id: str, error: str) -> list[str]:
        self.calls.append((model_id, error))
        return list(self._failed_request_ids)


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
    monkeypatch.setattr(
        training_routes.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: failed_future_calls.append((model_id, error)) or ["req-stale"],
    )
    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {
                "model_id": stale_session.model_id,
                "session_id": stale_session.session_id,
                "actor_name": "dedicated-stale-actor",
            },
            {
                "model_id": live_session.model_id,
                "session_id": live_session.session_id,
                "actor_name": "dedicated-live-actor",
            },
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: cleared_model_ids.append(model_id)),
    )

    cleaned = await training_routes.cleanup_stale_training_sessions_once(stale_after_s=123.0)

    assert cleaned == ["model-stale"]
    assert engine.shutdown_calls == ["model-stale"]
    assert manager.deleted == ["model-stale"]
    assert deleted_model_ids == ["model-stale"]
    assert cleared_model_ids == ["model-stale"]
    assert heartbeat_store.calls == [("sess-stale", 123.0), ("sess-live", 123.0)]
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
    monkeypatch.setattr(
        training_routes.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: [],
    )
    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {
                "model_id": restored.model_id,
                "session_id": restored.session_id,
                "actor_name": "dedicated-actor",
            }
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(training_routes, "_restore_training_session", lambda model_id: restored)
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: None),
    )

    cleaned = await training_routes.cleanup_stale_training_sessions_once(stale_after_s=60.0)

    assert cleaned == ["model-restore"]
    assert engine.shutdown_calls == ["model-restore"]
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
    monkeypatch.setattr(
        training_routes.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: (_ for _ in ()).throw(RuntimeError("future-store-down")),
    )
    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {
                "model_id": stale_session.model_id,
                "session_id": stale_session.session_id,
                "actor_name": "dedicated-stale-actor",
            }
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: None),
    )

    cleaned = await training_routes.cleanup_stale_training_sessions_once(stale_after_s=60.0)

    assert cleaned == []
    assert engine.shutdown_calls == []
    assert manager.deleted == []
    assert deleted_model_ids == []


@pytest.mark.anyio
async def test_issue_368_cleanup_skips_shared_actor_shutdown_after_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _StubSession("model-stale", "sess-stale")
    live = _StubSession("model-live", "sess-live")
    shared_worker = _StubSharedWorker()
    manager = _StubTrainingManager({})
    engine = _StubTrainingEngine()
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []

    engine._workers[stale.model_id] = shared_worker
    engine._resource_pool_actor_names[stale.model_id] = "shared-actor"

    monkeypatch.setattr(training_routes, "training_manager", manager)
    monkeypatch.setattr(training_routes, "training_engine", engine)
    monkeypatch.setattr(
        training_routes.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: [],
    )
    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {"model_id": stale.model_id, "session_id": stale.session_id, "actor_name": "shared-actor"},
            {"model_id": live.model_id, "session_id": live.session_id, "actor_name": "shared-actor"},
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(training_routes, "_restore_training_session", lambda model_id: stale)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value, timeout=None: value))
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: None),
    )

    cleaned = await training_routes.cleanup_stale_training_sessions_once(stale_after_s=60.0)

    assert cleaned == ["model-stale"]
    assert engine.shutdown_calls == []
    assert shared_worker.delete_calls == ["model-stale"]
    assert deleted_model_ids == ["model-stale"]
    assert stale.model_id not in engine._workers
    assert stale.model_id not in engine._resource_pool_actor_names


def test_issue_368_sync_training_session_step_bumps_when_result_has_no_step(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _StubTrainingStoreActor()
    monkeypatch.setattr(training_store_module, "bump_training_session_step", actor._bump_step)
    monkeypatch.setattr(training_store_module, "set_training_session_step", actor._set_step)

    result = future_store_module._sync_training_session_step(
        {"op": "training.optim_step", "model_id": "model-a"},
        {"metrics": {"loss": 1.0}},
    )

    assert actor.bump_calls == ["model-a"]
    assert actor.set_calls == []
    assert result["metrics"]["step"] == 7


def test_issue_368_sync_training_session_step_uses_reported_step_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _StubTrainingStoreActor()
    monkeypatch.setattr(training_store_module, "bump_training_session_step", actor._bump_step)
    monkeypatch.setattr(training_store_module, "set_training_session_step", actor._set_step)

    result = future_store_module._sync_training_session_step(
        {"op": "training.train_step", "model_id": "model-b"},
        {"metrics": {"step": 11, "loss": 0.5}},
    )

    assert actor.bump_calls == []
    assert actor.set_calls == [("model-b", 11)]
    assert result["metrics"]["step"] == 11


def test_issue_368_fail_training_requests_for_model_releases_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _StubFutureStoreActor(["req-1", "req-2"])
    store = future_store_module.FutureStore()
    released_request_ids = []

    monkeypatch.setattr(store, "_get_ray_actor", lambda: actor)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            get=lambda value: value,
            exceptions=SimpleNamespace(ActorDiedError=RuntimeError),
        ),
    )
    monkeypatch.setattr(
        "tinker_server.backend.capacity_manager.capacity_manager.release_all",
        lambda request_id: released_request_ids.append(request_id),
    )

    failed = store.fail_training_requests_for_model("model-z", "stale heartbeat")

    assert failed == ["req-1", "req-2"]
    assert actor.calls == [("model-z", "stale heartbeat")]
    assert released_request_ids == ["req-1", "req-2"]
