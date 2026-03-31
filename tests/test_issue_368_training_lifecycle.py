import importlib
import sys
from types import SimpleNamespace

import pytest

future_store_module = importlib.import_module("tinker_server.backend.future_store")
from tinker_server.backend import session_heartbeat_store as heartbeat_store_module
from tinker_server.backend import training_cleanup_executor as cleanup_executor_module
from tinker_server.backend import training_session_store as training_store_module


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
        self.deleted = []

    def is_stale(self, session_id: str, ttl_s: float) -> bool:
        self.calls.append((session_id, ttl_s))
        return session_id in self._stale

    async def async_is_stale(self, session_id: str, ttl_s: float) -> bool:
        return self.is_stale(session_id, ttl_s)

    def delete(self, session_id: str) -> bool:
        self.deleted.append(session_id)
        return True


class _StubRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _StubFutureStoreActor:
    def __init__(self, failed_request_ids):
        self.calls = []
        self._failed_request_ids = list(failed_request_ids)
        self.fail_training_requests_for_model = _StubRemoteMethod(self._fail_training_requests_for_model)

    def _fail_training_requests_for_model(self, *, model_id: str, error: str) -> list[str]:
        self.calls.append((model_id, error))
        return list(self._failed_request_ids)


@pytest.mark.anyio
async def test_issue_368_detached_training_cleanup_removes_stale_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []
    cleared_model_ids = []
    failed_future_calls = []
    killed = []
    shared_deletes = []

    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {"model_id": "model-stale", "session_id": "sess-stale", "actor_name": "dedicated-stale-actor"},
            {"model_id": "model-live", "session_id": "sess-live", "actor_name": "dedicated-live-actor"},
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        future_store_module.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: failed_future_calls.append((model_id, error)) or ["req-stale"],
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: cleared_model_ids.append(model_id)),
    )
    async def _kill(**kwargs):
        killed.append(kwargs)

    async def _delete_shared(**kwargs):
        shared_deletes.append(kwargs)

    monkeypatch.setattr(cleanup_executor_module, "_kill_training_actor", _kill)
    monkeypatch.setattr(cleanup_executor_module, "_delete_shared_worker_session", _delete_shared)

    cleaned = await cleanup_executor_module.cleanup_stale_training_sessions_once_impl(stale_after_s=123.0)

    assert cleaned == ["model-stale"]
    assert failed_future_calls == [
        ("model-stale", "Training session terminated due to stale heartbeat (> 123.0s)")
    ]
    assert deleted_model_ids == ["model-stale"]
    assert cleared_model_ids == ["model-stale"]
    assert heartbeat_store.calls == [("sess-stale", 123.0), ("sess-live", 123.0)]
    assert heartbeat_store.deleted == ["sess-stale"]
    assert [item["actor_name"] for item in killed] == ["dedicated-stale-actor"]
    assert shared_deletes == []


@pytest.mark.anyio
async def test_issue_368_detached_training_cleanup_uses_shared_delete_for_shared_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []
    shared_deletes = []
    killed = []

    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {"model_id": "model-stale", "session_id": "sess-stale", "actor_name": "shared-actor"},
            {"model_id": "model-live", "session_id": "sess-live", "actor_name": "shared-actor"},
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        future_store_module.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: [],
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: None),
    )
    async def _kill(**kwargs):
        killed.append(kwargs)

    async def _delete_shared(**kwargs):
        shared_deletes.append(kwargs)

    monkeypatch.setattr(cleanup_executor_module, "_kill_training_actor", _kill)
    monkeypatch.setattr(cleanup_executor_module, "_delete_shared_worker_session", _delete_shared)

    cleaned = await cleanup_executor_module.cleanup_stale_training_sessions_once_impl(stale_after_s=60.0)

    assert cleaned == ["model-stale"]
    assert deleted_model_ids == ["model-stale"]
    assert killed == []
    assert [item["actor_name"] for item in shared_deletes] == ["shared-actor"]


@pytest.mark.anyio
async def test_issue_368_detached_training_cleanup_aborts_if_future_fail_path_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []
    killed = []

    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {"model_id": "model-stale", "session_id": "sess-stale", "actor_name": "dedicated-stale-actor"},
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        future_store_module.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: (_ for _ in ()).throw(RuntimeError("future-store-down")),
    )
    async def _kill(**kwargs):
        killed.append(kwargs)

    monkeypatch.setattr(cleanup_executor_module, "_kill_training_actor", _kill)

    cleaned = await cleanup_executor_module.cleanup_stale_training_sessions_once_impl(stale_after_s=60.0)

    assert cleaned == []
    assert deleted_model_ids == []
    assert heartbeat_store.deleted == []
    assert killed == []


@pytest.mark.anyio
async def test_issue_368_detached_training_cleanup_continues_when_actor_actuation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_store = _StubHeartbeatStore({"sess-stale"})
    deleted_model_ids = []
    cleared_model_ids = []

    monkeypatch.setattr(
        training_store_module,
        "list_training_sessions",
        lambda: [
            {"model_id": "model-stale", "session_id": "sess-stale", "actor_name": "dedicated-stale-actor"},
        ],
    )
    monkeypatch.setattr(
        training_store_module,
        "delete_training_session",
        lambda model_id: deleted_model_ids.append(model_id),
    )
    monkeypatch.setattr(heartbeat_store_module, "session_heartbeat_store", heartbeat_store)
    monkeypatch.setattr(
        future_store_module.future_store,
        "fail_training_requests_for_model",
        lambda model_id, error: [],
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: cleared_model_ids.append(model_id)),
    )

    async def _boom(**_kwargs):
        raise RuntimeError("actor-down")

    monkeypatch.setattr(cleanup_executor_module, "_kill_training_actor", _boom)

    cleaned = await cleanup_executor_module.cleanup_stale_training_sessions_once_impl(stale_after_s=60.0)

    assert cleaned == ["model-stale"]
    assert deleted_model_ids == ["model-stale"]
    assert cleared_model_ids == ["model-stale"]
    assert heartbeat_store.deleted == ["sess-stale"]


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
