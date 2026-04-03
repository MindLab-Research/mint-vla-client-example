import anyio
import pytest
from types import SimpleNamespace

import tinker_server.config as config_module
import tinker_server.backend.future_replay as future_replay_module
from tinker_server.backend.future_replay import _future_replay_sweeper_actor_name, future_replay_store
from tinker_server.backend.future_store import FutureStatus, _meta_with_request_op
from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


@pytest.fixture(autouse=True)
def _reset_retrieve_future_state(monkeypatch, tmp_path):
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())
    monkeypatch.setattr(futures_route, "_PENDING_HINTS", futures_route.OrderedDict())
    monkeypatch.setattr(config_module.config, "future_replay_root_dir", str(tmp_path / "future-replay"), raising=False)
    monkeypatch.setattr(config_module.config, "future_replay_disk_ttl_s", 60.0, raising=False)
    monkeypatch.setattr(config_module.config, "future_replay_hot_ttl_s", 60.0, raising=False)


class _StubFutureStore:
    _UNSET = object()

    def __init__(self, status: FutureStatus, *, result=_UNSET, error=_UNSET, meta=None):
        self._status = status
        self._result = {"ok": "default"} if result is self._UNSET else result
        self._error = "error:default" if error is self._UNSET else error
        self._meta = dict(meta or {})
        self.cleanup_calls: list[str] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        return self._status

    async def async_get_result(self, request_id: str):
        return self._result

    async def async_get_error(self, request_id: str):
        return self._error

    async def async_get_meta(self, request_id: str):
        return dict(self._meta)

    async def async_cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)


class _UnknownFutureStore:
    async def async_get_status(self, request_id: str) -> FutureStatus:
        raise KeyError(f"Unknown request_id: {request_id}")

    async def async_debug_snapshot(self):
        return {"stub": True}


def _request_stub(*, admin: bool = True):
    user_data = {"user_id": "admin"} if admin else {"user_id": "user"}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_issue_440_train_step_result_persists_and_replays(monkeypatch):
    import tinker_server.backend.future_replay as future_replay_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(futures_route.time, "time", lambda: clock["now"])
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])

    stub = _StubFutureStore(
        FutureStatus.DONE,
        result={"ok": "rid-train-step"},
        meta={"op": "training.train_step", "model_id": "m1", "done_at": 990.0},
    )
    monkeypatch.setattr(futures_route, "future_store", stub)

    body = FutureRetrieveRequest(request_id="rid-train-step")
    payload = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())

    assert payload == {"ok": "rid-train-step"}
    assert stub.cleanup_calls == ["rid-train-step"]

    entry = future_replay_store().index_get("rid-train-step")
    assert entry is not None
    assert entry.op == "training.train_step"
    assert entry.model_id == "m1"

    monkeypatch.setattr(futures_route, "future_store", _UnknownFutureStore())
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())

    replayed = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())
    assert replayed == {"ok": "rid-train-step"}


def test_issue_440_random_unknown_without_replay_stays_404(monkeypatch):
    monkeypatch.setattr(futures_route, "future_store", _UnknownFutureStore())

    body = FutureRetrieveRequest(request_id="rid-unknown")
    with pytest.raises(futures_route.HTTPException) as exc:
        anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())
    assert exc.value.status_code == 404


def test_issue_440_known_terminal_future_evicted_not_unknown(monkeypatch):
    import tinker_server.backend.future_replay as future_replay_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(futures_route.time, "time", lambda: clock["now"])
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(config_module.config, "future_replay_disk_ttl_s", 1.0, raising=False)

    stub = _StubFutureStore(
        FutureStatus.DONE,
        result={"ok": "rid-evicted"},
        meta={"op": "training.forward_backward", "model_id": "m2", "done_at": 995.0},
    )
    monkeypatch.setattr(futures_route, "future_store", stub)

    body = FutureRetrieveRequest(request_id="rid-evicted")
    first = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())
    assert first == {"ok": "rid-evicted"}

    monkeypatch.setattr(futures_route, "future_store", _UnknownFutureStore())
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())
    clock["now"] = 1010.0

    payload = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())
    assert payload["error"] == "Known terminal future evicted"
    assert payload["request_id"] == "rid-evicted"
    assert payload["op"] == "training.forward_backward"


def test_issue_440_failure_replay_keeps_public_masking(monkeypatch):
    monkeypatch.setattr(config_module.config, "api_key", "secret", raising=False)

    clock = {"now": 2000.0}
    monkeypatch.setattr(futures_route.time, "time", lambda: clock["now"])
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])

    stub = _StubFutureStore(
        FutureStatus.FAILED,
        error="secret backend trace",
        meta={"op": "training.optim_step", "model_id": "m3", "done_at": 1990.0},
    )
    monkeypatch.setattr(futures_route, "future_store", stub)

    body = FutureRetrieveRequest(request_id="rid-failed")
    first = anyio.run(futures_route.retrieve_future, body, _request_stub(admin=False), _response_stub())
    assert first == {"error": futures_route.GENERIC_ERROR_MESSAGE, "category": "system"}

    monkeypatch.setattr(futures_route, "future_store", _UnknownFutureStore())
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())

    replayed = anyio.run(futures_route.retrieve_future, body, _request_stub(admin=False), _response_stub())
    assert replayed == {"error": futures_route.GENERIC_ERROR_MESSAGE, "category": "system"}


def test_issue_440_concurrent_first_retrieve_returns_equivalent_payloads(monkeypatch):
    clock = {"now": 3000.0}
    monkeypatch.setattr(futures_route.time, "time", lambda: clock["now"])
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])

    stub = _StubFutureStore(
        FutureStatus.DONE,
        result={"ok": "rid-race"},
        meta={"op": "training.train_step", "model_id": "m-race", "done_at": 2990.0},
    )
    monkeypatch.setattr(futures_route, "future_store", stub)

    body = FutureRetrieveRequest(request_id="rid-race")

    async def _retrieve():
        return await futures_route.retrieve_future(body, _request_stub(), _response_stub())

    async def _run():
        results = [None, None]

        async def _slot(i: int):
            results[i] = await _retrieve()

        async with anyio.create_task_group() as tg:
            tg.start_soon(_slot, 0)
            tg.start_soon(_slot, 1)
        return results

    results = anyio.run(_run)

    assert results == [{"ok": "rid-race"}, {"ok": "rid-race"}]
    assert future_replay_store().index_get("rid-race") is not None


def test_issue_440_restart_like_replay_survives_process_local_state_reset(monkeypatch):
    clock = {"now": 4000.0}
    monkeypatch.setattr(futures_route.time, "time", lambda: clock["now"])
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])

    stub = _StubFutureStore(
        FutureStatus.DONE,
        result={"ok": "rid-restart"},
        meta={"op": "training.train_step", "model_id": "m-restart", "done_at": 3990.0},
    )
    monkeypatch.setattr(futures_route, "future_store", stub)

    body = FutureRetrieveRequest(request_id="rid-restart")
    first = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())
    assert first == {"ok": "rid-restart"}

    monkeypatch.setattr(futures_route, "future_store", _UnknownFutureStore())
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())

    replayed = anyio.run(futures_route.retrieve_future, body, _request_stub(), _response_stub())
    assert replayed == {"ok": "rid-restart"}


def test_issue_440_lookup_with_payload_missing_returns_evicted(monkeypatch):
    clock = {"now": 5000.0}
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])

    store = future_replay_store()
    entry = store.persist_terminal_payload(
        request_id="rid-missing-payload",
        op="training.train_step",
        model_id="m-missing",
        final_status=FutureStatus.DONE.value,
        payload={"ok": True},
        done_at_ts=4990.0,
        retrieved_at_ts=5000.0,
    )
    payload_path = store.root_dir / entry.object_relpath
    payload_path.unlink()

    lookup = store.lookup("rid-missing-payload", now_ts=5001.0)

    assert lookup.state == "evicted"
    assert lookup.entry is not None
    assert lookup.entry.payload_deleted_at_ts is not None


def test_issue_440_orphaned_payload_without_index_stays_a_miss(monkeypatch):
    clock = {"now": 6000.0}
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])

    store = future_replay_store()
    object_path = store.root_dir / future_replay_module._object_relpath("rid-orphan")
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_text('{"schema_version":1,"request_id":"rid-orphan"}', encoding="utf-8")

    lookup = store.lookup("rid-orphan", now_ts=6000.0)

    assert lookup.state == "miss"
    assert store.index_get("rid-orphan") is None
    assert object_path.exists()


def test_issue_440_sweeper_deletes_expired_payloads_and_is_idempotent(monkeypatch):
    clock = {"now": 7000.0}
    monkeypatch.setattr(future_replay_module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(config_module.config, "future_replay_disk_ttl_s", 5.0, raising=False)

    store = future_replay_store()
    expired = store.persist_terminal_payload(
        request_id="rid-expired",
        op="training.train_step",
        model_id="m-expired",
        final_status=FutureStatus.DONE.value,
        payload={"ok": "expired"},
        done_at_ts=6990.0,
        retrieved_at_ts=7000.0,
    )
    live = store.persist_terminal_payload(
        request_id="rid-live",
        op="training.train_step",
        model_id="m-live",
        final_status=FutureStatus.DONE.value,
        payload={"ok": "live"},
        done_at_ts=7008.0,
        retrieved_at_ts=7008.0,
    )

    deleted = store.sweep_expired_payloads(now_ts=7006.0)
    deleted_again = store.sweep_expired_payloads(now_ts=7006.0)

    assert deleted == {"deleted": 1}
    assert deleted_again == {"deleted": 0}
    assert not (store.root_dir / expired.object_relpath).exists()
    assert (store.root_dir / live.object_relpath).exists()
    assert store.lookup("rid-expired", now_ts=7006.0).state == "evicted"
    assert store.lookup("rid-live", now_ts=7006.0).state == "ok"


def test_issue_440_future_replay_sweeper_actor_name_is_overrideable(monkeypatch):
    monkeypatch.setenv("MINT_FUTURE_REPLAY_SWEEPER_ACTOR_NAME", "mint_future_replay_sweeper_issue440_ns4")

    assert _future_replay_sweeper_actor_name() == "mint_future_replay_sweeper_issue440_ns4"


def test_issue_440_meta_with_request_op_restores_missing_op():
    meta = {"model_id": "m4", "done_at": 1.0, "final_status": FutureStatus.FAILED.value}

    out = _meta_with_request_op(meta, "training.train_step")

    assert out["op"] == "training.train_step"
    assert out["model_id"] == "m4"
    assert out["final_status"] == FutureStatus.FAILED.value
