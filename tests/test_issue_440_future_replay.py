import anyio
import pytest
from types import SimpleNamespace

import tinker_server.config as config_module
from tinker_server.backend.future_replay import future_replay_store
from tinker_server.backend.future_store import FutureStatus
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

    def get_status(self, request_id: str) -> FutureStatus:
        return self._status

    def get_result(self, request_id: str):
        return self._result

    def get_error(self, request_id: str):
        return self._error

    def get_meta(self, request_id: str):
        return dict(self._meta)

    def cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)


class _UnknownFutureStore:
    def get_status(self, request_id: str) -> FutureStatus:
        raise KeyError(f"Unknown request_id: {request_id}")

    def debug_snapshot(self):
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
    import tinker_server.backend.future_replay as future_replay_module

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
