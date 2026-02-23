import anyio
from types import SimpleNamespace

from tinker_server.backend.future_store import FutureStatus
from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


class _StubFutureStore:
    def __init__(self, status: FutureStatus):
        self._status = status
        self.cleanup_calls: list[str] = []

    def get_status(self, request_id: str) -> FutureStatus:
        return self._status

    def get_result(self, request_id: str):
        return {"ok": request_id}

    def get_error(self, request_id: str):
        return f"error:{request_id}"

    def cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)


def _request_with_admin_user():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_done_retrieve_does_not_evict_terminal_future(monkeypatch):
    stub = _StubFutureStore(FutureStatus.DONE)
    monkeypatch.setattr(futures_route, "future_store", stub)

    body = FutureRetrieveRequest(request_id="rid_done")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), response)

    assert payload == {"ok": "rid_done"}
    assert stub.cleanup_calls == ["rid_done"]


def test_failed_retrieve_does_not_evict_terminal_future(monkeypatch):
    stub = _StubFutureStore(FutureStatus.FAILED)
    monkeypatch.setattr(futures_route, "future_store", stub)

    body = FutureRetrieveRequest(request_id="rid_failed")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), response)

    assert payload == {"error": "error:rid_failed", "category": "system"}
    assert stub.cleanup_calls == ["rid_failed"]
