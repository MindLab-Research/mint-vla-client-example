import anyio
from types import SimpleNamespace

from tinker_server.backend.task_state_store import FutureStatus
from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


class _StubTaskFutureService:
    _UNSET = object()

    def __init__(self, status: FutureStatus, *, result=_UNSET, error=_UNSET, meta=None):
        self._status = status
        self._result = {"ok": "default"} if result is self._UNSET else result
        self._error = "error:default" if error is self._UNSET else error
        self._meta = meta
        self.cleanup_calls: list[str] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        return self._status

    async def async_get_result(self, request_id: str):
        return self._result

    async def async_get_error(self, request_id: str):
        return self._error

    async def async_get_meta(self, request_id: str):
        return self._meta

    async def async_cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)


def _request_with_admin_user():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_done_retrieve_does_not_evict_terminal_future(monkeypatch):
    stub = _StubTaskFutureService(FutureStatus.DONE, result={"ok": "rid_done"})
    monkeypatch.setattr(futures_route, "task_futures", stub)

    body = FutureRetrieveRequest(request_id="rid_done")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), response)

    assert payload == {"ok": "rid_done"}
    assert stub.cleanup_calls == ["rid_done"]


def test_failed_retrieve_does_not_evict_terminal_future(monkeypatch):
    stub = _StubTaskFutureService(FutureStatus.FAILED, error="error:rid_failed")
    monkeypatch.setattr(futures_route, "task_futures", stub)

    body = FutureRetrieveRequest(request_id="rid_failed")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), response)

    assert payload == {"error": "error:rid_failed", "category": "system"}
    assert stub.cleanup_calls == ["rid_failed"]


def test_retrieved_result_is_served_idempotently(monkeypatch):
    stub = _StubTaskFutureService(FutureStatus.RETRIEVED, result={"ok": "rid_retrieved"})
    monkeypatch.setattr(futures_route, "task_futures", stub)

    body = FutureRetrieveRequest(request_id="rid_retrieved")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), response)

    assert payload == {"ok": "rid_retrieved"}
    assert stub.cleanup_calls == []


def test_retrieved_error_is_served_idempotently(monkeypatch):
    stub = _StubTaskFutureService(FutureStatus.RETRIEVED, result=None, error="error:rid_retrieved_failed")
    monkeypatch.setattr(futures_route, "task_futures", stub)

    body = FutureRetrieveRequest(request_id="rid_retrieved_failed")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), response)

    assert payload == {"error": "error:rid_retrieved_failed", "category": "system"}
    assert stub.cleanup_calls == []
