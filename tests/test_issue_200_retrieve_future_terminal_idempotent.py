import anyio
from types import SimpleNamespace

from mint_server.backend.task_state_store import FutureStatus
from mint_server.models.types import FutureRetrieveRequest
from mint_server.routes import futures as futures_route


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


class _TaskStateTerminalStub:
    def __init__(self, record):
        self._record = record

    async def async_get_task(self, *, request_id: str):
        if self._record.get("request_id") != request_id:
            raise KeyError(request_id)
        return self._record

    async def async_wait_task_status_change(self, *, request_id: str, timeout_s: float, **_kwargs):
        _ = timeout_s
        if self._record.get("request_id") != request_id:
            return {"changed": False, "missing": True, "request_id": request_id}
        return {
            "changed": False,
            "timeout": True,
            "missing": False,
            "record": dict(self._record),
            "request_id": request_id,
        }

    async def async_ensure_ready(self, **_kwargs):
        return {"ok": True}

    async def async_ping(self, **_kwargs):
        return {"ok": True}

    async def async_acquire_owner(self, **_kwargs):
        raise NotImplementedError

    async def async_renew_owner(self, **_kwargs):
        raise NotImplementedError

    async def async_create_task(self, **_kwargs):
        raise NotImplementedError

    async def async_assign_task(self, **_kwargs):
        raise NotImplementedError

    async def async_claim_task(self, **_kwargs):
        raise NotImplementedError

    async def async_renew_lease(self, **_kwargs):
        raise NotImplementedError

    async def async_begin_finalize(self, **_kwargs):
        raise NotImplementedError

    async def async_commit_finalize_success(self, **_kwargs):
        raise NotImplementedError

    async def async_commit_finalize_failure(self, **_kwargs):
        raise NotImplementedError

    async def async_complete_task_failure(self, **_kwargs):
        raise NotImplementedError

    async def async_requeue_task(self, **_kwargs):
        raise NotImplementedError

    async def async_forget_task(self, **_kwargs):
        raise NotImplementedError

    async def async_list_active_tasks(self, **_kwargs):
        raise NotImplementedError

    async def async_update_task_metadata(self, **_kwargs):
        raise NotImplementedError


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


def test_terminal_payload_evicted_is_served_as_known_future(monkeypatch):
    record = {
        "request_id": "rid_evicted",
        "op": "sampling.asample",
        "status": "done",
        "result_path": None,
        "result_checksum": None,
        "error": None,
        "updated_at": 200.0,
        "metadata": {
            "op": "sampling.asample",
            "done_at": 100.0,
            "payload_evicted_at": 150.0,
        },
    }
    stub = _StubTaskFutureService(FutureStatus.DONE, result=None, error=None)
    monkeypatch.setattr(futures_route, "task_futures", stub)
    monkeypatch.setattr(futures_route, "_recent_get", lambda _request_id: None)
    monkeypatch.setattr("mint_server.backend.task_state_store.task_state_store", _TaskStateTerminalStub(record))

    body = FutureRetrieveRequest(request_id="rid_evicted")
    response = _response_stub()
    payload = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), response)

    assert payload["error"] == "Known terminal future evicted"
    assert payload["request_id"] == "rid_evicted"
    assert payload["op"] == "sampling.asample"
    assert response.status_code == 200
