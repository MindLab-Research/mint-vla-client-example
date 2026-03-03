import asyncio
import pytest
from types import SimpleNamespace

from tinker_server.backend.future_store import FutureStatus
from tinker_server.backend.api_work_queue import ApiWorkQueueUnavailableError
from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


class _StubFutureStore:
    def __init__(self, meta: dict):
        self._meta = dict(meta)

    def get_status(self, request_id: str) -> FutureStatus:
        return FutureStatus.PENDING

    def get_meta(self, request_id: str):
        return dict(self._meta)


class _StubApiWorkQueue:
    def __init__(self, *, depth: int, position: int | None = None, ema_exec_s: float | None = None):
        self._depth = depth
        self._position = position
        self._ema_exec_s = ema_exec_s

    async def find_position(self, request_id: str) -> dict:
        return {"found": True, "position": self._position, "depth": self._depth}

    async def get_eta_state(self, op: str | None) -> dict:
        return {"ema_exec_s": self._ema_exec_s}


class _StubApiWorkQueueUnavailable:
    async def find_position(self, request_id: str) -> dict:
        raise ApiWorkQueueUnavailableError("stub unavailable")

    async def get_eta_state(self, op: str | None) -> dict:
        raise ApiWorkQueueUnavailableError("stub unavailable")


def _request_stub():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_issue_182_pending_payload_queue_backlog_reason(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=5, position=2, ema_exec_s=4.0))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_queue")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_depth") == 5
    assert payload.get("queue_state_reason") == "queue_backlog"
    assert payload.get("retry_after_s") == int(response.headers.get("Retry-After"))


def test_issue_182_pending_payload_reason_null_when_not_queued(monkeypatch):
    meta = {"queue_state": "running", "stage": "prefill", "op": "asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=0, position=None, ema_exec_s=None))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_running")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "prefill"
    assert payload.get("queue_state_reason") is None


def test_issue_182_pending_payload_queue_lookup_unavailable_maps_503(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueueUnavailable())

    body = FutureRetrieveRequest(request_id="rid_unavailable")
    response = _response_stub()
    with pytest.raises(futures_route.HTTPException) as e:
        asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))
    assert e.value.status_code == 503
