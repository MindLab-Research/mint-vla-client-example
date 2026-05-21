import asyncio
from types import SimpleNamespace

import pytest

from mint_server.backend.task_state_store import FutureStatus
from mint_server.models.types import FutureRetrieveRequest
from mint_server.routes import futures as futures_route


class _WatchingTaskFutureService:
    def __init__(self, *, wait_status: FutureStatus | None, result=None):
        self._wait_status = wait_status
        self._result = result
        self.status_calls = 0
        self.wait_calls: list[tuple[str, float]] = []
        self.cleaned: list[str] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        _ = request_id
        self.status_calls += 1
        return FutureStatus.PENDING

    async def async_wait_status_change(
        self,
        request_id: str,
        *,
        timeout_s: float,
        terminal_only: bool = False,
    ) -> FutureStatus | None:
        assert terminal_only is True
        self.wait_calls.append((str(request_id), float(timeout_s)))
        return self._wait_status

    async def async_get_meta(self, request_id: str):
        _ = request_id
        return {}

    async def async_get_result(self, request_id: str):
        _ = request_id
        return self._result

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(str(request_id))


def _request_stub():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


@pytest.fixture(autouse=True)
def _reset_retrieve_future_state(monkeypatch):
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())
    monkeypatch.setattr(futures_route, "_PENDING_HINTS", futures_route.OrderedDict())


def test_local_retrieve_future_long_poll_returns_terminal_result(monkeypatch):
    stub = _WatchingTaskFutureService(
        wait_status=FutureStatus.DONE,
        result={"ok": True},
    )
    metrics = []

    monkeypatch.setattr(futures_route, "task_futures", stub)
    monkeypatch.setattr(futures_route, "_retrieve_wait_timeout_s", lambda: 0.02)
    monkeypatch.setattr(futures_route, "_retrieve_pending_min_poll_s", lambda: 0.01)
    monkeypatch.setattr(futures_route, "record_retrieve_future_wait_metric", lambda **kwargs: metrics.append(kwargs))

    response = _response_stub()
    payload = asyncio.run(
        futures_route.retrieve_future(
            FutureRetrieveRequest(request_id="rid-long-poll-done"),
            _request_stub(),
            response,
        )
    )

    assert response.status_code == 200
    assert payload == {"ok": True}
    assert stub.status_calls == 1
    assert stub.wait_calls == [("rid-long-poll-done", 0.02)]
    assert stub.cleaned == ["rid-long-poll-done"]
    assert metrics == [{"path": "local", "outcome": "ready", "waited": True}]


def test_local_retrieve_future_long_poll_timeout_preserves_pending(monkeypatch):
    stub = _WatchingTaskFutureService(wait_status=None)
    metrics = []

    monkeypatch.setattr(futures_route, "task_futures", stub)
    monkeypatch.setattr(futures_route, "_retrieve_wait_timeout_s", lambda: 0.02)
    monkeypatch.setattr(futures_route, "_retrieve_pending_min_poll_s", lambda: 0.01)
    monkeypatch.setattr(futures_route, "record_retrieve_future_wait_metric", lambda **kwargs: metrics.append(kwargs))

    response = _response_stub()
    payload = asyncio.run(
        futures_route.retrieve_future(
            FutureRetrieveRequest(request_id="rid-long-poll-pending"),
            _request_stub(),
            response,
        )
    )

    assert response.status_code == 408
    assert response.headers.get("Retry-After") == "1"
    assert payload.get("request_id") == "rid-long-poll-pending"
    assert payload.get("type") == "try_again"
    assert stub.status_calls == 1
    assert stub.wait_calls == [("rid-long-poll-pending", 0.02)]
    assert metrics == [{"path": "local", "outcome": "timeout", "waited": True}]
