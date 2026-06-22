import asyncio
from types import SimpleNamespace

import pytest

from mint_server.backend.stores.task_state_store import FutureStatus, TaskStateStoreUnavailableError
from mint_server.models.types import FutureRetrieveRequest
from mint_server.routes import futures as futures_route


class _WatchingTaskFutureService:
    def __init__(
        self,
        *,
        wait_status: FutureStatus | None,
        result=None,
        status: FutureStatus = FutureStatus.PENDING,
        status_error: Exception | None = None,
        wait_error: Exception | None = None,
        result_error: Exception | None = None,
    ):
        self._wait_status = wait_status
        self._result = result
        self._status = status
        self._status_error = status_error
        self._wait_error = wait_error
        self._result_error = result_error
        self.status_calls = 0
        self.wait_calls: list[tuple[str, float]] = []
        self.cleaned: list[str] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        _ = request_id
        self.status_calls += 1
        if self._status_error is not None:
            raise self._status_error
        return self._status

    async def async_wait_status_change(
        self,
        request_id: str,
        *,
        timeout_s: float,
        terminal_only: bool = False,
    ) -> FutureStatus | None:
        assert terminal_only is True
        self.wait_calls.append((str(request_id), float(timeout_s)))
        if self._wait_error is not None:
            raise self._wait_error
        return self._wait_status

    async def async_get_meta(self, request_id: str):
        _ = request_id
        return {}

    async def async_get_result(self, request_id: str):
        _ = request_id
        if self._result_error is not None:
            raise self._result_error
        return self._result

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(str(request_id))


def _request_stub():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


class _ActorUnavailableError(Exception):
    pass


class _RayTaskNotFoundError(Exception):
    def __init__(self) -> None:
        super().__init__("RayTaskError(TaskStateNotFoundError): missing-rid")


class _MissingLegacyTaskStateStore:
    async def async_get_task(self, request_id: str):
        _ = request_id
        raise _RayTaskNotFoundError()


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


def test_local_retrieve_future_long_poll_store_unavailable_preserves_pending(monkeypatch):
    stub = _WatchingTaskFutureService(
        wait_status=None,
        wait_error=TaskStateStoreUnavailableError("transient actor unavailable"),
    )
    metrics = []

    monkeypatch.setattr(futures_route, "task_futures", stub)
    monkeypatch.setattr(futures_route, "_retrieve_wait_timeout_s", lambda: 0.02)
    monkeypatch.setattr(futures_route, "_retrieve_pending_min_poll_s", lambda: 0.01)
    monkeypatch.setattr(futures_route, "record_retrieve_future_wait_metric", lambda **kwargs: metrics.append(kwargs))

    response = _response_stub()
    payload = asyncio.run(
        futures_route.retrieve_future(
            FutureRetrieveRequest(request_id="rid-long-poll-store-unavailable"),
            _request_stub(),
            response,
        )
    )

    assert response.status_code == 408
    assert response.headers.get("Retry-After") == "1"
    assert payload.get("request_id") == "rid-long-poll-store-unavailable"
    assert payload.get("type") == "try_again"
    assert stub.status_calls == 1
    assert stub.wait_calls == [("rid-long-poll-store-unavailable", 0.02)]
    assert metrics == [{"path": "local", "outcome": "timeout", "waited": True}]


def test_local_retrieve_future_status_actor_unavailable_returns_503(monkeypatch):
    stub = _WatchingTaskFutureService(
        wait_status=None,
        status_error=_ActorUnavailableError("The actor is temporarily unavailable"),
    )
    metrics = []

    monkeypatch.setattr(futures_route, "task_futures", stub)
    monkeypatch.setattr(futures_route, "record_retrieve_future_wait_metric", lambda **kwargs: metrics.append(kwargs))

    with pytest.raises(futures_route.HTTPException) as exc_info:
        asyncio.run(
            futures_route.retrieve_future(
                FutureRetrieveRequest(request_id="rid-status-actor-unavailable"),
                _request_stub(),
                _response_stub(),
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "TaskStateStore unavailable"
    assert stub.status_calls == 1
    assert stub.wait_calls == []
    assert metrics == [{"path": "local", "outcome": "unknown", "waited": False}]


def test_legacy_task_state_lookup_unwraps_ray_task_not_found(monkeypatch):
    import mint_server.backend.stores.task_state_store as task_state_module

    monkeypatch.setattr(task_state_module, "task_state_store", _MissingLegacyTaskStateStore())

    payload = asyncio.run(
        futures_route._lookup_legacy_task_state_terminal(
            "missing-rid",
            _request_stub(),
        )
    )

    assert payload is None


def test_retrieve_future_done_payload_missing_returns_diagnostic_payload(monkeypatch):
    stub = _WatchingTaskFutureService(
        wait_status=None,
        status=FutureStatus.DONE,
        result_error=FileNotFoundError("/missing/payload.json"),
    )
    metrics = []

    monkeypatch.setattr(futures_route, "task_futures", stub)
    monkeypatch.setattr(futures_route, "record_retrieve_future_wait_metric", lambda **kwargs: metrics.append(kwargs))

    response = _response_stub()
    payload = asyncio.run(
        futures_route.retrieve_future(
            FutureRetrieveRequest(request_id="rid-payload-missing"),
            _request_stub(),
            response,
        )
    )

    assert response.status_code == 200
    assert payload["category"] == "system"
    assert payload["request_id"] == "rid-payload-missing"
    assert "Future result payload missing" in payload["error"]
    assert stub.cleaned == []
    assert metrics == [{"path": "local", "outcome": "ready", "waited": False}]
