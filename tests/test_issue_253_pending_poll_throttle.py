import anyio
from types import SimpleNamespace

from tinker_server.backend.task_state_store import FutureStatus
from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


class _StubTaskFutureService:
    def __init__(self):
        self.status_calls = 0
        self.meta_calls = 0

    async def async_get_status(self, request_id: str) -> FutureStatus:
        self.status_calls += 1
        return FutureStatus.PENDING

    async def async_get_meta(self, request_id: str):
        self.meta_calls += 1
        return {}


def _request_with_admin_user():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_pending_retrieve_short_circuits_repeat_polls(monkeypatch):
    stub = _StubTaskFutureService()
    clock = {"now": 1000.0}

    monkeypatch.setattr(futures_route, "task_futures", stub)
    monkeypatch.setattr(futures_route.time, "time", lambda: clock["now"])
    monkeypatch.setattr(futures_route, "_PENDING_HINTS", futures_route.OrderedDict())

    body = FutureRetrieveRequest(request_id="rid_pending")

    first_response = _response_stub()
    first = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), first_response)
    assert first_response.status_code == 408
    assert first_response.headers.get("X-Tinker-Poll-Throttled") is None
    assert first.get("queue_state") == "active"
    assert first.get("retry_after_s") == 1
    assert first.get("request_id") == "rid_pending"
    assert first.get("type") == "try_again"
    assert stub.status_calls == 1
    assert stub.meta_calls == 1

    second_response = _response_stub()
    second = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), second_response)
    assert second_response.status_code == 408
    assert second_response.headers.get("Retry-After") == "1"
    assert second_response.headers.get("X-Tinker-Poll-Throttled") == "1"
    assert second == {"queue_state": "active", "retry_after_s": 1}
    assert stub.status_calls == 1
    assert stub.meta_calls == 1

    clock["now"] += 1.1
    third_response = _response_stub()
    third = anyio.run(futures_route.retrieve_future, body, _request_with_admin_user(), third_response)
    assert third_response.status_code == 408
    assert third_response.headers.get("X-Tinker-Poll-Throttled") is None
    assert third.get("queue_state") == "active"
    assert third.get("retry_after_s") == 1
    assert third.get("request_id") == "rid_pending"
    assert third.get("type") == "try_again"
    assert stub.status_calls == 2
    assert stub.meta_calls == 2
