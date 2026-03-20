import asyncio
import pytest
from types import SimpleNamespace

from tinker_server.backend.future_store import FutureStatus
from tinker_server.backend.api_work_queue import ApiWorkQueueUnavailableError
from tinker_server.models.types import FutureRetrieveRequest
from tinker_server.routes import futures as futures_route


@pytest.fixture(autouse=True)
def _reset_retrieve_future_caches(monkeypatch):
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())
    monkeypatch.setattr(futures_route, "_PENDING_HINTS", futures_route.OrderedDict())
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "0")


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
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=5, position=2, ema_exec_s=4.0))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_queue")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("request_id") == "rid_queue"
    assert payload.get("type") == "try_again"
    assert payload.get("status") == "queued"
    assert payload.get("queue_depth") == 5
    assert payload.get("queue_state_reason") == "queue_backlog"
    assert payload.get("estimated_wait_s") == pytest.approx(6.0)
    assert response.headers.get("X-Queue-ETA-S") == "6.000"
    assert payload.get("retry_after_s") == int(response.headers.get("Retry-After"))


def test_issue_182_pending_payload_reason_null_when_not_queued(monkeypatch):
    meta = {"queue_state": "running", "stage": "prefill", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=0, position=None, ema_exec_s=None))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_running")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("request_id") == "rid_running"
    assert payload.get("type") == "try_again"
    assert payload.get("status") == "prefill"
    assert payload.get("queue_state_reason") is None


def test_issue_182_pending_payload_progress_headers(monkeypatch):
    meta = {
        "queue_state": "running",
        "stage": "decode",
        "op": "sampling.asample",
        "progress": {"tokens_generated": 5, "max_tokens": 12},
        "last_progress_at": 0.0,
    }
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=0, position=None, ema_exec_s=None))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_decode")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "decode"
    assert payload.get("progress") == {"tokens_generated": 5, "max_tokens": 12}
    assert response.headers.get("X-Queue-Tokens-Generated") == "5"
    assert response.headers.get("X-Queue-Max-Tokens") == "12"


def test_issue_182_pending_payload_queue_position_unknown_reason(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=None, position=None, ema_exec_s=None))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_unknown_pos")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_position") is None
    assert payload.get("queue_depth") is None
    assert payload.get("queue_state_reason") == "queue_position_unknown"
    assert response.headers.get("X-Queue-Position") is None
    assert response.headers.get("X-Queue-Depth") is None


def test_issue_182_pending_payload_queue_lookup_unavailable_maps_503(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueueUnavailable())

    body = FutureRetrieveRequest(request_id="rid_unavailable")
    response = _response_stub()
    with pytest.raises(futures_route.HTTPException) as e:
        asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))
    assert e.value.status_code == 503


def test_issue_182_gateway_request_id_overrides_upstream(monkeypatch):
    import httpx
    import tinker_server.gateway as gateway

    monkeypatch.setattr(gateway, "decode_request_id", lambda rid: ("upstream-a", "raw-123"))
    monkeypatch.setattr(
        gateway,
        "upstream_for_alias",
        lambda alias: SimpleNamespace(alias=alias, base_url="http://upstream", auth_mode="none", api_key=None),
    )

    async def _forward_json(**kwargs):
        return httpx.Response(
            status_code=408,
            json={"request_id": "raw-123", "type": "try_again"},
            headers={"Retry-After": "1"},
        )

    monkeypatch.setattr(gateway, "forward_json", _forward_json)
    monkeypatch.setattr(gateway, "maybe_register_sampling_session_from_retrieve_future", lambda **kwargs: None)

    body = FutureRetrieveRequest(request_id="gw:upstream-a:encoded-xyz")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("request_id") == body.request_id


def test_issue_182_gateway_request_id_overrides_upstream_200(monkeypatch):
    import httpx
    import tinker_server.gateway as gateway

    monkeypatch.setattr(gateway, "decode_request_id", lambda rid: ("upstream-a", "raw-123"))
    monkeypatch.setattr(
        gateway,
        "upstream_for_alias",
        lambda alias: SimpleNamespace(alias=alias, base_url="http://upstream", auth_mode="none", api_key=None),
    )

    async def _forward_json(**kwargs):
        return httpx.Response(
            status_code=200,
            json={"request_id": "raw-123", "type": "create_model", "model_id": "m1", "backend": "megatron"},
            headers={},
        )

    monkeypatch.setattr(gateway, "forward_json", _forward_json)
    monkeypatch.setattr(gateway, "maybe_register_sampling_session_from_retrieve_future", lambda **kwargs: None)

    body = FutureRetrieveRequest(request_id="gw:upstream-a:encoded-xyz")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 200
    assert payload.get("request_id") == body.request_id


def test_issue_182_scheduler_enabled_omits_queue_fields(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=3, position=1, ema_exec_s=2.0))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    body = FutureRetrieveRequest(request_id="rid_sched")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_state_reason") == "scheduler_enabled"
    assert payload.get("queue_depth") is None
    assert payload.get("queue_position") is None
    assert payload.get("estimated_wait_s") is None
    assert response.headers.get("X-Queue-Depth") is None
    assert response.headers.get("X-Queue-Position") is None
    assert response.headers.get("X-Queue-ETA-S") is None


def test_issue_182_scheduler_default_enabled_omits_queue_fields(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "training.forward_backward"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=7, position=3, ema_exec_s=4.0))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)
    monkeypatch.delenv("MINT_SCHEDULER_ENABLE", raising=False)

    body = FutureRetrieveRequest(request_id="rid_sched_default")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_state_reason") == "scheduler_enabled"
    assert payload.get("queue_depth") is None
    assert payload.get("queue_position") is None
    assert payload.get("estimated_wait_s") is None
    assert response.headers.get("X-Queue-Depth") is None
    assert response.headers.get("X-Queue-Position") is None
    assert response.headers.get("X-Queue-ETA-S") is None


def test_issue_182_non_sampling_status_is_generic(monkeypatch):
    meta = {"queue_state": "running", "stage": "prefill", "op": "training.train_step"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=0, position=None, ema_exec_s=None))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_train_running")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "running"
