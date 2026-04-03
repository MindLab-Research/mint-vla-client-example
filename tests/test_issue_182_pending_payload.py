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


class _StubFutureStore:
    def __init__(self, meta: dict):
        self._meta = dict(meta)

    def get_status(self, request_id: str) -> FutureStatus:
        return FutureStatus.PENDING

    async def async_get_status(self, request_id: str) -> FutureStatus:
        return self.get_status(request_id)

    def get_meta(self, request_id: str):
        return dict(self._meta)

    async def async_get_meta(self, request_id: str):
        return self.get_meta(request_id)

    async def async_debug_snapshot(self) -> dict:
        return {"meta": dict(self._meta)}


class _StubApiWorkQueue:
    def __init__(
        self,
        *,
        depth: int,
        position: int | None = None,
        ema_exec_s: float | None = None,
        result: dict | None = None,
    ):
        self._depth = depth
        self._position = position
        self._ema_exec_s = ema_exec_s
        self._result = dict(result) if result is not None else None

    async def find_position(self, request_id: str) -> dict:
        if self._result is not None:
            return dict(self._result)
        return {"found": True, "queue_kind": "legacy", "position": self._position, "depth": self._depth}

    async def get_eta_state(self, op: str | None) -> dict:
        return {"ema_exec_s": self._ema_exec_s}


class _StubApiWorkQueueUnavailable:
    async def find_position(self, request_id: str) -> dict:
        raise ApiWorkQueueUnavailableError("stub unavailable")

    async def get_eta_state(self, op: str | None) -> dict:
        raise ApiWorkQueueUnavailableError("stub unavailable")


class _StubApiWorkQueueEtaUnavailable(_StubApiWorkQueue):
    async def get_eta_state(self, op: str | None) -> dict:
        raise ApiWorkQueueUnavailableError("eta unavailable")


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


def test_issue_182_scheduler_enabled_preserves_legacy_queue_fields(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=3, position=1, ema_exec_s=2.0))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    body = FutureRetrieveRequest(request_id="rid_sched_legacy")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "legacy"
    assert payload.get("queue_state_reason") == "queue_backlog"
    assert payload.get("queue_depth") == 3
    assert payload.get("queue_position") == 1
    assert payload.get("estimated_wait_s") == pytest.approx(2.0)
    assert response.headers.get("X-Queue-Depth") == "3"
    assert response.headers.get("X-Queue-Position") == "1"
    assert response.headers.get("X-Queue-ETA-S") == "2.000"


def test_issue_182_scheduler_enabled_hides_legacy_position_under_mixed_queue(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(
        wq,
        "api_work_queue",
        _StubApiWorkQueue(
            depth=8,
            position=1,
            ema_exec_s=2.0,
            result={
                "found": True,
                "queue_kind": "legacy",
                "position": 1,
                "depth": 8,
                "depth_legacy": 1,
                "depth_scheduled": 7,
            },
        ),
    )
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    body = FutureRetrieveRequest(request_id="rid_sched_mixed")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "legacy"
    assert payload.get("queue_state_reason") == "mixed_queue_arbitration"
    assert payload.get("queue_depth") == 8
    assert payload.get("queue_position") is None
    assert payload.get("estimated_wait_s") is None
    assert response.headers.get("X-Queue-Depth") == "8"
    assert response.headers.get("X-Queue-Position") is None
    assert response.headers.get("X-Queue-ETA-S") is None


def test_issue_182_pending_payload_exposes_stage_timing(monkeypatch):
    meta = {
        "queue_state": "running",
        "stage": "generate",
        "op": "sampling.asample",
        "queue_wait_s": 12.5,
        "dequeue_at": 100.0,
        "executor_started_at": 101.0,
        "executor_done_at": 115.0,
        "executor_exec_s": 14.0,
        "engine_acquire_s": 2.0,
        "lora_load_s": 3.0,
        "generate_s": 8.0,
    }
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=0, position=None, ema_exec_s=None))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_stage_timing")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "prefill"
    assert payload.get("stage") == "generate"
    assert payload.get("queue_wait_s") == 12.5
    assert payload.get("dequeue_at") == 100.0
    assert payload.get("executor_started_at") == 101.0
    assert payload.get("executor_done_at") == 115.0
    assert payload.get("executor_exec_s") == 14.0
    assert payload.get("engine_acquire_s") == 2.0
    assert payload.get("lora_load_s") == 3.0
    assert payload.get("generate_s") == 8.0
    assert response.headers.get("X-Queue-Stage") == "generate"


def test_issue_182_running_payload_keeps_scheduled_metadata_when_queue_lookup_misses(monkeypatch):
    meta = {
        "queue_state": "running",
        "stage": "generate",
        "op": "sampling.asample",
        "queue_kind": "scheduled",
        "scheduler_domain": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "scheduler_session_id": "sess-182",
        "scheduler_domain_key_source": "replica_key",
    }
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(
        wq,
        "api_work_queue",
        _StubApiWorkQueue(
            depth=0,
            ema_exec_s=None,
            result={"found": False, "queue_kind": None, "depth": 0, "depth_legacy": 0, "depth_scheduled": 0},
        ),
    )
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    body = FutureRetrieveRequest(request_id="rid_sched_running")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "prefill"
    assert payload.get("stage") == "generate"
    assert payload.get("queue_kind") == "scheduled"
    assert payload.get("scheduler_domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert payload.get("scheduler_session_id") == "sess-182"
    assert payload.get("scheduler_domain_key_source") == "replica_key"
    assert response.headers.get("X-Queue-Scheduler-Domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_issue_182_queued_scheduled_payload_survives_queue_actor_unavailable(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "sampling.asample",
        "queue_kind": "scheduled",
        "scheduler_domain": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "scheduler_session_id": "sess-182",
    }
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueueUnavailable())
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_sched_queued_unavailable")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "scheduled"
    assert payload.get("scheduler_domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert payload.get("scheduler_session_id") == "sess-182"
    assert payload.get("estimated_wait_s") is None


def test_issue_182_running_payload_survives_queue_actor_unavailable(monkeypatch):
    meta = {
        "queue_state": "running",
        "stage": "generate",
        "op": "sampling.asample",
        "queue_kind": "scheduled",
        "scheduler_domain": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "scheduler_session_id": "sess-182",
    }
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueueUnavailable())
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_sched_running_unavailable")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "prefill"
    assert payload.get("stage") == "generate"
    assert payload.get("queue_kind") == "scheduled"
    assert payload.get("scheduler_domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_issue_182_scheduler_enabled_skips_eta_lookup_for_scheduled_queue(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(
        wq,
        "api_work_queue",
        _StubApiWorkQueueEtaUnavailable(
            depth=8,
            ema_exec_s=None,
            result={
                "found": True,
                "queue_kind": "scheduled",
                "position": None,
                "depth": 8,
                "depth_legacy": 1,
                "depth_scheduled": 7,
                "scheduler_domain": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                "scheduler_session_id": "sess-182",
                "domain_depth": 5,
                "session_depth": 2,
                "session_position": 1,
                "active_sessions": 3,
            },
        ),
    )
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    body = FutureRetrieveRequest(request_id="rid_sched_eta_skip")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("queue_kind") == "scheduled"
    assert payload.get("estimated_wait_s") is None
    assert response.headers.get("X-Queue-ETA-S") is None


def test_issue_182_scheduler_enabled_exposes_scheduled_queue_fields(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(
        wq,
        "api_work_queue",
        _StubApiWorkQueue(
            depth=8,
            ema_exec_s=2.0,
            result={
                "found": True,
                "queue_kind": "scheduled",
                "position": None,
                "depth": 8,
                "depth_legacy": 1,
                "depth_scheduled": 7,
                "scheduler_domain": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                "scheduler_session_id": "sess-182",
                "domain_depth": 5,
                "session_depth": 2,
                "session_position": 1,
                "active_sessions": 3,
            },
        ),
    )
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    body = FutureRetrieveRequest(request_id="rid_sched_domain")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "scheduled"
    assert payload.get("queue_state_reason") == "scheduled_queue"
    assert payload.get("queue_depth") == 8
    assert payload.get("queue_position") is None
    assert payload.get("estimated_wait_s") is None
    assert payload.get("scheduler_domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert payload.get("scheduler_session_id") == "sess-182"
    assert payload.get("queue_depth_legacy") == 1
    assert payload.get("queue_depth_scheduled") == 7
    assert payload.get("queue_depth_domain") == 5
    assert payload.get("queue_depth_session") == 2
    assert payload.get("queue_position_session") == 1
    assert payload.get("queue_active_sessions") == 3
    assert response.headers.get("X-Queue-Depth") == "8"
    assert response.headers.get("X-Queue-Scheduler-Domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert response.headers.get("X-Queue-Domain-Depth") == "5"
    assert response.headers.get("X-Queue-Session-Depth") == "2"
    assert response.headers.get("X-Queue-Session-Position") == "1"


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
