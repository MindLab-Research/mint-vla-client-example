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

    async def async_fail(self, request_id: str, error: str) -> None:
        self._meta["failed_request_id"] = request_id
        self._meta["failed_error"] = error

    async def async_get_error(self, request_id: str) -> str | None:
        _ = request_id
        error = self._meta.get("failed_error")
        return str(error) if error is not None else None


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
        self.describe_pending_request_calls: list[tuple[str, str | None]] = []
        self.find_position_calls: list[str] = []
        self.get_eta_state_calls: list[str | None] = []

    async def describe_pending_request(self, request_id: str, op: str | None) -> dict:
        self.describe_pending_request_calls.append((request_id, op))
        if self._result is not None:
            out = dict(self._result)
        else:
            out = {"found": True, "queue_kind": "legacy", "position": self._position, "depth": self._depth}
        out.setdefault("ema_exec_s", self._ema_exec_s)
        return out

    async def find_position(self, request_id: str, *, timeout_s: float = 5.0) -> dict:
        _ = timeout_s
        self.find_position_calls.append(request_id)
        if self._result is not None:
            return dict(self._result)
        return {"found": True, "queue_kind": "legacy", "position": self._position, "depth": self._depth}

    async def get_eta_state(self, op: str | None, *, timeout_s: float = 5.0) -> dict:
        _ = timeout_s
        self.get_eta_state_calls.append(op)
        return {"ema_exec_s": self._ema_exec_s}


class _StubApiWorkQueueUnavailable:
    async def describe_pending_request(self, request_id: str, op: str | None) -> dict:
        _ = request_id, op
        raise ApiWorkQueueUnavailableError("stub unavailable")

    async def find_position(self, request_id: str, *, timeout_s: float = 5.0) -> dict:
        _ = timeout_s
        raise ApiWorkQueueUnavailableError("stub unavailable")

    async def get_eta_state(self, op: str | None, *, timeout_s: float = 5.0) -> dict:
        _ = timeout_s
        raise ApiWorkQueueUnavailableError("stub unavailable")


class _StubApiWorkQueueProbeUnavailable(_StubApiWorkQueue):
    async def describe_pending_request(self, request_id: str, op: str | None) -> dict:
        _ = request_id, op
        raise ApiWorkQueueUnavailableError("probe unavailable")


class _StubModelWorkScheduler:
    def __init__(self, *, present: bool) -> None:
        self.present = bool(present)
        self.contains_calls: list[str] = []

    async def contains_request(self, *, request_id: str) -> dict:
        self.contains_calls.append(request_id)
        return {"ok": True, "request_id": request_id, "present": self.present}


class _StubCapacityManager:
    def __init__(self) -> None:
        self.released: list[str] = []

    async def async_release_all(self, request_id: str) -> None:
        self.released.append(request_id)


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
    assert wq.api_work_queue.describe_pending_request_calls == [("rid_queue", "sampling.asample")]
    assert wq.api_work_queue.find_position_calls == []
    assert wq.api_work_queue.get_eta_state_calls == []


def test_issue_182_pending_payload_reason_null_when_not_queued(monkeypatch):
    meta = {"queue_state": "running", "stage": "prefill", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    queue = _StubApiWorkQueue(depth=0, position=None, ema_exec_s=None)
    monkeypatch.setattr(wq, "api_work_queue", queue)
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
    assert queue.describe_pending_request_calls == []
    assert queue.find_position_calls == []
    assert queue.get_eta_state_calls == []


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


def test_issue_182_pending_payload_training_correlation_fields(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "training.forward_backward",
        "model_id": "run-429",
        "session_id": "sess-429",
        "seq_id": 30,
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "backend": "megatron",
    }
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueue(depth=1, position=0, ema_exec_s=2.0))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 1, raising=False)

    body = FutureRetrieveRequest(request_id="rid_training_meta")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("model_id") == "run-429"
    assert payload.get("session_id") == "sess-429"
    assert payload.get("seq_id") == 30
    assert payload.get("base_model") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert payload.get("backend") == "megatron"


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


def test_issue_593_model_work_scheduler_payload_skips_legacy_queue_probe(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "sampling.asample",
        "queue_kind": "model_work_scheduler",
        "domain_key": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "affinity_group": "lora:session-a:generation:7",
        "ordering_key": "session:session-a",
    }
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    queue = _StubApiWorkQueueUnavailable()
    monkeypatch.setattr(wq, "api_work_queue", queue)
    import tinker_server.backend.model_work_scheduler as mws

    scheduler = _StubModelWorkScheduler(present=True)
    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_model_work_scheduler_queued")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "model_work_scheduler"
    assert payload.get("queue_state_reason") == "model_work_scheduler"
    assert payload.get("domain_key") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert payload.get("affinity_group") == "lora:session-a:generation:7"
    assert payload.get("ordering_key") == "session:session-a"
    assert payload.get("estimated_wait_s") is None
    assert scheduler.contains_calls == ["rid_model_work_scheduler_queued"]


def test_issue_593_model_work_scheduler_orphan_fails_on_retrieve(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "sampling.asample",
        "queue_kind": "model_work_scheduler",
        "domain_key": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    }
    store = _StubFutureStore(meta)
    monkeypatch.setattr(futures_route, "future_store", store)
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueueUnavailable())
    import tinker_server.backend.capacity_manager as cm

    capacity = _StubCapacityManager()
    monkeypatch.setattr(cm, "capacity_manager", capacity)
    import tinker_server.backend.model_work_scheduler as mws

    scheduler = _StubModelWorkScheduler(present=False)
    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)

    body = FutureRetrieveRequest(request_id="rid_orphaned_model_work")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 200
    assert payload.get("error") == "model work scheduler recovered without this request; request must be retried"
    assert capacity.released == ["rid_orphaned_model_work"]
    assert scheduler.contains_calls == ["rid_orphaned_model_work"]


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


def test_issue_182_legacy_queue_probe_unavailable_maps_503(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(wq, "api_work_queue", _StubApiWorkQueueProbeUnavailable(depth=4, position=1, ema_exec_s=None))
    import tinker_server.config as config_module

    monkeypatch.setattr(config_module.config, "api_work_queue_num_workers", 2, raising=False)

    body = FutureRetrieveRequest(request_id="rid_eta_unavailable")
    response = _response_stub()
    with pytest.raises(futures_route.HTTPException) as e:
        asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))
    assert e.value.status_code == 503


def test_issue_182_scheduler_enabled_skips_eta_lookup_for_scheduled_queue(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "future_store", _StubFutureStore(meta))
    import tinker_server.backend.api_work_queue as wq

    monkeypatch.setattr(
        wq,
        "api_work_queue",
        _StubApiWorkQueue(
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
