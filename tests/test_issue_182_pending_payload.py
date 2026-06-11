import asyncio
from types import SimpleNamespace

import pytest

from mint_server.backend.control_plane_contracts import RetrieveTaskResult
from mint_server.backend.task_state_store import FutureStatus
from mint_server.models.types import FutureRetrieveRequest
from mint_server.routes import futures as futures_route


@pytest.fixture(autouse=True)
def _reset_retrieve_future_caches(monkeypatch):
    monkeypatch.setattr(futures_route, "_RECENT", futures_route.OrderedDict())
    monkeypatch.setattr(futures_route, "_PENDING_HINTS", futures_route.OrderedDict())
    monkeypatch.setattr(futures_route, "_retrieve_wait_timeout_s", lambda: 0.0)


class _StubTaskFutureService:
    def __init__(
        self,
        meta: dict,
        *,
        status: FutureStatus = FutureStatus.PENDING,
        result=None,
        error: str | None = None,
    ):
        self._meta = dict(meta)
        self._status = status
        self._result = result
        self._error = error
        self.cleaned: list[str] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        _ = request_id
        return self._status

    async def async_get_meta(self, request_id: str):
        _ = request_id
        return dict(self._meta)

    async def async_debug_snapshot(self) -> dict:
        return {"meta": dict(self._meta)}

    async def async_fail(self, request_id: str, error: str) -> None:
        self._status = FutureStatus.FAILED
        self._meta["failed_request_id"] = request_id
        self._error = error

    async def async_get_error(self, request_id: str) -> str | None:
        _ = request_id
        return self._error

    async def async_get_result(self, request_id: str):
        _ = request_id
        return self._result

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(str(request_id))


class _StubModelWorkScheduler:
    def __init__(self, *, present: bool = True) -> None:
        self.present = bool(present)
        self.contains_calls: list[dict] = []

    async def contains_request(
        self,
        *,
        request_id: str,
        hydrate_task_state: bool = True,
    ) -> dict:
        self.contains_calls.append(
            {
                "request_id": request_id,
                "hydrate_task_state": hydrate_task_state,
            }
        )
        return {"ok": True, "request_id": request_id, "present": self.present}


def _install_scheduler(monkeypatch, *, present: bool = True) -> _StubModelWorkScheduler:
    import mint_server.backend.model_work_scheduler as mws

    scheduler = _StubModelWorkScheduler(present=present)
    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    return scheduler


class _StubModelWorkTaskGateway:
    def __init__(self, result: RetrieveTaskResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def retrieve_task(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.result


def _install_task_gateway(monkeypatch, result: RetrieveTaskResult) -> _StubModelWorkTaskGateway:
    import mint_server.backend.model_work_task_gateway as gateway_module

    gateway = _StubModelWorkTaskGateway(result)
    monkeypatch.setattr(gateway_module, "model_work_task_gateway", gateway)
    return gateway


@pytest.fixture(autouse=True)
def _install_default_pending_task_gateway(monkeypatch):
    _install_task_gateway(
        monkeypatch,
        RetrieveTaskResult(status="pending", request_id="stub_pending", retry_after_s=1.0),
    )


def _request_stub():
    return SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "admin"}), headers={})


def _response_stub():
    return SimpleNamespace(status_code=200, headers={})


def test_issue_182_pending_payload_model_work_scheduler_queued(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "sampling.asample",
        "queue_kind": "model_work_scheduler",
        "domain_key": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "affinity_group": "lora:session-a:generation:7",
        "ordering_key": "session:session-a",
    }
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))
    scheduler = _install_scheduler(monkeypatch)

    body = FutureRetrieveRequest(request_id="rid_model_work_scheduler_queued")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("request_id") == "rid_model_work_scheduler_queued"
    assert payload.get("type") == "try_again"
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "model_work_scheduler"
    assert payload.get("queue_state_reason") == "model_work_scheduler"
    assert payload.get("domain_key") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert payload.get("affinity_group") == "lora:session-a:generation:7"
    assert payload.get("ordering_key") == "session:session-a"
    assert payload.get("estimated_wait_s") is None
    assert response.headers.get("X-Queue-ETA-S") is None
    assert scheduler.contains_calls == []


def test_issue_182_pending_payload_defaults_queued_to_model_work_scheduler(monkeypatch):
    meta = {"queue_state": "queued", "stage": "queued", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))
    scheduler = _install_scheduler(monkeypatch)

    body = FutureRetrieveRequest(request_id="rid_queue_default")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "model_work_scheduler"
    assert payload.get("queue_state_reason") == "model_work_scheduler"
    assert payload.get("queue_position") is None
    assert payload.get("queue_depth") is None
    assert response.headers.get("Retry-After") == "1"
    assert scheduler.contains_calls == []


def test_issue_182_pending_payload_reason_null_when_not_queued(monkeypatch):
    meta = {"queue_state": "running", "stage": "prefill", "op": "sampling.asample"}
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))

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
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))

    body = FutureRetrieveRequest(request_id="rid_decode")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "decode"
    assert payload.get("progress") == {"tokens_generated": 5, "max_tokens": 12}
    assert response.headers.get("X-Queue-Tokens-Generated") == "5"
    assert response.headers.get("X-Queue-Max-Tokens") == "12"


def test_issue_648_pending_payload_includes_stable_queue_stage_timing(monkeypatch):
    meta = {
        "queue_state": "running",
        "stage": "decode",
        "op": "sampling.asample",
        "queued_at": 10.0,
        "dequeue_at": 15.0,
        "executor_started_at": 16.0,
        "lora_load_s": 2.0,
        "generate_s": 3.0,
    }
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))
    monkeypatch.setattr(futures_route.time, "time", lambda: 20.0)

    body = FutureRetrieveRequest(request_id="rid_issue648_timing")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload["queue_stage_timing"] == {
        "schema_version": 1,
        "scheduler_wait_s": 5.0,
        "executor_wait_s": 1.0,
        "lora_s": 2.0,
        "vllm_generate_s": 3.0,
        "finalization_s": None,
        "total_observed_s": 10.0,
    }


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
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))
    _install_scheduler(monkeypatch)

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


def test_issue_182_gateway_request_id_overrides_upstream(monkeypatch):
    import httpx
    import mint_server.gateway as gateway

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
    import mint_server.gateway as gateway

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
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))

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


def test_issue_182_scheduled_alias_is_normalized_to_model_work_scheduler(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "sampling.asample",
        "queue_kind": "scheduled",
        "scheduler_domain": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "scheduler_session_id": "sess-182",
        "scheduler_domain_key_source": "replica_key",
    }
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))
    _install_scheduler(monkeypatch)

    body = FutureRetrieveRequest(request_id="rid_sched_alias")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "queued"
    assert payload.get("queue_kind") == "model_work_scheduler"
    assert payload.get("scheduler_domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert payload.get("scheduler_session_id") == "sess-182"
    assert payload.get("scheduler_domain_key_source") == "replica_key"
    assert response.headers.get("X-Queue-Scheduler-Domain") == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_issue_593_model_work_scheduler_orphan_stays_pending_on_retrieve(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "sampling.asample",
        "queue_kind": "model_work_scheduler",
        "domain_key": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
    }
    store = _StubTaskFutureService(meta)
    monkeypatch.setattr(futures_route, "task_futures", store)
    scheduler = _install_scheduler(monkeypatch, present=False)

    body = FutureRetrieveRequest(request_id="rid_orphaned_model_work")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("request_id") == "rid_orphaned_model_work"
    assert payload.get("type") == "try_again"
    assert payload.get("queue_kind") == "model_work_scheduler"
    assert scheduler.contains_calls == []


def test_issue_182_pending_model_work_terminal_uses_gateway_result(monkeypatch):
    meta = {
        "queue_state": "queued",
        "stage": "queued",
        "op": "sampling.asample",
        "queue_kind": "model_work_scheduler",
    }
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))
    gateway = _install_task_gateway(
        monkeypatch,
        RetrieveTaskResult(
            status="failed",
            request_id="rid_gateway_failed",
            error={"message": "gateway terminal failure"},
        ),
    )

    body = FutureRetrieveRequest(request_id="rid_gateway_failed")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 200
    assert payload == {"error": "gateway terminal failure", "category": "system"}
    assert gateway.calls == [
        {
            "request_id": "rid_gateway_failed",
            "wait_timeout_s": 0.0,
            "privileged": True,
        }
    ]


def test_issue_182_non_sampling_status_is_generic(monkeypatch):
    meta = {"queue_state": "running", "stage": "prefill", "op": "training.train_step"}
    monkeypatch.setattr(futures_route, "task_futures", _StubTaskFutureService(meta))

    body = FutureRetrieveRequest(request_id="rid_train_running")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 408
    assert payload.get("status") == "running"


def test_issue_182_done_result_cleans_future_without_scheduler_release(monkeypatch):
    store = _StubTaskFutureService({}, status=FutureStatus.DONE, result={"ok": True})
    monkeypatch.setattr(futures_route, "task_futures", store)

    body = FutureRetrieveRequest(request_id="rid_done")
    response = _response_stub()
    payload = asyncio.run(futures_route.retrieve_future(body, _request_stub(), response))

    assert response.status_code == 200
    assert payload == {"ok": True}
    assert store.cleaned == ["rid_done"]
