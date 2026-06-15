from __future__ import annotations

import json
from types import SimpleNamespace

import anyio

from mint_server.backend.scheduling.model_work_admission import ModelWorkAdmissionRejectedError
from mint_server.models.types import ModelInput, SampleRequest, SamplingParams
from mint_server.routes import sampling as sampling_route


class _StubTaskFutureService:
    def __init__(self, *, fail_update_meta: bool = False):
        self.created: list[str] = []
        self.created_model_work: list[dict] = []
        self.queued: list[tuple[str, dict | None]] = []
        self.updated: list[tuple[str, dict | None]] = []
        self.cleaned: list[str] = []
        self.fail_update_meta = bool(fail_update_meta)

    async def async_create_with_id(self, request_id: str):
        self.created.append(request_id)
        return request_id

    async def async_create_model_work_with_id(self, request_id: str, **kwargs):
        self.created_model_work.append({"request_id": request_id, **dict(kwargs)})
        return {"request_id": request_id, "created": True}

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.queued.append((request_id, None if meta is None else dict(meta)))

    async def async_update_meta(self, request_id: str, meta: dict | None = None) -> None:
        if self.fail_update_meta:
            raise RuntimeError("update meta failed")
        self.updated.append((request_id, None if meta is None else dict(meta)))

    async def async_get_meta(self, _request_id: str) -> dict | None:
        return None

    async def async_get_status(self, _request_id: str) -> str:
        raise KeyError("unknown request_id")

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)


class _CaptureModelWorkScheduler:
    def __init__(self, *, append_error: Exception | None = None):
        self.calls: list[dict] = []
        self.cancelled: list[dict] = []
        self.append_error = append_error

    async def append(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.append_error is not None:
            raise self.append_error
        return {"ok": True, "scheduler_instance_id": "scheduler-instance-a"}

    async def cancel_request(self, **kwargs):
        self.cancelled.append(dict(kwargs))
        return {"ok": True, "cancelled": True}


class _StubSamplingSessionManager:
    def is_multi_lora_session(self, _session_id: str) -> bool:
        return True

    def get_session_base_model(self, _session_id: str) -> str:
        return "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def get_session_lora_rank(self, _session_id: str) -> int:
        return 32

    def get_session_adapter_path(self, _session_id: str) -> str:
        return "/tmp/adapter"

    def is_session_lora_loaded(self, _session_id: str) -> bool:
        return False

    def get_session_lora_int_id(self, _session_id: str):
        return None

    def is_base_model_session(self, _session_id: str) -> bool:
        return False

    def get_session_metadata_version(self, _session_id: str) -> int:
        return 7


def _patch_sampling_snapshot(monkeypatch) -> None:
    async def _snapshot(session_id: str):
        return sampling_route.SamplingSessionSnapshot(
            session_id=session_id,
            base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            lora_rank=32,
            adapter_path="/tmp/adapter",
            lora_loaded=False,
            lora_int_id=None,
            uses_multi_lora=True,
            uses_base_model=False,
            metadata_version=7,
        )

    monkeypatch.setattr(sampling_route, "_async_get_http_sampling_snapshot", _snapshot)


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id, "apikey_id": "key-a"}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def test_issue_593_asample_routes_multi_lora_to_model_work_scheduler(monkeypatch):
    stub_fs = _StubTaskFutureService()
    scheduler = _CaptureModelWorkScheduler()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)
    _patch_sampling_snapshot(monkeypatch)

    import mint_server.backend.scheduling.model_work_scheduler as mws
    import mint_server.backend.core.result_size_estimator as rse

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

    req = SampleRequest(
        sampling_session_id="session-a",
        num_samples=3,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    out = anyio.run(sampling_route.asample, req, _dummy_request("user-a"))

    assert isinstance(out.request_id, str) and out.request_id
    assert stub_fs.created == []
    assert len(scheduler.calls) == 1
    call = scheduler.calls[0]
    assert len(stub_fs.created_model_work) == 1
    created = stub_fs.created_model_work[0]
    assert created["request_id"] == out.request_id
    assert created["op"] == "sampling.asample"
    assert created["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert created["meta"]["queue_state"] == "queued"
    assert created["payload_hash"] == call["extra"]["payload_hash"]
    assert call["request_id"] == out.request_id
    assert call["op"] == "sampling.asample"
    assert json.loads(call["request_json"].decode("utf-8"))["sampling_session_id"] == "session-a"
    assert call["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert call["affinity_group"] == "lora:session-a:generation:7"
    assert call["ordering_key"] == "session:session-a"
    assert call["token_cost"] == 21
    assert call["extra"]["prompt_tokens"] == 3
    assert call["extra"]["max_tokens"] == 4
    assert call["extra"]["num_samples"] == 3
    assert call["extra"]["token_cost"] == 21
    assert call["assign"] is True
    assert call["assign_max_items"] == 1
    assert call["extra"]["model_work_scheduler"] is True
    assert isinstance(call["extra"]["model_work_attempt_id"], str)
    assert call["extra"]["model_work_attempt_id"]
    assert stub_fs.queued == []
    assert call["extra"]["queue_state"] == "queued"
    assert call["extra"]["queue_kind"] == "model_work_scheduler"
    assert call["extra"]["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert isinstance(call["extra"]["payload_hash"], str)
    assert call["extra"]["payload_hash"]
    assert stub_fs.updated == []


def test_issue_593_asample_does_not_mutate_future_meta_after_scheduler_append(monkeypatch):
    stub_fs = _StubTaskFutureService(fail_update_meta=True)
    scheduler = _CaptureModelWorkScheduler()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)
    _patch_sampling_snapshot(monkeypatch)

    import mint_server.backend.scheduling.model_work_scheduler as mws
    import mint_server.backend.core.result_size_estimator as rse

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

    req = SampleRequest(
        sampling_session_id="session-a",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    out = anyio.run(sampling_route.asample, req, _dummy_request("user-a"))

    assert isinstance(out.request_id, str) and out.request_id
    assert len(scheduler.calls) == 1
    assert scheduler.cancelled == []
    assert len(stub_fs.created_model_work) == 1
    assert stub_fs.updated == []
    assert stub_fs.cleaned == []


def test_issue_593_asample_does_not_cancel_scheduler_item_when_append_rejects(monkeypatch):
    stub_fs = _StubTaskFutureService()
    scheduler = _CaptureModelWorkScheduler(append_error=RuntimeError("duplicate request_id"))

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)
    _patch_sampling_snapshot(monkeypatch)

    import mint_server.backend.scheduling.model_work_scheduler as mws
    import mint_server.backend.core.result_size_estimator as rse

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

    req = SampleRequest(
        sampling_session_id="session-a",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    try:
        anyio.run(sampling_route.asample, req, _dummy_request("user-a"))
    except Exception:
        pass

    assert len(scheduler.calls) == 1
    assert scheduler.cancelled == []
    assert stub_fs.created == []
    assert stub_fs.queued == []
    assert len(stub_fs.created_model_work) == 1
    assert stub_fs.cleaned == [stub_fs.created_model_work[0]["request_id"]]


def test_asample_returns_429_for_durable_inflight_admission_rejection(monkeypatch):
    stub_fs = _StubTaskFutureService()
    metrics: list[dict] = []

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)
    _patch_sampling_snapshot(monkeypatch)
    monkeypatch.setattr(
        sampling_route,
        "record_sampling_admission_metric",
        lambda **kwargs: metrics.append(dict(kwargs)),
    )

    async def _reject(**_kwargs):
        raise ModelWorkAdmissionRejectedError(
            {
                "ok": False,
                "reason": "domain_inflight_limit_exceeded",
                "domain_key": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
                "principal": "apikey:key-a",
                "current": 10240,
                "limit": 10240,
                "retry_after_s": 5,
            }
        )

    monkeypatch.setattr(sampling_route, "enqueue_model_work", _reject)

    req = SampleRequest(
        sampling_session_id="session-a",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    response = anyio.run(sampling_route.asample, req, _dummy_request("user-a"))

    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"
    assert json.loads(response.body.decode("utf-8")) == {
        "error": "sampling_backpressure",
        "reason": "domain_inflight_limit_exceeded",
        "domain": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "principal": "apikey:key-a",
        "current": 10240,
        "limit": 10240,
        "retry_after_s": 5,
    }
    assert metrics == [
        {
            "route": "/api/v1/asample",
            "decision": "rejected",
            "reason": "domain_inflight_limit_exceeded",
            "scope": "api_key",
            "domain_key": "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        }
    ]


def test_issue_593_asample_ignores_legacy_flag_and_uses_model_work_scheduler(monkeypatch):
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASAMPLE", "0")
    stub_fs = _StubTaskFutureService()
    scheduler = _CaptureModelWorkScheduler()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)
    _patch_sampling_snapshot(monkeypatch)

    import mint_server.backend.scheduling.model_work_scheduler as mws
    import mint_server.backend.core.result_size_estimator as rse

    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

    req = SampleRequest(
        sampling_session_id="session-a",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    out = anyio.run(sampling_route.asample, req, _dummy_request("user-a"))

    assert isinstance(out.request_id, str) and out.request_id
    assert len(scheduler.calls) == 1
    assert json.loads(scheduler.calls[0]["request_json"].decode("utf-8"))[
        "sampling_session_id"
    ] == "session-a"
