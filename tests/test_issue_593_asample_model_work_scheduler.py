from __future__ import annotations

import json
from types import SimpleNamespace

import anyio

from tinker_server.models.types import ModelInput, SampleRequest, SamplingParams
from tinker_server.routes import sampling as sampling_route


class _StubTaskFutureService:
    def __init__(self, *, fail_update_meta: bool = False):
        self.created: list[str] = []
        self.queued: list[tuple[str, dict | None]] = []
        self.updated: list[tuple[str, dict | None]] = []
        self.cleaned: list[str] = []
        self.fail_update_meta = bool(fail_update_meta)

    async def async_create_with_id(self, request_id: str):
        self.created.append(request_id)
        return request_id

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.queued.append((request_id, None if meta is None else dict(meta)))

    async def async_update_meta(self, request_id: str, meta: dict | None = None) -> None:
        if self.fail_update_meta:
            raise RuntimeError("update meta failed")
        self.updated.append((request_id, None if meta is None else dict(meta)))

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


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id, "apikey_id": "key-a"}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def test_issue_593_asample_routes_multi_lora_to_model_work_scheduler(monkeypatch):
    stub_fs = _StubTaskFutureService()
    scheduler = _CaptureModelWorkScheduler()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)

    import tinker_server.backend.model_work_scheduler as mws
    import tinker_server.backend.result_size_estimator as rse

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
    assert call["request_id"] == out.request_id
    assert call["op"] == "sampling.asample"
    assert json.loads(call["request_json"].decode("utf-8"))["sampling_session_id"] == "session-a"
    assert call["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert call["affinity_group"] == "lora:session-a:generation:7"
    assert call["ordering_key"] == "session:session-a"
    assert call["token_cost"] == 12
    assert call["assign"] is True
    assert call["assign_max_items"] == 1
    assert call["extra"]["model_work_scheduler"] is True
    assert isinstance(call["extra"]["model_work_attempt_id"], str)
    assert call["extra"]["model_work_attempt_id"]
    assert stub_fs.queued[0][1]["queue_state"] == "queued"
    assert stub_fs.queued[0][1]["queue_kind"] == "model_work_scheduler"
    assert stub_fs.queued[0][1]["domain_key"] == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert stub_fs.queued[0][1]["model_work_attempt_id"] == call["extra"]["model_work_attempt_id"]
    assert stub_fs.updated == [
        (
            out.request_id,
            {
                "model_work_scheduler_instance_id": "scheduler-instance-a",
                "model_work_attempt_id": call["extra"]["model_work_attempt_id"],
            },
        )
    ]


def test_issue_593_asample_cancels_scheduler_item_if_post_append_meta_update_fails(monkeypatch):
    stub_fs = _StubTaskFutureService(fail_update_meta=True)
    scheduler = _CaptureModelWorkScheduler()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)

    import tinker_server.backend.model_work_scheduler as mws
    import tinker_server.backend.result_size_estimator as rse

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
    assert scheduler.cancelled == [
        {
            "request_id": scheduler.calls[0]["request_id"],
            "reason": "asample_enqueue_failed",
        }
    ]
    assert stub_fs.cleaned == [scheduler.calls[0]["request_id"]]


def test_issue_593_asample_does_not_cancel_scheduler_item_when_append_rejects(monkeypatch):
    stub_fs = _StubTaskFutureService()
    scheduler = _CaptureModelWorkScheduler(append_error=RuntimeError("duplicate request_id"))

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)

    import tinker_server.backend.model_work_scheduler as mws
    import tinker_server.backend.result_size_estimator as rse

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
    assert stub_fs.cleaned == []


def test_issue_593_asample_ignores_legacy_flag_and_uses_model_work_scheduler(monkeypatch):
    monkeypatch.setenv("MINT_MODEL_WORK_SCHEDULER_ASAMPLE", "0")
    stub_fs = _StubTaskFutureService()
    scheduler = _CaptureModelWorkScheduler()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", stub_fs)

    import tinker_server.backend.model_work_scheduler as mws
    import tinker_server.backend.result_size_estimator as rse

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
