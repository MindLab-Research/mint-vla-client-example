from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from mint_server.models.types import ComputeLogprobsRequest, ForwardBackwardInput, ForwardBackwardRequest, ModelInput, SampleRequest, SamplingParams
from mint_server.queue_priority import (
    effective_queue_priority,
    extract_queue_priority_from_headers,
    merge_queue_priority_extra,
    normalize_queue_priority,
)
from mint_server.routes import internal as internal_route
from mint_server.routes import sampling as sampling_route
from mint_server.routes import training as training_route


class _DummyRequest:
    def __init__(self, *, user_id: str | None = None, apikey_id: str | None = None, headers: dict | None = None) -> None:
        user_data = None
        if user_id is not None:
            user_data = {
                "user_id": user_id,
                "apikey_id": apikey_id,
                "account_id": user_id,
                "user_role": "user",
                "is_admin": False,
            }
        self.state = SimpleNamespace(user_data=user_data)
        self.headers = {} if headers is None else dict(headers)


class _StubTaskFutureService:
    async def async_create_with_id(self, _request_id: str):
        return None

    async def async_mark_queued(self, _request_id: str, meta: dict | None = None) -> None:
        _ = meta

    async def async_update_meta(self, _request_id: str, meta: dict | None = None) -> None:
        _ = meta

    async def async_ensure_pending(self, request_id: str, meta: dict | None = None) -> dict:
        _ = (request_id, meta)
        return {"created": True, "meta": None}

    async def async_cleanup(self, _request_id: str) -> None:
        return None

    async def async_forget(self, _request_id: str) -> None:
        return None


class _CaptureModelWorkScheduler:
    def __init__(self):
        self.calls: list[dict] = []

    async def append(self, **kwargs) -> dict:
        self.calls.append(dict(kwargs))
        return {"ok": True, "scheduler_instance_id": "scheduler-445"}

    async def cancel_request(self, **kwargs) -> dict:
        return {"ok": True, **dict(kwargs)}


class _StubSamplingSessionManager:
    def is_multi_lora_session(self, _session_id: str) -> bool:
        return False

    def get_engine(self, _session_id: str):
        return object()

    def get_session_base_model(self, _session_id: str) -> str:
        return "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_issue_445_queue_priority_normalization_and_aging():
    assert normalize_queue_priority(2) == 2
    assert normalize_queue_priority(99) == 2
    assert normalize_queue_priority(-5) == 0
    assert normalize_queue_priority("bad") == 0
    assert extract_queue_priority_from_headers({"X-MinT-Priority": "1"}) == 1
    assert extract_queue_priority_from_headers({"x-mint-priority": "9"}) == 2
    assert merge_queue_priority_extra(request=None) == {"queue_priority": 0}
    assert merge_queue_priority_extra({"foo": 1, "queue_priority": 7}) == {"foo": 1, "queue_priority": 2}
    assert effective_queue_priority(raw_priority=0, created_at=0.0, now=121.0, aging_s=60.0) == 2


@pytest.mark.anyio
async def test_issue_445_asample_enqueues_normalized_priority(monkeypatch):
    import mint_server.backend.model_work_scheduler as mws

    scheduler = _CaptureModelWorkScheduler()
    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", _StubTaskFutureService())
    monkeypatch.setattr(sampling_route, "record_sampling_admission_metric", lambda **_kwargs: None)
    async def _no_snapshot(_sid):
        return None

    monkeypatch.setattr(sampling_route, "_async_get_detached_sampling_snapshot", _no_snapshot)
    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)

    req = SampleRequest(
        sampling_session_id="sess",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    http_request = _DummyRequest(
        user_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        headers={"X-MinT-Priority": "9"},
    )

    await sampling_route.asample(req, cast(Request, http_request))

    assert scheduler.calls
    assert scheduler.calls[0]["extra"]["queue_priority"] == 2


@pytest.mark.anyio
async def test_issue_445_internal_noop_enqueues_normalized_priority(monkeypatch):
    import importlib

    import mint_server.backend.model_work_scheduler as mws

    task_state_store_module = importlib.import_module("mint_server.backend.task_state_store")

    scheduler = _CaptureModelWorkScheduler()
    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(task_state_store_module, "task_futures", _StubTaskFutureService())

    http_request = _DummyRequest(headers={"X-MinT-Priority": "9"})

    await internal_route.model_work_scheduler_noop(cast(Request, http_request))

    assert scheduler.calls
    assert scheduler.calls[0]["extra"]["queue_priority"] == 2


@pytest.mark.anyio
async def test_issue_445_forward_backward_enqueues_default_priority_on_invalid_header(monkeypatch):
    import mint_server.backend.model_work_scheduler as mws

    scheduler = _CaptureModelWorkScheduler()
    monkeypatch.setattr(training_route, "training_engine", object())
    async def _route_info(_model_id):
        return {"base_model": "Qwen/Qwen3-0.6B", "backend": "peft"}

    async def _protect(_info):
        return None

    monkeypatch.setattr(training_route, "_get_training_route_session_info", _route_info)
    monkeypatch.setattr(training_route, "_protect_training_session_enqueue_window", _protect)
    monkeypatch.setattr(training_route, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(training_route, "task_futures", _StubTaskFutureService())
    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(training_route, "_mark_training_inflight", lambda *_args, **_kwargs: None)

    req = ForwardBackwardRequest(
        model_id="run-445",
        seq_id=8,
        forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )
    http_request = _DummyRequest(
        user_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        headers={"X-MinT-Priority": "bad-value"},
    )

    await training_route.forward_backward(req, cast(Request, http_request))

    assert scheduler.calls
    assert scheduler.calls[0]["extra"]["queue_priority"] == 0
    assert scheduler.calls[0]["extra"]["scheduler_session_key"] == "run-445"
    assert scheduler.calls[0]["apikey_id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"


@pytest.mark.anyio
async def test_issue_445_compute_logprobs_enqueues_apikey_id(monkeypatch):
    import mint_server.backend.model_registry as model_registry
    import mint_server.backend.model_work_scheduler as mws

    scheduler = _CaptureModelWorkScheduler()
    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "task_futures", _StubTaskFutureService())
    monkeypatch.setattr(sampling_route, "record_sampling_admission_metric", lambda **_kwargs: None)
    monkeypatch.setattr(mws, "model_work_scheduler", scheduler)
    monkeypatch.setattr(model_registry, "get_model_config", lambda _model: SimpleNamespace(max_model_len=4096))

    req = ComputeLogprobsRequest(
        sampling_session_id="sess",
        seq_id=3,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )
    http_request = _DummyRequest(
        user_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        headers={"X-MinT-Priority": "9"},
    )

    await sampling_route.compute_logprobs(req, cast(Request, http_request))

    assert scheduler.calls
    assert scheduler.calls[0]["op"] == "sampling.compute_logprobs"
    assert scheduler.calls[0]["apikey_id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"
    assert scheduler.calls[0]["extra"]["queue_priority"] == 2
