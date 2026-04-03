from __future__ import annotations

import asyncio
import importlib
import importlib.machinery
import sys
import time
import types
from types import SimpleNamespace

import pytest

from tinker_server.models.types import ForwardBackwardInput, ForwardBackwardRequest, ModelInput, SampleRequest, SamplingParams
from tinker_server.queue_priority import (
    effective_queue_priority,
    extract_queue_priority_from_headers,
    merge_queue_priority_extra,
    normalize_queue_priority,
)
from tinker_server.routes import internal as internal_route
from tinker_server.routes import sampling as sampling_route
from tinker_server.routes import training as training_route


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


class _StubFutureStore:
    async def async_create_with_id(self, _request_id: str):
        return None

    async def async_mark_queued(self, _request_id: str, meta: dict | None = None) -> None:
        _ = meta

    async def async_ensure_pending(self, request_id: str, meta: dict | None = None) -> dict:
        _ = (request_id, meta)
        return {"created": True, "meta": None}

    async def async_cleanup(self, _request_id: str) -> None:
        return None

    async def async_forget(self, _request_id: str) -> None:
        return None


class _StubCapacityManager:
    async def async_try_reserve(self, request_id: str, queue_bytes: int, object_store_bytes: int) -> dict:
        _ = (request_id, queue_bytes, object_store_bytes)
        return {"ok": True}

    async def async_release_all(self, _request_id: str) -> None:
        return None


class _CaptureQueue:
    def __init__(self):
        self.calls: list[dict] = []

    async def enqueue(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


class _StubSamplingSessionManager:
    def is_multi_lora_session(self, _session_id: str) -> bool:
        return False

    def get_engine(self, _session_id: str):
        return object()

    def get_session_base_model(self, _session_id: str) -> str:
        return "Qwen/Qwen3-4B-Instruct-2507"


def _install_ray_stub(monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    class _Ctx:
        def get_task_id(self) -> str:
            return "task-mock"

        def get_job_id(self) -> str:
            return "job-mock"

    class _MethodProxy:
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    class _ActorProxy:
        def __init__(self, obj):
            self._obj = obj

        def __getattr__(self, name):
            attr = getattr(self._obj, name)
            if callable(attr):
                return _MethodProxy(attr)
            return attr

    def remote(*_args, **_kwargs):
        def _decorator(cls):
            class _RemoteWrapped(cls):
                @classmethod
                def options(cls_, **_opts):
                    class _OptionsHandle:
                        def remote(self, *args, **kwargs):
                            return _ActorProxy(cls_(*args, **kwargs))

                    return _OptionsHandle()

            return _RemoteWrapped

        return _decorator

    def get_actor(*_args, **_kwargs):
        raise ValueError("named actor not found")

    ray.remote = remote  # type: ignore[attr-defined]
    ray.get = lambda obj, timeout=None: obj  # type: ignore[attr-defined]
    ray.get_actor = get_actor  # type: ignore[attr-defined]
    ray.cluster_resources = lambda: {}  # type: ignore[attr-defined]
    ray.get_runtime_context = lambda: _Ctx()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", ray)


def _load_api_work_queue_module(monkeypatch):
    _install_ray_stub(monkeypatch)
    monkeypatch.setenv("RAY_ADDRESS", "ray://test")
    import tinker_server.config as config_module
    import tinker_server.backend.api_work_queue as api_work_queue

    monkeypatch.setattr(config_module, "PFS_PYTHONPATH", "")
    monkeypatch.setattr(config_module, "actor_runtime_env", lambda pythonpath, extra=None: {})
    return importlib.reload(api_work_queue)


def _scheduled_item(
    request_id: str,
    *,
    raw_priority: int,
    domain: str,
    session_key: str,
    created_at: float,
) -> dict:
    scheduler_domain = domain if ":" in str(domain) else f"peft:{domain}"
    return {
        "request_id": request_id,
        "op": "training.forward_backward",
        "request_json": b"{}",
        "user_id": None,
        "apikey_id": None,
        "throttle_principal": None,
        "webhook_url": None,
        "extra": {
            "queue_priority": raw_priority,
            "scheduler_enabled": True,
            "scheduler_domain": scheduler_domain,
            "scheduler_session_key": session_key,
        },
        "created_at": created_at,
    }


def _legacy_item(request_id: str, *, raw_priority: int, created_at: float) -> dict:
    return {
        "request_id": request_id,
        "op": "sampling.asample",
        "request_json": b"{}",
        "user_id": None,
        "apikey_id": None,
        "throttle_principal": None,
        "webhook_url": None,
        "extra": {"queue_priority": raw_priority},
        "created_at": created_at,
    }


async def _enqueue_many(actor, items: list[dict]) -> None:
    for it in items:
        await actor.enqueue(it)


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


def test_issue_445_actor_prefers_higher_priority_bucket(monkeypatch):
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._create_ray_actor()

    now = time.time()
    asyncio.run(
        _enqueue_many(
            actor,
            [
                _legacy_item("low", raw_priority=0, created_at=now - 5.0),
                _legacy_item("high", raw_priority=2, created_at=now - 1.0),
            ],
        )
    )

    first = asyncio.run(actor.dequeue("consumer-job"))
    assert first["request_id"] == "high"
    assert first["extra"]["_queue_priority_raw"] == 2
    assert first["extra"]["_queue_priority_effective"] == 2
    assert first["extra"]["_queue_kind"] == "legacy"


def test_issue_445_actor_aging_promotes_waiting_low_priority(monkeypatch):
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._create_ray_actor()

    now = time.time()
    asyncio.run(
        _enqueue_many(
            actor,
            [
                _legacy_item("aged-low", raw_priority=0, created_at=now - 61.0),
                _legacy_item("fresh-mid", raw_priority=1, created_at=now - 1.0),
            ],
        )
    )

    first = asyncio.run(actor.dequeue("consumer-job"))
    assert first["request_id"] == "aged-low"
    assert first["extra"]["_queue_priority_raw"] == 0
    assert first["extra"]["_queue_priority_effective"] == 1


def test_issue_445_actor_keeps_training_scheduler_behavior_within_priority_tier(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    monkeypatch.setenv("MINT_SCHEDULER_FAIRNESS", "rr")
    monkeypatch.setenv("MINT_SCHEDULER_MAX_CONSECUTIVE", "2")
    monkeypatch.setenv("MINT_SCHEDULER_STARVATION_S", "1000000000000")
    monkeypatch.setenv("MINT_SCHEDULER_COALESCE_MS", "0")
    api_work_queue = _load_api_work_queue_module(monkeypatch)
    actor = api_work_queue._create_ray_actor()

    now = time.time()
    asyncio.run(
        _enqueue_many(
            actor,
            [
                _legacy_item("legacy-low", raw_priority=0, created_at=now - 100.0),
                _scheduled_item("r1", raw_priority=2, domain="d", session_key="A", created_at=now - 4.0),
                _scheduled_item("r2", raw_priority=2, domain="d", session_key="B", created_at=now - 3.0),
                _scheduled_item("r3", raw_priority=2, domain="d", session_key="A", created_at=now - 2.0),
                _scheduled_item("r4", raw_priority=2, domain="d", session_key="B", created_at=now - 1.0),
            ],
        )
    )

    out = []
    for _ in range(4):
        item = asyncio.run(actor.dequeue("consumer-job"))
        out.append(item)
        asyncio.run(actor.finalize_request(item["request_id"]))
    sessions = [str((item.get("extra") or {}).get("scheduler_session_key")) for item in out]
    assert sessions == ["A", "A", "B", "B"]
    assert all((item.get("extra") or {}).get("_queue_priority_raw") == 2 for item in out)


@pytest.mark.anyio
async def test_issue_445_asample_enqueues_normalized_priority(monkeypatch):
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.result_size_estimator as rse

    queue = _CaptureQueue()
    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", _StubFutureStore())
    monkeypatch.setattr(sampling_route, "record_sampling_admission_metric", lambda **_kwargs: None)
    async def _no_snapshot(_sid):
        return None

    monkeypatch.setattr(sampling_route, "_async_get_detached_sampling_snapshot", _no_snapshot)
    monkeypatch.setattr(awq, "api_work_queue", queue)
    monkeypatch.setattr(cm, "capacity_manager", _StubCapacityManager())
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

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

    await sampling_route.asample(req, http_request)

    assert queue.calls
    assert queue.calls[0]["extra"]["queue_priority"] == 2


@pytest.mark.anyio
async def test_issue_445_internal_noop_enqueues_normalized_priority(monkeypatch):
    import importlib

    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.result_size_estimator as rse

    future_store_module = importlib.import_module("tinker_server.backend.future_store")

    queue = _CaptureQueue()
    monkeypatch.setattr(awq, "api_work_queue", queue)
    monkeypatch.setattr(cm, "capacity_manager", _StubCapacityManager())
    monkeypatch.setattr(future_store_module, "future_store", _StubFutureStore())
    monkeypatch.setattr(rse, "estimate_small_result_bytes", lambda: 0)

    http_request = _DummyRequest(headers={"X-MinT-Priority": "9"})

    await internal_route.work_queue_noop(http_request)

    assert queue.calls
    assert queue.calls[0]["extra"]["queue_priority"] == 2


@pytest.mark.anyio
async def test_issue_445_forward_backward_enqueues_default_priority_on_invalid_header(monkeypatch):
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.result_size_estimator as rse

    queue = _CaptureQueue()
    monkeypatch.setattr(training_route, "training_engine", object())
    async def _route_info(_model_id):
        return {"base_model": "Qwen/Qwen3-0.6B", "backend": "peft"}

    async def _protect(_info):
        return None

    monkeypatch.setattr(training_route, "_get_training_route_session_info", _route_info)
    monkeypatch.setattr(training_route, "_protect_training_session_enqueue_window", _protect)
    monkeypatch.setattr(training_route, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(training_route, "future_store", _StubFutureStore())
    monkeypatch.setattr(awq, "api_work_queue", queue)
    monkeypatch.setattr(cm, "capacity_manager", _StubCapacityManager())
    monkeypatch.setattr(rse, "estimate_forward_backward_result_bytes", lambda _req: 0)
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

    await training_route.forward_backward(req, http_request)

    assert queue.calls
    assert queue.calls[0]["extra"]["queue_priority"] == 0
    assert queue.calls[0]["extra"]["scheduler_session_key"] == "run-445"
