import anyio
import pytest
from fastapi import HTTPException
from types import SimpleNamespace

import tinker_server.backend.api_work_queue as awq
from tinker_server.backend.api_work_queue import ApiWorkQueueThrottleError
from tinker_server.models.types import ModelInput, SampleRequest, SamplingParams
from tinker_server.routes import sampling as sampling_route


class _StubSamplingSessionManager:
    def is_multi_lora_session(self, _session_id: str) -> bool:
        return False

    def get_engine(self, _session_id: str):
        return object()

    def get_session_base_model(self, _session_id: str) -> str:
        return "Qwen/Qwen3-4B-Instruct-2507"


class _StubFutureStore:
    def create_with_id(self, _request_id: str):
        return None

    def mark_queued(self, _request_id: str, meta: dict | None = None) -> None:
        _ = meta

    def ensure_pending(self, request_id: str, meta: dict | None = None) -> dict:
        _ = (request_id, meta)
        return {"created": True, "meta": None}

    def cleanup(self, _request_id: str) -> None:
        return None

    def forget(self, _request_id: str) -> None:
        return None


class _StubCapacityManager:
    def try_reserve(self, request_id: str, queue_bytes: int, object_store_bytes: int) -> dict:
        _ = (request_id, queue_bytes, object_store_bytes)
        return {"ok": True}

    def release_all(self, _request_id: str) -> None:
        return None


class _StubApiWorkQueue:
    async def enqueue(self, **kwargs) -> None:
        _ = kwargs
        raise ApiWorkQueueThrottleError(scope="api_key", limit=1, pending=1)


def _dummy_request(*, user_id: str, apikey_id: str):
    return SimpleNamespace(
        state=SimpleNamespace(
            user_data={
                "user_id": user_id,
                "apikey_id": apikey_id,
                "account_id": user_id,
                "user_role": "user",
                "is_admin": False,
            }
        ),
        headers={},
    )


def _sample_request() -> SampleRequest:
    return SampleRequest(
        sampling_session_id="sess",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )


def test_issue_324_asample_maps_queue_throttle_to_429(monkeypatch):
    stub_q = _StubApiWorkQueue()
    stub_fs = _StubFutureStore()
    stub_cap = _StubCapacityManager()
    metric_calls: list[dict] = []

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", stub_fs)
    monkeypatch.setattr(
        sampling_route,
        "record_sampling_admission_metric",
        lambda **kwargs: metric_calls.append(dict(kwargs)),
    )

    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.result_size_estimator as rse

    monkeypatch.setattr(awq, "api_work_queue", stub_q)
    monkeypatch.setattr(cm, "capacity_manager", stub_cap)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

    with pytest.raises(HTTPException) as exc:
        anyio.run(
            sampling_route.asample,
            _sample_request(),
            _dummy_request(
                user_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            ),
        )
    assert exc.value.status_code == 429
    assert exc.value.detail == {
        "code": "sampling_principal_backpressure",
        "scope": "api_key",
        "limit": 1,
        "pending": 1,
        "message": "Sampling backpressure: principal budget exhausted",
    }
    assert metric_calls == [
        {
            "route": "/api/v1/asample",
            "decision": "rejected",
            "reason": "queue_throttled",
            "scope": "api_key",
        }
    ]


def test_issue_324_compute_logprobs_records_capacity_rejection_metric(monkeypatch):
    metric_calls: list[dict] = []

    class _RejectingCapacityManager:
        def try_reserve(self, request_id: str, queue_bytes: int, object_store_bytes: int) -> dict:
            _ = (request_id, queue_bytes, object_store_bytes)
            return {"ok": False, "queue_bytes": 123, "object_store_bytes": 456}

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(
        sampling_route,
        "record_sampling_admission_metric",
        lambda **kwargs: metric_calls.append(dict(kwargs)),
    )

    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.result_size_estimator as rse

    monkeypatch.setattr(cm, "capacity_manager", _RejectingCapacityManager())
    monkeypatch.setattr(rse, "estimate_compute_logprobs_result_bytes", lambda _req: 0)

    request = sampling_route.ComputeLogprobsRequest(
        sampling_session_id="sess",
        seq_id=1,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )

    with pytest.raises(HTTPException) as exc:
        anyio.run(
            sampling_route.compute_logprobs,
            request,
            _dummy_request(
                user_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            ),
        )

    assert exc.value.status_code == 429
    assert metric_calls == [
        {
            "route": "/api/v1/compute_logprobs",
            "decision": "rejected",
            "reason": "capacity_rejected",
        }
    ]


def test_issue_324_unwrap_queue_throttle_error_from_ray_wrapper():
    expected = ApiWorkQueueThrottleError(scope="api_key", limit=1, pending=1)

    class _Wrapped(Exception):
        def as_instanceof_cause(self):
            return expected

    unwrapped = awq._unwrap_queue_throttle_error(_Wrapped("wrapped"))
    assert unwrapped is expected
