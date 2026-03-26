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

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", stub_fs)

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


def test_issue_324_unwrap_queue_throttle_error_from_ray_wrapper():
    expected = ApiWorkQueueThrottleError(scope="api_key", limit=1, pending=1)

    class _Wrapped(Exception):
        def as_instanceof_cause(self):
            return expected

    unwrapped = awq._unwrap_queue_throttle_error(_Wrapped("wrapped"))
    assert unwrapped is expected
