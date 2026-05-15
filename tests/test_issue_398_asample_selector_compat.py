from types import SimpleNamespace

import anyio
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tinker_server.models.types import ModelInput, SampleRequest, SamplingParams
from tinker_server.routes import sampling as sampling_route
from tinker_server.routes import service as service_route


class _StubSamplingSessionManager:
    def is_multi_lora_session(self, _session_id: str) -> bool:
        return True

    def get_engine(self, _session_id: str):
        return object()

    def get_session_base_model(self, _session_id: str):
        return "Qwen/Qwen3-0.6B"

    def get_session_lora_rank(self, _session_id: str):
        return 0

    def get_session_adapter_path(self, _session_id: str):
        return None

    def is_base_model_session(self, _session_id: str):
        return True

    def get_session_metadata_version(self, _session_id: str):
        return 1


class _StubFutureStore:
    def __init__(self):
        self.pending: dict[str, dict | None] = {}
        self.marked: list[str] = []
        self.created: list[str] = []

    async def async_create_with_id(self, request_id: str):
        self.created.append(request_id)
        return request_id

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.marked.append(request_id)
        cur = self.pending.get(request_id) or {}
        if meta is not None:
            cur.update(dict(meta))
        self.pending[request_id] = cur

    async def async_update_meta(self, request_id: str, meta: dict | None = None) -> None:
        cur = self.pending.get(request_id) or {}
        if meta is not None:
            cur.update(dict(meta))
        self.pending[request_id] = cur

    async def async_cleanup(self, _request_id: str) -> None:
        return None


class _StubCapacityManager:
    def __init__(self):
        self.reserved: list[str] = []
        self.released: list[str] = []

    async def async_try_reserve(self, request_id: str, queue_bytes: int, object_store_bytes: int):
        self.reserved.append(request_id)
        return {"ok": True}

    async def async_release_all(self, request_id: str) -> None:
        self.released.append(request_id)


class _StubApiWorkQueue:
    def __init__(self):
        self.calls: list[dict] = []

    async def enqueue(self, **kwargs):
        self.calls.append(dict(kwargs))


class _StubModelWorkScheduler:
    def __init__(self):
        self.calls: list[dict] = []

    async def append(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"ok": True, "scheduler_instance_id": "scheduler-test"}

    async def cancel_request(self, **_kwargs):
        return {"ok": True}


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def test_sample_request_preserves_base_model_selector():
    req = SampleRequest(
        base_model="Qwen/Qwen3-0.6B",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    assert req.base_model == "Qwen/Qwen3-0.6B"
    assert req.model_dump()["base_model"] == "Qwen/Qwen3-0.6B"
    assert req.needs_session_creation() is True


def test_sample_request_preserves_model_path_selector():
    req = SampleRequest(
        model_path="tinker://run-123/weights/checkpoint-001",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    assert req.model_path == "tinker://run-123/weights/checkpoint-001"
    assert req.model_dump()["model_path"] == "tinker://run-123/weights/checkpoint-001"
    assert req.needs_session_creation() is True


def test_sample_request_rejects_mixed_selectors():
    with pytest.raises(ValidationError):
        SampleRequest(
            sampling_session_id="sess",
            base_model="Qwen/Qwen3-0.6B",
            num_samples=1,
            prompt=ModelInput.from_ints([1, 2, 3]),
            sampling_params=SamplingParams(max_tokens=4),
        )


def test_sample_request_rejects_seq_id_without_session_selector():
    with pytest.raises(ValidationError):
        SampleRequest(
            base_model="Qwen/Qwen3-0.6B",
            seq_id=1,
            num_samples=1,
            prompt=ModelInput.from_ints([1, 2, 3]),
            sampling_params=SamplingParams(max_tokens=4),
        )


@pytest.mark.parametrize(
    ("selector_field", "selector_value"),
    [
        ("base_model", "Qwen/Qwen3-0.6B"),
        ("model_path", "tinker://run-123/weights/checkpoint-001"),
    ],
)
def test_asample_normalizes_direct_selector_before_enqueue(monkeypatch, selector_field: str, selector_value: str):
    stub_fs = _StubFutureStore()
    stub_scheduler = _StubModelWorkScheduler()
    created_sessions: list[tuple[str, str]] = []

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", stub_fs)

    import tinker_server.backend.model_registry as model_registry
    import tinker_server.backend.model_work_scheduler as mws

    monkeypatch.setattr(model_registry, "get_model_config", lambda _model: SimpleNamespace(max_model_len=4096))
    monkeypatch.setattr(mws, "model_work_scheduler", stub_scheduler)

    async def _ensure_sampling_session(*, model_path: str, http_request, parent_session_id: str | None = None):
        created_sessions.append((model_path, parent_session_id or ""))
        return "sess-created", "Qwen/Qwen3-0.6B"

    monkeypatch.setattr(service_route, "ensure_sampling_session", _ensure_sampling_session)

    req = SampleRequest(
        **{selector_field: selector_value},
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    out = anyio.run(sampling_route.asample, req, _dummy_request("u"))

    assert isinstance(out.request_id, str) and out.request_id
    assert created_sessions == [(selector_value, "")]
    assert len(stub_scheduler.calls) == 1
    payload = SampleRequest.model_validate_json(stub_scheduler.calls[0]["request_json"])
    assert payload.sampling_session_id == "sess-created"
    assert payload.model_id is None
    assert payload.base_model is None
    assert payload.model_path is None


def test_asample_keeps_seq_id_gate_for_existing_session_selector(monkeypatch):
    stub_fs = _StubFutureStore()
    stub_cap = _StubCapacityManager()
    stub_q = _StubApiWorkQueue()

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", stub_fs)

    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.result_size_estimator as rse

    monkeypatch.setattr(cm, "capacity_manager", stub_cap)
    monkeypatch.setattr(awq, "api_work_queue", stub_q)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)
    monkeypatch.setattr(sampling_route.server_config, "sampling_require_seq_id", True)

    req = SampleRequest(
        sampling_session_id="sess",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    with pytest.raises(HTTPException) as exc_info:
        anyio.run(sampling_route.asample, req, _dummy_request("u"))

    assert exc_info.value.status_code == 422
