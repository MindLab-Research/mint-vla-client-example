import anyio
from types import SimpleNamespace

from fastapi import HTTPException

from tinker_server.models.types import CreateSamplingSessionRequest, ModelInput, SampleRequest, SamplingParams
from tinker_server.routes import sampling as sampling_route
from tinker_server.routes import service as service_route


class _StubSessionManager:
    def __init__(self):
        self.sessions: dict[str, dict] = {}
        self.register_base_calls: list[str] = []

    async def get_engine_for_model(self, _base_model: str):
        return None

    def get_session_base_model(self, session_id: str):
        info = self.sessions.get(session_id)
        return None if info is None else info.get("base_model")

    def get_session_adapter_path(self, session_id: str):
        info = self.sessions.get(session_id)
        return None if info is None else info.get("adapter_path")

    def get_session_lora_rank(self, session_id: str):
        info = self.sessions.get(session_id)
        return None if info is None else info.get("lora_rank")

    def register_multi_lora_session(self, session_id: str, base_model: str, lora_rank: int, adapter_path: str | None, **_kwargs):
        self.sessions[session_id] = {
            "base_model": base_model,
            "lora_rank": int(lora_rank),
            "adapter_path": adapter_path,
        }

    def register_base_model_session(self, session_id: str, base_model: str):
        self.register_base_calls.append(session_id)
        self.sessions[session_id] = {
            "base_model": base_model,
            "lora_rank": 0,
            "adapter_path": None,
        }


class _StubSamplingSessionManager:
    def is_multi_lora_session(self, _session_id: str) -> bool:
        return False

    def get_engine(self, _session_id: str):
        return object()


class _StubFutureStore:
    def __init__(self):
        self.pending: dict[str, dict | None] = {}
        self.marked: list[str] = []
        self.created: list[str] = []
        self.forgotten: list[str] = []

    def ensure_pending(self, request_id: str, meta: dict | None = None) -> dict:
        if request_id in self.pending:
            return {"created": False, "meta": self.pending.get(request_id)}
        self.pending[request_id] = dict(meta) if meta is not None else None
        return {"created": True, "meta": None}

    def create_with_id(self, request_id: str):
        self.created.append(request_id)
        return request_id

    def mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.marked.append(request_id)
        cur = self.pending.get(request_id) or {}
        if meta is not None:
            cur.update(dict(meta))
        self.pending[request_id] = cur

    def cleanup(self, _request_id: str) -> None:
        return None

    def forget(self, request_id: str) -> None:
        self.forgotten.append(request_id)
        self.pending.pop(request_id, None)


class _StubCapacityManager:
    def __init__(self):
        self.reserved: list[str] = []
        self.released: list[str] = []

    def try_reserve(self, request_id: str, queue_bytes: int, object_store_bytes: int):
        self.reserved.append(request_id)
        return {"ok": True}

    def release_all(self, request_id: str) -> None:
        self.released.append(request_id)


class _StubApiWorkQueue:
    def __init__(self):
        self.calls: list[dict] = []

    async def enqueue(self, **kwargs):
        self.calls.append(dict(kwargs))


def _dummy_request(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


def test_create_sampling_session_deterministic_idempotent(monkeypatch):
    stub = _StubSessionManager()
    monkeypatch.setattr(service_route, "session_manager", stub)

    import tinker_server.supported_models_gate as gate
    import tinker_server.gateway as gw

    async def _allow(base_model: str, http_request=None):
        return base_model

    monkeypatch.setattr(gate, "enforce_base_model_allowed", _allow)
    monkeypatch.setattr(service_route, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(gw, "upstream_for_model", lambda _model: None)

    req = CreateSamplingSessionRequest(
        session_id="sess",
        sampling_session_seq_id=3,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )
    out1 = anyio.run(service_route.create_sampling_session, req, _dummy_request("u"))
    out2 = anyio.run(service_route.create_sampling_session, req, _dummy_request("u"))

    assert out1.sampling_session_id == "sess:sample:3"
    assert out2.sampling_session_id == "sess:sample:3"
    assert stub.register_base_calls == ["sess:sample:3"]


def test_create_sampling_session_conflict(monkeypatch):
    stub = _StubSessionManager()
    monkeypatch.setattr(service_route, "session_manager", stub)

    import tinker_server.supported_models_gate as gate
    import tinker_server.gateway as gw

    async def _allow(base_model: str, http_request=None):
        return base_model

    monkeypatch.setattr(gate, "enforce_base_model_allowed", _allow)
    monkeypatch.setattr(service_route, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(gw, "upstream_for_model", lambda _model: None)

    req1 = CreateSamplingSessionRequest(
        session_id="sess",
        sampling_session_seq_id=7,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )
    anyio.run(service_route.create_sampling_session, req1, _dummy_request("u"))

    req2 = CreateSamplingSessionRequest(
        session_id="sess",
        sampling_session_seq_id=7,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    try:
        anyio.run(service_route.create_sampling_session, req2, _dummy_request("u"))
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("expected HTTPException")


def test_asample_deterministic_request_id_dedup(monkeypatch):
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

    req = SampleRequest(
        sampling_session_id="sess",
        seq_id=9,
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    out1 = anyio.run(sampling_route.asample, req, _dummy_request("u"))
    out2 = anyio.run(sampling_route.asample, req, _dummy_request("u"))

    expected = sampling_route._deterministic_request_id("sess", 9)
    assert out1.request_id == expected
    assert out2.request_id == expected
    assert stub_cap.reserved == [expected]
    assert len(stub_q.calls) == 1


def test_asample_duplicate_payload_conflict(monkeypatch):
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

    req1 = SampleRequest(
        sampling_session_id="sess",
        seq_id=10,
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    req2 = SampleRequest(
        sampling_session_id="sess",
        seq_id=10,
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=5),
    )

    anyio.run(sampling_route.asample, req1, _dummy_request("u"))
    try:
        anyio.run(sampling_route.asample, req2, _dummy_request("u"))
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("expected HTTPException")


def test_asample_requires_seq_id_when_enabled(monkeypatch):
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
        seq_id=None,
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    try:
        anyio.run(sampling_route.asample, req, _dummy_request("u"))
    except HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("expected HTTPException")
