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

    async def async_ensure_pending(self, request_id: str, meta: dict | None = None) -> dict:
        if request_id in self.pending:
            return {"created": False, "meta": self.pending.get(request_id)}
        self.pending[request_id] = dict(meta) if meta is not None else None
        return {"created": True, "meta": None}

    async def async_create_with_id(self, request_id: str):
        self.created.append(request_id)
        return request_id

    async def async_mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.marked.append(request_id)
        cur = self.pending.get(request_id) or {}
        if meta is not None:
            cur.update(dict(meta))
        self.pending[request_id] = cur

    def mark_queued(self, request_id: str, meta: dict | None = None) -> None:
        self.marked.append(request_id)
        cur = self.pending.get(request_id) or {}
        if meta is not None:
            cur.update(dict(meta))
        self.pending[request_id] = cur

    async def async_get_status(self, request_id: str) -> str:
        if request_id in self.pending:
            return "PENDING"
        raise KeyError(f"Unknown request_id: {request_id}")

    async def async_cleanup(self, _request_id: str) -> None:
        return None

    async def async_forget(self, request_id: str) -> None:
        self.forgotten.append(request_id)
        self.pending.pop(request_id, None)


class _StubCapacityManager:
    def __init__(self):
        self.reserved: list[str] = []
        self.released: list[str] = []

    async def async_try_reserve(self, request_id: str, queue_bytes: int, object_store_bytes: int):
        self.reserved.append(request_id)
        return {"ok": True}

    async def async_release_all(self, request_id: str) -> None:
        self.released.append(request_id)

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


def _install_detached_sampling_store(monkeypatch):
    detached_sessions: dict[str, dict] = {}

    import tinker_server.backend.sampling_session_store as sss
    import tinker_server.backend.session_index_store as sis

    def _upsert_sampling_session(info: dict) -> None:
        detached_sessions[str(info["session_id"])] = dict(info)

    async def _async_get_sampling_session_info(session_id: str):
        info = detached_sessions.get(str(session_id))
        return None if info is None else dict(info)

    monkeypatch.setattr(sss, "upsert_sampling_session", _upsert_sampling_session)
    monkeypatch.setattr(sss, "async_get_sampling_session_info", _async_get_sampling_session_info)
    monkeypatch.setattr(sis, "add_sampler_to_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sis, "upsert_sampler_index", lambda *_args, **_kwargs: None)
    return detached_sessions


def test_create_sampling_session_deterministic_idempotent(monkeypatch):
    stub = _StubSessionManager()
    monkeypatch.setattr(service_route, "session_manager", stub)
    _install_detached_sampling_store(monkeypatch)

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
    _install_detached_sampling_store(monkeypatch)

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


def test_create_sampling_session_keeps_generic_samplers_out_of_heartbeat_fanout(monkeypatch):
    stub = _StubSessionManager()
    sampler_calls: list[tuple[str, str, str | None, str | None]] = []
    heartbeat_calls: list[tuple[str, str, str | None, str | None]] = []
    sampler_index_updates: list[dict] = []
    detached_sampling_updates = _install_detached_sampling_store(monkeypatch)

    monkeypatch.setattr(service_route, "session_manager", stub)

    import tinker_server.backend.session_index_store as sis
    import tinker_server.supported_models_gate as gate
    import tinker_server.gateway as gw

    async def _allow(base_model: str, http_request=None):
        return base_model

    monkeypatch.setattr(gate, "enforce_base_model_allowed", _allow)
    monkeypatch.setattr(service_route, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(gw, "upstream_for_model", lambda _model: None)
    monkeypatch.setattr(
        sis,
        "add_sampler_to_session",
        lambda session_id, sampler_id, user_id=None, created_at=None: sampler_calls.append(
            (session_id, sampler_id, user_id, created_at)
        ),
    )
    monkeypatch.setattr(
        sis,
        "add_heartbeat_sampler_to_session",
        lambda session_id, sampler_id, user_id=None, created_at=None: heartbeat_calls.append(
            (session_id, sampler_id, user_id, created_at)
        ),
    )
    monkeypatch.setattr(sis, "upsert_sampler_index", sampler_index_updates.append)
    req = CreateSamplingSessionRequest(
        session_id="sess",
        sampling_session_seq_id=11,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )
    out = anyio.run(service_route.create_sampling_session, req, _dummy_request("u"))

    assert out.sampling_session_id == "sess:sample:11"
    assert sampler_calls == [("sess", "sess:sample:11", "u", sampler_calls[0][3])]
    assert heartbeat_calls == []
    assert detached_sampling_updates == {
        "sess:sample:11": {
            "session_id": "sess:sample:11",
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "lora_rank": 0,
            "adapter_path": None,
            "lora_loaded": False,
            "lora_int_id": None,
            "uses_base_model": True,
            "last_activity": detached_sampling_updates["sess:sample:11"]["last_activity"],
            "inflight_requests": 0,
            "metadata_version": 1,
        }
    }
    assert sampler_index_updates == [
        {
            "sampler_id": "sess:sample:11",
            "session_id": "sess",
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "user_id": "u",
            "created_at": sampler_calls[0][3],
            "source_type": "base_model",
        }
    ]


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


def test_asample_sets_deterministic_request_id_in_logging_context_first(monkeypatch):
    stub_fs = _StubFutureStore()
    stub_cap = _StubCapacityManager()
    stub_q = _StubApiWorkQueue()
    request_id_bindings: list[str] = []

    monkeypatch.setattr(sampling_route, "session_manager", _StubSamplingSessionManager())
    monkeypatch.setattr(sampling_route, "future_store", stub_fs)
    monkeypatch.setattr(sampling_route, "set_request_id", lambda rid: request_id_bindings.append(rid))

    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.result_size_estimator as rse

    monkeypatch.setattr(cm, "capacity_manager", stub_cap)
    monkeypatch.setattr(awq, "api_work_queue", stub_q)
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)

    req = SampleRequest(
        sampling_session_id="sess",
        seq_id=42,
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    out = anyio.run(sampling_route.asample, req, _dummy_request("u"))
    expected = sampling_route._deterministic_request_id("sess", 42)
    assert out.request_id == expected
    assert request_id_bindings == [expected]


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
