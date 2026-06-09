from __future__ import annotations

import asyncio
import sys
import time
import types
from types import SimpleNamespace

import anyio
import pytest
from fastapi import HTTPException

from mint_server import app as app_module
from mint_server.backend.task_state_store import FutureStatus
from mint_server.backend.session_manager import SessionManager
from mint_server.models.types import SampleResponse, SampledSequence
from mint_server.routes import sampling as sampling_route
from mint_server.routes import service as service_route


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeRegistry:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    async def get_lora_id(self, session_id: str) -> int | None:
        return self.ids.get(session_id)


class _FakeEngine:
    def __init__(self) -> None:
        self.registry = _FakeRegistry()
        self.restore_calls: list[tuple[str, str, int]] = []
        self.generate_calls: list[dict] = []
        self.logprob_calls: list[dict] = []
        self.actor_name = "actor-364"

    async def restore_loaded_session(
        self,
        *,
        sampling_session_id: str,
        adapter_path: str,
        lora_int_id: int,
    ) -> int:
        self.restore_calls.append((sampling_session_id, adapter_path, lora_int_id))
        self.registry.ids[sampling_session_id] = int(lora_int_id)
        return int(lora_int_id)

    async def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        return SimpleNamespace(
            token_ids=[101, 102],
            logprobs=None,
            routed_experts=None,
            stop_reason="length",
        )

    async def compute_logprobs(self, **kwargs):
        self.logprob_calls.append(dict(kwargs))
        return [None, -0.1, -0.2]


class _FakeMultiModelManager:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine
        self.calls: list[str] = []

    async def get_engine(self, model_name: str) -> _FakeEngine:
        self.calls.append(model_name)
        return self.engine


def _request_stub(user_id: str | None = None):
    user_data = None if user_id is None else {"user_id": user_id}
    return SimpleNamespace(state=SimpleNamespace(user_data=user_data), headers={})


class _FakeModelActorInventory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def mark_inflight(self, actor_name: str, delta: int) -> None:
        self.calls.append((actor_name, int(delta)))


def _install_fake_model_actor_inventory(monkeypatch: pytest.MonkeyPatch, pool: _FakeModelActorInventory) -> None:
    module = types.ModuleType("mint_server.backend.model_actor_supervisor")
    module.get_model_actor_supervisor = lambda: pool
    monkeypatch.setitem(sys.modules, "mint_server.backend.model_actor_supervisor", module)


def test_issue_364_register_multi_lora_session_persists_detached_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    persisted: list[dict] = []

    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.upsert_sampling_session",
        lambda info: persisted.append(dict(info)),
    )

    manager = SessionManager()
    manager.register_multi_lora_session(
        session_id="sess-364",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=64,
        adapter_path="/tmp/adapter-364",
        lora_loaded=False,
    )
    manager.mark_session_lora_loaded("sess-364", True, lora_int_id=17)

    assert persisted[0]["session_id"] == "sess-364"
    assert persisted[0]["metadata_version"] == 1
    assert persisted[0]["lora_loaded"] is False
    assert persisted[-1]["metadata_version"] == 2
    assert persisted[-1]["lora_loaded"] is True
    assert persisted[-1]["lora_int_id"] == 17


def test_issue_364_get_engine_for_session_rehydrates_loaded_lora_mapping() -> None:
    manager = SessionManager()
    engine = _FakeEngine()
    multi_model_manager = _FakeMultiModelManager(engine)
    manager.set_multi_model_manager(multi_model_manager)
    manager.restore_sampling_session(
        {
            "session_id": "sess-364-restore",
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "lora_rank": 32,
            "adapter_path": "/shared/adapter-364",
            "lora_loaded": True,
            "lora_int_id": 9,
            "metadata_version": 4,
            "uses_base_model": False,
            "last_activity": 123.0,
        }
    )

    restored_engine = asyncio.run(manager.get_engine_for_session("sess-364-restore"))

    assert restored_engine is engine
    assert multi_model_manager.calls == ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert engine.restore_calls == [("sess-364-restore", "/shared/adapter-364", 9)]
    assert manager.get_session_lora_int_id("sess-364-restore") == 9
    snapshot = manager.get_sampling_session_snapshot("sess-364-restore")
    assert snapshot is not None
    assert snapshot.metadata_version == 4
    assert snapshot.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_issue_364_get_engine_for_session_reuses_existing_loaded_lora_mapping() -> None:
    manager = SessionManager()
    engine = _FakeEngine()
    engine.registry.ids["sess-364-existing"] = 23
    multi_model_manager = _FakeMultiModelManager(engine)
    manager.set_multi_model_manager(multi_model_manager)
    manager.restore_sampling_session(
        {
            "session_id": "sess-364-existing",
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "lora_rank": 32,
            "adapter_path": "/shared/adapter-existing",
            "lora_loaded": True,
            "lora_int_id": 9,
            "metadata_version": 5,
            "uses_base_model": False,
            "last_activity": 456.0,
        }
    )

    restored_engine = asyncio.run(manager.get_engine_for_session("sess-364-existing"))

    assert restored_engine is engine
    assert multi_model_manager.calls == ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert engine.restore_calls == []
    assert manager.get_session_lora_int_id("sess-364-existing") == 23


@pytest.mark.anyio
async def test_issue_364_app_restore_sampling_sessions_reads_detached_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager()

    async def _async_list_sampling_sessions():
        return [
            {
                "session_id": "sess-364-base",
                "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                "metadata_version": 2,
                "uses_base_model": True,
                "last_activity": 1.0,
            },
            {
                "session_id": "sess-364-lora",
                "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "metadata_version": 7,
                "lora_rank": 32,
                "adapter_path": "/shared/lora-364",
                "lora_loaded": True,
                "lora_int_id": 4,
                "uses_base_model": False,
                "last_activity": 2.0,
            },
        ]

    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.async_list_sampling_sessions",
        _async_list_sampling_sessions,
    )

    restored = await app_module._restore_sampling_sessions(manager)

    assert restored == 2
    assert manager.is_base_model_session("sess-364-base") is True
    assert manager.get_session_base_model("sess-364-lora") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert manager.get_session_lora_int_id("sess-364-lora") == 4
    snapshot = manager.get_sampling_session_snapshot("sess-364-lora")
    assert snapshot is not None
    assert snapshot.metadata_version == 7


def test_issue_364_get_session_no_longer_uses_process_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _async_get_session_index(_session_id: str):
        return None

    monkeypatch.setattr(
        "mint_server.backend.session_index_store.async_get_session_index",
        _async_get_session_index,
    )
    monkeypatch.setattr(
        service_route,
        "sessions",
        {"sess-local-only": {"user_id": "admin", "created_at": "2026-03-25T00:00:00"}},
        raising=False,
    )

    with pytest.raises(HTTPException, match="not found"):
        anyio.run(service_route.get_session, "sess-local-only", _request_stub("admin"))


def test_issue_364_list_sessions_no_longer_uses_process_local_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _async_list_session_index():
        return []

    monkeypatch.setattr(
        "mint_server.backend.session_index_store.async_list_session_index",
        _async_list_session_index,
    )
    monkeypatch.setattr(
        service_route,
        "sessions",
        {"sess-local-only": {"user_id": "admin", "created_at": "2026-03-25T00:00:00"}},
        raising=False,
    )

    out = anyio.run(service_route.list_sessions, 20, 0, _request_stub("admin"))

    assert out.sessions == []


def test_issue_364_end_session_cleans_sampler_index_and_parent_session_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager()
    manager.register_base_model_session(
        "sess-364-cleanup",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )

    calls: list[tuple[str, str, str | None]] = []
    detached_info = {
        "session_id": "sess-364-cleanup",
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "uses_base_model": True,
        "metadata_version": 1,
        "last_activity": 1.0,
    }

    monkeypatch.setattr(
        "mint_server.backend.session_index_store.get_sampler_index",
        lambda sampler_id: {
            "sampler_id": sampler_id,
            "session_id": "parent-session-364",
        },
    )
    monkeypatch.setattr(
        "mint_server.backend.session_index_store.delete_sampler_index",
        lambda sampler_id: calls.append(("delete", sampler_id, None)),
    )
    monkeypatch.setattr(
        "mint_server.backend.session_index_store.remove_sampler_from_session",
        lambda session_id, sampler_id: calls.append(("unlink", sampler_id, session_id)),
    )
    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.get_sampling_session_info",
        lambda session_id: dict(detached_info) if session_id == detached_info.get("session_id") else None,
    )

    def _delete_sampling_session(session_id: str) -> None:
        calls.append(("sampling-store", session_id, None))
        detached_info.clear()

    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.delete_sampling_session",
        _delete_sampling_session,
    )

    ended = anyio.run(manager.end_session, "sess-364-cleanup")

    assert ended is True
    assert calls == [
        ("sampling-store", "sess-364-cleanup", None),
        ("delete", "sess-364-cleanup", None),
        ("unlink", "sess-364-cleanup", "parent-session-364"),
    ]
    assert manager.get_session_base_model("sess-364-cleanup") is None


@pytest.mark.anyio
async def test_issue_364_sample_once_uses_detached_store_and_scheduler_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mint_server.backend.model_work_scheduler as mws

    async def _async_get_sampling_session_info(session_id: str):
        assert session_id == "sess-364-live"
        return {
            "session_id": session_id,
            "base_model": "Qwen/Qwen3-0.6B",
            "metadata_version": 3,
            "uses_base_model": True,
            "last_activity": 1.0,
        }

    async def _async_remote_sampling_session(_session_id: str):
        raise AssertionError("local detached sampler should not be routed as remote")

    from mint_server.routes import futures as futures_route

    captured: dict = {}
    futures: dict[str, SampleResponse] = {}

    class _FakeScheduler:
        async def append(self, **kwargs):
            captured.update(kwargs)
            request_id = str(kwargs["request_id"])
            futures[request_id] = SampleResponse(
                sequences=[SampledSequence(tokens=[101, 102], logprobs=None, stop_reason="length")]
            )
            return {"ok": True, "scheduler_instance_id": "scheduler-364"}

    class _FakeTaskFutures:
        async def async_ensure_pending(self, request_id: str, meta=None) -> dict:
            return {"created": True, "meta": dict(meta or {}), "request_id": request_id}

        async def async_update_meta(self, _request_id: str, meta=None) -> None:
            return None

        async def async_get_status(self, request_id: str) -> FutureStatus:
            assert request_id in futures
            return FutureStatus.DONE

        async def async_get_result(self, request_id: str):
            return futures[request_id].model_dump()

    fake_task_futures = _FakeTaskFutures()
    monkeypatch.setattr(sampling_route, "task_futures", fake_task_futures)
    monkeypatch.setattr(futures_route, "task_futures", fake_task_futures)
    monkeypatch.setattr(mws, "model_work_scheduler", _FakeScheduler())
    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.async_get_sampling_session_info",
        _async_get_sampling_session_info,
    )
    monkeypatch.setattr(
        "mint_server.gateway.async_remote_sampling_session",
        _async_remote_sampling_session,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_registry.get_model_config",
        lambda _model_name: SimpleNamespace(max_model_len=8192),
    )

    sequence = await sampling_route.sample_once(
        session_id="sess-364-live",
        token_ids=[1, 2, 3],
        max_tokens=4,
        temperature=1.0,
        top_p=1.0,
        stop=None,
        request_id="req-364-sample",
        http_request=_request_stub(),
        user_id=None,
    )

    assert captured["op"] == "sampling.asample"
    assert captured["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert captured["affinity_group"] == "base:Qwen/Qwen3-0.6B"
    assert captured["ordering_key"] == "session:sess-364-live"
    assert sequence.tokens == [101, 102]


@pytest.mark.anyio
async def test_issue_364_refreshes_local_sampler_on_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager()
    manager.register_base_model_session(
        "sess-364-stale",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        metadata_version=1,
    )

    async def _async_get_sampling_session_info(session_id: str):
        assert session_id == "sess-364-stale"
        return {
            "session_id": session_id,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "metadata_version": 2,
            "uses_base_model": True,
            "last_activity": 5.0,
        }

    monkeypatch.setattr(sampling_route, "session_manager", manager)
    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.async_get_sampling_session_info",
        _async_get_sampling_session_info,
    )

    await sampling_route._restore_local_sampling_session_if_needed("sess-364-stale")

    snapshot = manager.get_sampling_session_snapshot("sess-364-stale")
    assert snapshot is not None
    assert snapshot.metadata_version == 2
    assert snapshot.base_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_issue_364_restore_sampling_session_merges_last_activity_without_version_bump() -> None:
    manager = SessionManager()
    manager.register_multi_lora_session(
        session_id="sess-364-activity",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=8,
        adapter_path="/tmp/lora-364",
        lora_loaded=True,
        lora_int_id=7,
        last_activity=10.0,
        metadata_version=3,
    )

    restored = manager.restore_sampling_session(
        {
            "session_id": "sess-364-activity",
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "lora_rank": 8,
            "adapter_path": "/tmp/lora-364",
            "lora_loaded": True,
            "lora_int_id": 7,
            "metadata_version": 3,
            "last_activity": 25.0,
            "uses_base_model": False,
        }
    )

    assert restored is True
    snapshot = manager.get_sampling_session_snapshot("sess-364-activity")
    assert snapshot is not None
    info = manager._sessions["sess-364-activity"]
    assert info.last_activity >= 25.0
    assert snapshot.metadata_version == 3


def test_issue_364_model_actor_inventory_wrapper_preserves_metadata_without_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.model_actor_inventory as model_actor_inventory_module
    from mint_server.backend.model_actor_supervisor import ActorType, ModelActorSupervisor
    monkeypatch.setattr(model_actor_inventory_module.ray, "is_initialized", lambda: False)
    pool = ModelActorSupervisor()
    pool.clear(kill_actors=False)
    actor_name = "actor-364-wrapper-local"
    pool.unregister(actor_name)
    pool.register(
        actor_name=actor_name,
        actor_type=ActorType.DENSE,
        num_gpus=1,
        namespace="mint",
        base_model="Qwen/Qwen3-0.6B",
        session_id="session-364-local",
        metadata={"max_lora_rank": 64},
    )
    pool.mark_ready(actor_name)

    listed = [entry for entry in pool.list_actors() if entry["actor_name"] == actor_name]
    assert len(listed) == 1
    assert listed[0]["metadata"]["max_lora_rank"] == 64
    assert listed[0]["backend"] == "peft"

    pool.clear_session("session-364-local", actor_type=ActorType.DENSE)
    entry = pool.get(actor_name)
    assert entry is not None
    assert entry.current_session is None
    pool.unregister(actor_name)


@pytest.mark.anyio
async def test_issue_364_save_weights_for_sampler_persists_lora_int_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from mint_server.models.types import SaveWeightsForSamplerRequest
    from mint_server.routes import training as training_route

    ckpt_dir = tmp_path / "issue364_ephemeral_sampler"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    registered: dict = {}
    resolved: dict = {}

    class _FakeInferenceEngine:
        async def add_lora_for_session_from_path(self, *, sampling_session_id: str, lora_path: str) -> int:
            assert sampling_session_id
            assert lora_path == str(ckpt_dir)
            return 41

    class _FakeInferenceManager:
        tensor_parallel_size = 1
        data_parallel_size = 1
        gpu_memory_utilization = 0.8
        max_model_len = 8192

        async def get_engine_for_model(self, model_name: str):
            assert model_name == "Qwen/Qwen3-4B-Instruct-2507"
            return _FakeInferenceEngine()

        def register_multi_lora_session(self, **kwargs) -> None:
            registered.update(kwargs)

    session = SimpleNamespace(
        model_id="run-364",
        session_id="parent-364",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        current_step=13,
        backend="peft",
        lora_config=SimpleNamespace(rank=8, train_mlp=False),
        inference_engine=None,
    )

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    async def _identity_materialize(session_arg):
        return session_arg

    monkeypatch.setattr(training_route, "_materialize_training_session_for_stateful_use", _identity_materialize)
    monkeypatch.setattr(
        training_route,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: session,
            mark_inflight=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        training_route,
        "training_engine",
        SimpleNamespace(save_weights_for_sampler=_fake_save_weights_for_sampler),
    )
    monkeypatch.setattr(training_route, "inference_manager", _FakeInferenceManager())
    async def _async_resolve(request_id: str, response, **_kwargs):
        resolved.update({"request_id": request_id, "response": response})

    async def _async_fail(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        training_route,
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )
    monkeypatch.setattr(training_route, "checkpoint_has_optimizer_state", lambda _path: False)
    monkeypatch.setattr(training_route, "validate_sampler_checkpoint_for_sampling", lambda _path: None)
    monkeypatch.setattr(training_route, "write_checkpoint_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(training_route, "build_ephemeral_checkpoint_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(training_route, "begin_async_checkpoint_mirror", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "mint_server.client_compat.checkpoint_uri",
        lambda *_args, **_kwargs: "mint://ckpt/run-364/_ephemeral_364",
    )
    monkeypatch.setattr(
        "mint_server.backend.session_index_store.add_sampler_to_session",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "mint_server.backend.session_index_store.add_heartbeat_sampler_to_session",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "mint_server.backend.session_index_store.upsert_sampler_index",
        lambda _payload: None,
    )

    request = SaveWeightsForSamplerRequest(model_id="run-364", seq_id=0, path=None)
    await training_route._do_save_weights_for_sampler(
        request_id="req-364-save-sampler",
        request=request,
        user_id="owner-364",
        prefer_tinker=True,
    )

    assert registered["lora_loaded"] is False
    assert "lora_int_id" not in registered
    assert registered["adapter_path"] == str(ckpt_dir)
    assert resolved["request_id"] == "req-364-save-sampler"
    assert resolved["response"]["sampling_session_id"] is not None


@pytest.mark.anyio
async def test_issue_364_owner_cleanup_respects_detached_sampling_last_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(inactivity_timeout=1)
    manager.register_multi_lora_session(
        session_id="sess-364-cleanup-live",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=8,
        adapter_path="/tmp/lora-364-live",
        lora_loaded=True,
        lora_int_id=9,
        last_activity=0.0,
        metadata_version=3,
    )
    ended: list[str] = []

    async def _async_get_sampling_session_info(session_id: str):
        assert session_id == "sess-364-cleanup-live"
        return {
            "session_id": session_id,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "lora_rank": 8,
            "adapter_path": "/tmp/lora-364-live",
            "lora_loaded": True,
            "lora_int_id": 9,
            "metadata_version": 3,
            "last_activity": time.time(),
            "uses_base_model": False,
        }

    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.async_get_sampling_session_info",
        _async_get_sampling_session_info,
    )

    async def _end_session(session_id: str):
        ended.append(session_id)
        return True

    monkeypatch.setattr(manager, "end_session", _end_session)

    await manager._cleanup_inactive()

    assert ended == []


@pytest.mark.anyio
async def test_issue_364_sampling_restore_drops_stale_local_snapshot_when_store_entry_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager()
    manager.register_base_model_session(
        "sess-364-gone",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        metadata_version=2,
    )

    monkeypatch.setattr(sampling_route, "session_manager", manager)
    async def _async_get_sampling_session_info(_session_id: str):
        return None

    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.async_get_sampling_session_info",
        _async_get_sampling_session_info,
    )
    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.get_sampling_session_info",
        lambda _session_id: None,
    )
    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.delete_sampling_session",
        lambda _session_id: None,
    )

    restored = await sampling_route._restore_local_sampling_session_if_needed("sess-364-gone")

    assert restored is False
    assert manager.get_sampling_session_snapshot("sess-364-gone") is None


@pytest.mark.anyio
async def test_issue_364_asample_missing_detached_sampler_is_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _async_get_sampling_session_info(_session_id: str):
        return None

    async def _async_remote_sampling_session(_session_id: str):
        return None

    monkeypatch.setattr(sampling_route, "session_manager", None)
    monkeypatch.setattr(
        "mint_server.backend.sampling_session_store.async_get_sampling_session_info",
        _async_get_sampling_session_info,
    )
    monkeypatch.setattr(
        "mint_server.gateway.async_remote_sampling_session",
        _async_remote_sampling_session,
    )
    monkeypatch.setattr(
        "mint_server.gateway.remote_sampling_session",
        lambda _session_id: None,
    )

    from mint_server.models.types import ModelInput, SampleRequest, SamplingParams

    req = SampleRequest(
        sampling_session_id="sess-364-missing",
        seq_id=1,
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )

    with pytest.raises(HTTPException) as exc_info:
        await sampling_route.asample(req, SimpleNamespace(state=SimpleNamespace(user_data=None), headers={}))

    assert exc_info.value.status_code == 404
    assert "sess-364-missing" in str(exc_info.value.detail)


@pytest.mark.anyio
async def test_issue_364_compute_logprobs_marks_model_actor_inventory_inflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mint_server.models.types import ComputeLogprobsRequest, ModelInput

    manager = SessionManager()
    engine = _FakeEngine()
    manager.set_multi_model_manager(_FakeMultiModelManager(engine))
    manager.register_base_model_session(
        "sess-364-logprobs",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        metadata_version=3,
    )
    model_actor_inventory = _FakeModelActorInventory()
    resolved: dict = {}
    failed: list[str] = []

    monkeypatch.setattr(sampling_route, "session_manager", manager)
    async def _async_resolve(request_id: str, response, **_kwargs):
        resolved.update({"request_id": request_id, "response": response})

    async def _async_fail(_request_id: str, error: str):
        failed.append(str(error))

    monkeypatch.setattr(
        sampling_route,
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )
    _install_fake_model_actor_inventory(monkeypatch, model_actor_inventory)

    request = ComputeLogprobsRequest(
        sampling_session_id="sess-364-logprobs",
        seq_id=0,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )
    await sampling_route._do_compute_logprobs(
        request_id="req-364-logprobs",
        request=request,
        user_id=None,
    )

    assert failed == []
    assert resolved["request_id"] == "req-364-logprobs"
    assert model_actor_inventory.calls == [("actor-364", 1), ("actor-364", -1)]
