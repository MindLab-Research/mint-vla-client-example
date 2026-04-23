from __future__ import annotations

from types import SimpleNamespace

import pytest

from tinker_server.backend.training_session_manager import (
    MATERIALIZATION_STATE_READY,
    MATERIALIZATION_STATE_UNMATERIALIZED,
    TrainingSessionManager,
)
from tinker_server.models.types import CreateModelRequest, ForwardBackwardInput, ForwardBackwardRequest, LoRAConfig
from tinker_server.routes import training as training_route


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_issue_517_do_create_model_persists_unmaterialized_session_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    persisted: dict[str, object] = {}
    resolved: dict[str, object] = {}
    create_calls: list[str] = []

    async def _unexpected_create_training_session(_session) -> None:
        create_calls.append("called")

    async def _async_upsert_training_session(info: dict[str, object]) -> None:
        persisted.update(info)

    async def _best_effort_tokenizer_metadata(_session):
        return {
            "tokenizer_source_path": "/hf/snapshots/qwen3",
            "tokenizer_identity": "/hf/snapshots/qwen3#meta",
            "tokenizer_info": {"vocab_size": 151936},
        }

    monkeypatch.setattr(manager, "get_session", lambda model_id: manager.get_local_session(model_id))
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(
        training_route,
        "training_engine",
        SimpleNamespace(create_training_session=_unexpected_create_training_session),
    )
    monkeypatch.setattr(training_route, "_best_effort_local_tokenizer_metadata_for_session", _best_effort_tokenizer_metadata)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(
        "tinker_server.backend.session_index_store.add_training_run_to_session",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        training_route,
        "future_store",
        SimpleNamespace(resolve=lambda request_id, payload: resolved.update({"request_id": request_id, "payload": payload}), async_fail=lambda *_args, **_kwargs: None),
    )

    req = CreateModelRequest(
        session_id="s517",
        model_seq_id=0,
        base_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        lora_config=LoRAConfig(rank=8),
    )
    await training_route._do_create_model("rid-517-create", req, user_id="owner-a", webhook_url=None)

    assert create_calls == []
    assert persisted["materialization_state"] == MATERIALIZATION_STATE_UNMATERIALIZED
    assert persisted["actor_name"] is None
    assert persisted["backend"] == "megatron"
    assert persisted["tokenizer_info"] == {"vocab_size": 151936}
    assert resolved["request_id"] == "rid-517-create"
    assert resolved["payload"]["model_id"] == "s517_0"
    assert resolved["payload"]["backend"] == "megatron"
    session = manager.get_local_session("s517_0")
    assert session is not None
    assert session.materialization_state == MATERIALIZATION_STATE_UNMATERIALIZED
    assert session.pending_persist is False


@pytest.mark.anyio
async def test_issue_517_forward_backward_materializes_unmaterialized_session_on_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "run-517",
            "session_id": "session-517",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "metadata_version": 2,
            "materialization_state": MATERIALIZATION_STATE_UNMATERIALIZED,
        }
    )
    persisted: list[dict[str, object]] = []
    resolved: dict[str, object] = {}
    order: list[str] = []

    async def _create_training_session(session) -> None:
        order.append("create")
        session.is_active = True
        engine._resource_pool_actor_names[session.model_id] = "actor-517"

    async def _forward_backward(session, request):
        order.append("forward_backward")
        assert session.is_active is True
        assert session.materialization_state == MATERIALIZATION_STATE_READY
        assert request.model_id == "run-517"
        return {"type": "mint_forward_backward", "metrics": {"loss:mean": 0.0}}

    async def _collect_tokenizer_metadata(session):
        return {
            "tokenizer_source_path": "/hf/snapshots/qwen3",
            "tokenizer_identity": "/hf/snapshots/qwen3#ready",
            "tokenizer_info": {"vocab_size": 151936, "bos_token_id": 151643},
        }

    async def _async_upsert_training_session(info: dict[str, object]) -> None:
        persisted.append(dict(info))

    async def _async_fail(*_args, **_kwargs) -> None:
        raise AssertionError("forward_backward should not fail")

    engine = SimpleNamespace(
        _workers={},
        _resource_pool_actor_names={},
        create_training_session=_create_training_session,
        forward_backward=_forward_backward,
    )

    monkeypatch.setattr(manager, "get_session", lambda model_id: manager.get_local_session(model_id))
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_collect_control_plane_tokenizer_metadata", _collect_tokenizer_metadata)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(training_route, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(
        training_route,
        "future_store",
        SimpleNamespace(resolve=lambda request_id, payload: resolved.update({"request_id": request_id, "payload": payload}), async_fail=_async_fail),
    )

    req = ForwardBackwardRequest(
        model_id="run-517",
        forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )
    await training_route._do_forward_backward("rid-517-fb", req, user_id="owner-a")

    assert order == ["create", "forward_backward"]
    assert [item["materialization_state"] for item in persisted] == ["materializing", "ready"]
    assert persisted[-1]["actor_name"] == "actor-517"
    assert resolved["request_id"] == "rid-517-fb"
    assert resolved["payload"]["type"] == "mint_forward_backward"
    session = manager.get_local_session("run-517")
    assert session is not None
    assert session.materialization_state == MATERIALIZATION_STATE_READY
    assert session.actor_name == "actor-517"


def test_issue_517_build_create_scheduler_extra_uses_control_plane_lane_only_for_plain_create() -> None:
    extra_create = training_route._build_create_scheduler_extra(
        base_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        model_id="run-517",
        training_op="create_model",
    )
    extra_restore = training_route._build_create_scheduler_extra(
        base_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        model_id="run-517",
        training_op="create_model_from_state",
    )

    assert extra_create["scheduler_domain"] == "megatron:create:megatron_qwen3_235b_a22b_instruct_2507"
    assert extra_restore["scheduler_domain"] == "megatron:megatron_qwen3_235b_a22b_instruct_2507"


@pytest.mark.anyio
async def test_issue_517_create_model_route_enqueues_without_local_training_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tinker_server.backend.api_work_queue as api_work_queue_module
    import tinker_server.backend.capacity_manager as capacity_manager_module
    import tinker_server.gateway as gateway_module
    import tinker_server.supported_models_gate as supported_models_gate_module

    enqueued: dict[str, object] = {}

    class _Queue:
        def enqueue(self, **kwargs):
            async def _run() -> None:
                enqueued.update(kwargs)

            return _run()

    class _Capacity:
        async def async_try_reserve(self, *_args, **_kwargs):
            return {"ok": True}

        async def async_release_all(self, *_args, **_kwargs) -> None:
            return None

    async def _allow_model(*, base_model: str, http_request):
        return base_model

    async def _trace_enqueue(*, enqueue_coro, **_kwargs) -> None:
        await enqueue_coro

    async def _future_create(_request_id: str) -> None:
        return None

    async def _future_mark_queued(_request_id: str, meta=None) -> None:
        return None

    async def _future_cleanup(_request_id: str) -> None:
        return None

    monkeypatch.setattr(training_route, "training_manager", None)
    monkeypatch.setattr(training_route, "training_engine", None)
    monkeypatch.setattr(training_route, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(training_route, "_enqueue_training_request_with_trace", _trace_enqueue)
    monkeypatch.setattr(
        training_route,
        "future_store",
        SimpleNamespace(
            async_create_with_id=_future_create,
            async_mark_queued=_future_mark_queued,
            async_cleanup=_future_cleanup,
        ),
    )
    monkeypatch.setattr(supported_models_gate_module, "enforce_base_model_allowed", _allow_model)
    monkeypatch.setattr(gateway_module, "upstream_for_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gateway_module, "get_gateway_config", lambda: None)
    monkeypatch.setattr(api_work_queue_module, "api_work_queue", _Queue())
    monkeypatch.setattr(capacity_manager_module, "capacity_manager", _Capacity())

    req = CreateModelRequest(
        session_id="route-517",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=LoRAConfig(rank=8),
    )
    http_request = SimpleNamespace(
        state=SimpleNamespace(user_data={"user_id": "user-1", "user_role": "user"}),
        headers={},
    )

    out = await training_route.create_model(req, http_request)

    assert isinstance(out.request_id, str) and out.request_id
    assert enqueued["op"] == "training.create_model"
    assert ":create:" in str((enqueued.get("extra") or {}).get("scheduler_domain") or "")


@pytest.mark.anyio
async def test_issue_517_delete_model_route_uses_detached_route_info_without_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_session_info = {
        "model_id": "run-517",
        "session_id": "session-517",
        "model_seq_id": 0,
        "base_model": "Qwen/Qwen3-0.6B",
        "backend": "peft",
        "user_id": "owner-517",
    }
    captured: dict[str, object] = {}

    async def _resolve(_model_id: str):
        return None, dict(route_session_info)

    async def _enqueue(**kwargs):
        captured.update(kwargs)
        return "rid-delete-517"

    async def _wait(_request_id: str):
        return {"model_id": "run-517", "status": "deleted"}

    monkeypatch.setattr(training_route, "training_manager", None)
    monkeypatch.setattr(training_route, "training_engine", None)
    monkeypatch.setattr(training_route, "_resolve_training_route_session", _resolve)
    monkeypatch.setattr(training_route, "_enqueue_internal_serialized_model_op", _enqueue)
    monkeypatch.setattr(training_route, "_wait_internal_future_result", _wait)

    http_request = SimpleNamespace(
        state=SimpleNamespace(user_data={"user_id": "user-1", "user_role": "user"}),
        headers={},
    )
    out = await training_route.delete_model("run-517", http_request)

    assert out["status"] == "deleted"
    assert captured["model_id"] == "run-517"
    assert captured["user_id"] == "owner-517"
    assert str((captured.get("extra") or {}).get("scheduler_domain") or "").startswith("peft:")


@pytest.mark.anyio
async def test_issue_528_do_create_model_dense_is_metadata_only_until_first_stateful_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    persisted: dict[str, object] = {}
    create_calls: list[str] = []

    async def _unexpected_create_training_session(_session) -> None:
        create_calls.append("called")

    async def _async_upsert_training_session(info: dict[str, object]) -> None:
        persisted.update(info)

    async def _best_effort_tokenizer_metadata(_session):
        return {
            "tokenizer_source_path": "/hf/snapshots/qwen3-4b",
            "tokenizer_identity": "/hf/snapshots/qwen3-4b#meta",
            "tokenizer_info": {"vocab_size": 151936},
        }

    monkeypatch.setattr(manager, "get_session", lambda model_id: manager.get_local_session(model_id))
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(
        training_route,
        "training_engine",
        SimpleNamespace(create_training_session=_unexpected_create_training_session),
    )
    monkeypatch.setattr(training_route, "_best_effort_local_tokenizer_metadata_for_session", _best_effort_tokenizer_metadata)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(
        "tinker_server.backend.session_index_store.add_training_run_to_session",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        training_route,
        "future_store",
        SimpleNamespace(resolve=lambda *_args, **_kwargs: None, async_fail=lambda *_args, **_kwargs: None),
    )

    req = CreateModelRequest(
        session_id="s528",
        model_seq_id=0,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        lora_config=LoRAConfig(rank=8),
    )
    await training_route._do_create_model("rid-528-create", req, user_id="owner-a", webhook_url=None)

    assert create_calls == []
    assert persisted["backend"] == "peft"
    assert persisted["materialization_state"] == MATERIALIZATION_STATE_UNMATERIALIZED
    assert persisted["actor_name"] is None


@pytest.mark.anyio
async def test_issue_528_dense_materialization_happens_once_on_first_stateful_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    manager.restore_training_session_info(
        {
            "model_id": "run-528",
            "session_id": "session-528",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "backend": "peft",
            "metadata_version": 2,
            "materialization_state": MATERIALIZATION_STATE_UNMATERIALIZED,
        }
    )

    persisted: list[dict[str, object]] = []
    create_calls: list[str] = []
    forward_calls: list[str] = []

    async def _create_training_session(session) -> None:
        create_calls.append(session.model_id)
        session.is_active = True
        engine._resource_pool_actor_names[session.model_id] = "dense-actor-528"

    async def _forward_backward(session, request):
        forward_calls.append(request.model_id)
        assert session.is_active is True
        return {"type": "mint_forward_backward", "metrics": {"loss:mean": 0.0}}

    async def _collect_tokenizer_metadata(_session):
        return {
            "tokenizer_source_path": "/hf/snapshots/qwen3-4b",
            "tokenizer_identity": "/hf/snapshots/qwen3-4b#ready",
            "tokenizer_info": {"vocab_size": 151936},
        }

    async def _async_upsert_training_session(info: dict[str, object]) -> None:
        persisted.append(dict(info))

    engine = SimpleNamespace(
        _workers={},
        _resource_pool_actor_names={},
        create_training_session=_create_training_session,
        forward_backward=_forward_backward,
    )

    monkeypatch.setattr(manager, "get_session", lambda model_id: manager.get_local_session(model_id))
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_collect_control_plane_tokenizer_metadata", _collect_tokenizer_metadata)
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(training_route, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(
        training_route,
        "future_store",
        SimpleNamespace(resolve=lambda *_args, **_kwargs: None, async_fail=lambda *_args, **_kwargs: None),
    )

    req = ForwardBackwardRequest(
        model_id="run-528",
        forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )
    await training_route._do_forward_backward("rid-528-fb-1", req, user_id="owner-a")
    await training_route._do_forward_backward("rid-528-fb-2", req, user_id="owner-a")

    assert create_calls == ["run-528"]
    assert forward_calls == ["run-528", "run-528"]
    assert [item["materialization_state"] for item in persisted] == ["materializing", "ready"]
    session = manager.get_local_session("run-528")
    assert session is not None
    assert session.materialization_state == MATERIALIZATION_STATE_READY
