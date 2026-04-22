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
