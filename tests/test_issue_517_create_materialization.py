from __future__ import annotations

from types import SimpleNamespace

import pytest

from mint_server.backend.training.training_session_manager import (
    MATERIALIZATION_STATE_READY,
    MATERIALIZATION_STATE_UNMATERIALIZED,
    TrainingSessionManager,
)
from mint_server.models.types import CreateModelRequest, ForwardBackwardInput, ForwardBackwardRequest, LoRAConfig
from mint_server.routes import training as training_route


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _AsyncTaskFutureService:
    def __init__(self) -> None:
        self.resolved: dict[str, object] = {}
        self.failed: list[tuple[str, str]] = []
        self.created: list[str] = []
        self.queued: list[tuple[str, dict | None]] = []
        self.cleaned: list[str] = []

    async def async_create_with_id(self, request_id: str) -> None:
        self.created.append(request_id)

    async def async_create_model_work_with_id(self, request_id: str, **_kwargs) -> None:
        self.created.append(request_id)

    async def async_mark_queued(self, request_id: str, meta=None) -> None:
        self.queued.append((request_id, meta))

    async def async_update_meta(self, _request_id: str, _meta: dict) -> None:
        return None

    async def async_cleanup(self, request_id: str) -> None:
        self.cleaned.append(request_id)

    async def async_resolve(self, request_id: str, payload: object, **_kwargs) -> None:
        self.resolved.update({"request_id": request_id, "payload": payload})

    async def async_fail(self, request_id: str, error: str) -> None:
        self.failed.append((request_id, error))


@pytest.mark.anyio
async def test_issue_517_do_create_model_persists_unmaterialized_session_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = TrainingSessionManager()
    persisted: dict[str, object] = {}
    task_futures = _AsyncTaskFutureService()
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
        "mint_server.backend.stores.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(
        "mint_server.backend.stores.session_index_store.add_training_run_to_session",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        training_route,
        "task_futures",
        task_futures,
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
    assert persisted["backend"] == "bumblebee"
    assert persisted["tokenizer_info"] == {"vocab_size": 151936}
    assert task_futures.resolved["request_id"] == "rid-517-create"
    assert task_futures.resolved["payload"]["model_id"] == "s517_0"
    assert task_futures.resolved["payload"]["backend"] == "bumblebee"
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
    task_futures = _AsyncTaskFutureService()
    order: list[str] = []

    async def _create_training_session(session) -> None:
        order.append("create")
        session.is_active = True
        engine._model_actor_supervisor_actor_names[session.model_id] = "actor-517"

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

    engine = SimpleNamespace(
        _workers={},
        _model_actor_supervisor_actor_names={},
        create_training_session=_create_training_session,
        forward_backward=_forward_backward,
    )

    monkeypatch.setattr(manager, "get_session", lambda model_id: manager.get_local_session(model_id))
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_collect_control_plane_tokenizer_metadata", _collect_tokenizer_metadata)
    monkeypatch.setattr(
        "mint_server.backend.stores.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(training_route, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(
        training_route,
        "task_futures",
        task_futures,
    )

    req = ForwardBackwardRequest(
        model_id="run-517",
        forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )
    await training_route._do_forward_backward("rid-517-fb", req, user_id="owner-a")

    assert order == ["create", "forward_backward"]
    assert [item["materialization_state"] for item in persisted] == ["materializing", "ready"]
    assert persisted[-1]["actor_name"] == "actor-517"
    assert task_futures.resolved["request_id"] == "rid-517-fb"
    assert task_futures.resolved["payload"]["type"] == "mint_forward_backward"
    assert task_futures.failed == []
    session = manager.get_local_session("run-517")
    assert session is not None
    assert session.materialization_state == MATERIALIZATION_STATE_READY
    assert session.actor_name == "actor-517"


def test_issue_517_build_create_scheduler_extra_uses_control_plane_lane_only_for_plain_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINT_QWEN3_235B_TRAINING_BACKEND", raising=False)
    monkeypatch.delenv("MINT_MOE_TRAINING_BACKEND", raising=False)

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

    assert extra_create["scheduler_domain"] == "bumblebee:mint_megatron_qwen3_235b_a22b_instruct_2507"
    assert extra_restore["scheduler_domain"] == "bumblebee:mint_megatron_qwen3_235b_a22b_instruct_2507"


@pytest.mark.anyio
async def test_issue_517_create_model_route_enqueues_without_local_training_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mint_server.gateway as gateway_module
    import mint_server.supported_models_gate as supported_models_gate_module

    enqueued: dict[str, object] = {}

    async def _allow_model(*, base_model: str, http_request):
        return base_model

    async def _fake_enqueue_model_work(**kwargs):
        enqueued.update(kwargs)
        return SimpleNamespace(scheduler_result={"ok": True})

    task_futures = _AsyncTaskFutureService()

    monkeypatch.setattr(training_route, "training_manager", None)
    monkeypatch.setattr(training_route, "training_engine", None)
    monkeypatch.setattr(training_route, "can_access_model", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        training_route,
        "task_futures",
        task_futures,
    )
    monkeypatch.setattr(supported_models_gate_module, "enforce_base_model_allowed", _allow_model)
    monkeypatch.setattr(gateway_module, "upstream_for_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gateway_module, "get_gateway_config", lambda: None)
    monkeypatch.setattr("mint_server.backend.scheduling.model_work_admission.enqueue_model_work", _fake_enqueue_model_work)

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
    assert enqueued["domain_key"] == "training:Qwen/Qwen3-0.6B"
    assert str((enqueued.get("extra") or {}).get("scheduler_domain") or "") == "training:Qwen/Qwen3-0.6B"


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
    assert (captured.get("extra") or {}).get("scheduler_domain") == "training:Qwen/Qwen3-0.6B"


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
        "mint_server.backend.stores.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(
        "mint_server.backend.stores.session_index_store.add_training_run_to_session",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        training_route,
        "task_futures",
        _AsyncTaskFutureService(),
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
        engine._model_actor_supervisor_actor_names[session.model_id] = "dense-actor-528"

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
        _model_actor_supervisor_actor_names={},
        create_training_session=_create_training_session,
        forward_backward=_forward_backward,
    )

    monkeypatch.setattr(manager, "get_session", lambda model_id: manager.get_local_session(model_id))
    monkeypatch.setattr(training_route, "training_manager", manager)
    monkeypatch.setattr(training_route, "training_engine", engine)
    monkeypatch.setattr(training_route, "_collect_control_plane_tokenizer_metadata", _collect_tokenizer_metadata)
    monkeypatch.setattr(
        "mint_server.backend.stores.training_session_store.async_upsert_training_session",
        _async_upsert_training_session,
    )
    monkeypatch.setattr(training_route, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(
        training_route,
        "task_futures",
        _AsyncTaskFutureService(),
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
