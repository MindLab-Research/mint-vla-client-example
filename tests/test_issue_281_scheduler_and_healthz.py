from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from tinker_server.backend.future_store import FutureStatus


class _DummyRequest:
    def __init__(self, user_id: str | None = None) -> None:
        self.state = SimpleNamespace(user_data=None if user_id is None else {"user_id": user_id})


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _install_ray_stub(monkeypatch, *, available: dict | None = None, total: dict | None = None) -> None:
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.available_resources = lambda: available or {}  # type: ignore[attr-defined]
    ray.cluster_resources = lambda: total or {}  # type: ignore[attr-defined]
    ray.util = SimpleNamespace(
        placement_group_table=lambda *args, **kwargs: {},
        get_placement_group=lambda name: None,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)


def _manager_stub(session, *, delete_session=None):
    return SimpleNamespace(
        get_session=lambda _model_id: session,
        mark_inflight=lambda *_args, **_kwargs: None,
        delete_session=delete_session or (lambda _model_id: None),
    )


def _install_ray_cluster_health_stub(monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.available_resources = lambda: {"CPU": 12, "GPU": 3}  # type: ignore[attr-defined]
    ray.cluster_resources = lambda: {"CPU": 16, "GPU": 8}  # type: ignore[attr-defined]
    ray.nodes = lambda: [  # type: ignore[attr-defined]
        {"Alive": True, "NodeManagerAddress": "10.0.0.1"},
        {
            "Alive": False,
            "NodeManagerAddress": "10.0.0.2",
            "DeathReasonMessage": "health check failed due to missing too many heartbeats",
        },
    ]
    ray.util = SimpleNamespace(
        placement_group_table=lambda *args, **kwargs: {
            "pg1": {"name": "pg-ready", "state": "CREATED", "bundles": {0: {"GPU": 1}}},
            "pg2": {"name": "pg-pending", "state": "PENDING", "bundles": {0: {"GPU": 4}}},
        },
        get_placement_group=lambda name: None,
        list_named_actors=lambda all_namespaces=True: [
            {"name": "a", "namespace": "tinker"},
            {"name": "b", "namespace": "other"},
        ],
    )
    monkeypatch.setitem(sys.modules, "ray", ray)


def _install_ray_gcs_metrics_stub(monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.nodes = lambda: [  # type: ignore[attr-defined]
        {
            "Alive": True,
            "IsHeadNode": True,
            "NodeManagerAddress": "10.0.0.1",
            "MetricsExportPort": 8080,
        },
        {
            "Alive": True,
            "IsHeadNode": False,
            "NodeManagerAddress": "10.0.0.2",
            "MetricsExportPort": 8081,
        },
    ]
    monkeypatch.setitem(sys.modules, "ray", ray)


@pytest.mark.anyio
async def test_issue_281_forward_enqueues_scheduler_metadata(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    from tinker_server.models.types import ForwardBackwardInput, ForwardRequest
    from tinker_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(backend="peft", base_model="Qwen/Qwen3-0.6B")
    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    async def _async_create_with_id(_request_id):
        return None

    async def _async_mark_queued(_request_id, meta=None):
        _ = meta
        return None

    async def _async_try_reserve(*args, **kwargs):
        _ = (args, kwargs)
        return {"ok": True}

    async def _async_release_all(*_args, **_kwargs):
        return None

    monkeypatch.setattr(tr, "training_manager", _manager_stub(session))
    monkeypatch.setattr(tr, "training_engine", object())
    async def _restore_training_session(_model_id):
        return None

    monkeypatch.setattr(tr, "_restore_training_session", _restore_training_session)
    monkeypatch.setattr(tr, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            async_create_with_id=_async_create_with_id,
            async_mark_queued=_async_mark_queued,
            cleanup=lambda _request_id: None,
        ),
    )
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(
        cm,
        "capacity_manager",
        SimpleNamespace(
            async_try_reserve=_async_try_reserve,
            async_release_all=_async_release_all,
        ),
    )

    req = ForwardRequest(
        model_id="run-281",
        seq_id=7,
        forward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )
    await tr.forward(req, _DummyRequest())

    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "peft:Qwen/Qwen3-0.6B"
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == "forward"
    assert captured["extra"]["seq_id"] == 7


@pytest.mark.anyio
async def test_issue_281_create_model_enqueues_execution_serial_key(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.gateway as gateway
    import tinker_server.supported_models_gate as gate
    from tinker_server.models.types import CreateModelRequest, LoRAConfig
    from tinker_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")
    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    async def _allow_model(*, base_model, http_request):
        return base_model

    monkeypatch.setattr(gate, "enforce_base_model_allowed", _allow_model)
    monkeypatch.setattr(gateway, "upstream_for_model", lambda _base_model: None)
    monkeypatch.setattr(gateway, "get_gateway_config", lambda: None)
    monkeypatch.setattr(gateway, "remote_training_model", lambda _model_id: None)
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "training_manager", object())
    monkeypatch.setattr(tr, "can_access_model", lambda _base_model, _user_data: True)
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            create_with_id=lambda _request_id: None,
            mark_queued=lambda _request_id, meta=None: None,
            cleanup=lambda _request_id: None,
        ),
    )
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(
        cm,
        "capacity_manager",
        SimpleNamespace(
            try_reserve=lambda *args, **kwargs: {"ok": True},
            release_all=lambda *_args, **_kwargs: None,
        ),
    )

    req = CreateModelRequest(
        session_id="s281",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_config=LoRAConfig(rank=8),
    )
    await tr.create_model(req, _DummyRequest(user_id="owner-a"))

    assert captured["op"] == "training.create_model"
    assert captured["extra"]["execution_serial_key"] == "training_session:s281_0"
    assert captured["extra"]["scheduler_session_key"] == "s281_0"
    assert captured["extra"]["training_op"] == "create_model"


@pytest.mark.anyio
async def test_issue_281_save_weights_for_sampler_enqueues_scheduler_metadata(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.client_compat as client_compat
    from tinker_server.models.types import SaveWeightsForSamplerRequest
    from tinker_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(backend="megatron", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    async def _async_create_with_id(_request_id):
        return None

    async def _async_mark_queued(_request_id, meta=None):
        _ = meta
        return None

    async def _async_try_reserve(*args, **kwargs):
        _ = (args, kwargs)
        return {"ok": True}

    async def _async_release_all(*_args, **_kwargs):
        return None

    monkeypatch.setattr(tr, "training_manager", _manager_stub(session))
    monkeypatch.setattr(tr, "training_engine", object())
    async def _restore_training_session(_model_id):
        return None

    monkeypatch.setattr(tr, "_restore_training_session", _restore_training_session)
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            async_create_with_id=_async_create_with_id,
            async_mark_queued=_async_mark_queued,
            cleanup=lambda _request_id: None,
        ),
    )
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(
        cm,
        "capacity_manager",
        SimpleNamespace(
            async_try_reserve=_async_try_reserve,
            async_release_all=_async_release_all,
        ),
    )
    monkeypatch.setattr(client_compat, "prefer_tinker_uri", lambda _request: True)

    req = SaveWeightsForSamplerRequest(model_id="run-281", seq_id=9, path=None)
    await tr.save_weights_for_sampler(req, _DummyRequest(user_id="owner-a"))

    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "megatron:megatron_qwen3_30b_a3b_instruct_2507"
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == "save_weights_for_sampler"
    assert captured["extra"]["seq_id"] == 9
    assert captured["extra"]["prefer_tinker"] is True


@pytest.mark.anyio
async def test_issue_281_reset_expert_bias_enqueues_scheduler_metadata(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    from tinker_server.models.types import ResetExpertBiasRequest
    from tinker_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(backend="megatron", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tr, "training_manager", _manager_stub(session))
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            create_with_id=lambda _request_id: None,
            mark_queued=lambda _request_id, meta=None: None,
            get_status=lambda _request_id: FutureStatus.DONE,
            get_result=lambda _request_id: {"model_id": "run-281", "modules_reset": 3, "status": "success"},
            cleanup=lambda _request_id: None,
        ),
    )
    monkeypatch.setattr(tr, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(
        cm,
        "capacity_manager",
        SimpleNamespace(
            try_reserve=lambda *args, **kwargs: {"ok": True},
            release_all=lambda *_args, **_kwargs: None,
        ),
    )

    out = await tr.reset_expert_bias(ResetExpertBiasRequest(model_id="run-281"), _DummyRequest(user_id="owner-a"))

    assert out.model_id == "run-281"
    assert out.modules_reset == 3
    assert out.status == "success"
    assert captured["op"] == "training.reset_expert_bias"
    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "megatron:megatron_qwen3_30b_a3b_instruct_2507"
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == "reset_expert_bias"


@pytest.mark.anyio
async def test_issue_281_delete_model_enqueues_scheduler_metadata(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    from tinker_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(
        backend="megatron",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        user_id="owner-a",
    )
    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tr, "training_manager", _manager_stub(session))
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            create_with_id=lambda _request_id: None,
            mark_queued=lambda _request_id, meta=None: None,
            get_status=lambda _request_id: FutureStatus.DONE,
            get_result=lambda _request_id: {"model_id": "run-281", "status": "deleted"},
            cleanup=lambda _request_id: None,
        ),
    )
    monkeypatch.setattr(tr, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(
        cm,
        "capacity_manager",
        SimpleNamespace(
            try_reserve=lambda *args, **kwargs: {"ok": True},
            release_all=lambda *_args, **_kwargs: None,
        ),
    )

    out = await tr.delete_model("run-281")

    assert out == {"model_id": "run-281", "status": "deleted"}
    assert captured["op"] == "training.delete_model"
    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "megatron:megatron_qwen3_30b_a3b_instruct_2507"
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == "delete_model"


@pytest.mark.anyio
async def test_issue_281_do_create_model_active_duplicate_fails_without_deleting_existing(monkeypatch) -> None:
    from tinker_server.models.types import CreateModelRequest, LoRAConfig
    from tinker_server.routes import training as tr

    deleted: list[str] = []
    failed: dict = {}

    existing = SimpleNamespace(is_active=True)

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: existing,
            delete_session=lambda model_id: deleted.append(model_id),
        ),
    )
    monkeypatch.setattr(tr, "training_engine", SimpleNamespace(shutdown_session=lambda _session: None))
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            fail=lambda request_id, error: failed.update({"request_id": request_id, "error": error}),
        ),
    )

    await tr._do_create_model(
        "rid-dup",
        CreateModelRequest(
            session_id="sdup",
            model_seq_id=0,
            base_model="Qwen/Qwen3-4B-Instruct-2507",
            lora_config=LoRAConfig(rank=8),
        ),
        user_id="owner-a",
        webhook_url=None,
    )

    assert deleted == []
    assert failed["request_id"] == "rid-dup"
    assert "already exists" in failed["error"]


@pytest.mark.anyio
async def test_issue_281_do_create_model_from_state_active_duplicate_fails_without_deleting_existing(monkeypatch) -> None:
    from tinker_server.models.types import CreateModelFromStateRequest, LoRAConfig
    from tinker_server.routes import training as tr

    deleted: list[str] = []
    failed: dict = {}

    existing = SimpleNamespace(is_active=True)

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: existing,
            delete_session=lambda model_id: deleted.append(model_id),
        ),
    )
    monkeypatch.setattr(tr, "training_engine", SimpleNamespace(shutdown_session=lambda _session: None))
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            fail=lambda request_id, error: failed.update({"request_id": request_id, "error": error}),
        ),
    )

    await tr._do_create_model_from_state(
        "rid-dup2",
        CreateModelFromStateRequest(
            session_id="sdup2",
            model_seq_id=0,
            base_model="Qwen/Qwen3-4B-Instruct-2507",
            state_path="/tmp/fake-checkpoint",
            lora_config=LoRAConfig(rank=8),
            load_optimizer=False,
        ),
        user_id="owner-a",
    )

    assert deleted == []
    assert failed["request_id"] == "rid-dup2"
    assert "already exists" in failed["error"]


@pytest.mark.anyio
async def test_issue_281_do_reset_expert_bias_resolves_future(monkeypatch) -> None:
    from tinker_server.models.types import ResetExpertBiasRequest
    from tinker_server.routes import training as tr

    resolved: dict = {}

    async def _fake_reset(_session):
        return {"modules_reset": 2}

    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(reset_expert_bias=_fake_reset),
    )
    monkeypatch.setattr(
        tr,
        "training_manager",
        _manager_stub(SimpleNamespace(model_id="run-281")),
    )
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            resolve=lambda request_id, payload: resolved.update({"request_id": request_id, "payload": payload}),
            fail=lambda request_id, error: resolved.update({"failed_request_id": request_id, "error": error}),
        ),
    )

    await tr._do_reset_expert_bias("rid-281", ResetExpertBiasRequest(model_id="run-281"))

    assert resolved["request_id"] == "rid-281"
    assert resolved["payload"] == {
        "model_id": "run-281",
        "modules_reset": 2,
        "status": "success",
    }
    assert "error" not in resolved


@pytest.mark.anyio
async def test_issue_281_do_delete_model_shutdowns_then_resolves(monkeypatch) -> None:
    _install_ray_stub(monkeypatch)
    import tinker_server.backend.resource_pool as resource_pool
    import tinker_server.backend.training_session_store as training_session_store
    from tinker_server.routes import training as tr

    calls: dict[str, list] = {
        "shutdown": [],
        "delete_session": [],
        "delete_store": [],
        "clear_session": [],
    }
    resolved: dict = {}
    session = SimpleNamespace(model_id="run-281")

    async def _fake_shutdown(target_session):
        calls["shutdown"].append(target_session)

    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(shutdown_session=_fake_shutdown),
    )
    monkeypatch.setattr(
        tr,
        "training_manager",
        _manager_stub(session, delete_session=lambda model_id: calls["delete_session"].append(model_id)),
    )
    monkeypatch.setattr(training_session_store, "delete_training_session", lambda model_id: calls["delete_store"].append(model_id))
    monkeypatch.setattr(
        resource_pool,
        "get_resource_pool",
        lambda: SimpleNamespace(clear_session=lambda model_id: calls["clear_session"].append(model_id)),
    )
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            resolve=lambda request_id, payload: resolved.update({"request_id": request_id, "payload": payload}),
            fail=lambda request_id, error: resolved.update({"failed_request_id": request_id, "error": error}),
        ),
    )

    await tr._do_delete_model("rid-282", "run-281")

    assert calls["shutdown"] == [session]
    assert calls["delete_session"] == ["run-281"]
    assert calls["delete_store"] == ["run-281"]
    assert calls["clear_session"] == ["run-281"]
    assert resolved["request_id"] == "rid-282"
    assert resolved["payload"] == {"model_id": "run-281", "status": "deleted"}
    assert "error" not in resolved


@pytest.mark.anyio
async def test_issue_281_internal_serialized_op_marks_inflight_until_worker_finishes(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    from tinker_server.backend.training_session_manager import TrainingSessionManager
    from tinker_server.models.types import ResetExpertBiasRequest
    from tinker_server.routes import training as tr

    manager = TrainingSessionManager()
    manager.create_session("run-281", "sess-281", 0, "Qwen/Qwen3-30B-A3B-Instruct-2507")
    resolved: dict = {}

    async def _fake_enqueue(**_kwargs):
        return None

    async def _fake_reset(_session):
        return {"modules_reset": 1}

    monkeypatch.setattr(tr, "training_manager", manager)
    monkeypatch.setattr(tr, "training_engine", SimpleNamespace(reset_expert_bias=_fake_reset))
    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            create_with_id=lambda _request_id: None,
            mark_queued=lambda _request_id, meta=None: None,
            resolve=lambda request_id, payload: resolved.update({"request_id": request_id, "payload": payload}),
            fail=lambda request_id, error: resolved.update({"failed_request_id": request_id, "error": error}),
        ),
    )
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(
        cm,
        "capacity_manager",
        SimpleNamespace(
            try_reserve=lambda *args, **kwargs: {"ok": True},
            release_all=lambda *_args, **_kwargs: None,
        ),
    )

    request_id = await tr._enqueue_internal_serialized_model_op(
        model_id="run-281",
        op="training.reset_expert_bias",
        request_json=b"{}",
        extra={},
    )
    assert manager.get_session("run-281").inflight_ops == 1

    await tr._do_reset_expert_bias(request_id, ResetExpertBiasRequest(model_id="run-281"))

    assert manager.get_session("run-281").inflight_ops == 0
    assert resolved["request_id"] == request_id
    assert resolved["payload"]["modules_reset"] == 1


def test_issue_281_restore_training_session_uses_persisted_last_activity(monkeypatch) -> None:
    from tinker_server.backend.training_session_manager import TrainingSessionManager
    from tinker_server.routes import training as tr

    _install_ray_stub(monkeypatch)
    manager = TrainingSessionManager()
    monkeypatch.setattr(tr, "training_manager", manager)
    monkeypatch.setattr(tr, "training_engine", SimpleNamespace(_workers={}, _resource_pool_actor_names={}))
    monkeypatch.setattr(
        "tinker_server.backend.training_session_store.get_training_session_info",
        lambda _model_id: {
            "model_id": "run-restore",
            "session_id": "sess-restore",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-0.6B",
            "learning_rate": 1e-4,
            "backend": "peft",
            "created_at": "2026-03-20T10:00:00",
            "last_activity": 1234.5,
        },
    )

    session = tr._restore_training_session("run-restore")

    assert session is not None
    assert session.last_activity == pytest.approx(1234.5)
    assert session.created_at == "2026-03-20T10:00:00"


@pytest.mark.anyio
async def test_issue_281_internal_wait_releases_capacity_and_cleans_future(monkeypatch) -> None:
    import tinker_server.backend.capacity_manager as cm
    from tinker_server.routes import training as tr

    released: list[str] = []
    cleaned: list[str] = []

    monkeypatch.setattr(
        tr,
        "future_store",
        SimpleNamespace(
            get_status=lambda _request_id: FutureStatus.DONE,
            get_result=lambda _request_id: {"ok": True},
            cleanup=lambda request_id: cleaned.append(request_id),
        ),
    )
    monkeypatch.setattr(tr, "run_in_threadpool", _run_inline)
    monkeypatch.setattr(cm, "capacity_manager", SimpleNamespace(release_all=lambda request_id: released.append(request_id)))

    out = await tr._wait_internal_future_result("rid-283")

    assert out == {"ok": True}
    assert released == ["rid-283"]
    assert cleaned == ["rid-283"]


@pytest.mark.anyio
async def test_issue_281_public_healthz_stays_ready_without_ray_probe(monkeypatch) -> None:
    from tinker_server.routes import service

    _install_ray_stub(monkeypatch)

    def _boom(*args, **kwargs):
        raise AssertionError("public /api/v1/healthz should not call Ray")

    ray = sys.modules["ray"]
    monkeypatch.setattr(ray, "is_initialized", _boom)
    monkeypatch.setattr(ray, "available_resources", _boom)
    monkeypatch.setattr(ray, "cluster_resources", _boom)
    monkeypatch.setattr(ray.util, "placement_group_table", _boom)
    monkeypatch.setattr(ray.util, "get_placement_group", _boom)

    payload = await service.healthz()
    assert payload == {"status": "ready"}


@pytest.mark.anyio
async def test_issue_281_public_healthz_reports_startup_degraded_state() -> None:
    from tinker_server.health_state import clear_startup_degraded_state, set_startup_degraded_state
    from tinker_server.routes import service

    set_startup_degraded_state(reason="startup_degraded", error="boom", details={"phase": "init"})
    try:
        payload = await service.healthz()
        assert isinstance(payload, JSONResponse)
        assert payload.status_code == 503
        assert json.loads(payload.body) == {
            "status": "degraded",
            "reason": "startup_degraded",
            "error": "boom",
            "details": {"phase": "init"},
        }
    finally:
        clear_startup_degraded_state()


@pytest.mark.anyio
async def test_issue_281_internal_deep_healthz_timeout_is_observation_not_failure(monkeypatch) -> None:
    from tinker_server import health_checks
    from tinker_server.routes import internal

    _install_ray_stub(monkeypatch, available={"GPU": 2}, total={"GPU": 8})

    async def _raise_timeout(awaitable, timeout):
        awaitable.close()
        raise health_checks.asyncio.TimeoutError()

    monkeypatch.setattr(health_checks.asyncio, "wait_for", _raise_timeout)

    payload = await internal.deep_health_check()
    assert payload["status"] == "ready"
    assert payload["ray_observation"]["reason"] == "ray_healthz_timeout"


@pytest.mark.anyio
async def test_issue_281_internal_deep_healthz_pending_pg_is_observation_not_failure(monkeypatch) -> None:
    from tinker_server import health_checks
    from tinker_server.routes import internal

    _install_ray_stub(monkeypatch, available={"GPU": 2}, total={"GPU": 8})

    async def _return_pending(awaitable, timeout):
        awaitable.close()
        return ["pg-a"]

    monkeypatch.setattr(health_checks.asyncio, "wait_for", _return_pending)

    payload = await internal.deep_health_check()
    assert payload["status"] == "ready"
    assert payload["ray_observation"]["reason"] == "pending_placement_groups"
    assert payload["ray_observation"]["pending_pg_names"] == ["pg-a"]


@pytest.mark.anyio
async def test_issue_281_healthz_ray_connect_failure_is_503(monkeypatch) -> None:
    from tinker_server.routes import service

    _install_ray_stub(monkeypatch)

    async def _raise_connect_error(*, timeout_s: float):
        _ = timeout_s
        raise RuntimeError("ray disconnected")

    monkeypatch.setattr(service, "async_pending_gpu_pg_observation", _raise_connect_error)

    response = await service.healthz()

    assert response.status_code == 503
    assert response.body
    assert b"ray_unavailable" in response.body


@pytest.mark.anyio
async def test_issue_281_healthz_uninitialized_ray_is_503(monkeypatch) -> None:
    from tinker_server.routes import service

    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)

    response = await service.healthz()

    assert response.status_code == 503
    assert b"ray_unavailable" in response.body


@pytest.mark.anyio
async def test_issue_281_kill_dense_actors_uses_named_actor_helper(monkeypatch) -> None:
    from tinker_server.backend.resource_pool import ActorType
    from tinker_server.routes import service

    killed: list[tuple[str, str, str | None]] = []
    unregistered: list[str] = []

    async def _fake_kill(actor_name: str, namespace: str, *, base_model: str | None, timeout_s: float = 10.0):
        _ = timeout_s
        killed.append((actor_name, namespace, base_model))
        return True

    pool = SimpleNamespace(
        iter_entries=lambda: [
            SimpleNamespace(
                actor_type=ActorType.DENSE,
                actor_name="dense-a",
                namespace="ns-a",
                base_model="model-a",
                actor_handle=None,
            )
        ],
        unregister=lambda actor_name: unregistered.append(actor_name),
    )

    monkeypatch.setattr(service, "async_kill_named_actor", _fake_kill)
    monkeypatch.setattr("tinker_server.backend.resource_pool.get_resource_pool", lambda: pool)

    killed_count = await service._kill_dense_actors("model-a")

    assert killed_count == 1
    assert killed == [("dense-a", "ns-a", "model-a")]
    assert unregistered == ["dense-a"]


@pytest.mark.anyio
async def test_issue_281_kill_dense_actors_keeps_best_effort_unregister(monkeypatch) -> None:
    from tinker_server.backend.resource_pool import ActorType
    from tinker_server.routes import service

    unregistered: list[str] = []

    async def _fake_kill(actor_name: str, namespace: str, *, base_model: str | None, timeout_s: float = 10.0):
        _ = (actor_name, namespace, base_model, timeout_s)
        raise RuntimeError("kill failed")

    pool = SimpleNamespace(
        iter_entries=lambda: [
            SimpleNamespace(
                actor_type=ActorType.DENSE,
                actor_name="dense-b",
                namespace="ns-b",
                base_model="model-b",
                actor_handle=None,
            )
        ],
        unregister=lambda actor_name: unregistered.append(actor_name),
    )

    monkeypatch.setattr(service, "async_kill_named_actor", _fake_kill)
    monkeypatch.setattr("tinker_server.backend.resource_pool.get_resource_pool", lambda: pool)

    killed_count = await service._kill_dense_actors("model-b")

    assert killed_count == 1
    assert unregistered == ["dense-b"]


def test_issue_281_http_route_label_prefers_route_template() -> None:
    from starlette.requests import Request

    from tinker_server.app import _http_route_label

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/training_runs/run-281",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "route": SimpleNamespace(path="/api/v1/training_runs/{training_run_id}"),
        }
    )

    assert _http_route_label(request) == "/api/v1/training_runs/{training_run_id}"


def test_ray_cluster_health_snapshot_summarizes_cluster_state(monkeypatch) -> None:
    from tinker_server import ray_cluster_health as rch

    _install_ray_cluster_health_stub(monkeypatch)
    monkeypatch.setattr(rch, "_CACHE_VALUE", None)
    monkeypatch.setattr(rch, "_CACHE_AT_MONO", 0.0)

    snapshot = rch.get_ray_cluster_health_snapshot(force_refresh=True)

    assert snapshot["status"] == "degraded"
    assert snapshot["up"] is True
    assert snapshot["nodes"]["alive"] == 1
    assert snapshot["nodes"]["dead"] == 1
    assert snapshot["nodes"]["dead_missing_heartbeats"] == 1
    assert snapshot["placement_groups"]["pending_gpu"] == 1
    assert snapshot["placement_groups"]["pending_gpu_names"] == ["pg-pending"]
    assert snapshot["named_actors"]["total"] == 2
    assert snapshot["named_actors"]["namespace"] == 1
    assert "slow_control_plane_probes" not in snapshot["warnings"]


@pytest.mark.anyio
async def test_internal_metrics_exports_ray_cluster_metrics(monkeypatch) -> None:
    from tinker_server.routes import internal

    async def _fake_admission_stats() -> dict:
        return {
            "capacity": {},
            "work_queue": {},
            "future_store": {},
            "actors": {},
            "process": {},
            "ray_cluster": {
                "up": True,
                "warning_count": 2,
                "probe_error_count": 1,
                "slow_probe_count": 1,
                "total_probe_latency_ms": 321.5,
                "cache_age_s": 7.0,
                "nodes": {"alive": 6, "dead": 2, "dead_missing_heartbeats": 2},
                "resources": {"gpu_total": 72, "gpu_available": 56, "cpu_total": 100, "cpu_available": 84},
                "placement_groups": {"total": 9, "created": 6, "removed": 1, "pending": 2, "pending_gpu": 2},
                "named_actors": {"total": 11, "namespace": 8},
                "probes": {
                    "nodes": {"ok": True, "latency_ms": 12.5},
                    "placement_groups": {"ok": False, "latency_ms": 2500.0},
                },
            },
        }

    monkeypatch.setattr(internal, "admission_stats", _fake_admission_stats)

    response = await internal.metrics()

    assert isinstance(response, Response)
    body = response.body.decode()
    assert 'mint_ray_cluster_up 1' in body
    assert 'mint_ray_cluster_dead_nodes_missing_heartbeats 2' in body
    assert 'mint_ray_cluster_gpu_total 72' in body
    assert 'mint_ray_cluster_placement_groups_pending_gpu 2' in body
    assert 'mint_ray_cluster_probe_success{probe="nodes"} 1' in body
    assert 'mint_ray_cluster_probe_success{probe="placement_groups"} 0' in body


def test_ray_gcs_metrics_snapshot_extracts_selected_metrics(monkeypatch) -> None:
    from tinker_server import ray_gcs_metrics as rgm

    _install_ray_gcs_metrics_stub(monkeypatch)
    monkeypatch.setattr(rgm, "_CACHE_VALUE", None)
    monkeypatch.setattr(rgm, "_CACHE_AT_MONO", 0.0)

    prom_text = """
# HELP gcs_task_manager_task_events_reported reported
gcs_task_manager_task_events_reported{Component="gcs_server"} 1000
gcs_task_manager_task_events_stored{Component="gcs_server"} 995
gcs_task_manager_task_events_dropped{Component="gcs_server"} 5
gcs_storage_operation_count{Component="gcs_server"} 200
gcs_storage_operation_latency_ms_sum{Component="gcs_server"} 80
gcs_storage_operation_latency_ms_count{Component="gcs_server"} 4
gcs_placement_group_count{Component="gcs_server"} 9
gcs_actors_count{Component="gcs_server"} 11
grpc_server_req_handling{Component="gcs_server",grpc_server_method="GetAllNodeInfo"} 3
grpc_server_req_failed{Component="gcs_server",grpc_server_method="GetAllNodeInfo",grpc_server_status="DEADLINE_EXCEEDED"} 7
grpc_server_req_process_time_ms_sum{Component="gcs_server",grpc_server_method="GetAllNodeInfo"} 50
grpc_server_req_process_time_ms_count{Component="gcs_server",grpc_server_method="GetAllNodeInfo"} 2
health_check_rpc_latency_ms_sum{Component="gcs_server"} 20
health_check_rpc_latency_ms_count{Component="gcs_server"} 2
grpc_server_req_handling{Component="raylet",grpc_server_method="GetAllNodeInfo"} 99
""".strip()

    monkeypatch.setattr(rgm, "_scrape_metrics_text", lambda address, timeout_s: prom_text)

    snapshot = rgm.get_ray_gcs_metrics_snapshot(force_refresh=True)

    assert snapshot["status"] == "ready"
    assert snapshot["up"] is True
    assert snapshot["candidate_addresses"] == ["10.0.0.1:8080"]
    assert snapshot["sources_with_metrics"] == ["10.0.0.1:8080"]
    assert snapshot["aggregates"]["gcs_task_manager_task_events_dropped"] == 5.0
    assert snapshot["derived"]["gcs_task_manager_task_events_drop_ratio"] == 0.005
    assert snapshot["derived"]["gcs_storage_operation_latency_ms_mean"] == 20.0
    assert snapshot["derived"]["health_check_rpc_latency_ms_mean"] == 10.0
    sample_names = {sample["name"] for sample in snapshot["samples"]}
    assert "gcs_task_manager_task_events_reported" in sample_names
    assert "grpc_server_req_handling" in sample_names
    assert snapshot["sample_count"] == 14


@pytest.mark.anyio
async def test_internal_metrics_exports_ray_gcs_bridge_metrics(monkeypatch) -> None:
    from tinker_server.routes import internal

    async def _fake_admission_stats() -> dict:
        return {
            "capacity": {},
            "work_queue": {},
            "future_store": {},
            "actors": {},
            "process": {},
            "ray_cluster": {},
            "ray_gcs_metrics": {
                "up": True,
                "scrape_error_count": 0,
                "sample_count": 3,
                "scrape_latency_ms": 42.0,
                "cache_age_s": 5.0,
                "derived": {
                    "gcs_task_manager_task_events_drop_ratio": 0.01,
                },
                "samples": [
                    {
                        "name": "gcs_task_manager_task_events_dropped",
                        "labels": {"Component": "gcs_server"},
                        "value": 12,
                    },
                    {
                        "name": "grpc_server_req_handling",
                        "labels": {"Component": "gcs_server", "grpc_server_method": "GetAllNodeInfo"},
                        "value": 4,
                    },
                    {
                        "name": "health_check_rpc_latency_ms_count",
                        "labels": {"Component": "gcs_server"},
                        "value": 2,
                    },
                ],
            },
        }

    monkeypatch.setattr(internal, "admission_stats", _fake_admission_stats)

    response = await internal.metrics()

    assert isinstance(response, Response)
    body = response.body.decode()
    assert "mint_ray_gcs_metrics_bridge_up 1" in body
    assert "mint_ray_gcs_gcs_task_manager_task_events_drop_ratio 0.01" in body
    assert 'gcs_task_manager_task_events_dropped{Component="gcs_server"} 12' in body
    assert 'grpc_server_req_handling{Component="gcs_server",grpc_server_method="GetAllNodeInfo"} 4' in body
