from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

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

    monkeypatch.setattr(tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session))
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
    monkeypatch.setattr(tr, "_get_max_model_len", lambda _base_model: 4096)
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

    monkeypatch.setattr(tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session))
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
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
    monkeypatch.setattr(client_compat, "prefer_tinker_uri", lambda _request: True)

    req = SaveWeightsForSamplerRequest(model_id="run-281", seq_id=9, path=None)
    await tr.save_weights_for_sampler(req, _DummyRequest(user_id="owner-a"))

    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507"
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

    monkeypatch.setattr(tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session))
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
    assert captured["extra"]["scheduler_domain"] == "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507"
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

    monkeypatch.setattr(tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session))
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
    assert captured["extra"]["scheduler_domain"] == "megatron:Qwen/Qwen3-30B-A3B-Instruct-2507"
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
        SimpleNamespace(get_session=lambda _model_id: SimpleNamespace(model_id="run-281")),
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
        SimpleNamespace(
            get_session=lambda _model_id: session,
            delete_session=lambda model_id: calls["delete_session"].append(model_id),
        ),
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
