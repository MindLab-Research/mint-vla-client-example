from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse


class _DummyRequest:
    def __init__(self, user_id: str | None = None) -> None:
        self.state = SimpleNamespace(user_data=None if user_id is None else {"user_id": user_id})


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
    assert captured["extra"]["training_op"] == "forward"
    assert captured["extra"]["seq_id"] == 7


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
    assert captured["extra"]["training_op"] == "save_weights_for_sampler"
    assert captured["extra"]["seq_id"] == 9
    assert captured["extra"]["prefer_tinker"] is True


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
