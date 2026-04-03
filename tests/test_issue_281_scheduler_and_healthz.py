from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


class _DummyRequest:
    def __init__(self, user_id: str | None = None) -> None:
        self.state = SimpleNamespace(user_data=None if user_id is None else {"user_id": user_id})
        self.headers = {}


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


class _AsyncFutureStore:
    async def async_create_with_id(self, _request_id: str) -> None:
        return None

    async def async_mark_queued(self, _request_id: str, meta=None) -> None:
        return None

    async def async_cleanup(self, _request_id: str) -> None:
        return None

    async def async_forget(self, _request_id: str) -> None:
        return None

    async def async_ensure_pending(self, request_id: str, meta=None) -> dict:
        return {"created": True, "meta": dict(meta or {}), "request_id": request_id}

    async def async_get_status(self, _request_id: str) -> str:
        return "pending"


class _AsyncCapacityManager:
    async def async_try_reserve(self, *args, **kwargs) -> dict:
        return {"ok": True}

    async def async_release_all(self, *_args, **_kwargs) -> None:
        return None


async def _route_session_info(model_id: str, *, backend: str, base_model: str) -> dict[str, str]:
    return {
        "model_id": str(model_id),
        "session_id": str(model_id),
        "backend": str(backend),
        "base_model": str(base_model),
        "user_id": "owner-a",
    }


async def _noop_async(*_args, **_kwargs) -> None:
    return None


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
        "_get_training_route_session_info",
        lambda model_id: _route_session_info(model_id, backend="peft", base_model="Qwen/Qwen3-0.6B"),
    )
    monkeypatch.setattr(tr, "_protect_training_session_enqueue_window", _noop_async)
    monkeypatch.setattr(tr, "future_store", _AsyncFutureStore())
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(cm, "capacity_manager", _AsyncCapacityManager())

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
    assert captured["extra"]["queue_priority"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("route_name", "request_obj", "training_op"),
    [
        (
            "forward_backward",
            lambda types: types.ForwardBackwardRequest(
                model_id="run-281",
                seq_id=8,
                forward_backward_input=types.ForwardBackwardInput(data=[], loss_fn="noop"),
            ),
            "forward_backward",
        ),
        (
            "train_step",
            lambda types: types.TrainStepRequest(
                model_id="run-281",
                seq_id=9,
                forward_backward_input=types.ForwardBackwardInput(data=[], loss_fn="noop"),
                adam_params=types.AdamParams(),
            ),
            "train_step",
        ),
        (
            "optim_step",
            lambda types: types.OptimStepRequest(
                model_id="run-281",
                seq_id=10,
                adam_params=types.AdamParams(),
            ),
            "optim_step",
        ),
    ],
)
async def test_issue_281_training_routes_mark_queued_stage_metadata(
    monkeypatch, route_name: str, request_obj, training_op: str
) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    from tinker_server.models import types as model_types
    from tinker_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(backend="megatron", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    captured: dict = {}
    queued_meta: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    class _QueuedFutureStore(_AsyncFutureStore):
        async def async_mark_queued(self, _request_id: str, meta=None) -> None:
            queued_meta.update(meta or {})

    monkeypatch.setattr(tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session))
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
    monkeypatch.setattr(tr, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(
        tr,
        "_get_training_route_session_info",
        lambda model_id: _route_session_info(
            model_id,
            backend="megatron",
            base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        ),
    )
    monkeypatch.setattr(tr, "_protect_training_session_enqueue_window", _noop_async)
    monkeypatch.setattr(tr, "future_store", _QueuedFutureStore())
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(cm, "capacity_manager", _AsyncCapacityManager())

    await getattr(tr, route_name)(request_obj(model_types), _DummyRequest())

    assert queued_meta == {"op": f"training.{training_op}", "model_id": "run-281"}
    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "megatron:megatron_qwen3_30b_a3b_instruct_2507"
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == training_op
    assert captured["extra"]["queue_priority"] == 0


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
        "_get_training_route_session_info",
        lambda model_id: _route_session_info(
            model_id,
            backend="megatron",
            base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        ),
    )
    monkeypatch.setattr(tr, "_protect_training_session_enqueue_window", _noop_async)
    monkeypatch.setattr(tr, "future_store", _AsyncFutureStore())
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(cm, "capacity_manager", _AsyncCapacityManager())
    monkeypatch.setattr(client_compat, "prefer_tinker_uri", lambda _request: True)

    req = SaveWeightsForSamplerRequest(model_id="run-281", seq_id=9, path=None)
    await tr.save_weights_for_sampler(req, _DummyRequest(user_id="owner-a"))

    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "megatron:megatron_qwen3_30b_a3b_instruct_2507"
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == "save_weights_for_sampler"
    assert captured["extra"]["seq_id"] == 9
    assert captured["extra"]["queue_priority"] == 0
    assert captured["extra"]["prefer_tinker"] is True
    assert captured["extra"]["is_admin"] is False


@pytest.mark.anyio
async def test_issue_281_asample_enqueues_scheduler_metadata(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.model_registry as model_registry
    import tinker_server.backend.result_size_estimator as rse
    from tinker_server.models.types import ModelInput, SampleRequest, SamplingParams
    from tinker_server.routes import sampling as sr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        sr,
        "session_manager",
        SimpleNamespace(
            is_multi_lora_session=lambda _session_id: True,
            get_engine=lambda _session_id: None,
            get_session_base_model=lambda _session_id: "Qwen/Qwen3-0.6B",
            get_session_replica_key=lambda _session_id: "Qwen/Qwen3-0.6B::replica::1",
        ),
    )
    monkeypatch.setattr(sr, "future_store", _AsyncFutureStore())
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(cm, "capacity_manager", _AsyncCapacityManager())
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)
    monkeypatch.setattr(model_registry, "get_model_config", lambda _model: SimpleNamespace(max_model_len=4096))

    req = SampleRequest(
        sampling_session_id="sess-281",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    await sr.asample(req, _DummyRequest(user_id="owner-a"))

    assert captured["extra"] == {"queue_priority": 0}


@pytest.mark.anyio
async def test_issue_281_compute_logprobs_enqueues_scheduler_metadata(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.model_registry as model_registry
    import tinker_server.backend.result_size_estimator as rse
    from tinker_server.models.types import ComputeLogprobsRequest, ModelInput
    from tinker_server.routes import sampling as sr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        sr,
        "session_manager",
        SimpleNamespace(
            is_multi_lora_session=lambda _session_id: True,
            get_engine=lambda _session_id: None,
            get_session_base_model=lambda _session_id: "Qwen/Qwen3-0.6B",
            get_session_replica_key=lambda _session_id: "Qwen/Qwen3-0.6B::replica::1",
        ),
    )
    monkeypatch.setattr(sr, "future_store", _AsyncFutureStore())
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(cm, "capacity_manager", _AsyncCapacityManager())
    monkeypatch.setattr(rse, "estimate_compute_logprobs_result_bytes", lambda _req: 0)
    monkeypatch.setattr(model_registry, "get_model_config", lambda _model: SimpleNamespace(max_model_len=4096))

    req = ComputeLogprobsRequest(
        sampling_session_id="sess-281",
        seq_id=1,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )
    await sr.compute_logprobs(req, _DummyRequest(user_id="owner-a"))

    assert captured["extra"] == {"queue_priority": 0}


@pytest.mark.anyio
async def test_issue_281_asample_falls_back_to_base_model_scheduler_domain(monkeypatch) -> None:
    import tinker_server.backend.api_work_queue as awq
    import tinker_server.backend.capacity_manager as cm
    import tinker_server.backend.model_registry as model_registry
    import tinker_server.backend.result_size_estimator as rse
    from tinker_server.models.types import ModelInput, SampleRequest, SamplingParams
    from tinker_server.routes import sampling as sr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    captured: dict = {}

    async def _fake_enqueue(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        sr,
        "session_manager",
        SimpleNamespace(
            is_multi_lora_session=lambda _session_id: True,
            get_engine=lambda _session_id: None,
            get_session_base_model=lambda _session_id: "Qwen/Qwen3-0.6B",
            get_session_replica_key=lambda _session_id: None,
        ),
    )
    monkeypatch.setattr(sr, "future_store", _AsyncFutureStore())
    monkeypatch.setattr(awq, "api_work_queue", SimpleNamespace(enqueue=_fake_enqueue))
    monkeypatch.setattr(cm, "capacity_manager", _AsyncCapacityManager())
    monkeypatch.setattr(rse, "estimate_sampling_result_bytes", lambda _req: 0)
    monkeypatch.setattr(model_registry, "get_model_config", lambda _model: SimpleNamespace(max_model_len=4096))

    req = SampleRequest(
        sampling_session_id="sess-281",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    await sr.asample(req, _DummyRequest(user_id="owner-a"))

    assert captured["extra"] == {"queue_priority": 0}


@pytest.mark.anyio
async def test_issue_281_public_healthz_ignores_timeout_observation(monkeypatch) -> None:
    from tinker_server.routes import service

    _install_ray_stub(monkeypatch)

    async def _raise_timeout(awaitable, timeout):
        awaitable.close()
        raise service.asyncio.TimeoutError()

    monkeypatch.setattr(service.asyncio, "wait_for", _raise_timeout)

    payload = await service.healthz()
    assert payload["status"] == "ready"
    assert "ray_observation" not in payload


@pytest.mark.anyio
async def test_issue_281_public_healthz_ignores_pending_pg_observation(monkeypatch) -> None:
    from tinker_server.routes import service

    _install_ray_stub(monkeypatch, available={"GPU": 2}, total={"GPU": 8})

    async def _return_pending(awaitable, timeout):
        awaitable.close()
        return ["pg-a"]

    monkeypatch.setattr(service.asyncio, "wait_for", _return_pending)

    payload = await service.healthz()
    assert payload["status"] == "ready"
    assert "ray_observation" not in payload


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
