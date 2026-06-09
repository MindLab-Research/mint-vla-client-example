from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


class _DummyRequest:
    def __init__(self, user_id: str | None = None) -> None:
        self.state = SimpleNamespace(
            user_data=None if user_id is None else {"user_id": user_id}
        )
        self.headers = {}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _install_ray_stub(
    monkeypatch, *, available: dict | None = None, total: dict | None = None
) -> None:
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True  # type: ignore[attr-defined]
    ray.available_resources = lambda: available or {}  # type: ignore[attr-defined]
    ray.cluster_resources = lambda: total or {}  # type: ignore[attr-defined]
    ray.util = SimpleNamespace(
        placement_group_table=lambda *args, **kwargs: {},
        get_placement_group=lambda name: None,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)


def _manager_stub(session, *, delete_session=None, mark_inflight=None):
    return SimpleNamespace(
        get_session=lambda _model_id: session,
        mark_inflight=mark_inflight or (lambda *_args, **_kwargs: None),
        delete_session=delete_session or (lambda _model_id: None),
    )


async def _async_none(*_args, **_kwargs):
    return None


class _AsyncTaskFutureService:
    async def async_create_with_id(self, _request_id: str) -> None:
        return None

    async def async_mark_queued(self, _request_id: str, meta=None) -> None:
        return None

    async def async_update_meta(self, _request_id: str, meta=None) -> None:
        return None

    async def async_cleanup(self, _request_id: str) -> None:
        return None

    async def async_forget(self, _request_id: str) -> None:
        return None

    async def async_ensure_pending(self, request_id: str, meta=None) -> dict:
        return {"created": True, "meta": dict(meta or {}), "request_id": request_id}

    async def async_get_status(self, _request_id: str) -> str:
        return "pending"


def _sampling_snapshot(
    sr, *, session_id: str, base_model: str, lora_int_id: int | None = 1
):
    return sr.SamplingSessionSnapshot(
        session_id=session_id,
        uses_multi_lora=True,
        uses_base_model=False,
        base_model=base_model,
        lora_rank=0,
        adapter_path=None,
        lora_loaded=False,
        lora_int_id=lora_int_id,
        metadata_version=1,
    )


class _AsyncModelWorkScheduler:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def append(self, **kwargs) -> dict:
        self._captured.update(kwargs)
        return {"ok": True, "scheduler_instance_id": "scheduler-281"}


async def _route_session_info(
    model_id: str, *, backend: str, base_model: str
) -> dict[str, str]:
    return {
        "model_id": str(model_id),
        "session_id": str(model_id),
        "backend": str(backend),
        "base_model": str(base_model),
        "user_id": "owner-a",
    }


async def _noop_async(*_args, **_kwargs) -> None:
    return None


def _stub_training_inflight(
    monkeypatch, route_module, calls: list[tuple[str, int]] | None = None
) -> None:
    async def _mark_training_inflight(model_id: str, delta: int) -> None:
        if calls is not None:
            calls.append((model_id, delta))

    monkeypatch.setattr(
        route_module, "_mark_training_inflight", _mark_training_inflight
    )


@pytest.mark.anyio
async def test_issue_281_forward_enqueues_scheduler_metadata(monkeypatch) -> None:
    import mint_server.backend.model_work_scheduler as mws
    from mint_server.models.types import ForwardBackwardInput, ForwardRequest
    from mint_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(backend="peft", base_model="Qwen/Qwen3-0.6B")
    captured: dict = {}

    monkeypatch.setattr(
        tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session)
    )
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
    _stub_training_inflight(monkeypatch, tr)
    monkeypatch.setattr(tr, "_get_max_model_len", lambda _base_model: 4096)
    monkeypatch.setattr(
        tr,
        "_get_training_route_session_info",
        lambda model_id: _route_session_info(
            model_id, backend="peft", base_model="Qwen/Qwen3-0.6B"
        ),
    )
    monkeypatch.setattr(tr, "_protect_training_session_enqueue_window", _noop_async)
    monkeypatch.setattr(tr, "task_futures", _AsyncTaskFutureService())
    monkeypatch.setattr(mws, "model_work_scheduler", _AsyncModelWorkScheduler(captured))

    req = ForwardRequest(
        model_id="run-281",
        seq_id=7,
        forward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )
    await tr.forward(req, _DummyRequest())

    assert captured["extra"]["scheduler_enabled"] is True
    assert captured["extra"]["scheduler_domain"] == "training:Qwen/Qwen3-0.6B"
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == "forward"
    assert captured["extra"]["seq_id"] == 7
    assert captured["extra"]["queue_priority"] == 0
    assert captured["domain_key"] == "training:Qwen/Qwen3-0.6B"
    assert captured["affinity_group"] == "training_session:run-281"
    assert captured["ordering_key"] == "training_session:run-281"


@pytest.mark.anyio
async def test_issue_281_do_forward_backward_releases_durable_inflight_when_runtime_missing(
    monkeypatch,
) -> None:
    from mint_server.models.types import ForwardBackwardInput, ForwardBackwardRequest
    from mint_server.routes import training as tr

    inflight_calls: list[tuple[str, int]] = []
    failures: list[tuple[str, str]] = []

    async def _async_fail(request_id: str, error: str) -> None:
        failures.append((request_id, str(error)))

    _stub_training_inflight(monkeypatch, tr, inflight_calls)
    monkeypatch.setattr(tr, "training_engine", None)
    monkeypatch.setattr(tr, "training_manager", None)
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(async_fail=_async_fail),
    )

    req = ForwardBackwardRequest(
        model_id="run-281-missing-runtime",
        forward_backward_input=ForwardBackwardInput(data=[], loss_fn="noop"),
    )

    await tr._do_forward_backward("req-281-missing-runtime", req, None)

    assert inflight_calls == [("run-281-missing-runtime", -1)]
    assert failures == [("req-281-missing-runtime", "Training engine not initialized")]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("route_name", "request_obj", "training_op"),
    [
        (
            "forward_backward",
            lambda types: types.ForwardBackwardRequest(
                model_id="run-281",
                seq_id=8,
                forward_backward_input=types.ForwardBackwardInput(
                    data=[], loss_fn="noop"
                ),
            ),
            "forward_backward",
        ),
        (
            "train_step",
            lambda types: types.TrainStepRequest(
                model_id="run-281",
                seq_id=9,
                forward_backward_input=types.ForwardBackwardInput(
                    data=[], loss_fn="noop"
                ),
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
    import mint_server.backend.model_work_scheduler as mws
    from mint_server.models import types as model_types
    from mint_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(
        backend="megatron", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )
    captured: dict = {}
    queued_meta: dict = {}

    class _QueuedTaskFutureService(_AsyncTaskFutureService):
        async def async_mark_queued(self, _request_id: str, meta=None) -> None:
            queued_meta.update(meta or {})

    monkeypatch.setattr(
        tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session)
    )
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
    _stub_training_inflight(monkeypatch, tr)
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
    monkeypatch.setattr(tr, "task_futures", _QueuedTaskFutureService())
    monkeypatch.setattr(mws, "model_work_scheduler", _AsyncModelWorkScheduler(captured))

    req = request_obj(model_types)
    await getattr(tr, route_name)(req, _DummyRequest(user_id="owner-a"))

    assert queued_meta == {}
    assert captured["extra"]["op"] == f"training.{training_op}"
    assert captured["extra"]["model_id"] == "run-281"
    assert captured["extra"]["session_id"] == "run-281"
    assert captured["extra"]["base_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert captured["extra"]["backend"] == "megatron"
    assert captured["extra"]["seq_id"] == req.seq_id
    assert captured["extra"]["queue_state"] == "queued"
    assert captured["extra"]["stage"] == "queued"
    assert isinstance(captured["extra"]["queued_at"], float)
    assert captured["extra"]["scheduler_enabled"] is True
    assert (
        captured["extra"]["scheduler_domain"]
        == "megatron:mint_megatron_qwen3_30b_a3b_instruct_2507"
    )
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == training_op
    assert captured["extra"]["queue_priority"] == 0
    assert (
        captured["domain_key"] == "megatron:mint_megatron_qwen3_30b_a3b_instruct_2507"
    )
    assert captured["affinity_group"] == "training_session:run-281"
    assert captured["ordering_key"] == "training_session:run-281"
    assert captured["extra"]["model_work_scheduler"] is True


@pytest.mark.anyio
async def test_issue_281_save_weights_for_sampler_enqueues_scheduler_metadata(
    monkeypatch,
) -> None:
    import mint_server.backend.model_work_scheduler as mws
    import mint_server.client_compat as client_compat
    from mint_server.models.types import SaveWeightsForSamplerRequest
    from mint_server.routes import training as tr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    session = SimpleNamespace(
        backend="megatron", base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )
    captured: dict = {}

    monkeypatch.setattr(
        tr, "training_manager", SimpleNamespace(get_session=lambda _model_id: session)
    )
    monkeypatch.setattr(tr, "training_engine", object())
    monkeypatch.setattr(tr, "_restore_training_session", lambda _model_id: None)
    _stub_training_inflight(monkeypatch, tr)
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
    monkeypatch.setattr(tr, "task_futures", _AsyncTaskFutureService())
    monkeypatch.setattr(mws, "model_work_scheduler", _AsyncModelWorkScheduler(captured))
    monkeypatch.setattr(client_compat, "prefer_tinker_uri", lambda _request: True)

    req = SaveWeightsForSamplerRequest(model_id="run-281", seq_id=9, path=None)
    await tr.save_weights_for_sampler(req, _DummyRequest(user_id="owner-a"))

    assert captured["extra"]["scheduler_enabled"] is True
    assert (
        captured["extra"]["scheduler_domain"]
        == "megatron:mint_megatron_qwen3_30b_a3b_instruct_2507"
    )
    assert captured["extra"]["scheduler_session_key"] == "run-281"
    assert captured["extra"]["execution_serial_key"] == "training_session:run-281"
    assert captured["extra"]["training_op"] == "save_weights_for_sampler"
    assert captured["extra"]["seq_id"] == 9
    assert captured["extra"]["queue_priority"] == 0
    assert captured["extra"]["prefer_tinker"] is True
    assert captured["extra"]["is_admin"] is False
    assert (
        captured["domain_key"] == "megatron:mint_megatron_qwen3_30b_a3b_instruct_2507"
    )
    assert captured["affinity_group"] == "training_session:run-281"
    assert captured["ordering_key"] == "training_session:run-281"
    assert captured["extra"]["model_work_scheduler"] is True


@pytest.mark.anyio
async def test_issue_281_asample_enqueues_scheduler_metadata(monkeypatch) -> None:
    import mint_server.backend.model_work_scheduler as mws
    import mint_server.backend.model_registry as model_registry
    from mint_server.models.types import ModelInput, SampleRequest, SamplingParams
    from mint_server.routes import sampling as sr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    captured: dict = {}

    async def _async_get_http_sampling_snapshot(session_id: str):
        assert session_id == "sess-281"
        return _sampling_snapshot(
            sr, session_id=session_id, base_model="Qwen/Qwen3-0.6B", lora_int_id=1
        )

    monkeypatch.setattr(
        sr, "_async_get_http_sampling_snapshot", _async_get_http_sampling_snapshot
    )
    monkeypatch.setattr(sr, "task_futures", _AsyncTaskFutureService())
    monkeypatch.setattr(mws, "model_work_scheduler", _AsyncModelWorkScheduler(captured))
    monkeypatch.setattr(
        model_registry,
        "get_model_config",
        lambda _model: SimpleNamespace(max_model_len=4096),
    )

    req = SampleRequest(
        sampling_session_id="sess-281",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    await sr.asample(req, _DummyRequest(user_id="owner-a"))

    assert captured["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert captured["affinity_group"] == "lora:sess-281:generation:1"
    assert captured["ordering_key"] == "session:sess-281"
    assert captured["extra"]["queue_priority"] == 0
    assert captured["extra"]["model_work_scheduler"] is True
    assert captured["extra"]["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert captured["extra"]["affinity_group"] == "lora:sess-281:generation:1"


@pytest.mark.anyio
async def test_issue_281_compute_logprobs_enqueues_scheduler_metadata(
    monkeypatch,
) -> None:
    import mint_server.backend.model_work_scheduler as mws
    import mint_server.backend.model_registry as model_registry
    from mint_server.models.types import ComputeLogprobsRequest, ModelInput
    from mint_server.routes import sampling as sr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    captured: dict = {}
    queued_meta: dict = {}

    class _QueuedTaskFutureService(_AsyncTaskFutureService):
        async def async_mark_queued(self, _request_id: str, meta=None) -> None:
            queued_meta.update(meta or {})

    async def _async_get_http_sampling_snapshot(session_id: str):
        assert session_id == "sess-281"
        return sr.SamplingSessionSnapshot(
            session_id=session_id,
            uses_multi_lora=True,
            uses_base_model=False,
            base_model="Qwen/Qwen3-0.6B",
            lora_rank=0,
            adapter_path=None,
            lora_loaded=False,
            lora_int_id=None,
            metadata_version=1,
        )

    monkeypatch.setattr(
        sr, "_async_get_http_sampling_snapshot", _async_get_http_sampling_snapshot
    )
    monkeypatch.setattr(sr, "task_futures", _QueuedTaskFutureService())
    monkeypatch.setattr(mws, "model_work_scheduler", _AsyncModelWorkScheduler(captured))
    monkeypatch.setattr(
        model_registry,
        "get_model_config",
        lambda _model: SimpleNamespace(max_model_len=4096),
    )

    req = ComputeLogprobsRequest(
        sampling_session_id="sess-281",
        seq_id=1,
        sequence=ModelInput.from_ints([1, 2, 3]),
    )
    await sr.compute_logprobs(req, _DummyRequest(user_id="owner-a"))

    assert queued_meta == {}
    assert captured["extra"]["op"] == "sampling.compute_logprobs"
    assert captured["extra"]["sampling_session_id"] == "sess-281"
    assert captured["extra"]["queue_state"] == "queued"
    assert captured["extra"]["stage"] == "queued"
    assert isinstance(captured["extra"]["queued_at"], float)
    assert captured["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert captured["affinity_group"] == "lora:sess-281:generation:1"
    assert captured["ordering_key"] == "session:sess-281"
    assert captured["extra"]["queue_priority"] == 0
    assert captured["extra"]["model_work_scheduler"] is True


@pytest.mark.anyio
async def test_issue_281_do_create_model_active_duplicate_fails_without_deleting_existing(
    monkeypatch,
) -> None:
    from mint_server.models.types import CreateModelRequest, LoRAConfig
    import mint_server.backend.training_session_store as training_session_store
    from mint_server.routes import training as tr

    deleted: list[str] = []
    deleted_store: list[str] = []
    failed: dict = {}

    existing = SimpleNamespace(is_active=True)

    async def _async_fail(request_id, error):
        failed.update({"request_id": request_id, "error": error})

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: existing,
            delete_session=lambda model_id: deleted.append(model_id),
        ),
    )
    monkeypatch.setattr(
        tr, "training_engine", SimpleNamespace(shutdown_session=lambda _session: None)
    )
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(
            async_fail=_async_fail,
        ),
    )
    monkeypatch.setattr(
        training_session_store, "delete_training_session", deleted_store.append
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
    assert deleted_store == []
    assert failed["request_id"] == "rid-dup"
    assert "already exists" in failed["error"]


@pytest.mark.anyio
async def test_issue_281_do_create_model_from_state_active_duplicate_fails_without_deleting_existing(
    monkeypatch,
) -> None:
    from mint_server.models.types import CreateModelFromStateRequest, LoRAConfig
    import mint_server.backend.training_session_store as training_session_store
    from mint_server.routes import training as tr

    deleted: list[str] = []
    deleted_store: list[str] = []
    failed: dict = {}

    existing = SimpleNamespace(is_active=True)

    async def _async_fail(request_id, error):
        failed.update({"request_id": request_id, "error": error})

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: existing,
            delete_session=lambda model_id: deleted.append(model_id),
        ),
    )
    monkeypatch.setattr(
        tr, "training_engine", SimpleNamespace(shutdown_session=lambda _session: None)
    )
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(
            async_fail=_async_fail,
        ),
    )
    monkeypatch.setattr(
        training_session_store, "delete_training_session", deleted_store.append
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
    assert deleted_store == []
    assert failed["request_id"] == "rid-dup2"
    assert "already exists" in failed["error"]


@pytest.mark.anyio
async def test_issue_281_do_reset_expert_bias_resolves_future(monkeypatch) -> None:
    from mint_server.models.types import ResetExpertBiasRequest
    from mint_server.routes import training as tr

    resolved: dict = {}

    async def _fake_reset(_session):
        return {"modules_reset": 2}

    async def _async_fail(request_id, error):
        resolved.update({"failed_request_id": request_id, "error": error})

    async def _async_resolve(request_id, payload):
        resolved.update({"request_id": request_id, "payload": payload})

    inflight_calls: list[tuple[str, int]] = []
    _stub_training_inflight(monkeypatch, tr, inflight_calls)
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
    monkeypatch.setattr(tr, "_restore_training_session", _async_none)
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )

    await tr._do_reset_expert_bias(
        "rid-281", ResetExpertBiasRequest(model_id="run-281")
    )

    assert resolved["request_id"] == "rid-281"
    assert resolved["payload"] == {
        "model_id": "run-281",
        "modules_reset": 2,
        "status": "success",
    }
    assert inflight_calls == [("run-281", -1)]
    assert "error" not in resolved


@pytest.mark.anyio
async def test_issue_281_do_delete_model_deletes_then_resolves(monkeypatch) -> None:
    _install_ray_stub(monkeypatch)
    import mint_server.backend.model_actor_supervisor as model_actor_inventory
    import mint_server.backend.training_session_store as training_session_store
    from mint_server.routes import training as tr

    calls: dict[str, list] = {
        "engine_delete": [],
        "delete_session": [],
        "delete_store": [],
        "clear_session": [],
        "mark_inflight": [],
    }
    resolved: dict = {}
    session = SimpleNamespace(model_id="run-281")

    async def _fake_delete(target_session):
        calls["engine_delete"].append(target_session)

    async def _async_resolve(request_id, payload):
        resolved.update({"request_id": request_id, "payload": payload})

    async def _async_fail(request_id, error):
        resolved.update({"failed_request_id": request_id, "error": error})

    inflight_calls: list[tuple[str, int]] = []
    _stub_training_inflight(monkeypatch, tr, inflight_calls)
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(shutdown_session=_fake_delete),
    )
    monkeypatch.setattr(
        tr,
        "training_manager",
        _manager_stub(
            session,
            delete_session=lambda model_id: calls["delete_session"].append(model_id),
            mark_inflight=lambda model_id, delta: calls["mark_inflight"].append(
                (model_id, delta)
            ),
        ),
    )
    monkeypatch.setattr(
        training_session_store,
        "delete_training_session",
        lambda model_id: calls["delete_store"].append(model_id),
    )
    monkeypatch.setattr(
        model_actor_inventory,
        "get_model_actor_supervisor",
        lambda: SimpleNamespace(
            clear_session=lambda model_id: calls["clear_session"].append(model_id)
        ),
    )
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )

    await tr._do_delete_model("rid-282", "run-281")

    assert calls["engine_delete"] == [session]
    assert calls["delete_session"] == ["run-281"]
    assert calls["delete_store"] == ["run-281"]
    assert calls["clear_session"] == ["run-281"]
    assert calls["mark_inflight"] == []
    assert resolved["request_id"] == "rid-282"
    assert resolved["payload"] == {"model_id": "run-281", "status": "deleted"}
    assert inflight_calls == [("run-281", -1)]
    assert "error" not in resolved


@pytest.mark.anyio
async def test_issue_281_do_get_session_guard_state_releases_durable_inflight(
    monkeypatch,
) -> None:
    from mint_server.routes import training as tr

    resolved: dict = {}
    session = SimpleNamespace(model_id="run-281")

    async def _fake_get_guard_state(target_session):
        assert target_session is session
        return {"blocked": False}

    async def _async_resolve(request_id, payload):
        resolved.update({"request_id": request_id, "payload": payload})

    async def _async_fail(request_id, error):
        resolved.update({"failed_request_id": request_id, "error": error})

    inflight_calls: list[tuple[str, int]] = []
    _stub_training_inflight(monkeypatch, tr, inflight_calls)
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(get_session_guard_state=_fake_get_guard_state),
    )
    monkeypatch.setattr(tr, "training_manager", _manager_stub(session))
    monkeypatch.setattr(tr, "_restore_training_session", _async_none)
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )

    await tr._do_get_session_guard_state("rid-guard", "run-281")

    assert resolved == {"request_id": "rid-guard", "payload": {"blocked": False}}
    assert inflight_calls == [("run-281", -1)]


@pytest.mark.anyio
async def test_issue_281_do_get_tokenizer_info_releases_durable_inflight(
    monkeypatch,
) -> None:
    from mint_server.routes import training as tr

    resolved: dict = {}
    session = SimpleNamespace(model_id="run-281")

    async def _fake_get_tokenizer_info(target_session):
        assert target_session is session
        return {"vocab_size": 42}

    async def _async_resolve(request_id, payload):
        resolved.update({"request_id": request_id, "payload": payload})

    async def _async_fail(request_id, error):
        resolved.update({"failed_request_id": request_id, "error": error})

    inflight_calls: list[tuple[str, int]] = []
    _stub_training_inflight(monkeypatch, tr, inflight_calls)
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(get_tokenizer_info=_fake_get_tokenizer_info),
    )
    monkeypatch.setattr(tr, "training_manager", _manager_stub(session))
    monkeypatch.setattr(tr, "_restore_training_session", _async_none)
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )

    await tr._do_get_tokenizer_info("rid-tokenizer", "run-281")

    assert resolved == {"request_id": "rid-tokenizer", "payload": {"vocab_size": 42}}
    assert inflight_calls == [("run-281", -1)]


@pytest.mark.anyio
async def test_issue_281_asample_falls_back_to_base_model_scheduler_domain(
    monkeypatch,
) -> None:
    import mint_server.backend.model_work_scheduler as mws
    import mint_server.backend.model_registry as model_registry
    from mint_server.models.types import ModelInput, SampleRequest, SamplingParams
    from mint_server.routes import sampling as sr

    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "1")

    captured: dict = {}

    async def _async_get_http_sampling_snapshot(session_id: str):
        assert session_id == "sess-281"
        return _sampling_snapshot(
            sr, session_id=session_id, base_model="Qwen/Qwen3-0.6B", lora_int_id=None
        )

    monkeypatch.setattr(
        sr, "_async_get_http_sampling_snapshot", _async_get_http_sampling_snapshot
    )
    monkeypatch.setattr(sr, "task_futures", _AsyncTaskFutureService())
    monkeypatch.setattr(mws, "model_work_scheduler", _AsyncModelWorkScheduler(captured))
    monkeypatch.setattr(
        model_registry,
        "get_model_config",
        lambda _model: SimpleNamespace(max_model_len=4096),
    )

    req = SampleRequest(
        sampling_session_id="sess-281",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    await sr.asample(req, _DummyRequest(user_id="owner-a"))

    assert captured["domain_key"] == "vllm:Qwen/Qwen3-0.6B"
    assert captured["affinity_group"] == "lora:sess-281:generation:1"
    assert captured["ordering_key"] == "session:sess-281"
    assert captured["extra"]["queue_priority"] == 0
    assert captured["extra"]["model_work_scheduler"] is True


@pytest.mark.anyio
async def test_issue_281_internal_serialized_op_marks_inflight_until_worker_finishes(
    monkeypatch,
) -> None:
    import mint_server.backend.model_work_scheduler as mws
    from mint_server.backend.training_session_manager import TrainingSessionManager
    from mint_server.models.types import ResetExpertBiasRequest
    from mint_server.routes import training as tr

    manager = TrainingSessionManager()
    manager.create_session("run-281", "sess-281", 0, "Qwen/Qwen3-30B-A3B-Instruct-2507")
    captured: dict = {}
    resolved: dict = {}

    async def _fake_reset(_session):
        return {"modules_reset": 1}

    async def _async_fail(request_id, error):
        resolved.update({"failed_request_id": request_id, "error": error})

    async def _async_resolve(request_id, payload):
        resolved.update({"request_id": request_id, "payload": payload})

    async def _restore_training_session(_model_id: str):
        return manager.get_local_session(_model_id)

    monkeypatch.setattr(tr, "training_manager", manager)
    monkeypatch.setattr(
        tr, "training_engine", SimpleNamespace(reset_expert_bias=_fake_reset)
    )
    monkeypatch.setattr(tr, "_restore_training_session", _restore_training_session)
    inflight_calls: list[tuple[str, int]] = []
    _stub_training_inflight(monkeypatch, tr, inflight_calls)
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(
            async_create_with_id=_async_none,
            async_mark_queued=_async_none,
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )
    monkeypatch.setattr(mws, "model_work_scheduler", _AsyncModelWorkScheduler(captured))

    request_id = await tr._enqueue_internal_serialized_model_op(
        model_id="run-281",
        op="training.reset_expert_bias",
        request_json=b"{}",
        extra={},
    )
    assert inflight_calls == [("run-281", 1)]
    assert captured["op"] == "training.reset_expert_bias"
    assert captured["affinity_group"] == "training_session:run-281"
    assert captured["ordering_key"] == "training_session:run-281"

    await tr._do_reset_expert_bias(
        request_id, ResetExpertBiasRequest(model_id="run-281")
    )

    assert manager.get_local_session("run-281").inflight_ops == 0
    assert inflight_calls == [("run-281", 1), ("run-281", -1)]
    assert resolved["request_id"] == request_id
    assert resolved["payload"]["modules_reset"] == 1


@pytest.mark.anyio
async def test_issue_281_public_healthz_ignores_timeout_observation(
    monkeypatch,
) -> None:
    import mint_server.backend.model_work_scheduler as mws
    import mint_server.backend.task_state_store as tss
    from mint_server import health_checks
    from mint_server.health_state import (
        clear_runtime_degraded_state,
        clear_startup_degraded_state,
    )
    from mint_server.routes import service

    clear_startup_degraded_state()
    clear_runtime_degraded_state()
    health_checks.reset_public_healthz_cache()
    _install_ray_stub(monkeypatch)

    class _Ping:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            return {"ok": True}

    monkeypatch.setattr(mws, "model_work_scheduler", _Ping())
    monkeypatch.setattr(tss, "task_state_store", _Ping())
    monkeypatch.setattr(tss, "task_futures", _Ping())

    payload = await service.healthz()
    assert payload["status"] == "ready"
    assert "ray_observation" not in payload


@pytest.mark.anyio
async def test_issue_281_public_healthz_ignores_pending_pg_observation(
    monkeypatch,
) -> None:
    import mint_server.backend.model_work_scheduler as mws
    import mint_server.backend.task_state_store as tss
    from mint_server import health_checks
    from mint_server.health_state import (
        clear_runtime_degraded_state,
        clear_startup_degraded_state,
    )
    from mint_server.routes import service

    clear_startup_degraded_state()
    clear_runtime_degraded_state()
    health_checks.reset_public_healthz_cache()
    _install_ray_stub(monkeypatch, available={"GPU": 2}, total={"GPU": 8})

    class _Ping:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            return {"ok": True}

    monkeypatch.setattr(mws, "model_work_scheduler", _Ping())
    monkeypatch.setattr(tss, "task_state_store", _Ping())
    monkeypatch.setattr(tss, "task_futures", _Ping())

    payload = await service.healthz()
    assert payload["status"] == "ready"
    assert "ray_observation" not in payload


def test_issue_281_http_route_label_prefers_route_template() -> None:
    from starlette.requests import Request

    from mint_server.app import _http_route_label

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
