from __future__ import annotations

import asyncio
import importlib
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _import_training_route():
    sys.modules.setdefault("tinker_server.routes.service", types.ModuleType("tinker_server.routes.service"))
    return importlib.import_module("tinker_server.routes.training")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@contextmanager
def _span_recorder(store: list[tuple[str, dict[str, object]]], name: str, **kwargs):
    store.append((name, dict(kwargs)))

    class _Span:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key, value) -> None:
            self.attributes[str(key)] = value

    span = _Span()
    yield span


@pytest.mark.anyio
async def test_issue_408_save_weights_for_sampler_emits_trace_spans(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.backend import session_index_store as sis
    from tinker_server.models.types import SaveWeightsForSamplerRequest

    tr = _import_training_route()

    async def _identity_materialize(session):
        return session

    monkeypatch.setattr(tr, "_materialize_training_session_for_stateful_use", _identity_materialize)

    ckpt_dir = tmp_path / "sampler_ephemeral"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    span_calls: list[tuple[str, dict[str, object]]] = []
    resolved: dict[str, object] = {}

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    class _InferenceManagerStub:
        tensor_parallel_size = 1
        data_parallel_size = 1
        gpu_memory_utilization = 0.8
        max_model_len = 4096

        async def get_engine_for_model(self, _base_model: str):
            return object()

        def register_multi_lora_session(self, **_kwargs) -> None:
            return None

    def _resolve(request_id: str, response: dict) -> None:
        resolved["request_id"] = request_id
        resolved["response"] = response

    async def _async_resolve(request_id: str, response: dict) -> None:
        _resolve(request_id, response)

    async def _async_fail(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        tr,
        "training_manager",
        SimpleNamespace(
            get_session=lambda _model_id: SimpleNamespace(
                model_id="run-408",
                session_id="sess-408",
                base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                current_step=11,
                backend="megatron",
                lora_config=SimpleNamespace(rank=32, train_mlp=True),
                inference_engine=None,
            ),
            mark_inflight=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(
        tr,
        "training_engine",
        SimpleNamespace(save_weights_for_sampler=_fake_save_weights_for_sampler),
    )
    monkeypatch.setattr(tr, "inference_manager", _InferenceManagerStub())
    monkeypatch.setattr(
        tr,
        "task_state_futures",
        SimpleNamespace(async_resolve=_async_resolve, async_fail=_async_fail),
    )
    monkeypatch.setattr(tr, "checkpoint_has_optimizer_state", lambda _path: False)
    monkeypatch.setattr(tr, "validate_sampler_checkpoint_for_sampling", lambda _path: None)
    monkeypatch.setattr(tr, "write_checkpoint_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tr, "build_ephemeral_checkpoint_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(sis, "add_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "add_heartbeat_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "upsert_sampler_index", lambda _payload: None)
    monkeypatch.setattr(
        tr,
        "start_as_current_span",
        lambda name, **kwargs: _span_recorder(span_calls, name, **kwargs),
    )
    monkeypatch.setattr(
        tr,
        "start_as_current_span_from_traceparent",
        lambda name, **kwargs: _span_recorder(span_calls, name, **kwargs),
    )
    monkeypatch.setattr(
        tr,
        "get_current_traceparent",
        lambda: "00-" + ("a" * 32) + "-" + ("1" * 16) + "-01",
    )

    request = SaveWeightsForSamplerRequest(model_id="run-408", seq_id=0)
    await tr._do_save_weights_for_sampler(
        request_id="req-408-save",
        request=request,
        user_id="owner-408",
        prefer_tinker=True,
    )
    await asyncio.sleep(0)

    assert resolved["request_id"] == "req-408-save"
    span_by_name = {name: kwargs for name, kwargs in span_calls}
    assert "training.save_weights_for_sampler.validate_checkpoint" in span_by_name
    assert "training.save_weights_for_sampler.write_checkpoint_metadata" in span_by_name
    assert "training.save_weights_for_sampler.schedule_background_engine_warm" in span_by_name
    assert "training.save_weights_for_sampler.background_engine_warm" in span_by_name
    assert "training.save_weights_for_sampler.register_sampling_session" in span_by_name
    assert "training.save_weights_for_sampler.session_index_write" in span_by_name
    assert span_by_name["training.save_weights_for_sampler.validate_checkpoint"]["request_id"] == "req-408-save"
    assert span_by_name["training.save_weights_for_sampler.validate_checkpoint"]["attributes"]["model_id"] == "run-408"
    assert span_by_name["training.save_weights_for_sampler.background_engine_warm"]["traceparent"] == (
        "00-" + ("a" * 32) + "-" + ("1" * 16) + "-01"
    )
    assert span_by_name["training.save_weights_for_sampler.background_engine_warm"]["request_id"] == "req-408-save"
    assert span_by_name["training.save_weights_for_sampler.background_engine_warm"]["attributes"]["base_model"] == (
        "Qwen/Qwen3-30B-A3B-Instruct-2507"
    )


def test_issue_408_megatron_create_path_emits_trace_spans(monkeypatch) -> None:
    from tinker_server.backend import megatron_distributed as md
    from tinker_server.backend import model_registry as model_registry
    from tinker_server.backend import model_actor_registry as model_actor_registry_mod
    from tinker_server import config as config_mod

    span_calls: list[tuple[str, dict[str, object]]] = []
    removed: list[object] = []
    created: list[dict[str, object]] = []
    actor_ctor_calls: list[dict[str, object]] = []
    fake_pg = object()
    fake_actor = object()

    class _FakePool:
        def ensure_gpus_available(self, *_args, **_kwargs) -> bool:
            return True

        def reserve_gpus(self, *_args, **_kwargs) -> bool:
            return True

        def release_pending_gpus(self, *_args, **_kwargs) -> None:
            return None

        def register(self, **kwargs) -> None:
            created.append(dict(kwargs))

        def mark_ready(self, *_args, **_kwargs) -> bool:
            return True

    class _Options:
        def __init__(self, kwargs: dict[str, object]) -> None:
            self.kwargs = kwargs

        def remote(self, **kwargs):
            actor_ctor_calls.append(dict(kwargs))
            return fake_actor

    class _FakeMegatronWorkerGroup:
        @staticmethod
        def options(**kwargs):
            created.append({"options": dict(kwargs)})
            return _Options(kwargs)

    monkeypatch.setattr(model_actor_registry_mod, "get_model_actor_registry", lambda: _FakePool())
    monkeypatch.setattr(config_mod, "actor_runtime_env_vars", lambda **_kwargs: {})
    monkeypatch.setattr(config_mod, "otel_env_vars", lambda: {})
    monkeypatch.setattr(model_actor_registry_mod, "actor_observability_metadata", lambda _actor: {})
    monkeypatch.setattr(model_registry, "is_persistent_model", lambda _base_model: False)
    monkeypatch.setattr(md, "MegatronWorkerGroup", _FakeMegatronWorkerGroup)
    monkeypatch.setattr(md.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(md.ray, "get_actor", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("missing")))
    monkeypatch.setattr(md.ray.util, "get_placement_group", lambda _name: fake_pg, raising=False)
    monkeypatch.setattr(md.ray.util, "remove_placement_group", lambda pg: removed.append(pg), raising=False)
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        md,
        "start_as_current_span_from_traceparent",
        lambda name, **kwargs: _span_recorder(span_calls, name, **kwargs),
    )
    monkeypatch.setattr(
        md,
        "start_as_current_span",
        lambda name, **kwargs: _span_recorder(span_calls, name, **kwargs),
    )

    actor = md.get_or_create_megatron_worker_group(
        base_model="/tmp/qwen",
        lora_rank=8,
        learning_rate=1e-4,
        session_id="sess-408",
        traceparent="00-" + ("b" * 32) + "-" + ("2" * 16) + "-01",
        request_id="req-408-create",
    )

    assert actor is fake_actor
    assert removed == [fake_pg]
    assert actor_ctor_calls == [{
        "base_model": "/tmp/qwen",
        "lora_rank": 8,
        "learning_rate": 1e-4,
        "distributed_config": md.DistributedConfig(),
        "observability_base_model": "/tmp/qwen",
        "traceparent": "00-" + ("b" * 32) + "-" + ("2" * 16) + "-01",
        "request_id": "req-408-create",
    }]
    span_by_name = {name: kwargs for name, kwargs in span_calls}
    assert "training.create_model.megatron.actor_lookup" in span_by_name
    assert "training.create_model.megatron.orphan_pg_probe" in span_by_name
    assert "training.create_model.megatron.orphan_pg_race_guard" in span_by_name
    assert "training.create_model.megatron.orphan_pg_remove" in span_by_name
    assert "training.create_model.megatron.ensure_gpus_available" in span_by_name
    assert "training.create_model.megatron.reserve_gpus" in span_by_name
    assert "training.create_model.megatron.actor_create" in span_by_name
    assert "training.create_model.megatron.register_new_actor" in span_by_name
    assert span_by_name["training.create_model.megatron.actor_lookup"]["traceparent"] == (
        "00-" + ("b" * 32) + "-" + ("2" * 16) + "-01"
    )
    assert span_by_name["training.create_model.megatron.actor_lookup"]["request_id"] == "req-408-create"
    assert span_by_name["training.create_model.megatron.actor_create"]["attributes"]["base_model"] == "/tmp/qwen"
    assert span_by_name["training.create_model.megatron.reserve_gpus"]["attributes"]["world_size"] == 1


def test_issue_572_megatron_existing_actor_rank_mismatch_recreates(monkeypatch) -> None:
    from tinker_server.backend import megatron_distributed as md
    from tinker_server.backend import model_registry as model_registry
    from tinker_server.backend import model_actor_registry as model_actor_registry_mod
    from tinker_server import config as config_mod

    fake_new_actor = object()
    killed: list[dict[str, object]] = []
    created: list[dict[str, object]] = []

    class _RemoteMethod:
        def remote(self):
            return "diagnostics-ref"

    class _ActorHandle:
        get_diagnostics = _RemoteMethod()

    existing_actor = _ActorHandle()

    class _FakePool:
        def ensure_gpus_available(self, *_args, **_kwargs) -> bool:
            return True

        def reserve_gpus(self, *_args, **_kwargs) -> bool:
            return True

        def release_pending_gpus(self, *_args, **_kwargs) -> None:
            return None

        def register(self, **kwargs) -> None:
            created.append(dict(kwargs))

        def mark_ready(self, *_args, **_kwargs) -> bool:
            return True

    class _Options:
        def remote(self, **_kwargs):
            return fake_new_actor

    class _FakeMegatronWorkerGroup:
        @staticmethod
        def options(**_kwargs):
            return _Options()

    get_actor_calls = {"count": 0}

    def _fake_get_actor(*_args, **_kwargs):
        get_actor_calls["count"] += 1
        if get_actor_calls["count"] == 1:
            return existing_actor
        raise ValueError("missing")

    def _fake_get(value, timeout=None):
        if value == "diagnostics-ref":
            return {"lora_rank": 16}
        return value

    def _fake_kill(actor, **kwargs):
        assert actor is existing_actor
        killed.append(dict(kwargs))

    monkeypatch.setattr(model_actor_registry_mod, "get_model_actor_registry", lambda: _FakePool())
    monkeypatch.setattr(config_mod, "actor_runtime_env_vars", lambda **_kwargs: {})
    monkeypatch.setattr(config_mod, "otel_env_vars", lambda: {})
    monkeypatch.setattr(model_actor_registry_mod, "actor_observability_metadata", lambda _actor: {})
    monkeypatch.setattr(model_registry, "is_persistent_model", lambda _base_model: False)
    monkeypatch.setattr(md, "MegatronWorkerGroup", _FakeMegatronWorkerGroup)
    monkeypatch.setattr(md.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(md.ray, "get_actor", _fake_get_actor)
    monkeypatch.setattr(md.ray, "get", _fake_get)
    monkeypatch.setattr(md.ray_kill, "kill", _fake_kill)
    monkeypatch.setattr(
        md.ray.util,
        "get_placement_group",
        lambda _name: (_ for _ in ()).throw(ValueError("missing")),
        raising=False,
    )
    monkeypatch.setattr(
        md,
        "start_as_current_span_from_traceparent",
        lambda name, **kwargs: _span_recorder([], name, **kwargs),
    )

    actor = md.get_or_create_megatron_worker_group(
        base_model="/tmp/qwen",
        lora_rank=64,
        learning_rate=1e-4,
        session_id="sess-572-rank-mismatch",
    )

    assert actor is fake_new_actor
    assert killed[0]["reason"] == "megatron_actor_lora_rank_mismatch"
    assert killed[0]["observed_lora_rank"] == 16
    assert killed[0]["expected_lora_rank"] == 64
    assert created[-1]["metadata"]["max_lora_rank"] == 64


@pytest.mark.anyio
async def test_issue_408_async_get_or_create_megatron_worker_group_propagates_context(monkeypatch) -> None:
    from tinker_server.backend import megatron_distributed as md

    captured: dict[str, object] = {}

    def _fake_get_or_create(*args):
        captured["args"] = args
        return "actor-handle"

    async def _fake_to_thread(fn, *args):
        captured["fn"] = fn
        return fn(*args)

    monkeypatch.setattr(md, "get_or_create_megatron_worker_group", _fake_get_or_create)
    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(md, "get_current_traceparent", lambda: "00-" + ("c" * 32) + "-" + ("3" * 16) + "-01")
    monkeypatch.setattr(md, "get_request_id", lambda: "req-408-async")

    out = await md.async_get_or_create_megatron_worker_group(
        base_model="/tmp/qwen",
        lora_rank=16,
        learning_rate=2e-4,
        session_id="sess-async",
        observability_base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )

    assert out == "actor-handle"
    assert captured["fn"] is _fake_get_or_create
    assert captured["args"] == (
        "/tmp/qwen",
        16,
        2e-4,
        None,
        "sess-async",
        None,
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "00-" + ("c" * 32) + "-" + ("3" * 16) + "-01",
        "req-408-async",
    )
