from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _import_training_route():
    sys.modules.setdefault("tinker_server.routes.service", types.ModuleType("tinker_server.routes.service"))
    return importlib.import_module("tinker_server.routes.training")


async def _identity_materialize(session):
    return session


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_issue_408_save_weights_for_sampler_registers_lazy_multilora_session(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.backend import session_index_store as sis
    from tinker_server.models.types import SaveWeightsForSamplerRequest

    tr = _import_training_route()
    monkeypatch.setattr(tr, "_materialize_training_session_for_stateful_use", _identity_materialize)

    ckpt_dir = tmp_path / "sampler_ephemeral"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, object] = {}
    failures: list[tuple[str, str]] = []
    registration: dict[str, object] = {}
    warm_calls: list[str] = []
    warm_gate = asyncio.Event()
    scheduled_tasks: list[asyncio.Task[object]] = []

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    async def _async_fail(request_id: str, error: str) -> None:
        failures.append((request_id, error))

    class _InferenceManagerStub:
        tensor_parallel_size = 1
        data_parallel_size = 1
        gpu_memory_utilization = 0.8
        max_model_len = 4096

        async def get_engine_for_model(self, base_model: str):
            warm_calls.append(base_model)
            await warm_gate.wait()
            return object()

        def register_multi_lora_session(self, **kwargs) -> None:
            registration.update(kwargs)

    def _resolve(request_id: str, response: dict) -> None:
        resolved["request_id"] = request_id
        resolved["response"] = response

    async def _async_resolve(request_id: str, response: dict) -> None:
        _resolve(request_id, response)

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
        "task_futures",
        SimpleNamespace(async_resolve=_async_resolve, async_fail=_async_fail),
    )
    monkeypatch.setattr(tr, "checkpoint_has_optimizer_state", lambda _path: False)
    monkeypatch.setattr(tr, "validate_sampler_checkpoint_for_sampling", lambda _path: None)
    monkeypatch.setattr(tr, "write_checkpoint_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tr, "build_ephemeral_checkpoint_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(sis, "add_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "add_heartbeat_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "upsert_sampler_index", lambda _payload: None)

    orig_create_task = tr.asyncio.create_task

    def _capture_task(coro):
        task = orig_create_task(coro)
        scheduled_tasks.append(task)
        return task

    monkeypatch.setattr(tr.asyncio, "create_task", _capture_task)

    request = SaveWeightsForSamplerRequest(model_id="run-408", seq_id=0)
    await asyncio.wait_for(
        tr._do_save_weights_for_sampler(
        request_id="req-408-save",
        request=request,
        user_id="owner-408",
        prefer_tinker=True,
        ),
        timeout=0.1,
    )
    await asyncio.sleep(0)

    assert failures == []
    assert warm_calls == ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert len(scheduled_tasks) == 1
    assert resolved["request_id"] == "req-408-save"
    response = resolved["response"]
    assert isinstance(response, dict)
    sampling_session_id = response["sampling_session_id"]
    assert isinstance(sampling_session_id, str) and sampling_session_id
    assert response["path"] is None
    assert registration == {
        "session_id": sampling_session_id,
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "lora_rank": 32,
        "adapter_path": str(ckpt_dir),
        "lora_loaded": False,
    }

    for task in scheduled_tasks:
        task.cancel()
    await asyncio.gather(*scheduled_tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_issue_408_save_weights_for_sampler_reuses_pending_warm_task(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.backend import session_index_store as sis
    from tinker_server.models.types import SaveWeightsForSamplerRequest

    tr = _import_training_route()
    monkeypatch.setattr(tr, "_materialize_training_session_for_stateful_use", _identity_materialize)

    ckpt_dir = tmp_path / "sampler_ephemeral"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved: list[tuple[str, dict[str, object]]] = []
    registration_calls: list[dict[str, object]] = []
    warm_calls: list[str] = []
    warm_gate = asyncio.Event()
    scheduled_tasks: list[asyncio.Task[object]] = []

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    class _InferenceManagerStub:
        tensor_parallel_size = 1
        data_parallel_size = 1
        gpu_memory_utilization = 0.8
        max_model_len = 4096

        async def get_engine_for_model(self, base_model: str):
            warm_calls.append(base_model)
            await warm_gate.wait()
            return object()

        def register_multi_lora_session(self, **kwargs) -> None:
            registration_calls.append(dict(kwargs))

    def _resolve(request_id: str, response: dict) -> None:
        resolved.append((request_id, response))

    async def _async_resolve(request_id: str, response: dict) -> None:
        _resolve(request_id, response)

    inference_manager = _InferenceManagerStub()
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
    monkeypatch.setattr(tr, "inference_manager", inference_manager)
    monkeypatch.setattr(
        tr,
        "task_futures",
        SimpleNamespace(async_resolve=_async_resolve, async_fail=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(tr, "checkpoint_has_optimizer_state", lambda _path: False)
    monkeypatch.setattr(tr, "validate_sampler_checkpoint_for_sampling", lambda _path: None)
    monkeypatch.setattr(tr, "write_checkpoint_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tr, "build_ephemeral_checkpoint_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(sis, "add_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "add_heartbeat_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "upsert_sampler_index", lambda _payload: None)

    orig_create_task = tr.asyncio.create_task

    def _capture_task(coro):
        task = orig_create_task(coro)
        scheduled_tasks.append(task)
        return task

    monkeypatch.setattr(tr.asyncio, "create_task", _capture_task)

    request = SaveWeightsForSamplerRequest(model_id="run-408", seq_id=0)
    await tr._do_save_weights_for_sampler(
        request_id="req-408-save-1",
        request=request,
        user_id="owner-408",
        prefer_tinker=True,
    )
    await tr._do_save_weights_for_sampler(
        request_id="req-408-save-2",
        request=request,
        user_id="owner-408",
        prefer_tinker=True,
    )

    assert warm_calls == ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert len(scheduled_tasks) == 1
    assert len(resolved) == 2
    assert len(registration_calls) == 2
    assert all(call[1]["path"] is None for call in resolved)

    for task in scheduled_tasks:
        task.cancel()
    await asyncio.gather(*scheduled_tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_issue_408_save_weights_for_sampler_fails_fast_on_immediate_engine_error(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.backend import session_index_store as sis
    from tinker_server.models.types import SaveWeightsForSamplerRequest

    tr = _import_training_route()
    monkeypatch.setattr(tr, "_materialize_training_session_for_stateful_use", _identity_materialize)

    ckpt_dir = tmp_path / "sampler_ephemeral"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, object] = {}
    failures: list[tuple[str, str]] = []
    registration: dict[str, object] = {}
    warm_calls: list[str] = []

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    async def _async_fail(request_id: str, error: str) -> None:
        failures.append((request_id, error))

    async def _async_resolve(request_id: str, response: dict) -> None:
        resolved.update(request_id=request_id, response=response)

    class _InferenceManagerStub:
        tensor_parallel_size = 1
        data_parallel_size = 1
        gpu_memory_utilization = 0.8
        max_model_len = 4096

        async def get_engine_for_model(self, base_model: str):
            warm_calls.append(base_model)
            raise RuntimeError("ray not connected")

        def register_multi_lora_session(self, **kwargs) -> None:
            registration.update(kwargs)

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
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )
    monkeypatch.setattr(tr, "checkpoint_has_optimizer_state", lambda _path: False)
    monkeypatch.setattr(tr, "validate_sampler_checkpoint_for_sampling", lambda _path: None)
    monkeypatch.setattr(tr, "write_checkpoint_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tr, "build_ephemeral_checkpoint_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(sis, "add_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "add_heartbeat_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "upsert_sampler_index", lambda _payload: None)

    request = SaveWeightsForSamplerRequest(model_id="run-408", seq_id=0)
    await tr._do_save_weights_for_sampler(
        request_id="req-408-save",
        request=request,
        user_id="owner-408",
        prefer_tinker=True,
    )

    assert warm_calls == ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert failures == [("req-408-save", "ray not connected")]
    assert resolved == {}
    assert registration == {}


@pytest.mark.anyio
async def test_issue_408_save_weights_for_sampler_fails_fast_on_async_warm_error(
    monkeypatch, tmp_path: Path
) -> None:
    from tinker_server.backend import session_index_store as sis
    from tinker_server.models.types import SaveWeightsForSamplerRequest

    tr = _import_training_route()
    monkeypatch.setattr(tr, "_materialize_training_session_for_stateful_use", _identity_materialize)

    ckpt_dir = tmp_path / "sampler_ephemeral"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, object] = {}
    failures: list[tuple[str, str]] = []
    registration: dict[str, object] = {}
    warm_calls: list[str] = []

    async def _fake_save_weights_for_sampler(**_kwargs):
        return str(ckpt_dir)

    async def _async_fail(request_id: str, error: str) -> None:
        failures.append((request_id, error))

    async def _async_resolve(request_id: str, response: dict) -> None:
        resolved.update(request_id=request_id, response=response)

    class _InferenceManagerStub:
        tensor_parallel_size = 1
        data_parallel_size = 1
        gpu_memory_utilization = 0.8
        max_model_len = 4096

        async def get_engine_for_model(self, base_model: str):
            warm_calls.append(base_model)
            await asyncio.sleep(0)
            raise RuntimeError("async warm failed")

        def register_multi_lora_session(self, **kwargs) -> None:
            registration.update(kwargs)

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
        "task_futures",
        SimpleNamespace(
            async_resolve=_async_resolve,
            async_fail=_async_fail,
        ),
    )
    monkeypatch.setattr(tr, "checkpoint_has_optimizer_state", lambda _path: False)
    monkeypatch.setattr(tr, "validate_sampler_checkpoint_for_sampling", lambda _path: None)
    monkeypatch.setattr(tr, "write_checkpoint_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tr, "build_ephemeral_checkpoint_dir", lambda **_kwargs: str(ckpt_dir))
    monkeypatch.setattr(sis, "add_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "add_heartbeat_sampler_to_session", lambda **_kwargs: None)
    monkeypatch.setattr(sis, "upsert_sampler_index", lambda _payload: None)

    request = SaveWeightsForSamplerRequest(model_id="run-408", seq_id=0)
    await tr._do_save_weights_for_sampler(
        request_id="req-408-save",
        request=request,
        user_id="owner-408",
        prefer_tinker=True,
    )

    assert warm_calls == ["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    assert failures == [("req-408-save", "async warm failed")]
    assert resolved == {}
    assert registration == {}
