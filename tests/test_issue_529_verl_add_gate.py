from __future__ import annotations

import asyncio
import sys
import types

import pytest

import tinker_server.backend.lora_utils as lora_utils
import tinker_server.backend.verl_inference as verl_inference


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeBaseVLLMHttpServer:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.config = types.SimpleNamespace(enable_rollout_routing_replay=False)


class _FakeLoRARequest:
    def __init__(self, *, lora_name: str, lora_int_id: int, lora_path: str):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path


class _FakeEngine:
    def __init__(self) -> None:
        self.list_started = asyncio.Event()
        self.add_started = asyncio.Event()
        self.list_calls = 0
        self.calls: list[tuple[int, str]] = []

    async def list_loras(self):
        self.list_calls += 1
        self.list_started.set()
        return set()

    async def add_lora(self, lora_request):
        self.calls.append((int(lora_request.lora_int_id), str(lora_request.lora_name)))
        self.add_started.set()
        await asyncio.sleep(0)


def _install_fake_runtime(monkeypatch):
    monkeypatch.setattr(verl_inference, "init_actor_observability", lambda: None)

    verl_mod = types.ModuleType("verl")
    workers_mod = types.ModuleType("verl.workers")
    rollout_mod = types.ModuleType("verl.workers.rollout")
    vllm_rollout_mod = types.ModuleType("verl.workers.rollout.vllm_rollout")
    async_server_mod = types.ModuleType("verl.workers.rollout.vllm_rollout.vllm_async_server")
    async_server_mod.vLLMHttpServer = _FakeBaseVLLMHttpServer
    utils_mod = types.ModuleType("verl.workers.rollout.vllm_rollout.utils")
    utils_mod.VLLM_LORA_INT_ID = 1
    utils_mod.VLLM_LORA_NAME = "default"
    utils_mod.VLLM_LORA_PATH = "/tmp/default-lora"
    monkeypatch.setitem(sys.modules, "verl", verl_mod)
    monkeypatch.setitem(sys.modules, "verl.workers", workers_mod)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout", rollout_mod)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout", vllm_rollout_mod)
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        async_server_mod,
    )
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.utils", utils_mod)

    vllm_mod = types.ModuleType("vllm")
    lora_mod = types.ModuleType("vllm.lora")
    request_mod = types.ModuleType("vllm.lora.request")
    request_mod.LoRARequest = _FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_mod)


def _make_actor_impl(monkeypatch):
    _install_fake_runtime(monkeypatch)
    remote_cls = verl_inference._create_extended_server_class()
    return remote_cls.__ray_metadata__.modified_class


@pytest.mark.anyio
async def test_issue_529_verl_default_add_lora_from_path_does_not_wait_for_active_generate(monkeypatch):
    monkeypatch.delenv("MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE", raising=False)
    monkeypatch.setattr(lora_utils, "validate_peft_adapter_checkpoint_shapes", lambda *_a, **_k: None)
    monkeypatch.setattr(verl_inference, "_mint_present_expert_ids_from_adapter_dir", lambda _path: {})

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor)
    actor.engine = _FakeEngine()
    actor.model_config = types.SimpleNamespace(local_path="/tmp/fake-model")

    await actor._register_generate_start()

    add_task = asyncio.create_task(
        impl_cls.add_lora_from_path(
            actor,
            lora_int_id=7,
            lora_path="/tmp/fake-lora",
            lora_name="tenant-b",
        )
    )

    await asyncio.wait_for(actor.engine.list_started.wait(), timeout=1.0)
    await asyncio.wait_for(actor.engine.add_started.wait(), timeout=1.0)
    assert actor._active_generates == 1

    await actor._register_generate_end()
    await asyncio.wait_for(add_task, timeout=1.0)
    assert actor.engine.list_calls == 1
    assert actor.engine.calls == [(7, "tenant-b")]


@pytest.mark.anyio
async def test_issue_529_verl_env_enables_idle_gate_for_add_lora_from_path(monkeypatch):
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE", "1")
    monkeypatch.setattr(lora_utils, "validate_peft_adapter_checkpoint_shapes", lambda *_a, **_k: None)
    monkeypatch.setattr(verl_inference, "_mint_present_expert_ids_from_adapter_dir", lambda _path: {})

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor)
    actor.engine = _FakeEngine()
    actor.model_config = types.SimpleNamespace(local_path="/tmp/fake-model")

    await actor._register_generate_start()

    add_task = asyncio.create_task(
        impl_cls.add_lora_from_path(
            actor,
            lora_int_id=8,
            lora_path="/tmp/fake-lora",
            lora_name="tenant-b",
        )
    )

    await asyncio.sleep(0.05)
    assert not actor.engine.list_started.is_set()
    assert not actor.engine.add_started.is_set()

    await actor._register_generate_end()
    await asyncio.wait_for(actor.engine.list_started.wait(), timeout=1.0)
    await asyncio.wait_for(actor.engine.add_started.wait(), timeout=1.0)
    await asyncio.wait_for(add_task, timeout=1.0)
    assert actor.engine.list_calls == 1
    assert actor.engine.calls == [(8, "tenant-b")]


def test_issue_529_verl_request_timing_defaults_enabled(monkeypatch):
    monkeypatch.delenv("MINT_VLLM_REQUEST_TIMING", raising=False)

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor)

    assert actor._timing is True


def test_issue_529_verl_request_timing_env_can_disable(monkeypatch):
    monkeypatch.setenv("MINT_VLLM_REQUEST_TIMING", "0")

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor)

    assert actor._timing is False
