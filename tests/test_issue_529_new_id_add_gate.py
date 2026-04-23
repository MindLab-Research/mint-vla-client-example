from __future__ import annotations

import asyncio
import sys
import types

import pytest

import tinker_server.backend.lora_utils as lora_utils
import tinker_server.backend.multinode_inference as mni


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeLoRARequest:
    def __init__(self, *, lora_name: str, lora_int_id: int, lora_path: str):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path


class _FakeEngine:
    def __init__(self) -> None:
        self.add_started = asyncio.Event()
        self.calls: list[tuple[int, str]] = []

    async def add_lora(self, lora_request):
        self.calls.append((int(lora_request.lora_int_id), str(lora_request.lora_name)))
        self.add_started.set()
        await asyncio.sleep(0)


def _install_fake_vllm(monkeypatch):
    vllm_mod = types.ModuleType("vllm")
    lora_mod = types.ModuleType("vllm.lora")
    request_mod = types.ModuleType("vllm.lora.request")
    request_mod.LoRARequest = _FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_mod)


def _make_actor_impl(monkeypatch):
    monkeypatch.setattr(mni, "init_actor_observability", lambda: None)
    remote_cls = mni._create_multinode_vllm_actor()
    return remote_cls.__ray_metadata__.modified_class


@pytest.mark.anyio
async def test_issue_529_default_new_id_add_waits_for_active_generate(monkeypatch):
    _install_fake_vllm(monkeypatch)
    monkeypatch.delenv("MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE", raising=False)
    monkeypatch.setattr(lora_utils, "validate_peft_adapter_checkpoint_shapes", lambda *_a, **_k: None)

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)
    actor.engine = _FakeEngine()

    await actor._register_generate_start()

    add_task = asyncio.create_task(
        impl_cls.add_lora(
            actor,
            lora_int_id=7,
            lora_path="/tmp/fake-lora",
            lora_name="tenant-b",
        )
    )

    await asyncio.sleep(0.05)
    assert not actor.engine.add_started.is_set()

    await actor._register_generate_end()
    await asyncio.wait_for(actor.engine.add_started.wait(), timeout=1.0)
    await asyncio.wait_for(add_task, timeout=1.0)
    assert actor.engine.calls == [(7, "tenant-b")]


@pytest.mark.anyio
async def test_issue_529_env_disables_idle_gate_for_new_id_add(monkeypatch):
    _install_fake_vllm(monkeypatch)
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE", "0")
    monkeypatch.setattr(lora_utils, "validate_peft_adapter_checkpoint_shapes", lambda *_a, **_k: None)

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)
    actor.engine = _FakeEngine()

    await actor._register_generate_start()

    add_task = asyncio.create_task(
        impl_cls.add_lora(
            actor,
            lora_int_id=8,
            lora_path="/tmp/fake-lora",
            lora_name="tenant-b",
        )
    )

    await asyncio.wait_for(actor.engine.add_started.wait(), timeout=1.0)
    assert actor._active_generates == 1

    await actor._register_generate_end()
    await asyncio.wait_for(add_task, timeout=1.0)
    assert actor.engine.calls == [(8, "tenant-b")]
