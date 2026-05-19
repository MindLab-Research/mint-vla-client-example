from __future__ import annotations

import asyncio
import sys
import types

import pytest

import mint_server.backend.lora_utils as lora_utils
import mint_server.backend.multinode_inference as mni


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


def test_mint_vllm_multinode_async_engine_imports_top_level_exports(monkeypatch):
    class TopLevelArgs:
        pass

    class TopLevelEngine:
        pass

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.AsyncEngineArgs = TopLevelArgs
    vllm_mod.AsyncLLMEngine = TopLevelEngine
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)

    assert mni._import_vllm_async_engine_components() == (TopLevelArgs, TopLevelEngine)


def test_mint_vllm_multinode_async_engine_imports_legacy_submodules(monkeypatch):
    class LegacyArgs:
        pass

    class LegacyEngine:
        pass

    vllm_mod = types.ModuleType("vllm")
    engine_pkg = types.ModuleType("vllm.engine")
    arg_utils_mod = types.ModuleType("vllm.engine.arg_utils")
    async_engine_mod = types.ModuleType("vllm.engine.async_llm_engine")
    arg_utils_mod.AsyncEngineArgs = LegacyArgs
    async_engine_mod.AsyncLLMEngine = LegacyEngine
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.async_llm_engine", async_engine_mod)

    assert mni._import_vllm_async_engine_components() == (LegacyArgs, LegacyEngine)


def test_mint_vllm_multinode_child_env_disables_tensorflow_and_flax(monkeypatch):
    monkeypatch.delenv("USE_TF", raising=False)
    monkeypatch.delenv("USE_FLAX", raising=False)
    monkeypatch.setattr(mni.os.path, "isdir", lambda _path: False)
    monkeypatch.setattr(mni, "preferred_torch_lib_dirs", lambda: [])

    mni._stabilize_vllm_child_environment()

    assert mni.os.environ["USE_TF"] == "0"
    assert mni.os.environ["USE_FLAX"] == "0"


def test_mint_vllm_multinode_runtime_env_disables_tensorflow_and_flax_by_default():
    env_vars: dict[str, str] = {}

    mni._set_default_vllm_runtime_env(env_vars)

    assert env_vars["USE_TF"] == "0"
    assert env_vars["USE_FLAX"] == "0"


def _make_actor_impl(monkeypatch):
    monkeypatch.setattr(mni, "init_actor_observability", lambda: None)
    remote_cls = mni._create_mint_vllm_multinode_actor()
    return remote_cls.__ray_metadata__.modified_class


@pytest.mark.anyio
async def test_issue_529_default_new_id_add_does_not_wait_for_active_generate(monkeypatch):
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

    await asyncio.wait_for(actor.engine.add_started.wait(), timeout=1.0)
    assert actor._active_generates == 1

    await actor._register_generate_end()
    await asyncio.wait_for(add_task, timeout=1.0)
    assert actor.engine.calls == [(7, "tenant-b")]


@pytest.mark.anyio
async def test_issue_529_env_enables_idle_gate_for_new_id_add(monkeypatch):
    _install_fake_vllm(monkeypatch)
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE", "1")
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

    await asyncio.sleep(0.05)
    assert not actor.engine.add_started.is_set()

    await actor._register_generate_end()
    await asyncio.wait_for(actor.engine.add_started.wait(), timeout=1.0)
    await asyncio.wait_for(add_task, timeout=1.0)
    assert actor.engine.calls == [(8, "tenant-b")]


def test_issue_529_request_timing_defaults_enabled(monkeypatch):
    _install_fake_vllm(monkeypatch)
    monkeypatch.delenv("MINT_VLLM_REQUEST_TIMING", raising=False)

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)

    assert actor._timing is True


def test_issue_529_request_timing_env_can_disable(monkeypatch):
    _install_fake_vllm(monkeypatch)
    monkeypatch.setenv("MINT_VLLM_REQUEST_TIMING", "0")

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)

    assert actor._timing is False
