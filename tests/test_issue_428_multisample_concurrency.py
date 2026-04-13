from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

import tinker_server.backend.multinode_inference as mni
import tinker_server.backend.vllm_stop as vllm_stop


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTokensPrompt:
    def __init__(self, *, prompt_token_ids):
        self.prompt_token_ids = list(prompt_token_ids)


class _FakeLoRARequest:
    def __init__(self, *, lora_name: str, lora_int_id: int, lora_path: str):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path


class _FakeOutput:
    def __init__(self, *, index: int):
        self.index = index
        self.token_ids = [11 + index, 21 + index]
        self.logprobs = None
        self.finish_reason = "stop"
        self.routed_experts = None


class _FakeBatch:
    def __init__(self, outputs, *, finished: bool = True):
        self.outputs = list(outputs)
        self.finished = finished


class _FakeCollector:
    def __init__(self, *, payload, wait_event: asyncio.Event | None = None):
        self._payload = payload
        self._wait_event = wait_event

    def get_nowait(self):
        return None

    async def get(self):
        if self._wait_event is not None:
            await self._wait_event.wait()
        return self._payload


class _FakeEngine:
    def __init__(
        self,
        *,
        multisample_enqueued: asyncio.Event,
        ordinary_enqueued: asyncio.Event,
        release_multisample: asyncio.Event,
        second_multisample_enqueued: asyncio.Event | None = None,
    ):
        self.multisample_enqueued = multisample_enqueued
        self.ordinary_enqueued = ordinary_enqueued
        self.release_multisample = release_multisample
        self.second_multisample_enqueued = second_multisample_enqueued
        self.calls: list[tuple[str, int]] = []
        self.abort_calls: list[str] = []

    async def add_request(self, *, request_id, prompt, params, lora_request=None):
        del prompt, lora_request
        self.calls.append((str(request_id), int(params.n)))
        if request_id == "multi":
            self.multisample_enqueued.set()
            return _FakeCollector(
                payload=_FakeBatch([_FakeOutput(index=0), _FakeOutput(index=1)]),
                wait_event=self.release_multisample,
            )
        if request_id == "multi2":
            assert self.second_multisample_enqueued is not None
            self.second_multisample_enqueued.set()
            return _FakeCollector(
                payload=_FakeBatch([_FakeOutput(index=0), _FakeOutput(index=1)]),
            )
        if request_id == "ordinary":
            self.ordinary_enqueued.set()
            return _FakeCollector(payload=_FakeBatch([_FakeOutput(index=0)]))
        raise AssertionError(f"unexpected request_id: {request_id}")

    async def abort(self, request_id):
        self.abort_calls.append(str(request_id))


def _install_fake_vllm(monkeypatch):
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.SamplingParams = _FakeSamplingParams

    inputs_mod = types.ModuleType("vllm.inputs")
    inputs_mod.TokensPrompt = _FakeTokensPrompt

    lora_mod = types.ModuleType("vllm.lora")
    request_mod = types.ModuleType("vllm.lora.request")
    request_mod.LoRARequest = _FakeLoRARequest

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_mod)


def _make_actor_impl(monkeypatch):
    monkeypatch.setattr(mni, "init_actor_observability", lambda: None)
    remote_cls = mni._create_multinode_vllm_actor()
    impl_cls = remote_cls.__ray_metadata__.modified_class
    return impl_cls


def _stub_future_store(monkeypatch):
    async def _noop_async_update_meta(*args, **kwargs):
        return None

    future_store_module = importlib.import_module("tinker_server.backend.future_store")
    monkeypatch.setattr(
        future_store_module.future_store,
        "async_update_meta",
        _noop_async_update_meta,
    )


@pytest.mark.anyio
async def test_issue_428_default_multisample_mode_preserves_cross_request_concurrency(monkeypatch):
    monkeypatch.delenv("MINT_VLLM_MULTISAMPLE_MODE", raising=False)
    monkeypatch.delenv("MINT_VLLM_SERIALIZE_MULTISAMPLE", raising=False)
    monkeypatch.delenv("MINT_VLLM_SERIALIZE_GENERATE", raising=False)
    monkeypatch.delenv("MINT_VLLM_SERIALIZE_ADD_REQUEST", raising=False)

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)

    assert actor._multisample_mode == "vllm_n"
    assert actor._serialize_multisample is True
    assert actor._multisample_lock is not None


@pytest.mark.anyio
async def test_issue_428_multisample_request_does_not_block_ordinary_request_entry(monkeypatch):
    _install_fake_vllm(monkeypatch)
    _stub_future_store(monkeypatch)
    monkeypatch.setattr(mni, "init_actor_observability", lambda: None)
    monkeypatch.setattr(mni.server_config, "router_replay_mode", "disabled", raising=False)
    monkeypatch.setattr(vllm_stop, "vllm_stop_kwargs", lambda stop, default_stop_token_ids=None: {})

    monkeypatch.setenv("MINT_VLLM_MULTISAMPLE_MODE", "vllm_n")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_MULTISAMPLE", "1")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_GENERATE", "0")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_ADD_REQUEST", "1")

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)

    multisample_enqueued = asyncio.Event()
    ordinary_enqueued = asyncio.Event()
    release_multisample = asyncio.Event()
    actor.engine = _FakeEngine(
        multisample_enqueued=multisample_enqueued,
        ordinary_enqueued=ordinary_enqueued,
        release_multisample=release_multisample,
    )

    multi_task = asyncio.create_task(
        impl_cls.generate(
            actor,
            prompt_ids=[1, 2, 3],
            request_id="multi",
            lora_int_id=None,
            lora_path=None,
            max_tokens=8,
            n=2,
        )
    )
    await asyncio.wait_for(multisample_enqueued.wait(), timeout=5.0)

    ordinary_task = asyncio.create_task(
        impl_cls.generate(
            actor,
            prompt_ids=[9, 8, 7],
            request_id="ordinary",
            lora_int_id=None,
            lora_path=None,
            max_tokens=8,
            n=1,
        )
    )
    await asyncio.wait_for(ordinary_enqueued.wait(), timeout=5.0)

    assert not multi_task.done()

    release_multisample.set()
    multi_result, ordinary_result = await asyncio.gather(multi_task, ordinary_task)

    assert len(multi_result) == 2
    assert ordinary_result["token_ids"] == [11, 21]
    assert actor.engine.calls[0] == ("multi", 2)
    assert actor.engine.calls[1] == ("ordinary", 1)


@pytest.mark.anyio
async def test_issue_428_vllm_n_requests_remain_isolated_from_each_other(monkeypatch):
    _install_fake_vllm(monkeypatch)
    _stub_future_store(monkeypatch)
    monkeypatch.setattr(mni, "init_actor_observability", lambda: None)
    monkeypatch.setattr(mni.server_config, "router_replay_mode", "disabled", raising=False)
    monkeypatch.setattr(vllm_stop, "vllm_stop_kwargs", lambda stop, default_stop_token_ids=None: {})

    monkeypatch.setenv("MINT_VLLM_MULTISAMPLE_MODE", "vllm_n")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_MULTISAMPLE", "1")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_GENERATE", "0")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_ADD_REQUEST", "1")

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)

    first_multisample_enqueued = asyncio.Event()
    second_multisample_enqueued = asyncio.Event()
    release_multisample = asyncio.Event()
    actor.engine = _FakeEngine(
        multisample_enqueued=first_multisample_enqueued,
        ordinary_enqueued=asyncio.Event(),
        release_multisample=release_multisample,
        second_multisample_enqueued=second_multisample_enqueued,
    )

    task1 = asyncio.create_task(
        impl_cls.generate(
            actor,
            prompt_ids=[1, 2, 3],
            request_id="multi",
            lora_int_id=None,
            lora_path=None,
            max_tokens=8,
            n=2,
        )
    )
    await asyncio.wait_for(first_multisample_enqueued.wait(), timeout=5.0)

    task2 = asyncio.create_task(
        impl_cls.generate(
            actor,
            prompt_ids=[4, 5, 6],
            request_id="multi2",
            lora_int_id=None,
            lora_path=None,
            max_tokens=8,
            n=2,
        )
    )

    await asyncio.wait_for(second_multisample_enqueued.wait(), timeout=5.0)
    assert not task1.done()

    release_multisample.set()
    result1, result2 = await asyncio.gather(task1, task2)

    assert len(result1) == 2
    assert len(result2) == 2
    assert actor.engine.calls[0] == ("multi", 2)
    assert actor.engine.calls[1] == ("multi2", 2)


@pytest.mark.anyio
async def test_issue_428_concurrent_n1_failure_aborts_remaining_subrequests(monkeypatch):
    _install_fake_vllm(monkeypatch)
    _stub_future_store(monkeypatch)
    monkeypatch.setattr(mni, "init_actor_observability", lambda: None)
    monkeypatch.setattr(mni.server_config, "router_replay_mode", "disabled", raising=False)
    monkeypatch.setattr(vllm_stop, "vllm_stop_kwargs", lambda stop, default_stop_token_ids=None: {})

    monkeypatch.setenv("MINT_VLLM_MULTISAMPLE_MODE", "concurrent_n1")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_MULTISAMPLE", "0")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_GENERATE", "0")
    monkeypatch.setenv("MINT_VLLM_SERIALIZE_ADD_REQUEST", "1")

    impl_cls = _make_actor_impl(monkeypatch)
    actor = object.__new__(impl_cls)
    impl_cls.__init__(actor, model_path="fake-model", tensor_parallel_size=1)

    release_hanging = asyncio.Event()

    class _FailureCollector(_FakeCollector):
        async def get(self):
            raise RuntimeError("subrequest boom")

    class _FailingEngine:
        def __init__(self):
            self.abort_calls: list[str] = []
            self.calls: list[tuple[str, int]] = []

        async def add_request(self, *, request_id, prompt, params, lora_request=None):
            del prompt, lora_request
            self.calls.append((str(request_id), int(params.n)))
            if request_id.endswith("_s0"):
                return _FailureCollector(payload=_FakeBatch([_FakeOutput(index=0)]))
            return _FakeCollector(
                payload=_FakeBatch([_FakeOutput(index=0)]),
                wait_event=release_hanging,
            )

        async def abort(self, request_id):
            self.abort_calls.append(str(request_id))
            release_hanging.set()

    actor.engine = _FailingEngine()

    with pytest.raises(RuntimeError, match="subrequest boom"):
        await impl_cls.generate(
            actor,
            prompt_ids=[1, 2, 3],
            request_id="multi",
            lora_int_id=None,
            lora_path=None,
            max_tokens=8,
            n=2,
        )

    assert set(actor.engine.abort_calls) == {"multi", "multi_s0", "multi_s1"}
    assert actor._outer_to_subreq_ids == {}

    # After cleanup, a fresh single-sample request should still execute normally.
    actor.engine = _FakeEngine(
        multisample_enqueued=asyncio.Event(),
        ordinary_enqueued=asyncio.Event(),
        release_multisample=asyncio.Event(),
    )
    result = await impl_cls.generate(
        actor,
        prompt_ids=[9, 8, 7],
        request_id="ordinary",
        lora_int_id=None,
        lora_path=None,
        max_tokens=8,
        n=1,
    )
    assert result["token_ids"] == [11, 21]
