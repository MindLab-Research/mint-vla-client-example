from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import tinker_server.backend.dense_trainer as dense_trainer
from tinker_server.backend.verl_training import VerlTrainingEngine
from tinker_server.backend.training_session_manager import TrainingSession


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _AwaitableRemote:
    def __init__(self, value):
        self._value = value

    def remote(self, *args, **kwargs):
        _ = args, kwargs
        fut = asyncio.Future()
        fut.set_result(self._value)
        return fut


class _RemoteCall:
    def remote(self, *args, **kwargs):
        _ = args, kwargs
        return object()


@pytest.mark.anyio
async def test_issue_364_get_tokenizer_info_allows_dense_recover(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="run-364-rebind",
        session_id="session-364",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    calls: list[tuple[str, bool]] = []
    worker = SimpleNamespace(get_tokenizer_info=_AwaitableRemote({"model_name": session.base_model}))

    async def _fake_get_live_worker(s, *, op: str, allow_recover: bool = False):
        assert s is session
        calls.append((op, allow_recover))
        return worker

    monkeypatch.setattr(engine, "_get_live_worker", _fake_get_live_worker)

    out = await engine.get_tokenizer_info(session)

    assert out["model_name"] == "Qwen/Qwen3-0.6B"
    assert calls == [("get_tokenizer_info", False)]


@pytest.mark.anyio
async def test_issue_364_save_dense_lora_weights_allows_dense_recover(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="run-364-save",
        session_id="session-364",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    calls: list[tuple[str, bool]] = []
    worker = SimpleNamespace(save_lora_weights=_AwaitableRemote({"ok": True}))

    async def _fake_get_live_worker(s, *, op: str, allow_recover: bool = False):
        assert s is session
        calls.append((op, allow_recover))
        return worker

    async def _fake_await_with_keepalive(ref, _session, interval_s: float = 30.0, timeout_s=None):
        _ = _session, interval_s, timeout_s
        return await ref

    monkeypatch.setattr(engine, "_get_live_worker", _fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", _fake_await_with_keepalive)
    monkeypatch.setenv("MINT_SAVE_LORA_TIMEOUT_S", "30")

    out = await engine.save_dense_lora_weights_for_sampler(session, str(tmp_path / "ckpt"))

    assert out == str((tmp_path / "ckpt").resolve())
    assert calls == [("save_dense_lora_weights_for_sampler", False)]


@pytest.mark.anyio
async def test_issue_561_dense_fatal_error_retires_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="run-561-fatal",
        session_id="session-561-fatal",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = "peft_trainer_qwen__qwen3_0_6b_maxr64"
    session.namespace = "tinker"
    engine._resource_pool_actor_names[session.model_id] = session.actor_name

    worker = SimpleNamespace(forward_backward=_RemoteCall())

    class AcceleratorError(RuntimeError):
        pass

    class _RayTaskError(RuntimeError):
        def __init__(self, msg: str, *, cause=None) -> None:
            super().__init__(msg)
            self.cause = cause

    async def _fake_get_live_worker(s, *, op: str, allow_recover: bool = False):
        assert s is session
        assert op == "forward_backward"
        assert allow_recover is False
        return worker

    async def _fake_await_with_keepalive(ref, _session, interval_s: float = 30.0, timeout_s=None):
        _ = ref, _session, interval_s, timeout_s
        raise _RayTaskError(
            "RayTaskError(AcceleratorError)",
            cause=AcceleratorError("CUDA error: device-side assert triggered"),
        )

    retire_calls: list[dict[str, object]] = []

    def _fake_retire_dense_trainer(**kwargs) -> None:
        retire_calls.append(dict(kwargs))

    monkeypatch.setattr(engine, "_get_live_worker", _fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", _fake_await_with_keepalive)
    monkeypatch.setattr(dense_trainer, "retire_dense_trainer", _fake_retire_dense_trainer)

    request = SimpleNamespace(
        forward_backward_input=SimpleNamespace(
            data=[SimpleNamespace(model_dump=lambda: {"model_input": {}, "loss_fn_inputs": {}})],
            loss_fn="cross_entropy",
            loss_fn_config={},
        )
    )

    with pytest.raises(_RayTaskError, match="RayTaskError"):
        await engine.forward_backward(session, request)

    assert len(retire_calls) == 1
    assert retire_calls[0]["actor_name"] == "peft_trainer_qwen__qwen3_0_6b_maxr64"
    assert retire_calls[0]["session_id"] == session.model_id
    assert "forward_backward" in str(retire_calls[0]["reason"])
    assert "device-side assert" in str(retire_calls[0]["reason"])
    assert session.model_id not in engine._resource_pool_actor_names
    assert session.actor_name is None
    assert session.namespace is None


@pytest.mark.anyio
async def test_issue_561_rebind_refuses_poisoned_dense_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="run-561-rebind",
        session_id="session-561-rebind",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = "peft_trainer_qwen__qwen3_0_6b_maxr64"
    session.namespace = "tinker"

    monkeypatch.setattr(
        dense_trainer,
        "dense_trainer_reuse_block_reason",
        lambda actor_name: "forward_backward:CUDA error: device-side assert triggered",
    )
    monkeypatch.setattr(
        "tinker_server.backend.verl_training.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("poisoned actor must not be rebound")),
    )

    worker = await engine._rebind_worker_from_session_metadata(session, reason="issue-561")

    assert worker is None
