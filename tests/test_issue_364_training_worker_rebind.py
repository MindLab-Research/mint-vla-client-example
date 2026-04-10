from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

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
