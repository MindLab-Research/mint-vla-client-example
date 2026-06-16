from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import mint_server.backend.training.dense.dense_trainer as dense_trainer
import mint_server.backend.observability.runtime_observability as runtime_obs_module
from mint_server.backend.training.verl.verl_training import TrainingWorker, VerlTrainingEngine
from mint_server.backend.training.training_session_manager import TrainingSession


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
async def test_save_dense_lora_weights_materializes_worker_payload_on_api_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import json

    import torch
    from safetensors.torch import load_file

    engine = VerlTrainingEngine()
    session = TrainingSession(
        model_id="run-dense-sampler-materialize",
        session_id="session-dense-sampler-materialize",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    payload = {
        "state_dict": {"adapter.lora_A.weight": torch.zeros(2, 3)},
        "peft_config": {"r": 2, "lora_alpha": 4, "target_modules": ["adapter"]},
        "current_step": 7,
        "learning_rate": 3e-4,
    }
    worker = SimpleNamespace(save_lora_weights=_AwaitableRemote(payload))

    async def _fake_get_live_worker(s, *, op: str, allow_recover: bool = False):
        assert s is session
        return worker

    async def _fake_await_with_keepalive(ref, _session, interval_s: float = 30.0, timeout_s=None):
        _ = _session, interval_s, timeout_s
        return await ref

    monkeypatch.setattr(engine, "_get_live_worker", _fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", _fake_await_with_keepalive)

    out = await engine.save_dense_lora_weights_for_sampler(session, str(tmp_path / "ckpt"))

    assert out == str((tmp_path / "ckpt").resolve())
    assert set(load_file(str(tmp_path / "ckpt" / "adapter_model.safetensors")).keys()) == {
        "adapter.lora_A.weight"
    }
    assert json.loads((tmp_path / "ckpt" / "adapter_config.json").read_text())["r"] == 2
    assert json.loads((tmp_path / "ckpt" / "training_meta.json").read_text()) == {
        "current_step": 7,
        "learning_rate": 3e-4,
    }


@pytest.mark.anyio
async def test_issue_561_dense_fatal_error_retires_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    obs = runtime_obs_module.RuntimeObservability()
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", obs)
    session = TrainingSession(
        model_id="run-561-fatal",
        session_id="session-561-fatal",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = "mint_dense_qwen__qwen3_0_6b"
    session.namespace = "mint"
    engine._model_actor_supervisor_actor_names[session.model_id] = session.actor_name

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

    def _fake_retire_dense_trainer(**kwargs) -> str:
        retire_calls.append(dict(kwargs))
        return "ok"

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
    assert retire_calls[0]["actor_name"] == "mint_dense_qwen__qwen3_0_6b"
    assert retire_calls[0]["session_id"] == session.model_id
    assert retire_calls[0]["fatal_op"] == "forward_backward"
    assert "forward_backward" in str(retire_calls[0]["reason"])
    assert "device-side assert" in str(retire_calls[0]["reason"])
    assert session.model_id not in engine._model_actor_supervisor_actor_names
    assert session.actor_name is None
    assert session.namespace is None
    snap = obs.snapshot()
    assert len(snap["training_operation_latency"]) == 1
    op_row = snap["training_operation_latency"][0]
    assert op_row["base_model"] == "Qwen/Qwen3-0.6B"
    assert op_row["backend"] == "peft"
    assert op_row["op"] == "forward_backward"
    assert op_row["status"] == "error"
    assert op_row["failure_class"] == "cuda_fatal"
    assert op_row["count"] == 1
    assert op_row["duration_s_total"] >= 0.0
    assert op_row["duration_s_max"] >= 0.0
    assert snap["dense_actor_fatal"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "op": "forward_backward",
            "failure_class": "cuda_fatal",
            "count": 1,
        }
    ]


@pytest.mark.anyio
async def test_issue_561_dense_retire_failure_hard_poisons_session(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    obs = runtime_obs_module.RuntimeObservability()
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", obs)
    session = TrainingSession(
        model_id="run-561-retire-failed",
        session_id="session-561-retire-failed",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = "mint_dense_qwen__qwen3_0_6b"
    session.namespace = "mint"
    engine._model_actor_supervisor_actor_names[session.model_id] = session.actor_name

    worker = SimpleNamespace(forward_backward=_RemoteCall())

    class AcceleratorError(RuntimeError):
        pass

    class _RayTaskError(RuntimeError):
        def __init__(self, msg: str, *, cause=None) -> None:
            super().__init__(msg)
            self.cause = cause

    async def _fake_get_live_worker(s, *, op: str, allow_recover: bool = False):
        assert s is session
        return worker

    async def _fake_await_with_keepalive(ref, _session, interval_s: float = 30.0, timeout_s=None):
        _ = ref, _session, interval_s, timeout_s
        raise _RayTaskError(
            "RayTaskError(AcceleratorError)",
            cause=AcceleratorError("CUDA error: device-side assert triggered"),
        )

    monkeypatch.setattr(engine, "_get_live_worker", _fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", _fake_await_with_keepalive)
    monkeypatch.setattr(dense_trainer, "retire_dense_trainer", lambda **kwargs: "kill_failed")

    request = SimpleNamespace(
        forward_backward_input=SimpleNamespace(
            data=[SimpleNamespace(model_dump=lambda: {"model_input": {}, "loss_fn_inputs": {}})],
            loss_fn="cross_entropy",
            loss_fn_config={},
        )
    )

    with pytest.raises(_RayTaskError, match="RayTaskError"):
        await engine.forward_backward(session, request)

    hard_error = engine._hard_poisoned_sessions[session.model_id]
    assert "dense actor retirement failed" in hard_error
    assert "outcome=kill_failed" in hard_error
    with pytest.raises(RuntimeError, match="dense actor retirement failed"):
        engine._raise_if_session_poisoned(session, op="load_weights")
    assert session.model_id not in engine._model_actor_supervisor_actor_names
    assert session.actor_name is None
    assert session.namespace is None
    assert any(
        row["kind"] == "dense_actor_retire_failed" and row["failure_class"] == "kill_failed"
        for row in obs.snapshot()["recent_training_incidents"]
    )


@pytest.mark.anyio
async def test_issue_561_rebind_refuses_poisoned_dense_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = VerlTrainingEngine()
    obs = runtime_obs_module.RuntimeObservability()
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", obs)
    session = TrainingSession(
        model_id="run-561-rebind",
        session_id="session-561-rebind",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.actor_name = "mint_dense_qwen__qwen3_0_6b"
    session.namespace = "mint"

    monkeypatch.setattr(
        dense_trainer,
        "dense_trainer_reuse_block_reason",
        lambda actor_name: "forward_backward:CUDA error: device-side assert triggered",
    )
    monkeypatch.setattr(
        "mint_server.backend.training.verl.verl_training.ray.get_actor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("poisoned actor must not be rebound")),
    )

    worker = await engine._rebind_worker_from_session_metadata(session, reason="issue-561")

    assert worker is None
    assert obs.snapshot()["dense_actor_bind_decision"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "decision": "rebind_refused_poisoned",
            "count": 1,
        }
    ]


def test_issue_561_training_worker_validates_input_contract_before_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    impl_cls = TrainingWorker.__ray_metadata__.modified_class
    worker = object.__new__(impl_cls)

    worker.device = "cuda"
    worker._base_model = "Qwen/Qwen3-0.6B"
    worker._touch = lambda: None
    worker._bind_traceparent = lambda traceparent: None
    worker._ensure_session_loaded = lambda session_id, actual_rank=None: None
    worker.model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=16),
        train=lambda: None,
    )
    worker.tokenizer = SimpleNamespace(vocab_size=16)

    obs = runtime_obs_module.RuntimeObservability()
    events: list[tuple[str, dict[str, object] | None]] = []
    monkeypatch.setattr("mint_server.backend.training.verl.verl_training._get_torch", lambda: SimpleNamespace())
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", obs)
    monkeypatch.setattr(
        "mint_server.backend.training.verl.verl_training.record_span_event_otel",
        lambda name, *, attributes=None: events.append((name, attributes)),
    )

    bad_item = {
        "model_input": {
            "chunks": [
                {"type": "encoded_text", "tokens": [0, 1, 99]},
            ]
        },
        "loss_fn_inputs": {
            "target_tokens": {"data": [0, 1, 2]},
            "weights": {"data": [1.0, 1.0, 1.0]},
        },
    }

    with pytest.raises(ValueError, match="input_ids_out_of_range"):
        worker.forward_backward([bad_item], loss_fn="cross_entropy", session_id="session-561-contract")

    assert events == [
        (
            "mint.training_input_contract_violation",
            {
                "session_id": "session-561-contract",
                "loss_fn": "cross_entropy",
                "reason": "input_ids_out_of_range",
                "vocab_size": 16,
                "input_len": 3,
                "target_len": 3,
                "weights_len": 3,
                "old_logprobs_len": 0,
                "advantages_len": 0,
                "input_min": "0",
                "input_max": "99",
                "target_min": "0",
                "target_max": "2",
                "bad_input_positions": "[2]",
                "bad_target_positions": "[]",
                "bad_weight_positions": "[]",
                "bad_old_logprob_positions": "[]",
                "bad_advantage_positions": "[]",
            },
        )
    ]
    incidents = obs.snapshot()["recent_training_incidents"]
    assert len(incidents) == 1
    incident = incidents[0]
    assert incident["kind"] == "contract_violation"
    assert incident["base_model"] == "Qwen/Qwen3-0.6B"
    assert incident["backend"] == "peft"
    assert incident["op"] == "forward_backward"
    assert incident["failure_class"] == "input_contract"
    assert incident["session_id"] == "session-561-contract"
    assert incident["detail"] == "input_ids_out_of_range"
    assert incident["context"]["bad_input_positions"] == "[2]"


def test_issue_561_rejects_non_finite_weight_inputs_before_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    impl_cls = TrainingWorker.__ray_metadata__.modified_class
    worker = object.__new__(impl_cls)

    worker.device = "cuda"
    worker._base_model = "Qwen/Qwen3-4B-Instruct-2507"
    worker._touch = lambda: None
    worker._bind_traceparent = lambda traceparent: None
    worker._ensure_session_loaded = lambda session_id, actual_rank=None: None
    worker.model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=16),
        train=lambda: None,
    )
    worker.tokenizer = SimpleNamespace(vocab_size=16)

    monkeypatch.setattr("mint_server.backend.training.verl.verl_training._get_torch", lambda: SimpleNamespace())
    monkeypatch.setattr("mint_server.backend.training.verl.verl_training.record_span_event_otel", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", runtime_obs_module.RuntimeObservability())

    bad_item = {
        "model_input": {
            "chunks": [
                {"type": "encoded_text", "tokens": [0, 1, 2]},
            ]
        },
        "loss_fn_inputs": {
            "target_tokens": {"data": [0, 1, 2]},
            "weights": {"data": [1.0, float("nan"), 1.0]},
        },
    }

    with pytest.raises(ValueError, match="weights_non_finite_or_non_numeric"):
        worker.forward_backward([bad_item], loss_fn="cross_entropy", session_id="session-561-nan")


def test_issue_561_classifies_wrapped_input_contract_failure() -> None:
    err = RuntimeError("ray wrapper")
    err.__cause__ = ValueError("dense_input_contract_violation: reason=weights_non_finite_or_non_numeric")

    status, failure_class = VerlTrainingEngine._classify_training_failure(err)

    assert status == "error"
    assert failure_class == "input_contract"
