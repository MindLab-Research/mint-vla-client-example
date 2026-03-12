import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
import time
import sys
import types

import pytest

pytest.importorskip("ray")

import ray

from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.backend.verl_training import TrainingWorker, VerlTrainingEngine


class _RecordingRemoteMethod:
    def __init__(self, ref: str):
        self._ref = ref
        self.calls: list[tuple[tuple, dict]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._ref


class _FakeWorker:
    def __init__(self, ref: str = "fake-save-checkpoint-ref"):
        self.save_checkpoint = _RecordingRemoteMethod(ref)


class _FakeSamplerWorker:
    def __init__(self, ref: str = "fake-save-lora-ref"):
        self.save_lora_weights = _RecordingRemoteMethod(ref)


class _FakeLoadWorker:
    def __init__(self, ref: str = "fake-load-checkpoint-ref"):
        self.load_checkpoint = _RecordingRemoteMethod(ref)


def test_issue_193_training_worker_load_checkpoint_without_optimizer_resets_state(monkeypatch, tmp_path):
    impl_cls = TrainingWorker.__ray_metadata__.modified_class
    worker = object.__new__(impl_cls)

    worker.device = "cpu"
    worker.model = object()
    worker._step_count = 0
    worker._touch = lambda: None

    ensure_calls: list[str] = []
    reset_calls: list[float | None] = []

    worker._ensure_session_loaded = lambda session_id: ensure_calls.append(session_id)
    worker.reset_optimizer = lambda learning_rate=None: reset_calls.append(learning_rate) or {
        "status": "ok",
        "learning_rate": learning_rate,
    }

    class _FakeOptimizer:
        def load_state_dict(self, state):
            raise AssertionError("optimizer state should not load when load_optimizer=False")

    worker.optimizer = _FakeOptimizer()

    ckpt_dir = tmp_path / "worker_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"stub")
    (ckpt_dir / "training_meta.json").write_text('{"current_step": 17, "learning_rate": 0.0005}')

    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda *args, **kwargs: {"fake": "state"}
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors.torch = fake_safetensors_torch
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)

    fake_save_load = types.ModuleType("peft.utils.save_and_load")
    fake_save_load.set_peft_model_state_dict = lambda model, state_dict: None
    fake_peft_utils = types.ModuleType("peft.utils")
    fake_peft_utils.save_and_load = fake_save_load
    fake_peft = types.ModuleType("peft")
    fake_peft.utils = fake_peft_utils
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "peft.utils", fake_peft_utils)
    monkeypatch.setitem(sys.modules, "peft.utils.save_and_load", fake_save_load)

    monkeypatch.setattr(
        "tinker_server.backend.verl_training._get_torch",
        lambda: SimpleNamespace(load=lambda *args, **kwargs: {"unexpected": True}),
    )

    meta = worker.load_checkpoint(
        str(ckpt_dir),
        load_optimizer=False,
        session_id="existing_session",
    )

    assert meta["current_step"] == 17
    assert ensure_calls == ["existing_session"]
    assert reset_calls == [pytest.approx(5e-4)]
    assert worker._step_count == 17


def test_issue_193_megatron_save_weights_passes_explicit_session_id(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193"
    worker = _FakeWorker()
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
        lora_config=SimpleNamespace(
            train_attn=False,
            train_mlp=True,
            train_unembed=False,
        ),
    )

    def fake_ray_get(ref, timeout=None):
        assert ref == "fake-save-checkpoint-ref"
        assert timeout is not None
        return {"current_step": 7}

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_weights(
            session=session,
            save_path="/tmp/issue_193_ckpt",
            use_per_expert_lora=True,
        )

    saved_path = asyncio.run(_run())

    assert saved_path == str(Path("/tmp/issue_193_ckpt").resolve())
    assert session.current_step == 7

    assert len(worker.save_checkpoint.calls) == 1
    args, kwargs = worker.save_checkpoint.calls[0]
    assert args == (str(Path("/tmp/issue_193_ckpt").resolve()),)
    assert kwargs["session_id"] == model_id
    assert kwargs["use_per_expert_lora"] is True
    assert kwargs["train_attn"] is False
    assert kwargs["train_mlp"] is True
    assert kwargs["train_unembed"] is False


@pytest.mark.parametrize(
    "raised_error",
    [
        ray.exceptions.GetTimeoutError("checkpoint save timed out"),
        RuntimeError("checkpoint save failed"),
    ],
    ids=["timeout", "runtime_error"],
)
def test_issue_193_megatron_save_weights_propagates_ray_get_errors_without_step_pollution(
    monkeypatch,
    raised_error,
):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_error"
    worker = _FakeWorker(ref="fake-save-checkpoint-ref-error")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_error",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 123

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert awaitable == "fake-save-checkpoint-ref-error"
        assert keepalive_session is session
        assert timeout_s is not None
        raise raised_error

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        return await engine.save_weights(
            session=session,
            save_path="/tmp/issue_193_ckpt_error",
            use_per_expert_lora=False,
        )

    with pytest.raises(type(raised_error)):
        asyncio.run(_run())

    assert session.current_step == 123
    assert len(worker.save_checkpoint.calls) == 1
    _, kwargs = worker.save_checkpoint.calls[0]
    assert kwargs["session_id"] == model_id


def test_issue_193_megatron_save_weights_concurrent_sessions_are_isolated(monkeypatch):
    engine = VerlTrainingEngine()
    worker_a = _FakeWorker(ref="checkpoint-ref-a")
    worker_b = _FakeWorker(ref="checkpoint-ref-b")
    engine._workers["model_a"] = worker_a
    engine._workers["model_b"] = worker_b

    session_a = TrainingSession(
        model_id="model_a",
        session_id="session_a",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session_b = TrainingSession(
        model_id="model_b",
        session_id="session_b",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )

    def fake_ray_get(ref, timeout=None):
        assert timeout is not None
        if ref == "checkpoint-ref-a":
            time.sleep(0.02)
            return {"current_step": 101}
        if ref == "checkpoint-ref-b":
            time.sleep(0.01)
            return {"current_step": 202}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await asyncio.gather(
            engine.save_weights(session_a, "/tmp/issue_193_ckpt_a"),
            engine.save_weights(session_b, "/tmp/issue_193_ckpt_b"),
        )

    saved_paths = asyncio.run(_run())

    assert saved_paths == [
        str(Path("/tmp/issue_193_ckpt_a").resolve()),
        str(Path("/tmp/issue_193_ckpt_b").resolve()),
    ]
    assert session_a.current_step == 101
    assert session_b.current_step == 202

    args_a, kwargs_a = worker_a.save_checkpoint.calls[0]
    args_b, kwargs_b = worker_b.save_checkpoint.calls[0]
    assert args_a == (str(Path("/tmp/issue_193_ckpt_a").resolve()),)
    assert args_b == (str(Path("/tmp/issue_193_ckpt_b").resolve()),)
    assert kwargs_a["session_id"] == "model_a"
    assert kwargs_b["session_id"] == "model_b"


def test_issue_193_megatron_save_weights_retry_same_session_is_idempotent(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_retry"
    worker = _FakeWorker(ref="checkpoint-ref-retry")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_retry",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 9

    call_count = {"n": 0}

    def fake_ray_get(ref, timeout=None):
        assert ref == "checkpoint-ref-retry"
        assert timeout is not None
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient save failure")
        return {"current_step": 42}

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run_save():
        return await engine.save_weights(
            session=session,
            save_path="/tmp/issue_193_ckpt_retry",
            use_per_expert_lora=True,
        )

    with pytest.raises(RuntimeError, match="transient save failure"):
        asyncio.run(_run_save())
    assert session.current_step == 9

    saved_path = asyncio.run(_run_save())
    assert saved_path == str(Path("/tmp/issue_193_ckpt_retry").resolve())
    assert session.current_step == 42

    assert len(worker.save_checkpoint.calls) == 2
    first_args, first_kwargs = worker.save_checkpoint.calls[0]
    second_args, second_kwargs = worker.save_checkpoint.calls[1]
    assert first_args == second_args == (str(Path("/tmp/issue_193_ckpt_retry").resolve()),)
    assert first_kwargs["session_id"] == model_id
    assert second_kwargs["session_id"] == model_id
    assert first_kwargs["use_per_expert_lora"] is True
    assert second_kwargs["use_per_expert_lora"] is True


def test_issue_193_megatron_save_weights_concurrent_shared_actor_is_isolated(monkeypatch):
    engine = VerlTrainingEngine()

    class _SharedCheckpointRemote:
        def __init__(self):
            self.calls: list[tuple[tuple, dict]] = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return f"shared-ref-{kwargs['session_id']}"

    shared_remote = _SharedCheckpointRemote()
    shared_worker = SimpleNamespace(save_checkpoint=shared_remote)

    # Two model_ids share the same actor handle.
    engine._workers["model_shared_a"] = shared_worker
    engine._workers["model_shared_b"] = shared_worker

    session_a = TrainingSession(
        model_id="model_shared_a",
        session_id="session_shared_a",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session_b = TrainingSession(
        model_id="model_shared_b",
        session_id="session_shared_b",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )

    def fake_ray_get(ref, timeout=None):
        assert timeout is not None
        if ref == "shared-ref-model_shared_a":
            time.sleep(0.02)
            return {"current_step": 31}
        if ref == "shared-ref-model_shared_b":
            time.sleep(0.01)
            return {"current_step": 47}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await asyncio.gather(
            engine.save_weights(session_a, "/tmp/issue_193_ckpt_shared_a"),
            engine.save_weights(session_b, "/tmp/issue_193_ckpt_shared_b"),
        )

    saved_paths = asyncio.run(_run())

    assert saved_paths == [
        str(Path("/tmp/issue_193_ckpt_shared_a").resolve()),
        str(Path("/tmp/issue_193_ckpt_shared_b").resolve()),
    ]
    assert session_a.current_step == 31
    assert session_b.current_step == 47
    assert len(shared_remote.calls) == 2
    session_ids = {kwargs["session_id"] for _, kwargs in shared_remote.calls}
    assert session_ids == {"model_shared_a", "model_shared_b"}


def test_issue_193_dense_save_weights_passes_explicit_session_id_and_keepalive(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_dense_save"
    worker = _FakeWorker(ref="dense-save-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "dense-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_dense_save",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"current_step": 17}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        return await engine.save_weights(session, "/tmp/issue_193_dense_ckpt")

    saved_path = asyncio.run(_run())

    assert saved_path == str(Path("/tmp/issue_193_dense_ckpt").resolve())
    assert session.current_step == 17
    assert keepalive_calls == [("dense-save-ref", model_id, 30.0, 300)]
    args, kwargs = worker.save_checkpoint.calls[0]
    assert args == (str(Path("/tmp/issue_193_dense_ckpt").resolve()),)
    assert kwargs["session_id"] == model_id


def test_issue_193_dense_save_lora_passes_explicit_session_id_and_keepalive(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_dense_lora"
    worker = _FakeSamplerWorker(ref="dense-save-lora-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "dense-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_dense_lora",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return None

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        return await engine.save_dense_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_dense_lora",
        )

    saved_path = asyncio.run(_run())

    assert saved_path == str(Path("/tmp/issue_193_dense_lora").resolve())
    assert keepalive_calls == [("dense-save-lora-ref", model_id, 30.0, 300)]
    args, kwargs = worker.save_lora_weights.calls[0]
    assert args == (str(Path("/tmp/issue_193_dense_lora").resolve()),)
    assert kwargs["session_id"] == model_id


def test_issue_193_dense_load_weights_passes_explicit_session_id_and_keepalive(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_dense_load"
    worker = _FakeLoadWorker(ref="dense-load-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "shared-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_dense_load",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.learning_rate = 1e-4

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"current_step": 23, "learning_rate": 2e-4}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_dense_load",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert keepalive_calls == [("dense-load-ref", model_id, 30.0, 120)]
    args, kwargs = worker.load_checkpoint.calls[0]
    assert args == ("/tmp/issue_193_dense_load", True)
    assert kwargs["session_id"] == model_id
    assert session.current_step == 23
    assert session.learning_rate == pytest.approx(2e-4)


def test_issue_193_megatron_load_weights_passes_explicit_session_id_and_keepalive(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load"
    worker = _FakeLoadWorker(ref="megatron-load-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
        lora_config=SimpleNamespace(
            train_attn=False,
            train_mlp=True,
            train_unembed=False,
        ),
    )

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"current_step": 5, "learning_rate": 3e-4}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load",
            load_optimizer=False,
        )

    asyncio.run(_run())

    assert keepalive_calls == [("megatron-load-ref", model_id, 30.0, 120)]
    args, kwargs = worker.load_checkpoint.calls[0]
    assert args == ("/tmp/issue_193_megatron_load", False)
    assert kwargs["session_id"] == model_id
    assert kwargs["train_attn"] is False
    assert kwargs["train_mlp"] is True
    assert kwargs["train_unembed"] is False
    assert session.current_step == 5
    assert session.learning_rate == pytest.approx(3e-4)


@pytest.mark.parametrize(
    "raised_error",
    [
        ray.exceptions.GetTimeoutError("save_lora_weights timed out"),
        RuntimeError("save_lora_weights failed"),
    ],
    ids=["timeout", "runtime_error"],
)
def test_issue_193_save_lora_weights_for_sampler_propagates_errors_without_step_pollution(
    monkeypatch,
    raised_error,
):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_sampler_error"
    worker = _FakeSamplerWorker(ref="fake-save-lora-ref-error")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_sampler_error",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 88

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert awaitable == "fake-save-lora-ref-error"
        assert keepalive_session is session
        assert timeout_s is not None
        raise raised_error

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_lora_error",
            use_per_expert_lora=True,
        )

    with pytest.raises(type(raised_error)):
        asyncio.run(_run())

    assert session.current_step == 88
    assert len(worker.save_lora_weights.calls) == 1
    _, kwargs = worker.save_lora_weights.calls[0]
    assert kwargs["session_id"] == model_id


def test_issue_193_save_lora_weights_for_sampler_retry_same_session_is_idempotent(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_sampler_retry"
    worker = _FakeSamplerWorker(ref="fake-save-lora-ref-retry")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_sampler_retry",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 15

    call_count = {"n": 0}

    def fake_ray_get(ref, timeout=None):
        assert ref == "fake-save-lora-ref-retry"
        assert timeout is not None
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient save_lora failure")
        return {"current_step": 66}

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run_save():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_lora_retry",
            use_per_expert_lora=False,
        )

    with pytest.raises(RuntimeError, match="transient save_lora failure"):
        asyncio.run(_run_save())
    assert session.current_step == 15

    saved_path = asyncio.run(_run_save())
    assert saved_path == str(Path("/tmp/issue_193_lora_retry").resolve())
    assert session.current_step == 66

    assert len(worker.save_lora_weights.calls) == 2
    first_args, first_kwargs = worker.save_lora_weights.calls[0]
    second_args, second_kwargs = worker.save_lora_weights.calls[1]
    assert first_args == second_args == (str(Path("/tmp/issue_193_lora_retry").resolve()),)
    assert first_kwargs["session_id"] == model_id
    assert second_kwargs["session_id"] == model_id


def test_issue_193_same_model_concurrent_save_weights_out_of_order_steps_do_not_regress(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_same_model_save"

    class _SequenceCheckpointRemote:
        def __init__(self):
            self.calls: list[tuple[tuple, dict]] = []
            self._count = 0

        def remote(self, *args, **kwargs):
            self._count += 1
            self.calls.append((args, kwargs))
            return f"same-model-save-ref-{self._count}"

    remote = _SequenceCheckpointRemote()
    engine._workers[model_id] = SimpleNamespace(save_checkpoint=remote)

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_same_model_save",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 0

    def fake_ray_get(ref, timeout=None):
        assert timeout is not None
        if ref == "same-model-save-ref-1":
            time.sleep(0.01)
            return {"current_step": 101}
        if ref == "same-model-save-ref-2":
            time.sleep(0.02)
            return {"current_step": 99}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await asyncio.gather(
            engine.save_weights(session, "/tmp/issue_193_same_model_ckpt_a"),
            engine.save_weights(session, "/tmp/issue_193_same_model_ckpt_b"),
        )

    _ = asyncio.run(_run())
    assert session.current_step == 101


def test_issue_193_same_model_concurrent_save_lora_out_of_order_steps_do_not_regress(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_same_model_lora"

    class _SequenceLoraRemote:
        def __init__(self):
            self.calls: list[tuple[tuple, dict]] = []
            self._count = 0

        def remote(self, *args, **kwargs):
            self._count += 1
            self.calls.append((args, kwargs))
            return f"same-model-lora-ref-{self._count}"

    remote = _SequenceLoraRemote()
    engine._workers[model_id] = SimpleNamespace(save_lora_weights=remote)

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_same_model_lora",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 0

    def fake_ray_get(ref, timeout=None):
        assert timeout is not None
        if ref == "same-model-lora-ref-1":
            time.sleep(0.01)
            return {"current_step": 101}
        if ref == "same-model-lora-ref-2":
            time.sleep(0.02)
            return {"current_step": 99}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await asyncio.gather(
            engine.save_lora_weights_for_sampler(session, "/tmp/issue_193_same_model_lora_a"),
            engine.save_lora_weights_for_sampler(session, "/tmp/issue_193_same_model_lora_b"),
        )

    _ = asyncio.run(_run())
    assert session.current_step == 101


@pytest.mark.parametrize(
    "meta_payload",
    [
        {},
        {"current_step": "42"},
    ],
    ids=["missing_current_step", "current_step_wrong_type"],
)
def test_issue_193_save_weights_invalid_meta_fails_in_strict_megatron_mode(monkeypatch, meta_payload):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_save"
    worker = _FakeWorker(ref="invalid-meta-save-ref")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_save",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-save-ref"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_weights(session, "/tmp/issue_193_invalid_meta_save")

    with pytest.raises(ValueError, match="save_weights"):
        asyncio.run(_run())

    assert session.current_step == 77


@pytest.mark.parametrize(
    "meta_payload",
    [
        {},
        {"current_step": "42"},
    ],
    ids=["missing_current_step", "current_step_wrong_type"],
)
def test_issue_193_save_lora_invalid_meta_fails_in_strict_megatron_mode(monkeypatch, meta_payload):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_lora"
    worker = _FakeSamplerWorker(ref="invalid-meta-lora-ref")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_lora",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-lora-ref"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_invalid_meta_lora",
        )

    with pytest.raises(ValueError, match="save_lora_weights_for_sampler"):
        asyncio.run(_run())

    assert session.current_step == 77


@pytest.mark.parametrize(
    "meta_payload",
    [
        {},
        {"current_step": "42"},
    ],
    ids=["missing_current_step", "current_step_wrong_type"],
)
def test_issue_193_save_weights_invalid_meta_non_strict_mode_warns_without_pollution(
    monkeypatch,
    caplog,
    meta_payload,
):
    monkeypatch.setenv("MINT_MEGATRON_STRICT_SAVE_META", "0")

    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_save_non_strict"
    worker = _FakeWorker(ref="invalid-meta-save-ref-non-strict")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_save_non_strict",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-save-ref-non-strict"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_weights(session, "/tmp/issue_193_invalid_meta_save_non_strict")

    with caplog.at_level(logging.WARNING):
        _ = asyncio.run(_run())

    assert session.current_step == 77
    assert any("save_weights" in rec.getMessage() for rec in caplog.records)


@pytest.mark.parametrize(
    "meta_payload",
    [
        {},
        {"current_step": "42"},
    ],
    ids=["missing_current_step", "current_step_wrong_type"],
)
def test_issue_193_save_lora_invalid_meta_non_strict_mode_warns_without_pollution(
    monkeypatch,
    caplog,
    meta_payload,
):
    monkeypatch.setenv("MINT_MEGATRON_STRICT_SAVE_META", "0")

    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_lora_non_strict"
    worker = _FakeSamplerWorker(ref="invalid-meta-lora-ref-non-strict")
    engine._workers[model_id] = worker

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_lora_non_strict",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="megatron",
    )
    session.current_step = 77

    def fake_ray_get(ref, timeout=None):
        assert ref == "invalid-meta-lora-ref-non-strict"
        assert timeout is not None
        return meta_payload

    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        return await engine.save_lora_weights_for_sampler(
            session=session,
            save_path="/tmp/issue_193_invalid_meta_lora_non_strict",
        )

    with caplog.at_level(logging.WARNING):
        _ = asyncio.run(_run())

    assert session.current_step == 77
    assert any("save_lora_weights_for_sampler" in rec.getMessage() for rec in caplog.records)
