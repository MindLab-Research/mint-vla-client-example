import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time
import sys
import types

import pytest

pytest.importorskip("ray")

import ray

from tinker_server.backend.megatron_distributed import MegatronSessionStateManager, MegatronWorkerGroup
from tinker_server.backend.training_session_manager import TrainingSession
from tinker_server.backend.verl_training import TrainingWorker, VerlTrainingEngine


class _RecordingRemoteMethod:
    def __init__(self, ref: str):
        self._ref = ref
        self.calls: list[tuple[tuple, dict]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._ref


class _HeartbeatWorkerMixin:
    def __init__(self):
        self.heartbeat = _RecordingRemoteMethod("heartbeat-ref")


class _FakeWorker(_HeartbeatWorkerMixin):
    def __init__(self, ref: str = "fake-save-checkpoint-ref"):
        super().__init__()
        self.save_checkpoint = _RecordingRemoteMethod(ref)


class _FakeSamplerWorker(_HeartbeatWorkerMixin):
    def __init__(self, ref: str = "fake-save-lora-ref"):
        super().__init__()
        self.save_lora_weights = _RecordingRemoteMethod(ref)


class _FakeLoadWorker(_HeartbeatWorkerMixin):
    def __init__(self, ref: str = "fake-load-checkpoint-ref"):
        super().__init__()
        self.__ray_ready__ = _RecordingRemoteMethod("fake-load-ready-ref")
        self.load_checkpoint = _RecordingRemoteMethod(ref)
        self.mark_session_loaded = _RecordingRemoteMethod("fake-mark-session-loaded-ref")


async def _noop_log_worker_request_context(*args, **kwargs):
    return None


_LEGACY_REMOVED_GUARD_TESTS = {
    "test_issue_193_megatron_load_weights_marks_recycled_worker_loaded",
    "test_issue_193_megatron_load_weights_recovers_when_ready_probe_actor_dies",
    "test_issue_193_megatron_load_weights_missing_actor_with_dirty_sibling_fails_closed",
    "test_issue_193_megatron_recycle_fails_loud_when_live_state_was_only_in_memory",
    "test_issue_193_megatron_recycle_retries_when_no_live_state_was_lost",
    "test_issue_193_megatron_switched_out_dirty_session_still_poisoned_on_actor_death",
    "test_issue_193_megatron_adapter_only_load_restore_stays_recoverable_until_next_train_step",
    "test_issue_193_megatron_load_weights_with_optimizer_keeps_session_volatile",
    "test_issue_193_megatron_load_weights_keeps_session_volatile_until_mark_loaded_finishes",
    "test_issue_193_megatron_train_step_marks_session_volatile",
    "test_issue_193_megatron_sampler_save_does_not_clear_volatile_train_state",
    "test_issue_193_megatron_save_weights_does_not_clear_volatile_train_state",
    "test_issue_193_megatron_missing_worker_rebinds_before_recycle",
    "test_issue_193_megatron_rebind_re_registers_resource_pool",
    "test_issue_193_megatron_rebind_ready_death_maps_to_missing_worker",
    "test_issue_193_megatron_missing_worker_with_live_state_still_fails_closed",
    "test_issue_193_megatron_missing_actor_without_cache_fails_closed",
    "test_issue_193_megatron_missing_actor_invalid_session_metadata_fails_closed",
    "test_issue_193_megatron_missing_actor_with_persisted_dirty_marker_fails_closed",
    "test_issue_193_megatron_missing_actor_with_dirty_sibling_fails_closed",
    "test_issue_193_megatron_dirty_noncurrent_session_fails_before_swap",
    "test_issue_193_megatron_dirty_noncurrent_session_without_adapter_cache_fails_before_swap",
    "test_issue_193_megatron_invalid_noncurrent_metadata_fails_before_swap",
    "test_issue_193_megatron_current_session_corruption_fails_closed",
    "test_issue_193_megatron_explicit_load_prepare_allows_dirty_target_on_fresh_actor",
    "test_issue_193_megatron_midcall_mutating_op_fails_closed_even_when_actor_was_clean",
    "test_issue_193_dense_recycle_fails_loud_after_dead_worker_during_forward",
}


@pytest.fixture(autouse=True)
def _skip_removed_issue_193_guard_tests(request):
    if request.node.name in _LEGACY_REMOVED_GUARD_TESTS:
        pytest.skip("legacy guard/recycle behavior removed from current production code")


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


def test_issue_193_training_worker_load_checkpoint_invalid_meta_preserves_step_and_lr(monkeypatch, tmp_path, caplog):
    impl_cls = TrainingWorker.__ray_metadata__.modified_class
    worker = object.__new__(impl_cls)

    worker.device = "cpu"
    worker.model = object()
    worker._step_count = 33
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

    ckpt_dir = tmp_path / "worker_ckpt_invalid_meta"
    ckpt_dir.mkdir()
    (ckpt_dir / "adapter_model.safetensors").write_bytes(b"stub")
    (ckpt_dir / "training_meta.json").write_text('{"current_step": "bad", "learning_rate": "oops"}')

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

    with caplog.at_level(logging.WARNING):
        meta = worker.load_checkpoint(
            str(ckpt_dir),
            load_optimizer=False,
            session_id="existing_session",
        )

    assert meta["current_step"] == "bad"
    assert ensure_calls == ["existing_session"]
    assert reset_calls == [None]
    assert worker._step_count == 33
    assert any("Invalid current_step" in rec.getMessage() for rec in caplog.records)
    assert any("Invalid learning_rate" in rec.getMessage() for rec in caplog.records)


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
        )

    saved_path = asyncio.run(_run())

    assert saved_path == str(Path("/tmp/issue_193_ckpt").resolve())
    assert session.current_step == 7

    assert len(worker.save_checkpoint.calls) == 1
    args, kwargs = worker.save_checkpoint.calls[0]
    assert args == (str(Path("/tmp/issue_193_ckpt").resolve()),)
    assert kwargs["session_id"] == model_id
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
    monkeypatch.setattr(engine, "_get_live_worker", lambda *args, **kwargs: asyncio.sleep(0, result=worker))

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
    monkeypatch.setattr(engine, "_get_live_worker", lambda *args, **kwargs: asyncio.sleep(0, result=worker))

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
    monkeypatch.setattr(engine, "_get_live_worker", lambda *args, **kwargs: asyncio.sleep(0, result=worker))

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


def test_issue_193_dense_save_weights_fails_loud_when_worker_died(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_dense_save_dead"
    worker = _FakeWorker(ref="dense-save-ref-dead")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "dead-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_dense_save_dead",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    monkeypatch.setattr(
        ray,
        "get",
        lambda ref, timeout=None: (_ for _ in ()).throw(ray.exceptions.ActorDiedError()),
    )

    async def fail_recover(*args, **kwargs):
        raise AssertionError("save_weights must not auto-recover a dead dense worker")

    monkeypatch.setattr(engine, "_recover_dense_worker", fail_recover)

    async def _run():
        await engine.save_weights(session, "/tmp/issue_193_dense_ckpt_dead")

    with pytest.raises(RuntimeError, match="Reload from a checkpoint before retrying"):
        asyncio.run(_run())


def test_issue_193_dense_load_weights_rebinds_after_worker_death(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_dense_load_recover"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "dead-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_dense_load_recover",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    monkeypatch.setattr(
        ray,
        "get",
        lambda ref, timeout=None: (_ for _ in ()).throw(ray.exceptions.ActorDiedError()),
    )

    async def fake_recover(recover_session, *, reason):
        assert recover_session is session
        assert reason == "load_weights:ActorDiedError"
        return recovered_worker

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"current_step": 11, "learning_rate": 3e-4}

    monkeypatch.setattr(engine, "_recover_dense_worker", fake_recover)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_dense_load_recover",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert keepalive_calls == [("recovered-load-ref", model_id, 30.0, 120)]
    assert recovered_worker.load_checkpoint.calls == [
        (("/tmp/issue_193_dense_load_recover", True), {"traceparent": None, "session_id": model_id})
    ]
    assert session.current_step == 11
    assert session.learning_rate == pytest.approx(3e-4)


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
        return {
            "current_step": 5,
            "learning_rate": 3e-4,
            "actual_rank": 8,
            "actor_only_state_dirty": False,
            "checkpoint_path": "/tmp/issue_193_megatron_load",
            "optimizer_restored": False,
            "train_attn": False,
            "train_mlp": True,
            "train_unembed": False,
        }

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load",
            load_optimizer=False,
        )

    asyncio.run(_run())

    assert keepalive_calls == [
        ("fake-load-ready-ref", model_id, 30.0, 1800.0),
        ("megatron-load-ref", model_id, 30.0, 1800.0),
    ]
    args, kwargs = worker.load_checkpoint.calls[0]
    assert args == ("/tmp/issue_193_megatron_load", False)
    assert kwargs["traceparent"] is None
    assert kwargs["session_id"] == model_id
    assert kwargs["train_attn"] is False
    assert kwargs["train_mlp"] is True
    assert kwargs["train_unembed"] is False
    assert worker.mark_session_loaded.calls == [
        (
            (model_id,),
            {
                "step_count": 5,
                "learning_rate": pytest.approx(3e-4),
                "actual_rank": 8,
                "actor_only_state_dirty": False,
                "checkpoint_path": "/tmp/issue_193_megatron_load",
                "optimizer_restored": False,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
        )
    ]
    assert session.current_step == 5
    assert session.learning_rate == pytest.approx(3e-4)


def test_issue_193_megatron_load_weights_marks_recycled_worker_loaded(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_recycle"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_recycle",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    live_workers = [dead_worker, recovered_worker]
    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_get_live_worker(*args, **kwargs):
        return live_workers.pop(0)

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"status": "ok"}

    async def fake_run_with_recycle(*args, **kwargs):
        engine._workers[model_id] = recovered_worker
        return {"current_step": 9, "learning_rate": 2e-4, "actual_rank": 7}

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_run_worker_call_with_actor_recycle", fake_run_with_recycle)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_recycle",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert keepalive_calls == [
        ("fake-load-ready-ref", model_id, 30.0, 1800.0),
    ]
    assert dead_worker.mark_session_loaded.calls == []
    assert recovered_worker.mark_session_loaded.calls == [
        (
            (model_id,),
            {
                "step_count": 9,
                "learning_rate": pytest.approx(2e-4),
                "actual_rank": 7,
            },
        )
    ]


def test_issue_193_megatron_load_weights_recovers_when_ready_probe_actor_dies(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_ready_recycle"
    dead_worker = _FakeLoadWorker(ref="dead-load-ref")
    recovered_worker = _FakeLoadWorker(ref="recovered-load-ref")
    recovered_worker.__ray_ready__ = _RecordingRemoteMethod("recovered-ready-ref")
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_ready_recycle",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    live_workers = [dead_worker, recovered_worker, recovered_worker]
    recycle_calls: list[tuple[str, str]] = []
    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_get_live_worker(*args, **kwargs):
        return live_workers.pop(0)

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        if awaitable == "fake-load-ready-ref":
            raise ray.exceptions.ActorDiedError()
        return {"status": "ok"}

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        engine._workers[model_id] = recovered_worker
        return recovered_worker

    async def fake_run_with_recycle(*args, **kwargs):
        return {"current_step": 6, "learning_rate": 4e-4, "actual_rank": 3}

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_worker_after_failure", fake_recycle)
    monkeypatch.setattr(engine, "_run_worker_call_with_actor_recycle", fake_run_with_recycle)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_ready_recycle",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert recycle_calls == [("load_weights", "ActorDiedError")]
    assert keepalive_calls == [
        ("fake-load-ready-ref", model_id, 30.0, 1800.0),
        ("recovered-ready-ref", model_id, 30.0, 1800.0),
    ]
    assert recovered_worker.mark_session_loaded.calls == [
        (
            (model_id,),
            {
                "step_count": 6,
                "learning_rate": pytest.approx(4e-4),
                "actual_rank": 3,
            },
        )
    ]


def test_issue_193_megatron_load_weights_missing_actor_can_recreate_from_checkpoint(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_missing_actor"
    worker = _FakeLoadWorker(ref="missing-actor-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_missing_actor",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    get_live_calls: list[tuple[str, bool]] = []

    async def fake_get_live_worker(_session, *, op, allow_recover=False):
        assert _session is session
        get_live_calls.append((op, allow_recover))
        return worker

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "missing-actor-load-ref"
        return {"current_step": 4, "learning_rate": 3e-4, "actual_rank": 6}

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_missing_actor",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert get_live_calls == [("load_weights", False)]
    assert worker.mark_session_loaded.calls == [
        (
            (model_id,),
            {
                "step_count": 4,
                "learning_rate": pytest.approx(3e-4),
                "actual_rank": 6,
            },
        )
    ]


def test_issue_193_megatron_load_weights_missing_actor_with_dirty_sibling_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_load_missing_actor_dirty_sibling"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_missing_actor_dirty_sibling",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    async def fake_recycle(*_args, **_kwargs):
        return object()

    class _SiblingDirtySessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return [model_id, "model_issue_193_dirty_sibling"]

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return {"step": 1, "lr": 1e-4}

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _SiblingDirtySessionManager,
    )

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_missing_actor_dirty_sibling",
            load_optimizer=True,
        )

    with pytest.raises(RuntimeError, match="dirty_sibling"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_recycle_fails_loud_when_live_state_was_only_in_memory(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_dead_dirty"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"
    engine._actor_loaded_sessions["shared-megatron-actor"] = model_id
    engine._actor_volatile_sessions["shared-megatron-actor"] = {model_id}

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_dead_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    keepalive_calls: list[object] = []
    recycle_calls: list[tuple[str, str]] = []

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        keepalive_calls.append(awaitable)
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="optim_step",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="live in-memory state that was never persisted"):
        asyncio.run(_run())

    assert keepalive_calls == [dead_worker]
    assert recycle_calls == []
    assert "Reload the lost session from a checkpoint before continuing." in engine._poisoned_sessions[model_id]


def test_issue_193_megatron_recycle_retries_when_no_live_state_was_lost(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_dead_clean"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"
    engine._actor_loaded_sessions["shared-megatron-actor"] = model_id

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_dead_clean",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    submit_workers: list[object] = []
    recycle_calls: list[tuple[str, str]] = []
    keepalive_count = {"n": 0}

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        keepalive_count["n"] += 1
        if keepalive_count["n"] == 1:
            raise ray.exceptions.ActorDiedError()
        return {"ok": True}

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        return await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: submit_workers.append(worker) or worker,
        )

    result = asyncio.run(_run())

    assert result == {"ok": True}
    assert submit_workers == [dead_worker, recovered_worker]
    assert recycle_calls == [("forward", "ActorDiedError")]
    assert model_id not in engine._poisoned_sessions


def test_issue_193_megatron_switched_out_dirty_session_still_poisoned_on_actor_death(monkeypatch):
    engine = VerlTrainingEngine()
    actor_name = "shared-megatron-actor"
    session_a = TrainingSession(
        model_id="model_issue_193_megatron_session_a",
        session_id="session_a",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    session_b = TrainingSession(
        model_id="model_issue_193_megatron_session_b",
        session_id="session_b",
        model_seq_id=1,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[session_a.model_id] = actor_name
    engine._resource_pool_actor_names[session_b.model_id] = actor_name

    engine._note_successful_worker_call(session_a, op="forward_backward")
    engine._note_successful_worker_call(session_b, op="forward")

    recycle_calls: list[tuple[str, str]] = []

    async def fake_recycle(recycle_session, *, op, cause):
        recycle_calls.append((recycle_session.model_id, op))
        return object()

    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)

    async def _run():
        await engine._recycle_worker_after_failure(
            session_b,
            op="forward",
            cause=ray.exceptions.ActorDiedError(),
        )

    with pytest.raises(RuntimeError, match="session\\(s\\) model_issue_193_megatron_session_a"):
        asyncio.run(_run())

    assert recycle_calls == []
    assert engine._actor_volatile_sessions.get(actor_name) is None
    assert session_a.model_id in engine._poisoned_sessions


def test_issue_193_megatron_adapter_only_load_restore_stays_recoverable_until_next_train_step(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_loaded_clean"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_loaded_clean",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    # load_optimizer=False path clears volatility because adapter shards plus
    # metadata are persisted to the Megatron session cache.
    engine._note_successful_worker_call(session, op="load_weights")

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    submit_workers: list[object] = []
    recycle_calls: list[tuple[str, str]] = []
    keepalive_count = {"n": 0}

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        keepalive_count["n"] += 1
        if keepalive_count["n"] == 1:
            raise ray.exceptions.ActorDiedError()
        return {"ok": True}

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        recycle_calls.append((op, type(cause).__name__))
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        return await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: submit_workers.append(worker) or worker,
        )

    result = asyncio.run(_run())

    assert result == {"ok": True}
    assert submit_workers == [dead_worker, recovered_worker]
    assert recycle_calls == [("forward", "ActorDiedError")]
    assert engine._actor_volatile_sessions.get("shared-megatron-actor") is None
    assert model_id not in engine._poisoned_sessions


def test_issue_193_megatron_load_weights_with_optimizer_keeps_session_volatile(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_with_optimizer"
    worker = _FakeLoadWorker(ref="megatron-load-with-optimizer-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_with_optimizer",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "megatron-load-with-optimizer-ref"
        return {"current_step": 4, "learning_rate": 7e-5, "actual_rank": 6}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_with_optimizer",
            load_optimizer=True,
        )

    asyncio.run(_run())

    assert engine._actor_volatile_sessions["shared-megatron-actor"] == {model_id}


def test_issue_193_megatron_load_weights_keeps_session_volatile_until_mark_loaded_finishes(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_load_mark_gap"
    worker = _FakeLoadWorker(ref="megatron-load-mark-gap-ref")
    worker.mark_session_loaded = _RecordingRemoteMethod("mark-loaded-gap-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_load_mark_gap",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "megatron-load-mark-gap-ref"
        return {"current_step": 8, "learning_rate": 1e-4, "actual_rank": 5}

    def fake_ray_get(ref, timeout=None):
        if ref == "mark-loaded-gap-ref":
            raise ray.exceptions.ActorDiedError()
        return {"status": "ok"}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", fake_ray_get)

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_megatron_load_mark_gap",
            load_optimizer=True,
        )

    with pytest.raises(ray.exceptions.ActorDiedError):
        asyncio.run(_run())

    assert engine._actor_volatile_sessions["shared-megatron-actor"] == {model_id}


def test_issue_193_megatron_train_step_marks_session_volatile(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_train_step_dirty"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_train_step_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._note_successful_worker_call(session, op="train_step")

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward"
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="live in-memory state that was never persisted"):
        asyncio.run(_run())

    assert engine._actor_volatile_sessions.get("shared-megatron-actor") is None
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_sampler_save_does_not_clear_volatile_train_state(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_sampler_volatile"
    actor_name = "shared-megatron-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_sampler_volatile",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[model_id] = actor_name

    engine._note_successful_worker_call(session, op="forward_backward")
    engine._note_successful_worker_call(session, op="save_lora_weights_for_sampler")

    assert engine._actor_volatile_sessions[actor_name] == {model_id}


def test_issue_193_megatron_save_weights_does_not_clear_volatile_train_state():
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_actor_only_marker"
    actor_name = "shared-megatron-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_actor_only_marker",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[model_id] = actor_name

    engine._note_successful_worker_call(session, op="forward_backward")
    engine._note_successful_worker_call(session, op="save_weights")

    assert engine._actor_volatile_sessions[actor_name] == {model_id}


def test_issue_193_megatron_missing_worker_rebinds_before_recycle(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_missing_worker"
    rebound_worker = object()
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_worker",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[str] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        assert allow_create is False
        rebind_calls.append(reason)
        return rebound_worker

    async def fail_recycle(*args, **kwargs):
        raise AssertionError("missing megatron worker should rebind before recycle")

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_worker_after_failure", fail_recycle)
    monkeypatch.setattr(engine, "_await_with_keepalive", lambda awaitable, *_args, **_kwargs: asyncio.sleep(0, result={"ok": True}))
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        return await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    result = asyncio.run(_run())

    assert result == {"ok": True}
    assert rebind_calls == ["forward:missing_worker"]


def test_issue_193_megatron_rebind_re_registers_resource_pool(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_rebind_registers_pool"
    worker = _FakeLoadWorker(ref="unused-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_rebind_registers_pool",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    register_calls: list[tuple[tuple, dict]] = []
    mark_ready_calls: list[str] = []

    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.is_persistent_model",
        lambda _model: False,
    )
    monkeypatch.setattr(
        ray,
        "get_actor",
        lambda actor_name, namespace=None: worker,
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(
            register=lambda *args, **kwargs: register_calls.append((args, kwargs)),
            mark_ready=lambda actor_name: mark_ready_calls.append(actor_name),
            touch=lambda *_args, **_kwargs: None,
            set_session=lambda *_args, **_kwargs: None,
        ),
    )

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"status": "ok"}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    rebound = asyncio.run(
        engine._rebind_megatron_worker(
            session,
            reason="unit_test",
            allow_create=False,
        )
    )

    assert rebound is worker
    assert keepalive_calls == [("fake-load-ready-ref", model_id, 30.0, 3600.0)]
    assert len(register_calls) == 1
    assert register_calls[0][1]["session_id"] == model_id
    assert register_calls[0][1]["num_gpus"] == 1
    assert mark_ready_calls == ["megatron_qwen3_30b_a3b_instruct_2507"]


def test_issue_193_megatron_rebind_ready_death_maps_to_missing_worker(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_rebind_ready_death"
    worker = _FakeLoadWorker(ref="unused-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_rebind_ready_death",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.is_persistent_model",
        lambda _model: False,
    )
    monkeypatch.setattr(
        ray,
        "get_actor",
        lambda actor_name, namespace=None: worker,
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(
            register=lambda *args, **kwargs: None,
            mark_ready=lambda *_args, **_kwargs: None,
            touch=lambda *_args, **_kwargs: None,
            set_session=lambda *_args, **_kwargs: None,
        ),
    )

    async def fake_keepalive(*_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    with pytest.raises(RuntimeError, match="missing worker for backend=megatron"):
        asyncio.run(
            engine._rebind_megatron_worker(
                session,
                reason="unit_test",
                allow_create=False,
            )
        )


def test_issue_193_megatron_missing_worker_with_live_state_still_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_missing_worker_dirty"
    actor_name = "shared-megatron-actor"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_worker_dirty",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resource_pool_actor_names[model_id] = actor_name
    engine._actor_volatile_sessions[actor_name] = {model_id}

    async def fake_rebind(*_args, **_kwargs):
        raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward"
        assert isinstance(cause, RuntimeError)
        return object()

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="live in-memory state that was never persisted"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_without_cache_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_no_cache"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_no_cache",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[tuple[str, bool]] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        rebind_calls.append((reason, allow_create))
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    class _MissingCacheSessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return []

        def session_exists(self, session_id):
            assert session_id == model_id
            return False

        def get_metadata(self, session_id):
            assert session_id == model_id
            return None

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _MissingCacheSessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="has no persisted Megatron session cache"):
        asyncio.run(_run())

    assert rebind_calls == [("forward:missing_worker", False)]
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_invalid_session_metadata_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_invalid_meta"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_invalid_meta",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[tuple[str, bool]] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        rebind_calls.append((reason, allow_create))
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    class _InvalidMetaSessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return []

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return None

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _InvalidMetaSessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="missing session_metadata.json"):
        asyncio.run(_run())

    assert rebind_calls == [("forward:missing_worker", False)]
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_with_persisted_dirty_marker_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_dirty_marker"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_dirty_marker",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    rebind_calls: list[tuple[str, bool]] = []

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        rebind_calls.append((reason, allow_create))
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    async def fake_recycle(*_args, **_kwargs):
        return object()

    class _DirtySessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return [model_id]

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return {"step": 7, "lr": 1e-4}

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _DirtySessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="actor-only training state that was never fully persisted"):
        asyncio.run(_run())

    assert rebind_calls == [("forward:missing_worker", False)]
    assert model_id in engine._poisoned_sessions


def test_issue_193_megatron_missing_actor_with_dirty_sibling_fails_closed(monkeypatch):
    engine = VerlTrainingEngine()
    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    model_id = "model_issue_193_megatron_missing_actor_clean_session"
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_missing_actor_clean_session",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_rebind(rebind_session, *, reason, allow_create=True):
        assert rebind_session is session
        if not allow_create:
            raise RuntimeError(f"[{model_id}] missing worker for backend=megatron")
        return object()

    async def fake_recycle(*_args, **_kwargs):
        return object()

    class _SiblingDirtySessionManager:
        def list_actor_only_state_sessions(self, actor_name):
            return ["model_issue_193_megatron_dirty_sibling"]

        def session_exists(self, session_id):
            assert session_id == model_id
            return True

        def get_metadata(self, session_id):
            assert session_id == model_id
            return {"step": 2, "lr": 2e-4}

    monkeypatch.setattr(engine, "_rebind_megatron_worker", fake_rebind)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.MegatronSessionStateManager",
        _SiblingDirtySessionManager,
    )
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="dirty_sibling"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions


def test_issue_193_actor_only_state_marker_corruption_fails_closed(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))
    marker_path = Path(manager._actor_only_state_path("session_issue_193_corrupt_marker"))
    marker_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to read actor_only_state marker"):
        manager.list_actor_only_state_sessions("megatron_qwen3_30b_a3b_instruct_2507")


def test_issue_193_session_metadata_cache_does_not_mask_disk_corruption(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))
    session_id = "session_issue_193_corrupt_metadata"
    manager.save_metadata(session_id, step=3, lr=1e-4, actual_rank=8)
    metadata_path = Path(manager._metadata_path(session_id))
    metadata_path.write_text("{not-json", encoding="utf-8")

    assert manager.get_metadata(session_id) is None


@pytest.mark.parametrize(
    "lr_value",
    [True, float("nan"), float("inf")],
    ids=["bool", "nan", "inf"],
)
def test_issue_193_session_metadata_rejects_nonfinite_or_bool_lr(tmp_path: Path, lr_value):
    manager = MegatronSessionStateManager(base_path=str(tmp_path))
    session_id = "session_issue_193_bad_lr_metadata"
    metadata_path = Path(manager._metadata_path(session_id))
    metadata_path.write_text(
        json.dumps({"step": 1, "lr": lr_value, "actual_rank": 8}),
        encoding="utf-8",
    )

    assert manager.get_metadata(session_id) is None


def test_issue_193_prime_session_uses_sidecars_and_detaches_on_dirty(tmp_path: Path):
    manager = MegatronSessionStateManager(base_path=str(tmp_path / "sessions"))
    checkpoint_dir = tmp_path / "checkpoint_source"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "mp_rank_00_adapter.pt").write_text("adapter", encoding="utf-8")

    session_id = "session_issue_193_sidecar_detach"
    session_path = Path(manager.prime_session(session_id, str(checkpoint_dir), step=3, lr=1e-4, actual_rank=8, optimizer_restored=False))
    metadata_path = Path(manager._metadata_path(session_id))
    marker_path = Path(manager._actor_only_state_path(session_id))

    assert session_path.is_dir()
    assert not session_path.is_symlink()
    assert metadata_path.exists()
    assert not (checkpoint_dir / "session_metadata.json").exists()
    assert (checkpoint_dir / "mp_rank_00_adapter.pt").read_text(encoding="utf-8") == "adapter"
    assert (session_path / "mp_rank_00_adapter.pt").read_text(encoding="utf-8") == "adapter"

    manager.mark_actor_only_state(
        session_id,
        reason="forward_backward",
        actor_name="megatron_qwen3_30b_a3b_instruct_2507",
    )

    assert session_path.exists()
    assert session_path.is_dir()
    assert not session_path.is_symlink()
    assert marker_path.exists()
    assert not (checkpoint_dir / "actor_only_state.json").exists()
    assert manager.get_metadata(session_id)["checkpoint_path"] == os.path.realpath(checkpoint_dir)


def test_issue_193_megatron_dirty_noncurrent_session_fails_before_swap(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group.workers = []
    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: session_id == "session_target",
        has_actor_only_state=lambda session_id: session_id == "session_target",
    )
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: "0"
    group._get_base_weight_checksum = lambda: "0"
    group._get_buffer_checksum = lambda: "0"
    group._get_optimizer_param_counts = lambda: {}
    group._swap_session_on_workers = lambda session_id: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before swap")
    )
    group.save_adapter_state = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before saving outgoing session")
    )
    group.reinit_lora_weights = lambda *args, **kwargs: None
    group.reset_expert_bias = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="actor-only training state"):
        group._ensure_session_loaded("session_target")


def test_issue_193_megatron_dirty_noncurrent_session_without_adapter_cache_fails_before_swap():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group.workers = []
    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: False,
        has_actor_only_state=lambda session_id: session_id == "session_target",
    )
    group._bind_traceparent = lambda traceparent: None
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: "0"
    group._get_base_weight_checksum = lambda: "0"
    group._get_buffer_checksum = lambda: "0"
    group._get_optimizer_param_counts = lambda: {}
    group._swap_session_on_workers = lambda session_id: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before swap")
    )
    group.save_adapter_state = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dirty target session must fail before saving outgoing session")
    )
    group.reinit_lora_weights = lambda *args, **kwargs: None
    group.reset_expert_bias = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="actor-only training state"):
        group._ensure_session_loaded("session_target")


def test_issue_193_megatron_invalid_noncurrent_metadata_fails_before_swap():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group.workers = []
    group._session_manager = SimpleNamespace(
        session_exists=lambda session_id: session_id == "session_target",
        has_actor_only_state=lambda session_id: False,
        get_metadata=lambda session_id: None,
    )
    group._bind_traceparent = lambda traceparent: None
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: "0"
    group._get_base_weight_checksum = lambda: "0"
    group._get_buffer_checksum = lambda: "0"
    group._get_optimizer_param_counts = lambda: {}
    group._swap_session_on_workers = lambda session_id: (_ for _ in ()).throw(
        AssertionError("invalid metadata must fail before swap")
    )
    group.save_adapter_state = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("invalid metadata must fail before saving outgoing session")
    )
    group.reinit_lora_weights = lambda *args, **kwargs: None
    group.reset_expert_bias = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="missing session_metadata.json"):
        group._ensure_session_loaded("session_target")


def test_issue_193_megatron_current_session_corruption_fails_closed():
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = "session_current"
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group.workers = []
    group._session_manager = SimpleNamespace(
        has_actor_only_state=lambda session_id: (_ for _ in ()).throw(
            RuntimeError("Failed to read actor_only_state marker")
        ),
        session_exists=lambda session_id: True,
        get_metadata=lambda session_id: {"step": 1, "lr": 1e-4, "actual_rank": 8},
    )
    group._bind_traceparent = lambda traceparent: None

    with pytest.raises(RuntimeError, match="Failed to read actor_only_state marker"):
        group._ensure_session_loaded("session_current")


def test_issue_193_megatron_explicit_load_prepare_allows_dirty_target_on_fresh_actor(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    group._current_session = None
    group.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group._session_unknown_due_to_partial_swap = False
    swap_calls: list[str] = []
    clear_calls: list[str] = []

    class _FakeWorker:
        class clear_session_state:
            @staticmethod
            def remote(session_id, traceparent=None):
                clear_calls.append(session_id)
                return object()

    group.workers = [_FakeWorker()]
    group._session_manager = SimpleNamespace(
        get_session_path=lambda session_id: f"/tmp/{session_id}",
        has_actor_only_state=lambda session_id: (_ for _ in ()).throw(
            AssertionError("explicit checkpoint prepare must not consult target dirty marker")
        ),
    )
    group._bind_traceparent = lambda traceparent: None
    group._swap_session_on_workers = lambda session_id: swap_calls.append(session_id)
    monkeypatch.setattr(ray, "get", lambda refs, timeout=None: None)

    group._prepare_session_for_explicit_load("session_target")

    assert clear_calls == ["session_target"]
    assert swap_calls == ["session_target"]
    assert group._current_session == "session_target"


def test_issue_193_megatron_forward_uses_ensure_session_loaded(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    ensure_calls: list[str] = []

    class _FakeWorker:
        class forward:
            @staticmethod
            def remote(data_items, reset_bias, traceparent=None):
                return {"loss_fn_outputs": [{"loss": {"data": [0.0]}}], "loss_value": 0.0, "num_tokens": 0}

    group.workers = [_FakeWorker()]
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda session_id, **kwargs: ensure_calls.append(session_id) or {"switched": False}
    group._prepare_session_for_explicit_load = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("forward must not use explicit-load preparation")
    )
    monkeypatch.setattr(ray, "get", lambda futures, timeout=None: futures)

    result = group.forward([{"x": 1}], session_id="session_target")

    assert ensure_calls == ["session_target"]
    assert result["loss_fn_outputs"][0]["loss"]["data"] == [0.0]


def test_issue_193_megatron_load_checkpoint_uses_ensure_session_loaded(tmp_path: Path, monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_actor_class__
    group = group_cls.__new__(group_cls)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_000_000_adapter.pt").write_bytes(b"adapter")
    (ckpt_dir / "adapter_config.json").write_text(
        json.dumps({"r": 8, "target_modules": ["gate_proj", "up_proj", "down_proj"]}),
        encoding="utf-8",
    )
    (ckpt_dir / "training_meta.json").write_text(
        json.dumps({"current_step": 3, "learning_rate": 2e-4}),
        encoding="utf-8",
    )

    ensure_calls: list[tuple[str, dict]] = []
    load_adapter_calls: list[tuple[str, dict]] = []
    reset_optimizer_calls: list[tuple[tuple, dict]] = []
    group.workers = []
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = (
        lambda session_id, **kwargs: ensure_calls.append((session_id, kwargs)) or {"switched": False}
    )
    group.load_adapter_state = lambda load_path, **kwargs: load_adapter_calls.append((load_path, kwargs)) or {}
    group.reset_optimizer = lambda *args, **kwargs: reset_optimizer_calls.append((args, kwargs)) or None
    group._step_count = 0
    group.learning_rate = 1e-4
    group._actual_rank = None
    group.lora_rank = 8
    monkeypatch.setattr(ray, "get", lambda refs, timeout=None: None)

    result = group.load_checkpoint(str(ckpt_dir), load_optimizer=False, session_id="session_target")

    assert ensure_calls == [
        (
            "session_target",
            {
                "traceparent": None,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
        )
    ]
    assert load_adapter_calls == [
        (
            str(ckpt_dir),
            {
                "actual_rank": 8,
                "traceparent": None,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
        )
    ]
    assert reset_optimizer_calls == [
        (
            (2e-4,),
            {"traceparent": None},
        )
    ]
    assert result["optimizer_reset"] is True


def test_issue_193_megatron_resolution_falls_back_to_default_base_model():
    engine = VerlTrainingEngine(default_base_model="/tmp/wrong-default")
    session = TrainingSession(
        model_id="model_issue_193_megatron_resolution_strict",
        session_id="session_issue_193_megatron_resolution_strict",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    engine._resolve_hf_model_path = lambda requested_model: None

    resolved_path, requested_model = engine._resolve_session_base_model(session)
    assert resolved_path == "/tmp/wrong-default"
    assert requested_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_issue_193_megatron_midcall_mutating_op_fails_closed_even_when_actor_was_clean(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_midcall_mutating"
    dead_worker = object()
    recovered_worker = object()
    actor_name = "shared-megatron-actor"
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = actor_name

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_midcall_mutating",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward_backward"
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_megatron_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward_backward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="operation may have partially executed before the crash"):
        asyncio.run(_run())

    assert engine._poisoned_sessions[model_id].startswith(
        f"[{model_id}] megatron actor died during op=forward_backward"
    )


def test_issue_193_dense_recycle_fails_loud_after_dead_worker_during_forward(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_dense_dead_midcall"
    dead_worker = object()
    recovered_worker = object()
    engine._workers[model_id] = dead_worker
    engine._resource_pool_actor_names[model_id] = "dense-actor"

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_dense_dead_midcall",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )

    async def fake_get_live_worker(*args, **kwargs):
        return dead_worker

    async def fake_keepalive(awaitable, *_args, **_kwargs):
        raise ray.exceptions.ActorDiedError()

    async def fake_recycle(recycle_session, *, op, cause):
        assert recycle_session is session
        assert op == "forward"
        return recovered_worker

    monkeypatch.setattr(engine, "_get_live_worker", fake_get_live_worker)
    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(engine, "_recycle_dense_actor", fake_recycle)
    monkeypatch.setattr(engine, "_log_worker_request_context", _noop_log_worker_request_context)

    async def _run():
        await engine._run_worker_call_with_actor_recycle(
            session,
            op="forward",
            submit_fn=lambda worker: worker,
        )

    with pytest.raises(RuntimeError, match="dense actor recycle detected after op=forward"):
        asyncio.run(_run())

    assert model_id in engine._poisoned_sessions
    assert model_id in engine._poisoned_sessions


@pytest.mark.parametrize(
    "meta_payload",
    [
        {},
        {"current_step": "42", "learning_rate": "oops"},
        ["not-a-dict"],
    ],
    ids=["missing_current_step", "current_step_wrong_type", "meta_wrong_type"],
)
def test_issue_193_load_weights_invalid_meta_warns_without_pollution(monkeypatch, caplog, meta_payload):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_load"
    worker = _FakeLoadWorker(ref="invalid-meta-load-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "shared-actor"
    monkeypatch.setattr(engine, "_get_live_worker", lambda *args, **kwargs: asyncio.sleep(0, result=worker))

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_load",
        model_seq_id=0,
        base_model="Qwen/Qwen3-0.6B",
        backend="peft",
    )
    session.current_step = 77
    session.learning_rate = 1.5e-4

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert awaitable == "invalid-meta-load-ref"
        assert keepalive_session is session
        return meta_payload

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_invalid_meta_load",
            load_optimizer=True,
        )

    with caplog.at_level(logging.WARNING):
        asyncio.run(_run())

    assert session.current_step == 77
    assert session.learning_rate == pytest.approx(1.5e-4)
    assert any("load_weights" in rec.getMessage() for rec in caplog.records)


def test_issue_193_megatron_load_weights_invalid_meta_fails_loud(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_invalid_meta_megatron_load"
    worker = _FakeLoadWorker(ref="invalid-meta-megatron-load-ref")
    engine._workers[model_id] = worker
    engine._resource_pool_actor_names[model_id] = "shared-megatron-actor"
    monkeypatch.setattr(engine, "_get_live_worker", lambda *args, **kwargs: asyncio.sleep(0, result=worker))

    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_invalid_meta_megatron_load",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="megatron",
    )
    session.current_step = 12
    session.learning_rate = 9e-5

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        assert keepalive_session is session
        if awaitable == "fake-load-ready-ref":
            return {"status": "ok"}
        assert awaitable == "invalid-meta-megatron-load-ref"
        return ["not-a-dict"]

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)
    monkeypatch.setattr(ray, "get", lambda ref, timeout=None: {"status": "ok"})

    async def _run():
        await engine.load_weights(
            session=session,
            load_path="/tmp/issue_193_invalid_meta_megatron_load",
            load_optimizer=True,
        )

    with pytest.raises(RuntimeError, match="Megatron load_checkpoint returned invalid metadata"):
        asyncio.run(_run())

    assert session.current_step == 12
    assert session.learning_rate == pytest.approx(9e-5)
    assert worker.mark_session_loaded.calls == []


def test_issue_193_megatron_create_training_session_marks_ready_without_waiting(monkeypatch):
    engine = VerlTrainingEngine()
    model_id = "model_issue_193_megatron_create_ready"
    worker = _FakeLoadWorker(ref="unused-load-ref")
    session = TrainingSession(
        model_id=model_id,
        session_id="session_issue_193_megatron_create_ready",
        model_seq_id=0,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        backend="peft",
    )

    monkeypatch.setattr(engine, "_resolve_hf_model_path", lambda requested_model: f"/resolved/{requested_model}")
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_model_config",
        lambda _model: SimpleNamespace(is_moe=True, train_use_fp8=False),
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_registry.get_training_parallelism",
        lambda _model: (1, 1, 1, 1, 1),
    )
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.async_get_or_create_megatron_worker_group",
        lambda **kwargs: asyncio.sleep(0, result=worker),
    )
    monkeypatch.setattr(
        "tinker_server.backend.resource_pool.get_resource_pool",
        lambda: SimpleNamespace(
            touch=lambda *_args, **_kwargs: None,
            set_session=lambda *_args, **_kwargs: None,
            mark_ready=lambda *_args, **_kwargs: None,
            clear_session=lambda *_args, **_kwargs: None,
        ),
    )

    keepalive_calls: list[tuple[object, str, float, float | None]] = []

    async def fake_keepalive(awaitable, keepalive_session, interval_s=30.0, timeout_s=None):
        keepalive_calls.append((awaitable, keepalive_session.model_id, interval_s, timeout_s))
        return {"status": "ok"}

    monkeypatch.setattr(engine, "_await_with_keepalive", fake_keepalive)

    async def _run():
        await engine.create_training_session(session)

    asyncio.run(_run())

    assert keepalive_calls == []
    assert engine._workers[model_id] is worker
    assert session.backend == "megatron"
    assert session.is_active is True


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
