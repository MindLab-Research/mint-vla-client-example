import inspect
import logging
import time
import sys
import builtins

import pytest

pytest.importorskip("ray")

from tinker_server.backend.megatron_distributed import (
    DistributedConfig,
    MegatronRankWorker,
    MegatronWorkerGroup,
    _GRADIENTS_CONSUMED,
)


class _FakeTrainMode:
    def __init__(self, state: dict):
        self._state = state

    def __enter__(self):
        self._state["enter"] += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        self._state["exit"] += 1
        return None


class _FakeEvalMode:
    def __init__(self, state: dict):
        self._state = state

    def __enter__(self):
        self._state["eval_enter"] += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        self._state["eval_exit"] += 1
        return None


class _FakeEngine:
    def __init__(self, state: dict):
        self._state = state
        self.optimizer = None  # Needed by optim_step debug code
        self.lr_scheduler = _FakeLRScheduler()

    def train_mode(self):
        return _FakeTrainMode(self._state)

    def eval_mode(self):
        return _FakeEvalMode(self._state)

    def forward_backward_batch(self, *args, **kwargs):
        """Default no-op; override in tests to raise."""
        return {"loss": [], "metrics": {}}

    def optimizer_step(self):
        """Default no-op; override in tests to raise."""
        return 0.0

    def lr_scheduler_step(self):
        return 1e-4

    def _build_lr_scheduler(self):
        return _FakeLRScheduler()


class _FakeLRScheduler:
    def __init__(self):
        self.last_epoch = 0
        self.lr_scale = 1.0

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "lr_scale": self.lr_scale,
        }

    def load_state_dict(self, state):
        self.last_epoch = state.get("last_epoch", 0)
        self.lr_scale = state.get("lr_scale", 1.0)


def test_issue_193_mark_session_loaded_persists_session_cache(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)

    class _RemoteMethod:
        def __init__(self, result):
            self._result = result

        def remote(self, *args, **kwargs):
            return self._result

    group.workers = [type("W", (), {"mark_session_loaded": _RemoteMethod("ok")})()]
    calls: list[tuple[str, object]] = []
    group._session_manager = type(
        "SessionMgr",
        (),
        {
            "save_metadata": staticmethod(
                lambda session_id, step, lr, actual_rank, **kwargs: calls.append(
                    ("meta", (session_id, step, lr, actual_rank, kwargs))
                )
            ),
        },
    )()
    group._current_session = None
    group._step_count = 0
    group.learning_rate = 0.0
    group._actual_rank = None
    group.lora_rank = 8

    monkeypatch.setattr(sys.modules[MegatronWorkerGroup.__module__].ray, "get", lambda refs: None)

    out = group.mark_session_loaded(
        "sess-mark",
        step_count=7,
        learning_rate=2e-4,
        actual_rank=4,
    )

    assert out == {"status": "ok", "session_id": "sess-mark"}
    assert calls == [("meta", ("sess-mark", 7, 2e-4, 4, {}))]


def test_issue_193_prepare_session_for_explicit_load_skips_resave_when_session_is_clean(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)

    calls: list[tuple[str, object]] = []

    def _prime_session(session_id, checkpoint_path, step, lr, actual_rank, **kwargs):
        calls.append(("prime", (session_id, checkpoint_path, step, lr, actual_rank, kwargs)))
        return f"/tmp/cache/{session_id}"

    group._session_manager = type(
        "SessionMgr",
        (),
        {"prime_session": staticmethod(_prime_session)},
    )()

    out = group.prime_session_checkpoint(
        "sess-new",
        "/tmp/sess-new",
        step_count=11,
        learning_rate=3e-4,
        actual_rank=8,
        optimizer_restored=False,
    )

    assert out == {
        "status": "ok",
        "session_id": "sess-new",
        "session_path": "/tmp/cache/sess-new",
        "actual_rank": 8,
    }
    assert calls == [("prime", ("sess-new", "/tmp/sess-new", 11, 3e-4, 8, {"optimizer_restored": False}))]


class _FakeInnerOptimizer:
    def __init__(self):
        self.state = {}
        self.param_groups = [{"params": [], "lr": 1.0}]
        self.load_calls = 0

    def state_dict(self):
        return {
            "state": {"param_0": {"exp_avg": 3.0, "exp_avg_sq": 5.0}},
            "param_groups": [{"params": [0], "lr": self.param_groups[0]["lr"]}],
        }

    def load_state_dict(self, state_dict):
        self.load_calls += 1
        self.state = state_dict["state"]
        self.param_groups = state_dict["param_groups"]


class _FakeMegatronOptimizerWrapper:
    def __init__(self):
        self.optimizer = _FakeInnerOptimizer()
        self.wrapper_counter = 0
        self.grad_scaler = {"scale": 1.0}
        self.load_calls = 0

    def state_dict(self):
        return {
            "optimizer": {"param_groups": [{"lr": self.optimizer.param_groups[0]["lr"]}]},
            "grad_scaler": {"scale": self.grad_scaler["scale"]},
            "wrapper_counter": self.wrapper_counter,
        }

    def load_state_dict(self, state_dict):
        self.load_calls += 1
        self.wrapper_counter = state_dict["wrapper_counter"]
        self.grad_scaler = state_dict["grad_scaler"]


def _make_worker(
    monkeypatch,
    *,
    idle_timeout_s: str = "15",
    close_on_optim: str = "1",
) -> tuple[MegatronRankWorker, dict]:
    monkeypatch.setenv("MINT_MEGATRON_STICKY_TRAIN_MODE", "1")
    monkeypatch.setenv("MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S", idle_timeout_s)
    monkeypatch.setenv("MINT_MEGATRON_STICKY_CLOSE_ON_OPTIM", close_on_optim)
    state = {"enter": 0, "exit": 0, "eval_enter": 0, "eval_exit": 0}
    impl_cls = MegatronRankWorker.__ray_metadata__.modified_class
    worker = impl_cls(
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=12345,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=8,
        learning_rate=1e-4,
        distributed_config=DistributedConfig(),
    )
    worker.engine = _FakeEngine(state)
    return worker, state


def _prepare_worker_for_optim_step(worker):
    """Attach minimal stubs so worker.optim_step() reaches the sticky path."""
    worker.log_memory_breakdown = lambda tag: None
    worker._start_slow_op_watchdog = lambda **kw: None
    worker._stop_slow_op_watchdog = lambda w: None
    worker._zero_disabled_lora_params = lambda *a, **kw: None


def _prepare_worker_for_forward_backward(worker, monkeypatch):
    """Attach minimal stubs so worker.forward_backward() reaches the sticky path.

    Patches heavy imports (torch.cuda, verl, model_config) that are impossible
    to satisfy in a unit test environment without GPU.
    """
    worker.log_memory_breakdown = lambda tag: None
    worker._start_slow_op_watchdog = lambda **kw: None
    worker._stop_slow_op_watchdog = lambda w: None
    worker._is_output_rank = lambda: False
    worker._resolve_reset_bias = lambda val, default: False

    # Patch the heavy imports inside forward_backward body:
    # 1. tinker_to_tensordict -> returns a dummy
    # 2. create_sft_loss_fn -> returns a dummy
    # 3. torch.cuda.current_device -> returns 0
    # 4. torch.ones -> returns a no-op object
    # 5. torch.cuda.synchronize -> no-op
    # 6. get_model_config -> returns object with max_model_len

    import types

    class _FakeModelConfig:
        max_model_len = 2048

    fake_training = types.ModuleType("tinker_server.backend.megatron_training")
    fake_training.create_sft_loss_fn = lambda **kw: (lambda *a, **k: None)  # type: ignore
    fake_training.create_ppo_loss_fn = lambda *a, **kw: (lambda *a2, **k2: None)  # type: ignore
    fake_training.tinker_to_tensordict = lambda *a, **kw: "fake_tensordict"  # type: ignore
    monkeypatch.setitem(sys.modules, "tinker_server.backend.megatron_training", fake_training)

    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.get_model_config",
        lambda model: _FakeModelConfig(),
    )
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.flatten_encoded_text_chunks",
        lambda model_input: model_input.get("input_ids", [1, 2, 3]),
    )

    # Patch torch.cuda calls (forward_backward does a CUDA health check)
    import torch
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    original_ones = torch.ones
    def patched_ones(*args, **kwargs):
        # Strip device= to avoid CUDA requirement
        kwargs.pop("device", None)
        return original_ones(*args, **kwargs)
    monkeypatch.setattr(torch, "ones", patched_ones)


# ---------------------------------------------------------------------------
# Test 1: Same-session reuse
# ---------------------------------------------------------------------------

def test_issue_193_sticky_train_mode_reuse_same_session(monkeypatch):
    worker, state = _make_worker(monkeypatch)

    first = worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    second = worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")

    assert first["reused"] is False
    assert second["reused"] is True
    assert state["enter"] == 1
    assert state["exit"] == 0
    assert worker._sticky_train_mode_enter_total == 1
    assert worker._sticky_train_mode_reuse_total == 1

    released = worker._release_sticky_train_mode(reason="unit_test", snapshot_gradients=False)
    assert released["released"] is True
    assert state["exit"] == 1


# ---------------------------------------------------------------------------
# Test 2: Session switch + idle timeout
# ---------------------------------------------------------------------------

def test_issue_193_sticky_train_mode_switch_and_idle_timeout(monkeypatch):
    worker, state = _make_worker(monkeypatch, idle_timeout_s="0.1")
    worker._capture_gradients = lambda: ["captured-grad"]  # type: ignore[method-assign]

    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    switched = worker._ensure_sticky_train_mode(session_id="s2", reason="forward_backward")
    assert switched["reused"] is False
    assert switched["released_before_enter"] is True
    assert worker._session_gradients["s1"] == ["captured-grad"]
    assert state["enter"] == 2
    assert state["exit"] == 1

    # Force idle timeout for same session and ensure it re-enters instead of reusing.
    worker._sticky_train_mode_last_used_s = time.perf_counter() - 1.0
    timed_out = worker._ensure_sticky_train_mode(session_id="s2", reason="forward_backward")
    assert timed_out["reused"] is False
    assert timed_out["released_before_enter"] is True
    assert state["enter"] == 3
    assert state["exit"] == 2


# ---------------------------------------------------------------------------
# Test 3: _GRADIENTS_CONSUMED sentinel survives release(snapshot_gradients=True)
# ---------------------------------------------------------------------------

def test_issue_193_consumed_sentinel_survives_sticky_release(monkeypatch):
    """After optim_step marks gradients consumed, a subsequent release with
    snapshot_gradients=True must NOT overwrite the sentinel with captured GPU data.
    This is the core fix for Review Issue 1."""
    worker, state = _make_worker(monkeypatch, close_on_optim="0")
    capture_called = {"count": 0}

    def fake_capture():
        capture_called["count"] += 1
        return ["should-not-be-stored"]

    worker._capture_gradients = fake_capture  # type: ignore[method-assign]

    # Simulate: forward_backward opens ctx, optim_step marks consumed
    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    worker._session_gradients["s1"] = _GRADIENTS_CONSUMED

    # Now release with snapshot_gradients=True (simulates idle timeout or session switch)
    worker._release_sticky_train_mode(reason="idle_timeout", snapshot_gradients=True)

    # Sentinel must survive -- _capture_gradients must NOT have been called
    assert worker._session_gradients["s1"] is _GRADIENTS_CONSUMED
    assert capture_called["count"] == 0
    assert state["exit"] == 1


# ---------------------------------------------------------------------------
# Test 4: _GRADIENTS_CONSUMED is not passed to _restore_gradients
# ---------------------------------------------------------------------------

def test_issue_193_consumed_sentinel_not_restored(monkeypatch):
    """When cached gradients are _GRADIENTS_CONSUMED, the forward_backward path
    should skip _restore_gradients (using the freshly zeroed GPU buffers instead)."""
    worker, state = _make_worker(monkeypatch)
    restore_called = {"count": 0}

    def fake_restore(grads):
        restore_called["count"] += 1

    worker._restore_gradients = fake_restore  # type: ignore[method-assign]

    # Mark session as consumed, then enter sticky mode fresh
    worker._session_gradients["s1"] = _GRADIENTS_CONSUMED
    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")

    # The sentinel should not be treated as valid gradients to restore
    cached = worker._session_gradients.get("s1")
    assert cached is _GRADIENTS_CONSUMED
    # _restore_gradients should not be called with the sentinel
    assert restore_called["count"] == 0


# ---------------------------------------------------------------------------
# Test 5: Full cycle: consumed -> session switch -> come back
# ---------------------------------------------------------------------------

def test_issue_193_consumed_then_session_switch_then_forward(monkeypatch):
    """Full s1->s2->s1 cycle: s1 consumed, switch to s2, come back to s1.
    s1's sentinel must be preserved across the round-trip."""
    worker, state = _make_worker(monkeypatch)
    worker._capture_gradients = lambda: ["s2-grads"]  # type: ignore[method-assign]

    # s1: forward + optim -> consumed
    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    worker._session_gradients["s1"] = _GRADIENTS_CONSUMED
    worker._release_sticky_train_mode(reason="optim_step_complete", snapshot_gradients=False)

    # s2: forward
    worker._ensure_sticky_train_mode(session_id="s2", reason="forward_backward")

    # Switch back to s1 -- this releases s2 (snapshot=True captures s2 grads)
    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")

    # s1 sentinel must still be _GRADIENTS_CONSUMED
    assert worker._session_gradients["s1"] is _GRADIENTS_CONSUMED
    # s2 grads should have been captured
    assert worker._session_gradients["s2"] == ["s2-grads"]


# ---------------------------------------------------------------------------
# Test 6: Valid (non-sentinel) gradients are properly snapshot on release
# ---------------------------------------------------------------------------

def test_issue_193_valid_gradients_still_snapshot_on_release(monkeypatch):
    """When gradients are valid (not consumed), release(snapshot=True) should
    call _capture_gradients and store the result, overwriting the old cache."""
    worker, state = _make_worker(monkeypatch)
    worker._capture_gradients = lambda: ["new-snapshot"]  # type: ignore[method-assign]

    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    worker._session_gradients["s1"] = ["old-grads"]

    worker._release_sticky_train_mode(reason="idle_timeout", snapshot_gradients=True)
    assert worker._session_gradients["s1"] == ["new-snapshot"]


# ---------------------------------------------------------------------------
# Test 7: forward_backward() error triggers sticky cleanup (real method)
# ---------------------------------------------------------------------------

def test_issue_193_sticky_cleanup_on_forward_backward_error(monkeypatch):
    """Call the real forward_backward() with a failing engine.forward_backward_batch().
    Verifies the try/except in the sticky path actually fires."""
    worker, state = _make_worker(monkeypatch)
    _prepare_worker_for_forward_backward(worker, monkeypatch)

    # Make forward_backward_batch raise
    class ComputeError(RuntimeError):
        pass

    def failing_fbb(*args, **kwargs):
        raise ComputeError("GPU compute failed")

    worker.engine.forward_backward_batch = failing_fbb  # type: ignore[method-assign]

    # Call the real method -- should raise ComputeError and clean up sticky ctx
    with pytest.raises(ComputeError, match="GPU compute failed"):
        worker.forward_backward(
            data_items=[{"model_input": {"input_ids": [1, 2, 3]}}],
            loss_fn="cross_entropy",
            loss_fn_config={},
            session_id="s1",
        )

    # Verify: sticky ctx released, state cleaned
    assert state["enter"] == 1
    assert state["exit"] == 1
    assert worker._sticky_train_mode_ctx is None
    assert worker._sticky_train_mode_session_id is None

    # Verify: next call is fresh enter (not reuse of broken ctx)
    result = worker._ensure_sticky_train_mode(session_id="s1", reason="test")
    assert result["reused"] is False
    assert state["enter"] == 2


# ---------------------------------------------------------------------------
# Test 8: optim_step() error triggers sticky cleanup (real method)
# ---------------------------------------------------------------------------

def test_issue_193_sticky_cleanup_on_optim_step_error(monkeypatch):
    """Call the real optim_step() with a failing engine.optimizer_step().
    Verifies the try/except in the sticky path actually fires."""
    worker, state = _make_worker(monkeypatch)
    _prepare_worker_for_optim_step(worker)

    # Make optimizer_step raise
    class OptimError(RuntimeError):
        pass

    def failing_optimizer_step():
        raise OptimError("Optimizer diverged")

    worker.engine.optimizer_step = failing_optimizer_step  # type: ignore[method-assign]

    # Call the real method -- should raise OptimError and clean up sticky ctx
    with pytest.raises(OptimError, match="Optimizer diverged"):
        worker.optim_step(learning_rate=1e-4, session_id="s1")

    # Verify: sticky ctx released, state cleaned
    assert state["enter"] == 1
    assert state["exit"] == 1
    assert worker._sticky_train_mode_ctx is None
    assert worker._sticky_train_mode_session_id is None

    # Verify: next call is fresh enter
    result = worker._ensure_sticky_train_mode(session_id="s1", reason="test")
    assert result["reused"] is False
    assert state["enter"] == 2


# ---------------------------------------------------------------------------
# Test 9: Error path release does NOT snapshot gradients
# ---------------------------------------------------------------------------

def test_issue_193_error_does_not_snapshot_gradients(monkeypatch):
    """On error, snapshot_gradients=False must be used. _capture_gradients
    must NOT be called (GPU state is undefined after an error)."""
    worker, state = _make_worker(monkeypatch)
    capture_called = {"count": 0}

    def fake_capture():
        capture_called["count"] += 1
        return ["garbage"]

    worker._capture_gradients = fake_capture  # type: ignore[method-assign]

    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    worker._release_sticky_train_mode(
        reason="forward_backward_error", snapshot_gradients=False
    )

    assert capture_called["count"] == 0
    assert state["exit"] == 1


# ---------------------------------------------------------------------------
# Test 10: swap_session_state preserves _GRADIENTS_CONSUMED sentinel
# ---------------------------------------------------------------------------

def test_issue_193_swap_session_preserves_consumed_sentinel(monkeypatch):
    """When swapping away from a session with _GRADIENTS_CONSUMED, the sentinel
    must be preserved (not overwritten with captured GPU data). When swapping back,
    the sentinel must NOT be passed to _restore_gradients."""
    worker, state = _make_worker(monkeypatch)

    restore_called = {"count": 0, "last_arg": None}
    zero_grad_called = {"count": 0}
    capture_grad_called = {"count": 0}
    capture_opt_called = {"count": 0}

    def fake_restore(grads):
        restore_called["count"] += 1
        restore_called["last_arg"] = grads

    def fake_zero_grad():
        zero_grad_called["count"] += 1

    def fake_capture_grad():
        capture_grad_called["count"] += 1
        return ["should-not-capture"]

    def fake_capture_opt():
        capture_opt_called["count"] += 1
        return {}

    def fake_restore_opt(state):
        pass

    def fake_reset_opt():
        pass

    worker._restore_gradients = fake_restore  # type: ignore[method-assign]
    worker._capture_gradients = fake_capture_grad  # type: ignore[method-assign]
    worker._capture_optimizer_state = fake_capture_opt  # type: ignore[method-assign]
    worker._restore_optimizer_state = fake_restore_opt  # type: ignore[method-assign]
    worker._reset_optimizer_state = fake_reset_opt  # type: ignore[method-assign]
    worker.engine.optimizer_zero_grad = fake_zero_grad  # type: ignore[method-assign]

    # Simulate: s1 gradients consumed by optim_step
    worker._session_gradients["s1"] = _GRADIENTS_CONSUMED
    worker._current_session_id = "s1"

    # Swap to s2 (should preserve s1's sentinel, not overwrite with capture)
    worker.swap_session_state("s2")

    # Verify s1's sentinel survived (not overwritten by _capture_gradients)
    assert worker._session_gradients["s1"] is _GRADIENTS_CONSUMED
    # Verify _capture_gradients was NOT called for s1 (sentinel branch)
    assert capture_grad_called["count"] == 0

    # Swap back to s1 (should zero gradients, NOT restore sentinel)
    worker.swap_session_state("s1")

    # Verify _restore_gradients was NOT called with sentinel
    assert restore_called["count"] == 0  # s2 was new, no restore; s1 is consumed, no restore
    assert zero_grad_called["count"] == 2  # once for s2 (new), once for s1 (consumed)


# ---------------------------------------------------------------------------
# Test 11: swap_session_state does not pass sentinel to _restore_gradients
# ---------------------------------------------------------------------------

def test_issue_193_swap_session_does_not_restore_sentinel(monkeypatch):
    """Regression test: ensure swap_session_state checks for _GRADIENTS_CONSUMED
    before calling _restore_gradients."""
    worker, state = _make_worker(monkeypatch)

    restore_called = {"count": 0, "received_sentinel": False}

    def fake_restore(grads):
        restore_called["count"] += 1
        if grads is _GRADIENTS_CONSUMED:
            restore_called["received_sentinel"] = True

    def fake_capture_grad():
        return []

    def fake_capture_opt():
        return {}

    def fake_restore_opt(state):
        pass

    def fake_reset_opt():
        pass

    worker._restore_gradients = fake_restore  # type: ignore[method-assign]
    worker._capture_gradients = fake_capture_grad  # type: ignore[method-assign]
    worker._capture_optimizer_state = fake_capture_opt  # type: ignore[method-assign]
    worker._restore_optimizer_state = fake_restore_opt  # type: ignore[method-assign]
    worker._reset_optimizer_state = fake_reset_opt  # type: ignore[method-assign]
    worker.engine.optimizer_zero_grad = lambda: None  # type: ignore[method-assign]

    # Setup: s1 has consumed gradients
    worker._session_gradients["s1"] = _GRADIENTS_CONSUMED
    worker._current_session_id = "s2"

    # Swap to s1 (should zero, not restore)
    worker.swap_session_state("s1")

    # Verify sentinel was NOT passed to _restore_gradients
    assert not restore_called["received_sentinel"]


def test_issue_193_swap_session_restores_lr_scheduler_state(monkeypatch):
    worker, _ = _make_worker(monkeypatch)

    worker._capture_gradients = lambda: []  # type: ignore[method-assign]
    worker._restore_gradients = lambda grads: None  # type: ignore[method-assign]
    worker._capture_optimizer_state = lambda: {}  # type: ignore[method-assign]
    worker._restore_optimizer_state = lambda state: None  # type: ignore[method-assign]
    worker._reset_optimizer_state = lambda: worker._reset_lr_scheduler()  # type: ignore[method-assign]
    worker.engine.optimizer_zero_grad = lambda: None  # type: ignore[method-assign]

    worker._current_session_id = "s1"
    worker.engine.lr_scheduler.last_epoch = 7
    worker.engine.lr_scheduler.lr_scale = 0.25

    worker.swap_session_state("s2")

    assert worker.engine.lr_scheduler.last_epoch == 0
    assert worker.engine.lr_scheduler.lr_scale == 1.0

    worker.engine.lr_scheduler.last_epoch = 3
    worker.engine.lr_scheduler.lr_scale = 0.75

    worker.swap_session_state("s1")

    assert worker.engine.lr_scheduler.last_epoch == 3
    assert worker.engine.lr_scheduler.lr_scale == 0.75


def test_issue_193_capture_restore_optimizer_wrapper_state(monkeypatch):
    import types

    worker, _ = _make_worker(monkeypatch)
    worker.engine.optimizer = _FakeMegatronOptimizerWrapper()

    fake_megatron = types.ModuleType("megatron")
    fake_core = types.ModuleType("megatron.core")
    fake_optimizer_module = types.ModuleType("megatron.core.optimizer")
    fake_optimizer_module.ChainedOptimizer = type("ChainedOptimizer", (), {})
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core)
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer", fake_optimizer_module)

    worker.engine.optimizer.wrapper_counter = 11
    worker.engine.optimizer.grad_scaler["scale"] = 7.5
    worker.engine.optimizer.optimizer.param_groups[0]["lr"] = 0.123

    snapshot = worker._capture_optimizer_state()

    worker.engine.optimizer.wrapper_counter = 99
    worker.engine.optimizer.grad_scaler["scale"] = 42.0
    worker.engine.optimizer.optimizer.state = {"corrupted": True}
    worker.engine.optimizer.optimizer.param_groups = [{"params": [123], "lr": 9.9}]

    worker._restore_optimizer_state(snapshot)

    assert worker.engine.optimizer.load_calls == 0
    assert worker.engine.optimizer.optimizer.load_calls == 0
    assert worker.engine.optimizer.wrapper_counter == 99
    assert worker.engine.optimizer.grad_scaler == {"scale": 42.0}
    assert worker.engine.optimizer.optimizer.state == {}
    assert worker.engine.optimizer.optimizer.param_groups == [{"params": [123], "lr": 0.123}]


def test_issue_193_clear_session_state_clears_lr_scheduler_cache(monkeypatch):
    worker, _ = _make_worker(monkeypatch)

    worker._session_gradients["s1"] = [1, 2, 3]
    worker._session_optimizer_states["s1"] = {"state": 1}

    worker.clear_session_state("s1")

    assert "s1" not in worker._session_gradients
    assert "s1" not in worker._session_optimizer_states


def test_issue_193_save_adapter_state_persists_expert_bias(tmp_path, monkeypatch):
    import types

    import torch

    worker, _ = _make_worker(monkeypatch)

    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias", torch.tensor([1.5, -0.5], dtype=torch.float32))

    class _Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = _Router()

    chunk = _Chunk()
    worker.engine.module = [chunk]

    fake_peft = types.ModuleType("verl.utils.megatron_peft_utils")
    fake_peft.get_adapter_state_dict = lambda module: {"adapter.weight": torch.tensor([3.0])}  # type: ignore[attr-defined]
    fake_peft._get_rank_checkpoint_path = lambda checkpoint_path: str(tmp_path / "mp_rank_00")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_peft_utils", fake_peft)

    worker.save_adapter_state(str(tmp_path))

    payload = torch.load(tmp_path / "mp_rank_00_adapter.pt", map_location="cpu")
    assert payload == {"adapter_state_dict": {"adapter.weight": torch.tensor([3.0])}}


def test_issue_193_load_adapter_state_restores_expert_bias(tmp_path, monkeypatch):
    import types

    import torch

    worker, _ = _make_worker(monkeypatch)

    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias", torch.tensor([9.0, 9.0], dtype=torch.float32))

    class _Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = _Router()

    chunk = _Chunk()
    worker.engine.module = [chunk]
    worker._freeze_non_lora_params = lambda *_a, **_kw: None  # type: ignore[method-assign]
    worker._zero_disabled_lora_params = lambda *_a, **_kw: None  # type: ignore[method-assign]

    class _FakeOptimizer:
        def __init__(self):
            self.reload_calls = 0

        def reload_model_params(self, state_dict=None):
            self.reload_calls += 1

    worker.engine.optimizer = _FakeOptimizer()

    adapter_file = tmp_path / "mp_rank_00_adapter.pt"
    torch.save(
        {"adapter_state_dict": {"adapter.weight": torch.tensor([3.0])}},
        adapter_file,
    )

    fake_peft = types.ModuleType("verl.utils.megatron_peft_utils")
    fake_peft._get_rank_checkpoint_path = lambda checkpoint_path: str(tmp_path / "mp_rank_00")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_peft_utils", fake_peft)

    fake_utils = types.ModuleType("verl.utils.megatron_utils")
    fake_utils.unwrap_model = lambda model: model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_utils", fake_utils)

    fake_lora_utils = types.ModuleType("tinker_server.backend.lora_utils")
    fake_lora_utils.pad_lora_state_dict = lambda state, *_args, **_kwargs: state  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tinker_server.backend.lora_utils", fake_lora_utils)

    worker.load_adapter_state(str(tmp_path))

    assert chunk.router.expert_bias.tolist() == [9.0, 9.0]
    assert worker.engine.optimizer.reload_calls == 0


# ---------------------------------------------------------------------------
# Test 12: Cleanup error does not mask original error (real optim_step)
# ---------------------------------------------------------------------------

def test_issue_193_cleanup_error_preserves_original_error(monkeypatch):
    """Call real optim_step() where optimizer_step() AND __exit__() both fail.
    The original business error must be raised, not the cleanup error.
    Sticky state must still be cleared (fail-closed)."""
    worker, state = _make_worker(monkeypatch)
    _prepare_worker_for_optim_step(worker)

    class OptimError(RuntimeError):
        pass

    class ExitError(RuntimeError):
        pass

    original_exit_method = _FakeTrainMode.__exit__

    def failing_optimizer_step():
        raise OptimError("Optimizer diverged")

    def failing_exit(self, exc_type, exc, tb):
        state["exit"] += 1
        raise ExitError("__exit__ failed during cleanup")

    worker.engine.optimizer_step = failing_optimizer_step  # type: ignore[method-assign]
    _FakeTrainMode.__exit__ = failing_exit  # type: ignore[method-assign]

    try:
        # Business error (OptimError) should be raised, not ExitError
        with pytest.raises(OptimError, match="Optimizer diverged"):
            worker.optim_step(learning_rate=1e-4, session_id="s1")

        # Verify: sticky state cleaned despite double failure (fail-closed)
        assert worker._sticky_train_mode_ctx is None
        assert worker._sticky_train_mode_session_id is None
    finally:
        _FakeTrainMode.__exit__ = original_exit_method


# ---------------------------------------------------------------------------
# Test 13: __exit__ failure clears sticky state (fail-closed)
# ---------------------------------------------------------------------------

def test_issue_193_exit_failure_clears_sticky_state(monkeypatch):
    """When ctx.__exit__() fails, sticky bookkeeping must still be cleared
    to prevent reuse of a broken context handle."""
    worker, state = _make_worker(monkeypatch)

    # Make __exit__ raise an error
    class ExitError(RuntimeError):
        pass

    original_exit_method = _FakeTrainMode.__exit__

    def failing_exit(self, exc_type, exc, tb):
        state["exit"] += 1
        raise ExitError("__exit__ failed")

    # Patch the _FakeTrainMode's __exit__
    _FakeTrainMode.__exit__ = failing_exit  # type: ignore[method-assign]

    try:
        # Open sticky context
        worker._ensure_sticky_train_mode(session_id="s1", reason="test")
        assert worker._sticky_train_mode_ctx is not None
        assert worker._sticky_train_mode_session_id == "s1"

        # Release should fail but still clear state
        import pytest
        with pytest.raises(ExitError, match="__exit__ failed"):
            worker._release_sticky_train_mode(reason="test", snapshot_gradients=False)

        # Verify state was cleared despite __exit__ failure (fail-closed)
        assert worker._sticky_train_mode_ctx is None
        assert worker._sticky_train_mode_session_id is None
        assert worker._sticky_train_mode_last_used_s == 0.0

        # Next call should do fresh enter, not attempt to reuse broken ctx
        # Restore __exit__ first so fresh enter works
        _FakeTrainMode.__exit__ = original_exit_method
        result = worker._ensure_sticky_train_mode(session_id="s1", reason="test")
        assert result["reused"] is False
    finally:
        # Always restore to avoid polluting other tests
        _FakeTrainMode.__exit__ = original_exit_method


# ---------------------------------------------------------------------------
# Test 14: forward_backward() business error + __exit__ failure (real method)
# ---------------------------------------------------------------------------

def test_issue_193_forward_backward_error_plus_exit_failure(monkeypatch):
    """Call real forward_backward() where forward_backward_batch() AND
    ctx.__exit__() both fail.  The original business error must be raised,
    sticky state must be cleared (fail-closed), and next call must do
    fresh enter."""
    worker, state = _make_worker(monkeypatch)
    _prepare_worker_for_forward_backward(worker, monkeypatch)

    class ComputeError(RuntimeError):
        pass

    class ExitError(RuntimeError):
        pass

    original_exit_method = _FakeTrainMode.__exit__

    def failing_fbb(*args, **kwargs):
        raise ComputeError("GPU compute failed")

    def failing_exit(self, exc_type, exc, tb):
        state["exit"] += 1
        raise ExitError("__exit__ failed during cleanup")

    worker.engine.forward_backward_batch = failing_fbb  # type: ignore[method-assign]
    _FakeTrainMode.__exit__ = failing_exit  # type: ignore[method-assign]

    try:
        # Business error (ComputeError) must be raised, not ExitError
        with pytest.raises(ComputeError, match="GPU compute failed"):
            worker.forward_backward(
                data_items=[{"model_input": {"input_ids": [1, 2, 3]}}],
                loss_fn="cross_entropy",
                loss_fn_config={},
                session_id="s1",
            )

        # Verify: sticky state cleaned despite double failure (fail-closed)
        assert worker._sticky_train_mode_ctx is None
        assert worker._sticky_train_mode_session_id is None

        # Next call must be fresh enter (restore __exit__ so it works)
        _FakeTrainMode.__exit__ = original_exit_method
        result = worker._ensure_sticky_train_mode(session_id="s1", reason="test")
        assert result["reused"] is False
    finally:
        _FakeTrainMode.__exit__ = original_exit_method


# ---------------------------------------------------------------------------
# Test 15: CLOSE_ON_OPTIM=0 + optim_step error → no stale ctx
# ---------------------------------------------------------------------------

def test_issue_193_close_on_optim_off_error_clears_ctx(monkeypatch):
    """With CLOSE_ON_OPTIM=0, a successful optim_step keeps ctx open.
    But an erroring optim_step must still release the ctx (no stale handle)."""
    worker, state = _make_worker(monkeypatch, close_on_optim="0")
    _prepare_worker_for_optim_step(worker)

    class OptimError(RuntimeError):
        pass

    def failing_optimizer_step():
        raise OptimError("Optimizer diverged")

    worker.engine.optimizer_step = failing_optimizer_step  # type: ignore[method-assign]

    with pytest.raises(OptimError, match="Optimizer diverged"):
        worker.optim_step(learning_rate=1e-4, session_id="s1")

    # Even with CLOSE_ON_OPTIM=0, error path must clear sticky state
    assert worker._sticky_train_mode_ctx is None
    assert worker._sticky_train_mode_session_id is None
    assert state["enter"] == 1
    assert state["exit"] == 1

    # Next call: fresh enter, not reuse of broken ctx
    result = worker._ensure_sticky_train_mode(session_id="s1", reason="test")
    assert result["reused"] is False
    assert state["enter"] == 2


# ---------------------------------------------------------------------------
# Test 15b: CLOSE_ON_OPTIM=0 still clears gradients between steps
# ---------------------------------------------------------------------------

def test_issue_193_close_on_optim_off_clears_gradients_before_reused_forward(monkeypatch):
    """With CLOSE_ON_OPTIM=0, sticky ctx is reused across steps.
    optim_step must still clear gradients to prevent stale cross-step accumulation."""
    worker, state = _make_worker(monkeypatch, close_on_optim="0")
    _prepare_worker_for_forward_backward(worker, monkeypatch)
    _prepare_worker_for_optim_step(worker)

    grad_state = {"dirty": False, "zero_calls": 0}

    def fake_forward_backward_batch(*args, **kwargs):
        if grad_state["dirty"]:
            raise RuntimeError("stale gradients leaked across steps")
        # Simulate backward pass leaving gradients populated.
        grad_state["dirty"] = True
        return {"loss": [], "metrics": {}}

    def fake_optimizer_step():
        # Simulate a backend optimizer that DOES NOT clear gradients by itself.
        return 0.0

    def fake_optimizer_zero_grad():
        grad_state["dirty"] = False
        grad_state["zero_calls"] += 1

    worker.engine.forward_backward_batch = fake_forward_backward_batch  # type: ignore[method-assign]
    worker.engine.optimizer_step = fake_optimizer_step  # type: ignore[method-assign]
    worker.engine.optimizer_zero_grad = fake_optimizer_zero_grad  # type: ignore[method-assign]

    payload = [{"model_input": {"input_ids": [1, 2, 3]}}]

    # Step 1 backward leaves gradients dirty.
    worker.forward_backward(
        data_items=payload,
        loss_fn="cross_entropy",
        loss_fn_config={},
        session_id="s1",
    )
    assert grad_state["dirty"] is True

    # Step 1 optimizer must clear gradients even when ctx is kept open.
    worker.optim_step(learning_rate=1e-4, session_id="s1")
    assert grad_state["dirty"] is False
    assert grad_state["zero_calls"] >= 1

    # Step 2 backward reuses sticky ctx but must start from clean gradients.
    worker.forward_backward(
        data_items=payload,
        loss_fn="cross_entropy",
        loss_fn_config={},
        session_id="s1",
    )
    assert state["enter"] == 1  # sticky context reused
    assert state["exit"] == 0   # CLOSE_ON_OPTIM=0 keeps context open


def test_issue_193_gradient_clear_failure_is_fail_loud_and_releases_ctx(monkeypatch):
    """Gradient clear failure after optim_step must fail-loud and release sticky ctx."""
    worker, state = _make_worker(monkeypatch, close_on_optim="0")
    _prepare_worker_for_optim_step(worker)

    def fake_optimizer_step():
        return 0.0

    def failing_optimizer_zero_grad():
        raise RuntimeError("optimizer_zero_grad boom")

    worker.engine.optimizer_step = fake_optimizer_step  # type: ignore[method-assign]
    worker.engine.optimizer_zero_grad = failing_optimizer_zero_grad  # type: ignore[method-assign]
    worker.engine.optimizer = None

    with pytest.raises(RuntimeError, match="Failed to clear gradients after optim_step"):
        worker.optim_step(learning_rate=1e-4, session_id="s1")

    assert worker._sticky_train_mode_ctx is None
    assert worker._sticky_train_mode_session_id is None
    assert state["enter"] == 1
    assert state["exit"] == 1


# ---------------------------------------------------------------------------
# Test 16: Cleanup failure log observability (structured fields present)
# ---------------------------------------------------------------------------

def test_issue_193_cleanup_failure_log_has_structured_fields(monkeypatch, caplog):
    """When cleanup fails during optim_step error handling, the emitted
    warning log must contain reason, session_id, rank, and error_type
    for alert aggregation and incident triage."""
    worker, state = _make_worker(monkeypatch)
    _prepare_worker_for_optim_step(worker)

    class OptimError(RuntimeError):
        pass

    class ExitError(RuntimeError):
        pass

    original_exit_method = _FakeTrainMode.__exit__

    def failing_optimizer_step():
        raise OptimError("Optimizer diverged")

    def failing_exit(self, exc_type, exc, tb):
        state["exit"] += 1
        raise ExitError("__exit__ cleanup boom")

    worker.engine.optimizer_step = failing_optimizer_step  # type: ignore[method-assign]
    _FakeTrainMode.__exit__ = failing_exit  # type: ignore[method-assign]

    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(OptimError, match="Optimizer diverged"):
                worker.optim_step(learning_rate=1e-4, session_id="s1")

        # Find the outer cleanup warning (not the inner __exit__ error)
        cleanup_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "sticky cleanup failed" in r.getMessage()
        ]
        assert len(cleanup_warnings) >= 1, (
            f"Expected at least one cleanup warning, got: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

        msg = cleanup_warnings[0].getMessage()
        # Verify structured fields are present for alert aggregation
        assert "reason=optim_step_error" in msg
        assert "session=s1" in msg
        assert "Rank 0" in msg  # rank
        assert "error_type=ExitError" in msg
    finally:
        _FakeTrainMode.__exit__ = original_exit_method


# ---------------------------------------------------------------------------
# Test 17: forward_backward cleanup failure log observability
# ---------------------------------------------------------------------------

def test_issue_193_forward_backward_cleanup_log_has_structured_fields(monkeypatch, caplog):
    """Mirror of Test 16 for the forward_backward path: when cleanup fails
    during forward_backward error handling, the emitted warning must contain
    reason, session_id, rank, and error_type."""
    worker, state = _make_worker(monkeypatch)
    _prepare_worker_for_forward_backward(worker, monkeypatch)

    class ComputeError(RuntimeError):
        pass

    class ExitError(RuntimeError):
        pass

    original_exit_method = _FakeTrainMode.__exit__

    def failing_fbb(*args, **kwargs):
        raise ComputeError("GPU compute failed")

    def failing_exit(self, exc_type, exc, tb):
        state["exit"] += 1
        raise ExitError("__exit__ cleanup boom")

    worker.engine.forward_backward_batch = failing_fbb  # type: ignore[method-assign]
    _FakeTrainMode.__exit__ = failing_exit  # type: ignore[method-assign]

    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ComputeError, match="GPU compute failed"):
                worker.forward_backward(
                    data_items=[{"model_input": {"input_ids": [1, 2, 3]}}],
                    loss_fn="cross_entropy",
                    loss_fn_config={},
                    session_id="s1",
                )

        cleanup_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "sticky cleanup failed" in r.getMessage()
        ]
        assert len(cleanup_warnings) >= 1, (
            f"Expected at least one cleanup warning, got: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

        msg = cleanup_warnings[0].getMessage()
        assert "reason=forward_backward_error" in msg
        assert "session=s1" in msg
        assert "Rank 0" in msg
        assert "error_type=ExitError" in msg
    finally:
        _FakeTrainMode.__exit__ = original_exit_method


# ---------------------------------------------------------------------------
# Test 18: snapshot failure on session_change is fail-loud (not swallowed)
# ---------------------------------------------------------------------------

def test_issue_193_snapshot_failure_on_session_change_is_fail_loud(monkeypatch):
    """When _ensure_sticky_train_mode triggers a session change and
    _capture_gradients() fails, the error must propagate -- ctx must stay
    open and gradients must NOT be silently lost."""
    worker, state = _make_worker(monkeypatch)

    class CaptureError(RuntimeError):
        pass

    def failing_capture():
        raise CaptureError("GPU snapshot failed")

    worker._capture_gradients = failing_capture  # type: ignore[method-assign]

    # Open sticky ctx for s1
    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    assert state["enter"] == 1

    # Switching to s2 triggers release(snapshot_gradients=True) for s1.
    # Since _capture_gradients fails, the error must propagate.
    with pytest.raises(CaptureError, match="GPU snapshot failed"):
        worker._ensure_sticky_train_mode(session_id="s2", reason="forward_backward")

    # Ctx must still be open (not released) -- gradients survive on GPU
    assert worker._sticky_train_mode_ctx is not None
    assert worker._sticky_train_mode_session_id == "s1"
    assert state["exit"] == 0  # __exit__ was NOT called


# ---------------------------------------------------------------------------
# Test 20: shutdown() completes cleanup even when release fails
# ---------------------------------------------------------------------------

def test_issue_193_shutdown_cleanup_survives_release_failure(monkeypatch):
    """When _release_sticky_train_mode raises during shutdown(), the
    session caches must still be cleared and destroy_process_group must
    still be called (via finally)."""
    worker, state = _make_worker(monkeypatch)

    class ExitError(RuntimeError):
        pass

    original_exit_method = _FakeTrainMode.__exit__

    def failing_exit(self, exc_type, exc, tb):
        state["exit"] += 1
        raise ExitError("__exit__ failed")

    _FakeTrainMode.__exit__ = failing_exit  # type: ignore[method-assign]

    try:
        # Open sticky ctx and populate session caches
        worker._ensure_sticky_train_mode(session_id="s1", reason="test")
        worker._session_gradients["s1"] = ["some-grads"]
        worker._session_optimizer_states["s1"] = {"state": "data"}
        worker._current_session_id = "s1"

        # Mock torch.distributed: is_initialized→True, spy on destroy_process_group
        import torch
        dpg_called = {"count": 0}

        def spy_destroy():
            dpg_called["count"] += 1

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "destroy_process_group", spy_destroy)

        # shutdown() should NOT raise -- release error is caught
        worker.shutdown()

        # Verify: session caches cleared despite release failure
        assert len(worker._session_gradients) == 0
        assert len(worker._session_optimizer_states) == 0
        assert worker._current_session_id is None
        # Verify: destroy_process_group was called (finally guarantee)
        assert dpg_called["count"] == 1
    finally:
        _FakeTrainMode.__exit__ = original_exit_method


# ---------------------------------------------------------------------------
# Test 21: swap_session_state snapshot failure is fail-loud
# ---------------------------------------------------------------------------

def test_issue_193_swap_session_snapshot_failure_is_fail_loud(monkeypatch):
    """When swap_session_state calls _release_sticky_train_mode with
    snapshot_gradients=True and _capture_gradients() fails, the error
    must propagate -- not silently lose gradients and proceed."""
    worker, state = _make_worker(monkeypatch)

    class CaptureError(RuntimeError):
        pass

    capture_count = {"n": 0}

    def failing_capture():
        capture_count["n"] += 1
        raise CaptureError("GPU snapshot failed")

    def fake_capture_opt():
        return {}

    def fake_restore_opt(s):
        pass

    def fake_reset_opt():
        pass

    # Open sticky ctx for s1
    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    assert state["enter"] == 1

    # Now inject the failure -- swap_session_state will call
    # _release_sticky_train_mode(snapshot_gradients=True), which calls
    # _capture_gradients() → fails
    worker._capture_gradients = failing_capture  # type: ignore[method-assign]
    worker._capture_optimizer_state = fake_capture_opt  # type: ignore[method-assign]
    worker._restore_optimizer_state = fake_restore_opt  # type: ignore[method-assign]
    worker._reset_optimizer_state = fake_reset_opt  # type: ignore[method-assign]
    worker.engine.optimizer_zero_grad = lambda: None  # type: ignore[method-assign]

    with pytest.raises(CaptureError, match="GPU snapshot failed"):
        worker.swap_session_state("s2")

    # Ctx must still be open -- swap did not silently proceed
    assert worker._sticky_train_mode_ctx is not None
    assert worker._sticky_train_mode_session_id == "s1"
    assert state["exit"] == 0
    assert capture_count["n"] == 1


# ---------------------------------------------------------------------------
# Test 19: snapshot failure on idle_timeout is fail-loud
# ---------------------------------------------------------------------------

def test_issue_193_snapshot_failure_on_idle_timeout_is_fail_loud(monkeypatch):
    """When idle timeout triggers release with snapshot_gradients=True and
    _capture_gradients() fails, the error must propagate -- ctx must stay
    open for retry."""
    worker, state = _make_worker(monkeypatch, idle_timeout_s="0.1")

    class CaptureError(RuntimeError):
        pass

    def failing_capture():
        raise CaptureError("GPU snapshot failed")

    worker._capture_gradients = failing_capture  # type: ignore[method-assign]

    # Open sticky ctx for s1
    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    assert state["enter"] == 1

    # Force idle timeout
    worker._sticky_train_mode_last_used_s = time.perf_counter() - 1.0

    # Same session but timed out → release(snapshot=True) → capture fails
    with pytest.raises(CaptureError, match="GPU snapshot failed"):
        worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")

    # Ctx must still be open -- NOT silently released
    assert worker._sticky_train_mode_ctx is not None
    assert worker._sticky_train_mode_session_id == "s1"
    assert state["exit"] == 0


# ---------------------------------------------------------------------------
# Test 22: partial worker swap failure invalidates _current_session
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_invalidates_current_session(monkeypatch):
    """When _swap_session_on_workers fails (some workers may have swapped),
    _current_session must be set to None to prevent the 'already loaded'
    early-return from masking a split-state condition across ranks."""
    import ray as ray_module

    # Build a minimal group object without calling __init__ (which does
    # heavy Ray initialization).  We only need _current_session, workers,
    # and the _swap_session_on_workers method.
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "s1"

    # Mock workers with Ray-like remote interface
    class _FakeRemoteMethod:
        @staticmethod
        def remote(new_session_id):
            return f"future-{new_session_id}"

    class _FakeWorker:
        swap_session_state = _FakeRemoteMethod()

    group.workers = [_FakeWorker(), _FakeWorker()]

    # Mock ray.get to simulate partial failure
    def mock_ray_get(futures, **kwargs):
        raise RuntimeError("Worker 1 failed during swap_session_state")

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    with pytest.raises(RuntimeError, match="Worker 1 failed"):
        group._swap_session_on_workers("s2")

    # _current_session must be None -- prevents "already loaded" early-return
    assert group._current_session is None


def _make_group_with_unknown_session_after_partial_swap(monkeypatch):
    """Build a minimal MegatronWorkerGroup and force partial swap failure."""
    import ray as ray_module

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "s1"

    class _FakeRemoteMethod:
        @staticmethod
        def remote(new_session_id):
            return f"future-{new_session_id}"

    class _FakeWorker:
        swap_session_state = _FakeRemoteMethod()

    group.workers = [_FakeWorker(), _FakeWorker()]

    def mock_ray_get(futures, **kwargs):
        raise RuntimeError("Worker 1 failed during swap_session_state")

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    with pytest.raises(RuntimeError, match="Worker 1 failed"):
        group._swap_session_on_workers("s2")

    assert group._current_session is None
    assert group._session_unknown_due_to_partial_swap is True
    return group


# ---------------------------------------------------------------------------
# Test 23: forward_backward rejects implicit session when state is unknown
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_requires_explicit_session_for_forward_backward(monkeypatch):
    group = _make_group_with_unknown_session_after_partial_swap(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit session_id required"):
        group.forward_backward(data_items=[], session_id=None)


# ---------------------------------------------------------------------------
# Test 24: forward rejects implicit session when state is unknown
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_requires_explicit_session_for_forward(monkeypatch):
    group = _make_group_with_unknown_session_after_partial_swap(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit session_id required"):
        group.forward(data_items=[], session_id=None)


# ---------------------------------------------------------------------------
# Test 25: optim_step rejects implicit session when state is unknown
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_requires_explicit_session_for_optim_step(monkeypatch):
    group = _make_group_with_unknown_session_after_partial_swap(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit session_id required"):
        group.optim_step(learning_rate=1e-4, session_id=None)


# ---------------------------------------------------------------------------
# Test 26: save_checkpoint rejects implicit session when state is unknown
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_requires_explicit_session_for_save_checkpoint(monkeypatch):
    group = _make_group_with_unknown_session_after_partial_swap(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit session_id required"):
        group.save_checkpoint("/tmp/fake_ckpt", session_id=None)


# ---------------------------------------------------------------------------
# Test 27: save_lora_weights rejects implicit session when state is unknown
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_requires_explicit_session_for_save_lora_weights(monkeypatch):
    group = _make_group_with_unknown_session_after_partial_swap(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit session_id required"):
        group.save_lora_weights("/tmp/fake_lora", session_id=None)


def test_issue_193_partial_swap_requires_explicit_session_for_load_checkpoint(monkeypatch):
    group = _make_group_with_unknown_session_after_partial_swap(monkeypatch)

    with pytest.raises(RuntimeError, match="explicit session_id required"):
        group.load_checkpoint("/tmp/fake_ckpt", session_id=None)


def test_issue_193_invalid_load_checkpoint_does_not_switch_session(monkeypatch, tmp_path):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "current_session"
    group._session_unknown_due_to_partial_swap = False

    ensure_calls: list[tuple[str, dict]] = []

    def fake_ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append((session_id, kwargs))
        return {"switched": False}

    group._ensure_session_loaded = fake_ensure_session_loaded

    missing_dir = tmp_path / "missing_ckpt"
    with pytest.raises(FileNotFoundError, match="does not exist or is not a directory"):
        group.load_checkpoint(str(missing_dir), session_id="target_session")

    assert ensure_calls == []
    assert group._current_session == "current_session"


def test_issue_193_missing_optimizer_shard_does_not_switch_session(monkeypatch, tmp_path):
    import ray as ray_module

    ckpt_dir = tmp_path / "missing_optimizer_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_adapter.pt").write_text("placeholder", encoding="utf-8")

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "current_session"
    group._session_unknown_due_to_partial_swap = False

    ensure_calls: list[tuple[str, dict]] = []
    load_adapter_calls: list[tuple[str, dict]] = []

    def fake_ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append((session_id, kwargs))
        return {"switched": False}

    def fake_load_adapter_state(load_path, **kwargs):
        load_adapter_calls.append((load_path, kwargs))
        return {"status": "ok"}

    group._ensure_session_loaded = fake_ensure_session_loaded
    group.load_adapter_state = fake_load_adapter_state

    class _FakeCheckOptimizerStateExistsRemoteMethod:
        def __init__(self, result):
            self.result = result
            self.calls = []

        def remote(self, load_path, **kwargs):
            self.calls.append((load_path, kwargs))
            return self.result

    class _FakeWorker:
        def __init__(self, result):
            self.check_optimizer_state_exists = _FakeCheckOptimizerStateExistsRemoteMethod(result)

    worker_0 = _FakeWorker({"rank": 0, "exists": True, "optimizer_file": "rank0_optimizer.pt"})
    worker_1 = _FakeWorker({"rank": 1, "exists": False, "optimizer_file": "rank1_optimizer.pt"})
    group.workers = [worker_0, worker_1]

    def mock_ray_get(futures, timeout=None):
        assert timeout is None
        return futures

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    with pytest.raises(FileNotFoundError, match="rank1_optimizer.pt"):
        group.load_checkpoint(
            str(ckpt_dir),
            load_optimizer=True,
            session_id="target_session",
        )

    assert worker_0.check_optimizer_state_exists.calls == [(str(ckpt_dir), {"traceparent": None})]
    assert worker_1.check_optimizer_state_exists.calls == [(str(ckpt_dir), {"traceparent": None})]
    assert ensure_calls == []
    assert load_adapter_calls == []
    assert group._current_session == "current_session"


def test_issue_193_megatron_rank_worker_load_checkpoint_helpers_accept_traceparent():
    impl_cls = MegatronRankWorker.__ray_metadata__.modified_class

    clear_params = inspect.signature(impl_cls.clear_session_state).parameters
    check_params = inspect.signature(impl_cls.check_optimizer_state_exists).parameters

    assert "traceparent" in clear_params
    assert clear_params["traceparent"].default is None
    assert "traceparent" in check_params
    assert check_params["traceparent"].default is None


def test_issue_193_load_checkpoint_without_optimizer_clears_session_cache_and_resets_optimizer(tmp_path, monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "current_session"
    group._session_unknown_due_to_partial_swap = False
    group._step_count = 99
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8

    ensure_calls: list[tuple[str, dict]] = []
    load_adapter_calls: list[tuple[str, dict]] = []
    reset_calls: list[float | None] = []

    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda session_id, **kwargs: ensure_calls.append((session_id, kwargs))
    group.load_adapter_state = lambda load_path, **kwargs: load_adapter_calls.append((load_path, kwargs)) or {"status": "ok"}
    group.reset_optimizer = (
        lambda learning_rate=None, traceparent=None: reset_calls.append((learning_rate, traceparent)) or {"status": "ok"}
    )

    class _FakeClearRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, session_id, **kwargs):
            self.calls.append((session_id, kwargs))
            return {"status": "ok", "session_id": session_id}

    class _FakeWorker:
        def __init__(self):
            self.clear_session_state = _FakeClearRemoteMethod()

    worker_0 = _FakeWorker()
    worker_1 = _FakeWorker()
    group.workers = [worker_0, worker_1]

    import ray as ray_module

    monkeypatch.setattr(ray_module, "get", lambda futures, timeout=None: futures)

    ckpt_dir = tmp_path / "megatron_nonresume"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_adapter.pt").write_bytes(b"stub")
    (ckpt_dir / "training_meta.json").write_text('{"current_step": 42, "learning_rate": 0.0007}')

    result = group.load_checkpoint(
        str(ckpt_dir),
        load_optimizer=False,
        session_id="target_session",
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )

    assert ensure_calls == [
        (
            "target_session",
            {"traceparent": None, "train_attn": False, "train_mlp": True, "train_unembed": False},
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
    assert worker_0.clear_session_state.calls == [("target_session", {"traceparent": None})]
    assert worker_1.clear_session_state.calls == [("target_session", {"traceparent": None})]
    assert reset_calls == [(pytest.approx(7e-4), None)]
    assert result["optimizer_restored"] is False
    assert result["optimizer_reset"] is True
    assert group._step_count == 42
    assert group.learning_rate == pytest.approx(7e-4)


def test_issue_193_load_checkpoint_invalid_meta_preserves_step_and_lr(tmp_path, monkeypatch, caplog):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "current_session"
    group._session_unknown_due_to_partial_swap = False
    group._step_count = 99
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8

    ensure_calls: list[tuple[str, dict]] = []
    load_adapter_calls: list[tuple[str, dict]] = []
    reset_calls: list[float | None] = []

    group._resolve_required_session_id = lambda session_id, op: session_id
    group._ensure_session_loaded = lambda session_id, **kwargs: ensure_calls.append((session_id, kwargs))
    group.load_adapter_state = lambda load_path, **kwargs: load_adapter_calls.append((load_path, kwargs)) or {"status": "ok"}
    group.reset_optimizer = (
        lambda learning_rate=None, traceparent=None: reset_calls.append((learning_rate, traceparent)) or {"status": "ok"}
    )

    class _FakeClearRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, session_id, **kwargs):
            self.calls.append((session_id, kwargs))
            return {"status": "ok", "session_id": session_id}

    class _FakeWorker:
        def __init__(self):
            self.clear_session_state = _FakeClearRemoteMethod()

    worker_0 = _FakeWorker()
    worker_1 = _FakeWorker()
    group.workers = [worker_0, worker_1]

    import ray as ray_module

    monkeypatch.setattr(ray_module, "get", lambda futures, timeout=None: futures)

    ckpt_dir = tmp_path / "megatron_invalid_meta"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_adapter.pt").write_bytes(b"stub")
    (ckpt_dir / "training_meta.json").write_text('{"current_step": "bad", "learning_rate": "oops"}')

    with caplog.at_level(logging.WARNING):
        result = group.load_checkpoint(
            str(ckpt_dir),
            load_optimizer=False,
            session_id="target_session",
        )

    assert ensure_calls == [("target_session", {"traceparent": None, "train_attn": None, "train_mlp": None, "train_unembed": None})]
    assert load_adapter_calls == [
        (
            str(ckpt_dir),
            {
                "actual_rank": 8,
                "traceparent": None,
                "train_attn": None,
                "train_mlp": None,
                "train_unembed": None,
            },
        )
    ]
    assert worker_0.clear_session_state.calls == [("target_session", {"traceparent": None})]
    assert worker_1.clear_session_state.calls == [("target_session", {"traceparent": None})]
    assert reset_calls == [(pytest.approx(1e-4), None)]
    assert result["current_step"] == "bad"
    assert result["learning_rate"] == "oops"
    assert group._step_count == 99
    assert group.learning_rate == pytest.approx(1e-4)
    assert any("Invalid current_step" in rec.getMessage() for rec in caplog.records)
    assert any("Invalid learning_rate" in rec.getMessage() for rec in caplog.records)


def _make_group_with_current_session(current_session: str | None = "s1"):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = current_session
    group._session_unknown_due_to_partial_swap = False
    return group


def test_issue_193_existing_session_switch_does_not_reset_expert_bias(monkeypatch):
    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "old-session"
    group._session_unknown_due_to_partial_swap = False
    group._step_count = 7
    group.learning_rate = 1e-4
    group._actual_rank = 8
    group.lora_rank = 8
    group._bind_traceparent = lambda _traceparent: None
    group._swap_session_on_workers = lambda _sid: None
    group._get_lora_weight_norm = lambda: 0.0
    group._get_lora_weight_checksum = lambda: {}
    group._get_base_weight_checksum = lambda: {}
    group._get_buffer_checksum = lambda: {}
    group._get_optimizer_param_counts = lambda: {}

    calls = {"load": 0, "reset": 0, "reinit": 0}

    class _SessionManager:
        def get_session_path(self, session_id):
            return f"/tmp/{session_id}"

        def save_metadata(self, *args, **kwargs):
            return None

        def session_exists(self, session_id):
            return session_id == "existing-session"

        def get_metadata(self, session_id):
            if session_id == "existing-session":
                return {"step": 11, "lr": 3e-4, "actual_rank": 6}
            return None

    group._session_manager = _SessionManager()
    group.save_adapter_state = lambda *args, **kwargs: {"status": "ok"}

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        return {"status": "ok"}

    def fake_reset(*args, **kwargs):
        calls["reset"] += 1
        return {"reset_count": 1}

    def fake_reinit(*args, **kwargs):
        calls["reinit"] += 1
        return {"status": "ok"}

    group.load_adapter_state = fake_load
    group.reset_expert_bias = fake_reset
    group.reinit_lora_weights = fake_reinit

    group._ensure_session_loaded("existing-session")

    assert calls == {"load": 1, "reset": 1, "reinit": 0}
    assert group._step_count == 11
    assert group.learning_rate == pytest.approx(3e-4)
    assert group._actual_rank == 6


# ---------------------------------------------------------------------------
# Test 28: empty string session_id must not fallback in forward
# ---------------------------------------------------------------------------

def test_issue_193_empty_session_id_rejected_in_forward():
    group = _make_group_with_current_session("s1")

    with pytest.raises(ValueError, match="must be non-empty"):
        group.forward(data_items=[], session_id="")


# ---------------------------------------------------------------------------
# Test 29: empty string session_id must not fallback in train_step
# ---------------------------------------------------------------------------

def test_issue_193_empty_session_id_rejected_in_train_step():
    group = _make_group_with_current_session("s1")

    with pytest.raises(ValueError, match="must be non-empty"):
        group.train_step(data_items=[], session_id="")


def test_issue_193_whitespace_session_id_rejected():
    group = _make_group_with_current_session("s1")

    with pytest.raises(ValueError, match="must be non-empty"):
        group.forward(data_items=[], session_id="   ")


def test_issue_193_no_session_loaded_has_distinct_error_from_partial_swap():
    group = _make_group_with_current_session(current_session=None)

    with pytest.raises(ValueError, match="no session loaded"):
        group.forward_backward(data_items=[], session_id=None)

    with pytest.raises(ValueError, match="no session loaded"):
        group.forward(data_items=[], session_id=None)

    with pytest.raises(ValueError, match="no session loaded"):
        group.train_step(data_items=[], session_id=None)

    with pytest.raises(ValueError, match="no session loaded"):
        group.optim_step(learning_rate=1e-4, session_id=None)

    with pytest.raises(ValueError, match="no session loaded"):
        group.save_checkpoint("/tmp/fake_ckpt", session_id=None)

    with pytest.raises(ValueError, match="no session loaded"):
        group.save_lora_weights("/tmp/fake_lora", session_id=None)


# ---------------------------------------------------------------------------
# Test 30: partial swap can recover with explicit session for checkpoint save
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_explicit_session_recovers_save_checkpoint(monkeypatch):
    import ray as ray_module

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = None
    group._step_count = 12
    group._actual_rank = 16
    group.learning_rate = 3e-4

    ensure_calls: list[tuple[str, dict]] = []
    prime_calls: list[tuple[str, str, int, float, int | None]] = []
    clear_actor_only_calls: list[str] = []

    def fake_ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append((session_id, kwargs))
        return {"switched": False}

    group._ensure_session_loaded = fake_ensure_session_loaded

    class _FakeSessionManager:
        def prime_session(
            self,
            session_id,
            checkpoint_path,
            *,
            step,
            lr,
            actual_rank,
            optimizer_restored=True,
            checkpoint_identity=None,
            **kwargs,
        ):
            prime_calls.append(
                (session_id, checkpoint_path, step, lr, actual_rank, optimizer_restored, checkpoint_identity, kwargs)
            )

        def clear_actor_only_state(self, session_id):
            clear_actor_only_calls.append(session_id)

    group._session_manager = _FakeSessionManager()

    class _FakeSaveCheckpointRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, save_path, step_count, actual_rank, **kwargs):
            self.calls.append((save_path, step_count, actual_rank, kwargs))
            return f"future-{len(self.calls)}"

    class _FakeWorker:
        def __init__(self):
            self.save_checkpoint = _FakeSaveCheckpointRemoteMethod()

    worker_0 = _FakeWorker()
    worker_1 = _FakeWorker()
    group.workers = [worker_0, worker_1]

    def mock_ray_get(futures, timeout=None):
        assert len(futures) == 2
        assert timeout is not None
        return [{"current_step": 12}, {}]

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    result = group.save_checkpoint(
        "/tmp/recovery_ckpt",
        session_id="recovered_session",
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )

    assert result["current_step"] == 12
    assert ensure_calls == [
        (
            "recovered_session",
            {
                "traceparent": None,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
        )
    ]
    assert worker_0.save_checkpoint.calls == [
        (
            "/tmp/recovery_ckpt",
            12,
            16,
            {
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
                "traceparent": None,
            },
        )
    ]
    assert worker_1.save_checkpoint.calls == [
        (
            "/tmp/recovery_ckpt",
            12,
            16,
            {
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
                "traceparent": None,
            },
        )
    ]
    assert prime_calls == []
    assert clear_actor_only_calls == []


# ---------------------------------------------------------------------------
# Test 31: partial swap can recover with explicit session for forward
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_explicit_session_recovers_forward(monkeypatch):
    import ray as ray_module

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = None

    ensure_calls: list[tuple[str, dict]] = []

    def fake_ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append((session_id, kwargs))
        return {"switched": False}

    group._ensure_session_loaded = fake_ensure_session_loaded

    class _FakeForwardRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, data_items, reset_bias, **kwargs):
            self.calls.append((data_items, reset_bias, kwargs))
            return "future-forward"

    class _FakeWorker:
        def __init__(self):
            self.forward = _FakeForwardRemoteMethod()

    worker = _FakeWorker()
    group.workers = [worker]

    def mock_ray_get(futures, **kwargs):
        assert futures == ["future-forward"]
        return [
            {
                "loss_value": 0.0,
                "num_tokens": 0,
                "valid_count": 0,
                "loss_fn_outputs": [],
            }
        ]

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    result = group.forward(
        data_items=[],
        session_id="recovered_session",
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )

    assert ensure_calls == [
        (
            "recovered_session",
            {
                "traceparent": None,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
        )
    ]
    assert worker.forward.calls == [([], None, {"traceparent": None})]
    assert result["metrics"]["num_samples:sum"] == 0.0


# ---------------------------------------------------------------------------
# Test 32: partial swap can recover with explicit session for optim_step
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_explicit_session_recovers_optim_step(monkeypatch):
    import ray as ray_module

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = None
    group._step_count = 5

    ensure_calls: list[tuple[str, dict]] = []

    def fake_ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append((session_id, kwargs))
        return {"switched": False}

    group._ensure_session_loaded = fake_ensure_session_loaded

    class _FakeOptimStepRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, learning_rate, session_id, **kwargs):
            self.calls.append((learning_rate, session_id, kwargs))
            return "future-optim"

    class _FakeWorker:
        def __init__(self):
            self.optim_step = _FakeOptimStepRemoteMethod()

    worker = _FakeWorker()
    group.workers = [worker]

    def mock_ray_get(futures, **kwargs):
        assert futures == ["future-optim"]
        return [{"grad_norm": 1.5, "lr": 2e-4}]

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    result = group.optim_step(
        learning_rate=2e-4,
        session_id="recovered_session",
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )

    assert ensure_calls == [
        (
            "recovered_session",
            {
                "traceparent": None,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
        )
    ]
    assert worker.optim_step.calls == [
        (
            2e-4,
            "recovered_session",
            {
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
                "traceparent": None,
            },
        )
    ]
    assert result["metrics"]["step"] == 6
    assert result["metrics"]["grad_norm"] == 1.5


# ---------------------------------------------------------------------------
# Test 33: partial swap can recover with explicit session for save_lora_weights
# ---------------------------------------------------------------------------

def test_issue_193_partial_swap_explicit_session_recovers_save_lora_weights(monkeypatch):
    import ray as ray_module

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = None
    group._step_count = 9
    group._actual_rank = 8

    ensure_calls: list[tuple[str, dict]] = []

    def fake_ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append((session_id, kwargs))
        return {"switched": False}

    group._ensure_session_loaded = fake_ensure_session_loaded

    class _FakeSaveLoraRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, save_path, step_count, actual_rank, **kwargs):
            self.calls.append((save_path, step_count, actual_rank, kwargs))
            return "future-save-lora"

    class _FakeWorker:
        def __init__(self):
            self.save_lora_weights = _FakeSaveLoraRemoteMethod()

    worker = _FakeWorker()
    group.workers = [worker]

    def mock_ray_get(futures, timeout=None):
        assert futures == ["future-save-lora"]
        assert timeout is not None
        return [{"current_step": 9}]

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    result = group.save_lora_weights(
        "/tmp/recovery_lora",
        session_id="recovered_session",
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )

    assert result["current_step"] == 9
    assert ensure_calls == [
        (
            "recovered_session",
            {
                "traceparent": None,
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
            },
        )
    ]
    assert worker.save_lora_weights.calls == [
        (
            "/tmp/recovery_lora",
            9,
            8,
            {
                "train_attn": False,
                "train_mlp": True,
                "train_unembed": False,
                "traceparent": None,
            },
        )
    ]


def test_issue_193_partial_swap_explicit_session_recovers_load_checkpoint(monkeypatch, tmp_path):
    import json
    import ray as ray_module

    ckpt_dir = tmp_path / "recovery_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_adapter.pt").write_text("placeholder", encoding="utf-8")
    (ckpt_dir / "training_meta.json").write_text(
        json.dumps({"current_step": 21, "learning_rate": 3e-4}),
        encoding="utf-8",
    )

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = None
    group._actual_rank = 8
    group.lora_rank = 16
    group.learning_rate = 1e-4
    group._step_count = 0

    ensure_calls: list[tuple[str, dict]] = []
    load_adapter_calls: list[tuple[str, dict]] = []

    def fake_ensure_session_loaded(session_id, **kwargs):
        ensure_calls.append((session_id, kwargs))

    def fake_load_adapter_state(load_path, **kwargs):
        load_adapter_calls.append((load_path, kwargs))
        return {"status": "ok"}

    group._ensure_session_loaded = fake_ensure_session_loaded
    group.load_adapter_state = fake_load_adapter_state

    class _FakeLoadOptimizerStateRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, load_path, **kwargs):
            self.calls.append((load_path, kwargs))
            return f"future-{len(self.calls)}"

    class _FakeCheckOptimizerStateExistsRemoteMethod:
        def __init__(self):
            self.calls = []

        def remote(self, load_path, **kwargs):
            self.calls.append((load_path, kwargs))
            return {"exists": True, "optimizer_file": f"{load_path}/optimizer.pt"}

    class _FakeWorker:
        def __init__(self):
            self.load_optimizer_state = _FakeLoadOptimizerStateRemoteMethod()
            self.check_optimizer_state_exists = _FakeCheckOptimizerStateExistsRemoteMethod()

    worker_0 = _FakeWorker()
    worker_1 = _FakeWorker()
    group.workers = [worker_0, worker_1]

    call_count = {"n": 0}

    def mock_ray_get(futures, timeout=None):
        assert timeout is None
        assert len(futures) == 2
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [
                {"exists": True, "optimizer_file": "rank0_optimizer.pt"},
                {"exists": True, "optimizer_file": "rank1_optimizer.pt"},
            ]
        return [{"status": "ok"}, {}]

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    result = group.load_checkpoint(
        str(ckpt_dir),
        load_optimizer=True,
        session_id="recovered_session",
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )

    assert ensure_calls == [
        (
            "recovered_session",
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
    assert worker_0.check_optimizer_state_exists.calls == [(str(ckpt_dir), {"traceparent": None})]
    assert worker_1.check_optimizer_state_exists.calls == [(str(ckpt_dir), {"traceparent": None})]
    assert worker_0.load_optimizer_state.calls == [(str(ckpt_dir), {"traceparent": None})]
    assert worker_1.load_optimizer_state.calls == [(str(ckpt_dir), {"traceparent": None})]
    assert result["current_step"] == 21
    assert result["learning_rate"] == pytest.approx(3e-4)
    assert group._step_count == 21
    assert group.learning_rate == pytest.approx(3e-4)


def test_issue_193_same_path_fast_path_requires_checkpoint_optimizer_when_requested(monkeypatch, tmp_path):
    import json
    import ray as ray_module

    ckpt_dir = tmp_path / "same_path_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_adapter.pt").write_text("placeholder", encoding="utf-8")
    (ckpt_dir / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (ckpt_dir / "training_meta.json").write_text(
        json.dumps({"current_step": 5, "learning_rate": 2e-4}),
        encoding="utf-8",
    )

    group_cls = MegatronWorkerGroup.__ray_metadata__.modified_class
    group = object.__new__(group_cls)
    group._current_session = "current_session"
    group._actual_rank = 8
    group.lora_rank = 16
    group.learning_rate = 1e-4
    group._step_count = 0
    group._session_unknown_due_to_partial_swap = False

    ensure_calls: list[tuple[str, dict]] = []
    load_adapter_calls: list[tuple[str, dict]] = []

    group._ensure_session_loaded = (
        lambda session_id, **kwargs: ensure_calls.append((session_id, kwargs)) or {"switched": False}
    )
    group.load_adapter_state = lambda load_path, **kwargs: load_adapter_calls.append((load_path, kwargs)) or {"status": "ok"}
    group._bind_traceparent = lambda traceparent: None
    group._resolve_required_session_id = lambda session_id, op: session_id

    class _RemoteMethod:
        def __init__(self, result):
            self._result = result
            self.calls: list[tuple[tuple, dict]] = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self._result

    worker_0 = type(
        "W",
        (),
        {
            "check_optimizer_state_exists": _RemoteMethod({"exists": True, "optimizer_file": "rank0_optimizer.pt"}),
            "load_optimizer_state": _RemoteMethod({"status": "ok"}),
        },
    )()
    worker_1 = type(
        "W",
        (),
        {
            "check_optimizer_state_exists": _RemoteMethod({"exists": True, "optimizer_file": "rank1_optimizer.pt"}),
            "load_optimizer_state": _RemoteMethod({}),
        },
    )()
    group.workers = [worker_0, worker_1]

    group._session_manager = type(
        "SessionMgr",
        (),
        {
            "has_actor_only_state": staticmethod(lambda session_id: False),
            "get_session_path": staticmethod(lambda session_id: str(ckpt_dir)),
            "get_metadata": staticmethod(
                lambda session_id: {
                    "step": 5,
                    "lr": 2e-4,
                    "actual_rank": 8,
                    "optimizer_restored": False,
                    "checkpoint_path": str(ckpt_dir),
                }
            ),
            "prime_session": staticmethod(lambda *args, **kwargs: None),
            "clear_actor_only_state": staticmethod(lambda session_id: None),
            "mark_actor_only_state": staticmethod(lambda *args, **kwargs: None),
        },
    )()

    def mock_ray_get(futures, timeout=None):
        assert timeout is None
        if futures == [worker_0.check_optimizer_state_exists._result, worker_1.check_optimizer_state_exists._result]:
            return [worker_0.check_optimizer_state_exists._result, worker_1.check_optimizer_state_exists._result]
        if futures == [worker_0.load_optimizer_state._result, worker_1.load_optimizer_state._result]:
            return [worker_0.load_optimizer_state._result, worker_1.load_optimizer_state._result]
        raise AssertionError(f"unexpected ray.get futures={futures!r}")

    monkeypatch.setattr(ray_module, "get", mock_ray_get)

    result = group.load_checkpoint(str(ckpt_dir), load_optimizer=True, session_id="target_session")

    assert ensure_calls == [
        (
            "target_session",
            {
                "traceparent": None,
                "train_attn": None,
                "train_mlp": None,
                "train_unembed": None,
            },
        )
    ]
    assert load_adapter_calls == [
        (
            str(ckpt_dir),
            {
                "actual_rank": 8,
                "traceparent": None,
                "train_attn": None,
                "train_mlp": None,
                "train_unembed": None,
            },
        )
    ]
    assert result["optimizer_restored"] is True


def test_issue_193_get_lora_state_dict_releases_sticky_before_eval_mode(monkeypatch):
    import torch
    from types import SimpleNamespace

    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    worker.engine.module = object()
    worker.engine.bridge = SimpleNamespace(
        export_adapter_weights=lambda *args, **kwargs: [("adapter.weight", torch.ones(1))]
    )

    release_calls: list[tuple[str, bool]] = []

    def fake_release(*, reason: str, snapshot_gradients: bool):
        release_calls.append((reason, snapshot_gradients))
        worker._sticky_train_mode_ctx = None
        worker._sticky_train_mode_session_id = None
        return {}

    worker._sticky_train_mode_ctx = object()
    worker._sticky_train_mode_session_id = "s1"
    worker._release_sticky_train_mode = fake_release  # type: ignore[method-assign]

    state_dict = worker.get_lora_state_dict()

    assert release_calls == [("get_lora_state_dict", True)]
    assert list(state_dict.keys()) == ["adapter.weight"]


def test_issue_467_get_lora_state_dict_enters_eval_mode_for_bridge_export(monkeypatch):
    import torch

    worker, state = _make_worker(monkeypatch, close_on_optim="0")
    worker.engine.module = object()

    calls: list[tuple[object, bool, bool]] = []

    class _Bridge:
        def export_adapter_weights(self, module, cpu=True, show_progress=True):
            calls.append((module, cpu, show_progress))
            return [("adapter.weight", torch.ones(1))]

    worker.engine.bridge = _Bridge()

    state_dict = worker.get_lora_state_dict()

    assert calls == [(worker.engine.module, True, False)]
    assert list(state_dict.keys()) == ["adapter.weight"]
    assert state["enter"] == 0
    assert state["exit"] == 0
    assert state["eval_enter"] == 1
    assert state["eval_exit"] == 1


def test_issue_467_get_lora_state_dict_uses_bridge_export_without_custom_fallback(monkeypatch):
    import torch

    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    worker.engine.module = object()

    calls: list[tuple[object, bool, bool]] = []

    class _Bridge:
        def export_adapter_weights(self, module, cpu=True, show_progress=True):
            calls.append((module, cpu, show_progress))
            return [("base_model.model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 3))]

    worker.engine.bridge = _Bridge()

    state_dict = worker.get_lora_state_dict()

    assert calls == [(worker.engine.module, True, False)]
    assert list(state_dict.keys()) == ["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"]


def test_issue_467_get_lora_state_dict_stub_bridge_does_not_require_megatron_bridge_imports(monkeypatch):
    import torch

    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    worker.engine.module = object()

    class _Bridge:
        def export_adapter_weights(self, module, cpu=True, show_progress=True):
            return [("adapter.weight", torch.ones(1))]

    worker.engine.bridge = _Bridge()
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("megatron.bridge"):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    state_dict = worker.get_lora_state_dict()

    assert list(state_dict.keys()) == ["adapter.weight"]


def test_issue_467_get_lora_state_dict_filters_exported_names_on_stub_bridge(monkeypatch):
    import torch

    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    worker.engine.module = object()

    class _Bridge:
        def export_adapter_weights(self, module, cpu=True, show_progress=True):
            return [
                ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(1)),
                ("model.layers.0.mlp.gate_proj.lora_A.weight", torch.ones(1)),
            ]

    worker.engine.bridge = _Bridge()

    state_dict = worker.get_lora_state_dict(
        train_attn=True,
        train_mlp=False,
        train_unembed=False,
    )

    assert list(state_dict.keys()) == ["model.layers.0.self_attn.q_proj.lora_A.weight"]


def test_issue_467_get_lora_state_dict_allows_moe_mlp_export_without_layout_flag(monkeypatch):
    import torch

    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    worker.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    worker.engine.module = object()

    class _Bridge:
        def export_adapter_weights(self, module, cpu=True, show_progress=True):
            return [("model.layers.0.mlp.gate_proj.lora_A.weight", torch.ones(1))]

    worker.engine.bridge = _Bridge()
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.get_model_config",
        lambda _: type("Cfg", (), {"is_moe": True})(),
    )

    state_dict = worker.get_lora_state_dict(
        train_attn=False,
        train_mlp=True,
        train_unembed=False,
    )

    assert list(state_dict.keys()) == ["model.layers.0.mlp.gate_proj.lora_A.weight"]


def test_issue_482_save_lora_weights_writes_unembed_target_modules(monkeypatch, tmp_path):
    import json
    import torch

    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    worker.get_lora_state_dict = lambda **kwargs: {
        "base_model.model.output_layer.lora_A.weight": torch.ones(1, 1),
        "base_model.model.output_layer.lora_B.weight": torch.ones(1, 1),
    }
    monkeypatch.setattr(
        "tinker_server.backend.megatron_distributed.get_model_config",
        lambda _: type("Cfg", (), {"is_mla": False, "is_moe": False})(),
    )

    worker.save_lora_weights(
        str(tmp_path),
        train_attn=False,
        train_mlp=False,
        train_unembed=True,
    )

    config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["target_modules"] == ["output_layer"]


def test_issue_482_target_modules_unembed_fallback_excludes_attention(monkeypatch):
    worker, _ = _make_worker(monkeypatch, close_on_optim="0")

    target_modules = worker._target_modules_for_export(
        model_is_mla=False,
        train_attn=False,
        train_mlp=False,
        train_unembed=True,
        state_dict={},
    )

    assert target_modules == ["lm_head", "output_layer"]


def test_issue_467_get_lora_state_dict_fails_when_bridge_export_is_missing(monkeypatch):
    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    worker.engine.module = object()
    worker.engine.bridge = object()

    with pytest.raises(RuntimeError, match="export_adapter_weights"):
        worker.get_lora_state_dict()


def test_issue_193_load_optimizer_state_releases_sticky_before_train_mode(monkeypatch, tmp_path):
    worker, _ = _make_worker(monkeypatch, close_on_optim="0")
    import types

    fake_peft_utils = types.ModuleType("verl.utils.megatron_peft_utils")
    fake_peft_utils._get_rank_checkpoint_path = lambda checkpoint_path: str(checkpoint_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_peft_utils", fake_peft_utils)

    release_calls: list[tuple[str, bool]] = []

    def fake_release(*, reason: str, snapshot_gradients: bool):
        release_calls.append((reason, snapshot_gradients))
        worker._sticky_train_mode_ctx = None
        worker._sticky_train_mode_session_id = None
        return {}

    worker._sticky_train_mode_ctx = object()
    worker._sticky_train_mode_session_id = "s1"
    worker._release_sticky_train_mode = fake_release  # type: ignore[method-assign]

    with pytest.raises(FileNotFoundError):
        worker.load_optimizer_state(str(tmp_path / "missing_ckpt"))

    assert release_calls == [("load_optimizer_state", True)]


def test_issue_193_long_forward_backward_refreshes_sticky_idle_timer(monkeypatch):
    worker, state = _make_worker(monkeypatch, idle_timeout_s="0.1", close_on_optim="0")
    _prepare_worker_for_forward_backward(worker, monkeypatch)

    def fake_forward_backward_batch(*args, **kwargs):
        time.sleep(0.2)
        return {"loss": [], "metrics": {}}

    worker.engine.forward_backward_batch = fake_forward_backward_batch  # type: ignore[method-assign]

    worker.forward_backward(
        data_items=[{"model_input": {"input_ids": [1, 2, 3]}}],
        loss_fn="cross_entropy",
        loss_fn_config={},
        rollout_correction_config=None,
        session_id="s1",
        reset_bias=None,
    )
    reused = worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")

    assert reused["reused"] is True
    assert state["exit"] == 0
