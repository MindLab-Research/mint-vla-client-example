import time

from tinker_server.backend.megatron_distributed import (
    DistributedConfig,
    MegatronRankWorker,
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


class _FakeEngine:
    def __init__(self, state: dict):
        self._state = state

    def train_mode(self):
        return _FakeTrainMode(self._state)


def _make_worker(
    monkeypatch,
    *,
    idle_timeout_s: str = "15",
    close_on_optim: str = "1",
) -> tuple[MegatronRankWorker, dict]:
    monkeypatch.setenv("MINT_MEGATRON_STICKY_TRAIN_MODE", "1")
    monkeypatch.setenv("MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S", idle_timeout_s)
    monkeypatch.setenv("MINT_MEGATRON_STICKY_CLOSE_ON_OPTIM", close_on_optim)
    state = {"enter": 0, "exit": 0}
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
# Test 7: Sticky ctx cleaned up on forward_backward error
# ---------------------------------------------------------------------------

def test_issue_193_sticky_cleanup_on_forward_backward_error(monkeypatch):
    """When forward_backward computation raises, the sticky ctx must be released
    so the next call gets a fresh enter (not reuse of a broken ctx)."""
    worker, state = _make_worker(monkeypatch)

    worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    assert state["enter"] == 1
    assert worker._sticky_train_mode_ctx is not None

    # Simulate: forward_backward catches the error and releases ctx
    worker._release_sticky_train_mode(
        reason="forward_backward_error", snapshot_gradients=False
    )
    assert worker._sticky_train_mode_ctx is None
    assert state["exit"] == 1

    # Next call should do a fresh enter, not reuse
    result = worker._ensure_sticky_train_mode(session_id="s1", reason="forward_backward")
    assert result["reused"] is False
    assert state["enter"] == 2


# ---------------------------------------------------------------------------
# Test 8: Sticky ctx cleaned up on optim_step error
# ---------------------------------------------------------------------------

def test_issue_193_sticky_cleanup_on_optim_step_error(monkeypatch):
    """Same as test 7 but for optim_step error path."""
    worker, state = _make_worker(monkeypatch)

    worker._ensure_sticky_train_mode(session_id="s1", reason="optim_step")
    assert state["enter"] == 1

    worker._release_sticky_train_mode(
        reason="optim_step_error", snapshot_gradients=False
    )
    assert worker._sticky_train_mode_ctx is None
    assert state["exit"] == 1

    result = worker._ensure_sticky_train_mode(session_id="s1", reason="optim_step")
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

