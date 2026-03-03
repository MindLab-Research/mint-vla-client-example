import time

from tinker_server.backend.megatron_distributed import DistributedConfig, MegatronRankWorker


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


def _make_worker(monkeypatch, *, idle_timeout_s: str = "15") -> tuple[MegatronRankWorker, dict]:
    monkeypatch.setenv("MINT_MEGATRON_STICKY_TRAIN_MODE", "1")
    monkeypatch.setenv("MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S", idle_timeout_s)
    monkeypatch.setenv("MINT_MEGATRON_STICKY_CLOSE_ON_OPTIM", "1")
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
