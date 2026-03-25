from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import tinker_server.backend.openpi_fast_worker as fast_worker_module
import tinker_server.backend.openpi_pi05_worker as pi05_worker_module


def test_openpi_fast_worker_save_session_state_uses_manager() -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def save_state(self, session_id: str, **kwargs):
            calls["session_id"] = session_id
            calls.update(kwargs)
            return Path("/tmp/session-a")

    fake_session = SimpleNamespace(
        _session_state_manager=_FakeManager(),
        _session_state_signature=lambda: {"config_name": "pi0_fast_libero_low_mem_finetune"},
        _state=SimpleNamespace(step=7),
        _rng={"seed": [11, 12]},
        _pending_grads={"grad": [0.5, 0.25]},
        _learning_rate=1e-4,
        _save_train_state_checkpoint=lambda path, state: (path, state),
    )

    result = fast_worker_module.OpenPIFastWorkerSession.save_session_state(
        fake_session,
        {"session_id": "session-a"},
    )

    assert calls["session_id"] == "session-a"
    assert calls["worker_module"] == "tinker_server.backend.openpi_fast_worker"
    assert calls["runtime_signature"] == {"config_name": "pi0_fast_libero_low_mem_finetune"}
    assert calls["state"] is fake_session._state
    assert calls["rng"] == {"seed": [11, 12]}
    assert calls["pending_grads"] == {"grad": [0.5, 0.25]}
    assert calls["learning_rate"] == 1e-4
    assert calls["current_step"] == 7
    assert callable(calls["save_train_state_fn"])
    assert result == {"path": "/tmp/session-a", "current_step": 7, "learning_rate": 1e-4}


def test_openpi_fast_worker_load_session_state_restores_aux_state() -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def load_state(self, session_id: str, **kwargs):
            calls["session_id"] = session_id
            calls.update(kwargs)
            return {
                "state": "new-state",
                "rng": {"seed": [99]},
                "pending_grads": {"grad": [0.1]},
                "learning_rate": 0.002,
                "current_step": 9,
            }

    fake_session = SimpleNamespace(
        _session_state_manager=_FakeManager(),
        _session_state_signature=lambda: {"config_name": "pi0_fast_libero_low_mem_finetune"},
        _state="old-state",
        _rng={"seed": [11, 12]},
        _pending_grads=None,
        _learning_rate=1e-4,
        _load_train_state_checkpoint=lambda path: path,
    )

    result = fast_worker_module.OpenPIFastWorkerSession.load_session_state(
        fake_session,
        {"session_id": "session-a"},
    )

    assert calls["session_id"] == "session-a"
    assert calls["expected_worker_module"] == "tinker_server.backend.openpi_fast_worker"
    assert calls["expected_runtime_signature"] == {"config_name": "pi0_fast_libero_low_mem_finetune"}
    assert callable(calls["load_train_state_fn"])
    assert fake_session._state == "new-state"
    assert fake_session._rng == {"seed": [99]}
    assert fake_session._pending_grads == {"grad": [0.1]}
    assert fake_session._learning_rate == 0.002
    assert result == {"current_step": 9, "learning_rate": 0.002}


def test_openpi_pi05_worker_save_and_load_session_state_use_manager() -> None:
    save_calls: dict[str, object] = {}
    load_calls: dict[str, object] = {}

    class _FakeManager:
        def save_state(self, session_id: str, **kwargs):
            save_calls["session_id"] = session_id
            save_calls.update(kwargs)
            return Path("/tmp/session-b")

        def load_state(self, session_id: str, **kwargs):
            load_calls["session_id"] = session_id
            load_calls.update(kwargs)
            return {
                "state": "pi05-state",
                "rng": {"seed": [5]},
                "pending_grads": None,
                "learning_rate": 0.003,
                "current_step": 13,
            }

    fake_session = SimpleNamespace(
        _session_state_manager=_FakeManager(),
        _session_state_signature=lambda: {
            "config_name": "pi05_libero",
            "max_token_len": 48,
        },
        _state=SimpleNamespace(step=11),
        _rng={"seed": [1]},
        _pending_grads={"grad": [0.1]},
        _learning_rate=5e-4,
        _save_train_state_checkpoint=lambda path, state: (path, state),
        _load_train_state_checkpoint=lambda path: path,
    )

    save_result = pi05_worker_module.OpenPIPi05WorkerSession.save_session_state(
        fake_session,
        {"session_id": "session-b"},
    )
    load_result = pi05_worker_module.OpenPIPi05WorkerSession.load_session_state(
        fake_session,
        {"session_id": "session-b"},
    )

    assert save_calls["worker_module"] == "tinker_server.backend.openpi_pi05_worker"
    assert save_calls["runtime_signature"] == {
        "config_name": "pi05_libero",
        "max_token_len": 48,
    }
    assert load_calls["expected_worker_module"] == "tinker_server.backend.openpi_pi05_worker"
    assert load_calls["expected_runtime_signature"] == {
        "config_name": "pi05_libero",
        "max_token_len": 48,
    }
    assert save_result == {"path": "/tmp/session-b", "current_step": 11, "learning_rate": 5e-4}
    assert load_result == {"current_step": 13, "learning_rate": 0.003}
    assert fake_session._state == "pi05-state"
    assert fake_session._rng == {"seed": [5]}
    assert fake_session._pending_grads is None
    assert fake_session._learning_rate == 0.003


def test_worker_dispatch_accepts_session_state_ops() -> None:
    fast_session = SimpleNamespace(
        save_session_state=lambda payload: {"path": payload["session_id"]},
        load_session_state=lambda payload: {"session_id": payload["session_id"]},
    )
    pi05_session = SimpleNamespace(
        save_session_state=lambda payload: {"path": payload["session_id"]},
        load_session_state=lambda payload: {"session_id": payload["session_id"]},
    )

    fast_save, fast_close = fast_worker_module._dispatch(
        fast_session,
        "save_session_state",
        {"session_id": "session-a"},
    )
    fast_load, _ = fast_worker_module._dispatch(
        fast_session,
        "load_session_state",
        {"session_id": "session-a"},
    )
    pi05_save, pi05_close = pi05_worker_module._dispatch(
        pi05_session,
        "save_session_state",
        {"session_id": "session-b"},
    )
    pi05_load, _ = pi05_worker_module._dispatch(
        pi05_session,
        "load_session_state",
        {"session_id": "session-b"},
    )

    assert fast_save == {"path": "session-a"}
    assert fast_load == {"session_id": "session-a"}
    assert fast_close is False
    assert pi05_save == {"path": "session-b"}
    assert pi05_load == {"session_id": "session-b"}
    assert pi05_close is False


def test_openpi_fast_worker_checkpoint_save_normalizes_step_zero() -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def wait_until_finished(self) -> None:
            calls["waited"] = True

        def close(self) -> None:
            calls["closed"] = True

    class _FakeCheckpoints:
        def initialize_checkpoint_dir(self, checkpoint_path, *, keep_period, overwrite, resume):
            calls["checkpoint_path"] = checkpoint_path
            calls["init"] = {
                "keep_period": keep_period,
                "overwrite": overwrite,
                "resume": resume,
            }
            return _FakeManager(), False

        def save_state(self, manager, state, data_loader, step):
            calls["save"] = {
                "manager": manager,
                "state": state,
                "data_loader": data_loader,
                "step": step,
            }

    fake_session = SimpleNamespace(
        _checkpoints=_FakeCheckpoints(),
        _data_loader="loader",
    )
    state = SimpleNamespace(step=0)

    fast_worker_module.OpenPIFastWorkerSession._save_train_state_checkpoint(
        fake_session,
        Path("/tmp/openpi-fast-session"),
        state,
    )

    assert calls["save"]["step"] == 1
    assert calls["waited"] is True
    assert calls["closed"] is True


def test_openpi_pi05_worker_checkpoint_save_normalizes_step_zero() -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def wait_until_finished(self) -> None:
            calls["waited"] = True

        def close(self) -> None:
            calls["closed"] = True

    class _FakeCheckpoints:
        def initialize_checkpoint_dir(self, checkpoint_path, *, keep_period, overwrite, resume):
            calls["checkpoint_path"] = checkpoint_path
            calls["init"] = {
                "keep_period": keep_period,
                "overwrite": overwrite,
                "resume": resume,
            }
            return _FakeManager(), False

        def save_state(self, manager, state, data_loader, step):
            calls["save"] = {
                "manager": manager,
                "state": state,
                "data_loader": data_loader,
                "step": step,
            }

    fake_session = SimpleNamespace(
        _checkpoints=_FakeCheckpoints(),
        _data_loader="loader",
    )
    state = SimpleNamespace(step=0)

    pi05_worker_module.OpenPIPi05WorkerSession._save_train_state_checkpoint(
        fake_session,
        Path("/tmp/openpi-pi05-session"),
        state,
    )

    assert calls["save"]["step"] == 1
    assert calls["waited"] is True
    assert calls["closed"] is True
