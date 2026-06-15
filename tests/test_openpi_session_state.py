from __future__ import annotations

from pathlib import Path

import pytest


def test_openpi_session_state_roundtrip_preserves_auxiliary_state(tmp_path: Path) -> None:
    from mint_server.backend.openpi.openpi_session_state import OpenPISessionStateManager

    manager = OpenPISessionStateManager(tmp_path)
    state_store: dict[str, object] = {}

    def _save_train_state(path: Path, state: object) -> None:
        state_store[str(path)] = state

    def _load_train_state(path: Path) -> object:
        return state_store[str(path)]

    manager.save_state(
        "session-a",
        worker_module="mint_server.backend.openpi.openpi_fast_worker",
        runtime_signature={"base_model": "openpi/pi0-fast-libero-low-mem-finetune"},
        state={"params": {"w": [1, 2, 3]}},
        rng={"seed": [11, 12]},
        pending_grads={"grad": [0.5, 0.25]},
        learning_rate=1e-4,
        current_step=7,
        save_train_state_fn=_save_train_state,
    )

    restored = manager.load_state(
        "session-a",
        expected_worker_module="mint_server.backend.openpi.openpi_fast_worker",
        expected_runtime_signature={"base_model": "openpi/pi0-fast-libero-low-mem-finetune"},
        load_train_state_fn=_load_train_state,
    )

    assert restored["state"] == {"params": {"w": [1, 2, 3]}}
    assert restored["rng"] == {"seed": [11, 12]}
    assert restored["pending_grads"] == {"grad": [0.5, 0.25]}
    assert restored["learning_rate"] == 1e-4
    assert restored["current_step"] == 7
    assert manager.session_exists("session-a") is True


def test_openpi_session_state_rejects_worker_module_mismatch(tmp_path: Path) -> None:
    from mint_server.backend.openpi.openpi_session_state import OpenPISessionStateManager

    manager = OpenPISessionStateManager(tmp_path)
    state_store: dict[str, object] = {}

    manager.save_state(
        "session-a",
        worker_module="mint_server.backend.openpi.openpi_fast_worker",
        runtime_signature={"base_model": "openpi/pi0-fast-libero-low-mem-finetune"},
        state={"params": {"w": [1, 2, 3]}},
        rng={"seed": [11, 12]},
        pending_grads=None,
        learning_rate=1e-4,
        current_step=7,
        save_train_state_fn=lambda path, state: state_store.__setitem__(str(path), state),
    )

    with pytest.raises(ValueError, match="worker_module"):
        manager.load_state(
            "session-a",
            expected_worker_module="mint_server.backend.openpi.openpi_pi05_worker",
            expected_runtime_signature={"base_model": "openpi/pi0-fast-libero-low-mem-finetune"},
            load_train_state_fn=lambda path: state_store[str(path)],
        )


def test_openpi_session_state_rejects_runtime_signature_mismatch(tmp_path: Path) -> None:
    from mint_server.backend.openpi.openpi_session_state import OpenPISessionStateManager

    manager = OpenPISessionStateManager(tmp_path)
    state_store: dict[str, object] = {}

    manager.save_state(
        "session-a",
        worker_module="mint_server.backend.openpi.openpi_fast_worker",
        runtime_signature={"base_model": "openpi/pi0-fast-libero-low-mem-finetune"},
        state={"params": {"w": [1, 2, 3]}},
        rng={"seed": [11, 12]},
        pending_grads=None,
        learning_rate=1e-4,
        current_step=7,
        save_train_state_fn=lambda path, state: state_store.__setitem__(str(path), state),
    )

    with pytest.raises(ValueError, match="runtime_signature"):
        manager.load_state(
            "session-a",
            expected_worker_module="mint_server.backend.openpi.openpi_fast_worker",
            expected_runtime_signature={"base_model": "openpi/pi05-libero-low-mem-finetune"},
            load_train_state_fn=lambda path: state_store[str(path)],
        )


def test_openpi_session_state_load_raises_for_missing_session(tmp_path: Path) -> None:
    from mint_server.backend.openpi.openpi_session_state import OpenPISessionStateManager

    manager = OpenPISessionStateManager(tmp_path)

    with pytest.raises(FileNotFoundError, match="session-missing"):
        manager.load_state(
            "session-missing",
            expected_worker_module="mint_server.backend.openpi.openpi_fast_worker",
            expected_runtime_signature={"base_model": "openpi/pi0-fast-libero-low-mem-finetune"},
            load_train_state_fn=lambda path: path,
        )
