from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mint_server.backend.openpi.openpi_fast_worker as fast_worker_module
import mint_server.backend.openpi.openpi_pi05_worker as pi05_worker_module


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
        _session_state_tree=lambda: {"marker": "fast-session-tree"},
        _state=SimpleNamespace(step=7),
        _rng={"seed": [11, 12]},
        _pending_grads={"grad": [0.5, 0.25]},
        _learning_rate=1e-4,
        _save_session_train_state_checkpoint=lambda path, state: (path, state),
    )

    result = fast_worker_module.OpenPIFastWorkerSession.save_session_state(
        fake_session,
        {"session_id": "session-a"},
    )

    assert calls["session_id"] == "session-a"
    assert calls["worker_module"] == "mint_server.backend.openpi.openpi_fast_worker"
    assert calls["runtime_signature"] == {"config_name": "pi0_fast_libero_low_mem_finetune"}
    assert calls["state"] == {"marker": "fast-session-tree"}
    assert calls["rng"] == {"seed": [11, 12]}
    assert calls["pending_grads"] == {"grad": [0.5, 0.25]}
    assert calls["learning_rate"] == 1e-4
    assert calls["current_step"] == 7
    assert calls["save_train_state_fn"] is fake_session._save_session_train_state_checkpoint
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
        _load_session_train_state_checkpoint=lambda path: path,
    )

    result = fast_worker_module.OpenPIFastWorkerSession.load_session_state(
        fake_session,
        {"session_id": "session-a"},
    )

    assert calls["session_id"] == "session-a"
    assert calls["expected_worker_module"] == "mint_server.backend.openpi.openpi_fast_worker"
    assert calls["expected_runtime_signature"] == {"config_name": "pi0_fast_libero_low_mem_finetune"}
    assert calls["load_train_state_fn"] is fake_session._load_session_train_state_checkpoint
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
        _session_state_tree=lambda: {"marker": "pi05-session-tree"},
        _state=SimpleNamespace(step=11),
        _rng={"seed": [1]},
        _pending_grads={"grad": [0.1]},
        _learning_rate=5e-4,
        _save_session_train_state_checkpoint=lambda path, state: (path, state),
        _load_session_train_state_checkpoint=lambda path: path,
    )

    save_result = pi05_worker_module.OpenPIPi05WorkerSession.save_session_state(
        fake_session,
        {"session_id": "session-b"},
    )
    load_result = pi05_worker_module.OpenPIPi05WorkerSession.load_session_state(
        fake_session,
        {"session_id": "session-b"},
    )

    assert save_calls["worker_module"] == "mint_server.backend.openpi.openpi_pi05_worker"
    assert save_calls["runtime_signature"] == {
        "config_name": "pi05_libero",
        "max_token_len": 48,
    }
    assert save_calls["state"] == {"marker": "pi05-session-tree"}
    assert save_calls["save_train_state_fn"] is fake_session._save_session_train_state_checkpoint
    assert load_calls["expected_worker_module"] == "mint_server.backend.openpi.openpi_pi05_worker"
    assert load_calls["expected_runtime_signature"] == {
        "config_name": "pi05_libero",
        "max_token_len": 48,
    }
    assert load_calls["load_train_state_fn"] is fake_session._load_session_train_state_checkpoint
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
            assert Path(checkpoint_path).is_dir()
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

        class _normalize:
            @staticmethod
            def save(path, norm_stats):
                calls["asset_save"] = {"path": path, "norm_stats": norm_stats}

    class _FakeDataLoader:
        @staticmethod
        def data_config():
            return SimpleNamespace(norm_stats={"mean": [0.0]}, asset_id="asset")

    fake_session = SimpleNamespace(
        _checkpoints=_FakeCheckpoints(),
        _data_loader=_FakeDataLoader(),
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
            assert Path(checkpoint_path).is_dir()
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


def test_openpi_fast_worker_sampler_checkpoint_omits_train_state(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def save(self, step, *, items):
            calls["save"] = {"step": step, "items": items}

        def wait_until_finished(self) -> None:
            calls["waited"] = True

        def close(self) -> None:
            calls["closed"] = True

    class _FakeCheckpoints:
        def initialize_checkpoint_dir(self, checkpoint_path, *, keep_period, overwrite, resume):
            assert Path(checkpoint_path).is_dir()
            calls["checkpoint_path"] = checkpoint_path
            calls["init"] = {
                "keep_period": keep_period,
                "overwrite": overwrite,
                "resume": resume,
            }
            return _FakeManager(), False

        @staticmethod
        def _split_params(state):
            return object(), {"params": state.params}

        class _normalize:
            @staticmethod
            def save(path, norm_stats):
                calls["asset_save"] = {"path": path, "norm_stats": norm_stats}

    class _FakeDataLoader:
        @staticmethod
        def data_config():
            return SimpleNamespace(norm_stats=None, asset_id=None)

    fake_session = SimpleNamespace(
        _checkpoints=_FakeCheckpoints(),
        _data_loader=_FakeDataLoader(),
    )
    state = SimpleNamespace(step=0, params={"w": 1})

    fast_worker_module.OpenPIFastWorkerSession._save_sampler_checkpoint(
        fake_session,
        tmp_path / "missing-parent" / "openpi-fast-sampler",
        state,
    )

    assert calls["save"]["step"] == 1
    assert calls["init"]["overwrite"] is True
    assert (tmp_path / "missing-parent").is_dir()
    assert (tmp_path / "missing-parent" / "openpi-fast-sampler").is_dir()
    assert set(calls["save"]["items"].keys()) == {"assets", "params"}
    assert calls["waited"] is True
    assert calls["closed"] is True


def test_openpi_fast_worker_sampler_checkpoint_copies_seed_assets(tmp_path) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def save(self, step, *, items):
            calls["save"] = {"step": step, "items": items}

        def wait_until_finished(self) -> None:
            calls["waited"] = True

        def close(self) -> None:
            calls["closed"] = True

    class _FakeCheckpoints:
        def initialize_checkpoint_dir(self, checkpoint_path, *, keep_period, overwrite, resume):
            assert Path(checkpoint_path).is_dir()
            calls["checkpoint_path"] = checkpoint_path
            calls["init"] = {
                "keep_period": keep_period,
                "overwrite": overwrite,
                "resume": resume,
            }
            return _FakeManager(), False

        class _normalize:
            @staticmethod
            def save(path, norm_stats):
                raise AssertionError("normalize.save should not be used when seed assets are available")

    class _FakeDataLoader:
        @staticmethod
        def data_config():
            return SimpleNamespace(norm_stats=None, asset_id=None)

    seed_assets_dir = tmp_path / "seed_assets"
    norm_stats = seed_assets_dir / "physical-intelligence" / "libero" / "norm_stats.json"
    norm_stats.parent.mkdir(parents=True, exist_ok=True)
    norm_stats.write_text("{}", encoding="utf-8")

    fake_session = SimpleNamespace(
        _checkpoints=_FakeCheckpoints(),
        _data_loader=_FakeDataLoader(),
        _seed_assets_dir=seed_assets_dir,
    )
    state = SimpleNamespace(step=0, params={"w": 1})

    fast_worker_module.OpenPIFastWorkerSession._save_sampler_checkpoint(
        fake_session,
        tmp_path / "openpi-fast-sampler-seed-assets",
        state,
    )

    assets_callback = calls["save"]["items"]["assets"]
    export_assets_dir = tmp_path / "exported_assets"
    assets_callback(export_assets_dir)

    assert (export_assets_dir / "physical-intelligence" / "libero" / "norm_stats.json").read_text(encoding="utf-8") == "{}"
    assert calls["init"]["overwrite"] is True
    assert calls["waited"] is True
    assert calls["closed"] is True


def test_openpi_pi05_worker_sampler_checkpoint_omits_train_state(tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class _FakeManager:
        def save(self, step, *, items):
            calls["save"] = {"step": step, "items": items}

        def wait_until_finished(self) -> None:
            calls["waited"] = True

        def close(self) -> None:
            calls["closed"] = True

    class _FakeCheckpoints:
        def initialize_checkpoint_dir(self, checkpoint_path, *, keep_period, overwrite, resume):
            assert Path(checkpoint_path).is_dir()
            calls["checkpoint_path"] = checkpoint_path
            calls["init"] = {
                "keep_period": keep_period,
                "overwrite": overwrite,
                "resume": resume,
            }
            return _FakeManager(), False

        @staticmethod
        def _split_params(state):
            return object(), {"params": state.params}

        class _normalize:
            @staticmethod
            def save(path, norm_stats):
                calls["asset_save"] = {"path": path, "norm_stats": norm_stats}

    class _FakeDataLoader:
        @staticmethod
        def data_config():
            return SimpleNamespace(norm_stats=None, asset_id=None)

    fake_session = SimpleNamespace(
        _checkpoints=_FakeCheckpoints(),
        _data_loader=_FakeDataLoader(),
    )
    state = SimpleNamespace(step=0, params={"w": 1})

    pi05_worker_module.OpenPIPi05WorkerSession._save_sampler_checkpoint(
        fake_session,
        tmp_path / "missing-parent" / "openpi-pi05-sampler",
        state,
    )

    assert calls["save"]["step"] == 1
    assert calls["init"]["overwrite"] is True
    assert (tmp_path / "missing-parent").is_dir()
    assert (tmp_path / "missing-parent" / "openpi-pi05-sampler").is_dir()
    assert set(calls["save"]["items"].keys()) == {"assets", "params"}
    assert calls["waited"] is True
    assert calls["closed"] is True


def test_openpi_fast_worker_sampler_export_flattens_policy_checkpoint(tmp_path: Path) -> None:
    def _fake_save_sampler_checkpoint(path: Path, state) -> None:
        _ = state
        (path / "7" / "params").mkdir(parents=True)
        (path / "7" / "assets").mkdir(parents=True)
        (path / "7" / "params" / "_METADATA").write_text("params", encoding="utf-8")
        (path / "7" / "assets" / "asset.json").write_text("{}", encoding="utf-8")

    fake_session = SimpleNamespace(_save_sampler_checkpoint=_fake_save_sampler_checkpoint)
    export_dir = fast_worker_module.OpenPIFastWorkerSession._save_sampler_export(
        fake_session,
        tmp_path / "policy-export",
        SimpleNamespace(step=7),
    )

    assert export_dir == tmp_path / "policy-export"
    assert (export_dir / "params" / "_METADATA").read_text(encoding="utf-8") == "params"
    assert (export_dir / "assets" / "asset.json").read_text(encoding="utf-8") == "{}"
    assert not list(tmp_path.glob(".openpi_fast_sampler_export_*"))


def test_openpi_pi05_worker_sampler_export_flattens_policy_checkpoint(tmp_path: Path) -> None:
    def _fake_save_sampler_checkpoint(path: Path, state) -> None:
        _ = state
        (path / "3" / "params").mkdir(parents=True)
        (path / "3" / "assets").mkdir(parents=True)
        (path / "3" / "params" / "_METADATA").write_text("params", encoding="utf-8")
        (path / "3" / "assets" / "asset.json").write_text("{}", encoding="utf-8")

    fake_session = SimpleNamespace(_save_sampler_checkpoint=_fake_save_sampler_checkpoint)
    export_dir = pi05_worker_module.OpenPIPi05WorkerSession._save_sampler_export(
        fake_session,
        tmp_path / "policy-export",
        SimpleNamespace(step=3),
    )

    assert export_dir == tmp_path / "policy-export"
    assert (export_dir / "params" / "_METADATA").read_text(encoding="utf-8") == "params"
    assert (export_dir / "assets" / "asset.json").read_text(encoding="utf-8") == "{}"
    assert not list(tmp_path.glob(".openpi_pi05_sampler_export_*"))
