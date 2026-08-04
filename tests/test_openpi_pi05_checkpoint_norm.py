from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mint_server.backend.openpi.openpi_pi05_worker import (
    OpenPIPi05RuntimeInitOverrides,
    OpenPIPi05WorkerSession,
)


def test_runtime_override_authenticates_checkpoint_norm_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    norm_dir = tmp_path / "norm"
    norm_dir.mkdir()
    (norm_dir / "norm_stats.json").write_text('{"norm_stats": {}}\n')
    monkeypatch.setenv("MINT_OPENPI_PI05_CHECKPOINT_NORM_STATS_DIR", str(norm_dir))
    assert OpenPIPi05RuntimeInitOverrides.from_env().checkpoint_norm_stats_dir == str(
        norm_dir.resolve()
    )
    monkeypatch.setenv(
        "MINT_OPENPI_PI05_CHECKPOINT_NORM_STATS_DIR", str(tmp_path / "missing")
    )
    with pytest.raises(FileNotFoundError, match="must contain norm_stats.json"):
        OpenPIPi05RuntimeInitOverrides.from_env()


def test_sampler_assets_copy_exact_explicit_norm(tmp_path: Path) -> None:
    source = tmp_path / "norm"
    source.mkdir()
    exact = b'{"norm_stats":{"state":{"mean":[1]}}}\n'
    (source / "norm_stats.json").write_bytes(exact)

    class Loader:
        @staticmethod
        def data_config():
            return SimpleNamespace(
                norm_stats=None, asset_id="physical-intelligence/libero"
            )

    session = SimpleNamespace(
        _data_loader=Loader(),
        _seed_assets_dir=None,
        _checkpoint_norm_stats_dir=source,
    )
    output = tmp_path / "assets"
    OpenPIPi05WorkerSession._save_checkpoint_assets(session, output)
    assert (
        output / "physical-intelligence/libero/norm_stats.json"
    ).read_bytes() == exact


def test_checkpoint_root_records_norm_sha_and_widths(tmp_path: Path) -> None:
    norm = tmp_path / "norm"
    norm.mkdir()
    path = norm / "norm_stats.json"
    path.write_text('{"norm_stats": {}}\n')
    session = SimpleNamespace(
        _profile=None,
        _checkpoint_norm_stats_dir=norm,
        _state_dim=54,
        _action_dim=32,
    )
    root = tmp_path / "checkpoint"
    root.mkdir()
    OpenPIPi05WorkerSession._write_profile_manifest(session, root)
    payload = json.loads((root / "mint_pi05_norm_provenance.json").read_text())
    assert payload == {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "state_dim": 54,
        "action_dim": 32,
        "profile_id": None,
    }
