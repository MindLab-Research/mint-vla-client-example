from pathlib import Path

import pytest

from tinker_server.checkpoints import validate_sampler_checkpoint_for_sampling


def test_openpi_sampler_checkpoint_is_valid_without_adapter_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "openpi_sampler"
    (checkpoint_dir / "params").mkdir(parents=True)
    (checkpoint_dir / "params" / "_METADATA").write_bytes(b"")
    assets_dir = checkpoint_dir / "assets" / "physical-intelligence" / "libero"
    assets_dir.mkdir(parents=True)
    (assets_dir / "norm_stats.json").write_text("{}", encoding="utf-8")

    validate_sampler_checkpoint_for_sampling(str(checkpoint_dir))


def test_openpi_sampler_checkpoint_requires_real_policy_artifacts(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "openpi_sampler_incomplete"
    (checkpoint_dir / "params").mkdir(parents=True)
    (checkpoint_dir / "assets").mkdir()

    with pytest.raises(ValueError, match="Missing sampling weights"):
        validate_sampler_checkpoint_for_sampling(str(checkpoint_dir))
