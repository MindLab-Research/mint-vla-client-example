from pathlib import Path

from tinker_server.checkpoints import validate_sampler_checkpoint_for_sampling


def test_openpi_sampler_checkpoint_is_valid_without_adapter_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "openpi_sampler"
    (checkpoint_dir / "params").mkdir(parents=True)
    (checkpoint_dir / "assets").mkdir()

    validate_sampler_checkpoint_for_sampling(str(checkpoint_dir))
