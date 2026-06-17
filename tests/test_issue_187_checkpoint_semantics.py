import json
from pathlib import Path

import pytest

from mint_server.checkpoints.checkpoints import (
    checkpoint_has_optimizer_state,
    validate_checkpoint_dir,
    validate_checkpoint_load_contract,
    write_checkpoint_metadata,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_issue_187_training_checkpoint_allows_optimizer_restore(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_training"
    _touch(ckpt / "adapter_model.safetensors")
    _touch(ckpt / "optimizer.pt")
    write_checkpoint_metadata(
        str(ckpt),
        {
            "checkpoint_id": "ckpt_training",
            "owner_id": None,
            "model_id": "m1",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "step": 1,
            "checkpoint_type": "training",
            "optimizer_present": True,
            "backend": "dense",
            "type": "training",
        },
    )

    checkpoint_type, optimizer_present = validate_checkpoint_load_contract(
        str(ckpt), load_optimizer=True
    )
    assert checkpoint_type == "training"
    assert optimizer_present is True


def test_issue_187_orbax_training_checkpoint_reports_optimizer_state(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_orbax_training"
    _touch(ckpt / "1" / "train_state" / "_METADATA")

    assert checkpoint_has_optimizer_state(str(ckpt)) is True


def test_issue_187_orbax_training_checkpoint_allows_optimizer_restore(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_orbax_training"
    _touch(ckpt / "1" / "train_state" / "_METADATA")
    write_checkpoint_metadata(
        str(ckpt),
        {
            "checkpoint_id": "ckpt_orbax_training",
            "owner_id": None,
            "model_id": "m1",
            "model_name": "openpi/pi0-fast-libero-low-mem-finetune",
            "created_at": "2026-01-01T00:00:00Z",
            "step": 1,
            "checkpoint_type": "training",
            "backend": "openpi_fast",
            "type": "training",
        },
    )

    checkpoint_type, optimizer_present = validate_checkpoint_load_contract(
        str(ckpt), load_optimizer=True
    )
    assert checkpoint_type == "training"
    assert optimizer_present is True


def test_issue_187_openpi_training_checkpoint_dir_is_valid(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_openpi_training"
    _touch(ckpt / "1" / "params" / "_METADATA")
    _touch(ckpt / "1" / "assets" / "physical-intelligence" / "libero" / "norm_stats.json")
    _touch(ckpt / "1" / "train_state" / "_METADATA")

    validate_checkpoint_dir(str(ckpt), checkpoint_type="training")


def test_issue_187_openpi_pi05_training_checkpoint_dir_is_valid_without_sampler_assets(
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "ckpt_openpi_pi05_training"
    _touch(ckpt / "1" / "params" / "_METADATA")
    _touch(ckpt / "1" / "train_state" / "_METADATA")

    validate_checkpoint_dir(str(ckpt), checkpoint_type="training")


def test_issue_187_openpi_training_checkpoint_dir_rejects_empty_layout(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_openpi_training_incomplete"
    (ckpt / "1" / "params").mkdir(parents=True)
    (ckpt / "1" / "assets").mkdir()
    _touch(ckpt / "1" / "train_state" / "_METADATA")

    with pytest.raises(ValueError, match="Missing LoRA weights in extracted checkpoint"):
        validate_checkpoint_dir(str(ckpt), checkpoint_type="training")


def test_issue_187_openpi_pi05_training_checkpoint_dir_is_not_valid_sampler(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_openpi_pi05_sampler_incomplete"
    _touch(ckpt / "1" / "params" / "_METADATA")
    _touch(ckpt / "1" / "train_state" / "_METADATA")

    with pytest.raises(ValueError, match="Missing sampler weights in extracted checkpoint"):
        validate_checkpoint_dir(str(ckpt), checkpoint_type="sampler")


def test_issue_187_sampler_checkpoint_rejects_optimizer_restore(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_sampler"
    _touch(ckpt / "adapter_model.safetensors")
    write_checkpoint_metadata(
        str(ckpt),
        {
            "checkpoint_id": "ckpt_sampler",
            "owner_id": None,
            "model_id": "m1",
            "model_name": "Qwen/Qwen3-0.6B",
            "created_at": "2026-01-01T00:00:00Z",
            "step": 1,
            "checkpoint_type": "sampler",
            "optimizer_present": False,
            "backend": "dense",
            "type": "sampler",
        },
    )

    with pytest.raises(ValueError, match="checkpoint_type is not 'training'"):
        validate_checkpoint_load_contract(str(ckpt), load_optimizer=True)


def test_issue_187_sampler_checkpoint_allows_non_optimizer_load(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt_sampler"
    _touch(ckpt / "adapter_model.safetensors")
    (ckpt / "metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (ckpt / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "ckpt_sampler",
                "owner_id": None,
                "model_id": "m1",
                "model_name": "Qwen/Qwen3-0.6B",
                "created_at": "2026-01-01T00:00:00Z",
                "step": 1,
                "checkpoint_type": "sampler",
                "optimizer_present": False,
                "backend": "dense",
                "type": "sampler",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_type, optimizer_present = validate_checkpoint_load_contract(
        str(ckpt), load_optimizer=False
    )
    assert checkpoint_type == "sampler"
    assert optimizer_present is False
