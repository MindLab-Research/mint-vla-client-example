import io
import json
import tarfile
from pathlib import Path

import pytest


def test_issue_88_safe_extract_strips_single_root(tmp_path: Path) -> None:
    from tinker_server import checkpoints

    archive_path = tmp_path / "ckpt.tar.gz"
    root = "ckpt_original"
    payload = {
        f"{root}/adapter_model.safetensors": b"dummy-lora",
        f"{root}/optimizer.pt": b"dummy-optimizer",
        f"{root}/training_meta.json": json.dumps({"current_step": 7}).encode("utf-8"),
    }

    with tarfile.open(archive_path, "w:gz") as tf:
        for name, data in payload.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    out_dir = tmp_path / "extracted"
    checkpoints.safe_extract_checkpoint_archive(str(archive_path), str(out_dir))
    checkpoints.validate_checkpoint_dir(str(out_dir))

    assert (out_dir / "adapter_model.safetensors").exists()
    assert (out_dir / "optimizer.pt").exists()
    assert (out_dir / "training_meta.json").exists()
    assert not (out_dir / root).exists()


def test_issue_88_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    from tinker_server import checkpoints

    archive_path = tmp_path / "evil.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        data = b"evil"
        info = tarfile.TarInfo(name="ckpt_x/../../evil.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    out_dir = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe path"):
        checkpoints.safe_extract_checkpoint_archive(str(archive_path), str(out_dir))


def test_issue_88_resolve_ckpt_id(tmp_path: Path) -> None:
    from tinker_server import checkpoints

    ckpt_id = "ckpt_123456789abc"
    ckpt_dir = tmp_path / "anonymous" / ckpt_id
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "metadata.json").write_text(json.dumps({"checkpoint_id": ckpt_id}), encoding="utf-8")

    assert checkpoints.resolve_checkpoint_uri(ckpt_id, str(tmp_path)) == str(ckpt_dir)


def test_issue_88_validate_checkpoint_dir_accepts_megatron_shards(tmp_path: Path) -> None:
    from tinker_server import checkpoints

    ckpt_dir = tmp_path / "ckpt_x"
    ckpt_dir.mkdir()
    (ckpt_dir / "mp_rank_00_adapter.pt").write_bytes(b"dummy-lora")
    (ckpt_dir / "mp_rank_00_optimizer.pt").write_bytes(b"dummy-optimizer")

    checkpoints.validate_checkpoint_dir(str(ckpt_dir))
