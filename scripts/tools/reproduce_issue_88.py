import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from tinker_server import checkpoints  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

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

        if not (out_dir / "adapter_model.safetensors").exists():
            return _fail("missing adapter_model.safetensors after extract")
        if not (out_dir / "optimizer.pt").exists():
            return _fail("missing optimizer.pt after extract")
        if not (out_dir / "training_meta.json").exists():
            return _fail("missing training_meta.json after extract")
        if (out_dir / root).exists():
            return _fail("archive root dir not stripped during extract")

        # checkpoint id resolution: ckpt_{id} -> {checkpoints_dir}/anonymous/{id}
        ckpt_id = "ckpt_123456789abc"
        ckpt_dir = tmp_path / "anonymous" / ckpt_id
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "metadata.json").write_text(
            json.dumps({"checkpoint_id": ckpt_id}), encoding="utf-8"
        )
        got = checkpoints.resolve_checkpoint_uri(ckpt_id, str(tmp_path))
        if got != str(ckpt_dir):
            return _fail(f"resolve_checkpoint_uri returned {got!r} expected {str(ckpt_dir)!r}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
