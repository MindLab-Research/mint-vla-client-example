#!/usr/bin/env python3
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    ckpt_id = "ckpt_deadbeefcafe"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        owner = "user123"
        ckpt_dir = root / owner / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "metadata.json").write_text(f'{{"checkpoint_id": "{ckpt_id}"}}', encoding="utf-8")

        # Ensure module constant picks this up.
        os.environ["MINT_CHECKPOINT_DIR"] = str(root)

        from importlib import reload
        import mint_server.checkpoints.checkpoints as checkpoints

        reload(checkpoints)

        got = checkpoints.resolve_checkpoint_path(ckpt_id)
        if got != str(ckpt_dir):
            raise SystemExit(f"FAIL: resolve_checkpoint_path({ckpt_id!r}) returned {got!r}, want {str(ckpt_dir)!r}")

        got = checkpoints.resolve_checkpoint_path(f"mint://{owner}/{ckpt_id}")
        want = str(root / owner / ckpt_id)
        if got != want:
            raise SystemExit(f"FAIL: resolve_checkpoint_path(mint://...) returned {got!r}, want {want!r}")

        got = checkpoints.resolve_checkpoint_path("file:///tmp/x")
        if got != "/tmp/x":
            raise SystemExit(f"FAIL: resolve_checkpoint_path(file://...) returned {got!r}, want '/tmp/x'")

    print("PASS: checkpoint resolver handles ckpt_xxx and mint:// consistently")


if __name__ == "__main__":
    main()
