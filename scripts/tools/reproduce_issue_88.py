import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoints_py = repo_root / "tinker_server/checkpoints.py"
    weights_py = repo_root / "tinker_server/routes/weights.py"
    training_py = repo_root / "tinker_server/routes/training.py"
    types_py = repo_root / "tinker_server/models/types.py"

    required = {
        checkpoints_py: [
            "def safe_extract_checkpoint_archive",
            "def validate_checkpoint_dir",
            "def resolve_checkpoint_id",
            "def resolve_checkpoint_uri",
            "Unsafe path in archive",
        ],
        weights_py: [
            '@router.post("/checkpoints/upload"',
            "def upload_checkpoint_archive",
            "safe_extract_checkpoint_archive",
            "validate_checkpoint_dir",
            "CheckpointUploadResponse",
            "path=checkpoint_id",
        ],
        training_py: [
            "def _resolve_state_path",
            "resolve_checkpoint_uri",
        ],
        types_py: [
            "class CheckpointUploadResponse",
            "checkpoint_id: str",
            "path: str",
        ],
    }

    missing: list[str] = []
    for path, needles in required.items():
        txt = path.read_text(encoding="utf-8")
        for s in needles:
            if s not in txt:
                missing.append(f"{path}: {s}")

    if missing:
        print("FAIL: missing expected strings:", file=sys.stderr)
        for m in missing:
            print(f"- {m}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
