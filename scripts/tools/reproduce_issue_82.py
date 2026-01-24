import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "tinker_server/backend/multinode_inference.py"
    txt = src.read_text(encoding="utf-8")

    required = [
        "controller_gpus = 0",
        "controller_cpus = 0",
        "total_required_gpus = worker_gpus",
        "num_cpus=controller_cpus",
        "num_gpus=controller_gpus",
    ]
    missing = [s for s in required if s not in txt]
    if missing:
        print(f"FAIL: missing expected strings in {src}: {missing}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

