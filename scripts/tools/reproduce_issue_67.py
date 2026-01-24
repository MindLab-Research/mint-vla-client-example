import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "tinker_server/backend/megatron_distributed.py"
    txt = src.read_text(encoding="utf-8")

    required = [
        "def load_optimizer_state(",
        "_optimizer.pt",
        "torch.save(self._capture_optimizer_state(), optimizer_file)",
        "load_optimizer_state.remote",
        "optimizer_restored",
    ]
    missing = [s for s in required if s not in txt]
    if missing:
        print(f"FAIL: missing expected strings in {src}: {missing}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
