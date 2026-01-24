import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sampling_py = repo_root / "tinker_server/routes/sampling.py"
    utils_py = repo_root / "tinker_server/sampling_utils.py"

    required = {
        sampling_py: [
            "from ..sampling_utils import resolve_stop_reason",
            "resolve_stop_reason(",
            'getattr(result, "stop_reason", None)',
        ],
        utils_py: [
            "def resolve_stop_reason(",
            "DEFAULT_EOS_TOKENS",
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

