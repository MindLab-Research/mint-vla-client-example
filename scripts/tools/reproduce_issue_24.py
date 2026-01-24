import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "tinker_server/routes/futures.py"
    txt = src.read_text(encoding="utf-8")

    required = [
        "if status == FutureStatus.PENDING:",
        "response.status_code = 408",
        'response.headers["Retry-After"] = "1"',
    ]
    missing = [s for s in required if s not in txt]
    if missing:
        print(f"FAIL: missing expected strings in {src}: {missing}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

