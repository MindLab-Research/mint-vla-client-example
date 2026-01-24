import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tinker_server.sampling_utils import resolve_stop_reason  # noqa: E402


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    cases = [
        ("stop", [1, 2, 3], "stop"),
        ("length", [1, 2, 3], "length"),
        ("eos", [1, 2, 3], "eos"),
        (None, [151645], "stop"),
        ("unknown", [151643], "stop"),
        (None, [42], "length"),
    ]

    for stop_reason, token_ids, expected in cases:
        got = resolve_stop_reason(stop_reason=stop_reason, token_ids=token_ids)
        if got != expected:
            return _fail(
                f"resolve_stop_reason(stop_reason={stop_reason!r}, token_ids={token_ids!r}) "
                f"returned {got!r} expected {expected!r}"
            )

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
