import os
import sys
from pathlib import Path

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _read_text(rel: str) -> str:
    return (_repo_root / rel).read_text(encoding="utf-8")


def main() -> int:
    # Keep the standard tool interface even though this is a local invariant check.
    _ = BASE_URL, API_KEY

    rel = "tinker_server/backend/multi_lora_engine.py"
    txt = _read_text(rel)

    required = (
        "def idle_time(self) -> float:",
        "return time.time() - self.last_used",
        'os.environ.get("TINKER_LORA_EVICT_MIN_IDLE_S", "5.0")',
        "slot = self._slot_info.get(lora_id)",
        "if slot.idle_time() <= min_idle_s:",
    )
    for needle in required:
        if needle not in txt:
            return _fail(f"{rel} missing {needle!r}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
