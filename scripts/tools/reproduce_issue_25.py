import asyncio
import os
import sys
import time
from pathlib import Path

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


async def _run() -> int:
    # Issue #25: LoRARegistry.get_lru_candidates must not return recently-used ids.
    _ = BASE_URL, API_KEY

    from tinker_server.backend.multi_lora_engine import LoRARegistry

    os.environ.setdefault("TINKER_LORA_EVICT_MIN_IDLE_S", "5.0")
    min_idle_s = float(os.environ["TINKER_LORA_EVICT_MIN_IDLE_S"])

    reg = LoRARegistry()
    lora1 = await reg.allocate("session_1")
    lora2 = await reg.allocate("session_2")

    now = time.time()
    reg._slot_info[lora2].last_used = now
    reg._slot_info[lora1].last_used = now - (min_idle_s + 5.0)

    candidates = await reg.get_lru_candidates(2)
    if candidates != [lora1]:
        return _fail(
            f"expected candidates={[lora1]} with min_idle_s={min_idle_s}, got {candidates}"
        )

    print("PASS")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
