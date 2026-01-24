import sys
import time
from pathlib import Path

import asyncio

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tinker_server.backend.lora_registry import LoRARegistry  # noqa: E402


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


async def _run() -> None:
    reg = LoRARegistry()

    active = await reg.allocate("active")
    idle = await reg.allocate("idle")

    now = time.time()
    reg._slot_info[active].last_used = now
    reg._slot_info[idle].last_used = now - 60.0

    cands = await reg.get_lru_candidates(1)
    if cands != [idle]:
        raise RuntimeError(f"candidates={cands!r} expected {[idle]!r}")


def main() -> int:
    try:
        asyncio.run(_run())
    except Exception as e:
        return _fail(str(e))

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
