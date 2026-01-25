import os
import sys
from pathlib import Path

import requests

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
    # This issue is about log preservation in documented server start commands.
    _ = BASE_URL, API_KEY

    mint_dev = _read_text(".claude/skills/mint-dev/SKILL.md")
    if 'python scripts/run_server.py\\" > /tmp/tinker_server.log' in mint_dev:
        return _fail("mint-dev/SKILL.md still uses '>' redirection for /tmp/tinker_server.log (truncates logs)")
    if 'python scripts/run_server.py\\" >> /tmp/tinker_server.log' not in mint_dev:
        return _fail("mint-dev/SKILL.md missing '>> /tmp/tinker_server.log' redirection")

    auto_bugfix = _read_text(".claude/skills/auto-bugfix/SKILL.md")
    if 'python scripts/run_server.py\\" > /tmp/tinker_server_issue_$ISSUE.log' in auto_bugfix:
        return _fail("auto-bugfix/SKILL.md still uses '>' redirection for /tmp/tinker_server_issue_$ISSUE.log")
    if 'python scripts/run_server.py\\" >> /tmp/tinker_server_issue_$ISSUE.log' not in auto_bugfix:
        return _fail("auto-bugfix/SKILL.md missing '>> /tmp/tinker_server_issue_$ISSUE.log' redirection")

    # Regression guard: pid capture must not expand $! locally.
    if "echo \\\\$!" in auto_bugfix:
        return _fail("auto-bugfix/SKILL.md uses 'echo \\\\$!' (expands $! locally); expected 'echo \\$!'")
    if "echo \\$!" not in auto_bugfix:
        return _fail("auto-bugfix/SKILL.md missing 'echo \\$!' in pid capture")

    # zsh: $TINKER_PORT:localhost triggers :l modifier; braces required.
    if "ssh -f -N -L $TINKER_PORT:localhost:$TINKER_PORT volcano" in auto_bugfix:
        return _fail("auto-bugfix/SKILL.md uses unbraced $TINKER_PORT in -L (breaks in zsh)")
    if "ssh -f -N -L ${TINKER_PORT}:localhost:${TINKER_PORT} volcano" not in auto_bugfix:
        return _fail("auto-bugfix/SKILL.md missing braced ssh -L tunnel example for $TINKER_PORT")

    # Runtime check: server should expose server_info and report append-mode stdout/stderr.
    try:
        r = requests.get(
            f"{BASE_URL.rstrip('/')}/api/v1/server_info",
            headers={"X-API-Key": API_KEY},
            timeout=10,
        )
    except Exception as e:
        return _fail(f"GET /api/v1/server_info failed: {e}")

    if r.status_code != 200:
        return _fail(f"GET /api/v1/server_info returned {r.status_code}: {r.text[:200]!r}")

    try:
        info = r.json()
    except Exception as e:
        return _fail(f"/api/v1/server_info response is not JSON: {e}")

    try:
        stdout_append = bool(info["logging"]["stdout"]["append"])
        stderr_append = bool(info["logging"]["stderr"]["append"])
    except Exception as e:
        return _fail(f"/api/v1/server_info missing expected logging fields: {e}")

    if not stdout_append or not stderr_append:
        return _fail(f"server_info logging append=false (stdout={stdout_append}, stderr={stderr_append})")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
