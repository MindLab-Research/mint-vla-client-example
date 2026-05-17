#!/usr/bin/env python3
"""Watchdog for prod tinker-server-auth (server/ops side).

`/api/v1/healthz` is the public API-worker readiness endpoint and should remain
cheap. Trust it directly instead of falling back to root-path liveness.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _probe(url: str, timeout_s: float) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = int(resp.status)
            return 200 <= status < 300, f"http_{status}"
    except urllib.error.HTTPError as e:
        return False, f"http_{int(e.code)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _restart(program: str) -> int:
    r = subprocess.run(
        ["supervisorctl", "restart", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(f"{_ts()} watchdog: supervisorctl restart {program}: rc={r.returncode}", flush=True)
    if r.stdout:
        print(r.stdout.rstrip(), flush=True)
    return r.returncode


def main() -> int:
    url = os.environ.get("MINT_WATCHDOG_URL", "http://localhost:18000/api/v1/healthz")
    timeout_s = float(os.environ.get("MINT_WATCHDOG_TIMEOUT_S", "30"))
    interval_s = float(os.environ.get("MINT_WATCHDOG_INTERVAL_S", "60"))
    fails_to_restart = int(os.environ.get("MINT_WATCHDOG_FAILS_TO_RESTART", "6"))
    restart_cooldown_s = float(os.environ.get("MINT_WATCHDOG_RESTART_COOLDOWN_S", "60"))
    program = os.environ.get("MINT_WATCHDOG_PROGRAM", "tinker-server-auth")

    fails = 0

    # Give server time to boot after container start / deploy.
    boot_grace_s = float(os.environ.get("MINT_WATCHDOG_BOOT_GRACE_S", "30"))
    print(
        f"{_ts()} watchdog: starting url={url} "
        f"timeout_s={timeout_s} interval_s={interval_s} "
        f"fails_to_restart={fails_to_restart} restart_cooldown_s={restart_cooldown_s} boot_grace_s={boot_grace_s}",
        flush=True,
    )
    time.sleep(boot_grace_s)

    while True:
        ok, reason = _probe(url, timeout_s)
        if ok:
            fails = 0
            time.sleep(interval_s)
            continue

        fails += 1
        print(
            f"{_ts()} watchdog: failed probe {fails}/{fails_to_restart} (healthz={reason})",
            flush=True,
        )
        if fails < fails_to_restart:
            time.sleep(interval_s)
            continue

        print(
            f"{_ts()} watchdog: health check failed {fails} times; restarting {program}",
            flush=True,
        )
        _restart(program)
        fails = 0
        time.sleep(restart_cooldown_s)


if __name__ == "__main__":
    raise SystemExit(main())
