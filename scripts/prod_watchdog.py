#!/usr/bin/env python3
"""Watchdog for prod tinker-server-auth.

If /api/v1/healthz is unresponsive or non-200 for N consecutive checks,
restart the supervisord program `tinker-server-auth`.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.request


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _check(url: str, timeout_s: float) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


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
    url = os.environ.get("TINKER_WATCHDOG_URL", "http://localhost:18000/api/v1/healthz")
    timeout_s = float(os.environ.get("TINKER_WATCHDOG_TIMEOUT_S", "10"))
    interval_s = float(os.environ.get("TINKER_WATCHDOG_INTERVAL_S", "10"))
    fails_to_restart = int(os.environ.get("TINKER_WATCHDOG_FAILS_TO_RESTART", "6"))
    restart_cooldown_s = float(os.environ.get("TINKER_WATCHDOG_RESTART_COOLDOWN_S", "60"))
    program = os.environ.get("TINKER_WATCHDOG_PROGRAM", "tinker-server-auth")

    fails = 0

    # Give server time to boot after container start / deploy.
    boot_grace_s = float(os.environ.get("TINKER_WATCHDOG_BOOT_GRACE_S", "30"))
    print(
        f"{_ts()} watchdog: starting url={url} timeout_s={timeout_s} interval_s={interval_s} "
        f"fails_to_restart={fails_to_restart} restart_cooldown_s={restart_cooldown_s} boot_grace_s={boot_grace_s}",
        flush=True,
    )
    time.sleep(boot_grace_s)

    while True:
        ok = _check(url, timeout_s)
        if ok:
            fails = 0
            time.sleep(interval_s)
            continue

        fails += 1
        if fails < fails_to_restart:
            time.sleep(interval_s)
            continue

        print(f"{_ts()} watchdog: health check failed {fails} times; restarting {program}", flush=True)
        _restart(program)
        fails = 0
        time.sleep(restart_cooldown_s)


if __name__ == "__main__":
    raise SystemExit(main())
