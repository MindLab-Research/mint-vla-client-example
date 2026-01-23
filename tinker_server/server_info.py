from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


STARTED_AT = datetime.now(timezone.utc)


def _fd_info(fd: int) -> dict:
    target = None
    append = None
    isatty = None

    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
    except Exception:
        pass

    try:
        isatty = os.isatty(fd)
    except Exception:
        pass

    try:
        import fcntl

        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        append = bool(flags & os.O_APPEND)
    except Exception:
        pass

    return {"fd": fd, "target": target, "append": append, "isatty": isatty}


def _find_git_root() -> Path | None:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / ".git").exists():
            return p
    return None


@lru_cache(maxsize=1)
def _git_sha() -> str | None:
    for k in ("TINKER_GIT_SHA", "MINT_GIT_SHA", "GIT_SHA"):
        v = os.environ.get(k)
        if v:
            return v

    root = _find_git_root()
    if root is None:
        return None

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return None


def get_server_info() -> dict:
    return {
        "git_sha": _git_sha(),
        "started_at": STARTED_AT.isoformat(),
        "process": {"pid": os.getpid(), "argv": sys.argv},
        "logging": {"stdout": _fd_info(1), "stderr": _fd_info(2)},
    }

