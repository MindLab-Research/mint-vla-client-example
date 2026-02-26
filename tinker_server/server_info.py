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
    from .config import config as server_config
    return {
        "git_sha": _git_sha(),
        "started_at": STARTED_AT.isoformat(),
        "process": {"pid": os.getpid(), "argv": sys.argv},
        "logging": {"stdout": _fd_info(1), "stderr": _fd_info(2)},
        "config": {
            "config_path": server_config.config_path,
            "host": server_config.host,
            "port": server_config.port,
            "tensor_parallel_size": server_config.tensor_parallel_size,
            "data_parallel_size": server_config.data_parallel_size,
            "gpu_memory_utilization": server_config.gpu_memory_utilization,
            "max_model_len": server_config.max_model_len,
            "enable_multi_lora": server_config.enable_multi_lora,
            "max_loras": server_config.max_loras,
            "max_cpu_loras": server_config.max_cpu_loras,
            "max_lora_rank": server_config.max_lora_rank,
            "sampling_max_inflight_sample_tasks": server_config.sampling_max_inflight_sample_tasks,
            "sampling_max_concurrent_samples_per_request": server_config.sampling_max_concurrent_samples_per_request,
            "auth_enabled": server_config.auth_enabled,
            "router_replay_mode": server_config.router_replay_mode,
        },
    }
