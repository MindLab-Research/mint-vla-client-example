from __future__ import annotations

import os
from typing import Any


def ray_log_to_driver_enabled() -> bool:
    v = os.environ.get("MINT_RAY_LOG_TO_DRIVER", "").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def ray_log_to_driver_kwargs() -> dict[str, Any]:
    # Ray defaults log_to_driver=True in some contexts; explicitly set False unless enabled.
    return {"log_to_driver": ray_log_to_driver_enabled()}


def ray_client_working_dir() -> str | None:
    addr = os.environ.get("RAY_ADDRESS", "").strip()
    if not addr.startswith("ray://"):
        return None
    pfs_tinker_path = os.environ.get("PFS_TINKER_PATH", "").strip()
    return pfs_tinker_path or None


def init_ray(**kwargs: Any) -> Any:
    """Initialize Ray with optional log forwarding to driver.

    Adds log_to_driver=True when MINT_RAY_LOG_TO_DRIVER is enabled, unless explicitly
    set by the caller.
    """
    import ray

    addr = os.environ.get("RAY_ADDRESS", "").strip()
    if addr:
        current = kwargs.get("address")
        if current is None or current == "" or current == "auto":
            kwargs["address"] = addr

    working_dir = ray_client_working_dir()
    runtime_env = kwargs.get("runtime_env")
    if working_dir and (runtime_env is None or isinstance(runtime_env, dict)):
        payload = {} if runtime_env is None else dict(runtime_env)
        payload.setdefault("working_dir", working_dir)
        kwargs["runtime_env"] = payload

    for k, v in ray_log_to_driver_kwargs().items():
        kwargs.setdefault(k, v)

    return ray.init(**kwargs)
