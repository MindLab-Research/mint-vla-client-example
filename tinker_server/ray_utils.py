from __future__ import annotations

import os
from typing import Any


def ray_log_to_driver_enabled() -> bool:
    v = os.environ.get("MINT_RAY_LOG_TO_DRIVER", "").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def ray_log_to_driver_kwargs() -> dict[str, Any]:
    return {"log_to_driver": True} if ray_log_to_driver_enabled() else {}


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

    for k, v in ray_log_to_driver_kwargs().items():
        kwargs.setdefault(k, v)

    return ray.init(**kwargs)
