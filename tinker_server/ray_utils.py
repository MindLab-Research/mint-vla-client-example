from __future__ import annotations

import os
from typing import Any


class MissingRayAddressError(RuntimeError):
    pass


def ray_log_to_driver_enabled() -> bool:
    v = os.environ.get("MINT_RAY_LOG_TO_DRIVER", "").strip().lower()
    return v not in {"", "0", "false", "no", "off"}


def ray_log_to_driver_kwargs() -> dict[str, Any]:
    # Ray defaults log_to_driver=True in some contexts; explicitly set False unless enabled.
    return {"log_to_driver": ray_log_to_driver_enabled()}


def require_ray_address() -> str:
    addr = os.environ.get("RAY_ADDRESS", "").strip()
    if not addr:
        raise MissingRayAddressError("RAY_ADDRESS must be set before initializing Ray")
    return addr


def init_ray(**kwargs: Any) -> Any:
    """Initialize Ray with optional log forwarding to driver.

    Adds log_to_driver=True when MINT_RAY_LOG_TO_DRIVER is enabled, unless explicitly
    set by the caller.
    """
    import ray

    current = kwargs.get("address")
    if current is None or current == "" or current == "auto":
        kwargs["address"] = require_ray_address()
    elif isinstance(current, str):
        kwargs["address"] = current.strip()

    node_ip = os.environ.get("MINT_RAY_NODE_IP_ADDRESS", "").strip()
    if node_ip and "_node_ip_address" not in kwargs:
        kwargs["_node_ip_address"] = node_ip

    temp_dir = os.environ.get("MINT_RAY_TEMP_DIR", "").strip()
    if temp_dir and "_temp_dir" not in kwargs:
        kwargs["_temp_dir"] = temp_dir

    for k, v in ray_log_to_driver_kwargs().items():
        kwargs.setdefault(k, v)

    return ray.init(**kwargs)
