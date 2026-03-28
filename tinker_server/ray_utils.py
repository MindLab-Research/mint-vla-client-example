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


def preferred_driver_ray_address() -> str:
    for name in ("MINT_RAY_CLIENT_ADDRESS", "RAY_CLIENT_ADDRESS", "RAY_ADDRESS"):
        addr = os.environ.get(name, "").strip()
        if addr:
            return addr
    raise MissingRayAddressError(
        "RAY_ADDRESS must be set before initializing Ray "
        "(optionally set MINT_RAY_CLIENT_ADDRESS or RAY_CLIENT_ADDRESS for the driver)"
    )


def _driver_runtime_env() -> dict[str, Any]:
    runtime_env: dict[str, Any] = {}

    working_dir = (
        os.environ.get("MINT_RAY_WORKING_DIR", "").strip()
        or os.environ.get("PFS_TINKER_PATH", "").strip()
    )
    if working_dir:
        runtime_env["working_dir"] = working_dir

    py_modules_csv = os.environ.get("MINT_RAY_PY_MODULES_CSV", "").strip()
    if py_modules_csv:
        runtime_env["py_modules"] = [x.strip() for x in py_modules_csv.split(",") if x.strip()]

    return runtime_env


def init_ray(**kwargs: Any) -> Any:
    """Initialize Ray with optional log forwarding to driver.

    Adds log_to_driver=True when MINT_RAY_LOG_TO_DRIVER is enabled, unless explicitly
    set by the caller.
    """
    import ray

    current = kwargs.get("address")
    if current is None or current == "" or current == "auto":
        kwargs["address"] = preferred_driver_ray_address()
    elif isinstance(current, str):
        kwargs["address"] = current.strip()

    address = kwargs.get("address")
    if isinstance(address, str) and address.startswith("ray://") and "runtime_env" not in kwargs:
        runtime_env = _driver_runtime_env()
        if runtime_env:
            kwargs["runtime_env"] = runtime_env

    node_ip = os.environ.get("MINT_RAY_NODE_IP_ADDRESS", "").strip()
    if node_ip and "_node_ip_address" not in kwargs:
        kwargs["_node_ip_address"] = node_ip

    temp_dir = os.environ.get("MINT_RAY_TEMP_DIR", "").strip()
    if temp_dir and "_temp_dir" not in kwargs:
        kwargs["_temp_dir"] = temp_dir

    for k, v in ray_log_to_driver_kwargs().items():
        kwargs.setdefault(k, v)

    return ray.init(**kwargs)
