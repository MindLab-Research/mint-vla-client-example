from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger(__name__)

_RAY_INIT_THREAD_LOCK = threading.Lock()
_RAY_INIT_LOCK_PATH_ENV = "MINT_RAY_INIT_LOCK_PATH"
_RAY_INIT_LOCK_TIMEOUT_ENV = "MINT_RAY_INIT_LOCK_TIMEOUT_S"
_RAY_INIT_LOCK_POLL_ENV = "MINT_RAY_INIT_LOCK_POLL_S"


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


def _ray_init_lock_path() -> Path:
    raw = os.environ.get(_RAY_INIT_LOCK_PATH_ENV, "").strip()
    if raw:
        return Path(raw)
    return Path("/tmp/mint_ray_init.lock")


def _ray_init_lock_timeout_s() -> float:
    raw = os.environ.get(_RAY_INIT_LOCK_TIMEOUT_ENV, "").strip()
    if not raw:
        return 120.0
    try:
        return max(0.1, float(raw))
    except ValueError:
        logger.warning("Invalid %s=%r, using default", _RAY_INIT_LOCK_TIMEOUT_ENV, raw)
        return 120.0


def _ray_init_lock_poll_s() -> float:
    raw = os.environ.get(_RAY_INIT_LOCK_POLL_ENV, "").strip()
    if not raw:
        return 0.05
    try:
        return max(0.01, float(raw))
    except ValueError:
        logger.warning("Invalid %s=%r, using default", _RAY_INIT_LOCK_POLL_ENV, raw)
        return 0.05


@contextlib.contextmanager
def _ray_init_interprocess_lock() -> Iterator[float]:
    lock_path = _ray_init_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = open(lock_path, "a+", encoding="utf-8")
    timeout_s = _ray_init_lock_timeout_s()
    poll_s = _ray_init_lock_poll_s()
    wait_start = time.monotonic()

    try:
        while True:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                waited_s = time.monotonic() - wait_start
                if waited_s >= timeout_s:
                    raise TimeoutError(
                        "Timed out waiting for Ray init lock "
                        f"path={lock_path} waited_s={waited_s:.3f} timeout_s={timeout_s:.3f}"
                    ) from e
                time.sleep(poll_s)
        yield time.monotonic() - wait_start
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fh.close()


def init_ray(**kwargs: Any) -> Any:
    """Initialize Ray with optional log forwarding to driver.

    Adds log_to_driver=True when MINT_RAY_LOG_TO_DRIVER is enabled, unless explicitly
    set by the caller.
    """
    import ray

    if ray.is_initialized():
        return None

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

    with _RAY_INIT_THREAD_LOCK:
        if ray.is_initialized():
            return None
        with _ray_init_interprocess_lock() as lock_wait_s:
            if ray.is_initialized():
                return None
            pid = os.getpid()
            namespace = kwargs.get("namespace")
            t0 = time.monotonic()
            logger.info(
                "ray.init begin pid=%s address=%r namespace=%r lock_wait_s=%.3f",
                pid,
                kwargs.get("address"),
                namespace,
                lock_wait_s,
            )
            try:
                result = ray.init(**kwargs)
            except Exception:
                logger.exception(
                    "ray.init failed pid=%s address=%r namespace=%r lock_wait_s=%.3f elapsed_s=%.3f",
                    pid,
                    kwargs.get("address"),
                    namespace,
                    lock_wait_s,
                    time.monotonic() - t0,
                )
                raise
            logger.info(
                "ray.init ok pid=%s address=%r namespace=%r lock_wait_s=%.3f elapsed_s=%.3f",
                pid,
                kwargs.get("address"),
                namespace,
                lock_wait_s,
                time.monotonic() - t0,
            )
            return result
