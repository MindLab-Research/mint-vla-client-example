from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator


logger = logging.getLogger(__name__)

_RAY_INIT_THREAD_LOCK = threading.Lock()
_RAY_INIT_LOCK_PATH_ENV = "MINT_RAY_INIT_LOCK_PATH"
_RAY_INIT_LOCK_TIMEOUT_ENV = "MINT_RAY_INIT_LOCK_TIMEOUT_S"
_RAY_INIT_LOCK_POLL_ENV = "MINT_RAY_INIT_LOCK_POLL_S"
_RAY_HEAD_ADDRESS_PATH_ENV = "MINT_RAY_HEAD_ADDRESS_PATH"
_RAY_RECONNECT_POLL_ENV = "MINT_RAY_RECONNECT_POLL_S"
_DEFAULT_RAY_GCS_PORT = 6379
_RAY_LAST_INIT_ADDRESS: str | None = None
_RAY_CONNECTION_EPOCH = 0
_RAY_RECONNECT_INVALIDATORS: list[Callable[[], None]] = []


class MissingRayAddressError(RuntimeError):
    pass


def _normalize_ray_address(addr: str) -> str:
    value = str(addr).strip()
    if not value:
        raise MissingRayAddressError("Ray address is empty")
    if "://" in value or ":" in value:
        return value
    return f"{value}:{_DEFAULT_RAY_GCS_PORT}"


def _configured_ray_head_address_path() -> Path | None:
    raw = os.environ.get(_RAY_HEAD_ADDRESS_PATH_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _read_configured_ray_head_address(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise MissingRayAddressError(
            f"{_RAY_HEAD_ADDRESS_PATH_ENV} points to unreadable file: {path}"
        ) from e
    if not raw:
        raise MissingRayAddressError(f"{_RAY_HEAD_ADDRESS_PATH_ENV} points to empty file: {path}")
    return _normalize_ray_address(raw)


def ray_address_source_configured() -> bool:
    if _configured_ray_head_address_path() is not None:
        return True
    return any(
        bool(os.environ.get(name, "").strip())
        for name in ("MINT_RAY_CLIENT_ADDRESS", "RAY_CLIENT_ADDRESS", "RAY_ADDRESS")
    )


def ray_reconnect_poll_s() -> float:
    raw = os.environ.get(_RAY_RECONNECT_POLL_ENV, "").strip()
    if not raw:
        return 5.0
    try:
        return max(0.1, float(raw))
    except ValueError:
        logger.warning("Invalid %s=%r, using default", _RAY_RECONNECT_POLL_ENV, raw)
        return 5.0


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
    working_dir = os.environ.get("MINT_RAY_WORKING_DIR", "").strip()
    return working_dir or None


def require_ray_address() -> str:
    addr = os.environ.get("RAY_ADDRESS", "").strip()
    if not addr:
        raise MissingRayAddressError("RAY_ADDRESS must be set before initializing Ray")
    return _normalize_ray_address(addr)


def preferred_driver_ray_address() -> str:
    configured_path = _configured_ray_head_address_path()
    if configured_path is not None:
        return _read_configured_ray_head_address(configured_path)
    for name in ("MINT_RAY_CLIENT_ADDRESS", "RAY_CLIENT_ADDRESS", "RAY_ADDRESS"):
        addr = os.environ.get(name, "").strip()
        if addr:
            return _normalize_ray_address(addr)
    raise MissingRayAddressError(
        "RAY_ADDRESS must be set before initializing Ray "
        f"(or set {_RAY_HEAD_ADDRESS_PATH_ENV}; "
        "optionally set MINT_RAY_CLIENT_ADDRESS or RAY_CLIENT_ADDRESS for the driver)"
    )


def _preferred_ray_init_address() -> str:
    return preferred_driver_ray_address()


def preferred_ray_address() -> str | None:
    try:
        return preferred_driver_ray_address()
    except MissingRayAddressError:
        return None


def ray_connection_epoch() -> int:
    return int(_RAY_CONNECTION_EPOCH)


def is_wrong_cluster_error(exc: BaseException) -> bool:
    return "WrongClusterID" in str(exc)


def force_reconnect_ray(*, namespace: str | None = None) -> None:
    import ray

    global _RAY_LAST_INIT_ADDRESS

    if ray.is_initialized():
        ray.shutdown()
    _RAY_LAST_INIT_ADDRESS = None
    _run_ray_reconnect_invalidators()
    init_ray(address="auto", namespace=namespace, ignore_reinit_error=True)


def register_ray_reconnect_invalidator(callback: Callable[[], None]) -> None:
    if callback not in _RAY_RECONNECT_INVALIDATORS:
        _RAY_RECONNECT_INVALIDATORS.append(callback)


def _run_ray_reconnect_invalidators() -> None:
    for callback in list(_RAY_RECONNECT_INVALIDATORS):
        try:
            callback()
        except Exception:
            logger.exception("Ray reconnect invalidator failed: %r", callback)


def _driver_runtime_env() -> dict[str, Any]:
    runtime_env: dict[str, Any] = {}

    # Only package a Ray Client working_dir when the operator asks for it.
    # `PFS_TINKER_PATH` is already a shared cluster path in Mint deployments,
    # so auto-uploading it through runtime_env just creates redundant node-side
    # working_dir caches.
    working_dir = (
        os.environ.get("MINT_RAY_JOB_WORKING_DIR", "").strip()
        or os.environ.get("MINT_RAY_WORKING_DIR", "").strip()
    )
    if working_dir:
        runtime_env["working_dir"] = working_dir

    py_modules_csv = os.environ.get("MINT_RAY_PY_MODULES_CSV", "").strip()
    if py_modules_csv:
        runtime_env["py_modules"] = [x.strip() for x in py_modules_csv.split(",") if x.strip()]

    return runtime_env


def client_job_runtime_env(*, address: str | None = None) -> dict[str, Any] | None:
    addr = preferred_ray_address() if address is None else _normalize_ray_address(address)
    if not addr or not addr.startswith("ray://"):
        return None
    runtime_env = _job_level_runtime_env(addr, _driver_runtime_env())
    return runtime_env or None


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


def _job_level_runtime_env(address: str, existing: Any) -> dict[str, Any] | Any:
    if not isinstance(address, str) or not address.startswith("ray://"):
        return existing

    runtime_env: dict[str, Any]
    if existing is None:
        runtime_env = {}
    elif isinstance(existing, dict):
        runtime_env = dict(existing)
    else:
        return existing

    # Ray Client actor creation serializes code on the job side before actor-level
    # runtime_env takes effect. Only package a job-level working_dir when the
    # operator explicitly requests it. Auto-packaging `PFS_TINKER_PATH` causes
    # remote API-host startup to hash/upload the whole shared checkout on every
    # restart even though that path is already directly visible on the host.
    job_working_dir = os.environ.get("MINT_RAY_JOB_WORKING_DIR", "").strip()
    if job_working_dir and "working_dir" not in runtime_env and "py_modules" not in runtime_env:
        runtime_env["working_dir"] = job_working_dir

    return runtime_env or existing


def init_ray(**kwargs: Any) -> Any:
    """Initialize Ray with optional log forwarding to driver.

    Adds log_to_driver=True when MINT_RAY_LOG_TO_DRIVER is enabled, unless explicitly
    set by the caller. When the caller leaves address unset or uses "auto", prefer
    the configured head-address file, then Ray Client endpoints, then direct attach.

    When attaching through Ray Client, local repo paths must be supplied at the job
    level rather than per-actor runtime_env. `MINT_RAY_JOB_WORKING_DIR` provides
    that path for `ray.init(runtime_env=...)` without forcing every actor runtime_env
    to use a local-path working_dir, which Ray rejects in client mode.
    """
    import ray
    global _RAY_CONNECTION_EPOCH, _RAY_LAST_INIT_ADDRESS

    current = kwargs.get("address")
    if current is None or current == "" or current == "auto":
        configured_address = preferred_ray_address()
        if configured_address is None:
            if ray.is_initialized():
                return None
            raise MissingRayAddressError(
                "RAY_ADDRESS must be set before initializing Ray "
                f"(or set {_RAY_HEAD_ADDRESS_PATH_ENV}; "
                "optionally set MINT_RAY_CLIENT_ADDRESS or RAY_CLIENT_ADDRESS for the driver)"
            )
        desired_address: Any = configured_address
    elif isinstance(current, str):
        desired_address = _normalize_ray_address(current.strip())
    else:
        desired_address = current
    kwargs["address"] = desired_address

    existing_runtime_env = kwargs.get("runtime_env")
    if existing_runtime_env is None:
        runtime_env = client_job_runtime_env(address=desired_address)
    else:
        runtime_env = _job_level_runtime_env(desired_address, existing_runtime_env)
    if runtime_env is not None:
        kwargs["runtime_env"] = runtime_env

    node_ip = os.environ.get("MINT_RAY_NODE_IP_ADDRESS", "").strip()
    if node_ip and "_node_ip_address" not in kwargs:
        kwargs["_node_ip_address"] = node_ip

    temp_dir = os.environ.get("MINT_RAY_TEMP_DIR", "").strip()
    if temp_dir and "_temp_dir" not in kwargs:
        kwargs["_temp_dir"] = temp_dir

    job_working_dir = os.environ.get("MINT_RAY_JOB_WORKING_DIR", "").strip()
    if job_working_dir:
        runtime_env = dict(kwargs.get("runtime_env") or {})
        runtime_env.setdefault("working_dir", job_working_dir)
        kwargs["runtime_env"] = runtime_env
    else:
        working_dir = ray_client_working_dir()
        runtime_env = kwargs.get("runtime_env")
        if working_dir and (runtime_env is None or isinstance(runtime_env, dict)):
            payload = {} if runtime_env is None else dict(runtime_env)
            payload.setdefault("working_dir", working_dir)
            kwargs["runtime_env"] = payload

    for k, v in ray_log_to_driver_kwargs().items():
        kwargs.setdefault(k, v)

    def _reconnect_if_address_drift(namespace: Any) -> bool:
        nonlocal desired_address
        global _RAY_LAST_INIT_ADDRESS
        if not ray.is_initialized():
            return False
        if _RAY_LAST_INIT_ADDRESS == desired_address:
            return True
        logger.warning(
            "ray connection drift detected: current=%r desired=%r namespace=%r; reconnecting",
            _RAY_LAST_INIT_ADDRESS,
            desired_address,
            namespace,
        )
        ray.shutdown()
        _RAY_LAST_INIT_ADDRESS = None
        _run_ray_reconnect_invalidators()
        return False

    with _RAY_INIT_THREAD_LOCK:
        namespace = kwargs.get("namespace")
        if _reconnect_if_address_drift(namespace):
            return None
        with _ray_init_interprocess_lock() as lock_wait_s:
            if _reconnect_if_address_drift(namespace):
                return None
            pid = os.getpid()
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
            _RAY_LAST_INIT_ADDRESS = desired_address
            _RAY_CONNECTION_EPOCH += 1
            logger.info(
                "ray.init ok pid=%s address=%r namespace=%r lock_wait_s=%.3f elapsed_s=%.3f",
                pid,
                kwargs.get("address"),
                namespace,
                lock_wait_s,
                time.monotonic() - t0,
            )
            return result
