#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from types import ModuleType

_REPO_ROOT = pathlib.Path(os.environ.get("PFS_TINKER_PATH", "")).resolve()
if not str(_REPO_ROOT):
    raise RuntimeError("PFS_TINKER_PATH is required")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_local_runtime_env_module() -> ModuleType:
    module_name = "_tinker_runtime_env_bootstrap_workers"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name,
        _REPO_ROOT / "tinker_server" / "runtime_env.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local tinker_server.runtime_env bootstrap module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_pythonpath(entries: str) -> str:
    return ":".join(part for part in entries.split(":") if part)


def _is_local_checkout_path(path: str) -> bool:
    if not path:
        return False
    try:
        candidate = pathlib.Path(path).resolve()
    except OSError:
        return False
    try:
        candidate.relative_to(_REPO_ROOT)
        return True
    except ValueError:
        return False


def _is_interpreter_managed_path(path: str) -> bool:
    if not path:
        return False
    try:
        candidate = pathlib.Path(path).resolve()
    except OSError:
        return False
    prefixes = {
        pathlib.Path(getattr(sys, "prefix", "")).resolve(),
        pathlib.Path(getattr(sys, "exec_prefix", "")).resolve(),
        pathlib.Path(getattr(sys, "base_prefix", "")).resolve(),
        pathlib.Path(getattr(sys, "base_exec_prefix", "")).resolve(),
    }
    for prefix in prefixes:
        if not str(prefix):
            continue
        try:
            candidate.relative_to(prefix)
            return True
        except ValueError:
            continue
    return False


def _set_exact_pythonpath(entries: str) -> str:
    normalized = _normalize_pythonpath(entries)
    if not normalized:
        return ""
    parts = normalized.split(":")
    os.environ["PYTHONPATH"] = normalized
    existing = [
        part
        for part in sys.path
        if part not in parts and not _is_local_checkout_path(part) and _is_interpreter_managed_path(part)
    ]
    sys.path[:] = [*parts, *existing]
    return normalized


def _exact_torch_ld_library_path() -> str:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError("torch is not importable from the configured host runtime")
    torch_lib = pathlib.Path(torch_spec.origin).resolve().parent / "lib"
    if not torch_lib.is_dir():
        raise RuntimeError(f"Host torch lib directory missing for interpreter {sys.executable}: {torch_lib}")
    return ":".join(
        [
            str(torch_lib),
            "/usr/local/cuda/compat/lib",
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
            "/usr/local/cuda/lib64",
        ]
    )


def _set_exact_torch_ld_library_path() -> str:
    value = _exact_torch_ld_library_path()
    os.environ["LD_LIBRARY_PATH"] = value
    return value


def _reexec_if_env_mismatch(*, pythonpath: str, ld_library_path: str) -> None:
    desired = {
        "PYTHONPATH": pythonpath,
        "LD_LIBRARY_PATH": ld_library_path,
        "TINKER_SERVER_EXACT_ENV": "1",
    }
    current_matches = all(os.environ.get(k) == v for k, v in desired.items() if k != "TINKER_SERVER_EXACT_ENV")
    if current_matches:
        return
    if os.environ.get("TINKER_SERVER_EXACT_ENV") == "1":
        raise RuntimeError(
            "TINKER_SERVER_EXACT_ENV=1 but runtime environment is still incorrect: "
            f"PYTHONPATH={os.environ.get('PYTHONPATH')!r} LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')!r}"
        )
    new_env = dict(os.environ)
    new_env.update(desired)
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


def _reexec_to_runtime_host_python_if_needed() -> None:
    env_root = os.environ.get("PFS_RUNTIME_ENV_ROOT", "").strip()
    if not env_root:
        return
    runtime_env = _load_local_runtime_env_module()
    layout = runtime_env.validate_runtime_env_layout(env_root, require_host_python=True)
    target_python = pathlib.Path(layout.host_python).resolve()
    current_python = pathlib.Path(sys.executable).resolve()
    if current_python == target_python:
        return
    if os.environ.get("TINKER_SERVER_HOST_PYTHON") == "1":
        raise RuntimeError(
            f"TINKER_SERVER_HOST_PYTHON=1 but sys.executable={current_python} != {target_python}"
        )
    new_env = dict(os.environ)
    new_env["TINKER_SERVER_HOST_PYTHON"] = "1"
    os.execvpe(str(target_python), [str(target_python), *sys.argv], new_env)


class PollingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "408" in msg and "retrieve_future" in msg:
            return False
        if "telemetry" in msg:
            return False
        return True


if __name__ == "__main__":
    _reexec_to_runtime_host_python_if_needed()
    runtime_env = _load_local_runtime_env_module()
    pythonpath = _set_exact_pythonpath(
        runtime_env.bootstrap_runtime_pythonpath(
            os.environ,
            repo_root=str(_REPO_ROOT),
        )
    )
    ld_library_path = _set_exact_torch_ld_library_path()
    _reexec_if_env_mismatch(pythonpath=pythonpath, ld_library_path=ld_library_path)

    from tinker_server.logging_context import configure_logging

    configure_logging()
    logging.getLogger("uvicorn.access").addFilter(PollingLogFilter())

    import uvicorn
    from tinker_server.config import config

    workers = int(os.environ.get("MINT_UVICORN_WORKERS", "2"))
    timeout_worker_healthcheck = int(os.environ.get("MINT_UVICORN_WORKER_HEALTHCHECK_TIMEOUT", "120"))
    uvicorn.run(
        "tinker_server.app:app",
        host=config.host,
        port=config.port,
        log_level="info",
        workers=workers,
        timeout_worker_healthcheck=timeout_worker_healthcheck,
    )
