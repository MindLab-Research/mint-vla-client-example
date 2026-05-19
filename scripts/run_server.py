#!/usr/bin/env python
"""Run mint-server (server-side).

Usage:
    python scripts/run_server.py

Optional:
    python scripts/run_server.py --config /path/to/config.toml

Environment variables:
    MINT_HOST: Server host (default: 0.0.0.0)
    MINT_PORT: Server port (default: 8000)
    MINT_TP_SIZE: Tensor parallel size (default: auto-detected per model)
    MINT_DP_SIZE: Data parallel size for MoE (default: auto-detected per model)
    MINT_GPU_MEM_UTIL: GPU memory utilization (default: 0.9)
    MINT_MAX_MODEL_LEN: Maximum model context length (default: auto)

Note: No default model is configured. Clients specify models per-request.
Parallelism is auto-detected from the model registry when engines are created.
"""

import argparse
import importlib.util
import logging
import os
import pathlib
import sys
import tomllib
import traceback
from types import ModuleType, SimpleNamespace

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_local_runtime_env_module() -> ModuleType:
    module_name = "_mint_runtime_env_bootstrap"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name,
        _REPO_ROOT / "mint_server" / "runtime_env.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load local mint_server.runtime_env bootstrap module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _env_get(environ: dict[str, str], name: str, default: str = "") -> str:
    runtime_env = _load_local_runtime_env_module()
    if not hasattr(runtime_env, "env_get"):
        if name.startswith("MINT_"):
            return environ.get(name) or environ.get(f"TINKER_{name[len('MINT_'):]}") or default
        if name.startswith("TINKER_"):
            return environ.get(f"MINT_{name[len('TINKER_'):]}") or environ.get(name) or default
        return environ.get(name, default)
    value = runtime_env.env_get(environ, name, default)
    return default if value is None else str(value)


def _env_nonempty(environ: dict[str, str], name: str) -> str | None:
    runtime_env = _load_local_runtime_env_module()
    if not hasattr(runtime_env, "env_nonempty"):
        value = _env_get(environ, name, "")
        value = str(value).strip()
        return value or None
    return runtime_env.env_nonempty(environ, name)


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

def _normalize_pythonpath(entries: str) -> str:
    return ":".join(part for part in entries.split(":") if part)


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
        if part not in parts
        and not _is_local_checkout_path(part)
        and _is_interpreter_managed_path(part)
    ]
    sys.path[:] = [*parts, *existing]
    return normalized


def _exact_torch_ld_library_path() -> str:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError("torch is not importable from the configured host runtime")
    torch_lib = pathlib.Path(torch_spec.origin).resolve().parent / "lib"
    if not torch_lib.is_dir():
        raise RuntimeError(
            f"Host torch lib directory missing for interpreter {sys.executable}: {torch_lib}"
        )
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
        "MINT_SERVER_EXACT_ENV": "1",
    }
    current_pythonpath = _normalize_pythonpath(os.environ.get("PYTHONPATH", ""))
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_matches = all(
        _env_get(os.environ, k) == v for k, v in desired.items() if k != "MINT_SERVER_EXACT_ENV"
    )
    if current_matches:
        return
    if _env_get(os.environ, "MINT_SERVER_EXACT_ENV") == "1":
        raise RuntimeError(
            "MINT_SERVER_EXACT_ENV=1 but runtime environment is still incorrect: "
            f"PYTHONPATH={os.environ.get('PYTHONPATH')!r} LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')!r}"
        )
    new_env = dict(os.environ)
    new_env.update(desired)
    new_env["MINT_SERVER_ENV_NORMALIZED"] = "1"
    new_env["MINT_SERVER_PYTHONPATH_CHANGED"] = "1" if current_pythonpath != pythonpath else "0"
    new_env["MINT_SERVER_LD_LIBRARY_PATH_CHANGED"] = "1" if current_ld_library_path != ld_library_path else "0"
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
    if _env_get(os.environ, "MINT_SERVER_HOST_PYTHON") == "1":
        raise RuntimeError(
            f"MINT_SERVER_HOST_PYTHON=1 but sys.executable={current_python} != {target_python}"
        )
    new_env = dict(os.environ)
    new_env["MINT_SERVER_HOST_PYTHON"] = "1"
    os.execvpe(str(target_python), [str(target_python), *sys.argv], new_env)

def crash_handler(exc_type, exc_value, exc_tb):
    """Log uncaught exceptions before crash."""
    print(f"\n{'='*60}", flush=True)
    print("UNCAUGHT EXCEPTION - SERVER CRASHING", flush=True)
    print(f"{'='*60}", flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print(f"{'='*60}\n", flush=True)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = crash_handler


class PollingLogFilter(logging.Filter):
    """Filter out noisy 408 polling responses from access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Suppress 408 responses (polling for futures)
        if "408" in msg and "retrieve_future" in msg:
            return False
        # Suppress telemetry logs
        if "telemetry" in msg:
            return False
        return True


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to mint-server TOML config file (sets MINT_CONFIG_PATH before import).",
    )
    return p.parse_args(argv)


def _seed_runtime_env_from_config(config_path: str) -> None:
    data = tomllib.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    paths = data.get("paths", {})
    required = (
        "pfs_runtime_env_root",
        "mint_code_root",
        "pfs_hf_modules_path",
    )
    missing = [key for key in required if key not in paths]
    if missing:
        raise RuntimeError(
            f"{config_path} must define [paths] {required}; missing={missing}"
        )
    os.environ["PFS_RUNTIME_ENV_ROOT"] = str(paths["pfs_runtime_env_root"])
    os.environ["MINT_CODE_ROOT"] = str(paths["mint_code_root"])
    os.environ["PFS_HF_MODULES_PATH"] = str(paths["pfs_hf_modules_path"])


_DEFAULT_UVICORN_WORKERS = 8
_DEFAULT_WORKER_HEALTHCHECK_TIMEOUT_S = 120
_APP_IMPORT_STRING = "mint_server.app:app"


def _resolve_workers(environ: dict[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = str(env.get("MINT_UVICORN_WORKERS", "")).strip()
    if not raw:
        return _DEFAULT_UVICORN_WORKERS
    try:
        workers = int(raw)
    except ValueError as e:
        raise RuntimeError(f"MINT_UVICORN_WORKERS must be an integer, got {raw!r}") from e
    if workers < 1:
        raise RuntimeError(f"MINT_UVICORN_WORKERS must be >= 1, got {workers}")
    return workers


def _resolve_worker_healthcheck_timeout_s(environ: dict[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = str(env.get("MINT_UVICORN_WORKER_HEALTHCHECK_TIMEOUT", "")).strip()
    if not raw:
        return _DEFAULT_WORKER_HEALTHCHECK_TIMEOUT_S
    try:
        timeout_s = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"MINT_UVICORN_WORKER_HEALTHCHECK_TIMEOUT must be an integer, got {raw!r}"
        ) from e
    if timeout_s < 1:
        raise RuntimeError(
            f"MINT_UVICORN_WORKER_HEALTHCHECK_TIMEOUT must be >= 1, got {timeout_s}"
        )
    return timeout_s


def _uvicorn_target_and_kwargs(*, app: object, config: SimpleNamespace, environ: dict[str, str] | None = None) -> tuple[object, dict[str, object]]:
    workers = _resolve_workers(environ)
    kwargs: dict[str, object] = {
        "host": config.host,
        "port": config.port,
        "log_level": "info",
    }
    if workers > 1:
        kwargs["workers"] = workers
        kwargs["timeout_worker_healthcheck"] = _resolve_worker_healthcheck_timeout_s(environ)
        return _APP_IMPORT_STRING, kwargs
    return app, kwargs


def _launcher_observability(*, target: object, kwargs: dict[str, object], environ: dict[str, str] | None = None) -> dict[str, object]:
    env = os.environ if environ is None else environ
    workers = int(kwargs.get("workers", 1))
    pythonpath = _normalize_pythonpath(env.get("PYTHONPATH", ""))
    entry_count = 0 if not pythonpath else len(pythonpath.split(":"))
    return {
        "mode": "multi-worker" if workers > 1 else "single-worker",
        "workers": workers,
        "host": kwargs["host"],
        "port": kwargs["port"],
        "target": target if isinstance(target, str) else "inproc:app",
        "timeout_worker_healthcheck": kwargs.get("timeout_worker_healthcheck"),
        "namespace": _env_get(env, "MINT_RAY_NAMESPACE"),
        "ray_address": env.get("RAY_ADDRESS", ""),
        "ray_client_address": env.get("MINT_RAY_CLIENT_ADDRESS") or env.get("RAY_CLIENT_ADDRESS") or "",
        "env_normalized": _env_get(env, "MINT_SERVER_ENV_NORMALIZED") == "1",
        "pythonpath_changed": _env_get(env, "MINT_SERVER_PYTHONPATH_CHANGED") == "1",
        "ld_library_path_changed": _env_get(env, "MINT_SERVER_LD_LIBRARY_PATH_CHANGED") == "1",
        "pythonpath_entries": entry_count,
    }


def _log_launcher_observability(*, target: object, kwargs: dict[str, object], environ: dict[str, str] | None = None) -> None:
    meta = _launcher_observability(target=target, kwargs=kwargs, environ=environ)
    logging.getLogger("mint_server.launcher").info(
        "launcher mode=%s workers=%s host=%s port=%s target=%r timeout_worker_healthcheck=%r namespace=%r ray_address=%r ray_client_address=%r env_normalized=%s pythonpath_changed=%s ld_library_path_changed=%s pythonpath_entries=%s",
        meta["mode"],
        meta["workers"],
        meta["host"],
        meta["port"],
        meta["target"],
        meta["timeout_worker_healthcheck"],
        meta["namespace"],
        meta["ray_address"],
        meta["ray_client_address"],
        meta["env_normalized"],
        meta["pythonpath_changed"],
        meta["ld_library_path_changed"],
        meta["pythonpath_entries"],
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)
    config_path = str(args.config_path) if args.config_path else (_env_nonempty(os.environ, "MINT_CONFIG_PATH") or "")
    if config_path:
        os.environ["MINT_CONFIG_PATH"] = config_path
        _seed_runtime_env_from_config(config_path)

    _reexec_to_runtime_host_python_if_needed()

    runtime_env = _load_local_runtime_env_module()

    pythonpath = _set_exact_pythonpath(
        runtime_env.bootstrap_runtime_pythonpath(
            os.environ,
            repo_root=str(_REPO_ROOT),
        )
    )
    ld_library_path = _set_exact_torch_ld_library_path()
    _reexec_if_env_mismatch(
        pythonpath=pythonpath,
        ld_library_path=ld_library_path,
    )

    from mint_server.logging_context import configure_logging

    configure_logging()
    logging.getLogger("uvicorn.access").addFilter(PollingLogFilter())

    import uvicorn

    from mint_server.app import app
    from mint_server.config import config

    target, kwargs = _uvicorn_target_and_kwargs(
        app=app,
        config=SimpleNamespace(host=config.host, port=config.port),
    )
    _log_launcher_observability(target=target, kwargs=kwargs)
    uvicorn.run(target, **kwargs)


if __name__ == "__main__":
    main()
