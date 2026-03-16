#!/usr/bin/env python
"""Run the tinker-server (server-side).

Usage:
    python scripts/run_server.py

Optional:
    python scripts/run_server.py --config /path/to/config.toml

Environment variables:
    TINKER_HOST: Server host (default: 0.0.0.0)
    TINKER_PORT: Server port (default: 8000)
    TINKER_TP_SIZE: Tensor parallel size (default: auto-detected per model)
    TINKER_DP_SIZE: Data parallel size for MoE (default: auto-detected per model)
    TINKER_GPU_MEM_UTIL: GPU memory utilization (default: 0.9)
    TINKER_MAX_MODEL_LEN: Maximum model context length (default: auto)

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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _normalize_pythonpath(entries: str) -> str:
    return ":".join(part for part in entries.split(":") if part)


def _set_exact_pythonpath(entries: str) -> str:
    normalized = _normalize_pythonpath(entries)
    if not normalized:
        return ""
    parts = normalized.split(":")
    os.environ["PYTHONPATH"] = normalized
    existing = [part for part in sys.path if part not in parts]
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
        "TINKER_SERVER_EXACT_ENV": "1",
    }
    current_matches = all(
        os.environ.get(k) == v for k, v in desired.items() if k != "TINKER_SERVER_EXACT_ENV"
    )
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
    from tinker_server.runtime_env import validate_runtime_env_layout

    layout = validate_runtime_env_layout(env_root, require_host_python=True)
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
        help="Path to tinker-server TOML config file (sets TINKER_CONFIG_PATH before import).",
    )
    return p.parse_args(argv)


def _seed_runtime_env_from_config(config_path: str) -> None:
    data = tomllib.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
    paths = data.get("paths", {})
    if "pfs_runtime_env_root" in paths:
        os.environ["PFS_RUNTIME_ENV_ROOT"] = str(paths["pfs_runtime_env_root"])
    if "pfs_tinker_path" in paths:
        os.environ["PFS_TINKER_PATH"] = str(paths["pfs_tinker_path"])
    if "pfs_hf_modules_path" in paths:
        os.environ["PFS_HF_MODULES_PATH"] = str(paths["pfs_hf_modules_path"])


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if args.config_path:
        os.environ["TINKER_CONFIG_PATH"] = str(args.config_path)
        _seed_runtime_env_from_config(str(args.config_path))

    _reexec_to_runtime_host_python_if_needed()

    from tinker_server.runtime_env import bootstrap_runtime_pythonpath

    pythonpath = _set_exact_pythonpath(
        bootstrap_runtime_pythonpath(
            os.environ,
            repo_root=str(_REPO_ROOT),
        )
    )
    ld_library_path = _set_exact_torch_ld_library_path()
    _reexec_if_env_mismatch(
        pythonpath=pythonpath,
        ld_library_path=ld_library_path,
    )

    # Configure structured logging early with request_id support
    from tinker_server.logging_context import configure_logging
    configure_logging()

    # Suppress noisy 408 polling logs
    logging.getLogger("uvicorn.access").addFilter(PollingLogFilter())

    import uvicorn

    from tinker_server.app import app
    from tinker_server.config import config

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )
