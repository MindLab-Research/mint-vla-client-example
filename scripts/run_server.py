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

import logging
import pathlib
import argparse
import os
import sys
import traceback

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if args.config_path:
        os.environ["TINKER_CONFIG_PATH"] = str(args.config_path)

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
