#!/usr/bin/env python
"""Run the tinker-server.

Usage:
    python scripts/run_server.py

Environment variables:
    TINKER_HOST: Server host (default: 0.0.0.0)
    TINKER_PORT: Server port (default: 8000)
    TINKER_MODEL_PATH: HuggingFace model path (default: Qwen/Qwen2.5-7B-Instruct)
    TINKER_TP_SIZE: Tensor parallel size (default: auto-detected from model)
    TINKER_DP_SIZE: Data parallel size for MoE (default: auto-detected from model)
    TINKER_GPU_MEM_UTIL: GPU memory utilization (default: 0.9)
    TINKER_MAX_MODEL_LEN: Maximum model context length (default: auto)
"""

import logging
import os

import uvicorn

from tinker_server.app import app
from tinker_server.config import config


def extract_model_name(model_path: str) -> str:
    """Extract HuggingFace model name from path.

    Handles both:
    - Direct names: "Qwen/Qwen3-30B-A3B"
    - Snapshot paths: "/path/to/models--Qwen--Qwen3-30B-A3B/snapshots/abc123"
    """
    # Check if it's a snapshot path
    if "models--" in model_path:
        # Extract from path like: models--Qwen--Qwen3-30B-A3B
        for part in model_path.split("/"):
            if part.startswith("models--"):
                # Convert models--Qwen--Qwen3-30B-A3B to Qwen/Qwen3-30B-A3B
                name_parts = part.replace("models--", "").split("--")
                return "/".join(name_parts)
    return model_path


def apply_auto_detection():
    """Auto-detect parallelism settings from model name if not explicitly set."""
    from tinker_server.backend.model_registry import get_recommended_parallelism

    # Check if user explicitly set TP or DP via environment
    tp_explicit = "TINKER_TP_SIZE" in os.environ
    dp_explicit = "TINKER_DP_SIZE" in os.environ

    if tp_explicit or dp_explicit:
        # User specified parallelism, don't override
        logging.info(
            f"Using explicit parallelism: TP={config.tensor_parallel_size}, DP={config.data_parallel_size}"
        )
        return

    # Auto-detect from model name
    model_name = extract_model_name(config.model_path)
    tp, dp = get_recommended_parallelism(model_name)

    if tp != 1 or dp != 1:
        logging.info(
            f"Auto-detected parallelism for {model_name}: TP={tp}, DP={dp}"
        )
        # Update config (dataclass is mutable)
        config.tensor_parallel_size = tp
        config.data_parallel_size = dp
    else:
        logging.info(f"Using default parallelism for {model_name}: TP=1, DP=1")


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


if __name__ == "__main__":
    # Configure logging early
    logging.basicConfig(level=logging.INFO)

    # Apply auto-detection before app starts
    apply_auto_detection()

    # Suppress noisy 408 polling logs
    logging.getLogger("uvicorn.access").addFilter(PollingLogFilter())

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )
