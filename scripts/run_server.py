#!/usr/bin/env python
"""Run the tinker-server.

Usage:
    python scripts/run_server.py

Environment variables:
    TINKER_HOST: Server host (default: 0.0.0.0)
    TINKER_PORT: Server port (default: 8000)
    TINKER_MODEL_PATH: HuggingFace model path (default: Qwen/Qwen2.5-7B-Instruct)
    TINKER_TP_SIZE: Tensor parallel size (default: 1)
    TINKER_GPU_MEM_UTIL: GPU memory utilization (default: 0.9)
    TINKER_MAX_MODEL_LEN: Maximum model context length (default: auto)
"""

import uvicorn

from tinker_server.app import app
from tinker_server.config import config

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )
