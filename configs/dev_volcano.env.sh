#!/bin/sh

# Template configuration for multi-developer shared Ray cluster.
# Copy this file and customize for your namespace and node allocation.
#
# Example:
#   cp configs/dev_volcano_template.env.sh configs/dev_volcano_leixiang.env.sh
#   # Edit dev_volcano_leixiang.env.sh with your namespace and nodes
#   source configs/dev_volcano_leixiang.env.sh
#   python scripts/run_server.py

# ── Python / venv ──────────────────────────────────────────────────────────────
export PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213
export PYTHON_BIN=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
export PYTHONDONTWRITEBYTECODE=1

# ── Paths ──────────────────────────────────────────────────────────────────────
export HF_HOME=/vePFS-Mindverse/share/huggingface
export HF_HUB_OFFLINE=1
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/leixiang/tinker-server
export PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules
export LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# ── Ray ────────────────────────────────────────────────────────────────────────
export RAY_ADDRESS=192.168.37.185:6379
export MINT_RAY_NODE_IP_ADDRESS=192.168.32.124

# ── Namespace (CUSTOMIZE THIS) ─────────────────────────────────────────────────
# Use a unique namespace to avoid conflicts with other developers.
# Convention: tinker_<your_name>
export TINKER_RAY_NAMESPACE=tinker_leixiang
export MINT_RAY_NAMESPACE=tinker_leixiang

# ── Server ─────────────────────────────────────────────────────────────────────
export TINKER_HOST=0.0.0.0
export TINKER_PORT=8000

# ── Node Allocation (CUSTOMIZE THIS) ───────────────────────────────────────────

export MINT_MEGATRON_NODE_IPS_CSV=192.168.38.38
export MINT_MODEL_NODE_IPS_JSON='{"Qwen/Qwen3-30B-A3B-Instruct-2507":["192.168.38.38"]}'
export MINT_VLLM_PINNED_NODE_IP_JSON='{"Qwen/Qwen3-30B-A3B-Instruct-2507":"192.168.38.38"}'

# ── Models ─────────────────────────────────────────────────────────────────────
export MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY=1024
export MINT_SUPPORTED_MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507"
export MINT_PERSISTENT_MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507"

# ── Logging ────────────────────────────────────────────────────────────────────
export MINT_LOG_FILE=/tmp/tinker_server.log
