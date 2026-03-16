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
export PYTHON_BIN=/root/tinker_project/tinker-server/.venv31213/bin/python
export PYTHONDONTWRITEBYTECODE=1

# ── Paths ──────────────────────────────────────────────────────────────────────
export HF_HOME=/vePFS-Mindverse/share/huggingface
export HF_HUB_OFFLINE=1
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/leixiang/tinker-server
export PFS_VLLM_PATH=/vePFS-Mindverse/share/code/vllm-0.16.0-pkg
export PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH=/vePFS-Mindverse/share/code/megatron-bridge-hollowman/src
export PFS_VERL_PATH=/vePFS-Mindverse/share/code/verl
export PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules
export LD_LIBRARY_PATH=/root/tinker_project/tinker-server/.venv31213/lib/python3.12/site-packages/torch/lib

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
