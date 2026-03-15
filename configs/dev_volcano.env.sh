#!/bin/sh

# Authoritative non-secret runtime configuration for mint-dev.
# This file is version controlled on purpose.

# ── Python / venv ──────────────────────────────────────────────────────────────
export PYTHON_BIN=/root/tinker_project/tinker-server/.venv31213/bin/python
export PYTHONDONTWRITEBYTECODE=1

# ── Paths ──────────────────────────────────────────────────────────────────────
export HF_HOME=/vePFS-Mindverse/share/huggingface
export HF_HUB_OFFLINE=1
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/leixiang/tinker-server
export PYTHONPATH=/vePFS-Mindverse/share/code/vllm-0.16.0-pkg:/vePFS-Mindverse/share/code/megatron-bridge-hollowman/src:/vePFS-Mindverse/share/code/verl:/vePFS-Mindverse/share/code/leixiang/tinker-server:/vePFS-Mindverse/share/huggingface/modules:/root/tinker_project/tinker-server
export LD_LIBRARY_PATH=/root/tinker_project/tinker-server/.venv31213/lib/python3.12/site-packages/torch/lib

# ── Ray ────────────────────────────────────────────────────────────────────────
export RAY_ADDRESS=192.168.37.185:6379
export RAY_NODE_IP_ADDRESS=192.168.32.124
export RAY_WORKER_CPUS=8

# ── 30B main server ────────────────────────────────────────────────────────────
export TINKER_HOST=0.0.0.0
export TINKER_PORT=8000
export TINKER_RAY_NAMESPACE=tinker_leixiang
export MINT_RAY_NAMESPACE=tinker_leixiang

export MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507
export PINNED_NODE_IP=192.168.37.186
export MINT_MEGATRON_NODE_IPS_CSV=192.168.37.186
export MINT_MODEL_NODE_IPS_JSON='{"Qwen/Qwen3-30B-A3B-Instruct-2507":["192.168.37.186"]}'
export MINT_VLLM_PINNED_NODE_IP_JSON='{"Qwen/Qwen3-30B-A3B-Instruct-2507":"192.168.37.186"}'

export MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY=1024

# 30B is the persistent local model; 0.6B is gateway-routed so also listed here
# so enforce_base_model_allowed() lets it through to the gateway routing logic.
export MINT_SUPPORTED_MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507,Qwen/Qwen3-0.6B"
export MINT_PERSISTENT_MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507"

export LOG_FILE=/tmp/tinker_server.log

# ── 0.6B sidecar (gateway upstream) ───────────────────────────────────────────
export GATEWAY_06B_PORT=8002
export GATEWAY_06B_NODE_IP=192.168.37.240
export GATEWAY_06B_NAMESPACE=tinker_leixiang_06b
export GATEWAY_06B_MODEL=Qwen/Qwen3-0.6B

export MINT_LOG_FILE=/tmp/tinker_server.log

# Gateway config: 30B server (port 8000) routes 0.6B requests to sidecar (port 8002).
# auth_mode:none because the sidecar is started without TINKER_API_KEY.
export TINKER_GATEWAY_CONFIG_JSON='{"model_to_upstream":{"Qwen/Qwen3-0.6B":"sidecar-06b"},"upstreams":{"sidecar-06b":{"base_url":"http://127.0.0.1:8002","auth_mode":"none"}}}'
