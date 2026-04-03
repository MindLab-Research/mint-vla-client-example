#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <namespace> [port=18125] [log_file=/tmp/tinker_server_vla_<namespace>.log]" >&2
  exit 2
fi

NAMESPACE="$1"
PORT="${2:-18125}"
LOG_FILE="${3:-/tmp/tinker_server_vla_${NAMESPACE}.log}"

CODE_ROOT="/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402"
RUNTIME_ROOT="/vePFS-Mindverse/share/code/root/mint-runtime-py31213-openpi-pr422-20260402"
HF_MODULES_PATH="/vePFS-Mindverse/share/huggingface/modules"
FAST_WEIGHTS_PATH="${MINT_VLA_FAST_WEIGHTS_PATH:-/vePFS-Mindverse/share/code/root/.openpi-data-vla-pr422/openpi-assets/checkpoints/pi0_fast_base/params.partial/params}"
WORKER_IP="192.168.38.176"
HEAD_IP="192.168.38.184"
QUEUE_ACTOR_NAME="tinker_api_work_queue_vla_${NAMESPACE}"

cd "$CODE_ROOT"
. ./configs/prod_volcano.env.sh

unset \
  MINT_STARTUP_LEASE_ACTOR_NAME \
  MINT_FUTURE_STORE_ACTOR_NAME \
  MINT_OWNER_RUNTIME_SUPERVISOR_ACTOR_NAME \
  MINT_TRAINING_SESSION_STORE_ACTOR_NAME \
  MINT_SAMPLING_SESSION_STORE_ACTOR_NAME \
  MINT_SESSION_INDEX_ACTOR_NAME \
  MINT_SESSION_HEARTBEAT_ACTOR_NAME \
  MINT_GATEWAY_SESSION_STORE_ACTOR_NAME \
  MINT_RESOURCE_POOL_ACTOR_NAME \
  MINT_TRAINING_CLEANUP_EXECUTOR_ACTOR_NAME \
  MINT_SAMPLING_CLEANUP_EXECUTOR_ACTOR_NAME \
  TINKER_CAPACITY_MANAGER_ACTOR_NAME \
  MINT_QUEUE_EXECUTION_RUNTIME_ACTOR_NAME

export MINT_UVICORN_WORKERS=1
export TINKER_PORT="$PORT"
export TINKER_USAGE_LOG_DIR="$CODE_ROOT/results/usage"
mkdir -p "$TINKER_USAGE_LOG_DIR"

export RAY_ADDRESS="${HEAD_IP}:6379"
export MINT_RAY_CLIENT_ADDRESS="ray://${HEAD_IP}:10001"
export PFS_TINKER_PATH="$CODE_ROOT"
export PFS_RUNTIME_ENV_ROOT="$RUNTIME_ROOT"
export PFS_HF_MODULES_PATH="$HF_MODULES_PATH"
export MINT_VLLM_CHILD_PYTHON_EXECUTABLE="$CODE_ROOT/scripts/vllm_worker_python.py"
export TINKER_RAY_NAMESPACE="$NAMESPACE"
export MINT_RAY_NAMESPACE="$NAMESPACE"
export MINT_SUPPORTED_MODELS='openpi/pi0-fast-libero-low-mem-finetune,openpi/pi05-libero-low-mem-finetune'
export MINT_PERSISTENT_MODELS=
export MINT_PERSISTENT_PREWARM_INFERENCE=0
export MINT_PERSISTENT_PREWARM_TRAINING=0
export MINT_MODEL_NODE_IPS_JSON='{"openpi/pi0-fast-libero-low-mem-finetune":["192.168.38.176"],"openpi/pi05-libero-low-mem-finetune":["192.168.38.176"]}'
export MINT_DENSE_MODEL_NODE_IPS_JSON='{}'
export MINT_VLLM_MODEL_NODE_IPS_JSON='{}'
export MINT_MEGATRON_MODEL_NODE_IPS_JSON='{}'
export MINT_VLLM_PINNED_NODE_IP_JSON='{}'
export MINT_OPENPI_FAST_WEIGHTS_PATH="$FAST_WEIGHTS_PATH"
export MINT_API_WORK_QUEUE_PINNED_NODE_IP="$WORKER_IP"
export MINT_CONTROL_PLANE_PINNED_NODE_IP="$WORKER_IP"
export TINKER_API_WORK_QUEUE_ACTOR_NAME="$QUEUE_ACTOR_NAME"
export MINT_API_WORK_QUEUE_ACTOR_NAME="$QUEUE_ACTOR_NAME"
export MINT_LOG_FILE="$LOG_FILE"

exec "$RUNTIME_ROOT/host-venv/bin/python" scripts/run_server.py
