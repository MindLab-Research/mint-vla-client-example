#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <namespace> [port=18125] [log_file=/tmp/tinker_server_vla_<namespace>.log]" >&2
  exit 2
fi

NAMESPACE="$1"
PORT="${2:-18125}"
LOG_FILE="${3:-/tmp/tinker_server_vla_${NAMESPACE}.log}"
NAME_SUFFIX="$(printf '%s' "${NAMESPACE}" | tr -c '[:alnum:]' '_')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${MINT_VLA_RUNTIME_ROOT:-/vePFS-Mindverse/share/code/root/mint-runtime-py31213-origin-develop-vla-20260428r1}"
HF_MODULES_PATH="/vePFS-Mindverse/share/huggingface/modules"
FAST_WEIGHTS_PATH="${MINT_VLA_FAST_WEIGHTS_PATH:-/vePFS-Mindverse/share/models/openpi/pi0_fast_base/params}"
PI05_WEIGHTS_PATH="${MINT_VLA_PI05_WEIGHTS_PATH:-/vePFS-Mindverse/share/models/openpi/pi05_base/params}"
FAST_ASSETS_PATH="${MINT_VLA_FAST_ASSETS_PATH:-/vePFS-Mindverse/share/models/openpi/pi0_fast_base_official_20260428/assets}"
PI05_ASSETS_PATH="${MINT_VLA_PI05_ASSETS_PATH:-/vePFS-Mindverse/share/models/openpi/pi05_base/assets}"
FAST_NODE_IP="${MINT_VLA_FAST_NODE_IP:-192.168.39.110}"
PI05_NODE_IP="${MINT_VLA_PI05_NODE_IP:-192.168.39.110}"
HEAD_IP="${MINT_VLA_HEAD_IP:-192.168.39.87}"
CONTROL_PLANE_IP="${MINT_VLA_CONTROL_PLANE_IP:-192.168.39.87}"
OPENPI_CHECKPOINT_BASE_DIR="${MINT_VLA_OPENPI_CHECKPOINT_BASE_DIR:-$CODE_ROOT/checkpoints}"
OPENPI_XLA_CACHE_DIR="${MINT_VLA_OPENPI_XLA_CACHE_DIR:-$CODE_ROOT/results/xla_autotune_cache}"
OPENPI_XLA_FLAGS_DEFAULT="--xla_gpu_per_fusion_autotune_cache_dir=${OPENPI_XLA_CACHE_DIR} --xla_gpu_exclude_nondeterministic_ops"
BASE_CONFIG_ENV="${MINT_VLA_BASE_CONFIG_ENV:-/share/mint/prod/config/prod.env}"

for required_path in \
  "$CODE_ROOT" \
  "$BASE_CONFIG_ENV" \
  "$CODE_ROOT/scripts/run_server.py" \
  "$RUNTIME_ROOT/host-venv/bin/python" \
  "$CODE_ROOT/scripts/vllm_worker_python.py"
do
  if [[ ! -e "$required_path" ]]; then
    echo "required runtime path missing: $required_path" >&2
    exit 2
  fi
done

cd "$CODE_ROOT"
. "$BASE_CONFIG_ENV"

for required_path in \
  "$FAST_WEIGHTS_PATH" \
  "$PI05_WEIGHTS_PATH" \
  "$FAST_ASSETS_PATH" \
  "$PI05_ASSETS_PATH"
do
  if [[ ! -e "$required_path" ]]; then
    echo "required OpenPI path missing: $required_path" >&2
    exit 2
  fi
done

unset \
  MINT_STARTUP_LEASE_ACTOR_NAME \
  MINT_FUTURE_STORE_ACTOR_NAME \
  MINT_OWNER_RUNTIME_SUPERVISOR_ACTOR_NAME \
  MINT_API_WORK_QUEUE_ACTOR_NAME \
  MINT_CAPACITY_MANAGER_ACTOR_NAME \
  MINT_QUEUE_SUPERVISOR_ACTOR_NAME \
  TINKER_API_WORK_QUEUE_ACTOR_NAME \
  MINT_GATEWAY_SESSION_STORE_ACTOR_NAME \
  MINT_SAMPLING_SESSION_STORE_ACTOR_NAME \
  MINT_TRAINING_SESSION_STORE_ACTOR_NAME \
  MINT_SESSION_HEARTBEAT_ACTOR_NAME \
  MINT_SESSION_INDEX_ACTOR_NAME \
  MINT_RESOURCE_POOL_ACTOR_NAME \
  MINT_TRAINING_CLEANUP_EXECUTOR_ACTOR_NAME \
  MINT_SAMPLING_CLEANUP_EXECUTOR_ACTOR_NAME \
  MINT_FUTURE_REPLAY_SWEEPER_ACTOR_NAME \
  TINKER_CAPACITY_MANAGER_ACTOR_NAME \
  MINT_QUEUE_EXECUTION_RUNTIME_ACTOR_NAME

export MINT_UVICORN_WORKERS=1
export TINKER_PORT="$PORT"
export TINKER_USAGE_LOG_DIR="$CODE_ROOT/results/usage"
mkdir -p "$TINKER_USAGE_LOG_DIR" "$OPENPI_CHECKPOINT_BASE_DIR" "$OPENPI_XLA_CACHE_DIR"

export RAY_ADDRESS="${HEAD_IP}:6379"
export MINT_RAY_CLIENT_ADDRESS="ray://${HEAD_IP}:10001"
export MINT_RAY_INIT_LOCK_PATH="${MINT_RAY_INIT_LOCK_PATH:-/tmp/mint_ray_init_${NAMESPACE}.lock}"
export MINT_RAY_JOB_WORKING_DIR="$CODE_ROOT"
export TINKER_API_KEY="${TINKER_API_KEY:-dummy}"
export PFS_TINKER_PATH="$CODE_ROOT"
export PFS_RUNTIME_ENV_ROOT="$RUNTIME_ROOT"
export PFS_HF_MODULES_PATH="$HF_MODULES_PATH"
export MINT_VLLM_CHILD_PYTHON_EXECUTABLE="$CODE_ROOT/scripts/vllm_worker_python.py"
export TINKER_RAY_NAMESPACE="$NAMESPACE"
export MINT_RAY_NAMESPACE="$NAMESPACE"
export MINT_STARTUP_LEASE_ACTOR_NAME="tinker_startup_lease_${NAME_SUFFIX}"
export MINT_API_WORK_QUEUE_ACTOR_NAME="tinker_api_work_queue_${NAME_SUFFIX}"
export MINT_CAPACITY_MANAGER_ACTOR_NAME="tinker_capacity_manager_${NAME_SUFFIX}"
export MINT_QUEUE_EXECUTION_RUNTIME_ACTOR_NAME="tinker_queue_execution_runtime_${NAME_SUFFIX}"
export MINT_QUEUE_SUPERVISOR_ACTOR_NAME="tinker_queue_supervisor_${NAME_SUFFIX}"
export MINT_FUTURE_STORE_ACTOR_NAME="tinker_future_store_${NAME_SUFFIX}"
export MINT_GATEWAY_SESSION_STORE_ACTOR_NAME="tinker_gateway_session_store_${NAME_SUFFIX}"
export MINT_SAMPLING_SESSION_STORE_ACTOR_NAME="tinker_sampling_session_store_${NAME_SUFFIX}"
export MINT_TRAINING_SESSION_STORE_ACTOR_NAME="tinker_training_session_store_${NAME_SUFFIX}"
export MINT_SESSION_HEARTBEAT_ACTOR_NAME="tinker_session_heartbeat_store_${NAME_SUFFIX}"
export MINT_SESSION_INDEX_ACTOR_NAME="tinker_session_index_store_${NAME_SUFFIX}"
export MINT_RESOURCE_POOL_ACTOR_NAME="tinker_resource_pool_${NAME_SUFFIX}"
export MINT_OWNER_RUNTIME_SUPERVISOR_ACTOR_NAME="tinker_owner_runtime_supervisor_${NAME_SUFFIX}"
export MINT_TRAINING_CLEANUP_EXECUTOR_ACTOR_NAME="tinker_training_cleanup_executor_${NAME_SUFFIX}"
export MINT_SAMPLING_CLEANUP_EXECUTOR_ACTOR_NAME="tinker_sampling_cleanup_executor_${NAME_SUFFIX}"
export MINT_FUTURE_REPLAY_SWEEPER_ACTOR_NAME="mint_future_replay_sweeper_${NAME_SUFFIX}"
export MINT_SUPPORTED_MODELS='openpi/pi0-fast-libero-low-mem-finetune,openpi/pi05-libero-low-mem-finetune'
export MINT_PERSISTENT_MODELS=
export MINT_PERSISTENT_PREWARM_INFERENCE=0
export MINT_PERSISTENT_PREWARM_TRAINING=0
export MINT_MODEL_PLACEMENT_JSON="{\"openpi/pi0-fast-libero-low-mem-finetune\":{\"replica\":0,\"node_ip\":\"${FAST_NODE_IP}\",\"gpu_count\":1},\"openpi/pi05-libero-low-mem-finetune\":{\"replica\":0,\"node_ip\":\"${PI05_NODE_IP}\",\"gpu_count\":1}}"
export MINT_DENSE_MODEL_PLACEMENT_JSON='{}'
export MINT_VLLM_MODEL_PLACEMENT_JSON='{}'
export MINT_MEGATRON_MODEL_PLACEMENT_JSON='{}'
export MINT_OPENPI_FAST_WEIGHTS_PATH="$FAST_WEIGHTS_PATH"
export MINT_OPENPI_PI05_WEIGHTS_PATH="$PI05_WEIGHTS_PATH"
export MINT_OPENPI_FAST_CWD="$CODE_ROOT"
export MINT_OPENPI_PI05_CWD="$CODE_ROOT"
export MINT_OPENPI_FAST_CHECKPOINT_BASE_DIR="$OPENPI_CHECKPOINT_BASE_DIR"
export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR="$OPENPI_CHECKPOINT_BASE_DIR"
export MINT_OPENPI_FAST_ASSETS_BASE_DIR="$FAST_ASSETS_PATH"
export MINT_OPENPI_PI05_ASSETS_BASE_DIR="$PI05_ASSETS_PATH"
export MINT_OPENPI_FAST_REQUEST_TIMEOUT_S="1800"
export MINT_OPENPI_FAST_CREATE_SESSION_TIMEOUT_S="1800"
export MINT_OPENPI_FAST_SAVE_TIMEOUT_S="1800"
export MINT_OPENPI_FAST_LOAD_TIMEOUT_S="1800"
export MINT_OPENPI_FAST_ACTION_STARTUP_TIMEOUT_S="${MINT_VLA_OPENPI_FAST_ACTION_STARTUP_TIMEOUT_S:-180}"
export MINT_OPENPI_XLA_FLAGS="${MINT_VLA_OPENPI_XLA_FLAGS:-$OPENPI_XLA_FLAGS_DEFAULT}"
export MINT_API_WORK_QUEUE_PINNED_NODE_IP="$CONTROL_PLANE_IP"
export MINT_CONTROL_PLANE_PINNED_NODE_IP="$CONTROL_PLANE_IP"
export MINT_STARTUP_LEASE_PINNED_NODE_IP="${MINT_STARTUP_LEASE_PINNED_NODE_IP:-$CONTROL_PLANE_IP}"
export MINT_DETACHED_ACTOR_NODE_IP="${MINT_DETACHED_ACTOR_NODE_IP:-$CONTROL_PLANE_IP}"
export TINKER_API_WORK_QUEUE_ACTOR_NAME="$MINT_API_WORK_QUEUE_ACTOR_NAME"
export MINT_API_WORK_QUEUE_FAIL_FAST_ON_PROBE_TIMEOUT="1"
export MINT_API_WORK_QUEUE_DEBUG_LOG_PATH="/tmp/tinker_api_work_queue.${NAME_SUFFIX}.debug.jsonl"
export MINT_QUEUE_EXECUTION_RUNTIME_DEBUG_LOG_PATH="/tmp/tinker_queue_execution_runtime.${NAME_SUFFIX}.debug.jsonl"
export MINT_LOG_FILE="$LOG_FILE"

exec "$RUNTIME_ROOT/host-venv/bin/python" scripts/run_server.py
