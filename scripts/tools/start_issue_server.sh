#!/bin/bash
set -euo pipefail

# Start one isolated mint-dev issue server from the shared dev config template.
# This script is meant to run on the mint-dev host after the issue worktree has
# been synced to PFS. Model work is admitted through ModelWorkScheduler and
# executed by Ray-backed model runtime actors.

: "${ISSUE_SERVER_ROOT:=/root/mint_project/mint-server-issue-416}"

_INPUT_RAY_ADDRESS="${RAY_ADDRESS:-}"
_INPUT_MINT_RAY_CLIENT_ADDRESS="${MINT_RAY_CLIENT_ADDRESS:-}"
_INPUT_MINT_CODE_ROOT="${MINT_CODE_ROOT:-}"
_INPUT_MINT_RAY_PY_MODULES_CSV="${MINT_RAY_PY_MODULES_CSV:-}"
_INPUT_MINT_VLLM_CHILD_PYTHON_EXECUTABLE="${MINT_VLLM_CHILD_PYTHON_EXECUTABLE:-}"
_INPUT_MINT_TOPOLOGY_CONFIG_PATH="${MINT_TOPOLOGY_CONFIG_PATH:-}"

cd "${ISSUE_SERVER_ROOT}"
dev_config_env="${MINT_DEV_CONFIG_ENV:-/vePFS-Mindverse/share/mint/dev/config/common.env}"
if [[ ! -r "${dev_config_env}" ]]; then
  echo "missing dev config: ${dev_config_env}" >&2
  exit 1
fi
. "${dev_config_env}"

export RAY_ADDRESS="${_INPUT_RAY_ADDRESS:-${RAY_ADDRESS:-}}"
export MINT_RAY_CLIENT_ADDRESS="${_INPUT_MINT_RAY_CLIENT_ADDRESS:-${MINT_RAY_CLIENT_ADDRESS:-${RAY_ADDRESS}}}"
export MINT_CODE_ROOT="${_INPUT_MINT_CODE_ROOT:-${MINT_CODE_ROOT:-}}"
export MINT_RAY_PY_MODULES_CSV="${_INPUT_MINT_RAY_PY_MODULES_CSV:-}"
export MINT_VLLM_CHILD_PYTHON_EXECUTABLE="${_INPUT_MINT_VLLM_CHILD_PYTHON_EXECUTABLE:-}"
export MINT_TOPOLOGY_CONFIG_PATH="${_INPUT_MINT_TOPOLOGY_CONFIG_PATH:-${MINT_TOPOLOGY_CONFIG_PATH:-}}"

: "${RAY_ADDRESS:?set RAY_ADDRESS=ray://<head>:10001}"
: "${MINT_RAY_CLIENT_ADDRESS:?set MINT_RAY_CLIENT_ADDRESS=ray://<head>:10001}"
: "${MINT_CODE_ROOT:?set MINT_CODE_ROOT=/vePFS-Mindverse/share/code/<user>/mint-server-issue-<n>}"
: "${ISSUE_NAMESPACE:?set ISSUE_NAMESPACE=mint_<user>_issue_<n>}"
: "${ISSUE_PORT:?set ISSUE_PORT=10416}"
: "${ISSUE_LOG_FILE:?set ISSUE_LOG_FILE=/tmp/mint_server_issue.log}"
: "${ISSUE_SUPPORTED_MODELS:?set ISSUE_SUPPORTED_MODELS=Qwen/Qwen3-30B-A3B-Instruct-2507}"

export RAY_ADDRESS
export MINT_RAY_CLIENT_ADDRESS
export MINT_CODE_ROOT
export MINT_RAY_PY_MODULES_CSV="${MINT_RAY_PY_MODULES_CSV:-${MINT_CODE_ROOT}/mint_server}"
export MINT_VLLM_CHILD_PYTHON_EXECUTABLE="${MINT_VLLM_CHILD_PYTHON_EXECUTABLE:-${MINT_CODE_ROOT}/scripts/vllm_worker_python.py}"

export MINT_RAY_NAMESPACE="${ISSUE_NAMESPACE}"

ISSUE_NAME_SUFFIX="$(printf '%s' "${ISSUE_NAMESPACE}" | tr -c '[:alnum:]' '_')"
export MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME="mint_model_work_scheduler_${ISSUE_NAME_SUFFIX}"
export MINT_TASK_STATE_STORE_ACTOR_NAME="mint_task_state_store_${ISSUE_NAME_SUFFIX}"
export MINT_MAINTENANCE_CRON_ACTOR_NAME="mint_maintenance_cron_${ISSUE_NAME_SUFFIX}"
export MINT_MODEL_WORK_SCHEDULER_DEBUG_LOG_PATH="/tmp/mint_model_work_scheduler.${ISSUE_NAME_SUFFIX}.debug.jsonl"

export MINT_PORT="${ISSUE_PORT}"
export MINT_API_KEY="${MINT_API_KEY:-dummy}"
export MINT_LOG_FILE="${ISSUE_LOG_FILE}"

export MINT_SUPPORTED_MODELS="${ISSUE_SUPPORTED_MODELS}"
export MINT_UVICORN_WORKERS=1
export MINT_OAI_PRELOAD_TOKENIZERS=0
export MINT_DISABLE_MINT_ROUTE=1

if [[ "${ISSUE_STARTUP_PRINT_ENV:-0}" == "1" ]]; then
  env | grep -E '^(RAY_ADDRESS|MINT_RAY_CLIENT_ADDRESS|MINT_CODE_ROOT|MINT_RAY_PY_MODULES_CSV|MINT_VLLM_CHILD_PYTHON_EXECUTABLE|MINT_RAY_NAMESPACE|MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME|MINT_TASK_STATE_STORE_ACTOR_NAME|MINT_MAINTENANCE_CRON_ACTOR_NAME|MINT_DISABLE_MINT_ROUTE|MINT_API_KEY|MINT_PORT|MINT_LOG_FILE|MINT_TOPOLOGY_CONFIG_PATH)=' | sort
  exit 0
fi

"${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/bootstrap_control_plane.py
exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py
