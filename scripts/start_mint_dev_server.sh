#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_mint_dev_server.sh [start|stop|restart|status|healthz]

This script is intended to run on the mint-dev login machine. It:
1. Attaches the local host to the Ray head as a CPU-only worker.
2. Starts tinker-server with the verified mint-dev dev configuration.

Common overrides:
  REPO_DIR=/root/tinker_project/tinker-server
  PYTHON_BIN=.venv31213/bin/python
  RAY_HEAD_ADDRESS=192.168.37.185:6379
  RAY_NODE_IP_ADDRESS=192.168.32.124
  MODEL_NAME=Qwen/Qwen3-30B-A3B-Instruct-2507
  PINNED_NODE_IP=192.168.37.186
  TINKER_RAY_NAMESPACE=tinker_leixiang
  MINT_RAY_NAMESPACE=tinker_leixiang
  LOG_FILE=/tmp/tinker_server.log

Examples:
  scripts/start_mint_dev_server.sh start
  PINNED_NODE_IP=192.168.37.187 scripts/start_mint_dev_server.sh restart
  scripts/start_mint_dev_server.sh status
EOF
}

ACTION="${1:-start}"
if [[ "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="${REPO_DIR:-/root/tinker_project/tinker-server}"
PYTHON_BIN="${PYTHON_BIN:-.venv31213/bin/python}"
LOG_FILE="${LOG_FILE:-/tmp/tinker_server.log}"
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp/ray}"
RAY_ARCHIVE_OLD_TMP="${RAY_ARCHIVE_OLD_TMP:-1}"

RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-192.168.37.185:6379}"
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-192.168.32.124}"
RAY_WORKER_CPUS="${RAY_WORKER_CPUS:-8}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
HF_HOME="${HF_HOME:-/vePFS-Mindverse/share/huggingface}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
PFS_TINKER_PATH="${PFS_TINKER_PATH:-/vePFS-Mindverse/share/code/leixiang/tinker-server}"
TINKER_RAY_NAMESPACE="${TINKER_RAY_NAMESPACE:-tinker_leixiang}"
MINT_RAY_NAMESPACE="${MINT_RAY_NAMESPACE:-tinker_leixiang}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PINNED_NODE_IP="${PINNED_NODE_IP:-192.168.37.186}"
MINT_MEGATRON_NODE_IPS_CSV="${MINT_MEGATRON_NODE_IPS_CSV:-${PINNED_NODE_IP}}"
MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY="${MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY:-1024}"
TINKER_HOST="${TINKER_HOST:-0.0.0.0}"
TINKER_PORT="${TINKER_PORT:-8000}"
HEALTHZ_URL="${HEALTHZ_URL:-http://127.0.0.1:${TINKER_PORT}/api/v1/healthz}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "repo dir not found: ${REPO_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python bin not found or not executable: ${REPO_DIR}/${PYTHON_BIN}" >&2
  exit 1
fi

MINT_MODEL_NODE_IPS_JSON="${MINT_MODEL_NODE_IPS_JSON:-{\"${MODEL_NAME}\":[\"${PINNED_NODE_IP}\"]}}"
MINT_VLLM_PINNED_NODE_IP_JSON="${MINT_VLLM_PINNED_NODE_IP_JSON:-{\"${MODEL_NAME}\":\"${PINNED_NODE_IP}\"}}"

stop_server() {
  pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true
}

archive_ray_tmp() {
  if [[ "${RAY_ARCHIVE_OLD_TMP}" != "1" ]]; then
    return 0
  fi
  if [[ -d "${RAY_TMP_ROOT}" ]]; then
    local archive_path
    archive_path="${RAY_TMP_ROOT}.bak.$(date +%Y%m%d_%H%M%S)"
    mv "${RAY_TMP_ROOT}" "${archive_path}"
    echo "archived ray tmp: ${archive_path}"
  fi
}

start_ray_worker() {
  "${PYTHON_BIN}" -m ray.scripts.scripts stop --force >/tmp/mint_dev_ray_stop.log 2>&1 || true
  archive_ray_tmp
  "${PYTHON_BIN}" -m ray.scripts.scripts start \
    --address="${RAY_HEAD_ADDRESS}" \
    --node-ip-address="${RAY_NODE_IP_ADDRESS}" \
    --num-cpus="${RAY_WORKER_CPUS}" \
    --num-gpus=0 \
    --disable-usage-stats \
    --log-style=record
}

start_server() {
  : > "${LOG_FILE}"
  nohup env \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    HF_HOME="${HF_HOME}" \
    PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE}" \
    TINKER_HOST="${TINKER_HOST}" \
    TINKER_PORT="${TINKER_PORT}" \
    RAY_ADDRESS="${RAY_HEAD_ADDRESS}" \
    RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS}" \
    PFS_TINKER_PATH="${PFS_TINKER_PATH}" \
    TINKER_RAY_NAMESPACE="${TINKER_RAY_NAMESPACE}" \
    MINT_RAY_NAMESPACE="${MINT_RAY_NAMESPACE}" \
    MINT_PERSISTENT_MODELS="${MODEL_NAME}" \
    MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY="${MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY}" \
    MINT_MEGATRON_NODE_IPS_CSV="${MINT_MEGATRON_NODE_IPS_CSV}" \
    MINT_MODEL_NODE_IPS_JSON="${MINT_MODEL_NODE_IPS_JSON}" \
    MINT_VLLM_PINNED_NODE_IP_JSON="${MINT_VLLM_PINNED_NODE_IP_JSON}" \
    "${PYTHON_BIN}" scripts/run_server.py >>"${LOG_FILE}" 2>&1 < /dev/null &
  echo "$!" >/tmp/tinker_server.pid
  echo "server pid: $(cat /tmp/tinker_server.pid)"
}

healthz() {
  curl -fsS "${HEALTHZ_URL}"
}

wait_ready() {
  local attempt
  for attempt in $(seq 1 60); do
    if healthz >/dev/null 2>&1; then
      echo "healthz: ready"
      healthz
      return 0
    fi
    sleep 2
  done
  echo "healthz timeout: ${HEALTHZ_URL}" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  return 1
}

status() {
  echo "run_server:"
  pgrep -af "python scripts/run_server.py|scripts/run_server.py" || true
  echo "---"
  echo "healthz:"
  healthz || true
  echo
  echo "---"
  echo "tail:"
  tail -n 40 "${LOG_FILE}" || true
}

case "${ACTION}" in
  start)
    stop_server
    start_ray_worker
    start_server
    wait_ready
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_ray_worker
    start_server
    wait_ready
    ;;
  status)
    status
    ;;
  healthz)
    healthz
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
