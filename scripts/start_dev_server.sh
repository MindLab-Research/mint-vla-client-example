#!/bin/sh
set -eu

# Minimal one-shot dev launcher for mint-server.
#
# Contract: supply the smallest possible set of inputs; everything else uses
# code defaults (host/port, LD_LIBRARY_PATH, vLLM child python, HF modules,
# supported-model list). This script deliberately does NOT source ambient
# shared env files, because code checkout and Ray namespace are per-launch
# identity inputs.
#
# Required (script refuses to start if absent):
#   MINT_CODE_ROOT   Your personal mint-server checkout under /vePFS-Mindverse/share/.
#                    Must be visible to all Ray nodes (head + workers).
#                    Do NOT use: local paths (Ray head can't see them),
#                    /vePFS-Mindverse/user/... (not mounted on Ray nodes), or
#                    /vePFS-Mindverse/share/mint/dev/mint-server (shared, affects everyone).
#                    If validating a local worktree, set MINT_DEV_SOURCE_CHECKOUT
#                    and this launcher will sync it into MINT_CODE_ROOT first.
#
# Derived (override only if needed):
#   MINT_RAY_NAMESPACE        mint_<user>; refuses root/empty.
#   PFS_RUNTIME_ENV_ROOT      prebuilt host-venv (interpreter + torch/vllm/...),
#                             not business code; defaults to the dev runtime.
#   MINT_RAY_HEAD_ADDRESS_PATH  canonical head-address file; server reads the
#                             live head IP from it at init.
#   MINT_TMP_ROOT             scratch root for TMPDIR/cache.
#
# Optional:
#   MINT_DEV_DEPLOYMENT_ENV   extra deployment policy (model list, placement,
#                             prewarm, OTEL). Must NOT set MINT_CODE_ROOT or
#                             MINT_RAY_NAMESPACE; those are rejected below.
#   MINT_DEV_SOURCE_CHECKOUT  optional local/source checkout to rsync into
#                             MINT_CODE_ROOT before launching. Use this when
#                             MINT_CODE_ROOT is the worker-visible mirror of a
#                             local worktree.
#   MINT_DEV_RUN_ENV          per-run overrides (port, run-local task DB,
#                             placement, debug knobs). Same identity keys are
#                             rejected; set them in the launching shell or
#                             packet-owned launch env. If this does not set
#                             placement env vars, the launcher auto-generates
#                             placement from the current Ray dashboard.
#   MINT_DEV_AUTO_PLACEMENT   set to 0 to disable auto placement generation.
#   MINT_DEV_AUTO_PLACEMENT_ENV
#                             generated placement env path; defaults under
#                             MINT_TMP_ROOT/auto-placement/.
#   MINT_DEV_AUTO_PLACEMENT_GPU_COUNT
#                             fallback GPUs per model for auto placement.
#   MINT_DEV_AUTO_PLACEMENT_GPU_COUNTS_JSON
#                             optional JSON mapping model names to GPU counts.
#   MINT_DEV_GC_OLD_ACTORS    set to 1 to run scripts/tools/dev_ray_cleanup.py
#                             gc-stale-actors before bootstrap.
#   MINT_DEV_STOP_EXISTING_PORT_SERVER
#                             set to 1 to kill an existing scripts/run_server.py
#                             listener on MINT_PORT before launch. Refuses to
#                             kill non-Mint processes.
#   MINT_DEV_RESET_CONTROL_PLANE
#                             set to auto/force/1 to run reset-control-plane
#                             before bootstrap. Names and paths come from
#                             MINT_DEV_RESET_* env values.
#   MINT_DEV_BOOTSTRAP_TIMEOUT_S
#                             per-step timeout for bootstrap_control_plane.py.
#                             Defaults to 180s because first detached actor
#                             startup can be slow on a busy shared Ray cluster.
#   MINT_DEV_RESET_SKIP_RAY_WHEN_NO_ALIVE
#                             when set, force reset may skip Ray cleanup
#                             after the dashboard shows no ALIVE actors or
#                             active reset placement groups in this namespace.
#   MINT_DEV_TOPOLOGY_SOURCE_DIR
#                             optional packet-owned topology YAML directory to
#                             sync into dirname(MINT_TOPOLOGY_CONFIG_PATH).

reject_per_launch_keys() {
  env_file="$1"
  if grep -Eq '^[[:space:]]*(export[[:space:]]+)?(MINT_CODE_ROOT|MINT_RAY_NAMESPACE|MINT_RAY_HEAD_ADDRESS_PATH|MINT_VLLM_CHILD_PYTHON_EXECUTABLE|MINT_DEV_SOURCE_CHECKOUT)=' "${env_file}"; then
    echo "error: ${env_file} must not set MINT_CODE_ROOT, the Ray namespace," >&2
    echo "       MINT_RAY_HEAD_ADDRESS_PATH, MINT_VLLM_CHILD_PYTHON_EXECUTABLE," >&2
    echo "       or MINT_DEV_SOURCE_CHECKOUT" >&2
    echo "       (those are per-launch inputs derived by the launcher)." >&2
    exit 1
  fi
}

source_env_file() {
  env_name="$1"
  env_file="$2"
  if [ -z "${env_file}" ]; then
    return 0
  fi
  if [ ! -r "${env_file}" ]; then
    echo "error: ${env_name} not readable: ${env_file}" >&2
    exit 1
  fi
  reject_per_launch_keys "${env_file}"
  set -a
  . "${env_file}"
  set +a
  export MINT_CODE_ROOT MINT_RAY_NAMESPACE PFS_RUNTIME_ENV_ROOT
}

has_explicit_placement_env() {
  [ -n "${MINT_MODEL_PLACEMENT_JSON:-}" ] \
    || [ -n "${MINT_DENSE_MODEL_PLACEMENT_JSON:-}" ] \
    || [ -n "${MINT_VLLM_MODEL_PLACEMENT_JSON:-}" ] \
    || [ -n "${MINT_MEGATRON_MODEL_PLACEMENT_JSON:-}" ]
}

if [ -z "${MINT_CODE_ROOT:-}" ]; then
  echo "error: MINT_CODE_ROOT is required (mint-server checkout to run)." >&2
  echo "       Ask which checkout to use; do not default to the shared dev tree." >&2
  exit 1
fi
if [ -n "${MINT_DEV_SOURCE_CHECKOUT:-}" ]; then
  if [ ! -d "${MINT_DEV_SOURCE_CHECKOUT}" ]; then
    echo "error: MINT_DEV_SOURCE_CHECKOUT does not exist: ${MINT_DEV_SOURCE_CHECKOUT}" >&2
    exit 1
  fi
  mkdir -p "${MINT_CODE_ROOT}"
  if [ "$(cd "${MINT_DEV_SOURCE_CHECKOUT}" && pwd -P)" != "$(cd "${MINT_CODE_ROOT}" && pwd -P)" ]; then
    echo "syncing dev checkout ${MINT_DEV_SOURCE_CHECKOUT} -> ${MINT_CODE_ROOT}" >&2
    rsync -a --delete \
      --exclude '.git' \
      --exclude '.venv' \
      --exclude '.pytest_cache' \
      --exclude '.ruff_cache' \
      --exclude '__pycache__' \
      --exclude 'htmlcov' \
      --exclude '.coverage' \
      "${MINT_DEV_SOURCE_CHECKOUT}/" "${MINT_CODE_ROOT}/"
  fi
elif [ ! -d "${MINT_CODE_ROOT}" ]; then
  echo "error: MINT_CODE_ROOT does not exist: ${MINT_CODE_ROOT}" >&2
  exit 1
fi
cd "${MINT_CODE_ROOT}"

mint_git_source="${MINT_DEV_SOURCE_CHECKOUT:-${MINT_CODE_ROOT}}"
mint_git_sha="${MINT_GIT_SHA:-}"
if [ -z "${mint_git_sha}" ] && command -v git >/dev/null 2>&1; then
  mint_git_sha=$(git -C "${mint_git_source}" rev-parse HEAD 2>/dev/null || true)
fi
if [ -n "${mint_git_sha}" ]; then
  export MINT_GIT_SHA="${mint_git_sha}"
fi

mint_user="${MINT_DEV_USER:-${USER:-$(id -un)}}"
if [ -z "${mint_user}" ] || [ "${mint_user}" = "root" ]; then
  echo "error: cannot derive a non-root dev Ray namespace (user=${mint_user:-unset})." >&2
  echo "       Set MINT_RAY_NAMESPACE=mint_<you> or MINT_DEV_USER=<you>." >&2
  exit 1
fi
export MINT_RAY_NAMESPACE="${MINT_RAY_NAMESPACE:-mint_${mint_user}}"
case "${MINT_RAY_NAMESPACE}" in
  ""|mint|root|mint_root)
    echo "error: refusing shared/root namespace: ${MINT_RAY_NAMESPACE}" >&2
    exit 1
    ;;
esac

# Derive a stable port from the username if MINT_PORT is not set.
# Hash the namespace to a port in [30000, 40000) so multiple users
# can run dev servers concurrently without port conflicts.
if [ -z "${MINT_PORT:-}" ]; then
  _port_hash=0
  _port_ns="${MINT_RAY_NAMESPACE}"
  _port_i=1
  while [ "$_port_i" -le "${#_port_ns}" ]; do
    _port_c=$(printf '%d' "'$(printf '%.1s' "$_port_ns" "$_port_i")")
    _port_hash=$(( (_port_hash * 31 + _port_c) % 10000 ))
    _port_i=$(( _port_i + 1 ))
  done
  MINT_PORT=$(( 30000 + _port_hash ))
  echo "MINT_PORT derived from namespace: ${MINT_PORT} (override with MINT_PORT=<n>)" >&2
fi
export MINT_PORT

export PFS_RUNTIME_ENV_ROOT="${PFS_RUNTIME_ENV_ROOT:-/vePFS-Mindverse/share/mint/dev/runtime}"
export PFS_HF_MODULES_PATH="${PFS_HF_MODULES_PATH:-/vePFS-Mindverse/share/huggingface/modules}"
# Do NOT set MINT_RAY_JOB_WORKING_DIR. Even PFS paths cause Ray to package and
# upload the directory (~100-240 MB) through the Ray job/runtime-env path.
# Workers find mint_server via PFS_PYTHONPATH (built from MINT_CODE_ROOT) which
# is passed as env_vars to every actor's runtime_env.
# Local only: do NOT export MINT_RAY_HEAD_ADDRESS_PATH or RAY_ADDRESS. The
# driver gets the direct GCS address through MINT_RAY_GCS_ADDRESS, and Ray
# worker bootstrap treats inherited RAY_ADDRESS as an instruction to nested
# direct-attach before user runtime_env can blank it.
ray_head_ip_path="${MINT_RAY_HEAD_ADDRESS_PATH:-/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt}"
unset MINT_RAY_HEAD_ADDRESS_PATH || true
unset RAY_ADDRESS || true
export MINT_TMP_ROOT="${MINT_TMP_ROOT:-/vePFS-Mindverse/share/mint/dev/tmp}"

# ---------------------------------------------------------------------------
# Dev deployment defaults
#
# These values were previously in /vePFS-Mindverse/share/mint/dev/config/common.env.
# They are now baked into the launcher so that a bare `start_dev_server.sh`
# (without MINT_DEV_DEPLOYMENT_ENV) still gets sensible dev defaults.
#
# Secrets (OTLP API keys, etc.) should be placed in a secrets.env file and
# loaded via MINT_DEV_DEPLOYMENT_ENV=/path/to/secrets.env.
# ---------------------------------------------------------------------------

# Auth mode.
export MINT_AUTH_MODE="${MINT_AUTH_MODE:-no-auth}"

# Vendored deps used by some backends.
export MINT_BUMBLEBEE_REPO_PATH="${MINT_BUMBLEBEE_REPO_PATH:-/vePFS-Mindverse/share/mint/dev/vendor/bumblebee}"
export MINT_ACTOR_EXTRA_PYTHONPATH="${MINT_ACTOR_EXTRA_PYTHONPATH:-/vePFS-Mindverse/share/mint/dev/vendor/flash-attn-current}"

# MoE LoRA export policy.
export MINT_MOE_LORA_SHARED_EXPERT_EXPORT="${MINT_MOE_LORA_SHARED_EXPERT_EXPORT:-0}"
export MINT_MOE_LORA_SPARSE_EXPERT_EXPORT="${MINT_MOE_LORA_SPARSE_EXPERT_EXPORT:-1}"

# Model advertisement + persistence/prewarm policy.
export MINT_SUPPORTED_MODELS="${MINT_SUPPORTED_MODELS:-Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-4B-Thinking-2507,Qwen/Qwen3-30B-A3B-Instruct-2507}"
export MINT_PERSISTENT_MODELS="${MINT_PERSISTENT_MODELS:-Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-4B-Thinking-2507,Qwen/Qwen3-30B-A3B-Instruct-2507}"
export MINT_PERSISTENT_PREWARM_INFERENCE="${MINT_PERSISTENT_PREWARM_INFERENCE:-1}"
export MINT_PERSISTENT_PREWARM_TRAINING="${MINT_PERSISTENT_PREWARM_TRAINING:-1}"
export MINT_PERSISTENT_TRAIN_LORA_RANK="${MINT_PERSISTENT_TRAIN_LORA_RANK:-64}"
export MINT_PERSISTENT_TRAIN_LR="${MINT_PERSISTENT_TRAIN_LR:-5e-5}"
export MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S="${MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S:-3600}"
export MINT_SAVE_LORA_TIMEOUT_S="${MINT_SAVE_LORA_TIMEOUT_S:-1800}"
export MINT_SCHEDULER_ENABLE="${MINT_SCHEDULER_ENABLE:-1}"

# vLLM serving flags.
export MINT_ROUTER_REPLAY_MODE="${MINT_ROUTER_REPLAY_MODE:-disabled}"
export MINT_VLLM_ENABLE_CHUNKED_PREFILL="${MINT_VLLM_ENABLE_CHUNKED_PREFILL:-1}"
export MINT_VLLM_ENABLE_PREFIX_CACHING="${MINT_VLLM_ENABLE_PREFIX_CACHING:-1}"
export MINT_VLLM_FULLY_SHARDED_LORAS="${MINT_VLLM_FULLY_SHARDED_LORAS:-1}"
export MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE="${MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE:-0}"
export MINT_VLLM_ADMISSION_CONTROL="${MINT_VLLM_ADMISSION_CONTROL:-1}"

# Logging + observability.
export MINT_LOG_MAX_BYTES="${MINT_LOG_MAX_BYTES:-10485760}"
export MINT_LOG_BACKUP_COUNT="${MINT_LOG_BACKUP_COUNT:-5}"
export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-mint}"
export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-otel.macaron.xin:4317}"
export OTEL_EXPORTER_OTLP_INSECURE="${OTEL_EXPORTER_OTLP_INSECURE:-false}"
export OTEL_METRIC_EXPORT_INTERVAL_MS="${OTEL_METRIC_EXPORT_INTERVAL_MS:-10000}"
export OTEL_LOG_LEVEL="${OTEL_LOG_LEVEL:-DEBUG}"
export MINT_HEALTHZ_RAY_TIMEOUT_S="${MINT_HEALTHZ_RAY_TIMEOUT_S:-30.0}"
export MINT_VLLM_REQUEST_TIMING="${MINT_VLLM_REQUEST_TIMING:-1}"
export MINT_TIMING_DIAG="${MINT_TIMING_DIAG:-1}"
export MINT_VERL_DIAGNOSTICS="${MINT_VERL_DIAGNOSTICS:-1}"
export MINT_LOG_KILL_STACK="${MINT_LOG_KILL_STACK:-1}"

# Resource queues for GPU worker submission.
export MINT_MEGATRON_VOLC_RESOURCE_QUEUE_ID="${MINT_MEGATRON_VOLC_RESOURCE_QUEUE_ID:-q-20251126180002-26lwz}"
export MINT_VLLM_VOLC_RESOURCE_QUEUE_ID="${MINT_VLLM_VOLC_RESOURCE_QUEUE_ID:-q-20251126180002-26lwz}"

# Data + scratch.
export MINT_CHECKPOINT_DIR="${MINT_CHECKPOINT_DIR:-/tos-mindverse/tinker_checkpoints}"
export MINT_RUNTIME_CHECKPOINT_DIR="${MINT_RUNTIME_CHECKPOINT_DIR:-/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints}"
export MINT_TASK_PAYLOAD_ROOT_DIR="${MINT_TASK_PAYLOAD_ROOT_DIR:-/vePFS-Mindverse/share/mint/dev/data/task-state/payloads}"

export MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY="${MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY:-1024}"

# Dev defaults for optional services.
export MINT_USAGE_BACKEND="${MINT_USAGE_BACKEND:-disabled}"

# Load optional deployment env (OTLP API keys, etc.) and per-run overrides.
# Set MINT_DEV_DEPLOYMENT_ENV=/path/to/env explicitly when needed.
deployment_env="${MINT_DEV_DEPLOYMENT_ENV:-}"
source_env_file "MINT_DEV_DEPLOYMENT_ENV" "${deployment_env}"

run_env="${MINT_DEV_RUN_ENV:-}"
source_env_file "MINT_DEV_RUN_ENV" "${run_env}"

if [ "${MINT_DEV_STOP_EXISTING_PORT_SERVER:-0}" = "1" ]; then
  port="${MINT_PORT:-8000}"
  if command -v ss >/dev/null 2>&1; then
    port_pids=$(
      ss -ltnp "sport = :${port}" 2>/dev/null \
        | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
        | sort -u
    )
    for previous_pid in ${port_pids}; do
      cmd=$(ps -p "${previous_pid}" -o args= 2>/dev/null || true)
      case "${cmd}" in
        *"scripts/run_server.py"*)
          echo "stopping previous mint dev server on port ${port} pid=${previous_pid}" >&2
          kill "${previous_pid}" 2>/dev/null || true
          sleep 2
          ;;
        *)
          echo "error: port ${port} is occupied by non-Mint process pid=${previous_pid} cmd=${cmd}" >&2
          exit 1
          ;;
      esac
    done
  else
    echo "warning: ss not available; cannot preflight existing listener on port ${port}" >&2
  fi
fi

api_tmp_root="${MINT_TMP_ROOT}/api/${mint_user}"
api_tmp_link="/tmp/mda"
mkdir -p "${api_tmp_root}"
if [ -L "${api_tmp_link}" ] || [ -f "${api_tmp_link}" ]; then
  rm -f "${api_tmp_link}"
elif [ -e "${api_tmp_link}" ]; then
  echo "error: refusing to replace non-link temp path: ${api_tmp_link}" >&2
  exit 1
fi
ln -s "${api_tmp_root}" "${api_tmp_link}"
export TMPDIR="${api_tmp_link}/t"
export XDG_CACHE_HOME="${api_tmp_link}/c"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

ray_head_ip=""
if [ -r "${ray_head_ip_path}" ]; then
  ray_head_ip=$(tr -d '[:space:]' < "${ray_head_ip_path}")
fi
# Connection mode: direct attach only (GCS address).
if [ -z "${MINT_RAY_GCS_ADDRESS:-}" ] && [ -n "${ray_head_ip}" ]; then
  export MINT_RAY_GCS_ADDRESS="${ray_head_ip}:6379"
fi
unset RAY_CLIENT_ADDRESS 2>/dev/null || true
unset MINT_RAY_CLIENT_ADDRESS 2>/dev/null || true
if [ -z "${MINT_RAY_GCS_ADDRESS:-}" ]; then
  echo "error: no Ray head address. Expected an IP in ${ray_head_ip_path}" >&2
  echo "       or set MINT_RAY_GCS_ADDRESS=<head>:6379 for direct attach." >&2
  exit 1
fi

py="${PFS_RUNTIME_ENV_ROOT}/cpu/base-python/bin/python3.13"
if [ ! -x "${py}" ]; then
  echo "error: runtime python not found: ${py}" >&2
  exit 1
fi
vllm_worker_python="${MINT_CODE_ROOT}/scripts/vllm_worker_python.py"
if [ ! -r "${vllm_worker_python}" ]; then
  echo "error: vLLM worker python wrapper not found: ${vllm_worker_python}" >&2
  exit 1
fi
# Always derive this from MINT_CODE_ROOT. A stale shell or policy env pointing at
# another checkout bypasses this repo's Ray bootstrap sanitization.
export MINT_VLLM_CHILD_PYTHON_EXECUTABLE="${vllm_worker_python}"

auto_placement_env="${MINT_DEV_AUTO_PLACEMENT_ENV:-}"
if [ "${MINT_DEV_AUTO_PLACEMENT:-1}" != "0" ] && ! has_explicit_placement_env; then
  if [ -z "${ray_head_ip}" ]; then
    echo "error: cannot auto-generate placement; no Ray head IP in ${ray_head_ip_path}" >&2
    echo "       Set explicit placement env vars or MINT_DEV_AUTO_PLACEMENT=0 to bypass." >&2
    exit 1
  fi
  if [ -z "${auto_placement_env}" ]; then
    auto_placement_safe_ns=$(printf '%s' "${MINT_RAY_NAMESPACE}" | sed 's#[^A-Za-z0-9_.-]#_#g')
    auto_placement_env="${MINT_TMP_ROOT}/auto-placement/${auto_placement_safe_ns}.env"
  fi
  echo "auto-generating Mint dev placement from Ray head ${ray_head_ip}" >&2
  "${py}" scripts/tools/gen_dev_placement.py \
    --head-ip "${ray_head_ip}" \
    --models-from-env \
    --gpu-count "${MINT_DEV_AUTO_PLACEMENT_GPU_COUNT:-1}" \
    --output "${auto_placement_env}" \
    --force
  source_env_file "MINT_DEV_AUTO_PLACEMENT_ENV" "${auto_placement_env}"
elif has_explicit_placement_env; then
  echo "using explicit Mint placement env; auto placement skipped" >&2
else
  echo "Mint dev auto placement disabled; continuing without generated placement" >&2
fi

echo "=== mint-dev launch contract ===" >&2
echo "MINT_CODE_ROOT            ${MINT_CODE_ROOT}" >&2
echo "MINT_DEV_SOURCE_CHECKOUT  ${MINT_DEV_SOURCE_CHECKOUT:-<none>}" >&2
echo "MINT_GIT_SHA             ${MINT_GIT_SHA:-<unknown>}" >&2
echo "MINT_RAY_NAMESPACE        ${MINT_RAY_NAMESPACE}" >&2
echo "MINT_PORT                 ${MINT_PORT}" >&2
echo "PFS_RUNTIME_ENV_ROOT      ${PFS_RUNTIME_ENV_ROOT}" >&2
echo "MINT_VLLM_CHILD_PYTHON    ${MINT_VLLM_CHILD_PYTHON_EXECUTABLE}" >&2
echo "MINT_RAY_GCS_ADDRESS      ${MINT_RAY_GCS_ADDRESS:-<unset>}" >&2
echo "MINT_CONTROL_PLANE_NODE   ${MINT_CONTROL_PLANE_NODE_IP:-<auto>}" >&2
echo "RAY_ADDRESS               ${RAY_ADDRESS:-<unset>}" >&2
echo "ray head ip source        ${ray_head_ip_path}" >&2
echo "MINT_TMP_ROOT             ${MINT_TMP_ROOT}" >&2
echo "MINT_DEV_DEPLOYMENT_ENV   ${deployment_env:-<none, code defaults>}" >&2
echo "MINT_DEV_RUN_ENV          ${run_env:-<none>}" >&2
echo "MINT_DEV_AUTO_PLACEMENT   ${MINT_DEV_AUTO_PLACEMENT:-1}" >&2
echo "MINT_DEV_AUTO_PLACEMENT_ENV ${auto_placement_env:-<none>}" >&2
echo "================================" >&2

if [ -n "${MINT_DEV_TOPOLOGY_SOURCE_DIR:-}" ]; then
  if [ -z "${MINT_TOPOLOGY_CONFIG_PATH:-}" ]; then
    echo "error: MINT_DEV_TOPOLOGY_SOURCE_DIR requires MINT_TOPOLOGY_CONFIG_PATH" >&2
    exit 1
  fi
  if [ ! -d "${MINT_DEV_TOPOLOGY_SOURCE_DIR}" ]; then
    echo "error: MINT_DEV_TOPOLOGY_SOURCE_DIR does not exist: ${MINT_DEV_TOPOLOGY_SOURCE_DIR}" >&2
    exit 1
  fi
  topology_target_dir=$(dirname "${MINT_TOPOLOGY_CONFIG_PATH}")
  mkdir -p "${topology_target_dir}"
  echo "syncing topology env ${MINT_DEV_TOPOLOGY_SOURCE_DIR} -> ${topology_target_dir}" >&2
  rsync -a --delete --include '*/' --include '*.yaml' --exclude '*' \
    "${MINT_DEV_TOPOLOGY_SOURCE_DIR}/" "${topology_target_dir}/"
fi

export PYTHONPATH="${PFS_RUNTIME_ENV_ROOT}/cpu/site-packages:${MINT_CODE_ROOT}:${PYTHONPATH:-}"

"${py}" - <<'PY' >&2
import json
import os
import pathlib
import sys

repo = pathlib.Path(os.environ["MINT_CODE_ROOT"]).resolve()
sys.path.insert(0, str(repo))

from mint_server import ray_utils
from mint_server.runtime_env import bootstrap_runtime_pythonpath

pythonpath = bootstrap_runtime_pythonpath(os.environ, repo_root=str(repo))
job_env = ray_utils.client_job_runtime_env()

summary = {
    "cwd": os.getcwd(),
    "python": sys.executable,
    "pythonpath_entries": len([part for part in pythonpath.split(":") if part]),
    "pythonpath_has_code_root": str(repo) in pythonpath.split(":"),
    "ray_client_job_runtime_env": {
        "keys": sorted((job_env or {}).keys()) if isinstance(job_env, dict) else [],
        "env_vars": sorted((job_env or {}).get("env_vars", {}).keys())
        if isinstance(job_env, dict) and isinstance((job_env or {}).get("env_vars"), dict)
        else [],
        "working_dir": (job_env or {}).get("working_dir") if isinstance(job_env, dict) else None,
        "py_modules_count": len((job_env or {}).get("py_modules", []))
        if isinstance(job_env, dict) and isinstance((job_env or {}).get("py_modules"), list)
        else 0,
    },
}
print("=== mint-dev python/ray preflight ===")
print(json.dumps(summary, sort_keys=True))
print("=====================================")
PY

if [ "${MINT_DEV_GC_OLD_ACTORS:-0}" = "1" ]; then
  echo "=== mint-dev stale actor cleanup ===" >&2
  "${py}" scripts/tools/dev_ray_cleanup.py gc-stale-actors
  echo "====================================" >&2
fi

case "${MINT_DEV_RESET_CONTROL_PLANE:-0}" in
  0|false|False|FALSE|no|No|NO|"")
    ;;
  *)
    echo "=== mint-dev control-plane reset ===" >&2
    "${py}" scripts/tools/dev_ray_cleanup.py reset-control-plane
    echo "====================================" >&2
    ;;
esac

bootstrap_timeout_s="${MINT_DEV_BOOTSTRAP_TIMEOUT_S:-180}"
"${py}" scripts/bootstrap_control_plane.py --timeout-s "${bootstrap_timeout_s}"
exec "${py}" scripts/run_server.py
