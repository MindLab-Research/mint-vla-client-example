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
#                             packet-owned launch env.
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
#                             startup can be slow on a busy shared Ray Client.
#   MINT_DEV_RESET_SKIP_RAY_WHEN_NO_ALIVE
#                             when set, force reset may skip Ray Client cleanup
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

export PFS_RUNTIME_ENV_ROOT="${PFS_RUNTIME_ENV_ROOT:-/vePFS-Mindverse/share/mint/dev/runtime}"
export PFS_HF_MODULES_PATH="${PFS_HF_MODULES_PATH:-/vePFS-Mindverse/share/huggingface/modules}"
# Do NOT set MINT_RAY_JOB_WORKING_DIR. Even PFS paths cause Ray to package and
# upload the directory (~100-240 MB) over the Ray Client connection.
# Workers find mint_server via PFS_PYTHONPATH (built from MINT_CODE_ROOT) which
# is passed as env_vars to every actor's runtime_env.
# Local only: do NOT export MINT_RAY_HEAD_ADDRESS_PATH or RAY_ADDRESS. The driver
# must attach as a Ray client (ray://...:10001), and Ray worker bootstrap treats
# inherited RAY_ADDRESS as an instruction to nested direct-attach before user
# runtime_env can blank it. Mint code that needs the direct GCS address reads the
# explicit MINT_RAY_GCS_ADDRESS value instead.
ray_head_ip_path="${MINT_RAY_HEAD_ADDRESS_PATH:-/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt}"
unset MINT_RAY_HEAD_ADDRESS_PATH || true
unset RAY_ADDRESS || true
export MINT_TMP_ROOT="${MINT_TMP_ROOT:-/vePFS-Mindverse/share/mint/dev/tmp}"

# Dev defaults for optional services. Set to "postgres" only when a billing
# database is available; "disabled" avoids the startup health-check dependency.
export MINT_USAGE_BACKEND="${MINT_USAGE_BACKEND:-disabled}"

# Models the supervisor should pre-create workers for (comma-separated).
# Empty by default; set via deployment env or explicit override for the
# models you want to test.
export MINT_SUPPORTED_MODELS="${MINT_SUPPORTED_MODELS:-}"

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
# Driver attaches as a Ray client; Mint control-plane code gets the direct GCS
# address only through an explicit non-Ray bootstrap variable.
if [ -z "${MINT_RAY_CLIENT_ADDRESS:-}" ] && [ -n "${ray_head_ip}" ]; then
  export MINT_RAY_CLIENT_ADDRESS="ray://${ray_head_ip}:10001"
fi
if [ -z "${RAY_CLIENT_ADDRESS:-}" ] && [ -n "${MINT_RAY_CLIENT_ADDRESS:-}" ]; then
  export RAY_CLIENT_ADDRESS="${MINT_RAY_CLIENT_ADDRESS}"
fi
if [ -z "${MINT_RAY_GCS_ADDRESS:-}" ] && [ -n "${ray_head_ip}" ]; then
  export MINT_RAY_GCS_ADDRESS="${ray_head_ip}:6379"
fi
if [ -z "${MINT_RAY_CLIENT_ADDRESS:-}" ]; then
  echo "error: no Ray head address. Expected an IP in ${ray_head_ip_path}" >&2
  echo "       or set MINT_RAY_CLIENT_ADDRESS=ray://<head>:10001 explicitly." >&2
  exit 1
fi

py="${PFS_RUNTIME_ENV_ROOT}/cpu/base-python/bin/python3.12"
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

echo "=== mint-dev launch contract ===" >&2
echo "MINT_CODE_ROOT            ${MINT_CODE_ROOT}" >&2
echo "MINT_DEV_SOURCE_CHECKOUT  ${MINT_DEV_SOURCE_CHECKOUT:-<none>}" >&2
echo "MINT_GIT_SHA             ${MINT_GIT_SHA:-<unknown>}" >&2
echo "MINT_RAY_NAMESPACE        ${MINT_RAY_NAMESPACE}" >&2
echo "PFS_RUNTIME_ENV_ROOT      ${PFS_RUNTIME_ENV_ROOT}" >&2
echo "MINT_VLLM_CHILD_PYTHON    ${MINT_VLLM_CHILD_PYTHON_EXECUTABLE}" >&2
echo "MINT_RAY_CLIENT_ADDRESS   ${MINT_RAY_CLIENT_ADDRESS}" >&2
echo "MINT_RAY_GCS_ADDRESS      ${MINT_RAY_GCS_ADDRESS:-<unset>}" >&2
echo "MINT_CONTROL_PLANE_NODE   ${MINT_CONTROL_PLANE_NODE_IP:-<auto>}" >&2
echo "RAY_ADDRESS               ${RAY_ADDRESS:-<unset>}" >&2
echo "ray head ip source        ${ray_head_ip_path}" >&2
echo "MINT_TMP_ROOT             ${MINT_TMP_ROOT}" >&2
echo "MINT_DEV_DEPLOYMENT_ENV   ${deployment_env:-<none, code defaults>}" >&2
echo "MINT_DEV_RUN_ENV          ${run_env:-<none>}" >&2
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
