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
#                    Sync your checkout first, e.g.:
#                    rsync -a --delete <your-checkout>/ /vePFS-Mindverse/share/<your-path>/
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
#   MINT_DEV_RUN_ENV          per-run overrides (port, run-local task DB,
#                             placement, debug knobs). Same identity keys are
#                             rejected; set them in the launching shell/script.

reject_per_launch_keys() {
  env_file="$1"
  if grep -Eq '^[[:space:]]*(export[[:space:]]+)?(MINT_CODE_ROOT|MINT_RAY_NAMESPACE|TINKER_RAY_NAMESPACE|MINT_RAY_HEAD_ADDRESS_PATH|MINT_VLLM_CHILD_PYTHON_EXECUTABLE)=' "${env_file}"; then
    echo "error: ${env_file} must not set MINT_CODE_ROOT, the Ray namespace," >&2
    echo "       MINT_RAY_HEAD_ADDRESS_PATH, or MINT_VLLM_CHILD_PYTHON_EXECUTABLE" >&2
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
if [ ! -d "${MINT_CODE_ROOT}" ]; then
  echo "error: MINT_CODE_ROOT does not exist: ${MINT_CODE_ROOT}" >&2
  exit 1
fi
cd "${MINT_CODE_ROOT}"

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

deployment_env="${MINT_DEV_DEPLOYMENT_ENV:-}"
source_env_file "MINT_DEV_DEPLOYMENT_ENV" "${deployment_env}"

run_env="${MINT_DEV_RUN_ENV:-}"
source_env_file "MINT_DEV_RUN_ENV" "${run_env}"

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

py="${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python"
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
echo "MINT_RAY_NAMESPACE        ${MINT_RAY_NAMESPACE}" >&2
echo "PFS_RUNTIME_ENV_ROOT      ${PFS_RUNTIME_ENV_ROOT}" >&2
echo "MINT_VLLM_CHILD_PYTHON    ${MINT_VLLM_CHILD_PYTHON_EXECUTABLE}" >&2
echo "MINT_RAY_CLIENT_ADDRESS   ${MINT_RAY_CLIENT_ADDRESS}" >&2
echo "MINT_RAY_GCS_ADDRESS      ${MINT_RAY_GCS_ADDRESS:-<unset>}" >&2
echo "RAY_ADDRESS               ${RAY_ADDRESS:-<unset>}" >&2
echo "ray head ip source        ${ray_head_ip_path}" >&2
echo "MINT_TMP_ROOT             ${MINT_TMP_ROOT}" >&2
echo "MINT_DEV_DEPLOYMENT_ENV   ${deployment_env:-<none, code defaults>}" >&2
echo "MINT_DEV_RUN_ENV          ${run_env:-<none>}" >&2
echo "================================" >&2

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

"${py}" scripts/bootstrap_control_plane.py
exec "${py}" scripts/run_server.py
