#!/bin/sh
set -eu

# Minimal one-shot dev launcher for mint-server.
#
# Contract: supply the smallest possible set of inputs; everything else uses
# code defaults (host/port, LD_LIBRARY_PATH, vLLM child python, HF modules,
# supported-model list). This script deliberately does NOT source the legacy
# shared common.env, because that file hardcodes a fixed code checkout and a
# shared Ray namespace, both of which must be chosen per launch.
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
# Local only: do NOT export MINT_RAY_HEAD_ADDRESS_PATH. The driver must attach as
# a Ray client (ray://...:10001); the file holds a bare IP that the server would
# normalize to the GCS port (...:6379) and try to direct-attach, which hangs on a
# driver-only API host. We read the IP here and set the client/direct addresses.
ray_head_ip_path="${MINT_RAY_HEAD_ADDRESS_PATH:-/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt}"
unset MINT_RAY_HEAD_ADDRESS_PATH || true
export MINT_TMP_ROOT="${MINT_TMP_ROOT:-/vePFS-Mindverse/share/mint/dev/tmp}"

deployment_env="${MINT_DEV_DEPLOYMENT_ENV:-}"
if [ -n "${deployment_env}" ]; then
  if [ ! -r "${deployment_env}" ]; then
    echo "error: MINT_DEV_DEPLOYMENT_ENV not readable: ${deployment_env}" >&2
    exit 1
  fi
  if grep -Eq '^[[:space:]]*export[[:space:]]+(MINT_CODE_ROOT|MINT_RAY_NAMESPACE|TINKER_RAY_NAMESPACE|MINT_RAY_HEAD_ADDRESS_PATH)=' "${deployment_env}"; then
    echo "error: ${deployment_env} must not set MINT_CODE_ROOT, the Ray namespace," >&2
    echo "       or MINT_RAY_HEAD_ADDRESS_PATH (those are per-launch inputs)." >&2
    exit 1
  fi
  set -a
  . "${deployment_env}"
  set +a
  export MINT_CODE_ROOT MINT_RAY_NAMESPACE PFS_RUNTIME_ENV_ROOT
fi

secrets_env="${MINT_DEV_SECRETS_ENV:-/vePFS-Mindverse/share/mint/dev/config/secrets.env}"
if [ -r "${secrets_env}" ]; then
  set -a
  . "${secrets_env}"
  set +a
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
# Driver attaches as a Ray client; actors get the direct GCS address as a hint.
if [ -z "${MINT_RAY_CLIENT_ADDRESS:-}" ] && [ -n "${ray_head_ip}" ]; then
  export MINT_RAY_CLIENT_ADDRESS="ray://${ray_head_ip}:10001"
fi
if [ -z "${RAY_CLIENT_ADDRESS:-}" ] && [ -n "${MINT_RAY_CLIENT_ADDRESS:-}" ]; then
  export RAY_CLIENT_ADDRESS="${MINT_RAY_CLIENT_ADDRESS}"
fi
if [ -z "${RAY_ADDRESS:-}" ] && [ -n "${ray_head_ip}" ]; then
  export RAY_ADDRESS="${ray_head_ip}:6379"
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

echo "=== mint-dev launch contract ===" >&2
echo "MINT_CODE_ROOT            ${MINT_CODE_ROOT}" >&2
echo "MINT_RAY_NAMESPACE        ${MINT_RAY_NAMESPACE}" >&2
echo "PFS_RUNTIME_ENV_ROOT      ${PFS_RUNTIME_ENV_ROOT}" >&2
echo "MINT_RAY_CLIENT_ADDRESS   ${MINT_RAY_CLIENT_ADDRESS}" >&2
echo "RAY_ADDRESS               ${RAY_ADDRESS:-<unset>}" >&2
echo "ray head ip source        ${ray_head_ip_path}" >&2
echo "MINT_TMP_ROOT             ${MINT_TMP_ROOT}" >&2
echo "MINT_DEV_DEPLOYMENT_ENV   ${deployment_env:-<none, code defaults>}" >&2
echo "================================" >&2

"${py}" scripts/bootstrap_control_plane.py
exec "${py}" scripts/run_server.py
