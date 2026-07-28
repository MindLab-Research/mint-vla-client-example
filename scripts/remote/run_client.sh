#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONFIG_FILE=${VLA_CLIENT_CONFIG:-${REPO_ROOT}/config/remote.env}

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

: "${MINT_BASE_URL:=http://127.0.0.1:30532}"
: "${MINT_API_KEY:=tml-dummy}"
: "${MINT_CODE_ROOT:=/vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16}"
: "${MINT_OPENPI_ROOT:=/vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16}"
: "${MINT_GRB_ROOT:=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl}"
: "${MINT_CPU_ROOT:=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/cpu}"
: "${MINT_EXTRA_PYDEPS:=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps}"
: "${MINT_PI_FINETUNE_ROOT:=/vePFS-Mindverse/user/intern/wenxi/pi-finetune}"
: "${OPENPI_DATA_HOME:=/vePFS-Mindverse/share/models/openpi}"
: "${HF_HOME:=/vePFS-Mindverse/share/huggingface}"
: "${MINT_LANCE_DATASET:=/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance}"
: "${VLA_CLIENT_RESULTS_ROOT:=${REPO_ROOT}/results}"
: "${VLA_CLIENT_INFERENCE_ROOT:=${VLA_CLIENT_RESULTS_ROOT}/inference}"
: "${MINT_GLVND_SHIM:=/vePFS-Mindverse/share/zhouch-caches/.cache/openvla_full_a800/glvnd_shim}"

# The host exposes NVIDIA's vendor EGL library but not the vendor-neutral
# GLVND loader. Add the shared shim when present so MuJoCo/EGL evaluation can
# import reliably; training clients simply ignore these variables.
if [[ -f "${MINT_GLVND_SHIM}/libEGL.so.1" ]]; then
  export LD_LIBRARY_PATH="${MINT_GLVND_SHIM}:${LD_LIBRARY_PATH:-}"
  export __EGL_VENDOR_LIBRARY_DIRS="${__EGL_VENDOR_LIBRARY_DIRS:-/usr/share/glvnd/egl_vendor.d}"
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <client-script.py> [arguments...]" >&2
  exit 2
fi

CLIENT_SCRIPT=$1
shift
if [[ "${CLIENT_SCRIPT}" != /* ]]; then
  CLIENT_SCRIPT="${REPO_ROOT}/${CLIENT_SCRIPT}"
fi
if [[ ! -f "${CLIENT_SCRIPT}" ]]; then
  echo "client script does not exist: ${CLIENT_SCRIPT}" >&2
  exit 2
fi

PYTHON_BIN=${MINT_PYTHON_BIN:-${MINT_GRB_ROOT}/host-venv/bin/python}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "remote OpenPI interpreter is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

ACTION_LORA_R16_MODEL=openpi/pi05-action-lora-r16-finetune
SELECTED_MODEL=""
CLIENT_ARGS=("$@")
for ((i = 0; i < ${#CLIENT_ARGS[@]}; i++)); do
  case "${CLIENT_ARGS[i]}" in
    --model)
      if ((i + 1 < ${#CLIENT_ARGS[@]})); then
        SELECTED_MODEL=${CLIENT_ARGS[i + 1]}
      fi
      ;;
    --model=*) SELECTED_MODEL=${CLIENT_ARGS[i]#--model=} ;;
  esac
done
if [[ "${SELECTED_MODEL}" == "${ACTION_LORA_R16_MODEL}" && -z "${MINT_OPENPI_ROOT}" ]]; then
  echo "${ACTION_LORA_R16_MODEL} requires MINT_OPENPI_ROOT to select the isolated rank-16 OpenPI worktree" >&2
  exit 2
fi

for required_path in "${MINT_CODE_ROOT}" "${MINT_EXTRA_PYDEPS}" ${MINT_OPENPI_ROOT:+"${MINT_OPENPI_ROOT}"}; do
  if [[ ! -e "${required_path}" ]]; then
    echo "required remote path is missing: ${required_path}" >&2
    exit 2
  fi
done
if [[ -n "${MINT_OPENPI_ROOT}" && ! -f "${MINT_OPENPI_ROOT}/src/openpi/models/gemma.py" ]]; then
  echo "MINT_OPENPI_ROOT is not an OpenPI worktree root: ${MINT_OPENPI_ROOT}" >&2
  exit 2
fi

export MINT_BASE_URL MINT_API_KEY MINT_OPENPI_ROOT OPENPI_DATA_HOME HF_HOME MINT_LANCE_DATASET
export VLA_CLIENT_RESULTS_ROOT VLA_CLIENT_INFERENCE_ROOT
export JAX_PLATFORMS=cpu
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/scripts/train:${MINT_PI_FINETUNE_ROOT}/case/01_export_video:${MINT_OPENPI_ROOT:+${MINT_OPENPI_ROOT}/src:${MINT_OPENPI_ROOT}/packages/openpi-client/src:}${MINT_CODE_ROOT}:${MINT_EXTRA_PYDEPS}:${MINT_GRB_ROOT}/site-packages:${MINT_CPU_ROOT}/site-packages:${MINT_GRB_ROOT}/src/openpi/src:${MINT_GRB_ROOT}/src/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

if git -C "${REPO_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  VLA_CLIENT_GIT_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
elif [[ -f "${REPO_ROOT}/.vla_mint_commit" ]]; then
  VLA_CLIENT_GIT_COMMIT=$(<"${REPO_ROOT}/.vla_mint_commit")
else
  VLA_CLIENT_GIT_COMMIT=unknown
fi
export VLA_CLIENT_GIT_COMMIT

mkdir -p "${VLA_CLIENT_RESULTS_ROOT}" "${VLA_CLIENT_INFERENCE_ROOT}"

NEEDS_SERVER=1
for argument in "$@"; do
  if [[ "${argument}" == "--dry-run" ]]; then
    NEEDS_SERVER=0
    break
  fi
done

if [[ "${NEEDS_SERVER}" == "1" && "${VLA_SKIP_SERVER_CHECK:-0}" != "1" ]]; then
  HTTP_CODE=$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --max-time 10 "${MINT_BASE_URL%/}/openapi.json" || true)
  if [[ "${HTTP_CODE}" != "200" ]]; then
    echo "MINT server preflight failed: ${MINT_BASE_URL%/}/openapi.json returned ${HTTP_CODE:-no response}" >&2
    echo "Confirm the assigned port with the server owner before training." >&2
    exit 3
  fi
fi

echo "client_repo=${REPO_ROOT}"
echo "client_git_commit=${VLA_CLIENT_GIT_COMMIT}"
echo "mint_base_url=${MINT_BASE_URL}"
echo "default_dataset=${MINT_LANCE_DATASET}"
echo "results_root=${VLA_CLIENT_RESULTS_ROOT}"
echo "inference_root=${VLA_CLIENT_INFERENCE_ROOT}"
exec "${PYTHON_BIN}" -u "${CLIENT_SCRIPT}" "$@"
