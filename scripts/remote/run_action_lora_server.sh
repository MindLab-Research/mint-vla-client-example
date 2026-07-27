#!/usr/bin/env bash
set -Eeuo pipefail

PORT=30532
GPU_IDS=0,1,2,3,4,5,6,7
RUNTIME_ROOT=
CACHE_DIR=
PRINT_CONFIG=0
while (($#)); do
  case "$1" in
    --port) PORT=${2:?}; shift 2 ;;
    --gpus) GPU_IDS=${2:?}; shift 2 ;;
    --runtime-root) RUNTIME_ROOT=${2:?}; shift 2 ;;
    --cache-dir) CACHE_DIR=${2:?}; shift 2 ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help)
      cat <<'EOF'
usage: run_action_lora_server.sh --runtime-root PATH [options]

Options:
  --port PORT          MINT listen port (default: 30532)
  --gpus CSV           visible GPU IDs (default: 0,1,2,3,4,5,6,7)
  --cache-dir PATH     durable JAX cache; defaults by visible GPU count
  --print-config       validate and print configuration without starting
EOF
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

: "${RUNTIME_ROOT:?--runtime-root is required}"
[[ "$RUNTIME_ROOT" = /* ]] || { echo "--runtime-root must be absolute" >&2; exit 64; }
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT > 0 && PORT < 65536)) || {
  echo "invalid --port: $PORT" >&2; exit 64;
}
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "invalid --gpus CSV: $GPU_IDS" >&2; exit 64;
}
GPU_COMMAS=${GPU_IDS//[^,]/}
GPU_COUNT=$((1 + ${#GPU_COMMAS}))
if [[ -z "$CACHE_DIR" ]]; then
  CACHE_DIR="/vePFS-Mindverse/user/intern/wenxi/results/runtime/jax_compilation_cache/pi05_action_lora_r16_a800_${GPU_COUNT}gpu"
fi
[[ "$CACHE_DIR" = /* ]] || { echo "--cache-dir must be absolute" >&2; exit 64; }

MINT_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16
OPENPI_ROOT=/vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16
PYTHON_BIN=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python
MODEL=openpi/pi05-action-lora-r16-finetune
for path in "$MINT_ROOT" "$OPENPI_ROOT" "$PYTHON_BIN"; do
  [[ -e "$path" ]] || { echo "required path missing: $path" >&2; exit 2; }
done
mkdir -p "$RUNTIME_ROOT"/{openpi_checkpoint_base,runtime_checkpoints,tmp,action_state}
mkdir -p "$CACHE_DIR"

cat >&2 <<EOF
mint_action_lora_server:
  mint_root=$MINT_ROOT
  openpi_root=$OPENPI_ROOT
  runtime_root=$RUNTIME_ROOT
  jax_compilation_cache=$CACHE_DIR
  visible_gpus=$GPU_IDS
  port=$PORT
EOF
if ((PRINT_CONFIG)); then
  exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPU_IDS MINT_HOST=127.0.0.1 MINT_PORT=$PORT
export MINT_SUPPORTED_MODELS=$MODEL MINT_ALLOW_NO_RAY=1 MINT_SKIP_SUPERVISOR=1
export MINT_USAGE_BACKEND=disabled MINT_UVICORN_WORKERS=1
export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1
export MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params
export MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets
export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR="$RUNTIME_ROOT/openpi_checkpoint_base"
export MINT_RUNTIME_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints"
export MINT_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints/persistent_cache"
export MINT_PERSISTENT_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints/persistent_cache"
export MINT_TMP_ROOT="$RUNTIME_ROOT/tmp"
export MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT_BASE="$RUNTIME_ROOT/action_state"

# The worker applies the MINT alias explicitly before constructing any JIT.
# Standard JAX variables are also set so XLA autotune artifacts share the cache.
export MINT_OPENPI_JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true

export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi
export HF_HOME=/vePFS-Mindverse/share/huggingface
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONPATH=$MINT_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/site-packages
export PYTHONNOUSERSITE=1

exec "$PYTHON_BIN" -u -c \
  "import uvicorn; from mint_server.app import app; uvicorn.run(app,host='127.0.0.1',port=$PORT,workers=1,log_level='info')"
