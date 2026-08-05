#!/usr/bin/env bash
set -Eeuo pipefail

PORT=30532
GPU_IDS=0,1,2,3,4,5,6,7
RUNTIME_ROOT=
CACHE_DIR=
MINT_ROOT=${MINT_CODE_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint-state41-28dof}
OPENPI_ROOT=${MINT_OPENPI_ROOT:-/vePFS-Mindverse/user/intern/wenxi/openpi-state41-28dof}
PYTHON_BIN=${MINT_PYTHON_BIN:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python}
MODEL=openpi/pi05-action-lora-r16-state41-28dof-finetune
NORM_STATS_PATH=${MINT_OPENPI_PI05_NORM_STATS:-}
ENABLE_JAX_PERSISTENT_CACHE=0
PRINT_CONFIG=0
while (($#)); do
  case "$1" in
    --port) PORT=${2:?}; shift 2 ;;
    --gpus) GPU_IDS=${2:?}; shift 2 ;;
    --runtime-root) RUNTIME_ROOT=${2:?}; shift 2 ;;
    --cache-dir) CACHE_DIR=${2:?}; shift 2 ;;
    --mint-root) MINT_ROOT=${2:?}; shift 2 ;;
    --openpi-root) OPENPI_ROOT=${2:?}; shift 2 ;;
    --python-bin) PYTHON_BIN=${2:?}; shift 2 ;;
    --model) MODEL=${2:?}; shift 2 ;;
    --norm-stats) NORM_STATS_PATH=${2:?}; shift 2 ;;
    --enable-jax-persistent-cache) ENABLE_JAX_PERSISTENT_CACHE=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help)
      cat <<'EOF'
usage: run_action_lora_server.sh --runtime-root PATH [options]

Options:
  --port PORT          MINT listen port (default: 30532)
  --gpus CSV           visible GPU IDs (default: 0,1,2,3,4,5,6,7)
  --cache-dir PATH     durable JAX cache when persistent caching is enabled
  --mint-root PATH     MINT checkout (default: MINT_CODE_ROOT or project path)
  --openpi-root PATH   paired OpenPI checkout (default: MINT_OPENPI_ROOT or project path)
  --python-bin PATH    GPU runtime Python (default: MINT_PYTHON_BIN or project path)
  --model ID           enabled Action-LoRA model identity (default: state41 28DoF)
  --norm-stats PATH    locked norm_stats.json used for checkpoint assets
  --enable-jax-persistent-cache
                     opt in to JAX persistent executable serialization
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
  CACHE_PROFILE=state41
  if [[ "$MODEL" == openpi/pi05-action-lora-r16-state45-phase-28dof-finetune ]]; then
    CACHE_PROFILE=state45_phase
  fi
  CACHE_DIR="/vePFS-Mindverse/user/intern/wenxi/results/runtime/jax_compilation_cache/pi05_action_lora_r16_${CACHE_PROFILE}_a800_${GPU_COUNT}gpu"
fi
[[ "$CACHE_DIR" = /* ]] || { echo "--cache-dir must be absolute" >&2; exit 64; }

[[ "$MINT_ROOT" = /* && "$OPENPI_ROOT" = /* && "$PYTHON_BIN" = /* ]] || {
  echo "--mint-root, --openpi-root, and --python-bin must be absolute" >&2; exit 64;
}
for path in "$MINT_ROOT" "$OPENPI_ROOT"; do
  [[ -e "$path" ]] || { echo "required path missing: $path" >&2; exit 2; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "runtime Python is not executable: $PYTHON_BIN" >&2; exit 2; }
mkdir -p "$RUNTIME_ROOT"/{openpi_checkpoint_base,runtime_checkpoints,tmp,action_state}
ASSETS_BASE_DIR=${MINT_OPENPI_PI05_ASSETS_BASE_DIR:-/vePFS-Mindverse/share/code/conley/openpi/assets}
if [[ "$MODEL" == openpi/pi05-action-lora-r16-state41-28dof-finetune || \
      "$MODEL" == openpi/pi05-action-lora-r16-state45-phase-28dof-finetune ]]; then
  [[ -n "$NORM_STATS_PATH" ]] || {
    echo "--norm-stats is required for State41/State45 Action-LoRA servers" >&2
    exit 64
  }
fi
if [[ -n "$NORM_STATS_PATH" ]]; then
  [[ "$NORM_STATS_PATH" = /* ]] || { echo "--norm-stats must be absolute" >&2; exit 64; }
  [[ -f "$NORM_STATS_PATH" ]] || { echo "norm stats file missing: $NORM_STATS_PATH" >&2; exit 2; }
  ASSETS_BASE_DIR="$RUNTIME_ROOT/openpi_assets"
  NORM_DEST="$ASSETS_BASE_DIR/pi05_libero/physical-intelligence/libero/norm_stats.json"
  mkdir -p "$(dirname "$NORM_DEST")"
  cp "$NORM_STATS_PATH" "$NORM_DEST"
  NORM_STATS_SHA256=$(sha256sum "$NORM_DEST" | awk '{print $1}')
else
  NORM_STATS_SHA256=
fi
if ((ENABLE_JAX_PERSISTENT_CACHE)); then
  mkdir -p "$CACHE_DIR"
fi

cat >&2 <<EOF
mint_action_lora_server:
  mint_root=$MINT_ROOT
  openpi_root=$OPENPI_ROOT
  runtime_root=$RUNTIME_ROOT
  jax_persistent_executable_cache=$ENABLE_JAX_PERSISTENT_CACHE
  jax_compilation_cache=$([[ "$ENABLE_JAX_PERSISTENT_CACHE" == 1 ]] && printf '%s' "$CACHE_DIR" || printf 'disabled')
  visible_gpus=$GPU_IDS
  model=$MODEL
  port=$PORT
  assets_base_dir=$ASSETS_BASE_DIR
  norm_stats_path=${NORM_STATS_PATH:-none}
  norm_stats_sha256=${NORM_STATS_SHA256:-none}
EOF
if ((PRINT_CONFIG)); then
  exit 0
fi

export CUDA_VISIBLE_DEVICES=$GPU_IDS MINT_HOST=127.0.0.1 MINT_PORT=$PORT
export MINT_SUPPORTED_MODELS=$MODEL MINT_ALLOW_NO_RAY=1 MINT_SKIP_SUPERVISOR=1
export MINT_USAGE_BACKEND=disabled MINT_UVICORN_WORKERS=1
export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1
export MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params
export MINT_OPENPI_PI05_ASSETS_BASE_DIR="$ASSETS_BASE_DIR"
export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR="$RUNTIME_ROOT/openpi_checkpoint_base"
export MINT_RUNTIME_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints"
export MINT_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints/persistent_cache"
export MINT_PERSISTENT_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints/persistent_cache"
export MINT_TMP_ROOT="$RUNTIME_ROOT/tmp"
export MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT_BASE="$RUNTIME_ROOT/action_state"

# pi0.5's multi-GB executable cannot be serialized reliably on this runtime.
# Compile/JIT normally by default; opt in only when the runtime has been proven
# to serialize the executable successfully.
if ((ENABLE_JAX_PERSISTENT_CACHE)); then
  export MINT_OPENPI_JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
  export JAX_ENABLE_COMPILATION_CACHE=true
  export JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
  export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
  export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
else
  unset MINT_OPENPI_JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
  unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
  unset JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES
  unset JAX_RAISE_PERSISTENT_CACHE_ERRORS
fi

export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi
export HF_HOME=/vePFS-Mindverse/share/huggingface
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONPATH=$MINT_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/site-packages
export PYTHONNOUSERSITE=1

exec "$PYTHON_BIN" -u -c \
  "import uvicorn; from mint_server.app import app; uvicorn.run(app,host='127.0.0.1',port=$PORT,workers=1,log_level='info')"
