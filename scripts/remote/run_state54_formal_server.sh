#!/usr/bin/env bash
set -euo pipefail
SEED=${1:?training seed required}
GPUS=${2:-4,5,6,7}
PORT=${3:-30540}
RUNTIME_ROOT=${4:?absolute runtime root required}
SERVER_LOG=${5:?server log required}
case "$SEED" in 42|43|44) ;; *) echo "seed must be42/43/44" >&2; exit 64;; esac
ROOT=/vePFS-Mindverse/user/intern/rongenz/pi05-finetune
MINT_ROOT=$ROOT/mint-state54-formal-v1
OPENPI_ROOT=$ROOT/openpi-dev-v3
NORM_DIR=/vePFS-Mindverse/share/intern/rongenz/pi05-finetune/results/training/state54_replay_snapshot_train_split_norm_v1_20260804
EXPECTED_NORM_SHA=d6adccb613e555b5754367f38e11c91e7223b35d96e94dabceb6a170bca92c5c
EXPECTED_MINT_COMMIT=b9d5f7112ca64dae7311ff9ff754c1ee384f0166
EXPECTED_OPENPI_COMMIT=4515b26372882fda2d4ca2363d0a90774515f9bd
[[ "$RUNTIME_ROOT" = /* && "$SERVER_LOG" = /* ]] || { echo "runtime/log paths must be absolute" >&2;exit 64;}
[[ $(git -C "$MINT_ROOT" rev-parse HEAD) == "$EXPECTED_MINT_COMMIT" ]]
[[ $(git -C "$OPENPI_ROOT" rev-parse HEAD) == "$EXPECTED_OPENPI_COMMIT" ]]
[[ -z $(git -C "$MINT_ROOT" status --porcelain)$(git -C "$OPENPI_ROOT" status --porcelain) ]]
[[ $(sha256sum "$NORM_DIR/norm_stats.json"|awk '{print $1}') == "$EXPECTED_NORM_SHA" ]]
source "$ROOT/pi05_common_env.sh"
mkdir -p "$RUNTIME_ROOT"/{checkpoint_base,runtime_checkpoints,tmp,action_state} "$(dirname "$SERVER_LOG")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPUS" MINT_HOST=127.0.0.1 MINT_PORT="$PORT"
export MINT_CODE_ROOT="$MINT_ROOT" MINT_SUPPORTED_MODELS=openpi/pi05-action-lora-r16-state54-finetune
export MINT_ALLOW_NO_RAY=1 MINT_SKIP_SUPERVISOR=1 MINT_USAGE_BACKEND=disabled MINT_UVICORN_WORKERS=1
export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1
export MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params
export MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets
export MINT_OPENPI_PI05_CHECKPOINT_NORM_STATS_DIR="$NORM_DIR"
export MINT_OPENPI_PI05_SEED="$SEED"
export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR="$RUNTIME_ROOT/checkpoint_base"
export MINT_RUNTIME_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints"
export MINT_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints/persistent_cache"
export MINT_PERSISTENT_CHECKPOINT_DIR="$RUNTIME_ROOT/runtime_checkpoints/persistent_cache"
export MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT_BASE="$RUNTIME_ROOT/action_state"
export MINT_TMP_ROOT="$RUNTIME_ROOT/tmp" TMPDIR="$RUNTIME_ROOT/tmp"
unset MINT_OPENPI_JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE JAX_COMPILATION_CACHE_DIR
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONPATH="$MINT_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$EXTRA_PYDEPS:$GRB/site-packages"
unset JAX_PLATFORMS
cd "$MINT_ROOT"
printf 'state54 formal server seed=%s gpus=%s port=%s runtime=%s mint=%s openpi=%s norm=%s\n' "$SEED" "$GPUS" "$PORT" "$RUNTIME_ROOT" "$EXPECTED_MINT_COMMIT" "$EXPECTED_OPENPI_COMMIT" "$EXPECTED_NORM_SHA"
exec "$PY" -u -c "import uvicorn;from mint_server.app import app;uvicorn.run(app,host='127.0.0.1',port=$PORT,workers=1,log_level='info')" >"$SERVER_LOG" 2>&1
