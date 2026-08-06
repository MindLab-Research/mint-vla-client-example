#!/usr/bin/env bash
set -euo pipefail
TASK=${1:?task_a or task_b}
MODE=${2:-formal}
ROOT=/vePFS-Mindverse/user/intern/rongenz/pi05-finetune
SHARE=/vePFS-Mindverse/share/intern/rongenz/pi05-finetune
C=$ROOT/mint-vla-client-example-state54-replay-v1
M=$ROOT/mint-state54-formal-v1
O=$ROOT/openpi-dev-v3
SOURCE_ROOT=/vePFS-Mindverse/user/intern/wenxi/results/datas/28dof_manohand/release/mano_28d_native_replay_state41_rgb_v1
SOURCE=$SOURCE_ROOT/mano_28d_native_replay_state41_rgb_v1.lance
EXPERIMENTS=$SHARE/results/datasets/state56_native28_training_experiments_v1_20260805
NORMS=$SHARE/results/training/state56_native28_training_experiments_norms_v1_20260805
RUNTIME=$SHARE/runtime-checkpoints-state56-experiments-v1
EVAL_ROOT=$SHARE/results/eval/state56_native28_aug010_20k_mode4_v1
TRAIN_CLIENT_SHA=428bbc959686832ffe7ecfdffc4d378241362a92
MINT_SHA=d8005e7097e4c6ef46f7026eda7bcbc7990a308d
OPENPI_SHA=2d43d317552c9268bb40204ae7599e12508760c5
case "$TASK" in
 task_a)
  TASK_DIR=task_a_cube1_03_all_train;SELECTION=train_selection.json;WINDOWS=train_contact_windows.json
  CHECKPOINT=state56_aug010_task_a_cube1_03_all_train_step20000_v1
  OUT_NAME=task_a_all15
  ;;
 task_b)
  TASK_DIR=task_b_cube1_all_seed_disjoint;SELECTION=validation_selection.json;WINDOWS=validation_contact_windows.json
  CHECKPOINT=state56_aug010_task_b_cube1_all_seed_disjoint_step20000_v1
  OUT_NAME=task_b_validation115
  ;;
 *) echo "unsupported task: $TASK" >&2; exit 2;;
esac
case "$MODE" in
 smoke) MAX_FRAMES=3;OUT=$EVAL_ROOT/integration_${TASK}_3frames_v1;;
 formal) MAX_FRAMES=0;OUT=$EVAL_ROOT/$OUT_NAME;;
 *) echo "unsupported mode: $MODE" >&2;exit 2;;
esac
D=$EXPERIMENTS/$TASK_DIR
N=$NORMS/$TASK_DIR
MODEL_ROOT=$(find "$RUNTIME" -type d -name "$CHECKPOINT" -print)
[[ $(printf '%s\n' "$MODEL_ROOT" | sed '/^$/d' | wc -l) -eq 1 ]]
MODEL_PATH=$MODEL_ROOT/sampler
[[ -f "$MODEL_PATH/metadata.json" && -f "$MODEL_PATH/mint_pi05_profile.json" ]]
[[ $(git -C "$M" rev-parse HEAD) == "$MINT_SHA" ]]
[[ $(git -C "$O" rev-parse HEAD) == "$OPENPI_SHA" ]]
[[ -z $(git -C "$C" status --porcelain)$(git -C "$M" status --porcelain)$(git -C "$O" status --porcelain) ]]
EVAL_CLIENT_SHA=$(git -C "$C" rev-parse HEAD)
ROWS=$(/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python -c "import json;print(','.join(str(x['release_row_index']) for x in json.load(open('$D/$SELECTION'))['rows']))")
sha(){ sha256sum "$1"|awk '{print $1}'; }
SV_SHA=$(sha "$SOURCE_ROOT/release_verification.json")
SEL_SHA=$(sha "$D/$SELECTION")
WIN_SHA=$(sha "$D/$WINDOWS")
NORM_SHA=$(sha "$N/norm_stats.json")
DC_SHA=$(sha "$N/data_contract_v2.json")
[[ ! -e "$OUT" ]]
mkdir -p "$EVAL_ROOT"
source "$ROOT/pi05_common_env.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface JAX_PLATFORMS=cpu
export LD_LIBRARY_PATH=/vePFS-Mindverse/share/zhouch-caches/.cache/openvla_full_a800/glvnd_shim:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=4
export MANORL_EXPECTED_COMMIT=e17f0122decddffc348ec10d0ed42552a0540e1b MANORL_ALL_ASSETS_COMMIT=e7910212e54367008ecb7484e5e9354e822de03e
export PYTHONPATH=$C:$C/scripts/train:$C/scripts/eval:$M:$O/src:$O/packages/openpi-client/src:/vePFS-Mindverse/user/intern/wenxi/pi-finetune/case/01_export_video:$EXTRA_PYDEPS:$GRB/site-packages
LOG=$EVAL_ROOT/${OUT_NAME}_${MODE}.log
status=1
archive_launcher(){
 if [[ -d "$OUT" ]]; then cp "$0" "$OUT/launch.sh"; printf '%s\n' "$status" > "$OUT/exit_code.txt"; fi
}
trap archive_launcher EXIT
cd "$C"
set +e
$PY -u scripts/eval/infer_mano_mode4_state56_batch.py \
 --base-url http://127.0.0.1:30540 --api-key tml-dummy --model openpi/pi05-action-lora-r16-state56-28dof-finetune --model-path "$MODEL_PATH" --owner-id "rongenz-state56-${TASK}-20k-eval" \
 --lance-dataset "$SOURCE" --source-release-verification "$SOURCE_ROOT/release_verification.json" --source-release-verification-sha256 "$SV_SHA" \
 --row-indices "$ROWS" --selection "$D/$SELECTION" --selection-sha256 "$SEL_SHA" \
 --norm-stats-dir "$N" --norm-sha-expected "$NORM_SHA" --state56-data-contract "$N/data_contract_v2.json" --state56-data-contract-sha256 "$DC_SHA" \
 --contact-window-manifest "$D/$WINDOWS" --contact-window-manifest-sha256 "$WIN_SHA" --contact-context-frames 100 \
 --chunk-stride 5 --temporal-decay .4 --act-batch-size 4 --row-batch-size 4 --max-warm-request-seconds 10 --max-frames "$MAX_FRAMES" --width 640 --height 360 \
 --training-client-commit "$TRAIN_CLIENT_SHA" --evaluation-client-commit "$EVAL_CLIENT_SHA" --backend-commit "$MINT_SHA" --model-commit "$OPENPI_SHA" --output-dir "$OUT" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
