#!/usr/bin/env bash
set -euo pipefail
PERSONAL_ROOT=${PERSONAL_ROOT:-/vePFS-Mindverse/user/intern/rongenz/pi05-finetune}
SHARE_ROOT=${SHARE_ROOT:-/vePFS-Mindverse/share/intern/rongenz/pi05-finetune}
CLIENT_ROOT=${CLIENT_ROOT:-$PERSONAL_ROOT/mint-vla-client-example-state54-replay-v1}
SOURCE_RELEASE=${SOURCE_RELEASE:-/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example-state44/results/datasets/current_replayed_strict_allobjects_matched_new_image_state44_v1_20260801}
COMMON_DATA_RELEASE=${COMMON_DATA_RELEASE:-$SHARE_ROOT/results/datasets/state44_replay_cube1_cube2_state54_data_v1_20260803_v2}
FEATURE_RELEASE=${FEATURE_RELEASE:-$SHARE_ROOT/results/datasets/state44_replay_cube1_cube2_state54_snapshot_features_v1_20260803_v2}
SPLIT_ROOT=${SPLIT_ROOT:-$SHARE_ROOT/results/datasets/state44_replay_cube1_cube2_state54_split_v1_20260803}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SHARE_ROOT/results/training/state54_replay_snapshot_train_split_norm_v1_20260804}
EXPECTED_SOURCE_RELEASE_SHA=${EXPECTED_SOURCE_RELEASE_SHA:-e05598979dbc08827f169b44f4fa655a01b8af85efe23682fc620b7ba5c544bd}
EXPECTED_FEATURE_RELEASE_SHA=${EXPECTED_FEATURE_RELEASE_SHA:-5e0c83d791595f29393710f23b5b9935e4d361e676251e1d7df4a759c8d54be5}
EXPECTED_SPLIT_MANIFEST_SHA=${EXPECTED_SPLIT_MANIFEST_SHA:-694e3d405aaa0d574d2a64a8ca01ba3581340c03dd61dd697b02ae047d65a326}
EXPECTED_TRAIN_ROWS_SHA=${EXPECTED_TRAIN_ROWS_SHA:-196143fa7d45ca5be8ace5d688b4025cc753e7963a7b42490b1f79e9081c7e01}
EXPECTED_WINDOW_SHA=${EXPECTED_WINDOW_SHA:-4d6a4042316a6eb6a5a42bb3912bb043cc8c12747bc61d30acaa4f9bce21bdda}
sha256_check(){ local p=$1 e=$2 a; a=$(sha256sum "$p"|awk "{print \$1}");[[ "$a" == "$e" ]]||{ echo "SHA mismatch: $p: $a != $e" >&2;exit 2;};}
sha256_check "$SOURCE_RELEASE/release.json" "$EXPECTED_SOURCE_RELEASE_SHA"
sha256_check "$FEATURE_RELEASE/release.json" "$EXPECTED_FEATURE_RELEASE_SHA"
sha256_check "$SPLIT_ROOT/split_manifest.json" "$EXPECTED_SPLIT_MANIFEST_SHA"
sha256_check "$SPLIT_ROOT/train_rows.csv" "$EXPECTED_TRAIN_ROWS_SHA"
sha256_check "$COMMON_DATA_RELEASE/contact_pm60_window_manifest.json" "$EXPECTED_WINDOW_SHA"
[[ $(awk -F, "NR==1{print NF}" "$SPLIT_ROOT/train_rows.csv") == 813 ]] || { echo "train split must contain 813 rows" >&2;exit 2;}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "output exists: $OUTPUT_ROOT" >&2;exit 2;}
source "$PERSONAL_ROOT/pi05_common_env.sh"
mkdir -p "$OUTPUT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CLIENT_ROOT:$CLIENT_ROOT/scripts/train:$PERSONAL_ROOT/mint-state54-formal-v1:$PERSONAL_ROOT/openpi-dev-v3/src:$PERSONAL_ROOT/openpi-dev-v3/packages/openpi-client/src:/vePFS-Mindverse/user/intern/wenxi/pi-finetune/case/01_export_video:$EXTRA_PYDEPS:$GRB/site-packages"
cd "$CLIENT_ROOT"
"$PY" scripts/train/compute_state54_population.py \
  --lance-dataset "$SOURCE_RELEASE/dataset.lance" \
  --rows-csv "$SPLIT_ROOT/train_rows.csv" \
  --contact-window-manifest "$COMMON_DATA_RELEASE/contact_pm60_window_manifest.json" \
  --gesture-index "$SOURCE_RELEASE/gesture_index.json" \
  --output-dir "$OUTPUT_ROOT" \
  --state54-replay-feature-release "$FEATURE_RELEASE" \
  --state54-replay-feature-release-sha256 "$EXPECTED_FEATURE_RELEASE_SHA" \
  --mode both --max-token-len 256 --state-noise-std 0 --augmentation-seed 43 \
  | tee "$OUTPUT_ROOT/prepare.log"
sha256sum "$OUTPUT_ROOT/norm_stats.json" "$OUTPUT_ROOT/norm_summary.json" "$OUTPUT_ROOT/token_audit.json" > "$OUTPUT_ROOT/SHA256SUMS"
