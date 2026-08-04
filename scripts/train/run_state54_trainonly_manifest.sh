#!/usr/bin/env bash
set -euo pipefail
MANIFEST=${1:?launch manifest JSON required}
EXPECTED_MANIFEST_SHA=${2:?launch manifest SHA256 required}
ROOT=/vePFS-Mindverse/user/intern/rongenz/pi05-finetune
SHARE=/vePFS-Mindverse/share/intern/rongenz/pi05-finetune
CLIENT=$ROOT/mint-vla-client-example-state54-replay-v1
MINT=$ROOT/mint-state54-formal-v1
OPENPI=$ROOT/openpi-dev-v3
source "$ROOT/pi05_common_env.sh"
[[ $(sha256sum "$MANIFEST"|awk '{print $1}') == "$EXPECTED_MANIFEST_SHA" ]] || { echo "launch manifest SHA mismatch" >&2;exit 2;}
eval "$($PY - "$MANIFEST" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1]))
required={
 "status":"frozen_not_launched", "state_contract":"mano_object_dynamics_state54_v1",
 "batch_size":8, "learning_rate":5e-5, "state_noise_std":0.05, "target_noise_std":0.0,
 "sampling_strategy":"coverage", "slate_size":16, "anchors_per_row":8,
 "augmentation_seed":43,
 "train_rows":813, "train_active_frames":423450,
}
for k,v in required.items():
 if m.get(k)!=v: raise ValueError(f"manifest {k}: {m.get(k)!r} != {v!r}")
if m.get("seed") not in (42,43,44): raise ValueError("invalid seed")
if m.get("steps") not in (20,150000): raise ValueError("steps must be smoke20 or formal150000")
if m["steps"]==150000 and m.get("checkpoint_every")!=25000: raise ValueError("formal checkpoint schedule mismatch")
if m["steps"]==20 and m.get("checkpoint_every")!=0: raise ValueError("smoke must not use periodic checkpoints")
keys=("run_id","output_root","client_commit","mint_commit","openpi_commit","data_contract","data_contract_sha256","formal_protocol","formal_protocol_sha256","coverage_schedule","coverage_schedule_sha256","train_rows_csv","train_rows_csv_sha256","dataset","gesture_index","contact_window_manifest","feature_release","feature_release_sha256","norm_dir","norm_sha256","seed","augmentation_seed","steps","checkpoint_every","base_url")
for k in keys:
 if k not in m: raise ValueError(f"manifest missing {k}")
for k in keys:
 print(f"{k.upper()}={shlex.quote(str(m[k]))}")
PY
)"
[[ $(git -C "$CLIENT" rev-parse HEAD) == "$CLIENT_COMMIT" ]]
[[ $(git -C "$MINT" rev-parse HEAD) == "$MINT_COMMIT" ]]
[[ $(git -C "$OPENPI" rev-parse HEAD) == "$OPENPI_COMMIT" ]]
[[ -z $(git -C "$CLIENT" status --porcelain)$(git -C "$MINT" status --porcelain)$(git -C "$OPENPI" status --porcelain) ]]
[[ $(sha256sum "$DATA_CONTRACT"|awk '{print $1}') == "$DATA_CONTRACT_SHA256" ]]
[[ $(sha256sum "$FORMAL_PROTOCOL"|awk '{print $1}') == "$FORMAL_PROTOCOL_SHA256" ]]
[[ $(sha256sum "$COVERAGE_SCHEDULE"|awk '{print $1}') == "$COVERAGE_SCHEDULE_SHA256" ]]
[[ $(sha256sum "$TRAIN_ROWS_CSV"|awk '{print $1}') == "$TRAIN_ROWS_CSV_SHA256" ]]
[[ $(sha256sum "$NORM_DIR/norm_stats.json"|awk '{print $1}') == "$NORM_SHA256" ]]
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "output exists: $OUTPUT_ROOT" >&2;exit 2;}
python3 - "$BASE_URL" <<'PY'
import socket,sys
from urllib.parse import urlparse
u=urlparse(sys.argv[1]);s=socket.create_connection((u.hostname,u.port),timeout=2);s.close()
PY
mkdir -p "$OUTPUT_ROOT"
cp "$MANIFEST" "$OUTPUT_ROOT/launch_manifest.json"
printf '%s  launch_manifest.json\n' "$EXPECTED_MANIFEST_SHA" > "$OUTPUT_ROOT/launch_manifest.sha256"
export PYTHONDONTWRITEBYTECODE=1
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface
export LD_LIBRARY_PATH=/vePFS-Mindverse/share/zhouch-caches/.cache/openvla_full_a800/glvnd_shim:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$CLIENT:$CLIENT/scripts/train:$CLIENT/scripts/eval:/vePFS-Mindverse/user/intern/wenxi/pi-finetune/case/01_export_video:$OPENPI/src:$OPENPI/packages/openpi-client/src:$MINT:$EXTRA_PYDEPS:$GRB/site-packages"
export VLA_CLIENT_GIT_COMMIT="$CLIENT_COMMIT"
POPULATION_ROWS=$(cat "$TRAIN_ROWS_CSV")
ARGS=(
  --base-url "$BASE_URL" --api-key tml-dummy
  --model openpi/pi05-action-lora-r16-state54-finetune
  --lance-dataset "$DATASET" --target-lance-dataset "$DATASET"
  --row-indices "$POPULATION_ROWS"
  --action-source urdf_target_absolute --action-horizon 10
  --language-conditioning gesture --gesture-index "$GESTURE_INDEX"
  --frame-window contact --contact-context-frames 60
  --contact-window-manifest "$CONTACT_WINDOW_MANIFEST" --missing-contact-policy error
  --norm-stats-dir "$NORM_DIR" --state-contract mano_object_dynamics_state54_v1
  --state54-replay-feature-release "$FEATURE_RELEASE"
  --state54-replay-feature-release-sha256 "$FEATURE_RELEASE_SHA256"
  --state54-data-contract "$DATA_CONTRACT" --state54-data-contract-sha256 "$DATA_CONTRACT_SHA256"
  --steps "$STEPS" --batch-size 8 --learning-rate 5e-5
  --seed "$SEED" --augmentation-seed "$AUGMENTATION_SEED" --state-noise-std "$STATE_NOISE_STD" --target-noise-std "$TARGET_NOISE_STD"
  --sampling-strategy coverage --coverage-anchors-per-row 8 --slate-size 16
  --datum-cache-size 4096 --row-cache-size 813 --preload-selected-rows
  --batch-producers 2 --batch-build-workers 4 --prefetch-batches 2
  --save-path "${RUN_ID}_step${STEPS}"
  --metrics-jsonl "$OUTPUT_ROOT/metrics.jsonl" --output-json "$OUTPUT_ROOT/result.json"
)
if (( CHECKPOINT_EVERY > 0 )); then
  ARGS+=(--checkpoint-every "$CHECKPOINT_EVERY" --checkpoint-save-path-template "${RUN_ID}_step{step}")
fi
cd "$CLIENT"
set +e
"$PY" -u scripts/train/train_cube1_01_compare.py "${ARGS[@]}" 2>&1 | tee "$OUTPUT_ROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$OUTPUT_ROOT/exit_code.txt"
exit "$STATUS"
