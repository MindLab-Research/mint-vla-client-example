#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONFIG_FILE=${VLA_CLIENT_CONFIG:-${REPO_ROOT}/config/remote.env}
if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi
PROFILE_REPORT=${1:?usage: run_state41_cube1_gesture03_20k.sh <profile-report.json> [run-id]}
RUN_ID=${2:-state41_cube1_gesture03_bschema_contact_pm100_20k}

readarray -t CONTRACT < <(python3 - "$PROFILE_REPORT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(); r=json.loads(p.read_text())
if r.get('status')!='passed' or r.get('state_dim')!=41 or r.get('action_dim')!=32:
    raise SystemExit('state41 profile report is not passed state41/action32')
if r.get('object')!='cube1' or str(r.get('gesture'))!='03':
    raise SystemExit('profile report is not cube1 gesture03')
if r.get('frame_window')!='contact' or r.get('contact_context_frames')!=100:
    raise SystemExit('profile report is not contact ±100')
print(r['dataset'])
print(r['norm']['directory'])
print(r['norm']['sha256'])
print(r['contact_window_manifest'])
print(','.join(str(x['release_row_index']) for x in r['selection']))
PY
)
DATASET=${CONTRACT[0]}
NORM_DIR=${CONTRACT[1]}
NORM_SHA=${CONTRACT[2]}
WINDOW_MANIFEST=${CONTRACT[3]}
ROW_INDICES=${CONTRACT[4]}

OUT_ROOT=${VLA_CLIENT_RESULTS_ROOT:-${REPO_ROOT}/results}/training/${RUN_ID}
mkdir -p "$OUT_ROOT"
RESULT_JSON="$OUT_ROOT/result.json"
METRICS_JSONL="$OUT_ROOT/metrics.jsonl"
DRIVER_LOG="$OUT_ROOT/driver.log"
if [[ -e "$RESULT_JSON" || -e "$METRICS_JSONL" ]]; then
  echo "refusing existing training outputs under $OUT_ROOT" >&2
  exit 2
fi

MODEL=openpi/pi05-action-lora-r16-state41-28dof-finetune
STEPS=${STATE41_STEPS:-20000}
BATCH_SIZE=${STATE41_BATCH_SIZE:-8}
LEARNING_RATE=${STATE41_LEARNING_RATE:-1e-4}
CHECKPOINT_EVERY=${STATE41_CHECKPOINT_EVERY:-4000}
SLATE_SIZE=${STATE41_SLATE_SIZE:-15}
ROW_CACHE_SIZE=${STATE41_ROW_CACHE_SIZE:-15}
DATUM_CACHE_SIZE=${STATE41_DATUM_CACHE_SIZE:-4096}
PREFETCH_BATCHES=${STATE41_PREFETCH_BATCHES:-2}
BATCH_PRODUCERS=${STATE41_BATCH_PRODUCERS:-1}
BATCH_BUILD_WORKERS=${STATE41_BATCH_BUILD_WORKERS:-4}
STATE_NOISE_STD=${STATE41_STATE_NOISE_STD:-0}
FINAL_PATH="${RUN_ID}_step${STEPS}"
CHECKPOINT_TEMPLATE="${RUN_ID}_step{step}"
EXTRA_ARGS=()
if [[ "${STATE41_TRAIN_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run --augmentation-audit-samples 32)
fi
{
  printf 'started=%s\n' "$(date -Is)"
  printf 'profile_report=%s\n' "$(realpath "$PROFILE_REPORT")"
  printf 'dataset=%s\nrows=%s\nnorm_sha256=%s\n' "$DATASET" "$ROW_INDICES" "$NORM_SHA"
  printf 'model=%s\nsteps=%s\ncheckpoint_every=%s\nstate_noise_std=%s\nbatch_size=%s\nlearning_rate=%s\nbatch_producers=%s\nprefetch_batches=%s\nbatch_build_workers=%s\ndatum_cache_size=%s\n' "$MODEL" "$STEPS" "$CHECKPOINT_EVERY" "$STATE_NOISE_STD" "$BATCH_SIZE" "$LEARNING_RATE" "$BATCH_PRODUCERS" "$PREFETCH_BATCHES" "$BATCH_BUILD_WORKERS" "$DATUM_CACHE_SIZE"
} | tee "$DRIVER_LOG"

exec "${REPO_ROOT}/scripts/remote/run_client.sh" \
  scripts/train/train_cube1_01_compare.py \
  --model "$MODEL" \
  --lance-dataset "$DATASET" \
  --row-indices "$ROW_INDICES" \
  --action-source urdf_target_absolute \
  --language-conditioning object_only \
  --state-contract state41 \
  --action-horizon 10 \
  --frame-window contact \
  --contact-context-frames 100 \
  --contact-window-manifest "$WINDOW_MANIFEST" \
  --missing-contact-policy error \
  --norm-stats-dir "$NORM_DIR" \
  --norm-sha-expected "$NORM_SHA" \
  --steps "$STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --batch-size "$BATCH_SIZE" \
  --sampling-strategy coverage \
  --slate-size "$SLATE_SIZE" \
  --coverage-anchors-per-row 8 \
  --row-cache-size "$ROW_CACHE_SIZE" \
  --preload-selected-rows \
  --datum-cache-size "$DATUM_CACHE_SIZE" \
  --prefetch-batches "$PREFETCH_BATCHES" \
  --batch-producers "$BATCH_PRODUCERS" \
  --batch-build-workers "$BATCH_BUILD_WORKERS" \
  --state-noise-std "$STATE_NOISE_STD" \
  --target-noise-std 0 \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --checkpoint-save-path-template "$CHECKPOINT_TEMPLATE" \
  --save-path "$FINAL_PATH" \
  --metrics-jsonl "$METRICS_JSONL" \
  --output-json "$RESULT_JSON" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$DRIVER_LOG"
