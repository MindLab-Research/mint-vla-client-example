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

PROFILE_REPORT=${1:?usage: run_state41_gradea_100k.sh <profile-report.json> [run-id]}
RUN_ID=${2:-state41_gradeA_train95_aug01_qposonly_alora_r16_bs64_4gpu_contact_pm100_100k_v1}
MODEL=openpi/pi05-action-lora-r16-state41-28dof-finetune

readarray -t CONTRACT < <(python3 - "$PROFILE_REPORT" "$MODEL" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]).expanduser().resolve(); model=sys.argv[2]
report=json.loads(path.read_text())
required={
    'contract':'mano_state41_grade_a_train_profile_v1',
    'status':'passed',
    'population':'grade_a',
    'population_rows':4856,
    'state_dim':41,
    'action_dim':32,
    'action_horizon':10,
    'frame_window':'contact',
    'contact_context_frames':100,
    'missing_contact_policy':'error',
    'language_conditioning':'gesture',
    'prompt_template':'pick up the {object} using gesture {gesture}',
    'model':model,
    'norm_population':'train_only_contact_window',
}
for key,expected in required.items():
    if report.get(key)!=expected:
        raise SystemExit(f'profile mismatch {key}: {report.get(key)!r} != {expected!r}')
if report.get('delta_mask_segments') != [3,-3,22,-4]:
    raise SystemExit('profile B-mask mismatch')
if report.get('sampling_default') != {'strategy':'sqrt_tempered','coverage_anchors_per_row':8}:
    raise SystemExit('profile sampling default mismatch')
for key in ('train_selection_manifest','train_contact_window_manifest'):
    candidate=Path(report[key]).resolve()
    if not candidate.is_file(): raise SystemExit(f'missing profile artifact: {candidate}')
norm=report.get('norm') or {}; norm_path=Path(norm.get('path','')).resolve()
if not norm_path.is_file(): raise SystemExit(f'missing norm_stats.json: {norm_path}')
actual=hashlib.sha256(norm_path.read_bytes()).hexdigest()
if actual!=norm.get('sha256'): raise SystemExit(f'norm SHA mismatch {actual} != {norm.get("sha256")}')
selection_path=Path(report['train_selection_manifest']).resolve()
selection=json.loads(selection_path.read_text())
if selection.get('contract')!='mano_state41_grade_a_selection_v1' or selection.get('split')!='train':
    raise SystemExit('invalid train selection contract')
rows=selection.get('rows') or []
if len(rows)!=report.get('train_rows') or not rows:
    raise SystemExit('train selection count mismatch')
if any(row.get('grade')!='A' for row in rows): raise SystemExit('non-A row in train selection')
indices=','.join(str(int(row['release_row_index'])) for row in rows)
print(report['dataset'])
print(str(norm_path.parent))
print(actual)
print(str(Path(report['train_contact_window_manifest']).resolve()))
print(indices)
print(hashlib.sha256(path.read_bytes()).hexdigest())
print(report['train_uuid_sha256'])
print(report['validation_uuid_sha256'])
PY
)
DATASET=${CONTRACT[0]}
NORM_DIR=${CONTRACT[1]}
NORM_SHA=${CONTRACT[2]}
WINDOW_MANIFEST=${CONTRACT[3]}
ROW_INDICES=${CONTRACT[4]}
PROFILE_SHA=${CONTRACT[5]}
TRAIN_UUID_SHA=${CONTRACT[6]}
VALIDATION_UUID_SHA=${CONTRACT[7]}

STEPS=${STATE41_STEPS:-100000}
BATCH_SIZE=${STATE41_BATCH_SIZE:-64}
LEARNING_RATE=${STATE41_LEARNING_RATE:-5e-5}
CHECKPOINT_EVERY=${STATE41_CHECKPOINT_EVERY:-5000}
SLATE_SIZE=${STATE41_SLATE_SIZE:-16}
ROW_CACHE_SIZE=${STATE41_ROW_CACHE_SIZE:-16}
DATUM_CACHE_SIZE=${STATE41_DATUM_CACHE_SIZE:-256}
PREFETCH_BATCHES=${STATE41_PREFETCH_BATCHES:-2}
BATCH_PRODUCERS=${STATE41_BATCH_PRODUCERS:-2}
BATCH_BUILD_WORKERS=${STATE41_BATCH_BUILD_WORKERS:-16}
STATE_NOISE_STD=${STATE41_STATE_NOISE_STD:-0.1}
EXPECTED_DEVICE_COUNT=${STATE41_EXPECTED_DEVICE_COUNT:-4}

[[ "$STEPS" == 100000 ]] || { echo "full-A default requires STATE41_STEPS=100000" >&2; exit 64; }
[[ "$BATCH_SIZE" == 64 ]] || { echo "full-A default requires STATE41_BATCH_SIZE=64" >&2; exit 64; }
[[ "$LEARNING_RATE" == 5e-5 || "$LEARNING_RATE" == 0.00005 ]] || { echo "full-A default requires lr=5e-5" >&2; exit 64; }
[[ "$CHECKPOINT_EVERY" == 5000 ]] || { echo "full-A default requires checkpoint every5000" >&2; exit 64; }
[[ "$STATE_NOISE_STD" == 0.1 || "$STATE_NOISE_STD" == 0.10 ]] || { echo "full-A default requires qpos state noise0.1" >&2; exit 64; }
[[ "$EXPECTED_DEVICE_COUNT" == 4 ]] || { echo "full-A default requires four training GPUs" >&2; exit 64; }
[[ "$SLATE_SIZE" == 16 && "$ROW_CACHE_SIZE" == 16 ]] || { echo "full-A default requires slate/cache16" >&2; exit 64; }
[[ $((BATCH_SIZE * PREFETCH_BATCHES)) -le $((SLATE_SIZE * 8)) ]] || {
  echo "prefetch spans more than one sqrt-tempered slate" >&2; exit 64;
}

OUT_ROOT=${VLA_CLIENT_RESULTS_ROOT:-${REPO_ROOT}/results}/training/${RUN_ID}
RESULT_JSON="$OUT_ROOT/result.json"
METRICS_JSONL="$OUT_ROOT/metrics.jsonl"
CHECKPOINT_EVENTS_JSONL="$OUT_ROOT/checkpoint_events.jsonl"
DRIVER_LOG="$OUT_ROOT/driver.log"
FINAL_PATH="${RUN_ID}_step${STEPS}"
CHECKPOINT_TEMPLATE="${RUN_ID}_step{step}"

if [[ "${STATE41_GRADEA_PRINT_CONFIG:-0}" == 1 ]]; then
  python3 - <<PY
import json
print(json.dumps({
  'run_id':'$RUN_ID','profile_report':str(__import__('pathlib').Path('$PROFILE_REPORT').resolve()),
  'profile_sha256':'$PROFILE_SHA','dataset':'$DATASET','train_uuid_sha256':'$TRAIN_UUID_SHA',
  'validation_uuid_sha256':'$VALIDATION_UUID_SHA','norm_sha256':'$NORM_SHA',
  'model':'$MODEL','steps':$STEPS,'batch_size':$BATCH_SIZE,'per_device_batch_size':16,
  'expected_device_count':$EXPECTED_DEVICE_COUNT,'learning_rate':float('$LEARNING_RATE'),
  'sample_seed':42,'augmentation_seed':43,
  'state_noise_std':float('$STATE_NOISE_STD'),'target_noise_std':0.0,
  'sampling_strategy':'sqrt_tempered','slate_size':$SLATE_SIZE,'coverage_anchors_per_row':8,
  'checkpoint_every':$CHECKPOINT_EVERY,'language_conditioning':'gesture',
  'frame_window':'contact','contact_context_frames':100,'continuous_training':True,
  'interleaved_mode4':False,'checkpoint_events_jsonl':'$CHECKPOINT_EVENTS_JSONL'
},sort_keys=True,indent=2))
PY
  exit 0
fi

mkdir -p "$OUT_ROOT"
if [[ -e "$RESULT_JSON" || -e "$METRICS_JSONL" || -e "$CHECKPOINT_EVENTS_JSONL" ]]; then
  echo "refusing existing training outputs under $OUT_ROOT" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ "${STATE41_TRAIN_DRY_RUN:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--dry-run --augmentation-audit-samples 64)
fi
{
  printf 'started=%s\n' "$(date -Is)"
  printf 'run_id=%s\nprofile_report=%s\nprofile_sha256=%s\n' "$RUN_ID" "$(realpath "$PROFILE_REPORT")" "$PROFILE_SHA"
  printf 'dataset=%s\ntrain_uuid_sha256=%s\nvalidation_uuid_sha256=%s\nnorm_sha256=%s\n' "$DATASET" "$TRAIN_UUID_SHA" "$VALIDATION_UUID_SHA" "$NORM_SHA"
  printf 'model=%s\nsteps=%s\nbatch_size=%s\nexpected_device_count=%s\nlearning_rate=%s\nstate_noise_std=%s\ncheckpoint_every=%s\n' "$MODEL" "$STEPS" "$BATCH_SIZE" "$EXPECTED_DEVICE_COUNT" "$LEARNING_RATE" "$STATE_NOISE_STD" "$CHECKPOINT_EVERY"
} | tee "$DRIVER_LOG"

exec "${REPO_ROOT}/scripts/remote/run_client.sh" \
  scripts/train/train_cube1_01_compare.py \
  --model "$MODEL" \
  --lance-dataset "$DATASET" \
  --row-indices "$ROW_INDICES" \
  --action-source urdf_target_absolute \
  --language-conditioning gesture \
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
  --seed 42 \
  --augmentation-seed 43 \
  --batch-size "$BATCH_SIZE" \
  --expected-device-count "$EXPECTED_DEVICE_COUNT" \
  --sampling-strategy sqrt_tempered \
  --slate-size "$SLATE_SIZE" \
  --coverage-anchors-per-row 8 \
  --row-cache-size "$ROW_CACHE_SIZE" \
  --datum-cache-size "$DATUM_CACHE_SIZE" \
  --prefetch-batches "$PREFETCH_BATCHES" \
  --batch-producers "$BATCH_PRODUCERS" \
  --batch-build-workers "$BATCH_BUILD_WORKERS" \
  --state-noise-std "$STATE_NOISE_STD" \
  --target-noise-std 0 \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --checkpoint-save-path-template "$CHECKPOINT_TEMPLATE" \
  --checkpoint-events-jsonl "$CHECKPOINT_EVENTS_JSONL" \
  --save-path "$FINAL_PATH" \
  --metrics-jsonl "$METRICS_JSONL" \
  --output-json "$RESULT_JSON" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$DRIVER_LOG"
