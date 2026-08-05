#!/usr/bin/env bash
# Launch a user-authorized State45 phase-aware training run from a passed profile.
# This script has no implicit experiment size: STATE45_STEPS must be explicit.
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

PROFILE_REPORT=${1:?usage: run_state45_gradea_train.sh <profile-report.json> [run-id]}
RUN_ID=${2:-state45_gradeA_phase_fulltask_v1}
MODEL=openpi/pi05-action-lora-r16-state45-phase-28dof-finetune
: "${STATE45_STEPS:?set STATE45_STEPS explicitly; this launcher never chooses an experiment size}"
STEPS=$STATE45_STEPS
BATCH_SIZE=${STATE45_BATCH_SIZE:-64}
LEARNING_RATE=${STATE45_LEARNING_RATE:-5e-5}
CHECKPOINT_EVERY=${STATE45_CHECKPOINT_EVERY:-5000}
STATE_NOISE_STD=${STATE45_STATE_NOISE_STD:-0.1}
EXPECTED_DEVICE_COUNT=${STATE45_EXPECTED_DEVICE_COUNT:-4}
PREFETCH_BATCHES=${STATE45_PREFETCH_BATCHES:-2}
STATE45_OBJECT_FILTER=${STATE45_OBJECT:-}
STATE45_GESTURE_FILTER=${STATE45_GESTURE:-}
if [[ -n "$STATE45_OBJECT_FILTER" && -z "$STATE45_GESTURE_FILTER" || \
      -z "$STATE45_OBJECT_FILTER" && -n "$STATE45_GESTURE_FILTER" ]]; then
  echo 'STATE45_OBJECT and STATE45_GESTURE must be set together' >&2
  exit 64
fi

CONTRACT_RAW=$(STATE45_OBJECT_FILTER="$STATE45_OBJECT_FILTER" STATE45_GESTURE_FILTER="$STATE45_GESTURE_FILTER" python3 - "$PROFILE_REPORT" "$MODEL" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
path=Path(sys.argv[1]).expanduser().resolve();model=sys.argv[2]
report=json.loads(path.read_text())
required={
  'contract':'mano_state45_grade_a_train_profile_v1',
  'status':'passed','population':'grade_a','population_rows':4856,
  'state_contract':'mano_state45_phase_native_sim_28d_v1',
  'source_state_contract':'mano_state41_native_sim_28d_v1',
  'state_dim':45,'action_dim':32,'action_horizon':10,
  'frame_window':'contact','contact_context_frames':100,
  'missing_contact_policy':'error','language_conditioning':'gesture',
  'prompt_template':'pick up the {object} using gesture {gesture}, then place it back on the table',
  'model':model,
  'profile_id':'pi05_action_lora_r16_state45_phase_28dof_v1',
  'max_token_len':224,'norm_population':'train_only_contact_window',
  'fail_on_token_truncation':True,
}
for key,expected in required.items():
  if report.get(key)!=expected:
    raise SystemExit(f'profile mismatch {key}: {report.get(key)!r} != {expected!r}')
if report.get('delta_mask_segments') != [3,-3,22,-4]:
  raise SystemExit('State45 B-mask mismatch')
if report.get('token_audit',{}).get('overflow_count') != 0:
  raise SystemExit('State45 profile has observed token overflow')
counterfactual=report.get('counterfactual_token_audit') or {}
if counterfactual.get('overflow_count') != 0:
  raise SystemExit('State45 profile has counterfactual token overflow')
if int(counterfactual.get('max',10**9)) > int(report.get('max_token_len',0)):
  raise SystemExit('State45 counterfactual maximum exceeds configured token budget')
for key in ('train_selection_manifest','train_contact_window_manifest'):
  candidate=Path(report[key]).resolve()
  if not candidate.is_file(): raise SystemExit(f'missing profile artifact: {candidate}')
norm=report.get('norm') or {};norm_path=Path(norm.get('path','')).resolve()
if not norm_path.is_file(): raise SystemExit(f'missing norm_stats.json: {norm_path}')
actual=hashlib.sha256(norm_path.read_bytes()).hexdigest()
if actual!=norm.get('sha256'): raise SystemExit(f'norm SHA mismatch {actual} != {norm.get("sha256")}')
selection=json.loads(Path(report['train_selection_manifest']).read_text())
if selection.get('contract')!='mano_state45_grade_a_selection_v1' or selection.get('split')!='train':
  raise SystemExit('invalid State45 train selection contract')
if selection.get('split_contract')!='mano_state45_grade_a_object_gesture_split_v1':
  raise SystemExit('invalid State45 object/gesture split provenance')
all_rows=selection.get('rows') or []
if len(all_rows)!=report.get('train_rows') or any(row.get('grade')!='A' for row in all_rows):
  raise SystemExit('invalid State45 train selection rows')
expected_prompt='pick up the {object} using gesture {gesture}, then place it back on the table'
if any(row.get('prompt') != expected_prompt.format(object=row['object'],gesture=row['gesture']) for row in all_rows):
  raise SystemExit('State45 train selection prompt mismatch')
object_filter=os.environ.get('STATE45_OBJECT_FILTER','')
gesture_filter=os.environ.get('STATE45_GESTURE_FILTER','')
if object_filter:
  if not gesture_filter.isdigit() or len(gesture_filter)!=2:
    raise SystemExit('STATE45_GESTURE must be a two-digit formal gesture')
  rows=[row for row in all_rows if row.get('object')==object_filter and row.get('gesture')==gesture_filter]
else:
  rows=all_rows
if not rows:
  raise SystemExit(f'no State45 train rows match object={object_filter!r} gesture={gesture_filter!r}')
print(report['dataset'])
print(str(norm_path.parent))
print(actual)
print(str(Path(report['train_contact_window_manifest']).resolve()))
row_indices=','.join(str(int(row['release_row_index'])) for row in rows)
print(row_indices)
print(hashlib.sha256(path.read_bytes()).hexdigest())
print(len(rows))
print(object_filter)
print(gesture_filter)
print(hashlib.sha256(row_indices.encode()).hexdigest())
PY
)
readarray -t CONTRACT <<<"$CONTRACT_RAW"
DATASET=${CONTRACT[0]}
NORM_DIR=${CONTRACT[1]}
NORM_SHA=${CONTRACT[2]}
WINDOW_MANIFEST=${CONTRACT[3]}
ROW_INDICES=${CONTRACT[4]}
PROFILE_SHA=${CONTRACT[5]}
FILTER_ROW_COUNT=${CONTRACT[6]}
FILTER_OBJECT=${CONTRACT[7]}
FILTER_GESTURE=${CONTRACT[8]}
FILTER_ROWS_SHA=${CONTRACT[9]}

[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "STATE45_STEPS must be a positive integer" >&2; exit 64; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "STATE45_BATCH_SIZE must be positive" >&2; exit 64; }
[[ "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]] || { echo "STATE45_CHECKPOINT_EVERY must be positive" >&2; exit 64; }
[[ "$PREFETCH_BATCHES" =~ ^[1-9][0-9]*$ ]] || { echo "STATE45_PREFETCH_BATCHES must be positive" >&2; exit 64; }

OUT_ROOT=${VLA_CLIENT_RESULTS_ROOT:-${REPO_ROOT}/results}/training/${RUN_ID}
RESULT_JSON="$OUT_ROOT/result.json"
METRICS_JSONL="$OUT_ROOT/metrics.jsonl"
CHECKPOINT_EVENTS_JSONL="$OUT_ROOT/checkpoint_events.jsonl"
DRIVER_LOG="$OUT_ROOT/driver.log"
FINAL_PATH="${RUN_ID}_step${STEPS}"
CHECKPOINT_TEMPLATE="${RUN_ID}_step{step}"
if [[ "${STATE45_PRINT_CONFIG:-0}" == 1 ]]; then
  python3 - <<PY
import json
print(json.dumps({'run_id':'$RUN_ID','profile_sha256':'$PROFILE_SHA','model':'$MODEL','steps':int('$STEPS'),'batch_size':int('$BATCH_SIZE'),'learning_rate':float('$LEARNING_RATE'),'checkpoint_every':int('$CHECKPOINT_EVERY'),'state_noise_std':float('$STATE_NOISE_STD'),'expected_device_count':int('$EXPECTED_DEVICE_COUNT'),'prefetch_batches':int('$PREFETCH_BATCHES'),'row_filter':{'object':'$FILTER_OBJECT','gesture':'$FILTER_GESTURE','row_count':int('$FILTER_ROW_COUNT'),'rows_sha256':'$FILTER_ROWS_SHA'}},indent=2,sort_keys=True))
PY
  exit 0
fi
mkdir -p "$OUT_ROOT"
if [[ -e "$RESULT_JSON" || -e "$METRICS_JSONL" || -e "$CHECKPOINT_EVENTS_JSONL" ]]; then
  echo "refusing existing training outputs under $OUT_ROOT" >&2
  exit 2
fi
{
  printf 'started=%s\nrun_id=%s\nprofile_report=%s\nprofile_sha256=%s\n' "$(date -Is)" "$RUN_ID" "$(realpath "$PROFILE_REPORT")" "$PROFILE_SHA"
  printf 'model=%s\nsteps=%s\nbatch_size=%s\nlearning_rate=%s\nstate_noise_std=%s\nprefetch_batches=%s\n' "$MODEL" "$STEPS" "$BATCH_SIZE" "$LEARNING_RATE" "$STATE_NOISE_STD" "$PREFETCH_BATCHES"
  printf 'row_filter_object=%s\nrow_filter_gesture=%s\nrow_filter_count=%s\nrow_filter_rows_sha256=%s\n' "$FILTER_OBJECT" "$FILTER_GESTURE" "$FILTER_ROW_COUNT" "$FILTER_ROWS_SHA"
} | tee "$DRIVER_LOG"

exec "${REPO_ROOT}/scripts/remote/run_client.sh" \
  scripts/train/train_cube1_01_compare.py \
  --model "$MODEL" --lance-dataset "$DATASET" --row-indices "$ROW_INDICES" \
  --action-source urdf_target_absolute --language-conditioning gesture \
  --state-contract state45 --action-horizon 10 --frame-window contact \
  --contact-context-frames 100 --contact-window-manifest "$WINDOW_MANIFEST" \
  --missing-contact-policy error --norm-stats-dir "$NORM_DIR" \
  --norm-sha-expected "$NORM_SHA" --steps "$STEPS" \
  --learning-rate "$LEARNING_RATE" --seed 42 --augmentation-seed 43 \
  --batch-size "$BATCH_SIZE" --expected-device-count "$EXPECTED_DEVICE_COUNT" \
  --sampling-strategy sqrt_tempered --slate-size 16 --coverage-anchors-per-row 8 \
  --row-cache-size 16 --datum-cache-size 256 --prefetch-batches "$PREFETCH_BATCHES" \
  --batch-producers 2 --batch-build-workers 16 \
  --state-noise-std "$STATE_NOISE_STD" --target-noise-std 0 \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --checkpoint-save-path-template "$CHECKPOINT_TEMPLATE" \
  --checkpoint-events-jsonl "$CHECKPOINT_EVENTS_JSONL" \
  --save-path "$FINAL_PATH" --metrics-jsonl "$METRICS_JSONL" \
  --output-json "$RESULT_JSON" 2>&1 | tee -a "$DRIVER_LOG"
