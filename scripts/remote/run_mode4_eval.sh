#!/usr/bin/env bash
# Unified Mode4 launcher: evaluate one checkpoint/row set against either an
# existing MINT endpoint or a dedicated server owned by this invocation.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONFIG_FILE=${VLA_CLIENT_CONFIG:-${REPO_ROOT}/config/remote.env}
if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

: "${MINT_CODE_ROOT:=/vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16}"
: "${MINT_OPENPI_ROOT:=/vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16}"
: "${MINT_PYTHON_BIN:=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python}"
: "${MINT_API_KEY:=tml-dummy}"
: "${MANO_DATASET_RELEASE:=${REPO_ROOT}/config/datasets/mano_dataset_release.json}"
CANONICAL_DATASET=$(python3 "${REPO_ROOT}/scripts/mano_dataset_release.py" \
  --manifest "${MANO_DATASET_RELEASE}" resolve training_dataset)
CANONICAL_CONTACT_MANIFEST=$(python3 "${REPO_ROOT}/scripts/mano_dataset_release.py" \
  --manifest "${MANO_DATASET_RELEASE}" resolve contact_windows)
CANONICAL_GESTURE_INDEX=$(python3 "${REPO_ROOT}/scripts/mano_dataset_release.py" \
  --manifest "${MANO_DATASET_RELEASE}" resolve language_index)
MANO_RELEASE_ID=$(python3 "${REPO_ROOT}/scripts/mano_dataset_release.py" \
  --manifest "${MANO_DATASET_RELEASE}" release-id)
MANO_RELEASE_SHA256=$(sha256sum "${MANO_DATASET_RELEASE}" | awk '{print $1}')
: "${MINT_LANCE_DATASET:=${CANONICAL_DATASET}}"
: "${VLA_CLIENT_RESULTS_ROOT:=${REPO_ROOT}/results}"
: "${VLA_CLIENT_INFERENCE_ROOT:=${VLA_CLIENT_RESULTS_ROOT}/inference}"

MODEL=openpi/pi05-action-lora-r16-finetune
MODEL_PATH=
STATE_CONTRACT=state32
DATASET=${MINT_LANCE_DATASET}
ROWS=
NORMALIZATION_ROWS=
NORM_STATS_DIR=
NORM_SHA_EXPECTED=
OUTPUT_DIR=
RUN_NAME=
OWNER_ID=
BASE_URL=
ENDPOINT_LABEL=
BACKEND_COMMIT=
OPENPI_COMMIT=
REUSE_SERVER_INFO=
ACTION_SESSION_ID=
OWN_SERVER=0
KEEP_SERVER=0
SERVER_PORT=30532
SERVER_PORT_SET=0
SERVER_GPUS=
SERVER_GPUS_SET=0
SERVER_RUNTIME_ROOT=
SERVER_CACHE_DIR=
ENABLE_JAX_PERSISTENT_CACHE=0
CHUNK_STRIDE=5
TEMPORAL_DECAY=0.4
ACT_MODE=batch
ACT_BATCH_SIZE=4
ROW_EXECUTION=lockstep
ROW_BATCH_SIZE=4
MAX_WARM_REQUEST_SECONDS=2
MAX_FRAMES=0
FPS=10
WIDTH=640
HEIGHT=360
VIDEO_MODE=full
FRAME_WINDOW=contact
CONTACT_WINDOW_MANIFEST=
CONTACT_CONTEXT_FRAMES=100
MISSING_CONTACT_POLICY=error
LANGUAGE_CONDITIONING=gesture
GESTURE_INDEX=${CANONICAL_GESTURE_INDEX}
OVERWRITE_OUTPUT=0
ALLOW_DIRTY_SOURCES=0
PRINT_CONFIG=0

usage() {
  cat <<'EOF'
usage: run_mode4_eval.sh --model-path PATH --rows CSV \
  --normalization-rows CSV|all --norm-stats-dir PATH --owner ID \
  (--base-url URL --backend-commit SHA --model-commit SHA | \
   --reuse-server-info PATH | --own-server --server-runtime-root PATH) [options]

Required evaluation options:
  --model-path PATH           checkpoint identifier accepted by MINT
  --dataset, --lance-dataset PATH
                              Lance dataset; defaults to release role training_dataset
  --rows, --row-indices CSV   ordered evaluation row IDs
  --normalization-rows, --normalization-row-indices CSV|all
  --norm-stats-dir PATH       checkpoint's locked normalization directory
  --norm-sha-expected SHA     population-specific expected norm_stats.json SHA256
  --owner, --owner-id ID      MINT action-session owner ID; supplied by --reuse-server-info

Endpoint selection (choose exactly one):
  --base-url URL              existing endpoint; this launcher never stops it
  --backend-commit SHA        operator-declared MINT source for existing endpoint
  --model-commit SHA          operator-declared OpenPI source for existing endpoint
  --endpoint-label TEXT       optional allocation/deployment identifier
  --action-session-id ID      reuse an externally owned action session on --base-url
  --reuse-server-info PATH    attach through a prior keep-server marker and reuse its action session
  --own-server                start and stop a dedicated server for this run
  --server-runtime-root PATH  required with --own-server

Evaluation options:
  --model NAME                model identity (default: action-LoRA rank 16)
  --state-contract state32|state44 (default: state32)
  --chunk-stride N            query stride, 1..9 (default: 5)
  --temporal-decay FLOAT      ensemble decay in (0, 1] (default: 0.4)
  --act-mode batch|single     action endpoint mode (default: batch)
  --act-batch-size N          fixed model batch shape (default: 4)
  --row-execution lockstep|sequential
                              lockstep batches real observations from independent rows (default)
  --row-batch-size N          concurrent rows per lockstep group, <= act-batch-size (default: 4)
  --max-warm-request-seconds FLOAT (default: 2; 0 disables)
  --max-frames N              optional bounded smoke rollout; 0 means full
  --fps FLOAT                 output video FPS (default: 10)
  --width N --height N        per-panel render size (default: 640x360)
  --video-mode full|none      full writes videos; none keeps observation rendering but skips output-video encoding
  --frame-window contact|full contact initializes physics at the manifest window (default);
                              full is an explicit full-trajectory stress test
  --contact-window-manifest PATH
                              canonical dataset uses release role contact_windows;
                              non-release datasets derive <dataset-without-.lance>.contact_ctx100_error_v1.json
  --contact-context-frames N  must match the manifest (default: 100)
  --missing-contact-policy full|skip|error (default: error)
  --language-conditioning gesture|motion_variant|object_only
  --gesture-index PATH        canonical gesture index; required for gesture
  --api-key KEY               MINT API key (default: MINT_API_KEY)

Dedicated-server options:
  --server-port PORT          listen port (required with --own-server)
  --server-gpus CSV           CUDA GPU IDs (required with --own-server)
  --mint-root PATH            paired MINT checkout
  --openpi-root PATH          paired OpenPI checkout
  --python-bin PATH           GPU runtime Python
  --server-cache-dir PATH     cache path used only with explicit cache opt-in
  --keep-server               retain the owned server and compiled action session after success
  --enable-jax-persistent-cache
                              opt in to multi-GB executable serialization

Output/source controls:
  --output-dir PATH           explicit result root; existing roots are refused
  --run-name NAME             client-local results/inference/NAME; cannot combine with --output-dir
                              default: mode4_<UTC>_<pid> under VLA_CLIENT_INFERENCE_ROOT
  --overwrite-output          delete and recreate an existing output root
  --allow-dirty-sources       permit dirty client/owned-server worktrees and record it
  --print-config              validate and print effective JSON; do not start/run
  -h, --help                  show this help

Existing-endpoint commits are operator declarations because a client cannot
prove the source of an already-running process. Dedicated-server commits are
read directly from the exact worktrees launched by this command.
EOF
}

while (($#)); do
  case "$1" in
    --model) MODEL=${2:?}; shift 2 ;;
    --model-path) MODEL_PATH=${2:?}; shift 2 ;;
    --state-contract) STATE_CONTRACT=${2:?}; shift 2 ;;
    --dataset|--lance-dataset) DATASET=${2:?}; shift 2 ;;
    --rows|--row-indices) ROWS=${2:?}; shift 2 ;;
    --normalization-rows|--normalization-row-indices) NORMALIZATION_ROWS=${2:?}; shift 2 ;;
    --norm-stats-dir) NORM_STATS_DIR=${2:?}; shift 2 ;;
    --norm-sha-expected) NORM_SHA_EXPECTED=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --run-name) RUN_NAME=${2:?}; shift 2 ;;
    --owner|--owner-id) OWNER_ID=${2:?}; shift 2 ;;
    --base-url) BASE_URL=${2:?}; shift 2 ;;
    --endpoint-label) ENDPOINT_LABEL=${2:?}; shift 2 ;;
    --backend-commit) BACKEND_COMMIT=${2:?}; shift 2 ;;
    --model-commit) OPENPI_COMMIT=${2:?}; shift 2 ;;
    --action-session-id) ACTION_SESSION_ID=${2:?}; shift 2 ;;
    --reuse-server-info) REUSE_SERVER_INFO=${2:?}; shift 2 ;;
    --own-server) OWN_SERVER=1; shift ;;
    --keep-server) KEEP_SERVER=1; shift ;;
    --server-port) SERVER_PORT=${2:?}; SERVER_PORT_SET=1; shift 2 ;;
    --server-gpus) SERVER_GPUS=${2:?}; SERVER_GPUS_SET=1; shift 2 ;;
    --server-runtime-root) SERVER_RUNTIME_ROOT=${2:?}; shift 2 ;;
    --server-cache-dir) SERVER_CACHE_DIR=${2:?}; shift 2 ;;
    --mint-root) MINT_CODE_ROOT=${2:?}; shift 2 ;;
    --openpi-root) MINT_OPENPI_ROOT=${2:?}; shift 2 ;;
    --python-bin) MINT_PYTHON_BIN=${2:?}; shift 2 ;;
    --enable-jax-persistent-cache) ENABLE_JAX_PERSISTENT_CACHE=1; shift ;;
    --chunk-stride) CHUNK_STRIDE=${2:?}; shift 2 ;;
    --temporal-decay) TEMPORAL_DECAY=${2:?}; shift 2 ;;
    --act-mode) ACT_MODE=${2:?}; shift 2 ;;
    --act-batch-size) ACT_BATCH_SIZE=${2:?}; shift 2 ;;
    --row-execution) ROW_EXECUTION=${2:?}; shift 2 ;;
    --row-batch-size) ROW_BATCH_SIZE=${2:?}; shift 2 ;;
    --max-warm-request-seconds) MAX_WARM_REQUEST_SECONDS=${2:?}; shift 2 ;;
    --max-frames) MAX_FRAMES=${2:?}; shift 2 ;;
    --fps) FPS=${2:?}; shift 2 ;;
    --width) WIDTH=${2:?}; shift 2 ;;
    --height) HEIGHT=${2:?}; shift 2 ;;
    --video-mode) VIDEO_MODE=${2:?}; shift 2 ;;
    --frame-window) FRAME_WINDOW=${2:?}; shift 2 ;;
    --contact-window-manifest) CONTACT_WINDOW_MANIFEST=${2:?}; shift 2 ;;
    --contact-context-frames) CONTACT_CONTEXT_FRAMES=${2:?}; shift 2 ;;
    --missing-contact-policy) MISSING_CONTACT_POLICY=${2:?}; shift 2 ;;
    --language-conditioning) LANGUAGE_CONDITIONING=${2:?}; shift 2 ;;
    --gesture-index) GESTURE_INDEX=${2:?}; shift 2 ;;
    --api-key) MINT_API_KEY=${2:?}; shift 2 ;;
    --overwrite-output) OVERWRITE_OUTPUT=1; shift ;;
    --allow-dirty-sources) ALLOW_DIRTY_SOURCES=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

fail() { echo "run_mode4_eval: $*" >&2; exit 64; }
require_nonempty() { [[ -n "$2" ]] || fail "$1 is required"; }
require_absolute() { [[ "$2" = /* ]] || fail "$1 must be an absolute path: $2"; }
require_positive_int() {
  [[ "$2" =~ ^[0-9]+$ ]] && ((10#$2 > 0)) || fail "$1 must be a positive integer: $2"
}
require_nonnegative_int() {
  [[ "$2" =~ ^[0-9]+$ ]] || fail "$1 must be a non-negative integer: $2"
}
require_nonnegative_float() {
  python3 - "$2" <<'PY' || fail "$1 must be a finite non-negative number: $2"
import math, sys
value=float(sys.argv[1])
if not math.isfinite(value) or value < 0: raise SystemExit(1)
PY
}
require_positive_float() {
  python3 - "$2" <<'PY' || fail "$1 must be a finite positive number: $2"
import math, sys
value=float(sys.argv[1])
if not math.isfinite(value) or value <= 0: raise SystemExit(1)
PY
}
git_commit() {
  local label=$1 path=$2
  [[ -d "$path" ]] || fail "$label path is missing: $path"
  git -C "$path" rev-parse HEAD 2>/dev/null || fail "$label is not a Git checkout: $path"
}
git_dirty() {
  [[ -n "$(git -C "$1" status --porcelain)" ]] && printf true || printf false
}
validate_sha() {
  [[ "$2" =~ ^[0-9a-fA-F]{7,40}$ ]] || fail "$1 must be a 7-40 character hexadecimal Git SHA"
}

REUSE_MARKER_VERIFIED=0
if [[ -n "$REUSE_SERVER_INFO" ]]; then
  ((OWN_SERVER == 0)) || fail "--reuse-server-info and --own-server are mutually exclusive"
  [[ -z "$BASE_URL" ]] || fail "--reuse-server-info and --base-url are mutually exclusive"
  [[ -z "$ACTION_SESSION_ID" ]] || fail "--reuse-server-info and --action-session-id are mutually exclusive"
  require_absolute --reuse-server-info "$REUSE_SERVER_INFO"
  [[ -f "$REUSE_SERVER_INFO" ]] || fail "reuse server marker does not exist: $REUSE_SERVER_INFO"
  REUSE_DATA=$(python3 - "$REUSE_SERVER_INFO" <<'PY'
import json, sys
from pathlib import Path
marker=json.loads(Path(sys.argv[1]).read_text())
if marker.get('status') != 'owned_running':
    raise SystemExit(f"marker status must be owned_running, got {marker.get('status')!r}")
for key in (
    'pid','base_url','owner_id','backend_commit','model_commit','action_session_id',
    'model','model_path','act_mode','act_batch_size',
):
    if not marker.get(key):
        raise SystemExit(f"reuse marker missing {key}")
print(marker['pid'])
print(marker['base_url'])
print(marker['owner_id'])
print(marker['backend_commit'])
print(marker['model_commit'])
print(marker['action_session_id'])
print(marker['model'])
print(marker['model_path'])
print(marker['act_mode'])
print(marker['act_batch_size'])
PY
  ) || fail "invalid reuse server marker: $REUSE_SERVER_INFO"
  mapfile -t REUSE_FIELDS <<< "$REUSE_DATA"
  ((${#REUSE_FIELDS[@]} == 10)) || fail "invalid reuse server marker fields"
  REUSE_SERVER_PID=${REUSE_FIELDS[0]}
  [[ "$REUSE_SERVER_PID" =~ ^[0-9]+$ ]] || fail "reuse server marker contains an invalid PID"
  kill -0 "$REUSE_SERVER_PID" 2>/dev/null || fail "retained server PID is not running: $REUSE_SERVER_PID"
  REUSE_CMDLINE=$(tr '\0' ' ' < "/proc/$REUSE_SERVER_PID/cmdline" 2>/dev/null || true)
  [[ "$REUSE_CMDLINE" == *mint_server* || "$REUSE_CMDLINE" == *uvicorn* ]] || \
    fail "retained server PID is not recognizably MINT/uvicorn: $REUSE_SERVER_PID"
  BASE_URL=${REUSE_FIELDS[1]}
  if [[ -n "$OWNER_ID" && "$OWNER_ID" != "${REUSE_FIELDS[2]}" ]]; then
    fail "--owner-id does not match reuse server marker"
  fi
  if [[ -n "$BACKEND_COMMIT" && "$BACKEND_COMMIT" != "${REUSE_FIELDS[3]}" ]]; then
    fail "--backend-commit does not match reuse server marker"
  fi
  if [[ -n "$OPENPI_COMMIT" && "$OPENPI_COMMIT" != "${REUSE_FIELDS[4]}" ]]; then
    fail "--model-commit does not match reuse server marker"
  fi
  OWNER_ID=${REUSE_FIELDS[2]}
  BACKEND_COMMIT=${REUSE_FIELDS[3]}
  OPENPI_COMMIT=${REUSE_FIELDS[4]}
  ACTION_SESSION_ID=${REUSE_FIELDS[5]}
  [[ "$MODEL" == "${REUSE_FIELDS[6]}" ]] || fail "--model does not match retained action session"
  [[ "$MODEL_PATH" == "${REUSE_FIELDS[7]}" ]] || fail "--model-path does not match retained action session"
  [[ "$ACT_MODE" == "${REUSE_FIELDS[8]}" ]] || fail "--act-mode does not match retained action session"
  EXPECTED_REUSE_BATCH_SIZE=$ACT_BATCH_SIZE
  [[ "$ACT_MODE" == batch ]] || EXPECTED_REUSE_BATCH_SIZE=1
  [[ "$EXPECTED_REUSE_BATCH_SIZE" == "${REUSE_FIELDS[9]}" ]] || \
    fail "--act-batch-size does not match retained action session"
  ENDPOINT_LABEL=${ENDPOINT_LABEL:-retained-action-session}
  REUSE_MARKER_VERIFIED=1
fi

require_nonempty --model "$MODEL"
require_nonempty --model-path "$MODEL_PATH"
require_nonempty --dataset "$DATASET"
require_nonempty --rows "$ROWS"
require_nonempty --normalization-rows "$NORMALIZATION_ROWS"
require_nonempty --norm-stats-dir "$NORM_STATS_DIR"
require_nonempty --owner "$OWNER_ID"
if [[ -n "$OUTPUT_DIR" && -n "$RUN_NAME" ]]; then
  fail "--output-dir and --run-name are mutually exclusive"
fi
if [[ -n "$RUN_NAME" && ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  fail "--run-name must be a single portable path component: $RUN_NAME"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  require_absolute VLA_CLIENT_INFERENCE_ROOT "$VLA_CLIENT_INFERENCE_ROOT"
  RUN_NAME=${RUN_NAME:-"mode4_$(date -u +%Y%m%dT%H%M%SZ)_$$"}
  OUTPUT_DIR="${VLA_CLIENT_INFERENCE_ROOT%/}/${RUN_NAME}"
fi
case "$FRAME_WINDOW" in
  contact|full) ;;
  *) fail "--frame-window must be contact or full: $FRAME_WINDOW" ;;
esac
case "$MISSING_CONTACT_POLICY" in
  full|skip|error) ;;
  *) fail "--missing-contact-policy must be full, skip, or error: $MISSING_CONTACT_POLICY" ;;
esac
require_nonnegative_int --contact-context-frames "$CONTACT_CONTEXT_FRAMES"
if [[ "$FRAME_WINDOW" == contact ]]; then
  if [[ -z "$CONTACT_WINDOW_MANIFEST" ]]; then
    if [[ "$(readlink -f "$DATASET")" == "$(readlink -f "$CANONICAL_DATASET")" ]]; then
      CONTACT_WINDOW_MANIFEST=${CANONICAL_CONTACT_MANIFEST}
    else
      CONTACT_WINDOW_MANIFEST="${DATASET%.lance}.contact_ctx100_error_v1.json"
    fi
  fi
  require_absolute --contact-window-manifest "$CONTACT_WINDOW_MANIFEST"
  [[ -f "$CONTACT_WINDOW_MANIFEST" ]] || \
    fail "contact-window manifest does not exist: $CONTACT_WINDOW_MANIFEST"
else
  CONTACT_WINDOW_MANIFEST=
fi
require_absolute --dataset "$DATASET"
require_absolute --norm-stats-dir "$NORM_STATS_DIR"
require_absolute --output-dir "$OUTPUT_DIR"
[[ "$OUTPUT_DIR" != / ]] || fail "--output-dir must not be /"
[[ -e "$DATASET" ]] || fail "dataset does not exist: $DATASET"
[[ -d "$NORM_STATS_DIR" && -f "$NORM_STATS_DIR/norm_stats.json" ]] || \
  fail "norm stats directory must contain norm_stats.json: $NORM_STATS_DIR"
[[ "$ROWS" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "--rows must be comma-separated non-negative integers"
[[ "$NORMALIZATION_ROWS" == all || "$NORMALIZATION_ROWS" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
  fail "--normalization-rows must be 'all' or comma-separated non-negative integers"
normalize_csv() {
  python3 - "$1" <<'PY'
import sys
values=list(dict.fromkeys(int(x) for x in sys.argv[1].split(',')))
print(','.join(str(x) for x in values))
PY
}
ROWS=$(normalize_csv "$ROWS")
if [[ "$NORMALIZATION_ROWS" != all ]]; then
  NORMALIZATION_ROWS=$(normalize_csv "$NORMALIZATION_ROWS")
fi
case "$LANGUAGE_CONDITIONING" in
  gesture|motion_variant|object_only) ;;
  *) fail "invalid --language-conditioning: $LANGUAGE_CONDITIONING" ;;
esac
if [[ "$LANGUAGE_CONDITIONING" == gesture ]]; then
  [[ -f "$GESTURE_INDEX" ]] || fail "gesture index does not exist: $GESTURE_INDEX"
fi
[[ "$SERVER_PORT" =~ ^[0-9]+$ ]] && ((10#$SERVER_PORT > 0 && 10#$SERVER_PORT < 65536)) || \
  fail "invalid --server-port: $SERVER_PORT"
if [[ -n "$SERVER_GPUS" ]]; then
  [[ "$SERVER_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "invalid --server-gpus: $SERVER_GPUS"
fi
[[ "$CHUNK_STRIDE" =~ ^[0-9]+$ ]] && ((10#$CHUNK_STRIDE >= 1 && 10#$CHUNK_STRIDE < 10)) || \
  fail "--chunk-stride must be between 1 and 9"
[[ "$ACT_MODE" == batch || "$ACT_MODE" == single ]] || fail "--act-mode must be batch or single"
[[ "$ROW_EXECUTION" == lockstep || "$ROW_EXECUTION" == sequential ]] || \
  fail "--row-execution must be lockstep or sequential"
[[ "$STATE_CONTRACT" == state32 || "$STATE_CONTRACT" == state44 ]] || \
  fail "--state-contract must be state32 or state44"
if [[ "$STATE_CONTRACT" == state44 ]]; then
  [[ "$MODEL" == openpi/pi05-action-lora-r16-state44-finetune ]] || \
    fail "state44 requires model openpi/pi05-action-lora-r16-state44-finetune"
elif [[ "$MODEL" == openpi/pi05-action-lora-r16-state44-finetune ]]; then
  fail "state44 model identity requires --state-contract state44"
fi
[[ "$VIDEO_MODE" == full || "$VIDEO_MODE" == none ]] || fail "--video-mode must be full or none"
require_positive_int --act-batch-size "$ACT_BATCH_SIZE"
require_positive_int --row-batch-size "$ROW_BATCH_SIZE"
((10#$ROW_BATCH_SIZE <= 10#$ACT_BATCH_SIZE)) || \
  fail "--row-batch-size must be <= --act-batch-size"
[[ "$ROW_EXECUTION" != lockstep || "$ACT_MODE" == batch ]] || \
  fail "--row-execution lockstep requires --act-mode batch"
require_nonnegative_int --max-frames "$MAX_FRAMES"
((10#$MAX_FRAMES == 0 || 10#$MAX_FRAMES >= 2)) || fail "--max-frames must be 0 or at least 2"
require_positive_int --width "$WIDTH"
require_positive_int --height "$HEIGHT"
require_nonnegative_float --max-warm-request-seconds "$MAX_WARM_REQUEST_SECONDS"
require_positive_float --fps "$FPS"
python3 - "$TEMPORAL_DECAY" <<'PY' || fail "--temporal-decay must be in (0, 1]"
import math, sys
value=float(sys.argv[1])
if not math.isfinite(value) or not 0 < value <= 1: raise SystemExit(1)
PY

CLIENT_COMMIT=$(git_commit client "$REPO_ROOT")
CLIENT_DIRTY=$(git_dirty "$REPO_ROOT")
PROVENANCE_VERIFICATION=operator_declared
MINT_DIRTY=null
OPENPI_DIRTY=null
if ((KEEP_SERVER && !OWN_SERVER)); then
  fail "--keep-server requires --own-server"
fi
if ((OWN_SERVER)); then
  [[ -z "$BASE_URL" ]] || fail "--own-server and --base-url are mutually exclusive"
  [[ -z "$REUSE_SERVER_INFO" ]] || fail "--own-server and --reuse-server-info are mutually exclusive"
  [[ -z "$ACTION_SESSION_ID" ]] || fail "--action-session-id requires an existing or retained endpoint"
  ((SERVER_PORT_SET == 1)) || fail "--server-port is required with --own-server"
  ((SERVER_GPUS_SET == 1)) || fail "--server-gpus is required with --own-server"
  require_nonempty --server-runtime-root "$SERVER_RUNTIME_ROOT"
  require_absolute --server-runtime-root "$SERVER_RUNTIME_ROOT"
  require_absolute --mint-root "$MINT_CODE_ROOT"
  require_absolute --openpi-root "$MINT_OPENPI_ROOT"
  require_absolute --python-bin "$MINT_PYTHON_BIN"
  [[ -x "$MINT_PYTHON_BIN" ]] || fail "GPU runtime Python is not executable: $MINT_PYTHON_BIN"
  [[ -z "$SERVER_CACHE_DIR" || "$SERVER_CACHE_DIR" = /* ]] || fail "--server-cache-dir must be absolute"
  if [[ -n "$SERVER_CACHE_DIR" && "$ENABLE_JAX_PERSISTENT_CACHE" == 0 ]]; then
    fail "--server-cache-dir requires --enable-jax-persistent-cache"
  fi
  BASE_URL="http://127.0.0.1:${SERVER_PORT}"
  ACTUAL_BACKEND_COMMIT=$(git_commit MINT "$MINT_CODE_ROOT")
  ACTUAL_OPENPI_COMMIT=$(git_commit OpenPI "$MINT_OPENPI_ROOT")
  [[ -z "$BACKEND_COMMIT" ]] || validate_sha --backend-commit "$BACKEND_COMMIT"
  [[ -z "$OPENPI_COMMIT" ]] || validate_sha --model-commit "$OPENPI_COMMIT"
  MINT_DIRTY=$(git_dirty "$MINT_CODE_ROOT")
  OPENPI_DIRTY=$(git_dirty "$MINT_OPENPI_ROOT")
  if [[ -n "$BACKEND_COMMIT" && "$ACTUAL_BACKEND_COMMIT" != "$BACKEND_COMMIT"* ]]; then
    fail "--backend-commit does not match owned MINT worktree: $BACKEND_COMMIT != $ACTUAL_BACKEND_COMMIT"
  fi
  if [[ -n "$OPENPI_COMMIT" && "$ACTUAL_OPENPI_COMMIT" != "$OPENPI_COMMIT"* ]]; then
    fail "--model-commit does not match owned OpenPI worktree: $OPENPI_COMMIT != $ACTUAL_OPENPI_COMMIT"
  fi
  BACKEND_COMMIT=$ACTUAL_BACKEND_COMMIT
  OPENPI_COMMIT=$ACTUAL_OPENPI_COMMIT
  PROVENANCE_VERIFICATION=launcher_verified_worktrees
else
  [[ -n "$BASE_URL" ]] || fail "choose --base-url or --own-server"
  [[ -z "$SERVER_RUNTIME_ROOT" && -z "$SERVER_CACHE_DIR" && "$ENABLE_JAX_PERSISTENT_CACHE" == 0 \
     && "$SERVER_PORT_SET" == 0 && "$SERVER_GPUS_SET" == 0 ]] || \
    fail "dedicated-server options require --own-server"
  require_nonempty --backend-commit "$BACKEND_COMMIT"
  require_nonempty --model-commit "$OPENPI_COMMIT"
  validate_sha --backend-commit "$BACKEND_COMMIT"
  validate_sha --model-commit "$OPENPI_COMMIT"
  if ((REUSE_MARKER_VERIFIED)); then
    PROVENANCE_VERIFICATION=retained_action_session_marker
  fi
fi
BASE_URL=${BASE_URL%/}
[[ "$BASE_URL" =~ ^https?://[^/]+$ ]] || fail "--base-url must be an http(s) endpoint without a path"

if ((ALLOW_DIRTY_SOURCES == 0)); then
  [[ "$CLIENT_DIRTY" == false ]] || fail "client worktree is dirty; commit/revert or use --allow-dirty-sources"
  if ((OWN_SERVER)); then
    [[ "$MINT_DIRTY" == false ]] || fail "owned MINT worktree is dirty"
    [[ "$OPENPI_DIRTY" == false ]] || fail "owned OpenPI worktree is dirty"
  fi
fi

python3 - "$OUTPUT_DIR" "$REPO_ROOT" "$DATASET" "$NORM_STATS_DIR" \
  "$MINT_CODE_ROOT" "$MINT_OPENPI_ROOT" "$SERVER_RUNTIME_ROOT" <<'PY' || \
  fail "output directory overlaps a protected source/runtime path"
from pathlib import Path
import sys
out=Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    if not value:
        continue
    protected=Path(value).resolve()
    if out == protected or out in protected.parents:
        raise SystemExit(1)
    # Keep the owned server state and evaluation output as separate trees.
    if value == sys.argv[-1] and protected in out.parents:
        raise SystemExit(1)
PY

if [[ -e "$OUTPUT_DIR" && "$OVERWRITE_OUTPUT" == 0 ]]; then
  fail "output already exists: $OUTPUT_DIR"
fi
NORM_SHA256=$(sha256sum "$NORM_STATS_DIR/norm_stats.json" | awk '{print $1}')
if [[ -n "$NORM_SHA_EXPECTED" ]]; then
  [[ "$NORM_SHA_EXPECTED" =~ ^[0-9a-fA-F]{64}$ ]] || fail "--norm-sha-expected must be 64 hexadecimal characters"
  [[ "${NORM_SHA_EXPECTED,,}" == "$NORM_SHA256" ]] || \
    fail "norm SHA mismatch: expected ${NORM_SHA_EXPECTED,,}, got $NORM_SHA256"
  NORM_SHA_EXPECTED=${NORM_SHA_EXPECTED,,}
fi

write_config() {
  python3 - "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
(
    client_commit, client_dirty, backend_commit, model_commit, mint_dirty,
    openpi_dirty, norm_sha, norm_sha_expected, release_id, release_manifest, release_sha,
    verification, endpoint_mode, endpoint_label,
    base_url, model, model_path, state_contract, dataset, rows, norm_rows, norm_dir, output_dir,
    owner, stride, decay, act_mode, batch_size, row_execution, row_batch_size,
    max_warm, max_frames, fps, width, height, video_mode, frame_window,
    contact_manifest, contact_context,
    missing_contact_policy, language, gesture_index, server_port, server_gpus, runtime_root,
    cache_dir, persistent_cache, keep_server, reuse_server_info, action_session_id,
) = sys.argv[1:]
row_ids=list(dict.fromkeys(int(x) for x in rows.split(',')))
normalization='all' if norm_rows == 'all' else list(dict.fromkeys(int(x) for x in norm_rows.split(',')))
payload={
    'mode': 'mode4_policy_target_dof_mujoco_physics',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'endpoint': {
        'mode': endpoint_mode,
        'label': endpoint_label or None,
        'base_url': base_url,
        'source_verification': verification,
        'reuse_server_info': reuse_server_info or None,
    },
    'evaluation': {
        'model': model,
        'model_path': model_path,
        'state_contract': state_contract,
        'state_dim': 44 if state_contract == 'state44' else 32,
        'action_dim': 32,
        'lance_dataset': dataset,
        'row_indices': row_ids,
        'normalization_row_indices': normalization,
        'norm_stats_dir': norm_dir,
        'output_dir': output_dir,
        'owner_id': owner,
        'action_session_id': action_session_id or None,
        'chunk_stride': int(stride),
        'temporal_decay': float(decay),
        'act_mode': act_mode,
        'act_batch_size': 1 if act_mode == 'single' else int(batch_size),
        'row_execution': row_execution,
        'row_batch_size': int(row_batch_size),
        'max_warm_request_seconds': float(max_warm),
        'max_frames': int(max_frames),
        'fps': float(fps),
        'width': int(width),
        'height': int(height),
        'video_mode': video_mode,
        'frame_window': frame_window,
        'contact_window_manifest': contact_manifest or None,
        'contact_context_frames': int(contact_context),
        'missing_contact_policy': missing_contact_policy,
        'dataset_reference_video_window': 'full',
        'physics_comparison_video_window': frame_window,
        'language_conditioning': language,
        'gesture_index': gesture_index if language == 'gesture' else None,
        'extended_state': True,
    },
    'dedicated_server': {
        'port': int(server_port),
        'gpus': [int(x) for x in server_gpus.split(',')],
        'runtime_root': runtime_root,
        'cache_dir': cache_dir or None,
        'jax_persistent_executable_cache': persistent_cache == '1',
        'keep_server': keep_server == '1',
    } if endpoint_mode == 'dedicated' else None,
    'provenance': {
        'client_commit': client_commit,
        'client_dirty': client_dirty == 'true',
        'backend_commit': backend_commit,
        'backend_dirty': None if mint_dirty == 'null' else mint_dirty == 'true',
        'model_commit': model_commit,
        'model_dirty': None if openpi_dirty == 'null' else openpi_dirty == 'true',
        'dataset_release_id': release_id,
        'dataset_release_manifest': release_manifest,
        'dataset_release_manifest_sha256': release_sha,
        'norm_stats_sha256': norm_sha,
        'norm_stats_sha256_expected': norm_sha_expected or None,
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

if ((OWN_SERVER)); then
  ENDPOINT_MODE=dedicated
elif ((REUSE_MARKER_VERIFIED)); then
  ENDPOINT_MODE=retained
else
  ENDPOINT_MODE=existing
fi
CONFIG_ARGS=(
  "$CLIENT_COMMIT" "$CLIENT_DIRTY" "$BACKEND_COMMIT" "$OPENPI_COMMIT" "$MINT_DIRTY"
  "$OPENPI_DIRTY" "$NORM_SHA256" "$NORM_SHA_EXPECTED" "$MANO_RELEASE_ID" "$MANO_DATASET_RELEASE"
  "$MANO_RELEASE_SHA256" "$PROVENANCE_VERIFICATION" "$ENDPOINT_MODE"
  "$ENDPOINT_LABEL" "$BASE_URL" "$MODEL" "$MODEL_PATH" "$STATE_CONTRACT" "$DATASET" "$ROWS"
  "$NORMALIZATION_ROWS" "$NORM_STATS_DIR" "$OUTPUT_DIR" "$OWNER_ID" "$CHUNK_STRIDE"
  "$TEMPORAL_DECAY" "$ACT_MODE" "$ACT_BATCH_SIZE" "$ROW_EXECUTION" "$ROW_BATCH_SIZE"
  "$MAX_WARM_REQUEST_SECONDS" "$MAX_FRAMES" "$FPS" "$WIDTH" "$HEIGHT" "$VIDEO_MODE" "$FRAME_WINDOW"
  "$CONTACT_WINDOW_MANIFEST" "$CONTACT_CONTEXT_FRAMES" "$MISSING_CONTACT_POLICY"
  "$LANGUAGE_CONDITIONING" "$GESTURE_INDEX" "$SERVER_PORT" "$SERVER_GPUS"
  "$SERVER_RUNTIME_ROOT" "$SERVER_CACHE_DIR"
  "$ENABLE_JAX_PERSISTENT_CACHE" "$KEEP_SERVER" "$REUSE_SERVER_INFO" "$ACTION_SESSION_ID"
)
if ((PRINT_CONFIG)); then
  write_config "${CONFIG_ARGS[@]}"
  exit 0
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  rm -rf -- "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
write_config "${CONFIG_ARGS[@]}" > "$OUTPUT_DIR/effective_config.json"

server_pid=
run_started=1
cleanup_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
write_failure_marker() {
  local rc=$1
  python3 - "$OUTPUT_DIR/run.failed.json" "$rc" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'status': 'failed', 'exit_code': int(sys.argv[2]),
    'timestamp': datetime.now(timezone.utc).isoformat(),
}, indent=2) + '\n')
PY
}
write_keepalive_marker() {
  [[ -n "$server_pid" ]] || fail "cannot keep server alive without a server PID"
  kill -0 "$server_pid" 2>/dev/null || fail "dedicated server exited before keepalive handoff"
  python3 - "$OUTPUT_DIR/server.keepalive.json" "$server_pid" "$BASE_URL" \
    "$SERVER_PORT" "$SERVER_GPUS" "$SERVER_RUNTIME_ROOT" "$OWNER_ID" \
    "$CLIENT_COMMIT" "$BACKEND_COMMIT" "$OPENPI_COMMIT" "$ACTION_SESSION_ID" \
    "$ACTION_SESSION_MARKER" "$MODEL" "$MODEL_PATH" "$ACT_MODE" "$ACT_BATCH_SIZE" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
(
    path, pid, base_url, port, gpus, runtime_root, owner_id,
    client_commit, backend_commit, model_commit, action_session_id,
    action_session_marker, model, model_path, act_mode, act_batch_size,
) = sys.argv[1:]
Path(path).write_text(json.dumps({
    'status': 'owned_running',
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'pid': int(pid),
    'base_url': base_url,
    'port': int(port),
    'gpus': [int(x) for x in gpus.split(',')],
    'runtime_root': runtime_root,
    'owner_id': owner_id,
    'client_commit': client_commit,
    'backend_commit': backend_commit,
    'model_commit': model_commit,
    'action_session_id': action_session_id,
    'action_session_marker': action_session_marker,
    'model': model,
    'model_path': model_path,
    'act_mode': act_mode,
    'act_batch_size': 1 if act_mode == 'single' else int(act_batch_size),
    'source_output': str(Path(path).parent),
}, indent=2) + '\n')
PY
}
on_exit() {
  local rc=$?
  trap - EXIT
  cleanup_server
  if ((run_started && rc != 0)); then
    write_failure_marker "$rc" || true
  fi
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ((OWN_SERVER)); then
  SERVER_CMD=("$SCRIPT_DIR/run_action_lora_server.sh"
    --runtime-root "$SERVER_RUNTIME_ROOT" --port "$SERVER_PORT" --gpus "$SERVER_GPUS"
    --mint-root "$MINT_CODE_ROOT" --openpi-root "$MINT_OPENPI_ROOT" --python-bin "$MINT_PYTHON_BIN")
  if ((ENABLE_JAX_PERSISTENT_CACHE)); then
    SERVER_CMD+=(--enable-jax-persistent-cache)
    [[ -z "$SERVER_CACHE_DIR" ]] || SERVER_CMD+=(--cache-dir "$SERVER_CACHE_DIR")
  fi
  if ((KEEP_SERVER)); then
    nohup "${SERVER_CMD[@]}" > "$OUTPUT_DIR/server.log" 2>&1 &
  else
    "${SERVER_CMD[@]}" > "$OUTPUT_DIR/server.log" 2>&1 &
  fi
  server_pid=$!
  printf '%s\n' "$server_pid" > "$OUTPUT_DIR/server.pid"
  ready=000
  for _ in $(seq 1 240); do
    ready=$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
      --max-time 5 "$BASE_URL/openapi.json" 2>/dev/null || true)
    [[ "$ready" == 200 ]] && break
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 1
  done
  if [[ "$ready" != 200 ]]; then
    tail -n 80 "$OUTPUT_DIR/server.log" >&2 || true
    fail "dedicated MINT server did not become ready"
  fi
else
  ready=$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --max-time 10 "$BASE_URL/openapi.json" || true)
  [[ "$ready" == 200 ]] || fail "MINT endpoint preflight failed: $BASE_URL/openapi.json returned ${ready:-no response}"
fi

EVAL_ARGS=(
  --base-url "$BASE_URL" --api-key "$MINT_API_KEY" --model "$MODEL"
  --model-path "$MODEL_PATH" --owner-id "$OWNER_ID" --lance-dataset "$DATASET"
  --row-indices "$ROWS" --language-conditioning "$LANGUAGE_CONDITIONING"
  --normalization-row-indices "$NORMALIZATION_ROWS" --state-contract "$STATE_CONTRACT"
  --norm-stats-dir "$NORM_STATS_DIR" --output-dir "$OUTPUT_DIR/artifacts"
  --chunk-stride "$CHUNK_STRIDE" --temporal-decay "$TEMPORAL_DECAY"
  --act-mode "$ACT_MODE" --act-batch-size "$ACT_BATCH_SIZE"
  --row-execution "$ROW_EXECUTION" --row-batch-size "$ROW_BATCH_SIZE"
  --max-warm-request-seconds "$MAX_WARM_REQUEST_SECONDS" --max-frames "$MAX_FRAMES"
  --fps "$FPS" --width "$WIDTH" --height "$HEIGHT" --video-mode "$VIDEO_MODE"
  --frame-window "$FRAME_WINDOW" --contact-context-frames "$CONTACT_CONTEXT_FRAMES"
  --missing-contact-policy "$MISSING_CONTACT_POLICY"
  --client-commit "$CLIENT_COMMIT" --backend-commit "$BACKEND_COMMIT"
  --model-commit "$OPENPI_COMMIT"
)
if [[ -n "$NORM_SHA_EXPECTED" ]]; then
  EVAL_ARGS+=(--norm-sha-expected "$NORM_SHA_EXPECTED")
fi
if ((KEEP_SERVER)); then
  EVAL_ARGS+=(--keep-action-session)
elif [[ -n "$ACTION_SESSION_ID" ]]; then
  EVAL_ARGS+=(--action-session-id "$ACTION_SESSION_ID")
fi
if [[ "$FRAME_WINDOW" == contact ]]; then
  EVAL_ARGS+=(--contact-window-manifest "$CONTACT_WINDOW_MANIFEST")
fi
if [[ "$LANGUAGE_CONDITIONING" == gesture ]]; then
  EVAL_ARGS+=(--gesture-index "$GESTURE_INDEX")
fi

VLA_CLIENT_CONFIG=/dev/null MINT_CODE_ROOT="$MINT_CODE_ROOT" \
MINT_OPENPI_ROOT="$MINT_OPENPI_ROOT" MINT_PYTHON_BIN="$MINT_PYTHON_BIN" \
MINT_BASE_URL="$BASE_URL" \
MINT_API_KEY="$MINT_API_KEY" \
  "$SCRIPT_DIR/run_client.sh" scripts/eval/infer_mano_mode4.py "${EVAL_ARGS[@]}" \
  > "$OUTPUT_DIR/eval.log" 2>&1

SUMMARY_PATH="$OUTPUT_DIR/artifacts/summary.json"
[[ -s "$SUMMARY_PATH" ]] || fail "Mode4 completed without summary: $SUMMARY_PATH"
if ((KEEP_SERVER)); then
  ACTION_SESSION_MARKER="$OUTPUT_DIR/artifacts/action_session.retained.json"
  [[ -s "$ACTION_SESSION_MARKER" ]] || fail "Mode4 did not retain its action session: $ACTION_SESSION_MARKER"
  ACTION_SESSION_ID=$(python3 - "$ACTION_SESSION_MARKER" "$BASE_URL" "$MODEL_PATH" "$OWNER_ID" <<'PY'
import json, sys
from pathlib import Path
marker=json.loads(Path(sys.argv[1]).read_text())
if marker.get('status') != 'retained': raise SystemExit('action session marker is not retained')
if marker.get('base_url') != sys.argv[2]: raise SystemExit('action session base_url mismatch')
if marker.get('model_path') != sys.argv[3]: raise SystemExit('action session model_path mismatch')
if marker.get('owner_id') != sys.argv[4]: raise SystemExit('action session owner mismatch')
session_id=marker.get('action_session_id')
if not session_id: raise SystemExit('action session marker is missing its ID')
print(session_id)
PY
  ) || fail "invalid retained action-session marker"
  CONFIG_ARGS[$((${#CONFIG_ARGS[@]} - 1))]=$ACTION_SESSION_ID
  write_config "${CONFIG_ARGS[@]}" > "$OUTPUT_DIR/effective_config.json.tmp"
  mv "$OUTPUT_DIR/effective_config.json.tmp" "$OUTPUT_DIR/effective_config.json"
  write_keepalive_marker
  # on_exit must not clean a server and session explicitly handed off to the operator.
  server_pid=
fi
python3 - "$OUTPUT_DIR/run.completed.json" "$SUMMARY_PATH" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'status': 'completed',
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'effective_config': str(Path(sys.argv[1]).with_name('effective_config.json')),
    'summary': sys.argv[2],
}, indent=2) + '\n')
PY
rm -f "$OUTPUT_DIR/run.failed.json"
