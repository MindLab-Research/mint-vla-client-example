#!/usr/bin/env bash
# Run one phase-aware persistent State45 rollout against an existing MINT server.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
MODEL=openpi/pi05-action-lora-r16-state45-phase-28dof-finetune
BASE_URL=${MINT_BASE_URL:-http://127.0.0.1:30532}
API_KEY=${MINT_API_KEY:-tml-dummy}
MAX_CONTROL_SECONDS=15
CHUNK_STRIDE=1
VIDEO_MODE=none
PRINT_CONFIG=0
DATASET=
ROW=
NORM_DIR=
NORM_SHA=
WINDOW_MANIFEST=
MODEL_PATH=
OWNER_ID=
OUTPUT_DIR=
while (($#)); do
  case "$1" in
    --base-url) BASE_URL=${2:?}; shift 2 ;;
    --api-key) API_KEY=${2:?}; shift 2 ;;
    --lance-dataset) DATASET=${2:?}; shift 2 ;;
    --row) ROW=${2:?}; shift 2 ;;
    --norm-stats-dir) NORM_DIR=${2:?}; shift 2 ;;
    --norm-sha-expected) NORM_SHA=${2:?}; shift 2 ;;
    --contact-window-manifest) WINDOW_MANIFEST=${2:?}; shift 2 ;;
    --model-path) MODEL_PATH=${2:?}; shift 2 ;;
    --owner-id) OWNER_ID=${2:?}; shift 2 ;;
    --output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
    --max-control-seconds) MAX_CONTROL_SECONDS=${2:?}; shift 2 ;;
    --chunk-stride) CHUNK_STRIDE=${2:?}; shift 2 ;;
    --video-mode) VIDEO_MODE=${2:?}; shift 2 ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
for value in DATASET ROW NORM_DIR NORM_SHA WINDOW_MANIFEST MODEL_PATH OWNER_ID OUTPUT_DIR; do
  [[ -n "${!value}" ]] || { echo "missing required ${value}" >&2; exit 64; }
done
[[ "$ROW" =~ ^[0-9]+$ ]] || { echo "--row must be a non-negative integer" >&2; exit 64; }
[[ "$CHUNK_STRIDE" =~ ^[1-9]$ ]] || { echo "--chunk-stride must be in [1,9]" >&2; exit 64; }
[[ "$VIDEO_MODE" == none || "$VIDEO_MODE" == full ]] || { echo "invalid --video-mode" >&2; exit 64; }
python3 - "$MAX_CONTROL_SECONDS" <<'PY'
import math,sys
value=float(sys.argv[1])
if not math.isfinite(value) or value<=0: raise SystemExit('--max-control-seconds must be finite and positive')
PY
if ((PRINT_CONFIG)); then
  python3 - <<PY
import json
print(json.dumps({
  'model':'$MODEL','state_contract':'state45','state_dim':45,
  'action_dim':32,'action_horizon':10,'frame_window':'persistent_task',
  'max_control_seconds':float('$MAX_CONTROL_SECONDS'),
  'chunk_stride':int('$CHUNK_STRIDE'),'row':int('$ROW'),
  'video_mode':'$VIDEO_MODE'
},indent=2,sort_keys=True))
PY
  exit 0
fi
exec "$REPO_ROOT/scripts/remote/run_client.sh" \
  scripts/eval/infer_mano_mode4_state45.py \
  --base-url "$BASE_URL" --api-key "$API_KEY" --model "$MODEL" \
  --model-path "$MODEL_PATH" --owner-id "$OWNER_ID" \
  --lance-dataset "$DATASET" --row-indices "$ROW" \
  --normalization-row-indices "$ROW" --state-contract state45 \
  --norm-stats-dir "$NORM_DIR" --norm-sha-expected "$NORM_SHA" \
  --output-dir "$OUTPUT_DIR" --language-conditioning gesture \
  --contact-window-manifest "$WINDOW_MANIFEST" --contact-context-frames 100 \
  --missing-contact-policy error --frame-window persistent_task \
  --max-control-seconds "$MAX_CONTROL_SECONDS" --chunk-stride "$CHUNK_STRIDE" \
  --act-mode single --act-batch-size 1 --row-execution sequential \
  --row-batch-size 1 --video-mode "$VIDEO_MODE"
