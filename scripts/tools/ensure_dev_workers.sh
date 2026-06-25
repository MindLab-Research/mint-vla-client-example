#!/usr/bin/env bash
#
# Ensure N mint-dev GPU workers are running on the Volcano ML Platform.
#
# Usage:
#   scripts/tools/ensure_dev_workers.sh <count> [--queue-id <id>] [--template <path>]
#
# Examples:
#   ensure_dev_workers.sh 4                    # ensure workers 1-4 are running
#   ensure_dev_workers.sh 4 --dry-run          # check only, don't submit
#   ensure_dev_workers.sh 4 --queue-id q-xxx   # override queue ID
#
# Requires: volc CLI in PATH (v1.2+), credentials in ~/.volc/credentials
#

set -euo pipefail

# --- defaults ---
DEFAULT_QUEUE_ID="q-20251126180002-26lwz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_TEMPLATE="${REPO_ROOT}/docker/volc/dev-worker.yaml"

# --- args ---
COUNT=""
QUEUE_ID="${DEFAULT_QUEUE_ID}"
TEMPLATE="${DEFAULT_TEMPLATE}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue-id)   QUEUE_ID="$2"; shift 2 ;;
    --template)   TEMPLATE="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *)
      if [[ -z "$COUNT" ]]; then
        COUNT="$1"
      else
        echo "error: unexpected argument: $1" >&2; exit 1
      fi
      shift ;;
  esac
done

if [[ -z "$COUNT" ]]; then
  echo "usage: $0 <count> [--queue-id <id>] [--template <path>] [--dry-run]" >&2
  exit 1
fi

if ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: count must be a positive integer, got: $COUNT" >&2
  exit 1
fi

# --- locate volc CLI ---
VOLC_BIN=""
for candidate in "$(command -v volc 2>/dev/null || true)" \
                 "$HOME/.volc/bin/volc" \
                 "/root/.volc/bin/volc"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    VOLC_BIN="$candidate"
    break
  fi
done

if [[ -z "$VOLC_BIN" ]]; then
  echo "error: volc CLI not found in PATH or ~/.volc/bin/" >&2
  echo "hint: install with: curl -fsSL https://www.volcengine.com/docs/6459/80261 | bash" >&2
  exit 1
fi

if [[ ! -f "$TEMPLATE" ]]; then
  echo "error: template not found: $TEMPLATE" >&2
  exit 1
fi

# Suppress volc metrics noise (stderr)
export VOLC_LOG_LEVEL=error 2>/dev/null || true

# --- get currently running workers ---
echo "📋 Querying running mint-dev-worker tasks..."
RUNNING_JSON=$("$VOLC_BIN" ml_task list --output json 2>/dev/null | tail -1)

if [[ -z "$RUNNING_JSON" ]]; then
  echo "error: failed to query task list" >&2
  exit 1
fi

# Extract running worker names (mint-dev-worker-{idx} pattern)
RUNNING_NAMES=$(echo "$RUNNING_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for t in data:
    name = t.get('JobName', '')
    status = t.get('Status', '')
    if name.startswith('mint-dev-worker-') and status in ('Running', 'Queue', 'Staging'):
        print(name)
" 2>/dev/null || echo "")

# --- check each worker 1..count ---
MISSING=()
for i in $(seq 1 "$COUNT"); do
  NAME="mint-dev-worker-${i}"
  if echo "$RUNNING_NAMES" | grep -qx "$NAME"; then
    echo "  ✅ ${NAME} — already running"
  else
    echo "  ❌ ${NAME} — missing"
    MISSING+=("$i")
  fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo ""
  echo "🎉 All ${COUNT} workers are running. Nothing to do."
  exit 0
fi

echo ""
echo "🔧 Need to start ${#MISSING[@]} worker(s): ${MISSING[*]}"

if $DRY_RUN; then
  echo "(dry-run mode — skipping submission)"
  exit 0
fi

# --- submit missing workers ---
TMPDIR_WORK=$(mktemp -d)
trap 'rm -rf "$TMPDIR_WORK"' EXIT

for i in "${MISSING[@]}"; do
  NAME="mint-dev-worker-${i}"
  TMP_YAML="${TMPDIR_WORK}/${NAME}.yaml"

  # Render template: set task name and queue ID
  python3 -c "
import yaml, sys, copy

with open('$TEMPLATE') as f:
    cfg = yaml.safe_load(f)

cfg['TaskName'] = '$NAME'
cfg['ResourceQueueID'] = '$QUEUE_ID'

# Normalize storage fields: ensure 'Id' key exists
for s in cfg.get('Storages', []):
    if 'VepfsId' in s and 'Id' not in s:
        s['Id'] = s.pop('VepfsId')

with open('$TMP_YAML', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
"

  echo ""
  echo "🚀 Submitting ${NAME}..."
  RESULT=$("$VOLC_BIN" ml_task submit -c "$TMP_YAML" 2>&1 || true)
  TASK_ID=$(echo "$RESULT" | grep -oE 'task_id=[^ ]+' | cut -d= -f2 || echo "")

  if [[ -n "$TASK_ID" ]]; then
    echo "  ✅ ${NAME} submitted: ${TASK_ID}"
  else
    echo "  ❌ ${NAME} failed: ${RESULT}" >&2
    echo "$RESULT" | grep -v "Metrics" | tail -5 >&2
  fi
done

echo ""
echo "✅ Done. Use '$VOLC_BIN ml_task list -n mint-dev-worker' to check status."
