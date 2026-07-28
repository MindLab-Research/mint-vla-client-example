#!/usr/bin/env bash
# Stop only a server handed off by run_mode4_eval.sh --keep-server.
set -Eeuo pipefail

INFO=${1:?usage: stop_owned_mode4_server.sh SERVER_KEEPALIVE_JSON}
[[ -f "$INFO" ]] || { echo "keepalive marker missing: $INFO" >&2; exit 2; }

readarray -t FIELDS < <(python3 - "$INFO" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
d=json.loads(p.read_text())
if d.get('status') != 'owned_running':
    raise SystemExit(f"marker status is not owned_running: {d.get('status')!r}")
for key in ('pid','base_url','port','runtime_root'):
    if key not in d:
        raise SystemExit(f"keepalive marker missing {key}")
print(d['pid'])
print(d['base_url'])
print(d['port'])
print(d['runtime_root'])
PY
)
PID=${FIELDS[0]}
BASE_URL=${FIELDS[1]}
PORT=${FIELDS[2]}
RUNTIME_ROOT=${FIELDS[3]}
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "invalid PID in keepalive marker" >&2; exit 2; }

if ! kill -0 "$PID" 2>/dev/null; then
  echo "server PID $PID is not running; refusing to rewrite ownership state" >&2
  exit 1
fi
CMDLINE=$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true)
[[ "$CMDLINE" == *"mint_server"* || "$CMDLINE" == *"uvicorn"* ]] || {
  echo "PID $PID is not recognizably a MINT/uvicorn server; refusing to stop it" >&2
  exit 1
}

curl --silent --show-error --max-time 5 "$BASE_URL/openapi.json" >/dev/null 2>&1 || {
  echo "server ownership check failed: $BASE_URL/openapi.json is not reachable" >&2
  exit 1
}

echo "stopping owned MINT server pid=$PID port=$PORT runtime_root=$RUNTIME_ROOT"
kill -INT "$PID" 2>/dev/null || true
for _ in $(seq 1 60); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
  echo "server did not stop after SIGINT; sending SIGTERM" >&2
  kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
fi
if kill -0 "$PID" 2>/dev/null; then
  echo "owned server still running after graceful stop" >&2
  exit 1
fi
python3 - "$INFO" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
d['status']='stopped'
d['stopped_at']=datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(d, indent=2)+'\n')
PY
echo "owned server stopped"
