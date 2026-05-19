#!/usr/bin/env bash
set -euo pipefail

NS="${MINT_RAY_NAMESPACE:?set MINT_RAY_NAMESPACE}"
PORT="${MINT_PORT:-10427}"
ROOT="${ISSUE_SERVER_ROOT:-/root/mint_project/mint-server-issue-416}"
MINT_ROOT="${MINT_CODE_ROOT:?set MINT_CODE_ROOT}"
LOG="${ISSUE_LOG_FILE:-/tmp/mint_server_issue_416_r17.log}"
RAY_ADDR="${RAY_ADDRESS:?set RAY_ADDRESS=ray://<head>:10001}"
PY="${MINT_HOST_PYTHON:-/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python}"
SSH_HOST="${MINT_DEV_SSH_HOST:-mint-dev}"

ssh "$SSH_HOST" 'bash -s' <<EOF
set -euo pipefail
NS="$NS"
PORT="$PORT"
ROOT="$ROOT"
MINT_ROOT="$MINT_ROOT"
LOG="$LOG"
RAY_ADDR="$RAY_ADDR"
PY="$PY"

PID=\$(ps eww -C python -o pid= -o args= | grep "scripts/run_server.py" | grep "\$NS" | awk "NR==1{print \\\$1}" || true)
if [ -n "\$PID" ]; then
  kill "\$PID" || true
  sleep 2
  kill -9 "\$PID" 2>/dev/null || true
fi

RAY_ADDRESS="\$RAY_ADDR" MINT_RAY_NAMESPACE="\$NS" MINT_RAY_NAMESPACE="\$NS" "\$PY" - <<'PYCODE'
import json
import os
import ray
from ray.util.placement_group import PlacementGroup, placement_group_table, remove_placement_group

ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
ns = os.environ["MINT_RAY_NAMESPACE"]
killed = 0
for actor in ray.util.list_named_actors(all_namespaces=True):
    if actor.get("namespace") != ns:
        continue
    try:
        ray.kill(ray.get_actor(actor["name"], namespace=ns))
        killed += 1
    except Exception:
        pass

removed = []
for pgid, info in placement_group_table().items():
    if info.get("state") == "REMOVED":
        continue
    bundles = info.get("bundles") or []
    if "192.168.39.94" not in json.dumps(bundles):
        continue
    pg = PlacementGroup(ray._raylet.PlacementGroupID.from_hex(pgid))
    remove_placement_group(pg)
    removed.append({"id": pgid, "name": info.get("name")})

print(json.dumps({"killed_actors": killed, "removed_pgs": removed}, indent=2))
PYCODE

cd "\$ROOT"
nohup env \
  ISSUE_SERVER_ROOT="\$ROOT" \
  ISSUE_NAMESPACE="\$NS" \
  ISSUE_STARTUP_LEASE=mint_startup_lease_issue_416_r17 \
  ISSUE_PORT="\$PORT" \
  ISSUE_LOG_FILE="\$LOG" \
  ISSUE_USAGE_LOG_DIR=/tmp/mint_usage_issue_416_r17 \
  ISSUE_SUPPORTED_MODELS=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  RAY_ADDRESS="\$RAY_ADDR" \
  MINT_RAY_CLIENT_ADDRESS="\$RAY_ADDR" \
  MINT_CODE_ROOT="\$MINT_ROOT" \
  ISSUE_MODEL_PLACEMENT_JSON='{"Qwen/Qwen3-30B-A3B-Instruct-2507":{"replica":0,"node_ip":"192.168.39.159","gpu_count":4}}' \
  ISSUE_MEGATRON_MODEL_PLACEMENT_JSON='{"Qwen/Qwen3-30B-A3B-Instruct-2507":{"replica":0,"node_ip":"192.168.39.159","gpu_count":4}}' \
  bash scripts/tools/start_issue_server.sh >> "\$LOG" 2>&1 &

for _ in \$(seq 1 180); do
  if curl -sf "http://localhost:\$PORT/api/v1/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl -sf "http://localhost:\$PORT/api/v1/healthz"

cd /vePFS-Mindverse/share/code/yiwen/mint-server-issue-416
MINT_BASE_URL="http://localhost:\$PORT" \
MINT_API_KEY=dummy \
RAY_ADDRESS="\$RAY_ADDR" \
MINT_RAY_NAMESPACE="\$NS" \
MINT_DEV_SSH_HOST=local \
"\$PY" scripts/tools/reproduce_issue_476.py

MINT_BASE_URL="http://localhost:\$PORT" \
MINT_API_KEY=dummy \
RAY_ADDRESS="\$RAY_ADDR" \
MINT_RAY_NAMESPACE="\$NS" \
MINT_DEV_SSH_HOST=local \
"\$PY" scripts/tools/reproduce_issue_416.py
EOF
