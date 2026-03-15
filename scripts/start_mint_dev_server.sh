#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
. ./configs/dev_volcano.env.sh
set +a

# 0.6B sidecar: runs without auth. Explicitly unset auth vars in case they are
# present in the calling shell's environment (e.g. from a previous session).
(
  unset TINKER_API_KEY TOKEN_SECRET_KEY
  TINKER_PORT="$GATEWAY_06B_PORT"
  TINKER_RAY_NAMESPACE="$GATEWAY_06B_NAMESPACE"
  MINT_RAY_NAMESPACE="$GATEWAY_06B_NAMESPACE"
  MINT_SUPPORTED_MODELS="$GATEWAY_06B_MODEL"
  MINT_PERSISTENT_MODELS="$GATEWAY_06B_MODEL"
  MINT_MEGATRON_NODE_IPS_CSV="$GATEWAY_06B_NODE_IP"
  MINT_MODEL_NODE_IPS_JSON="{\"$GATEWAY_06B_MODEL\":[\"$GATEWAY_06B_NODE_IP\"]}"
  MINT_VLLM_PINNED_NODE_IP_JSON="{\"$GATEWAY_06B_MODEL\":\"$GATEWAY_06B_NODE_IP\"}"
  TINKER_GATEWAY_CONFIG_JSON=
  exec "$PYTHON_BIN" scripts/run_server.py
) >>/tmp/tinker_server_06b.log 2>&1 &

# 30B main server.
[ -f .secrets.env ] && . ./.secrets.env
exec "$PYTHON_BIN" scripts/run_server.py
