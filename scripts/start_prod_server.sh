#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
prod_config_env="${MINT_PROD_CONFIG_ENV:-/vePFS-Mindverse/share/mint/prod/config/prod.env}"
if [ ! -r "${prod_config_env}" ]; then
  echo "missing prod config: ${prod_config_env}" >&2
  exit 1
fi
. "${prod_config_env}"
prod_secrets_env="${MINT_PROD_SECRETS_ENV:-/vePFS-Mindverse/share/mint/prod/config/secrets.env}"
if [ ! -r "${prod_secrets_env}" ]; then
  echo "missing prod secrets: ${prod_secrets_env}" >&2
  exit 1
fi
. "${prod_secrets_env}"
set +a

if [ -z "${MINT_RUNTIME_CHECKPOINT_DIR:-}" ]; then
  export MINT_RUNTIME_CHECKPOINT_DIR="${TINKER_RUNTIME_CHECKPOINT_DIR:-/vePFS-Mindverse/share/mint/prod/data/runtime-checkpoints}"
fi

if [ -n "${MINT_GATEWAY_GLM51_BASE_URL:-}" ]; then
  if [ "${MINT_GATEWAY_GLM51_AUTH_MODE:-static_api_key}" != "static_api_key" ]; then
    echo "unsupported GLM5.1 gateway auth mode: ${MINT_GATEWAY_GLM51_AUTH_MODE}" >&2
    exit 1
  fi
  if [ -z "${MINT_API_KEY:-}" ]; then
    echo "missing MINT_API_KEY for GLM5.1 static gateway auth" >&2
    exit 1
  fi
  export MINT_GATEWAY_CONFIG_JSON="$(python - <<'PY'
import json
import os

raw = os.environ.get("MINT_GATEWAY_CONFIG_JSON", "").strip()
cfg = json.loads(raw) if raw else {}
model_to_upstream = dict(cfg.get("model_to_upstream") or {})
upstreams = dict(cfg.get("upstreams") or {})
alias = os.environ["MINT_GATEWAY_GLM51_ALIAS"].strip()
model = os.environ["MINT_GATEWAY_GLM51_MODEL"].strip()
base_url = os.environ["MINT_GATEWAY_GLM51_BASE_URL"].strip().rstrip("/")
if not alias or not model or not base_url:
    raise SystemExit("GLM5.1 gateway config is incomplete")
model_to_upstream[model] = alias
upstreams[alias] = {
    "base_url": base_url,
    "auth_mode": "static_api_key",
    "api_key": os.environ["MINT_API_KEY"].strip(),
}
cfg["model_to_upstream"] = model_to_upstream
cfg["upstreams"] = upstreams
print(json.dumps(cfg, separators=(",", ":")))
PY
)"
fi

api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"
api_tmp_link="/tmp/mpa"
mkdir -p "$api_tmp_root"
if [ -L "$api_tmp_link" ] || [ -f "$api_tmp_link" ]; then
  rm -f "$api_tmp_link"
elif [ -e "$api_tmp_link" ]; then
  echo "refusing to replace non-link temp path: $api_tmp_link" >&2
  exit 1
fi
ln -s "$api_tmp_root" "$api_tmp_link"

export TMPDIR="${api_tmp_link}/t"
export XDG_CACHE_HOME="${api_tmp_link}/c"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MINT_RUNTIME_CHECKPOINT_DIR}"

ray_node_ip="$(python - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
PY
)"
ray_temp_dir="${api_tmp_link}/ray"
mkdir -p "${ray_temp_dir}"
"${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/ray" start \
  --address="${RAY_ADDRESS}" \
  --node-ip-address="${ray_node_ip}" \
  --num-cpus=0 \
  --num-gpus=0 \
  --temp-dir="${ray_temp_dir}" \
  --disable-usage-stats >/tmp/mint_prod_api_ray_start.log 2>&1 || {
    cat /tmp/mint_prod_api_ray_start.log >&2
    exit 1
  }

"${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/bootstrap_control_plane.py
exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py
