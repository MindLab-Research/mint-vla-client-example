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
  export MINT_RUNTIME_CHECKPOINT_DIR="/vePFS-Mindverse/share/mint/prod/data/runtime-checkpoints"
fi
if [ -z "${MINT_CODE_ROOT:-}" ]; then
  export MINT_CODE_ROOT="$repo_root"
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
