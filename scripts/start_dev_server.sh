#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
. ./configs/dev_volcano.env.sh
set +a

api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"
api_tmp_link="/tmp/mda"
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
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TINKER_RUNTIME_CHECKPOINT_DIR}"

ray_head_ip_path="${MINT_RAY_HEAD_ADDRESS_PATH:-/vePFS-Mindverse/share/code/tinker-server/ray_head_ip.txt}"
ray_head_ip=""
if [ -r "$ray_head_ip_path" ]; then
  ray_head_ip=$(tr -d '[:space:]' < "$ray_head_ip_path")
fi
if [ -n "$ray_head_ip" ]; then
  case "${RAY_ADDRESS:-}" in
    ray://*) ;;
    *:6379|"") export RAY_ADDRESS="ray://${ray_head_ip}:10001" ;;
  esac
fi
if [ -z "${MINT_RAY_CLIENT_ADDRESS:-}" ]; then
  case "${RAY_ADDRESS:-}" in
    ray://*) export MINT_RAY_CLIENT_ADDRESS="${RAY_ADDRESS}" ;;
  esac
fi

exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py
