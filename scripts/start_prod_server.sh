#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
. ./configs/prod_volcano.env.sh
. ./.secrets.env
set +a

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
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TINKER_RUNTIME_CHECKPOINT_DIR}"

exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py
