#!/usr/bin/env bash
# 读取合并了推理结果的 Lance 数据集(封装解释器 + PYTHONPATH)。
#
# 用法:
#   bash lance_read.sh                              # 读默认 /tmp/pi05_replay_merged.lance
#   bash lance_read.sh /path/to/other.lance         # 读指定数据集
set -euo pipefail

GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LANCE_PATH="${1:-/tmp/pi05_replay_merged.lance}"

export PYTHONPATH="/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:${GRB}/site-packages"
exec "${GRB}/host-venv/bin/python" \
  "${HERE}/scripts/wip/read_replay_lance.py" "${LANCE_PATH}"
