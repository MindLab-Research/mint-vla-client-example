#!/usr/bin/env bash
# RLcheck_local.sh — 本机化 RLcheck（替代 RLcheck.sh 的 ssh 隧道版）
#
# 差异 vs 三机版：
#   - 无 ssh 隧道：server 就在本机 :30496，MINT_BASE_URL 直指它。
#   - client 用独立 venv（mindlab-toolkit 提供 import mint），与 server runtime 隔离。
#
# 前置：step0(ray) -> step1(rsync) -> step2(placement) -> step3(server) -> step4(healthz ready)
set -uo pipefail

PORT="${MINT_PORT:-30496}"
CLIENT_PY="${MINT_CLIENT_PYTHON:-/vePFS-Mindverse/user/intern/wenxi/mint/.venv-mindlab/bin/python}"
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi}"

if [ ! -x "${CLIENT_PY}" ]; then
  echo "error: client python 不存在: ${CLIENT_PY}" >&2
  echo "       期望仓库内的 .venv-mindlab（已装 mindlab-toolkit，提供 import mint）。" >&2
  exit 1
fi

MINT_BASE_URL="http://localhost:${PORT}" \
TINKER_API_KEY=tml-dummy \
MINT_API_KEY=tml-dummy \
"${CLIENT_PY}" "${CODE_ROOT}/scripts/tools/rl_check.py" \
  --model Qwen/Qwen3.6-27B \
  --steps 10 \
  --group-size 4 \
  --timeout-s 600
