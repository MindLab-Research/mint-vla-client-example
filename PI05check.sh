#!/usr/bin/env bash
# PI05check.sh — 本机化 openpi pi0.5 端到端冒烟（仿 RLcheck_local.sh）
#
# 跑通 openpi/pi05-libero-low-mem-finetune（pi0.5 flow-matching VLA）的全链路：
#   create_model -> train_step(flow_matching) -> save_weights_for_sampler
#   -> action_session -> act
#
# 差异 vs VLA_check.sh（三机 ssh 隧道版）：
#   - 无 ssh 隧道：server 就在本机 :30496，MINT_BASE_URL 直指它。
#   - client 只用 requests，复用仓库内 .venv-mindlab（与 server runtime 隔离）。
#
# 前置：
#   step0(ray) -> step1(rsync，含本次 openpi_ray_runtime.py 改动) ->
#   step2(placement) -> step3(server，已加 pi05 supported model + checkpoint/
#   assets/weights env) -> step4(healthz ready)
#   并且 gpu_rl tier 已装 JAX cuda13 栈（jax0.7.2/orbax0.11.40/flax0.10.2…）。
set -uo pipefail

PORT="${MINT_PORT:-30496}"
CLIENT_PY="${MINT_CLIENT_PYTHON:-/vePFS-Mindverse/user/intern/wenxi/mint/.venv-mindlab/bin/python}"
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi}"
MODEL="${MINT_PI05_MODEL:-openpi/pi05-libero-low-mem-finetune}"
OUT_JSON="${MINT_PI05_OUTPUT_JSON:-/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_check.json}"

if [ ! -x "${CLIENT_PY}" ]; then
  echo "error: client python 不存在: ${CLIENT_PY}" >&2
  echo "       期望仓库内的 .venv-mindlab（提供 requests）。" >&2
  exit 1
fi

DRIVER="${CODE_ROOT}/scripts/wip/openpi_vla_smoke.py"
if [ ! -f "${DRIVER}" ]; then
  echo "error: smoke driver 不存在: ${DRIVER}" >&2
  exit 1
fi

echo "== pi0.5 端到端冒烟 =="
echo "   base_url = http://localhost:${PORT}"
echo "   model    = ${MODEL}"
echo "   driver   = ${DRIVER}"
echo "   output   = ${OUT_JSON}"

MINT_BASE_URL="http://localhost:${PORT}" \
TINKER_API_KEY=tml-dummy \
MINT_API_KEY=tml-dummy \
"${CLIENT_PY}" "${DRIVER}" \
  --model "${MODEL}" \
  --output-json "${OUT_JSON}"
rc=$?

if [ "${rc}" -ne 0 ]; then
  echo "FAILED: smoke driver 退出码 ${rc}" >&2
  exit "${rc}"
fi
echo "OK: pi0.5 全链路通过，结果见 ${OUT_JSON}"
