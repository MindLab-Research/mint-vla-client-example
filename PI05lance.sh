#!/usr/bin/env bash
# PI05lance.sh — 用真实 lance 数据集跑 openpi pi0.5 全链路冒烟（仿 PI05check.sh）
#
# 与 PI05check.sh 的差异：
#   - driver 换成 scripts/wip/openpi_vla_smoke_lance.py（读 lance 数据集，
#     走完整 openpi transform：LiberoInputs→Normalize→PaliGemma分词→Pad）。
#   - client 解释器换成 gpu_rl host-venv，并拼三段 PYTHONPATH，使其同时能
#     import openpi(+jax+sentencepiece) 与 lance(pylance)。
#
# 前置（与 PI05check.sh 相同）：
#   step0(ray) -> step1(rsync) -> step3(server) -> step4(healthz ready)
#   且 gpu_rl tier 已装 JAX cuda13 栈；lance 依赖已 --no-deps 装进 extra-pydeps。
#
# 用法：
#   bash PI05lance.sh                 # 默认 4 步、batch 2、连活跃 server
#   bash PI05lance.sh --dry-run       # 只读 lance + 组 batch，不连 server
#   MINT_LANCE_DATASET=/path bash PI05lance.sh   # 换数据集
set -uo pipefail

PORT="${MINT_PORT:-30496}"
# 固定本地 runtime（与 step0/step3 一致，故意不读外部可能过期的 PFS_RUNTIME_ENV_ROOT）。
PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime
RUNTIME_ROOT="${PFS_RUNTIME_ENV_ROOT}"
GRB="${RUNTIME_ROOT}/gpu_rl"
CLIENT_PY="${MINT_CLIENT_PYTHON:-${GRB}/host-venv/bin/python}"
EXTRA_PYDEPS="${MINT_EXTRA_PYDEPS:-/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps}"

# client driver 在本机跑，直接用本仓库副本（无需 worker 可见）。
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint}"
DRIVER="${CODE_ROOT}/scripts/wip/openpi_vla_smoke_lance.py"

MODEL="${MINT_PI05_MODEL:-openpi/pi05-libero-low-mem-finetune}"
LANCE_DS="${MINT_LANCE_DATASET:-/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_lance_smoke.lance}"
STEPS="${MINT_PI05_STEPS:-1000}"
BATCH="${MINT_PI05_BATCH:-2}"
OUT_JSON="${MINT_PI05_OUTPUT_JSON:-/tmp/pi05_lance_smoke.json}"

# openpi 资产缓存（PaliGemma tokenizer 等）与 HF 缓存。
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/vePFS-Mindverse/share/code/conley/.openpi_cache}"
export HF_HOME="${HF_HOME:-/vePFS-Mindverse/share/huggingface}"

# 同时暴露 openpi 与 lance：本仓库 + extra-pydeps(lance) + gpu_rl 的 openpi/site-packages。
export PYTHONPATH="${CODE_ROOT}:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"

# --dry-run 透传给 driver（只读 lance + 组 batch，不连 server）。
DRY_RUN=""
for arg in "$@"; do
  [ "${arg}" = "--dry-run" ] && DRY_RUN="--dry-run"
done

# --- 前置校验 --------------------------------------------------------------- #
if [ ! -x "${CLIENT_PY}" ]; then
  echo "error: client python 不存在: ${CLIENT_PY}" >&2
  echo "       期望 gpu_rl host-venv（提供 openpi+jax+sentencepiece+requests）。" >&2
  exit 1
fi
if [ ! -f "${DRIVER}" ]; then
  echo "error: lance driver 不存在: ${DRIVER}" >&2
  exit 1
fi
if [ ! -d "${LANCE_DS}" ]; then
  echo "error: lance 数据集不存在: ${LANCE_DS}" >&2
  echo "       用 MINT_LANCE_DATASET=/path 指定，或先生成。" >&2
  exit 1
fi

echo "== pi0.5 lance 全链路冒烟 =="
echo "   base_url = http://localhost:${PORT}"
echo "   model    = ${MODEL}"
echo "   driver   = ${DRIVER}"
echo "   lance    = ${LANCE_DS}"
echo "   steps    = ${STEPS}  batch = ${BATCH}"
echo "   client   = ${CLIENT_PY}"
[ -n "${DRY_RUN}" ] && echo "   mode     = dry-run（不连 server）"

# --- 连 server 时先探健康 --------------------------------------------------- #
if [ -z "${DRY_RUN}" ]; then
  HEALTH=$(curl -s "http://localhost:${PORT}/api/v1/healthz" 2>/dev/null)
  if ! echo "${HEALTH}" | grep -q "ready"; then
    echo "error: server 未就绪 @ :${PORT}（healthz='${HEALTH}'）。" >&2
    echo "       先跑 step0(ray) -> step1(rsync) -> step3(server) -> step4(health)。" >&2
    exit 1
  fi
  echo "   health   = ${HEALTH}"
fi

# --- 跑 driver -------------------------------------------------------------- #
MINT_BASE_URL="http://localhost:${PORT}" \
TINKER_API_KEY=tml-dummy \
MINT_API_KEY=tml-dummy \
"${CLIENT_PY}" "${DRIVER}" \
  --model "${MODEL}" \
  --lance-dataset "${LANCE_DS}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH}" \
  --output-json "${OUT_JSON}" \
  ${DRY_RUN}
rc=$?

if [ "${rc}" -ne 0 ]; then
  echo "FAILED: lance driver 退出码 ${rc}" >&2
  exit "${rc}"
fi
[ -n "${DRY_RUN}" ] && { echo "OK: dry-run 数据路径通过。"; exit 0; }
echo "OK: pi0.5 lance 全链路通过，结果见 ${OUT_JSON}"
