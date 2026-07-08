#!/usr/bin/env bash
# PI05lance_infer.sh — 用已训练落盘的 sampler 权重做纯推理(不重训),仿 PI05lance.sh 环境。
#
# 与 PI05lance.sh 的差异:
#   - driver 换成 scripts/wip/openpi_vla_infer_lance.py:只 建 action_session + act,
#     不 create_model、不 train_step。
#   - 默认从上次训练输出 JSON(/tmp/pi05_lance_smoke.json)自动取 save_result.path
#     与 owner_id;也可用 MINT_SAMPLER_PATH / --model-path 显式指定。
#
# 用法:
#   bash PI05lance_infer.sh                       # 自动读上次训练的 sampler,推理 index 0
#   MINT_INFER_INDICES=0,1,2 bash PI05lance_infer.sh
#   MINT_SAMPLER_PATH=mint://... bash PI05lance_infer.sh
set -uo pipefail

PORT="${MINT_PORT:-30496}"
PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime
RUNTIME_ROOT="${PFS_RUNTIME_ENV_ROOT}"
GRB="${RUNTIME_ROOT}/gpu_rl"
CLIENT_PY="${MINT_CLIENT_PYTHON:-${GRB}/host-venv/bin/python}"
EXTRA_PYDEPS="${MINT_EXTRA_PYDEPS:-/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps}"

CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint}"
DRIVER="${CODE_ROOT}/scripts/wip/openpi_vla_infer_lance.py"

MODEL="${MINT_PI05_MODEL:-openpi/pi05-libero-low-mem-finetune}"
LANCE_DS="${MINT_LANCE_DATASET:-/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance}"
TRAIN_JSON="${MINT_PI05_OUTPUT_JSON:-/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_lance_smoke.json}"
OUT_JSON="${MINT_PI05_INFER_JSON:-/vePFS-Mindverse/user/intern/wenxi/results/datas/pi05_infer.json}"
INDICES="${MINT_INFER_INDICES:-0}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/vePFS-Mindverse/share/code/conley/.openpi_cache}"
export HF_HOME="${HF_HOME:-/vePFS-Mindverse/share/huggingface}"
export PYTHONPATH="${CODE_ROOT}:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"

# --- 解析 sampler 权重路径 -------------------------------------------------- #
SAMPLER_PATH="${MINT_SAMPLER_PATH:-}"
SAMPLER_OWNER="${MINT_SAMPLER_OWNER:-}"
if [ -z "${SAMPLER_PATH}" ]; then
  if [ ! -f "${TRAIN_JSON}" ]; then
    echo "error: 未指定 MINT_SAMPLER_PATH,且训练输出 ${TRAIN_JSON} 不存在。" >&2
    echo "       先跑 bash PI05lance.sh 训练,或显式 MINT_SAMPLER_PATH=mint://... 。" >&2
    exit 1
  fi
  read -r SAMPLER_PATH SAMPLER_OWNER < <("${CLIENT_PY}" - "${TRAIN_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sr = d.get("save_result") or {}
print(sr.get("path", ""), sr.get("owner_id", "anonymous"))
PY
)
  if [ -z "${SAMPLER_PATH}" ] || [ "${SAMPLER_PATH}" = "None" ]; then
    echo "error: ${TRAIN_JSON} 里没有 save_result.path(训练可能没走到 save 阶段)。" >&2
    exit 1
  fi
fi
SAMPLER_OWNER="${SAMPLER_OWNER:-anonymous}"

echo "== pi0.5 纯推理(复用训练权重) =="
echo "   base_url    = http://localhost:${PORT}"
echo "   model_path  = ${SAMPLER_PATH}"
echo "   owner_id    = ${SAMPLER_OWNER}"
echo "   lance       = ${LANCE_DS}"
echo "   indices     = ${INDICES}"
echo "   client      = ${CLIENT_PY}"

# --- 前置校验 --------------------------------------------------------------- #
[ -x "${CLIENT_PY}" ] || { echo "error: client python 不存在: ${CLIENT_PY}" >&2; exit 1; }
[ -f "${DRIVER}" ]    || { echo "error: infer driver 不存在: ${DRIVER}" >&2; exit 1; }
[ -d "${LANCE_DS}" ]  || { echo "error: lance 数据集不存在: ${LANCE_DS}" >&2; exit 1; }

HEALTH=$(curl -s "http://localhost:${PORT}/api/v1/healthz" 2>/dev/null)
if ! echo "${HEALTH}" | grep -q "ready"; then
  echo "error: server 未就绪 @ :${PORT}（healthz='${HEALTH}'）。" >&2
  exit 1
fi
echo "   health      = ${HEALTH}"

# --- 跑 driver -------------------------------------------------------------- #
MINT_BASE_URL="http://localhost:${PORT}" \
MINT_API_KEY=tml-dummy \
"${CLIENT_PY}" "${DRIVER}" \
  --model "${MODEL}" \
  --model-path "${SAMPLER_PATH}" \
  --owner-id "${SAMPLER_OWNER}" \
  --lance-dataset "${LANCE_DS}" \
  --indices "${INDICES}" \
  --output-json "${OUT_JSON}"
rc=$?

[ "${rc}" -ne 0 ] && { echo "FAILED: infer driver 退出码 ${rc}" >&2; exit "${rc}"; }
echo "OK: pi0.5 推理完成,结果见 ${OUT_JSON}"
