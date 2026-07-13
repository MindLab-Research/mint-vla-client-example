#!/usr/bin/env bash
# PI05lance_local_merge_infer.sh — Ray-free pi0.5 inference writeback to Lance.
#
# Companion to PI05lance_local_norray.sh (training) / PI05lance_local_eval.sh.
# Boots the SAME Ray-free mint-server env, then runs
# scripts/wip/openpi_vla_merge_infer_lance.py: load the no-ray-trained sampler,
# /act over every (episode, frame) of the lance dataset, and write a NEW lance
# that keeps all original columns + appends parallel per-frame prediction columns
# (pred_actions / pred_actions_physical / pred_action_mse / pred_meta).
# Output name carries "noray" to distinguish from Ray-era writebacks.
#
# Correctness: --norm-stats MUST be exported (openpi_export_norm_stats.py) from
# the SAME lance the model trained on — training computes stats on the fly and
# never persists them; unnormalization (pred_actions_physical) and normalized-
# space pred_action_mse are only meaningful with the matching stats.
#
# Usage:
#   bash scripts/vla/PI05lance_local_merge_infer.sh                   # auto-picks latest train json + full lance
#   MINT_SAMPLER_PATH=mint://... bash .../PI05lance_local_merge_infer.sh
#   MINT_MERGE_SKIP_SERVER=1 bash .../PI05lance_local_merge_infer.sh  # reuse a live server
set -uo pipefail

# --- knobs ------------------------------------------------------------------ #
PORT="${MINT_PORT:-30510}"
ACTION_HORIZON="${MINT_MERGE_ACTION_HORIZON:-10}"
MODEL="${MINT_PI05_MODEL:-openpi/pi05-libero-low-mem-finetune}"
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint}"
GRB="${MINT_GPU_RL_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl}"
EXTRA_PYDEPS="${MINT_EXTRA_PYDEPS:-/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps}"
PY="${GRB}/host-venv/bin/python"
LANCE_DS="${MINT_LANCE_DATASET:-/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance}"

LOG_DIR="${MINT_LOG_DIR:-/vePFS-Mindverse/user/intern/wenxi/results/logs}"
DATA_DIR="${MINT_DATA_DIR:-/vePFS-Mindverse/user/intern/wenxi/results/datas}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${LOG_DIR}/pi05_merge_noray_server_${STAMP}.log"
RUN_LOG="${LOG_DIR}/pi05_merge_noray_run_${STAMP}.log"
OUT_LANCE="${MINT_OUTPUT_LANCE:-${DATA_DIR}/pi05_replay_merged_noray_${STAMP}.lance}"
NORM_STATS="${MINT_NORM_STATS:-${DATA_DIR}/pi05_norm_stats_full_noray.json}"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

# --- resolve trained sampler path ------------------------------------------- #
SAMPLER_PATH="${MINT_SAMPLER_PATH:-}"
SAMPLER_OWNER="${MINT_SAMPLER_OWNER:-}"
TRAIN_JSON="${MINT_TRAIN_JSON:-}"
if [ -z "${SAMPLER_PATH}" ]; then
  if [ -z "${TRAIN_JSON}" ]; then
    TRAIN_JSON="$(ls -t "${DATA_DIR}"/pi05_norray_run_*.json 2>/dev/null | head -1)"
  fi
  if [ -z "${TRAIN_JSON}" ] || [ ! -f "${TRAIN_JSON}" ]; then
    echo "error: no MINT_SAMPLER_PATH and no training json (looked for ${DATA_DIR}/pi05_norray_run_*.json)." >&2
    exit 1
  fi
  read -r SAMPLER_PATH SAMPLER_OWNER < <("${PY}" - "${TRAIN_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sr = d.get("save_result") or {}
print(sr.get("path", ""), sr.get("owner_id", "anonymous"))
PY
)
fi
[ -z "${SAMPLER_OWNER}" ] && SAMPLER_OWNER="anonymous"
if [ -z "${SAMPLER_PATH}" ]; then
  echo "error: could not resolve a sampler path." >&2; exit 1
fi

echo "== Ray-free pi0.5 inference writeback (noray) =="
echo "   base_url    = http://localhost:${PORT}"
echo "   model       = ${MODEL}"
echo "   lance_in    = ${LANCE_DS}"
echo "   sampler     = ${SAMPLER_PATH}"
echo "   owner       = ${SAMPLER_OWNER}"
echo "   norm_stats  = ${NORM_STATS}"
echo "   out_lance   = ${OUT_LANCE}"
echo "   server_log  = ${SERVER_LOG}"
echo "   run_log     = ${RUN_LOG}"

# --- preflight -------------------------------------------------------------- #
if [ ! -x "${PY}" ]; then echo "error: gpu_rl python missing: ${PY}" >&2; exit 1; fi
if [ ! -d "${LANCE_DS}" ]; then echo "error: lance dataset missing: ${LANCE_DS}" >&2; exit 1; fi
if [ ! -f "${NORM_STATS}" ]; then
  echo "error: norm_stats json missing: ${NORM_STATS}" >&2
  echo "       export it first from the SAME lance the model trained on:" >&2
  echo "       ${PY} scripts/wip/openpi_export_norm_stats.py --lance-dataset ${LANCE_DS} --output ${NORM_STATS}" >&2
  exit 1
fi

# --- server env (Ray-free) — identical to PI05lance_local_norray.sh --------- #
export MINT_CODE_ROOT="${CODE_ROOT}"
export MINT_PORT="${PORT}" MINT_HOST=0.0.0.0
export MINT_UVICORN_WORKERS=1 MINT_SKIP_SUPERVISOR=1 MINT_ALLOW_NO_RAY=1 MINT_USAGE_BACKEND=disabled
export MINT_RAY_NAMESPACE=mint_wenxi_local
export MINT_SUPPORTED_MODELS="${MODEL}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/vePFS-Mindverse/share/models/openpi}"
export HF_HOME="${HF_HOME:-/vePFS-Mindverse/share/huggingface}"
export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR="${MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR:-/vePFS-Mindverse/share/mint/dev/data/wenxi/openpi-pi05-checkpoints}"
export MINT_OPENPI_PI05_ASSETS_BASE_DIR="${MINT_OPENPI_PI05_ASSETS_BASE_DIR:-/vePFS-Mindverse/share/code/conley/openpi/assets}"
export MINT_OPENPI_PI05_WEIGHTS_PATH="${MINT_OPENPI_PI05_WEIGHTS_PATH:-/vePFS-Mindverse/share/models/openpi/pi05_base/params}"
export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1
export MINT_RUNTIME_CHECKPOINT_DIR="${MINT_RUNTIME_CHECKPOINT_DIR:-/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints}"
export MINT_TMP_ROOT="${MINT_TMP_ROOT:-/vePFS-Mindverse/share/mint/dev/tmp}" TMPDIR="${TMPDIR:-/tmp/mda/t}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export CUDA_VISIBLE_DEVICES="${MINT_CUDA_DEVICES:-3,4,5,6}"
export PYTHONPATH="${CODE_ROOT}:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"

SERVER_PID=""
cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "-- stopping server pid=${SERVER_PID}"; kill "${SERVER_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

# --- start server (unless reusing a live one) ------------------------------- #
if [ "${MINT_MERGE_SKIP_SERVER:-0}" != "1" ]; then
  for pid in $(pgrep -f "_run_local_openpi_server.py" 2>/dev/null); do
    echo "-- killing prior server pid=${pid}"; kill "${pid}" 2>/dev/null
  done
  sleep 2
  echo "-- launching Ray-free server -> ${SERVER_LOG}"
  nohup "${PY}" -u "${CODE_ROOT}/scripts/wip/_run_local_openpi_server.py" > "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "-- server pid=${SERVER_PID}"
fi

# --- wait for the server to accept requests --------------------------------- #
echo "-- waiting for server on :${PORT} (up to 180s) ..."
READY=0
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://localhost:${PORT}/api/v1/healthz" 2>/dev/null)
  if [ "${code}" = "200" ] || [ "${code}" = "503" ]; then READY=1; break; fi
  if [ -n "${SERVER_PID}" ] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "error: server died during startup; see ${SERVER_LOG}" >&2
    tail -20 "${SERVER_LOG}" >&2; exit 1
  fi
  sleep 2
done
if [ "${READY}" != "1" ]; then
  echo "error: server not ready after 180s; see ${SERVER_LOG}" >&2; exit 1
fi
echo "-- server responding (healthz http=${code}; 'unhealthy'/503 expected in Ray-free degraded mode)"

# --- run the merge-infer driver --------------------------------------------- #
echo "-- inferring over all frames + writing merged lance -> ${RUN_LOG}"
MINT_BASE_URL="http://localhost:${PORT}" \
MINT_API_KEY=tml-dummy TINKER_API_KEY=tml-dummy \
JAX_PLATFORMS=cpu \
"${PY}" -u "${CODE_ROOT}/scripts/wip/openpi_vla_merge_infer_lance.py" \
  --base-url "http://localhost:${PORT}" \
  --api-key tml-dummy \
  --model-path "${SAMPLER_PATH}" \
  --owner-id "${SAMPLER_OWNER}" \
  --norm-stats "${NORM_STATS}" \
  --lance-dataset "${LANCE_DS}" \
  --action-horizon "${ACTION_HORIZON}" \
  --output-lance "${OUT_LANCE}" \
  > "${RUN_LOG}" 2>&1
rc=$?

# --- summarize -------------------------------------------------------------- #
echo ""
echo "== merge-infer exit=${rc} =="
if [ "${rc}" = "0" ] && [ -d "${OUT_LANCE}" ]; then
  FRAMES=$(grep -cE "ep=[0-9]+ frame=[0-9]+" "${RUN_LOG}" 2>/dev/null || echo 0)
  echo "frames inferred: ${FRAMES}"
  grep -E "^OK: " "${RUN_LOG}" | tail -1
  "${PY}" - "${RUN_LOG}" <<'PYEOF'
import sys, re
mses = [float(m.group(1)) for m in
        (re.search(r"mse=([0-9.]+)", ln) for ln in open(sys.argv[1])) if m]
if mses:
    import statistics
    print("per-frame mse (normalized): min/mean/max = %.5f / %.5f / %.5f  n=%d"
          % (min(mses), statistics.mean(mses), max(mses), len(mses)))
PYEOF
  echo "OK: merged lance written -> ${OUT_LANCE}"
else
  echo "FAILED: see ${RUN_LOG} and ${SERVER_LOG}" >&2
  echo "-- run tail --" >&2; tail -25 "${RUN_LOG}" >&2
  exit "${rc}"
fi

