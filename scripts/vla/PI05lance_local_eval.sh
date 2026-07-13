#!/usr/bin/env bash
# PI05lance_local_eval.sh — quantitative eval of a Ray-free-trained pi0.5 sampler.
#
# Companion to PI05lance_local_norray.sh (training). Boots the SAME Ray-free
# mint-server env, then runs scripts/wip/openpi_vla_eval_lance.py: load the
# trained sampler weights, /act on eval samples, compare against the Lance
# ground-truth actions (same normalized space), report MSE/L1 and the ratio
# vs the zero-prediction baseline. ratio < 1 => the model learned something.
#
# Correctness note: eval MUST use the SAME lance dataset as training so
# _compute_norm_stats produces identical normalization — otherwise pred and gt
# live in different spaces and MSE is meaningless. Training had no train/val
# split (random sampling over the full set), so reusing the full lance is right.
#
# Usage:
#   bash scripts/vla/PI05lance_local_eval.sh                    # auto-picks latest train json
#   MINT_EVAL_INDICES=0,1,2,5,10 bash .../PI05lance_local_eval.sh
#   MINT_TRAIN_JSON=/path/run.json bash .../PI05lance_local_eval.sh
#   MINT_SAMPLER_PATH=mint://... MINT_SAMPLER_OWNER=... bash .../PI05lance_local_eval.sh
#   MINT_EVAL_SKIP_SERVER=1 bash .../PI05lance_local_eval.sh    # reuse a live server
set -uo pipefail

# --- knobs ------------------------------------------------------------------ #
PORT="${MINT_PORT:-30510}"
INDICES="${MINT_EVAL_INDICES:-0,1,2,5,10,20,50,100}"
ACTION_HORIZON="${MINT_EVAL_ACTION_HORIZON:-10}"
MODEL="${MINT_PI05_MODEL:-openpi/pi05-libero-low-mem-finetune}"
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint}"
GRB="${MINT_GPU_RL_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl}"
EXTRA_PYDEPS="${MINT_EXTRA_PYDEPS:-/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps}"
PY="${GRB}/host-venv/bin/python"
LANCE_DS="${MINT_LANCE_DATASET:-/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance}"

LOG_DIR="${MINT_LOG_DIR:-/vePFS-Mindverse/user/intern/wenxi/results/logs}"
DATA_DIR="${MINT_DATA_DIR:-/vePFS-Mindverse/user/intern/wenxi/results/datas}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${LOG_DIR}/pi05_eval_server_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/pi05_eval_run_${STAMP}.log"
OUT_JSON="${DATA_DIR}/pi05_eval_run_${STAMP}.json"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

# --- resolve trained sampler path ------------------------------------------- #
# Prefer explicit MINT_SAMPLER_PATH; else read save_result from a training json
# (MINT_TRAIN_JSON, or the newest pi05_norray_run_*.json).
SAMPLER_PATH="${MINT_SAMPLER_PATH:-}"
SAMPLER_OWNER="${MINT_SAMPLER_OWNER:-}"
TRAIN_JSON="${MINT_TRAIN_JSON:-}"
if [ -z "${SAMPLER_PATH}" ]; then
  if [ -z "${TRAIN_JSON}" ]; then
    TRAIN_JSON="$(ls -t "${DATA_DIR}"/pi05_norray_run_*.json 2>/dev/null | head -1)"
  fi
  if [ -z "${TRAIN_JSON}" ] || [ ! -f "${TRAIN_JSON}" ]; then
    echo "error: no MINT_SAMPLER_PATH and no training json found (looked for ${DATA_DIR}/pi05_norray_run_*.json)." >&2
    echo "       run training first, or pass MINT_SAMPLER_PATH=mint://... ." >&2
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
  echo "error: could not resolve a sampler path (train json had no save_result.path?)." >&2
  exit 1
fi

echo "== Ray-free pi0.5 sampler eval over HTTP =="
echo "   base_url    = http://localhost:${PORT}"
echo "   model       = ${MODEL}"
echo "   lance       = ${LANCE_DS}"
echo "   sampler     = ${SAMPLER_PATH}"
echo "   owner       = ${SAMPLER_OWNER}"
echo "   train_json  = ${TRAIN_JSON:-<explicit sampler path>}"
echo "   indices     = ${INDICES}"
echo "   server_log  = ${SERVER_LOG}"
echo "   eval_log    = ${EVAL_LOG}"
echo "   out_json    = ${OUT_JSON}"

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

# --- preflight -------------------------------------------------------------- #
if [ ! -x "${PY}" ]; then echo "error: gpu_rl python missing: ${PY}" >&2; exit 1; fi
if [ ! -d "${LANCE_DS}" ]; then echo "error: lance dataset missing: ${LANCE_DS}" >&2; exit 1; fi

SERVER_PID=""
cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "-- stopping server pid=${SERVER_PID}"; kill "${SERVER_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

# --- start server (unless reusing a live one) ------------------------------- #
if [ "${MINT_EVAL_SKIP_SERVER:-0}" != "1" ]; then
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
    echo "error: server process died during startup; see ${SERVER_LOG}" >&2
    tail -20 "${SERVER_LOG}" >&2; exit 1
  fi
  sleep 2
done
if [ "${READY}" != "1" ]; then
  echo "error: server not ready after 180s; see ${SERVER_LOG}" >&2; exit 1
fi
echo "-- server responding (healthz http=${code}; 'unhealthy'/503 expected in Ray-free degraded mode)"

# --- run the eval driver ---------------------------------------------------- #
echo "-- running eval on indices ${INDICES} -> ${EVAL_LOG}"
MINT_BASE_URL="http://localhost:${PORT}" \
MINT_API_KEY=tml-dummy TINKER_API_KEY=tml-dummy \
JAX_PLATFORMS=cpu \
"${PY}" -u "${CODE_ROOT}/scripts/wip/openpi_vla_eval_lance.py" \
  --base-url "http://localhost:${PORT}" \
  --api-key tml-dummy \
  --model-path "${SAMPLER_PATH}" \
  --owner-id "${SAMPLER_OWNER}" \
  --lance-dataset "${LANCE_DS}" \
  --action-horizon "${ACTION_HORIZON}" \
  --indices "${INDICES}" \
  --output-json "${OUT_JSON}" \
  > "${EVAL_LOG}" 2>&1
rc=$?

# --- summarize -------------------------------------------------------------- #
echo ""
echo "== eval exit=${rc} =="
if [ "${rc}" = "0" ] && [ -f "${OUT_JSON}" ]; then
  "${PY}" - "${OUT_JSON}" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
agg = d.get("aggregate") or {}
rows = [r for r in d.get("per_sample", []) if "mse" in r]
print("sampler:", d.get("model_path"))
print("eval samples:", agg.get("num_samples"))
print("overall_mse:      %s" % agg.get("overall_mse"))
print("overall_l1:       %s" % agg.get("overall_l1"))
print("baseline_mse_zero:%s" % agg.get("baseline_mse_zero"))
ratio = agg.get("mse_vs_baseline_ratio")
print("mse_vs_baseline_ratio: %s" % ratio)
nan = any(r.get("pred_has_nan_inf") for r in rows)
print("any pred NaN/Inf:", nan)
verdict = "TRAINED MODEL BEATS ZERO BASELINE" if (isinstance(ratio,(int,float)) and ratio < 1 and not nan) \
    else ("INCONCLUSIVE/REGRESSION (ratio>=1 or NaN)" if ratio is not None else "NO RATIO")
print("VERDICT:", verdict)
PYEOF
  echo "OK: eval complete. json=${OUT_JSON}"
else
  echo "FAILED: see ${EVAL_LOG} and ${SERVER_LOG}" >&2
  echo "-- eval tail --" >&2; tail -25 "${EVAL_LOG}" >&2
  exit "${rc}"
fi

