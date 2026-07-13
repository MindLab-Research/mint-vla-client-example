#!/usr/bin/env bash
# PI05lance_local_norray.sh — Ray-free OpenPI (pi0.5) full-chain over real HTTP.
#
# Self-contained: boots a single-worker mint-server with NO Ray control plane
# (openpi runs in-process; see openpi_local_execution.py), waits for it to serve,
# then drives create_model -> train_step*N -> save_weights -> action -> act via
# the real tinker HTTP API using scripts/wip/openpi_vla_smoke_lance.py.
#
# Everything the server needs to come up Ray-free:
#   MINT_ALLOW_NO_RAY=1     init_ray failure degrades instead of crashing
#   MINT_USAGE_BACKEND=disabled   skip postgres usage store
#   MINT_SKIP_SUPERVISOR=1  no Ray model-actor supervisor
#   MINT_UVICORN_WORKERS=1  single worker (process-local engine/session/future)
#
# OOM fix: pi05 train_step recompiles an XLA graph per step (variable padded
# shapes); CUDA command buffers accumulate and exhaust VRAM around step ~17.
# XLA_FLAGS=--xla_gpu_enable_command_buffer= disables command buffers so graphs
# don't pile up. See results/logs/*server*.log RESOURCE_EXHAUSTED for the trail.
#
# Usage:
#   bash scripts/vla/PI05lance_local_norray.sh                 # 400 steps, batch 2
#   MINT_PI05_STEPS=50 MINT_PI05_BATCH=1 bash .../PI05lance_local_norray.sh
#   MINT_PI05_SKIP_SERVER=1 bash .../PI05lance_local_norray.sh  # reuse live server
set -uo pipefail

# --- knobs ------------------------------------------------------------------ #
PORT="${MINT_PORT:-30510}"
STEPS="${MINT_PI05_STEPS:-400}"
BATCH="${MINT_PI05_BATCH:-2}"
MODEL="${MINT_PI05_MODEL:-openpi/pi05-libero-low-mem-finetune}"
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint}"
GRB="${MINT_GPU_RL_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl}"
EXTRA_PYDEPS="${MINT_EXTRA_PYDEPS:-/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps}"
PY="${GRB}/host-venv/bin/python"
LANCE_DS="${MINT_LANCE_DATASET:-/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_full_lance.lance}"

LOG_DIR="${MINT_LOG_DIR:-/vePFS-Mindverse/user/intern/wenxi/results/logs}"
DATA_DIR="${MINT_DATA_DIR:-/vePFS-Mindverse/user/intern/wenxi/results/datas}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${LOG_DIR}/pi05_norray_server_${STAMP}.log"
DRIVER_LOG="${LOG_DIR}/pi05_norray_run_${STAMP}.log"
OUT_JSON="${DATA_DIR}/pi05_norray_run_${STAMP}.json"
mkdir -p "${LOG_DIR}" "${DATA_DIR}"

echo "== Ray-free pi0.5 full-chain over HTTP =="
echo "   code_root = ${CODE_ROOT}"
echo "   base_url  = http://localhost:${PORT}"
echo "   model     = ${MODEL}"
echo "   lance     = ${LANCE_DS}"
echo "   steps     = ${STEPS}  batch = ${BATCH}"
echo "   server_log= ${SERVER_LOG}"
echo "   driver_log= ${DRIVER_LOG}"
echo "   out_json  = ${OUT_JSON}"

# --- server env (Ray-free) -------------------------------------------------- #
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
# Disable XLA CUDA command buffers to stop per-step graph accumulation (OOM fix).
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
# Pin to idle GPUs by default — this box shares cards with other users, and the
# pi05 mesh spans all visible devices (fsdp). Landing on cards already ~75% full
# is what tipped training into OOM. Override MINT_CUDA_DEVICES to change/empty.
export CUDA_VISIBLE_DEVICES="${MINT_CUDA_DEVICES:-3,4,5,6}"
export PYTHONPATH="${CODE_ROOT}:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"

# --- preflight -------------------------------------------------------------- #
if [ ! -x "${PY}" ]; then echo "error: gpu_rl python missing: ${PY}" >&2; exit 1; fi
if [ ! -d "${LANCE_DS}" ]; then echo "error: lance dataset missing: ${LANCE_DS}" >&2; exit 1; fi

SERVER_PID=""
cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "-- stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

# --- start server (unless reusing a live one) ------------------------------- #
if [ "${MINT_PI05_SKIP_SERVER:-0}" != "1" ]; then
  # stop any prior norray launcher on this port (match our unique script name)
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
  # openpi endpoints work even while healthz reports degraded; probe create_model
  # reachability via a cheap GET on the app root instead.
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
echo "-- server responding (healthz http=${code}; 'unhealthy'/503 is expected in Ray-free degraded mode)"

# --- run the driver over real HTTP ------------------------------------------ #
echo "-- running driver: ${STEPS} steps, batch ${BATCH} -> ${DRIVER_LOG}"
MINT_BASE_URL="http://localhost:${PORT}" \
MINT_API_KEY=tml-dummy TINKER_API_KEY=tml-dummy \
JAX_PLATFORMS=cpu \
"${PY}" -u "${CODE_ROOT}/scripts/wip/openpi_vla_smoke_lance.py" \
  --base-url "http://localhost:${PORT}" \
  --lance-dataset "${LANCE_DS}" \
  --steps "${STEPS}" --batch-size "${BATCH}" \
  --output-json "${OUT_JSON}" \
  > "${DRIVER_LOG}" 2>&1
rc=$?

# --- summarize -------------------------------------------------------------- #
echo ""
echo "== driver exit=${rc} =="
DONE=$(grep -cE '"step": [0-9]+, "loss"' "${DRIVER_LOG}" 2>/dev/null || echo 0)
echo "steps logged: ${DONE}/${STEPS}"
echo "-- last 5 step lines --"
grep -E '"step": [0-9]+, "loss"' "${DRIVER_LOG}" | tail -5
if [ "${rc}" = "0" ] && [ -f "${OUT_JSON}" ]; then
  "${PY}" - "${OUT_JSON}" <<'PYEOF'
import json, sys, statistics
d = json.load(open(sys.argv[1]))
steps = d.get("steps", [])
vals = [(s.get("metrics") or {}).get("loss:mean") for s in steps]
vals = [v for v in vals if isinstance(v, (int, float))]
print(f"model_id: {d.get('model_id')}")
print(f"steps with loss: {len(vals)}/{len(steps)}")
if vals:
    print("loss min/max/mean: %.4f / %.4f / %.4f" % (min(vals), max(vals), statistics.mean(vals)))
    n = min(50, len(vals))
    print("mean first-%d: %.4f  last-%d: %.4f" % (n, statistics.mean(vals[:n]), n, statistics.mean(vals[-n:])))
sw = d.get("save_result") or {}
print("save uri:", sw.get("path"))
# driver writes the act payload under "action_result" (not "act")
act = (d.get("action_result") or d.get("act") or {}).get("actions") or {}
print("act shape:", act.get("shape"))
PYEOF
  echo "OK: full chain passed. json=${OUT_JSON}"
else
  echo "FAILED: see ${DRIVER_LOG} and ${SERVER_LOG}" >&2
  echo "-- driver tail --" >&2; tail -25 "${DRIVER_LOG}" >&2
  exit "${rc}"
fi

