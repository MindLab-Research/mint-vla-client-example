#!/usr/bin/env bash
# 08:00 post-deadline: wait for training completion, Mode 4 eval 5 rows × 2 arms,
# generate + validate 10/10 videos, write delivery.ready for local rsync+send.
# NO lark-cli on remote; sending happens on local machine.
set -Eeuo pipefail

CLEAN_ROOT=/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_clean_lr5e5_80k_20260726
AUG_ROOT=/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_stateaug005_lr5e5_80k_20260726
CLIENT=/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-example
MINT_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16
OPENPI_ROOT=/vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16
DATA=/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance
GESTURE=$CLIENT/config/datasets/new_all_generated_mano.index.json
NORM=/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726
ROWS=924,960,914,943,939
NORM_ROWS=$(python3 -c "print(','.join(str(i) for i in range(810,995)))")
DELIVERY_ROOT=/vePFS-Mindverse/user/intern/wenxi/results/training/mode4_delivery_v1_presence_20260727
READY_FILE=$DELIVERY_ROOT/delivery.ready
FAILED_FILE=$DELIVERY_ROOT/delivery.failed

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
fail(){ log "FAILED: $*"; echo "$(date -Is) $*" > "$FAILED_FILE"; exit 1; }

# ---------------------------------------------------------------------------
# Wait for training completion
# ---------------------------------------------------------------------------
wait_arm_done() {
  local root=$1 arm=$2
  log "waiting for $arm..."
  while true; do
    if [[ -f "$root/exit_code" ]]; then
      local rc
      rc=$(cat "$root/exit_code")
      if [[ "$rc" != "0" ]]; then
        fail "$arm exit_code=$rc"
      fi
      if [[ -f "$root/train/result.json" ]]; then
        local stop_reason completed_step
        stop_reason=$(python3 -c "import json; d=json.load(open('$root/train/result.json')); print(d.get('stop_reason',''))" 2>/dev/null || echo "")
        completed_step=$(python3 -c "import json; d=json.load(open('$root/train/result.json')); print(d.get('completed_step',0))" 2>/dev/null || echo "0")
        log "$arm: exit=0 stop_reason=$stop_reason completed_step=$completed_step"
        # Fail-closed: only accept deadline stop with steps, or normal 80K completion.
        if [[ "$stop_reason" == "deadline" && "$completed_step" -gt 0 ]]; then
          break
        elif [[ -z "$stop_reason" || "$stop_reason" == "None" ]] && [[ "$completed_step" -eq 80000 ]]; then
          log "$arm: completed normally (80K)"
          break
        else
          fail "$arm: unexpected stop_reason=$stop_reason completed_step=$completed_step"
        fi
      fi
    fi
    sleep 30
  done
  # Wait for server exit
  local server_pid=""
  if [[ -f "$root/server/pid" ]]; then
    server_pid=$(cat "$root/server/pid")
    while kill -0 "$server_pid" 2>/dev/null; do sleep 5; done
  fi
  log "$arm server exited"
}

# ---------------------------------------------------------------------------
# Read checkpoint model path from result.json
# ---------------------------------------------------------------------------
read_model_path() {
  python3 -c "
import json
d = json.load(open('$1/train/result.json'))
path = d.get('save_result',{}).get('path','')
if not path: raise SystemExit('no save_result.path')
print(path)
"
}

# ---------------------------------------------------------------------------
# Run Mode 4 eval for one arm (independent port, per-arm dirs, cleanup trap)
# ---------------------------------------------------------------------------
run_eval() {
  local arm=$1 arm_root=$2 model_path=$3 port=$4 gpu=$5 out_dir=$6
  mkdir -p "$out_dir"
  log "starting $arm eval on port $port, GPU $gpu"
  local server_pid=""

  cleanup_eval() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
      kill -INT "$server_pid" 2>/dev/null || true
      for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
      kill -TERM "$server_pid" 2>/dev/null || true
      server_pid=""
      log "$arm eval server cleaned up"
    fi
  }
  trap cleanup_eval RETURN EXIT

  (
    export CUDA_VISIBLE_DEVICES=$gpu MINT_HOST=127.0.0.1 MINT_PORT=$port
    export MINT_SUPPORTED_MODELS=openpi/pi05-action-lora-r16-finetune MINT_ALLOW_NO_RAY=1 MINT_SKIP_SUPERVISOR=1 MINT_USAGE_BACKEND=disabled MINT_UVICORN_WORKERS=1
    export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1 MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets
    export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR="$arm_root/server/openpi_checkpoint_base" MINT_RUNTIME_CHECKPOINT_DIR="$arm_root/server/runtime_checkpoints" MINT_CHECKPOINT_DIR="$arm_root/server/runtime_checkpoints/persistent_cache" MINT_PERSISTENT_CHECKPOINT_DIR="$arm_root/server/runtime_checkpoints/persistent_cache"
    export MINT_TMP_ROOT="$arm_root/server/tmp" MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT_BASE="$arm_root/server/action_state"
    export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface XLA_FLAGS=--xla_gpu_enable_command_buffer=
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    export PYTHONPATH=$MINT_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/site-packages PYTHONNOUSERSITE=1
    exec /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python -u -c "import uvicorn; from mint_server.app import app; uvicorn.run(app,host='127.0.0.1',port=$port,workers=1,log_level='info')"
  ) > "$out_dir/server.log" 2>&1 &
  server_pid=$!
  echo "$server_pid" > "$out_dir/server.pid"
  code=000; for _ in $(seq 1 180); do code=$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/openapi.json" 2>/dev/null||true); [[ "$code" == 200 ]] && break; sleep 1; done
  [[ "$code" == 200 ]] || fail "$arm server failed to start"
  log "$arm server ready pid=$server_pid"

  cd "$CLIENT"
  export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export LD_LIBRARY_PATH=/vePFS-Mindverse/share/zhouch-caches/.cache/openvla_full_a800/glvnd_shim:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=$CLIENT/scripts/eval:$CLIENT:$CLIENT/scripts/train:/vePFS-Mindverse/user/intern/wenxi/pi-finetune/case/01_export_video:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$MINT_ROOT:/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps:/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/site-packages
  /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python scripts/eval/infer_mano_mode4.py \
    --base-url http://127.0.0.1:$port --model-path "$model_path" --owner-id 000000000000000000000001 \
    --lance-dataset "$DATA" --row-indices "$ROWS" \
    --language-conditioning gesture --gesture-index "$GESTURE" \
    --normalization-row-indices "$NORM_ROWS" \
    --extended-state --norm-stats-dir "$NORM" \
    --output-dir "$out_dir/artifacts" --chunk-stride 1 --act-mode batch --act-batch-size 4 \
    > "$out_dir/eval.log" 2>&1
  local rc=$?
  log "$arm eval rc=$rc"
  return $rc
}

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
log "=== deliver_mode4_eval started ==="
rm -f "$READY_FILE" "$FAILED_FILE"

# ① Wait for both arms
wait_arm_done "$CLEAN_ROOT" "clean"
wait_arm_done "$AUG_ROOT" "stateaug005"

# ② Read model paths
CLEAN_MODEL_PATH=$(read_model_path "$CLEAN_ROOT")
AUG_MODEL_PATH=$(read_model_path "$AUG_ROOT")
log "clean: $CLEAN_MODEL_PATH"
log "aug:   $AUG_MODEL_PATH"

# ③ Wait for all GPUs free
log "waiting for all GPUs free..."
while true; do
  local_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100{c++} END{print c+0}')
  [[ "$local_used" == "0" ]] && break
  sleep 10
done
log "all GPUs free"

# ④ Run both evals in parallel (clean artifacts first to avoid stale false-positive)
rm -rf "$DELIVERY_ROOT/clean/artifacts" "$DELIVERY_ROOT/stateaug005/artifacts"
mkdir -p "$DELIVERY_ROOT/clean/artifacts" "$DELIVERY_ROOT/stateaug005/artifacts"
run_eval "clean" "$CLEAN_ROOT" "$CLEAN_MODEL_PATH" 30536 "0,1,2,3" "$DELIVERY_ROOT/clean" &
CLEAN_PID=$!
run_eval "stateaug005" "$AUG_ROOT" "$AUG_MODEL_PATH" 30537 "4,5,6,7" "$DELIVERY_ROOT/stateaug005" &
AUG_PID=$!
wait $CLEAN_PID $AUG_PID || true

# ⑤ Validate 10/10 videos
log "validating 10/10 videos..."
MISSING=0
VIDEO_PATHS=()
for arm in clean stateaug005; do
  for row in $(echo "$ROWS" | tr ',' ' '); do
    v="$DELIVERY_ROOT/$arm/artifacts/row${row}/mode4/mode4_physics_vs_dataset_head.mp4"
    if [[ -s "$v" && $(stat -c%s "$v") -gt 1000 ]]; then
      VIDEO_PATHS+=("$v")
    else
      log "MISSING: $arm row$row"
      MISSING=$((MISSING+1))
    fi
  done
done

if [[ $MISSING -gt 0 ]]; then
  fail "$MISSING videos missing/empty (need 10/10)"
fi
log "all 10 videos validated"

# ⑥ Build manifest
python3 - "$DELIVERY_ROOT" "$ROWS" "$CLEAN_ROOT" "$AUG_ROOT" "$CLIENT" <<'PYMANIFEST'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [int(x) for x in sys.argv[2].split(",")]
clean_root = Path(sys.argv[3])
aug_root = Path(sys.argv[4])
sys.path.insert(0, sys.argv[5])
from scripts.mano_state_contract import (
    CONTACT_RULE,
    CONTACT_SEMANTICS,
    EXPECTED_NORM_SHA256,
    STATE_CONTRACT_ID,
)

def read_steps(r):
    d = json.loads((r / "train/result.json").read_text())
    return {"stop_reason": d.get("stop_reason"), "completed_step": d.get("completed_step")}

manifest = {
    "deadline": "2026-07-27T08:00:00+08:00",
    "state_contract": STATE_CONTRACT_ID,
    "contact_semantics": CONTACT_SEMANTICS,
    "contact_rule": CONTACT_RULE,
    "norm_sha_expected": EXPECTED_NORM_SHA256,
    "arms": {},
    "clean_training": read_steps(clean_root),
    "aug_training": read_steps(aug_root),
}

for arm in ["clean", "stateaug005"]:
    entries = []
    for row in rows:
        mode4_dir = root / arm / "artifacts" / f"row{row}" / "mode4"
        head_video = mode4_dir / "mode4_physics_vs_dataset_head.mp4"
        result_json = mode4_dir / "result.json"
        entry = {"row": row, "arm": arm, "head_video": str(head_video)}
        if result_json.exists():
            r = json.loads(result_json.read_text())
            entry["result_json"] = str(result_json)
            entry["object_height"] = float(r.get("physics", {}).get("max_object_height", 0))
            entry["hand_object_contacts"] = r.get("physics", {}).get("contacts", {}).get("hand_object", 0)
        entries.append(entry)
    manifest["arms"][arm] = entries

out = root / "delivery_manifest.json"
out.write_text(json.dumps(manifest, indent=2))
print(f"manifest: {out}")
PYMANIFEST

# ⑦ Write delivery.ready
python3 - "$READY_FILE" "$DELIVERY_ROOT" <<'PYREADY'
import json, sys
from pathlib import Path

ready_file = Path(sys.argv[1])
root = Path(sys.argv[2])
manifest = json.loads((root / "delivery_manifest.json").read_text())

ready = {
    "status": "ready",
    "timestamp": __import__("datetime").datetime.now().isoformat(),
    "clean_completed_step": manifest["clean_training"]["completed_step"],
    "aug_completed_step": manifest["aug_training"]["completed_step"],
    "videos": [],
}
for arm, entries in manifest["arms"].items():
    for e in entries:
        ready["videos"].append({
            "arm": arm,
            "row": e["row"],
            "path": e["head_video"],
        })

ready_file.write_text(json.dumps(ready, indent=2))
print(f"delivery.ready written: {ready_file}")
print(f"  clean_completed_step: {ready['clean_completed_step']}")
print(f"  aug_completed_step: {ready['aug_completed_step']}")
print(f"  videos: {len(ready['videos'])}")
PYREADY

log "=== delivery complete: delivery.ready written ==="
