#!/usr/bin/env bash
# Latest clean/StateAug B-exact 32D checkpoints under historical kinematic Mode3.
set -Eeuo pipefail

CLEAN_ROOT=/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_clean_lr5e5_80k_20260726
AUG_ROOT=/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_stateaug005_lr5e5_80k_20260726
CLIENT=/vePFS-Mindverse/user/intern/wenxi/mint-vla-client-mode3-latest
MINT_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint-action-lora-r16
OPENPI_ROOT=/vePFS-Mindverse/user/intern/wenxi/openpi-action-lora-r16
DATA=/vePFS-Mindverse/user/intern/wenxi/results/datas/new_all_generated_mano_with_images.lance
MANIFEST=${DATA}.contact_windows.json
GESTURE=$CLIENT/config/datasets/new_all_generated_mano.index.json
NORM=/vePFS-Mindverse/user/intern/wenxi/results/training/gesture03_32d_extended_norm_v1_20260726
OUTPUT_ROOT=${MODE3_OUTPUT_ROOT:-/vePFS-Mindverse/user/intern/wenxi/results/training/mode3_latest_b_alora_clean_aug_20260727}
QUERY_STRIDE=${MODE3_QUERY_STRIDE:-10}
RUN_SMOKE=${MODE3_RUN_SMOKE:-1}
ROWS=924,960,914,943,939
NORM_ROWS=$(python3 -c "print(','.join(str(i) for i in range(810,995)))")
OWNER_ID=000000000000000000000001
PYTHON=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/host-venv/bin/python
SITE=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl/site-packages
EXTRA=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps
PI_FINETUNE=/vePFS-Mindverse/user/intern/wenxi/pi-finetune/case/01_export_video
EXPECTED_NORM_SHA=507bc329fe6cd44bbc8fd49de82be3459e225e35ce6adb0310602ce1e51a432d
EXPECTED_MINT_COMMIT=32f5316291be29cd290e3d07969dda5bc922e44e
EXPECTED_OPENPI_COMMIT=dac1605516df0efa8638e9a93daa00c5baf7f2ad
READY=$OUTPUT_ROOT/delivery.ready
FAILED=$OUTPUT_ROOT/delivery.failed

log(){ printf '%s %s\n' "$(date -Is)" "$*"; }
fail(){ mkdir -p "$OUTPUT_ROOT"; log "FAILED: $*" | tee "$FAILED" >&2; exit 1; }

[[ "$QUERY_STRIDE" == 1 || "$QUERY_STRIDE" == 10 ]] || {
  echo "MODE3_QUERY_STRIDE must be 1 or 10, got $QUERY_STRIDE" >&2; exit 2;
}
[[ "$RUN_SMOKE" == 0 || "$RUN_SMOKE" == 1 ]] || {
  echo "MODE3_RUN_SMOKE must be 0 or 1, got $RUN_SMOKE" >&2; exit 2;
}
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "output already exists: $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
trap 'rc=$?; if ((rc != 0)) && [[ ! -f "$FAILED" ]]; then echo "$(date -Is) top-level rc=$rc" > "$FAILED"; fi' EXIT

for path in "$CLIENT" "$MINT_ROOT" "$OPENPI_ROOT" "$DATA" "$MANIFEST" "$GESTURE" "$NORM/norm_stats.json" "$PYTHON"; do
  [[ -e "$path" ]] || fail "required path missing: $path"
done
CLIENT_COMMIT=$(git -C "$CLIENT" rev-parse HEAD)
MINT_COMMIT=$(git -C "$MINT_ROOT" rev-parse HEAD)
OPENPI_COMMIT=$(git -C "$OPENPI_ROOT" rev-parse HEAD)
[[ -z "$(git -C "$CLIENT" status --porcelain)" ]] || fail "Mode3 client worktree is dirty"
[[ -z "$(git -C "$MINT_ROOT" status --porcelain)" ]] || fail "MINT worktree is dirty"
[[ -z "$(git -C "$OPENPI_ROOT" status --porcelain)" ]] || fail "OpenPI worktree is dirty"
[[ "$MINT_COMMIT" == "$EXPECTED_MINT_COMMIT" ]] || fail "MINT commit drift: $MINT_COMMIT"
[[ "$OPENPI_COMMIT" == "$EXPECTED_OPENPI_COMMIT" ]] || fail "OpenPI commit drift: $OPENPI_COMMIT"
ACTUAL_NORM_SHA=$(sha256sum "$NORM/norm_stats.json" | awk '{print $1}')
[[ "$ACTUAL_NORM_SHA" == "$EXPECTED_NORM_SHA" ]] || fail "norm SHA drift: $ACTUAL_NORM_SHA"

python3 - "$MANIFEST" <<'PY'
import json, sys
p=sys.argv[1]; d=json.load(open(p))
assert d["context_frames"] == 100, d.get("context_frames")
assert d["missing_policy"] == "error", d.get("missing_policy")
for row in (924,960,914,943,939):
    assert str(row) in d["windows"], row
PY

read_model_path(){
  python3 - "$1/train/result.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])).get("save_result",{}).get("path")
if not p: raise SystemExit("missing final save_result.path")
print(p)
PY
}
CLEAN_MODEL=$(read_model_path "$CLEAN_ROOT")
AUG_MODEL=$(read_model_path "$AUG_ROOT")

# Snapshot-level resource check immediately before launch.
python3 - <<'PY'
import socket
for port in (30536,30537):
    s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    try: s.bind(("127.0.0.1",port))
    except OSError as exc: raise SystemExit(f"port {port} unavailable: {exc}")
    finally: s.close()
PY
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100{c++} END{print c+0}')
[[ "$USED" == 0 ]] || fail "$USED GPUs already use more than 100 MiB"

run_arm(){ (
  set -Eeuo pipefail
  arm=$1; train_root=$2; model_path=$3; port=$4; gpus=$5; arm_out=$6
  mkdir -p "$arm_out"/{server,smoke,artifacts}
  server_pid=""
  cleanup(){
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
      kill -INT "$server_pid" 2>/dev/null || true
      for _ in $(seq 1 60); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  (
    export CUDA_VISIBLE_DEVICES=$gpus MINT_HOST=127.0.0.1 MINT_PORT=$port
    export MINT_SUPPORTED_MODELS=openpi/pi05-action-lora-r16-finetune
    export MINT_ALLOW_NO_RAY=1 MINT_SKIP_SUPERVISOR=1 MINT_USAGE_BACKEND=disabled MINT_UVICORN_WORKERS=1
    export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1
    export MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params
    export MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets
    export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR="$train_root/server/openpi_checkpoint_base"
    export MINT_RUNTIME_CHECKPOINT_DIR="$train_root/server/runtime_checkpoints"
    export MINT_CHECKPOINT_DIR="$train_root/server/runtime_checkpoints/persistent_cache"
    export MINT_PERSISTENT_CHECKPOINT_DIR="$train_root/server/runtime_checkpoints/persistent_cache"
    export MINT_TMP_ROOT="$arm_out/server/tmp"
    export MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT_BASE="$arm_out/server/action_state"
    # Do not enable JAX persistent executable caching here. pi0.5 lowers a
    # multi-GB executable and this runtime cannot serialize it; the compile
    # itself succeeds when cache serialization is disabled (the production
    # Mode4 server contract uses the same no-cache path).
    unset MINT_OPENPI_JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
    unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
    unset JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES
    unset JAX_RAISE_PERSISTENT_CACHE_ERRORS
    export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface
    export XLA_FLAGS=--xla_gpu_enable_command_buffer=
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
    export PYTHONPATH=$MINT_ROOT:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$EXTRA:$SITE
    export PYTHONNOUSERSITE=1
    mkdir -p "$arm_out/server"/{tmp,action_state}
    exec "$PYTHON" -u -c "import uvicorn; from mint_server.app import app; uvicorn.run(app,host='127.0.0.1',port=$port,workers=1,log_level='info')"
  ) >"$arm_out/server/server.log" 2>&1 &
  server_pid=$!
  echo "$server_pid" > "$arm_out/server/server.pid"
  code=000
  for _ in $(seq 1 240); do
    code=$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/openapi.json" 2>/dev/null || true)
    [[ "$code" == 200 ]] && break
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 1
  done
  [[ "$code" == 200 ]] || { echo "$arm server failed" > "$arm_out/arm.failed"; exit 1; }
  log "$arm server ready pid=$server_pid GPUs=$gpus port=$port"

  export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi HF_HOME=/vePFS-Mindverse/share/huggingface
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export LD_LIBRARY_PATH=/vePFS-Mindverse/share/zhouch-caches/.cache/openvla_full_a800/glvnd_shim:${LD_LIBRARY_PATH:-}
  export PYTHONPATH=$CLIENT/scripts/eval:$CLIENT:$CLIENT/scripts/train:$PI_FINETUNE:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src:$MINT_ROOT:$EXTRA:$SITE
  common=(
    --base-url "http://127.0.0.1:$port" --model-path "$model_path" --owner-id "$OWNER_ID"
    --lance-dataset "$DATA" --language-conditioning gesture --gesture-index "$GESTURE"
    --normalization-row-indices "$NORM_ROWS" --contact-window-manifest "$MANIFEST"
    --missing-contact-policy error --extended-state --norm-stats-dir "$NORM"
    --act-batch-size 4 --query-stride "$QUERY_STRIDE" --max-warm-request-seconds 2
    --client-commit "$CLIENT_COMMIT" --backend-commit "$MINT_COMMIT" --model-commit "$OPENPI_COMMIT"
  )

  if [[ "$RUN_SMOKE" == 1 ]]; then
    # At least two requests exercise model load/JIT, live state, sharding, and warm latency.
    smoke_frames=$((QUERY_STRIDE + 1))
    "$PYTHON" -u "$CLIENT/scripts/eval/infer_mano_mode3.py" "${common[@]}" \
      --row-index 924 --max-frames "$smoke_frames" --output-dir "$arm_out/smoke" \
      >"$arm_out/smoke/eval.log" 2>&1
    [[ -s "$arm_out/smoke/mode3/result.json" ]] || { echo "$arm smoke missing result" > "$arm_out/arm.failed"; exit 1; }
    log "$arm smoke passed"
  else
    log "$arm standalone smoke skipped; full run is the fail-closed production probe"
  fi

  "$PYTHON" -u "$CLIENT/scripts/eval/infer_mano_mode3.py" "${common[@]}" \
    --row-indices "$ROWS" --output-dir "$arm_out/artifacts" \
    >"$arm_out/eval.log" 2>&1
  log "$arm full Mode3 complete"
) }

log "launching clean and aug Mode3; client=$CLIENT_COMMIT"
run_arm clean "$CLEAN_ROOT" "$CLEAN_MODEL" 30536 0,1,2,3 "$OUTPUT_ROOT/clean" & CLEAN_PID=$!
run_arm stateaug005 "$AUG_ROOT" "$AUG_MODEL" 30537 4,5,6,7 "$OUTPUT_ROOT/stateaug005" & AUG_PID=$!
set +e
wait "$CLEAN_PID"; CLEAN_RC=$?
wait "$AUG_PID"; AUG_RC=$?
set -e
[[ "$CLEAN_RC" == 0 && "$AUG_RC" == 0 ]] || fail "arm failure: clean=$CLEAN_RC aug=$AUG_RC"

log "validating Mode3 artifacts"
export PYTHONPATH=$SITE:$EXTRA
"$PYTHON" - "$OUTPUT_ROOT" "$ROWS" "$CLIENT_COMMIT" "$MINT_COMMIT" "$OPENPI_COMMIT" "$EXPECTED_NORM_SHA" "$QUERY_STRIDE" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); rows=[int(x) for x in sys.argv[2].split(',')]
client,mint,openpi,norm=sys.argv[3:7]
query_stride=int(sys.argv[7])
mode=("historical_kinematic_mode3_sim_no_smooth" if query_stride == 10
      else "kinematic_mode3_replan1_first_action")
action_execution=("consume_full_nonoverlap_chunk" if query_stride == 10
                  else "replan_each_frame_execute_action_0")
entries=[]
for arm in ('clean','stateaug005'):
    for row in rows:
        d=root/arm/'artifacts'/f'row{row}'/'mode3'
        p=d/'result.json'
        if not p.is_file(): raise SystemExit(f'missing {p}')
        r=json.loads(p.read_text())
        expected={
            'mode':mode,
            'physics_dynamics':False,
            'mujoco_update':'mj_forward_only; mj_step_never_called',
            'state_observation_source':'sim',
            'image_observation_source':'sim',
            'object_pose_source':'reference_trajectory',
            'temporal_ensemble':False,
            'query_stride':query_stride,
            'action_execution':action_execution,
            'act_batch_size':4,
            'client_commit':client,
            'backend_commit':mint,
            'model_commit':openpi,
            'norm_sha_expected':norm,
            'norm_sha_actual':norm,
            'row_index':row,
        }
        for key,value in expected.items():
            if r.get(key) != value: raise SystemExit(f'{p}: {key}={r.get(key)!r}, expected {value!r}')
        n=int(r['trajectory_frame_count']); q=math.ceil(n/query_stride)
        if r['query_count'] != q: raise SystemExit(f'{p}: query_count {r["query_count"]} != {q}')
        frames=[x['source_frame'] for x in r['query_timings']]
        if frames != list(range(r['frame_window']['start_frame'], r['frame_window']['end_frame']+1, query_stride)):
            raise SystemExit(f'{p}: query frame drift')
        state=np.load(d/'state_observation_32d.npy')
        raw=np.load(d/'actions_raw_pred_physical.npy')
        commanded=np.load(d/'actions_commanded_physical.npy')
        applied=np.load(d/'actions_applied_physical.npy')
        hands=np.load(d/'hand_state_sim.npy')
        if state.shape != (n,32) or raw.shape != (n,32) or commanded.shape != (n,32) or applied.shape != (n,32) or hands.shape != (n+1,26):
            raise SystemExit(f'{p}: array shape mismatch')
        if not all(np.isfinite(x).all() for x in (state,raw,commanded,applied,hands)):
            raise SystemExit(f'{p}: non-finite array')
        if not np.array_equal(raw[:,26:],np.zeros((n,6),np.float32)):
            raise SystemExit(f'{p}: raw action tail not exact zero')
        if not np.array_equal(commanded[:,26:],np.zeros((n,6),np.float32)) or not np.array_equal(applied[:,26:],np.zeros((n,6),np.float32)):
            raise SystemExit(f'{p}: executed action tail not exact zero')
        if not np.isin(state[:,26:31],[0.0,1.0]).all(): raise SystemExit(f'{p}: non-binary contacts')
        videos=[]
        for key in ('head_video','wrist_video'):
            v=Path(r[key])
            if not v.is_file() or v.stat().st_size < 1000: raise SystemExit(f'{p}: missing/empty {key}')
            videos.append(str(v))
        entries.append({
            'arm':arm,'row':row,'seed_uuid':r.get('seed_uuid'),'prompt':r.get('prompt'),
            'frame_window':r['frame_window'],'query_count':q,
            'contact_positive_frames':state[:,26:31].sum(axis=0).astype(int).tolist(),
            'reference_lift_min':float(state[:,31].min()),'reference_lift_max':float(state[:,31].max()),
            'head_video':videos[0],'wrist_video':videos[1],'result_json':str(p),
            'head_sha256':hashlib.sha256(Path(videos[0]).read_bytes()).hexdigest(),
        })
manifest={
    'status':'validated','mode':mode,'query_stride':query_stride,
    'action_execution':action_execution,
    'rows':rows,'client_commit':client,'mint_commit':mint,'openpi_commit':openpi,
    'norm_sha256':norm,'entries':entries,
}
(root/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'delivery.ready').write_text(json.dumps({'status':'ready','video_count':len(entries),'manifest':str(root/'delivery_manifest.json')},indent=2)+'\n')
print(json.dumps({'validated_entries':len(entries),'manifest':str(root/'delivery_manifest.json')}))
PY
[[ -s "$READY" ]] || fail "delivery.ready not written"
rm -f "$FAILED"
trap - EXIT
log "Mode3 delivery ready: $READY"
