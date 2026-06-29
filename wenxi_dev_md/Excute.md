# Excute.md — exact commands per goal (VLA / OpenPI on dev)

Companion to `Plan.md`. This file holds the HOW: copy-pasteable commands.

> Conventions
> - `<you>` = `wenxi`; namespace = `mint_wenxi_dev`.
> - `MINT_CODE_ROOT` MUST be under `/vePFS-Mindverse/share/...` (visible to all
>   Ray nodes). We use a shared mirror of this branch, not `/user/...`.
> - Client/tool scripts run LOCALLY (talk to server over HTTP). Server/ops
>   scripts run on `mint-dev` via SSH.
> - NEVER run `ray`/`volc` locally. Use the `volcano-cluster` skill.
> - NEVER `rsync --delete`.
> - Status markers below: [ ] todo, [x] done, [~] partial. Update as we go.

Shared variables (set once per shell where noted):

```bash
export YOU=wenxi
export BRANCH=dev-vla-wenxi
export NS=mint_wenxi_dev
export SHARE_CODE=/vePFS-Mindverse/share/code/$YOU/$BRANCH   # PFS mirror, all-node visible
export LOCAL_CHECKOUT=/vePFS-Mindverse/user/intern/wenxi/mint # this working copy

# Dev driver host (per user 2026-06-22): run the API server HERE, not on the Ray
# head. No more Ray Client mode. sshd is on port 2222.
export DRIVER_IP=192.168.42.106          # mint-dev-driver
export DRIVER_PORT=2222
export RAY_HEAD_IP=192.168.42.141
export SSH_KEY=$HOME/.ssh/ssh_worker_rsa_key
# Convenience ssh wrapper for the driver (use instead of `ssh mint-dev` until the
# alias is configured):
sshdrv() { ssh -i "$SSH_KEY" -p "$DRIVER_PORT" -o StrictHostKeyChecking=no root@"$DRIVER_IP" "$@"; }

# Runtime root (CORRECTED): the DEFAULT dev runtime is sufficient for pi0.5.
#  - cpu tier  -> API host interpreter (no torch/jax)
#  - gpu_rl tier -> GPU workers; already contains openpi+jax (commit matches
#    pyproject). OpenPI runtime requests gpu_rl by default; nothing requests
#    gpu_vla. Do NOT build a gpu_vla tier; do NOT use the flat candidate root.
export DEV_RUNTIME=/vePFS-Mindverse/share/mint/dev/runtime
export GPU_RL_BUILD=$DEV_RUNTIME/gpu_rl     # for read-only import probes
```

---

## Goal 0 — Prerequisite (must clear before Goal 1)  [ ]

Only ONE real blocker remains (the runtime is fine — see Plan.md "Findings A").

### 0.1 SSH into the dev driver  [blocked: key not on 106 local authorized_keys]

Host-key fingerprints proved: `106:2222` is a **system sshd** reading 106's LOCAL
`~/.ssh/authorized_keys` (our key is NOT there → Permission denied). The mint-sshd
that reads the shared `/share/mint/runtime/ssh/authorized_keys` runs on the Ray
**head** `141:22`, which the user excluded ("不是去 ray head 上"). So the shared-file
mechanism does not grant access to the driver.

```bash
# Our key IS correctly in the shared file (verified: privkey-derived pubkey ==
# the root@di-...985z8 line). That file just isn't what 106:2222 reads.
# Test (currently Permission denied):
sshdrv 'echo CONNECTED $(whoami)@$(hostname)'
```

Resolution needs the user: either push `ssh_worker_rsa_key.pub` into
`192.168.42.106:~/.ssh/authorized_keys`, or run driver commands locally via the
`!` prefix. Do NOT loop on ssh retries; do NOT ssh the Ray head.

> (Removed the earlier "build a gpu_vla symlink root" step — it was based on a
> wrong assumption. No runtime path requests gpu_vla; the default runtime works.)

---

## Goal 1 — Bring up the dev MinT server  [ ]

> NOTE: use the `sshdrv` wrapper (driver `192.168.42.106:2222`) once SSH works,
> and the DEFAULT `PFS_RUNTIME_ENV_ROOT` (the script's default — omit the var).

### 1.0 Pre-flight (read-only)

```bash
# Confirm dev Ray head address is published
sshdrv 'cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt'

# Is a server already listening / are there stale actors in our namespace?
sshdrv 'curl -s http://localhost:8000/api/v1/healthz; echo'
sshdrv 'ps aux | grep run_server | grep -v grep'
```

If the GPU cluster/worker is not up: STOP and use the `volcano-cluster` skill to
create it. Do not run `volc`/`ray` locally.

### 1.1 Sync this branch to the PFS mirror (no --delete)

```bash
mkdir -p "$SHARE_CODE"
rsync -a --exclude '.git' --exclude '__pycache__' \
  "$LOCAL_CHECKOUT"/ "$SHARE_CODE"/
```

### 1.2 Generate placement config (worker IPs change on cluster recreate)

```bash
cd "$LOCAL_CHECKOUT"
HEAD_IP=$(ssh mint-dev 'cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt')
python scripts/tools/gen_dev_placement.py --head-ip "$HEAD_IP" \
  --model Qwen/Qwen3-0.6B --gpu-count 1 \
  --output /tmp/mint_dev_run.env
scp /tmp/mint_dev_run.env mint-dev:/tmp/mint_dev_run.env
```

(Model here just seeds a placement entry; VLA placement is added in Goal 2.)

### 1.3 Start the server (NON-VLA baseline — proves the control plane)

```bash
ssh mint-dev 'MINT_CODE_ROOT='"$SHARE_CODE"' \
  MINT_DEV_USER=wenxi \
  MINT_RAY_NAMESPACE=mint_wenxi_dev \
  MINT_PORT=8000 \
  MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-wenxi-server.log \
  MINT_DISABLE_MINT_ROUTE=1 \
  MINT_UVICORN_WORKERS=1 \
  MINT_SUPERVISOR_STATE_BACKEND=memory \
  MINT_DEV_RUN_ENV=/tmp/mint_dev_run.env \
  nohup '"$SHARE_CODE"'/scripts/start_dev_server.sh \
  >> /tmp/mint_wenxi_launch.log 2>&1 &'
```

### 1.4 Wait for health + open tunnel

```bash
ssh mint-dev 'for i in $(seq 1 60); do curl -s http://localhost:8000/api/v1/healthz && break; sleep 5; done; echo'
# Expected: {"status":"ready"}

ssh -f -N -L 8000:localhost:8000 mint-dev   # local access
MINT_BASE_URL=http://localhost:8000 MINT_API_KEY=dummy \
  python "$LOCAL_CHECKOUT"/scripts/tools/smoke.py service
```

### 1.5 (optional) RL sanity to prove training path on a tiny text model

```bash
MINT_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy MINT_API_KEY=dummy \
  python "$LOCAL_CHECKOUT"/scripts/tools/rl_check.py \
  --model Qwen/Qwen3-0.6B --steps 10 --group-size 4 --timeout-s 600
```

Goal 1 done when: healthz ready, tunnel works, smoke `service` passes.

---

## Goal 2 — Run pi0.5 (VLA, flow_matching) end-to-end  [ ]

Restart the server with the **OpenPI runtime + VLA routes enabled + pi0.5
advertised + weights/assets pointed at the shared bundle**. This replaces the
Goal-1 baseline server.

### 2.0 Verify the OpenPI runtime imports (read-only, on a node that can see PFS)

```bash
# The DEFAULT runtime's gpu_rl build already has openpi+jax (this is what the
# OpenPI worker uses — from_env defaults to the gpu_rl tier):
"$GPU_RL_BUILD/host-venv/bin/python" -c "import jax; print('jax', jax.__version__)"
PYTHONPATH="$GPU_RL_BUILD/site-packages:$GPU_RL_BUILD/src/openpi/src:$GPU_RL_BUILD/src/openpi/packages/openpi-client/src" \
  "$GPU_RL_BUILD/host-venv/bin/python" -c "import openpi; print('openpi OK')"
```

### 2.1 Restart server with VLA config

If a baseline server is running, stop it first and clean stale control-plane
actors (see section 4 "Restart after code changes / reconfigure").

`PFS_RUNTIME_ENV_ROOT` is omitted → uses the script default
(`/vePFS-Mindverse/share/mint/dev/runtime`), which is correct for pi0.5.

```bash
sshdrv 'MINT_CODE_ROOT='"$SHARE_CODE"' \
  MINT_DEV_USER=wenxi \
  MINT_RAY_NAMESPACE=mint_wenxi_dev \
  MINT_PORT=8000 \
  MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-wenxi-server.log \
  MINT_UVICORN_WORKERS=1 \
  MINT_SUPERVISOR_STATE_BACKEND=memory \
  MINT_DEV_RUN_ENV=/tmp/mint_dev_run.env \
  MINT_SUPPORTED_MODELS="openpi/pi0-fast-libero-low-mem-finetune,openpi/pi05-libero-low-mem-finetune,Qwen/Qwen3-0.6B" \
  MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params \
  MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/models/openpi/pi05_base/assets \
  MINT_OPENPI_FAST_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi0_fast_base/params \
  MINT_OPENPI_FAST_ASSETS_BASE_DIR=/vePFS-Mindverse/share/models/openpi/pi0_fast_base_official_20260428/assets \
  MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1 \
  nohup '"$SHARE_CODE"'/scripts/start_dev_server.sh \
  >> /tmp/mint_wenxi_launch.log 2>&1 &'
```

Notes:
- `MINT_DISABLE_MINT_ROUTE` is intentionally UNSET → `/api/v1/mint/*` loads.
- **Base is develop `d86e1487` (VLA rollup #698):** OpenPI workers run directly
  in the Ray actor (no subprocess). Keep `MINT_UVICORN_WORKERS=1` (rollup's
  control-plane authority gate is not yet final).
- `MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1` is set so the pi0.5 action session
  can create its runtime actor directly instead of requiring the supervisor to
  reconcile it first. This is an explicit smoke-path bypass — fine for Goal 2
  validation, not the general production path.
- Confirm pi0.5 is advertised: `curl .../api/v1/server_info` (or supported models).
- The pi0.5 GPU worker (`num_gpus=1`) is created lazily on first request; first
  create/train_step can take a while (engine init). Poll the future, don't abort.

### 2.2 Synthetic smoke (no dataset — proves the service path)  [ ]

```bash
ssh -f -N -L 8000:localhost:8000 mint-dev   # if tunnel not already up
MINT_BASE_URL=http://localhost:8000 MINT_API_KEY=dummy \
  python "$LOCAL_CHECKOUT"/scripts/wip/openpi_vla_smoke.py \
  --model openpi/pi05-libero-low-mem-finetune \
  --output-json /tmp/pi05_smoke.json
```

Pass = JSON with a non-empty `action_result` (action tensor) and train_result
metrics. This covers create_model → vla/train_step(flow_matching) →
save_weights_for_sampler → action_session → act → cleanup.

### 2.3 Real-data LIBERO SFT (downward loss curve)  [ ]

Runs LOCALLY; reads LIBERO from PFS; needs the `openpi` package importable in the
LOCAL python (the script imports `openpi.training.config`). Use a python that has
openpi on its path (e.g. the openpi runtime interpreter) if the default lacks it.

```bash
export OPENPI_DATA_HOME=/vePFS-Mindverse/share/code/conley/.openpi_cache
export HF_HOME=/vePFS-Mindverse/share/huggingface
MINT_BASE_URL=http://localhost:8000 MINT_API_KEY=dummy \
  python "$LOCAL_CHECKOUT"/scripts/wip/openpi_libero_sft.py \
  --base-url http://localhost:8000 \
  --base-model openpi/pi05-libero-low-mem-finetune \
  --task-index 10 \
  --steps 12 --batch-size 2 --stride 5 --max-episodes 8 \
  --output-dir /tmp/sft_pi05_task10
```

Pass = `summary.json` + `loss_curve.png` with `final_loss < initial_loss`
(reference 2026-04-08: ~0.119 → ~0.063 on task 10 "put the bowl on the plate").

Goal 2 done when: 2.2 returns an action tensor AND 2.3 shows a real downward
LIBERO loss curve for pi0.5.

---

## Goal 3 — Trace the pi0.5 data path  [ ]

Mostly an analysis deliverable; commands here are for inspection.

### 3.1 Inspect the LIBERO dataset shape

```bash
DS=/vePFS-Mindverse/share/code/conley/.hf-lerobot/physical-intelligence/libero
head -5 "$DS"/meta/tasks.jsonl
head -3 "$DS"/meta/episodes.jsonl
ls "$DS"/data/ | head
```

### 3.2 Key code references to read (file:line)

- Client lowering (parquet → transform → Datum):
  `scripts/wip/openpi_libero_sft.py:126` (`_iter_windows_for_task`),
  `:186` (`_pi05_datum_from_transformed`), `:112` (`_build_transform`).
- Route: `mint_server/routes/mint.py` (`vla_train_step`).
- Dispatch: `mint_server/backend/scheduling/model_work_dispatch.py`
  (`mint.vla.train_step`).
- Server training engine: `mint_server/backend/openpi/openpi_pi05_training.py:300`
  (`forward_backward`, asserts `flow_matching`),
  `:145` (`build_openpi_pi05_sft_runtime_payload`).
- GPU worker: `mint_server/backend/openpi/openpi_pi05_worker.py`.
- Runtime env for OpenPI actors:
  `mint_server/backend/openpi/openpi_ray_runtime.py:26` (`_openpi_runtime_env_vars`).
- Registry entry: `mint_server/backend/core/model_registry.py:150-164`.

Deliverable: write the end-to-end trace (one frame → loss) into Plan.md's Goal 3
section or a sibling note.

---

## 4. Restart after code changes / reconfigure

Python does not hot-reload; detached actors do not hot-reload either.

```bash
# 1. Kill the API server
ssh mint-dev 'kill $(pgrep -f "scripts/run_server.py" | head -1) 2>/dev/null; sleep 2'

# 2. Clean stale control-plane actors in OUR namespace (only if fingerprint
#    mismatch or after code change). Use the runtime interpreter + head IP.
#    (See .claude/skills/mint-dev/SKILL.md "Restart After Code Changes" for the
#    full python snippet — kills mint_config, mint_task_state_store,
#    mint_model_work_scheduler, mint_maintenance_cron, mint_model_actor_supervisor
#    in namespace mint_wenxi_dev.)

# 3. Re-sync code (NO --delete)
rsync -a --exclude '.git' --exclude '__pycache__' "$LOCAL_CHECKOUT"/ "$SHARE_CODE"/

# 4. Restart with the appropriate command (Goal 1 baseline or Goal 2 VLA)

# 5. Verify a NEW process and the new log
ssh mint-dev 'ps aux | grep run_server | grep -v grep'
```

## 5. Cleanup (required closeout)

```bash
# Kill all named actors in namespace mint_wenxi_dev (full snippet in mint-dev
# SKILL.md "Cleanup"), stop the API server, confirm the port is free.
ssh mint-dev 'kill $(pgrep -f "scripts/run_server.py" | head -1) 2>/dev/null'
ssh mint-dev 'curl -s http://localhost:8000/api/v1/healthz; echo "(expect connection refused after stop)"'
```

Do not mark a goal done while owned actors/PGs remain in the namespace.

## 6. Health / logs / diagnostics

```bash
ssh mint-dev 'curl -s http://localhost:8000/api/v1/healthz; echo'
ssh mint-dev 'curl -s http://localhost:8000/api/v1/server_info; echo'
ssh mint-dev 'curl -s http://localhost:8000/internal/actors; echo'          # actor inventory
ssh mint-dev 'curl -s http://localhost:8000/internal/admission_stats; echo'  # scheduler/supervisor
ssh mint-dev 'tail -80 /vePFS-Mindverse/share/mint/dev/logs/mint-wenxi-server.log'
```

