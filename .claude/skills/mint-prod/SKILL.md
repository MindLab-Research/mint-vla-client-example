---
name: mint-prod
description: |
  Production environment operations for the Mint server on Volcano cluster.

  Use for: code sync, server start/stop, vLLM management, logs - all in PROD environment.

  Triggers: "prod server", "start prod", "restart prod", "prod logs", "sync to prod", "prod vLLM", "production"

  **Do NOT invoke this skill for development work. Use mint-dev instead.**

  For cluster lifecycle (create/teardown tasks), invoke the volcano-cluster skill.

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Mint Production Environment

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

> **STOP. USE THESE COMMANDS EXACTLY.**
>
> Do NOT guess SSH hosts, log locations, or process names. Everything is documented below.
>
> | Task | Command |
> |------|---------|
> | SSH to server | `ssh mint-prod-volcano` (NOT `mint-dev`, NOT direct IP) |
> | Server logs | `ssh mint-prod-volcano "tail -50 /tmp/tinker_server_auth.log"` |
> | Health check | `curl http://localhost:18000/api/v1/healthz` |
> | Restart server | `ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'` |
> | Stop server (fallback) | `ssh mint-prod-volcano 'fuser -k 18000/tcp'` (NOT pkill) |
> | Kill vLLM | `curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" -d '{"actor_type":"vllm"}' http://localhost:18000/api/v1/actors/kill` |
>
> If you find yourself guessing or trial-and-error debugging basic infrastructure, **STOP and re-read this skill**.
>
> Note: `GET /api/v1/healthz` is the cheap public API-worker health endpoint. For costly Ray / placement-group diagnostics, use the internal deep health surface instead of expecting `healthz` to reflect cluster capacity.

---

## NEVER Do These (Development Belongs to mint-dev)

- **NEVER** `ssh mint-dev` - that's development
- **NEVER** use port `8000` - that's development
- **NEVER** run `rsync` with `--delete` on production paths
- **NEVER** sync virtualenvs or temp dirs (exclude `.venv*/`, `.venv_cpu/`, `.unison*/`, `cpu-pydeps/`)
- **NEVER** interrupt a running `rsync` (partial trees cause hard-to-debug corruption)
- **NEVER** use `mint-dev-*.yaml` Ray configs - that's development
- **NEVER** use `tinker-server` directory (without `-auth`) - that's development
- **NEVER** use `pkill -f "run_server"` - may kill dev server; use `fuser -k 18000/tcp`
- **NEVER** omit `X-API-Key` header on API calls (except healthz)
- **NEVER** omit `PYTHONPATH` override - causes auth bypass

If user asks for development operations, **stop and invoke mint-dev skill instead**.

---

## Environment Config

| Property | Value |
|----------|-------|
| SSH Host | `mint-prod-volcano` |
| Port | 18000 |
| External URL | `https://mint.macaron.im` (international), `https://mint.macaron.xin` (China) |
| Code Directory | `tinker-server-auth` |
| PFS Path | `/vePFS-Mindverse/share/code/tinker-server-auth` |
| Ray Configs | `.claude/skills/volcano-cluster/configs/mint-prod-head.yaml`, `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml` |
| Prod GPU Queues | Volcano prod workers MUST be on `a800-mindverse-C1` (`q-20251126180002-26lwz`). Do not submit `mint-prod-worker` to C2. |
| API Key | **Required** (`X-API-Key` header) |
| Log File | `/tmp/tinker_server_auth.log` |

## Production Topology (Authoritative)

Cluster management:
- Volcano: exactly 2 worker nodes named mint-prod-worker (16 GPUs).
- Aliyun: exactly 3 worker nodes named mint-prod-worker (24 GPUs).
- Volcano: prod workers MUST be submitted to queue C1 (`q-20251126180002-26lwz`), not C2.

Model lineup:
- Main gateway on volcano; volcano hosts 0.6B, 4B and 30B itself.
- 235B on aliyun (public access address: http://123.57.26.97:18000/)
- K2: will be routed, skip since not ready yet.

Placement:
- Main volcano: 30B vllm + 30B megatron on node 1; 0.6B vllm + 0.6B peft + 4B vllm + 4B peft on node 2.
- Main aliyun: 235B vllm on node 1; 235B megatron on node 2+3.

API host Ray driver precondition:
- One-time setup only: if `mint-prod-volcano` has not yet joined the Ray cluster as a local 0-GPU node, `tinker-server-auth` startup can fail in `ray.init()`.
- If already joined, do not repeat this step during normal deploy/restart.
- One-time command:
  `ssh mint-prod-volcano '/root/tinker_project/tinker-server-auth/.venv31213/bin/ray start --address=192.168.37.147:6379 --node-ip-address=192.168.47.239 --num-cpus=0 --num-gpus=0 --disable-usage-stats'`

## Queue Placement SOP (C1/C2)

K2 is not in production service yet. Skip this section unless the task is explicitly K2 placement or K2 bringup.

### Queue IDs (prod)

| Alias | ResourceQueueId | Intended use |
|------|------------------|--------------|
| C1 | `q-20251126180002-26lwz` | Default GPU queue |
| C2 | `q-20260203101340-www2h` | K2 multinode vLLM queue |

### Routing policy (must hold)

| Actor/workload | Queue |
|----------------|-------|
| K2 Megatron (`moonshotai/Kimi-K2-Instruct`) | C1 |
| K2 vLLM (`moonshotai/Kimi-K2-Instruct`) | C2 |
| Everything else | C1 |

### Query placement (authoritative procedure)

1) Query actor inventory:
```bash
source .secrets.env
curl -s -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/actors | jq '.actors[] | {actor_name,actor_type,base_model,idle,current_session,num_gpus}'
```

2) Query queue-to-node IP mapping:
```bash
ssh mint-prod-volcano "cd /vePFS-Mindverse/share/code/tinker-server-auth && python3 - <<'PY'
from tinker_server.backend.volc_placement import list_node_ips_for_resource_queue
queues = {
    'C1': 'q-20251126180002-26lwz',
    'C2': 'q-20260203101340-www2h',
}
for name, rq in queues.items():
    ips = sorted(list_node_ips_for_resource_queue(resource_queue_id=rq))
    print(name, rq, len(ips), ','.join(ips))
PY"
```

3) Validate K2 Megatron run placement by run window (model_id-scoped):
```bash
mid='<k2_model_id>'
start_line=$(ssh mint-prod-volcano "rg -n '\\[$mid\\] Creating MegatronWorkerGroup' /tmp/tinker_server_auth.log | tail -n1 | cut -d: -f1")
ssh mint-prod-volcano "printf 'c2_megatron_lines='; tail -n +${start_line} /tmp/tinker_server_auth.log | rg '\\(MegatronRankWorker pid=[0-9]+, ip=192\\.168\\.33\\.(163|164|165|166|167|168|169|170)\\)' | wc -l"
ssh mint-prod-volcano "printf 'c1_megatron_lines='; tail -n +${start_line} /tmp/tinker_server_auth.log | rg '\\(MegatronRankWorker pid=[0-9]+, ip=192\\.168\\.33\\.(43|45|47|48|50|51|55|56|57|58|59|60|142|144|146|147)\\)' | wc -l"
```

Expected for K2 Megatron: `c2_megatron_lines=0`, `c1_megatron_lines>0`.

### Ensure placement (configuration + recycle actors)

Set queue env vars in prod `.secrets.env`:
```bash
MINT_MEGATRON_VOLC_RESOURCE_QUEUE_ID=q-20251126180002-26lwz
MINT_K2_INFER_VOLC_RESOURCE_QUEUE_ID=q-20260203101340-www2h
```

Apply and recycle K2 actors:
```bash
source .secrets.env
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
curl -s -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"actor_type":"megatron","model_name":"moonshotai/Kimi-K2-Instruct"}' \
  http://localhost:18000/api/v1/actors/kill
curl -s -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"actor_type":"vllm","model_name":"moonshotai/Kimi-K2-Instruct"}' \
  http://localhost:18000/api/v1/actors/kill
```

Then recreate actors as needed and re-run placement checks above.

**Reverse Proxy:** Port 18000 is reverse-proxied by Azure Gateway at `https://mint.macaron.im` (international) and `https://mint.macaron.xin` (China). No additional setup required.

**IMPORTANT:** All API calls (except `/api/v1/healthz` and `/`) require `X-API-Key` header.

---

**Worker queue selection:** `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml` uses a `<GPU_QUEUE_ID>` placeholder. For Volcano prod, set it to C1 (`q-20251126180002-26lwz`) and do not submit `mint-prod-worker` to C2.

## Finding the Server Process

**Always verify the actual log file location before tailing logs:**

```bash
# Find server process
ssh mint-prod-volcano 'ps aux | grep run_server | grep -v grep'

# Check where stdout goes (actual log file)
ssh mint-prod-volcano 'ls -la /proc/<PID>/fd/1'

# Example output: /proc/31501/fd/1 -> /tmp/tinker_server_auth.log
```

The log file is typically `/tmp/tinker_server_auth.log`, but verify with the above if logs seem stale.

---

## Quick Reference

```bash
# SSH tunnel
ssh -f -N -L 18000:localhost:18000 mint-prod-volcano

# Health check (no auth needed)
curl http://localhost:18000/api/v1/healthz

# Server logs
ssh mint-prod-volcano "tail -50 /tmp/tinker_server_auth.log"

# vLLM status (auth required)
curl -H "X-API-Key: $TINKER_API_KEY" "http://localhost:18000/api/v1/actors?type=vllm"

# Kill vLLM (auth required)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" -d '{"actor_type":"vllm"}' http://localhost:18000/api/v1/actors/kill
```

---

## 1. Code Synchronization

> **CRITICAL: DEPLOY VIA RSYNC (NO `--delete`)**
>
> Production code sync is **unidirectional rsync** from local to server.
>
> Hard rule: **do not use `--delete`**. Also exclude any environment or temp trees so rsync cannot clobber them.

### Deploy latest `origin/main`

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
```

### Sync code to production (dry-run first)

```bash
rsync -avz --dry-run \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='.secrets.env' \
  --exclude='.venv' \
  --exclude='.venv*/' \
  --exclude='.venv_cpu/' \
  --exclude='cpu-pydeps/' \
  --exclude='.unison*' \
  --exclude='LOG.md' \
  --exclude='PROMPT.md' \
  --exclude='ray_head_ip.txt' \
  --exclude='.claude' \
  ./ mint-prod-volcano:/vePFS-Mindverse/share/code/tinker-server-auth/
```

### Execute sync

```bash
rsync -avz \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='.secrets.env' \
  --exclude='.venv' \
  --exclude='.venv*/' \
  --exclude='.venv_cpu/' \
  --exclude='cpu-pydeps/' \
  --exclude='.unison*' \
  --exclude='LOG.md' \
  --exclude='PROMPT.md' \
  --exclude='ray_head_ip.txt' \
  --exclude='.claude' \
  ./ mint-prod-volcano:/vePFS-Mindverse/share/code/tinker-server-auth/
```

### Restart to load new code

```bash
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
```

**SSH server symlink setup** (one-time):
```bash
ssh mint-prod-volcano "rm -rf /root/tinker_project/tinker-server-auth && \
  ln -s /vePFS-Mindverse/share/code/tinker-server-auth /root/tinker_project/tinker-server-auth"
```

---

## 2. Server Management

### Environment Variables

**Prod runs under supervisord and sources `/root/tinker_project/tinker-server-auth/.secrets.env`.**

**Update secrets on prod (required when rotating credentials):**
```bash
# From repo root: copy local `.secrets.env` to prod and restart (single file only)
rsync -av .secrets.env mint-prod-volcano:/root/tinker_project/tinker-server-auth/.secrets.env
ssh mint-prod-volcano 'chmod 600 /root/tinker_project/tinker-server-auth/.secrets.env'
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
```

**Load secrets locally (for auth-required curl):**
```bash
source .secrets.env
```

```bash
export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/tinker_project/tinker-server-auth:$PYTHONPATH
export TINKER_API_KEY=$TINKER_API_KEY          # from .secrets.env
export TINKER_TOKEN_SECRET_KEY=$TINKER_TOKEN_SECRET_KEY  # from .secrets.env
export TINKER_PORT=18000
export TINKER_CHECKPOINT_DIR=/vePFS-Mindverse/share/tinker_checkpoints

 # Persistent actors (server startup prewarm; eviction-protected)
 # Configure these in `.secrets.env` and restart the server to apply.
 export MINT_PERSISTENT_MODELS="Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-30B-A3B-Instruct-2507"
 export MINT_PERSISTENT_TRAIN_LORA_RANK=16
 export MINT_PERSISTENT_TRAIN_LR=5e-5
 export MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S=3600
 export MINT_MEGATRON_EVICT_PROTECTED=1  # allow full-cluster Megatron to preempt idle protected actors
```

### Multi-target Model Routing (Gateway)

 Prod can run as a gateway/router that forwards selected base models to other tinker-server deployments.

 Deployment targets (current plan):
 - `mint-prod-volcano` (this server): `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-30B-A3B-Instruct-2507`
 - `mint-prod-aliyun`: `Qwen/Qwen3-235B-A22B-Instruct-2507`
 - K2: planned to be routed; do not treat K2 as production-ready.

 Router config (set on `mint-prod-volcano` only):
```bash
export TINKER_GATEWAY_CONFIG_JSON='
{
  "model_to_upstream": {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "mint-prod-aliyun"
  },
  "upstreams": {
    "mint-prod-aliyun": {
      "base_url": "http://<mint-prod-aliyun-host>:18000",
      "auth_mode": "pass_through"
    }
  }
}
'
```

Current Aliyun deployment (example):
```bash
export TINKER_GATEWAY_CONFIG_JSON='{"model_to_upstream":{"Qwen/Qwen3-235B-A22B-Instruct-2507":"mint-prod-aliyun"},"upstreams":{"mint-prod-aliyun":{"base_url":"http://123.57.26.97:18000","auth_mode":"pass_through"}}}'
```

Verify the gateway advertises the routed model (MUST include `Qwen/Qwen3-235B-A22B-Instruct-2507`):
```bash
source .secrets.env
curl -s -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/get_server_capabilities | python3 -m json.tool
```

Smoke test remote 235B training model creation (gateway forwards to Aliyun):
```bash
source .secrets.env
python3 - <<'PY'
import json, os, subprocess, time
api_key=os.environ["TINKER_API_KEY"]
hdr=["-H", f"X-API-Key: {api_key}", "-H", "Content-Type: application/json"]
def post(path, body):
    out=subprocess.check_output(["curl","-s","-w","\\n%{http_code}",*hdr,"-d",json.dumps(body),f"http://localhost:18000{path}"], text=True)
    payload, code_s = out.rsplit("\\n", 1)
    return int(code_s), payload
code, payload = post("/api/v1/create_session", {})
sid=json.loads(payload)["session_id"]
code, payload = post("/api/v1/create_model", {"session_id": sid, "model_seq_id": 0, "base_model": "Qwen/Qwen3-235B-A22B-Instruct-2507", "user_metadata": {}, "lora_config": {"rank": 16}})
req_id=json.loads(payload)["request_id"]
for _ in range(240):
    code, payload = post("/api/v1/retrieve_future", {"request_id": req_id})
    if code == 408:
        time.sleep(1); continue
    print(payload); break
PY
```

Upstream (remote) server config requirements:
- Set `MINT_SUPPORTED_MODELS` on each deployment to the models it MUST advertise.
- If `get_server_capabilities` on the upstream does not include a routed model, the gateway treats it as misconfiguration and fails requests for that model.

GPU-aware tuning knobs:
- GPU types: Volcano uses A800 80GB; Aliyun uses H (SM90).
- Tune per-model TP/EP/CP and vLLM memory caps in `tinker_server/backend/model_registry.py`.
- For environment-specific tuning without code changes, set `MINT_MODEL_CONFIG_OVERRIDES_JSON`:
  ```bash
  export MINT_MODEL_CONFIG_OVERRIDES_JSON='{"Qwen/Qwen3-235B-A22B-Instruct-2507":{"inference_tp":16,"train_pp":1,"train_ep":8,"max_num_batched_tokens":2048,"vllm_distributed_executor_backend":"ray"}}'
  ```

**IMPORTANT:** `PYTHONPATH` must prioritize `tinker-server-auth` to override pip-installed `tinker-server`. Without this, auth middleware is bypassed.

**Note:** No default model is configured. Clients specify models per-request. Model paths are resolved via `_resolve_model_path()` in `multi_lora_engine.py`.

### Restart Server (supervisord)

```bash
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
```

### Stop Server

**Use `fuser` to kill only port 18000. Do NOT use `pkill` - it may kill dev server too.**

```bash
ssh mint-prod-volcano 'fuser -k 18000/tcp'
```

### Check Status

```bash
ssh mint-prod-volcano "ps aux | grep run_server | grep -v grep"
```

---

## 3. GPU Actor Admin

| Operation | Time | When to use |
|-----------|------|-------------|
| Reconnect (existing) | ~2s | Server restart, GPU actor still alive |
| Kill + restart | ~80s | Base model changed, OOM, GPU actor code changed |

Current admin route:
- `POST /api/v1/actors/kill`
- `actor_type`: `vllm`, `megatron`, `dense`, or `all`
- optional filters: `model_name`, `actor_name`, `force`, `reason`

### Kill vLLM Actor

```bash
# Kill all tracked vLLM actors
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json"   -d '{"actor_type":"vllm"}' http://localhost:18000/api/v1/actors/kill

# Kill one model's vLLM actor
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json"   -d '{"actor_type":"vllm","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507"}'   http://localhost:18000/api/v1/actors/kill
```

### Kill Megatron Actor

```bash
# Kill all tracked Megatron actors
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json"   -d '{"actor_type":"megatron"}' http://localhost:18000/api/v1/actors/kill

# Kill one model's Megatron actor
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json"   -d '{"actor_type":"megatron","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507"}'   http://localhost:18000/api/v1/actors/kill
```

### Check Actor Status

```bash
# vLLM status
curl -s -H "X-API-Key: $TINKER_API_KEY" "http://localhost:18000/api/v1/actors?type=vllm" | jq

# Megatron status
curl -s -H "X-API-Key: $TINKER_API_KEY" "http://localhost:18000/api/v1/actors?type=megatron" | jq

# Full tracked actor inventory
curl -s -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/actors | jq '.actors[] | {actor_name,actor_type,base_model,idle,current_session,num_gpus}'
```

### Kill all tracked GPU actors

```bash
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json"   -d '{"actor_type":"all"}' http://localhost:18000/api/v1/actors/kill
```

`{"actor_type":"all"}` kills only ModelActorRegistry-tracked GPU actors. It does not clear detached scheduler/runtime/store actors.

---

## 4. Code Update SOP

### Rule 0: server restart does not reload detached actors

The API server is a Python process. Ray actors are separate Python processes.

Detached actors (GPU actors, queue/control-plane actors, stores) survive server restarts and keep running old code until killed.

Hard rule: never create/get/kill actors outside `TINKER_RAY_NAMESPACE` unless the user explicitly requests cross-namespace action.

### Kill criteria after code changes

Always restart the server after code changes.

Kill GPU actors only if the change can be imported/executed inside that actor process:
- vLLM: `tinker_server/backend/verl_inference.py`, `tinker_server/backend/multi_lora_engine.py`, `tinker_server/backend/multinode_inference.py`, `tinker_server/backend/vllm_*.py`
- Megatron: `tinker_server/backend/megatron_distributed.py`, `tinker_server/backend/megatron_training.py`, `tinker_server/backend/verl_patches.py`
- Dense training pool: `tinker_server/backend/verl_training.py`

Kill detached control-plane actors in the current namespace if the diff can be imported/executed inside the scheduled request path, detached stores, or detached cleanup loops:
- control plane core: `tinker_server/backend/model_work_scheduler.py`, `tinker_server/backend/model_runtime_actor.py`, `tinker_server/backend/model_runtime_executor.py`, `tinker_server/backend/maintenance_cron_actor.py`
- detached state: `tinker_server/backend/task_state_store.py`, `tinker_server/backend/task_payload_store.py`, `tinker_server/backend/startup_lease.py`, `tinker_server/backend/future_replay.py`
- detached stores: `tinker_server/backend/sampling_session_store.py`, `tinker_server/backend/training_session_store.py`, `tinker_server/backend/gateway_session_store.py`, `tinker_server/backend/session_heartbeat_store.py`, `tinker_server/backend/session_index_store.py`
- detached cleanup executors: `tinker_server/backend/sampling_cleanup_executor.py`, `tinker_server/backend/training_cleanup_executor.py`
- scheduled auth/handler path: `tinker_server/gateway_auth.py`, `tinker_server/routes/sampling.py`, `tinker_server/routes/training.py`, `tinker_server/routes/weights.py`, `tinker_server/routes/mint.py`
- shared imports used by both API and detached actors: `tinker_server/config.py`, `tinker_server/ray_utils.py`, `tinker_server/backend/ray_kill.py`, `tinker_server/backend/model_registry.py`

If none of the above changed: restart server only.

Checkpoint index rollout spans two detached actor families:
- `mint_model_runtime_*` and `mint_model_work_scheduler` handle scheduled checkpoint claim/write, so they must be recreated before fresh named checkpoint writes will emit non-null `checkpoint_record_id` and `metadata.json.ckpt_id`
- `mint_maintenance_cron` runs the periodic checkpoint mirror/publish loop, so it must also be recreated before fresh rows will drain from `checkpoint_staging` into `checkpoint_catalog`

If `TINKER_CHECKPOINT_INDEX_PG_DSN` or checkpoint publication code changes and only the first actor family is recycled, checkpoints can mirror to disk while remaining stranded in `checkpoint_staging`.

### Kill Actors

The current admin route is `POST /api/v1/actors/kill`.
Use it for ModelActorRegistry-tracked GPU actor families only:

```bash
# Kill one GPU actor family (admin only when auth is enabled)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H 'Content-Type: application/json'   -d '{"actor_type":"vllm"}' http://localhost:18000/api/v1/actors/kill
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H 'Content-Type: application/json'   -d '{"actor_type":"megatron"}' http://localhost:18000/api/v1/actors/kill
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H 'Content-Type: application/json'   -d '{"actor_type":"dense"}' http://localhost:18000/api/v1/actors/kill

# Kill all tracked GPU actors
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H 'Content-Type: application/json'   -d '{"actor_type":"all"}' http://localhost:18000/api/v1/actors/kill
```

`{"actor_type":"all"}` kills only `vllm`, `megatron`, and `dense` actors.
It does not kill detached scheduler/runtime/store actors.
It does not remove residual placement groups left behind by dead actors.
Do not treat it as a full control-plane reset.

No HTTP endpoint currently clears detached control-plane actors. Use Ray name lookup on the API host, scoped to the current namespace:

```bash
# Stop service first if you need a clean detached control-plane reset.
ssh mint-prod-volcano 'supervisorctl stop tinker-server-auth'

# Kill detached control-plane actors in current namespace and remove their named placement groups.
ssh mint-prod-volcano "TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:-tinker}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:-tinker}' python3 - <<'PY'
import os
import ray
from ray.util import get_placement_group, remove_placement_group
from ray.util.placement_group import placement_group_table

ray.init(address='auto', ignore_reinit_error=True)
ns = os.environ.get('TINKER_RAY_NAMESPACE', 'tinker')
exact_names = {
    'mint_model_work_scheduler',
    'mint_maintenance_cron',
    'mint_task_state_store',
    'tinker_startup_lease_store',
    'mint_future_replay_sweeper',
    'tinker_sampling_session_store',
    'tinker_training_session_store',
    'tinker_gateway_session_store',
    'tinker_session_heartbeat_store',
    'tinker_session_index_store',
    'tinker_sampling_cleanup_executor',
    'tinker_training_cleanup_executor',
}
prefixes = ('mint_model_runtime_',)
victims = []
for row in ray.util.list_named_actors(all_namespaces=True):
    if row.get('namespace') != ns:
        continue
    name = str(row.get('name') or '')
    if name in exact_names or any(name.startswith(p) for p in prefixes):
        victims.append(name)
for name in sorted(set(victims)):
    try:
        ray.kill(ray.get_actor(name, namespace=ns), no_restart=True)
        print(f'killed_actor={name} namespace={ns}')
    except ValueError:
        pass
    except Exception as e:
        print(f'kill_failed actor={name} namespace={ns} err={type(e).__name__}: {e}')
pg_names = {f'{name}_pg' for name in victims}
for info in placement_group_table().values():
    if info.get('state') == 'REMOVED':
        continue
    pg_name = str(info.get('name') or '')
    if pg_name not in pg_names:
        continue
    try:
        remove_placement_group(get_placement_group(pg_name))
        print(f'removed_pg={pg_name}')
    except Exception as e:
        print(f'remove_pg_failed pg={pg_name} err={type(e).__name__}: {e}')
PY"
```

Keep this cleanup namespace-scoped.
Do not touch unrelated namespaces or OpenPI-specific actor families unless the diff actually changes them.

### Restart Server

```bash
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
```

### Restart after killing vLLM

```bash
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H 'Content-Type: application/json'   -d '{"actor_type":"vllm"}' http://localhost:18000/api/v1/actors/kill
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
sleep 80 && curl -s http://localhost:18000/api/v1/healthz
```

### Restart after killing Megatron

```bash
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H 'Content-Type: application/json'   -d '{"actor_type":"megatron"}' http://localhost:18000/api/v1/actors/kill
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
curl -s http://localhost:18000/api/v1/healthz
```

### Restart after killing all tracked GPU actors

Use this after shared GPU actor code changes or when GPUs are exhausted.
Use the full detached control-plane reset above when queued-path or control-plane code changed.

```bash
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H 'Content-Type: application/json'   -d '{"actor_type":"all"}' http://localhost:18000/api/v1/actors/kill
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
sleep 80 && curl -s http://localhost:18000/api/v1/healthz
```

---

## 5. Ray Cluster

**Production join rule:**
- `mint-prod-volcano` may need the documented one-time local 0-GPU join step before `tinker-server-auth` can start cleanly.
- Do not repeat `ray start` during normal tests, inspection, or routine restarts once that join is already in place.
- For tests and connectivity checks, use `ray.init(address=...)` in Python or Ray CLI commands that connect to the head directly.

**Safe connectivity check for tests and inspection:**
```bash
ssh mint-prod-volcano "ray status --address='<RAY_HEAD_IP>:6379'"
ssh mint-prod-volcano "python3 - <<'PY'\nimport ray\nray.init(address='<RAY_HEAD_IP>:6379')\nprint(ray.cluster_resources())\nPY"
```

**Get Ray head IP from PFS:**
```bash
cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt
```

**For cluster create/teardown, invoke the `volcano-cluster` skill.**

Prod-specific values:
- Ray head config: `.claude/skills/volcano-cluster/configs/mint-prod-head.yaml`
- Ray worker config: `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml`
- Task names: include "prod" prefix

---

## 6. GPU Requirements (Production)

> **CRITICAL: ALWAYS verify cluster has enough GPUs before starting or switching model actors.**

### Current production model lineup (Volcano gateway + Aliyun 235B)

Production worker replica size: 8 GPUs (`.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml` runs `ray start --num-gpus=8`).

| Model | vLLM GPUs (inference) | Training GPUs | Total GPUs (simultaneous) | 8-GPU worker replicas (total) |
|-------|------------------------|--------------|----------------------------|-------------------------------|
| Qwen3-0.6B (Dense) | 1 | 1 | 2 | 1 |
| Qwen3-4B (Dense) | 1 | 1 | 2 | 1 |
| Qwen3-30B-A3B (MoE) | 4 | 4 | 8 | 1 |
| Qwen3-235B-A22B (MoE) | 8 | 16 | 24 | 3 |

### GPU Requirements by Model

| Model | vLLM (Inference) | Training (PEFT/Megatron) | Total (Simultaneous) |
|-------|------------------|---------------------|----------------------|
| **Qwen3-0.6B (Dense)** | TP=1 → **1 GPU** | **1 GPU** | **2 GPUs** |
| **Qwen3-4B (Dense)** | TP=1 → **1 GPU** | **1 GPU** | **2 GPUs** |
| **Qwen3-30B-A3B (MoE)** | TP=4 → **4 GPUs** | TP=4, EP=1 → **4 GPUs** | **8 GPUs** |
| **Qwen3-235B-A22B (MoE)** | TP=8 → **8 GPUs** | **16 GPUs** | **24 GPUs** |

### Pre-flight Check (MANDATORY)

```bash
# Quick status command (MANDATORY before any work)
ssh mint-prod-volcano 'python3 << "PYEOF"
import ray
ray.init(address="auto", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
gpu_avail = r.get("GPU", 0)
gpu_total = t.get("GPU", 0)
print(f"GPUs: {gpu_avail:.0f} / {gpu_total:.0f}")
# List actors by prefix (vLLM actors are named tinker_vllm_{model_name})
actors = ray.util.list_named_actors(all_namespaces=True)
for a in actors:
    name = a["name"]
    if name.startswith("tinker_vllm_") or name.startswith("megatron_"):
        print(f"{name}: ALIVE")
PYEOF'
```

---

## 7. Debugging

```bash
# Error search
ssh mint-prod-volcano "grep -i 'error\|exception\|traceback' /tmp/tinker_server_auth.log | tail -20"

# Training worker logs
ssh mint-prod-volcano "grep 'TrainingWorker' /tmp/tinker_server_auth.log | tail -20"

# Forward/backward issues
ssh mint-prod-volcano "grep 'loss_fn_inputs\|Missing' /tmp/tinker_server_auth.log | tail -10"
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Code out of sync | Re-run rsync deploy (NO `--delete`) and restart server (see Code Synchronization) |
| Symlink broken | Re-run symlink setup command |
| Server won't start | Check logs: `tail -100 /tmp/tinker_server_auth.log` |
| Auth bypass | Verify `PYTHONPATH` prioritizes `tinker-server-auth` |
| Can't connect | Check SSH tunnel (port 18000), Ray cluster connection |
| vLLM OOM | Kill vLLM actor, restart server |

---

## Common Mistakes

1. **Using `pkill` instead of `fuser -k 18000/tcp`** - may kill dev server running on same host
2. **Missing `PYTHONPATH` override** - causes auth bypass (loads pip-installed `tinker-server` without auth)
3. **Forgetting `X-API-Key` header** - all endpoints except healthz require auth
4. **Wrong port** - prod uses 18000, not 8000

---

## 8. Scripts Directory Structure

```
scripts/
├── run_server.py      # Server entry point (core)
├── tools/             # Reusable debug utilities (tracked in git)
└── wip/               # Work-in-progress investigations (gitignored)
```

**Workflow:**
- Active investigation scripts → `scripts/wip/` (not tracked)
- Scripts worth sharing/collaborating → promote to `scripts/tools/`
- Throwaway scripts → delete after use

**Do NOT** accumulate investigation scripts in `scripts/` root.
