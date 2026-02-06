---
name: mint-prod
description: |
  Production environment operations for the Mint server on Volcano cluster.

  Use for: code sync, server start/stop, vLLM management, logs - all in PROD environment.

  Triggers: "prod server", "start prod", "restart prod", "prod logs", "sync to prod", "prod vLLM", "production"

  **Do NOT invoke this skill for development work. Use mint-dev instead.**

  For cluster lifecycle (create/teardown tasks), invoke the volcano-cluster skill.
---

# Mint Production Environment

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
> | Kill vLLM | `curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm` |
>
> If you find yourself guessing or trial-and-error debugging basic infrastructure, **STOP and re-read this skill**.

---

## NEVER Do These (Development Belongs to mint-dev)

- **NEVER** `ssh mint-dev` - that's development
- **NEVER** use port `8000` - that's development
- **NEVER** use unison for production sync - use rsync (unidirectional)
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
| External URL | `https://mint.macaron.im` |
| Code Directory | `tinker-server-auth` |
| PFS Path | `/vePFS-Mindverse/share/code/tinker-server-auth` |
| Ray Configs | `.claude/skills/volcano-cluster/configs/mint-prod-head.yaml`, `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml` |
| Prod GPU Queues | Prefer `a800-mindverse-C1` (`q-20251126180002-26lwz`) or `a800-mindverse-C2` (`q-20260203101340-www2h`). Do not hard-code; confirm availability or ask user before submitting workers. |
| API Key | **Required** (`X-API-Key` header) |
| Log File | `/tmp/tinker_server_auth.log` |

**Reverse Proxy:** Port 18000 is automatically reverse-proxied by Azure Gateway at `https://mint.macaron.im`. No additional setup required.

**IMPORTANT:** All API calls (except `/api/v1/healthz` and `/`) require `X-API-Key` header.

---

**Worker queue selection:** `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml` uses a `<GPU_QUEUE_ID>` placeholder. Set it explicitly before submitting any new prod worker tasks.

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
curl -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/vllm_status

# Kill vLLM (auth required)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm
```

---

## 1. Code Synchronization

> **CRITICAL: USE RSYNC FOR PRODUCTION DEPLOYMENT**
>
> Production uses **unidirectional rsync** from local to server. This ensures:
> - Local code is the source of truth
> - No accidental overwrites from server
> - Explicit deployment step (not background sync)

```bash
# From the tinker-server-prod directory:

# Sync local code to production server (dry-run first)
rsync -avz --dry-run --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='.secrets.env' \
  --exclude='LOG.md' \
  --exclude='PROMPT.md' \
  --exclude='ray_head_ip.txt' \
  --exclude='.claude' \
  ./ mint-prod-volcano:/vePFS-Mindverse/share/code/tinker-server-auth/

# Execute sync (remove --dry-run)
rsync -avz --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='.secrets.env' \
  --exclude='LOG.md' \
  --exclude='PROMPT.md' \
  --exclude='ray_head_ip.txt' \
  --exclude='.claude' \
  ./ mint-prod-volcano:/vePFS-Mindverse/share/code/tinker-server-auth/
```

**Verify sync succeeded:**
```bash
# Compare specific file
ssh mint-prod-volcano "head -5 /vePFS-Mindverse/share/code/tinker-server-auth/tinker_server/backend/model_registry.py"

# Check git commit on server
ssh mint-prod-volcano "cd /vePFS-Mindverse/share/code/tinker-server-auth && git log -1 --oneline"
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
# From repo root: copy local `.secrets.env` to prod and restart
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
export MINT_PERSISTENT_MODELS="Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-30B-A3B-Instruct-2507,moonshotai/Kimi-K2-Thinking"
export MINT_PERSISTENT_TRAIN_LORA_RANK=16
export MINT_PERSISTENT_TRAIN_LR=5e-5
export MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S=3600
```

### Multi-target Model Routing (Gateway)

Prod can run as a gateway/router that forwards selected base models to other tinker-server deployments.

Deployment targets (current plan):
- `mint-prod-volcano` (this server): `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-30B-A3B-Instruct-2507`, `moonshotai/Kimi-K2-Thinking`
- `mint-prod-aliyun`: `Qwen/Qwen3-235B-A22B-Instruct-2507`

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
- GPU types: Volcano uses A800 80GB; Aliyun uses L20X 140GB.
- Tune per-model TP/EP/CP and vLLM memory caps in `tinker_server/backend/model_registry.py`.
- For environment-specific tuning without code changes, set `MINT_MODEL_CONFIG_OVERRIDES_JSON`:
  ```bash
  export MINT_MODEL_CONFIG_OVERRIDES_JSON='{"Qwen/Qwen3-235B-A22B-Instruct-2507":{"inference_tp":16}}'
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

## 3. vLLM Actor

| Operation | Time | When to use |
|-----------|------|-------------|
| Reconnect (existing) | ~2s | Server restart, vLLM actor still alive |
| Kill + restart | ~80s | Base model changed, OOM, vLLM code changed |

### Kill vLLM Actor

```bash
# Via API (admin only when auth is enabled)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm

# Kill specific model's vLLM actor
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:18000/api/v1/kill_vllm
```

### Kill Megatron Actor

```bash
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_megatron

# Kill specific model's Megatron actor
curl -X POST -H "X-API-Key: $TINKER_API_KEY" -H "Content-Type: application/json" \
  -d '{"base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:18000/api/v1/kill_megatron
```

### Check Actor Status

```bash
# vLLM status
curl -s -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/vllm_status | jq

# Megatron status
curl -s -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/megatron_status | jq

# Kill all actors (nuclear option; admin only when auth is enabled)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_all_actors
```

---

## 4. Code Update SOP

### Rule 0: server restart does not reload detached actors

The API server is a Python process. Ray actors are separate Python processes.

Detached actors (vLLM, Megatron, DenseTrainerPool, stores) survive server restarts and keep running old code until killed.

Hard rule: never create/get/kill actors outside `TINKER_RAY_NAMESPACE` unless the user explicitly requests cross-namespace action.

### Kill criteria after code changes

Always restart the server after code changes.

Kill detached actors only if the change can be imported/executed inside that actor process:
- vLLM: `tinker_server/backend/verl_inference.py`, `tinker_server/backend/multi_lora_engine.py`, `tinker_server/backend/multinode_inference.py`, `tinker_server/backend/vllm_*.py`
- Megatron: `tinker_server/backend/megatron_distributed.py`, `tinker_server/backend/megatron_training.py`, `tinker_server/backend/verl_patches.py`
- Dense training pool: `tinker_server/backend/verl_training.py`
- Detached stores: `tinker_server/backend/future_store.py`, `tinker_server/backend/training_session_store.py`, `tinker_server/backend/gateway_session_store.py`
- Shared (kills required for all GPU actor types): `tinker_server/config.py`, `tinker_server/ray_utils.py`, `tinker_server/backend/ray_kill.py`, `tinker_server/backend/model_registry.py`

If none of the above changed: restart server only.

### Kill Actors

Preferred: use the HTTP endpoints documented above (auth required) for vLLM and Megatron.

```bash
# Quick reference (admin only when auth is enabled)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_megatron
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm
```

No dedicated endpoints exist for dense trainer pool actors or detached store actors. Use Ray name lookup on the API host:

```bash
# Kill dense trainer pool actors in current namespace (prefix match)
ssh mint-prod-volcano "TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:-tinker}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:-tinker}' python3 -c \"
import os
import ray
ray.init(address='auto', ignore_reinit_error=True)
ns = os.environ.get('TINKER_RAY_NAMESPACE', 'tinker')
actors = ray.util.list_named_actors(all_namespaces=True)
killed = 0
for a in actors:
    if a.get('namespace') != ns:
        continue
    name = a.get('name') or ''
    if not name.startswith('dense_trainer_pool_'):
        continue
    try:
        ray.kill(ray.get_actor(name, namespace=ns))
        killed += 1
    except Exception as e:
        print(f\"kill_failed name={name!r} namespace={ns!r} err={e!r}\")
print(f\"killed={killed} prefix='dense_trainer_pool_' namespace={ns}\")
\""

# Kill detached store actors in current namespace (name match)
ssh mint-prod-volcano "TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:-tinker}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:-tinker}' python3 -c \"
import os
import ray
ray.init(address='auto', ignore_reinit_error=True)
ns = os.environ.get('TINKER_RAY_NAMESPACE', 'tinker')
names = ['tinker_future_store', 'tinker_training_session_store', 'tinker_gateway_session_store']
killed = 0
for name in names:
    try:
        ray.kill(ray.get_actor(name, namespace=ns))
        killed += 1
    except ValueError:
        pass
    except Exception as e:
        print(f\"kill_failed name={name!r} namespace={ns!r} err={e!r}\")
print(f\"killed={killed} kind='stores' namespace={ns}\")
\""
```

### Restart Server

```bash
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
```

### Restart after killing vLLM

```bash
# Kill vLLM (admin only when auth is enabled)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm

# Restart server
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'

# Wait for vLLM init (~80s)
sleep 80 && curl -s http://localhost:18000/api/v1/healthz
```

### Restart after killing Megatron

```bash
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_megatron
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
curl -s http://localhost:18000/api/v1/healthz
```

### Restart after killing all tracked GPU actors

Use this after shared actor code changes (for example `tinker_server/backend/model_registry.py`) or when GPUs are exhausted.

Note: `/api/v1/kill_all_actors` kills ResourcePool-tracked GPU actors (vLLM, Megatron, dense trainer pool). It does not kill detached store actors.

```bash
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_all_actors
ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'
sleep 80 && curl -s http://localhost:18000/api/v1/healthz
```

---

## 5. Ray Cluster

**Connect SSH server to cluster:**
```bash
ssh mint-prod-volcano "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
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

### Official supported model lineup (0.6B, 4B, 30B, 235B)

Production worker replica size: 8 GPUs (`.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml` runs `ray start --num-gpus=8`).

| Model | vLLM GPUs (inference) | Training GPUs | Total GPUs (simultaneous) | 8-GPU worker replicas (total) |
|-------|------------------------|--------------|----------------------------|-------------------------------|
| Qwen3-0.6B (Dense) | 1 | 1 | 2 | 1 |
| Qwen3-4B (Dense) | 1 | 1 | 2 | 1 |
| Qwen3-30B-A3B (MoE) | 4 | 4 | 8 | 1 |
| Qwen3-235B-A22B (MoE) | 16 | 32 | 48 | 6 |

**Full production lineup (all four models resident):** 60 GPUs total, so 8 worker replicas (64 GPUs) plus 1 head node.

### GPU Requirements by Model

| Model | vLLM (Inference) | Training (PEFT/Megatron) | Total (Simultaneous) |
|-------|------------------|---------------------|----------------------|
| **Qwen3-0.6B (Dense)** | TP=1 → **1 GPU** | **1 GPU** | **2 GPUs** |
| **Qwen3-4B (Dense)** | TP=1 → **1 GPU** | **1 GPU** | **2 GPUs** |
| **Qwen3-30B-A3B (MoE)** | TP=4 → **4 GPUs** | TP=4, EP=1 → **4 GPUs** | **8 GPUs** |
| **Qwen3-235B-A22B (MoE)** | TP=16 → **16 GPUs** | TP=4, EP=8 → **32 GPUs** | **48 GPUs** |

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
| Code out of sync | Run rsync from local to server (see Code Synchronization) |
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
