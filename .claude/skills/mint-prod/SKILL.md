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
> | SSH to server | `ssh mint-prod` (NOT `volcano`, NOT direct IP) |
> | Server logs | `ssh mint-prod "tail -50 /tmp/tinker_server_auth.log"` |
> | Health check | `curl http://localhost:18000/api/v1/healthz` |
> | Stop server | `ssh mint-prod 'fuser -k 18000/tcp'` (NOT pkill) |
> | Kill vLLM | `curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm` |
>
> If you find yourself guessing or trial-and-error debugging basic infrastructure, **STOP and re-read this skill**.

---

## NEVER Do These (Development Belongs to mint-dev)

- **NEVER** `ssh volcano` - that's development
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
| SSH Host | `mint-prod` |
| Port | 18000 |
| External URL | `https://mint.macaron.im` |
| Code Directory | `tinker-server-auth` |
| PFS Path | `/vePFS-Mindverse/share/code/tinker-server-auth` |
| Ray Configs | `.claude/skills/volcano-cluster/configs/mint-prod-head.yaml`, `.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml` |
| API Key | **Required** (`X-API-Key` header) |
| Log File | `/tmp/tinker_server_auth.log` |

**Reverse Proxy:** Port 18000 is automatically reverse-proxied by Azure Gateway at `https://mint.macaron.im`. No additional setup required.

**IMPORTANT:** All API calls (except `/api/v1/healthz` and `/`) require `X-API-Key` header.

---

## Finding the Server Process

**Always verify the actual log file location before tailing logs:**

```bash
# Find server process
ssh mint-prod 'ps aux | grep run_server | grep -v grep'

# Check where stdout goes (actual log file)
ssh mint-prod 'ls -la /proc/<PID>/fd/1'

# Example output: /proc/31501/fd/1 -> /tmp/tinker_server_auth.log
```

The log file is typically `/tmp/tinker_server_auth.log`, but verify with the above if logs seem stale.

---

## Quick Reference

```bash
# SSH tunnel
ssh -f -N -L 18000:localhost:18000 mint-prod

# Health check (no auth needed)
curl http://localhost:18000/api/v1/healthz

# Server logs
ssh mint-prod "tail -50 /tmp/tinker_server_auth.log"

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
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.claude' \
  ./ mint-prod:/vePFS-Mindverse/share/code/tinker-server-auth/

# Execute sync (remove --dry-run)
rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.claude' \
  ./ mint-prod:/vePFS-Mindverse/share/code/tinker-server-auth/
```

**Verify sync succeeded:**
```bash
# Compare specific file
ssh mint-prod "head -5 /vePFS-Mindverse/share/code/tinker-server-auth/tinker_server/backend/model_registry.py"

# Check git commit on server
ssh mint-prod "cd /vePFS-Mindverse/share/code/tinker-server-auth && git log -1 --oneline"
```

**SSH server symlink setup** (one-time):
```bash
ssh mint-prod "rm -rf /root/tinker_project/tinker-server-auth && \
  ln -s /vePFS-Mindverse/share/code/tinker-server-auth /root/tinker_project/tinker-server-auth"
```

---

## 2. Server Management

### Environment Variables

**Secrets are stored in `.secrets.env` (gitignored). Source before use:**
```bash
# Read secrets from local file
source /home/yiwen/tinker_project/tinker-server-prod/.secrets.env
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
```

**IMPORTANT:** `PYTHONPATH` must prioritize `tinker-server-auth` to override pip-installed `tinker-server`. Without this, auth middleware is bypassed.

**Note:** No default model is configured. Clients specify models per-request. Model paths are resolved via `_resolve_model_path()` in `multi_lora_engine.py`.

### Start Server

**First source secrets locally, then start server:**
```bash
# Source secrets (run locally)
source /home/yiwen/tinker_project/tinker-server-prod/.secrets.env

# Start server with secrets
ssh mint-prod "cd /root/tinker_project/tinker-server-auth && nohup env \
  PYTHONPATH=/root/tinker_project/tinker-server-auth:\$PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  HF_HOME=/vePFS-Mindverse/share/huggingface \
  PYTHONDONTWRITEBYTECODE=1 \
  TINKER_API_KEY=$TINKER_API_KEY \
  TINKER_TOKEN_SECRET_KEY=$TINKER_TOKEN_SECRET_KEY \
  TINKER_PORT=18000 \
  TINKER_CHECKPOINT_DIR=/vePFS-Mindverse/share/tinker_checkpoints \
  python scripts/run_server.py > /tmp/tinker_server_auth.log 2>&1 &"
```

### Stop Server

**Use `fuser` to kill only port 18000. Do NOT use `pkill` - it may kill dev server too.**

```bash
ssh mint-prod 'fuser -k 18000/tcp'
```

### Check Status

```bash
ssh mint-prod "ps aux | grep run_server | grep -v grep"
```

---

## 3. vLLM Actor

| Operation | Time | When to use |
|-----------|------|-------------|
| Reconnect (existing) | ~2s | Server restart, vLLM actor still alive |
| Kill + restart | ~80s | Base model changed, OOM, vLLM code changed |

### Kill vLLM Actor

```bash
# Via API (requires auth)
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

# Kill all actors (nuclear option)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_all_actors
```

---

## 4. Code Update SOP

### Decision Matrix

| Code Changed | Actors Running | Action |
|--------------|----------------|--------|
| `megatron_*.py`, `megatron_distributed.py` | Megatron alive | Kill Megatron + restart server |
| `verl_inference.py`, `multi_lora_engine.py`, `vllm_*.py` | vLLM alive | Kill vLLM + restart server |
| Routes, middleware only | Any | Restart server only |
| Any | 0 GPUs available | Kill idle actors first, free GPUs, then proceed |

### Kill Actors

See "Kill vLLM Actor" and "Kill Megatron Actor" sections above for API commands.

```bash
# Quick reference (requires auth)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_megatron
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm
```

### Legacy Reference

| Changed Code | Required Actions |
|--------------|------------------|
| `megatron_*.py`, `megatron_distributed.py` | Kill Megatron actor + restart server |
| `verl_inference.py`, `multi_lora_engine.py`, `vllm_*.py` | Kill vLLM actor + restart server |
| Route handlers, middleware, other server code | Restart server only |

### Fast Restart (no vLLM changes)

```bash
# Source secrets first
source /home/yiwen/tinker_project/tinker-server-prod/.secrets.env

ssh mint-prod 'fuser -k 18000/tcp 2>/dev/null; sleep 2'
ssh mint-prod "cd /root/tinker_project/tinker-server-auth && nohup env \
  PYTHONPATH=/root/tinker_project/tinker-server-auth:\$PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  HF_HOME=/vePFS-Mindverse/share/huggingface \
  PYTHONDONTWRITEBYTECODE=1 \
  TINKER_API_KEY=$TINKER_API_KEY \
  TINKER_TOKEN_SECRET_KEY=$TINKER_TOKEN_SECRET_KEY \
  TINKER_PORT=18000 \
  TINKER_CHECKPOINT_DIR=/vePFS-Mindverse/share/tinker_checkpoints \
  python scripts/run_server.py > /tmp/tinker_server_auth.log 2>&1 &"
```

### Full Restart (vLLM changes)

```bash
# Source secrets first
source /home/yiwen/tinker_project/tinker-server-prod/.secrets.env

# Kill vLLM (requires auth)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm

# Restart server
ssh mint-prod 'fuser -k 18000/tcp 2>/dev/null; sleep 2'
ssh mint-prod "cd /root/tinker_project/tinker-server-auth && nohup env \
  PYTHONPATH=/root/tinker_project/tinker-server-auth:\$PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  HF_HOME=/vePFS-Mindverse/share/huggingface \
  PYTHONDONTWRITEBYTECODE=1 \
  TINKER_API_KEY=$TINKER_API_KEY \
  TINKER_TOKEN_SECRET_KEY=$TINKER_TOKEN_SECRET_KEY \
  TINKER_PORT=18000 \
  TINKER_CHECKPOINT_DIR=/vePFS-Mindverse/share/tinker_checkpoints \
  python scripts/run_server.py > /tmp/tinker_server_auth.log 2>&1 &"

# Wait for vLLM init (~80s)
sleep 80 && curl -s http://localhost:18000/api/v1/healthz
```

---

## 5. Ray Cluster

**Connect SSH server to cluster:**
```bash
ssh mint-prod "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
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
ssh mint-prod 'python3 << "PYEOF"
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
ssh mint-prod "grep -i 'error\|exception\|traceback' /tmp/tinker_server_auth.log | tail -20"

# Training worker logs
ssh mint-prod "grep 'TrainingWorker' /tmp/tinker_server_auth.log | tail -20"

# Forward/backward issues
ssh mint-prod "grep 'loss_fn_inputs\|Missing' /tmp/tinker_server_auth.log | tail -10"
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
