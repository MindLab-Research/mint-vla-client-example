---
name: mint-dev
description: |
  Development environment operations for the Mint server on Volcano cluster.

  Use for: code sync, server start/stop, vLLM management, logs - all in DEV environment.

  Triggers: "dev server", "start dev", "restart dev", "dev logs", "sync to dev", "dev vLLM"

  **Do NOT invoke this skill for production deployment. Use mint-prod instead.**

  For cluster lifecycle (create/teardown tasks), invoke the volcano-cluster skill.
---

# Mint Development Environment

> **STOP. USE THESE COMMANDS EXACTLY.**
>
> Do NOT guess SSH hosts, log locations, or process names. Everything is documented below.
>
> | Task | Command |
> |------|---------|
> | SSH to server | `ssh volcano` (NOT direct IP) |
> | Server logs | `ssh volcano "tail -50 /tmp/tinker_server.log"` |
> | Health check | `curl http://localhost:8000/api/v1/healthz` |
> | Restart server | See "Start Server" section below |
> | Kill vLLM | `curl -X POST http://localhost:8000/api/v1/kill_vllm` |
>
> If you find yourself guessing or trial-and-error debugging basic infrastructure, **STOP and re-read this skill**.

---

## NEVER Do These (Production Belongs to mint-prod)

- **NEVER** `ssh mint-prod` - that's production
- **NEVER** use port `18000` - that's production
- **NEVER** use `volcano-tinker-auth` unison profile - that's production
- **NEVER** use `mint-prod-*.yaml` Ray configs - that's production
- **NEVER** use `tinker-server-auth` directory - that's production
- **NEVER** set `TINKER_PORT` - not needed for dev (uses default 8000)

If user asks for production operations, **stop and invoke mint-prod skill instead**.

---

## Environment Config

| Property | Value |
|----------|-------|
| SSH Host | `volcano` |
| Port | 8000 |
| Code Directory | `tinker-server` |
| PFS Path | `/vePFS-Mindverse/share/code/tinker-server` |
| Unison Profile | `volcano-tinker` |
| Ray Configs | `mint-dev-head.yaml`, `mint-dev-worker.yaml` |
| API Key | Not required (auth disabled when `TINKER_API_KEY` unset) |
| Log File | `/tmp/tinker_server.log` |

---

## Finding the Server Process

**Always verify the actual log file location before tailing logs:**

```bash
# Find server process
ssh volcano 'ps aux | grep run_server | grep -v grep'

# Check where stdout goes (actual log file)
ssh volcano 'ls -la /proc/<PID>/fd/1'

# Example output: /proc/12345/fd/1 -> /tmp/tinker_server.log
```

The log file is typically `/tmp/tinker_server.log`, but verify with the above if logs seem stale.

---

## Quick Reference

```bash
# SSH tunnel
ssh -f -N -L 8000:localhost:8000 volcano

# Health check
curl http://localhost:8000/api/v1/healthz

# Server logs
ssh volcano "tail -50 /tmp/tinker_server.log"

# vLLM status
curl http://localhost:8000/api/v1/vllm_status

# Kill vLLM
curl -X POST http://localhost:8000/api/v1/kill_vllm
```

---

## 1. Code Synchronization

> **CRITICAL: ALWAYS USE DAEMON MODE (`-repeat watch`)**
>
> **NEVER** run one-off `unison volcano-tinker -batch` commands. This causes stale code on workers.

> **PRE-FLIGHT CHECK:** Before ANY dev work, verify unison daemon is running:
> ```bash
> pgrep -af "unison.*volcano-tinker" || echo "WARNING: unison not running - server has outdated code!"
> ```
> If not running, start it first: `unison volcano-tinker -repeat watch`

```bash
# Start daemon (run first, keep running)
unison volcano-tinker -repeat watch

# Check if running
pgrep -af "unison.*volcano-tinker"

# Stop daemon
pkill -f "unison.*volcano-tinker"
```

**First-time setup:**
```bash
cp .claude/skills/mint-dev/configs/volcano-tinker.prf ~/.unison/
```

**SSH server symlink setup** (one-time):
```bash
ssh volcano "rm -rf /root/tinker_project/tinker-server && \
  ln -s /vePFS-Mindverse/share/code/tinker-server /root/tinker_project/tinker-server"
```

---

## 2. Server Management

### Environment Variables

```bash
export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/tinker_project/tinker-server:$PYTHONPATH
```

**Note:** No default model is configured. Clients specify models per-request. Model paths are resolved via `_resolve_model_path()` in `multi_lora_engine.py`.

### Start Server

```bash
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

### Stop Server

```bash
ssh volcano 'pkill -f "python scripts/run_server.py"'
```

### Check Status

```bash
ssh volcano "ps aux | grep run_server | grep -v grep"
```

---

## 3. vLLM Actor

| Operation | Time | When to use |
|-----------|------|-------------|
| Reconnect (existing) | ~2s | Server restart, vLLM actor still alive |
| Kill + restart | ~80s | Base model changed, OOM, vLLM code changed |

### Kill vLLM Actor

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Via script (if server down)
ssh volcano 'cd /root/tinker_project/tinker-server && python scripts/kill_vllm.py'
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

### Kill Scripts

```bash
# Kill Megatron (frees 8 GPUs for MoE)
ssh volcano 'cd /root/tinker_project/tinker-server && python scripts/kill_megatron.py'

# Kill vLLM (frees 1-4 GPUs depending on model)
curl -X POST http://localhost:8000/api/v1/kill_vllm
# OR if server is down:
ssh volcano 'cd /root/tinker_project/tinker-server && python scripts/kill_vllm.py'
```

### Legacy Reference (do not use these names directly)

| Changed Code | Required Actions |
|--------------|------------------|
| `megatron_*.py`, `megatron_distributed.py` | Kill Megatron actor + restart server |
| `verl_inference.py`, `multi_lora_engine.py`, `vllm_*.py` | Kill vLLM actor + restart server |
| Route handlers, middleware, other server code | Restart server only |

### Fast Restart (no vLLM changes)

```bash
ssh volcano 'pkill -f "run_server" 2>/dev/null'
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

### Full Restart (vLLM changes)

```bash
# Kill vLLM
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Restart server
ssh volcano 'pkill -f "run_server" 2>/dev/null'
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'

# Wait for vLLM init (~80s)
sleep 80 && curl -s http://localhost:8000/api/v1/healthz
```

---

## 5. Ray Cluster

**Connect SSH server to cluster:**
```bash
ssh volcano "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
```

**For cluster create/teardown, invoke the `volcano-cluster` skill.**

Dev-specific values:
- Ray head config: `.claude/skills/volcano-cluster/configs/mint-dev-head.yaml`
- Ray worker config: `.claude/skills/volcano-cluster/configs/mint-dev-worker.yaml`
- Task names: no "prod" prefix

---

## 6. GPU Requirements for MoE Models

> **CRITICAL: ALWAYS verify cluster has enough GPUs before starting MoE actors.**
>
> Insufficient GPUs cause pending placement groups that block the cluster.

### GPU Requirements by Model

| Model | vLLM (Inference) | Megatron (Training) | Total (Simultaneous) |
|-------|------------------|---------------------|----------------------|
| **Qwen3-30B-A3B** | TP=1, DP=4 → **4 GPUs** | TP=4, EP=2 → **8 GPUs** | **12 GPUs** |
| **Qwen3-235B-A22B** | TP=2, DP=4 → **8 GPUs** | Not tested | **8+ GPUs** |
| Dense models (7B-14B) | **1 GPU** | **1 GPU** | **2 GPUs** |

### Pre-flight Check (MANDATORY)

Before starting any MoE test, run:

```bash
# Quick status command (MANDATORY before any work)
ssh volcano 'python3 << "PYEOF"
import ray
ray.init(address="auto", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
gpu_avail = r.get("GPU", 0)
gpu_total = t.get("GPU", 0)
print(f"GPUs: {gpu_avail:.0f} / {gpu_total:.0f}")
for name in ["persistent_megatron_worker_group_v2", "tinker_vllm_server"]:
    try:
        ray.get_actor(name, namespace="tinker")
        print(f"{name}: ALIVE")
    except ValueError:
        print(f"{name}: not running")
PYEOF'

# Check pending placement groups (should be empty)
ssh volcano "ray status 2>/dev/null | grep -A5 'Pending Demands'"
```

**Required for Qwen3-30B-A3B tests:** At least 12 available GPUs and no pending placement groups.

### Parallelism Configuration

**vLLM (Inference)** - configured in `model_registry.py`:
- TP (tensor_parallel): Shards model weights across GPUs
- DP (data_parallel): Runs multiple model replicas
- MoE uses expert parallelism: EP = TP × DP

**Megatron (Training)** - configured in `verl_training.py`:
- TP=4: Tensor parallelism (shards attention/FFN)
- EP=2: Expert parallelism (distributes MoE experts)
- world_size = TP × PP × EP × CP = 4 × 1 × 2 × 1 = 8 GPUs

### Clearing Stuck Resources

If placement groups are pending (blocking GPUs):

```bash
# Kill all Megatron actors
ssh volcano 'cd /root/tinker_project/tinker-server && python scripts/kill_megatron.py'

# Kill vLLM actor
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Verify resources freed
ssh volcano "ray status 2>/dev/null | head -20"
```

---

## 7. Debugging

```bash
# Error search
ssh volcano "grep -i 'error\|exception\|traceback' /tmp/tinker_server.log | tail -20"

# Training worker logs
ssh volcano "grep 'TrainingWorker' /tmp/tinker_server.log | tail -20"

# Forward/backward issues
ssh volcano "grep 'loss_fn_inputs\|Missing' /tmp/tinker_server.log | tail -10"
```

---

## 8. Running Test Scripts

> **CRITICAL: Test Scripts Run LOCALLY, Not on Server**
>
> Test scripts that use HTTP API (pytest, test_client.py, etc.) run on your LOCAL machine.
> Local machine has internet access for downloading tokenizers from HuggingFace Hub.
>
> **Do NOT:**
> - Run test scripts on the server (no internet for tokenizer downloads)
> - Set `HF_HUB_OFFLINE=1` or `HF_HOME=/vePFS-...` for test scripts
>
> **Server commands** (ssh volcano '...') need HF_HUB_OFFLINE because the server has no internet.
> **Test commands** run locally and download tokenizers automatically.

```bash
# Ensure SSH tunnel is active
ssh -f -N -L 8000:localhost:8000 volcano

# Run test script LOCALLY (downloads tokenizer from HuggingFace)
TINKER_BASE_URL=http://localhost:8000 python scripts/test_client.py

# Run merge gate tests LOCALLY
TINKER_BASE_URL=http://localhost:8000 python -m pytest .claude/skills/merge-gate/tests/ -v
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Unison not running | `pgrep -af "unison.*volcano-tinker"` then restart daemon |
| Symlink broken | Re-run symlink setup command |
| Server won't start | Check logs: `tail -100 /tmp/tinker_server.log` |
| Can't connect | Check SSH tunnel, Ray cluster connection |
| vLLM OOM | Kill vLLM actor, restart server |
| Pending placement groups | Not enough GPUs. Kill stale actors (see section 6) |
| MoE test hangs on startup | Check GPU availability first. Need 12 GPUs for 30B MoE |
| Tokenizer download fails | Run test script locally, not on server (server has no internet) |
