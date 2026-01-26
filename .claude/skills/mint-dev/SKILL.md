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

> **CRITICAL: RESTART SERVER AFTER CODE CHANGES**
>
> Python servers do NOT hot-reload. After ANY code change:
> 1. Verify code synced: `ssh volcano 'grep "your_change" /path/to/file'`
> 2. **RESTART SERVER** (see section 2 below)
> 3. Verify new process: `ssh volcano 'ps aux | grep run_server'`
>
> **Server running old code = your fix does not exist.** This has wasted hours of debugging.

---

## NEVER Do These (Production Belongs to mint-prod)

- **NEVER** `ssh mint-prod` - that's production
- **NEVER** use port `18000` - that's production
- **NEVER** use `volcano-tinker-auth` unison profile - that's production
- **NEVER** use `mint-prod-*.yaml` Ray configs - that's production
- **NEVER** use `tinker-server-auth` directory - that's production
- **NEVER** sync to `/vePFS-Mindverse/share/code/tinker-server` (shared; causes devs to clobber each other)
- **NEVER** set `TINKER_PORT` - not needed for dev (uses default 8000)

If user asks for production operations, **stop and invoke mint-prod skill instead**.

---

## Environment Config

| Property | Value |
|----------|-------|
| SSH Host | `volcano` |
| Port | 8000 |
| Code Directory | `tinker-server` |
| PFS Path | Required: `/vePFS-Mindverse/share/code/$USER/tinker-server` |
| Unison Profile | Required: `volcano-tinker-$USER` |
| Ray Configs | `mint-dev-head.yaml`, `mint-dev-worker.yaml` |
| Dev GPU Queue | `q-20260124095758-ngkg7` (24 GPUs total) |
| API Key | Not required (auth disabled when `TINKER_API_KEY` unset) |
| Log File | `/tmp/tinker_server.log` |

---

## Concurrent Dev Runs (Issue #83)

Goal: isolate code + detached Ray actor state across developers sharing the same dev Ray cluster.

Required env vars:
- `TINKER_RAY_NAMESPACE`: Ray namespace for all server-owned actors (default `tinker`)
- `PFS_TINKER_PATH`: PFS code root used in Ray worker `runtime_env` `PYTHONPATH`

### Unison Profile (Per-Developer)

Create a per-developer profile (no shared PFS root):

```bash
mkdir -p ~/.unison
sed "s/__PFS_USER__/$USER/g" .claude/skills/mint-dev/configs/volcano-tinker.prf > ~/.unison/volcano-tinker-$USER.prf

# Start unison in background (explicit nohup)
nohup unison volcano-tinker-$USER -repeat watch > /tmp/unison-volcano-tinker-$USER.log 2>&1 &
```

### Volcano Symlink (Per-Developer)

Point the server working tree at the same per-developer PFS directory:

```bash
ssh volcano "rm -rf /root/tinker_project/tinker-server && \
  ln -s /vePFS-Mindverse/share/code/$USER/tinker-server /root/tinker_project/tinker-server"
```

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
> **NEVER** run one-off `unison volcano-tinker-$USER -batch` commands. This causes stale code on workers.

> **PRE-FLIGHT CHECK:** Before ANY dev work, verify unison daemon is running:
> ```bash
> pgrep -af "unison.*volcano-tinker-$USER" || echo "WARNING: unison not running - server has outdated code!"
> ```
> If not running, start it first (explicit nohup): `nohup unison volcano-tinker-$USER -repeat watch > /tmp/unison-volcano-tinker-$USER.log 2>&1 &`

```bash
# Start daemon (run first, keep running)
nohup unison volcano-tinker-$USER -repeat watch > /tmp/unison-volcano-tinker-$USER.log 2>&1 &

# Check if running
pgrep -af "unison.*volcano-tinker-$USER"

# Stop daemon
pkill -f "[u]nison.*volcano-tinker-$USER" || true
```

**First-time setup:**
```bash
mkdir -p ~/.unison
sed "s/__PFS_USER__/$USER/g" .claude/skills/mint-dev/configs/volcano-tinker.prf > ~/.unison/volcano-tinker-$USER.prf
```

**SSH server symlink setup** (one-time):
```bash
ssh volcano "rm -rf /root/tinker_project/tinker-server && \
  ln -s /vePFS-Mindverse/share/code/$USER/tinker-server /root/tinker_project/tinker-server"
```

---

## 2. Server Management

### Environment Variables

```bash
export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/tinker_project/tinker-server:$PYTHONPATH
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server
# For concurrent dev runs, set this to a unique value (example: tinker_$USER).
# export TINKER_RAY_NAMESPACE=tinker

# MoE vLLM placement mode:
# - Default: MINT_MOE_MULTINODE_MIN_GPUS=4, so Qwen3-30B (TP=4) uses MultiNodeInferenceEngine
#   (Ray distributed executor, can spread TP across nodes; slower but schedules under GPU fragmentation).
# - Set to 16 to force single-node MultiLoRAInferenceEngine for Qwen3-30B (requires 4 GPUs on one node).
# export MINT_MOE_MULTINODE_MIN_GPUS=16
```

**Note:** No default model is configured. Clients specify models per-request. Model paths are resolved via `_resolve_model_path()` in `multi_lora_engine.py`.

### Start Server

```bash
ssh volcano "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   TINKER_RAY_NAMESPACE=tinker_$USER \
   python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
```

### Stop Server

```bash
ssh volcano 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'

# If multiple server processes remain, force kill:
ssh volcano 'pkill -9 -f "python scripts/run_server.py" 2>/dev/null || true'

# Verify:
ssh volcano 'ps aux | grep run_server | grep -v grep'
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
# Via API (preferred)
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Kill specific model's vLLM actor
curl -X POST -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:8000/api/v1/kill_vllm
```

### Kill Megatron Actor

```bash
curl -X POST http://localhost:8000/api/v1/kill_megatron

# Kill specific model's Megatron actor
curl -X POST -H "Content-Type: application/json" \
  -d '{"base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:8000/api/v1/kill_megatron
```

### Check Actor Status

```bash
# vLLM status
curl -s http://localhost:8000/api/v1/vllm_status | jq

# Megatron status
curl -s http://localhost:8000/api/v1/megatron_status | jq

# Kill all actors (nuclear option)
curl -X POST http://localhost:8000/api/v1/kill_all_actors
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

> **Actor naming convention:**
> - vLLM: `tinker_vllm_{model_name}` (e.g., `tinker_vllm_kimi-k2-thinking`)
> - Megatron: `megatron_{model_name}` (e.g., `megatron_kimi-k2-thinking`)
> - Namespace: `TINKER_RAY_NAMESPACE` (default `tinker`)
>
> **When to kill actors:**
> - Implementation code changed (actors cache old code)
> - OOM or stuck state
> - Switching to different model

```bash
# Kill vLLM actor for K2
ssh volcano 'python3 -c "
import os
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
try:
    ns = os.environ.get(\"TINKER_RAY_NAMESPACE\", \"tinker\")
    actor = ray.get_actor(\"tinker_vllm_kimi-k2-thinking\", namespace=ns)
    ray.kill(actor)
    print(\"Killed vLLM actor\")
except ValueError as e:
    print(f\"Actor not found: {e}\")
"'

# Kill Megatron actor for K2
ssh volcano 'python3 -c "
import os
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
try:
    ns = os.environ.get(\"TINKER_RAY_NAMESPACE\", \"tinker\")
    actor = ray.get_actor(\"megatron_kimi-k2-thinking\", namespace=ns)
    ray.kill(actor)
    print(\"Killed Megatron actor\")
except ValueError as e:
    print(f\"Actor not found: {e}\")
"'

# List all actors in tinker namespace (to find actor names)
ssh volcano 'python3 -c "
import os
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
actors = ray.util.list_named_actors(all_namespaces=True)
for a in actors:
    if os.environ.get(\"TINKER_RAY_NAMESPACE\", \"tinker\") in str(a):
        print(a)
"'
```

### Legacy Reference (do not use these names directly)

| Changed Code | Required Actions |
|--------------|------------------|
| `megatron_*.py`, `megatron_distributed.py` | Kill Megatron actor + restart server |
| `verl_inference.py`, `multi_lora_engine.py`, `vllm_*.py` | Kill vLLM actor + restart server |
| Route handlers, middleware, other server code | Restart server only |

### Fast Restart (no vLLM changes)

```bash
ssh volcano 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh volcano "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
```

### Full Restart (vLLM changes)

```bash
# Kill vLLM
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Restart server
ssh volcano 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh volcano "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"

# Wait for vLLM init (~80s)
sleep 80 && curl -s http://localhost:8000/api/v1/healthz
```

---

## 5. Ray Cluster

**Find Ray head task (if task ID unknown):**
```bash
volc ml_task list --output json | jq '.[] | select(.Name | startswith("ray-head")) | {Id, Name, Status}'
```

**Get Ray head IP from task logs:**
```bash
volc ml_task logs -t <head_task_id> -i worker_0 | grep "Local node IP"
```

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
| **Qwen3-30B-A3B** | TP=4 → **4 GPUs** | TP=4, EP=1 → **4 GPUs** | **8 GPUs** |
| **Moonlight-16B-A3B** | TP=8 → **8 GPUs** | TP=8, EP=8 → **8 GPUs** | **16 GPUs** |
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
# List actors by prefix (vLLM actors are named tinker_vllm_{model_name})
actors = ray.util.list_named_actors(all_namespaces=True)
for a in actors:
    name = a["name"]
    if name.startswith("tinker_vllm_") or name.startswith("megatron_"):
        print(f"{name}: ALIVE")
PYEOF'

# Check pending placement groups (should be empty)
ssh volcano "ray status 2>/dev/null | grep -A5 'Pending Demands'"
```

**Required for Qwen3-30B-A3B tests:** At least 8 available GPUs and no pending placement groups.

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
# Kill Megatron actor (see "Kill Actors" section above for commands)
# Kill vLLM actor (see "Kill Actors" section above for commands)

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
> Test scripts that use HTTP API (pytest, `scripts/tools/smoke.py service`, etc.) run on your LOCAL machine.
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
# CRITICAL: Always set TINKER_TELEMETRY=0 to prevent log flooding
TINKER_BASE_URL=http://localhost:8000 TINKER_TELEMETRY=0 python scripts/tools/smoke.py service

# Run merge gate tests LOCALLY
TINKER_BASE_URL=http://localhost:8000 TINKER_TELEMETRY=0 python -m pytest .claude/skills/merge-gate/tests/ -v

# For training scripts (e.g., tinker_cookbook)
TINKER_BASE_URL=http://localhost:8000 TINKER_TELEMETRY=0 python -m tinker_cookbook.recipes.math_rl.train ...
```

---

## 9. Ray Actor Status and Logs

> **CRITICAL: NEVER assume actor state without verifying.**
>
> A failed `ray.get_actor()` lookup could mean: wrong name, wrong namespace, or actor actually dead.
> **ALWAYS list actors first** to see what exists before concluding anything.

### List All Actors (DO THIS FIRST)

```bash
# List all actors - this shows actual names and states
ssh volcano 'ray list actors 2>&1 | grep -E "(vllm|megatron|Extended)" | head -20'

# Or list with full details
ssh volcano 'ray list actors --filter "state=ALIVE" 2>&1 | head -30'
```

### Check Specific Actor Status

```bash
# WRONG: Guessing actor name and concluding "DEAD" if not found
# RIGHT: List first, then check with exact name from list

ssh volcano 'python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)

# List actors first to get exact names
actors = ray.util.list_named_actors(all_namespaces=True)
print(\"Named actors:\")
for a in actors:
    print(f\"  {a}\")
"'
```

### Get Actor Logs

```bash
# Get actor ID from ray list actors output, then:
ssh volcano 'ray logs actor --id <ACTOR_ID> --tail 100 2>&1'

# Example with actual ID:
ssh volcano 'ray logs actor --id 618fd2b45b4f8ac797dafdbd1e000000 --tail 100 2>&1'
```

### List Dead Actors (for crash investigation)

```bash
ssh volcano 'ray list actors --filter "state=DEAD" 2>&1 | head -30'
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Unison not running | `pgrep -af "unison.*volcano-tinker-$USER"` then restart daemon |
| Symlink broken | Re-run symlink setup command |
| Server won't start | Check logs: `tail -100 /tmp/tinker_server.log` |
| Can't connect | Check SSH tunnel, Ray cluster connection |
| vLLM OOM | Kill vLLM actor, restart server |
| Pending placement groups | Not enough GPUs. Kill stale actors (see section 6) |
| MoE test hangs on startup | Check GPU availability first. Need 12 GPUs for 30B MoE |
| Tokenizer download fails | Run test script locally, not on server (server has no internet) |
| Actor lookup fails | **LIST actors first** (`ray list actors`), don't assume dead |

---

## 10. Scripts Directory Structure

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
