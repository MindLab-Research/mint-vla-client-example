---
name: deployment-maintenance
description: |
  Load this skill when deploying code to test server features on the Volcano GPU cluster.

  The Tinker server runs on a remote Volcano ML platform with Ray-based distributed GPU workers.
  Code syncs via Unison directly to PFS - single source of truth for both SSH server and Ray workers.

  Triggers:
  - Deploy/test: "deploy", "test on server", "run remotely", "integration test"
  - Code sync: "sync code", "unison"
  - API server: "start server", "stop server", "restart server", "server logs"
  - vLLM: "vLLM status", "kill vLLM", "vLLM OOM"
  - Ray cluster: "create cluster", "tear down cluster", "Ray dashboard"
  - Volcano: "Volcano task", "GPU allocation", "list tasks", "cancel task"
---

# Deployment & Cluster Maintenance

## Architecture

```
Local Machine                      Volcano ML Platform
─────────────                      ───────────────────
                 ┌──Unison───>  PFS: /vePFS-Mindverse/share/code/tinker-server
Code edits <─────┤<──────────                  │
                 │             ┌───────────────┴───────────────┐
                 │             ▼                               ▼
                 │      SSH Server (volcano)             Ray Workers
                 │      symlink to PFS                   direct read
                 │      API server (no GPU)              2-8 GPUs each
                 └──────joins cluster: --num-gpus=0
```

**Key constraints:**
- SSH server has NO GPUs. All GPU workloads run on Ray workers (Volcano tasks).
- **PFS is single source of truth.** Edit on local machine OR server - Unison syncs bidirectionally (prefer newer).

| Component | Location | GPUs | Code Source |
|-----------|----------|------|-------------|
| API server | SSH server | 0 | PFS (via symlink) |
| Ray head | Volcano task | 0 | - |
| Ray workers | Volcano tasks | 2-8 each | PFS (direct) |

---

## Quick Reference

**Before running client tests:** Establish SSH tunnel first.

```bash
ssh -f -N -L 8000:localhost:8000 volcano
```

| Task | Command |
|------|---------|
| Start sync daemon | `unison volcano-tinker -repeat watch` |
| Health check | `curl http://localhost:8000/api/v1/healthz` |
| Server logs | `ssh volcano "tail -50 /tmp/tinker_server.log"` |
| Stop server | `ssh volcano 'pkill -f "python scripts/run_server.py"'` |
| vLLM status | `curl http://localhost:8000/api/v1/vllm_status` |
| Kill vLLM | `curl -X POST http://localhost:8000/api/v1/kill_vllm` |
| List tasks | `volc ml_task list --output json` |
| Cancel task | `volc ml_task cancel --id <task_id> --output json` |
| Ray dashboard | `http://<RAY_HEAD_IP>:8265` |

**Finding RAY_HEAD_IP:** Check task logs for "Local node IP: 192.x.x.x":
```bash
volc ml_task logs -t <task_id> -i worker_0 | grep "Local node IP"
```

---

## 1. Code Synchronization

**IMPORTANT:** Start the Unison daemon **before** any other deployment operations. Manual one-time syncs are error-prone and should be avoided.

Unison provides **bidirectional sync** between local machine and PFS. Edit on either side - newer changes win. Both SSH server and Ray workers read from PFS.

```bash
unison volcano-tinker -repeat watch   # Start daemon (runs continuously) - DO THIS FIRST
pgrep -af "unison.*volcano-tinker"    # Check if running
pkill -f "unison.*volcano-tinker"     # Stop daemon
```

First-time setup: `cp configs/volcano-tinker.prf ~/.unison/`

Syncs everything including `.git` - full repo state on both sides.

**Why daemon mode?** The daemon watches for file changes and syncs automatically. Without it, code edits won't reach PFS until manually synced, causing stale code issues on Ray workers.

**SSH server symlink setup** (one-time):
```bash
ssh volcano "rm -rf /root/tinker_project/tinker-server && \
  ln -s /vePFS-Mindverse/share/code/tinker-server /root/tinker_project/tinker-server"
```

---

## 2. API Server

### Environment

**Offline environment** - workers have no internet. SSH server has proxy access.

```bash
export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONDONTWRITEBYTECODE=1  # Disable .pyc cache - avoids bytecode/source mismatch on PFS
export TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

### Proxy (SSH Server Only)

SSH server has internet via proxy. Workers do NOT have internet.

```bash
# HTTP proxy
export http_proxy=http://localhost:1081
export https_proxy=http://localhost:1081

# SOCKS5 proxy
export ALL_PROXY=socks5://localhost:1080
```

Use for downloading packages, models, or accessing external APIs from SSH server.

### Start Server

```bash
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/healthz` | GET | Health check |
| `/api/v1/vllm_status` | GET | vLLM actor status |
| `/api/v1/kill_vllm` | POST | Kill vLLM actor |
| `/api/v1/create_session` | POST | Create training session |
| `/api/v1/create_sampling_session` | POST | Create sampling session |
| `/api/v1/asample` | POST | Async sample request |
| `/api/v1/retrieve_future` | POST | Poll result (408=pending) |

---

## 3. Ray Cluster

### Connect SSH Server to Existing Cluster

```bash
ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0
```

### Deploy New Cluster

Prerequisites: Install Volcano CLI with `scripts/setup_volc_cli.sh`, then `volc configure`

1. **Start head node:**
   ```bash
   volc ml_task submit -c configs/ray_head.yaml --output json
   ```

2. **Get head IP from logs:**
   ```bash
   volc ml_task logs -t <head_task_id> -i worker_0 | grep "Local node IP"
   ```

3. **Create worker config:** Copy `configs/ray_worker.yaml`, replace `<RAY_HEAD_IP>` with actual IP, adjust GPU count if needed.

4. **Submit worker:**
   ```bash
   volc ml_task submit -c ray_worker_configured.yaml --output json
   ```

### GPU Flavors

| Flavor | GPUs | Memory |
|--------|------|--------|
| `ml.pni2l.7xlarge` | 2 | 490 GiB |
| `ml.pni2l.14xlarge` | 4 | 980 GiB |
| `ml.pni2l.28xlarge` | 8 | 1960 GiB |

Update both `Flavor` and `--num-gpus=N` in YAML config.

### Manage Cluster

The cluster is fully dynamic. Use `volc` CLI to add, remove, or recreate workers as needed.

**Important:** Always use `--output json` to avoid interactive TUI mode.

```bash
volc ml_task list --output json                      # List all tasks
volc ml_task submit -c <config.yaml> --output json   # Add new worker
volc ml_task cancel --id <task_id> --output json     # Remove worker or head
volc ml_task logs -t <task_id> -i worker_0           # View logs (find IP here)
```

**Common operations:**
- **Scale up:** Submit additional worker tasks pointing to existing head
- **Scale down:** Cancel worker tasks to free GPUs
- **Recreate worker:** Cancel stale worker, submit new one
- **Tear down cluster:** Cancel both head and all worker tasks

---

## 4. vLLM Actor

| Operation | Time | Method |
|-----------|------|--------|
| First start | ~80s | Automatic on first request |
| Reconnect (existing) | ~2s | Server restart reconnects to detached actor |
| Kill actor | - | `curl -X POST .../kill_vllm` |

**Kill actor + restart server when:** Base model changed, OOM, need to free GPU memory, or vLLM code changed. See Section 6 for SOP.

---

## 5. Package Upgrades via PFS

Workers cannot install packages (no internet, no pip). To upgrade packages without rebuilding images:

1. **Download on SSH server** (has proxy):
   ```bash
   ssh volcano 'export http_proxy=http://localhost:1081 https_proxy=http://localhost:1081 && \
     pip download <package>==<version> --no-deps -d /tmp/wheels'
   ```

2. **Install to PFS target directory:**
   ```bash
   ssh volcano 'pip install --target=/vePFS-Mindverse/share/code/<package>-<version> \
     /tmp/wheels/<package>-*.whl --no-deps'
   ```

3. **Set PYTHONPATH in Ray runtime_env:**
   ```python
   runtime_env = {
       "env_vars": {
           "PYTHONPATH": "/vePFS-Mindverse/share/code/<package>-<version>:$PYTHONPATH",
       }
   }
   actor = SomeActor.options(runtime_env=runtime_env).remote()
   ```

### Current PFS Packages

| Package | Version | Path | Used By |
|---------|---------|------|---------|
| vLLM | 0.12.0 | `/vePFS-Mindverse/share/code/vllm-0.12.0/` | vLLM inference actors |

### Verification

Test that workers load the correct version:
```bash
ssh volcano 'cd /vePFS-Mindverse/share/code/tinker-server && python3 scripts/test_vllm_version.py'
```

---

## 6. Code Updates (SOP)

**Both actor kill AND server restart are required** after code changes. Ray actors and the API server can retain stale bytecode cache even after code syncs to PFS.

| Changed Code | Required Actions |
|--------------|------------------|
| `megatron_*.py`, `megatron_distributed.py` | Kill Megatron actor + restart server |
| `verl_inference.py`, `multi_lora_engine.py`, `vllm_*.py` | Kill vLLM actor + restart server |
| Route handlers, middleware, other server code | Restart server only |

### Standard Procedure

**After Megatron code changes:**
```bash
# 1. Kill Megatron actor
ssh volcano 'HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
try:
    actor = ray.get_actor(\"persistent_megatron_worker_group\", namespace=\"tinker\")
    ray.kill(actor)
    print(\"Killed megatron actor\")
except ValueError:
    print(\"Megatron actor not found\")
"'

# 2. Restart API server
ssh volcano 'pkill -f "run_server" 2>/dev/null; pkill -f "uvicorn" 2>/dev/null'
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

**After vLLM code changes:**
```bash
# 1. Kill vLLM actor
curl -X POST http://localhost:8000/api/v1/kill_vllm

# 2. Restart API server
ssh volcano 'pkill -f "run_server" 2>/dev/null; pkill -f "uvicorn" 2>/dev/null'
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

**Full cleanup** (kills both actors and restarts server):
```bash
# Kill server
ssh volcano 'pkill -f "run_server" 2>/dev/null; pkill -f "uvicorn" 2>/dev/null'

# Kill both actors
ssh volcano 'HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
for name in [\"persistent_megatron_worker_group\", \"persistent_vllm_actor\"]:
    try:
        actor = ray.get_actor(name, namespace=\"tinker\")
        ray.kill(actor)
        print(f\"Killed {name}\")
    except ValueError:
        print(f\"{name} not found\")
"'

# Restart server
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

---

## Troubleshooting

### Code Sync
| Symptom | Fix |
|---------|-----|
| Unison not running | `pgrep -af "unison.*volcano-tinker"` → restart |
| Symlink broken | Re-run symlink setup command |

### Server
| Symptom | Fix |
|---------|-----|
| Won't start | `ssh volcano "tail -100 /tmp/tinker_server.log"` |
| Can't connect | Check SSH tunnel, Ray cluster |

### GPU/Memory
| Symptom | Fix |
|---------|-----|
| vLLM OOM | Kill actor, restart server |
| Worker OOM | Check dashboard, kill stale actors |
| MoE models | Need ~57 GiB - ensure workers have free memory |

### Common Mistakes

1. **Running GPU workloads on SSH server** - it has no GPUs. Use `CUDA_VISIBLE_DEVICES=`.
2. **Starting Ray head on SSH server** - Ray head is a Volcano task. Connect to existing cluster.
3. **Forgetting `--num-gpus=0`** - SSH server must join cluster with zero GPUs.

---

## Resources

- [volcano-reference.md](volcano-reference.md) - Instance flavors, Volcano CLI
- [configs/volcano-tinker.prf](configs/volcano-tinker.prf) - Unison profile (bidirectional sync to PFS)
- [configs/ray_head.yaml](configs/ray_head.yaml) - Ray head node config (ready to use)
- [configs/ray_worker.yaml](configs/ray_worker.yaml) - Ray worker template (requires head IP)
- [scripts/setup_volc_cli.sh](scripts/setup_volc_cli.sh) - Install Volcano CLI
