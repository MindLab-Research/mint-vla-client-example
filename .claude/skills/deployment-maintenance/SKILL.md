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

## Environments

| Property | Development | Production |
|----------|-------------|------------|
| SSH Host | `volcano` | `mint-prod` |
| Port | 8000 | 18000 |
| Code Directory | `tinker-server` | `tinker-server-auth` |
| PFS Path | `/vePFS-Mindverse/share/code/tinker-server` | `/vePFS-Mindverse/share/code/tinker-server-auth` |
| Unison Profile | `volcano-tinker` | `volcano-tinker-auth` |
| Ray Configs | `ray_head.yaml`, `ray_worker.yaml` | `mint-prod-head.yaml`, `mint-prod-worker.yaml` |
| API Key | Not required | Required (`X-API-Key` header) |
| Log File | `/tmp/tinker_server.log` | `/tmp/tinker_server_auth.log` |

**IMPORTANT:** Production Ray tasks include "prod" in names. Never cancel or modify prod tasks for dev work.

---

## Architecture

```
Local Machine                      Volcano ML Platform
-------------                      -------------------
                 +--Unison--->  PFS: /vePFS-Mindverse/share/code/tinker-server[-auth]
Code edits <-----+<----------                  |
                 |             +---------------+---------------+
                 |             v                               v
                 |      SSH Server                       Ray Workers
                 |      symlink to PFS                   direct read
                 |      API server (no GPU)              2-8 GPUs each
                 +------joins cluster: --num-gpus=0
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

## Quick Reference - Development

**SSH tunnel:**
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

---

## Quick Reference - Production

**SSH tunnel:**
```bash
ssh -f -N -L 18000:localhost:18000 mint-prod
```

| Task | Command |
|------|---------|
| Start sync daemon | `unison volcano-tinker-auth -repeat watch` |
| Health check | `curl http://localhost:18000/api/v1/healthz` |
| Server logs | `ssh mint-prod "tail -50 /tmp/tinker_server_auth.log"` |
| Stop server | `ssh mint-prod 'fuser -k 18000/tcp'` |
| vLLM status | `curl -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/vllm_status` |
| Kill vLLM | `curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm` |

**Production API Key:** Set `TINKER_API_KEY` environment variable. All endpoints except `/api/v1/healthz` and `/` require the `X-API-Key` header.

---

## Common Operations

### List/Cancel Volcano Tasks

```bash
volc ml_task list --output json                      # List all tasks
volc ml_task cancel --id <task_id> --output json     # Cancel task
volc ml_task logs -t <task_id> -i worker_0           # View logs
```

**Ray dashboard:** `http://<RAY_HEAD_IP>:8265`

### Finding RAY_HEAD_IP

```bash
# Dev cluster
volc ml_task logs -t <head_task_id> -i worker_0 | grep "Local node IP"

# Prod cluster
volc ml_task logs -t <head_task_id> -i worker_0 | grep "MINT Production Ray head IP"
# Or read from PFS:
cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt
```

---

## 1. Code Synchronization

> **CRITICAL: ALWAYS USE DAEMON MODE (`-repeat watch`)**
>
> **NEVER** run one-off `unison <profile> -batch` commands. This causes:
> - Stale code on workers (edits not synced)
> - Race conditions during deployment
> - Mysterious "code not updated" bugs
>
> **ALWAYS** start the daemon first and let it run continuously.

```bash
# Development - START THIS FIRST
unison volcano-tinker -repeat watch

# Production - START THIS FIRST
unison volcano-tinker-auth -repeat watch
```

Check/stop daemon:
```bash
pgrep -af "unison.*volcano-tinker"    # Check if running
pkill -f "unison.*volcano-tinker"     # Stop daemon
```

**Verify daemon is running before any deployment operation.**

First-time setup:
```bash
cp configs/volcano-tinker.prf ~/.unison/       # Dev
cp configs/volcano-tinker-auth.prf ~/.unison/  # Prod
```

**SSH server symlink setup** (one-time):
```bash
# Development
ssh volcano "rm -rf /root/tinker_project/tinker-server && \
  ln -s /vePFS-Mindverse/share/code/tinker-server /root/tinker_project/tinker-server"

# Production
ssh mint-prod "rm -rf /root/tinker_project/tinker-server-auth && \
  ln -s /vePFS-Mindverse/share/code/tinker-server-auth /root/tinker_project/tinker-server-auth"
```

---

## 2. API Server

### Environment Variables

**Offline environment** - workers have no internet.

```bash
export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONDONTWRITEBYTECODE=1
export TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

### Start Development Server

```bash
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

### Start Production Server

**IMPORTANT:** Production requires `PYTHONPATH` override because `tinker-server` is pip-installed. Without this, Python loads the wrong package.

```bash
ssh mint-prod 'cd /root/tinker_project/tinker-server-auth && nohup env \
  PYTHONPATH=/root/tinker_project/tinker-server-auth:$PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  HF_HOME=/vePFS-Mindverse/share/huggingface \
  PYTHONDONTWRITEBYTECODE=1 \
  TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  TINKER_API_KEY=<API_KEY> \
  TINKER_PORT=18000 \
  python scripts/run_server.py > /tmp/tinker_server_auth.log 2>&1 &'
```

### Stop Server

```bash
# Development - can use pkill
ssh volcano 'pkill -f "python scripts/run_server.py"'

# Production - kill only port 18000 to avoid affecting dev server
ssh mint-prod 'fuser -k 18000/tcp'
```

### API Endpoints

| Endpoint | Method | Auth Required | Purpose |
|----------|--------|---------------|---------|
| `/api/v1/healthz` | GET | No | Health check |
| `/api/v1/vllm_status` | GET | Prod only | vLLM actor status |
| `/api/v1/kill_vllm` | POST | Prod only | Kill vLLM actor |
| `/api/v1/create_session` | POST | Prod only | Create training session |
| `/api/v1/create_sampling_session` | POST | Prod only | Create sampling session |
| `/api/v1/asample` | POST | Prod only | Async sample request |
| `/api/v1/retrieve_future` | POST | Prod only | Poll result (408=pending) |

---

## 3. Ray Cluster

### Development Cluster

**Connect SSH server to existing cluster:**
```bash
ssh volcano "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
```

**Deploy new cluster:**
1. Submit head: `volc ml_task submit -c configs/ray_head.yaml --output json`
2. Get head IP from logs
3. Edit `ray_worker.yaml` with head IP
4. Submit worker: `volc ml_task submit -c ray_worker.yaml --output json`

### Production Cluster

**Ray configs:** `mint-prod-head.yaml`, `mint-prod-worker.yaml`

Production head writes IP to PFS automatically. Workers read from PFS.

**Deploy new prod cluster:**
1. Submit head:
   ```bash
   volc ml_task submit -c configs/mint-prod-head.yaml --output json
   ```
2. Wait for head to start and write IP to PFS:
   ```bash
   cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt
   ```
3. Submit worker (reads head IP from PFS automatically):
   ```bash
   volc ml_task submit -c configs/mint-prod-worker.yaml --output json
   ```

**Connect SSH server to prod cluster:**
```bash
ssh mint-prod "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
```

### GPU Flavors

| Flavor | GPUs | Memory |
|--------|------|--------|
| `ml.pni2l.7xlarge` | 2 | 490 GiB |
| `ml.pni2l.14xlarge` | 4 | 980 GiB |
| `ml.pni2l.28xlarge` | 8 | 1960 GiB |

Update both `Flavor` and `--num-gpus=N` in YAML config.

---

## 4. vLLM Actor

| Operation | Time | Method |
|-----------|------|--------|
| First start | ~80s | Automatic on first request |
| Reconnect (existing) | ~2s | Server restart reconnects to detached actor |
| Kill actor | - | POST to `/kill_vllm` |

**Kill actor + restart server when:** Base model changed, OOM, need to free GPU memory, or vLLM code changed.

---

## 5. Code Updates (SOP)

**Both actor kill AND server restart are required** after code changes.

| Changed Code | Required Actions |
|--------------|------------------|
| `megatron_*.py`, `megatron_distributed.py` | Kill Megatron actor + restart server |
| `verl_inference.py`, `multi_lora_engine.py`, `vllm_*.py` | Kill vLLM actor + restart server |
| Route handlers, middleware, other server code | Restart server only |

### Development Code Update

```bash
# Kill vLLM
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Restart server
ssh volcano 'pkill -f "run_server" 2>/dev/null'
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
   python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

### Production Code Update

```bash
# Kill vLLM (requires API key)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm

# Restart server (kill only port 18000)
ssh mint-prod 'fuser -k 18000/tcp 2>/dev/null; sleep 2'
ssh mint-prod 'cd /root/tinker_project/tinker-server-auth && nohup env \
  PYTHONPATH=/root/tinker_project/tinker-server-auth:$PYTHONPATH \
  HF_HUB_OFFLINE=1 \
  HF_HOME=/vePFS-Mindverse/share/huggingface \
  PYTHONDONTWRITEBYTECODE=1 \
  TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  TINKER_API_KEY=<API_KEY> \
  TINKER_PORT=18000 \
  python scripts/run_server.py > /tmp/tinker_server_auth.log 2>&1 &'
```

---

## 6. Package Upgrades via PFS

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

3. **Set PYTHONPATH in Ray runtime_env**

### Current PFS Packages

| Package | Version | Path |
|---------|---------|------|
| vLLM | 0.12.0 | `/vePFS-Mindverse/share/code/vllm-0.12.0/` |

---

## Troubleshooting

### Code Sync
| Symptom | Fix |
|---------|-----|
| Unison not running | `pgrep -af "unison.*volcano-tinker"` then restart |
| Symlink broken | Re-run symlink setup command |

### Server
| Symptom | Fix |
|---------|-----|
| Won't start | Check logs: `tail -100 /tmp/tinker_server[_auth].log` |
| Can't connect | Check SSH tunnel, Ray cluster |
| Auth bypass (prod) | Verify `PYTHONPATH` is set to prioritize tinker-server-auth |

### GPU/Memory
| Symptom | Fix |
|---------|-----|
| vLLM OOM | Kill actor, restart server |
| Worker OOM | Check dashboard, kill stale actors |

### Common Mistakes

1. **Running GPU workloads on SSH server** - it has no GPUs.
2. **Modifying prod tasks for dev work** - use separate dev cluster.
3. **Missing PYTHONPATH in prod** - causes auth bypass (loads wrong package).
4. **Using pkill for prod server** - may kill dev server. Use `fuser -k 18000/tcp`.

---

## Resources

- [volcano-reference.md](volcano-reference.md) - Instance flavors, Volcano CLI
- **Development configs:**
  - [configs/volcano-tinker.prf](configs/volcano-tinker.prf) - Unison profile
  - [configs/ray_head.yaml](configs/ray_head.yaml) - Ray head
  - [configs/ray_worker.yaml](configs/ray_worker.yaml) - Ray worker
- **Production configs:**
  - [configs/volcano-tinker-auth.prf](configs/volcano-tinker-auth.prf) - Unison profile
  - [configs/mint-prod-head.yaml](configs/mint-prod-head.yaml) - Ray head
  - [configs/mint-prod-worker.yaml](configs/mint-prod-worker.yaml) - Ray worker
- [scripts/setup_volc_cli.sh](scripts/setup_volc_cli.sh) - Install Volcano CLI
