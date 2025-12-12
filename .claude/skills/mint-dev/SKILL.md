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

## NEVER Do These (Production Belongs to mint-prod)

- **NEVER** `ssh mint-prod` - that's production
- **NEVER** use port `18000` - that's production
- **NEVER** use `volcano-tinker-auth` unison profile - that's production
- **NEVER** use `mint-prod-*.yaml` Ray configs - that's production
- **NEVER** use `tinker-server-auth` directory - that's production
- **NEVER** set `TINKER_API_KEY` or `TINKER_PORT` - not needed for dev

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
| API Key | Not required |
| Log File | `/tmp/tinker_server.log` |

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
export TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

### Start Server

```bash
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c \
  "PYTHONPATH=/root/tinker_project/tinker-server:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
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
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
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
   TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
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

## 6. Debugging

```bash
# Error search
ssh volcano "grep -i 'error\|exception\|traceback' /tmp/tinker_server.log | tail -20"

# Training worker logs
ssh volcano "grep 'TrainingWorker' /tmp/tinker_server.log | tail -20"

# Forward/backward issues
ssh volcano "grep 'loss_fn_inputs\|Missing' /tmp/tinker_server.log | tail -10"
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
