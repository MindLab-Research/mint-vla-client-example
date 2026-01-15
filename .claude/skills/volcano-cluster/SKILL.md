---
name: volcano-cluster
description: |
  Ray cluster lifecycle management on Volcano ML platform.

  Use for: create cluster, tear down cluster, list tasks, cancel tasks, check Ray dashboard, GPU allocation, stale actor cleanup.

  Triggers: "create cluster", "tear down", "list tasks", "cancel task", "Ray dashboard", "stale actor", "GPU", "volc"

  This skill covers generic Volcano/Ray operations. For environment-specific server operations, use mint-dev or mint-prod.
---

# Volcano Cluster Management

## Quick Reference

```bash
# List all tasks
volc ml_task list --output json

# Cancel task
volc ml_task cancel --id <task_id> --output json

# View task logs (find Ray IP here)
volc ml_task logs -t <task_id> -i worker_0

# Ray dashboard
http://<RAY_HEAD_IP>:8265
```

**IMPORTANT:** Use `--output json` to avoid interactive TUI mode.

---

## 1. Instance Flavors

| Flavor | GPUs | Memory | Use Case |
|--------|------|--------|----------|
| `ml.g2a.xlarge` | 0 (CPU only) | - | Ray head node |
| `ml.hpcpni2l.28xlarge` | 8x A800 80GB | 1960 GiB | GPU workers (RDMA) |

**Worker scaling:** Adjust `RoleReplicas` in worker YAML. N replicas = 8N GPUs = 640N GB GPU memory.

## 1.1 Resource Queues

| Queue ID | Type | Use Case |
|----------|------|----------|
| `q-20251225183621-m2297` | CPU | Ray head node (CPU-only instances) |
| `q-20251126180002-26lwz` | GPU | GPU workers (A800 instances) |

**IMPORTANT:** CPU-only instances (ml.g2a.xlarge) MUST use the CPU queue. GPU instances MUST use the GPU queue.

---

## 2. Network Access

| Component | Internet | Proxy |
|-----------|----------|-------|
| SSH server | Via proxy | `localhost:1081` (HTTP), `localhost:1080` (SOCKS5) |
| Ray workers | None | N/A |

Workers must use packages pre-installed in image or via PFS PYTHONPATH.

---

## 3. PFS Directory Structure

| Path | Purpose |
|------|---------|
| `/vePFS-Mindverse/share/code/tinker-server/` | Dev code (synced via Unison) |
| `/vePFS-Mindverse/share/code/tinker-server-auth/` | Prod code (synced via Unison) |
| `/vePFS-Mindverse/share/code/vllm-0.12.0/` | vLLM package (PYTHONPATH override) |
| `/vePFS-Mindverse/share/huggingface/` | HuggingFace cache (models, tokenizers) |
| `/vePFS-Mindverse/share/models/` | Model checkpoints |
| `/vePFS-Mindverse/share/dataset/` | Training datasets |

---

## 4. Create Cluster

### Development Cluster

1. **Submit head:**
   ```bash
   volc ml_task submit -c .claude/skills/volcano-cluster/configs/mint-dev-head.yaml --output json
   ```

2. **Get head IP from logs:**
   ```bash
   volc ml_task logs -t <head_task_id> -i worker_0 | grep "Local node IP"
   ```

3. **Copy template and fill in head IP:**
   ```bash
   cp .claude/skills/volcano-cluster/configs/mint-dev-worker.yaml /tmp/mint-dev-worker.yaml
   sed -i "s/<RAY_HEAD_IP>/<actual_ip>/g" /tmp/mint-dev-worker.yaml
   ```

4. **Submit worker from temp file:**
   ```bash
   volc ml_task submit -c /tmp/mint-dev-worker.yaml --output json
   ```

5. **Connect SSH server to cluster:**
   ```bash
   ssh volcano "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
   ```

### Production Cluster

Prod head writes IP to PFS automatically. Workers read from PFS.

1. **Submit head:**
   ```bash
   volc ml_task submit -c .claude/skills/volcano-cluster/configs/mint-prod-head.yaml --output json
   ```

2. **Wait for head to write IP to PFS:**
   ```bash
   # Check PFS file (head writes on startup)
   cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt
   ```

3. **Submit worker (reads head IP from PFS):**
   ```bash
   volc ml_task submit -c .claude/skills/volcano-cluster/configs/mint-prod-worker.yaml --output json
   ```

4. **Connect SSH server to cluster:**
   ```bash
   ssh mint-prod "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
   ```

---

## 5. Tear Down Cluster

### List Tasks

```bash
volc ml_task list --output json
```

Look for task names containing your cluster identifier.

### Cancel Tasks

```bash
# Cancel single task
volc ml_task cancel --id <task_id> --output json

# Cancel multiple (run sequentially)
volc ml_task cancel --id <worker_task_id> --output json
volc ml_task cancel --id <head_task_id> --output json
```

**WARNING:** Production tasks include "prod" in names. Never cancel prod tasks for dev work.

### Disconnect SSH Server from Cluster

```bash
# Dev
ssh volcano "ray stop"

# Prod
ssh mint-prod "ray stop"
```

---

## 6. Stale Actor Cleanup

When actors become unresponsive (OOM, stuck, orphaned):

### Check GPU and Actor Status

```bash
# Dev cluster
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

# Prod cluster - use mint-prod instead of volcano
```

### Kill Actors

```bash
# Kill Megatron (dev)
curl -X POST http://localhost:8000/api/v1/kill_megatron

# Kill Megatron (prod - requires auth)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_megatron

# Kill vLLM (dev)
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Kill vLLM (prod - requires auth)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm

# Kill all actors (dev)
curl -X POST http://localhost:8000/api/v1/kill_all_actors

# Check status
curl -s http://localhost:8000/api/v1/megatron_status | jq
curl -s http://localhost:8000/api/v1/vllm_status | jq
```

### Actor Names Reference

| Actor | Name Pattern | Namespace |
|-------|--------------|-----------|
| Megatron | `megatron_{model_name}` (e.g., `megatron_kimi_k2_thinking`) | `tinker` |
| vLLM | `vllm_{model_name}` (e.g., `vllm_kimi_k2_thinking`) | `tinker` |

Model name is derived from HuggingFace model ID: lowercase, replace `-` and `.` with `_`, take last component.

### Nuclear Option

If actors cannot be killed via API:

1. Cancel all worker tasks
2. Cancel head task
3. Redeploy cluster

---

## 7. Monitoring

### Ray Dashboard

```
http://<RAY_HEAD_IP>:8265
```

Shows: actors, tasks, resources, logs, errors.

### Task Logs

```bash
volc ml_task logs -t <task_id> -i worker_0
```

### Finding Ray Head IP

**Dev cluster:**
```bash
volc ml_task logs -t <head_task_id> -i worker_0 | grep "Local node IP"
```

**Prod cluster:**
```bash
# From logs
volc ml_task logs -t <head_task_id> -i worker_0 | grep "MINT Production Ray head IP"

# From PFS (preferred)
cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt
```

---

## 8. Package Upgrades via PFS

Workers cannot install packages (no internet). To upgrade without rebuilding images:

1. **Download on SSH server** (has proxy):
   ```bash
   ssh volcano 'export http_proxy=http://localhost:1081 https_proxy=http://localhost:1081 && \
     pip download <package>==<version> --no-deps -d /tmp/wheels'
   ```

2. **Install to PFS:**
   ```bash
   ssh volcano 'pip install --target=/vePFS-Mindverse/share/code/<package>-<version> \
     /tmp/wheels/<package>-*.whl --no-deps'
   ```

3. **Set PYTHONPATH in Ray runtime_env**

### Current PFS Packages

| Package | Version | Path |
|---------|---------|------|
| PyTorch | 2.9.0 | `/vePFS-Mindverse/share/code/torch-2.9.0/` |
| vLLM | 0.12.0 | `/vePFS-Mindverse/share/code/vllm-0.12.0/` |

**PYTHONPATH order matters:** torch must come before vllm.

---

## 9. Common YAML Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ActiveDeadlineSeconds` | Max runtime (0 = unlimited) | 0 |
| `DelayExitTimeSeconds` | Keep instance alive after completion | 0 |
| `ResourceQueueID` | CPU queue for heads, GPU queue for workers | See configs |
| `RoleReplicas` | Number of instances (workers: N = 8N GPUs) | 1 |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Task stuck in Queue | Check queue capacity, try different queue |
| Worker fails to join | Verify HEAD_IP correct, check network, same image version |
| Out of memory | Reduce batch size, enable gradient checkpointing |
| Stale actors | Kill via API or tear down cluster |

---

## Config Files

| Environment | Head Config | Worker Config |
|-------------|-------------|---------------|
| Dev | `mint-dev-head.yaml` | `mint-dev-worker.yaml` |
| Prod | `mint-prod-head.yaml` | `mint-prod-worker.yaml` |

All configs in `.claude/skills/volcano-cluster/configs/`
