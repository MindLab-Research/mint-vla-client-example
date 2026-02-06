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
# Volcano CLI is pre-installed and configured on the SSH bastions:
# - Dev:  ssh mint-dev
# - Prod: ssh mint-prod-volcano
#
# Use the absolute path to avoid PATH mismatches:
#   /root/.volc/bin/volc
#
# Note: /root/.volc/bin/volc prints a version banner before JSON output.
# When parsing JSON, strip everything before the first '[' or '{'.

# Version (do NOT use "--version")
/root/.volc/bin/volc version

# List all tasks
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task list --output json'

# Cancel task
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task cancel --id <task_id>'

# View task logs (find Ray IP here)
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task logs -t <task_id> -i worker_0'

# Ray dashboard
http://<RAY_HEAD_IP>:8265
```

**IMPORTANT:** Use `--output json` to avoid interactive TUI mode.
**IMPORTANT:** `volc ml_task list --output json` emits a banner before the JSON payload.
**NOTE:** `volc ml_task cancel` does not support `--output json` on the current CLI version.

---

## 1. Instance Flavors

| Flavor | GPUs | Memory | Use Case |
|--------|------|--------|----------|
| `ml.g2a.xlarge` | 0 (CPU only) | - | Ray head node |
| `ml.hpcpni2l.28xlarge` | 8x A800 80GB | 1960 GiB | GPU workers (RDMA) |

**Worker scaling:** Adjust `RoleReplicas` in worker YAML. N replicas = 8N GPUs = 640N GB GPU memory.

## 1.1 Resource Queues

| Queue Name | Queue ID | Type | Default Use |
|------------|----------|------|-------------|
| `cpu-mindverse` | `q-20251225183621-m2297` | CPU | Ray head nodes (CPU-only) |
| `a800-mindverse-B` | `q-20260124095758-ngkg7` | GPU | Dev GPU workers (preferred) |
| `a800-mindverse-C1` | `q-20251126180002-26lwz` | GPU | Prod GPU workers (preferred) |
| `a800-mindverse-C2` | `q-20260203101340-www2h` | GPU | Prod GPU workers (fallback) |

**IMPORTANT:** CPU-only instances (ml.g2a.xlarge) MUST use the CPU queue. GPU instances MUST use a GPU queue.

## 1.2 Queue Availability (CLI)

`volc ml_task submit` supports selecting a queue via `--resource_queue_id` / `--resource_queue_name`, but this CLI does not appear to provide a "show free GPUs per queue" command (only task listing). Use the Volcano console for queue capacity (used/total) and treat the CLI as "tasks demand and state".

If you need a CLI-only estimate of current demand per queue, aggregate non-terminal tasks by `ResourceQueueId` and multiply `RoleReplicas` by GPUs-per-flavor:

```bash
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task list --output json --limit 200' | python3 - <<'PY'
import json, re, sys
s=sys.stdin.read()
m=re.search(r'[\[{]', s)
if not m:
    raise SystemExit("no JSON in volc output")
j=json.loads(s[m.start():])

flavor_gpus = {
    "ml.hpcpni2l.28xlarge": 8,
    "ml.hpcpni2l.14xlarge": 4,
    "ml.hpcpni2l.7xlarge": 2,
    "ml.g2a.xlarge": 0,
}

def task_gpus(t):
    g = 0
    for spec in t.get("TaskRoleSpecs") or []:
        reps = spec.get("RoleReplicas") or 0
        flavor = spec.get("ResourceSpecId") or (spec.get("ResourceSpec") or {}).get("FlavorID")
        g += reps * flavor_gpus.get(flavor, 0)
    return g

by_queue = {}
for t in j:
    q = t.get("ResourceQueueId") or ""
    st = t.get("Status") or ""
    if st not in ("Queue", "Staging", "Running", "Killing", "Initialized"):
        continue
    by_queue.setdefault(q, {"Running": 0, "Queue": 0, "Other": 0})
    g = task_gpus(t)
    if st == "Running":
        by_queue[q]["Running"] += g
    elif st == "Queue":
        by_queue[q]["Queue"] += g
    else:
        by_queue[q]["Other"] += g

for q, v in sorted(by_queue.items(), key=lambda kv: (-(kv[1]["Running"]+kv[1]["Queue"]), kv[0])):
    print(q, "running_gpus", v["Running"], "queued_gpus", v["Queue"], "other_gpus", v["Other"])
PY
```

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
| `/vePFS-Mindverse/share/code/vllm-0.13.0-pkg/` | vLLM package (PYTHONPATH override) |
| `/vePFS-Mindverse/share/huggingface/` | HuggingFace cache (models, tokenizers) |
| `/vePFS-Mindverse/share/models/` | Model checkpoints |
| `/vePFS-Mindverse/share/dataset/` | Training datasets |

---

## 4. Create Cluster

## 4.0 Pre-flight: identify current usage (dev vs prod)

Always measure what is already running before submitting or canceling tasks.

### List Volcano jobs (from prod bastion)

```bash
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task list --output json --limit 200' | python3 - <<'PY'
import json, re, sys
s=sys.stdin.read()
m=re.search(r'[\[{]', s)
if not m:
    raise SystemExit("no JSON in volc output")
j=json.loads(s[m.start():])
for job in j:
    name=job.get("JobName","")
    jid=job.get("JobId","")
    status=job.get("Status","")
    start=job.get("Start","")
    rq=job.get("ResourceQueueId","")
    specs=job.get("TaskRoleSpecs") or []
    reps=[]
    flavors=[]
    for spec in specs:
        reps.append(str(spec.get("RoleReplicas")))
        flavors.append(str(spec.get("ResourceSpecId")))
    print(f"{status}\t{name}\t{jid}\t{start}\t{rq}\treps={','.join(reps)}\tflavors={','.join(flavors)}")
PY
```

Conventions used by these configs:
- Prod head: `mint-prod-head`
- Prod workers: `mint-prod-worker*`
- Dev head: `ray-head`
- Dev workers: `ray-worker-*`

### List Ray GPU nodes (no Volcano CLI required)

```bash
ssh mint-prod-volcano python3 - <<'PY'
import ray
ray.init(address="auto")
nodes=[n for n in ray.nodes() if n.get("Alive")]
gpu_nodes=[n for n in nodes if (n.get("Resources") or {}).get("GPU",0)]
print("prod_gpu_nodes", len(gpu_nodes), "gpu_total", ray.cluster_resources().get("GPU"), "gpu_avail", ray.available_resources().get("GPU"))
PY

ssh mint-dev python3 - <<'PY'
import ray
ray.init(address="auto")
nodes=[n for n in ray.nodes() if n.get("Alive")]
gpu_nodes=[n for n in nodes if (n.get("Resources") or {}).get("GPU",0)]
print("dev_gpu_nodes", len(gpu_nodes), "gpu_total", ray.cluster_resources().get("GPU"), "gpu_avail", ray.available_resources().get("GPU"))
PY
```

### Development Cluster

1. **Submit head:**
   ```bash
   ssh mint-dev '/root/.volc/bin/volc ml_task submit -c /vePFS-Mindverse/share/code/tinker-server/.claude/skills/volcano-cluster/configs/mint-dev-head.yaml --output json'
   ```

2. **Get head IP from logs:**
   ```bash
   ssh mint-dev '/root/.volc/bin/volc ml_task logs -t <head_task_id> -i worker_0 | grep \"Local node IP\"'
   ```

3. **Copy template and fill in head IP + choose GPU queue:**
   ```bash
   cp .claude/skills/volcano-cluster/configs/mint-dev-worker.yaml /tmp/mint-dev-worker.yaml
   sed -i "s/<RAY_HEAD_IP>/<actual_ip>/g" /tmp/mint-dev-worker.yaml
   sed -i "s/<GPU_QUEUE_ID>/q-20260124095758-ngkg7/g" /tmp/mint-dev-worker.yaml  # prefer a800-mindverse-B for dev
   ```

4. **Submit worker from temp file:**
   ```bash
   ssh mint-dev '/root/.volc/bin/volc ml_task submit -c /tmp/mint-dev-worker.yaml --output json'
   ```

5. **Connect SSH server to cluster:**
   ```bash
   ssh mint-dev "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
   ```

### Production Cluster

Prod head writes IP to PFS automatically. Workers read from PFS.

1. **Submit head:**
   ```bash
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task submit -c /vePFS-Mindverse/share/code/tinker-server-auth/.claude/skills/volcano-cluster/configs/mint-prod-head.yaml --output json'
   ```

2. **Wait for head to write IP to PFS:**
   ```bash
   # Check PFS file (head MUST write on startup)
   ssh mint-prod-volcano 'ls -la /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt && cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt'
   ```

   If the file is missing but the Ray cluster is running, recover it from Ray and write it:

   ```bash
ssh mint-prod-volcano 'ip=$(python3 - <<\"PY\"
import ray
ray.init(address=\"auto\")
nodes=[n for n in ray.nodes() if n.get(\"Alive\")]
head=[n for n in nodes if (n.get(\"Resources\") or {}).get(\"node:__internal_head__\")]
print(head[0][\"NodeManagerAddress\"] if head else \"\")
PY
); test -n \"$ip\" && echo \"$ip\" > /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt && cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt'
   ```

3. **Copy template, choose GPU queue, submit worker (reads head IP from PFS):**
   ```bash
   # Prefer a800-mindverse-C1 or a800-mindverse-C2 for prod workers.
   ssh mint-prod-volcano 'tmp=/tmp/mint-prod-worker.yaml && \
     cp /vePFS-Mindverse/share/code/tinker-server-auth/.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml "$tmp" && \
     sed -i "s/<GPU_QUEUE_ID>/q-20251126180002-26lwz/g" "$tmp" && \
     /root/.volc/bin/volc ml_task submit -c "$tmp" --output json'
   ```

4. **Connect SSH server to cluster:**
   ```bash
ssh mint-prod-volcano "ray start --address='<RAY_HEAD_IP>:6379' --num-gpus=0"
   ```

---

## 5. Tear Down Cluster

### List Tasks

```bash
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task list --output json --limit 200'
```

Look for task names containing your cluster identifier.

### Cancel Tasks

```bash
# Cancel single task
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task cancel --id <task_id>'

# Cancel multiple (run sequentially)
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task cancel --id <worker_task_id>'
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task cancel --id <head_task_id>'
```

**WARNING:** Production tasks include "prod" in names. Never cancel prod tasks for dev work.

### Disconnect SSH Server from Cluster

```bash
# Dev
ssh mint-dev "ray stop"

# Prod
ssh mint-prod-volcano "ray stop"
```

---

## 6. Stale Actor Cleanup

When actors become unresponsive (OOM, stuck, orphaned):

### Check GPU and Actor Status

```bash
# Dev cluster
ssh mint-dev 'python3 << "PYEOF"
import os
import ray
ray.init(address="auto", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
gpu_avail = r.get("GPU", 0)
gpu_total = t.get("GPU", 0)
print(f"GPUs: {gpu_avail:.0f} / {gpu_total:.0f}")
ns = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE") or "tinker"
actors = ray.util.list_named_actors(all_namespaces=True)
for a in actors:
    if a.get("namespace") != ns:
        continue
    name = a.get("name", "")
    if (
        name.startswith("tinker_vllm_")
        or name.startswith("multinode_vllm_")
        or name.startswith("megatron_")
        or name.startswith("dense_trainer_pool_")
    ):
        print(f"{name}: ALIVE")
PYEOF'

# Prod cluster - use mint-prod-volcano instead of mint-dev
```

### Kill Actors

```bash
# Kill Megatron (dev)
curl -X POST http://localhost:8000/api/v1/kill_megatron

# Kill Megatron (prod - admin only when auth is enabled)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_megatron

# Kill vLLM (dev)
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Kill vLLM (prod - admin only when auth is enabled)
curl -X POST -H "X-API-Key: $TINKER_API_KEY" http://localhost:18000/api/v1/kill_vllm

# Kill all actors (dev; admin only when auth is enabled)
curl -X POST http://localhost:8000/api/v1/kill_all_actors

# Check status
curl -s http://localhost:8000/api/v1/megatron_status | jq
curl -s http://localhost:8000/api/v1/vllm_status | jq
```

### Actor Names Reference

| Actor | Name Pattern | Namespace |
|-------|--------------|-----------|
| Megatron | `megatron_{model_name}` (example: `megatron_kimi_k2_thinking`) | `TINKER_RAY_NAMESPACE` |
| vLLM (single-node) | `tinker_vllm_{model_part}` (example: `tinker_vllm_kimi-k2-thinking`) | `TINKER_RAY_NAMESPACE` |
| vLLM (multi-node) | `multinode_vllm_{model_part}` | `TINKER_RAY_NAMESPACE` |
| Dense trainer pool | `dense_trainer_pool_{model_part}_maxr{rank}` | `TINKER_RAY_NAMESPACE` |
| Stores | `tinker_future_store`, `tinker_training_session_store`, `tinker_gateway_session_store` | `MINT_RAY_NAMESPACE` (defaults to `TINKER_RAY_NAMESPACE` if set) |

Model name normalization differs by subsystem:
- Megatron: last component, lowercase, replace `-` and `.` with `_`.
- vLLM: last component, lowercase, replace spaces with `_` (hyphens/dots preserved).

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
ssh mint-prod-volcano '/root/.volc/bin/volc ml_task logs -t <task_id> -i worker_0'
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

# From PFS (MUST if present)
cat /vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt
```

---

## 8. Package Upgrades via PFS

Workers cannot install packages (no internet). To upgrade without rebuilding images:

1. **Download on SSH server** (has proxy):
   ```bash
   ssh mint-dev 'export http_proxy=http://localhost:1081 https_proxy=http://localhost:1081 && \
     pip download <package>==<version> --no-deps -d /tmp/wheels'
   ```

2. **Install to PFS:**
   ```bash
   ssh mint-dev 'pip install --target=/vePFS-Mindverse/share/code/<package>-<version> \
     /tmp/wheels/<package>-*.whl --no-deps'
   ```

3. **Set PYTHONPATH in Ray runtime_env**

### Current PFS Packages

| Package | Version | Path |
|---------|---------|------|
| PyTorch | 2.9.0 | `/vePFS-Mindverse/share/code/torch-2.9.0/` |
| vLLM | 0.13.0 | `/vePFS-Mindverse/share/code/vllm-0.13.0-pkg/` |

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

### Queued even though the queue has free GPUs

Sometimes a large worker submit stays in `Queue` even when the resource queue has enough free GPUs (as shown in the Volcano console). In that case, split one big task into multiple smaller tasks (fewer `RoleReplicas` per task). Example: if `RoleReplicas: 10` is queued, try `4 + 4 + 2`.

This reduces scheduler coupling: each task is an independent allocation request, so partial capacity can be used immediately instead of waiting for one big request to fit.

**Prod example (split 10 replicas into 4+4+2):**
```bash
ssh mint-prod-volcano 'set -euo pipefail
base=/vePFS-Mindverse/share/code/tinker-server-auth/.claude/skills/volcano-cluster/configs/mint-prod-worker.yaml
queue_id=q-20251126180002-26lwz  # a800-mindverse-C1 (set explicitly)

tmp4a=/tmp/mint-prod-worker-scale4a-$(date +%Y%m%d%H%M%S).yaml
cp "$base" "$tmp4a"
sed -i "s/^TaskName: \\\"mint-prod-worker\\\"/TaskName: \\\"mint-prod-worker-scale4a\\\"/" "$tmp4a"
sed -i "s/^\\s*RoleReplicas: 1/      RoleReplicas: 4/" "$tmp4a"
sed -i "s/<GPU_QUEUE_ID>/$queue_id/g" "$tmp4a"
/root/.volc/bin/volc ml_task submit -c "$tmp4a" --output json

tmp4b=/tmp/mint-prod-worker-scale4b-$(date +%Y%m%d%H%M%S).yaml
cp "$base" "$tmp4b"
sed -i "s/^TaskName: \\\"mint-prod-worker\\\"/TaskName: \\\"mint-prod-worker-scale4b\\\"/" "$tmp4b"
sed -i "s/^\\s*RoleReplicas: 1/      RoleReplicas: 4/" "$tmp4b"
sed -i "s/<GPU_QUEUE_ID>/$queue_id/g" "$tmp4b"
/root/.volc/bin/volc ml_task submit -c "$tmp4b" --output json

tmp2=/tmp/mint-prod-worker-scale2-$(date +%Y%m%d%H%M%S).yaml
cp "$base" "$tmp2"
sed -i "s/^TaskName: \\\"mint-prod-worker\\\"/TaskName: \\\"mint-prod-worker-scale2\\\"/" "$tmp2"
sed -i "s/^\\s*RoleReplicas: 1/      RoleReplicas: 2/" "$tmp2"
sed -i "s/<GPU_QUEUE_ID>/$queue_id/g" "$tmp2"
/root/.volc/bin/volc ml_task submit -c "$tmp2" --output json
'
```

| Symptom | Fix |
|---------|-----|
| Task stuck in Queue | Check queue capacity. If capacity is sufficient but still queued, split into multiple tasks with smaller `RoleReplicas` (example: `4 + 4 + 2`) and unique `TaskName`s. |
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
