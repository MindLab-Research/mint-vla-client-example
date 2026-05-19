---
name: aliyun-cluster
description: |
  DLC cluster lifecycle management on Aliyun PAI (DLC CLI).

  Use for: create/stop/list DLC jobs, fetch logs, and extract pod/IP details for multi-node jobs.

  Triggers: "aliyun cluster", "dlc", "mint-prod-aliyun", "create aliyun job", "stop aliyun job", "aliyun logs"

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Aliyun Cluster Management (PAI-DLC)

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

This skill mirrors `volcano-cluster`, but uses Aliyun PAI DLC CLI (`dlc`) instead of `volc`.

**Requirement:** `mint-prod-aliyun` is a CPU-only API host. Do not use `nvidia-smi` there. GPUs MUST be allocated to DLC jobs and consumed via Ray workers.

## Project invariants (Aliyun)

- DLC region: `cn-beijing` (endpoint `pai-dlc.cn-beijing.aliyuncs.com`)
- Worker image (GPU nodes): `acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:11`
- CPFS (Mindverse PFS) must be mounted on every pod at `/vePFS-Mindverse`
  - File system id: `bmcpfs-03001407yug37qgafv7j5`
  - Mount target: `cpfs-03001407yug37qgafv7j5-vpc-q9ctsd.cn-beijing.cpfs.aliyuncs.com`
  - File system path: `/`

## Code and environment invariants (Aliyun)

- Deploy code via **unidirectional rsync** from local to server.
- Never run any sync with `--delete` against a production path.
- Treat `.venv_cpu/` as environment state, not code. Always exclude it from code deployments.
- Exclude `.unison*` and other temp trees from deployments. Do not copy `.unison*.unison.tmp` artifacts into any live directory.

### Safe rsync pattern (Aliyun code root)

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
  --exclude='.claude' \
  ./ mint-prod-aliyun:/vePFS-Mindverse/share/code/mint-server-aliyun/
```

## Quick Reference

```bash
# SSH bastion for Aliyun deployment
ssh mint-prod-aliyun

# Ensure DLC CLI exists (remote)
ssh mint-prod-aliyun 'command -v dlc || echo "dlc missing"'

# List jobs (remote)
ssh mint-prod-aliyun 'dlc get job --page_num 1 --page_size 50'

# Stop job (remote)
ssh mint-prod-aliyun 'dlc stop job <JOB_ID> --force'

# Job details (pods + IPs)
ssh mint-prod-aliyun 'dlc get job <JOB_ID>'

# Logs for a specific pod
ssh mint-prod-aliyun 'dlc logs <JOB_ID> <POD_ID> --max_events_num 200'
```

## 0. DLC CLI Availability (mint-prod-aliyun)

`dlc` may or may not exist on first boot. If missing, run these on `mint-prod-aliyun` before doing anything else:

```bash
ssh mint-prod-aliyun 'apt-get update'
ssh mint-prod-aliyun 'apt-get install -y ca-certificates wget rsync'
ssh mint-prod-aliyun 'wget -O /usr/local/bin/dlc "https://dlc-release.oss-cn-zhangjiakou.aliyuncs.com/console/public/latest/dlc?spm=a2c4g.11186623.0.0.5b9c722e2xGL3x"'
ssh mint-prod-aliyun 'chmod +x /usr/local/bin/dlc'
```

If TLS/CA errors persist, reinstall certificates:
```bash
ssh mint-prod-aliyun 'apt-get update && apt-get install -y ca-certificates'
```

## 1. Authentication (required once per user)

### Credentials source (mint-prod-aliyun)

For this project, credentials are stored in the repo-root `.env`:
- `ALI_ACCESSKEY_ID`
- `ALI_ACCESSKEY_SECRET`

Workflow:
```bash
cd /path/to/mint-server-prod
set -a
source .env
set +a
```

Then run `dlc` with explicit auth flags (do not paste credentials into logs or committed files):
```bash
dlc -I -a "$ALI_ACCESSKEY_ID" -k "$ALI_ACCESSKEY_SECRET" -r cn-beijing -e pai-dlc.cn-beijing.aliyuncs.com <COMMAND...>
```

Note: do not rely on `~/.dlc/config` in this environment. Always pass `-I -a -k -r -e`.

## 2. Create Cluster (Ray: mint-prod-aliyun head + 3 DLC workers, 24 GPUs)

Cluster layout used for `mint-prod-aliyun` (REQUIRED):
- Ray head MUST run on the `mint-prod-aliyun` API host (CPU-only: 4 CPU)
- Ray workers MUST be 3 DLC jobs (8 GPUs each) = 24 GPUs total

Reason (observed behavior): the `mint-prod-aliyun` API host cannot open TCP connections to DLC pod IPs on port 6379, so running a DLC-hosted Ray head makes the API server unable to connect to GCS (`ray.init(address=<pod_ip>:6379)` times out). Running the Ray head locally on `mint-prod-aliyun` avoids inbound-to-pod networking entirely (workers connect outbound to the head).

### CPFS mount (required, must be read-write)

In this environment, mounting CPFS via `--data_sources` results in a read-only mount (pods show `/vePFS-Mindverse ... (ro,...)`) even when passing `{readOnly:false}`.

Use `--data_source_uris` with the BMCPFS URI format instead. This produces a read-write mount (pods show `/vePFS-Mindverse ... (rw,...)`).

```bash
--data_source_uris "bmcpfs://bmcpfs-03001407yug37qgafv7j5.cn-beijing/::/vePFS-Mindverse"
```

CPFS details:
- File system id: `bmcpfs-03001407yug37qgafv7j5`
- Mount target (control-plane): `cpfs-03001407yug37qgafv7j5-vpc-q9ctsd.cn-beijing.cpfs.aliyuncs.com`
- Mount root: `/` (contains `/share`, `/user`, etc)

### Dedicated resource group (prepaid) note

When submitting into a dedicated resource group (`--resource_id ...`), the backend rejects `*_spec`/EcsSpec parameters.
Use the per-role ResourceConfig flags instead (CPU/GPU/memory/gpu_type), for example:
- `--master_cpu ... --master_gpu 0 --master_memory ...`
- `--worker_gpu 8 --worker_gpu_type <SM90_GPU_TYPE> --worker_cpu ... --worker_memory ...`

If jobs are stuck in `Queuing` while GPUs exist, check whether CPU/memory is the limiting dimension. You can keep GPU requirements fixed and reduce CPU/memory to fit the remaining capacity, e.g. lower `--worker_cpu` and/or `--worker_memory` (and `--worker_shared_memory` if set).

Two distinct knobs (do not confuse them):
- DLC scheduling (whether the pod can start): controlled by `dlc submit ... --master_cpu/--master_memory/--master_gpu`.
- Ray scheduling (how many CPUs Ray thinks exist on the node): controlled by `RAY_NUM_CPUS` in `ray_entrypoint.sh`.

### Variables (copy/paste once)

Run on your workstation:
```bash
export DLC_REGION=cn-beijing
export DLC_ENDPOINT=pai-dlc.cn-beijing.aliyuncs.com
export DLC_WORKSPACE_ID=341495
export DLC_RESOURCE_ID=quotatbifdwjdnhp
export DLC_DATA_SOURCE_URIS='bmcpfs://bmcpfs-03001407yug37qgafv7j5.cn-beijing/::/vePFS-Mindverse'
export DLC_IMAGE='acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:11'

# Code root on CPFS (workers run entrypoint from here)
export ALIYUN_CODE_ROOT=/vePFS-Mindverse/share/code/mint-server-aliyun
export ALIYUN_RAY_ENTRYPOINT="${ALIYUN_CODE_ROOT}/.claude/skills/aliyun-cluster/scripts/ray_entrypoint.sh"
```

### Ray entrypoint (no CPFS writes)

`${ALIYUN_RAY_ENTRYPOINT}` uses:
- `MINT_RAY_ROLE=head|worker` (MUST set; overrides DLC-injected `RANK`/`PAI_TASK_ROLE`)
- `HEAD_IP` or `MASTER_ADDR` for worker-to-head join

This avoids writing the head IP into CPFS (DLC BMCPFS mounts are read-only by default in this environment).

### 2.1 Start Ray head on `mint-prod-aliyun` (CPU-only)

Run on `mint-prod-aliyun`:
```bash
ssh mint-prod-aliyun 'cd /vePFS-Mindverse/share/code/mint-server-aliyun && \
  nohup bash -c "PYTHONPATH=$PWD:$PYTHONPATH RAY_START_HEAD=1 RAY_HEAD_PORT=6379 RAY_HEAD_NUM_CPUS=4 \
    .venv_cpu/bin/python .claude/skills/aliyun-cluster/scripts/start_local_ray_head.py" \
  >> /tmp/ray_head_aliyun.log 2>&1 &'
```

Verify:
```bash
ssh mint-prod-aliyun 'tail -20 /tmp/ray_head_aliyun.log'
# Expect: RAY_NODE_IP 10.11.26.94 (or similar VPC IP)
ssh mint-prod-aliyun 'pgrep -af raylet || true'
```

### 2.2 Submit 3 worker jobs (8 GPUs each, connect to mint-prod-aliyun head)

Run on `mint-prod-aliyun` (repeat 3 times; each job gets 8 GPUs):
```bash
ssh mint-prod-aliyun "dlc submit pytorchjob \
  --name=ray-worker-235b-1 \
  --workspace_id=${DLC_WORKSPACE_ID} \
  --resource_id=${DLC_RESOURCE_ID} \
  --data_source_uris \"${DLC_DATA_SOURCE_URIS}\" \
  --masters 1 --workers 0 \
  --master_image \"${DLC_IMAGE}\" \
  --master_cpu 160 --master_memory 1600Gi --master_gpu 8 \
  --envs MINT_RAY_ROLE=worker,HEAD_IP=<MINT_PROD_ALIYUN_RAY_HEAD_IP>,RAY_NUM_GPUS=8 \
  --command \"bash ${ALIYUN_RAY_ENTRYPOINT}\""
```

Verify each worker joins the head:
```bash
ssh mint-prod-aliyun 'dlc logs <WORKER_JOB_ID> <POD_ID> --max_events_num 200'
# Look for: RAY_WORKER_IP <ip> head <MINT_PROD_ALIYUN_RAY_HEAD_IP> num_gpus 8
```

### 2.3 Verify Ray sees 24 GPUs

Run from `mint-prod-aliyun`:
```bash
ssh mint-prod-aliyun "cd /vePFS-Mindverse/share/code/mint-server-aliyun && \
  RAY_ADDRESS=<MINT_PROD_ALIYUN_RAY_HEAD_IP>:6379 PYTHONPATH=$PWD .venv_cpu/bin/python - <<'PY'
import os
import ray

ray.init(address=os.environ['RAY_ADDRESS'])
print(ray.cluster_resources())
PY"
```

## 3. GPU/Parallelism Notes

Hardware differences matter for vLLM TP/PP/DP and overall GPU count:
- Volcano: A800 80GB
- Aliyun: H (SM90)

For mint-server deployments on Aliyun, tune `mint_server/backend/model_registry.py` or set `MINT_MODEL_CONFIG_OVERRIDES_JSON` on the Aliyun server.

Current 235B defaults in `mint_server/backend/model_registry.py` are tuned for 24xSM90-class Aliyun GPUs with concurrent prewarm:
- Inference: `inference_tp=8`, `inference_dp=1` (8 GPUs)
- Training: `train_tp=4`, `train_pp=2`, `train_ep=2` (16 GPUs)

## 3.1 API Host: Ray head + server start (mint-prod-aliyun)

`mint-prod-aliyun` is a CPU-only API host.

In this environment, run the Ray head locally on `mint-prod-aliyun` (section 2.1) and run `mint-server` as a Ray driver against that local head.

Invariants:
- Do not use `nvidia-smi` here.
- Do not use `/opt/venv` on the API host. `/opt/venv` exists inside DLC worker images.
- Do not restart the local Ray head while actors exist; that SIGTERMs workers and breaks `save_weights_for_sampler`.

### Start mint-server on mint-prod-aliyun

Key env vars:
- `RAY_ADDRESS=<MINT_PROD_ALIYUN_RAY_HEAD_IP>:6379`
- `MINT_SUPPORTED_MODELS=Qwen/Qwen3-235B-A22B-Instruct-2507`
- `MINT_PERSISTENT_MODELS=Qwen/Qwen3-235B-A22B-Instruct-2507`
- `MINT_CHECKPOINT_DIR=/vePFS-Mindverse/share/mint_checkpoints`
- `MINT_CODE_ROOT=/vePFS-Mindverse/share/code/mint-server-aliyun` (so Ray worker `runtime_env` points at the correct code root)
- Do not shadow in-image vLLM with an incomplete CPFS checkout (missing `vllm._C`).

Start (uses project-owned `.venv_cpu`; logs to `/tmp/mint_server_auth.log`):
```bash
ssh mint-prod-aliyun 'cd /vePFS-Mindverse/share/code/mint-server-aliyun && \
  set -a && source .secrets.env && set +a && \
  pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true; \
  nohup bash -c "PYTHONPATH=$PWD:$PYTHONPATH HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
    PYTHONDONTWRITEBYTECODE=1 RAY_ADDRESS=<MINT_PROD_ALIYUN_RAY_HEAD_IP>:6379 MINT_PORT=18000 \
    MINT_SUPPORTED_MODELS=Qwen/Qwen3-235B-A22B-Instruct-2507 \
    MINT_PERSISTENT_MODELS=Qwen/Qwen3-235B-A22B-Instruct-2507 \
    MINT_PERSISTENT_TRAIN_LORA_RANK=16 MINT_PERSISTENT_TRAIN_LR=5e-5 \
    .venv_cpu/bin/python scripts/run_server.py" >> /tmp/mint_server_auth.log 2>&1 &'
```

Sanity:
```bash
ssh mint-prod-aliyun 'curl -s http://localhost:18000/api/v1/healthz'
```

## 4. Extra Python deps for Megatron prewarm (modelopt)

Megatron-Bridge imports `modelopt` (`import modelopt.torch.distill as mtd`). If the DLC image lacks it, install into CPFS and add via `PFS_EXTRA_PYTHONPATH`.

Install `modelopt` into CPFS and add it to `PFS_EXTRA_PYTHONPATH` for Ray worker `runtime_env`.

Target directory:
```bash
/vePFS-Mindverse/share/code/modelopt-pkg
```

## 5. transformer_engine verification (required for Megatron)

Prefer using the DLC image's built-in `transformer_engine` (expected present in `mint:11`). If you add `transformer_engine` via `PFS_EXTRA_PYTHONPATH`, verify it is a complete install (Python extension modules present).

Run this on a DLC GPU worker pod:
```bash
python -c 'import transformer_engine.pytorch as te; print(te.__file__)'
```

If it fails with `StopIteration` inside `transformer_engine/pytorch/__init__.py`, your `transformer_engine` directory is missing the `transformer_engine.pytorch` extension `*.so` files. Do not proceed until the import works on a worker.

Install on `mint-prod-aliyun`:
```bash
ssh mint-prod-aliyun 'set -euo pipefail
target=/vePFS-Mindverse/share/code/modelopt-pkg
rm -rf "$target" && mkdir -p "$target"
py=/vePFS-Mindverse/share/code/mint-server-aliyun/.venv/bin/python

# WARNING: do NOT install nvidia-modelopt with dependencies; it will pull a full torch+CUDA wheel set.
$py -m pip install -q --target "$target" --no-deps nvidia-modelopt==0.41.0
$py -m pip install -q --target "$target" ninja packaging nvidia-ml-py rich tqdm pulp regex safetensors pydantic
'
```

Sanity check on a GPU worker (forces execution on a GPU pod):
```bash
ssh mint-prod-aliyun 'RAY_ADDRESS=<HEAD_IP>:6379 /vePFS-Mindverse/share/code/mint-server-aliyun/.venv/bin/python - <<'"'"'PY'"'"'
import os, ray
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)

@ray.remote(num_gpus=1)
def probe():
    import sys
    sys.path.insert(0, "/vePFS-Mindverse/share/code/modelopt-pkg")
    import modelopt
    import torch
    return {"modelopt": modelopt.__version__, "torch": torch.__version__}

print(ray.get(probe.remote()))
PY'
'
```

## 6. Command Manual

Read `.claude/skills/aliyun-cluster/references/dlc_cli_manual.md` for the consolidated `dlc` CLI usage manual (commands, flags, and examples).
