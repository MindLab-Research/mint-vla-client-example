# Mint Docker Images

Three independent Dockerfiles per GPU architecture, per CUDA version.
Pick the one that matches your deployment strategy — no `--target` needed,
each file is self-contained.

## Architectures

| File suffix | Target GPU | DeepEP variant | CUDA | Python |
|-------------|-----------|----------------|------|--------|
| `.sm80.*` | A800 (SM80) | SM80 patched (NVSHMEM disabled) | 13.0.3 | 3.13 |
| `.sm90.*` | H100/H800 (SM90) | base (unmodified, NVSHMEM enabled) | 13.0.3 | 3.13 |
| `.sm80.cuda129.*` | A800 (SM80) | SM80 patched | 12.9.2 (legacy) | 3.12 |

> **CUDA 13.0.3** is the default for all architectures. The `.cuda129`
> files are preserved for backward compatibility with existing deployments
> that depend on CUDA 12.9 + torch 2.9.1.

To add a new architecture, copy the three `.sm80.*` files and adjust:
- `TORCH_CUDA_ARCH_LIST` (e.g. `9.0` for SM90)
- `DEEPEP_VARIANT` (`base` for SM90, `sm80` for SM80)
- Base image if a different CUDA version is needed

## Files

```
docker/
├── Dockerfile.sm80.base           ← CUDA 13.0.3 + system tools, no Python
├── Dockerfile.sm80.overlay         ← + Python 3.13 + Ray  (RECOMMENDED)
├── Dockerfile.sm80.full            ← + torch 2.11/vLLM 0.23/DeepEP all baked in
├── Dockerfile.sm90.base           ← CUDA 13.0.3 + system tools, no Python
├── Dockerfile.sm90.overlay         ← + Python 3.13 + Ray  (RECOMMENDED)
├── Dockerfile.sm90.full            ← + torch 2.11/vLLM 0.23/DeepEP all baked in
├── Dockerfile.sm80.cuda129.base    ← CUDA 12.9.2 legacy (Python 3.12)
├── Dockerfile.sm80.cuda129.overlay ← CUDA 12.9.2 legacy (Python 3.12)
├── Dockerfile.sm80.cuda129.full    ← CUDA 12.9.2 legacy (private Volcano base)
└── README.md
```

| File | Python | Ray | torch/vLLM/DeepEP | Image size | Update speed |
|------|--------|-----|---------------------|-----------|--------------|
| `.base` | ❌ | ❌ | ❌ | ~3 GB | N/A |
| **`.overlay`** ✅ | ✅ system | ✅ system | ❌ PFS overlay | ~4 GB | Fast |
| `.full` | ✅ venv | ✅ venv | ✅ venv | ~15 GB | Slow (full rebuild) |

**Why Ray is in the overlay image (not PFS):**
Ray must be available at container start to form the cluster. If Ray
lived only in the PFS overlay, the container would need PFS mounted
before Ray could even start — a chicken-and-egg problem. By baking
Python + Ray into the image, the cluster forms instantly; heavy ML
packages (torch, vLLM, DeepEP) load from PFS via `PYTHONPATH` injection
at Ray worker launch time.

## Build

```bash
# ── SM80 (A800) ──
# Overlay (recommended)
docker build -t mint:overlay-sm80 -f docker/Dockerfile.sm80.overlay .
# Base
docker build -t mint:base-sm80 -f docker/Dockerfile.sm80.base .
# Full
docker build -t mint:full-sm80 \
    --build-arg DEEPEP_VARIANT=sm80 \
    -f docker/Dockerfile.sm80.full .

# ── SM90 (H100/H800) ──
# Overlay (recommended)
docker build -t mint:overlay-sm90 -f docker/Dockerfile.sm90.overlay .
# Base
docker build -t mint:base-sm90 -f docker/Dockerfile.sm90.base .
# Full
docker build -t mint:full-sm90 \
    --build-arg DEEPEP_VARIANT=base \
    -f docker/Dockerfile.sm90.full .

# ── SM80 CUDA 12.9 legacy ──
docker build -t mint:overlay-sm80-cuda129-py312 -f docker/Dockerfile.sm80.cuda129.overlay .
docker build -t mint:base-sm80-cuda129 -f docker/Dockerfile.sm80.cuda129.base .
docker build -t mint:full-sm80-cuda129-py312 \
    --build-arg DEEPEP_VARIANT=sm80 \
    -f docker/Dockerfile.sm80.cuda129.full .
```

### Docker tag convention

**Local build tags** (for development):

```
mint:<mode>-<arch>[-cuda<major><minor>][-py<major><minor>]
```

| Component | Values | Notes |
|-----------|--------|-------|
| `<mode>` | `base`, `overlay`, `full` | Image type |
| `<arch>` | `sm80`, `sm90` | GPU architecture |
| `-cuda<major><minor>` | `-cuda129` | Omit for current default (CUDA 13.0) |
| `-py<major><minor>` | `-py313` | Omit for current default (Python 3.13) |

**Default (CUDA 13 + Python 3.13) — omit suffixes:**

```bash
mint:overlay-sm80        # SM80 overlay, CUDA 13, Python 3.13
mint:overlay-sm90        # SM90 overlay, CUDA 13, Python 3.13
mint:full-sm80           # SM80 full, CUDA 13, Python 3.13
mint:full-sm90           # SM90 full, CUDA 13, Python 3.13
mint:base-sm80           # SM80 base, CUDA 13, no Python
mint:base-sm90           # SM90 base, CUDA 13, no Python
```

**Legacy (CUDA 12.9 + Python 3.12):**

```bash
mint:overlay-sm80-cuda129-py312
mint:full-sm80-cuda129-py312
mint:base-sm80-cuda129
```

When the default CUDA or Python version changes, existing tags should be
re-pushed with the explicit suffix to preserve traceability.

**Registry tags** (for deployment to `image-mindverse-cn-beijing.cr.volces.com/mint/mint-ray-node`):

```
<registry>/mint-ray-node:<date>-cuda<major><minor>-py<major><minor>-<arch>
<registry>/mint-ray-node:latest-<arch>
```

The date-prefixed tag is unique and self-describing; `latest-<arch>` is
a mutable pointer to the most recent push.

```bash
# Example: push SM80 overlay built on 2026-06-24
REGISTRY=image-mindverse-cn-beijing.cr.volces.com/mint/mint-ray-node
DATE=20260624

docker tag mint:overlay-sm80 ${REGISTRY}:${DATE}-cuda130-py313-sm80
docker tag mint:overlay-sm80 ${REGISTRY}:latest-sm80
docker push ${REGISTRY}:${DATE}-cuda130-py313-sm80
docker push ${REGISTRY}:latest-sm80
```

| Tag | Mutability | Purpose |
|-----|------------|---------|
| `20260624-cuda130-py313-sm80` | Immutable | Pin a specific build for rollback |
| `latest-sm80` | Mutable | Always points to the latest SM80 overlay |

### Build args

| Arg | Files | Default | Purpose |
|-----|-------|---------|---------|
| `CUDA_IMAGE` | `.base`, `.overlay` | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` | Public CUDA base |
| `BASE_IMAGE` | `.cuda129.full` | `image-mindverse-…/verl@sha256:…` | Private Volcano base (CUDA 12.9) |
| `TORCH_INDEX` | `.full` | `https://download.pytorch.org/whl/cu130` | PyTorch wheel index |
| `TORCH_VERSION` | `.full` | `2.11.0` | torch version (pinned by vLLM 0.23.0) |
| `VLLM_VERSION` | `.full` | `0.23.0` | vLLM version |
| `RAY_VERSION` | `.overlay`, `.full` | `2.51.1` | Ray version |
| `PIP_VERSION` | `.overlay`, `.full` | `26.1.2` | pip version |
| `SETUPTOOLS_VERSION` | `.overlay`, `.full` | `82.0.1` | setuptools version |
| `DEEPEP_VARIANT` | `.full` | `sm80` (SM80) / `base` (SM90) | DeepEP build variant |
| `DEEPEP_REPO` | `.full` | `https://github.com/deepseek-ai/DeepEP` | DeepEP source repo |
| `DEEPEP_BRANCH` | `.full` | `main` | DeepEP git branch |
| `NVIDIA_MODELOPT_VERSION` | `.full` | `0.44.0` | NVIDIA ModelOpt version |
| `PIP_INDEX_URL` | `.overlay` | `https://mirrors.ivolces.com/pypi/simple/` | Volcano PyPI mirror |

### Network: proxies and mirrors

If `pip install` times out during build, use the Volcano PyPI mirror
(recommended, already the default in `PIP_INDEX_URL`):

```bash
docker build --build-arg PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple/ \
    -t mint:overlay-sm80 -f docker/Dockerfile.sm80.overlay .
```

Alternatively, use an HTTP/SOCKS5 proxy:

```bash
docker build --network host \
    --build-arg HTTP_PROXY=http://192.168.4.70:17890 \
    --build-arg HTTPS_PROXY=http://192.168.4.70:17890 \
    -t mint:overlay-sm80 -f docker/Dockerfile.sm80.overlay .
```

## Overlay runtime

```bash
docker run --gpus all --shm-size=16g \
  -v /vePFS-Mindverse:/vePFS-Mindverse:ro \
  -e PFS_OVERLAY_ROOT=/vePFS-Mindverse \
  mint:overlay-sm80   # or mint:overlay-sm90
```

### PFS overlay directory structure

```
/vePFS-Mindverse/
└── runtime-env/
    ├── manifest.json          ← read by mint_server/ray/runtime_env.py
    ├── site-packages/         ← torch, vLLM, DeepEP, transformers, …
    └── src/
        ├── Megatron-LM/
        ├── Megatron-Bridge/
        ├── verl/
        └── openpi/
```

The `.overlay` image has Python 3.13 + Ray in system site-packages
(no venv). The PFS overlay only needs `site-packages/` and `src/`.

The `.full` image uses `/opt/venv` for package isolation.

The `.base` image has no Python. Use it only if you want to pull Python
from PFS too (requires `base-python/` in the overlay and a custom entrypoint).

### manifest.json format

```json
{
  "runtime_env": {
    "site_packages_dir": "site-packages",
    "source_dir": "src",
    "host_venv_dir": "host-venv",
    "base_python_dir": "base-python"
  },
  "sources": [
    {"name": "Megatron-LM", "tier": "gpu_rl", "pythonpath": ["."]},
    {"name": "Megatron-Bridge", "tier": "gpu_rl", "pythonpath": ["src", "."]},
    {"name": "verl", "tier": "gpu_rl", "pythonpath": ["."]}
  ]
}
```

### Updating dependencies (overlay mode)

```bash
# Update a package in PFS — no image rebuild
pip install --target /vePFS-Mindverse/runtime-env/site-packages \
    --no-deps new-package==1.2.3

# Ray version change — rebuild image (fast, ~2 min)
docker build --build-arg RAY_VERSION=2.52.0 \
    -t mint:overlay-sm80 -f docker/Dockerfile.sm80.overlay .
```

## Cluster deployment templates

Volcano and (future) Aliyun task YAML templates live under `docker/volc/`
and `docker/aliyun/` respectively. These are the canonical deployment
configs — the `.claude/skills/volcano-cluster/configs/` copies are kept
in sync.

```
docker/
├── volc/
│   ├── dev-head.yaml      ← dev CPU head (ml.r3i.4xlarge)
│   ├── dev-worker.yaml     ← dev GPU worker (ml.hpcpni2l.28xlarge)
│   ├── prod-head.yaml     ← prod CPU head
│   └── prod-worker.yaml   ← prod GPU worker
└── aliyun/                  ← (future)
```

Submit:
```bash
# Start head first
volc ml_task submit -c docker/volc/dev-head.yaml

# Wait for Ray head IP, then start workers
volc ml_task submit -c docker/volc/dev-worker.yaml
```

The worker template uses `TaskName: mint-dev-worker`. To launch multiple
workers, change the `TaskName` field for each submission.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PFS_OVERLAY_ROOT` | `/vePFS-Mindverse` | PFS mount root |
| `PFS_RUNTIME_ENV_ROOT` | `${PFS_OVERLAY_ROOT}/runtime-env` | Overlay env root |
| `PFS_HF_MODULES_PATH` | `${PFS_OVERLAY_ROOT}/hf-modules` | HF modules cache |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit path |
| `NCCL_IB_DISABLE` | `0` | Enable RDMA |
| `NCCL_NET_GDR_LEVEL` | `2` | GDR level |
| `NCCL_DEBUG` | `WARN` | NCCL log level |
