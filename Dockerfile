# syntax=docker/dockerfile:1.6
#
# Mint runtime image (`mint:16-*`), used by:
# - Ray head/worker tasks on Volcano/Aliyun (GPU worker image)
# - (Optionally) an API server container, but in our deployments the API server / Ray driver
#   typically runs on the host using a dedicated venv (see deployment_scales.md).
#
# This Dockerfile must not assume the build context contains mint-server code.
# (The platform build environment may only provide this Dockerfile.)

# Base image from the private Volcano registry with:
# - torch 2.9.1+cu129
# - DeepEP preinstalled (import names: `deep_ep`, `deep_ep_cpp`)
# - vLLM 0.16.x preinstalled
#
# Pin by digest for reproducibility.
ARG BASE_IMAGE=image-mindverse-cn-beijing.cr.volces.com/namespace-mindverse/verl@sha256:a4599b0ebbf8fed7fb469886d90ecf0b6be9b36e55ef31e4df329c7b2c1922c6
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# System deps (keep minimal; avoid compiling large deps in this image).
RUN apt-get update && apt-get install -y --no-install-recommends \
  aria2 \
  ca-certificates \
  curl \
  git \
  build-essential \
  ninja-build \
  python3.12-venv \
  ibverbs-providers \
  ibverbs-utils \
  libibverbs-dev \
  && rm -rf /var/lib/apt/lists/*

# Virtualenv + tooling.
# Use `--system-site-packages` so we can import torch/vLLM/DeepEP from the base image.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ARG PIP_VERSION=24.2
ARG SETUPTOOLS_VERSION=79.0.0
RUN python -m pip install --no-cache-dir --upgrade \
  "pip==${PIP_VERSION}" \
  "setuptools==${SETUPTOOLS_VERSION}" \
  wheel

# Ensure torch shared libs are visible to CUDA extensions (flash-attn, transformer_engine, etc.).
# Prefer the system-site-packages torch path from the base image.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Verify torch + DeepEP + vLLM from the base image.
# NOTE: DeepEP's compiled extension is ABI-sensitive to torch; keep them consistent.
RUN python -c "import torch; import deep_ep; import deep_ep_cpp; import vllm; print('torch.__version__:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('deep_ep:', deep_ep.__file__); print('vllm:', getattr(vllm, '__version__', 'unknown')); assert torch.__version__.startswith('2.9.'), f\"expected torch 2.9.*, got {torch.__version__}\"; assert (torch.version.cuda or '').startswith('12.9'), f\"expected CUDA 12.9.*, got {torch.version.cuda}\""

# DeepEP is SM90-centric upstream; for A800 (SM80) we need a build with SM90 features disabled.
# Build selection:
# - base: use DeepEP from the base image
# - sm80: rebuild DeepEP in this image (DISABLE_SM90_FEATURES=1, TORCH_CUDA_ARCH_LIST=8.0),
#         force-disable NVSHMEM kernels, and avoid a hard crash when RDMA hints are requested.
ARG DEEPEP_VARIANT=base
ARG DEEPEP_SOURCE_DIR=/home/dpsk_a2a/DeepEP
RUN set -euo pipefail; \
  if [ "${DEEPEP_VARIANT}" = "base" ]; then \
    echo "DeepEP variant: base"; \
  elif [ "${DEEPEP_VARIANT}" = "sm80" ]; then \
    echo "DeepEP variant: sm80"; \
    test -d "${DEEPEP_SOURCE_DIR}"; \
    rm -rf /tmp/DeepEP; \
    cp -a "${DEEPEP_SOURCE_DIR}" /tmp/DeepEP; \
    python -c $'from __future__ import annotations\n\nimport os\nfrom pathlib import Path\n\nsetup_py = Path(\"/tmp/DeepEP/setup.py\")\nif not setup_py.exists():\n    raise SystemExit(f\"DeepEP setup.py not found at {setup_py}\")\nsetup_txt = setup_py.read_text(encoding=\"utf-8\")\n\nif \"Patched: force-disable NVSHMEM\" not in setup_txt:\n    lines = setup_txt.splitlines(True)\n    insert_at = None\n    for i, line in enumerate(lines):\n        if line.strip() == \"# NVSHMEM flags\":\n            insert_at = i\n            break\n    if insert_at is None:\n        raise SystemExit(\"DeepEP setup.py anchor not found: # NVSHMEM flags\")\n\n    nl = chr(10)\n    env_key = \"DISABLE_SM90_FEATURES\"\n    inject_lines = [\n        \"    # Patched: force-disable NVSHMEM when DISABLE_SM90_FEATURES=1 (SM80 build)\" + nl,\n        f\"    if int(os.getenv({env_key!r}, 0)):\" + nl,\n        \"        disable_nvshmem = True\" + nl,\n        nl,\n    ]\n    lines[insert_at:insert_at] = inject_lines\n    setup_py.write_text(\"\".join(lines), encoding=\"utf-8\")\n\ncfg = Path(\"/tmp/DeepEP/csrc/config.hpp\")\nif not cfg.exists():\n    raise SystemExit(f\"DeepEP config.hpp not found at {cfg}\")\ncfg_txt = cfg.read_text(encoding=\"utf-8\")\n\ncfg_lines = cfg_txt.splitlines(True)\nmatch_idxs = [i for i, line in enumerate(cfg_lines) if \"NVSHMEM is disable during compilation\" in line]\nif len(match_idxs) != 1:\n    raise SystemExit(\"DeepEP config.hpp needle not found for NVSHMEM-disabled RDMA hint\")\nidx = match_idxs[0]\nindent = cfg_lines[idx][: (len(cfg_lines[idx]) - len(cfg_lines[idx].lstrip()))]\ncfg_lines[idx] = indent + \"return 0;\" + chr(10)\ncfg.write_text(\"\".join(cfg_lines), encoding=\"utf-8\")\n'; \
    DISABLE_SM90_FEATURES=1 TORCH_CUDA_ARCH_LIST=8.0 python -m pip install --no-cache-dir --no-deps --no-build-isolation --force-reinstall /tmp/DeepEP; \
    python -c "import deep_ep_cpp; print('deep_ep_cpp:', getattr(deep_ep_cpp, '__file__', None)); print('deep_ep_cpp.is_sm90_compiled:', deep_ep_cpp.is_sm90_compiled()); assert deep_ep_cpp.is_sm90_compiled() is False"; \
    rm -rf /tmp/DeepEP; \
  else \
    echo "ERROR: unknown DEEPEP_VARIANT='${DEEPEP_VARIANT}' (expected 'base' or 'sm80')" >&2; \
    exit 1; \
  fi

# Core runtime deps used by the API server and Ray driver processes.
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    "ray[default]==2.51.1" \
    "fastapi[standard]==0.121.2" \
    "uvicorn[standard]==0.38.0" \
    "pydantic==2.12.4" \
    "httpx==0.27.2" \
    "structlog>=25.5.0" \
    "opentelemetry-api>=1.39.1" \
    "opentelemetry-sdk>=1.39.1" \
    "opentelemetry-exporter-otlp>=1.39.1" \
    "transformers==4.57.0" \
    "accelerate==1.11.0" \
    "einops" \
    "onnxscript" \
    "codetiming==1.4.0" \
    "torchdata==0.11.0" \
    "datasets==4.4.2" \
    "tensordict" \
    "peft==0.18.0"

# NVIDIA ModelOpt (import name: modelopt).
ARG NVIDIA_PYPI_INDEX=https://pypi.nvidia.com
ARG NVIDIA_MODELOPT_VERSION=0.41.0
RUN python -m pip install --no-cache-dir \
  --index-url "${NVIDIA_PYPI_INDEX}" \
  --extra-index-url https://pypi.org/simple \
  --retries 10 \
  --timeout 60 \
  --only-binary=:all: \
  "nvidia-modelopt==${NVIDIA_MODELOPT_VERSION}"

# `pybind11` is used by some training stacks and must come from a prebuilt wheel.
RUN python -m pip install --no-cache-dir --only-binary=:all: \
  --retries 10 --timeout 60 \
  "pybind11"

# OmegaConf requires antlr4-python3-runtime==4.9.* (and later runtimes are not compatible).
# PyPI does not publish 4.9.* wheels, so install from sdist (reproducible source build, no runtime copies).
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps "omegaconf==2.3.0"
RUN python -m pip install --no-cache-dir --no-build-isolation "antlr4-python3-runtime==4.9.3"
RUN python -c "import antlr4; print('antlr4:', antlr4.__file__); import omegaconf; print('omegaconf:', omegaconf.__version__)"

# Do not bake mutable source trees into the image. The supported runtime contract
# is: image owns ABI-bound packages, while `PFS_RUNTIME_ENV_ROOT` owns pinned
# source trees (`Megatron-LM`, `Megatron-Bridge`, `verl`) plus shared Python deps.

# Default command is intentionally minimal; Ray task YAMLs and server ops override this.
RUN python -c "import onnxscript, modelopt; print('onnxscript', getattr(onnxscript, '__version__', 'unknown')); print('modelopt', getattr(modelopt, '__version__', 'unknown'))"
RUN python -c "import torch, deep_ep_cpp, vllm; import ray; print('torch:', torch.__version__); print('ray:', ray.__version__); print('vllm:', getattr(vllm, '__version__', 'unknown'))"
CMD ["bash", "-lc", "python -c 'import torch, deep_ep_cpp, vllm; print(torch.__version__, torch.version.cuda); print(getattr(vllm, \"__version__\", \"unknown\"))' && sleep infinity"]
