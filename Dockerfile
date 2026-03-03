# syntax=docker/dockerfile:1.6
#
# Mint runtime image (replacement for `mint:4`), used by:
# - API server container
# - Ray head/worker tasks
#
# This Dockerfile must not assume the build context contains tinker-server code.
# (The platform build environment may only provide this Dockerfile.)

ARG CUDA_IMAGE=nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# System deps: match volcano host baseline (Ubuntu 22.04) + build deps for python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    aria2 \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-dev \
    python3-venv \
    build-essential \
    pkg-config \
    ninja-build \
    ibverbs-providers \
    ibverbs-utils \
    libibverbs-dev \
  && rm -rf /var/lib/apt/lists/*

# Virtualenv + tooling.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ARG PIP_VERSION=24.2
ARG SETUPTOOLS_VERSION=79.0.0
RUN python -m pip install --no-cache-dir --upgrade \
  "pip==${PIP_VERSION}" \
  "setuptools==${SETUPTOOLS_VERSION}" \
  wheel

# Install torch/cu129 from PyTorch index (use PyPI for non-torch deps).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
ARG NVIDIA_PYPI_INDEX=https://pypi.nvidia.com
ARG TORCH_VERSION=2.9.1+cu129
ARG TORCHVISION_VERSION=0.24.1+cu129
ARG TORCHAUDIO_VERSION=2.9.1+cu129
# Install torch via a direct wheel URL (avoid pip hash enforcement issues observed with the
# PyTorch simple index fragment in earlier builds).
ARG TORCH_WHEEL_URL=https://download.pytorch.org/whl/cu129/torch-2.9.1%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_FILENAME=torch-2.9.1+cu129-cp310-cp310-manylinux_2_28_x86_64.whl
RUN set -euo pipefail \
  && aria2c -x 8 -s 8 -k 1M --max-tries=10 --retry-wait=2 \
    -d /tmp -o "${TORCH_WHEEL_FILENAME}" "${TORCH_WHEEL_URL}" \
  && python -m pip install --no-cache-dir --only-binary=:all: \
    --index-url https://pypi.org/simple \
    --extra-index-url "${TORCH_INDEX_URL}" \
    --extra-index-url "${NVIDIA_PYPI_INDEX}" \
    "/tmp/${TORCH_WHEEL_FILENAME}" \
  && rm -f "/tmp/${TORCH_WHEEL_FILENAME}"

RUN python -m pip install --no-cache-dir --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --extra-index-url "${TORCH_INDEX_URL}" \
  --extra-index-url "${NVIDIA_PYPI_INDEX}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

# Verify torch CUDA build + critical CUDA user-space package versions.
# (Do not override torch's pinned nvidia-* deps: mismatched versions can silently break runtime.)
RUN python -c "import importlib.metadata as im; import torch; print('torch.__version__:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); assert (torch.version.cuda or '').startswith('12.9'), f\"expected CUDA 12.9.*, got {torch.version.cuda}\"; print('nvidia-cuda-nvrtc-cu12:', im.version('nvidia-cuda-nvrtc-cu12')); print('nvidia-nvjitlink-cu12:', im.version('nvidia-nvjitlink-cu12')); assert im.version('nvidia-cuda-nvrtc-cu12') == '12.9.86'; assert im.version('nvidia-nvjitlink-cu12') == '12.9.86'"

# Core runtime deps used by the API server and Ray workers (versions from `ssh mint-dev` baseline).
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    "ray==2.51.1" \
    "fastapi[standard]==0.121.2" \
    "uvicorn[standard]==0.38.0" \
    "pydantic==2.12.4" \
    "httpx==0.27.2" \
    "transformers==4.57.1" \
    "accelerate==1.11.0" \
    "einops" \
    "onnxscript" \
    "codetiming==1.4.0" \
    "torchdata==0.11.0" \
    "datasets==4.4.2" \
    "tensordict" \
    "peft==0.18.0"

# NVIDIA ModelOpt (import name: modelopt).
ARG NVIDIA_MODELOPT_VERSION=0.41.0
RUN python -m pip install --no-cache-dir \
  --index-url "${NVIDIA_PYPI_INDEX}/simple" \
  --extra-index-url https://pypi.org/simple \
  --retries 10 \
  --timeout 60 \
  --only-binary=:all: \
  "nvidia-modelopt==${NVIDIA_MODELOPT_VERSION}"

# `pybind11` is used by some training stacks and must come from a prebuilt wheel.
RUN python -m pip install --no-cache-dir --only-binary=:all: "pybind11"

# Transformer Engine (Megatron dependency): install from NVIDIA index and verify the torch extension loads.
ARG TRANSFORMER_ENGINE_VERSION=2.11.0
RUN python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url "${NVIDIA_PYPI_INDEX}" \
  --only-binary=:all: \
  "transformer-engine[core_cu12]==${TRANSFORMER_ENGINE_VERSION}"
# NOTE: `transformer_engine_torch` is sdist-only on public indices. This is an intentional
# exception to "wheel-only installs", and matches the build-from-source approach in `mint:12`.
RUN python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url "${NVIDIA_PYPI_INDEX}" \
  --no-build-isolation \
  "transformer_engine_torch==${TRANSFORMER_ENGINE_VERSION}"
RUN python -c "import transformer_engine; import transformer_engine.pytorch as te; print('transformer_engine.__version__:', getattr(transformer_engine, '__version__', 'unknown')); print('transformer_engine.pytorch:', te)"

# vLLM: use the official prebuilt wheel (CUDA 12.9 / cu129).
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/89a77b10846fd96273cce78d86d2556ea582d26e/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl
RUN python -m pip install --no-cache-dir --only-binary=:all: --upgrade "${VLLM_WHEEL_URL}"

# FlashAttention (optional accel used by some inference/training stacks).
# Prefer a matching prebuilt wheel to avoid compiling CUDA extensions in the Docker build.
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_WHEEL_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.11/flash_attn-2.8.3%2Bcu129torch2.9-cp310-cp310-linux_x86_64.whl
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps "${FLASH_ATTN_WHEEL_URL}"
RUN python -c "import flash_attn; import flash_attn.flash_attn_interface; print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))"

# OmegaConf requires antlr4-python3-runtime==4.9.* (and later runtimes are not compatible).
# PyPI does not publish 4.9.* wheels, so install from sdist (reproducible source build, no runtime copies).
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps "omegaconf==2.3.0"
RUN python -m pip install --no-cache-dir --no-build-isolation "antlr4-python3-runtime==4.9.3"
RUN python -c "import antlr4; print('antlr4:', antlr4.__file__); import omegaconf; print('omegaconf:', omegaconf.__version__)"

# Megatron-LM + Megatron-Bridge + verl: clone from pinned commits (no local modifications).
ARG MEGATRON_LM_REPO=https://github.com/NVIDIA/Megatron-LM.git
ARG MEGATRON_LM_COMMIT=0810e6390280672f9c87c388ce4f559571d54365
ARG MEGATRON_BRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
ARG MEGATRON_BRIDGE_COMMIT=0034ddaad7fae7c658c3df7e12d13522a4935770
ARG VERL_REPO=https://github.com/verl-project/verl.git
# Sync to the same git commit as `/vePFS-Mindverse/share/code/leixiang/verl` on Volcano.
ARG VERL_COMMIT=9433f8a8f2771256ea4f8f94e4401bcfe9703228
RUN mkdir -p /workspace \
  && git clone --filter=blob:none --depth=1 --branch main "${MEGATRON_LM_REPO}" /workspace/Megatron-LM \
  && git -C /workspace/Megatron-LM checkout "${MEGATRON_LM_COMMIT}" \
  && git clone --filter=blob:none --depth=1 --branch main "${MEGATRON_BRIDGE_REPO}" /workspace/Megatron-Bridge \
  && git -C /workspace/Megatron-Bridge checkout "${MEGATRON_BRIDGE_COMMIT}" \
  && git clone --filter=blob:none "${VERL_REPO}" /workspace/verl \
  && git -C /workspace/verl checkout "${VERL_COMMIT}"

ENV PYTHONPATH="/workspace/Megatron-LM:/workspace/Megatron-Bridge:/workspace/verl:${PYTHONPATH}"
RUN python -c "import verl; print('verl:', verl.__file__)"

# Default command is intentionally minimal; Ray task YAMLs and server ops override this.
RUN python -c "import onnxscript, modelopt; print('onnxscript', getattr(onnxscript, '__version__', 'unknown')); print('modelopt', getattr(modelopt, '__version__', 'unknown'))"
CMD ["bash", "-lc", "python -c 'import torch, vllm; print(torch.__version__, torch.version.cuda); print(vllm.__version__)' && sleep infinity"]
