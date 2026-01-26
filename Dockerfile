# syntax=docker/dockerfile:1.6
#
# Mint/Tinker runtime image used by:
# - API server container (tinker-server)
# - Ray head/worker tasks (GPU workers)
#
# Target constraints (per user request):
# - torch 2.9.0 with CUDA 12.9 user-space libs
# - CUDA user-space libs pinned via pip nvidia-* packages (12.9.*)
#
# Note: CUDA driver is provided by the host via NVIDIA Container Toolkit.

ARG CUDA_IMAGE=nvidia/cuda:12.9.0-cudnn9-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# System deps: match volcano host baseline (Ubuntu 22.04) + build deps for vLLM extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
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
RUN python -m pip install --no-cache-dir --upgrade \
  "pip==${PIP_VERSION}" \
  setuptools \
  wheel \
  uv

WORKDIR /root/tinker_project/tinker-server

# Install Python deps first for build caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Install project code.
COPY . .
RUN uv sync --frozen --no-dev

# Pin CUDA 12.9 user-space libs used by torch.
# (Torch wheels depend on these packages; pinning here forces CUDA 12.9 instead of CUDA 12.8.)
ARG NVIDIA_CUDA_RUNTIME_CU12=12.9.79
ARG NVIDIA_CUDA_CUPTI_CU12=12.9.79
ARG NVIDIA_CUDA_NVRTC_CU12=12.9.86
ARG NVIDIA_NVJITLINK_CU12=12.9.86
ARG NVIDIA_NVTX_CU12=12.9.79
ARG NVIDIA_CUBLAS_CU12=12.9.1.4
ARG NVIDIA_NCCL_CU12=2.29.2
RUN python -m pip install --no-cache-dir --upgrade \
    "nvidia-cuda-runtime-cu12==${NVIDIA_CUDA_RUNTIME_CU12}" \
    "nvidia-cuda-cupti-cu12==${NVIDIA_CUDA_CUPTI_CU12}" \
    "nvidia-cuda-nvrtc-cu12==${NVIDIA_CUDA_NVRTC_CU12}" \
    "nvidia-nvjitlink-cu12==${NVIDIA_NVJITLINK_CU12}" \
    "nvidia-nvtx-cu12==${NVIDIA_NVTX_CU12}" \
    "nvidia-cublas-cu12==${NVIDIA_CUBLAS_CU12}" \
    "nvidia-nccl-cu12==${NVIDIA_NCCL_CU12}" \
  && python - <<'PY'
import torch
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
assert (torch.version.cuda or "").startswith("12.9"), f"expected CUDA 12.9.*, got {torch.version.cuda}"
PY

# vLLM patch for MoE expert LoRA support (idempotent).
RUN python patches/apply_vllm_patch.py

# Install Megatron-LM + Megatron-Bridge from pinned commits (mirrors mint:4 editable installs).
ARG MEGATRON_LM_REPO=https://github.com/NVIDIA/Megatron-LM.git
ARG MEGATRON_LM_COMMIT=aa4ec99205a52187adead37cabceb678a2b6b975
ARG MEGATRON_BRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
ARG MEGATRON_BRIDGE_COMMIT=b2bb00a0e01112c2738b1865ca6e7cb65ae2f5c4
RUN mkdir -p /workspace \
  && git clone "${MEGATRON_LM_REPO}" /workspace/Megatron-LM \
  && git -C /workspace/Megatron-LM checkout "${MEGATRON_LM_COMMIT}" \
  && python -m pip install --no-cache-dir -e /workspace/Megatron-LM \
  && git clone "${MEGATRON_BRIDGE_REPO}" /workspace/Megatron-Bridge \
  && git -C /workspace/Megatron-Bridge checkout "${MEGATRON_BRIDGE_COMMIT}" \
  && python -m pip install --no-cache-dir -e /workspace/Megatron-Bridge

# Install verl from pinned commit (mirrors mint:4 editable install).
ARG VERL_REPO=https://github.com/volcengine/verl.git
ARG VERL_COMMIT=38246890efb50e60d2471ac2518cb512ba8361ba
RUN git clone "${VERL_REPO}" /root/verl \
  && git -C /root/verl checkout "${VERL_COMMIT}" \
  && python -m pip install --no-cache-dir -e /root/verl

# Default command is intentionally minimal; Ray task YAMLs and server ops override this.
CMD ["bash", "-lc", "python -c 'import torch; import vllm; print(torch.__version__, torch.version.cuda); print(vllm.__version__)' && sleep infinity"]

