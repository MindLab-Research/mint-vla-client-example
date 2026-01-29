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
ARG TORCH_VERSION=2.9.0+cu129
ARG TORCHVISION_VERSION=0.24.0+cu129
ARG TORCHAUDIO_VERSION=2.9.0+cu129
# Work around a hash mismatch between the PyTorch simple index fragment and served bytes
# for torch==2.9.0+cu129 (observed on 2026-01-27). Install torch via a direct wheel URL
# (no fragment hash enforcement).
ARG TORCH_WHEEL_URL=https://download.pytorch.org/whl/cu129/torch-2.9.0%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_FILENAME=torch-2.9.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl
RUN set -euo pipefail \
  && aria2c -x 8 -s 8 -k 1M --max-tries=10 --retry-wait=2 \
    -d /tmp -o "${TORCH_WHEEL_FILENAME}" "${TORCH_WHEEL_URL}" \
  && python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url "${TORCH_INDEX_URL}" \
    --extra-index-url "${NVIDIA_PYPI_INDEX}" \
    "/tmp/${TORCH_WHEEL_FILENAME}" \
  && rm -f "/tmp/${TORCH_WHEEL_FILENAME}"

RUN python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url "${TORCH_INDEX_URL}" \
  --extra-index-url "${NVIDIA_PYPI_INDEX}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

# Verify torch CUDA build + critical CUDA user-space package versions.
# (Do not override torch's pinned nvidia-* deps: mismatched versions can silently break runtime.)
RUN python - <<'PY'
import importlib.metadata as im

import torch

print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
assert (torch.version.cuda or "").startswith("12.9"), f"expected CUDA 12.9.*, got {torch.version.cuda}"

print("nvidia-cuda-nvrtc-cu12:", im.version("nvidia-cuda-nvrtc-cu12"))
print("nvidia-nvjitlink-cu12:", im.version("nvidia-nvjitlink-cu12"))
assert im.version("nvidia-cuda-nvrtc-cu12") == "12.9.86"
assert im.version("nvidia-nvjitlink-cu12") == "12.9.86"
PY

# Core runtime deps used by the API server and Ray workers (versions from `ssh mint-dev` baseline).
RUN python -m pip install --no-cache-dir \
    "ray==2.51.1" \
    "fastapi==0.121.2" \
    "uvicorn[standard]==0.38.0" \
    "pydantic==2.12.4" \
    "httpx==0.27.2" \
    "transformers==4.57.1" \
    "accelerate==1.11.0" \
    "omegaconf==2.3.0" \
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
  "nvidia-modelopt==${NVIDIA_MODELOPT_VERSION}"

# Build deps for editable Megatron installs (avoid PEP 517 build isolation re-downloading torch).
RUN python -m pip install --no-cache-dir "pybind11"

# Transformer Engine (Megatron dependency): install from NVIDIA index and verify the torch extension loads.
ARG TRANSFORMER_ENGINE_VERSION=2.11.0
RUN python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url "${NVIDIA_PYPI_INDEX}" \
  --no-build-isolation \
  "transformer-engine[pytorch,core_cu12]==${TRANSFORMER_ENGINE_VERSION}"
RUN python -c "import transformer_engine; import transformer_engine.pytorch as te; print('transformer_engine.__version__:', getattr(transformer_engine, '__version__', 'unknown')); print('transformer_engine.pytorch:', te)"

# vLLM: use a specific prebuilt wheel (matches volcano override commit g811cdf519).
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/811cdf5197acb4d6ab42250a5b0f822887d1190a/vllm-0.13.0rc2.dev207%2Bg811cdf519-cp38-abi3-manylinux_2_31_x86_64.whl
RUN python -m pip install --no-cache-dir --upgrade "${VLLM_WHEEL_URL}"

# FlashAttention (optional accel used by some inference/training stacks).
# Prefer a matching prebuilt wheel to avoid compiling CUDA extensions in the Docker build.
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_WHEEL_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.11/flash_attn-2.8.3%2Bcu129torch2.9-cp310-cp310-linux_x86_64.whl
RUN python -m pip install --no-cache-dir "${FLASH_ATTN_WHEEL_URL}"
RUN python -c "import flash_attn; import flash_attn.flash_attn_interface; print('flash_attn', getattr(flash_attn, '__version__', 'unknown'))"

# Megatron-LM + Megatron-Bridge + verl: install from pinned commits (clean, no local dirty state).
ARG MEGATRON_LM_REPO=https://github.com/NVIDIA/Megatron-LM.git
ARG MEGATRON_LM_COMMIT=aa4ec99205a52187adead37cabceb678a2b6b975
ARG MEGATRON_BRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
# Includes HollowMan6 PRs (e.g. #1811, #1834) without requiring newer Megatron-LM experimental specs.
ARG MEGATRON_BRIDGE_COMMIT=2e73984734ca17658b834633995a15b2536e1911
ARG VERL_REPO=https://github.com/volcengine/verl.git
# Includes vLLM LoRA import compatibility for vllm>=0.13 (vllm.lora.lora_model).
ARG VERL_COMMIT=2bb42bae6078359c3fdc56ba6c7533e76fc05407
RUN mkdir -p /workspace \
  && git clone "${MEGATRON_LM_REPO}" /workspace/Megatron-LM \
  && git -C /workspace/Megatron-LM checkout "${MEGATRON_LM_COMMIT}" \
  && python -m pip install --no-cache-dir --no-build-isolation --no-deps -e /workspace/Megatron-LM \
  && git clone "${MEGATRON_BRIDGE_REPO}" /workspace/Megatron-Bridge \
  && git -C /workspace/Megatron-Bridge checkout "${MEGATRON_BRIDGE_COMMIT}" \
  && python -m pip install --no-cache-dir --no-build-isolation --no-deps -e /workspace/Megatron-Bridge \
  && git clone "${VERL_REPO}" /workspace/verl \
  && git -C /workspace/verl checkout "${VERL_COMMIT}" \
  && python -m pip install --no-cache-dir --no-build-isolation --no-deps -e /workspace/verl

# Default command is intentionally minimal; Ray task YAMLs and server ops override this.
RUN python -c "import onnxscript, modelopt; print('onnxscript', getattr(onnxscript, '__version__', 'unknown')); print('modelopt', getattr(modelopt, '__version__', 'unknown'))"
CMD ["bash", "-lc", "python -c 'import torch, vllm; print(torch.__version__, torch.version.cuda); print(vllm.__version__)' && sleep infinity"]
