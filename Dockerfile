# syntax=docker/dockerfile:1.6
#
# Mint runtime image (`mint:14`), used by:
# - API server container
# - Ray head/worker tasks
#
# This Dockerfile must not assume the build context contains tinker-server code.
# (The platform build environment may only provide this Dockerfile.)

# Base image with DeepEP preinstalled (import name: `deep_ep`).
# Pin by digest for reproducibility.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.3-cu129@sha256:9971a17304b3e3688a41d591ccd6538f2f3a0baf9c73c4ab21e1e4ee993327e8
FROM ${SGLANG_IMAGE}

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# System deps: build deps for python sdists (TransformerEngine torch extension, antlr runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
  aria2 \
  ca-certificates \
  curl \
  git \
  build-essential \
  pkg-config \
  ninja-build \
  ibverbs-providers \
  ibverbs-utils \
  libibverbs-dev \
  && rm -rf /var/lib/apt/lists/*

# Virtualenv + tooling. Use `--system-site-packages` so we can import DeepEP + torch from the base image.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ARG PIP_VERSION=24.2
ARG SETUPTOOLS_VERSION=79.0.0
RUN python -m pip install --no-cache-dir --upgrade \
  "pip==${PIP_VERSION}" \
  "setuptools==${SETUPTOOLS_VERSION}" \
  wheel

# Verify DeepEP + torch from the base image.
# NOTE: DeepEP's compiled extension is ABI-sensitive to torch; do not upgrade torch unless DeepEP is rebuilt.
RUN python -c "import torch; import deep_ep; import deep_ep_cpp; print('torch.__version__:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); assert torch.__version__.startswith('2.8.'), f\"expected torch 2.8.*, got {torch.__version__}\"; assert (torch.version.cuda or '').startswith('12.9'), f\"expected CUDA 12.9.*, got {torch.version.cuda}\""

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

# vLLM: use the official prebuilt wheel.
# IMPORTANT: install with `--no-deps` to avoid upgrading torch (which would break DeepEP).
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/89a77b10846fd96273cce78d86d2556ea582d26e/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps --upgrade "${VLLM_WHEEL_URL}"
# vLLM runtime deps (installed explicitly because we install the vLLM wheel with `--no-deps`).
RUN python -m pip install --no-cache-dir --only-binary=:all: \
  "aiohttp>=3.13.3" \
  "anthropic>=0.71.0" \
  "blake3" \
  "cachetools" \
  "cbor2" \
  "compressed-tensors==0.13.0" \
  "depyf==0.20.0" \
  "diskcache==5.6.3" \
  "flashinfer-python==0.6.3" \
  "gguf>=0.17.0" \
  "grpcio" \
  "grpcio-reflection" \
  "ijson" \
  "lark==1.2.2" \
  "llguidance>=1.3.0,<1.4.0" \
  "lm-format-enforcer==0.11.3" \
  "mcp" \
  "mistral_common[image]>=1.9.0" \
  "model-hosting-container-standards>=0.1.13,<1.0.0" \
  "msgspec" \
  "numba==0.61.2" \
  "opencv-python-headless>=4.13.0" \
  "openai==1.99.9" \
  "openai-harmony>=0.0.3" \
  "outlines_core==0.2.11" \
  "partial-json-parser" \
  "prometheus-fastapi-instrumentator>=7.0.0" \
  "protobuf==6.33.5" \
  "py-cpuinfo" \
  "pybase64" \
  "python-json-logger" \
  "pyzmq>=25.0.0" \
  "setproctitle" \
  "xgrammar==0.1.29"
RUN python -c "import torch, deep_ep, deep_ep_cpp; import vllm; from vllm import LLM; print('vllm:', vllm.__version__); print('torch:', torch.__version__); print('deep_ep:', deep_ep.__file__)"

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
RUN set -euo pipefail \
  && mkdir -p /workspace \
  && git init /workspace/Megatron-LM \
  && git -C /workspace/Megatron-LM remote add origin "${MEGATRON_LM_REPO}" \
  && git -C /workspace/Megatron-LM fetch --depth=1 origin "${MEGATRON_LM_COMMIT}" \
  && git -C /workspace/Megatron-LM checkout --detach FETCH_HEAD \
  && git init /workspace/Megatron-Bridge \
  && git -C /workspace/Megatron-Bridge remote add origin "${MEGATRON_BRIDGE_REPO}" \
  && git -C /workspace/Megatron-Bridge fetch --depth=1 origin "${MEGATRON_BRIDGE_COMMIT}" \
  && git -C /workspace/Megatron-Bridge checkout --detach FETCH_HEAD \
  && git init /workspace/verl \
  && git -C /workspace/verl remote add origin "${VERL_REPO}" \
  && git -C /workspace/verl fetch --depth=1 origin "${VERL_COMMIT}" \
  && git -C /workspace/verl checkout --detach FETCH_HEAD

ENV PYTHONPATH="/workspace/Megatron-LM:/workspace/Megatron-Bridge:/workspace/verl:${PYTHONPATH}"
RUN python -c "import verl; print('verl:', verl.__file__)"

# Default command is intentionally minimal; Ray task YAMLs and server ops override this.
RUN python -c "import onnxscript, modelopt; print('onnxscript', getattr(onnxscript, '__version__', 'unknown')); print('modelopt', getattr(modelopt, '__version__', 'unknown'))"
RUN python -c "import torch, deep_ep_cpp; import ray, vllm; import transformer_engine.pytorch as te; print('torch:', torch.__version__); print('ray:', ray.__version__); print('vllm:', vllm.__version__); print('te:', te)"
CMD ["bash", "-lc", "python -c 'import torch, deep_ep_cpp, vllm; print(torch.__version__, torch.version.cuda); print(vllm.__version__)' && sleep infinity"]
