# syntax=docker/dockerfile:1.6
#
# Mint runtime image (replacement for `mint:4`), used by:
# - API server container
# - Ray head/worker tasks
#
# This Dockerfile must not assume the build context contains tinker-server code.
# (The platform build environment may only provide this Dockerfile.)

ARG CUDA_IMAGE=nvidia/cuda:12.9.0-cudnn9-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# System deps: match volcano host baseline (Ubuntu 22.04) + build deps for python wheels.
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
  wheel

# Install torch/cu129 from PyTorch index.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
ARG TORCH_VERSION=2.9.0+cu129
ARG TORCHVISION_VERSION=0.24.0+cu129
ARG TORCHAUDIO_VERSION=2.9.0+cu129
RUN python -m pip install --no-cache-dir \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

# Pin CUDA 12.9 user-space libs used by torch.
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

# Core runtime deps used by the API server and Ray workers (versions from `ssh volcano` baseline).
RUN python -m pip install --no-cache-dir \
    "ray==2.51.1" \
    "fastapi==0.121.2" \
    "uvicorn[standard]==0.38.0" \
    "pydantic==2.12.4" \
    "httpx==0.27.2" \
    "transformers==4.57.1" \
    "accelerate==1.11.0" \
    "omegaconf==2.3.0" \
    "peft==0.18.0"

# vLLM: use a specific prebuilt wheel (matches volcano override commit g811cdf519).
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/811cdf5197acb4d6ab42250a5b0f822887d1190a/vllm-0.13.0rc2.dev207%2Bg811cdf519-cp38-abi3-manylinux_2_31_x86_64.whl
RUN python -m pip install --no-cache-dir --upgrade "${VLLM_WHEEL_URL}"

# vLLM MoE expert LoRA: patch LoRAModel.from_local_checkpoint unexpected-module check.
RUN python - <<'PY'
import pathlib

import vllm

p = pathlib.Path(vllm.__file__).resolve().parent / "lora" / "lora_model.py"
content = p.read_text()

old = "\n".join([
    '                if ".experts" in module_name:',
    '                    expert_idx = module_name.find(".experts")',
    '                    expert_suffix = module_name[expert_idx + 1 :]',
    '                    if expert_suffix not in expected_lora_modules:',
    '                        unexpected_modules.append(module_name)',
    '',
])

new = "\n".join([
    '                if ".experts" in module_name:',
    '                    # Handle expert patterns like: experts.0.gate_proj, experts.1.down_proj',
    '                    # Extract the module name after experts.{N}.',
    '                    import re',
    '                    VALID_EXPERT_SUFFIXES = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}',
    '                    expert_match = re.search(r"\\.experts\\.(\\d+)\\.(\\w+)$", module_name)',
    '                    if expert_match:',
    '                        expert_module = expert_match.group(2)',
    '                        if expert_module not in VALID_EXPERT_SUFFIXES and \\',
    '                           expert_module not in expected_lora_modules and \\',
    '                           "experts" not in expected_lora_modules:',
    '                            unexpected_modules.append(module_name)',
    '                    else:',
    '                        if "experts" not in expected_lora_modules:',
    '                            unexpected_modules.append(module_name)',
    '',
])

if "VALID_EXPERT_SUFFIXES" in content:
    print("vLLM MoE LoRA patch: already applied", p)
elif old not in content:
    raise RuntimeError(f"vLLM MoE LoRA patch: pattern not found in {p}")
else:
    p.write_text(content.replace(old, new))
    print("vLLM MoE LoRA patch: applied", p)
PY

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
  && python -m pip install --no-cache-dir -e /workspace/Megatron-LM \
  && git clone "${MEGATRON_BRIDGE_REPO}" /workspace/Megatron-Bridge \
  && git -C /workspace/Megatron-Bridge checkout "${MEGATRON_BRIDGE_COMMIT}" \
  && python -m pip install --no-cache-dir -e /workspace/Megatron-Bridge \
  && git clone "${VERL_REPO}" /workspace/verl \
  && git -C /workspace/verl checkout "${VERL_COMMIT}" \
  && python -m pip install --no-cache-dir -e /workspace/verl

# Default command is intentionally minimal; Ray task YAMLs and server ops override this.
CMD ["bash", "-lc", "python -c 'import torch, vllm; print(torch.__version__, torch.version.cuda); print(vllm.__version__)' && sleep infinity"]
