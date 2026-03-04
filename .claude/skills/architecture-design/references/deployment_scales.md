# Deployment scales (image vs venv)

This repo uses two distinct dependency surfaces. Mixing them causes silent drift and hard-to-debug Ray failures.

## 1) GPU worker image (Mint Docker image)

Purpose:
- Runs on Volcano/Aliyun GPU tasks (Ray head + Ray workers).
- Provides GPU runtime dependencies (CUDA, torch+cu*, vLLM, DeepEP, Megatron-LM, Megatron-Bridge, verl).

Image series and arch variants:
- Start at version 15:
  - `mint:15-sm80` for Volcano A-cards (A800, SM80)
  - `mint:15-sm90` for Aliyun H-cards (H, SM90)

Rule:
- Use the Mint Docker image for GPU workers. Do not depend on host venv state for worker runtime.

## 2) API server / Ray driver venv (host Python)

Purpose:
- Runs on the API server host / Ray driver host (CPU-only).
- Starts the API server process and joins or connects to the Ray cluster.

Hard requirements:
- `ray` must be installed in the driver venv and kept aligned with the cluster version.
- `torch` must be installed in the driver venv (CPU wheel is fine) to deserialize tensors from the Ray object store.
- Python patch version must match the Ray cluster Python exactly (Ray is strict).

Reference artifact:
- `requirements/api_server_driver_py31213.requirements.in` is the top-level dependency list.
- `requirements/api_server_driver_py31213.freeze.txt` is the pinned snapshot captured from `mint-dev:/root/venv_k2_py31213` (Python 3.12.13).

Rule:
- Use the host venv for the API server / driver. Do not rely on the Mint image to "patch" the driver environment at runtime.
