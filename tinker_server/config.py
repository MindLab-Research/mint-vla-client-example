"""Server configuration."""

import os
import secrets
from dataclasses import dataclass

# PFS paths for Ray worker runtime_env
# NOTE: vLLM 0.12.0 requires PyTorch 2.9.0, which requires NCCL 2.21+
# System has NCCL 2.x (older) - cannot use PFS PyTorch 2.9.0
# MoE LoRA blocked until Docker image upgraded with newer CUDA stack
#
# Default to the production code path so Ray actors use the same code as the
# API server deployment.
PFS_TINKER_PATH = os.environ.get("PFS_TINKER_PATH", "/vePFS-Mindverse/share/code/tinker-server-auth")

# PFS verl path with _mutable_fields patch for LoRA config assignment
PFS_VERL_PATH = "/vePFS-Mindverse/share/code/verl"

# PFS vLLM 0.13.0 with raw logits dump instrumentation
PFS_VLLM_PATH = "/vePFS-Mindverse/share/code/vllm-0.13.0-pkg"

# PFS megatron-bridge path for MoE LoRA ETP fix (PR #1380)
# Clone from: https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
PFS_MEGATRON_BRIDGE_PATH = "/vePFS-Mindverse/share/code/megatron-bridge/src"

# HollowMan fork with export_adapter_weights API for LoRA export
# Clone from: https://github.com/HollowMan6/Megatron-Bridge.git branch merged
PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH = "/vePFS-Mindverse/share/code/megatron-bridge-hollowman/src"

# Toggle to use HollowMan fork of Megatron-Bridge (affects training forward pass)
# Default: true - HollowMan fork fixes train-inference KL divergence (verified 2026-01-07)
USE_HOLLOWMAN_MBRIDGE = os.environ.get("USE_HOLLOWMAN_MBRIDGE", "true").lower() in ("true", "1", "yes")

# Toggle to use Megatron-Bridge export_adapter_weights API instead of custom implementation
USE_MBRIDGE_LORA_EXPORT = os.environ.get("USE_MBRIDGE_LORA_EXPORT", "false").lower() in ("true", "1", "yes")

# HuggingFace modules path for trust_remote_code models (K2, etc.)
# Custom model code is cached here when models are first loaded
PFS_HF_MODULES_PATH = "/vePFS-Mindverse/share/huggingface/modules"

# PYTHONPATH for Ray actors - vLLM first (for instrumentation), then megatron-bridge, verl, tinker-server, HF modules
# USE_HOLLOWMAN_MBRIDGE controls which megatron-bridge version is used
if USE_HOLLOWMAN_MBRIDGE:
    PFS_PYTHONPATH = f"{PFS_VLLM_PATH}:{PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH}:{PFS_VERL_PATH}:{PFS_TINKER_PATH}:{PFS_HF_MODULES_PATH}"
else:
    PFS_PYTHONPATH = f"{PFS_VLLM_PATH}:{PFS_MEGATRON_BRIDGE_PATH}:{PFS_VERL_PATH}:{PFS_TINKER_PATH}:{PFS_HF_MODULES_PATH}"


@dataclass
class ServerConfig:
    """Configuration for tinker-server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication
    api_key: str = ""  # Hardcoded API key (legacy). If set, accepts this key directly.
    token_secret_key: str = ""  # Secret for sk- token decryption. If set, accepts encrypted tokens.

    # Usage logging
    usage_log_dir: str = "/tmp/tinker_usage"  # Directory to store usage logs

    # Model settings (no default model - clients specify per-request)
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1  # For MoE: EP = TP * DP
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None

    # Multi-LoRA settings
    enable_multi_lora: bool = True  # Enable shared multi-LoRA engine
    max_loras: int = 64  # GPU slots for concurrent LoRA adapters (~2.5GB for 64 rank-32 Qwen-7B)
    max_cpu_loras: int = 1024  # CPU cache for evicted adapters
    max_lora_rank: int = 64  # Maximum supported LoRA rank

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        api_key = os.environ.get("TINKER_API_KEY", "")
        token_secret_key = os.environ.get("TINKER_TOKEN_SECRET_KEY", "")
        # Auth disabled (dev mode) if neither api_key nor token_secret_key is set

        return cls(
            host=os.environ.get("TINKER_HOST", "0.0.0.0"),
            port=int(os.environ.get("TINKER_PORT", "8000")),
            api_key=api_key,
            token_secret_key=token_secret_key,
            usage_log_dir=os.environ.get("TINKER_USAGE_LOG_DIR", "/tmp/tinker_usage"),
            tensor_parallel_size=int(os.environ.get("TINKER_TP_SIZE", "1")),
            data_parallel_size=int(os.environ.get("TINKER_DP_SIZE", "1")),
            gpu_memory_utilization=float(os.environ.get("TINKER_GPU_MEM_UTIL", "0.85")),
            max_model_len=int(os.environ["TINKER_MAX_MODEL_LEN"])
            if os.environ.get("TINKER_MAX_MODEL_LEN")
            else None,
            # Multi-LoRA settings
            enable_multi_lora=os.environ.get("TINKER_ENABLE_MULTI_LORA", "true").lower()
            in ("true", "1", "yes"),
            max_loras=int(os.environ.get("TINKER_MAX_LORAS", "64")),
            max_cpu_loras=int(os.environ.get("TINKER_MAX_CPU_LORAS", "1024")),
            max_lora_rank=int(os.environ.get("TINKER_MAX_LORA_RANK", "64")),
        )

    @property
    def auth_enabled(self) -> bool:
        """Check if any authentication is configured."""
        return bool(self.api_key or self.token_secret_key)

    def validate_api_key(self, provided_key: str) -> bool:
        """Validate hardcoded API key using constant-time comparison."""
        if not self.api_key:
            return False
        return secrets.compare_digest(self.api_key, provided_key)

# Global config instance
config = ServerConfig.from_env()
