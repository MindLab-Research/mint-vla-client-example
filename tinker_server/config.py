"""Server configuration."""

import os
import secrets
from dataclasses import dataclass

# PFS paths for Ray worker runtime_env
# NOTE: vLLM 0.12.0 requires PyTorch 2.9.0, which requires NCCL 2.21+
# System has NCCL 2.x (older) - cannot use PFS PyTorch 2.9.0
# MoE LoRA blocked until Docker image upgraded with newer CUDA stack
PFS_TINKER_PATH = "/vePFS-Mindverse/share/code/tinker-server"

# PFS verl path with _mutable_fields patch for LoRA config assignment
PFS_VERL_PATH = "/vePFS-Mindverse/share/code/verl"

# HuggingFace modules path for trust_remote_code models (K2, etc.)
# Custom model code is cached here when models are first loaded
PFS_HF_MODULES_PATH = "/vePFS-Mindverse/share/huggingface/modules"

# PYTHONPATH for Ray actors - verl first (for _mutable_fields patch), then tinker-server, then HF modules
PFS_PYTHONPATH = f"{PFS_VERL_PATH}:{PFS_TINKER_PATH}:{PFS_HF_MODULES_PATH}"


@dataclass
class ServerConfig:
    """Configuration for tinker-server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication
    api_key: str = ""  # If empty, auth disabled; if set, all endpoints require it

    # Model settings (no default model - clients specify per-request)
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1  # For MoE: EP = TP * DP
    gpu_memory_utilization: float = 0.9
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
        # If no API key set, auth is disabled (dev mode)

        return cls(
            host=os.environ.get("TINKER_HOST", "0.0.0.0"),
            port=int(os.environ.get("TINKER_PORT", "8000")),
            api_key=api_key,
            tensor_parallel_size=int(os.environ.get("TINKER_TP_SIZE", "1")),
            data_parallel_size=int(os.environ.get("TINKER_DP_SIZE", "1")),
            gpu_memory_utilization=float(os.environ.get("TINKER_GPU_MEM_UTIL", "0.9")),
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

    def validate_api_key(self, provided_key: str) -> bool:
        """Validate API key using constant-time comparison."""
        return secrets.compare_digest(self.api_key, provided_key)


# Global config instance
config = ServerConfig.from_env()
