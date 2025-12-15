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

# PYTHONPATH for Ray actors - verl first (for _mutable_fields patch), then tinker-server
PFS_PYTHONPATH = f"{PFS_VERL_PATH}:{PFS_TINKER_PATH}"


@dataclass
class ServerConfig:
    """Configuration for tinker-server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Authentication
    api_key: str = ""  # Required - server refuses to start without it

    # Model settings
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
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
        if not api_key:
            raise ValueError(
                "TINKER_API_KEY environment variable is required. "
                "Set it to a secure random string (e.g., 32+ bytes, base64 encoded)."
            )

        return cls(
            host=os.environ.get("TINKER_HOST", "0.0.0.0"),
            port=int(os.environ.get("TINKER_PORT", "8000")),
            api_key=api_key,
            model_path=os.environ.get("TINKER_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
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
