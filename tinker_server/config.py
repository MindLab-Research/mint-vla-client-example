"""Server configuration."""

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for tinker-server."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Model settings
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("TINKER_HOST", "0.0.0.0"),
            port=int(os.environ.get("TINKER_PORT", "8000")),
            model_path=os.environ.get("TINKER_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
            tensor_parallel_size=int(os.environ.get("TINKER_TP_SIZE", "1")),
            gpu_memory_utilization=float(os.environ.get("TINKER_GPU_MEM_UTIL", "0.9")),
            max_model_len=int(os.environ["TINKER_MAX_MODEL_LEN"])
            if os.environ.get("TINKER_MAX_MODEL_LEN")
            else None,
        )


# Global config instance
config = ServerConfig.from_env()
