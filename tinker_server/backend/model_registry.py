"""Model configuration registry for hardware requirements."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hardware configuration for a model.

    Inference parallelism (vLLM):
        - recommended_tp: Tensor parallelism (shards weights)
        - recommended_dp: Data parallelism (replicas)

    Training parallelism (Megatron):
        - train_tp: Tensor parallelism
        - train_ep: Expert parallelism (MoE only, 1 for dense)
    """

    is_moe: bool
    recommended_tp: int  # vLLM inference TP
    recommended_dp: int  # vLLM inference DP
    train_tp: int = 1  # Megatron training TP
    train_ep: int = 1  # Megatron training EP (MoE only)
    use_fp8: bool = False  # Use FP8 quantization (K2, etc.)

    @property
    def total_gpus(self) -> int:
        """Total GPUs for inference."""
        return self.recommended_tp * self.recommended_dp

    @property
    def train_gpus(self) -> int:
        """Total GPUs for training (world_size = TP * EP)."""
        return self.train_tp * self.train_ep


# Supported models - only these are allowed
MODEL_CONFIGS = {
    # Dense models (train_tp=1, train_ep=1 - uses PEFT backend)
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        is_moe=False, recommended_tp=1, recommended_dp=1, train_tp=1, train_ep=1
    ),
    "Qwen/Qwen3-0.6B": ModelConfig(
        is_moe=False, recommended_tp=1, recommended_dp=1, train_tp=1, train_ep=1
    ),
    # MoE models - Qwen3 30B variants
    # Inference: TP=4, DP=1 (4 GPUs) - EP not supported in vLLM LoRA
    # Training: TP=4, EP=1 (4 GPUs) - reduced from TP=4,EP=2 for smaller clusters
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelConfig(
        is_moe=True, recommended_tp=4, recommended_dp=1, train_tp=4, train_ep=1
    ),
    "Qwen/Qwen3-30B-A3B": ModelConfig(
        is_moe=True, recommended_tp=4, recommended_dp=1, train_tp=4, train_ep=1
    ),
    "Qwen/Qwen3-30B-A3B-Base": ModelConfig(
        is_moe=True, recommended_tp=4, recommended_dp=1, train_tp=4, train_ep=1
    ),
    # Kimi K2 - 1T param MoE (384 experts, 8+1 active per token)
    # Reference: TP=8, EP=64 (64 GPUs)
    # Minimum: TP=8, EP=16 (16 GPUs) with FP8 + offload
    # Block-FP8 quantization required for memory efficiency
    "moonshotai/Kimi-K2-Instruct": ModelConfig(
        is_moe=True,
        recommended_tp=8,  # Inference: 8 GPUs minimum
        recommended_dp=1,
        train_tp=8,
        train_ep=8,  # Training: 64 GPUs (8×8), can reduce to 16 with offload
        use_fp8=True,
    ),
    "moonshotai/Kimi-K2-Thinking": ModelConfig(
        is_moe=True,
        recommended_tp=8,
        recommended_dp=1,
        train_tp=8,
        train_ep=8,
        use_fp8=True,
    ),
}


def normalize_model_name(model_name_or_path: str) -> str:
    """Normalize a model path or name to HuggingFace model name format.

    Args:
        model_name_or_path: Either a HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
            or a full path (e.g., "/vePFS/.../models--Qwen--Qwen3-0.6B/snapshots/...")

    Returns:
        Normalized model name

    Raises:
        ValueError: If model cannot be identified
    """
    # If already a valid model name, return as-is
    if model_name_or_path in MODEL_CONFIGS:
        return model_name_or_path

    # Try to extract model name from HuggingFace cache path
    # Path format: .../models--{org}--{model}/snapshots/...
    import re
    match = re.search(r'models--([^/]+)--([^/]+)', model_name_or_path)
    if match:
        org, model = match.groups()
        candidate = f"{org}/{model}"
        if candidate in MODEL_CONFIGS:
            return candidate

    # Try substring matching as fallback
    for model_name in MODEL_CONFIGS:
        # Normalize for comparison: Qwen/Qwen3-0.6B -> qwen--qwen3-0.6b
        path_pattern = model_name.replace("/", "--").lower()
        if path_pattern in model_name_or_path.lower():
            return model_name

    raise ValueError(
        f"Cannot identify model from: {model_name_or_path}. "
        f"Supported models: {list(MODEL_CONFIGS.keys())}"
    )


def get_model_config(model_name: str) -> ModelConfig:
    """Get config for a supported model.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B") or full path

    Returns:
        ModelConfig with parallelism recommendations

    Raises:
        ValueError: If model is not in the supported list
    """
    # Normalize path to model name
    normalized = normalize_model_name(model_name)
    return MODEL_CONFIGS[normalized]


def is_supported_model(model_name: str) -> bool:
    """Check if a model is in the supported list."""
    try:
        normalize_model_name(model_name)
        return True
    except ValueError:
        return False


def is_moe_model(model_name: str) -> bool:
    """Check if a model uses MoE architecture (requires Megatron backend).

    Args:
        model_name: HuggingFace model name

    Returns:
        True if model is MoE, False if dense

    Raises:
        ValueError: If model is not supported
    """
    config = get_model_config(model_name)
    return config.is_moe


def get_recommended_parallelism(model_name: str) -> tuple[int, int]:
    """Return (tensor_parallel_size, data_parallel_size) for inference.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Tuple of (TP, DP) sizes for vLLM inference
    """
    config = get_model_config(model_name)
    return config.recommended_tp, config.recommended_dp


def get_training_parallelism(model_name: str) -> tuple[int, int]:
    """Return (tensor_parallel_size, expert_parallel_size) for training.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (TP, EP) sizes for Megatron training
    """
    config = get_model_config(model_name)
    return config.train_tp, config.train_ep


def requires_fp8(model_name: str) -> bool:
    """Check if model requires FP8 quantization.

    Args:
        model_name: HuggingFace model name

    Returns:
        True if model uses FP8 (e.g., Kimi-K2)
    """
    config = get_model_config(model_name)
    return config.use_fp8
