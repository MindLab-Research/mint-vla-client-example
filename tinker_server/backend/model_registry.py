"""Model configuration registry for hardware requirements."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hardware configuration for a model."""

    is_moe: bool
    recommended_tp: int
    recommended_dp: int

    @property
    def total_gpus(self) -> int:
        return self.recommended_tp * self.recommended_dp


# Supported models - only these are allowed
MODEL_CONFIGS = {
    # Dense models
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(False, 1, 1),
    "Qwen/Qwen3-0.6B": ModelConfig(False, 1, 1),
    # MoE models (TP=4, DP=1 for vLLM LoRA support - EP not supported in vLLM 0.12.0)
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelConfig(True, 4, 1),
    "Qwen/Qwen3-30B-A3B": ModelConfig(True, 4, 1),
    "Qwen/Qwen3-30B-A3B-Base": ModelConfig(True, 4, 1),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get config for a supported model.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")

    Returns:
        ModelConfig with parallelism recommendations

    Raises:
        ValueError: If model is not in the supported list
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    raise ValueError(
        f"Unsupported model: {model_name}. "
        f"Supported models: {list(MODEL_CONFIGS.keys())}"
    )


def is_supported_model(model_name: str) -> bool:
    """Check if a model is in the supported list."""
    return model_name in MODEL_CONFIGS


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
    """Return (tensor_parallel_size, data_parallel_size) for model.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Tuple of (TP, DP) sizes
    """
    config = get_model_config(model_name)
    return config.recommended_tp, config.recommended_dp
