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


# Model-specific parallelism configurations
MODEL_CONFIGS = {
    # Dense models - Qwen2.5 series
    "Qwen/Qwen2.5-0.5B-Instruct": ModelConfig(False, 1, 1),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelConfig(False, 1, 1),
    "Qwen/Qwen2.5-3B-Instruct": ModelConfig(False, 1, 1),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(False, 1, 1),
    "Qwen/Qwen2.5-14B-Instruct": ModelConfig(False, 2, 1),
    "Qwen/Qwen2.5-32B-Instruct": ModelConfig(False, 2, 1),
    "Qwen/Qwen2.5-72B-Instruct": ModelConfig(False, 4, 1),
    # MoE models - Qwen3 series
    # EP = TP * DP, so Qwen3-30B with TP=1, DP=4 -> EP=4
    "Qwen/Qwen3-30B-A3B": ModelConfig(True, 1, 4),
    "Qwen/Qwen3-30B-A3B-Instruct": ModelConfig(True, 1, 4),
    # Qwen3-235B with TP=2, DP=4 -> EP=8
    "Qwen/Qwen3-235B-A22B": ModelConfig(True, 2, 4),
    "Qwen/Qwen3-235B-A22B-Instruct": ModelConfig(True, 2, 4),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get config for model, with fallback heuristics.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        ModelConfig with parallelism recommendations
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Extract short name for heuristic matching
    short_name = model_name.split("/")[-1]

    # Detect MoE from naming pattern "XXB-AXXB" (activated params suffix)
    if "-A" in short_name:
        return ModelConfig(True, 1, 4)  # Default MoE: TP=1, DP=4, EP=4

    return ModelConfig(False, 1, 1)  # Default dense: single GPU


def get_recommended_parallelism(model_name: str) -> tuple[int, int]:
    """Return (tensor_parallel_size, data_parallel_size) for model.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Tuple of (TP, DP) sizes
    """
    config = get_model_config(model_name)
    return config.recommended_tp, config.recommended_dp
