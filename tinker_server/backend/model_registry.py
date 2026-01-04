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
        - train_cp: Context parallelism (shards sequence for long context)
        - train_etp: Expert tensor parallelism (None = use TP, 1 = no expert splitting)

    MoE Parallel Folding:
        When both train_ep > 1 and train_cp > 1, MoE Parallel Folding is used.
        This folds CP and EP onto the same GPU ranks:
        - Attention uses CP groups (sequence sharded)
        - MoE uses EP groups (experts distributed)
        - world_size = TP * max(EP, CP) instead of TP * EP * CP

    Memory configuration:
        - gpu_memory_utilization: Override for vLLM memory utilization (None = use global default 0.85).
          Large MoE models (K2) need lower utilization to leave room for LoRA buffers.

    LoRA configuration:
        - max_loras: Override max number of LoRAs (None = use default, 0 = disable LoRA).
          K2 with 384 experts cannot fit LoRA buffers on 8 GPUs.
        - max_lora_rank: Override max LoRA rank for inference (None = use global default).
          Large MoE models (K2) need lower rank to fit LoRA buffers in GPU memory.
    """

    is_moe: bool
    recommended_tp: int  # vLLM inference TP
    recommended_dp: int  # vLLM inference DP
    train_tp: int = 1  # Megatron training TP (attention/shared layers)
    train_ep: int = 1  # Megatron training EP (MoE expert distribution, 1 for dense)
    train_cp: int = 1  # Megatron training CP (context parallelism for long sequences)
    train_etp: int | None = None  # Expert tensor parallelism (None = use TP, 1 = no expert splitting)
    quantization: str | None = None  # vLLM quantization: None (auto-detect), "fp8", "compressed-tensors", etc.
    gpu_memory_utilization: float | None = None  # None = use global default (0.85), or override for large models
    max_loras: int | None = None  # None = use default (1 for MoE, 64 for dense), 0 = disable LoRA
    max_lora_rank: int | None = None  # None = use global default, or override for large models
    max_model_len: int | None = None  # None = use model's default, or override for large models with KV cache constraints
    max_num_seqs: int | None = None  # None = use default (256), or lower for large MoE models with KV cache constraints
    kv_cache_dtype: str | None = None  # None = use model's default, "fp8_e5m2" halves KV cache memory

    @property
    def total_gpus(self) -> int:
        """Total GPUs for inference."""
        return self.recommended_tp * self.recommended_dp

    @property
    def train_gpus(self) -> int:
        """Total GPUs for training.

        MoE Parallel Folding cases:
        1. EP > TP with ETP < TP: world_size = EP
           (TP is a subgroup for attention within EP dimension)
        2. CP > 1 and EP > 1: world_size = TP * max(EP, CP)
           (CP and EP share GPU ranks)
        3. Traditional: world_size = TP * EP * CP
        """
        etp = self.train_etp if self.train_etp is not None else self.train_tp

        if self.train_ep > self.train_tp and etp < self.train_tp:
            # MoE Parallel Folding with ETP: TP is subgroup within EP
            return self.train_ep * self.train_cp
        elif self.train_ep > 1 and self.train_cp > 1:
            # CP/EP Folding: CP and EP share GPU ranks
            return self.train_tp * max(self.train_ep, self.train_cp)
        else:
            # Traditional: all dimensions orthogonal
            return self.train_tp * self.train_ep * self.train_cp


# Supported models - only these are allowed
MODEL_CONFIGS = {
    # Dense models (train_tp=1, train_ep=1 - uses PEFT backend)
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        is_moe=False, recommended_tp=1, recommended_dp=1, train_tp=1, train_ep=1
    ),
    "Qwen/Qwen3-0.6B": ModelConfig(
        is_moe=False, recommended_tp=1, recommended_dp=1, train_tp=1, train_ep=1
    ),
    "Qwen/Qwen3-4B-Instruct-2507": ModelConfig(
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
    "Qwen/Qwen3-30B-A3B-Thinking-2507": ModelConfig(
        is_moe=True, recommended_tp=4, recommended_dp=1, train_tp=4, train_ep=1
    ),
    # Kimi K2 - 1.04T param MoE (384 experts × 61 layers, 8 active per token)
    # Architecture: hidden=7168, moe_intermediate=2048 per expert
    # PROMPT.md settings for K2-Instruct (same as K2-Thinking):
    # - Megatron: TP=16, EP=64, ETP=1, lora_rank=16 (64 GPUs)
    # - vLLM: TP=32, max_lora_rank=16 (32 GPUs)
    # - Total: 96 GPUs
    # MoE Parallel Folding: world_size = EP = 64 GPUs (TP folds into EP)
    # 384 experts / 64 GPUs = 6 experts per GPU
    "moonshotai/Kimi-K2-Instruct": ModelConfig(
        is_moe=True,
        recommended_tp=32,  # Inference: TP=32 (PROMPT.md spec)
        recommended_dp=1,
        train_tp=16,  # Training: TP=16 (folds into EP)
        train_ep=64,  # Training: EP=64 (64 GPUs total)
        train_cp=1,  # Training: CP=1 (no context parallelism)
        train_etp=1,  # Expert tensor parallelism = 1 (each expert on 1 GPU)
        quantization=None,  # Let vLLM auto-detect from config.json
        gpu_memory_utilization=0.85,
        max_loras=1,  # LoRA REQUIRED for weight transfer
        max_lora_rank=16,  # Rank 16: matches training lora_rank
        max_model_len=65536,  # 64K context
    ),
    "moonshotai/Kimi-K2-Thinking": ModelConfig(
        is_moe=True,
        recommended_tp=32,  # Inference: TP=32 (PROMPT.md spec)
        recommended_dp=1,
        train_tp=16,  # Training: TP=16 (PROMPT.md: lora_rank=16 requires TP<=16)
        train_ep=64,  # Training: EP=64 (64 GPUs total)
        train_cp=1,  # Training: CP=1 (no context parallelism)
        train_etp=1,  # Expert tensor parallelism = 1 (each expert on 1 GPU)
        # PROMPT.md settings:
        # - Megatron: TP=16, EP=64, ETP=1, lora_rank=16 (64 GPUs)
        # - vLLM: TP=32, max_lora_rank=16 (32 GPUs)
        # - Total: 96 GPUs
        # MoE Parallel Folding: world_size = EP = 64 GPUs
        # 384 experts / 64 GPUs = 6 experts per GPU
        quantization=None,  # INT4 compressed-tensors, vLLM auto-detects
        gpu_memory_utilization=0.85,  # Lower util to avoid OOM on long sequences
        max_loras=1,  # LoRA for weight transfer
        max_lora_rank=16,  # Rank 16: matches training lora_rank
        max_model_len=65536,  # 64K context
    ),
    # Moonlight-16B-A3B - smaller K2-like model (64 experts, 27 layers)
    # Same DeepseekV3ForCausalLM architecture
    "moonshotai/Moonlight-16B-A3B-Instruct": ModelConfig(
        is_moe=True,
        recommended_tp=4,  # Inference: 4 GPUs
        recommended_dp=1,
        train_tp=2,
        train_ep=4,  # Training: 8 GPUs (TP=2, EP=4) - 64 experts / 4 = 16 per rank
        quantization=None,  # BF16, no quantization needed
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


def list_supported_models() -> list[str]:
    """Return list of all supported model names."""
    return list(MODEL_CONFIGS.keys())


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


def is_mla_model(model_name: str) -> bool:
    """Check if a model uses Multi-Latent Attention (MLA) architecture.

    MLA models (DeepSeek V3 / Moonlight / Kimi-K2) use different attention
    projections that require special handling for LoRA:
    - linear_q_proj, linear_kv_down_proj, linear_kv_up_proj instead of linear_qkv

    Args:
        model_name: HuggingFace model name

    Returns:
        True if model uses MLA, False otherwise

    Raises:
        ValueError: If model is not supported
    """
    # Normalize the model name first
    normalized = normalize_model_name(model_name)

    # MLA models: DeepSeek V3 architecture (Moonlight, Kimi-K2)
    mla_models = {
        "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Kimi-K2-Thinking",
        "moonshotai/Moonlight-16B-A3B-Instruct",
    }
    return normalized in mla_models


def get_recommended_parallelism(model_name: str) -> tuple[int, int]:
    """Return (tensor_parallel_size, data_parallel_size) for inference.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        Tuple of (TP, DP) sizes for vLLM inference
    """
    config = get_model_config(model_name)
    return config.recommended_tp, config.recommended_dp


def get_training_parallelism(model_name: str) -> tuple[int, int, int, int | None]:
    """Return (tensor_parallel_size, expert_parallel_size, context_parallel_size, expert_tensor_parallel_size) for training.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (TP, EP, CP, ETP) sizes for Megatron training.
        ETP is None if not specified (defaults to TP in Megatron).

    Note:
        When both EP > 1 and CP > 1, MoE Parallel Folding is used.
        world_size = TP * max(EP, CP) instead of TP * EP * CP.
    """
    config = get_model_config(model_name)
    return config.train_tp, config.train_ep, config.train_cp, config.train_etp


def get_quantization(model_name: str) -> str | None:
    """Get quantization method for model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Quantization method (None for auto-detect, "fp8", "compressed-tensors", etc.)
    """
    config = get_model_config(model_name)
    return config.quantization


def requires_fp8(model_name: str) -> bool:
    """Check if model requires FP8 quantization (deprecated, use get_quantization).

    Args:
        model_name: HuggingFace model name

    Returns:
        True if model uses FP8
    """
    return get_quantization(model_name) == "fp8"


def get_max_lora_rank(model_name: str) -> int | None:
    """Get per-model max LoRA rank override.

    Large MoE models (K2) need reduced LoRA rank to fit buffers in GPU memory.

    Args:
        model_name: HuggingFace model name

    Returns:
        Max LoRA rank for this model, or None to use global default.
    """
    config = get_model_config(model_name)
    return config.max_lora_rank


def get_max_model_len(model_name: str) -> int | None:
    """Get per-model max_model_len override.

    Large models (K2) need reduced max_model_len to fit KV cache in GPU memory.
    K2's default 262K context requires 17GB KV cache, but only ~9GB available at 80% util.

    Args:
        model_name: HuggingFace model name

    Returns:
        Max model length for this model, or None to use model's default.
    """
    config = get_model_config(model_name)
    return config.max_model_len


def get_max_num_seqs(model_name: str) -> int | None:
    """Get per-model max_num_seqs override.

    Large models (K2) with high max_model_len need reduced max_num_seqs to fit
    KV cache in GPU memory. With 8K context and 256 max_seqs, KV cache is massive.

    Args:
        model_name: HuggingFace model name

    Returns:
        Max concurrent sequences for this model, or None to use default (256).
    """
    config = get_model_config(model_name)
    return config.max_num_seqs


def get_max_loras(model_name: str) -> int | None:
    """Get per-model max_loras override.

    K2 with 384 experts requires max_loras=0 (LoRA disabled) because
    vLLM pre-allocates LoRA buffers for ALL experts, exceeding GPU memory.

    Args:
        model_name: HuggingFace model name

    Returns:
        Max LoRAs for this model (0 = disable), or None to use default.
    """
    config = get_model_config(model_name)
    return config.max_loras


def get_kv_cache_dtype(model_name: str) -> str | None:
    """Get per-model kv_cache_dtype override.

    FP8 KV cache (fp8_e5m2) halves KV cache memory usage, enabling
    larger context or batch sizes.

    Args:
        model_name: HuggingFace model name

    Returns:
        KV cache dtype for this model, or None to use model's default.
    """
    config = get_model_config(model_name)
    return config.kv_cache_dtype
