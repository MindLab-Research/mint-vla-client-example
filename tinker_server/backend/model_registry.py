"""Model configuration registry for hardware requirements."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hardware configuration for a model.

    Inference parallelism (vLLM):
        - inference_tp: Tensor parallelism (shards weights)
        - inference_dp: Data parallelism (replicas)

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
        - max_loras: Override max number of LoRAs (None = use default, 0 = disable LoRA).
        - max_cpu_loras: Override vLLM CPU LoRA cache size (None = vLLM default).
        - max_lora_rank: Override max LoRA rank for inference (None = use global default).
        - max_model_len: vLLM context limit (required for all models).

    Architecture flags:
        - is_mla: Uses Multi-Latent Attention (DeepSeek V3/Moonlight/K2)
    """

    is_moe: bool
    inference_tp: int  # vLLM inference TP
    inference_dp: int  # vLLM inference DP
    max_model_len: int  # vLLM context limit (required - no fallback)
    train_tp: int = 1  # Megatron training TP (attention/shared layers)
    train_ep: int = 1  # Megatron training EP (MoE expert distribution, 1 for dense)
    train_cp: int = 1  # Megatron training CP (context parallelism for long sequences)
    train_etp: int | None = None  # Expert tensor parallelism (None = use TP, 1 = no expert splitting)
    quantization: str | None = None  # vLLM quantization: None (auto-detect), "fp8", "compressed-tensors", etc.
    # vLLM memory settings
    gpu_memory_utilization: float | None = None  # None = use global default (0.85), or override for large models
    max_loras: int | None = None  # None = use default (1 for MoE, 64 for dense), 0 = disable LoRA
    max_cpu_loras: int | None = None  # None = vLLM default (max_cpu_loras=max_loras)
    max_lora_rank: int | None = None  # None = use global default, or override for large models
    max_model_len: int  # vLLM context limit (required - no fallback)
    max_num_seqs: int | None = None  # None = use default (256), or lower for large MoE models with KV cache constraints
    # prompt_logprobs/logits can spike memory; lower values reduce peak usage at the cost of speed.
    max_num_batched_tokens: int | None = None  # None = use engine default heuristic
    kv_cache_dtype: str | None = None  # None = use model's default, "fp8_e5m2" halves KV cache memory
    gradient_checkpointing: bool = False  # Enable for large dense models to reduce VRAM usage
    is_mla: bool = False  # Uses Multi-Latent Attention (DeepSeek V3 architecture)

    @property
    def total_gpus(self) -> int:
        """Total GPUs for inference."""
        return self.inference_tp * self.inference_dp

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
    # 7B+ models: gradient_checkpointing=True to avoid OOM with long sequences
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=32768,  # 32K context
        max_num_seqs=32,  # Leave vLLM headroom for 32K prompt_logprobs
        gradient_checkpointing=True,  # Required for sequences >5000 tokens
    ),
    "Qwen/Qwen3-0.6B": ModelConfig(
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=40960,  # 40K context
        max_num_seqs=64,  # Leave headroom for long-context prompt_logprobs
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-4B": ModelConfig(
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=40960,  # 40K context
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-4B-Instruct-2507": ModelConfig(
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=40960,  # 40K context (reduced from 256K for faster vLLM init)
        max_num_seqs=32,  # Leave headroom for long-context prompt_logprobs
        max_num_batched_tokens=2048,  # Cap prompt_logprobs peak allocations at long context
        gradient_checkpointing=True,  # Required for sequences >8000 tokens
    ),
    "Qwen/Qwen3-8B": ModelConfig(
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=40960,  # 40K context
        gradient_checkpointing=True,
    ),
    # MoE models - Qwen3 30B variants (40K context per model config)
    # Inference: TP=4, DP=1 (4 GPUs) - EP not supported in vLLM LoRA
    # Training: TP=4, EP=1 (4 GPUs) - reduced from TP=4,EP=2 for smaller clusters
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelConfig(
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=40960,  # 40K context - full model capability
        max_num_seqs=8,  # Constrain KV cache; prompt_logprobs needs headroom at long context
        max_num_batched_tokens=1024,  # Avoid multi-GB logits buffers during prompt_logprobs
        max_loras=8,
        max_cpu_loras=16,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B": ModelConfig(
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=40960,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B-Base": ModelConfig(
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=40960,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": ModelConfig(
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=40960,
        gradient_checkpointing=True,
    ),
    # Qwen3 235B MoE variants (235B total, 22B active)
    # Model config: num_attention_heads=64, num_key_value_heads=4 (GQA)
    # vLLM DP placement requires DP ranks colocated on one node (8 GPUs/node),
    # so DP=4 is not feasible. Use multi-node TP=16 instead (16 GPUs total).
    # Training: TP=4, EP=8 => 32 GPUs total
    "Qwen/Qwen3-235B-A22B-Instruct-2507": ModelConfig(
        is_moe=True,
        inference_tp=16,
        inference_dp=1,
        train_tp=4,
        train_ep=8,
        max_lora_rank=16,  # Match cookbook lora_rank=16; avoid MoE LoRA buffer blowup at rank=64
        max_model_len=32768,  # 32K context
        max_num_seqs=4,  # Constrain KV cache; prompt_logprobs needs extra headroom
        max_num_batched_tokens=512,  # prompt_logprobs memory spike at 32K
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": ModelConfig(
        is_moe=True,
        inference_tp=16,
        inference_dp=1,
        train_tp=4,
        train_ep=8,
        max_lora_rank=16,
        max_model_len=32768,  # 32K context
        gradient_checkpointing=True,
    ),
    # Kimi K2 - 1.04T param MoE (384 experts × 61 layers, 8 active per token)
    # Architecture: hidden=7168, moe_intermediate=2048 per expert
    # Uses MLA (Multi-Latent Attention) from DeepSeek V3 architecture
    # PROMPT.md settings for K2-Instruct (same as K2-Thinking):
    # - Megatron: TP=16, EP=64, ETP=1, lora_rank=16 (64 GPUs)
    # - vLLM: TP=32, max_lora_rank=16 (32 GPUs)
    # - Total: 96 GPUs
    # MoE Parallel Folding: world_size = EP = 64 GPUs (TP folds into EP)
    # 384 experts / 64 GPUs = 6 experts per GPU
    "moonshotai/Kimi-K2-Instruct": ModelConfig(
        is_moe=True,
        inference_tp=32,  # Inference: TP=32 (PROMPT.md spec)
        inference_dp=1,
        train_tp=16,  # Training: TP=16 (folds into EP)
        train_ep=64,  # Training: EP=64 (64 GPUs total)
        train_cp=1,  # Training: CP=1 (no context parallelism)
        train_etp=1,  # Expert tensor parallelism = 1 (each expert on 1 GPU)
        quantization=None,  # Let vLLM auto-detect from config.json
        gpu_memory_utilization=0.98,  # K2 uses 77 GiB/79 GiB, need high utilization
        max_loras=1,  # LoRA REQUIRED for weight transfer
        max_lora_rank=16,  # Rank 16: matches training lora_rank
        max_model_len=10240,  # Reduced from 64K to save GPU memory (train uses 8K)
        is_mla=True,  # DeepSeek V3 MLA architecture
    ),
    "moonshotai/Kimi-K2-Thinking": ModelConfig(
        is_moe=True,
        inference_tp=32,  # Inference: TP=32 (PROMPT.md spec)
        inference_dp=1,
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
        gpu_memory_utilization=0.98,  # K2 uses 77 GiB/79 GiB, need high utilization
        max_loras=1,  # LoRA for weight transfer
        max_lora_rank=16,  # Rank 16: matches training lora_rank
        max_model_len=10240,  # Reduced from 64K to save GPU memory (train uses 8K)
        is_mla=True,  # DeepSeek V3 MLA architecture
    ),
    # Moonlight-16B-A3B - smaller DeepSeek V3 MLA model (64 experts, 27 layers)
    # Merge gate settings:
    # - Megatron: TP=2, EP=4 (8 GPUs)
    # - vLLM: TP=4 (4 GPUs)
    "moonshotai/Moonlight-16B-A3B-Instruct": ModelConfig(
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        train_tp=2,
        train_ep=4,
        train_cp=1,
        quantization=None,  # BF16, no quantization needed
        max_loras=1,
        max_lora_rank=32,
        max_model_len=8192,  # 8K context
        is_mla=True,  # DeepSeek V3 MLA architecture
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


def get_training_parallelism(model_name: str) -> tuple[int, int, int, int]:
    """Get Megatron training parallelism settings for a model.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B") or full path

    Returns:
        Tuple of (train_tp, train_ep, train_cp, train_etp)
    """
    config = get_model_config(model_name)
    return (config.train_tp, config.train_ep, config.train_cp, config.train_etp)


def requires_fp8(model_name: str) -> bool:
    """Check if a model requires FP8 inference.

    Args:
        model_name: HuggingFace model name or full path

    Returns:
        True if the model requires FP8 inference
    """
    config = get_model_config(model_name)
    return config.quantization == "fp8"


def list_supported_models() -> list[str]:
    """Return list of supported model names."""
    allowed = [
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-0.6B",
    ]
    return [m for m in MODEL_CONFIGS.keys() if m in allowed]
