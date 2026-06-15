"""Model configuration registry for hardware requirements."""

import json
import logging
import os
import types
from dataclasses import dataclass, replace
from typing import Any, Literal, get_args, get_origin

from mint_server.runtime_env import env_get

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Hardware configuration for a model.

    Model parameters:
        - num_parameters: Total parameter count in billions (e.g., 30.0 for 30B model)

    Inference parallelism (vLLM):
        - inference_tp: Tensor parallelism (shards weights)
        - inference_pp: Pipeline parallelism (stages)
        - inference_dp: Data parallelism (replicas)

    Training parallelism (Megatron):
        - train_tp: Tensor parallelism
        - train_pp: Pipeline parallelism
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
    NCCL flags:
        - train_nccl_ib_disable: Set NCCL_IB_DISABLE=1 for Megatron ranks (multi-node stability).
    """

    num_parameters: float  # Total parameter count in billions
    is_moe: bool
    inference_tp: int  # vLLM inference TP
    inference_dp: int  # vLLM inference DP
    max_model_len: int  # vLLM context limit (required - no fallback)
    inference_pp: int = 1  # vLLM inference PP
    train_tp: int = 1  # Megatron training TP (attention/shared layers)
    train_pp: int = 1  # Megatron training PP (pipeline parallelism)
    train_ep: int = 1  # Megatron training EP (MoE expert distribution, 1 for dense)
    train_cp: int = 1  # Megatron training CP (context parallelism for long sequences)
    train_etp: int | None = None  # Expert tensor parallelism (None = use TP, 1 = no expert splitting)
    quantization: str | None = None  # vLLM quantization: None (auto-detect), "fp8", "compressed-tensors", etc.
    train_use_fp8: bool = False  # Enable FP8 params during Megatron init (memory savings)
    train_nccl_ib_disable: bool = False  # Set NCCL_IB_DISABLE=1 for Megatron ranks
    # vLLM memory settings
    gpu_memory_utilization: float | None = None  # None = use global default (0.85), or override for large models
    max_loras: int | None = None  # None = use default (1 for MoE, 64 for dense), 0 = disable LoRA
    max_cpu_loras: int | None = None  # None = vLLM default (max_cpu_loras=max_loras)
    max_lora_rank: int | None = None  # None = use global default, or override for large models
    max_num_seqs: int | None = None  # None = use default (256), or lower for large MoE models with KV cache constraints
    # prompt_logprobs/logits can spike memory; lower values reduce peak usage at the cost of speed.
    max_num_batched_tokens: int | None = None  # None = use engine default heuristic
    kv_cache_dtype: str | None = None  # None = use model's default, "fp8_e5m2" halves KV cache memory
    gradient_checkpointing: bool = False  # Enable for large dense models to reduce VRAM usage
    is_mla: bool = False  # Uses Multi-Latent Attention (DeepSeek V3 architecture)
    # vLLM engine selection:
    # - "verl_http": verl's single-node HTTP-style vLLM server (default; used for 1-GPU small models)
    # - "async": direct vLLM AsyncLLMEngine wrapper (supports multi-GPU TP and per-token logprobs)
    vllm_engine: Literal["verl_http", "async"] = "verl_http"
    # Only used when vllm_engine="async":
    # - "mp": single-node multiprocessing TP (avoid Ray compiled DAG)
    # - "ray": Ray distributed executor (uses Ray compiled DAG; keep for K2)
    vllm_distributed_executor_backend: Literal["mp", "ray"] = "mp"
    # Cross-family dispatch metadata. Defaults preserve current text-model behavior.
    policy_family: Literal["text_lm", "ar_action_tokens", "flow_action"] = "text_lm"
    inference_modality: Literal["tokens", "actions"] = "tokens"
    supported_modalities: tuple[Literal["text", "image", "video"], ...] = ("text",)
    training_backend: str = "mint_text"
    camera_layout: tuple[str, ...] = ()
    action_dim: int | None = None
    action_horizon: int | None = None
    action_token_budget: int | None = None

    @property
    def total_gpus(self) -> int:
        """Total GPUs for inference."""
        return self.inference_tp * self.inference_pp * self.inference_dp

    @property
    def train_gpus(self) -> int:
        """Total GPUs for training.

        MoE Parallel Folding cases:
        1. EP >= TP with ETP < TP: world_size = EP
           (TP is a subgroup for attention within EP dimension)
        2. CP > 1 and EP > 1: world_size = TP * max(EP, CP)
           (CP and EP share GPU ranks)
        3. Traditional: world_size = TP * EP * CP
        """
        etp = self.train_etp if self.train_etp is not None else self.train_tp

        if self.train_ep >= self.train_tp and etp < self.train_tp:
            # MoE Parallel Folding with ETP: TP is subgroup within EP
            return self.train_ep * self.train_pp * self.train_cp
        elif self.train_ep > 1 and self.train_cp > 1:
            # CP/EP Folding: CP and EP share GPU ranks
            return self.train_tp * self.train_pp * max(self.train_ep, self.train_cp)
        else:
            # Traditional: all dimensions orthogonal
            return self.train_tp * self.train_pp * self.train_ep * self.train_cp


# Supported models - only these are allowed
MODEL_CONFIGS = {
    # Read-only OpenPI FAST reference profiles.
    # These are capability descriptors for Mint-owned integration code, not a claim that
    # the full runtime path is already enabled everywhere.
    "openpi/pi0-fast-libero-low-mem-finetune": ModelConfig(
        num_parameters=2.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        train_tp=1,
        train_ep=1,
        max_model_len=180,
        policy_family="ar_action_tokens",
        inference_modality="actions",
        training_backend="openpi_fast",
        camera_layout=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        action_dim=7,
        action_horizon=10,
        # Leave headroom so FAST action decoding can emit the full action suffix,
        # terminator, and EOS without truncating at the Mint-side cap.
        action_token_budget=64,
    ),
    "openpi/pi05-libero-low-mem-finetune": ModelConfig(
        num_parameters=3.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        train_tp=1,
        train_ep=1,
        max_model_len=200,
        policy_family="flow_action",
        inference_modality="actions",
        training_backend="openpi_pi05",
        camera_layout=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        action_dim=32,
        action_horizon=10,
    ),
    # Dense models (train_tp=1, train_ep=1 - uses PEFT backend)
    # 7B+ models: gradient_checkpointing=True to avoid OOM with long sequences
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        num_parameters=7.0,
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=32768,  # 32K context
        max_num_seqs=32,  # Leave vLLM headroom for 32K prompt_logprobs
        gradient_checkpointing=True,  # Required for sequences >5000 tokens
    ),
    "Qwen/Qwen3-0.6B": ModelConfig(
        num_parameters=0.6,
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=32768,  # 32K context
        max_num_seqs=608,  # 32 * N_long (N_long=19) for short-request headroom
        max_num_batched_tokens=1024,  # Prompt-logprob path chunks internally at 1024
        gpu_memory_utilization=0.90,
        max_loras=18,
        max_cpu_loras=180,
        max_lora_rank=64,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-4B": ModelConfig(
        num_parameters=4.0,
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=32768,  # 32K context
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-4B-Instruct-2507": ModelConfig(
        num_parameters=4.0,
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=32768,  # 32K context
        max_num_seqs=416,  # 32 * N_long (N_long=13) for short-request headroom
        max_num_batched_tokens=1024,  # Prompt-logprob path chunks internally at 1024
        gpu_memory_utilization=0.90,
        max_loras=12,
        max_cpu_loras=120,
        max_lora_rank=64,
        gradient_checkpointing=True,  # Required for sequences >8000 tokens
    ),
    "Qwen/Qwen3-4B-Thinking-2507": ModelConfig(
        num_parameters=4.0,
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=32768,  # 32K context
        max_num_seqs=416,  # Same dense serving/training envelope as 4B instruct
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.90,
        max_loras=12,
        max_cpu_loras=120,
        max_lora_rank=64,
        gradient_checkpointing=True,  # Required for sequences >8000 tokens
    ),
    "Qwen/Qwen3-8B": ModelConfig(
        num_parameters=8.0,
        is_moe=False, inference_tp=1, inference_dp=1, train_tp=1, train_ep=1,
        max_model_len=32768,  # 32K context
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3.5-27B": ModelConfig(
        num_parameters=27.0,
        is_moe=False,
        inference_tp=4,
        inference_dp=1,
        train_tp=4,
        train_ep=1,
        max_model_len=32768,
        max_num_seqs=128,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.90,
        max_loras=4,
        max_cpu_loras=16,
        max_lora_rank=64,
        gradient_checkpointing=True,
        vllm_engine="async",
        vllm_distributed_executor_backend="mp",
        supported_modalities=("text",),
    ),
    # MoE models - Qwen3 30B variants (40K context per model config)
    # Inference: TP=4, DP=1 (4 GPUs) - EP not supported in vLLM LoRA
    # Training: TP=4, EP=1 (4 GPUs) - reduced from TP=4,EP=2 for smaller clusters
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelConfig(
        num_parameters=30.0,
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=32768,  # 32K context
        # 32 * N_long (N_long=22) for short-request headroom. Keep
        # max_num_batched_tokens >= max_num_seqs for scheduler validity.
        max_num_seqs=704,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.85,  # Leave prompt_logprobs/runtime headroom above KV cache
        max_loras=21,
        max_cpu_loras=210,
        max_lora_rank=64,
        gradient_checkpointing=True,
        vllm_engine="async",
        vllm_distributed_executor_backend="mp",
    ),
    "Qwen/Qwen3-30B-A3B": ModelConfig(
        num_parameters=30.0,
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=32768,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B-Base": ModelConfig(
        num_parameters=30.0,
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=32768,
        gradient_checkpointing=True,
    ),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": ModelConfig(
        num_parameters=30.0,
        is_moe=True, inference_tp=4, inference_dp=1, train_tp=4, train_ep=1,
        max_model_len=32768,
        gradient_checkpointing=True,
    ),
    # Qwen3 235B MoE variants (235B total, 22B active)
    # Default profile is the current Volcano A800 shape:
    # - Inference: TP=16 via backend="ray" (16 GPUs)
    # - Training: TP=4, PP=1, EP=8 (32 GPUs)
    "Qwen/Qwen3-235B-A22B-Instruct-2507": ModelConfig(
        num_parameters=235.0,
        is_moe=True,
        inference_tp=16,
        inference_dp=1,
        train_tp=4,
        train_pp=1,
        train_ep=8,
        gpu_memory_utilization=0.75,
        max_loras=8,
        max_cpu_loras=80,
        max_lora_rank=64,
        max_model_len=32768,  # 32K context
        max_num_seqs=288,  # 32 * N_long (N_long=9) for short-request headroom
        max_num_batched_tokens=1024,  # Prompt-logprob path chunks internally at 1024
        gradient_checkpointing=True,
        vllm_engine="async",
        vllm_distributed_executor_backend="ray",
    ),
    "Qwen/Qwen3-235B-A22B-Thinking-2507": ModelConfig(
        num_parameters=235.0,
        is_moe=True,
        inference_tp=16,
        inference_dp=1,
        train_tp=4,
        train_pp=1,
        train_ep=8,
        gpu_memory_utilization=0.75,
        max_loras=8,
        max_cpu_loras=80,
        max_lora_rank=64,
        max_model_len=32768,  # 32K context
        max_num_seqs=288,  # 32 * N_long (N_long=9) for short-request headroom
        max_num_batched_tokens=1024,  # Prompt-logprob path chunks internally at 1024
        gradient_checkpointing=True,
        vllm_engine="async",
        vllm_distributed_executor_backend="ray",
    ),
    "zai-org/GLM-5": ModelConfig(
        num_parameters=355.0,
        is_moe=True,
        inference_tp=1,
        inference_dp=1,
        train_tp=4,
        train_pp=1,
        train_ep=8,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.90,
        max_loras=0,
        max_cpu_loras=0,
        max_lora_rank=64,
    ),
    # Kimi K2 - 1.04T param MoE (384 experts × 61 layers, 8 active per token)
    # Architecture: hidden=7168, moe_intermediate=2048 per expert
    # Uses MLA (Multi-Latent Attention) from DeepSeek V3 architecture
    # mint-prod deployment target:
    # - vLLM inference: TP=64, PP=1, DP=1 (64 GPUs total)
    # - Megatron training: TP=64, EP=64, CP=2, ETP=1 (128 GPUs total; EP-fold + CP)
    # Notes:
    # - vLLM DP>1 has failed on Volcano with "Not enough resources to allocate 2 DP ranks on DP master node".
    # - vLLM PP>1 for K2 has hit an init-time KeyError in vllm/config/vllm.py (forward_context lookup).
    # - LoRA rank must be divisible by TP (adapter sharding); keep max_lora_rank=64.
    # With 384 experts: EP=64 => 6 experts per GPU.
    "moonshotai/Kimi-K2-Instruct": ModelConfig(
        num_parameters=1040.0,
        is_moe=True,
        inference_tp=64,
        inference_pp=1,
        inference_dp=1,
        train_tp=64,
        train_ep=64,
        train_cp=2,
        train_etp=1,  # Expert tensor parallelism = 1 (each expert on 1 GPU)
        quantization=None,  # Let vLLM auto-detect from config.json
        train_use_fp8=False,
        train_nccl_ib_disable=False,
        # Leave headroom for vLLM MoE LoRA buffers (fused_moe creates weights at engine init).
        # We have observed OOM during LoRA weight creation (alloc ~336 MiB with ~58 MiB free) when
        # vLLM preallocates too aggressively.
        gpu_memory_utilization=0.87,
        max_loras=1,  # LoRA REQUIRED for weight transfer
        max_lora_rank=64,
        # Leave a small generation headroom above a 32k prompt without materially increasing KV cache size.
        # This avoids hard failures when prompt_target_tokens=32000 (effective_max_tokens would be 0).
        max_model_len=32768,
        max_num_seqs=8,  # Must be >= default SamplingParams(n=8)
        max_num_batched_tokens=1024,  # Cap logits/prefill peak allocations at long context
        is_mla=True,  # DeepSeek V3 MLA architecture
        vllm_engine="async",
        vllm_distributed_executor_backend="ray",
        gradient_checkpointing=True,  # Required to fit model with long context
    ),
    "unsloth/Kimi-K2-Instruct-0905-BF16": ModelConfig(
        num_parameters=1040.0,
        is_moe=True,
        inference_tp=64,
        inference_pp=1,
        inference_dp=1,
        train_tp=64,
        train_ep=64,
        train_cp=2,
        train_etp=1,
        quantization=None,
        train_use_fp8=False,
        train_nccl_ib_disable=False,
        gpu_memory_utilization=0.87,
        max_loras=1,
        max_lora_rank=64,
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        is_mla=True,
        vllm_engine="async",
        vllm_distributed_executor_backend="ray",
        gradient_checkpointing=True,
    ),
    "moonshotai/Kimi-K2-Thinking": ModelConfig(
        num_parameters=1040.0,
        is_moe=True,
        inference_tp=32,  # Inference: TP=32 (PROMPT.md spec)
        inference_dp=1,
        train_tp=16,  # Training: TP=16 (PROMPT.md: lora_rank=16 requires TP<=16)
        train_ep=64,  # Training: EP=64 (64 GPUs total)
        train_cp=2,  # Training: CP=2 (context parallelism)
        train_etp=1,  # Expert tensor parallelism = 1 (each expert on 1 GPU)
        # PROMPT.md settings:
        # - Megatron: TP=16, EP=64, CP=2, ETP=1, lora_rank=16 (128 GPUs)
        # - vLLM: TP=32, max_lora_rank=16 (32 GPUs)
        # - Total: 192 GPUs
        # MoE Parallel Folding: world_size = EP = 64 GPUs
        # 384 experts / 64 GPUs = 6 experts per GPU
        quantization=None,  # INT4 compressed-tensors, vLLM auto-detects
        train_use_fp8=False,
        train_nccl_ib_disable=True,
        gpu_memory_utilization=0.98,  # K2 uses 77 GiB/79 GiB, need high utilization
        max_loras=1,  # LoRA for weight transfer
        max_lora_rank=16,  # Rank 16: matches training lora_rank
        max_model_len=32768,  # Reduced from 64K to 32K to save GPU memory (train uses 8K)
        is_mla=True,  # DeepSeek V3 MLA architecture
        vllm_engine="async",
        vllm_distributed_executor_backend="ray",
        gradient_checkpointing=True,  # Required to fit model with long context
    ),
    # Moonlight-16B-A3B - smaller DeepSeek V3 MLA model (64 experts, 27 layers)
    # Merge gate settings:
    # - Megatron: TP=1, EP=4 (4 GPUs)
    # - vLLM: TP=4 (4 GPUs)
    "moonshotai/Moonlight-16B-A3B-Instruct": ModelConfig(
        num_parameters=16.0,
        is_moe=True,
        inference_tp=4,
        inference_dp=1,
        train_tp=1,
        train_ep=4,
        train_cp=1,
        quantization=None,  # BF16, no quantization needed
        max_loras=1,
        max_lora_rank=32,
        max_model_len=32768,  # 32K context
        is_mla=True,  # DeepSeek V3 MLA architecture
        gradient_checkpointing=True,
        vllm_engine="async",
        vllm_distributed_executor_backend="mp",
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

    raise ValueError(
        f"Cannot identify model from: {model_name_or_path}. "
        f"Supported models: {list(MODEL_CONFIGS.keys())}"
    )


def maybe_normalize_model_name(model_name_or_path: str) -> str | None:
    """Best-effort normalization for model matching.

    Returns a supported HF model name if it can be identified from a name/path,
    otherwise returns None.
    """
    if model_name_or_path in MODEL_CONFIGS:
        return model_name_or_path

    import re

    match = re.search(r"models--([^/]+)--([^/]+)", model_name_or_path)
    if match:
        org, model = match.groups()
        candidate = f"{org}/{model}"
        if candidate in MODEL_CONFIGS:
            return candidate

    return None


def _topology_model_names_from_env() -> set[str]:
    path = str(os.environ.get("MINT_TOPOLOGY_CONFIG_PATH") or "").strip()
    if not path:
        return set()
    try:
        import yaml

        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return set()
    if not isinstance(payload, dict):
        return set()
    models = payload.get("models") or {}
    if not isinstance(models, dict):
        return set()
    return {str(name).strip() for name in models if str(name).strip()}


def is_topology_desired_model(model_name_or_path: str) -> bool:
    raw = _topology_model_names_from_env()
    if not raw:
        return False
    normalized = {m for s in raw if (m := maybe_normalize_model_name(s)) is not None}
    model_ids = raw | normalized

    if model_name_or_path in model_ids:
        return True

    m = maybe_normalize_model_name(model_name_or_path)
    return m in model_ids if m is not None else False


def get_model_config(model_name: str) -> ModelConfig:
    """Get config for a supported model.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B") or full path

    Returns:
        ModelConfig with parallelism recommendations

    Raises:
        ValueError: If model is not in the supported list
    """
    def _int_env(name: str) -> int | None:
        v = os.environ.get(name)
        if v is None or v.strip() == "":
            return None
        try:
            return int(v)
        except ValueError:
            logger.warning(f"Ignoring invalid {name}={v!r} (expected int)")
            return None

    # Normalize path to model name
    normalized = normalize_model_name(model_name)
    cfg = MODEL_CONFIGS[normalized]

    overrides: dict[str, int] = {}
    for field, env_name in (
        ("max_num_seqs", "MINT_VLLM_MAX_NUM_SEQS"),
        ("max_num_batched_tokens", "MINT_VLLM_MAX_NUM_BATCHED_TOKENS"),
        ("max_loras", "MINT_VLLM_MAX_LORAS"),
        ("max_cpu_loras", "MINT_VLLM_MAX_CPU_LORAS"),
        ("max_lora_rank", "MINT_VLLM_MAX_LORA_RANK"),
    ):
        v = _int_env(env_name)
        if v is not None:
            overrides[field] = v

    if overrides:
        cfg = replace(cfg, **overrides)

    raw_overrides = (env_get(os.environ, "MINT_MODEL_CONFIG_OVERRIDES_JSON", "") or "").strip()
    if raw_overrides:
        import json

        data = json.loads(raw_overrides)
        if not isinstance(data, dict):
            raise ValueError("MINT_MODEL_CONFIG_OVERRIDES_JSON must be a JSON object mapping model_name to overrides")
        model_override = data.get(normalized)
        if model_override is not None:
            if not isinstance(model_override, dict):
                raise ValueError(f"MINT_MODEL_CONFIG_OVERRIDES_JSON[{normalized!r}] must be a JSON object")
            allowed_fields = set(ModelConfig.__dataclass_fields__.keys())
            unknown = sorted(set(model_override.keys()) - allowed_fields)
            if unknown:
                raise ValueError(f"MINT_MODEL_CONFIG_OVERRIDES_JSON[{normalized!r}] has unknown fields: {unknown}")

            def _type_ok(field: str, v: Any) -> bool:
                ann = ModelConfig.__annotations__.get(field)
                if ann is None:
                    return True

                def _ok(value: Any, annotation: Any) -> bool:
                    if annotation is Any:
                        return True
                    if annotation is type(None):
                        return value is None
                    origin = get_origin(annotation)
                    if origin in (types.UnionType, getattr(__import__("typing"), "Union")):
                        return any(_ok(value, a) for a in get_args(annotation))
                    if annotation is bool:
                        return isinstance(value, bool)
                    if annotation is int:
                        return isinstance(value, int) and not isinstance(value, bool)
                    if annotation is float:
                        return isinstance(value, (int, float)) and not isinstance(value, bool)
                    if annotation is str:
                        return isinstance(value, str)
                    return True

                return _ok(v, ann)

            bad: list[str] = []
            for k, v in model_override.items():
                if not _type_ok(k, v):
                    bad.append(f"{k}={v!r}")
            if bad:
                raise ValueError(f"MINT_MODEL_CONFIG_OVERRIDES_JSON[{normalized!r}] has invalid values: {bad}")

            cfg = replace(cfg, **model_override)
    return cfg


def get_training_parallelism(model_name: str) -> tuple[int, int, int, int, int]:
    """Get Megatron training parallelism settings for a model.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B") or full path

    Returns:
        Tuple of (train_tp, train_pp, train_ep, train_cp, train_etp)
    """
    config = get_model_config(model_name)
    return (config.train_tp, config.train_pp, config.train_ep, config.train_cp, config.train_etp)


def requires_fp8(model_name: str) -> bool:
    """Check if a model requires FP8 inference.

    Args:
        model_name: HuggingFace model name or full path

    Returns:
        True if the model requires FP8 inference
    """
    config = get_model_config(model_name)
    return config.quantization == "fp8"


def _gateway_supported_models() -> set[str]:
    raw = (env_get(os.environ, "MINT_GATEWAY_CONFIG_JSON", "") or "").strip()
    if not raw:
        return set()

    data = json.loads(raw)
    model_map = (
        data.get("model_to_upstream")
        or data.get("model_to_deployment_target")
        or data.get("model_to_target")
        or {}
    )
    if not isinstance(model_map, dict):
        raise ValueError("MINT_GATEWAY_CONFIG_JSON model routing must be a JSON object")
    return {str(name).strip() for name in model_map if str(name).strip()}


def list_supported_models() -> list[str]:
    """Return list of supported model names."""
    raw = (env_get(os.environ, "MINT_SUPPORTED_MODELS", "") or "").strip()
    if raw:
        items = [s.strip() for s in raw.split(",") if s.strip()]
        seen: set[str] = set()
        models: list[str] = []
        for m in items:
            if m in seen:
                continue
            seen.add(m)
            models.append(m)
        gateway_models = _gateway_supported_models()
        unknown = [m for m in models if m not in MODEL_CONFIGS and m not in gateway_models]
        if unknown:
            raise ValueError(f"Unsupported models in MINT_SUPPORTED_MODELS: {unknown}")
        return models

    allowed = [
        "openpi/pi0-fast-libero-low-mem-finetune",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-0.6B",
        "zai-org/GLM-5",
        "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Moonlight-16B-A3B-Instruct",
    ]
    return [m for m in allowed if m in MODEL_CONFIGS]
