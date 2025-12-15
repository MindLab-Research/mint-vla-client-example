"""MegatronTrainingWorker - MoE training via verl's Megatron backend.

This module provides Tinker API compatibility for MoE model training using
verl's MegatronEngine, which handles:
- Expert Parallelism (EP) for MoE layers
- Tensor/Pipeline/Context Parallelism
- LoRA via Megatron-Bridge
- Offloading for memory efficiency
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import ray
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MegatronTrainingConfig:
    """Configuration for MegatronTrainingWorker.

    Translates Tinker API parameters to verl/Megatron config.
    """
    model_path: str
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    # Parallelism config - single process for now (TP=1 to avoid distributed)
    # TODO: Implement proper multi-process parallelism for 8 GPUs
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_size: int = 1
    # Offloading - enable to fit large models
    param_offload: bool = True
    optimizer_offload: bool = True
    grad_offload: bool = True
    # Training
    dtype: str = "bfloat16"
    seed: int = 42


def tinker_to_tensordict(
    data_items: list[dict],
    max_token_len_per_gpu: int = 8192,
    device: str | torch.device | None = None,
) -> TensorDict:
    """Convert Tinker Datum format to verl TensorDict.

    Tinker format:
        {
            "model_input": {"chunks": [{"tokens": [...]}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": [...]},
                "weights": {"data": [...]},
                "logprobs": {"data": [...]},      # for RL
                "advantages": {"data": [...]}     # for RL
            }
        }

    verl TensorDict format:
        TensorDict({
            "input_ids": tensor [batch, seq_len],
            "attention_mask": tensor [batch, seq_len],
            "loss_mask": tensor [batch, seq_len],
            "position_ids": tensor [batch, seq_len],  # 0 to seq_len-1
            "temperature": tensor [batch, 1],         # logits scaling (1.0 = no scaling)
            "old_log_probs": tensor [batch, seq_len],  # for RL
            "advantages": tensor [batch, seq_len],     # for RL
            # Non-tensor metadata for verl's prepare_micro_batches:
            "use_dynamic_bsz": NonTensorData(True),
            "max_token_len_per_gpu": NonTensorData(int),
        })

    Args:
        data_items: List of Tinker Datum dicts.
        max_token_len_per_gpu: Max tokens per GPU for dynamic batch sizing.
            Used by verl's prepare_micro_batches when use_dynamic_bsz=True.
        device: Target device for tensors. If None, uses CPU.
            Creating tensors directly on GPU avoids CPU-to-GPU copy issues with nested tensors.
    """
    input_ids_list = []
    attention_mask_list = []
    loss_mask_list = []
    old_log_probs_list = []
    advantages_list = []

    max_len = 0
    has_rl_inputs = False

    # First pass: collect data and find max length
    for item in data_items:
        model_input = item.get("model_input", {})
        loss_fn_inputs = item.get("loss_fn_inputs", {})

        # Extract input tokens
        chunks = model_input.get("chunks", [])
        if chunks and "tokens" in chunks[0]:
            tokens = chunks[0]["tokens"]
        else:
            continue

        max_len = max(max_len, len(tokens))
        input_ids_list.append(tokens)

        # Extract weights (loss mask)
        # Try "weights" first, then "mask"; if neither has data, default to all 1s
        weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("mask")
        if weights_data and weights_data.get("data"):
            weights = weights_data["data"]
        else:
            # Default: all tokens contribute to loss
            weights = [1.0] * len(tokens)
        loss_mask_list.append(weights)

        # RL inputs (optional)
        logprobs_data = loss_fn_inputs.get("logprobs", {})
        advantages_data = loss_fn_inputs.get("advantages", {})

        if logprobs_data.get("data"):
            has_rl_inputs = True
            old_log_probs_list.append(logprobs_data["data"])
        if advantages_data.get("data"):
            advantages_list.append(advantages_data["data"])

    if not input_ids_list:
        raise ValueError("No valid data items found")

    batch_size = len(input_ids_list)

    # verl expects nested/jagged tensors for variable-length sequences
    # Create tensors directly on target device (GPU) to avoid .to() issues
    # verl's forward_step calls batch.to(device) which fails for nested tensors on CPU
    input_ids_tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in input_ids_list]
    loss_mask_tensors = [torch.tensor(seq, dtype=torch.float, device=device) for seq in loss_mask_list]
    position_ids_tensors = [torch.arange(len(seq), dtype=torch.long, device=device) for seq in input_ids_list]

    # Create nested tensors with jagged layout
    input_ids = torch.nested.as_nested_tensor(input_ids_tensors, layout=torch.jagged)
    loss_mask = torch.nested.as_nested_tensor(loss_mask_tensors, layout=torch.jagged)
    position_ids = torch.nested.as_nested_tensor(position_ids_tensors, layout=torch.jagged)

    # TensorDict (don't pass device= as it triggers .to() on nested tensors)
    td = TensorDict({
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }, batch_size=[batch_size])

    # Temperature for logits scaling (1.0 = no scaling during training)
    # verl's forward_step does logits.div_(batch["temperature"])
    # Must be a scalar float to broadcast correctly with logits shape [?, ?, vocab_size]
    # Using set_non_tensor with float prevents batching/slicing by prepare_micro_batches
    td.set_non_tensor("temperature", 1.0)

    # Add RL inputs if present (also as nested tensors on target device)
    if has_rl_inputs and old_log_probs_list:
        old_log_probs_tensors = [torch.tensor(seq, dtype=torch.float, device=device) for seq in old_log_probs_list]
        td["old_log_probs"] = torch.nested.as_nested_tensor(old_log_probs_tensors, layout=torch.jagged)
    if advantages_list:
        advantages_tensors = [torch.tensor(seq, dtype=torch.float, device=device) for seq in advantages_list]
        td["advantages"] = torch.nested.as_nested_tensor(advantages_tensors, layout=torch.jagged)

    # Add non-tensor metadata for verl's prepare_micro_batches
    # These control how verl splits the batch into micro-batches
    td["use_dynamic_bsz"] = NonTensorData(True)
    td["max_token_len_per_gpu"] = NonTensorData(max_token_len_per_gpu)

    # Add fields required by verl's sft_loss
    # Compute total tokens in batch for loss normalization
    # Use set_non_tensor to avoid batch dimension validation (these are scalar values)
    total_tokens = sum(len(seq) for seq in input_ids_list)
    td.set_non_tensor("dp_size", 1)  # Single data parallel group
    td.set_non_tensor("batch_num_tokens", total_tokens)

    return td


def create_sft_loss_fn(return_logprobs: bool = True) -> Callable:
    """Create SFT loss function that also extracts per-token log_probs.

    verl's postprocess_micro_batch_func calls:
        loss, metrics = loss_function(model_output=..., data=..., dp_group=...)

    Args:
        return_logprobs: If True, include log_probs in metrics (for cookbook train_nll)

    Returns:
        Loss function compatible with verl's forward_backward_batch.
    """
    def sft_loss_with_logprobs(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """SFT cross-entropy loss that also returns log_probs for metrics.

        Args:
            model_output: dict with "log_probs" key containing log probabilities
            data: TensorDict with loss_mask, dp_size, batch_num_tokens
            dp_group: data parallel group (unused)

        Returns:
            Tuple of (loss_tensor, metrics_dict_with_logprobs)
        """
        log_probs = model_output.get("log_probs")

        if log_probs is None:
            loss = torch.tensor(0.0)
            return loss, {"error": "no_log_probs", "num_tokens": 0}

        # Handle NestedTensor format from verl (NO_PADDING mode)
        if hasattr(log_probs, 'values'):
            log_probs_flat = log_probs.values()
        else:
            log_probs_flat = log_probs

        # Get loss_mask to identify which tokens contribute to loss
        loss_mask = data.get("loss_mask")
        if loss_mask is not None and hasattr(loss_mask, 'values'):
            loss_mask_flat = loss_mask.values()
        elif loss_mask is not None:
            loss_mask_flat = loss_mask
        else:
            # Default: all tokens contribute
            loss_mask_flat = torch.ones_like(log_probs_flat, dtype=torch.bool)

        # Ensure types are compatible
        loss_mask_float = loss_mask_flat.float()

        # Compute SFT loss: -sum(log_probs * mask) / sum(mask)
        # log_probs are already the target token log probs from verl's forward
        weighted_log_probs = log_probs_flat * loss_mask_float
        num_tokens = loss_mask_float.sum()

        if num_tokens > 0:
            nll = -weighted_log_probs.sum() / num_tokens
        else:
            nll = -weighted_log_probs.sum()

        # Clone log_probs for metrics (detach to avoid affecting gradients)
        log_probs_cpu = log_probs_flat.detach().cpu()

        metrics = {
            "loss": nll.detach(),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, 'item') else int(num_tokens),
        }

        if return_logprobs:
            metrics["log_probs"] = log_probs_cpu

        return nll, metrics

    return sft_loss_with_logprobs


def create_loss_fn() -> Callable:
    """Create default loss function for warmup/initialization.

    Returns SFT loss function as the default for forward-backward warmup.
    """
    return create_sft_loss_fn(return_logprobs=False)


def create_ppo_loss_fn(epsilon: float = 0.2) -> Callable:
    """Create PPO loss function compatible with verl's model_output format.

    verl's postprocess_micro_batch_func calls:
        loss, metrics = loss_function(model_output=..., data=..., dp_group=...)

    verl's model_output is a dict with:
        - "log_probs": log probabilities from model forward (already computed)
        - "entropy": (optional) entropy values

    Args:
        epsilon: Clipping parameter for PPO.
    """
    def ppo_loss_fn(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """PPO clipped objective loss.

        Args:
            model_output: dict with "log_probs" key containing log probabilities
            data: TensorDict with input_ids, loss_mask, old_log_probs, advantages
            dp_group: data parallel group (may be used for normalization)

        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        # Extract log_probs from model output (verl provides this, not raw logits)
        new_log_probs = model_output["log_probs"]

        # Handle nested tensor format from verl (NO_PADDING mode)
        if hasattr(new_log_probs, 'values'):
            new_log_probs = new_log_probs.values()

        # Get old log probs and advantages from data
        old_log_probs = data["old_log_probs"]
        advantages = data["advantages"]
        loss_mask = data["loss_mask"]

        # Handle nested tensors (verl NO_PADDING mode)
        if hasattr(old_log_probs, 'values'):
            old_log_probs = old_log_probs.values()
        if hasattr(advantages, 'values'):
            advantages = advantages.values()
        if hasattr(loss_mask, 'values'):
            loss_mask = loss_mask.values()

        loss_mask = loss_mask.float()

        # Note: No shift needed for RL training
        # Client sends aligned old_log_probs and loss_mask for response tokens
        # verl's new_log_probs are already aligned with old_log_probs

        # Importance ratio
        log_ratio = new_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        # PPO clipped objective
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        pg_loss = torch.maximum(pg_loss1, pg_loss2)

        # Weighted average
        masked_loss = pg_loss * loss_mask
        num_tokens = loss_mask.sum()

        # Get dp_size and batch_num_tokens for proper normalization
        # TensorDict.get() may return NonTensorData wrappers, extract .data attribute
        dp_size = data.get("dp_size", 1)
        if hasattr(dp_size, 'data'):
            dp_size = dp_size.data
        batch_num_tokens = data.get("batch_num_tokens", num_tokens)
        if hasattr(batch_num_tokens, 'data'):
            batch_num_tokens = batch_num_tokens.data

        # Convert to Python scalar for comparison
        if hasattr(batch_num_tokens, 'item'):
            batch_num_tokens_scalar = batch_num_tokens.item()
        else:
            batch_num_tokens_scalar = float(batch_num_tokens)

        if batch_num_tokens_scalar > 0:
            loss = masked_loss.sum() / batch_num_tokens * dp_size
        else:
            loss = masked_loss.sum()

        # Clip fraction metric
        clipped = ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).float()
        clip_frac = (clipped * loss_mask).sum() / max(num_tokens, 1)
        ratio_mean = (ratio * loss_mask).sum() / max(num_tokens, 1)

        metrics = {
            "loss": loss.detach().item(),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, 'item') else int(num_tokens),
            "clip_frac": clip_frac.detach().item() if hasattr(clip_frac, 'item') else float(clip_frac),
            "ratio_mean": ratio_mean.detach().item() if hasattr(ratio_mean, 'item') else float(ratio_mean),
        }
        return loss, metrics

    return ppo_loss_fn


def create_logprob_extractor_fn() -> Callable:
    """Create a function that extracts per-token log_probs from model forward.

    Used for Chat SL and DPO which need per-token log probabilities
    rather than aggregate loss. verl computes log_probs during forward
    and passes them to the loss function.

    Returns a function that returns:
        - Zero loss (no training signal)
        - Metrics dict containing per-token log_probs
    """
    def logprob_extractor(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """Extract per-token log_probs from model_output.

        Args:
            model_output: dict with "log_probs" containing per-token log probabilities
            data: TensorDict with input_ids, loss_mask, etc.
            dp_group: data parallel group (unused)

        Returns:
            Tuple of (zero_loss, metrics_with_logprobs)
        """
        log_probs = model_output.get("log_probs")

        if log_probs is None:
            # Fallback: model didn't compute log_probs
            loss = torch.tensor(0.0)
            return loss, {"error": "no_log_probs", "num_tokens": 0}

        # Handle NestedTensor format from verl (NO_PADDING mode)
        if hasattr(log_probs, 'values'):
            log_probs = log_probs.values()

        # Get loss_mask to identify response tokens
        loss_mask = data.get("loss_mask")
        if loss_mask is not None and hasattr(loss_mask, 'values'):
            loss_mask = loss_mask.values()

        # Clone and detach log_probs
        log_probs_cpu = log_probs.detach().cpu()

        # Compute aggregate loss for compatibility (NLL)
        if loss_mask is not None:
            loss_mask_float = loss_mask.float() if loss_mask is not None else torch.ones_like(log_probs)
            nll = -(log_probs * loss_mask_float).sum()
            num_tokens = loss_mask_float.sum().item()
            if num_tokens > 0:
                nll = nll / num_tokens
        else:
            nll = -log_probs.mean()
            num_tokens = log_probs.numel()

        # Return log_probs in metrics
        metrics = {
            "loss": nll.detach().item() if hasattr(nll, 'item') else float(nll),
            "num_tokens": int(num_tokens),
            "log_probs": log_probs_cpu,  # Per-token log probabilities tensor
        }
        return nll, metrics

    return logprob_extractor


@ray.remote(num_gpus=1)  # TODO: Implement multi-GPU parallelism
class MegatronTrainingWorker:
    """Ray actor for MoE training via verl's Megatron backend.

    Provides Tinker API compatibility:
    - forward_backward(data_items, loss_fn) -> {loss_fn_outputs, metrics}
    - optim_step(learning_rate) -> {metrics}
    - get_lora_state_dict() -> {name: tensor}
    """

    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        config: MegatronTrainingConfig | None = None,
    ):
        """Initialize Megatron training worker.

        Currently runs single-GPU without parallelism. Multi-GPU parallelism
        requires launching distributed processes (torchrun) which is not yet
        implemented for Ray actors.

        Args:
            base_model: HuggingFace model path or local path.
            lora_rank: LoRA adapter rank.
            learning_rate: Initial learning rate.
            config: Optional full MegatronTrainingConfig.
        """
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate

        if config is None:
            config = MegatronTrainingConfig(
                model_path=base_model,
                lora_rank=lora_rank,
                learning_rate=learning_rate,
            )
        self.config = config

        # Will be set during initialization
        self.engine = None
        self.bridge = None
        self._step_count = 0

        # Initialize the Megatron backend
        self._initialize_megatron()

        logger.info(f"[MegatronTrainingWorker] Ready with model={base_model}, lora_rank={lora_rank}")

    def _initialize_megatron(self):
        """Initialize verl's MegatronEngine.

        This sets up:
        1. Distributed process group
        2. Model parallel groups (TP, PP, EP, CP)
        3. Model with LoRA via Megatron-Bridge
        4. Optimizer with offloading
        """
        # Import verl components
        from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
        from verl.trainer.config import CheckpointConfig
        from verl.utils.fs import copy_to_local
        from verl.utils.torch_dtypes import PrecisionType

        # Initialize distributed if not already done
        # For Ray actor running single-process multi-GPU, we set up minimal distributed env
        if not torch.distributed.is_initialized():
            # Set required env vars for torch.distributed if not present
            if "RANK" not in os.environ:
                os.environ["RANK"] = "0"
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = "1"
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = "0"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

        # Copy model to local if needed
        local_path = copy_to_local(self.base_model)

        # Build HFModelConfig
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True, local_files_only=True)

        model_config = HFModelConfig(
            path=self.base_model,  # HuggingFace model name for tokenizer
            local_path=local_path,
            hf_config=hf_config,
            architectures=hf_config.architectures,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules="all-linear",
            trust_remote_code=True,
        )

        # Build McoreEngineConfig
        engine_config = McoreEngineConfig(
            tensor_model_parallel_size=self.config.tensor_parallel_size,
            pipeline_model_parallel_size=self.config.pipeline_parallel_size,
            expert_model_parallel_size=self.config.expert_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
            param_offload=self.config.param_offload,
            optimizer_offload=self.config.optimizer_offload,
            grad_offload=self.config.grad_offload,
            dtype=self.config.dtype,
            seed=self.config.seed,
            use_mbridge=True,
            use_distributed_optimizer=True,
        )

        # Build McoreOptimizerConfig
        # Use constant LR decay style for online learning (no fixed schedule)
        optimizer_config = McoreOptimizerConfig(
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            clip_grad=1.0,
            lr_decay_steps=100000,  # Large value for online learning
            lr_decay_style="constant",  # Don't decay learning rate
            lr_warmup_steps=0,
        )

        # Build CheckpointConfig (minimal)
        checkpoint_config = CheckpointConfig()

        # Create and initialize the engine
        # Use MegatronEngineWithLMHead which implements forward_step for LM training
        self.engine = MegatronEngineWithLMHead(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        self.engine.initialize()

        # Store bridge reference for weight export
        self.bridge = self.engine.bridge

        logger.info("[MegatronTrainingWorker] MegatronEngineWithLMHead initialized")

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> dict:
        """Forward + backward pass via MegatronEngine.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type ("cross_entropy", "importance_sampling", "ppo").
            loss_fn_config: Optional config (e.g., {"epsilon": 0.2} for PPO).

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
        loss_fn_config = loss_fn_config or {}

        # Convert data to TensorDict
        data = tinker_to_tensordict(data_items)
        device = f"cuda:{torch.distributed.get_rank() % 8}"
        data = data.to(device)

        # Select loss function
        if loss_fn == "cross_entropy":
            loss_function = create_sft_loss_fn()
        elif loss_fn == "ppo":
            epsilon = loss_fn_config.get("epsilon", 0.2)
            loss_function = create_ppo_loss_fn(epsilon)
        elif loss_fn == "importance_sampling":
            # Importance sampling is PPO without clipping
            loss_function = create_ppo_loss_fn(epsilon=float("inf"))
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        # Zero gradients
        self.engine.optimizer_zero_grad()

        # Forward + backward via engine
        result = self.engine.forward_backward_batch(
            data=data,
            loss_function=loss_function,
            forward_only=False,
        )

        # Extract metrics from result
        # Result structure depends on verl internals
        loss_value = 0.0
        num_tokens = 0
        if result and len(result) > 0:
            # Aggregate losses from micro batches
            for micro_result in result:
                if isinstance(micro_result, dict):
                    loss_value += micro_result.get("loss", 0.0)
                    num_tokens += micro_result.get("num_tokens", 0)

        metrics = {
            "loss:mean": float(loss_value),
            "num_samples:sum": float(len(data_items)),
            "num_tokens:sum": float(num_tokens),
        }

        # Add PPO-specific metrics
        if loss_fn == "ppo" and result:
            clip_frac = sum(r.get("clip_frac", 0) for r in result if isinstance(r, dict))
            ratio_mean = sum(r.get("ratio_mean", 1) for r in result if isinstance(r, dict))
            n = len([r for r in result if isinstance(r, dict)]) or 1
            metrics["clipfrac:mean"] = float(clip_frac / n)
            metrics["ratio:mean"] = float(ratio_mean / n)

        logger.info(f"[MegatronTrainingWorker] forward_backward ({loss_fn}): loss={loss_value:.4f}")

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": [],  # Simplified - logprobs computed on demand
            "metrics": metrics,
        }

    def forward(self, data_items: list[dict]) -> dict:
        """Forward pass only (no backward). Returns per-token logprobs.

        Args:
            data_items: List of Tinker Datum dicts.

        Returns:
            Dict with loss_fn_outputs (including per-token logprobs) and metrics.
        """
        data = tinker_to_tensordict(data_items)
        device = f"cuda:{torch.distributed.get_rank() % 8}"
        data = data.to(device)

        # Use logprob extractor to get per-token log probabilities
        loss_function = create_logprob_extractor_fn()

        # Forward only via engine
        with torch.no_grad():
            result = self.engine.forward_backward_batch(
                data=data,
                loss_function=loss_function,
                forward_only=True,
            )

        # Extract per-token log_probs from result
        loss_value = 0.0
        num_tokens = 0
        all_log_probs = []
        loss_fn_outputs = []

        if result and len(result) > 0:
            for micro_result in result:
                if isinstance(micro_result, dict):
                    loss = micro_result.get("loss", 0.0)
                    if hasattr(loss, "item"):
                        loss = loss.item()
                    loss_value += float(loss)

                    tokens = micro_result.get("num_tokens", 0)
                    if hasattr(tokens, "item"):
                        tokens = tokens.item()
                    num_tokens += int(tokens)

                    log_probs = micro_result.get("log_probs")
                    if log_probs is not None:
                        if hasattr(log_probs, "cpu"):
                            log_probs = log_probs.cpu()
                        all_log_probs.append(log_probs)
                        # Note: use "logprobs" (no underscore) for cookbook NLL evaluator compatibility
                        loss_fn_outputs.append({
                            "loss": {"data": [float(loss)], "shape": [1], "dtype": "float32"},
                            "logprobs": log_probs.tolist() if hasattr(log_probs, "tolist") else list(log_probs),
                        })

        # Combine log_probs if multiple micro-batches
        log_probs_data = None
        if all_log_probs:
            combined = torch.cat(all_log_probs, dim=0) if len(all_log_probs) > 1 else all_log_probs[0]
            log_probs_data = {
                "data": combined.tolist(),
                "shape": list(combined.shape),
                "dtype": str(combined.dtype),
            }

        logger.info(f"[MegatronTrainingWorker] forward: loss={loss_value:.4f}, log_probs={'present' if log_probs_data else 'none'}")

        return {
            "loss_fn_output_type": "logprob_extractor",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": {
                "loss:mean": float(loss_value),
                "num_samples:sum": float(len(data_items)),
                "num_tokens:sum": float(num_tokens),
            },
            "log_probs": log_probs_data,
        }

    def optim_step(self, learning_rate: float | None = None) -> dict:
        """Optimizer step.

        Args:
            learning_rate: Optional new learning rate.

        Returns:
            Dict with metrics.
        """
        # Update learning rate if provided
        # Note: verl's optimizer handles LR scheduling differently
        # For now, we skip dynamic LR updates

        # Optimizer step via engine
        grad_norm = self.engine.optimizer_step()

        # LR scheduler step
        current_lr = self.engine.lr_scheduler_step()

        self._step_count += 1

        logger.info(f"[MegatronTrainingWorker] optim_step: grad_norm={grad_norm:.4f}, step={self._step_count}")

        return {
            "metrics": {
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "step": self._step_count,
                "lr": float(current_lr[0]) if current_lr else self.learning_rate,
            },
            "type": "optim_step",
        }

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract LoRA adapter weights in PEFT format.

        Uses bridge.export_weights() and filters for LoRA parameters.
        Converts mbridge HuggingFace names to PEFT format for vLLM compatibility.

        mbridge format: layers.0.self_attn.q_proj.lora_A.weight
        PEFT format:    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

        Returns:
            Dict mapping LoRA parameter names (PEFT format) to CPU tensors.
        """
        if self.bridge is None:
            raise RuntimeError("Bridge not initialized - cannot export weights")

        # Export all weights via bridge
        full_state_dict = dict(self.bridge.export_weights(self.engine.module))

        # Filter for LoRA parameters and convert to PEFT format
        lora_state_dict = {}
        for name, tensor in full_state_dict.items():
            if "lora" in name.lower():
                # Convert mbridge HuggingFace format to PEFT format
                peft_name = f"base_model.model.model.{name}"
                # Move to CPU for Ray serialization
                lora_state_dict[peft_name] = tensor.cpu() if tensor.is_cuda else tensor

        logger.info(f"[MegatronTrainingWorker] Extracted {len(lora_state_dict)} LoRA parameters (PEFT format)")
        if lora_state_dict:
            sample_keys = list(lora_state_dict.keys())[:3]
            logger.debug(f"[MegatronTrainingWorker] Sample LoRA keys: {sample_keys}")

        return lora_state_dict

    def get_lora_config(self) -> dict:
        """Get LoRA configuration as dictionary.

        Returns:
            PEFT config dict compatible with vLLM's PEFTHelper.
        """
        return {
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }

    def get_tokenizer_info(self) -> dict:
        """Return tokenizer configuration.

        Returns:
            Dict with tokenizer info.
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, local_files_only=True)

        return {
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        }


    def save_checkpoint(self, save_path: str) -> dict:
        """Save checkpoint: LoRA weights + config + training metadata.

        For distributed Megatron training, optimizer state is sharded across ranks
        and not saved here. Only rank 0 saves the checkpoint.

        Args:
            save_path: Directory path to save checkpoint files.

        Returns:
            Dict with training metadata.
        """
        import json
        import os

        from safetensors.torch import save_file

        os.makedirs(save_path, exist_ok=True)

        # 1. LoRA weights
        state_dict = self.get_lora_state_dict()
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # 2. LoRA config
        config = self.get_lora_config()
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 3. Training metadata
        meta = {
            "current_step": self._step_count,
            "learning_rate": self.learning_rate,
        }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info(f"[MegatronTrainingWorker] Saved checkpoint to {abs_path} (step={self._step_count})")
        return meta

    def shutdown(self) -> None:
        """Release resources."""
        logger.info("[MegatronTrainingWorker] Shutting down")
        # MegatronEngine cleanup handled by garbage collection
        self.engine = None
        self.bridge = None
        torch.cuda.empty_cache()


def is_moe_model(model_name: str) -> bool:
    """Check if a model is an MoE model requiring Megatron training.

    Args:
        model_name: Model name or path (e.g., "Qwen/Qwen3-30B-A3B").

    Returns:
        True if model is MoE and should use MegatronTrainingWorker.
    """
    import re

    # Qwen3 MoE pattern: Qwen3-*-A*B (e.g., Qwen3-30B-A3B)
    # The "A" followed by a number indicates active parameters (MoE)
    moe_patterns = [
        r"Qwen3-\d+B-A\d+B",  # Qwen3 MoE (e.g., Qwen3-30B-A3B)
        r"Mixtral",           # Mixtral models
        r"DeepSeek.*MoE",     # DeepSeek MoE
    ]

    for pattern in moe_patterns:
        if re.search(pattern, model_name, re.IGNORECASE):
            return True

    return False
