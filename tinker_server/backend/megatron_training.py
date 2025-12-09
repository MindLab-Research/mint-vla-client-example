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


def tinker_to_tensordict(data_items: list[dict]) -> TensorDict:
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
            "old_log_probs": tensor [batch, seq_len],  # for RL
            "advantages": tensor [batch, seq_len],     # for RL
        })
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
        weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("mask", {})
        weights = weights_data.get("data", []) if weights_data else [1.0] * len(tokens)
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

    # Pad sequences to max_len
    def pad_sequence(seq_list: list[list], pad_value: float = 0.0) -> torch.Tensor:
        padded = []
        for seq in seq_list:
            if len(seq) < max_len:
                seq = seq + [pad_value] * (max_len - len(seq))
            padded.append(seq[:max_len])
        return torch.tensor(padded)

    input_ids = pad_sequence(input_ids_list, pad_value=0).long()
    attention_mask = (input_ids != 0).long()  # Simple attention mask
    loss_mask = pad_sequence(loss_mask_list, pad_value=0.0).float()

    td = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }, batch_size=[batch_size])

    # Add RL inputs if present
    if has_rl_inputs and old_log_probs_list:
        td["old_log_probs"] = pad_sequence(old_log_probs_list, pad_value=0.0).float()
    if advantages_list:
        td["advantages"] = pad_sequence(advantages_list, pad_value=0.0).float()

    return td


def create_sft_loss_fn() -> Callable:
    """Create cross-entropy loss function for SFT.

    verl expects loss functions that take (logits, data) and return loss dict.
    """
    def sft_loss_fn(logits: torch.Tensor, data: TensorDict) -> dict:
        """Cross-entropy loss with loss_mask weighting.

        Args:
            logits: [batch, seq_len, vocab_size]
            data: TensorDict with input_ids, loss_mask
        """
        # Shift for language modeling: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data["input_ids"][..., 1:].contiguous()
        shift_mask = data["loss_mask"][..., 1:].contiguous()

        # Flatten
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_mask = shift_mask.view(-1)

        # Cross-entropy per token
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="none"
        )

        # Weighted average
        masked_loss = ce_loss * shift_mask
        num_tokens = shift_mask.sum()

        if num_tokens > 0:
            loss = masked_loss.sum() / num_tokens
        else:
            loss = masked_loss.sum()

        return {"loss": loss, "num_tokens": num_tokens}

    return sft_loss_fn


def create_ppo_loss_fn(epsilon: float = 0.2) -> Callable:
    """Create PPO loss function.

    Args:
        epsilon: Clipping parameter for PPO.
    """
    def ppo_loss_fn(logits: torch.Tensor, data: TensorDict) -> dict:
        """PPO clipped objective loss.

        Args:
            logits: [batch, seq_len, vocab_size]
            data: TensorDict with input_ids, loss_mask, old_log_probs, advantages
        """
        # Shift for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data["input_ids"][..., 1:].contiguous()
        shift_mask = data["loss_mask"][..., 1:].contiguous()
        old_log_probs = data["old_log_probs"][..., 1:].contiguous()
        advantages = data["advantages"][..., 1:].contiguous()

        # Compute new log probs
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        new_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Importance ratio
        log_ratio = new_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        # PPO clipped objective
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        pg_loss = torch.maximum(pg_loss1, pg_loss2)

        # Weighted average
        masked_loss = pg_loss * shift_mask
        num_tokens = shift_mask.sum()

        if num_tokens > 0:
            loss = masked_loss.sum() / num_tokens
        else:
            loss = masked_loss.sum()

        # Clip fraction metric
        clipped = ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).float()
        clip_frac = (clipped * shift_mask).sum() / max(num_tokens, 1)

        return {
            "loss": loss,
            "num_tokens": num_tokens,
            "clip_frac": clip_frac,
            "ratio_mean": (ratio * shift_mask).sum() / max(num_tokens, 1),
        }

    return ppo_loss_fn


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
        from verl.workers.engine.megatron.transformer_impl import MegatronEngine
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
        self.engine = MegatronEngine(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        self.engine.initialize()

        # Store bridge reference for weight export
        self.bridge = self.engine.bridge

        logger.info("[MegatronTrainingWorker] MegatronEngine initialized")

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
        """Forward pass only (no backward). Returns logprobs.

        Args:
            data_items: List of Tinker Datum dicts.

        Returns:
            Dict with loss_fn_outputs (including logprobs) and metrics.
        """
        data = tinker_to_tensordict(data_items)
        device = f"cuda:{torch.distributed.get_rank() % 8}"
        data = data.to(device)

        loss_function = create_sft_loss_fn()

        # Forward only via engine
        result = self.engine.forward_backward_batch(
            data=data,
            loss_function=loss_function,
            forward_only=True,
        )

        logger.info("[MegatronTrainingWorker] forward completed")

        return {
            "loss_fn_output_type": "sft_loss",
            "loss_fn_outputs": [],
            "metrics": {
                "num_samples:sum": float(len(data_items)),
            },
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
        """Extract LoRA adapter weights.

        Uses bridge.export_weights() and filters for LoRA parameters.

        Returns:
            Dict mapping LoRA parameter names to CPU tensors.
        """
        if self.bridge is None:
            raise RuntimeError("Bridge not initialized - cannot export weights")

        # Export all weights via bridge
        full_state_dict = dict(self.bridge.export_weights(self.engine.module))

        # Filter for LoRA parameters
        lora_state_dict = {}
        for name, tensor in full_state_dict.items():
            if "lora" in name.lower():
                # Move to CPU for Ray serialization
                lora_state_dict[name] = tensor.cpu() if tensor.is_cuda else tensor

        logger.info(f"[MegatronTrainingWorker] Extracted {len(lora_state_dict)} LoRA parameters")

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
