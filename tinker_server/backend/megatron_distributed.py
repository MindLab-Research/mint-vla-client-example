"""Distributed MegatronTrainingWorker using Ray placement groups.

Replaces single-actor MegatronTrainingWorker with coordinator + N workers.
Each worker runs one rank of distributed Megatron training.

Based on patterns from verl/single_controller/ray/base.py and
verl/workers/megatron_workers.py.
"""

from __future__ import annotations  # Allow forward references in type hints

import os
import socket
import logging
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tensordict import TensorDict

import ray
# NOTE: torch and tensordict imports are LAZY - done inside MegatronRankWorker.__init__
# to ensure CUDA_VISIBLE_DEVICES is set before torch initializes CUDA
# (tensordict imports torch internally)

logger = logging.getLogger(__name__)

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH

# Persistent actor configuration - matches vLLM pattern
PERSISTENT_MEGATRON_ACTOR_NAME = "persistent_megatron_worker_group_v2"
PERSISTENT_NAMESPACE = "tinker"  # Same namespace as vLLM


@dataclass
class DistributedConfig:
    """Configuration for distributed Megatron training.

    Defaults configured for 1-GPU setup (GPU 0 has leaked memory).
    For multi-GPU: tensor_parallel_size=2 (2-GPU) or TP=2,PP=2,EP=2 (8-GPU)
    """

    tensor_parallel_size: int = 1  # Single GPU - GPU 0 has corrupted memory
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_size: int = 1

    @property
    def world_size(self) -> int:
        """Total number of processes needed."""
        # For MoE models: world_size = TP * PP * EP * CP
        # Each rank handles one shard of tensor/pipeline/expert parallelism
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.context_parallel_size
        )


@ray.remote(num_gpus=0)
def get_node_ip_and_free_port() -> tuple[str, int]:
    """Get node IP and free port for master address.

    Self-contained to avoid module import issues on Ray workers.
    """
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    # Get free port inline
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return ip, port


@ray.remote(num_gpus=1)
class MegatronRankWorker:
    """Single-rank worker for distributed Megatron training.

    Each worker:
    - Owns 1 GPU
    - Runs one rank of torch.distributed
    - Holds shard of model based on TP/PP/EP configuration
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        distributed_config: DistributedConfig,
    ):
        """Create worker but don't initialize distributed yet.
        
        Distributed init is deferred to initialize() to avoid deadlock.
        All workers must be created first, then initialize() called on all
        simultaneously so they can reach init_process_group barrier together.
        """
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.config = distributed_config
        self.engine = None  # Set in initialize()
        
        logger.info(f"[MegatronRankWorker] Worker {rank}/{world_size} created (not yet initialized)")

    def initialize(self):
        """Initialize distributed backend and Megatron engine.
        
        Must be called on all workers simultaneously after all workers are created.
        This ensures all workers reach init_process_group barrier together.
        """
        # Ray sets CUDA_VISIBLE_DEVICES before process starts when using num_gpus=1
        # Import torch HERE (lazy) - CUDA_VISIBLE_DEVICES must be set before torch initializes CUDA
        import torch

        cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ray_gpu_ids = ray.get_gpu_ids()
        device_count = torch.cuda.device_count()

        logger.info(
            f"[Rank {self.rank}] initialize() starting: CUDA_VISIBLE_DEVICES={cuda_device!r}, "
            f"ray_gpu_ids={ray_gpu_ids}, torch.cuda.device_count()={device_count}"
        )

        if device_count != 1:
            raise RuntimeError(
                f"MegatronRankWorker rank {self.rank} expected 1 GPU, but torch sees {device_count}. "
                f"CUDA_VISIBLE_DEVICES={cuda_device}, ray_gpu_ids={ray_gpu_ids}. "
                f"Check that Ray actor was created with num_gpus=1."
            )

        # Set environment for torch.distributed
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        # LOCAL_RANK is always 0 because CUDA_VISIBLE_DEVICES limits to single GPU
        os.environ["LOCAL_RANK"] = "0"

        # HuggingFace offline mode
        os.environ["HF_HOME"] = "/vePFS-Mindverse/share/huggingface"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self._initialize_distributed()
        self._initialize_megatron()

        logger.info(f"[Rank {self.rank}] initialize() complete")

    def _initialize_distributed(self):
        """Initialize torch.distributed with NCCL backend."""
        import torch

        logger.info(f"[Rank {self.rank}] _initialize_distributed starting...")

        if torch.distributed.is_initialized():
            logger.info(f"[Rank {self.rank}] torch.distributed already initialized")
            return

        # Set CUDA device before init
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        logger.info(
            f"[Rank {self.rank}] Calling init_process_group: "
            f"master={master_addr}:{master_port}, world_size={self.world_size}"
        )

        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=self.world_size,
            rank=self.rank,
        )

        logger.info(f"[Rank {self.rank}] torch.distributed initialized")

    def _initialize_megatron(self):
        """Initialize Megatron model parallel and engine."""
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
        from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
        from verl.trainer.config import CheckpointConfig
        from verl.utils.fs import copy_to_local
        from transformers import AutoConfig

        # Note: mpu.initialize_model_parallel() is called by MegatronEngine
        # Copy model to local (returns path unchanged if not HDFS)
        local_path = copy_to_local(self.base_model)

        # Build configs
        hf_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=True, local_files_only=True
        )

        # Build lora sub-config for get_peft_config() compatibility
        # verl's get_peft_config expects model_config.lora with both .get() and .rank access
        # Must be passed at construction time since HFModelConfig.lora is not in _mutable_fields
        lora_config = {}
        if self.lora_rank > 0:
            # Create a config object that supports both dict and attribute access
            class LoraConfigDict(dict):
                """Dict subclass with attribute access for verl compatibility."""
                def __getattr__(self, key):
                    try:
                        return self[key]
                    except KeyError:
                        raise AttributeError(key)

            lora_config = LoraConfigDict(
                rank=self.lora_rank,
                alpha=self.lora_rank * 2,
                type="lora",
                target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
                dropout=0.0,
            )
            logger.info(f"[Rank {self.rank}] LoRA config: rank={self.lora_rank}, alpha={self.lora_rank * 2}")

        model_config = HFModelConfig(
            path=self.base_model,
            local_path=local_path,
            hf_config=hf_config,
            architectures=hf_config.architectures,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            lora=lora_config,
            target_modules="all-linear",
            trust_remote_code=True,
        )

        # Build override_transformer_config for MoE models
        override_tf_config = {}
        num_experts = getattr(hf_config, "num_experts", None)
        if num_experts is not None:
            # MoE model - pass expert parameters to TransformerConfig
            override_tf_config["num_moe_experts"] = num_experts
            # moe_router_topk = num_experts_per_tok (active experts per token)
            num_experts_per_tok = getattr(hf_config, "num_experts_per_tok", 2)
            override_tf_config["moe_router_topk"] = num_experts_per_tok
            # Disable permute fusion due to Triton version incompatibility
            # (triton 3.5.0 + transformer_engine 2.9.0 causes get_int_dtype error)
            override_tf_config["moe_permute_fusion"] = False
            logger.info(
                f"[Rank {self.rank}] MoE config: {num_experts} experts, "
                f"top-{num_experts_per_tok} routing"
            )

        # For LoRA training, disable grad_offload to keep gradient buffers allocated.
        # The distributed optimizer needs gradient storage, but offloading resizes it to 0.
        # LoRA adapter grads are small so grad_offload isn't needed for memory.
        use_grad_offload = False if self.lora_rank > 0 else True
        if self.lora_rank > 0:
            logger.info(f"[Rank {self.rank}] LoRA enabled (rank={self.lora_rank}), disabling grad_offload")

        engine_config = McoreEngineConfig(
            tensor_model_parallel_size=self.config.tensor_parallel_size,
            pipeline_model_parallel_size=self.config.pipeline_parallel_size,
            expert_model_parallel_size=self.config.expert_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
            param_offload=True,
            optimizer_offload=True,
            grad_offload=use_grad_offload,
            dtype="bfloat16",
            use_mbridge=True,
            vanilla_mbridge=False,  # Required for LoRA - enables provider initialization
            use_distributed_optimizer=True,  # Keep distributed optimizer for efficiency
            override_transformer_config=override_tf_config,
        )

        optimizer_config = McoreOptimizerConfig(
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            clip_grad=1.0,
            lr_decay_steps=100000,
            lr_decay_style="constant",
            lr_warmup_steps=0,
        )

        checkpoint_config = CheckpointConfig()

        # Create and initialize engine
        # Use MegatronEngineWithLMHead which implements forward_step for LM training
        self.engine = MegatronEngineWithLMHead(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        self.engine.initialize()
        logger.info(f"[Rank {self.rank}] MegatronEngineWithLMHead initialized")

        # CUDA sync and test to detect corruption early
        import torch
        torch.cuda.synchronize()
        try:
            test_tensor = torch.ones(1, device="cuda:0")
            torch.cuda.synchronize()  # Force error detection
            logger.info(f"[Rank {self.rank}] Post-init CUDA test passed: {test_tensor.item()}")
            del test_tensor
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Post-init CUDA test FAILED: {e}")
            raise RuntimeError(f"CUDA corrupted after engine init: {e}")

        # Warmup disabled: nested tensors with CUDA cause issues
        # If warmup is needed in the future, ensure CUDA operations work correctly first
        # if self.lora_rank > 0:
        #     self._warmup_lora_weights()
        logger.info(f"[Rank {self.rank}] Skipping warmup (nested tensor CUDA issues)")

    def _warmup_lora_weights(self):
        """Run warmup forward_backward to initialize LoRA weight storage.

        When param_offload=True, LoRA adapter weights may not be allocated until
        the first forward_backward pass. If forward_only=True is called first
        (e.g., NLL evaluation in SL recipes), the weights remain unallocated,
        causing "storage size of 0" errors.

        This warmup runs a dummy forward_backward to force weight allocation.
        verl's forward_backward_batch expects loss_mask for token counting.
        """
        import torch
        from tensordict import TensorDict
        from tensordict.tensorclass import NonTensorData

        logger.info(f"[Rank {self.rank}] Running LoRA warmup forward_backward...")

        # Create minimal dummy batch (4 tokens) with fields verl expects
        device = torch.cuda.current_device()
        seq_len = 4

        # Use nested tensors to match actual training data format
        dummy_input_ids_t = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
        dummy_position_ids_t = torch.arange(seq_len, dtype=torch.long, device=device)
        dummy_loss_mask_t = torch.ones(seq_len, dtype=torch.bool, device=device)

        # Create nested tensors with jagged layout (batch size 1)
        input_ids = torch.nested.as_nested_tensor([dummy_input_ids_t], layout=torch.jagged)
        position_ids = torch.nested.as_nested_tensor([dummy_position_ids_t], layout=torch.jagged)
        loss_mask = torch.nested.as_nested_tensor([dummy_loss_mask_t], layout=torch.jagged)

        dummy_data = TensorDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=[1],
            device=device,
        )

        # Add non-tensor metadata for verl's prepare_micro_batches
        dummy_data["use_dynamic_bsz"] = NonTensorData(True)
        dummy_data["max_token_len_per_gpu"] = NonTensorData(8192)
        dummy_data.set_non_tensor("dp_size", 1)
        dummy_data.set_non_tensor("batch_num_tokens", seq_len)
        dummy_data.set_non_tensor("temperature", 1.0)

        # Run forward_backward (not forward_only) to allocate LoRA weights
        from tinker_server.backend.megatron_training import create_loss_fn

        loss_function = create_loss_fn()
        try:
            self.engine.forward_backward_batch(
                data=dummy_data,
                loss_function=loss_function,
                forward_only=False,
            )
            # Zero gradients after warmup
            self.engine.optimizer.zero_grad()
            logger.info(f"[Rank {self.rank}] LoRA warmup complete")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] LoRA warmup failed (non-fatal): {e}")
            import traceback
            logger.warning(f"[Rank {self.rank}] Warmup traceback: {traceback.format_exc()}")

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str,
        loss_fn_config: dict,
    ) -> dict:
        """Run forward and backward pass on this rank's shard.

        Gradients are synchronized via NCCL allreduce.
        Returns metrics from rank 0 only, including per-sample loss_fn_outputs.

        Note: TensorDict with nested tensors is created locally to avoid Ray
        serialization issues that cause CUDA memory corruption.
        """
        import torch
        from tinker_server.backend.megatron_training import (
            create_sft_loss_fn, create_ppo_loss_fn, tinker_to_tensordict
        )

        # Get sequence lengths from raw data (before tensor creation)
        seq_lengths = []
        for item in data_items:
            model_input = item.get("model_input", {})
            chunks = model_input.get("chunks", [])
            if chunks and "tokens" in chunks[0]:
                seq_lengths.append(len(chunks[0]["tokens"]))

        # Test CUDA context before creating TensorDict
        device = torch.cuda.current_device()
        try:
            test_tensor = torch.ones(1, device=f"cuda:{device}")
            torch.cuda.synchronize()  # Force error detection here
            logger.info(f"[Rank {self.rank}] CUDA context valid, test tensor on cuda:{device}")
            del test_tensor
        except Exception as e:
            logger.error(f"[Rank {self.rank}] CUDA context INVALID at forward_backward start: {e}")
            raise RuntimeError(f"CUDA context corrupted before forward_backward: {e}")

        # Create TensorDict directly on GPU to avoid .to() issues with nested tensors
        # verl's forward_step calls batch.to(device) which fails for nested tensors on CPU
        data = tinker_to_tensordict(data_items, device=f"cuda:{device}")

        # Select loss function (SFT returns log_probs in metrics for train_nll)
        if loss_fn == "cross_entropy":
            loss_function = create_sft_loss_fn(return_logprobs=True)
        elif loss_fn == "ppo":
            epsilon = loss_fn_config.get("epsilon", 0.2)
            loss_function = create_ppo_loss_fn(epsilon)
        elif loss_fn == "importance_sampling":
            loss_function = create_ppo_loss_fn(epsilon=float("inf"))
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        # Use train_mode context to load model from CPU to GPU (required for param_offload)
        # The context manager handles: load to GPU on __enter__, offload to CPU on __exit__
        with self.engine.train_mode():
            # Zero gradients (must be after model is on GPU)
            self.engine.optimizer_zero_grad()

            # Run forward-backward (engine handles gradient sync)
            result = self.engine.forward_backward_batch(
                data=data,
                loss_function=loss_function,
                forward_only=False,
            )

        # Only rank 0 returns metrics
        if self.rank == 0:
            loss_value = 0.0
            num_tokens = 0
            clip_frac_sum = 0.0
            ratio_mean_sum = 0.0
            n_ppo_results = 0
            all_log_probs = []
            loss_fn_outputs = []

            # verl's forward_backward_batch returns a single dict:
            # {
            #     "model_output": {"log_probs": nested_tensor, ...},
            #     "loss": [loss1, loss2, ...],  # list from each micro-batch
            #     "metrics": {
            #         "log_probs": [tensor1, tensor2, ...],
            #         "num_tokens": [n1, n2, ...],
            #         "loss": [l1, l2, ...],
            #     }
            # }
            if result and isinstance(result, dict):
                metrics = result.get("metrics", {})
                losses = result.get("loss", [])

                # Sum losses from all micro-batches
                for loss in losses:
                    if hasattr(loss, "item"):
                        loss = loss.item()
                    loss_value += float(loss)

                # Sum num_tokens from metrics (list of values from micro-batches)
                num_tokens_list = metrics.get("num_tokens", [])
                for tokens in num_tokens_list:
                    if hasattr(tokens, "item"):
                        tokens = tokens.item()
                    num_tokens += int(tokens)

                # Extract per-token log_probs from metrics (list of tensors)
                log_probs_list = metrics.get("log_probs", [])
                for log_probs in log_probs_list:
                    if log_probs is not None:
                        if hasattr(log_probs, "cpu"):
                            log_probs = log_probs.cpu()
                        all_log_probs.append(log_probs)

                # Extract PPO metrics if present (list of values)
                clip_frac_list = metrics.get("clip_frac", [])
                for cf in clip_frac_list:
                    if hasattr(cf, "item"):
                        cf = cf.item()
                    clip_frac_sum += float(cf)
                    n_ppo_results += 1

                ratio_mean_list = metrics.get("ratio_mean", [])
                for rm in ratio_mean_list:
                    if hasattr(rm, "item"):
                        rm = rm.item()
                    ratio_mean_sum += float(rm)

            # Concatenate and split log_probs into per-sample tensors for loss_fn_outputs
            if all_log_probs and seq_lengths:
                combined_log_probs = torch.cat(all_log_probs, dim=0) if len(all_log_probs) > 1 else all_log_probs[0]
                offset = 0
                avg_loss_per_sample = loss_value / max(len(seq_lengths), 1)
                for seq_len in seq_lengths:
                    sample_log_probs = combined_log_probs[offset:offset + seq_len]
                    offset += seq_len
                    loss_fn_outputs.append({
                        "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                        "logprobs": {
                            "data": sample_log_probs.tolist(),
                            "shape": list(sample_log_probs.shape),
                            "dtype": "float32",
                        },
                    })

            # Return CPU-safe scalars and loss_fn_outputs
            return {
                "loss_value": float(loss_value),
                "num_tokens": int(num_tokens),
                "clip_frac_sum": float(clip_frac_sum),
                "ratio_mean_sum": float(ratio_mean_sum),
                "n_ppo_results": int(n_ppo_results),
                "loss_fn_outputs": loss_fn_outputs,
            }
        return {}

    def forward(
        self,
        data_items: list[dict],
    ) -> dict:
        """Run forward pass only (no backward). Returns per-token logprobs.

        Similar to forward_backward but skips gradient computation.
        Used for computing reference model logprobs in DPO/SL.
        Returns per-token log probabilities from rank 0 only.

        Important: Creates one loss_fn_outputs entry per sample (not per micro-batch)
        to match cookbook's NLL evaluator expectations.

        Note: TensorDict with nested tensors is created locally to avoid Ray
        serialization issues that cause CUDA memory corruption.
        """
        import torch
        from tinker_server.backend.megatron_training import tinker_to_tensordict

        # Get sequence lengths from raw data (before tensor creation)
        seq_lengths = []
        for item in data_items:
            model_input = item.get("model_input", {})
            chunks = model_input.get("chunks", [])
            if chunks and "tokens" in chunks[0]:
                seq_lengths.append(len(chunks[0]["tokens"]))

        # Create TensorDict directly on GPU to avoid .to() issues with nested tensors
        device = torch.cuda.current_device()
        data = tinker_to_tensordict(data_items, device=f"cuda:{device}")

        # Use logprob extractor to get per-token log probabilities
        from tinker_server.backend.megatron_training import create_logprob_extractor_fn
        loss_function = create_logprob_extractor_fn()

        # Use eval_mode context to load model from CPU to GPU (required for param_offload)
        # eval_mode is used for forward-only operations
        with self.engine.eval_mode():
            # Run forward only (no gradient sync)
            with torch.no_grad():
                result = self.engine.forward_backward_batch(
                    data=data,
                    loss_function=loss_function,
                    forward_only=True,
                )

        # Only rank 0 returns metrics
        if self.rank == 0:
            loss_value = 0.0
            num_tokens = 0
            all_log_probs = []
            loss_fn_outputs = []

            # verl's forward_backward_batch returns a single dict (not a list):
            # {
            #     "model_output": {"log_probs": nested_tensor, ...},
            #     "loss": [loss1, loss2, ...],  # list from each micro-batch
            #     "metrics": {
            #         "log_probs": [tensor1, tensor2, ...],  # our extracted log_probs (concatenated per micro-batch)
            #         "num_tokens": [n1, n2, ...],
            #         "loss": [l1, l2, ...],
            #     }
            # }
            if result and isinstance(result, dict):
                metrics = result.get("metrics", {})
                losses = result.get("loss", [])

                # Sum losses from all micro-batches
                for loss in losses:
                    if hasattr(loss, "item"):
                        loss = loss.item()
                    loss_value += float(loss)

                # Sum num_tokens from all micro-batches (list of values)
                num_tokens_list = metrics.get("num_tokens", [])
                for tokens in num_tokens_list:
                    if hasattr(tokens, "item"):
                        tokens = tokens.item()
                    num_tokens += int(tokens)

                # Extract per-token log_probs from metrics (list of tensors, one per micro-batch)
                log_probs_list = metrics.get("log_probs", [])
                for log_probs in log_probs_list:
                    if log_probs is not None:
                        # log_probs is already CPU tensor from extractor
                        if hasattr(log_probs, "cpu"):
                            log_probs = log_probs.cpu()
                        all_log_probs.append(log_probs)

                # Concatenate all micro-batch log_probs into a single tensor
                if all_log_probs:
                    combined_log_probs = torch.cat(all_log_probs, dim=0) if len(all_log_probs) > 1 else all_log_probs[0]
                else:
                    combined_log_probs = None

                # Split combined log_probs back into per-sample tensors using seq_lengths
                # This is critical for cookbook compatibility - it expects one loss_fn_outputs per sample
                if combined_log_probs is not None and seq_lengths:
                    offset = 0
                    avg_loss_per_sample = loss_value / max(len(seq_lengths), 1)
                    for seq_len in seq_lengths:
                        sample_log_probs = combined_log_probs[offset:offset + seq_len]
                        offset += seq_len
                        loss_fn_outputs.append({
                            "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                            "logprobs": {
                                "data": sample_log_probs.tolist(),
                                "shape": list(sample_log_probs.shape),
                                "dtype": "float32",
                            },
                        })
                elif combined_log_probs is not None:
                    # Fallback: single entry with all log_probs (legacy behavior)
                    loss_fn_outputs.append({
                        "loss": {"data": [loss_value], "shape": [1], "dtype": "float32"},
                        "logprobs": {
                            "data": combined_log_probs.tolist(),
                            "shape": list(combined_log_probs.shape),
                            "dtype": "float32",
                        },
                    })

            return {
                "loss_value": float(loss_value),
                "num_tokens": int(num_tokens),
                "loss_fn_outputs": loss_fn_outputs,
                "log_probs": combined_log_probs,  # Combined per-token log_probs tensor (for backward compat)
            }
        return {}

    def optim_step(self, learning_rate: float) -> dict:
        """Run optimizer step (synchronized across ranks)."""
        # Note: learning_rate not used directly - verl engine handles LR scheduling
        # Use train_mode context to ensure model/gradients are on GPU (required for param_offload)
        # Without this, optimizer would access offloaded tensors with invalid CUDA pointers
        with self.engine.train_mode():
            grad_norm = self.engine.optimizer_step()
            current_lr = self.engine.lr_scheduler_step()

        if self.rank == 0:
            # Handle current_lr being either a float or a list
            if current_lr is not None:
                lr_value = current_lr[0] if isinstance(current_lr, (list, tuple)) else current_lr
            else:
                lr_value = learning_rate
            # Return CPU-safe scalars only
            return {
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "lr": float(lr_value),
                "step": "completed",
            }
        return {}

    def get_lora_state_dict(self) -> dict:
        """Get LoRA state dict in PEFT format (rank 0 gathers from all ranks).

        With use_mbridge=True, weights must be accessed via bridge.export_weights()
        instead of model.named_parameters().

        IMPORTANT: ALL ranks must call bridge.export_weights() because it uses NCCL
        collectives internally. Only rank 0 processes and returns the actual data;
        other ranks return empty dict AFTER the collective completes.

        The bridge returns HuggingFace-style names like:
            model.layers.0.self_attn.q_proj.lora_A.weight
            model.layers.0.self_attn.q_proj.lora_B.weight

        PEFT expects names like:
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
        """
        logger.info(f"[Rank {self.rank}] get_lora_state_dict: ENTRY")
        
        # Write diagnostic output to shared PFS for debugging
        debug_file = "/vePFS-Mindverse/share/code/tinker-server/debug_lora_export.log"

        adapter_state = {}

        # Patterns for identifying adapter/LoRA parameters
        # MBridge uses HF-style names with "lora" suffix
        # Megatron may use different naming conventions
        adapter_patterns = ['lora', 'adapter', 'peft']

        # Check if we have mbridge with export_weights (vanilla_mbridge=True only)
        # Non-vanilla bridge uses different API - skip to fallback
        bridge = getattr(self.engine, 'bridge', None)
        has_export_weights = bridge is not None and hasattr(bridge, 'export_weights')

        if has_export_weights:
            logger.info(f"[Rank {self.rank}] Using bridge.export_weights() for vanilla mbridge mode")
            try:
                # export_weights returns a generator of (name, tensor) tuples
                # with HuggingFace-style parameter names
                # ALL ranks must iterate through this generator for NCCL sync
                all_param_names = []
                adapter_param_names = []
                for name, tensor in bridge.export_weights(self.engine.module):
                    all_param_names.append(name)
                    # Only rank 0 collects the actual data
                    if self.rank == 0:
                        # Check multiple patterns for adapter parameters
                        name_lower = name.lower()
                        if any(pattern in name_lower for pattern in adapter_patterns):
                            adapter_param_names.append(name)
                            # Clone and move to CPU
                            adapter_state[name] = tensor.data.clone().cpu() if tensor.is_cuda else tensor.data.clone()

                # Log diagnostic info
                logger.info(f"[Rank {self.rank}] bridge.export_weights() returned {len(all_param_names)} total params")
                
                # Only rank 0 logs detailed info and writes debug file
                if self.rank == 0:
                    logger.info(f"[Rank 0] Found {len(adapter_param_names)} adapter params (patterns: {adapter_patterns})")

                    # Write to PFS for debugging (append mode)
                    try:
                        import datetime
                        with open(debug_file, "a") as f:
                            f.write(f"\n=== {datetime.datetime.now()} ===\n")
                            f.write(f"Total params from bridge.export_weights(): {len(all_param_names)}\n")
                            f.write(f"Adapter params found: {len(adapter_param_names)}\n")
                            f.write(f"First 30 param names:\n")
                            for n in all_param_names[:30]:
                                f.write(f"  {n}\n")
                            if len(all_param_names) > 100:
                                f.write(f"Middle 10 param names (around index {len(all_param_names)//2}):\n")
                                mid = len(all_param_names) // 2
                                for n in all_param_names[mid:mid+10]:
                                    f.write(f"  {n}\n")
                            if adapter_param_names:
                                f.write(f"Adapter param names:\n")
                                for n in adapter_param_names[:30]:
                                    f.write(f"  {n}\n")
                            f.write(f"===\n")
                    except Exception as e:
                        logger.warning(f"Failed to write debug file: {e}")

                    # Show sample of ALL param names to debug naming convention
                    if all_param_names:
                        sample_all = all_param_names[:10]
                        logger.info(f"[Rank 0] Sample ALL param names (first 10): {sample_all}")
                        # Also show some from the middle
                        if len(all_param_names) > 50:
                            mid_idx = len(all_param_names) // 2
                            sample_mid = all_param_names[mid_idx:mid_idx+5]
                            logger.info(f"[Rank 0] Sample ALL param names (middle 5): {sample_mid}")

                    if adapter_param_names:
                        sample_adapter = adapter_param_names[:5]
                        logger.info(f"[Rank 0] Sample adapter param names: {sample_adapter}")
                    else:
                        logger.warning("[Rank 0] NO adapter params found via bridge.export_weights()!")

            except Exception as e:
                logger.error(f"[Rank {self.rank}] bridge.export_weights() failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Non-rank-0 workers return empty dict AFTER participating in NCCL collective
        if self.rank != 0:
            logger.info(f"[Rank {self.rank}] get_lora_state_dict: returning empty dict (non-rank-0)")
            return {}

        # Fallback: use get_adapter_state_dict if bridge method returned empty or doesn't exist
        if not adapter_state:
            logger.info("[Rank 0] Trying get_adapter_state_dict() fallback...")
            try:
                from verl.utils.megatron_peft_utils import get_adapter_state_dict
                raw_state = get_adapter_state_dict(self.engine.module)
                logger.info(f"[Rank 0] get_adapter_state_dict returned {len(raw_state)} params")
                # Move all tensors to CPU for serialization to CPU-only WorkerGroup
                for name, tensor in raw_state.items():
                    adapter_state[name] = tensor.cpu() if tensor.is_cuda else tensor.clone()
                if adapter_state:
                    sample_keys = list(adapter_state.keys())[:5]
                    logger.info(f"[Rank 0] Sample Megatron keys: {sample_keys}")
            except ImportError:
                logger.warning("[Rank 0] verl.utils.megatron_peft_utils not available")
            except Exception as e:
                logger.error(f"[Rank 0] get_adapter_state_dict failed: {e}")

        if not adapter_state:
            logger.warning("[Rank 0] No adapter/LoRA parameters found!")
            return {}

        # Convert to PEFT format
        # HF names: model.layers.X.self_attn.q_proj.lora_A.weight
        # PEFT names: base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight
        lora_state_dict = {}
        for name, tensor in adapter_state.items():
            # Add PEFT prefix if not already present
            if name.startswith("model."):
                peft_name = f"base_model.model.{name}"
            elif name.startswith("base_model."):
                peft_name = name  # Already in PEFT format
            else:
                # Megatron format - need more complex conversion
                peft_name = self._convert_megatron_to_peft(name)
                if peft_name is None:
                    logger.warning(f"Could not convert to PEFT: {name}")
                    continue
            lora_state_dict[peft_name] = tensor

        logger.info(f"[Rank 0] Extracted {len(lora_state_dict)} LoRA parameters (PEFT format)")
        if lora_state_dict:
            sample_peft_keys = list(lora_state_dict.keys())[:3]
            logger.info(f"[Rank 0] Sample PEFT keys: {sample_peft_keys}")

        return lora_state_dict

    def _convert_megatron_to_peft(self, name: str) -> str | None:
        """Convert Megatron LoRA param name to PEFT format.

        Megatron-Bridge names (with vanilla_mbridge=False):
            decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight
            decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight
            decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight
            decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight
            decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight

        PEFT names (HuggingFace style):
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight
            base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight
        """
        import re

        # Extract layer number
        match = re.search(r'layers\.(\d+)\.', name)
        if not match:
            logger.warning(f"Could not extract layer number from: {name}")
            return None
        layer_num = match.group(1)

        # Map adapter naming to LoRA naming
        # adapter.linear_in -> lora_A, adapter.linear_out -> lora_B
        if 'adapter.linear_in' in name:
            lora_type = 'lora_A'
        elif 'adapter.linear_out' in name:
            lora_type = 'lora_B'
        elif 'lora_A' in name.lower():
            lora_type = 'lora_A'
        elif 'lora_B' in name.lower():
            lora_type = 'lora_B'
        else:
            logger.warning(f"Could not determine LoRA type from: {name}")
            return None

        # Determine the target module
        # Handle both dense and MoE (experts) variants
        if 'linear_qkv.adapter' in name or 'linear_qkv.lora_' in name.lower():
            # Fused QKV - map to q_proj (vLLM will handle the fused format)
            target = 'self_attn.q_proj'
        elif 'self_attention.linear_proj' in name:
            target = 'self_attn.o_proj'
        elif 'mlp.experts.linear_fc1' in name or 'mlp.linear_fc1' in name:
            # gate_proj for Qwen-style models (gate is part of gate_up_proj)
            target = 'mlp.gate_proj'
        elif 'mlp.experts.linear_fc2' in name or 'mlp.linear_fc2' in name:
            target = 'mlp.down_proj'
        else:
            logger.warning(f"Unknown module pattern: {name}")
            return None

        peft_name = f"base_model.model.model.layers.{layer_num}.{target}.{lora_type}.weight"
        return peft_name


    def save_checkpoint(self, save_path: str, step_count: int = 0) -> dict:
        """Save checkpoint: LoRA weights + config + training metadata.

        Only rank 0 saves the checkpoint.

        Args:
            save_path: Directory path to save checkpoint files.
            step_count: Current training step (passed from MegatronWorkerGroup).

        Returns:
            Dict with training metadata.
        """
        import json
        import os

        from safetensors.torch import save_file

        if self.rank != 0:
            return {}

        os.makedirs(save_path, exist_ok=True)

        # 1. LoRA weights (PEFT format)
        state_dict = self.get_lora_state_dict()
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # 2. LoRA config
        config = {
            "r": self.lora_rank,
            "lora_alpha": self.lora_rank,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": self.base_model,
        }
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 3. Training metadata
        meta = {
            "current_step": step_count,
            "learning_rate": self.learning_rate,
        }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info(f"[MegatronRankWorker] Saved checkpoint to {abs_path} (step={step_count})")
        return meta


    # ========================================================================
    # Phase 6: Multi-Session Support Methods
    # ========================================================================

    def load_adapter_state(
        self, checkpoint_path: str, actual_rank: int | None = None, trainer_rank: int | None = None
    ) -> dict:
        """Load LoRA adapter weights from checkpoint.

        Phase 7: Supports padding for unified rank training.

        ALL ranks must call this method - uses NCCL collectives internally.
        Used for session swapping: loading a different session's adapter weights.

        Args:
            checkpoint_path: Base directory containing adapter checkpoint files.
                Each rank loads from its own mp_rank_XX_adapter.pt file.
            actual_rank: The rank of the checkpoint being loaded. If less than
                trainer_rank, padding will be applied.
            trainer_rank: The trainer's max rank. Required if actual_rank is specified.

        Returns:
            Dict with status info (rank 0 only returns meaningful data).
        """
        import os

        import torch

        from tinker_server.backend.lora_utils import pad_lora_state_dict
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path
        from verl.utils.megatron_utils import unwrap_model

        # Use train_mode context to ensure model is on GPU for loading
        with self.engine.train_mode():
            # Get rank-specific checkpoint path
            rank_path = _get_rank_checkpoint_path(checkpoint_path)
            adapter_file = rank_path + "_adapter.pt"

            if not os.path.isfile(adapter_file):
                raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_file}")

            checkpoint = torch.load(adapter_file, map_location="cpu")
            adapter_state = checkpoint.get("adapter_state_dict", {})

            # Phase 7: Apply padding if actual_rank < trainer_rank
            if actual_rank is not None and trainer_rank is not None and actual_rank < trainer_rank:
                logger.info(
                    f"[Rank {self.rank}] Padding adapter from rank {actual_rank} to {trainer_rank}"
                )
                adapter_state = pad_lora_state_dict(adapter_state, actual_rank, trainer_rank)

            # Load into model
            model = self.engine.module
            if isinstance(model, list):
                model = model[0]
            unwrapped = unwrap_model(model)
            if isinstance(unwrapped, list):
                unwrapped = unwrapped[0]

            _, unexpected = unwrapped.load_state_dict(adapter_state, strict=False)
            if unexpected:
                logger.warning(f"[Rank {self.rank}] Unexpected keys in checkpoint: {unexpected[:5]}...")

        logger.info(f"[Rank {self.rank}] Loaded adapter state from {checkpoint_path}")

        if self.rank == 0:
            return {"status": "ok", "path": checkpoint_path, "actual_rank": actual_rank}
        return {}

    def save_adapter_state(
        self, checkpoint_path: str, actual_rank: int | None = None, trainer_rank: int | None = None
    ) -> dict:
        """Save LoRA adapter weights to checkpoint.

        Phase 7: Supports truncation for unified rank training.

        ALL ranks must call this method - uses NCCL collectives internally.
        Used for session swapping: saving current session's adapter weights.

        Args:
            checkpoint_path: Base directory to save adapter checkpoint files.
                Each rank saves to its own mp_rank_XX_adapter.pt file.
            actual_rank: The rank to save as. If less than trainer_rank,
                truncation will be applied to strip zero-padded dimensions.
            trainer_rank: The trainer's max rank. Required if actual_rank is specified.

        Returns:
            Dict with status info (rank 0 only returns meaningful data).
        """
        import os
        from pathlib import Path

        import torch

        from tinker_server.backend.lora_utils import truncate_lora_state_dict
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path, get_adapter_state_dict

        os.makedirs(checkpoint_path, exist_ok=True)

        # Use train_mode context to ensure model is on GPU for saving
        with self.engine.train_mode():
            # Get adapter state dict
            adapter_state = get_adapter_state_dict(self.engine.module)

            # Phase 7: Apply truncation if actual_rank < trainer_rank
            if actual_rank is not None and trainer_rank is not None and actual_rank < trainer_rank:
                logger.info(
                    f"[Rank {self.rank}] Truncating adapter from rank {trainer_rank} to {actual_rank}"
                )
                adapter_state = truncate_lora_state_dict(adapter_state, trainer_rank, actual_rank)

            # Get rank-specific path
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            rank_path = _get_rank_checkpoint_path(checkpoint_path)
            adapter_file = rank_path + "_adapter.pt"

            torch.save({"adapter_state_dict": adapter_state}, adapter_file)

        logger.info(f"[Rank {self.rank}] Saved adapter state to {checkpoint_path}")

        if self.rank == 0:
            return {"status": "ok", "path": checkpoint_path, "actual_rank": actual_rank}
        return {}

    def reset_optimizer(self, learning_rate: float | None = None) -> dict:
        """Reset optimizer state for a new session.

        Updates learning rate and zeros gradients. Note: With distributed optimizer,
        full momentum/variance reset is complex and often unnecessary - momentum
        from prior training can actually help convergence.

        Args:
            learning_rate: Optional new learning rate. If None, keeps current.

        Returns:
            Dict with status info.
        """
        # Update learning rate if specified
        if learning_rate is not None:
            self.learning_rate = learning_rate
            # Update all param groups
            for group in self.engine.optimizer.param_groups:
                group['lr'] = learning_rate

        # Zero gradients (always safe)
        self.engine.optimizer_zero_grad()

        # Note: Full optimizer state reset (momentum/variance) is skipped for distributed
        # optimizer because:
        # 1. ProxyDict doesn't support standard iteration patterns
        # 2. Momentum from prior training often helps convergence
        # If full reset is needed, the actor should be killed and recreated.

        logger.info(f"[Rank {self.rank}] Reset optimizer (lr={learning_rate or self.learning_rate}, grads zeroed)")

        if self.rank == 0:
            return {"status": "ok", "learning_rate": learning_rate or self.learning_rate}
        return {}

    def get_session_info(self) -> dict:
        """Get current session info for diagnostics.

        Returns:
            Dict with model, LoRA, and optimizer info.
        """
        if self.rank != 0:
            return {}

        lr = self.learning_rate
        if self.engine.optimizer.param_groups:
            lr = self.engine.optimizer.param_groups[0].get('lr', self.learning_rate)

        return {
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "learning_rate": float(lr),
            "world_size": self.world_size,
            "rank": self.rank,
        }

    def shutdown(self):
        """Clean shutdown of distributed process."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@ray.remote(num_gpus=0)
class MegatronWorkerGroup:
    """Manages N distributed MegatronRankWorkers.

    Creates placement group, spawns workers, routes API calls.
    This is the Tinker API surface for MoE training.

    This is a Ray actor (num_gpus=0) to match MegatronTrainingWorker interface.
    """

    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        distributed_config: DistributedConfig | None = None,
    ):
        self.base_model = base_model
        self.lora_rank = lora_rank  # This is max_lora_rank for Phase 7
        self.learning_rate = learning_rate
        self.config = distributed_config or DistributedConfig()

        self.workers: list[ray.actor.ActorHandle] = []
        self.placement_group = None
        self._step_count = 0
        self._current_session: str | None = None  # Phase 6: session tracking
        self._actual_rank: int | None = None  # Phase 7: actual LoRA rank for current session

        self._initialize()

    def _initialize(self):
        """Create placement group, spawn workers, then initialize them all together."""
        world_size = self.config.world_size

        # Create placement group with N GPU bundles
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(world_size)]
        self.placement_group = ray.util.placement_group(
            bundles,
            strategy="STRICT_PACK",  # All on same node for NVLink
        )
        ray.get(self.placement_group.ready())

        logger.info(f"[MegatronWorkerGroup] Placement group ready with {world_size} GPUs")

        # Runtime env for workers
        runtime_env = {
            "env_vars": {
                "PYTHONPATH": PFS_PYTHONPATH,
                "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",  # Avoid stale bytecode on PFS
            },
        }

        # Get master address from first bundle's node
        master_addr, master_port = ray.get(
            get_node_ip_and_free_port.options(
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=0,
                ),
                runtime_env=runtime_env,
            ).remote()
        )

        logger.info(f"[MegatronWorkerGroup] Master: {master_addr}:{master_port}")

        # Spawn workers - __init__ is lightweight, no distributed init yet
        for rank in range(world_size):
            logger.info(f"[MegatronWorkerGroup] Spawning rank {rank}")
            worker = MegatronRankWorker.options(
                num_gpus=1,  # Ray sets CUDA_VISIBLE_DEVICES before process starts
                num_cpus=1,
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=rank,
                ),
                runtime_env=runtime_env,
            ).remote(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                base_model=self.base_model,
                lora_rank=self.lora_rank,
                learning_rate=self.learning_rate,
                distributed_config=self.config,
            )
            self.workers.append(worker)

        # Wait for all worker actors to be created (lightweight __init__ only)
        ray.get([w.__ray_ready__.remote() for w in self.workers])
        logger.info(f"[MegatronWorkerGroup] All {world_size} worker actors created")

        # Now initialize all workers simultaneously - they will reach
        # init_process_group barrier together, avoiding deadlock
        logger.info(f"[MegatronWorkerGroup] Calling initialize() on all workers...")
        ray.get([w.initialize.remote() for w in self.workers])

        logger.info(f"[MegatronWorkerGroup] All {world_size} workers initialized and ready")

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> dict:
        """Run forward-backward on all workers.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type.
            loss_fn_config: Optional loss config.

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
        loss_fn_config = loss_fn_config or {}

        # Send raw data_items to workers (TensorDict created locally on each worker
        # to avoid Ray serialization issues with nested tensors)
        futures = [
            w.forward_backward.remote(data_items, loss_fn, loss_fn_config)
            for w in self.workers
        ]
        results = ray.get(futures)

        # Rank 0 result has metrics
        rank0_result = results[0]
        loss_value = rank0_result.get("loss_value", 0.0)
        num_tokens = rank0_result.get("num_tokens", 0)

        metrics = {
            "loss:mean": float(loss_value),
            "num_samples:sum": float(len(data_items)),
            "num_tokens:sum": float(num_tokens),
        }

        # Add PPO metrics if present (now pre-extracted as scalars)
        n_ppo = rank0_result.get("n_ppo_results", 0)
        if loss_fn == "ppo" and n_ppo > 0:
            clip_frac_sum = rank0_result.get("clip_frac_sum", 0.0)
            ratio_mean_sum = rank0_result.get("ratio_mean_sum", 0.0)
            metrics["clipfrac:mean"] = float(clip_frac_sum / n_ppo)
            metrics["ratio:mean"] = float(ratio_mean_sum / n_ppo)

        logger.info(f"[MegatronWorkerGroup] forward_backward ({loss_fn}): loss={loss_value:.4f}")

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": rank0_result.get("loss_fn_outputs", []),
            "metrics": metrics,
        }

    def forward(
        self,
        data_items: list[dict],
    ) -> dict:
        """Run forward pass only on all workers. Returns per-token logprobs.

        Similar to forward_backward but skips gradient computation.
        Used for computing reference model logprobs in DPO/SL.

        Args:
            data_items: List of Tinker Datum dicts.

        Returns:
            Dict with loss_fn_outputs (including per-token logprobs) and metrics.
        """
        # Send raw data_items to workers (TensorDict created locally on each worker
        # to avoid Ray serialization issues with nested tensors)
        futures = [w.forward.remote(data_items) for w in self.workers]
        results = ray.get(futures)

        # Rank 0 result has metrics, loss_fn_outputs, and log_probs
        rank0_result = results[0]
        loss_value = rank0_result.get("loss_value", 0.0)
        num_tokens = rank0_result.get("num_tokens", 0)
        loss_fn_outputs = rank0_result.get("loss_fn_outputs", [])
        log_probs = rank0_result.get("log_probs")  # Per-token log_probs tensor

        metrics = {
            "loss:mean": float(loss_value),
            "num_samples:sum": float(len(data_items)),
            "num_tokens:sum": float(num_tokens),
        }

        # Convert log_probs tensor to serializable format if present
        log_probs_data = None
        if log_probs is not None:
            import torch
            if isinstance(log_probs, torch.Tensor):
                log_probs_data = {
                    "data": log_probs.tolist(),
                    "shape": list(log_probs.shape),
                    "dtype": str(log_probs.dtype),
                }

        logger.info(f"[MegatronWorkerGroup] forward: loss={loss_value:.4f}, log_probs={'present' if log_probs_data else 'none'}")

        return {
            "loss_fn_output_type": "logprob_extractor",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
            "log_probs": log_probs_data,  # Per-token log probabilities
        }

    def optim_step(self, learning_rate: float) -> dict:
        """Run optimizer step on all workers."""
        futures = [w.optim_step.remote(learning_rate) for w in self.workers]
        ray.get(futures)

        self._step_count += 1
        return {"metrics": {"step": self._step_count}}

    def get_lora_state_dict(self) -> dict:
        """Get LoRA state dict from all workers (rank 0 returns data, others empty).

        IMPORTANT: Must call ALL workers in parallel because bridge.export_weights()
        may use NCCL collectives internally. Calling only rank 0 would deadlock.
        """
        logger.info("[MegatronWorkerGroup] get_lora_state_dict: ENTRY")
        logger.info(f"[MegatronWorkerGroup] Calling get_lora_state_dict.remote() on all {len(self.workers)} workers...")

        try:
            # Call ALL workers - bridge.export_weights() may use NCCL allgather
            # Rank 0 returns actual data, other ranks return empty dict
            futures = [w.get_lora_state_dict.remote() for w in self.workers]
            results = ray.get(futures, timeout=120)

            # Rank 0's result has the actual data
            result = results[0]
            logger.info(f"[MegatronWorkerGroup] get_lora_state_dict: returning {len(result)} params from rank 0")
            return result
        except ray.exceptions.GetTimeoutError:
            logger.error("[MegatronWorkerGroup] get_lora_state_dict timed out after 120s")
            raise RuntimeError("get_lora_state_dict timed out - workers not responding (possible NCCL deadlock)")
        except ray.exceptions.RayActorError as e:
            logger.error(f"[MegatronWorkerGroup] get_lora_state_dict failed - worker died: {e}")
            raise RuntimeError(f"get_lora_state_dict failed - worker died: {e}")

    def get_lora_config(self) -> dict:
        """Get LoRA configuration as dictionary.
        
        Returns:
            PEFT config dict compatible with vLLM's PEFTHelper.
        """
        return {
            "r": self.lora_rank,
            "lora_alpha": self.lora_rank * 2,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }

    def get_tokenizer_info(self) -> dict:
        """Get tokenizer info for the model.

        Returns:
            Dict with tokenizer configuration.
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        return {
            "vocab_size": len(tokenizer),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

    def load_checkpoint(self, load_path: str, load_optimizer: bool = True) -> dict:
        """Load checkpoint from path.

        Note: Megatron distributed checkpoints use verl's checkpoint format.
        This is a placeholder - actual implementation needs verl engine integration.

        Args:
            load_path: Path to checkpoint directory.
            load_optimizer: Whether to restore optimizer state.

        Returns:
            Dict with load metadata.
        """
        logger.warning(f"[MegatronWorkerGroup] load_checkpoint not fully implemented: {load_path}")
        # For now, return empty metadata - full implementation requires
        # coordinated loading across all workers
        return {"status": "not_implemented", "path": load_path}


    def save_checkpoint(self, save_path: str) -> dict:
        """Save checkpoint using rank 0 worker.

        Args:
            save_path: Directory path to save checkpoint files.

        Returns:
            Dict with training metadata.
        """
        logger.info(f"[MegatronWorkerGroup] save_checkpoint: {save_path}")
        result = ray.get(self.workers[0].save_checkpoint.remote(save_path, self._step_count))
        logger.info(f"[MegatronWorkerGroup] save_checkpoint: completed, step={result.get('current_step', 'unknown')}")
        return result

    def get_diagnostics(self) -> dict:
        """Return diagnostic info about the worker group."""
        return {
            "code_version": "test-reload-v1",  # Trivial change to test code reload
            "world_size": self.config.world_size,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
            "expert_parallel_size": self.config.expert_parallel_size,
            "num_workers": len(self.workers),
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
        }


    # ========================================================================
    # Phase 6: Multi-Session Support Methods
    # ========================================================================

    def load_adapter_state(
        self, checkpoint_path: str, actual_rank: int | None = None
    ) -> dict:
        """Load LoRA adapter weights from checkpoint on all workers.

        Phase 7: Supports loading adapters with rank < trainer's max_rank.
        Padding is applied automatically.

        ALL workers must participate due to NCCL collectives.

        Args:
            checkpoint_path: Base directory containing adapter checkpoint.
            actual_rank: The actual rank of the adapter being loaded.
                If None, assumes adapter rank matches trainer rank (no padding).

        Returns:
            Dict with status info from rank 0.
        """
        logger.info(
            f"[MegatronWorkerGroup] Loading adapter state from {checkpoint_path} "
            f"(actual_rank={actual_rank}, trainer_rank={self.lora_rank})"
        )
        futures = [
            w.load_adapter_state.remote(
                checkpoint_path, actual_rank=actual_rank, trainer_rank=self.lora_rank
            )
            for w in self.workers
        ]
        results = ray.get(futures)
        result = results[0]  # Rank 0 result
        self._actual_rank = actual_rank if actual_rank is not None else self.lora_rank
        logger.info(f"[MegatronWorkerGroup] Adapter state loaded: {result}")
        return result

    def save_adapter_state(
        self, checkpoint_path: str, actual_rank: int | None = None
    ) -> dict:
        """Save LoRA adapter weights to checkpoint from all workers.

        Phase 7: Supports saving adapters with rank < trainer's max_rank.
        Truncation is applied automatically to strip zero-padded dimensions.

        ALL workers must participate due to NCCL collectives.

        Args:
            checkpoint_path: Base directory to save adapter checkpoint.
            actual_rank: The actual rank to save (truncate to). If None, uses
                self._actual_rank or trainer rank.

        Returns:
            Dict with status info from rank 0.
        """
        effective_rank = actual_rank or self._actual_rank or self.lora_rank
        logger.info(
            f"[MegatronWorkerGroup] Saving adapter state to {checkpoint_path} "
            f"(actual_rank={effective_rank}, trainer_rank={self.lora_rank})"
        )
        futures = [
            w.save_adapter_state.remote(
                checkpoint_path, actual_rank=effective_rank, trainer_rank=self.lora_rank
            )
            for w in self.workers
        ]
        results = ray.get(futures)
        result = results[0]  # Rank 0 result
        logger.info(f"[MegatronWorkerGroup] Adapter state saved: {result}")
        return result

    def reset_optimizer(self, learning_rate: float | None = None) -> dict:
        """Reset optimizer state on all workers.

        Used for new sessions to start fresh without prior momentum.

        Args:
            learning_rate: Optional new learning rate.

        Returns:
            Dict with status info from rank 0.
        """
        logger.info(f"[MegatronWorkerGroup] Resetting optimizer (lr={learning_rate})")
        futures = [w.reset_optimizer.remote(learning_rate) for w in self.workers]
        results = ray.get(futures)
        result = results[0]  # Rank 0 result
        self._step_count = 0  # Reset step counter for new session
        logger.info(f"[MegatronWorkerGroup] Optimizer reset: {result}")
        return result

    def swap_session(
        self,
        old_session_id: str | None,
        new_session_id: str,
        old_checkpoint_path: str | None,
        new_checkpoint_path: str | None,
        new_learning_rate: float,
        new_actual_rank: int | None = None,
    ) -> dict:
        """Atomically swap from old session to new session.

        Phase 7: Supports actual_rank tracking for unified rank training.

        Steps:
        1. Save old session state (if any and checkpoint_path provided)
        2. Load new session state (if checkpoint exists) or reset
        3. Update current session marker and actual_rank

        Args:
            old_session_id: ID of session being swapped out (None if first session).
            new_session_id: ID of session being swapped in.
            old_checkpoint_path: Where to save old session state (None to skip).
            new_checkpoint_path: Where to load new session state (None to reset).
            new_learning_rate: Learning rate for new session.
            new_actual_rank: Actual LoRA rank for new session (default: trainer's rank).

        Returns:
            Dict with swap status.
        """
        import os

        logger.info(f"[MegatronWorkerGroup] Swapping session: {old_session_id} -> {new_session_id}")

        # 1. Save old session state (if applicable)
        if old_session_id and old_checkpoint_path:
            logger.info(f"[MegatronWorkerGroup] Saving old session {old_session_id}")
            self.save_adapter_state(old_checkpoint_path)

        # 2. Load new session state or reset
        if new_checkpoint_path and os.path.exists(new_checkpoint_path):
            logger.info(f"[MegatronWorkerGroup] Loading new session {new_session_id} from {new_checkpoint_path}")
            self.load_adapter_state(new_checkpoint_path, actual_rank=new_actual_rank)
            # Keep optimizer state from checkpoint (momentum preserved)
        else:
            logger.info(f"[MegatronWorkerGroup] Resetting for new session {new_session_id}")
            self.reset_optimizer(new_learning_rate)

        # 3. Update session tracking (Phase 7: include actual_rank)
        self._current_session = new_session_id
        self._step_count = 0
        self.learning_rate = new_learning_rate
        self._actual_rank = new_actual_rank if new_actual_rank is not None else self.lora_rank

        logger.info(
            f"[MegatronWorkerGroup] Session swap complete: now on {new_session_id} "
            f"(actual_rank={self._actual_rank})"
        )
        return {
            "status": "ok",
            "old_session": old_session_id,
            "new_session": new_session_id,
            "actual_rank": self._actual_rank,
        }

    def get_session_info(self) -> dict:
        """Get current session info.

        Phase 7: Includes actual_rank and max_lora_rank for unified rank training.

        Returns:
            Dict with session and worker info.
        """
        futures = [self.workers[0].get_session_info.remote()]
        results = ray.get(futures)
        worker_info = results[0]

        return {
            **worker_info,
            "current_session": getattr(self, '_current_session', None),
            "step_count": self._step_count,
            "num_workers": len(self.workers),
            "max_lora_rank": self.lora_rank,  # Phase 7: trainer's max rank
            "actual_rank": getattr(self, '_actual_rank', None),  # Phase 7: session's actual rank
        }

    def shutdown(self):
        """Shutdown all workers and release placement group."""
        for w in self.workers:
            try:
                ray.get(w.shutdown.remote())
            except Exception:
                pass

        if self.placement_group:
            ray.util.remove_placement_group(self.placement_group)

        self.workers = []
        self.placement_group = None


# ============================================================================
# Phase 6: MegatronActorPool - Manages actors keyed by (base_model, lora_rank)
# ============================================================================

@dataclass
class MegatronActorEntry:
    """Entry in the actor pool with session tracking.

    Phase 7: Supports unified rank training where max_lora_rank >= actual session ranks.
    Phase 9: Adds LRU tracking for adaptive resource management.
    """
    actor: ray.actor.ActorHandle
    base_model: str
    max_lora_rank: int  # Trainer's max rank (used for pooling)
    num_gpus: int = 8  # GPUs used by this actor
    current_session: str | None = None
    actual_rank: int | None = None  # Current session's actual rank
    created_at: float = 0.0
    last_accessed: float = 0.0  # Phase 9: LRU tracking

    def __post_init__(self):
        import time
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_accessed == 0.0:
            self.last_accessed = now

    def touch(self):
        """Update last_accessed timestamp (call on each access)."""
        import time
        self.last_accessed = time.time()

    def is_idle(self) -> bool:
        """Check if actor has no active session."""
        return self.current_session is None

    def age(self) -> float:
        """Seconds since creation."""
        import time
        return time.time() - self.created_at

    def idle_time(self) -> float:
        """Seconds since last access."""
        import time
        return time.time() - self.last_accessed


class MegatronActorPool:
    """Pool of Megatron actors keyed by base_model.

    Phase 7: Unified rank support - actors are initialized with max_lora_rank
    and can train adapters with any rank <= max_lora_rank via padding/truncation.

    Phase 9: LRU-based eviction for adaptive resource management.

    Thread-safe for concurrent access from multiple API requests.
    """

    # Default max rank for pooled actors
    DEFAULT_MAX_LORA_RANK = 64
    # Default GPUs per Megatron actor (TP=4, EP=2)
    DEFAULT_NUM_GPUS = 8
    # Minimum actor age before eligible for eviction (seconds)
    MIN_ACTOR_AGE = 300  # 5 minutes

    def __init__(self):
        import threading
        self._actors: dict[str, MegatronActorEntry] = {}  # key: base_model
        self._lock = threading.Lock()

    def _make_key(self, base_model: str) -> str:
        """Create pool key from model path."""
        return base_model

    def _make_actor_name(self, base_model: str, max_rank: int) -> str:
        """Create unique actor name for this config."""
        model_name = base_model.split("/")[-1].replace("-", "_").lower()
        return f"megatron_pool_{model_name}_maxr{max_rank}"

    def get(self, base_model: str) -> MegatronActorEntry | None:
        """Get existing actor entry if it exists.

        Args:
            base_model: HuggingFace model path.

        Returns:
            MegatronActorEntry if actor exists, None otherwise.
        """
        key = self._make_key(base_model)
        with self._lock:
            entry = self._actors.get(key)
            if entry:
                entry.touch()  # Phase 9: Update LRU
            return entry

    def _get_idle_actors_lru(self) -> list[MegatronActorEntry]:
        """Get idle actors sorted by last access time (LRU first).

        Must be called with lock held.
        """
        idle = [e for e in self._actors.values() if e.is_idle() and e.age() > self.MIN_ACTOR_AGE]
        return sorted(idle, key=lambda e: e.last_accessed)

    def _evict_for_gpus(self, needed_gpus: int, save_state: bool = True) -> int:
        """Evict idle actors to free up GPUs.

        Must be called with lock held.

        Args:
            needed_gpus: Number of GPUs needed.
            save_state: If True, save session state before eviction.

        Returns:
            Number of GPUs freed.
        """
        freed_gpus = 0
        idle_actors = self._get_idle_actors_lru()

        for entry in idle_actors:
            if freed_gpus >= needed_gpus:
                break

            logger.info(
                f"[MegatronActorPool] Evicting idle actor: {entry.base_model} "
                f"(idle {entry.idle_time():.1f}s, frees {entry.num_gpus} GPUs)"
            )

            # Remove from pool
            key = self._make_key(entry.base_model)
            self._actors.pop(key, None)

            # Kill actor (no state to save since it's idle)
            try:
                ray.get(entry.actor.shutdown.remote(), timeout=30)
                ray.kill(entry.actor)
            except Exception as e:
                logger.warning(f"[MegatronActorPool] Error killing evicted actor: {e}")

            freed_gpus += entry.num_gpus

        return freed_gpus

    def get_or_create(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        distributed_config: DistributedConfig | None = None,
        session_id: str | None = None,
        max_lora_rank: int | None = None,
    ) -> MegatronActorEntry:
        """Get existing or create new Megatron actor for this base model.

        Phase 7: If actor exists, it will be reused regardless of lora_rank
        (padding/truncation happens at session swap time).

        Phase 9: If resources exhausted, tries LRU eviction of idle actors.

        Args:
            base_model: HuggingFace model path.
            lora_rank: Requested LoRA rank for this session.
            learning_rate: Initial learning rate (only used if creating new).
            distributed_config: Parallelism config.
            session_id: Optional session ID to register with actor.
            max_lora_rank: Override max rank for new actor creation.

        Returns:
            MegatronActorEntry with actor handle.

        Raises:
            ValueError: If rank exceeds max or insufficient resources.
        """
        key = self._make_key(base_model)
        config = distributed_config or DistributedConfig()

        with self._lock:
            # Return existing actor if available
            if key in self._actors:
                entry = self._actors[key]
                if lora_rank > entry.max_lora_rank:
                    raise ValueError(
                        f"Requested rank {lora_rank} exceeds actor's max_lora_rank {entry.max_lora_rank}. "
                        f"Kill the actor and restart with higher max_lora_rank."
                    )
                entry.touch()  # Phase 9: Update LRU
                logger.info(
                    f"[MegatronActorPool] Reusing existing actor for {base_model} "
                    f"(max_rank={entry.max_lora_rank}, session_rank={lora_rank})"
                )
                return entry

            # Create new actor with max rank
            effective_max_rank = max(
                lora_rank,
                max_lora_rank or 0,
                self.DEFAULT_MAX_LORA_RANK,
            )
            actor_name = self._make_actor_name(base_model, effective_max_rank)
            num_gpus = config.tensor_parallel_size * config.expert_parallel_size

            logger.info(
                f"[MegatronActorPool] Creating new actor: {actor_name} for {base_model}, "
                f"max_rank={effective_max_rank}, gpus={num_gpus}"
            )

            # Check if detached actor already exists (e.g., from previous server run)
            try:
                actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
                logger.info(f"[MegatronActorPool] Found existing detached actor: {actor_name}")
                # Verify it's alive
                ray.get(actor.get_diagnostics.remote(), timeout=10)
            except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                # Create new detached actor
                runtime_env = {
                    "env_vars": {
                        "PYTHONPATH": PFS_PYTHONPATH,
                        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    }
                }

                actor = MegatronWorkerGroup.options(
                    name=actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    lifetime="detached",
                    runtime_env=runtime_env,
                ).remote(
                    base_model=base_model,
                    lora_rank=effective_max_rank,  # Initialize with max rank
                    learning_rate=learning_rate,
                    distributed_config=config,
                )

                # Wait for initialization
                ray.get(actor.__ray_ready__.remote())
                logger.info(f"[MegatronActorPool] Actor {actor_name} initialized")

            entry = MegatronActorEntry(
                actor=actor,
                base_model=base_model,
                max_lora_rank=effective_max_rank,
                num_gpus=num_gpus,
                current_session=session_id,
                actual_rank=lora_rank,
            )
            self._actors[key] = entry
            return entry

    def remove(self, base_model: str, kill_actor: bool = True) -> bool:
        """Remove actor from pool.

        Args:
            base_model: HuggingFace model path.
            kill_actor: If True, also kill the Ray actor.

        Returns:
            True if actor was removed, False if not found.
        """
        key = self._make_key(base_model)

        with self._lock:
            if key not in self._actors:
                return False

            entry = self._actors.pop(key)
            if kill_actor:
                try:
                    ray.get(entry.actor.shutdown.remote(), timeout=30)
                    ray.kill(entry.actor)
                except Exception as e:
                    logger.warning(f"[MegatronActorPool] Error killing actor: {e}")

            logger.info(f"[MegatronActorPool] Removed actor for {base_model}")
            return True

    def list_actors(self) -> list[dict]:
        """List all actors in the pool.

        Returns:
            List of actor info dicts.
        """
        with self._lock:
            return [
                {
                    "base_model": entry.base_model,
                    "max_lora_rank": entry.max_lora_rank,
                    "actual_rank": entry.actual_rank,
                    "current_session": entry.current_session,
                    "num_gpus": entry.num_gpus,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "idle_time": entry.idle_time(),
                }
                for entry in self._actors.values()
            ]

    def total_gpus(self) -> int:
        """Total GPUs used by all actors in pool."""
        with self._lock:
            return sum(e.num_gpus for e in self._actors.values())

    def evict_idle(self, min_idle_seconds: float = 600) -> int:
        """Evict all idle actors that have been idle longer than threshold.

        Args:
            min_idle_seconds: Minimum idle time before eviction.

        Returns:
            Number of actors evicted.
        """
        with self._lock:
            to_evict = [
                e for e in self._actors.values()
                if e.is_idle() and e.idle_time() > min_idle_seconds
            ]

            for entry in to_evict:
                key = self._make_key(entry.base_model)
                self._actors.pop(key, None)
                try:
                    ray.get(entry.actor.shutdown.remote(), timeout=30)
                    ray.kill(entry.actor)
                except Exception as e:
                    logger.warning(f"[MegatronActorPool] Error killing idle actor: {e}")
                logger.info(f"[MegatronActorPool] Evicted idle actor: {entry.base_model}")

            return len(to_evict)

    def clear(self, kill_actors: bool = True) -> int:
        """Remove all actors from pool.

        Args:
            kill_actors: If True, also kill the Ray actors.

        Returns:
            Number of actors removed.
        """
        with self._lock:
            count = len(self._actors)
            if kill_actors:
                for entry in self._actors.values():
                    try:
                        ray.get(entry.actor.shutdown.remote(), timeout=30)
                        ray.kill(entry.actor)
                    except Exception as e:
                        logger.warning(f"[MegatronActorPool] Error killing actor: {e}")
            self._actors.clear()
            logger.info(f"[MegatronActorPool] Cleared {count} actors")
            return count


# Global actor pool instance
_megatron_actor_pool: MegatronActorPool | None = None


def get_megatron_actor_pool() -> MegatronActorPool:
    """Get or create the global Megatron actor pool."""
    global _megatron_actor_pool
    if _megatron_actor_pool is None:
        _megatron_actor_pool = MegatronActorPool()
    return _megatron_actor_pool


def get_or_create_megatron_worker_group(
    base_model: str,
    lora_rank: int,
    learning_rate: float,
    distributed_config: DistributedConfig | None = None,
) -> ray.actor.ActorHandle:
    """Get existing or create new persistent MegatronWorkerGroup.

    Uses detached Ray actor pattern like vLLM for crash resilience.
    First tries to connect to existing actor, creates new one if not found.

    Args:
        base_model: HuggingFace model path.
        lora_rank: LoRA rank.
        learning_rate: Initial learning rate.
        distributed_config: Parallelism config. Defaults to single-GPU.

    Returns:
        Ray actor handle to MegatronWorkerGroup.
    """
    config = distributed_config or DistributedConfig()

    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    # Try to get existing persistent actor
    try:
        actor = ray.get_actor(PERSISTENT_MEGATRON_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        logger.info(
            f"Connected to existing Megatron actor: {PERSISTENT_MEGATRON_ACTOR_NAME}"
        )
        return actor
    except ValueError:
        # Actor doesn't exist, create new one
        logger.info(
            f"Creating new detached Megatron actor: {PERSISTENT_MEGATRON_ACTOR_NAME}"
        )

    # Runtime env for PFS code access
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": PFS_PYTHONPATH,
            "HF_HOME": "/vePFS-Mindverse/share/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",  # Avoid stale bytecode on PFS
        }
    }

    # Create detached Ray actor with well-known name
    # lifetime="detached" ensures actor survives owner process termination
    actor = MegatronWorkerGroup.options(
        name=PERSISTENT_MEGATRON_ACTOR_NAME,
        namespace=PERSISTENT_NAMESPACE,
        lifetime="detached",
        runtime_env=runtime_env,
    ).remote(
        base_model=base_model,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        distributed_config=config,
    )

    # Wait for initialization
    ray.get(actor.__ray_ready__.remote())
    logger.info("Megatron worker group initialized (detached actor)")

    return actor


async def async_get_or_create_megatron_worker_group(
    base_model: str,
    lora_rank: int,
    learning_rate: float,
    distributed_config: DistributedConfig | None = None,
) -> ray.actor.ActorHandle:
    """Async version of get_or_create_megatron_worker_group.

    Wraps blocking Ray operations in asyncio.to_thread() to avoid blocking
    the uvicorn event loop during FastAPI request handling.

    Args:
        base_model: HuggingFace model path.
        lora_rank: LoRA rank.
        learning_rate: Initial learning rate.
        distributed_config: Parallelism config. Defaults to single-GPU.

    Returns:
        Ray actor handle to MegatronWorkerGroup.
    """
    import asyncio

    return await asyncio.to_thread(
        get_or_create_megatron_worker_group,
        base_model,
        lora_rank,
        learning_rate,
        distributed_config,
    )


def kill_megatron_actor() -> bool:
    """Kill persistent Megatron actor and release resources.

    Returns:
        True if actor was killed, False if not found.
    """
    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    try:
        actor = ray.get_actor(PERSISTENT_MEGATRON_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        # Graceful shutdown first
        try:
            ray.get(actor.shutdown.remote(), timeout=10)
        except Exception:
            pass
        ray.kill(actor, no_restart=True)
        logger.info(f"Killed Megatron actor: {PERSISTENT_MEGATRON_ACTOR_NAME}")
        return True
    except ValueError:
        logger.info("No Megatron actor to kill")
        return False


def is_megatron_actor_running() -> bool:
    """Check if persistent Megatron actor is running.

    Returns:
        True if actor exists and is actually alive (not dead).
    """
    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    try:
        actor = ray.get_actor(PERSISTENT_MEGATRON_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        # ray.get_actor() returns handle even for DEAD actors
        # Need to actually verify actor is alive by calling a method
        # Use short timeout to avoid hanging on dead actors
        ray.get(actor.get_diagnostics.remote(), timeout=5)
        return True
    except ValueError:
        # Actor doesn't exist
        return False
    except ray.exceptions.RayActorError:
        # Actor is dead
        return False
    except ray.exceptions.GetTimeoutError:
        # Actor not responding
        return False
    except Exception:
        # Any other error - assume not running
        return False
