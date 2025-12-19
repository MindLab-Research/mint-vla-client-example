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
from tinker_server.backend.model_registry import is_moe_model

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
    use_fp8: bool = False  # FP8 quantization for K2 and similar models

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

    Uses Ray's node IP which correctly identifies the inter-node network interface.
    Self-contained to avoid module import issues on Ray workers.
    """
    import socket
    import ray
    # Use Ray's IP detection which respects --node-ip-address and finds the correct interface
    ip = ray.util.get_node_ip_address()
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
        # Apply MLA patches for DeepseekV3/K2/Moonlight models BEFORE importing Megatron
        # These patches enable Flash Attention 2 with MLA by padding value tensors
        # Must be applied before MLASelfAttention class is imported/instantiated
        try:
            from verl.models.mcore.patch_v012 import apply_patch
            apply_patch()
            logger.info(f"[Rank {self.rank}] Applied MLA patches from verl.models.mcore.patch_v012")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Could not apply MLA patch: {e}")

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
        # Different HF configs use different attribute names:
        # - Qwen3-MoE: num_experts
        # - DeepseekV3/Moonlight: n_routed_experts
        override_tf_config = {}
        num_experts = getattr(hf_config, "num_experts", None) or getattr(hf_config, "n_routed_experts", None)
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
                f"top-{num_experts_per_tok} routing, permute_fusion=False"
            )

        # For LoRA training, disable grad_offload to keep gradient buffers allocated.
        # The distributed optimizer needs gradient storage, but offloading resizes it to 0.
        # LoRA adapter grads are small so grad_offload isn't needed for memory.
        use_grad_offload = False if self.lora_rank > 0 else True
        if self.lora_rank > 0:
            logger.info(f"[Rank {self.rank}] LoRA enabled (rank={self.lora_rank}), disabling grad_offload")

        # FP8 support for large models like Kimi-K2
        # FP8 is configured via override_transformer_config (not override_mcore_model_config)
        # because TransformerConfig.fp8 and .fp8_param control FP8 during model creation
        if self.config.use_fp8:
            # Use e4m3 format for FP8 (8-bit floating point with 4-bit exponent, 3-bit mantissa)
            # fp8_param=True stores parameters in FP8 precision for memory savings
            # use_cpu_initialization=True initializes on CPU first, then converts to FP8 before GPU transfer
            # This avoids OOM during BF16→FP8 conversion which would require both in GPU memory
            override_tf_config["fp8"] = "e4m3"
            override_tf_config["fp8_param"] = True
            override_tf_config["use_cpu_initialization"] = True
            logger.info(f"[Rank {self.rank}] FP8 enabled (format: e4m3, fp8_param=True, cpu_init=True) for memory-efficient training")

        # MLA attention (Multi-Latent Attention) for DeepSeekV3/K2/Moonlight models
        # These models have qk_nope_head_dim + qk_rope_head_dim = head_dim_qk
        # MLA has head_dim_qk=192 (qk_nope=128 + qk_rope=64) and head_dim_v=128
        # Flash Attention 2 requires head_dim_qk == head_dim_v
        # The MLA patch in verl/models/mcore/patch_v012.py pads value tensor to 192
        # to match query dimension, enabling FA2 with THD format on sm80 (A100/A800)
        #
        # IMPORTANT: Do NOT force unfused backend here - it conflicts with THD format
        # (TE disables unfused for THD). Let FA2 work with the value padding instead.
        qk_nope = getattr(hf_config, "qk_nope_head_dim", 0)
        qk_rope = getattr(hf_config, "qk_rope_head_dim", 0)
        head_dim_qk = qk_nope + qk_rope
        if head_dim_qk > 0:
            logger.info(
                f"[Rank {self.rank}] MLA attention detected: head_dim_qk={head_dim_qk} "
                f"(qk_nope={qk_nope} + qk_rope={qk_rope}). Using FA2 with value padding."
            )

        logger.info(f"[Rank {self.rank}] override_transformer_config: {override_tf_config}")

        engine_config = McoreEngineConfig(
            tensor_model_parallel_size=self.config.tensor_parallel_size,
            pipeline_model_parallel_size=self.config.pipeline_parallel_size,
            expert_model_parallel_size=self.config.expert_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
            param_offload=True,
            optimizer_offload=True,
            grad_offload=use_grad_offload,
            dtype="bfloat16",  # Base dtype, FP8 handled via override_transformer_config
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
                    logprobs_list = sample_log_probs.tolist()
                    loss_fn_outputs.append({
                        "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                        "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
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
                        logprobs_list = sample_log_probs.tolist()
                        loss_fn_outputs.append({
                            "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                            "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                        })
                elif combined_log_probs is not None:
                    # Fallback: single entry with all log_probs (legacy behavior)
                    logprobs_list = combined_log_probs.tolist()
                    loss_fn_outputs.append({
                        "loss": {"data": [loss_value], "shape": [1], "dtype": "float32"},
                        "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                    })

            return {
                "loss_value": float(loss_value),
                "num_tokens": int(num_tokens),
                "loss_fn_outputs": loss_fn_outputs,
                "log_probs": combined_log_probs,  # Combined per-token log_probs tensor (for backward compat)
            }
        return {}

    def optim_step(self, learning_rate: float) -> dict:
        """Run optimizer step (synchronized across ranks).

        WARNING: With param_offload=True, gradients are zeroed when entering train_mode.
        Use train_step() instead for combined forward_backward + optim_step.
        """
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

    def train_step(
        self,
        data_items: list[dict],
        loss_fn: str,
        loss_fn_config: dict,
        learning_rate: float,
    ) -> dict:
        """Run forward, backward, and optimizer step in a single train_mode context.

        This is the correct way to train with param_offload=True. When train_mode exits,
        gradients are offloaded to CPU. A subsequent train_mode entry zeros the gradients.
        By keeping everything in one context, gradients persist for the optimizer step.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type ("cross_entropy", "importance_sampling", "ppo").
            loss_fn_config: Loss function config (e.g., {"epsilon": 0.2} for PPO).
            learning_rate: Learning rate (passed for compatibility, verl handles scheduling).

        Returns:
            Dict with loss metrics and optimizer results (from rank 0 only).
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

        # Create TensorDict directly on GPU
        device = torch.cuda.current_device()
        data = tinker_to_tensordict(data_items, device=f"cuda:{device}")

        # Select loss function
        if loss_fn == "cross_entropy":
            loss_function = create_sft_loss_fn(return_logprobs=True)
        elif loss_fn == "ppo":
            epsilon = loss_fn_config.get("epsilon", 0.2)
            loss_function = create_ppo_loss_fn(epsilon)
        elif loss_fn == "importance_sampling":
            loss_function = create_ppo_loss_fn(epsilon=float("inf"))
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        # CRITICAL: All operations in single train_mode context
        # This ensures gradients survive for optimizer step
        with self.engine.train_mode():
            # Zero gradients
            self.engine.optimizer_zero_grad()

            # Forward-backward (computes gradients)
            result = self.engine.forward_backward_batch(
                data=data,
                loss_function=loss_function,
                forward_only=False,
            )

            # Optimizer step (uses gradients computed above)
            grad_norm = self.engine.optimizer_step()
            current_lr = self.engine.lr_scheduler_step()

        # Only rank 0 returns metrics
        if self.rank == 0:
            loss_value = 0.0
            num_tokens = 0
            clip_frac_sum = 0.0
            ratio_mean_sum = 0.0
            n_ppo_results = 0
            all_log_probs = []
            loss_fn_outputs = []

            if result and isinstance(result, dict):
                metrics = result.get("metrics", {})
                losses = result.get("loss", [])

                for loss in losses:
                    if hasattr(loss, "item"):
                        loss = loss.item()
                    loss_value += float(loss)

                num_tokens_list = metrics.get("num_tokens", [])
                for tokens in num_tokens_list:
                    if hasattr(tokens, "item"):
                        tokens = tokens.item()
                    num_tokens += int(tokens)

                log_probs_list = metrics.get("log_probs", [])
                for log_probs in log_probs_list:
                    if log_probs is not None:
                        if hasattr(log_probs, "cpu"):
                            log_probs = log_probs.cpu()
                        all_log_probs.append(log_probs)

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

            # Build loss_fn_outputs
            if all_log_probs and seq_lengths:
                combined_log_probs = torch.cat(all_log_probs, dim=0) if len(all_log_probs) > 1 else all_log_probs[0]
                offset = 0
                avg_loss_per_sample = loss_value / max(len(seq_lengths), 1)
                for seq_len in seq_lengths:
                    sample_log_probs = combined_log_probs[offset:offset + seq_len]
                    offset += seq_len
                    logprobs_list = sample_log_probs.tolist()
                    loss_fn_outputs.append({
                        "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                        "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                    })

            # Handle learning rate
            if current_lr is not None:
                lr_value = current_lr[0] if isinstance(current_lr, (list, tuple)) else current_lr
            else:
                lr_value = learning_rate

            logger.info(f"[Rank {self.rank}] train_step: loss={loss_value:.4f}, grad_norm={grad_norm:.4f}")

            return {
                "loss_value": float(loss_value),
                "num_tokens": int(num_tokens),
                "clip_frac_sum": float(clip_frac_sum),
                "ratio_mean_sum": float(ratio_mean_sum),
                "n_ppo_results": int(n_ppo_results),
                "loss_fn_outputs": loss_fn_outputs,
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "lr": float(lr_value),
            }
        return {}

    def get_lora_state_dict(self) -> dict:
        """Get LoRA state dict in PEFT format (rank 0 gathers from all ranks).

        MBridge stores LoRA weights as lora_a/lora_b submodules, but bridge.export_weights()
        MERGES them into base weights. We must extract directly from model.named_parameters()
        to get separate lora_A/lora_B matrices for vLLM multi-LoRA inference.

        IMPORTANT: ALL ranks must participate in extraction if using any NCCL collectives.
        Only rank 0 processes and returns the actual data.

        MBridge internal names (from named_parameters):
            decoder.layers.0.self_attention.linear_qkv.lora_a.weight
            decoder.layers.0.self_attention.linear_qkv.lora_b.weight

        PEFT expects names like:
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight

        NOTE: vLLM 0.12.0 supports FusedMoE LoRA via FusedMoEWithLoRA class.
        All modules (attention + MLP/expert) are now exported for both MoE and Dense models.
        """
        logger.info(f"[Rank {self.rank}] get_lora_state_dict: ENTRY")

        # Write diagnostic output to shared PFS for debugging
        debug_file = "/vePFS-Mindverse/share/code/tinker-server/debug_lora_export.log"

        adapter_state = {}

        # CRITICAL: With param_offload=True, model is on CPU. Must use eval_mode() context
        # to load model onto GPU before accessing parameters.
        with self.engine.eval_mode():
            # FIRST: Try direct extraction from model.named_parameters()
            # MBridge stores LoRA weights as .adapter.linear_in/.adapter.linear_out submodules
            # This avoids the merge that export_weights() does
            logger.info(f"[Rank {self.rank}] Trying direct named_parameters() extraction (within eval_mode)...")
            try:
                from verl.utils.megatron_utils import unwrap_model

                unwrapped = unwrap_model(self.engine.module)
                if isinstance(unwrapped, list):
                    unwrapped = unwrapped[0]

                all_param_names = []
                lora_param_names = []

                for name, param in unwrapped.named_parameters():
                    all_param_names.append(name)
                    # Look for adapter patterns in name (MBridge uses .adapter.linear_in/.adapter.linear_out)
                    name_lower = name.lower()
                    if '.adapter.' in name_lower or 'lora_a' in name_lower or 'lora_b' in name_lower:
                        lora_param_names.append(name)
                        if self.rank == 0:
                            # Copy to CPU for serialization
                            adapter_state[name] = param.data.clone().cpu() if param.is_cuda else param.data.clone()

                logger.info(f"[Rank {self.rank}] named_parameters() returned {len(all_param_names)} total, {len(lora_param_names)} LoRA params")

                if self.rank == 0:
                    # Write diagnostic info
                    try:
                        import datetime
                        with open(debug_file, "a") as f:
                            f.write(f"\n=== {datetime.datetime.now()} (named_parameters within eval_mode) ===\n")
                            f.write(f"Total params: {len(all_param_names)}\n")
                            f.write(f"LoRA params found: {len(lora_param_names)}\n")
                            f.write(f"Sample all params (first 20):\n")
                            for n in all_param_names[:20]:
                                f.write(f"  {n}\n")
                            if lora_param_names:
                                f.write(f"LoRA params:\n")
                                for n in lora_param_names[:30]:
                                    f.write(f"  {n}\n")
                            f.write(f"===\n")
                    except Exception as e:
                        logger.warning(f"Failed to write debug file: {e}")

                    if lora_param_names:
                        logger.info(f"[Rank 0] Sample LoRA params: {lora_param_names[:5]}")
                    else:
                        logger.warning("[Rank 0] NO .adapter. params found in named_parameters()")
            except Exception as e:
                logger.error(f"[Rank {self.rank}] named_parameters() extraction failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # SECOND: If direct extraction failed, try bridge.export_weights() (may give merged weights)
            if not adapter_state:
                bridge = getattr(self.engine, 'bridge', None)
                has_export_weights = bridge is not None and hasattr(bridge, 'export_weights')

                if has_export_weights:
                    logger.info(f"[Rank {self.rank}] Fallback to bridge.export_weights() (may return merged weights)")
                    try:
                        adapter_patterns = ['lora', 'adapter', 'peft']
                        all_param_names = []
                        adapter_param_names = []

                        for name, tensor in bridge.export_weights(self.engine.module):
                            all_param_names.append(name)
                            if self.rank == 0:
                                name_lower = name.lower()
                                if any(pattern in name_lower for pattern in adapter_patterns):
                                    adapter_param_names.append(name)
                                    adapter_state[name] = tensor.data.clone().cpu() if tensor.is_cuda else tensor.data.clone()

                        logger.info(f"[Rank {self.rank}] bridge.export_weights() returned {len(all_param_names)} params, {len(adapter_param_names)} adapter")

                        if self.rank == 0 and not adapter_param_names:
                            logger.warning("[Rank 0] NO adapter params found via export_weights() - weights were merged!")
                    except Exception as e:
                        logger.error(f"[Rank {self.rank}] bridge.export_weights() failed: {e}")

            # THIRD: Try verl's get_adapter_state_dict as last resort (also within eval_mode)
            if not adapter_state and self.rank == 0:
                logger.info("[Rank 0] Trying get_adapter_state_dict() fallback...")
                try:
                    from verl.utils.megatron_peft_utils import get_adapter_state_dict
                    raw_state = get_adapter_state_dict(self.engine.module)
                    logger.info(f"[Rank 0] get_adapter_state_dict returned {len(raw_state)} params")
                    for name, tensor in raw_state.items():
                        adapter_state[name] = tensor.cpu() if tensor.is_cuda else tensor.clone()
                    if adapter_state:
                        sample_keys = list(adapter_state.keys())[:5]
                        logger.info(f"[Rank 0] Sample Megatron keys: {sample_keys}")
                except ImportError:
                    logger.warning("[Rank 0] verl.utils.megatron_peft_utils not available")
                except Exception as e:
                    logger.error(f"[Rank 0] get_adapter_state_dict failed: {e}")

        # Non-rank-0 workers return empty dict
        if self.rank != 0:
            logger.info(f"[Rank {self.rank}] get_lora_state_dict: returning empty dict (non-rank-0)")
            return {}

        if not adapter_state:
            logger.warning("[Rank 0] No adapter/LoRA parameters found!")
            return {}

        # Convert to PEFT format
        # HF names: model.layers.X.self_attn.q_proj.lora_A.weight
        # PEFT names: base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight
        #
        # NOTE: vLLM 0.13.0+ supports expert LoRA for MoE models via FusedMoEWithLoRA.
        # MLP/expert modules are now included in the exported LoRA state_dict.
        def _is_mlp_module(param_name: str) -> bool:
            """Check if parameter is from MLP/expert layer."""
            mlp_patterns = ['.mlp.', '.experts.', 'linear_fc1', 'linear_fc2',
                           'gate_proj', 'up_proj', 'down_proj']
            name_lower = param_name.lower()
            return any(p in name_lower for p in mlp_patterns)

        # Check if model is MoE - if so, filter MLP modules
        # vLLM's FusedMoEWithLoRA expects per-expert LoRA weights, but Megatron exports
        # shared LoRA (single adapter for all experts). This causes weight format mismatch.
        # Solution: Only export attention modules for MoE models.
        model_is_moe = is_moe_model(self.base_model)
        if model_is_moe:
            logger.info("[Rank 0] MoE model detected - filtering MLP/expert modules (vLLM weight format incompatible)")

        lora_state_dict = {}
        mlp_filtered_count = 0
        logger.info(f"[Rank 0] Processing {len(adapter_state)} params from adapter_state")
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

            # For MoE models, filter out MLP/expert modules
            if model_is_moe and _is_mlp_module(peft_name):
                mlp_filtered_count += 1
                continue

            lora_state_dict[peft_name] = tensor

        if model_is_moe:
            logger.info(f"[Rank 0] Filtered {mlp_filtered_count} MLP/expert modules for MoE model")
        logger.info(f"[Rank 0] Extracted {len(lora_state_dict)} LoRA parameters (PEFT format, attention-only for MoE)")
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

        NOTE: vLLM 0.12.0 has FusedMoEWithLoRA class but two blockers remain:
        1. Module validation patch needed (patches/apply_vllm_patch.py) - DONE
        2. Weight format adapter needed - PEFT format -> vLLM packed format - NOT DONE
        MLP/expert modules are filtered until weight format adapter is implemented.
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
        if 'linear_qkv.adapter' in name or 'linear_qkv.lora_' in name.lower():
            # Fused QKV - map to q_proj (vLLM will handle the fused format)
            target = 'self_attn.q_proj'
        elif 'self_attention.linear_proj' in name:
            target = 'self_attn.o_proj'
        elif 'linear_fc1' in name:
            # MLP gate/up projection - map to gate_proj
            # For MoE, this is the expert's first linear layer
            target = 'mlp.gate_proj'
        elif 'linear_fc2' in name:
            # MLP down projection
            target = 'mlp.down_proj'
        else:
            logger.warning(f"Unknown module pattern: {name}")
            return None

        peft_name = f"base_model.model.model.layers.{layer_num}.{target}.{lora_type}.weight"
        return peft_name

    def reinit_lora_weights(self, learning_rate: float | None = None) -> dict:
        """Reinitialize LoRA weights AND optimizer state for fresh session.

        Uses verl's default initialization:
        - lora_A: xavier_uniform
        - lora_B: zeros

        Also resets Adam optimizer state (exp_avg, exp_avg_sq) to prevent
        momentum from previous sessions affecting new training.

        Args:
            learning_rate: New learning rate for the session. If provided,
                updates all optimizer param_groups. Critical for session reuse.

        This allows reusing the actor for a new session with fresh weights.
        ALL ranks must call this method for distributed sync.

        Returns:
            dict with status and count of reinitialized parameters.
        """
        import torch
        import torch.nn.init as init
        from megatron.core.optimizer import ChainedOptimizer

        logger.info(f"[Rank {self.rank}] reinit_lora_weights: ENTRY (lr={learning_rate})")
        print(f"[REINIT DEBUG Rank {self.rank}] reinit_lora_weights ENTRY lr={learning_rate}", flush=True)

        # Must use train_mode context for param_offload
        reinit_count = 0
        opt_state_reset_count = 0
        lr_updated = False

        with self.engine.train_mode():
            # Access model parameters (unwrap from list/DDP like verl does)
            from verl.utils.megatron_utils import unwrap_model
            model = unwrap_model(self.engine.module)
            # Unwrap nested lists until we get a module
            while isinstance(model, list):
                model = model[0]

            # Find and reinitialize all LoRA parameters
            for name, param in model.named_parameters():
                name_lower = name.lower()
                if 'lora' not in name_lower and 'adapter' not in name_lower:
                    continue

                if not param.requires_grad:
                    continue

                # Determine if this is lora_A or lora_B type
                is_lora_a = ('lora_a' in name_lower or 'linear_in' in name_lower)
                is_lora_b = ('lora_b' in name_lower or 'linear_out' in name_lower)

                if is_lora_a:
                    # xavier_uniform for lora_A
                    init.xavier_uniform_(param.data)
                    reinit_count += 1
                elif is_lora_b:
                    # zeros for lora_B
                    init.zeros_(param.data)
                    reinit_count += 1

            # Zero gradients
            if hasattr(self.engine, 'optimizer') and self.engine.optimizer is not None:
                self.engine.optimizer.zero_grad(set_to_none=True)

            # Reset optimizer state (Adam momentum and variance)
            # This is critical: without resetting, accumulated momentum from
            # previous session causes unexpected convergence behavior
            optimizer = self.engine.optimizer
            logger.info(f"[Rank {self.rank}] Optimizer type: {type(optimizer)}")
            if optimizer is not None:
                # Handle ChainedOptimizer (multiple optimizers)
                def iter_optimizers(opt):
                    if isinstance(opt, ChainedOptimizer):
                        return opt.chained_optimizers
                    return [opt]

                opt_list = list(iter_optimizers(optimizer))
                logger.info(f"[Rank {self.rank}] Number of optimizers in chain: {len(opt_list)}")

                for i, _opt in enumerate(opt_list):
                    logger.info(f"[Rank {self.rank}] Optimizer {i}: type={type(_opt)}, has .optimizer={hasattr(_opt, 'optimizer')}")

                    # Update learning rate in param_groups (critical for session reuse!)
                    if learning_rate is not None and hasattr(_opt, 'param_groups'):
                        old_lr = _opt.param_groups[0].get('lr', 'N/A') if _opt.param_groups else 'N/A'
                        for pg in _opt.param_groups:
                            pg['lr'] = learning_rate
                        lr_updated = True
                        logger.info(f"[Rank {self.rank}] Updated optimizer {i} LR: {old_lr} -> {learning_rate}")

                    # Access the underlying PyTorch optimizer
                    if hasattr(_opt, 'optimizer') and _opt.optimizer is not None:
                        inner_opt = _opt.optimizer
                        logger.info(f"[Rank {self.rank}] Inner optimizer type: {type(inner_opt)}")
                        state = inner_opt.state
                        state_count = len(state)
                        logger.info(f"[Rank {self.rank}] Optimizer state has {state_count} entries BEFORE clear, state type: {type(state)}")

                        # Update learning rate in inner optimizer's param_groups too
                        if learning_rate is not None and hasattr(inner_opt, 'param_groups'):
                            for pg in inner_opt.param_groups:
                                pg['lr'] = learning_rate

                        # CRITICAL FIX: Clear entire optimizer state dict instead of zeroing
                        # individual entries. With distributed optimizer + param_offload, the
                        # param objects used as keys may change identity when offloaded/reloaded.
                        # Clearing forces fresh state initialization on next training step.
                        #
                        # Handle ProxyDict from ChainedOptimizer - it wraps multiple optimizer
                        # states and doesn't have .clear(). Access underlying dicts directly.
                        print(f"[REINIT DEBUG Rank {self.rank}] state type={type(state).__name__}, has _inner_dicts={hasattr(state, '_inner_dicts')}, has clear={hasattr(state, 'clear')}, count={state_count}", flush=True)
                        if hasattr(state, '_inner_dicts'):
                            # ProxyDict from ChainedOptimizer
                            for inner_dict in state._inner_dicts:
                                inner_dict.clear()
                            logger.info(f"[Rank {self.rank}] Cleared ProxyDict optimizer state ({state_count} entries from {len(state._inner_dicts)} inner dicts)")
                            print(f"[REINIT DEBUG Rank {self.rank}] Cleared ProxyDict, now len={len(state)}", flush=True)
                        elif hasattr(state, 'clear'):
                            # Regular dict
                            state.clear()
                            logger.info(f"[Rank {self.rank}] Cleared optimizer state dict ({state_count} entries)")
                            print(f"[REINIT DEBUG Rank {self.rank}] Cleared dict via .clear(), now len={len(state)}", flush=True)
                        else:
                            # Unknown type - try to clear via iteration
                            keys = list(state.keys()) if hasattr(state, 'keys') else []
                            for key in keys:
                                del state[key]
                            logger.info(f"[Rank {self.rank}] Cleared optimizer state via key deletion ({len(keys)} entries)")
                            print(f"[REINIT DEBUG Rank {self.rank}] Cleared via key deletion, now len={len(state)}", flush=True)
                        opt_state_reset_count = state_count
                    else:
                        logger.warning(f"[Rank {self.rank}] Optimizer {i} has no inner optimizer or it's None")

        # Update instance learning_rate for future reference
        if learning_rate is not None:
            self.learning_rate = learning_rate

        logger.info(f"[Rank {self.rank}] Reinitialized {reinit_count} LoRA params, reset {opt_state_reset_count} optimizer states, lr_updated={lr_updated}")
        return {"status": "ok", "reinit_count": reinit_count, "opt_state_reset": opt_state_reset_count, "lr_updated": lr_updated, "learning_rate": learning_rate}

    def get_optimizer_info(self) -> dict:
        """Return detailed info about optimizer structure for debugging."""
        from megatron.core.optimizer import ChainedOptimizer

        info = {
            "rank": self.rank,
            "optimizer_type": str(type(self.engine.optimizer)),
            "has_optimizer": self.engine.optimizer is not None,
        }

        if self.engine.optimizer is None:
            return info

        optimizer = self.engine.optimizer

        def iter_optimizers(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        opt_list = list(iter_optimizers(optimizer))
        info["num_optimizers"] = len(opt_list)
        info["optimizer_details"] = []

        for i, _opt in enumerate(opt_list):
            opt_info = {
                "index": i,
                "type": str(type(_opt)),
                "has_inner_optimizer": hasattr(_opt, 'optimizer') and _opt.optimizer is not None,
            }
            if hasattr(_opt, 'optimizer') and _opt.optimizer is not None:
                inner_opt = _opt.optimizer
                opt_info["inner_type"] = str(type(inner_opt))
                opt_info["state_count"] = len(inner_opt.state)
                # Sample first few state entries
                state_samples = []
                for j, (param_id, param_state) in enumerate(inner_opt.state.items()):
                    if j >= 3:
                        break
                    sample = {
                        "param_type": str(type(param_id)),
                        "state_keys": list(param_state.keys()),
                    }
                    if 'exp_avg' in param_state:
                        sample["exp_avg_norm"] = param_state['exp_avg'].norm().item()
                    if 'exp_avg_sq' in param_state:
                        sample["exp_avg_sq_norm"] = param_state['exp_avg_sq'].norm().item()
                    state_samples.append(sample)
                opt_info["state_samples"] = state_samples
            info["optimizer_details"].append(opt_info)

        return info

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
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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

        # Return state_dict and peft_config for vLLM multi-LoRA registration
        meta["state_dict"] = state_dict
        meta["peft_config"] = config
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
        # PACK: try to colocate but allow multi-node for large models (K2: 16+ GPUs)
        # STRICT_PACK would require single node, blocking on 8-GPU nodes
        self.placement_group = ray.util.placement_group(
            bundles,
            strategy="PACK",
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
        session_id: str | None = None,  # Accepted for API consistency with TrainingWorker
    ) -> dict:
        """Run forward-backward on all workers.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type.
            loss_fn_config: Optional loss config.
            session_id: Unused - session management handled via swap_session().

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
        # Note: session_id is not used here. For Megatron, session state is managed
        # via swap_session() before calling forward_backward. This parameter exists
        # for API consistency with TrainingWorker.
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
        session_id: str | None = None,  # Accepted for API consistency with TrainingWorker
    ) -> dict:
        """Run forward pass only on all workers. Returns per-token logprobs.

        Similar to forward_backward but skips gradient computation.
        Used for computing reference model logprobs in DPO/SL.

        Args:
            data_items: List of Tinker Datum dicts.
            session_id: Unused - session management handled via swap_session().

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

    def optim_step(
        self,
        learning_rate: float,
        session_id: str | None = None,  # Accepted for API consistency with TrainingWorker
    ) -> dict:
        """Run optimizer step on all workers.

        WARNING: With param_offload=True, gradients are zeroed when entering train_mode.
        Use train_step() instead for combined forward_backward + optim_step.

        Args:
            learning_rate: Learning rate for this step.
            session_id: Unused - session management handled via swap_session().
        """
        futures = [w.optim_step.remote(learning_rate) for w in self.workers]
        ray.get(futures)

        self._step_count += 1
        return {"metrics": {"step": self._step_count}}

    def train_step(
        self,
        data_items: list[dict],
        loss_fn: str,
        loss_fn_config: dict,
        learning_rate: float,
    ) -> dict:
        """Run forward, backward, and optimizer step in a single train_mode context.

        This is the correct way to train with param_offload=True. Keeps all operations
        in one train_mode context so gradients survive for optimizer step.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type.
            loss_fn_config: Loss function config.
            learning_rate: Learning rate.

        Returns:
            Dict with loss metrics and optimizer results.
        """
        # Call all workers in parallel
        futures = [
            w.train_step.remote(data_items, loss_fn, loss_fn_config, learning_rate)
            for w in self.workers
        ]
        results = ray.get(futures)

        # Rank 0 returns the actual result
        rank0_result = results[0]

        self._step_count += 1

        # Extract metrics from rank 0 result
        loss_value = rank0_result.get("loss_value", 0.0)
        num_tokens = rank0_result.get("num_tokens", 0)
        clip_frac_sum = rank0_result.get("clip_frac_sum", 0.0)
        ratio_mean_sum = rank0_result.get("ratio_mean_sum", 0.0)
        n_ppo_results = rank0_result.get("n_ppo_results", 0)
        grad_norm = rank0_result.get("grad_norm", 0.0)
        lr = rank0_result.get("lr", learning_rate)

        n_samples = len(data_items) or 1

        metrics = {
            "loss:mean": loss_value / n_samples if n_samples > 0 else loss_value,
            "num_samples:sum": float(n_samples),
            "num_tokens:sum": float(num_tokens),
            "grad_norm": grad_norm,
            "lr": lr,
            "step": self._step_count,
        }
        if n_ppo_results > 0:
            metrics["clipfrac:mean"] = clip_frac_sum / n_ppo_results
            metrics["ratio:mean"] = ratio_mean_sum / n_ppo_results

        logger.info(
            f"[MegatronWorkerGroup] train_step: loss={loss_value:.4f}, "
            f"grad_norm={grad_norm:.4f}, step={self._step_count}"
        )

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": rank0_result.get("loss_fn_outputs", []),
            "metrics": metrics,
        }

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

    def reinit_lora_weights(self, learning_rate: float | None = None) -> dict:
        """Reinitialize LoRA weights to fresh random state on all workers.

        This must be called when reusing an actor for a new session to ensure
        fresh weights instead of inheriting trained weights from previous session.

        Args:
            learning_rate: New learning rate for the session. If provided,
                updates all optimizer param_groups on all workers.

        Returns:
            dict with status and total count of reinitialized parameters.
        """
        logger.info(f"[MegatronWorkerGroup] reinit_lora_weights: reinitializing on all workers (lr={learning_rate})")

        # Call reinit on all workers (required for distributed sync)
        futures = [w.reinit_lora_weights.remote(learning_rate) for w in self.workers]
        results = ray.get(futures)

        # Aggregate results
        total_reinit = sum(r.get("reinit_count", 0) for r in results)
        total_opt_state_reset = sum(r.get("opt_state_reset", 0) for r in results)
        lr_updated = any(r.get("lr_updated", False) for r in results)

        # Reset step count for fresh training
        self._step_count = 0

        # Update instance learning_rate
        if learning_rate is not None:
            self.learning_rate = learning_rate

        logger.info(f"[MegatronWorkerGroup] reinit_lora_weights: reinitialized {total_reinit} params, reset {total_opt_state_reset} optimizer states, lr_updated={lr_updated}")
        return {"status": "ok", "reinit_count": total_reinit, "opt_state_reset": total_opt_state_reset, "lr_updated": lr_updated, "learning_rate": learning_rate}

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

    def get_optimizer_info(self) -> dict:
        """Get detailed optimizer info for debugging."""
        futures = [self.workers[0].get_optimizer_info.remote()]
        results = ray.get(futures)
        return results[0]

    def shutdown(self):
        """Shutdown all workers and release placement group."""
        # First graceful shutdown
        for w in self.workers:
            try:
                ray.get(w.shutdown.remote(), timeout=5)
            except Exception:
                pass

        # Then force kill all workers to release GPU memory
        for w in self.workers:
            try:
                ray.kill(w, no_restart=True)
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
                # Also update unified resource pool
                from tinker_server.backend.resource_pool import get_resource_pool
                actor_name = self._make_actor_name(base_model, entry.max_lora_rank)
                resource_pool = get_resource_pool()
                pool_entry = resource_pool.get(actor_name)  # This calls touch()
                if pool_entry:
                    pool_entry.current_session = session_id

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

            # Phase 9: Check available GPUs and evict LRU actors if necessary
            from tinker_server.backend.resource_pool import get_resource_pool, ActorType
            resource_pool = get_resource_pool()
            resource_pool.ensure_gpus_available(num_gpus)

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

            # Register with unified resource pool for LRU tracking
            resource_pool.register(
                actor_name=actor_name,
                actor_type=ActorType.MEGATRON,
                num_gpus=num_gpus,
                actor_handle=actor,
                namespace=PERSISTENT_NAMESPACE,
                base_model=base_model,
                session_id=session_id,
            )

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
    from tinker_server.backend.resource_pool import get_resource_pool, ActorType

    config = distributed_config or DistributedConfig()
    num_gpus = config.world_size

    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    resource_pool = get_resource_pool()

    # Try to get existing persistent actor
    try:
        actor = ray.get_actor(PERSISTENT_MEGATRON_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        logger.info(
            f"Connected to existing Megatron actor: {PERSISTENT_MEGATRON_ACTOR_NAME}"
        )
        # Register with resource pool (reconnection case)
        resource_pool.register(
            actor_name=PERSISTENT_MEGATRON_ACTOR_NAME,
            actor_type=ActorType.MEGATRON,
            num_gpus=num_gpus,
            actor_handle=actor,
            namespace=PERSISTENT_NAMESPACE,
            base_model=base_model,
        )
        # Reinitialize LoRA weights for fresh session
        # This ensures each new session starts with fresh random weights
        # instead of inheriting trained weights from previous session
        # CRITICAL: Pass learning_rate to update optimizer param_groups
        logger.info(f"Reinitializing LoRA weights for new session (lr={learning_rate})...")
        print(f"[API SERVER] Calling reinit_lora_weights with lr={learning_rate}", flush=True)
        result = ray.get(actor.reinit_lora_weights.remote(learning_rate))
        print(f"[API SERVER] reinit_lora_weights returned: reinit_count={result.get('reinit_count', 0)}, opt_state_reset={result.get('opt_state_reset', 0)}, lr_updated={result.get('lr_updated', False)}", flush=True)
        logger.info(f"LoRA weights reinitialized: {result.get('reinit_count', 0)} params, lr_updated={result.get('lr_updated', False)}")
        return actor
    except ValueError:
        # Actor doesn't exist, create new one
        logger.info(
            f"Creating new detached Megatron actor: {PERSISTENT_MEGATRON_ACTOR_NAME}"
        )

    # Phase 9: Check available GPUs and evict LRU actors if necessary
    resource_pool.ensure_gpus_available(num_gpus)

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

    # Register with unified resource pool for LRU tracking
    resource_pool.register(
        actor_name=PERSISTENT_MEGATRON_ACTOR_NAME,
        actor_type=ActorType.MEGATRON,
        num_gpus=num_gpus,
        actor_handle=actor,
        namespace=PERSISTENT_NAMESPACE,
        base_model=base_model,
    )

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
    from tinker_server.backend.resource_pool import get_resource_pool

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

        # Unregister from resource pool
        resource_pool = get_resource_pool()
        resource_pool.unregister(PERSISTENT_MEGATRON_ACTOR_NAME)

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
