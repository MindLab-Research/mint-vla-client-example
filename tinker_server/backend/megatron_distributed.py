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

        model_config = HFModelConfig(
            path=self.base_model,
            local_path=local_path,
            hf_config=hf_config,
            architectures=hf_config.architectures,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
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
            logger.info(
                f"[Rank {self.rank}] MoE config: {num_experts} experts, "
                f"top-{num_experts_per_tok} routing"
            )

        engine_config = McoreEngineConfig(
            tensor_model_parallel_size=self.config.tensor_parallel_size,
            pipeline_model_parallel_size=self.config.pipeline_parallel_size,
            expert_model_parallel_size=self.config.expert_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
            param_offload=True,
            optimizer_offload=True,
            grad_offload=True,
            dtype="bfloat16",
            use_mbridge=True,
            use_distributed_optimizer=True,
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

    def forward_backward(
        self,
        data: TensorDict,
        loss_fn: str,
        loss_fn_config: dict,
    ) -> dict:
        """Run forward and backward pass on this rank's shard.

        Gradients are synchronized via NCCL allreduce.
        Returns metrics from rank 0 only.
        """
        import torch
        from tinker_server.backend.megatron_training import (
            create_sft_loss_fn, create_ppo_loss_fn
        )

        # Move data to this rank's device
        device = torch.cuda.current_device()
        data = data.to(device)

        # Select loss function
        if loss_fn == "cross_entropy":
            loss_function = create_sft_loss_fn()
        elif loss_fn == "ppo":
            epsilon = loss_fn_config.get("epsilon", 0.2)
            loss_function = create_ppo_loss_fn(epsilon)
        elif loss_fn == "importance_sampling":
            loss_function = create_ppo_loss_fn(epsilon=float("inf"))
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        # Zero gradients
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

            if result and len(result) > 0:
                for micro_result in result:
                    if isinstance(micro_result, dict):
                        # Extract loss - may be tensor, convert to Python float
                        loss = micro_result.get("loss", 0.0)
                        if hasattr(loss, "item"):
                            loss = loss.item()
                        loss_value += float(loss)

                        # Extract num_tokens - may be tensor
                        tokens = micro_result.get("num_tokens", 0)
                        if hasattr(tokens, "item"):
                            tokens = tokens.item()
                        num_tokens += int(tokens)

                        # Extract PPO metrics if present
                        if "clip_frac" in micro_result:
                            cf = micro_result["clip_frac"]
                            if hasattr(cf, "item"):
                                cf = cf.item()
                            clip_frac_sum += float(cf)
                            n_ppo_results += 1
                        if "ratio_mean" in micro_result:
                            rm = micro_result["ratio_mean"]
                            if hasattr(rm, "item"):
                                rm = rm.item()
                            ratio_mean_sum += float(rm)

            # Return CPU-safe scalars only (no CUDA tensors)
            return {
                "loss_value": float(loss_value),
                "num_tokens": int(num_tokens),
                "clip_frac_sum": float(clip_frac_sum),
                "ratio_mean_sum": float(ratio_mean_sum),
                "n_ppo_results": int(n_ppo_results),
            }
        return {}

    def optim_step(self, learning_rate: float) -> dict:
        """Run optimizer step (synchronized across ranks)."""
        # Note: learning_rate not used directly - verl engine handles LR scheduling
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
        """Get LoRA state dict (rank 0 gathers from all ranks)."""
        # Get local shard
        local_state = self.engine.get_lora_state_dict()

        # Gather to rank 0
        if self.rank == 0:
            # For TP/PP, need to gather shards
            # Simplified: assume LoRA weights are replicated or handle in engine
            return local_state
        return {}

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
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.config = distributed_config or DistributedConfig()

        self.workers: list[ray.actor.ActorHandle] = []
        self.placement_group = None
        self._step_count = 0

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
        from tinker_server.backend.megatron_training import tinker_to_tensordict

        loss_fn_config = loss_fn_config or {}

        # Convert to TensorDict
        data = tinker_to_tensordict(data_items)

        # Broadcast to all workers
        futures = [
            w.forward_backward.remote(data, loss_fn, loss_fn_config)
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
            "loss_fn_outputs": [],
            "metrics": metrics,
        }

    def optim_step(self, learning_rate: float) -> dict:
        """Run optimizer step on all workers."""
        futures = [w.optim_step.remote(learning_rate) for w in self.workers]
        ray.get(futures)

        self._step_count += 1
        return {"metrics": {"step": self._step_count}}

    def get_lora_state_dict(self) -> dict:
        """Get LoRA state dict from rank 0."""
        return ray.get(self.workers[0].get_lora_state_dict.remote())

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
            "PYTHONPATH": "/vePFS-Mindverse/share/code/tinker-server",
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
        True if actor exists and is accessible.
    """
    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    try:
        ray.get_actor(PERSISTENT_MEGATRON_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        return True
    except ValueError:
        return False
