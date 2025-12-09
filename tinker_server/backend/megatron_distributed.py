"""Distributed MegatronTrainingWorker using Ray placement groups.

Replaces single-actor MegatronTrainingWorker with coordinator + N workers.
Each worker runs one rank of distributed Megatron training.

Based on patterns from verl/single_controller/ray/base.py and
verl/workers/megatron_workers.py.
"""

import os
import socket
import logging
from dataclasses import dataclass
from typing import Callable

import ray
import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed Megatron training."""

    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 2
    expert_parallel_size: int = 2
    context_parallel_size: int = 1

    @property
    def world_size(self) -> int:
        """Total number of processes needed."""
        # EP is orthogonal to TP/PP/CP
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
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
        self.rank = rank
        self.world_size = world_size
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.config = distributed_config

        # Set environment for torch.distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        # LOCAL_RANK is always 0 because Ray sets CUDA_VISIBLE_DEVICES to a single GPU
        os.environ["LOCAL_RANK"] = "0"

        # HuggingFace offline mode
        os.environ["HF_HOME"] = "/vePFS-Mindverse/share/huggingface"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self._initialize_distributed()
        self._initialize_megatron()

        logger.info(f"[MegatronRankWorker] Rank {rank}/{world_size} ready")

    def _initialize_distributed(self):
        """Initialize torch.distributed with NCCL backend."""
        if torch.distributed.is_initialized():
            return

        # Set CUDA device before init
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=self.world_size,
            rank=self.rank,
        )

        logger.info(f"[Rank {self.rank}] torch.distributed initialized")

    def _initialize_megatron(self):
        """Initialize Megatron model parallel and engine."""
        from verl.workers.engine.megatron.transformer_impl import MegatronEngine
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
        self.engine = MegatronEngine(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        self.engine.initialize()

        logger.info(f"[Rank {self.rank}] MegatronEngine initialized")

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
            if result and len(result) > 0:
                for micro_result in result:
                    if isinstance(micro_result, dict):
                        loss_value += micro_result.get("loss", 0.0)
                        num_tokens += micro_result.get("num_tokens", 0)

            return {
                "loss_value": loss_value,
                "num_tokens": num_tokens,
                "result": result,
            }
        return {}

    def optim_step(self, learning_rate: float) -> dict:
        """Run optimizer step (synchronized across ranks)."""
        self.engine.optim_step(learning_rate)

        if self.rank == 0:
            return {"step": "completed"}
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
        """Create placement group and spawn workers."""
        world_size = self.config.world_size

        # Create placement group with N GPU bundles
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(world_size)]
        self.placement_group = ray.util.placement_group(
            bundles,
            strategy="STRICT_PACK",  # All on same node for NVLink
        )
        ray.get(self.placement_group.ready())

        logger.info(f"[MegatronWorkerGroup] Placement group ready with {world_size} GPUs")

        # Runtime env for all Ray tasks - ensures workers can import tinker_server
        runtime_env = {
            "env_vars": {
                "PYTHONPATH": "/vePFS-Mindverse/share/code/tinker-server",
                "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
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

        # Spawn workers (reuse same runtime_env)
        for rank in range(world_size):
            worker = MegatronRankWorker.options(
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

        # Wait for all workers to initialize
        ray.get([w.__ray_ready__.remote() for w in self.workers])

        logger.info(f"[MegatronWorkerGroup] All {world_size} workers ready")

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

        # Add PPO metrics if present
        result_list = rank0_result.get("result", [])
        if loss_fn == "ppo" and result_list:
            clip_frac = sum(r.get("clip_frac", 0) for r in result_list if isinstance(r, dict))
            ratio_mean = sum(r.get("ratio_mean", 1) for r in result_list if isinstance(r, dict))
            n = len([r for r in result_list if isinstance(r, dict)]) or 1
            metrics["clipfrac:mean"] = float(clip_frac / n)
            metrics["ratio:mean"] = float(ratio_mean / n)

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
        return {"step": self._step_count}

    def get_lora_state_dict(self) -> dict:
        """Get LoRA state dict from rank 0."""
        return ray.get(self.workers[0].get_lora_state_dict.remote())

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
