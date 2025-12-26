"""Multi-LoRA inference engine for multi-tenant serving.

Provides shared vLLM engine with per-sampling-session LoRA weights.
Each sampling session gets frozen weights via unique lora_int_id.
Supports GPU slots with CPU cache for overflow (LRU eviction).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import ray

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default configuration for 80GB GPU
DEFAULT_MAX_LORAS = 64  # GPU slots (~2.5GB for rank-32 Qwen-7B)
DEFAULT_MAX_CPU_LORAS = 1024  # CPU cache for evicted adapters
DEFAULT_MAX_LORA_RANK = 64  # Maximum supported rank

# Well-known name for persistent vLLM actor
PERSISTENT_VLLM_ACTOR_NAME = "tinker_vllm_server"
# Fixed namespace for persistent actors (without this, each process gets random namespace)
PERSISTENT_NAMESPACE = "tinker"

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH


def _get_actor_node_id(actor_handle: ray.actor.ActorHandle) -> str | None:
    """Get the node_id where an actor is running.

    Uses Ray's state API to get actor location.
    Returns None if unable to determine.
    """
    try:
        actor_id = actor_handle._actor_id
        # Convert ActorID to hex string for the state API
        actor_id_hex = actor_id.hex()
        from ray._private.state import actors as state_actors
        actor_info = state_actors(actor_id_hex)
        if actor_info:
            # NodeID is nested under Address
            address = actor_info.get("Address", {})
            return address.get("NodeID")
    except Exception as e:
        logger.debug(f"Could not get node_id for actor: {e}")
    return None


@dataclass
class LoRASlotInfo:
    """Metadata for a loaded LoRA adapter."""

    lora_int_id: int
    sampling_session_id: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class LoRARegistry:
    """Maps sampling_session_id to lora_int_id with LRU tracking.

    Each sampling session gets a unique lora_int_id for frozen weights.
    Tracks usage for LRU eviction when GPU slots are exhausted.
    """

    def __init__(self):
        self._session_to_id: dict[str, int] = {}
        self._id_to_session: dict[int, str] = {}
        self._slot_info: dict[int, LoRASlotInfo] = {}
        self._lru_order: OrderedDict[int, None] = OrderedDict()  # LRU at front
        self._next_id: int = 1
        self._lock = asyncio.Lock()

    async def allocate(self, sampling_session_id: str) -> int:
        """Allocate a unique lora_int_id for a sampling session.

        Args:
            sampling_session_id: Unique identifier for the sampling session.

        Returns:
            Unique lora_int_id for this session.

        Raises:
            ValueError: If session already has an allocated ID.
        """
        async with self._lock:
            if sampling_session_id in self._session_to_id:
                raise ValueError(
                    f"Session {sampling_session_id} already has lora_int_id "
                    f"{self._session_to_id[sampling_session_id]}"
                )

            lora_id = self._next_id
            self._next_id += 1

            self._session_to_id[sampling_session_id] = lora_id
            self._id_to_session[lora_id] = sampling_session_id
            self._slot_info[lora_id] = LoRASlotInfo(
                lora_int_id=lora_id,
                sampling_session_id=sampling_session_id,
            )
            self._lru_order[lora_id] = None

            logger.debug(
                f"Allocated lora_int_id={lora_id} for session {sampling_session_id}"
            )
            return lora_id

    async def get_lora_id(self, sampling_session_id: str) -> int | None:
        """Get lora_int_id for a sampling session and update LRU.

        Args:
            sampling_session_id: The sampling session identifier.

        Returns:
            The lora_int_id if session exists, None otherwise.
        """
        async with self._lock:
            lora_id = self._session_to_id.get(sampling_session_id)
            if lora_id is not None:
                # Update LRU order (move to end = most recently used)
                self._lru_order.move_to_end(lora_id)
                # Update last_used timestamp
                if lora_id in self._slot_info:
                    self._slot_info[lora_id].last_used = time.time()
            return lora_id

    async def get_lru_candidates(self, count: int) -> list[int]:
        """Get the least recently used lora_int_ids for eviction.

        Args:
            count: Number of candidates to return.

        Returns:
            List of lora_int_ids in LRU order (oldest first).
        """
        async with self._lock:
            candidates = []
            for lora_id in self._lru_order:
                if len(candidates) >= count:
                    break
                candidates.append(lora_id)
            return candidates

    async def remove(self, lora_id: int) -> str | None:
        """Remove a lora_int_id from the registry.

        Args:
            lora_id: The lora_int_id to remove.

        Returns:
            The sampling_session_id that was removed, or None if not found.
        """
        async with self._lock:
            session_id = self._id_to_session.pop(lora_id, None)
            if session_id:
                self._session_to_id.pop(session_id, None)
                self._slot_info.pop(lora_id, None)
                self._lru_order.pop(lora_id, None)
                logger.debug(f"Removed lora_int_id={lora_id} (session {session_id})")
            return session_id

    async def count(self) -> int:
        """Get the number of registered sessions."""
        async with self._lock:
            return len(self._session_to_id)

    async def list_sessions(self) -> list[str]:
        """List all registered sampling session IDs."""
        async with self._lock:
            return list(self._session_to_id.keys())


@dataclass
class GenerateResult:
    """Result of a generate call."""

    token_ids: list[int]
    logprobs: list[float] | None = None
    stop_reason: str | None = None


class MultiLoRAInferenceEngine:
    """Shared inference engine with multi-tenant LoRA support.

    Single vLLM engine serves multiple sampling sessions, each with
    frozen LoRA weights identified by unique lora_int_id.

    Features:
    - GPU slots (max_loras) for hot adapters
    - CPU cache (max_cpu_loras) for evicted adapters
    - Automatic swap between GPU and CPU on demand
    - Per-session weight isolation (Tinker SDK semantics)
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        max_loras: int = DEFAULT_MAX_LORAS,
        max_cpu_loras: int = DEFAULT_MAX_CPU_LORAS,
        max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
        actor_name: str | None = None,
        quantization: str | None = None,  # "fp8" for FP8 models like K2
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_lora_rank = max_lora_rank
        # Custom actor name for multi-model support (one actor per base model)
        self.actor_name = actor_name or PERSISTENT_VLLM_ACTOR_NAME

        self.registry = LoRARegistry()
        self.server = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the shared vLLM engine with multi-LoRA support.

        Uses a detached Ray actor that survives server restarts.
        First tries to connect to existing actor, creates new one if not found.
        """
        async with self._init_lock:
            if self._initialized:
                return

            if not ray.is_initialized():
                # Use fixed namespace so detached actors can be found across process restarts
                ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

            # Try to get existing persistent actor
            # Note: ray.get_actor succeeds even for dead actors (name still registered)
            # We must verify the actor is alive AND engine is initialized
            try:
                self.server = ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                # Health check: try calling a method to verify actor is alive
                # This will raise RayActorError if actor is dead
                try:
                    ray.get(self.server.__ray_ready__.remote(), timeout=5)

                    # Check if engine is actually initialized (not just actor alive)
                    # A broken actor can be "alive" but have failed engine init
                    engine_ready = ray.get(self.server.is_engine_ready.remote(), timeout=10)
                    if not engine_ready:
                        logger.warning(
                            f"vLLM actor {self.actor_name} has broken engine, killing and recreating"
                        )
                        try:
                            ray.kill(self.server, no_restart=True)
                        except Exception as kill_err:
                            logger.debug(f"Failed to kill broken actor: {kill_err}")
                        self.server = None
                        self._initialized = False
                    else:
                        logger.info(
                            f"Connected to existing persistent vLLM actor: {self.actor_name}"
                        )
                        self._initialized = True

                        # Register existing actor with resource pool for LRU tracking
                        # Include node_id for proper per-node GPU scheduling
                        from tinker_server.backend.resource_pool import get_resource_pool, ActorType
                        total_gpus = self.tensor_parallel_size * self.data_parallel_size
                        resource_pool = get_resource_pool()
                        actor_node_id = _get_actor_node_id(self.server)
                        logger.info(f"Registering existing actor {self.actor_name} with ResourcePool (node={actor_node_id[:8] if actor_node_id else 'unknown'})")
                        resource_pool.register(
                            actor_name=self.actor_name,
                            actor_type=ActorType.VLLM,
                            num_gpus=total_gpus,
                            actor_handle=self.server,
                            namespace=PERSISTENT_NAMESPACE,
                            base_model=self.model_path,
                            node_id=actor_node_id,
                        )
                        # Mark as ready since it's an existing actor that responded to health check
                        resource_pool.mark_ready(self.actor_name)
                        return
                except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                    # Actor is dead or unresponsive, need to create new one
                    logger.warning(
                        f"vLLM actor {self.actor_name} is dead/unresponsive, creating new one"
                    )
                    # Must kill dead actor to free the name for reuse
                    # Ray keeps names registered even for dead actors
                    try:
                        ray.kill(self.server, no_restart=True)
                        logger.debug(f"Killed dead actor to free name: {self.actor_name}")
                    except Exception as kill_err:
                        logger.debug(f"Could not kill dead actor: {kill_err}")
                    self.server = None
                    # Reset initialized flag so we'll create new actor below
                    self._initialized = False
            except ValueError:
                # Actor doesn't exist, create new one
                logger.info(
                    f"No existing vLLM actor found, creating new detached actor: {self.actor_name}"
                )

            from verl.workers.config import HFModelConfig, RolloutConfig
            from verl.workers.rollout.replica import RolloutMode

            from .verl_inference import _create_extended_server_class

            # Pass multi-LoRA config to vLLM engine
            ExtendedVLLMHttpServer = _create_extended_server_class(
                max_loras=self.max_loras,
                max_cpu_loras=self.max_cpu_loras,
            )

            # Compute total GPUs needed for MoE models
            # For EP (expert parallelism), total_gpus = TP * DP
            total_gpus = self.tensor_parallel_size * self.data_parallel_size

            # Ensure GPUs available, evicting idle actors if needed (LRU)
            # This is critical to prevent server hangs when no GPUs are free.
            from tinker_server.backend.resource_pool import get_resource_pool
            resource_pool = get_resource_pool()
            try:
                resource_pool.ensure_gpus_available(total_gpus)
            except ValueError as e:
                # Unable to free enough GPUs even after eviction
                logger.error(f"Cannot create vLLM actor: {e}")
                raise

            # Build engine_kwargs for expert parallelism
            # Pass enable_expert_parallel directly to vLLM, bypassing verl's worker-based EP
            # This allows vLLM to handle EP internally via multiprocessing
            engine_kwargs = {}
            if self.data_parallel_size > 1:
                engine_kwargs = {
                    "vllm": {
                        "enable_expert_parallel": True,
                    }
                }
                logger.info(f"Enabling expert parallelism via vLLM (DP={self.data_parallel_size})")

            # Configure rollout with multi-LoRA support
            # NOTE: Keep expert_parallel_size=1 to avoid verl's worker-based EP assertion
            # Expert parallelism is enabled via engine_kwargs instead
            rollout_config = RolloutConfig(
                name="vllm",
                tensor_model_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                prompt_length=2048,
                response_length=2048,
                max_num_seqs=256,
                dtype="auto",
                load_format="auto",
                enforce_eager=False,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
                enable_prefix_caching=True,
                disable_log_stats=True,
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
                data_parallel_size=1,  # Keep at 1 to avoid verl's worker assertion
                expert_parallel_size=1,  # Keep at 1 to avoid verl's worker assertion
                engine_kwargs=engine_kwargs,
                quantization=self.quantization,  # "fp8" for FP8 models like K2
            )
            if self.max_model_len is not None:
                rollout_config.max_model_len = self.max_model_len
            if self.quantization:
                logger.info(f"vLLM quantization enabled: {self.quantization}")

            # Model config with multi-LoRA parameters
            model_config = HFModelConfig(
                path=self.model_path,
                trust_remote_code=True,
                lora_rank=self.max_lora_rank,  # Max rank to support
                lora_adapter_path=None,  # No initial adapter
            )

            logger.info(
                f"Initializing MultiLoRAInferenceEngine: "
                f"TP={self.tensor_parallel_size}, DP={self.data_parallel_size}, total_gpus={total_gpus}, "
                f"max_loras={self.max_loras}, max_cpu_loras={self.max_cpu_loras}, "
                f"max_lora_rank={self.max_lora_rank}"
            )

            # Create detached Ray actor with well-known name
            # lifetime="detached" ensures actor survives owner process termination
            # Request total_gpus for MoE expert parallelism
            # runtime_env prepends vLLM 0.12.0 from PFS for MoE LoRA support
            #
            # Find a node with enough free GPUs to avoid memory conflicts.
            # When training and inference coexist, Megatron holds GPU memory even
            # though Ray doesn't track actual CUDA memory usage. We must place
            # vLLM on a separate node with completely free GPUs.
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            # Find a node with enough AVAILABLE GPUs (not just total GPUs).
            # Ray's node.Resources shows total, but other actors (Megatron) may hold GPUs.
            # We compute available by checking:
            # 1. GPUs used by active placement groups (Megatron training)
            # 2. GPUs used by actors tracked in ResourcePool (vLLM, dense training)
            target_node = None
            logger.debug(f"Looking for node with {total_gpus} available GPUs for vLLM")

            # Get cluster-wide available resources for validation
            cluster_available = ray.available_resources()
            cluster_gpus = cluster_available.get("GPU", 0)
            logger.debug(f"Cluster has {cluster_gpus} GPUs available")

            # Find GPUs used by active placement groups on each node
            # Each bundle in a placement group typically uses 1 GPU
            pg_table = ray.util.placement_group_table()
            gpus_used_by_pg = {}  # node_id -> count of GPUs used by placement groups
            for pg_id, pg_info in pg_table.items():
                if pg_info.get("state") == "CREATED":
                    # Count bundles per node (each bundle typically uses 1 GPU)
                    bundles_to_node = pg_info.get("bundles_to_node_id", {})
                    for bundle_idx, node_id in bundles_to_node.items():
                        gpus_used_by_pg[node_id] = gpus_used_by_pg.get(node_id, 0) + 1

            # Also get GPUs used by actors tracked in ResourcePool (vLLM, dense training)
            # This is critical - without it, we may schedule to a node that already has vLLM actors
            gpus_used_by_actors = resource_pool.gpus_used_by_node()

            # Collect candidate nodes based on AVAILABLE GPUs (total - pg_used - actor_used)
            candidates = []
            for node in ray.nodes():
                node_id = node["NodeID"]
                node_id_short = node_id[:8]
                if node["Alive"]:
                    total_res = node.get("Resources", {})
                    total_gpu = total_res.get("GPU", 0)
                    obj_store = total_res.get("object_store_memory", 0)
                    pg_gpus = gpus_used_by_pg.get(node_id, 0)
                    actor_gpus = gpus_used_by_actors.get(node_id, 0)
                    available_gpu = total_gpu - pg_gpus - actor_gpus
                    logger.debug(f"Node {node_id_short}: total={total_gpu}, pg_used={pg_gpus}, actor_used={actor_gpus}, available={available_gpu}")

                    # Node must have enough AVAILABLE GPUs (after subtracting all usage) and enough object store
                    if available_gpu >= total_gpus and obj_store > 100_000_000_000:
                        candidates.append((node_id, available_gpu))

            # Prefer nodes with NO placement groups first (to avoid GPU assignment conflicts)
            # Among those, prefer nodes with more GPUs (more room)
            if candidates:
                # Separate into "clean" nodes (no PG or actors) and "partial" nodes
                clean_nodes = [(nid, gpus) for nid, gpus in candidates
                               if gpus_used_by_pg.get(nid, 0) == 0 and gpus_used_by_actors.get(nid, 0) == 0]
                partial_nodes = [(nid, gpus) for nid, gpus in candidates
                                 if gpus_used_by_pg.get(nid, 0) > 0 or gpus_used_by_actors.get(nid, 0) > 0]

                if clean_nodes:
                    # Prefer clean nodes
                    clean_nodes.sort(key=lambda x: -x[1])  # Sort by available GPUs descending
                    target_node = clean_nodes[0][0]
                    available_gpus = clean_nodes[0][1]
                    logger.info(f"Selected clean node {target_node[:8]} with {available_gpus} available GPUs")
                else:
                    # Fall back to partial nodes
                    partial_nodes.sort(key=lambda x: -x[1])  # Sort by available GPUs descending
                    target_node = partial_nodes[0][0]
                    available_gpus = partial_nodes[0][1]
                    pg_count = gpus_used_by_pg.get(target_node, 0)
                    actor_count = gpus_used_by_actors.get(target_node, 0)
                    logger.info(f"Selected partial node {target_node[:8]} with {available_gpus} available GPUs ({pg_count} PG, {actor_count} actors)")

            scheduling_opts = {}
            if target_node:
                scheduling_opts["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=target_node,
                    soft=False,  # Hard constraint - fail if node unavailable
                )
            else:
                scheduling_opts["scheduling_strategy"] = "SPREAD"
                logger.warning("No suitable node found, using SPREAD scheduling")

            self.server = ExtendedVLLMHttpServer.options(
                num_gpus=total_gpus,
                name=self.actor_name,
                namespace=PERSISTENT_NAMESPACE,
                lifetime="detached",
                **scheduling_opts,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": PFS_PYTHONPATH,
                        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                        "HF_HUB_OFFLINE": "1",
                    }
                },
            ).remote(
                config=rollout_config,
                model_config=model_config,
                rollout_mode=RolloutMode.STANDALONE,
                workers=[],
                replica_rank=0,
                node_rank=0,
                gpus_per_node=total_gpus,
                nnodes=1,
            )

            # Run blocking ray.get() in thread executor to avoid blocking the event loop.
            # This allows the server to remain responsive during vLLM initialization.
            # Timeout scales with model size:
            # - Small models (1-2 GPUs): 300s (5 min)
            # - Medium models (4 GPUs): 600s (10 min)
            # - Large models (8+ GPUs, e.g., K2): 1800s (30 min)
            import asyncio
            loop = asyncio.get_event_loop()

            # Compute timeout based on model size (proxy: number of GPUs)
            if total_gpus >= 8:
                init_timeout = 1800  # 30 min for K2 and similar large models
            elif total_gpus >= 4:
                init_timeout = 600   # 10 min for medium MoE models
            else:
                init_timeout = 300   # 5 min for small models

            logger.info(f"Launching vLLM server (timeout={init_timeout}s for {total_gpus} GPUs)...")
            try:
                await loop.run_in_executor(
                    None,  # Use default thread pool
                    lambda: ray.get(self.server.launch_server.remote(), timeout=init_timeout)
                )
            except ray.exceptions.GetTimeoutError:
                logger.error(f"vLLM launch timed out after {init_timeout}s for {self.actor_name}")
                # Kill the stuck actor
                try:
                    ray.kill(self.server, no_restart=True)
                except Exception:
                    pass
                self.server = None
                raise RuntimeError(f"vLLM actor {self.actor_name} launch timed out after {init_timeout}s")

            self._initialized = True
            logger.info(f"MultiLoRAInferenceEngine initialized (detached actor: {self.actor_name})")

            # Register with unified resource pool for LRU tracking
            # Include node_id for proper per-node GPU scheduling
            from tinker_server.backend.resource_pool import get_resource_pool, ActorType
            resource_pool = get_resource_pool()
            actor_node_id = _get_actor_node_id(self.server)
            resource_pool.register(
                actor_name=self.actor_name,
                actor_type=ActorType.VLLM,
                num_gpus=total_gpus,
                actor_handle=self.server,
                namespace=PERSISTENT_NAMESPACE,
                base_model=self.model_path,
                node_id=actor_node_id,
            )
            # Mark as ready now that launch completed successfully
            resource_pool.mark_ready(self.actor_name)
            if actor_node_id:
                logger.info(f"vLLM actor {self.actor_name} running on node {actor_node_id[:8]}")

    async def add_lora_for_session(
        self,
        sampling_session_id: str,
        state_dict: dict,
        peft_config: dict,
    ) -> int:
        """Add frozen LoRA weights for a sampling session.

        Each call creates a new lora_int_id with frozen weights.
        Old sessions retain their weights.

        Uses tensor transfer via Ray to support distributed deployments
        where API server and inference worker have different filesystems.
        Worker saves tensors locally then creates file-based LoRARequest.
        File-based loading supports vLLM's GPU/CPU swapping.

        Auto-restarts vLLM actor if dead.

        Args:
            sampling_session_id: Unique identifier for the sampling session.
            state_dict: LoRA weight tensors (transferred via Ray object store).
            peft_config: PEFT adapter configuration dict.

        Returns:
            The allocated lora_int_id for this session.
        """
        if not self._initialized:
            await self.initialize()

        # Allocate unique ID for this session
        lora_id = await self.registry.allocate(sampling_session_id)

        # Check if we need to evict from GPU
        current_count = await self.registry.count()
        if current_count > self.max_loras:
            # vLLM handles GPU/CPU swapping automatically via max_cpu_loras
            # We just need to track in registry
            logger.debug(
                f"GPU slots full ({self.max_loras}), "
                f"vLLM will manage CPU cache for overflow"
            )

        # Transfer tensors via Ray and save locally on worker node.
        # Worker creates file-based LoRARequest for GPU/CPU swap support.
        # Auto-restart vLLM if actor died (e.g. killed for GPU allocation).
        start_time = time.time()
        try:
            await self.server.add_lora_with_id.remote(
                lora_int_id=lora_id,
                state_dict=state_dict,
                peft_config=peft_config,
            )
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
            logger.warning(f"vLLM actor dead/unresponsive, reinitializing: {e}")
            self._initialized = False
            self.server = None
            await self.initialize()
            # Retry after restart
            await self.server.add_lora_with_id.remote(
                lora_int_id=lora_id,
                state_dict=state_dict,
                peft_config=peft_config,
            )
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} "
            f"(lora_int_id={lora_id}, load_time={load_time:.3f}s)"
        )
        return lora_id

    async def generate(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
    ) -> GenerateResult:
        """Generate tokens using session-specific LoRA or base model.

        If sampling_session_id is None or has no registered LoRA, uses base model.

        Args:
            sampling_session_id: The sampling session to use, or None for base model.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            logprobs: Whether to return log probabilities.

        Returns:
            GenerateResult with generated tokens and metadata.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)

        if lora_id is not None:
            # Generate with session-specific LoRA
            result = await self.server.generate_with_lora.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                lora_int_id=lora_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=logprobs,
            )
        else:
            # Generate with base model (no LoRA)
            result = await self.server.generate_base.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=logprobs,
            )

        return GenerateResult(
            token_ids=result["token_ids"],
            logprobs=result.get("logprobs"),
            stop_reason=result.get("stop_reason"),
        )

    async def compute_logprobs(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
    ) -> list[float]:
        """Compute logprobs using session-specific LoRA or base model.

        If sampling_session_id is None or has no registered LoRA, uses base model.

        Returns logprobs[i] = log P(token[i+1] | token[0:i+1]).
        Output length is len(prompt_ids) - 1.

        Args:
            sampling_session_id: The sampling session to use, or None for base model.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.

        Returns:
            List of logprobs, length = len(prompt_ids) - 1.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)

        if lora_id is not None:
            # Compute logprobs with session-specific LoRA
            result = await self.server.compute_prompt_logprobs_with_lora.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                lora_int_id=lora_id,
            )
        else:
            # Compute logprobs with base model (no LoRA)
            result = await self.server.compute_prompt_logprobs_base.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
            )

        return list(result)

    async def remove_session(self, sampling_session_id: str) -> bool:
        """Remove a sampling session and its LoRA.

        Args:
            sampling_session_id: The session to remove.

        Returns:
            True if session was removed, False if not found.
        """
        lora_id = await self.registry.get_lora_id(sampling_session_id)
        if lora_id is None:
            return False

        # Remove from vLLM engine
        try:
            await self.server.remove_lora.remote(lora_id)
        except Exception as e:
            logger.warning(f"Failed to remove LoRA {lora_id} from engine: {e}")

        # Remove from registry
        await self.registry.remove(lora_id)
        logger.info(f"Removed session {sampling_session_id} (lora_int_id={lora_id})")
        return True

    async def shutdown(self, kill_actor: bool = False) -> None:
        """Disconnect from the engine (optionally kill the persistent actor).

        Args:
            kill_actor: If True, kill the persistent vLLM actor.
                        If False (default), just disconnect - actor survives for reuse.
        """
        if self.server is not None and kill_actor:
            try:
                ray.kill(self.server)
                logger.info("Killed persistent vLLM actor")
            except Exception as e:
                logger.warning(f"Error killing server actor: {e}")
        self.server = None
        self._initialized = False
        logger.info("MultiLoRAInferenceEngine disconnected")


def _model_to_actor_name(model_name: str) -> str:
    """Convert model name to a valid Ray actor name.

    Examples:
        "Qwen/Qwen2.5-7B-Instruct" -> "tinker_vllm_qwen2.5-7b-instruct"
        "Qwen/Qwen3-30B-A3B-Instruct-2507" -> "tinker_vllm_qwen3-30b-a3b-instruct-2507"
    """
    # Extract model part after "/"
    if "/" in model_name:
        model_part = model_name.split("/")[-1]
    else:
        model_part = model_name
    # Lowercase and replace invalid chars
    safe_name = model_part.lower().replace(" ", "_")
    return f"tinker_vllm_{safe_name}"


def _resolve_model_path(model_name: str) -> str:
    """Resolve model name to full path on PFS.

    Uses cached paths for known models.
    """
    # Map of model names to local paths
    MODEL_PATHS = {
        # Dense models
        "Qwen/Qwen2.5-7B-Instruct": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "Qwen/Qwen3-0.6B": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
        "Qwen/Qwen3-4B-Instruct-2507": "/vePFS-Mindverse/share/modelscope/models/Qwen/Qwen3-4B-Instruct-2507",
        # MoE models (all share same architecture, different checkpoints)
        "Qwen/Qwen3-30B-A3B-Instruct-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "Qwen/Qwen3-30B-A3B": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/main",
        "Qwen/Qwen3-30B-A3B-Base": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/main",
        "Qwen/Qwen3-30B-A3B-Thinking-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/main",
        # K2 models (1T params MoE, requires FP8)
        "moonshotai/Kimi-K2-Thinking": "/vePFS-Mindverse/share/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd",
    }

    if model_name in MODEL_PATHS:
        return MODEL_PATHS[model_name]

    # Fall back to model name as path
    return model_name


class MultiModelInferenceManager:
    """Manages multiple vLLM engines, one per base model.

    Provides dynamic engine creation based on model name.
    Each base model gets its own persistent vLLM actor.
    """

    def __init__(
        self,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        max_loras: int = DEFAULT_MAX_LORAS,
        max_cpu_loras: int = DEFAULT_MAX_CPU_LORAS,
        max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
    ):
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_lora_rank = max_lora_rank

        # Dict of model_name -> engine
        self._engines: dict[str, MultiLoRAInferenceEngine] = {}
        self._init_lock = asyncio.Lock()

    async def get_engine(self, model_name: str) -> MultiLoRAInferenceEngine:
        """Get or create engine for a model.

        If cached engine exists, verifies actor is alive before returning.
        If actor is dead, removes from cache and creates new engine.

        For models requiring TP > 8 (multi-node), uses MultiNodeInferenceEngine
        which leverages vLLM's native Ray distributed backend.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")

        Returns:
            Initialized inference engine for the model.
        """
        async with self._init_lock:
            if model_name in self._engines:
                engine = self._engines[model_name]
                # Check if actor is still alive before returning cached engine
                # Handle both MultiLoRAInferenceEngine and MultiNodeInferenceEngine
                actor_handle = getattr(engine, 'server', None) or getattr(engine, 'engine', None)
                if actor_handle is not None:
                    try:
                        # MultiNodeInferenceEngine uses is_ready, MultiLoRAInferenceEngine uses __ray_ready__
                        if hasattr(engine, 'server'):
                            ray.get(actor_handle.__ray_ready__.remote(), timeout=5)
                        else:
                            ray.get(actor_handle.is_ready.remote(), timeout=5)
                        # Actor alive, return cached engine
                        return engine
                    except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                        # Actor is dead, remove from cache and recreate
                        logger.warning(
                            f"Cached vLLM engine for {model_name} has dead actor, recreating"
                        )
                        del self._engines[model_name]
                else:
                    # Engine has no actor handle, remove stale entry
                    del self._engines[model_name]

            # Get model config for parallelism settings
            from tinker_server.backend.model_registry import get_model_config
            config = get_model_config(model_name)

            # Create unique actor name for this model
            actor_name = _model_to_actor_name(model_name)
            model_path = _resolve_model_path(model_name)

            # Determine quantization from model config (None = vLLM auto-detect from config.json)
            quantization = config.quantization

            # Determine max_loras: per-model override > MoE default (1) > global default
            from .model_registry import get_max_loras
            model_max_loras_override = get_max_loras(model_name)
            if model_max_loras_override is not None:
                model_max_loras = model_max_loras_override
            elif config.is_moe:
                model_max_loras = 1  # Default for MoE: minimal LoRA support
            else:
                model_max_loras = self.max_loras
            # Disable CPU LoRA cache if LoRA is disabled or MoE model
            model_max_cpu_loras = 0 if model_max_loras == 0 or config.is_moe else self.max_cpu_loras

            # Use per-model max_lora_rank override if specified (K2 needs rank 8 to fit).
            # Large MoE models have huge LoRA buffers: 384 experts × rank × hidden_size.
            from .model_registry import get_max_lora_rank
            model_max_lora_rank = get_max_lora_rank(model_name) or self.max_lora_rank

            # Use per-model gpu_memory_utilization if specified (K2 needs lower for LoRA headroom)
            model_gpu_util = config.gpu_memory_utilization or self.gpu_memory_utilization

            # Use per-model max_model_len if specified (K2 needs 128K, default 262K exceeds KV cache)
            from .model_registry import get_max_model_len
            model_max_model_len = get_max_model_len(model_name) or self.max_model_len

            # Use per-model kv_cache_dtype if specified (FP8 KV cache halves memory)
            from .model_registry import get_kv_cache_dtype
            model_kv_cache_dtype = get_kv_cache_dtype(model_name)

            # Check if model needs multi-node (TP > 8)
            # K2 requires TP=16 across 2 nodes for LoRA support
            needs_multinode = config.recommended_tp > 8

            if needs_multinode:
                # Use MultiNodeInferenceEngine with vLLM's native Ray distributed backend
                from .multinode_inference import MultiNodeInferenceEngine

                total_gpus = config.recommended_tp  # TP=16 for K2
                logger.info(
                    f"Creating multi-node vLLM engine for model {model_name}: "
                    f"actor={actor_name}, TP={config.recommended_tp}, total_gpus={total_gpus}, "
                    f"gpu_util={model_gpu_util}, quant={quantization}, "
                    f"max_loras={model_max_loras}, max_lora_rank={model_max_lora_rank}, "
                    f"max_model_len={model_max_model_len}, kv_cache_dtype={model_kv_cache_dtype}"
                )

                # Ensure GPUs available, evicting idle actors if needed (LRU)
                # This waits up to 10 minutes for resources to become available
                from tinker_server.backend.resource_pool import get_resource_pool
                resource_pool = get_resource_pool()
                try:
                    resource_pool.ensure_gpus_available(total_gpus)
                except ValueError as e:
                    logger.error(f"Cannot create multi-node vLLM actor: {e}")
                    raise

                engine = MultiNodeInferenceEngine(
                    model_path=model_path,
                    tensor_parallel_size=config.recommended_tp,
                    gpu_memory_utilization=model_gpu_util,
                    max_model_len=model_max_model_len,
                    max_loras=model_max_loras,
                    max_lora_rank=model_max_lora_rank,
                    quantization=quantization,
                    kv_cache_dtype=model_kv_cache_dtype,
                    actor_name=actor_name,
                )
            else:
                # Use standard MultiLoRAInferenceEngine for single-node models
                logger.info(
                    f"Creating vLLM engine for model {model_name}: "
                    f"actor={actor_name}, TP={config.recommended_tp}, DP={config.recommended_dp}, "
                    f"quant={quantization}, max_loras={model_max_loras}, max_lora_rank={model_max_lora_rank}"
                )

                engine = MultiLoRAInferenceEngine(
                    model_path=model_path,
                    tensor_parallel_size=config.recommended_tp,
                    data_parallel_size=config.recommended_dp,
                    gpu_memory_utilization=model_gpu_util,
                    max_model_len=model_max_model_len,
                    max_loras=model_max_loras,
                    max_cpu_loras=model_max_cpu_loras,
                    max_lora_rank=model_max_lora_rank,
                    actor_name=actor_name,
                    quantization=quantization,
                )

            await engine.initialize()

            self._engines[model_name] = engine
            logger.info(f"Engine created for {model_name}")
            return engine

    def get_engine_if_exists(self, model_name: str) -> MultiLoRAInferenceEngine | None:
        """Get engine for model if already created, None otherwise."""
        return self._engines.get(model_name)

    async def shutdown_all(self, kill_actors: bool = False) -> None:
        """Shutdown all engines.

        Args:
            kill_actors: If True, kill the persistent actors. If False, just disconnect.
        """
        for model_name, engine in self._engines.items():
            logger.info(f"Shutting down engine for {model_name}")
            await engine.shutdown(kill_actor=kill_actors)
        self._engines.clear()

    def list_models(self) -> list[str]:
        """List all models with active engines."""
        return list(self._engines.keys())


def kill_persistent_vllm_actor() -> bool:
    """Kill the persistent vLLM actor if it exists.

    Use this to force a full restart of the vLLM engine (e.g., after model changes).
    After calling this, the next server startup will create a new actor (~80s init).

    Returns:
        True if actor was killed, False if not found.
    """
    from tinker_server.backend.resource_pool import get_resource_pool

    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    try:
        actor = ray.get_actor(PERSISTENT_VLLM_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        ray.kill(actor)
        logger.info(f"Killed persistent vLLM actor: {PERSISTENT_VLLM_ACTOR_NAME}")

        # Unregister from resource pool
        resource_pool = get_resource_pool()
        resource_pool.unregister(PERSISTENT_VLLM_ACTOR_NAME)

        return True
    except ValueError:
        logger.info(f"No persistent vLLM actor found: {PERSISTENT_VLLM_ACTOR_NAME}")
        return False


def check_persistent_vllm_actor() -> bool:
    """Check if persistent vLLM actor exists and is alive.

    Returns:
        True if actor exists and is alive, False otherwise.
    """
    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    try:
        actor = ray.get_actor(PERSISTENT_VLLM_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        # ray.get_actor succeeds even for dead actors (name still registered)
        # Verify actor is alive by calling a method on it
        ray.get(actor.__ray_ready__.remote(), timeout=5)
        return True
    except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
        return False


# Global instance (initialized in app lifespan)
multi_lora_engine: MultiLoRAInferenceEngine | None = None
