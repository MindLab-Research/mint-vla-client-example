"""Multi-LoRA inference engine for multi-tenant serving.

Provides shared vLLM engine with per-sampling-session LoRA weights.
Each sampling session gets frozen weights via unique lora_int_id.
Supports GPU slots with CPU cache for overflow (LRU eviction).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import ray

from . import ray_kill

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

# Import model registry
from tinker_server.backend.model_registry import get_model_config


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
        max_num_seqs: int = 256,
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
        self.max_num_seqs = max_num_seqs
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
                            ray_kill.kill(
                                self.server,
                                reason="vllm_engine_broken",
                                actor_name=self.actor_name,
                                namespace=PERSISTENT_NAMESPACE,
                                no_restart=True,
                            )
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
                except ray.exceptions.RayActorError:
                    # Actor is dead, need to create new one
                    logger.warning(
                        f"vLLM actor {self.actor_name} is dead, creating new one"
                    )
                    # Must kill dead actor to free the name for reuse
                    # Ray keeps names registered even for dead actors
                    try:
                        ray_kill.kill(
                            self.server,
                            reason="vllm_actor_dead_free_name",
                            actor_name=self.actor_name,
                            namespace=PERSISTENT_NAMESPACE,
                            no_restart=True,
                        )
                        logger.debug(f"Killed dead actor to free name: {self.actor_name}")
                    except Exception as kill_err:
                        logger.debug(f"Could not kill dead actor: {kill_err}")
                    self.server = None
                    # Reset initialized flag so we'll create new actor below
                    self._initialized = False
                except ray.exceptions.GetTimeoutError:
                    # Actor might be busy (queued tasks) rather than dead.
                    # Killing on timeout will terminate active inference and training flows.
                    logger.warning(
                        f"vLLM actor {self.actor_name} __ray_ready__/is_engine_ready timed out; assuming busy and reusing actor"
                    )
                    logger.info(
                        f"Connected to existing persistent vLLM actor: {self.actor_name}"
                    )
                    self._initialized = True

                    from tinker_server.backend.resource_pool import get_resource_pool, ActorType
                    total_gpus = self.tensor_parallel_size * self.data_parallel_size
                    resource_pool = get_resource_pool()
                    actor_node_id = _get_actor_node_id(self.server)
                    logger.info(
                        f"Registering existing actor {self.actor_name} with ResourcePool (node={actor_node_id[:8] if actor_node_id else 'unknown'})"
                    )
                    resource_pool.register(
                        actor_name=self.actor_name,
                        actor_type=ActorType.VLLM,
                        num_gpus=total_gpus,
                        actor_handle=self.server,
                        namespace=PERSISTENT_NAMESPACE,
                        base_model=self.model_path,
                        node_id=actor_node_id,
                    )
                    resource_pool.mark_ready(self.actor_name)
                    return
            except ValueError:
                # Actor doesn't exist, create new one
                logger.info(
                    f"No existing vLLM actor found, creating new detached actor: {self.actor_name}"
                )

            from verl.workers.config import HFModelConfig, RolloutConfig
            from verl.workers.rollout.replica import RolloutMode

            from .verl_inference import _create_extended_server_class

            def _pg_name() -> str:
                return f"{self.actor_name}_pg"

            def _get_or_create_pg() -> ray.util.placement_group.PlacementGroup:
                pg_name = _pg_name()
                try:
                    pg = ray.util.get_placement_group(pg_name)
                except Exception:
                    pg = ray.util.placement_group(
                        [{"GPU": total_gpus, "CPU": 1}],
                        strategy="PACK",
                        name=pg_name,
                        lifetime="detached",
                    )
                try:
                    ray.get(pg.ready())
                except SystemExit as e:
                    # Ray can raise SystemExit for some internal failures; do not let this
                    # terminate the API server process.
                    if getattr(e, "code", None) == 15:
                        raise
                    try:
                        ray.util.remove_placement_group(pg)
                    except Exception:
                        pass
                    raise RuntimeError(f"ray.get(pg.ready()) triggered SystemExit for {pg_name}: {e}") from e
                return pg

            def _remove_pg() -> None:
                pg_name = _pg_name()
                try:
                    pg = ray.util.get_placement_group(pg_name)
                except Exception:
                    return
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:
                    pass

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
                import asyncio
                await asyncio.to_thread(resource_pool.ensure_gpus_available, total_gpus)
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

            # Determine effective context limit for vLLM
            model_cfg = get_model_config(self.model_path)
            # Use model registry's max_model_len (single source of truth)
            max_model_len = model_cfg.max_model_len
            # prompt_logprobs uses float32 log_softmax over [tokens, vocab], which can spike memory.
            # max_num_batched_tokens limits chunk size during prefill/logprobs to cap peak allocations.
            max_num_batched_tokens = model_cfg.max_num_batched_tokens
            if max_num_batched_tokens is None:
                max_num_batched_tokens = 4096 if max_model_len >= 32768 else 8192
            # verl calculates max_model_len = prompt_length + response_length
            # We split evenly, but this does NOT restrict actual prompt/response sizes:
            # - The split only affects verl's default max_new_tokens (response_length)
            # - Our generate() computes actual limit dynamically: max_model_len - len(prompt)
            # - A 32K model can do 25K prompt + 7K response, or 5K prompt + 27K response
            # The sum (total context window) is what matters, not the individual values.
            prompt_length = max_model_len // 2
            response_length = max_model_len - prompt_length
            logger.info(f"vLLM max_model_len={max_model_len} (prompt={prompt_length}, response={response_length})")

            # Configure rollout with multi-LoRA support
            # NOTE: Keep expert_parallel_size=1 to avoid verl's worker-based EP assertion
            # Expert parallelism is enabled via engine_kwargs instead
            rollout_config = RolloutConfig(
                name="vllm",
                tensor_model_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                prompt_length=prompt_length,
                response_length=response_length,
                max_model_len=max_model_len,
                max_num_seqs=self.max_num_seqs,
                dtype="auto",
                load_format="auto",
                enforce_eager=False,
                enable_chunked_prefill=True,
                max_num_batched_tokens=max_num_batched_tokens,
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
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            pg = _get_or_create_pg()
            scheduling_opts = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                )
            }

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
            except SystemExit as e:
                if getattr(e, "code", None) == 15:
                    raise
                try:
                    ray_kill.kill(
                        self.server,
                        reason="vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
                _remove_pg()
                self.server = None
                raise RuntimeError(f"ray.get(launch_server) triggered SystemExit for {self.actor_name}: {e}") from e
            except ray.exceptions.GetTimeoutError:
                logger.error(f"vLLM launch timed out after {init_timeout}s for {self.actor_name}")
                # Kill the stuck actor
                try:
                    ray_kill.kill(
                        self.server,
                        reason="vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
                _remove_pg()
                self.server = None
                raise RuntimeError(f"vLLM actor {self.actor_name} launch timed out after {init_timeout}s")
            except Exception:
                # launch_server can fail fast on GPU placement collisions (vLLM checks free memory on startup).
                # In this case, the detached actor name is already taken; kill+remove the placement group so a
                # retry can reschedule to a different node/GPU set.
                try:
                    ray_kill.kill(
                        self.server,
                        reason="vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
                _remove_pg()
                self.server = None
                raise

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
        print(f"[DEBUG add_lora_for_session] About to call add_lora_with_id.remote, state_dict has {len(state_dict)} keys", flush=True)
        try:
            print(f"[DEBUG add_lora_for_session] Calling server.add_lora_with_id.remote", flush=True)
            await self.server.add_lora_with_id.remote(
                lora_int_id=lora_id,
                state_dict=state_dict,
                peft_config=peft_config,
            )
            print(f"[DEBUG add_lora_for_session] add_lora_with_id.remote returned", flush=True)
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
            logger.warning(f"vLLM actor dead/unresponsive, reinitializing: {e}")
            print(f"[DEBUG add_lora_for_session] RayActorError/GetTimeoutError: {e}", flush=True)
            self._initialized = False
            self.server = None
            await self.initialize()
            # Retry after restart
            await self.server.add_lora_with_id.remote(
                lora_int_id=lora_id,
                state_dict=state_dict,
                peft_config=peft_config,
            )
        except Exception as e:
            print(f"[DEBUG add_lora_for_session] UNEXPECTED EXCEPTION: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} "
            f"(lora_int_id={lora_id}, load_time={load_time:.3f}s)"
        )
        return lora_id

    async def add_lora_for_session_from_path(
        self,
        sampling_session_id: str,
        lora_path: str,
    ) -> int:
        """Add frozen LoRA weights for a sampling session from filesystem path.

        More efficient than add_lora_for_session for large MoE models:
        - Avoids transferring 30k+ tensors through Ray object store
        - vLLM worker loads directly from shared filesystem (PFS)

        Requires shared filesystem access between API server and vLLM worker.

        Args:
            sampling_session_id: Unique identifier for the sampling session.
            lora_path: Path to saved LoRA adapter directory (must be accessible to vLLM worker).

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
            logger.debug(
                f"GPU slots full ({self.max_loras}), "
                f"vLLM will manage CPU cache for overflow"
            )

        # Load directly from shared filesystem path
        # Much faster than Ray tensor transfer for large MoE models
        start_time = time.time()
        print(f"[DEBUG add_lora_for_session_from_path] Loading from path: {lora_path}", flush=True)
        try:
            await self.server.add_lora_from_path.remote(
                lora_int_id=lora_id,
                lora_path=lora_path,
                lora_name=sampling_session_id,
            )
            print(f"[DEBUG add_lora_for_session_from_path] add_lora_from_path.remote returned", flush=True)
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError) as e:
            logger.warning(f"vLLM actor dead/unresponsive, reinitializing: {e}")
            print(f"[DEBUG add_lora_for_session_from_path] RayActorError/GetTimeoutError: {e}", flush=True)
            self._initialized = False
            self.server = None
            await self.initialize()
            # Retry after restart
            await self.server.add_lora_from_path.remote(
                lora_int_id=lora_id,
                lora_path=lora_path,
                lora_name=sampling_session_id,
            )
        except Exception as e:
            print(f"[DEBUG add_lora_for_session_from_path] UNEXPECTED EXCEPTION: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} from path "
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

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

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
    ) -> list[float | None]:
        """Compute logprobs using session-specific LoRA or base model.

        If sampling_session_id is None or has no registered LoRA, uses base model.

        Returns a list of length len(prompt_ids), where:
        - logprobs[0] is None (first token has no conditioning context)
        - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1

        Args:
            sampling_session_id: The sampling session to use, or None for base model.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.

        Returns:
            List of logprobs, length = len(prompt_ids).
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)

        if lora_id is not None:
            # Compute logprobs with session-specific LoRA
            try:
                result = await self.server.compute_prompt_logprobs_with_lora.remote(
                    prompt_ids=prompt_ids,
                    request_id=request_id,
                    lora_int_id=lora_id,
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if any(s in msg for s in ("OutOfMemoryError", "CUDA out of memory", "EngineDeadError")):
                    logger.error(f"vLLM compute_logprobs failed; killing actor {self.actor_name}: {msg}")
                    try:
                        from tinker_server.backend.resource_pool import get_resource_pool

                        ray_kill.kill(
                            self.server,
                            reason="vllm_compute_logprobs_failed",
                            actor_name=self.actor_name,
                            namespace=PERSISTENT_NAMESPACE,
                            no_restart=True,
                        )
                        get_resource_pool().unregister(self.actor_name)
                    except Exception:
                        pass
                    self.server = None
                    self._initialized = False
                raise
        else:
            # Compute logprobs with base model (no LoRA)
            try:
                result = await self.server.compute_prompt_logprobs_base.remote(
                    prompt_ids=prompt_ids,
                    request_id=request_id,
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if any(s in msg for s in ("OutOfMemoryError", "CUDA out of memory", "EngineDeadError")):
                    logger.error(f"vLLM compute_logprobs failed; killing actor {self.actor_name}: {msg}")
                    try:
                        from tinker_server.backend.resource_pool import get_resource_pool

                        ray_kill.kill(
                            self.server,
                            reason="vllm_compute_logprobs_failed",
                            actor_name=self.actor_name,
                            namespace=PERSISTENT_NAMESPACE,
                            no_restart=True,
                        )
                        get_resource_pool().unregister(self.actor_name)
                    except Exception:
                        pass
                    self.server = None
                    self._initialized = False
                raise

        return list(result)

    async def compute_topk(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        k: int = 10,
    ) -> list[dict[int, float] | None]:
        """Compute top-K tokens using session-specific LoRA or base model.

        If sampling_session_id is None or has no registered LoRA, uses base model.

        Returns a list of length len(prompt_ids), where:
        - topk[0] is None (first token has no conditioning context)
        - topk[i] is a dict of token_id -> logprob for token i's distribution (i >= 1)

        Args:
            sampling_session_id: The sampling session to use, or None for base model.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.
            k: Number of top tokens to return.

        Returns:
            List of dicts mapping token_id to logprob, length = len(prompt_ids).
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

        # Look up LoRA ID for this session (None = base model)
        lora_id = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)

        if lora_id is not None:
            # Compute top-K with session-specific LoRA
            result = await self.server.compute_prompt_topk_with_lora.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                lora_int_id=lora_id,
                k=k,
            )
        else:
            # Compute top-K with base model (no LoRA)
            result = await self.server.compute_prompt_topk_base.remote(
                prompt_ids=prompt_ids,
                request_id=request_id,
                k=k,
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
                ray_kill.kill(
                    self.server,
                    reason="vllm_shutdown",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                )
                logger.info("Killed persistent vLLM actor")
            except Exception as e:
                logger.warning(f"Error killing server actor: {e}")
            try:
                pg = ray.util.get_placement_group(f"{self.actor_name}_pg")
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
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
        "Qwen/Qwen3-4B-Instruct-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/main",
        # MoE models (all share same architecture, different checkpoints)
        "Qwen/Qwen3-30B-A3B-Instruct-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "Qwen/Qwen3-30B-A3B": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/main",
        "Qwen/Qwen3-30B-A3B-Base": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/main",
        "Qwen/Qwen3-30B-A3B-Thinking-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/main",
        "Qwen/Qwen3-235B-A22B-Instruct-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e",
        "Qwen/Qwen3-235B-A22B-Thinking-2507": "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5",
        # K2 models (1T params MoE, requires FP8)
        "moonshotai/Kimi-K2-Instruct": "/vePFS-Mindverse/share/huggingface/hub/models--unsloth--Kimi-K2-Instruct-0905-BF16/snapshots/fbaf30b3baf5fdc2b2170ae04f4ff4948b0487cb",
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
        # Per-model init locks: avoid blocking other models while one
        # model's vLLM actor is (re)initializing.
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _get_model_lock(self, model_name: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(model_name)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[model_name] = lock
            return lock

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
        lock = await self._get_model_lock(model_name)
        async with lock:
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
            model_max_loras_override = config.max_loras
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
            model_max_lora_rank = config.max_lora_rank or self.max_lora_rank

            # Use per-model gpu_memory_utilization if specified (K2 needs lower for LoRA headroom)
            model_gpu_util = config.gpu_memory_utilization or self.gpu_memory_utilization

            # Use per-model max_model_len if specified (K2 needs 128K, default 262K exceeds KV cache)
            model_max_model_len = config.max_model_len or self.max_model_len

            # Use per-model max_num_seqs if specified (K2 needs reduced value for KV cache)
            model_max_num_seqs = config.max_num_seqs or 256  # Default: 256

            # Use per-model kv_cache_dtype if specified (FP8 KV cache halves memory)
            model_kv_cache_dtype = config.kv_cache_dtype

            # Multi-node required if total GPUs exceed a single 8xA800 node.
            # Examples:
            # - K2: TP=32 => 32 GPUs
            # - Qwen3-235B: TP=16 => 16 GPUs
            needs_multinode = config.total_gpus > 8

            if needs_multinode:
                # Use MultiNodeInferenceEngine with vLLM's native Ray distributed backend
                from .multinode_inference import MultiNodeInferenceEngine

                total_gpus = config.total_gpus
                enable_expert_parallel = config.is_moe and config.inference_dp > 1
                logger.info(
                    f"Creating multi-node vLLM engine for model {model_name}: "
                    f"actor={actor_name}, TP={config.inference_tp}, DP={config.inference_dp}, "
                    f"expert_parallel={enable_expert_parallel}, total_gpus={total_gpus}, "
                    f"gpu_util={model_gpu_util}, quant={quantization}, "
                    f"max_loras={model_max_loras}, max_lora_rank={model_max_lora_rank}, "
                    f"max_model_len={model_max_model_len}, max_num_seqs={model_max_num_seqs}, "
                    f"kv_cache_dtype={model_kv_cache_dtype}"
                )

                # MultiNodeInferenceEngine.initialize() handles:
                # - reconnecting to existing detached actor (no extra GPUs needed)
                # - creating a new actor (includes its own GPU availability checks)
                engine = MultiNodeInferenceEngine(
                    model_path=model_path,
                    model_name=model_name,
                    tensor_parallel_size=config.inference_tp,
                    data_parallel_size=config.inference_dp,
                    enable_expert_parallel=enable_expert_parallel,
                    gpu_memory_utilization=model_gpu_util,
                    max_model_len=model_max_model_len,
                    max_loras=model_max_loras,
                    max_lora_rank=model_max_lora_rank,
                    max_num_seqs=model_max_num_seqs,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    quantization=quantization,
                    kv_cache_dtype=model_kv_cache_dtype,
                    actor_name=actor_name,
                )
            else:
                # Use standard MultiLoRAInferenceEngine for single-node models
                logger.info(
                    f"Creating vLLM engine for model {model_name}: "
                    f"actor={actor_name}, TP={config.inference_tp}, DP={config.inference_dp}, "
                    f"quant={quantization}, max_loras={model_max_loras}, max_lora_rank={model_max_lora_rank}"
                )

                engine = MultiLoRAInferenceEngine(
                    model_path=model_path,
                    tensor_parallel_size=config.inference_tp,
                    data_parallel_size=config.inference_dp,
                    gpu_memory_utilization=model_gpu_util,
                    max_model_len=model_max_model_len,
                    max_num_seqs=model_max_num_seqs,
                    max_loras=model_max_loras,
                    max_cpu_loras=model_max_cpu_loras,
                    max_lora_rank=model_max_lora_rank,
                    actor_name=actor_name,
                    quantization=quantization,
                )

            await engine.initialize()

            self._engines[model_name] = engine
            persistent_csv = os.environ.get("MINT_PERSISTENT_MODELS", "").strip()
            if persistent_csv:
                persistent_models = {m.strip() for m in persistent_csv.split(",") if m.strip()}
                if model_name in persistent_models:
                    from tinker_server.backend.resource_pool import get_resource_pool

                    resource_pool = get_resource_pool()
                    if not resource_pool.set_protected(actor_name, True):
                        logger.warning(
                            f"Failed to protect vLLM actor for persistent model {model_name}: actor={actor_name}"
                        )
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


def kill_persistent_vllm_actor(model_name: str | None = None) -> bool:
    """Kill persistent vLLM actor(s).

    Args:
        model_name: If provided, kill actor for this specific model.
                   If None, kill ALL vLLM actors.

    Returns:
        True if any actor was killed, False if none found.
    """
    from tinker_server.backend.resource_pool import ActorType, get_resource_pool

    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    resource_pool = get_resource_pool()

    if model_name:
        # Kill specific model's actor
        actor_name = _model_to_actor_name(model_name)
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            ray_kill.kill(
                actor,
                reason="vllm_kill_by_name",
                actor_name=actor_name,
                namespace=PERSISTENT_NAMESPACE,
            )
            logger.info(f"Killed vLLM actor: {actor_name}")
            resource_pool.unregister(actor_name)
            try:
                pg = ray.util.get_placement_group(f"{actor_name}_pg")
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
            return True
        except ValueError:
            logger.info(f"No vLLM actor found: {actor_name}")
            try:
                pg = ray.util.get_placement_group(f"{actor_name}_pg")
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
            return False
    else:
        # Kill ALL vLLM actors via resource pool
        killed_any = False
        for entry in resource_pool.iter_entries():
            if entry.actor_type == ActorType.VLLM:
                try:
                    actor = ray.get_actor(entry.actor_name, namespace=entry.namespace)
                    ray_kill.kill(
                        actor,
                        reason="vllm_kill_by_name",
                        actor_name=entry.actor_name,
                        namespace=entry.namespace,
                    )
                    logger.info(f"Killed vLLM actor: {entry.actor_name}")
                    resource_pool.unregister(entry.actor_name)
                    try:
                        pg = ray.util.get_placement_group(f"{entry.actor_name}_pg")
                        ray.util.remove_placement_group(pg)
                    except Exception:
                        pass
                    killed_any = True
                except ValueError:
                    logger.warning(f"vLLM actor not found in Ray: {entry.actor_name}")
                    resource_pool.unregister(entry.actor_name)
                    try:
                        pg = ray.util.get_placement_group(f"{entry.actor_name}_pg")
                        ray.util.remove_placement_group(pg)
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"Error killing vLLM actor {entry.actor_name}: {e}")
        if not killed_any:
            logger.info("No vLLM actors found in resource pool")
        return killed_any


def check_persistent_vllm_actor(model_name: str | None = None) -> bool:
    """Check if vLLM actor(s) exist and are alive.

    Args:
        model_name: If provided, check for this specific model's actor.
                   If None, check for ANY running vLLM actor.

    Returns:
        True if matching actor exists and is alive.
    """
    from tinker_server.backend.resource_pool import ActorType, get_resource_pool

    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    if model_name:
        # Check specific model's actor
        actor_name = _model_to_actor_name(model_name)
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            ray.get(actor.__ray_ready__.remote(), timeout=5)
            return True
        except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
            return False
    else:
        # Check for ANY vLLM actor via resource pool
        resource_pool = get_resource_pool()
        for entry in resource_pool.iter_entries():
            if entry.actor_type == ActorType.VLLM:
                try:
                    actor = ray.get_actor(entry.actor_name, namespace=entry.namespace)
                    ray.get(actor.__ray_ready__.remote(), timeout=5)
                    return True
                except (ValueError, ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                    continue
        return False


def list_vllm_actors() -> list[dict]:
    """List all vLLM actors from resource pool.

    Returns:
        List of actor info dicts with name, gpus, base_model.
    """
    from tinker_server.backend.resource_pool import ActorType, get_resource_pool

    resource_pool = get_resource_pool()
    return [
        {"name": e.actor_name, "gpus": e.num_gpus, "base_model": e.base_model}
        for e in resource_pool.iter_entries()
        if e.actor_type == ActorType.VLLM
    ]


# Global instance (initialized in app lifespan)
multi_lora_engine: MultiLoRAInferenceEngine | None = None
