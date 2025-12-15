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
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_lora_rank = max_lora_rank

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
            # We must verify the actor is alive by calling a method on it
            try:
                self.server = ray.get_actor(PERSISTENT_VLLM_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
                # Health check: try calling a method to verify actor is alive
                # This will raise RayActorError if actor is dead
                try:
                    ray.get(self.server.__ray_ready__.remote(), timeout=5)
                    logger.info(
                        f"Connected to existing persistent vLLM actor: {PERSISTENT_VLLM_ACTOR_NAME}"
                    )
                    self._initialized = True
                    return
                except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
                    # Actor is dead or unresponsive, need to create new one
                    logger.warning(
                        f"vLLM actor {PERSISTENT_VLLM_ACTOR_NAME} is dead/unresponsive, creating new one"
                    )
                    self.server = None
                    # Reset initialized flag so we'll create new actor below
                    self._initialized = False
            except ValueError:
                # Actor doesn't exist, create new one
                logger.info(
                    f"No existing vLLM actor found, creating new detached actor: {PERSISTENT_VLLM_ACTOR_NAME}"
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
            )
            if self.max_model_len is not None:
                rollout_config.max_model_len = self.max_model_len

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
            self.server = ExtendedVLLMHttpServer.options(
                num_gpus=total_gpus,
                name=PERSISTENT_VLLM_ACTOR_NAME,
                namespace=PERSISTENT_NAMESPACE,
                lifetime="detached",
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

            # ray.get() blocks until launch_server completes (sets self.engine)
            # Note: await on ObjectRef doesn't work - must use ray.get()
            ray.get(self.server.launch_server.remote())
            self._initialized = True
            logger.info("MultiLoRAInferenceEngine initialized (detached actor)")

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


def kill_persistent_vllm_actor() -> bool:
    """Kill the persistent vLLM actor if it exists.

    Use this to force a full restart of the vLLM engine (e.g., after model changes).
    After calling this, the next server startup will create a new actor (~80s init).

    Returns:
        True if actor was killed, False if not found.
    """
    if not ray.is_initialized():
        ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

    try:
        actor = ray.get_actor(PERSISTENT_VLLM_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
        ray.kill(actor)
        logger.info(f"Killed persistent vLLM actor: {PERSISTENT_VLLM_ACTOR_NAME}")
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
