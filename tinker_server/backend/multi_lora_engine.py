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
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        max_loras: int = DEFAULT_MAX_LORAS,
        max_cpu_loras: int = DEFAULT_MAX_CPU_LORAS,
        max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
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
        """Initialize the shared vLLM engine with multi-LoRA support."""
        async with self._init_lock:
            if self._initialized:
                return

            from verl.workers.config import HFModelConfig, RolloutConfig
            from verl.workers.rollout.replica import RolloutMode

            from .verl_inference import _create_extended_server_class

            # Pass multi-LoRA config to vLLM engine
            ExtendedVLLMHttpServer = _create_extended_server_class(
                max_loras=self.max_loras,
                max_cpu_loras=self.max_cpu_loras,
            )

            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)

            # Configure rollout with multi-LoRA support
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
                data_parallel_size=1,
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
                f"max_loras={self.max_loras}, max_cpu_loras={self.max_cpu_loras}, "
                f"max_lora_rank={self.max_lora_rank}"
            )

            # Create Ray actor with GPU resources
            self.server = ExtendedVLLMHttpServer.options(
                num_gpus=self.tensor_parallel_size
            ).remote(
                config=rollout_config,
                model_config=model_config,
                rollout_mode=RolloutMode.STANDALONE,
                workers=[],
                replica_rank=0,
                node_rank=0,
                gpus_per_node=self.tensor_parallel_size,
                nnodes=1,
            )

            await self.server.launch_server.remote()
            self._initialized = True
            logger.info("MultiLoRAInferenceEngine initialized")

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
        start_time = time.time()
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
        sampling_session_id: str,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
    ) -> GenerateResult:
        """Generate tokens using session-specific LoRA weights.

        Args:
            sampling_session_id: The sampling session to use.
            prompt_ids: Input token IDs.
            request_id: Unique request identifier.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p (nucleus) sampling parameter.
            logprobs: Whether to return log probabilities.

        Returns:
            GenerateResult with generated tokens and metadata.

        Raises:
            ValueError: If sampling_session_id has no loaded LoRA.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        lora_id = await self.registry.get_lora_id(sampling_session_id)
        if lora_id is None:
            raise ValueError(
                f"No LoRA loaded for sampling session {sampling_session_id}"
            )

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

        return GenerateResult(
            token_ids=result["token_ids"],
            logprobs=result.get("logprobs"),
            stop_reason=result.get("stop_reason"),
        )

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

    async def shutdown(self) -> None:
        """Shutdown the engine and cleanup resources."""
        if self.server is not None:
            try:
                ray.kill(self.server)
            except Exception as e:
                logger.warning(f"Error killing server actor: {e}")
            self.server = None
        self._initialized = False
        logger.info("MultiLoRAInferenceEngine shutdown")


# Global instance (initialized in app lifespan)
multi_lora_engine: MultiLoRAInferenceEngine | None = None
