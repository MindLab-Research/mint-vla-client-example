"""Multi-node vLLM inference engine for large MoE models like K2.

Uses vLLM's native Ray distributed backend for TP across multiple nodes.
This bypasses verl's single-node vLLMHttpServerBase to enable TP > 8.

For K2 (1T params, 384 experts):
- TP=8 (single node): Base model uses 79GB/80GB, no room for LoRA
- TP=16 (2 nodes): ~40GB/GPU, leaves room for LoRA buffers
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ray

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH

# Namespace for actors
PERSISTENT_NAMESPACE = "tinker"


@dataclass
class MultiNodeLoRASlot:
    """Metadata for a loaded LoRA adapter in multi-node engine."""

    lora_int_id: int
    sampling_session_id: str
    adapter_path: str  # Shared filesystem path
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class MultiNodeLoRARegistry:
    """Maps sampling_session_id to lora_int_id for multi-node engine."""

    def __init__(self):
        self._session_to_id: dict[str, int] = {}
        self._id_to_slot: dict[int, MultiNodeLoRASlot] = {}
        self._next_id: int = 1
        self._lock = asyncio.Lock()

    async def allocate(self, sampling_session_id: str, adapter_path: str) -> int:
        """Allocate a unique lora_int_id for a sampling session."""
        async with self._lock:
            if sampling_session_id in self._session_to_id:
                raise ValueError(
                    f"Session {sampling_session_id} already has lora_int_id "
                    f"{self._session_to_id[sampling_session_id]}"
                )

            lora_id = self._next_id
            self._next_id += 1

            self._session_to_id[sampling_session_id] = lora_id
            self._id_to_slot[lora_id] = MultiNodeLoRASlot(
                lora_int_id=lora_id,
                sampling_session_id=sampling_session_id,
                adapter_path=adapter_path,
            )

            logger.debug(
                f"Allocated lora_int_id={lora_id} for session {sampling_session_id}"
            )
            return lora_id

    async def get_lora_id(self, sampling_session_id: str) -> int | None:
        """Get lora_int_id for a sampling session."""
        async with self._lock:
            lora_id = self._session_to_id.get(sampling_session_id)
            if lora_id is not None and lora_id in self._id_to_slot:
                self._id_to_slot[lora_id].last_used = time.time()
            return lora_id

    async def get_adapter_path(self, lora_id: int) -> str | None:
        """Get adapter path for a lora_int_id."""
        async with self._lock:
            slot = self._id_to_slot.get(lora_id)
            return slot.adapter_path if slot else None

    async def remove(self, lora_id: int) -> str | None:
        """Remove a lora_int_id from the registry."""
        async with self._lock:
            slot = self._id_to_slot.pop(lora_id, None)
            if slot:
                self._session_to_id.pop(slot.sampling_session_id, None)
                logger.debug(f"Removed lora_int_id={lora_id}")
                return slot.sampling_session_id
            return None

    async def count(self) -> int:
        """Get the number of registered sessions."""
        async with self._lock:
            return len(self._session_to_id)


def _create_multinode_vllm_actor(
    max_loras: int = 1,
    max_lora_rank: int = 8,
):
    """Create a Ray actor class that wraps vLLM's AsyncLLMEngine for multi-node TP.

    Unlike verl's vLLMHttpServerBase which is single-node, this uses vLLM's native
    Ray distributed backend for TP across multiple nodes.

    Args:
        max_loras: Maximum LoRAs in a single batch.
        max_lora_rank: Maximum LoRA rank.
    """

    @ray.remote(num_cpus=1)  # num_gpus=0: vLLM's internal Ray backend manages GPU allocation
    class MultiNodeVLLMEngine:
        """vLLM engine with native Ray backend for multi-node TP."""

        def __init__(
            self,
            model_path: str,
            tensor_parallel_size: int,
            gpu_memory_utilization: float = 0.80,
            max_model_len: int | None = None,
            quantization: str | None = None,
            enable_lora: bool = True,
        ):
            self.model_path = model_path
            self.tensor_parallel_size = tensor_parallel_size
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_model_len = max_model_len
            self.quantization = quantization
            self.enable_lora = enable_lora
            self.max_loras = max_loras
            self.max_lora_rank = max_lora_rank

            self.engine = None
            self._initialized = False

        async def initialize(self) -> None:
            """Initialize vLLM engine with Ray distributed backend."""
            if self._initialized:
                return

            # Force vLLM v0 engine BEFORE importing vLLM
            # v1's multiprocess architecture requires coordinator to have GPU
            import os
            os.environ["VLLM_USE_V1"] = "0"

            # Import vLLM components AFTER setting env var
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            # Build engine args for multi-node TP
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                distributed_executor_backend="ray",  # Key: use Ray for multi-node
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="auto",
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                max_num_seqs=256,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
                enable_prefix_caching=True,
                disable_log_stats=True,
                enforce_eager=True,  # CUDA graphs OOM on K2 at 0.98 util
                quantization=self.quantization,
                # LoRA config
                enable_lora=self.enable_lora,
                max_loras=self.max_loras if self.enable_lora else None,
                max_lora_rank=self.max_lora_rank if self.enable_lora else None,
            )

            logger.info(
                f"Creating AsyncLLMEngine: TP={self.tensor_parallel_size}, "
                f"backend=ray, enable_lora={self.enable_lora}"
            )

            # Create engine - vLLM will spawn Ray workers across nodes
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

            self._initialized = True
            logger.info("MultiNodeVLLMEngine initialized")

        async def is_ready(self) -> bool:
            """Check if engine is initialized and ready."""
            return self._initialized and self.engine is not None

        async def add_lora(self, lora_int_id: int, lora_path: str, lora_name: str) -> None:
            """Add LoRA adapter from shared filesystem path.

            For multi-node: all workers must have access to the same path.
            Use shared filesystem (e.g., /vePFS-Mindverse/share/).

            Args:
                lora_int_id: Unique identifier for this LoRA adapter.
                lora_path: Path to PEFT adapter directory (must be on shared FS).
                lora_name: Human-readable name for the adapter.
            """
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            await self.engine.add_lora(lora_request)
            logger.info(f"Added LoRA {lora_name} (id={lora_int_id}) from {lora_path}")

        async def remove_lora(self, lora_int_id: int) -> None:
            """Remove a LoRA adapter."""
            await self.engine.remove_lora(lora_int_id)
            logger.info(f"Removed LoRA id={lora_int_id}")

        async def list_loras(self) -> set[int]:
            """List loaded LoRA adapter IDs."""
            return await self.engine.list_loras()

        async def generate(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int | None,
            lora_path: str | None,
            max_tokens: int,
            temperature: float = 1.0,
            top_k: int = -1,
            top_p: float = 1.0,
            logprobs: bool = True,
        ) -> dict:
            """Generate tokens with optional LoRA adapter.

            Args:
                prompt_ids: Input token IDs.
                request_id: Unique request identifier.
                lora_int_id: LoRA adapter ID to use, or None for base model.
                lora_path: Path to LoRA adapter (for LoRARequest).
                max_tokens: Maximum tokens to generate.
                temperature: Sampling temperature.
                top_k: Top-k sampling parameter.
                top_p: Top-p sampling parameter.
                logprobs: Whether to return log probabilities.

            Returns:
                Dict with token_ids, logprobs, stop_reason.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=0 if logprobs else None,
                stop_token_ids=[151645, 151643],  # Qwen EOS tokens
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Build LoRA request if specified
            lora_request = None
            if lora_int_id is not None and lora_path is not None:
                lora_request = LoRARequest(
                    lora_name=str(lora_int_id),
                    lora_int_id=lora_int_id,
                    lora_path=lora_path,
                )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            # Get final response
            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            token_ids = list(final_res.outputs[0].token_ids)
            log_probs = None
            if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:
                log_probs = [
                    logprobs[token_ids[i]].logprob
                    for i, logprobs in enumerate(final_res.outputs[0].logprobs)
                ]

            # Determine stop reason
            stop_reason = "length"
            if final_res.outputs[0].finish_reason == "stop":
                stop_reason = "stop"
            elif any(tid in [151645, 151643] for tid in token_ids[-3:]):
                stop_reason = "stop"

            return {
                "token_ids": token_ids,
                "logprobs": log_probs,
                "stop_reason": stop_reason,
            }

        async def compute_prompt_logprobs(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int | None,
            lora_path: str | None,
        ) -> list[float]:
            """Compute logprobs for prompt tokens.

            Returns logprobs[i] = log P(token[i+1] | token[0:i+1]).
            Output length is len(prompt_ids) - 1.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            if len(prompt_ids) < 2:
                return []

            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=1,
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

            # Build LoRA request if specified
            lora_request = None
            if lora_int_id is not None and lora_path is not None:
                lora_request = LoRARequest(
                    lora_name=str(lora_int_id),
                    lora_int_id=lora_int_id,
                    lora_path=lora_path,
                )

            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )

            # Get final response
            final_res = None
            async for output in generator:
                final_res = output
            assert final_res is not None

            # Extract prompt logprobs
            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return []

            logprobs = []
            for i in range(1, len(prompt_logprobs)):
                if prompt_logprobs[i] is None:
                    continue
                token_id = prompt_ids[i]
                if token_id in prompt_logprobs[i]:
                    logprobs.append(prompt_logprobs[i][token_id].logprob)
                else:
                    logprobs.append(-100.0)

            return logprobs

    return MultiNodeVLLMEngine


@dataclass
class GenerateResult:
    """Result of a generate call."""

    token_ids: list[int]
    logprobs: list[float] | None = None
    stop_reason: str | None = None


class MultiNodeInferenceEngine:
    """Multi-node inference engine for large MoE models.

    Uses vLLM's native Ray distributed backend for TP across multiple nodes.
    Designed for K2 (1T params) which needs TP=16 for LoRA support.

    Key differences from MultiLoRAInferenceEngine:
    - Uses vLLM's Ray backend instead of verl's single-node ZMQ pattern
    - Outer actor has num_gpus=0 (vLLM manages GPU workers internally)
    - LoRA adapters must be on shared filesystem (all nodes access same path)
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 16,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int | None = None,
        max_loras: int = 1,
        max_lora_rank: int = 8,
        quantization: str | None = None,
        actor_name: str | None = None,
        shared_adapter_dir: str = "/vePFS-Mindverse/share/tinker_adapters",
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.quantization = quantization
        self.actor_name = actor_name or f"multinode_vllm_{model_path.split('/')[-1].lower()}"
        self.shared_adapter_dir = shared_adapter_dir

        self.registry = MultiNodeLoRARegistry()
        self.engine = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the multi-node vLLM engine."""
        print(f"[DEBUG] MultiNodeInferenceEngine.initialize() called", flush=True)
        async with self._init_lock:
            if self._initialized:
                print(f"[DEBUG] Already initialized, returning", flush=True)
                return

            if not ray.is_initialized():
                ray.init(address="auto", namespace=PERSISTENT_NAMESPACE, ignore_reinit_error=True)

            # Try to connect to existing actor
            existing_actor = None
            try:
                existing_actor = ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                is_ready = ray.get(existing_actor.is_ready.remote(), timeout=30)
                if is_ready:
                    logger.info(f"Connected to existing MultiNodeVLLMEngine: {self.actor_name}")
                    self.engine = existing_actor
                    self._initialized = True
                    return
                else:
                    logger.warning(f"Actor {self.actor_name} exists but not ready, will recreate")
            except (ValueError, ray.exceptions.RayActorError):
                logger.info(f"No existing actor found, creating new: {self.actor_name}")
            except ray.exceptions.GetTimeoutError:
                logger.warning(f"Actor {self.actor_name} exists but is_ready timed out, will recreate")

            # Kill existing actor if any before creating new
            if existing_actor is not None:
                try:
                    ray.kill(existing_actor, no_restart=True)
                    # Wait for Ray to clean up the actor name
                    import time
                    for _ in range(10):
                        time.sleep(1)
                        try:
                            ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                        except ValueError:
                            break  # Actor name is available
                except Exception as e:
                    logger.warning(f"Error killing actor {self.actor_name}: {e}")

            # Ensure shared adapter directory exists
            os.makedirs(self.shared_adapter_dir, exist_ok=True)

            # Step 1: Ensure enough GPUs are available (evict idle actors if needed)
            from tinker_server.backend.resource_pool import get_resource_pool
            resource_pool = get_resource_pool()
            logger.info(f"Ensuring {self.tensor_parallel_size} GPUs available for vLLM")
            resource_pool.ensure_gpus_available(self.tensor_parallel_size, timeout=300)

            # Step 2: Find nodes with AVAILABLE GPUs
            # Available = Total - PG_used - Actor_used (same logic as multi_lora_engine.py)
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            target_node = None

            # Count GPUs used by placement groups per node
            pg_table = ray.util.placement_group_table()
            gpus_used_by_pg: dict[str, int] = {}
            for pg_id, pg_info in pg_table.items():
                if pg_info.get("state") != "CREATED":
                    continue
                bundles_to_node = pg_info.get("bundles_to_node_id", {})
                for bundle_idx, node_id in bundles_to_node.items():
                    if node_id:
                        # Each Megatron bundle uses 1 GPU
                        gpus_used_by_pg[node_id] = gpus_used_by_pg.get(node_id, 0) + 1

            # Count GPUs used by actors tracked in ResourcePool
            gpus_used_by_actors = resource_pool.gpus_used_by_node()

            # Find nodes with enough available GPUs
            # For multi-node vLLM (TP=16), we need nodes with 8 free GPUs each
            gpus_per_node = 8  # Standard GPU node size
            candidates: list[tuple[str, str, int]] = []  # (node_id, ip, available_gpus)

            for node in ray.nodes():
                if not node["Alive"]:
                    continue
                node_id = node["NodeID"]
                node_ip = node["NodeManagerAddress"]
                total_gpus = node.get("Resources", {}).get("GPU", 0)
                if total_gpus == 0:
                    continue

                pg_gpus = gpus_used_by_pg.get(node_id, 0)
                actor_gpus = gpus_used_by_actors.get(node_id, 0)
                available_gpus = int(total_gpus - pg_gpus - actor_gpus)

                logger.debug(
                    f"Node {node_id[:8]} ({node_ip}): total={total_gpus}, "
                    f"pg_used={pg_gpus}, actor_used={actor_gpus}, available={available_gpus}"
                )

                if available_gpus >= gpus_per_node:
                    candidates.append((node_id, node_ip, available_gpus))

            if not candidates:
                raise RuntimeError(
                    f"No nodes with {gpus_per_node}+ free GPUs after ensure_gpus_available({self.tensor_parallel_size}). "
                    f"Check ResourcePool and placement group state."
                )

            # Prefer nodes with most available GPUs (cleanest placement)
            candidates.sort(key=lambda x: -x[2])
            target_node, target_ip, available = candidates[0]
            logger.info(
                f"Selected node {target_ip} ({target_node[:8]}) with {available} available GPUs"
            )

            # Create new engine actor
            MultiNodeVLLMEngine = _create_multinode_vllm_actor(
                max_loras=self.max_loras,
                max_lora_rank=self.max_lora_rank,
            )

            # target_node is guaranteed set (RuntimeError raised above if no candidates)
            # soft=False: fail if target unavailable, no fallback
            scheduling_opts = {
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    node_id=target_node, soft=False
                )
            }

            # Actor has num_gpus=0 because vLLM's Ray backend manages GPU workers
            self.engine = MultiNodeVLLMEngine.options(
                name=self.actor_name,
                namespace=PERSISTENT_NAMESPACE,
                lifetime="detached",
                **scheduling_opts,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": PFS_PYTHONPATH,
                        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                        "HF_HUB_OFFLINE": "1",
                        # Force vLLM v0 engine - v1's multiprocess architecture
                        # conflicts with Ray distributed executor backend
                        "VLLM_USE_V1": "0",
                    }
                },
            ).remote(
                model_path=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                quantization=self.quantization,
                enable_lora=self.max_loras > 0,
            )

            # Initialize engine (this spawns vLLM's Ray workers)
            logger.info(f"Initializing MultiNodeVLLMEngine with TP={self.tensor_parallel_size}")
            # K2 (TP=16): ~30 min shard loading + post-init (CUDA graphs, KV cache)
            # Need 3600s (1h) to be safe; 1800s times out at edge cases
            init_timeout = 3600 if self.tensor_parallel_size >= 16 else (1800 if self.tensor_parallel_size >= 8 else 600)

            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: ray.get(self.engine.initialize.remote(), timeout=init_timeout)
                )
            except ray.exceptions.GetTimeoutError:
                logger.error(f"Engine initialization timed out after {init_timeout}s")
                ray.kill(self.engine, no_restart=True)
                self.engine = None
                raise RuntimeError(f"MultiNodeVLLMEngine init timed out")

            self._initialized = True
            logger.info(f"MultiNodeInferenceEngine initialized: {self.actor_name}")

            # Register with unified resource pool for LRU tracking
            # Multi-node vLLM internally manages GPU workers, but we track total GPUs for eviction
            from tinker_server.backend.resource_pool import get_resource_pool, ActorType
            resource_pool = get_resource_pool()
            resource_pool.register(
                actor_name=self.actor_name,
                actor_type=ActorType.VLLM,
                num_gpus=self.tensor_parallel_size,  # TP=16 for K2
                actor_handle=self.engine,
                namespace=PERSISTENT_NAMESPACE,
                base_model=self.model_path,
            )
            # Mark as ready since initialization completed
            resource_pool.mark_ready(self.actor_name)
            logger.info(f"Registered {self.actor_name} with ResourcePool ({self.tensor_parallel_size} GPUs)")

    async def add_lora_for_session(
        self,
        sampling_session_id: str,
        state_dict: dict,
        peft_config: dict,
    ) -> int:
        """Add LoRA weights for a sampling session.

        For multi-node: saves adapter to shared filesystem, then loads via path.
        All vLLM workers access the same shared path.

        Args:
            sampling_session_id: Unique identifier for the sampling session.
            state_dict: LoRA weight tensors.
            peft_config: PEFT adapter configuration dict.

        Returns:
            The allocated lora_int_id for this session.
        """
        if not self._initialized:
            await self.initialize()

        from safetensors.torch import save_file

        # Save adapter to shared filesystem
        adapter_dir = os.path.join(self.shared_adapter_dir, sampling_session_id)
        os.makedirs(adapter_dir, exist_ok=True)

        weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        config_path = os.path.join(adapter_dir, "adapter_config.json")

        save_file(state_dict, weights_path)
        with open(config_path, "w") as f:
            json.dump(peft_config, f, indent=2)

        # Allocate lora_int_id
        lora_id = await self.registry.allocate(sampling_session_id, adapter_dir)

        # Add to engine (all workers load from shared path)
        start_time = time.time()
        await self.engine.add_lora.remote(
            lora_int_id=lora_id,
            lora_path=adapter_dir,
            lora_name=sampling_session_id,
        )
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} "
            f"(lora_int_id={lora_id}, path={adapter_dir}, load_time={load_time:.3f}s)"
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
        """Generate tokens using session-specific LoRA or base model."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        # Look up LoRA for this session
        lora_id = None
        lora_path = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
            if lora_id is not None:
                lora_path = await self.registry.get_adapter_path(lora_id)

        result = await self.engine.generate.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
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
        """Compute logprobs using session-specific LoRA or base model."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        # Look up LoRA for this session
        lora_id = None
        lora_path = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
            if lora_id is not None:
                lora_path = await self.registry.get_adapter_path(lora_id)

        result = await self.engine.compute_prompt_logprobs.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )

        return list(result)

    async def remove_session(self, sampling_session_id: str) -> bool:
        """Remove a sampling session and its LoRA."""
        lora_id = await self.registry.get_lora_id(sampling_session_id)
        if lora_id is None:
            return False

        try:
            await self.engine.remove_lora.remote(lora_id)
        except Exception as e:
            logger.warning(f"Failed to remove LoRA {lora_id} from engine: {e}")

        await self.registry.remove(lora_id)
        logger.info(f"Removed session {sampling_session_id} (lora_int_id={lora_id})")
        return True

    async def shutdown(self, kill_actor: bool = False) -> None:
        """Disconnect from the engine."""
        if self.engine is not None and kill_actor:
            try:
                ray.kill(self.engine)
                logger.info("Killed MultiNodeVLLMEngine actor")
            except Exception as e:
                logger.warning(f"Error killing actor: {e}")
        self.engine = None
        self._initialized = False
        logger.info("MultiNodeInferenceEngine disconnected")
