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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ray

from . import ray_kill

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH, RAY_NAMESPACE
from tinker_server.ray_utils import ray_log_to_driver_kwargs

# Namespace for actors
PERSISTENT_NAMESPACE = RAY_NAMESPACE


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


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


class _AsyncRWLock:
    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @asynccontextmanager
    async def read_locked(self):
        async with self._cond:
            while self._writer or self._writers_waiting > 0:
                await self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @asynccontextmanager
    async def write_locked(self):
        async with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    await self._cond.wait()
                self._writer = True
            finally:
                self._writers_waiting -= 1
        try:
            yield
        finally:
            async with self._cond:
                self._writer = False
                self._cond.notify_all()


def _create_multinode_vllm_actor(
    max_loras: int = 1,
    max_cpu_loras: int | None = None,
    max_lora_rank: int = 8,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int | None = None,
):
    """Create a Ray actor class that wraps vLLM's AsyncLLMEngine for multi-node TP.

    Unlike verl's vLLMHttpServerBase which is single-node, this uses vLLM's native
    Ray distributed backend for TP across multiple nodes.

    Args:
        max_loras: Maximum LoRAs in a single batch.
        max_cpu_loras: Maximum LoRAs to store in CPU memory (None = vLLM default).
        max_lora_rank: Maximum LoRA rank.
        max_num_seqs: Maximum concurrent sequences (reduce for large models with KV cache constraints).
    """

    @ray.remote(num_cpus=1)  # num_gpus=0: vLLM's internal Ray backend manages GPU allocation
    class MultiNodeVLLMEngine:
        """vLLM engine with native Ray backend for multi-node TP."""

        def __init__(
            self,
            model_path: str,
            tensor_parallel_size: int,
            pipeline_parallel_size: int = 1,
            data_parallel_size: int = 1,
            enable_expert_parallel: bool = False,
            gpu_memory_utilization: float = 0.80,
            max_model_len: int | None = None,
            quantization: str | None = None,
            enable_lora: bool = True,
            kv_cache_dtype: str | None = None,
            max_num_batched_tokens: int | None = None,
        ):
            self.model_path = model_path
            self.tensor_parallel_size = tensor_parallel_size
            self.pipeline_parallel_size = pipeline_parallel_size
            self.data_parallel_size = data_parallel_size
            self.enable_expert_parallel = enable_expert_parallel
            self.gpu_memory_utilization = gpu_memory_utilization
            self.max_model_len = max_model_len
            self.quantization = quantization
            self.enable_lora = enable_lora
            self.max_loras = max_loras
            self.max_cpu_loras = max_cpu_loras
            self.max_lora_rank = max_lora_rank
            self.max_num_seqs = max_num_seqs
            self.kv_cache_dtype = kv_cache_dtype
            self.max_num_batched_tokens = max_num_batched_tokens

            self.engine = None
            self._initialized = False
            self._rw_lock = _AsyncRWLock()
            self._lock_mode = os.environ.get("MINT_VLLM_ENGINE_LOCK_MODE", "rw").strip().lower()
            self._timing = _env_flag("MINT_VLLM_REQUEST_TIMING", default=False)
            self._serialize_prompt_logprobs = _env_flag("MINT_VLLM_PROMPT_LOGPROBS_SERIALIZE", default=False)
            self._prompt_logprobs_lock = asyncio.Lock() if self._serialize_prompt_logprobs else None
            # vLLM supports concurrent requests for continuous batching, but some engine calls
            # (notably list_loras) must not race active generation on multinode.
            #
            # Default to serialized generate for safety; set MINT_VLLM_SERIALIZE_GENERATE=0 to
            # allow concurrent generate() calls (still protected by the RW lock).
            self._serialize_generate = _env_flag("MINT_VLLM_SERIALIZE_GENERATE", default=True)
            self._generate_lock = asyncio.Lock() if self._serialize_generate else None
            # vLLM SamplingParams(n>1) has shown hangs under concurrent multinode traffic.
            # Default to serializing multi-sample requests (num_samples>1) only.
            self._serialize_multisample = _env_flag("MINT_VLLM_SERIALIZE_MULTISAMPLE", default=True)
            self._multisample_lock = asyncio.Lock() if self._serialize_multisample else None
            # AsyncLLMEngine.add_request has shown hangs when called concurrently on multinode.
            # Serialize add_request() while still allowing concurrent in-flight requests.
            self._serialize_add_request = _env_flag("MINT_VLLM_SERIALIZE_ADD_REQUEST", default=True)
            self._add_request_lock = asyncio.Lock() if self._serialize_add_request else None
            # For multinode, vLLM's `SamplingParams(n>1)` path has shown hangs even at low
            # concurrency. Optionally implement multi-sample by issuing N independent n=1
            # requests; rely on vLLM prefix caching to reuse the long prompt KV across calls.
            #
            # Modes:
            # - "vllm_n": use `SamplingParams(n=N)` (default vLLM multisample)
            # - "sequential_n1": run N sequential `SamplingParams(n=1)` requests
            self._multisample_mode = os.environ.get("MINT_VLLM_MULTISAMPLE_MODE", "sequential_n1").strip().lower()
            self._outer_to_subreq_ids: dict[str, set[str]] = {}
            self._outer_to_subreq_lock = asyncio.Lock()
            self._generate_timeout_s = float(os.environ.get("MINT_VLLM_GENERATE_TIMEOUT_S", "0"))
            self._gate_lock = asyncio.Lock()
            self._active_generates = 0
            self._active_generates_cond = asyncio.Condition()
            self._is_ready_timeout_s = float(os.environ.get("MINT_VLLM_IS_READY_TIMEOUT_S", "0.05"))
            # vLLM's `max_num_seqs` is a hard cap on active sequences. Under multinode + long-context
            # + multi-sample (SamplingParams(n>1)), vLLM can hang when the server oversubscribes this
            # cap and relies on vLLM internal queueing. Use explicit admission control to avoid
            # exceeding `max_num_seqs` from the server side (no client API change).
            self._admission_control = _env_flag("MINT_VLLM_ADMISSION_CONTROL", default=True)
            self._active_seq_slots = 0
            self._seq_slots_cond = asyncio.Condition()

        @asynccontextmanager
        async def _reserve_seq_slots(self, n_req: int):
            if (not self._admission_control) or (self.max_num_seqs is None):
                yield
                return
            need = max(1, int(n_req))
            async with self._seq_slots_cond:
                while self._active_seq_slots + need > int(self.max_num_seqs):
                    await self._seq_slots_cond.wait()
                self._active_seq_slots += need
            try:
                yield
            finally:
                async with self._seq_slots_cond:
                    self._active_seq_slots -= need
                    self._seq_slots_cond.notify_all()

        @asynccontextmanager
        async def _lock_read(self):
            if self._lock_mode == "all":
                async with self._rw_lock.write_locked():
                    yield
            else:
                async with self._rw_lock.read_locked():
                    yield

        @asynccontextmanager
        async def _lock_write(self):
            async with self._rw_lock.write_locked():
                yield

        @asynccontextmanager
        async def _maybe_prompt_logprobs_lock(self):
            if self._prompt_logprobs_lock is None:
                yield
                return
            async with self._prompt_logprobs_lock:
                yield

        @asynccontextmanager
        async def _maybe_generate_lock(self):
            if self._generate_lock is None:
                yield
                return
            async with self._generate_lock:
                yield

        @asynccontextmanager
        async def _maybe_multisample_lock(self, n_req: int):
            if self._multisample_lock is None or n_req <= 1:
                yield
                return
            async with self._multisample_lock:
                yield

        @asynccontextmanager
        async def _maybe_add_request_lock(self):
            if self._add_request_lock is None:
                yield
                return
            async with self._add_request_lock:
                yield

        async def _register_generate_start(self) -> None:
            async with self._gate_lock:
                async with self._active_generates_cond:
                    self._active_generates += 1

        async def _register_generate_end(self) -> None:
            async with self._active_generates_cond:
                self._active_generates -= 1
                if self._active_generates == 0:
                    self._active_generates_cond.notify_all()

        @asynccontextmanager
        async def _exclusive_engine_op(self):
            async with self._gate_lock:
                async with self._active_generates_cond:
                    while self._active_generates > 0:
                        await self._active_generates_cond.wait()
                yield

        async def initialize(self) -> None:
            """Initialize vLLM engine with Ray distributed backend."""
            if self._initialized:
                return

            # Force vLLM v0 engine BEFORE importing vLLM
            # v1's multiprocess architecture requires coordinator to have GPU
            import os
            os.environ["VLLM_USE_V1"] = "0"
            # PyNcclCommunicator has hit NCCL internal errors in multi-node init;
            # disable to fall back to torch.distributed collectives.
            os.environ["VLLM_DISABLE_PYNCCL"] = "1"

            # Import vLLM components AFTER setting env var
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            # Build engine args for multi-node TP
            # prompt_logprobs uses float32 log_softmax over [tokens, vocab], which can spike memory.
            max_num_batched_tokens = self.max_num_batched_tokens
            if max_num_batched_tokens is None:
                max_num_batched_tokens = 4096 if (self.max_model_len or 0) >= 32768 else 8192
            max_num_batched_tokens = int(os.environ.get("MINT_VLLM_MAX_NUM_BATCHED_TOKENS", str(max_num_batched_tokens)))
            enable_chunked_prefill = _env_flag("MINT_VLLM_ENABLE_CHUNKED_PREFILL", default=True)
            enable_prefix_caching = _env_flag("MINT_VLLM_ENABLE_PREFIX_CACHING", default=True)
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                data_parallel_size=self.data_parallel_size,
                data_parallel_backend="ray" if self.data_parallel_size > 1 else "mp",
                enable_expert_parallel=self.enable_expert_parallel,
                distributed_executor_backend="ray",  # Key: use Ray for multi-node
                disable_custom_all_reduce=True,  # Avoid PyNcclCommunicator issues in multi-node
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="auto",
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                enable_chunked_prefill=enable_chunked_prefill,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_caching=enable_prefix_caching,
                disable_log_stats=not _env_flag("MINT_VLLM_LOG_STATS", default=False),
                enforce_eager=True,  # CUDA graphs OOM on K2 at 0.98 util
                quantization=self.quantization,
                kv_cache_dtype=self.kv_cache_dtype or "auto",  # None -> "auto" for vLLM CacheConfig validation
                # LoRA config
                enable_lora=self.enable_lora,
                max_loras=self.max_loras if self.enable_lora else None,
                max_lora_rank=self.max_lora_rank if self.enable_lora else None,
                max_cpu_loras=self.max_cpu_loras if self.enable_lora else None,
            )

            logger.info(
                f"Creating AsyncLLMEngine: "
                f"TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, "
                f"DP={self.data_parallel_size}, expert_parallel={self.enable_expert_parallel}, "
                f"backend=ray, enable_lora={self.enable_lora}, gpu_util={self.gpu_memory_utilization}, "
                f"chunked_prefill={enable_chunked_prefill}, max_num_batched_tokens={max_num_batched_tokens}, "
                f"prefix_caching={enable_prefix_caching}"
            )

            # Create engine - vLLM will spawn Ray workers across nodes
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

            self._initialized = True
            logger.info("MultiNodeVLLMEngine initialized")

        async def is_ready(self) -> bool:
            """Check if engine is initialized and the EngineCore is responsive."""
            if not self._initialized or self.engine is None:
                return False
            try:
                # Touch EngineCore. The Ray actor can be alive while EngineCore is dead.
                #
                # IMPORTANT: `list_loras()` must not run concurrently with `generate()`.
                # Also: do not block live traffic for liveness checks; when busy, assume alive.
                if self._gate_lock.locked():
                    return True
                async with self._gate_lock:
                    async with self._active_generates_cond:
                        if self._active_generates > 0:
                            return True
                    async with self._lock_read():
                        try:
                            await asyncio.wait_for(self.engine.list_loras(), timeout=self._is_ready_timeout_s)
                        except asyncio.TimeoutError:
                            return True
            except Exception as e:
                logger.warning(f"MultiNodeVLLMEngine is_ready failed: {type(e).__name__}: {e}")
                return False
            return True

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

            t0 = time.perf_counter()
            async with self._exclusive_engine_op():
                async with self._lock_write():
                    t1 = time.perf_counter()
                    await self.engine.add_lora(lora_request)
            t2 = time.perf_counter()
            if self._timing:
                print(
                    f"[vLLM timing] add_lora id={lora_int_id} lock_wait_s={t1 - t0:.3f} engine_s={t2 - t1:.3f} total_s={t2 - t0:.3f}"
                    ,
                    flush=True,
                )
            logger.info(f"Added LoRA {lora_name} (id={lora_int_id}) from {lora_path}")

        async def remove_lora(self, lora_int_id: int) -> None:
            """Remove a LoRA adapter."""
            t0 = time.perf_counter()
            async with self._exclusive_engine_op():
                async with self._lock_write():
                    t1 = time.perf_counter()
                    await self.engine.remove_lora(lora_int_id)
            t2 = time.perf_counter()
            if self._timing:
                print(
                    f"[vLLM timing] remove_lora id={lora_int_id} lock_wait_s={t1 - t0:.3f} engine_s={t2 - t1:.3f} total_s={t2 - t0:.3f}"
                    ,
                    flush=True,
                )
            logger.info(f"Removed LoRA id={lora_int_id}")

        async def list_loras(self) -> set[int]:
            """List loaded LoRA adapter IDs."""
            # NOTE: list_loras must not race active generation on multinode.
            async with self._exclusive_engine_op():
                async with self._lock_read():
                    return await self.engine.list_loras()

        async def abort_request(self, request_id: str) -> None:
            """Abort an in-flight request in vLLM."""
            try:
                async with self._outer_to_subreq_lock:
                    sub_ids = list(self._outer_to_subreq_ids.get(request_id, ()))
            except Exception:
                sub_ids = []
            try:
                for sid in sub_ids:
                    try:
                        await self.engine.abort(sid)
                    except Exception:
                        pass
                await self.engine.abort(request_id)
            except Exception as e:
                logger.warning(f"MultiNodeVLLMEngine.abort_request failed: {type(e).__name__}: {e}")

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
            n: int = 1,
        ) -> dict | list[dict]:
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
                n: Number of sequences to sample for the same prompt.

            Returns:
                Dict with token_ids, logprobs, stop_reason.
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            n_req = max(1, int(n))
            if n_req > 1 and self._multisample_mode == "sequential_n1":
                sub_ids = {f"{request_id}_s{i}" for i in range(n_req)}
                try:
                    async with self._outer_to_subreq_lock:
                        self._outer_to_subreq_ids[request_id] = sub_ids
                    outs: list[dict] = []
                    for i in range(n_req):
                        sub_id = f"{request_id}_s{i}"
                        out = await self.generate(
                            prompt_ids=prompt_ids,
                            request_id=sub_id,
                            lora_int_id=lora_int_id,
                            lora_path=lora_path,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            logprobs=logprobs,
                            n=1,
                        )
                        assert isinstance(out, dict)
                        outs.append(out)
                    return outs
                finally:
                    async with self._outer_to_subreq_lock:
                        self._outer_to_subreq_ids.pop(request_id, None)

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                logprobs=0 if logprobs else None,
                n=n_req,
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

            t0 = time.perf_counter()
            first_tok_s: float | None = None
            # Get final response
            async with self._maybe_generate_lock():
                t_lock = time.perf_counter()
                async with self._reserve_seq_slots(n_req):
                    await self._register_generate_start()
                    try:
                        async with self._maybe_multisample_lock(n_req):
                            async with self._lock_read():
                                t1 = time.perf_counter()
                                try:
                                    async with self._maybe_add_request_lock():
                                        if self._generate_timeout_s > 0:
                                            collector = await asyncio.wait_for(
                                                self.engine.add_request(
                                                    request_id=request_id,
                                                    prompt=prompt,
                                                    params=sampling_params,
                                                    lora_request=lora_request,
                                                ),
                                                timeout=self._generate_timeout_s,
                                            )
                                        else:
                                            collector = await self.engine.add_request(
                                                request_id=request_id,
                                                prompt=prompt,
                                                params=sampling_params,
                                                lora_request=lora_request,
                                            )
                                except asyncio.TimeoutError as e:
                                    try:
                                        await self.engine.abort(request_id)
                                    except Exception:
                                        pass
                                    raise RuntimeError(
                                        f"vllm_add_request_timeout_s={self._generate_timeout_s} request_id={request_id}"
                                    ) from e
                                final_res = None
                                by_index: dict[int, Any] | None = {} if n_req > 1 else None
                                deadline = None
                                if self._generate_timeout_s > 0:
                                    deadline = time.perf_counter() + self._generate_timeout_s
                                while True:
                                    try:
                                        if deadline is None:
                                            out = await collector.get()
                                        else:
                                            remaining = deadline - time.perf_counter()
                                            if remaining <= 0:
                                                raise asyncio.TimeoutError()
                                            out = await asyncio.wait_for(collector.get(), timeout=remaining)
                                    except asyncio.TimeoutError as e:
                                        try:
                                            await self.engine.abort(request_id)
                                        except Exception:
                                            pass
                                        raise RuntimeError(
                                            f"vllm_generate_timeout_s={self._generate_timeout_s} request_id={request_id}"
                                        ) from e
                                    if first_tok_s is None:
                                        first_tok_s = time.perf_counter() - t0
                                    if by_index is not None:
                                        for oo in out.outputs:
                                            try:
                                                idx = int(getattr(oo, "index"))
                                            except Exception:
                                                idx = -1
                                            by_index[idx] = oo
                                    final_res = out
                                    if out.finished:
                                        break
                        assert final_res is not None
                    finally:
                        await self._register_generate_end()
            t2 = time.perf_counter()
            if self._timing:
                print(
                    f"[vLLM timing] generate req={request_id} prompt_len={len(prompt_ids)} max_tokens={max_tokens} "
                    f"lora_id={lora_int_id} serialize_wait_s={t_lock - t0:.3f} rw_lock_wait_s={t1 - t_lock:.3f} "
                    f"total_s={t2 - t0:.3f} first_tok_s={first_tok_s}"
                    ,
                    flush=True,
                )

            if n_req == 1:
                token_ids = list(final_res.outputs[0].token_ids)  # type: ignore[union-attr]
                log_probs = None
                if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:  # type: ignore[union-attr]
                    log_probs = [
                        logprobs[token_ids[i]].logprob
                        for i, logprobs in enumerate(final_res.outputs[0].logprobs)  # type: ignore[union-attr]
                    ]

                # Determine stop reason
                stop_reason = "length"
                if final_res.outputs[0].finish_reason == "stop":  # type: ignore[union-attr]
                    stop_reason = "stop"
                elif any(tid in [151645, 151643] for tid in token_ids[-3:]):
                    stop_reason = "stop"

                return {
                    "token_ids": token_ids,
                    "logprobs": log_probs,
                    "stop_reason": stop_reason,
                }

            outs = list(final_res.outputs)  # type: ignore[union-attr]
            if len(outs) != n_req:
                assert by_index is not None
                if len(by_index) != n_req:
                    raise RuntimeError(
                        f"vLLM n={n_req} outputs_len={len(outs)} indices={sorted(by_index)}"
                    )
                keys = sorted(by_index)
                if keys and keys[0] == 1 and keys[-1] == n_req:
                    outs = [by_index[i + 1] for i in range(n_req)]
                else:
                    outs = [by_index[i] for i in range(n_req)]

            indices = []
            for i, o in enumerate(outs):
                try:
                    indices.append(int(getattr(o, "index")))
                except Exception:
                    indices.append(i)
            if len(set(indices)) == n_req:
                if min(indices) == 1 and max(indices) == n_req:
                    outs = [o for _, o in sorted(zip(indices, outs, strict=True))]
                elif min(indices) == 0 and max(indices) == n_req - 1:
                    outs = [o for _, o in sorted(zip(indices, outs, strict=True))]

            multi_results: list[dict] = []
            for out in outs:
                out_token_ids = list(out.token_ids)
                out_log_probs = None
                if sampling_params.logprobs is not None and out.logprobs:
                    out_log_probs = [
                        lp[out_token_ids[i]].logprob
                        for i, lp in enumerate(out.logprobs)
                    ]

                out_stop_reason = "length"
                if out.finish_reason == "stop":
                    out_stop_reason = "stop"
                elif any(tid in [151645, 151643] for tid in out_token_ids[-3:]):
                    out_stop_reason = "stop"

                multi_results.append(
                    {
                        "token_ids": out_token_ids,
                        "logprobs": out_log_probs,
                        "stop_reason": out_stop_reason,
                    }
                )

            return multi_results

        async def compute_prompt_logprobs(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int | None,
            lora_path: str | None,
        ) -> list[float | None]:
            """Compute logprobs for prompt tokens.

            Returns a list of length len(prompt_ids), where:
            - logprobs[0] is None (first token has no conditioning context)
            - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1
            """
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]

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

            t0 = time.perf_counter()
            async with self._maybe_prompt_logprobs_lock():
                async with self._lock_read():
                    t1 = time.perf_counter()
                    collector = await self.engine.add_request(
                        request_id=request_id,
                        prompt=prompt,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                    final_res = None
                    while True:
                        out = await collector.get()
                        final_res = out
                        if out.finished:
                            break
                    assert final_res is not None
            t2 = time.perf_counter()
            if self._timing:
                print(
                    f"[vLLM timing] prompt_logprobs req={request_id} prompt_len={len(prompt_ids)} "
                    f"lora_id={lora_int_id} lock_wait_s={t1 - t0:.3f} total_s={t2 - t0:.3f}"
                    ,
                    flush=True,
                )

            # Extract prompt logprobs
            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            out: list[float | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    out.append(None)
                    continue
                token_id = prompt_ids[i]
                token_lp = prompt_logprobs[i].get(token_id)
                out.append(token_lp.logprob if token_lp is not None else None)

            return out

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
    Designed for large models that need >8 GPUs (multi-node) due to weight + KV + LoRA memory.

    Key differences from MultiLoRAInferenceEngine:
    - Uses vLLM's Ray backend instead of verl's single-node ZMQ pattern
    - Controller actor runs CPU-only; vLLM spawns 1-GPU workers in Ray
    - LoRA adapters must be on shared filesystem (all nodes access same path)
    """

    def __init__(
        self,
        model_path: str,
        model_name: str | None = None,
        tensor_parallel_size: int = 16,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int | None = None,
        max_loras: int = 1,
        max_cpu_loras: int | None = None,
        max_lora_rank: int = 8,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int | None = None,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        actor_name: str | None = None,
        shared_adapter_dir: str = "/vePFS-Mindverse/share/tinker_adapters",
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_lora_rank = max_lora_rank
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.quantization = quantization
        self.kv_cache_dtype = kv_cache_dtype
        self.actor_name = actor_name or f"multinode_vllm_{model_path.split('/')[-1].lower()}"
        self.shared_adapter_dir = shared_adapter_dir

        self.registry = MultiNodeLoRARegistry()
        self.engine = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the multi-node vLLM engine."""
        async with self._init_lock:
            if self._initialized:
                return

            if not ray.is_initialized():
                ray.init(
                    address="auto",
                    namespace=PERSISTENT_NAMESPACE,
                    ignore_reinit_error=True,
                    **ray_log_to_driver_kwargs(),
                )

            # MultiNodeVLLMEngine itself does not need Ray GPU resources (vLLM's Ray backend
            # manages the 1-GPU worker actors). Reserving an extra GPU for the controller
            # breaks TP=16 scheduling on 2x8-GPU clusters (would require 17 GPUs).
            controller_gpus = 0
            controller_cpus = 1
            worker_gpus = (
                self.tensor_parallel_size
                * self.pipeline_parallel_size
                * self.data_parallel_size
            )
            total_required_gpus = worker_gpus
            ray_cgraph_get_timeout = (
                os.environ.get("RAY_CGRAPH_get_timeout")
                or os.environ.get("MINT_RAY_CGRAPH_GET_TIMEOUT_S")
                or "1800"
            )

            is_persistent = False
            persistent_csv = os.environ.get("MINT_PERSISTENT_MODELS", "").strip()
            if persistent_csv and self.model_name:
                persistent_models = {m.strip() for m in persistent_csv.split(",") if m.strip()}
                is_persistent = self.model_name in persistent_models

            # Try to connect to existing actor
            existing_actor = None
            try:
                existing_actor = ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                try:
                    is_ready = await asyncio.to_thread(ray.get, existing_actor.is_ready.remote(), timeout=30)
                except SystemExit as e:
                    if getattr(e, "code", None) == 15:
                        raise
                    logger.warning(
                        f"ray.get(is_ready) triggered SystemExit for {self.actor_name}: {e}; treating as not-ready"
                    )
                    is_ready = False
                if is_ready:
                    logger.info(f"Connected to existing MultiNodeVLLMEngine: {self.actor_name}")
                    self.engine = existing_actor
                    self._initialized = True
                    from tinker_server.backend.resource_pool import get_resource_pool, ActorType

                    resource_pool = get_resource_pool()
                    resource_pool.register(
                        actor_name=self.actor_name,
                        actor_type=ActorType.VLLM,
                        num_gpus=total_required_gpus,
                        actor_handle=self.engine,
                        namespace=PERSISTENT_NAMESPACE,
                        base_model=self.model_path,
                        protected=is_persistent,
                    )
                    resource_pool.mark_ready(self.actor_name)
                    return
                else:
                    logger.warning(f"Actor {self.actor_name} exists but not ready, will recreate")
            except (ValueError, ray.exceptions.RayActorError):
                logger.info(f"No existing actor found, creating new: {self.actor_name}")
            except ray.exceptions.GetTimeoutError:
                # Actor might be busy (queued tasks) rather than dead.
                # Do not kill on timeout; reuse and allow requests to queue.
                logger.warning(f"Actor {self.actor_name} is_ready timed out; assuming busy and reusing actor")
                self.engine = existing_actor
                self._initialized = True
                from tinker_server.backend.resource_pool import get_resource_pool, ActorType

                resource_pool = get_resource_pool()
                resource_pool.register(
                    actor_name=self.actor_name,
                    actor_type=ActorType.VLLM,
                    num_gpus=total_required_gpus,
                    actor_handle=self.engine,
                    namespace=PERSISTENT_NAMESPACE,
                    base_model=self.model_path,
                    protected=is_persistent,
                )
                resource_pool.mark_ready(self.actor_name)
                return

            # Kill existing actor if any before creating new
            if existing_actor is not None:
                try:
                    ray_kill.kill(
                        existing_actor,
                        reason="multinode_vllm_recreate",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                    try:
                        pg = ray.util.get_placement_group(f"{self.actor_name}_pg")
                        ray.util.remove_placement_group(pg)
                    except Exception:
                        pass
                    # Wait for Ray to clean up the actor name
                    import time
                    for _ in range(10):
                        await asyncio.sleep(1)
                        try:
                            ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                        except ValueError:
                            break  # Actor name is available
                except Exception as e:
                    logger.warning(f"Error killing actor {self.actor_name}: {e}")

            # Ensure shared adapter directory exists
            os.makedirs(self.shared_adapter_dir, exist_ok=True)

            # Step 1: Ensure enough GPUs are available (evict idle actors if needed)
            from tinker_server.backend.resource_pool import get_resource_pool, ActorType
            resource_pool = get_resource_pool()
            logger.info(
                f"Ensuring {total_required_gpus} GPUs available for multi-node vLLM "
                f"(TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, "
                f"DP={self.data_parallel_size}, expert_parallel={self.enable_expert_parallel}, "
                f"controller_gpus={controller_gpus}, worker_gpus={worker_gpus})"
            )
            await asyncio.to_thread(resource_pool.ensure_gpus_available, total_required_gpus, 300)

            # Step 2: Create a detached placement group and capture child tasks.
            #
            # vLLM's Ray backend spawns 1-GPU RayWorkerWrapper actors. Without a placement group,
            # those workers can collide with Megatron placement groups, leading to vLLM init failures
            # like "Free memory on device ... is less than desired GPU memory utilization".
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            pg_name = f"{self.actor_name}_pg"
            try:
                pg = ray.util.get_placement_group(pg_name)
            except Exception:
                pg_bundles = [{"GPU": 1, "CPU": 1}] * total_required_gpus + [{"CPU": controller_cpus}]
                pg = ray.util.placement_group(
                    pg_bundles,
                    # PACK to minimize fragmentation: multi-node vLLM uses many 1-GPU workers.
                    # SPREAD can occupy 1-3 GPUs on every node, preventing later 4-GPU actors
                    # (e.g., Qwen3-30B) from finding a node with 4 free GPUs.
                    strategy="PACK",
                    name=pg_name,
                    lifetime="detached",
                )
            try:
                await asyncio.to_thread(ray.get, pg.ready())
            except SystemExit as e:
                if getattr(e, "code", None) == 15:
                    raise
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:
                    pass
                raise RuntimeError(f"ray.get(pg.ready()) triggered SystemExit for {pg_name}: {e}") from e

            # Create new engine actor
            MultiNodeVLLMEngine = _create_multinode_vllm_actor(
                max_loras=self.max_loras,
                max_cpu_loras=self.max_cpu_loras,
                max_lora_rank=self.max_lora_rank,
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
            )

            scheduling_opts = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    # vLLM's Ray backend places worker ranks into bundles [0..TP-1] by index.
                    # Place the controller into a CPU-only bundle to avoid reserving an extra GPU
                    # while keeping child task capture for vLLM's Ray worker actors.
                    placement_group_bundle_index=total_required_gpus,
                    placement_group_capture_child_tasks=True,
                )
            }

            env_vars = {
                "PYTHONPATH": PFS_PYTHONPATH,
                "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                "HF_HUB_OFFLINE": "1",
                # vLLM Ray executor uses Ray compiled DAG (cgraph); vLLM defaults to 300s.
                # If a model execution takes longer, EngineCore can die and the actor becomes unusable.
                "RAY_CGRAPH_get_timeout": str(ray_cgraph_get_timeout),
                # Force vLLM v0 engine - v1's multiprocess architecture
                # conflicts with Ray distributed executor backend
                "VLLM_USE_V1": "0",
                "VLLM_DISABLE_PYNCCL": "1",
            }
            for k in (
                "MINT_VLLM_LOG_STATS",
                "MINT_VLLM_ENGINE_LOCK_MODE",
                "MINT_VLLM_REQUEST_TIMING",
                "MINT_VLLM_PROMPT_LOGPROBS_SERIALIZE",
                "MINT_VLLM_SERIALIZE_GENERATE",
            ):
                v = os.environ.get(k)
                if v is not None:
                    env_vars[k] = v

            self.engine = MultiNodeVLLMEngine.options(
                name=self.actor_name,
                namespace=PERSISTENT_NAMESPACE,
                lifetime="detached",
                num_cpus=controller_cpus,
                num_gpus=controller_gpus,
                max_concurrency=int(os.environ.get("MINT_VLLM_ACTOR_MAX_CONCURRENCY", "64")),
                **scheduling_opts,
                runtime_env={"env_vars": env_vars},
            ).remote(
                model_path=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                data_parallel_size=self.data_parallel_size,
                enable_expert_parallel=self.enable_expert_parallel,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                quantization=self.quantization,
                enable_lora=self.max_loras > 0,
                kv_cache_dtype=self.kv_cache_dtype or "auto",
                max_num_batched_tokens=self.max_num_batched_tokens,
            )

            # Initialize engine (this spawns vLLM's Ray workers)
            logger.info(
                f"Initializing MultiNodeVLLMEngine: "
                f"TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, "
                f"DP={self.data_parallel_size}, expert_parallel={self.enable_expert_parallel}, "
                f"total_gpus={total_required_gpus}"
            )
            # Large models can spend 10-30+ minutes loading shards across many GPUs.
            if total_required_gpus >= 16:
                init_timeout = 3600
            elif total_required_gpus >= 8:
                init_timeout = 1800
            elif total_required_gpus >= 4:
                init_timeout = 1800
            else:
                init_timeout = 600

            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: ray.get(self.engine.initialize.remote(), timeout=init_timeout)
                )
            except SystemExit as e:
                if getattr(e, "code", None) == 15:
                    raise
                try:
                    ray_kill.kill(
                        self.engine,
                        reason="multinode_vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                        timeout_s=init_timeout,
                    )
                except Exception:
                    pass
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:
                    pass
                self.engine = None
                raise RuntimeError(f"ray.get(initialize) triggered SystemExit for {self.actor_name}: {e}") from e
            except ray.exceptions.GetTimeoutError:
                logger.error(f"Engine initialization timed out after {init_timeout}s")
                ray_kill.kill(
                    self.engine,
                    reason="multinode_vllm_init_timeout",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=init_timeout,
                )
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:
                    pass
                self.engine = None
                raise RuntimeError(f"MultiNodeVLLMEngine init timed out")
            except Exception:
                try:
                    ray_kill.kill(
                        self.engine,
                        reason="multinode_vllm_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                        timeout_s=init_timeout,
                    )
                except Exception:
                    pass
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:
                    pass
                self.engine = None
                raise

            self._initialized = True
            logger.info(f"MultiNodeInferenceEngine initialized: {self.actor_name}")

            # Register with unified resource pool for LRU tracking
            # Multi-node vLLM internally manages GPU workers, but we track total GPUs for eviction
            resource_pool.register(
                actor_name=self.actor_name,
                actor_type=ActorType.VLLM,
                num_gpus=total_required_gpus,
                actor_handle=self.engine,
                namespace=PERSISTENT_NAMESPACE,
                base_model=self.model_path,
                protected=is_persistent,
            )
            # Mark as ready since initialization completed
            resource_pool.mark_ready(self.actor_name)
            logger.info(
                f"Registered {self.actor_name} with ResourcePool ({total_required_gpus} GPUs)"
            )

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

    async def add_lora_for_session_from_path(
        self,
        sampling_session_id: str,
        lora_path: str,
    ) -> int:
        """Add frozen LoRA weights for a sampling session from filesystem path.

        Used by the ephemeral sampling flow (save_weights_and_get_sampling_client),
        where training workers save adapters to shared filesystem and vLLM loads
        directly from that path.
        """
        if not self._initialized:
            await self.initialize()

        lora_id = await self.registry.allocate(sampling_session_id, lora_path)

        start_time = time.time()
        await self.engine.add_lora.remote(
            lora_int_id=lora_id,
            lora_path=lora_path,
            lora_name=sampling_session_id,
        )
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} from path "
            f"(lora_int_id={lora_id}, path={lora_path}, load_time={load_time:.3f}s)"
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

        ray_get_timeout_s = float(os.environ.get("MINT_VLLM_RAY_GET_TIMEOUT_S", "0"))

        # Look up LoRA for this session
        lora_id = None
        lora_path = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
            if lora_id is not None:
                lora_path = await self.registry.get_adapter_path(lora_id)

        ref = self.engine.generate.remote(
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
        try:
            if ray_get_timeout_s > 0:
                result = await asyncio.to_thread(ray.get, ref, timeout=ray_get_timeout_s)
            else:
                result = await asyncio.to_thread(ray.get, ref)
        except ray.exceptions.GetTimeoutError as e:
            # Avoid killing the actor: killing forces a 60-90s re-init and pollutes latency measurements.
            # Try aborting just this request, then fail loud to the client.
            try:
                abort_ref = self.engine.abort_request.remote(request_id)
                await asyncio.to_thread(ray.get, abort_ref, timeout=10)
            except Exception:
                pass
            raise RuntimeError(
                f"multinode_vllm_ray_get_timeout_s={ray_get_timeout_s} request_id={request_id}"
            ) from e
        except Exception as e:
            try:
                ray_kill.kill(
                    self.engine,
                    reason="multinode_vllm_ray_get_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=10,
                )
            except Exception:
                pass
            raise e

        return GenerateResult(
            token_ids=result["token_ids"],
            logprobs=result.get("logprobs"),
            stop_reason=result.get("stop_reason"),
        )

    async def generate_many(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        num_samples: int,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        logprobs: bool = True,
    ) -> list[GenerateResult]:
        """Generate multiple sequences for the same prompt in a single vLLM request."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        ray_get_timeout_s = float(os.environ.get("MINT_VLLM_RAY_GET_TIMEOUT_S", "0"))

        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1 (got {num_samples})")

        # Look up LoRA for this session
        lora_id = None
        lora_path = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
            if lora_id is not None:
                lora_path = await self.registry.get_adapter_path(lora_id)

        ref = self.engine.generate.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            logprobs=logprobs,
            n=num_samples,
        )
        try:
            if ray_get_timeout_s > 0:
                raw = await asyncio.to_thread(ray.get, ref, timeout=ray_get_timeout_s)
            else:
                raw = await asyncio.to_thread(ray.get, ref)
        except ray.exceptions.GetTimeoutError as e:
            try:
                abort_ref = self.engine.abort_request.remote(request_id)
                await asyncio.to_thread(ray.get, abort_ref, timeout=10)
            except Exception:
                pass
            raise RuntimeError(
                f"multinode_vllm_ray_get_timeout_s={ray_get_timeout_s} request_id={request_id}"
            ) from e
        except Exception as e:
            try:
                ray_kill.kill(
                    self.engine,
                    reason="multinode_vllm_ray_get_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=10,
                )
            except Exception:
                pass
            raise e

        if isinstance(raw, dict):
            raw_list: list[dict] = [raw]
        else:
            raw_list = list(raw)

        return [
            GenerateResult(
                token_ids=r["token_ids"],
                logprobs=r.get("logprobs"),
                stop_reason=r.get("stop_reason"),
            )
            for r in raw_list
        ]

    async def compute_logprobs(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
    ) -> list[float | None]:
        """Compute logprobs using session-specific LoRA or base model."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        ray_get_timeout_s = float(os.environ.get("MINT_VLLM_RAY_GET_TIMEOUT_S", "0"))

        # Look up LoRA for this session
        lora_id = None
        lora_path = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
            if lora_id is not None:
                lora_path = await self.registry.get_adapter_path(lora_id)

        ref = self.engine.compute_prompt_logprobs.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )
        try:
            if ray_get_timeout_s > 0:
                result = await asyncio.to_thread(ray.get, ref, timeout=ray_get_timeout_s)
            else:
                result = await asyncio.to_thread(ray.get, ref)
        except ray.exceptions.GetTimeoutError as e:
            raise RuntimeError(
                f"multinode_vllm_ray_get_timeout_s={ray_get_timeout_s} request_id={request_id}"
            ) from e
        except Exception as e:
            try:
                ray_kill.kill(
                    self.engine,
                    reason="multinode_vllm_ray_get_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=10,
                )
            except Exception:
                pass
            raise e

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
                ray_kill.kill(
                    self.engine,
                    reason="multinode_vllm_shutdown",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                )
                logger.info("Killed MultiNodeVLLMEngine actor")
            except Exception as e:
                logger.warning(f"Error killing actor: {e}")
        self.engine = None
        self._initialized = False
        logger.info("MultiNodeInferenceEngine disconnected")
