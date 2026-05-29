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
import math
import os
import sys
import time
import traceback
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ray

from mint_server.config import PFS_PYTHONPATH, RAY_NAMESPACE, preferred_torch_lib_dirs
from mint_server.config import config as server_config
from mint_server.logging_context import (
    get_current_traceparent,
    init_actor_observability,
    restore_trace_id_from_traceparent,
    start_as_current_span,
    traced_async_from_traceparent,
)
from mint_server.ray_utils import init_ray
from mint_server.runtime_env import join_pythonpath, sanitize_worker_pythonpath

from . import ray_kill
from .async_ray_control import async_get_ray_ref
from .gpu_binding_helpers import gpu_bindings_from_ray_gpu_ids
from .multinode_resources import MultiNodeEngineResources, compute_multinode_engine_resources
from .ray_placement_groups import PlacementGroupMismatchError, get_named_placement_group
from .ray_keepalive import ray_get_with_model_actor_supervisor_keepalive
from .runtime_actor_metrics import current_ray_actor_name, init_vllm_runtime_otel_metrics
from .vllm_scheduler_observability import (
    VllmStatsObserver,
    install_vllm_iteration_observability_patches,
    make_vllm_stats_logger_factory,
)
from .node_placement import (
    assert_node_ip_capacity,
    ModelGpuPlacement,
    parse_model_gpu_placement,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
VLLM_TOKEN_BUDGET_RATIO = 0.95
VLLM_NO_COMPILED_DAG_ENV = "MINT_VLLM_RAY_EXECUTOR_NO_COMPILED_DAG_SAMPLE"


def _progress_meta(tokens_generated: int, max_tokens: int) -> dict[str, Any]:
    return {
        "tokens_generated": int(tokens_generated),
        "max_tokens": int(max_tokens),
        "last_progress_at": time.time(),
    }

# Namespace for actors
PERSISTENT_NAMESPACE = RAY_NAMESPACE


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _import_vllm_async_engine_components() -> tuple[type[Any], type[Any]]:
    try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
    except ImportError as top_level_error:
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
        except ImportError:
            raise top_level_error
    return AsyncEngineArgs, AsyncLLMEngine


def _enforce_vllm_no_compiled_dag(
    env_vars: dict[str, str],
    *,
    distributed_executor_backend: str,
) -> None:
    """Keep compiled DAG disabled through the one patch flag the current build honors."""
    env_vars.pop(VLLM_NO_COMPILED_DAG_ENV, None)
    env_vars.pop("VLLM_DISABLE_RAY_COMPILED_DAG", None)
    if distributed_executor_backend == "ray":
        env_vars[VLLM_NO_COMPILED_DAG_ENV] = "1"


def _set_default_vllm_runtime_env(env_vars: dict[str, str]) -> None:
    env_vars.setdefault("MINT_ENABLE_VLLM_IMPORT_PATCHES", "1")
    env_vars.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env_vars.setdefault("USE_TF", "0")
    env_vars.setdefault("USE_FLAX", "0")


def _prepare_mint_vllm_multinode_runtime_env(env_vars: dict[str, str]) -> None:
    _set_default_vllm_runtime_env(env_vars)
    # Driver/head temp-dir hints are not portable into Ray worker-hosted vLLM
    # EngineCore subprocesses. Let each controller actor stamp its local node IP.
    env_vars.pop("MINT_RAY_TEMP_DIR", None)
    env_vars.pop("MINT_RAY_NODE_IP_ADDRESS", None)


def _prepend_env_path_entries(raw: str | None, entries: list[str], *, blocked: set[str] | None = None) -> str:
    blocked = blocked or set()
    out: list[str] = []
    seen: set[str] = set()
    for entry in [*entries, *str(raw or "").split(":")]:
        if not entry:
            continue
        norm = os.path.normcase(os.path.abspath(entry))
        if norm in blocked or norm in seen:
            continue
        seen.add(norm)
        out.append(entry)
    return ":".join(out)


def _stabilize_vllm_child_environment() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    try:
        os.environ["MINT_RAY_NODE_IP_ADDRESS"] = ray.util.get_node_ip_address()
    except Exception:
        pass
    os.environ.pop("MINT_RAY_TEMP_DIR", None)

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    python_entries = [
        path
        for path in (
            "/vllm",
            f"/opt/venv/lib/{pyver}/site-packages",
        )
        if os.path.isdir(path)
    ]
    if python_entries:
        os.environ["PYTHONPATH"] = _prepend_env_path_entries(os.environ.get("PYTHONPATH"), python_entries)
        for entry in reversed(python_entries):
            if entry not in sys.path:
                sys.path.insert(0, entry)

    blocked_ld = {
        os.path.normcase("/usr/local/lib/python3.10/dist-packages/torch/lib"),
        os.path.normcase("/usr/local/lib/python3.10/site-packages/torch/lib"),
    }
    ld_entries = [
        *preferred_torch_lib_dirs(),
        "/usr/local/cuda/compat/lib",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
        "/usr/local/cuda/lib64",
    ]
    os.environ["LD_LIBRARY_PATH"] = _prepend_env_path_entries(
        os.environ.get("LD_LIBRARY_PATH"),
        ld_entries,
        blocked=blocked_ld,
    )

    try:
        import multiprocessing

        preferred_executable = os.environ.get("MINT_VLLM_CHILD_PYTHON_EXECUTABLE", "").strip() or sys.executable
        if preferred_executable and os.path.exists(preferred_executable):
            multiprocessing.set_executable(preferred_executable)
    except Exception as e:
        logger.warning("failed to pin multiprocessing executable: %s: %s", type(e).__name__, e)


def _patch_vllm_fused_moe_slice_for_fully_sharded_loras() -> None:
    """Patch vLLM fused-MoE LoRA slicing for fully-sharded dummy weights.

    In vLLM, `FusedMoEWithLoRA` allocates W13 LoRA-A stacked tensors with rank
    divided by TP when `fully_sharded_loras=True`. During adapter activation,
    vLLM can call `_slice_w13_a()` on tensors already in per-rank shape, which
    triggers an assertion on `current_lora_rank % tp_size`.
    """

    import vllm.lora.layers.fused_moe as fused_moe_mod

    def _patch_cls(cls: type) -> None:
        original = getattr(cls, "_slice_w13_a", None)
        if original is None:
            raise RuntimeError(f"vLLM class {cls.__name__} has no _slice_w13_a")
        if getattr(original, "_mint_patched_fully_sharded", False):
            return

        def _slice_w13_a(self, w13_lora_a):  # type: ignore[no-untyped-def]
            if self.tp_size == 1 or not self.fully_sharded:
                return w13_lora_a

            expected_rank = int(self.w13_lora_a_stacked[0].shape[2])
            current_rank = int(w13_lora_a.shape[1])
            if current_rank == expected_rank:
                return w13_lora_a

            if current_rank % self.tp_size != 0:
                raise RuntimeError(
                    "vLLM fused_moe _slice_w13_a unexpected rank: "
                    f"current_rank={current_rank} tp_size={self.tp_size} expected_rank={expected_rank}"
                )

            sliced_rank = current_rank // self.tp_size
            start_idx = self.tp_rank * sliced_rank
            end_idx = (self.tp_rank + 1) * sliced_rank
            return w13_lora_a[:, start_idx:end_idx, :]

        _slice_w13_a._mint_patched_fully_sharded = True  # type: ignore[attr-defined]
        cls._slice_w13_a = _slice_w13_a  # type: ignore[method-assign]

    for name in ("FusedMoEWithLoRA", "FusedMoE3DWithLoRA"):
        cls = getattr(fused_moe_mod, name, None)
        if cls is None:
            raise RuntimeError(f"vLLM fused_moe is missing class {name}")
        _patch_cls(cls)


def _node_affinity_scheduling_opts_for_model(
    model_name: str | None,
    *,
    required_gpus: int,
) -> dict[str, Any]:
    """Optional single-node pinning for vLLM actors (mp backend).

    Use-case: pack MoE inference+training on the same 8-GPU node under topology placement.
    Controlled by MINT_VLLM_MODEL_PLACEMENT_JSON or MINT_MODEL_PLACEMENT_JSON.
    """
    if not model_name:
        return {}
    placement = _model_gpu_placement_for_model(model_name)
    if placement is None:
        return {}
    if len(placement.slices) != 1:
        raise RuntimeError(
            f"mp vLLM requires exactly 1 placement slice, got {len(placement.slices)} "
            f"for model={model_name!r}"
        )
    pinned_ip = placement.slices[0].node_ip
    if placement.total_gpus != int(required_gpus):
        raise RuntimeError(
            f"mp vLLM placement GPU count mismatch for model={model_name!r}: "
            f"need {required_gpus} GPUs, got {placement.total_gpus}"
        )
    assert_node_ip_capacity(
        required_gpus_by_node_ip={pinned_ip: int(required_gpus)},
        context=f"mint_vllm_multinode_pin model={model_name}",
    )

    node_res = f"node:{pinned_ip}"
    logger.info(f"mint_vllm_multinode_pin model={model_name} resources={node_res!r}")
    return {"resources": {node_res: 0.001}}


def _model_gpu_placement_for_model(model_name: str | None) -> ModelGpuPlacement | None:
    if not model_name:
        return None
    lookup_keys = [model_name, model_name.lower()]
    context = f"mint_vllm_multinode_placement model={model_name}"
    placement = parse_model_gpu_placement(
        raw_json=os.environ.get("MINT_VLLM_MODEL_PLACEMENT_JSON"),
        lookup_keys=lookup_keys,
        env_var_name="MINT_VLLM_MODEL_PLACEMENT_JSON",
        context=context,
        replica=0,
    )
    if placement is None:
        placement = parse_model_gpu_placement(
            raw_json=os.environ.get("MINT_MODEL_PLACEMENT_JSON"),
            lookup_keys=lookup_keys,
            env_var_name="MINT_MODEL_PLACEMENT_JSON",
            context=context,
            replica=0,
        )
    if placement is not None:
        logger.info(
            "mint_vllm_multinode_placement model=%s total_gpus=%s slices=%s",
            model_name,
            placement.total_gpus,
            [
                {
                    "replica": slice_.replica,
                    "worker_index": slice_.worker_index,
                    "gpu_count": slice_.gpu_count,
                    "node_ip": slice_.node_ip,
                }
                for slice_ in placement.slices
            ],
        )
    return placement


def _raise_serializable_vllm_error(*, where: str, request_id: str, extra: dict[str, Any]) -> None:
    # Do not chain the original exception object (it may not be picklable across Ray).
    tb = traceback.format_exc()
    extra_str = " ".join(f"{k}={v!r}" for k, v in sorted(extra.items()))
    raise RuntimeError(f"{where} request_id={request_id} {extra_str}\n{tb}") from None


def _raise_serializable_vllm_engine_error(
    *, where: str, request_id: str, extra: dict[str, Any]
) -> None:
    # Same as _raise_serializable_vllm_error, but semantically used inside the vLLM Ray actor
    # to avoid propagating non-picklable vLLM exception objects back to the CPU driver.
    _raise_serializable_vllm_error(where=where, request_id=request_id, extra=extra)


def _is_request_validation_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "vllm_prompt_logprobs_add_request_failed",
            "vllm_prompt_topk_add_request_failed",
            "vllm_generate_add_request_failed",
            "Requested prompt logprobs of",
            "Prompt+max_tokens length",
            "exceeds max_model_len",
            "maximum model length",
            "Prompt length (",
            "model context limit",
        )
    )


def _is_missing_lora_path_error(exc: BaseException) -> bool:
    """Return True for request-scoped missing-adapter/path failures.

    These indicate missing request inputs (evicted/deleted adapter path), not a
    broken vLLM actor. They must fail the request without killing the actor.
    """
    text_candidates: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        message = str(cur)
        if message:
            text_candidates.append(message.lower())
        cur = cur.__cause__ if isinstance(cur.__cause__, BaseException) else None

    joined = "\n".join(text_candidates)
    if any(
        marker in joined
        for marker in (
            "missing lora adapter path",
            "no adapter found for",
            "failed: no adapter found",
            "no path found for lora_int_id",
        )
    ):
        return True

    has_missing_file_signal = any(
        marker in joined
        for marker in (
            "file not found",
            "no such file or directory",
        )
    )
    if not has_missing_file_signal:
        return False

    return any(
        marker in joined
        for marker in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "lora_int_id",
            "lora adapter path",
            "lora_path",
        )
    )


def _raise_if_missing_lora_path(
    *,
    sampling_session_id: str | None,
    lora_int_id: int | None,
    lora_path: str | None,
) -> None:
    """Fail fast when a request-scoped LoRA path is missing.

    This protects the multi-node path from escalating a missing adapter path
    into a generic engine failure that kills the actor.
    """
    if lora_int_id is None:
        return
    if not lora_path or not os.path.isdir(lora_path):
        raise ValueError(
            "Missing LoRA adapter path for "
            f"sampling_session_id={sampling_session_id!r} "
            f"lora_int_id={lora_int_id!r} path={lora_path!r}"
        )


@dataclass
class MultiNodeLoRASlot:
    """Metadata for a loaded LoRA adapter in multi-node engine."""

    lora_int_id: int
    sampling_session_id: str
    adapter_path: str  # Shared filesystem path
    session_ids: set[str] = field(default_factory=set)
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

            for lora_id, slot in self._id_to_slot.items():
                if slot.adapter_path != adapter_path:
                    continue
                self._session_to_id[sampling_session_id] = lora_id
                slot.session_ids.add(sampling_session_id)
                slot.last_used = time.time()
                logger.debug(
                    "Reused lora_int_id=%s for session %s (adapter_path=%s)",
                    lora_id,
                    sampling_session_id,
                    adapter_path,
                )
                return lora_id

            lora_id = self._next_id
            self._next_id += 1

            self._session_to_id[sampling_session_id] = lora_id
            self._id_to_slot[lora_id] = MultiNodeLoRASlot(
                lora_int_id=lora_id,
                sampling_session_id=sampling_session_id,
                adapter_path=adapter_path,
                session_ids={sampling_session_id},
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

    async def restore_existing_session(
        self,
        sampling_session_id: str,
        *,
        adapter_path: str,
        lora_int_id: int,
    ) -> int:
        """Rehydrate an already-loaded LoRA mapping after API restart."""
        async with self._lock:
            existing_id = self._session_to_id.get(sampling_session_id)
            if existing_id is not None:
                return existing_id

            lora_id = int(lora_int_id)
            slot = self._id_to_slot.get(lora_id)
            if slot is None:
                slot = MultiNodeLoRASlot(
                    lora_int_id=lora_id,
                    sampling_session_id=sampling_session_id,
                    adapter_path=str(adapter_path),
                    session_ids={sampling_session_id},
                )
                self._id_to_slot[lora_id] = slot
            else:
                if str(slot.adapter_path) != str(adapter_path):
                    raise ValueError(
                        f"lora_int_id={lora_id} already mapped to adapter_path={slot.adapter_path}, "
                        f"cannot restore adapter_path={adapter_path}"
                    )
                slot.session_ids.add(sampling_session_id)
                slot.last_used = time.time()
            self._session_to_id[sampling_session_id] = lora_id
            self._next_id = max(self._next_id, lora_id + 1)
            return lora_id

    async def remove_session(self, sampling_session_id: str) -> tuple[int | None, bool]:
        """Remove a session mapping and report whether engine unload is needed."""
        async with self._lock:
            lora_id = self._session_to_id.pop(sampling_session_id, None)
            if lora_id is None:
                return None, False
            slot = self._id_to_slot.get(lora_id)
            if slot is None:
                return lora_id, False
            slot.session_ids.discard(sampling_session_id)
            if slot.sampling_session_id == sampling_session_id and slot.session_ids:
                slot.sampling_session_id = next(iter(slot.session_ids))
            if slot.session_ids:
                logger.debug(
                    "Removed session %s from shared lora_int_id=%s (remaining_sessions=%s)",
                    sampling_session_id,
                    lora_id,
                    sorted(slot.session_ids),
                )
                return lora_id, False
            self._id_to_slot.pop(lora_id, None)
            logger.debug("Removed final session %s for lora_int_id=%s", sampling_session_id, lora_id)
            return lora_id, True

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


def _create_mint_vllm_multinode_actor(
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

    max_concurrency = int(os.environ.get("MINT_MULTINODE_VLLM_ACTOR_MAX_CONCURRENCY", "128"))

    @ray.remote(
        num_cpus=1,
        max_concurrency=max_concurrency,
    )  # num_gpus=0: vLLM's internal Ray backend manages GPU allocation
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
            init_actor_observability()
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
            env_max_num_batched_tokens = os.environ.get("MINT_VLLM_MAX_NUM_BATCHED_TOKENS")
            self.max_num_batched_tokens = (
                int(env_max_num_batched_tokens)
                if env_max_num_batched_tokens is not None and env_max_num_batched_tokens.strip()
                else max_num_batched_tokens
            )

            self.engine = None
            self._initialized = False
            self._rw_lock = _AsyncRWLock()
            self._lock_mode = os.environ.get("MINT_VLLM_ENGINE_LOCK_MODE", "rw").strip().lower()
            self._timing = _env_flag("MINT_VLLM_REQUEST_TIMING", default=True)
            self._serialize_prompt_logprobs = _env_flag("MINT_VLLM_PROMPT_LOGPROBS_SERIALIZE", default=False)
            self._prompt_logprobs_lock = asyncio.Lock() if self._serialize_prompt_logprobs else None
            # vLLM supports concurrent requests for continuous batching, but some engine calls
            # (notably list_loras) must not race active generation on multinode.
            #
            # Allow concurrent generate() calls by default so vLLM can continuously batch
            # requests across clients. Set MINT_VLLM_SERIALIZE_GENERATE=1 to serialize.
            self._serialize_generate = _env_flag("MINT_VLLM_SERIALIZE_GENERATE", default=False)
            self._generate_lock = asyncio.Lock() if self._serialize_generate else None
            # vLLM SamplingParams(n>1) has shown hangs under concurrent multinode traffic.
            # Default to issuing N concurrent n=1 requests (see MINT_VLLM_MULTISAMPLE_MODE),
            # avoiding vLLM's `SamplingParams(n>1)` path. Serialize only when using vLLM's n>1.
            # AsyncLLMEngine.add_request has shown hangs when called concurrently on multinode.
            # Serialize add_request() while still allowing concurrent in-flight requests.
            self._serialize_add_request = _env_flag("MINT_VLLM_SERIALIZE_ADD_REQUEST", default=True)
            self._add_request_lock = asyncio.Lock() if self._serialize_add_request else None
            # Multinode multi-sample can run either through vLLM's native `n>1` path or
            # by expanding into repeated `n=1` requests.
            #
            # Keep the default on vLLM's native `SamplingParams(n>1)` path. The issue428
            # experiment should compare that native behavior before/after reducing the
            # multisample-specific full-request mutex, rather than changing the default
            # serving semantics at the mint-server layer.
            #
            # Modes:
            # - "vllm_n": use `SamplingParams(n=N)` (default)
            # - "sequential_n1": run N sequential `SamplingParams(n=1)` requests
            # - "concurrent_n1": run N concurrent `SamplingParams(n=1)` requests
            self._multisample_mode = os.environ.get("MINT_VLLM_MULTISAMPLE_MODE", "vllm_n").strip().lower()
            default_serialize_multisample = self._multisample_mode == "vllm_n"
            self._serialize_multisample = _env_flag(
                "MINT_VLLM_SERIALIZE_MULTISAMPLE",
                default=default_serialize_multisample,
            )
            self._multisample_lock = asyncio.Lock() if self._serialize_multisample else None
            self._outer_to_subreq_ids: dict[str, set[str]] = {}
            self._outer_to_subreq_lock = asyncio.Lock()
            self._generate_timeout_s = float(os.environ.get("MINT_VLLM_GENERATE_TIMEOUT_S", "0"))
            self._post_generate_delay_s = float(os.environ.get("MINT_VLLM_POST_GENERATE_DELAY_S", "0"))
            self._gate_lock = asyncio.Lock()
            # Default to allowing add_lora to run without draining active generation first.
            # Set MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE=1 to restore the conservative gate.
            self._serialize_add_lora_until_idle = _env_flag(
                "MINT_VLLM_SERIALIZE_ADD_LORA_UNTIL_IDLE", default=False
            )
            self._vllm_stats_observer = VllmStatsObserver()
            self._kv_cache_observability: dict[str, int] = {}
            self._otel_runtime_metrics_enabled = init_vllm_runtime_otel_metrics(
                snapshot_fn=self._vllm_stats_observer.snapshot,
                actor_name=current_ray_actor_name("unknown"),
                base_model=str(self.model_path or "unknown"),
            )
            self._active_generates = 0
            self._active_generates_cond = asyncio.Condition()
            self._is_ready_timeout_s = float(os.environ.get("MINT_VLLM_IS_READY_TIMEOUT_S", "0.05"))
            self._progress_by_outer: dict[str, dict[str, int]] = {}
            self._progress_total_by_outer: dict[str, int] = {}
            self._progress_last_by_outer: dict[str, float] = {}
            self._progress_lock = asyncio.Lock()
            # vLLM's `max_num_seqs` is a hard cap on active sequences. Under multinode + long-context
            # + multi-sample (SamplingParams(n>1)), vLLM can hang when the server oversubscribes this
            # cap and relies on vLLM internal queueing. Use explicit admission control to avoid
            # exceeding `max_num_seqs` from the server side (no client API change).
            self._admission_control = _env_flag("MINT_VLLM_ADMISSION_CONTROL", default=True)
            self._active_seq_slots = 0
            self._seq_slots_cond = asyncio.Condition()
        def get_node_ip(self) -> str:
            return ray.util.get_node_ip_address()

        def get_rss_bytes(self) -> int:
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) < 2:
                raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
            rss_pages = int(parts[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return rss_pages * page_size

        async def get_observability_binding(self) -> dict[str, object]:
            import socket

            hostname = socket.gethostname()
            node_id = None
            try:
                node_id = str(ray.get_runtime_context().get_node_id())
            except Exception:
                node_id = None
            gpu_bindings = gpu_bindings_from_ray_gpu_ids(hostname=hostname, node_id=node_id)
            return {
                "hostname": hostname,
                "node_id": node_id,
                "gpu_indices": [binding["gpu_index"] for binding in gpu_bindings if "gpu_index" in binding],
                "gpu_bindings": gpu_bindings,
                **self._vllm_stats_observer.snapshot(),
                **self._kv_cache_observability,
            }

        def _bind_traceparent(self, traceparent: str | None) -> None:
            if isinstance(traceparent, str) and traceparent:
                restore_trace_id_from_traceparent(traceparent)

        def _seq_slot_need(self, n_req: int) -> int:
            if (not self._admission_control) or (self.max_num_seqs is None):
                return 0
            return max(1, int(n_req))

        async def _acquire_seq_slots(self, n_req: int) -> int:
            need = self._seq_slot_need(n_req)
            if need <= 0:
                return 0
            async with self._seq_slots_cond:
                while self._active_seq_slots + need > int(self.max_num_seqs):
                    await self._seq_slots_cond.wait()
                self._active_seq_slots += need
            return need

        async def _release_seq_slots(self, need: int) -> None:
            if int(need) <= 0:
                return
            async with self._seq_slots_cond:
                self._active_seq_slots -= int(need)
                self._seq_slots_cond.notify_all()

        @asynccontextmanager
        async def _reserve_seq_slots(self, n_req: int):
            need = await self._acquire_seq_slots(n_req)
            try:
                yield
            finally:
                await self._release_seq_slots(need)

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

        async def _update_progress(
            self,
            *,
            outer_request_id: str,
            sub_request_id: str | None,
            tokens_generated: int,
            max_tokens: int,
            min_interval_s: float = 5.0,
        ) -> None:
            now = time.time()
            async with self._progress_lock:
                last = self._progress_last_by_outer.get(outer_request_id)
                if last is not None and (now - last) < min_interval_s:
                    return
                self._progress_last_by_outer[outer_request_id] = now
                if sub_request_id is not None:
                    bucket = self._progress_by_outer.setdefault(outer_request_id, {})
                    bucket[str(sub_request_id)] = int(tokens_generated)
                    total = self._progress_total_by_outer.get(outer_request_id)
                    values = list(bucket.values())
                    if total is not None and len(values) < int(total):
                        values.append(0)
                    min_tokens = min(values) if values else int(tokens_generated)
                else:
                    min_tokens = int(tokens_generated)
            try:
                from .task_state_store import task_futures

                await task_futures.async_update_meta(
                    outer_request_id,
                    meta={
                        "stage": "decode",
                        "progress": _progress_meta(min_tokens, max_tokens),
                        "last_progress_at": time.time(),
                    },
                )
            except Exception as e:
                logger.warning(
                    "mint_vllm_multinode_progress_update_failed request_id=%s err=%s: %s",
                    outer_request_id,
                    type(e).__name__,
                    e,
                )

        async def _clear_progress(self, outer_request_id: str) -> None:
            async with self._progress_lock:
                self._progress_by_outer.pop(outer_request_id, None)
                self._progress_total_by_outer.pop(outer_request_id, None)
                self._progress_last_by_outer.pop(outer_request_id, None)

        @asynccontextmanager
        async def _exclusive_engine_op(self):
            async with self._gate_lock:
                async with self._active_generates_cond:
                    while self._active_generates > 0:
                        await self._active_generates_cond.wait()
                yield

        @asynccontextmanager
        async def _maybe_add_lora_idle_gate(self):
            if self._serialize_add_lora_until_idle:
                async with self._exclusive_engine_op():
                    yield
                return
            yield

        async def initialize(self) -> None:
            """Initialize vLLM engine with Ray distributed backend."""
            if self._initialized:
                return

            import os

            _stabilize_vllm_child_environment()
            distributed_executor_backend = os.environ.get("MINT_VLLM_DISTRIBUTED_EXECUTOR_BACKEND", "ray").strip().lower()
            if "VLLM_USE_V1" not in os.environ:
                os.environ["VLLM_USE_V1"] = "1" if distributed_executor_backend == "mp" else "0"
            # PyNcclCommunicator has hit NCCL internal errors in multi-node init;
            # disable to fall back to torch.distributed collectives.
            os.environ["VLLM_DISABLE_PYNCCL"] = "1"
            logger.info(
                "vllm_child_env python=%s ray_address=%s py_path_head=%s ld_library_path=%s",
                sys.executable,
                os.environ.get("RAY_ADDRESS", ""),
                sys.path[:8],
                os.environ.get("LD_LIBRARY_PATH", ""),
            )

            # Import vLLM components AFTER setting env var.
            AsyncEngineArgs, AsyncLLMEngine = _import_vllm_async_engine_components()

            # Build engine args for multi-node TP
            # prompt_logprobs uses float32 log_softmax over [tokens, vocab], which can spike memory.
            max_num_batched_tokens = self.max_num_batched_tokens
            if max_num_batched_tokens is None:
                max_num_batched_tokens = 4096 if (self.max_model_len or 0) >= 32768 else 8192
            max_num_batched_tokens = int(os.environ.get("MINT_VLLM_MAX_NUM_BATCHED_TOKENS", str(max_num_batched_tokens)))
            enable_chunked_prefill = _env_flag("MINT_VLLM_ENABLE_CHUNKED_PREFILL", default=True)
            enable_prefix_caching = _env_flag("MINT_VLLM_ENABLE_PREFIX_CACHING", default=True)
            if distributed_executor_backend not in ("ray", "mp"):
                raise ValueError(
                    f"Invalid MINT_VLLM_DISTRIBUTED_EXECUTOR_BACKEND={distributed_executor_backend!r} "
                    f"(expected 'ray' or 'mp')"
                )
            fully_sharded_loras = (
                _env_flag("MINT_VLLM_FULLY_SHARDED_LORAS", default=True)
                and self.enable_lora
                and self.max_lora_rank is not None
                and self.max_lora_rank % self.tensor_parallel_size == 0
            )
            if fully_sharded_loras:
                _patch_vllm_fused_moe_slice_for_fully_sharded_loras()
            lora_dtype_env = os.environ.get("MINT_VLLM_LORA_DTYPE", "auto").strip()
            lora_dtype_resolved = lora_dtype_env
            if self.enable_lora and lora_dtype_env.lower() == "auto":
                def _infer_hf_torch_dtype_str(model_path: str) -> str | None:
                    from transformers import AutoConfig
                    import torch

                    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    torch_dtype = getattr(cfg, "torch_dtype", None)
                    if torch_dtype is None:
                        return None
                    if isinstance(torch_dtype, str):
                        dtype_str = torch_dtype
                    elif isinstance(torch_dtype, torch.dtype):
                        dtype_str = str(torch_dtype)
                    else:
                        dtype_str = str(torch_dtype)
                    dtype_str = dtype_str.replace("torch.", "").strip().lower()
                    if dtype_str in ("fp16", "float16", "half"):
                        return "float16"
                    if dtype_str in ("bf16", "bfloat16"):
                        return "bfloat16"
                    if dtype_str in ("fp32", "float32"):
                        return "float32"
                    return None

                # vLLM fused MoE LoRA Triton kernels reinterpret LoRA weight pointers as the
                # output dtype (see vllm/lora/ops/triton_ops/fused_moe_lora_op.py). If LoRA
                # weights are loaded in fp16 while model output is bf16, the kernel will
                # read fp16 memory as bf16 and can produce NaNs. Default to bf16 for BF16
                # model snapshots unless explicitly overridden.
                inferred = _infer_hf_torch_dtype_str(str(self.model_path))
                if inferred is not None:
                    lora_dtype_resolved = inferred
                elif "BF16" in str(self.model_path).upper():
                    lora_dtype_resolved = "bfloat16"
                else:
                    logger.warning(
                        "Unable to infer HF torch_dtype for model=%r with enable_lora=1; "
                        "leaving vLLM lora_dtype='auto' (set MINT_VLLM_LORA_DTYPE to override)",
                        self.model_path,
                    )

            logger.info(
                "vLLM LoRA dtype: env=%r resolved=%r", lora_dtype_env, lora_dtype_resolved
            )
            enable_return_routed_experts = (server_config.router_replay_mode == "R3")
            if enable_return_routed_experts:
                import inspect

                sig = inspect.signature(AsyncEngineArgs.__init__)
                has_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
                if "enable_return_routed_experts" not in sig.parameters and not has_kwargs:
                    import vllm  # type: ignore

                    raise RuntimeError(
                        "router_replay_mode=R3 requires vLLM AsyncEngineArgs(enable_return_routed_experts=...). "
                        f"Installed vllm={getattr(vllm, '__version__', 'unknown')!r} does not support it "
                        f"(AsyncEngineArgs.__init__ signature={sig})."
                    )
            all2all_backend_env = os.environ.get("MINT_VLLM_ALL2ALL_BACKEND", "").strip().lower()
            default_all2all_backend = (
                "deepep_high_throughput"
                if self.enable_expert_parallel
                else "allgather_reducescatter"
            )
            all2all_backend = all2all_backend_env or default_all2all_backend
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                data_parallel_size=self.data_parallel_size,
                data_parallel_backend="ray" if self.data_parallel_size > 1 else "mp",
                enable_expert_parallel=self.enable_expert_parallel,
                all2all_backend=all2all_backend,
                distributed_executor_backend=distributed_executor_backend,
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
                fully_sharded_loras=fully_sharded_loras if self.enable_lora else False,
                lora_dtype=lora_dtype_resolved,
                enable_return_routed_experts=enable_return_routed_experts,
            )

            logger.info(
                f"Creating AsyncLLMEngine: "
                f"TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, "
                f"DP={self.data_parallel_size}, expert_parallel={self.enable_expert_parallel}, a2a_backend={all2all_backend}, "
                f"backend={distributed_executor_backend}, enable_lora={self.enable_lora}, gpu_util={self.gpu_memory_utilization}, "
                f"fully_sharded_loras={fully_sharded_loras}, chunked_prefill={enable_chunked_prefill}, "
                f"max_num_batched_tokens={max_num_batched_tokens}, "
                f"prefix_caching={enable_prefix_caching}"
            )

            # Install vLLM-side hooks before engine creation so scheduler stats
            # include per-iteration token and executor timing signals.
            install_vllm_iteration_observability_patches()

            # Create engine - vLLM will spawn Ray workers across nodes
            self.engine = AsyncLLMEngine.from_engine_args(
                engine_args,
                stat_loggers=[make_vllm_stats_logger_factory(self._vllm_stats_observer)],
            )
            try:
                info = await self.get_kv_debug_info()
                capacity = int(info.get("kv_cache_capacity_tokens") or 0)
                if capacity > 0:
                    self._kv_cache_observability = {
                        "kv_cache_capacity_tokens": capacity,
                        "kv_cache_token_budget": max(
                            1,
                            int(math.floor(float(capacity) * VLLM_TOKEN_BUDGET_RATIO)),
                        ),
                    }
                    block_size = int(info.get("kv_cache_block_size") or 0)
                    if block_size > 0:
                        self._kv_cache_observability["kv_cache_block_size"] = block_size
                    num_blocks = int(info.get("min_kv_cache_num_blocks") or 0)
                    if num_blocks > 0:
                        self._kv_cache_observability["kv_cache_num_blocks"] = num_blocks
            except Exception:
                logger.warning("vllm_kv_cache_observability_init_failed", exc_info=True)

            self._initialized = True
            logger.info("MultiNodeVLLMEngine initialized")

        async def is_ready(self) -> bool:
            """Check if engine is initialized and the EngineCore is responsive.

            Semantics:
            - Returns False when the engine is busy (cannot safely probe EngineCore).
              Callers must treat False as "not ready / unknown", not as "dead".
            - When idle, returns True without touching EngineCore. (EngineCore probes
              can hang indefinitely under Ray distributed executor and will block
              the actor event loop, preventing later `generate()` calls.)
            """
            if not self._initialized or self.engine is None:
                return False
            try:
                # Touch EngineCore. The Ray actor can be alive while EngineCore is dead.
                #
                # IMPORTANT: `list_loras()` must not run concurrently with `generate()`.
                # Also: avoid cancelling vLLM engine coroutines during liveness checks. Cancellation
                # can leave engine state inconsistent and later generations can hang.
                if self._gate_lock.locked():
                    return False
                async with self._gate_lock:
                    async with self._active_generates_cond:
                        if self._active_generates > 0:
                            return False
                    return True
            except Exception as e:
                logger.warning(f"MultiNodeVLLMEngine is_ready failed: {type(e).__name__}: {e}")
                return False

        @traced_async_from_traceparent(
            "sampling.mint_vllm_multinode_actor.add_lora",
            component="mint_vllm_multinode_actor",
            op="sampling.add_lora_from_path",
            request_id_arg="lora_name",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "lora_path": a.get("lora_path"),
            },
        )
        async def add_lora(
            self,
            lora_int_id: int,
            lora_path: str,
            lora_name: str,
            traceparent: str | None = None,
        ) -> None:
            """Add LoRA adapter from shared filesystem path.

            For multi-node: all workers must have access to the same path.
            Use shared filesystem (e.g., /vePFS-Mindverse/share/).

            Args:
                lora_int_id: Unique identifier for this LoRA adapter.
                lora_path: Path to PEFT adapter directory (must be on shared FS).
                lora_name: Human-readable name for the adapter.
            """
            self._bind_traceparent(traceparent)
            from vllm.lora.request import LoRARequest
            from .lora_utils import maybe_validate_peft_adapter_checkpoint_shapes

            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            )

            t0 = time.perf_counter()
            async with self._maybe_add_lora_idle_gate():
                async with self._lock_write():
                    t1 = time.perf_counter()
                    try:
                        fully_sharded_loras = (
                            _env_flag("MINT_VLLM_FULLY_SHARDED_LORAS", default=True)
                            and self.enable_lora
                            and self.max_lora_rank is not None
                            and self.tensor_parallel_size > 1
                            and self.max_lora_rank % self.tensor_parallel_size == 0
                        )
                        maybe_validate_peft_adapter_checkpoint_shapes(
                            lora_path,
                            self.model_path,
                            tensor_parallel_size=self.tensor_parallel_size,
                            fully_sharded_loras=fully_sharded_loras,
                        )
                        await self.engine.add_lora(lora_request)
                    except Exception:
                        import json
                        from pathlib import Path

                        adapter_dir = Path(lora_path)
                        summary: dict[str, object] = {
                            "lora_int_id": lora_int_id,
                            "lora_name": lora_name,
                            "lora_path": lora_path,
                            "exists": adapter_dir.exists(),
                            "is_dir": adapter_dir.is_dir(),
                        }
                        try:
                            if adapter_dir.is_dir():
                                entries = sorted(p.name for p in adapter_dir.iterdir())
                                summary["entries"] = entries[:20]
                                for fname in ("adapter_config.json", "adapter_model.safetensors"):
                                    p = adapter_dir / fname
                                    summary[f"{fname}_exists"] = p.exists()
                                    if p.exists():
                                        summary[f"{fname}_size_bytes"] = p.stat().st_size
                                cfg_path = adapter_dir / "adapter_config.json"
                                if cfg_path.exists():
                                    with cfg_path.open("r", encoding="utf-8") as f:
                                        cfg = json.load(f)
                                    summary["adapter_r"] = cfg.get("r")
                                    summary["target_modules"] = cfg.get("target_modules")
                                    summary["peft_type"] = cfg.get("peft_type")
                                    summary["base_model_name_or_path"] = cfg.get(
                                        "base_model_name_or_path"
                                    )
                        except Exception as summarize_e:
                            summary["summary_error"] = f"{type(summarize_e).__name__}: {summarize_e}"

                        logger.exception("vLLM add_lora failed; adapter_summary=%s", summary)
                        raise
            t2 = time.perf_counter()
            if self._timing:
                logger.info(
                    "[vLLM timing] add_lora id=%s lock_wait_s=%.3f engine_s=%.3f total_s=%.3f",
                    lora_int_id,
                    t1 - t0,
                    t2 - t1,
                    t2 - t0,
                )
            logger.info(f"Added LoRA {lora_name} (id={lora_int_id}) from {lora_path}")

        async def remove_lora(self, lora_int_id: int, traceparent: str | None = None) -> None:
            """Remove a LoRA adapter."""
            self._bind_traceparent(traceparent)
            t0 = time.perf_counter()
            async with self._exclusive_engine_op():
                async with self._lock_write():
                    t1 = time.perf_counter()
                    await self.engine.remove_lora(lora_int_id)
            t2 = time.perf_counter()
            if self._timing:
                logger.info(
                    "[vLLM timing] remove_lora id=%s lock_wait_s=%.3f engine_s=%.3f total_s=%.3f",
                    lora_int_id,
                    t1 - t0,
                    t2 - t1,
                    t2 - t0,
                )
            logger.info(f"Removed LoRA id={lora_int_id}")

        async def list_loras(self) -> set[int]:
            """List loaded LoRA adapter IDs."""
            # NOTE: list_loras must not race active generation on multinode.
            async with self._exclusive_engine_op():
                async with self._lock_read():
                    return await self.engine.list_loras()

        async def get_debug_env_info(self) -> dict:
            import os
            import sys
            import vllm

            try:
                from vllm.lora import lora_weights as lw  # type: ignore

                Packed = getattr(lw, "PackedLoRALayerWeights", None)
                LoRA = getattr(lw, "LoRALayerWeights", None)
                pack_moe_cm = Packed.__dict__.get("pack_moe") if Packed is not None else None
                pack_moe_orig = getattr(pack_moe_cm, "__func__", None)
                pack_moe_sparse_ok = bool(getattr(pack_moe_orig, "__mint_sparse_ok__", False))
                lora_opt_safe = bool(
                    getattr(getattr(LoRA, "optimize", None), "_mint_overlap_safe", False)
                )
                packed_opt_safe = bool(
                    getattr(getattr(Packed, "optimize", None), "_mint_overlap_safe", False)
                )
            except Exception as e:
                pack_moe_sparse_ok = False
                lora_opt_safe = False
                packed_opt_safe = False
                lora_patch_err = f"{type(e).__name__}: {e}"
            else:
                lora_patch_err = None

            return {
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "mint_enable_vllm_import_patches": os.environ.get(
                    "MINT_ENABLE_VLLM_IMPORT_PATCHES"
                ),
                "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
                "vllm_file": vllm.__file__,
                "vllm_lora_patch_error": lora_patch_err,
                "vllm_lora_pack_moe_sparse_ok": pack_moe_sparse_ok,
                "vllm_lora_opt_overlap_safe": lora_opt_safe,
                "vllm_lora_packed_opt_overlap_safe": packed_opt_safe,
                "sys_path_first_8": sys.path[:8],
            }

        async def get_kv_debug_info(self) -> dict:
            def _collect_kv_info(worker_wrapper):
                worker = getattr(worker_wrapper, "worker", None)
                target = worker if worker is not None else worker_wrapper
                model_runner = getattr(target, "model_runner", None)
                kv_cfg = getattr(model_runner, "kv_cache_config", None)
                groups = getattr(kv_cfg, "kv_cache_groups", []) or []
                block_sizes = []
                for group in groups:
                    spec = getattr(group, "kv_cache_spec", None)
                    block_size = getattr(spec, "block_size", 0) if spec is not None else 0
                    try:
                        if int(block_size) > 0:
                            block_sizes.append(int(block_size))
                    except (TypeError, ValueError):
                        pass
                if not block_sizes:
                    cache_config = getattr(target, "cache_config", None)
                    block_size = getattr(cache_config, "block_size", 0) if cache_config is not None else 0
                    try:
                        if int(block_size) > 0:
                            block_sizes.append(int(block_size))
                    except (TypeError, ValueError):
                        pass
                block_size = min(block_sizes) if block_sizes else 0
                num_blocks = int(getattr(kv_cfg, "num_blocks", 0) or 0) if kv_cfg is not None else 0
                return {
                    "available_kv_cache_memory_bytes": int(
                        getattr(target, "available_kv_cache_memory_bytes", 0) or 0
                    ),
                    "requested_memory_bytes": int(
                        getattr(target, "requested_memory", 0) or 0
                    ),
                    "non_torch_memory_bytes": int(
                        getattr(target, "non_torch_memory", 0) or 0
                    ),
                    "peak_activation_memory_bytes": int(
                        getattr(target, "peak_activation_memory", 0) or 0
                    ),
                    "kv_cache_num_blocks": num_blocks,
                    "kv_cache_block_size": block_size,
                    "kv_cache_capacity_tokens": num_blocks * block_size if num_blocks > 0 and block_size > 0 else 0,
                    "kv_cache_groups": len(groups),
                }

            infos = await self.engine.collective_rpc(_collect_kv_info)
            capacities = [int(x.get("kv_cache_capacity_tokens", 0) or 0) for x in infos]
            block_sizes = [int(x.get("kv_cache_block_size", 0) or 0) for x in infos]
            num_blocks = [int(x.get("kv_cache_num_blocks", 0) or 0) for x in infos]
            return {
                "per_worker": infos,
                "min_available_kv_cache_memory_bytes": min(
                    int(x.get("available_kv_cache_memory_bytes", 0) or 0) for x in infos
                ) if infos else 0,
                "max_available_kv_cache_memory_bytes": max(
                    int(x.get("available_kv_cache_memory_bytes", 0) or 0) for x in infos
                ) if infos else 0,
                "kv_cache_capacity_tokens": min((x for x in capacities if x > 0), default=0),
                "kv_cache_block_size": min((x for x in block_sizes if x > 0), default=0),
                "min_kv_cache_num_blocks": min((x for x in num_blocks if x > 0), default=0),
                "max_kv_cache_num_blocks": max(num_blocks) if num_blocks else 0,
            }

        async def abort_request(self, request_id: str, traceparent: str | None = None) -> None:
            """Abort an in-flight request in vLLM."""
            self._bind_traceparent(traceparent)
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

        @traced_async_from_traceparent(
            "sampling.mint_vllm_multinode_actor.generate",
            component="mint_vllm_multinode_actor",
            op="sampling.generate",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "prompt_tokens": len(a.get("prompt_ids") or []),
                "max_tokens": a.get("max_tokens"),
                "num_samples": a.get("n"),
            },
        )
        async def generate(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int | None,
            lora_path: str | None,
            max_tokens: int,
            outer_request_id: str | None = None,
            stop: object | None = None,
            temperature: float = 1.0,
            top_k: int = -1,
            top_p: float = 1.0,
            logprobs: bool = True,
            n: int = 1,
            traceparent: str | None = None,
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
            self._bind_traceparent(traceparent)
            import math

            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest
            from .vllm_stop import vllm_stop_kwargs

            outer_request_id = str(outer_request_id or request_id)
            n_req = max(1, int(n))
            if n_req > 1 and self._multisample_mode in ("sequential_n1", "concurrent_n1"):
                logger.info(
                    "mint_vllm_multinode_multisample_mode request_id=%s outer_request_id=%s mode=%s n=%s",
                    request_id,
                    outer_request_id,
                    self._multisample_mode,
                    n_req,
                )
                sub_ids = {f"{request_id}_s{i}" for i in range(n_req)}
                try:
                    async with self._outer_to_subreq_lock:
                        self._outer_to_subreq_ids[request_id] = sub_ids
                    async with self._progress_lock:
                        self._progress_total_by_outer[outer_request_id] = int(n_req)
                    if self._multisample_mode == "sequential_n1":
                        outs: list[dict] = []
                        for i in range(n_req):
                            sub_id = f"{request_id}_s{i}"
                            out = await self.generate(
                                prompt_ids=prompt_ids,
                                request_id=sub_id,
                                lora_int_id=lora_int_id,
                                lora_path=lora_path,
                                max_tokens=max_tokens,
                                outer_request_id=outer_request_id,
                                stop=stop,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                logprobs=logprobs,
                                n=1,
                                traceparent=traceparent,
                            )
                            assert isinstance(out, dict)
                            outs.append(out)
                        return outs

                    tasks: dict[str, asyncio.Task] = {}
                    for i in range(n_req):
                        sub_id = f"{request_id}_s{i}"
                        tasks[sub_id] = asyncio.create_task(
                            self.generate(
                                prompt_ids=prompt_ids,
                                request_id=sub_id,
                                lora_int_id=lora_int_id,
                                lora_path=lora_path,
                                max_tokens=max_tokens,
                                outer_request_id=outer_request_id,
                                stop=stop,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                logprobs=logprobs,
                                n=1,
                                traceparent=traceparent,
                            )
                        )
                    try:
                        outs_raw = await asyncio.gather(*tasks.values())
                    except Exception:
                        # One subrequest failed. Best-effort abort the remaining in-flight
                        # subrequests while the outer->subrequest mapping is still present,
                        # then await task cleanup before dropping the mapping in finally.
                        try:
                            await self.abort_request(request_id)
                        except Exception:
                            pass
                        for task in tasks.values():
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks.values(), return_exceptions=True)
                        raise
                    outs: list[dict] = []
                    for out in outs_raw:
                        assert isinstance(out, dict)
                        outs.append(out)
                    return outs
                finally:
                    async with self._outer_to_subreq_lock:
                        self._outer_to_subreq_ids.pop(request_id, None)
                    await self._clear_progress(outer_request_id)
            if n_req > 1:
                logger.info(
                    "mint_vllm_multinode_multisample_mode request_id=%s outer_request_id=%s mode=%s n=%s",
                    request_id,
                    outer_request_id,
                    self._multisample_mode,
                    n_req,
                )

            effective_max_tokens = int(max_tokens)
            if self.max_model_len is not None:
                effective_max_tokens = min(effective_max_tokens, int(self.max_model_len) - len(prompt_ids))
            if effective_max_tokens < 1:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids):,} tokens) exceeds model context limit "
                    f"({self.max_model_len:,} tokens). Reduce prompt or use a model with larger context."
                )

            sampling_params = SamplingParams(
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                # vLLM multi-node engine returns `None` logprob entries when logprobs=0.
                # Use a positive value so per-token logprobs are populated.
                logprobs=1 if logprobs else None,
                n=n_req,
                **vllm_stop_kwargs(stop, default_stop_token_ids=[151645, 151643, 163586, 163585]),
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
            generate_lock_wait_s = 0.0
            seq_slot_wait_s = 0.0
            engine_read_lock_wait_s = 0.0
            add_request_wait_s = 0.0
            add_request_exec_s = 0.0
            try:
                from .task_state_store import task_futures

                await task_futures.async_update_meta(
                    outer_request_id,
                    meta={
                        "stage": "prefill",
                        "progress": None,
                        "max_tokens": int(effective_max_tokens),
                    },
                )
            except Exception as e:
                logger.warning(
                    "mint_vllm_multinode_prefill_meta_update_failed request_id=%s err=%s: %s",
                    outer_request_id,
                    type(e).__name__,
                    e,
                )
            first_tok_s: float | None = None
            reserved_seq_slots = 0
            generate_registered = False
            # Get final response
            t_generate_lock_start = time.perf_counter()
            async with self._maybe_generate_lock():
                t_after_generate_lock = time.perf_counter()
                generate_lock_wait_s = t_after_generate_lock - t_generate_lock_start
                with start_as_current_span(
                    "sampling.mint_vllm_multinode_actor.seq_slot_wait",
                    component="mint_vllm_multinode_actor",
                    op="sampling.seq_slot_wait",
                    request_id=request_id,
                    attributes={
                        "prompt_tokens": len(prompt_ids),
                        "num_samples": n_req,
                        "max_tokens": effective_max_tokens,
                    },
                ) as seq_slot_span:
                    t_seq_slot_start = time.perf_counter()
                    reserved_seq_slots = await self._acquire_seq_slots(n_req)
                    seq_slot_wait_s = time.perf_counter() - t_seq_slot_start
                    if seq_slot_span is not None:
                        seq_slot_span.set_attribute("wait_s", float(seq_slot_wait_s))
                try:
                    await self._register_generate_start()
                    generate_registered = True
                    t_read_lock_start = time.perf_counter()
                    async with self._lock_read():
                        engine_read_lock_wait_s = time.perf_counter() - t_read_lock_start

                        async def _enqueue_request(*, multisample_lock_already_held: bool) -> Any:
                            nonlocal add_request_wait_s, add_request_exec_s
                            # vLLM's AsyncLLMEngine.generate() is an async generator whose
                            # first `__anext__()` both enqueues the request (add_request)
                            # and waits for the first engine output. Serializing that
                            # `__anext__()` across concurrent requests destroys continuous
                            # batching for long prompts.
                            #
                            # Keep serialization scoped to the enqueue window that is known to
                            # be unsafe. Preserve native `vllm_n`, but do not hold the
                            # multisample-specific lock across the whole decode path.
                            with start_as_current_span(
                                "sampling.mint_vllm_multinode_actor.add_request",
                                component="mint_vllm_multinode_actor",
                                op="sampling.vllm_add_request",
                                request_id=request_id,
                                attributes={
                                    "prompt_tokens": len(prompt_ids),
                                    "num_samples": n_req,
                                    "max_tokens": effective_max_tokens,
                                },
                            ) as add_request_span:
                                t_add_request_start = time.perf_counter()
                                try:
                                    async with AsyncExitStack() as stack:
                                        if (
                                            not multisample_lock_already_held
                                            and n_req > 1
                                            and self._multisample_lock is not None
                                        ):
                                            await stack.enter_async_context(self._maybe_multisample_lock(n_req))
                                        if self._add_request_lock is not None:
                                            await stack.enter_async_context(self._maybe_add_request_lock())
                                        t_before_add_request = time.perf_counter()
                                        add_request_wait_s = t_before_add_request - t_add_request_start
                                        result = await self.engine.add_request(
                                            request_id=request_id,
                                            prompt=prompt,
                                            params=sampling_params,
                                            lora_request=lora_request,
                                        )
                                        add_request_exec_s = time.perf_counter() - t_before_add_request
                                        if add_request_span is not None:
                                            add_request_span.set_attribute("wait_s", float(add_request_wait_s))
                                            add_request_span.set_attribute("exec_s", float(add_request_exec_s))
                                        return result
                                except Exception:
                                    _raise_serializable_vllm_error(
                                        request_id=request_id,
                                        where="vllm_add_request_failed",
                                        extra={
                                            "prompt_len": len(prompt_ids),
                                            "max_tokens": effective_max_tokens,
                                            "n": n_req,
                                            "model_path": self.model_path,
                                            "tp": self.tensor_parallel_size,
                                            "pp": self.pipeline_parallel_size,
                                        },
                                    )

                        final_res = None
                        by_index: dict[int, Any] | None = {} if n_req > 1 else None
                        deadline = None
                        if self._generate_timeout_s > 0:
                            deadline = time.perf_counter() + self._generate_timeout_s
                        try:
                            collector = await _enqueue_request(
                                multisample_lock_already_held=False,
                            )
                            while True:
                                try:
                                    if deadline is None:
                                        remaining = None
                                    else:
                                        remaining = deadline - time.perf_counter()
                                        if remaining <= 0:
                                            raise asyncio.TimeoutError()

                                    if remaining is None:
                                        out = collector.get_nowait() or await collector.get()
                                    else:
                                        out = collector.get_nowait() or await asyncio.wait_for(
                                            collector.get(),
                                            timeout=remaining,
                                        )
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
                                try:
                                    if n_req == 1:
                                        tokens_generated = len(out.outputs[0].token_ids)
                                    else:
                                        lengths = [len(oo.token_ids) for oo in out.outputs]
                                        tokens_generated = min(lengths) if lengths else 0
                                    sub_request_id = None if outer_request_id == request_id else request_id
                                    await self._update_progress(
                                        outer_request_id=outer_request_id,
                                        sub_request_id=sub_request_id,
                                        tokens_generated=tokens_generated,
                                        max_tokens=effective_max_tokens,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "mint_vllm_multinode_progress_compute_failed request_id=%s err=%s: %s",
                                        outer_request_id,
                                        type(e).__name__,
                                        e,
                                    )
                                if out.finished:
                                    break
                        except Exception:
                            _raise_serializable_vllm_engine_error(
                                request_id=request_id,
                                where="vllm_generate_collect_failed",
                                extra={
                                    "prompt_len": len(prompt_ids),
                                    "max_tokens": effective_max_tokens,
                                    "n": n_req,
                                    "model_path": self.model_path,
                                    "tp": self.tensor_parallel_size,
                                    "pp": self.pipeline_parallel_size,
                                },
                            )
                    assert final_res is not None
                finally:
                    try:
                        if generate_registered:
                            await self._register_generate_end()
                    finally:
                        try:
                            await self._release_seq_slots(reserved_seq_slots)
                        finally:
                            if self._post_generate_delay_s > 0:
                                await asyncio.sleep(self._post_generate_delay_s)
            t2 = time.perf_counter()
            total_s = t2 - t0
            self._vllm_stats_observer.observe_actor_timing(
                seq_slot_wait_s=seq_slot_wait_s,
                generate_lock_wait_s=generate_lock_wait_s,
                engine_read_lock_wait_s=engine_read_lock_wait_s,
                add_request_wait_s=add_request_wait_s,
                add_request_exec_s=add_request_exec_s,
                first_token_observed_s=first_tok_s,
            )
            try:
                slow_log_threshold_s = float(os.environ.get("MINT_VLLM_SLOW_REQUEST_LOG_THRESHOLD_S", "30"))
            except ValueError:
                slow_log_threshold_s = 30.0
            if total_s >= slow_log_threshold_s or n_req > 1:
                logger.info(
                    "mint_vllm_multinode_generate_summary request_id=%s outer_request_id=%s prompt_len=%s max_tokens=%s n=%s total_s=%.3f first_tok_s=%s generate_lock_wait_s=%.3f seq_slot_wait_s=%.3f engine_read_lock_wait_s=%.3f add_request_wait_s=%.3f add_request_exec_s=%.3f",
                    request_id,
                    outer_request_id,
                    len(prompt_ids),
                    effective_max_tokens,
                    n_req,
                    total_s,
                    None if first_tok_s is None else round(float(first_tok_s), 3),
                    generate_lock_wait_s,
                    seq_slot_wait_s,
                    engine_read_lock_wait_s,
                    add_request_wait_s,
                    add_request_exec_s,
                )
            if self._timing:
                logger.info(
                    "[vLLM timing] generate req=%s prompt_len=%s max_tokens=%s lora_id=%s "
                    "generate_lock_wait_s=%.3f seq_slot_wait_s=%.3f engine_read_lock_wait_s=%.3f "
                    "add_request_wait_s=%.3f add_request_exec_s=%.3f total_s=%.3f first_tok_s=%s",
                    request_id,
                    len(prompt_ids),
                    max_tokens,
                    lora_int_id,
                    generate_lock_wait_s,
                    seq_slot_wait_s,
                    engine_read_lock_wait_s,
                    add_request_wait_s,
                    add_request_exec_s,
                    total_s,
                    first_tok_s,
                )

            if n_req == 1:
                token_ids = list(final_res.outputs[0].token_ids)  # type: ignore[union-attr]
                log_probs = None
                if sampling_params.logprobs is not None and final_res.outputs[0].logprobs:  # type: ignore[union-attr]
                    log_probs = []
                    non_finite_count = 0
                    non_finite_samples: list[tuple[int, int, float]] = []
                    for i, lps in enumerate(final_res.outputs[0].logprobs):  # type: ignore[union-attr]
                        tid = token_ids[i]
                        getter = getattr(lps, "get", None)
                        lp_obj = getter(tid) if callable(getter) else None
                        if lp_obj is None:
                            raise RuntimeError(
                                f"vLLM missing sampled-token logprob: request_id={request_id} idx={i} token_id={tid}"
                            )
                        if isinstance(lp_obj, (float, int)):
                            lp_f = float(lp_obj)
                            if not math.isfinite(lp_f):
                                non_finite_count += 1
                                if len(non_finite_samples) < 3:
                                    non_finite_samples.append((i, tid, lp_f))
                            log_probs.append(lp_f)
                            continue
                        lp_val = getattr(lp_obj, "logprob", None)
                        if lp_val is None and isinstance(lp_obj, dict):
                            lp_val = lp_obj.get("logprob")
                        if lp_val is None:
                            raise RuntimeError(
                                f"vLLM returned None sampled-token logprob: request_id={request_id} idx={i} token_id={tid}"
                            )
                        lp_f = float(lp_val)
                        if not math.isfinite(lp_f):
                            non_finite_count += 1
                            if len(non_finite_samples) < 3:
                                non_finite_samples.append((i, tid, lp_f))
                        log_probs.append(lp_f)

                    if non_finite_count:
                        token_preview = {
                            "head": token_ids[:8],
                            "tail": token_ids[-8:] if len(token_ids) > 8 else token_ids[:],
                        }
                        raise RuntimeError(
                            f"Non-finite sampled-token logprobs: request_id={request_id} "
                            f"count={non_finite_count} samples(idx,token,lp)={non_finite_samples} "
                            f"token_preview={token_preview}"
                        )
                routed_experts = None
                raw_re = getattr(final_res.outputs[0], "routed_experts", None)  # type: ignore[union-attr]
                if server_config.router_replay_mode == "R3" and raw_re is None:
                    raise RuntimeError(
                        "router_replay_mode=R3 but vLLM returned no routed_experts for single-output generate"
                    )
                if raw_re is not None:
                    routed_experts = raw_re.tolist() if hasattr(raw_re, "tolist") else raw_re

                # Determine stop reason
                stop_reason = "length"
                if final_res.outputs[0].finish_reason == "stop":  # type: ignore[union-attr]
                    stop_reason = "stop"
                elif any(tid in [151645, 151643, 163586, 163585] for tid in token_ids[-3:]):
                    stop_reason = "stop"

                result = {
                    "token_ids": token_ids,
                    "logprobs": log_probs,
                    "stop_reason": stop_reason,
                    "routed_experts": routed_experts,
                    "_timing_total_s": float(t2 - t0),
                    "_timing_first_tok_s": float(first_tok_s) if first_tok_s is not None else None,
                }
                if outer_request_id == request_id:
                    await self._clear_progress(outer_request_id)
                return result

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
            non_finite_count = 0
            non_finite_samples: list[tuple[int, int, int, float]] = []
            affected_seq_preview: dict[int, dict[str, list[int]]] = {}
            for seq_i, out in enumerate(outs):
                out_token_ids = list(out.token_ids)
                out_log_probs = None
                if sampling_params.logprobs is not None and out.logprobs:
                    out_log_probs = []
                    for i, lps in enumerate(out.logprobs):
                        tid = out_token_ids[i]
                        getter = getattr(lps, "get", None)
                        lp_obj = getter(tid) if callable(getter) else None
                        if lp_obj is None:
                            raise RuntimeError(
                                f"vLLM missing sampled-token logprob: request_id={request_id} idx={i} token_id={tid}"
                            )
                        if isinstance(lp_obj, (float, int)):
                            lp_f = float(lp_obj)
                            if not math.isfinite(lp_f):
                                non_finite_count += 1
                                if len(non_finite_samples) < 3:
                                    non_finite_samples.append((seq_i, i, tid, lp_f))
                                if seq_i not in affected_seq_preview and len(affected_seq_preview) < 3:
                                    affected_seq_preview[seq_i] = {
                                        "head": out_token_ids[:8],
                                        "tail": out_token_ids[-8:] if len(out_token_ids) > 8 else out_token_ids[:],
                                    }
                            out_log_probs.append(lp_f)
                            continue
                        lp_val = getattr(lp_obj, "logprob", None)
                        if lp_val is None and isinstance(lp_obj, dict):
                            lp_val = lp_obj.get("logprob")
                        if lp_val is None:
                            raise RuntimeError(
                                f"vLLM returned None sampled-token logprob: request_id={request_id} idx={i} token_id={tid}"
                            )
                        lp_f = float(lp_val)
                        if not math.isfinite(lp_f):
                            non_finite_count += 1
                            if len(non_finite_samples) < 3:
                                non_finite_samples.append((seq_i, i, tid, lp_f))
                            if seq_i not in affected_seq_preview and len(affected_seq_preview) < 3:
                                affected_seq_preview[seq_i] = {
                                    "head": out_token_ids[:8],
                                    "tail": out_token_ids[-8:] if len(out_token_ids) > 8 else out_token_ids[:],
                                }
                        out_log_probs.append(lp_f)
                out_routed_experts = None
                raw_re = getattr(out, "routed_experts", None)
                if server_config.router_replay_mode == "R3" and raw_re is None:
                    raise RuntimeError(
                        "router_replay_mode=R3 but vLLM returned no routed_experts for multi-output generate"
                    )
                if raw_re is not None:
                    out_routed_experts = raw_re.tolist() if hasattr(raw_re, "tolist") else raw_re

                out_stop_reason = "length"
                if out.finish_reason == "stop":
                    out_stop_reason = "stop"
                elif any(tid in [151645, 151643, 163586, 163585] for tid in out_token_ids[-3:]):
                    out_stop_reason = "stop"

                multi_results.append(
                    {
                        "token_ids": out_token_ids,
                        "logprobs": out_log_probs,
                        "stop_reason": out_stop_reason,
                        "routed_experts": out_routed_experts,
                        "_timing_total_s": float(t2 - t0),
                        "_timing_first_tok_s": float(first_tok_s) if first_tok_s is not None else None,
                            }
                )

            if non_finite_count:
                raise RuntimeError(
                    f"Non-finite sampled-token logprobs: request_id={request_id} "
                    f"count={non_finite_count} samples(seq,idx,token,lp)={non_finite_samples} "
                    f"seq_token_preview={affected_seq_preview}"
                )

            if outer_request_id == request_id:
                await self._clear_progress(outer_request_id)
            return multi_results

        @traced_async_from_traceparent(
            "sampling.mint_vllm_multinode_actor.compute_prompt_logprobs",
            component="mint_vllm_multinode_actor",
            op="sampling.compute_prompt_logprobs",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "prompt_tokens": len(a.get("prompt_ids") or []),
            },
        )
        async def compute_prompt_logprobs(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int | None,
            lora_path: str | None,
            traceparent: str | None = None,
        ) -> list[float | None]:
            """Compute logprobs for prompt tokens.

            Returns a list of length len(prompt_ids), where:
            - logprobs[0] is None (first token has no conditioning context)
            - logprobs[i] = log P(token[i] | token[0:i]) for i >= 1
            """
            self._bind_traceparent(traceparent)
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
            async with self._exclusive_engine_op():
                async with self._maybe_prompt_logprobs_lock():
                    async with self._lock_read():
                        t1 = time.perf_counter()
                        # Prompt-logprobs requests explicitly skip prefix-cache reads in vLLM.
                        # After a long sample finishes, its prompt KV blocks can remain cached and
                        # occupy the KV pool while being unusable for the follow-on prompt-logprobs
                        # request. Clearing prefix cache here frees those cached blocks without
                        # changing prompt-logprobs semantics for this request.
                        await self.engine.reset_prefix_cache()
                        try:
                            collector = await self.engine.add_request(
                                request_id=request_id,
                                prompt=prompt,
                                params=sampling_params,
                                lora_request=lora_request,
                            )
                        except Exception:
                            _raise_serializable_vllm_error(
                                request_id=request_id,
                                where="vllm_prompt_logprobs_add_request_failed",
                                extra={
                                    "prompt_len": len(prompt_ids),
                                    "model_path": self.model_path,
                                    "tp": self.tensor_parallel_size,
                                    "pp": self.pipeline_parallel_size,
                                },
                            )
                        final_res = None
                        try:
                            while True:
                                out = await collector.get()
                                final_res = out
                                if out.finished:
                                    break
                        except Exception:
                            _raise_serializable_vllm_engine_error(
                                request_id=request_id,
                                where="vllm_prompt_logprobs_collect_failed",
                                extra={
                                    "prompt_len": len(prompt_ids),
                                    "model_path": self.model_path,
                                    "tp": self.tensor_parallel_size,
                                    "pp": self.pipeline_parallel_size,
                                },
                            )
                        assert final_res is not None
            t2 = time.perf_counter()
            if self._timing:
                logger.info(
                    "[vLLM timing] prompt_logprobs req=%s prompt_len=%s lora_id=%s lock_wait_s=%.3f total_s=%.3f",
                    request_id,
                    len(prompt_ids),
                    lora_int_id,
                    t1 - t0,
                    t2 - t0,
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

        @traced_async_from_traceparent(
            "sampling.mint_vllm_multinode_actor.compute_prompt_topk",
            component="mint_vllm_multinode_actor",
            op="sampling.compute_prompt_topk",
            request_id_arg="request_id",
            attributes_builder=lambda a: {
                "lora_int_id": a.get("lora_int_id"),
                "prompt_tokens": len(a.get("prompt_ids") or []),
                "topk": a.get("k"),
            },
        )
        async def compute_prompt_topk(
            self,
            prompt_ids: list[int],
            request_id: str,
            lora_int_id: int | None,
            lora_path: str | None,
            k: int,
            traceparent: str | None = None,
        ) -> list[list[tuple[int, float]] | None]:
            """Compute top-K prompt logprobs.

            Returns a list of length len(prompt_ids), where:
            - topk[0] is None (first token has no conditioning context)
            - topk[i] is a list of (token_id, logprob) pairs for i >= 1
            """
            self._bind_traceparent(traceparent)
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            from vllm.lora.request import LoRARequest

            if not prompt_ids:
                return []
            if len(prompt_ids) == 1:
                return [None]

            kk = int(k)
            if kk <= 0:
                return [None] * len(prompt_ids)

            sampling_params = SamplingParams(
                max_tokens=1,
                prompt_logprobs=kk,
                temperature=1.0,
            )

            prompt = TokensPrompt(prompt_token_ids=prompt_ids)

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
                    try:
                        collector = await self.engine.add_request(
                            request_id=request_id,
                            prompt=prompt,
                            params=sampling_params,
                            lora_request=lora_request,
                        )
                    except Exception:
                        _raise_serializable_vllm_error(
                            request_id=request_id,
                            where="vllm_prompt_topk_add_request_failed",
                            extra={
                                "prompt_len": len(prompt_ids),
                                "k": kk,
                                "model_path": self.model_path,
                                "tp": self.tensor_parallel_size,
                                "pp": self.pipeline_parallel_size,
                            },
                        )
                    final_res = None
                    try:
                        while True:
                            out = await collector.get()
                            final_res = out
                            if out.finished:
                                break
                    except Exception:
                        _raise_serializable_vllm_engine_error(
                            request_id=request_id,
                            where="vllm_prompt_topk_collect_failed",
                            extra={
                                "prompt_len": len(prompt_ids),
                                "k": kk,
                                "model_path": self.model_path,
                                "tp": self.tensor_parallel_size,
                                "pp": self.pipeline_parallel_size,
                            },
                        )
                    assert final_res is not None
            t2 = time.perf_counter()
            if self._timing:
                logger.info(
                    "[vLLM timing] prompt_topk req=%s prompt_len=%s k=%s lora_id=%s lock_wait_s=%.3f total_s=%.3f",
                    request_id,
                    len(prompt_ids),
                    kk,
                    lora_int_id,
                    t1 - t0,
                    t2 - t0,
                )

            prompt_logprobs = final_res.prompt_logprobs
            if prompt_logprobs is None:
                return [None] * len(prompt_ids)

            out: list[list[tuple[int, float]] | None] = [None]
            for i in range(1, len(prompt_ids)):
                if i >= len(prompt_logprobs) or prompt_logprobs[i] is None:
                    out.append(None)
                    continue
                entry = prompt_logprobs[i]
                items = getattr(entry, "items", None)
                if not callable(items):
                    out.append(None)
                    continue

                pairs: list[tuple[int, float]] = []
                for tok, lp_obj in entry.items():
                    if isinstance(lp_obj, (float, int)):
                        lp_val = float(lp_obj)
                    else:
                        lp_val = getattr(lp_obj, "logprob", None)
                        if lp_val is None and isinstance(lp_obj, dict):
                            lp_val = lp_obj.get("logprob")
                        if lp_val is None:
                            continue
                        lp_val = float(lp_val)
                    pairs.append((int(tok), float(lp_val)))

                pairs.sort(key=lambda kv: kv[1], reverse=True)
                out.append(pairs[:kk])

            return out

    return MultiNodeVLLMEngine


def _model_to_mint_vllm_actor_name(model_name: str) -> str:
    model_part = model_name.split("/")[-1] if "/" in model_name else model_name
    return f"mint_vllm_{model_part.lower().replace(' ', '_')}"


@dataclass
class GenerateResult:
    """Result of a generate call."""

    token_ids: list[int]
    logprobs: list[float] | None = None
    stop_reason: str | None = None
    routed_experts: list | None = None


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
        shared_adapter_dir: str = "/vePFS-Mindverse/share/mint/adapters",
        distributed_executor_backend: str = "ray",
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
        self.actor_name = actor_name or _model_to_mint_vllm_actor_name(model_name)
        self.shared_adapter_dir = shared_adapter_dir
        self.distributed_executor_backend = distributed_executor_backend.strip().lower()
        if self.distributed_executor_backend not in ("ray", "mp"):
            raise ValueError(
                f"distributed_executor_backend must be one of: 'ray', 'mp' (got {distributed_executor_backend!r})"
            )

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
                init_ray(
                    namespace=PERSISTENT_NAMESPACE,
                    ignore_reinit_error=True,
                )

            # MultiNodeVLLMEngine itself does not need Ray GPU resources (vLLM's Ray backend
            # manages the 1-GPU worker actors). Reserving an extra GPU for the controller
            # breaks TP=16 scheduling on 2x8-GPU clusters (would require 17 GPUs).
            worker_gpus = (
                self.tensor_parallel_size
                * self.pipeline_parallel_size
                * self.data_parallel_size
            )
            distributed_executor_backend = self.distributed_executor_backend
            mp_pinned_node_ip: str | None = None
            if distributed_executor_backend == "mp":
                # vLLM runs locally (single-node) inside this Ray actor and needs direct GPU access.
                # Reserve all GPUs on a single node for this actor. vLLM will spawn local processes
                # for TP and communicate via NCCL (no Ray compiled DAG).
                controller_gpus = int(worker_gpus)
                controller_cpus = 1
                total_required_gpus = int(worker_gpus)
                resources = None
            else:
                resources = compute_multinode_engine_resources(
                    worker_gpus,
                    preferred_node_ips=[],
                )
                controller_gpus = resources.controller_gpus
                controller_cpus = resources.controller_cpus
                total_required_gpus = resources.total_required_gpus
            ray_cgraph_get_timeout = (
                os.environ.get("RAY_CGRAPH_get_timeout")
                or os.environ.get("MINT_RAY_CGRAPH_GET_TIMEOUT_S")
                or "1800"
            )
            logger.info(f"mint_vllm_multinode_init actor={self.actor_name} RAY_CGRAPH_get_timeout={ray_cgraph_get_timeout}")

            # Large models can spend 10-30+ minutes loading shards across many GPUs.
            if total_required_gpus >= 16:
                init_timeout = 3600
            elif total_required_gpus >= 8:
                init_timeout = 1800
            elif total_required_gpus >= 4:
                init_timeout = 1800
            else:
                init_timeout = 600

            from .model_registry import is_topology_desired_model

            is_topology_desired = bool(self.model_name and is_topology_desired_model(self.model_name))

            def _attach_existing_actor(existing_actor_handle) -> None:
                self.engine = existing_actor_handle
                self._initialized = True
                from mint_server.backend.model_actor_inventory import ActorType
                from mint_server.backend.model_actor_publication import (
                    BackendModelActorLaunch,
                    publish_backend_model_actor,
                )

                publish_backend_model_actor(BackendModelActorLaunch(
                    actor_name=self.actor_name,
                    actor_type=ActorType.VLLM,
                    num_gpus=total_required_gpus,
                    actor_handle=self.engine,
                    namespace=PERSISTENT_NAMESPACE,
                    base_model=self.model_path,
                    protected=is_topology_desired,
                ))

            # Try to connect to existing actor
            existing_actor = None
            try:
                existing_actor = ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                try:
                    is_ready = await async_get_ray_ref(existing_actor.is_ready.remote(), timeout_s=30)
                except ray.exceptions.GetTimeoutError:
                    # A timed-out readiness probe means the actor exists but its event loop is
                    # currently occupied. That happens both during long initialization and while
                    # serving long requests. Reusing the detached actor lets later calls queue
                    # behind the actor instead of wedging this session on initialize().
                    logger.warning(
                        f"ray.get(is_ready) timed out for {self.actor_name}; assuming existing actor is busy and reusing it"
                    )
                    _attach_existing_actor(existing_actor)
                    return
                except ray.exceptions.RayTaskError as e:
                    logger.warning(
                        f"ray.get(is_ready) failed for {self.actor_name}: {type(e).__name__}: {e}; treating as not-ready"
                    )
                    is_ready = False
                except SystemExit as e:
                    if getattr(e, "code", None) == 15:
                        raise
                    logger.warning(
                        f"ray.get(is_ready) triggered SystemExit for {self.actor_name}: {e}; treating as not-ready"
                    )
                    is_ready = False
                if is_ready:
                    logger.info(f"Connected to existing MultiNodeVLLMEngine: {self.actor_name}")
                    _attach_existing_actor(existing_actor)
                    return
                else:
                    # vLLM engine initialization can take a long time. If an actor exists but is not
                    # ready, assume it is still initializing and wait, rather than killing and
                    # recreating (which can lead to repeated Volcano placement selection and
                    # "insufficient free nodes" during initialization).
                    logger.info(
                        f"Actor {self.actor_name} exists but not ready; waiting for initialize (timeout={init_timeout}s)"
                    )
                    try:
                        await async_get_ray_ref(existing_actor.initialize.remote(), timeout_s=init_timeout)
                    except ray.exceptions.GetTimeoutError:
                        logger.warning(
                            f"Actor {self.actor_name} initialize timed out after {init_timeout}s; will recreate"
                        )
                    except ray.exceptions.RayTaskError as e:
                        logger.warning(
                            f"ray.get(initialize) failed for {self.actor_name}: {type(e).__name__}: {e}; will recreate"
                        )
                    except SystemExit as e:
                        if getattr(e, "code", None) == 15:
                            raise
                        logger.warning(
                            f"ray.get(initialize) triggered SystemExit for {self.actor_name}: {e}; will recreate"
                        )
                    else:
                        logger.info(f"Connected to existing MultiNodeVLLMEngine after init: {self.actor_name}")
                        _attach_existing_actor(existing_actor)
                        return
            except (ValueError, ray.exceptions.RayActorError):
                logger.info(f"No existing actor found, creating new: {self.actor_name}")
                try:
                    stale_pg = get_named_placement_group(
                        f"{self.actor_name}_pg",
                        namespace=PERSISTENT_NAMESPACE,
                    )
                except Exception:
                    stale_pg = None
                if stale_pg is not None:
                    logger.warning(
                        "Removing stale placement group for actor_name=%s before recreation",
                        self.actor_name,
                    )
                    try:
                        ray.util.remove_placement_group(stale_pg)
                    except Exception as e:
                        logger.warning(
                            "Failed removing stale placement group for actor_name=%s: %s",
                            self.actor_name,
                            e,
                        )

            # Kill existing actor if any before creating new
            if existing_actor is not None:
                try:
                    ray_kill.kill(
                        existing_actor,
                        reason="mint_vllm_multinode_recreate",
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
                    for _ in range(10):
                        await asyncio.sleep(1)
                        try:
                            ray.get_actor(self.actor_name, namespace=PERSISTENT_NAMESPACE)
                        except ValueError:
                            break  # Actor name is available
                except Exception as e:
                    logger.warning(f"Error killing actor {self.actor_name}: {e}")

            node_ips: list[str] | None = None

            # Preferred node pinning takes precedence over queue-based selection.
            preferred_placement = _model_gpu_placement_for_model(self.model_name)
            if preferred_placement is not None:
                if preferred_placement.total_gpus != int(worker_gpus):
                    raise RuntimeError(
                        f"MINT_MODEL_PLACEMENT_JSON GPU count mismatch for model={self.model_name!r}: "
                        f"need {worker_gpus} GPUs, got {preferred_placement.total_gpus}"
                )
                node_ips = preferred_placement.node_ips
                logger.info(
                    "[MultiNodeInferenceEngine] Using pinned placement for model=%s slices=%s",
                    self.model_name,
                    [
                        {
                            "replica": slice_.replica,
                            "worker_index": slice_.worker_index,
                            "gpu_count": slice_.gpu_count,
                            "node_ip": slice_.node_ip,
                        }
                        for slice_ in preferred_placement.slices
                    ],
                )
                if distributed_executor_backend == "mp":
                    if len(preferred_placement.slices) != 1:
                        raise RuntimeError(
                            f"mp vLLM requires exactly 1 pinned worker slice, got slices={len(preferred_placement.slices)} "
                            f"for model={self.model_name!r}"
                        )
                    mp_pinned_node_ip = node_ips[0]
                    logger.info(
                        f"[MultiNodeInferenceEngine] mp pin model={self.model_name} node={mp_pinned_node_ip}"
                    )
                else:
                    pg_name = f"{self.actor_name}_pg"
                    assert_node_ip_capacity(
                        required_gpus_by_node_ip=preferred_placement.required_gpus_by_node_ip(),
                        context=f"mint_vllm_multinode_node_pin model={self.model_name}",
                        ignore_placement_group_names={pg_name},
                        ignore_placement_group_namespace=PERSISTENT_NAMESPACE,
                    )
                    pg_bundles = preferred_placement.pg_bundles()
                    if resources is None:
                        raise RuntimeError("internal error: Ray vLLM placement resources are not initialized")
                    resources = MultiNodeEngineResources(
                        worker_gpus=resources.worker_gpus,
                        controller_gpus=resources.controller_gpus,
                        controller_cpus=resources.controller_cpus,
                        total_required_gpus=resources.total_required_gpus,
                        pg_bundles=pg_bundles + [resources.pg_bundles[-1]],
                        controller_bundle_index=int(worker_gpus),
                    )
            # Ensure shared adapter directory exists
            os.makedirs(self.shared_adapter_dir, exist_ok=True)

            from mint_server.backend.model_actor_inventory import ActorType
            from mint_server.backend.model_actor_publication import (
                BackendModelActorLaunch,
                publish_backend_model_actor,
            )
            logger.info(
                f"Creating multi-node vLLM requiring {total_required_gpus} GPUs "
                f"(TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, "
                f"DP={self.data_parallel_size}, expert_parallel={self.enable_expert_parallel}, "
                f"controller_gpus={controller_gpus}, worker_gpus={worker_gpus})"
            )

            # Step 1: Create a detached placement group and capture child tasks.
            #
            # vLLM's Ray backend spawns 1-GPU RayWorkerWrapper actors. Without a placement group,
            # those workers can collide with Megatron placement groups, leading to vLLM init failures
            # like "Free memory on device ... is less than desired GPU memory utilization".
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            pg = None
            if distributed_executor_backend == "ray":
                pg_name = f"{self.actor_name}_pg"
                pg_bundles = resources.pg_bundles
                try:
                    pg = get_named_placement_group(
                        pg_name,
                        namespace=PERSISTENT_NAMESPACE,
                        expected_bundles=pg_bundles,
                    )
                except PlacementGroupMismatchError as e:
                    logger.warning(
                        "Removing incompatible placement group for actor_name=%s: %s",
                        self.actor_name,
                        e,
                    )
                    ray.util.remove_placement_group(e.pg)
                    pg = ray.util.placement_group(
                        pg_bundles,
                        # PACK to minimize fragmentation: multi-node vLLM uses many 1-GPU workers.
                        # SPREAD can occupy 1-3 GPUs on every node, preventing later 4-GPU actors
                        # (e.g., Qwen3-30B) from finding a node with 4 free GPUs.
                        strategy="PACK",
                        name=pg_name,
                        lifetime="detached",
                    )
                except Exception:
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
                    await async_get_ray_ref(pg.ready())
                except SystemExit as e:
                    if getattr(e, "code", None) == 15:
                        raise
                    try:
                        if pg is not None:
                            ray.util.remove_placement_group(pg)
                    except Exception:
                        pass
                    raise RuntimeError(f"ray.get(pg.ready()) triggered SystemExit for {pg_name}: {e}") from e

            # Create new engine actor
            MultiNodeVLLMEngine = _create_mint_vllm_multinode_actor(
                max_loras=self.max_loras,
                max_cpu_loras=self.max_cpu_loras,
                max_lora_rank=self.max_lora_rank,
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
            )

            if distributed_executor_backend == "ray":
                scheduling_opts = {
                    "scheduling_strategy": PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        # vLLM's Ray backend places worker ranks into bundles [0..TP-1] by index.
                        # Place the controller into a CPU-only bundle to avoid reserving an extra GPU
                        # while keeping child task capture for vLLM's Ray worker actors.
                        placement_group_bundle_index=resources.controller_bundle_index,
                        placement_group_capture_child_tasks=True,
                    )
                }
            else:
                # mp backend: no Ray child actors; schedule this actor directly onto 1 node with all GPUs.
                if mp_pinned_node_ip:
                    scheduling_opts = {"resources": {f"node:{mp_pinned_node_ip}": 0.001}}
                else:
                    scheduling_opts = _node_affinity_scheduling_opts_for_model(
                        self.model_name,
                        required_gpus=int(worker_gpus),
                    )

            from ..config import (
                actor_ld_library_path,
                actor_runtime_env_vars,
                otel_env_vars,
                preferred_vllm_python_executable,
            )
            worker_pythonpath = join_pythonpath(
                "/vllm",
                sanitize_worker_pythonpath(
                    PFS_PYTHONPATH,
                    env_root=os.environ.get("PFS_RUNTIME_ENV_ROOT"),
                ),
            )
            env_vars = actor_runtime_env_vars(
                pythonpath=worker_pythonpath,
                extra={
                "LD_LIBRARY_PATH": actor_ld_library_path(),
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                "HF_HUB_OFFLINE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "OMP_NUM_THREADS": os.environ.get("MINT_VLLM_OMP_NUM_THREADS", "1"),
                "MKL_NUM_THREADS": os.environ.get("MINT_VLLM_MKL_NUM_THREADS", "1"),
                "OPENBLAS_NUM_THREADS": os.environ.get("MINT_VLLM_OPENBLAS_NUM_THREADS", "1"),
                "NUMEXPR_NUM_THREADS": os.environ.get("MINT_VLLM_NUMEXPR_NUM_THREADS", "1"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("MINT_VLLM_VECLIB_MAXIMUM_THREADS", "1"),
                "BLIS_NUM_THREADS": os.environ.get("MINT_VLLM_BLIS_NUM_THREADS", "1"),
                # Some environments import tvm_ffi during vLLM init and try to JIT-build a
                # torch c-dlpack addon on every Ray worker process, which can spawn dozens
                # of concurrent compilers and stall engine startup. Disable the optional
                # build to keep vLLM init deterministic.
                "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
                # vLLM Ray executor uses Ray compiled DAG (cgraph); vLLM defaults to 300s.
                # If a model execution takes longer, EngineCore can die and the actor becomes unusable.
                "RAY_CGRAPH_get_timeout": str(ray_cgraph_get_timeout),
                "MINT_VLLM_DISTRIBUTED_EXECUTOR_BACKEND": distributed_executor_backend,
                "VLLM_DISABLE_PYNCCL": "1",
                **otel_env_vars(),
                },
            )
            if "CUDA_LAUNCH_BLOCKING" in os.environ:
                env_vars["CUDA_LAUNCH_BLOCKING"] = os.environ["CUDA_LAUNCH_BLOCKING"]
            _prepare_mint_vllm_multinode_runtime_env(env_vars)
            if "MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE" in os.environ:
                env_vars["MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE"] = os.environ[
                    "MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE"
                ]
            if "VLLM_USE_V1" in os.environ:
                env_vars["VLLM_USE_V1"] = os.environ["VLLM_USE_V1"]
            else:
                # mp backend requires local multiprocessing; default to v1 there unless explicitly overridden.
                env_vars["VLLM_USE_V1"] = "1" if distributed_executor_backend == "mp" else "0"
            for k in (
                # Ray compiled DAG knobs (driver-side env; propagated via runtime_env).
                "RAY_CGRAPH_submit_timeout",
                "RAY_CGRAPH_teardown_timeout",
                "RAY_CGRAPH_read_iteration_timeout_s",
                "RAY_CGRAPH_buffer_size_bytes",
                "RAY_CGRAPH_max_inflight_executions",
                "RAY_CGRAPH_max_buffered_results",
                "RAY_CGRAPH_overlap_gpu_communication",
                "RAY_CGRAPH_ENABLE_DETECT_DEADLOCK",
                "RAY_CGRAPH_ENABLE_PROFILING",
                "RAY_CGRAPH_ENABLE_NVTX_PROFILING",
                "RAY_CGRAPH_ENABLE_TORCH_PROFILING",
                "RAY_CGRAPH_VISUALIZE_SCHEDULE",
                "MINT_ENABLE_VLLM_IMPORT_PATCHES",
                "MINT_VLLM_LOG_STATS",
                "MINT_VLLM_DISTRIBUTED_EXECUTOR_BACKEND",
                "MINT_VLLM_ALL2ALL_BACKEND",
                "MINT_VLLM_ENGINE_LOCK_MODE",
                "MINT_VLLM_REQUEST_TIMING",
                "MINT_VLLM_PROMPT_LOGPROBS_SERIALIZE",
                "MINT_VLLM_SERIALIZE_GENERATE",
                "MINT_VLLM_SERIALIZE_MULTISAMPLE",
                "MINT_VLLM_MULTISAMPLE_MODE",
                "MINT_VLLM_SERIALIZE_ADD_REQUEST",
                "MINT_VLLM_GENERATE_TIMEOUT_S",
                "MINT_VLLM_IS_READY_TIMEOUT_S",
                "MINT_VLLM_ENABLE_CHUNKED_PREFILL",
                "MINT_VLLM_ENABLE_PREFIX_CACHING",
                "MINT_VLLM_FULLY_SHARDED_LORAS",
                "MINT_VLLM_LORA_DTYPE",
                "MINT_VLLM_WORKER_LORA_LOAD_TO_DEVICE",
                "MINT_VLLM_MAX_NUM_BATCHED_TOKENS",
                "MINT_VLLM_ADMISSION_CONTROL",
                "VLLM_USE_RAY_WRAPPED_PP_COMM",
                "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE",
                "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM",
            ):
                v = os.environ.get(k)
                if v is not None:
                    env_vars[k] = v
            env_vars.setdefault("VLLM_ATTENTION_BACKEND", server_config.vllm_attention_backend)
            _enforce_vllm_no_compiled_dag(
                env_vars,
                distributed_executor_backend=distributed_executor_backend,
            )

            # Performance defaults: do not disable prefix caching or grouped-topk in code.
            # If stability requires toggling other vLLM knobs, do it via env.

            # Expose selected vLLM debug/perf knobs via API host env without code deploys.
            for k in (
                "VLLM_LOGGING_LEVEL",
                "VLLM_LOG_LEVEL",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING",
                "VLLM_FUSED_MOE_CHUNK_SIZE",
                "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
                "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE",
                "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM",
            ):
                v = os.environ.get(k)
                if v is not None:
                    env_vars[k] = v

            # Default generate timeout: prevents hung /asample futures when EngineCore dies
            # (e.g. Triton illegal memory access) but the Ray actor stays alive.
            if "MINT_VLLM_GENERATE_TIMEOUT_S" not in env_vars:
                env_vars["MINT_VLLM_GENERATE_TIMEOUT_S"] = "3600"
            env_vars["MINT_VLLM_BASE_MODEL_NAME"] = str(self.model_name or self.model_path)

            # Fully sharded LoRAs are the default for multinode MoE actors when
            # max_lora_rank is divisible by TP. Operators can still turn this off via:
            #   export MINT_VLLM_FULLY_SHARDED_LORAS=0
            runtime_env = {"env_vars": env_vars}
            preferred_python = (preferred_vllm_python_executable() or "").strip()
            if preferred_python:
                runtime_env["py_executable"] = preferred_python

            self.engine = MultiNodeVLLMEngine.options(
                name=self.actor_name,
                namespace=PERSISTENT_NAMESPACE,
                lifetime="detached",
                num_cpus=controller_cpus,
                num_gpus=controller_gpus,
                max_concurrency=int(os.environ.get("MINT_VLLM_ACTOR_MAX_CONCURRENCY", "64")),
                **scheduling_opts,
                runtime_env=runtime_env,
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

            try:
                await async_get_ray_ref(self.engine.initialize.remote(), timeout_s=init_timeout)
            except SystemExit as e:
                if getattr(e, "code", None) == 15:
                    raise
                try:
                    ray_kill.kill(
                        self.engine,
                        reason="mint_vllm_multinode_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                        timeout_s=init_timeout,
                    )
                except Exception:
                    pass
                try:
                    if pg is not None:
                        ray.util.remove_placement_group(pg)
                except Exception:
                    pass
                self.engine = None
                raise RuntimeError(f"ray.get(initialize) triggered SystemExit for {self.actor_name}: {e}") from e
            except ray.exceptions.GetTimeoutError:
                logger.error(f"Engine initialization timed out after {init_timeout}s")
                ray_kill.kill(
                    self.engine,
                    reason="mint_vllm_multinode_init_timeout",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=init_timeout,
                )
                try:
                    if pg is not None:
                        ray.util.remove_placement_group(pg)
                except Exception:
                    pass
                self.engine = None
                raise RuntimeError("MultiNodeVLLMEngine init timed out")
            except Exception:
                try:
                    ray_kill.kill(
                        self.engine,
                        reason="mint_vllm_multinode_init_failed",
                        actor_name=self.actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                        timeout_s=init_timeout,
                    )
                except Exception:
                    pass
                try:
                    if pg is not None:
                        ray.util.remove_placement_group(pg)
                except Exception:
                    pass
                self.engine = None
                raise

            self._initialized = True
            logger.info(f"MultiNodeInferenceEngine initialized: {self.actor_name}")

            # Register with unified model actor registry for LRU tracking
            # Multi-node vLLM internally manages GPU workers, but we track total GPUs for eviction
            publish_backend_model_actor(BackendModelActorLaunch(
                actor_name=self.actor_name,
                actor_type=ActorType.VLLM,
                num_gpus=total_required_gpus,
                actor_handle=self.engine,
                namespace=PERSISTENT_NAMESPACE,
                base_model=self.model_path,
                protected=is_topology_desired,
            ))
            logger.info(
                f"Published {self.actor_name} through ModelActorSupervisor inventory ({total_required_gpus} GPUs)"
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
        traceparent = get_current_traceparent()

        # Add to engine (all workers load from shared path)
        start_time = time.time()
        try:
            ref = self.engine.add_lora.remote(
                lora_int_id=lora_id,
                lora_path=adapter_dir,
                lora_name=sampling_session_id,
                traceparent=traceparent,
            )
            await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
        except Exception:
            # Roll back registry on load failure so retries don't trip
            # "already has lora_int_id" for the same session.
            try:
                await self.registry.remove_session(sampling_session_id)
            except Exception as cleanup_e:
                logger.warning(
                    f"Failed to roll back lora_int_id={lora_id} after add_lora failure: "
                    f"{type(cleanup_e).__name__}: {cleanup_e}"
                )
            raise
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

        logger.info(
            "add_lora_for_session_from_path start sampling_session_id=%s path=%s stage=before_registry_allocate",
            sampling_session_id,
            lora_path,
        )
        lora_id = await self.registry.allocate(sampling_session_id, lora_path)
        logger.info(
            "add_lora_for_session_from_path start sampling_session_id=%s path=%s lora_int_id=%s stage=after_registry_allocate",
            sampling_session_id,
            lora_path,
            lora_id,
        )
        traceparent = get_current_traceparent()

        start_time = time.time()
        try:
            ref = self.engine.add_lora.remote(
                lora_int_id=lora_id,
                lora_path=lora_path,
                lora_name=sampling_session_id,
                traceparent=traceparent,
            )
            logger.info(
                "add_lora_for_session_from_path start sampling_session_id=%s path=%s lora_int_id=%s stage=after_add_lora_remote",
                sampling_session_id,
                lora_path,
                lora_id,
            )
            await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
            logger.info(
                "add_lora_for_session_from_path start sampling_session_id=%s path=%s lora_int_id=%s stage=after_add_lora_wait",
                sampling_session_id,
                lora_path,
                lora_id,
            )
        except Exception:
            logger.exception(
                "Failed to add LoRA from path for sampling_session_id=%s lora_int_id=%s path=%s",
                sampling_session_id,
                lora_id,
                lora_path,
            )
            # Roll back registry on load failure so retries don't trip
            # "already has lora_int_id" for the same session.
            try:
                await self.registry.remove_session(sampling_session_id)
            except Exception as cleanup_e:
                logger.warning(
                    f"Failed to roll back lora_int_id={lora_id} after add_lora failure: "
                    f"{type(cleanup_e).__name__}: {cleanup_e}"
                )
            raise
        load_time = time.time() - start_time

        logger.info(
            f"Added LoRA for session {sampling_session_id} from path "
            f"(lora_int_id={lora_id}, path={lora_path}, load_time={load_time:.3f}s)"
        )
        return lora_id

    async def abort_request(self, request_id: str) -> None:
        """Best-effort abort for an in-flight vLLM request."""
        if not self._initialized:
            return
        traceparent = get_current_traceparent()

        try:
            ref = self.engine.abort_request.remote(request_id, traceparent=traceparent)
            await async_get_ray_ref(ref, timeout_s=10)
        except Exception as e:
            logger.warning(f"MultiNodeInferenceEngine.abort_request failed: {type(e).__name__}: {e}")

    async def generate(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        max_tokens: int,
        stop: object | None = None,
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
        _raise_if_missing_lora_path(
            sampling_session_id=sampling_session_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )
        traceparent = get_current_traceparent()

        ref = self.engine.generate.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
            max_tokens=max_tokens,
            outer_request_id=request_id,
            stop=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            logprobs=logprobs,
            traceparent=traceparent,
        )
        try:
            timeout_s = ray_get_timeout_s if ray_get_timeout_s > 0 else None
            result = await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name, timeout_s=timeout_s)
        except asyncio.TimeoutError as e:
            # Avoid killing the actor: killing forces a 60-90s re-init and pollutes latency measurements.
            # Try aborting just this request, then fail loud to the client.
            try:
                abort_ref = self.engine.abort_request.remote(request_id, traceparent=traceparent)
                await async_get_ray_ref(abort_ref, timeout_s=10)
            except Exception:
                pass
            raise RuntimeError(
                f"mint_vllm_multinode_ray_get_timeout_s={ray_get_timeout_s} request_id={request_id}"
            ) from e
        except Exception as e:
            if _is_request_validation_error(e) or _is_missing_lora_path_error(e):
                raise
            logger.exception(
                "mint_vllm_multinode_ray_get_failed generate actor=%s request_id=%s sampling_session_id=%s prompt_len=%s max_tokens=%s",
                self.actor_name,
                request_id,
                sampling_session_id,
                len(prompt_ids),
                max_tokens,
            )
            try:
                ray_kill.kill(
                    self.engine,
                    reason="mint_vllm_multinode_ray_get_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=10,
                )
            except Exception:
                pass
            raise

        return GenerateResult(
            token_ids=result["token_ids"],
            logprobs=result.get("logprobs"),
            stop_reason=result.get("stop_reason"),
            routed_experts=result.get("routed_experts"),
        )

    async def generate_many(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        num_samples: int,
        max_tokens: int,
        stop: object | None = None,
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
        _raise_if_missing_lora_path(
            sampling_session_id=sampling_session_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )
        traceparent = get_current_traceparent()

        ref = self.engine.generate.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
            max_tokens=max_tokens,
            outer_request_id=request_id,
            stop=stop,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            logprobs=logprobs,
            n=num_samples,
            traceparent=traceparent,
        )
        try:
            timeout_s = ray_get_timeout_s if ray_get_timeout_s > 0 else None
            raw = await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name, timeout_s=timeout_s)
        except asyncio.TimeoutError as e:
            try:
                abort_ref = self.engine.abort_request.remote(request_id, traceparent=traceparent)
                await async_get_ray_ref(abort_ref, timeout_s=10)
            except Exception:
                pass
            raise RuntimeError(
                f"mint_vllm_multinode_ray_get_timeout_s={ray_get_timeout_s} request_id={request_id}"
            ) from e
        except Exception as e:
            if _is_request_validation_error(e) or _is_missing_lora_path_error(e):
                raise
            logger.exception(
                "mint_vllm_multinode_ray_get_failed generate_many actor=%s request_id=%s sampling_session_id=%s prompt_len=%s num_samples=%s max_tokens=%s",
                self.actor_name,
                request_id,
                sampling_session_id,
                len(prompt_ids),
                num_samples,
                max_tokens,
            )
            try:
                ray_kill.kill(
                    self.engine,
                    reason="mint_vllm_multinode_ray_get_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=10,
                )
            except Exception:
                pass
            raise

        if isinstance(raw, dict):
            raw_list: list[dict] = [raw]
        else:
            raw_list = list(raw)

        return [
            GenerateResult(
                token_ids=r["token_ids"],
                logprobs=r.get("logprobs"),
                stop_reason=r.get("stop_reason"),
                routed_experts=r.get("routed_experts"),
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
        _raise_if_missing_lora_path(
            sampling_session_id=sampling_session_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )
        traceparent = get_current_traceparent()

        ref = self.engine.compute_prompt_logprobs.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
            traceparent=traceparent,
        )
        try:
            timeout_s = ray_get_timeout_s if ray_get_timeout_s > 0 else None
            result = await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name, timeout_s=timeout_s)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"mint_vllm_multinode_ray_get_timeout_s={ray_get_timeout_s} request_id={request_id}"
            ) from e
        except Exception as e:
            if _is_request_validation_error(e) or _is_missing_lora_path_error(e):
                raise
            logger.exception(
                "mint_vllm_multinode_ray_get_failed compute_logprobs actor=%s request_id=%s sampling_session_id=%s prompt_len=%s",
                self.actor_name,
                request_id,
                sampling_session_id,
                len(prompt_ids),
            )
            try:
                ray_kill.kill(
                    self.engine,
                    reason="mint_vllm_multinode_ray_get_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=10,
                )
            except Exception:
                pass
            raise

        return list(result)

    async def compute_topk(
        self,
        sampling_session_id: str | None,
        prompt_ids: list[int],
        request_id: str,
        k: int = 10,
    ) -> list[list[tuple[int, float]] | None]:
        """Compute top-K prompt logprobs using session-specific LoRA or base model."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if not prompt_ids:
            return []
        if len(prompt_ids) == 1:
            return [None]

        if self.max_model_len is not None and len(prompt_ids) > self.max_model_len:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeds max_model_len={self.max_model_len}. "
                "Reduce prompt or use a model with larger context."
            )

        kk = int(k)
        if kk <= 0:
            return [None] * len(prompt_ids)

        ray_get_timeout_s = float(os.environ.get("MINT_VLLM_RAY_GET_TIMEOUT_S", "0"))

        lora_id = None
        lora_path = None
        if sampling_session_id is not None:
            lora_id = await self.registry.get_lora_id(sampling_session_id)
            if lora_id is not None:
                lora_path = await self.registry.get_adapter_path(lora_id)
        _raise_if_missing_lora_path(
            sampling_session_id=sampling_session_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )
        traceparent = get_current_traceparent()

        ref = self.engine.compute_prompt_topk.remote(
            prompt_ids=prompt_ids,
            request_id=request_id,
            lora_int_id=lora_id,
            lora_path=lora_path,
            k=kk,
            traceparent=traceparent,
        )
        try:
            timeout_s = ray_get_timeout_s if ray_get_timeout_s > 0 else None
            result = await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name, timeout_s=timeout_s)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"mint_vllm_multinode_ray_get_timeout_s={ray_get_timeout_s} request_id={request_id}"
            ) from e
        except Exception as e:
            if _is_request_validation_error(e) or _is_missing_lora_path_error(e):
                raise
            logger.exception(
                "mint_vllm_multinode_ray_get_failed compute_topk actor=%s request_id=%s sampling_session_id=%s prompt_len=%s k=%s",
                self.actor_name,
                request_id,
                sampling_session_id,
                len(prompt_ids),
                kk,
            )
            try:
                ray_kill.kill(
                    self.engine,
                    reason="mint_vllm_multinode_ray_get_failed",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    timeout_s=10,
                )
            except Exception:
                pass
            raise

        return list(result)

    async def remove_session(self, sampling_session_id: str) -> bool:
        """Remove a sampling session and its LoRA."""
        lora_id = await self.registry.get_lora_id(sampling_session_id)
        if lora_id is None:
            return False

        removed_lora_id, should_unload = await self.registry.remove_session(sampling_session_id)
        if removed_lora_id is None:
            return False
        if should_unload:
            traceparent = get_current_traceparent()
            try:
                ref = self.engine.remove_lora.remote(removed_lora_id, traceparent=traceparent)
                await ray_get_with_model_actor_supervisor_keepalive(ref, actor_name=self.actor_name)
            except Exception as e:
                logger.warning(f"Failed to remove LoRA {removed_lora_id} from engine: {e}")

        logger.info(
            "Removed session %s (lora_int_id=%s should_unload=%s)",
            sampling_session_id,
            removed_lora_id,
            should_unload,
        )
        return True

    async def restore_loaded_session(
        self,
        *,
        sampling_session_id: str,
        adapter_path: str,
        lora_int_id: int,
    ) -> int:
        """Restore a detached control-plane mapping for an already-loaded LoRA."""
        return await self.registry.restore_existing_session(
            sampling_session_id,
            adapter_path=adapter_path,
            lora_int_id=lora_int_id,
        )

    async def shutdown(self, kill_actor: bool = False) -> None:
        """Disconnect from the engine."""
        if self.engine is not None and kill_actor:
            try:
                ray_kill.kill(
                    self.engine,
                    reason="mint_vllm_multinode_shutdown",
                    actor_name=self.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                )
                logger.info("Killed MultiNodeVLLMEngine actor")
            except Exception as e:
                logger.warning(f"Error killing actor: {e}")
        self.engine = None
        self._initialized = False
        logger.info("MultiNodeInferenceEngine disconnected")
