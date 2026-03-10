"""Distributed Megatron training backend using Ray placement groups.

Defines MegatronWorkerGroup (detached Ray actor) and MegatronRankWorker (per-rank workers)
used by VerlTrainingEngine for MoE model training.

Shared loss functions and Tinker Datum conversion utilities live in megatron_training.py.
"""

from __future__ import annotations  # Allow forward references in type hints

import os
import json
import math
import socket
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import ray
# NOTE: torch and tensordict imports are LAZY - done inside MegatronRankWorker.__init__
# to ensure CUDA_VISIBLE_DEVICES is set before torch initializes CUDA
# (tensordict imports torch internally)

from . import ray_kill
from ..logging_context import get_request_id, init_actor_observability

logger = logging.getLogger(__name__)

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH, PFS_TINKER_PATH, RAY_NAMESPACE, otel_env_vars
from tinker_server.backend.model_registry import get_model_config
from tinker_server.ray_utils import init_ray
from tinker_server.model_input_utils import flatten_encoded_text_chunks

# Persistent actor configuration
PERSISTENT_NAMESPACE = RAY_NAMESPACE  # Same namespace as vLLM

_megatron_create_locks: dict[str, threading.Lock] = {}
_megatron_create_locks_guard = threading.Lock()

# Sentinel indicating that optim_step has consumed the session's gradients.
# Using a unique object() (not None) so dict.get() returning None ("never cached")
# is distinguishable from "consumed".  All consumers must check with `is`.
_GRADIENTS_CONSUMED = object()


def _get_megatron_create_lock(actor_name: str) -> threading.Lock:
    with _megatron_create_locks_guard:
        lock = _megatron_create_locks.get(actor_name)
        if lock is None:
            lock = threading.Lock()
            _megatron_create_locks[actor_name] = lock
        return lock


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default


def _collect_python_thread_stacks(*, limit: int = 64) -> str:
    import sys
    import traceback

    lines: list[str] = []
    threads_by_ident = {t.ident: t for t in threading.enumerate()}
    frames = sys._current_frames()
    for tid, frame in frames.items():
        thread = threads_by_ident.get(tid)
        tname = thread.name if thread is not None else "unknown"
        lines.append(f"\n--- thread_id={tid} name={tname} ---\n")
        lines.extend(traceback.format_stack(frame, limit=limit))
    return "".join(lines)


def _model_key_from_base_model(base_model: str) -> str:
    import re

    hf_cache_pattern = r"models--([^/]+)--([^/]+)/snapshots"
    match = re.search(hf_cache_pattern, base_model)
    if match:
        org, model = match.groups()
        return f"{org}/{model}"
    return base_model


def _preferred_worker_node_ips_for_model(base_model: str) -> list[str]:
    raw = os.environ.get("MINT_MODEL_NODE_IPS_JSON", "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        logger.warning("MINT_MODEL_NODE_IPS_JSON is not valid JSON; ignoring")
        return []
    if not isinstance(data, dict):
        logger.warning("MINT_MODEL_NODE_IPS_JSON must be a JSON object; ignoring")
        return []

    model_key = _model_key_from_base_model(base_model)
    candidates = None
    for key in (model_key, model_key.lower(), base_model, base_model.lower()):
        value = data.get(key)
        if value is not None:
            candidates = value
            break
    if not isinstance(candidates, list):
        return []

    cleaned = [str(ip).strip() for ip in candidates if str(ip).strip()]
    if not cleaned:
        return []

    try:
        cluster = ray.cluster_resources() or {}
    except Exception:
        cluster = {}
    usable = [ip for ip in cleaned if f"node:{ip}" in cluster]
    if not usable:
        logger.warning(f"[MegatronWorkerGroup] node pinning has no usable node IPs for model={model_key}")
        return []
    if len(usable) < len(cleaned):
        skipped = [ip for ip in cleaned if ip not in usable]
        logger.warning(f"[MegatronWorkerGroup] node pinning skipped missing nodes for model={model_key}: {skipped}")
    logger.info(f"[MegatronWorkerGroup] node pinning for model={model_key}: {usable}")
    return usable


def _make_megatron_actor_name(base_model: str) -> str:
    """Generate per-model Megatron actor name.

    Normalizes input to handle both:
    - HuggingFace model ID: "Qwen/Qwen3-30B-A3B-Instruct-2507"
    - Resolved cache path: "/vePFS/.../models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/..."

    Both produce the same actor name for consistent lookup.
    """
    import re

    # Check if this is a resolved HuggingFace cache path
    # Pattern: models--{org}--{model}/snapshots/{hash}
    hf_cache_pattern = r"models--([^/]+)--([^/]+)/snapshots"
    match = re.search(hf_cache_pattern, base_model)

    if match:
        # Extract org and model from cache path
        org, model = match.groups()
        model_name = model.lower().replace("-", "_").replace(".", "_")
    else:
        # HuggingFace model ID or plain path - take last component
        model_name = base_model.split("/")[-1].lower().replace("-", "_").replace(".", "_")

    return f"megatron_{model_name}"


@dataclass
class DistributedConfig:
    """Configuration for distributed Megatron training.

    Defaults configured for 1-GPU setup (GPU 0 has leaked memory).
    For multi-GPU: tensor_parallel_size=2 (2-GPU) or TP=2,PP=2,EP=2 (8-GPU)

    MoE Parallel Folding cases:
    1. EP > TP with ETP < TP: world_size = EP
       (TP is a subgroup for attention within EP dimension)
    2. CP > 1 and EP > 1: world_size = TP * PP * max(EP, CP)
       (CP and EP share GPU ranks)
    3. Traditional: world_size = TP * PP * EP * CP
    """

    tensor_parallel_size: int = 1  # Single GPU - GPU 0 has corrupted memory
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None  # None = use TP, 1 = no expert splitting
    context_parallel_size: int = 1
    use_fp8: bool = False  # FP8 quantization for K2 and similar models
    router_replay_mode: str = "disabled"

    @property
    def world_size(self) -> int:
        """Total number of processes needed.

        MoE Parallel Folding cases:
        1. EP > TP with ETP < TP: world_size = EP
           (TP is a subgroup for attention within EP dimension)
        2. CP > 1 and EP > 1: world_size = TP * PP * max(EP, CP)
           (CP and EP share GPU ranks)
        3. Traditional: world_size = TP * PP * EP * CP
        """
        etp = self.expert_tensor_parallel_size
        if etp is None:
            etp = self.tensor_parallel_size

        if self.expert_parallel_size >= self.tensor_parallel_size and etp < self.tensor_parallel_size:
            # MoE Parallel Folding with ETP: TP is subgroup within EP
            return (
                self.expert_parallel_size
                * self.pipeline_parallel_size
                * self.context_parallel_size
            )
        elif self.expert_parallel_size > 1 and self.context_parallel_size > 1:
            # CP/EP Folding: CP and EP share GPU ranks
            return (
                self.tensor_parallel_size
                * self.pipeline_parallel_size
                * max(self.expert_parallel_size, self.context_parallel_size)
            )
        else:
            # Traditional: all dimensions are orthogonal
            return (
                self.tensor_parallel_size
                * self.pipeline_parallel_size
                * self.expert_parallel_size
                * self.context_parallel_size
            )


@ray.remote(num_gpus=0, num_cpus=0)
def get_node_ip_and_free_port() -> tuple[str, int]:
    """Get node IP and free port for master address.

    Uses Ray's node IP which correctly identifies the inter-node network interface.
    Self-contained to avoid module import issues on Ray workers.
    """
    import ray
    # Use Ray's IP detection which respects --node-ip-address and finds the correct interface
    ip = ray.util.get_node_ip_address()
    # Get free port inline
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return ip, port


@ray.remote(num_gpus=1, num_cpus=0)
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
        init_actor_observability()
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.config = distributed_config
        self.engine = None  # Set in initialize()
        self._current_session_id = None
        # Per-session gradient storage (CPU).
        # Values: list[torch.Tensor] (valid gradients) or _GRADIENTS_CONSUMED (consumed by optim_step)
        self._session_gradients: dict[str, list[torch.Tensor] | object] = {}
        self._session_optimizer_states: dict[str, dict] = {}  # Per-session optimizer state (CPU)
        self._sticky_train_mode_enabled = _env_flag("MINT_MEGATRON_STICKY_TRAIN_MODE", default=False)
        self._sticky_train_mode_idle_timeout_s = _env_float("MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S", default=15.0)
        self._sticky_train_mode_close_on_optim = _env_flag("MINT_MEGATRON_STICKY_CLOSE_ON_OPTIM", default=True)
        self._sticky_train_mode_diag = _env_flag("MINT_MEGATRON_STICKY_TIMING_DIAG", default=False)
        self._sticky_train_mode_ctx = None
        self._sticky_train_mode_session_id: str | None = None
        self._sticky_train_mode_last_used_s: float = 0.0
        self._sticky_train_mode_enter_total: int = 0
        self._sticky_train_mode_reuse_total: int = 0
        self._sticky_train_mode_exit_total: int = 0

        logger.info(f"[MegatronRankWorker] Worker {rank}/{world_size} created (not yet initialized)")

    def _sticky_enabled_for(self, session_id: str | None) -> bool:
        return bool(self._sticky_train_mode_enabled and session_id)

    def _release_sticky_train_mode(self, *, reason: str, snapshot_gradients: bool) -> dict:
        """Release (close) the currently open sticky train_mode() context.

        Two responsibilities before calling ctx.__exit__():
          1. Optionally snapshot GPU gradients to CPU (if snapshot_gradients=True),
             so the session's accumulated gradients survive the offload.
             Skipped when gradients were already consumed (_GRADIENTS_CONSUMED).
          2. Call ctx.__exit__() which triggers GPU->CPU model parameter offload (~620ms).

        Idempotent: if no ctx is open, returns {"released": False} without side effects.

        Args:
            reason: Why we're releasing. Used in diagnostic logs. Common values:
                "session_change", "idle_timeout", "optim_step_complete",
                "forward_backward_error", "optim_step_error", "swap_session_state".
            snapshot_gradients: Whether to capture GPU gradients to CPU before exit.
                True  -- on session switch / idle timeout (gradients still useful).
                False -- after optim_step (gradients consumed) or on error (GPU state
                         undefined; snapshotting would persist garbage).

        Returns:
            dict with: released, exit_s, snapshot_count, exit_total, session_id.
        """
        released = False
        exit_s = 0.0
        snap_count = 0
        active_session = self._sticky_train_mode_session_id
        if self._sticky_train_mode_ctx is not None:
            # -- Optional gradient snapshot before __exit__ destroys GPU state --
            if snapshot_gradients and active_session is not None:
                # Guard: if optim_step already consumed the gradients
                # (_GRADIENTS_CONSUMED sentinel), skip snapshot.  Capturing
                # post-optimizer residual GPU data would revive stale gradients
                # and cause double-apply on the next forward_backward.
                existing = self._session_gradients.get(active_session)
                if existing is _GRADIENTS_CONSUMED:
                    logger.debug(
                        f"[Rank {self.rank}] sticky_train_mode release: "
                        f"skipping gradient snapshot for session={active_session} "
                        f"(already consumed by optim_step), reason={reason}"
                    )
                else:
                    try:
                        captured = self._capture_gradients()
                        self._session_gradients[active_session] = captured
                        snap_count = len(captured)
                    except Exception as e:
                        logger.warning(
                            f"[Rank {self.rank}] sticky_train_mode snapshot failed "
                            f"(session={active_session}, reason={reason}): {e}"
                        )

            # -- ctx.__exit__: triggers GPU->CPU model parameter offload --
            t0 = time.perf_counter()
            self._sticky_train_mode_ctx.__exit__(None, None, None)
            exit_s = time.perf_counter() - t0

            # -- Reset all sticky bookkeeping --
            self._sticky_train_mode_ctx = None
            self._sticky_train_mode_session_id = None
            self._sticky_train_mode_last_used_s = 0.0
            self._sticky_train_mode_exit_total += 1
            released = True

            if self._sticky_train_mode_diag:
                logger.info(
                    f"[Rank {self.rank}] sticky_train_mode release: reason={reason} "
                    f"exit_ms={exit_s * 1000.0:.2f} snapshot_count={snap_count}"
                )

        return {
            "released": released,
            "exit_s": exit_s,
            "snapshot_count": snap_count,
            "exit_total": self._sticky_train_mode_exit_total,
            "session_id": active_session,
        }

    def _ensure_sticky_train_mode(self, *, session_id: str, reason: str) -> dict:
        """Ensure a train_mode() context is open for the given session_id.

        Three mutually exclusive outcomes:
          1. Reuse  -- ctx already open for same session, not expired -> zero IO cost.
          2. Rotate -- ctx open for a different session or idle-expired
                       -> release old ctx (snapshot gradients), then fresh enter.
          3. Enter  -- no ctx -> fresh enter (CPU->GPU weight load, ~620ms).

        Why this method exists:
          With param_offload=True, train_mode().__enter__ triggers a full model
          CPU->GPU transfer and __exit__ triggers the reverse.  Without sticky mode,
          each forward_backward / optim_step call independently enters/exits,
          producing 2*(N+1) full-model round-trips for N chunks per step.
          Sticky mode reuses one open context across chunks, exiting only once
          at step end.

        Args:
            session_id: Which session this operation belongs to.  Used to detect
                session changes that require rotating the context.
            reason: Caller identifier ("forward_backward" / "optim_step") for
                diagnostic logs.

        Returns:
            dict containing:
              reused (bool): True if an existing ctx was reused (zero IO cost).
              enter_s (float): Wall-clock seconds for fresh enter (0.0 if reused).
              released_before_enter (bool): True if an old ctx was released first.
              enter_total (int): Cumulative enter count.
              reuse_total (int): Cumulative reuse count.
        """
        # -- Fast path: feature disabled or no session_id --
        if not self._sticky_enabled_for(session_id):
            return {
                "reused": False,
                "enter_s": 0.0,
                "released_before_enter": False,
                "enter_total": self._sticky_train_mode_enter_total,
                "reuse_total": self._sticky_train_mode_reuse_total,
            }

        released_before_enter = False
        now = time.perf_counter()

        # -- Phase 1: Check whether the existing ctx must be released --
        # Two release triggers (either one suffices):
        #   a) Session changed -> old session's GPU grads must be snapshot-saved first.
        #   b) Idle timeout    -> holding GPU memory indefinitely is wasteful.
        if self._sticky_train_mode_ctx is not None:
            idle_s = (
                now - self._sticky_train_mode_last_used_s
                if self._sticky_train_mode_last_used_s > 0
                else 0.0
            )
            session_changed = self._sticky_train_mode_session_id != session_id
            idle_expired = (
                self._sticky_train_mode_idle_timeout_s > 0
                and idle_s > self._sticky_train_mode_idle_timeout_s
            )
            if session_changed or idle_expired:
                # snapshot_gradients=True: GPU grads will be destroyed by __exit__,
                # so capture them to CPU first (unless already _GRADIENTS_CONSUMED).
                self._release_sticky_train_mode(
                    reason="session_change" if session_changed else "idle_timeout",
                    snapshot_gradients=True,
                )
                released_before_enter = True

        # -- Phase 2: Reuse (ctx still alive = same session, not expired) --
        # This is the performance-critical fast path: consecutive chunks within
        # the same session hit this branch with zero IO cost.
        if self._sticky_train_mode_ctx is not None:
            self._sticky_train_mode_reuse_total += 1
            self._sticky_train_mode_last_used_s = time.perf_counter()
            if self._sticky_train_mode_diag:
                logger.info(
                    f"[Rank {self.rank}] sticky_train_mode reuse: "
                    f"session={session_id} reason={reason} reuse_total={self._sticky_train_mode_reuse_total}"
                )
            return {
                "reused": True,
                "enter_s": 0.0,
                "released_before_enter": released_before_enter,
                "enter_total": self._sticky_train_mode_enter_total,
                "reuse_total": self._sticky_train_mode_reuse_total,
            }

        # -- Phase 3: Fresh enter (no ctx, or Phase 1 just released the old one) --
        # ctx.__enter__() calls load_megatron_model_to_gpu():
        #   - All model parameters: CPU -> GPU (PCIe DMA, ~620ms)
        #   - Zero all GPU gradient buffers
        t0 = time.perf_counter()
        ctx = self.engine.train_mode()
        ctx.__enter__()
        enter_s = time.perf_counter() - t0
        self._sticky_train_mode_ctx = ctx
        self._sticky_train_mode_session_id = session_id
        self._sticky_train_mode_last_used_s = time.perf_counter()
        self._sticky_train_mode_enter_total += 1
        if self._sticky_train_mode_diag:
            logger.info(
                f"[Rank {self.rank}] sticky_train_mode enter: "
                f"session={session_id} reason={reason} enter_ms={enter_s * 1000.0:.2f}"
            )
        return {
            "reused": False,
            "enter_s": enter_s,
            "released_before_enter": released_before_enter,
            "enter_total": self._sticky_train_mode_enter_total,
            "reuse_total": self._sticky_train_mode_reuse_total,
        }

    def _start_slow_op_watchdog(
        self,
        *,
        op: str,
        session_id: str | None,
        extra: str = "",
    ) -> tuple[threading.Event, threading.Thread] | None:
        timeout_s = _env_float("MINT_MEGATRON_STACK_DUMP_TIMEOUT_S", 0.0)
        if timeout_s <= 0:
            return None
        stack_limit = max(8, _env_int("MINT_MEGATRON_STACK_DUMP_LIMIT", 96))
        stop_event = threading.Event()
        started_at = time.perf_counter()

        def _watch() -> None:
            if stop_event.wait(timeout_s):
                return
            elapsed_s = time.perf_counter() - started_at
            try:
                stack_dump = _collect_python_thread_stacks(limit=stack_limit)
            except Exception as e:
                logger.error(
                    f"[Rank {self.rank}] slow_op_watchdog failed to collect stacks "
                    f"op={op} session={session_id} elapsed_s={elapsed_s:.3f}: {type(e).__name__}: {e}"
                )
                return
            logger.error(
                f"[Rank {self.rank}] slow_op_watchdog timeout op={op} "
                f"session={session_id} elapsed_s={elapsed_s:.3f} {extra}\n{stack_dump}"
            )

        thread = threading.Thread(
            target=_watch,
            name=f"rank{self.rank}-{op}-watchdog",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    def _stop_slow_op_watchdog(self, token: tuple[threading.Event, threading.Thread] | None) -> None:
        if token is None:
            return
        stop_event, thread = token
        stop_event.set()
        try:
            thread.join(timeout=0.05)
        except Exception:
            pass

    def get_rss_bytes(self) -> int:
        with open("/proc/self/statm", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
        rss_pages = int(parts[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return rss_pages * page_size

    def log_memory_breakdown(self, phase: str) -> dict:
        """Log detailed GPU memory breakdown for profiling.

        Args:
            phase: Description of current phase (e.g., "after_init", "after_forward")

        Returns:
            Dict with memory stats in GiB for analysis.
        """
        import torch

        if not torch.cuda.is_available():
            return {}

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GiB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GiB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GiB
        max_reserved = torch.cuda.max_memory_reserved() / (1024**3)  # GiB

        # Get detailed stats
        stats = torch.cuda.memory_stats()
        active_bytes = stats.get("active_bytes.all.current", 0) / (1024**3)
        inactive_bytes = stats.get("inactive_split_bytes.all.current", 0) / (1024**3)

        memory_info = {
            "phase": phase,
            "allocated_gib": round(allocated, 3),
            "reserved_gib": round(reserved, 3),
            "max_allocated_gib": round(max_allocated, 3),
            "max_reserved_gib": round(max_reserved, 3),
            "active_gib": round(active_bytes, 3),
            "inactive_gib": round(inactive_bytes, 3),
            "fragmentation_gib": round(reserved - allocated, 3),
        }

        # Only rank 0 logs to avoid spam
        if self.rank == 0:
            logger.info(
                f"[MEMORY] {phase}: "
                f"allocated={allocated:.2f}GiB, reserved={reserved:.2f}GiB, "
                f"peak={max_allocated:.2f}GiB, fragmentation={reserved-allocated:.2f}GiB"
            )

        return memory_info

    def _capture_gradients(self) -> list[torch.Tensor]:
        """Capture all gradient buffers from DDP modules to CPU.

        Returns a list of CPU tensors containing gradient data.
        Must be called while in train_mode context (gradients on GPU).
        """
        import logging
        from megatron.core.distributed import DistributedDataParallel as DDP

        grads = []
        want_total_norm = self.rank == 0 and logger.isEnabledFor(logging.DEBUG)
        total_norm_sq = 0.0
        for model_chunk in self.engine.module:
            if isinstance(model_chunk, DDP):
                for buffers in [model_chunk.buffers, model_chunk.expert_parallel_buffers]:
                    for buffer in buffers:
                        if buffer.grad_data is not None and buffer.grad_data.storage().size() > 0:
                            cpu_grad = buffer.grad_data.detach().cpu()
                            grads.append(cpu_grad)
                            if want_total_norm:
                                total_norm_sq += cpu_grad.float().norm().item() ** 2

        if want_total_norm:
            total_norm = total_norm_sq ** 0.5
            logger.debug(
                f"[Rank {self.rank}] _capture_gradients: {len(grads)} buffers, total_norm={total_norm:.6f}"
            )
        logger.debug(f"[Rank {self.rank}] Captured {len(grads)} gradient buffers")
        return grads

    def _restore_gradients(self, grads: list[torch.Tensor]) -> None:
        """Restore gradient buffers from CPU back to GPU.

        Must be called while in train_mode context (after model loaded to GPU).
        Args:
            grads: List of CPU tensors from _capture_gradients.
        """
        import logging
        from megatron.core.distributed import DistributedDataParallel as DDP

        idx = 0
        want_total_norm = self.rank == 0 and logger.isEnabledFor(logging.DEBUG)
        total_norm_sq = 0.0
        for model_chunk in self.engine.module:
            if isinstance(model_chunk, DDP):
                for buffers in [model_chunk.buffers, model_chunk.expert_parallel_buffers]:
                    for buffer in buffers:
                        if buffer.grad_data is not None and buffer.grad_data.storage().size() > 0:
                            if idx >= len(grads):
                                raise RuntimeError(
                                    f"[Rank {self.rank}] _restore_gradients: expected >= {idx+1} grads, got {len(grads)}"
                                )

                            cpu_grad = grads[idx]
                            if cpu_grad.is_cuda:
                                raise RuntimeError(
                                    f"[Rank {self.rank}] _restore_gradients: expected CPU grads, got CUDA tensor at idx={idx}"
                                )

                            buffer.grad_data.copy_(cpu_grad, non_blocking=True)
                            if want_total_norm:
                                total_norm_sq += cpu_grad.float().norm().item() ** 2
                            idx += 1

        if idx != len(grads):
            raise RuntimeError(
                f"[Rank {self.rank}] _restore_gradients: restored {idx} buffers but grads has {len(grads)} tensors"
            )

        if want_total_norm:
            total_norm = total_norm_sq ** 0.5
            logger.debug(
                f"[Rank {self.rank}] _restore_gradients: restored {idx} buffers, total_norm={total_norm:.6f}"
            )
        logger.debug(f"[Rank {self.rank}] Restored {idx} gradient buffers")

    def _capture_optimizer_state(self) -> dict:
        """Capture optimizer state (momentum, variance) to CPU.

        Returns a dict containing optimizer state.
        """
        import torch
        from megatron.core.optimizer import ChainedOptimizer

        state_dict = {}
        optimizer = self.engine.optimizer

        if optimizer is None:
            return state_dict

        def iter_optimizers(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        for i, _opt in enumerate(iter_optimizers(optimizer)):
            if hasattr(_opt, 'optimizer') and _opt.optimizer is not None:
                inner_opt = _opt.optimizer
                # Deep copy state to CPU
                opt_state = {}
                for param, state in inner_opt.state.items():
                    opt_state[id(param)] = {
                        k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                        for k, v in state.items()
                    }
                state_dict[f"optimizer_{i}"] = {
                    "state": opt_state,
                    "param_groups": [
                        {k: v for k, v in pg.items() if k != 'params'}
                        for pg in inner_opt.param_groups
                    ]
                }

        logger.debug(f"[Rank {self.rank}] Captured optimizer state for {len(state_dict)} optimizers")
        return state_dict

    def _restore_optimizer_state(self, state_dict: dict) -> None:
        """Restore optimizer state from CPU.

        CRITICAL: Always clears existing state first to prevent contamination
        between sessions. Even if state_dict is empty, we must clear the existing
        optimizer state so the session starts fresh.

        Args:
            state_dict: Dict from _capture_optimizer_state. May be empty for new sessions.
        """
        import torch
        from megatron.core.optimizer import ChainedOptimizer

        optimizer = self.engine.optimizer
        if optimizer is None:
            return

        def iter_optimizers(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        for i, _opt in enumerate(iter_optimizers(optimizer)):
            if hasattr(_opt, 'optimizer') and _opt.optimizer is not None:
                inner_opt = _opt.optimizer

                # CRITICAL: Always clear existing state first to prevent session contamination
                # Without this, optimizer momentum from previous session persists
                state = inner_opt.state
                if hasattr(state, '_inner_dicts'):
                    # ProxyDict from ChainedOptimizer
                    for inner_dict in state._inner_dicts:
                        inner_dict.clear()
                elif hasattr(state, 'clear'):
                    # Regular dict
                    state.clear()
                else:
                    # Unknown type - try to clear via iteration
                    keys = list(state.keys()) if hasattr(state, 'keys') else []
                    for key in keys:
                        del state[key]
                
                # If no state to restore, we're done (state is now clean)
                key = f"optimizer_{i}"
                if not state_dict or key not in state_dict:
                    continue

                saved_state = state_dict[key]["state"]
                saved_param_ids = list(saved_state.keys())

                # Get all params from param_groups (not from state.keys() which could be empty)
                all_params = []
                for pg in inner_opt.param_groups:
                    all_params.extend(pg['params'])

                # Restore state by position mapping
                for j, param in enumerate(all_params):
                    if j < len(saved_param_ids):
                        saved_id = saved_param_ids[j]
                        if saved_id in saved_state:
                            # Initialize state dict for this param
                            inner_opt.state[param] = {}
                            for k, v in saved_state[saved_id].items():
                                if isinstance(v, torch.Tensor):
                                    inner_opt.state[param][k] = v.cuda()
                                else:
                                    inner_opt.state[param][k] = v

                # Restore param group settings (like lr)
                saved_groups = state_dict[key].get("param_groups", [])
                for j, pg in enumerate(inner_opt.param_groups):
                    if j < len(saved_groups):
                        for k, v in saved_groups[j].items():
                            pg[k] = v

        logger.debug(f"[Rank {self.rank}] Restored optimizer state (cleared first)")

    def _reset_optimizer_state(self) -> None:
        """Reset optimizer state (momentum, variance) for a new session.

        Clears all momentum and variance buffers so the new session starts fresh,
        without momentum contamination from previous sessions.
        """
        from megatron.core.optimizer import ChainedOptimizer

        optimizer = self.engine.optimizer
        if optimizer is None:
            print(f"[Rank {self.rank}] _reset_optimizer_state: optimizer is None, skipping", flush=True)
            return

        def iter_optimizers(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        reset_count = 0
        for i, _opt in enumerate(iter_optimizers(optimizer)):
            if hasattr(_opt, 'optimizer') and _opt.optimizer is not None:
                inner_opt = _opt.optimizer
                state = inner_opt.state
                state_size_before = len(state) if hasattr(state, '__len__') else 'unknown'
                print(f"[Rank {self.rank}] _reset_optimizer_state: opt[{i}] state size BEFORE clear = {state_size_before}", flush=True)
                # Handle ProxyDict from ChainedOptimizer - it wraps multiple optimizer
                # states and doesn't have .clear(). Access underlying dicts directly.
                if hasattr(state, '_inner_dicts'):
                    # ProxyDict from ChainedOptimizer
                    for inner_dict in state._inner_dicts:
                        inner_dict.clear()
                    logger.debug(f"[Rank {self.rank}] Cleared ProxyDict optimizer state ({len(state._inner_dicts)} inner dicts)")
                    print(f"[Rank {self.rank}] _reset_optimizer_state: Cleared ProxyDict ({len(state._inner_dicts)} inner dicts)", flush=True)
                elif hasattr(state, 'clear'):
                    # Regular dict
                    state.clear()
                    logger.debug(f"[Rank {self.rank}] Cleared optimizer state dict")
                    print(f"[Rank {self.rank}] _reset_optimizer_state: Cleared state dict", flush=True)
                else:
                    # Unknown type - try to clear via iteration
                    keys = list(state.keys()) if hasattr(state, 'keys') else []
                    for key in keys:
                        del state[key]
                    logger.debug(f"[Rank {self.rank}] Cleared optimizer state via key deletion ({len(keys)} entries)")
                    print(f"[Rank {self.rank}] _reset_optimizer_state: Cleared via key deletion ({len(keys)} keys)", flush=True)
                state_size_after = len(state) if hasattr(state, '__len__') else 'unknown'
                print(f"[Rank {self.rank}] _reset_optimizer_state: opt[{i}] state size AFTER clear = {state_size_after}", flush=True)
                reset_count += 1

        logger.debug(f"[Rank {self.rank}] Reset optimizer state for {reset_count} optimizers")
        print(f"[Rank {self.rank}] _reset_optimizer_state: Reset {reset_count} optimizers total", flush=True)

        # Reset LR scheduler so new sessions start with fresh schedule
        self._reset_lr_scheduler()

        # Rebuild optimizer + scheduler to clear any hidden state
        self._rebuild_optimizer_and_scheduler()

    def _reset_lr_scheduler(self) -> None:
        """Reset LR scheduler for a fresh session."""
        lr_scheduler = getattr(self.engine, "lr_scheduler", None)
        if lr_scheduler is None:
            return

        # Rebuild scheduler if engine supports it
        if hasattr(self.engine, "_build_lr_scheduler"):
            try:
                self.engine.lr_scheduler = self.engine._build_lr_scheduler()
                # Keep checkpoint manager in sync if present
                for attr_name in ("checkpoint_mananager", "checkpoint_manager"):
                    manager = getattr(self.engine, attr_name, None)
                    if manager is not None and hasattr(manager, "optimizer_scheduler"):
                        manager.optimizer_scheduler = self.engine.lr_scheduler
                logger.info(f"[Rank {self.rank}] Reset lr_scheduler via _build_lr_scheduler")
                return
            except Exception as e:
                logger.warning(f"[Rank {self.rank}] Failed to rebuild lr_scheduler: {e}")

        # Fallback: try to reset state_dict if supported
        try:
            lr_scheduler.load_state_dict({})
            logger.info(f"[Rank {self.rank}] Reset lr_scheduler via empty state_dict")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to reset lr_scheduler: {e}")

    def _rebuild_optimizer_and_scheduler(self) -> None:
        """Rebuild optimizer and LR scheduler to ensure a clean state."""
        try:
            if hasattr(self.engine, "_build_optimizer") and hasattr(self.engine, "_build_lr_scheduler"):
                self.engine.optimizer = self.engine._build_optimizer()
                self.engine.lr_scheduler = self.engine._build_lr_scheduler()
                # Keep checkpoint manager in sync if present
                for attr_name in ("checkpoint_mananager", "checkpoint_manager"):
                    manager = getattr(self.engine, attr_name, None)
                    if manager is not None:
                        if hasattr(manager, "optimizer"):
                            manager.optimizer = self.engine.optimizer
                        if hasattr(manager, "optimizer_scheduler"):
                            manager.optimizer_scheduler = self.engine.lr_scheduler
                logger.info(f"[Rank {self.rank}] Rebuilt optimizer and lr_scheduler")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to rebuild optimizer/lr_scheduler: {e}")

    def swap_session_state(self, new_session_id: str) -> None:
        """Swap session state: save outgoing session's gradients/optimizer, load incoming.

        For new sessions (not in cache), resets optimizer state to avoid momentum
        contamination from previous sessions.

        IMPORTANT: Does NOT overwrite gradients if forward_backward already cached them.
        This is critical because entering train_mode() zeros GPU gradients, so we must
        preserve gradients that were captured by forward_backward before the session switch.

        Args:
            new_session_id: Session ID to switch to.
        """
        if self._current_session_id == new_session_id:
            return

        if self._sticky_train_mode_ctx is not None:
            # Avoid nested train_mode contexts during session switch routines.
            self._release_sticky_train_mode(reason="swap_session_state", snapshot_gradients=True)

        # Must use train_mode to access GPU gradient buffers (required for param_offload)
        with self.engine.train_mode():
            # Save outgoing session's state
            if self._current_session_id is not None:
                # Only capture gradients if not already cached by forward_backward
                # CRITICAL: train_mode() zeros GPU gradients, so if forward_backward
                # already captured valid gradients, we must NOT overwrite them with zeros
                #
                # Three cases:
                # 1. Not in cache → capture (first forward_backward hasn't run)
                # 2. In cache AND valid list → preserve (valid gradients from forward_backward)
                # 3. In cache AND _GRADIENTS_CONSUMED → preserve sentinel (consumed by optim_step)
                cached = self._session_gradients.get(self._current_session_id)
                if self._current_session_id not in self._session_gradients:
                    grads = self._capture_gradients()
                    self._session_gradients[self._current_session_id] = grads
                    logger.debug(
                        f"[Rank {self.rank}] Captured gradients for session {self._current_session_id}: "
                        f"{len(grads)} buffers"
                    )
                elif cached is not None and cached is not _GRADIENTS_CONSUMED:
                    logger.debug(
                        f"[Rank {self.rank}] Preserving existing gradients for session {self._current_session_id}"
                    )
                elif cached is _GRADIENTS_CONSUMED:
                    # Gradients consumed by optim_step - preserve sentinel
                    logger.debug(
                        f"[Rank {self.rank}] Session {self._current_session_id} gradients were consumed, "
                        "preserving _GRADIENTS_CONSUMED sentinel"
                    )
                else:
                    # cached is None (should not happen with current logic, but handle defensively)
                    logger.debug(
                        f"[Rank {self.rank}] Session {self._current_session_id} has None gradients"
                    )

                # Always capture optimizer state (not affected by train_mode zeroing)
                opt_state = self._capture_optimizer_state()
                self._session_optimizer_states[self._current_session_id] = opt_state
                logger.debug(
                    f"[Rank {self.rank}] Saved optimizer state for session {self._current_session_id}: "
                    f"{len(opt_state)} optimizers"
                )

            # Restore incoming session's gradients (or zero for new session)
            # _GRADIENTS_CONSUMED means gradients were consumed by optim_step - zero them
            cached_incoming = self._session_gradients.get(new_session_id)
            if cached_incoming is not None and cached_incoming is not _GRADIENTS_CONSUMED:
                self._restore_gradients(cached_incoming)
                logger.debug(f"[Rank {self.rank}] Restored gradients for session {new_session_id}")
            else:
                # New session OR gradients consumed - zero gradients
                self.engine.optimizer_zero_grad()
                if cached_incoming is _GRADIENTS_CONSUMED:
                    logger.debug(
                        f"[Rank {self.rank}] Session {new_session_id} gradients were consumed, zeroed gradients"
                    )
                else:
                    logger.debug(f"[Rank {self.rank}] Session {new_session_id} - zeroed gradients (new session)")

            # Restore incoming session's optimizer state (or reset for new session)
            if new_session_id in self._session_optimizer_states:
                self._restore_optimizer_state(self._session_optimizer_states[new_session_id])
                print(f"[Rank {self.rank}] RESTORED optimizer state for session {new_session_id}", flush=True)
                logger.debug(f"[Rank {self.rank}] Restored optimizer state for session {new_session_id}")
            else:
                # New session - reset optimizer state (clear momentum/variance)
                self._reset_optimizer_state()
                print(f"[Rank {self.rank}] RESET optimizer state for NEW session {new_session_id}", flush=True)
                logger.info(f"[Rank {self.rank}] New session {new_session_id} - reset optimizer state")

        self._current_session_id = new_session_id

    def clear_session_state(self, session_id: str) -> None:
        """Clear saved state for a session (call after session completes).

        Args:
            session_id: Session ID to clear.
        """
        if session_id in self._session_gradients:
            del self._session_gradients[session_id]
        if session_id in self._session_optimizer_states:
            del self._session_optimizer_states[session_id]
        logger.debug(f"[Rank {self.rank}] Cleared state for session {session_id}")

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
            f"ray_gpu_ids={ray_gpu_ids}, torch.cuda.device_count()={device_count}, "
            f"request_id={get_request_id() or '-'}"
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

        # Log memory after initialization
        self.log_memory_breakdown("after_init")

        logger.info(f"[Rank {self.rank}] initialize() complete")

    def _initialize_distributed(self):
        """Initialize torch.distributed with NCCL backend."""
        import torch
        from datetime import timedelta

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

        # K2 / multi-node runs can spend >10 minutes in HF->Megatron weight loading before the
        # first NCCL communicator is fully established. torch.distributed defaults (10 min) can
        # time out during the initial collectives, causing the whole worker group to die.
        #
        # Keep a per-worker override so operators can tune without code changes.
        timeout_s = int(os.environ.get("MINT_TORCH_DIST_TIMEOUT_S") or (3600 if self.world_size >= 32 else 600))

        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=self.world_size,
            rank=self.rank,
            timeout=timedelta(seconds=timeout_s),
        )

        logger.info(f"[Rank {self.rank}] torch.distributed initialized")

    def _initialize_megatron(self):
        """Initialize Megatron model parallel and engine."""
        # CRITICAL: Enable determinism FIRST, before ANY Megatron/TE imports
        # This must happen before FlashAttention code is loaded to take effect
        # Without this, consecutive forward passes differ by ~0.46 nats
        from tinker_server.backend.verl_patches import _enable_megatron_determinism
        _enable_megatron_determinism(seed=42)

        # Apply MLA patches for DeepseekV3/K2/Moonlight models BEFORE importing Megatron
        # These patches enable Flash Attention 2 with MLA by padding value tensors
        # Must be applied before MLASelfAttention class is imported/instantiated
        try:
            from verl.models.mcore.patch_v012 import apply_patch
            apply_patch()
            logger.info(f"[Rank {self.rank}] Applied MLA patches from verl.models.mcore.patch_v012")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Could not apply MLA patch: {e}")

        # Apply label shift patch to fix log_prob alignment
        # Must be applied BEFORE importing MegatronEngineWithLMHead
        try:
            from tinker_server.backend.verl_patches import apply_verl_patches
            apply_verl_patches()
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Could not apply verl patches: {e}")

        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
        from verl.workers.config import (
            HFModelConfig,
            McoreEngineConfig,
            McoreOptimizerConfig,
        )
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

        num_experts = getattr(hf_config, "num_experts", None) or getattr(hf_config, "n_routed_experts", None)
        try:
            model_uses_mla = get_model_config(self.base_model).is_mla
        except ValueError:
            # Unknown model, check HF config for MLA-specific attributes
            model_uses_mla = hasattr(hf_config, 'qk_nope_head_dim') and hasattr(hf_config, 'kv_lora_rank')

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

            # Determine target modules based on model architecture
            # MLA models (DeepSeek V3 / Moonlight / K2) use different attention projections
            if model_uses_mla:
                # MLA attention projections (DeepSeek V3 / Moonlight / K2)
                # Megatron module names (from named_parameters):
                # - linear_q_proj: Q projection
                # - linear_kv_down_proj: KV down projection
                # - linear_kv_up_proj: KV up projection
                # - linear_proj: output projection
                attention_modules = [
                    "linear_q_proj",
                    "linear_kv_down_proj", "linear_kv_up_proj",
                    "linear_proj",
                ]
                logger.info(f"[Rank {self.rank}] MLA model detected - using MLA target modules")
            else:
                # Standard attention projections
                # Megatron uses: linear_qkv (fused QKV), linear_proj (output)
                attention_modules = ["linear_qkv", "linear_proj"]

            target_modules = attention_modules + ["linear_fc1", "linear_fc2"]

            lora_config = LoraConfigDict(
                rank=self.lora_rank,
                alpha=self.lora_rank * 2,
                type="lora",
                target_modules=target_modules,
                dropout=0.0,
            )
            logger.info(f"[Rank {self.rank}] LoRA config: rank={self.lora_rank}, alpha={self.lora_rank * 2}, target_modules={target_modules}")

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
        if num_experts is not None:
            # MoE model - pass expert parameters to TransformerConfig
            override_tf_config["num_moe_experts"] = num_experts
            # moe_router_topk = num_experts_per_tok (active experts per token)
            num_experts_per_tok = getattr(hf_config, "num_experts_per_tok", 2)
            override_tf_config["moe_router_topk"] = num_experts_per_tok
            # need TE 2.1+
            override_tf_config["moe_permute_fusion"] = True
            override_tf_config["moe_shared_expert_overlap"] = True
            override_tf_config["moe_grouped_gemm"] = True
            logger.info(
                f"[Rank {self.rank}] MoE config: {num_experts} experts, "
                f"top-{num_experts_per_tok} routing, permute_fusion=True"
            )
            enable_deepep = (
                _env_flag("MINT_MEGATRON_ENABLE_DEEPEP", default=False)
                and int(getattr(self.config, "expert_parallel_size", 1) or 1) > 1
            )
            if enable_deepep:
                moe_token_dispatcher_type = (
                    os.environ.get("MINT_MEGATRON_MOE_TOKEN_DISPATCHER_TYPE", "flex").strip().lower()
                )
                override_tf_config["moe_token_dispatcher_type"] = moe_token_dispatcher_type
                override_tf_config["moe_flex_dispatcher_backend"] = (
                    os.environ.get("MINT_MEGATRON_MOE_FLEX_DISPATCHER_BACKEND", "deepep").strip().lower()
                )
                # NOTE: Megatron-LM still supports moe_enable_deepep but warns it's deprecated;
                # keep it alongside moe_flex_dispatcher_backend for compatibility.
                override_tf_config["moe_enable_deepep"] = True
                override_tf_config["moe_router_dtype"] = (
                    os.environ.get("MINT_MEGATRON_MOE_ROUTER_DTYPE", "fp32").strip().lower()
                )
                if moe_token_dispatcher_type != "alltoall":
                    if override_tf_config.get("moe_shared_expert_overlap") is True:
                        override_tf_config["moe_shared_expert_overlap"] = False
                        logger.info(
                            f"[Rank {self.rank}] Disabled moe_shared_expert_overlap for moe_token_dispatcher_type={moe_token_dispatcher_type}"
                        )
                sms = os.environ.get("MINT_MEGATRON_MOE_DEEPEP_NUM_SMS", "").strip()
                if sms:
                    override_tf_config["moe_deepep_num_sms"] = int(sms)
                logger.info(
                    f"[Rank {self.rank}] DeepEP enabled: dispatcher={override_tf_config['moe_token_dispatcher_type']}, "
                    f"backend={override_tf_config['moe_flex_dispatcher_backend']}, "
                    f"router_dtype={override_tf_config['moe_router_dtype']}, "
                    f"deepep_num_sms={override_tf_config.get('moe_deepep_num_sms', 'default')}"
                )

        override_tf_config["deallocate_pipeline_outputs"] = True
        grad_accum_fusion_available = False
        try:
            import fused_weight_gradient_mlp_cuda  # noqa: F401

            grad_accum_fusion_available = True
        except Exception:
            grad_accum_fusion_available = False
        override_tf_config["gradient_accumulation_fusion"] = grad_accum_fusion_available
        if not grad_accum_fusion_available:
            logger.info(
                f"[Rank {self.rank}] fused_weight_gradient_mlp_cuda not found; "
                "gradient_accumulation_fusion=False"
            )
        override_tf_config["persist_layer_norm"] = True
        override_tf_config["bias_activation_fusion"] = True
        override_tf_config["bias_dropout_fusion"] = True
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
            import torch

            major, minor = torch.cuda.get_device_capability()
            if (major, minor) < (8, 9):
                raise ValueError(
                    f"FP8 requested (train_use_fp8=True) but GPU compute capability is {major}.{minor}. "
                    "TransformerEngine FP8 requires compute capability >= 8.9."
                )
            # Use e4m3 format for FP8 (8-bit floating point with 4-bit exponent, 3-bit mantissa)
            # fp8_param=True stores parameters in FP8 precision for memory savings
            # use_cpu_initialization=True initializes on CPU first, then converts to FP8 before GPU transfer
            # This avoids OOM during BF16→FP8 conversion which would require both in GPU memory
            override_tf_config["fp8"] = "e4m3"
            override_tf_config["fp8_param"] = True
            override_tf_config["use_cpu_initialization"] = True
            logger.info(f"[Rank {self.rank}] FP8 enabled (format: e4m3, fp8_param=True, cpu_init=True) for memory-efficient training")

        # Activation recomputation (gradient checkpointing) for memory-constrained training
        # Required for long context training on large models (e.g., 40K tokens on 30B)
        try:
            use_gradient_checkpointing = get_model_config(self.base_model).gradient_checkpointing
        except ValueError:
            use_gradient_checkpointing = False
        if use_gradient_checkpointing:
            # Use FULL recomputation for maximum memory savings
            # For 40K context on 30B MoE, "selective" isn't enough - need full recompute
            # This recomputes ALL activations during backward pass, trading compute for memory
            override_tf_config["recompute_granularity"] = "full"
            override_tf_config["recompute_method"] = "uniform"
            override_tf_config["recompute_num_layers"] = 1
            logger.info(f"[Rank {self.rank}] Activation recomputation enabled (full: all layers)")

        # MLA attention (Multi-Latent Attention) detection for DeepSeekV3/K2/Moonlight models
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
        has_mla_attention = head_dim_qk > 0
        if has_mla_attention:
            logger.info(
                f"[Rank {self.rank}] MLA attention detected: head_dim_qk={head_dim_qk} "
                f"(qk_nope={qk_nope} + qk_rope={qk_rope}). Using FA2 with value padding."
            )

        # Activation checkpointing for memory-efficient training
        # When enabled, activations are checkpointed and recomputed during backward pass
        # This trades ~30-40% extra compute for significant memory savings on long sequences
        # Essential for K2 (1.04T params) with variable-length thinking outputs
        #
        # Available modules for selective recompute:
        # - "moe": recompute MoE layer (biggest saver for MoE models)
        # - "mla_up_proj": recompute MLA up projection and RoPE (important for K2)
        # - "shared_experts": recompute shared experts
        # - "core_attn": recompute core attention
        # - "layernorm": recompute layernorms
        # Only use selective recompute if full recompute wasn't enabled via model_registry
        if num_experts is not None and not use_gradient_checkpointing:
            override_tf_config["recompute_granularity"] = "selective"
            # Aggressive recompute for maximum memory savings:
            # - moe: MoE FFN activations (~40% of memory)
            # - mla_up_proj: MLA up projections for K2/DeepSeekV3 (~15% of memory)
            # NOTE: shared_experts conflicts with moe-shared-expert-overlap (enabled by default in new megatron-bridge)
            recompute_modules = ["moe"]
            # Add MLA-specific recompute for MLA models
            if has_mla_attention:
                recompute_modules.append("mla_up_proj")
            override_tf_config["recompute_modules"] = recompute_modules
            logger.info(f"[Rank {self.rank}] Selective recompute enabled: {recompute_modules}")

        logger.info(f"[Rank {self.rank}] override_transformer_config: {override_tf_config}")

        engine_kwargs: dict[str, object] = {
            "tensor_model_parallel_size": self.config.tensor_parallel_size,
            "pipeline_model_parallel_size": self.config.pipeline_parallel_size,
            "expert_model_parallel_size": self.config.expert_parallel_size,
            "expert_tensor_parallel_size": self.config.expert_tensor_parallel_size,
            "context_parallel_size": self.config.context_parallel_size,
            "param_offload": True,
            "optimizer_offload": True,
            "grad_offload": use_grad_offload,
            "dtype": "bfloat16",  # Base dtype, FP8 handled via override_transformer_config
            # THD ("remove padding") path in TransformerEngine disables FlashAttention when there is
            # any padding between sequences; verl's THD preprocessing pads sequences for alignment,
            # causing long-context training to fall back to O(seq^2) softmax and OOM at ~38K tokens.
            #
            # Disable remove-padding for non-MLA models so FlashAttention can be selected in BSHD.
            "use_remove_padding": has_mla_attention,
            "use_mbridge": True,
            "vanilla_mbridge": False,  # Required for LoRA - enables provider initialization
            "use_distributed_optimizer": True,  # Keep distributed optimizer for efficiency
            "override_transformer_config": override_tf_config,
        }

        # Compatibility: older verl branches do not expose EngineRouterReplayConfig nor
        # McoreEngineConfig.router_replay. Skip router replay config in that case.
        engine_fields = getattr(McoreEngineConfig, "__dataclass_fields__", {}) or {}
        if "router_replay" in engine_fields:
            router_replay_cfg = None
            try:
                from verl.workers.config.engine import EngineRouterReplayConfig

                router_replay_cfg = EngineRouterReplayConfig(mode=self.config.router_replay_mode)
            except Exception:
                try:
                    from verl.workers.config.actor import RouterReplayConfig

                    router_replay_cfg = RouterReplayConfig(mode=self.config.router_replay_mode)
                except Exception as e:
                    logger.warning(
                        f"[Rank {self.rank}] router_replay config unavailable; disabling router replay: {e}"
                    )
            if router_replay_cfg is not None:
                engine_kwargs["router_replay"] = router_replay_cfg
        elif self.config.router_replay_mode != "disabled":
            logger.warning(
                f"[Rank {self.rank}] McoreEngineConfig has no router_replay field; "
                "router replay mode ignored"
            )

        engine_config = McoreEngineConfig(**engine_kwargs)
        print(
            f"[Rank {self.rank}] McoreEngineConfig: TP={engine_config.tensor_model_parallel_size}, "
            f"EP={engine_config.expert_model_parallel_size}, ETP={engine_config.expert_tensor_parallel_size}, "
            f"CP={engine_config.context_parallel_size}, PP={engine_config.pipeline_model_parallel_size}",
            flush=True
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
        dummy_data["max_token_len_per_gpu"] = NonTensorData(get_model_config(self.base_model).max_model_len)
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

    def reset_expert_bias(self) -> dict:
        """Reset expert_bias buffers to zero in all MoE router modules.

        The expert_bias buffer accumulates during training (via finalize_model_grads)
        to balance token distribution across experts. However, this buffer is NOT
        exported with LoRA weights, causing train-inference mismatch:
        - Megatron (trained): has accumulated expert_bias != 0
        - vLLM (loaded LoRA): has expert_bias = 0

        This causes different routing decisions and thus different logprobs even
        with identical LoRA weights.

        Call this before computing logprobs to ensure consistent behavior with vLLM.

        Returns:
            dict with reset counts (only from rank 0)
        """
        import torch
        import sys
        import time

        # DEBUG: Write to stderr (should appear in Ray logs)
        print(f"[DEBUG {time.strftime('%H:%M:%S')}] reset_expert_bias ENTRY, rank={self.rank}", file=sys.stderr, flush=True)

        reset_count = 0
        bias_values_before = []

        # self.engine.module is a list (one per pipeline stage)
        for model_chunk in self.engine.module:
            # Iterate through all submodules
            for name, module in model_chunk.named_modules():
                if hasattr(module, 'expert_bias') and module.expert_bias is not None:
                    bias_before = module.expert_bias.clone()
                    if torch.any(bias_before != 0):
                        bias_values_before.append((name, bias_before.cpu().tolist()))
                    module.expert_bias.zero_()
                    reset_count += 1

                # Also reset local_tokens_per_expert if present
                if hasattr(module, 'local_tokens_per_expert') and module.local_tokens_per_expert is not None:
                    module.local_tokens_per_expert.zero_()

        # DEBUG: Write results to stderr
        print(f"[DEBUG {time.strftime('%H:%M:%S')}] reset_count={reset_count}, non_zero_before={len(bias_values_before)}, rank={self.rank}", file=sys.stderr, flush=True)

        if self.rank == 0:
            logger.info(f"[Rank {self.rank}] Reset {reset_count} expert_bias buffers to zero")
            if bias_values_before:
                logger.info(f"[Rank {self.rank}] Non-zero expert_bias found before reset: {len(bias_values_before)} modules")
                for name, vals in bias_values_before[:3]:  # Log first 3
                    logger.info(f"[Rank {self.rank}]   {name}: {vals[:5]}...")  # First 5 values

        return {"reset_count": reset_count} if self.rank == 0 else {}

    def _resolve_reset_bias(self, reset_bias: bool | None, default: bool) -> bool:
        if reset_bias is not None:
            return reset_bias
        return _env_flag("MINT_RESET_EXPERT_BIAS", default=default)

    def _is_output_rank(self) -> bool:
        """Return True if this rank should emit metrics/logprobs."""
        try:
            from megatron.core import mpu

            return (
                mpu.is_pipeline_last_stage(ignore_virtual=True)
                and mpu.get_data_parallel_rank() == 0
                and mpu.get_tensor_model_parallel_rank() == 0
            )
        except Exception:
            return self.rank == 0

    def check_determinism_status(self) -> dict:
        """Check determinism settings on this worker.

        Returns:
            dict with determinism status info (only from rank 0)
        """
        import torch
        import os
        import glob

        status = {
            "rank": self.rank,
            "hostname": socket.gethostname(),
            "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "FLASH_ATTENTION_DETERMINISTIC": os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "NOT SET"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET"),
            "NCCL_DETERMINISTIC": os.environ.get("NCCL_DETERMINISTIC", "NOT SET"),
        }

        # Check for status files
        status_files = glob.glob("/tmp/determinism_status_*.txt")
        status["status_files"] = status_files

        if self.rank == 0:
            logger.info(f"[Rank 0] Determinism status: {status}")

        return status if self.rank == 0 else {}

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str,
        loss_fn_config: dict,
        rollout_correction_config: dict | None = None,
        session_id: str | None = None,
        reset_bias: bool | None = None,
    ) -> dict:
        """Run forward and backward pass on this rank's shard.

        Gradients are synchronized via NCCL allreduce.
        Returns metrics from rank 0 only, including per-sample loss_fn_outputs.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type ("cross_entropy", "importance_sampling", "ppo").
            loss_fn_config: Config for loss function (e.g., {"epsilon": 0.2} for PPO).
            rollout_correction_config: Optional verl rollout correction config passed to policy_loss.
            session_id: Optional session ID for gradient isolation.
            reset_bias: If True, reset expert_bias to zero before forward pass.
                If None, uses MINT_RESET_EXPERT_BIAS (default False for training).
                This ensures logprobs match vLLM (which always has bias=0).

        Note: TensorDict with nested tensors is created locally to avoid Ray
        serialization issues that cause CUDA memory corruption.
        """
        import torch
        from tinker_server.backend.megatron_training import (
            create_sft_loss_fn, create_ppo_loss_fn, tinker_to_tensordict
        )

        reset_bias = self._resolve_reset_bias(reset_bias, default=False)
        output_rank = self._is_output_rank()

        # Reset expert_bias for train-inference consistency
        # expert_bias accumulates during training but isn't exported to vLLM
        # Reset ensures Megatron routing matches vLLM routing for correct KL calculation
        if reset_bias:
            self.reset_expert_bias()

        # Session state swap moved inside train_mode() - see below

        seq_lengths: list[int] = []
        for item_index, item in enumerate(data_items):
            model_input = item.get("model_input", {})
            tokens = flatten_encoded_text_chunks(model_input)
            if not tokens:
                raise ValueError(f"Item {item_index}: model_input has no tokens")
            seq_lengths.append(len(tokens))

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
        max_token_len = get_model_config(self.base_model).max_model_len
        data = tinker_to_tensordict(data_items, max_token_len_per_gpu=max_token_len, device=f"cuda:{device}")

        # Log memory before forward-backward
        self.log_memory_breakdown("before_forward_backward")

        # Select loss function (SFT returns log_probs in metrics for train_nll)
        if loss_fn == "cross_entropy":
            loss_function = create_sft_loss_fn(return_logprobs=True)
        elif loss_fn == "ppo":
            epsilon = loss_fn_config.get("epsilon", 0.2)
            loss_function = create_ppo_loss_fn(
                epsilon,
                rollout_correction_config=rollout_correction_config,
            )
        elif loss_fn == "importance_sampling":
            loss_function = create_ppo_loss_fn(
                epsilon=float("inf"),
                rollout_correction_config=rollout_correction_config,
            )
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        result = None
        train_mode_enter_ms = 0.0
        train_mode_exit_ms = 0.0
        train_mode_reused = False
        grad_restore_skipped = False
        forward_backward_batch_ms = 0.0

        def _run_forward_backward_compute():
            nonlocal result, forward_backward_batch_ms
            t_fb0 = time.perf_counter()
            watchdog = self._start_slow_op_watchdog(
                op="forward_backward_batch",
                session_id=session_id,
                extra=f"items={len(data_items)} loss_fn={loss_fn}",
            )
            try:
                result = self.engine.forward_backward_batch(
                    data=data,
                    loss_function=loss_function,
                    forward_only=False,
                )
            finally:
                self._stop_slow_op_watchdog(watchdog)
            forward_backward_batch_ms = (time.perf_counter() - t_fb0) * 1000.0

            if result:
                logger.debug(
                    f"[Rank {self.rank} DEBUG] forward_backward_batch result keys: "
                    f"{list(result.keys()) if isinstance(result, dict) else type(result)}"
                )
                if isinstance(result, dict):
                    losses = result.get("loss", [])
                    logger.debug(f"[Rank {self.rank} DEBUG] losses: {losses}")
                    metrics = result.get("metrics", {})
                    logger.debug(f"[Rank {self.rank} DEBUG] metrics keys: {list(metrics.keys())}")
                    log_probs_list = metrics.get("log_probs", [])
                    logger.debug(
                        f"[Rank {self.rank} DEBUG] log_probs_list len: {len(log_probs_list)}, "
                        f"first is None: {log_probs_list[0] is None if log_probs_list else 'empty'}"
                    )
            else:
                logger.debug(f"[Rank {self.rank} DEBUG] forward_backward_batch returned empty/None result")

            self.log_memory_breakdown("after_forward_backward")

        # Use train_mode context to load model from CPU to GPU (required for param_offload)
        # The context manager handles: load to GPU on __enter__, offload to CPU on __exit__
        # IMPORTANT: train_mode() entry zeros gradients via load_megatron_model_to_gpu().
        # For gradient isolation: restore cached grads after entry, capture before exit.
        if self._sticky_enabled_for(session_id):
            sticky = self._ensure_sticky_train_mode(session_id=session_id, reason="forward_backward")
            train_mode_reused = bool(sticky.get("reused", False))
            train_mode_enter_ms = float(sticky.get("enter_s", 0.0)) * 1000.0

            cached_grads = self._session_gradients.get(session_id)
            if cached_grads is not None and cached_grads is not _GRADIENTS_CONSUMED and not train_mode_reused:
                self._restore_gradients(cached_grads)
                logger.debug(f"[Rank {self.rank}] Restored gradients for session {session_id}")
            elif train_mode_reused:
                grad_restore_skipped = True
                logger.debug(f"[Rank {self.rank}] Reused sticky train_mode for session {session_id}, skip gradient restore")
            else:
                logger.debug(
                    f"[Rank {self.rank}] No cached gradients for session {session_id}, "
                    "using zeroed grads from train_mode entry"
                )

            try:
                _run_forward_backward_compute()
            except Exception as original_error:
                # On error, GPU state is undefined -- release sticky context without
                # snapshotting gradients (would persist garbage).
                # Wrap cleanup in try/except to prevent cleanup errors from masking
                # the original business error.
                try:
                    self._release_sticky_train_mode(
                        reason="forward_backward_error",
                        snapshot_gradients=False,
                    )
                except Exception as cleanup_error:
                    logger.error(
                        f"[Rank {self.rank}] Failed to release sticky train_mode during "
                        f"forward_backward error cleanup: {cleanup_error}",
                        exc_info=True,
                    )
                # Always re-raise the original error, not the cleanup error
                raise original_error
        else:
            tm_enter_t0 = time.perf_counter()
            with self.engine.train_mode():
                train_mode_enter_ms = (time.perf_counter() - tm_enter_t0) * 1000.0
                cached_grads = self._session_gradients.get(session_id) if session_id else None
                if cached_grads is not None and cached_grads is not _GRADIENTS_CONSUMED:
                    self._restore_gradients(cached_grads)
                    logger.debug(f"[Rank {self.rank}] Restored gradients for session {session_id}")
                else:
                    logger.debug(
                        f"[Rank {self.rank}] No cached gradients for session {session_id}, "
                        "using zeroed grads from train_mode entry"
                    )

                _run_forward_backward_compute()

                # Capture gradients BEFORE exiting train_mode (exit destroys GPU grads)
                if session_id is not None:
                    self._session_gradients[session_id] = self._capture_gradients()
                    logger.debug(f"[Rank {self.rank}] Captured gradients for session {session_id}")
                tm_exit_t0 = time.perf_counter()
            train_mode_exit_ms = (time.perf_counter() - tm_exit_t0) * 1000.0

        # Only one output rank returns metrics
        if output_rank:
            loss_value = 0.0
            num_tokens = 0
            clip_frac_sum = 0.0
            ratio_mean_sum = 0.0
            n_ppo_results = 0
            all_log_probs = []
            loss_fn_outputs = []
            per_sample_log_probs = None

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

                model_output = result.get("model_output", {})
                logger.info(f"[Rank {self.rank}] model_output keys: {list(model_output.keys())}")
                model_log_probs = model_output.get("log_probs")
                # Extract top-k tensors if present (from verl_patches.py)
                model_topk_indices = model_output.get("topk_indices")  # (batch, seq_len, k)
                model_topk_logits = model_output.get("topk_logits")    # (batch, seq_len, k)
                if model_topk_indices is not None:
                    logger.info(f"[Rank {self.rank}] Got topk_indices shape={model_topk_indices.shape}")
                else:
                    logger.info(f"[Rank {self.rank}] topk_indices is None")
                if model_log_probs is not None:
                    if hasattr(model_log_probs, "unbind"):
                        per_sample_log_probs = [lp.detach().cpu() for lp in model_log_probs.unbind()]
                    elif seq_lengths and hasattr(model_log_probs, "dim") and model_log_probs.dim() >= 2:
                        per_sample_log_probs = []
                        for idx, row in enumerate(model_log_probs):
                            seq_len = seq_lengths[idx] if idx < len(seq_lengths) else row.shape[0]
                            per_sample_log_probs.append(row[:seq_len].detach().cpu())

            # Concatenate and split log_probs into per-sample tensors for loss_fn_outputs
            # Also split topk tensors which are in THD format (1, total_tokens, k)
            topk_offset = 0  # Track offset into concatenated topk tensor

            # NaN guard: detect and report NaN/Inf in loss_value before it propagates to JSON
            # (orjson converts NaN to null, which causes pydantic validation failures on client)
            if math.isnan(loss_value) or math.isinf(loss_value):
                logger.error(f"[Rank {self.rank}] NaN/Inf detected in loss_value={loss_value}. "
                             f"num_tokens={num_tokens}, loss_fn={loss_fn}. "
                             "This will cause client-side validation failures.")
                # Set to a large but valid number to allow training to continue with a warning
                loss_value = 1e6

            if per_sample_log_probs:
                avg_loss_per_sample = loss_value / max(len(per_sample_log_probs), 1)
                for sample_idx, sample_log_probs in enumerate(per_sample_log_probs):
                    logprobs_list = sample_log_probs.tolist()
                    seq_len = len(logprobs_list)
                    output_entry = {
                        "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                        "logprobs": {"data": logprobs_list, "shape": [seq_len], "dtype": "float32"},
                    }
                    # Add top-k if available
                    # topk tensors have shape (1, total_tokens, k) - all sequences concatenated
                    if model_topk_indices is not None and model_topk_logits is not None:
                        # Check if THD format (batch=1, concatenated sequences)
                        if model_topk_indices.dim() == 3 and model_topk_indices.shape[0] == 1:
                            # Convert to int to avoid symbolic shape comparison issues
                            total_tokens = int(model_topk_indices.shape[1])
                            k = int(model_topk_indices.shape[2])
                            if topk_offset + seq_len <= total_tokens:
                                # TensorData.data must be flattened (per tinker schema)
                                topk_idx = model_topk_indices[0, topk_offset:topk_offset + seq_len, :].flatten().tolist()
                                topk_lp = model_topk_logits[0, topk_offset:topk_offset + seq_len, :].flatten().tolist()
                                output_entry["topk_indices"] = {"data": topk_idx, "shape": [seq_len, k], "dtype": "int64"}
                                output_entry["topk_logits"] = {"data": topk_lp, "shape": [seq_len, k], "dtype": "float32"}
                                topk_offset += seq_len
                        elif sample_idx < model_topk_indices.shape[0]:
                            # Per-sample format: (num_samples, seq_len, k)
                            k = model_topk_indices.shape[2]
                            # TensorData.data must be flattened (per tinker schema)
                            topk_idx = model_topk_indices[sample_idx, :seq_len, :].flatten().tolist()
                            topk_lp = model_topk_logits[sample_idx, :seq_len, :].flatten().tolist()
                            output_entry["topk_indices"] = {"data": topk_idx, "shape": [seq_len, k], "dtype": "int64"}
                            output_entry["topk_logits"] = {"data": topk_lp, "shape": [seq_len, k], "dtype": "float32"}
                    loss_fn_outputs.append(output_entry)
            elif loss_fn == "cross_entropy" and all_log_probs and seq_lengths:
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

            valid_count = len(seq_lengths)
            expected_outputs = valid_count
            if expected_outputs and len(loss_fn_outputs) < expected_outputs:
                avg_loss_per_sample = loss_value / max(expected_outputs, 1)
                for _ in range(expected_outputs - len(loss_fn_outputs)):
                    loss_fn_outputs.append({
                        "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                        "logprobs": {"data": [], "shape": [0], "dtype": "float32"},
                    })

            # Calculate precision difference metrics if log_probs present
            debug_metrics = {}
            if result and isinstance(result, dict):
                metrics = result.get("metrics", {})
                logger.debug(f"[Rank {self.rank} DEBUG] Available metric keys: {list(metrics.keys())}")

                # Debug metrics are stored as lists (one per micro-batch)
                # Take the last value from each list (most recent micro-batch)
                for key in ["training/rollout_probs_diff_valid", "training/rollout_probs_diff_mean",
                           "training/rollout_probs_diff_max", "training/rollout_probs_diff_std",
                           "training/rollout_actor_probs_pearson_corr"]:
                    if key in metrics:
                        values = metrics[key]
                        logger.debug(f"[Rank {self.rank} DEBUG] Found {key}: {values}")
                        # metrics[key] is a list, take the last (or average)
                        if isinstance(values, list) and values:
                            # Average debug metrics across micro-batches
                            if isinstance(values[0], (int, float)):
                                # Filter out nan values before averaging (math imported at module level)
                                valid_values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
                                if valid_values:
                                    debug_metrics[f"{key}:mean"] = sum(valid_values) / len(valid_values)
                                else:
                                    debug_metrics[f"{key}:mean"] = 0.0
                            else:
                                # If not numeric, just take the last value
                                debug_metrics[f"{key}:mean"] = values[-1]
                        else:
                            # Skip if values is None, empty list, or non-numeric
                            if values is not None and isinstance(values, (int, float)):
                                debug_metrics[f"{key}:mean"] = float(values)
                    else:
                        logger.debug(f"[Rank {self.rank} DEBUG] Key {key} NOT found in metrics")

            logger.debug(f"[Rank {self.rank} DEBUG] Extracted debug_metrics: {debug_metrics}")

            # Return CPU-safe scalars and loss_fn_outputs
            routing_replay_enabled = int("routed_experts" in data.keys())
            routing_replay_items = int(valid_count) if routing_replay_enabled else 0
            result_dict = {
                "loss_value": float(loss_value),
                "num_tokens": int(num_tokens),
                "clip_frac_sum": float(clip_frac_sum),
                "ratio_mean_sum": float(ratio_mean_sum),
                "n_ppo_results": int(n_ppo_results),
                "valid_count": int(valid_count),
                "loss_fn_outputs": loss_fn_outputs,
                "routing_replay_enabled": routing_replay_enabled,
                "routing_replay_items": routing_replay_items,
                "train_mode_enter_ms": float(train_mode_enter_ms),
                "train_mode_exit_ms": float(train_mode_exit_ms),
                "train_mode_reused": float(1.0 if train_mode_reused else 0.0),
                "grad_restore_skipped": float(1.0 if grad_restore_skipped else 0.0),
                "forward_backward_batch_ms": float(forward_backward_batch_ms),
                "train_mode_enter_total": float(self._sticky_train_mode_enter_total),
                "train_mode_reuse_total": float(self._sticky_train_mode_reuse_total),
                "train_mode_exit_total": float(self._sticky_train_mode_exit_total),
            }
            # Add debug metrics if present
            if debug_metrics:
                result_dict["debug_metrics"] = debug_metrics
                logger.debug(f"[Rank {self.rank}] Debug metrics: {debug_metrics}")
            else:
                logger.debug(f"[Rank {self.rank} DEBUG] No debug metrics to add!")
            return result_dict
        return {}

    def forward(
        self,
        data_items: list[dict],
        reset_bias: bool | None = None,
    ) -> dict:
        """Run forward pass only (no backward). Returns per-token logprobs.

        Similar to forward_backward but skips gradient computation.
        Used for computing reference model logprobs in DPO/SL.
        Returns per-token log probabilities from rank 0 only.

        Important: Creates one loss_fn_outputs entry per sample (not per micro-batch)
        to match cookbook's NLL evaluator expectations.

        Note: TensorDict with nested tensors is created locally to avoid Ray
        serialization issues that cause CUDA memory corruption.

        Args:
            data_items: List of data items with model_input and loss_fn_inputs.
            reset_bias: If True, reset expert_bias to zero before forward pass.
                If None, uses MINT_RESET_EXPERT_BIAS (default True for logprob-only).
                This ensures logprobs match vLLM (which always has bias=0).
        """
        import torch
        from tinker_server.backend.megatron_training import tinker_to_tensordict

        reset_bias = self._resolve_reset_bias(reset_bias, default=True)
        output_rank = self._is_output_rank()

        # Reset expert_bias for train-inference consistency
        # expert_bias accumulates during training but isn't exported to vLLM
        # Reset ensures Megatron routing matches vLLM routing
        if reset_bias:
            self.reset_expert_bias()

        seq_lengths: list[int] = []
        for item_index, item in enumerate(data_items):
            model_input = item.get("model_input", {})
            tokens = flatten_encoded_text_chunks(model_input)
            if not tokens:
                raise ValueError(f"Item {item_index}: model_input has no tokens")
            seq_lengths.append(len(tokens))

        # Create TensorDict directly on GPU to avoid .to() issues with nested tensors
        device = torch.cuda.current_device()
        max_token_len = get_model_config(self.base_model).max_model_len
        data = tinker_to_tensordict(data_items, max_token_len_per_gpu=max_token_len, device=f"cuda:{device}")

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

        # Only one output rank returns metrics
        if output_rank:
            valid_count = len(seq_lengths)
            loss_value = 0.0
            num_tokens = 0
            all_log_probs = []
            loss_fn_outputs = []
            per_sample_log_probs = None
            combined_log_probs = None

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

                model_output = result.get("model_output", {})
                model_log_probs = model_output.get("log_probs")
                if model_log_probs is not None:
                    if hasattr(model_log_probs, "unbind"):
                        per_sample_log_probs = [lp.detach().cpu() for lp in model_log_probs.unbind()]
                    elif seq_lengths and hasattr(model_log_probs, "dim") and model_log_probs.dim() >= 2:
                        per_sample_log_probs = []
                        for idx, row in enumerate(model_log_probs):
                            seq_len = seq_lengths[idx] if idx < len(seq_lengths) else row.shape[0]
                            per_sample_log_probs.append(row[:seq_len].detach().cpu())

                if per_sample_log_probs:
                    combined_log_probs = (
                        torch.cat(per_sample_log_probs, dim=0)
                        if len(per_sample_log_probs) > 1
                        else per_sample_log_probs[0]
                    )
                    avg_loss_per_sample = loss_value / max(len(per_sample_log_probs), 1)
                    for sample_log_probs in per_sample_log_probs:
                        logprobs_list = sample_log_probs.tolist()
                        loss_fn_outputs.append({
                            "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                            "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                        })
                else:
                    # Fallback to metrics order (may be shuffled under dynamic bsz)
                    if all_log_probs:
                        combined_log_probs = (
                            torch.cat(all_log_probs, dim=0) if len(all_log_probs) > 1 else all_log_probs[0]
                        )
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
                "valid_count": int(valid_count),
                "loss_fn_outputs": loss_fn_outputs,
                "log_probs": combined_log_probs,  # Combined per-token log_probs tensor (for backward compat)
            }
        return {}

    def optim_step(
        self,
        learning_rate: float,
        session_id: str | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Run optimizer step (synchronized across ranks).

        Restores session's cached gradients before applying optimizer step.
        Clears cache after - gradients are consumed.

        Note: Session state swap (gradients + optimizer) is handled by
        MegatronWorkerGroup._ensure_session_loaded() calling swap_session_state()
        BEFORE forward_backward/optim_step. This method only needs to restore
        this session's cached gradients from the most recent forward_backward.
        """
        train_attn = True if train_attn is None else bool(train_attn)
        train_mlp = True if train_mlp is None else bool(train_mlp)
        train_unembed = True if train_unembed is None else bool(train_unembed)
        train_mode_enter_ms = 0.0
        train_mode_exit_ms = 0.0
        train_mode_reused = False
        grad_restore_skipped = False
        optim_step_batch_ms = 0.0

        def _run_optim_core():
            nonlocal grad_norm, current_lr, optim_step_batch_ms
            t_opt0 = time.perf_counter()
            watchdog = self._start_slow_op_watchdog(
                op="optim_step_core",
                session_id=session_id,
                extra=f"lr={learning_rate}",
            )
            try:

                # DEBUG: Check optimizer state size BEFORE step
                if self.rank == 0:
                    from megatron.core.optimizer import ChainedOptimizer

                    optimizer = self.engine.optimizer
                    if optimizer is not None:
                        def iter_optimizers(opt):
                            if isinstance(opt, ChainedOptimizer):
                                return opt.chained_optimizers
                            return [opt]

                        for i, _opt in enumerate(iter_optimizers(optimizer)):
                            if hasattr(_opt, "optimizer") and _opt.optimizer is not None:
                                inner_opt = _opt.optimizer
                                state = inner_opt.state
                                state_size = len(state) if hasattr(state, "__len__") else "unknown"
                                first_state = None
                                if state:
                                    first_param = next(iter(state), None)
                                    if first_param is not None:
                                        first_state = {
                                            k: v.shape if hasattr(v, "shape") else v
                                            for k, v in state[first_param].items()
                                        }
                                print(
                                    f"[Rank {self.rank}] optim_step BEFORE: opt[{i}] state size = {state_size}, "
                                    f"first_state_keys = {first_state}",
                                    flush=True,
                                )

                grad_norm = self.engine.optimizer_step()
                current_lr = self.engine.lr_scheduler_step()

                try:
                    from verl.utils.megatron_utils import unwrap_model

                    model = unwrap_model(self.engine.module)
                    while isinstance(model, list):
                        model = model[0]
                    self._zero_disabled_lora_params(
                        model, train_attn=train_attn, train_mlp=train_mlp, train_unembed=train_unembed
                    )
                except Exception:
                    pass

                self.log_memory_breakdown("after_optim_step")
                optim_step_batch_ms = (time.perf_counter() - t_opt0) * 1000.0

                if session_id is not None:
                    self._session_gradients[session_id] = _GRADIENTS_CONSUMED
                    logger.debug(f"[Rank {self.rank}] Marked gradients as consumed for session {session_id}")
            finally:
                self._stop_slow_op_watchdog(watchdog)

        grad_norm = None
        current_lr = None
        if self._sticky_enabled_for(session_id):
            sticky = self._ensure_sticky_train_mode(session_id=session_id, reason="optim_step")
            train_mode_reused = bool(sticky.get("reused", False))
            train_mode_enter_ms = float(sticky.get("enter_s", 0.0)) * 1000.0

            cached_grads = self._session_gradients.get(session_id)
            if cached_grads is not None and cached_grads is not _GRADIENTS_CONSUMED and not train_mode_reused:
                self._restore_gradients(cached_grads)
                logger.debug(f"[Rank {self.rank}] Restored gradients for optim_step, session {session_id}")
            elif train_mode_reused:
                grad_restore_skipped = True
                logger.debug(
                    f"[Rank {self.rank}] Reused sticky train_mode for optim_step, "
                    f"session {session_id}, skip gradient restore"
                )

            try:
                _run_optim_core()
            except Exception as original_error:
                # On error, GPU state is undefined -- release sticky context without
                # snapshotting gradients (would persist garbage).
                # Wrap cleanup in try/except to prevent cleanup errors from masking
                # the original business error.
                try:
                    self._release_sticky_train_mode(
                        reason="optim_step_error",
                        snapshot_gradients=False,
                    )
                except Exception as cleanup_error:
                    logger.error(
                        f"[Rank {self.rank}] Failed to release sticky train_mode during "
                        f"optim_step error cleanup: {cleanup_error}",
                        exc_info=True,
                    )
                # Always re-raise the original error, not the cleanup error
                raise original_error

            if self._sticky_train_mode_close_on_optim:
                released = self._release_sticky_train_mode(
                    reason="optim_step_complete",
                    snapshot_gradients=False,
                )
                train_mode_exit_ms = float(released.get("exit_s", 0.0)) * 1000.0
        else:
            tm_enter_t0 = time.perf_counter()
            with self.engine.train_mode():
                train_mode_enter_ms = (time.perf_counter() - tm_enter_t0) * 1000.0
                cached_grads = self._session_gradients.get(session_id) if session_id else None
                if cached_grads is not None and cached_grads is not _GRADIENTS_CONSUMED:
                    self._restore_gradients(cached_grads)
                    logger.debug(f"[Rank {self.rank}] Restored gradients for optim_step, session {session_id}")
                _run_optim_core()
                tm_exit_t0 = time.perf_counter()
            train_mode_exit_ms = (time.perf_counter() - tm_exit_t0) * 1000.0

        if self.rank == 0:
            # Handle current_lr being either a float or a list
            if current_lr is not None:
                lr_value = current_lr[0] if isinstance(current_lr, (list, tuple)) else current_lr
            else:
                lr_value = learning_rate
            print(f"[Rank {self.rank}] optim_step: grad_norm={grad_norm}, lr={lr_value}, session={session_id}", flush=True)
            # Return CPU-safe scalars only
            return {
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "lr": float(lr_value),
                "step": "completed",
                "train_mode_enter_ms": float(train_mode_enter_ms),
                "train_mode_exit_ms": float(train_mode_exit_ms),
                "train_mode_reused": float(1.0 if train_mode_reused else 0.0),
                "grad_restore_skipped": float(1.0 if grad_restore_skipped else 0.0),
                "optim_step_batch_ms": float(optim_step_batch_ms),
                "train_mode_enter_total": float(self._sticky_train_mode_enter_total),
                "train_mode_reuse_total": float(self._sticky_train_mode_reuse_total),
                "train_mode_exit_total": float(self._sticky_train_mode_exit_total),
            }
        return {}

    def get_lora_state_dict(self, use_per_expert_lora: bool = False) -> dict:
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

        Args:
            use_per_expert_lora: If True, expand shared MLP LoRA to per-expert format
                for vLLM FusedMoEWithLoRA compatibility. If False (default), filter out
                MLP/expert modules for MoE models to avoid weight format mismatch.

        MoE Weight Format Issue:
            Megatron-Bridge exports shared LoRA weights (single adapter applied to all experts):
                decoder.layers.X.mlp.experts.linear_fc1.adapter.linear_in.weight

            vLLM FusedMoEWithLoRA expects per-expert weights:
                base_model.model.model.layers.X.mlp.experts.0.gate_proj.lora_A.weight
                base_model.model.model.layers.X.mlp.experts.1.gate_proj.lora_A.weight
                ...

            When use_per_expert_lora=True, we expand the shared weights to per-expert format
            by replicating the shared weights for each expert. This ensures train-inference
        consistency but may use more memory during inference.
        """
        import os

        logger.info(f"[Rank {self.rank}] get_lora_state_dict: ENTRY")

        # ========== Try HollowMan Megatron-Bridge export_adapter_weights API ==========
        # This is enabled via USE_MBRIDGE_LORA_EXPORT=true environment variable which
        # prepends HollowMan fork to PYTHONPATH
        bridge = getattr(self.engine, 'bridge', None)
        try:
            model_is_moe = get_model_config(self.base_model).is_moe
        except Exception:
            model_is_moe = False

        # For MoE models when we are NOT exporting per-expert LoRA (train_mlp=False),
        # avoid export_adapter_weights: it materializes the entire MLP LoRA tree
        # (hundreds of thousands of tensors) even though we will filter it out.
        if model_is_moe and not use_per_expert_lora and bridge is not None and hasattr(bridge, 'export_adapter_weights'):
            logger.info(
                f"[Rank {self.rank}] Skipping Megatron-Bridge export_adapter_weights API (MoE attention-only export)"
            )
        if bridge is not None and hasattr(bridge, 'export_adapter_weights') and not (model_is_moe and not use_per_expert_lora):
            logger.info(f"[Rank {self.rank}] Using Megatron-Bridge export_adapter_weights API")
            adapter_state = {}

            # Default to exporting a shared-expert LoRA artifact (expert 0 only) for MoE models.
            # vLLM hot-load will broadcast the shared expert weights at load time.
            shared_mode = os.environ.get("MINT_MOE_LORA_SHARED_EXPERT_EXPORT", "1").strip().lower()
            # Default to exporting the full per-expert LoRA tree so vLLM can pack MoE LoRAs.
            # Operators can force a "shared expert" export (keep only expert 0) to reduce
            # artifact size, but this requires downstream broadcasting support.
            shared_export = (
                bool(model_is_moe)
                and bool(use_per_expert_lora)
                and shared_mode in {"1", "true", "yes"}
            )
            import re

            expert_pat = re.compile(r"\.mlp\.(?:shared_)?experts\.(\d+)\.")

            # Debug: verify whether "shared" expert LoRA weights are actually identical across
            # expert-parallel (EP) shards. If they differ, "expert-0-only" export is not valid.
            #
            # We only gather small checksum scalars (not full tensors) to keep overhead low.
            ep_group = None
            ep_size = 1
            ep_rank = 0
            try:
                from megatron.core import parallel_state as mpu

                ep_size = int(mpu.get_expert_model_parallel_world_size())
                ep_rank = int(mpu.get_expert_model_parallel_rank())
                if ep_size > 1:
                    ep_group = mpu.get_expert_model_parallel_group()
            except Exception:
                ep_group = None
                ep_size = 1
                ep_rank = 0

            did_ep_checksum = False
            with self.engine.eval_mode():
                for name, tensor in bridge.export_adapter_weights(self.engine.module, cpu=True, show_progress=False):
                    if (
                        (not did_ep_checksum)
                        and ep_group is not None
                        and ep_size > 1
                        and use_per_expert_lora
                        and ".mlp.experts." in name
                        and ".mlp.shared_experts." not in name
                        and name.endswith(".adapter.linear_in.weight")
                    ):
                        import torch
                        import torch.distributed as dist

                        # tensor is already on CPU here; compute checksum on CPU, then all_gather
                        # the tiny checksum tensor on GPU/NCCL.
                        t = tensor.detach()
                        chk = torch.tensor(
                            [
                                float(ep_rank),
                                float(t.float().sum().item()),
                                float(t.float().mean().item()),
                                float(t.float().abs().max().item()),
                            ],
                            device=torch.cuda.current_device(),
                            dtype=torch.float32,
                        )
                        gathered = [torch.empty_like(chk) for _ in range(ep_size)]
                        dist.all_gather(gathered, chk, group=ep_group)
                        if self.rank == 0:
                            rows = [
                                (
                                    int(g[0].item()),
                                    float(g[1].item()),
                                    float(g[2].item()),
                                    float(g[3].item()),
                                )
                                for g in gathered
                            ]
                            rows = sorted(rows, key=lambda x: x[0])
                            logger.info(
                                f"[Rank 0] EP LoRA checksum probe ({name}): "
                                f"ep_size={ep_size} rows={rows}"
                            )
                        did_ep_checksum = True
                    if self.rank == 0:
                        if shared_export:
                            m = expert_pat.search(name)
                            if m is not None and int(m.group(1)) != 0:
                                continue
                        adapter_state[name] = tensor.clone()

            # Non-rank-0 workers return empty dict
            if self.rank != 0:
                logger.info(f"[Rank {self.rank}] get_lora_state_dict: returning empty dict (non-rank-0)")
                return {}

            if model_is_moe and not use_per_expert_lora:
                before = len(adapter_state)
                adapter_state = {
                    k: v
                    for k, v in adapter_state.items()
                    if (".mlp." not in k and ".experts." not in k)
                }
                logger.info(f"[Rank 0] Filtered {before - len(adapter_state)} MLP/expert params (export_adapter_weights)")

            # For MoE models exporting per-expert LoRA, the full per-expert tree can be
            # extremely large (K2: ~140k tensors, ~77GB).
            #
            # Current vLLM hot-load support in this repo only supports:
            # - full per-expert exports (all experts present), OR
            # - "shared-expert" exports (expert 0 only) where vLLM broadcasts expert 0
            #   weights to all experts at load time.
            #
            # A "one representative expert per EP shard" sparse export is not supported
            # by the current vLLM loader patch (it fail-fast) because correct filling
            # requires EP shard aware mapping.
            if model_is_moe and use_per_expert_lora:
                import os
                import re

                if os.environ.get("MINT_MOE_LORA_SPARSE_EXPERT_EXPORT", "1").strip() != "0":
                    # Sparse expert export: keep one representative expert per EP shard (linear placement).
                    #
                    # This reduces the export artifact size drastically (K2 full per-expert adapters are very large).
                    # The downstream vLLM hot-load path must reconstruct full per-expert tensors at load time.
                    #
                    # If MINT_MOE_LORA_SHARED_EXPERT_EXPORT=1 is also enabled, we additionally drop to expert-0-only.

                    # Determine EP size.
                    try:
                        from megatron.core import parallel_state as mpu

                        ep_size = int(mpu.get_expert_model_parallel_world_size())
                    except Exception:
                        ep_size = 1

                    # Determine num_experts.
                    num_experts = 0
                    if ep_size > 1:
                        try:
                            from transformers import AutoConfig

                            hf_config = AutoConfig.from_pretrained(
                                self.base_model, trust_remote_code=True
                            )
                            num_experts = int(
                                getattr(hf_config, "num_experts", None)
                                or getattr(hf_config, "n_routed_experts", 0)
                                or 0
                            )
                        except Exception:
                            num_experts = 0

                    if ep_size > 1 and num_experts > 0:
                        base = num_experts // ep_size
                        rem = num_experts % ep_size
                        reps = {
                            (r * base + min(r, rem))
                            for r in range(ep_size)
                            if (base + (1 if r < rem else 0)) > 0
                        }

                        # Keep all non-expert keys + only representative expert keys.
                        expert_pat = re.compile(r"\.mlp\.experts\.(\d+)\.")
                        before = len(adapter_state)
                        dropped = 0
                        filtered = {}
                        for k, v in adapter_state.items():
                            m = expert_pat.search(k)
                            if m is None:
                                filtered[k] = v
                                continue
                            idx = int(m.group(1))
                            if idx in reps:
                                filtered[k] = v
                            else:
                                dropped += 1

                        adapter_state = filtered
                        logger.info(
                            f"[Rank 0] Sparse expert export (export_adapter_weights): "
                            f"ep_size={ep_size} num_experts={num_experts} kept_experts={len(reps)} "
                            f"dropped_keys={dropped} before={before} after={len(adapter_state)}"
                        )

                # Default to shared-expert export (expert 0 only). vLLM load-time patch will
                # broadcast the shared expert weights across experts.
                mode = os.environ.get("MINT_MOE_LORA_SHARED_EXPERT_EXPORT", "1").strip().lower()
                if mode in {"1", "true", "yes"}:
                    expert_pat = re.compile(r"\.mlp\.(?:shared_)?experts\.(\d+)\.")
                    before = len(adapter_state)
                    dropped = 0
                    filtered = {}
                    for k, v in adapter_state.items():
                        m = expert_pat.search(k)
                        if m is None:
                            filtered[k] = v
                            continue
                        idx = int(m.group(1))
                        if idx == 0:
                            filtered[k] = v
                        else:
                            dropped += 1
                    adapter_state = filtered
                    logger.info(
                        f"[Rank 0] Shared-expert export (export_adapter_weights): kept_expert=0 "
                        f"dropped_keys={dropped} before={before} after={len(adapter_state)}"
                    )

            logger.info(f"[Rank 0] export_adapter_weights returned {len(adapter_state)} params")
            return adapter_state

        # ========== Fall back to custom implementation ==========
        logger.info(f"[Rank {self.rank}] Using custom LoRA extraction (export_adapter_weights not available)")

        # Write diagnostic output to shared PFS for debugging
        debug_file = "/vePFS-Mindverse/share/code/tinker-server/debug_lora_export.log"

        # ========== TP Gathering Helpers ==========
        # LoRA weights are sharded across TP ranks. We must gather before export.
        # Based on Megatron-Bridge gpt_bridge.py _get_tp_split_dim() logic.
        #
        # Sharding rules for LoRA:
        # - linear_proj, linear_fc2 (RowParallel): lora_A sharded on dim 1
        # - linear_q_proj, linear_kv_up_proj (ColumnParallel): lora_B sharded on dim 0
        # - linear_fc1 (fused gate+up): lora_B sharded on dim 1 (special case)
        # - linear_kv_down_proj: lora_B sharded on dim 0 (for mcore < 0.14)

        def _get_lora_tp_split_dim(param_name: str, etp_size: int = 1) -> int | None:
            """Determine TP split dimension for a LoRA parameter.

            Returns None if parameter is not TP-sharded.
            Returns 0 for column-parallel (output sharded).
            Returns 1 for row-parallel (input sharded) or fused linear_fc1.

            CRITICAL: For MoE models with ETP=1, expert MLP LoRAs are NOT tensor-parallelized.
            Only attention LoRAs use TP sharding. Expert MLP LoRAs use ETP sharding.
            """
            import re

            # Identify lora_A vs lora_B
            is_lora_a = '.adapter.linear_in.' in param_name or '.lora_a.' in param_name.lower()
            is_lora_b = '.adapter.linear_out.' in param_name or '.lora_b.' in param_name.lower()

            if not (is_lora_a or is_lora_b):
                return None

            # Check if this is an expert MLP layer (MoE)
            # Expert layers have patterns like: mlp.experts.*, mlp.shared_experts.*
            # These use ETP (Expert Tensor Parallel) instead of TP
            # Note: Dense MLP layers (e.g., layer 0's .mlp.linear_fc1 without .experts.)
            # still use TP sharding and should be gathered
            is_expert_layer = '.mlp.experts.' in param_name or '.mlp.shared_experts.' in param_name

            if is_expert_layer:
                # For expert layers, sharding depends on ETP, not TP
                # With ETP=1 (no expert tensor parallelism), these are NOT sharded
                if etp_size <= 1:
                    return None
                # With ETP > 1, would need ETP-based gathering (not implemented)
                # For now, return None and log warning
                logger.warning(f"ETP={etp_size} > 1 not yet supported for LoRA gathering")
                return None

            # RowParallel layers (attention output projection, MLP down projection):
            # - lora_A is sharded on dim 1 (input dimension)
            row_parallel_keys = {'linear_proj', 'linear_fc2'}

            # ColumnParallel layers (attention QKV, MLA projections, MLP up/gate):
            # - lora_A is sharded on dim 0 (rank dimension)
            # - lora_B is sharded on dim 0 (output dimension)
            col_parallel_keys = {'linear_qkv', 'linear_q_proj', 'linear_kv_up_proj', 'linear_kv_down_proj', 'linear_fc1'}

            # Extract base layer name (e.g., "linear_proj" from "decoder.layers.0.self_attn.linear_proj.adapter.linear_in.weight")
            match = re.search(r'(linear_\w+)\.(?:adapter|lora)', param_name)
            if not match:
                return None
            base_layer = match.group(1)

            if is_lora_a:
                # lora_A is sharded if base layer is RowParallel (dim 1) or ColumnParallel (dim 0)
                if base_layer in row_parallel_keys:
                    return 1  # RowParallel: lora_A sharded on input dim
                elif base_layer in col_parallel_keys:
                    return 0  # ColumnParallel: lora_A sharded on rank dim
            elif is_lora_b:
                # CRITICAL: linear_out in ParallelLinearAdapter is ALWAYS ColumnParallelLinear
                # regardless of whether base layer is RowParallel or ColumnParallel.
                # So lora_B is ALWAYS sharded on dim 0 (output dimension).
                if base_layer in col_parallel_keys or base_layer in row_parallel_keys:
                    return 0  # lora_B always sharded on output dim

            return None

        def _gather_tensor_across_tp(tensor, tp_group, split_dim: int):
            """Gather tensor across TP ranks along the specified dimension.

            Args:
                tensor: Local shard of the tensor
                tp_group: Tensor parallel process group
                split_dim: Dimension along which tensor is sharded (0 or 1)

            Returns:
                Gathered tensor (full tensor on all ranks)
            """
            import torch
            import torch.distributed as dist

            tp_size = dist.get_world_size(tp_group)
            if tp_size == 1:
                return tensor

            # Ensure tensor is on GPU for NCCL
            device = torch.cuda.current_device()
            tensor = tensor.to(device) if not tensor.is_cuda else tensor

            if split_dim == 0:
                # Prefer list-based all_gather over all_gather_into_tensor: the latter
                # has been observed to hang in large EP-folded configurations.
                gathered_list = [torch.empty_like(tensor) for _ in range(tp_size)]
                dist.all_gather(gathered_list, tensor.contiguous(), group=tp_group)
                return torch.cat(gathered_list, dim=0)
            else:
                # Gather along other dimension (RowParallel lora_A, fused lora_B)
                # Need to use all_gather then cat
                gathered_list = [torch.empty_like(tensor) for _ in range(tp_size)]
                dist.all_gather(gathered_list, tensor.contiguous(), group=tp_group)
                return torch.cat(gathered_list, dim=split_dim)

        def _split_fused_gate_up_lora_b(tensor, tp_size_for_split: int):
            """Split fused gate+up lora_B tensor with TP-aware interleaved layout.

            When TP > 1, lora_B for fused gate+up is interleaved:
            Layout: [gate_shard0, up_shard0, gate_shard1, up_shard1, ...]

            M-Bridge reference: peft_bridge.py:282-311

            Args:
                tensor: Fused lora_B tensor of shape (2*intermediate_size, rank)
                tp_size_for_split: TP size (use ETP for expert layers, TP for dense)

            Returns:
                (gate_tensor, up_tensor) each of shape (intermediate_size, rank)
            """
            import torch

            if tp_size_for_split <= 1:
                # Simple contiguous split
                half_size = tensor.shape[0] // 2
                return tensor[:half_size, :].clone(), tensor[half_size:, :].clone()

            # TP > 1: Deinterleave the shards
            shard_size = tensor.shape[0] // tp_size_for_split
            if shard_size * tp_size_for_split != tensor.shape[0] or shard_size % 2 != 0:
                # Fallback if dimensions don't match expected interleaved layout
                logger.warning(f"Unexpected tensor shape for TP-aware split: {tensor.shape}, tp_size={tp_size_for_split}")
                half_size = tensor.shape[0] // 2
                return tensor[:half_size, :].clone(), tensor[half_size:, :].clone()

            shards = torch.split(tensor, shard_size, dim=0)
            gate_parts = []
            up_parts = []
            for shard in shards:
                gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
                gate_parts.append(gate_shard)
                up_parts.append(up_shard)
            return torch.cat(gate_parts, dim=0), torch.cat(up_parts, dim=0)

        # Get TP group and size for gathering
        try:
            from megatron.core import parallel_state as mpu
            tp_group = mpu.get_tensor_model_parallel_group()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            logger.info(f"[Rank {self.rank}] TP size: {tp_size}")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Could not get TP group: {e}, assuming TP=1")
            tp_group = None
            tp_size = 1

        # Get ETP size for MoE expert layers
        try:
            from megatron.core import parallel_state as mpu
            etp_size = mpu.get_expert_tensor_parallel_world_size()
            logger.info(f"[Rank {self.rank}] ETP size: {etp_size}")
        except Exception:
            # ETP not available, assume 1 (no expert tensor parallelism)
            etp_size = 1

        # Get EP (Expert Parallelism) group, size, and rank
        # CRITICAL: With EP=8, each rank holds different experts (64/8 = 8 experts per rank)
        # We must gather expert LoRAs from ALL EP ranks to export complete state
        try:
            from megatron.core import parallel_state as mpu
            ep_group = mpu.get_expert_model_parallel_group()
            ep_size = mpu.get_expert_model_parallel_world_size()
            ep_rank = mpu.get_expert_model_parallel_rank()
            logger.info(f"[Rank {self.rank}] EP size: {ep_size}, EP rank: {ep_rank}")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Could not get EP group: {e}, assuming EP=1")
            ep_group = None
            ep_size = 1
            ep_rank = 0

        # For MoE models, exporting routed expert LoRAs requires EP gathering and
        # per-expert expansion for vLLM. If we're not exporting per-expert weights,
        # skip the MLP subtree early to avoid expensive TP/EP collectives.
        try:
            model_is_moe = get_model_config(self.base_model).is_moe
        except Exception:
            model_is_moe = False

        skip_mlp_params = model_is_moe and not use_per_expert_lora
        if self.rank == 0:
            print(
                f"[Rank 0] get_lora_state_dict groups: tp_size={tp_size} etp_size={etp_size} "
                f"ep_size={ep_size} use_per_expert_lora={use_per_expert_lora} skip_mlp={skip_mlp_params}",
                flush=True,
            )

        import os

        debug_lora_export = os.environ.get("MINT_DEBUG_LORA_EXPORT", "0") == "1"
        debug_lora_export_max_params = int(os.environ.get("MINT_DEBUG_LORA_EXPORT_MAX_PARAMS", "10"))

        def _gather_expert_lora_across_ep(tensor, ep_group, ep_size: int):
            """Gather expert LoRA tensor from all EP ranks.

            Each EP rank holds a shard of experts. We gather all shards to
            get complete expert weights.

            Args:
                tensor: Local expert LoRA tensor
                ep_group: Expert parallel process group
                ep_size: Number of EP ranks

            Returns:
                List of tensors, one per EP rank (indexed by ep_rank)
            """
            import torch
            import torch.distributed as dist

            if ep_size == 1:
                return [tensor]

            # Ensure tensor is on GPU for NCCL
            device = torch.cuda.current_device()
            tensor = tensor.to(device) if not tensor.is_cuda else tensor

            # Gather from all EP ranks
            gathered_list = [torch.empty_like(tensor) for _ in range(ep_size)]
            dist.all_gather(gathered_list, tensor.contiguous(), group=ep_group)
            return gathered_list

        # ========== End TP/EP Gathering Helpers ==========

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
                lora_seen = 0

                for name, param in unwrapped.named_parameters():
                    all_param_names.append(name)
                    # Look for adapter patterns in name (MBridge uses .adapter.linear_in/.adapter.linear_out)
                    name_lower = name.lower()
                    if '.adapter.' in name_lower or 'lora_a' in name_lower or 'lora_b' in name_lower:
                        lora_param_names.append(name)
                        lora_seen += 1
                        if self.rank == 0 and (lora_seen == 1 or lora_seen % 500 == 0):
                            print(
                                f"[Rank 0] get_lora_state_dict progress: lora_seen={lora_seen} last={name}",
                                flush=True,
                            )
                        original_shape = list(param.data.shape)
                        do_debug_param = debug_lora_export and (len(lora_param_names) <= debug_lora_export_max_params)

                        # MoE MLP/expert LoRAs are only needed when exporting per-expert weights.
                        # If we aren't exporting per-expert LoRAs, skip the MLP subtree early to
                        # avoid expensive TP/EP collectives and CPU copies.
                        if skip_mlp_params and ".mlp." in name_lower:
                            continue

                        # DEBUG: Log original tensor shape with more detail
                        if self.rank == 0 and do_debug_param:
                            is_lora_a = '.adapter.linear_in.' in name or '.lora_a.' in name_lower
                            lora_type = "lora_A" if is_lora_a else "lora_B"
                            logger.info(f"[Rank 0] Megatron {lora_type} {name}: shape={original_shape}")
                        # CRITICAL: ALL ranks must participate in gathering to avoid NCCL deadlock!
                        # Check if this parameter needs TP gathering
                        split_dim = _get_lora_tp_split_dim(name, etp_size=etp_size)
                        if self.rank == 0 and do_debug_param:
                            # Detailed logging for shape debugging
                            is_expert = '.mlp.experts.' in name or '.mlp.shared_experts.' in name or '.mlp.linear_fc' in name
                            logger.info(f"[Rank 0] {name}: split_dim={split_dim}, is_expert={is_expert}, etp_size={etp_size}, tp_size={tp_size}")
                        if split_dim is not None and tp_group is not None and tp_size > 1:
                            # Gather across TP ranks (NCCL collective).
                            gathered = _gather_tensor_across_tp(param.data, tp_group, split_dim)
                            tensor_for_export = gathered
                            if self.rank == 0 and do_debug_param:
                                logger.info(f"[Rank 0] GATHERED {name}: {original_shape} -> {list(gathered.shape)} (dim={split_dim})")
                        else:
                            tensor_for_export = param.data
                            if self.rank == 0 and do_debug_param:
                                logger.info(f"[Rank 0] NO_GATHER {name}: shape={list(tensor_for_export.shape)}")

                        # EP gather is only required when exporting per-expert expert LoRAs.
                        is_routed_expert = '.mlp.experts.' in name and '.mlp.shared_experts.' not in name
                        if use_per_expert_lora and is_routed_expert and ep_group is not None and ep_size > 1:
                            # ALL ranks participate in EP all_gather (NCCL collective).
                            ep_gathered = _gather_expert_lora_across_ep(tensor_for_export, ep_group, ep_size)
                            if self.rank == 0:
                                if do_debug_param:
                                    logger.info(f"[Rank 0] EP_GATHERED {name}: {ep_size} shards, shape={list(ep_gathered[0].shape)}")
                                # Store with EP rank index for later expansion
                                # Key format: original_name::EP_RANK::{i}
                                for i, tensor in enumerate(ep_gathered):
                                    ep_key = f"{name}::EP_RANK::{i}"
                                    adapter_state[ep_key] = tensor.detach().cpu()
                        else:
                            # Non-expert param or EP=1: store normally (only rank 0)
                            if self.rank == 0:
                                adapter_state[name] = tensor_for_export.detach().cpu()

                logger.info(f"[Rank {self.rank}] named_parameters() returned {len(all_param_names)} total, {len(lora_param_names)} LoRA params")

                if self.rank == 0:
                    # Write diagnostic info with shapes
                    try:
                        import datetime
                        with open(debug_file, "a") as f:
                            f.write(f"\n=== {datetime.datetime.now()} (named_parameters within eval_mode) ===\n")
                            f.write(f"Total params: {len(all_param_names)}\n")
                            f.write(f"LoRA params found: {len(lora_param_names)}\n")
                            f.write(f"TP size: {tp_size}, ETP size: {etp_size}, EP size: {ep_size}\n")
                            f.write("LoRA params with shapes and norms:\n")
                            total_norm = 0.0
                            nonzero_count = 0
                            for n in lora_param_names[:50]:
                                if n in adapter_state:
                                    t = adapter_state[n]
                                    shape = list(t.shape)
                                    norm = float(t.norm().item())
                                    max_val = float(t.abs().max().item())
                                    total_norm += norm
                                    if norm > 1e-8:
                                        nonzero_count += 1
                                    f.write(f"  {n}: shape={shape}, norm={norm:.6f}, max={max_val:.6f}\n")
                            f.write(f"\nSummary: {nonzero_count}/{len(lora_param_names[:50])} tensors non-zero, total_norm={total_norm:.6f}\n")
                            f.write("===\n")
                        # ALSO LOG TO STDOUT for visibility in server logs
                        logger.info(f"[Rank 0] LoRA TENSOR NORMS: {nonzero_count}/{len(lora_param_names[:50])} non-zero, total_norm={total_norm:.6f}")
                        for n in lora_param_names[:5]:
                            if n in adapter_state:
                                t = adapter_state[n]
                                norm = float(t.norm().item())
                                logger.info(f"[Rank 0] TENSOR {n}: norm={norm:.6f}")
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

            # Log before exiting eval_mode to detect if exit is blocked
            logger.info(f"[Rank {self.rank}] get_lora_state_dict: extraction complete, about to exit eval_mode (adapter_state has {len(adapter_state)} keys)")

            # SECOND: If direct extraction failed, try bridge.export_hf_weights()
            # NOTE: Modern Megatron-Bridge uses export_hf_weights(), not the old export_weights() API.
            # export_weights() merges LoRA into base weights, which is not what we want.
            # CRITICAL: Use lora_param_names (same across all ranks) not adapter_state (only rank 0 has data)
            adapter_param_names = []  # Track across fallbacks for symmetric branching
            if not lora_param_names:
                bridge = getattr(self.engine, 'bridge', None)
                has_export_hf_weights = bridge is not None and hasattr(bridge, 'export_hf_weights')

                if has_export_hf_weights:
                    logger.info(f"[Rank {self.rank}] Fallback to bridge.export_hf_weights()")
                    try:
                        adapter_patterns = ['lora', 'adapter', 'peft']
                        all_param_names = []

                        for name, tensor in bridge.export_hf_weights(self.engine.module):
                            all_param_names.append(name)
                            name_lower = name.lower()
                            if any(pattern in name_lower for pattern in adapter_patterns):
                                adapter_param_names.append(name)
                                # CRITICAL: ALL ranks must do the same work to avoid NCCL deadlock!
                                cloned = tensor.data.clone().cpu() if tensor.is_cuda else tensor.data.clone()
                                if self.rank == 0:
                                    adapter_state[name] = cloned

                        logger.info(f"[Rank {self.rank}] bridge.export_hf_weights() returned {len(all_param_names)} params, {len(adapter_param_names)} adapter")

                        if self.rank == 0 and not adapter_param_names:
                            logger.warning("[Rank 0] NO adapter params found via export_hf_weights()")
                    except Exception as e:
                        logger.error(f"[Rank {self.rank}] bridge.export_hf_weights() failed: {e}")

            # THIRD: Try verl's get_adapter_state_dict as last resort (also within eval_mode)
            # CRITICAL: ALL ranks must attempt this to stay synchronized
            # Use symmetric condition: no params found from prior methods
            if not lora_param_names and not adapter_param_names:
                logger.info(f"[Rank {self.rank}] Trying get_adapter_state_dict() fallback...")
                try:
                    from verl.utils.megatron_peft_utils import get_adapter_state_dict
                    raw_state = get_adapter_state_dict(self.engine.module)
                    logger.info(f"[Rank {self.rank}] get_adapter_state_dict returned {len(raw_state)} params")
                    for name, tensor in raw_state.items():
                        # ALL ranks must participate in gathering
                        split_dim = _get_lora_tp_split_dim(name, etp_size=etp_size)
                        if split_dim is not None and tp_group is not None and tp_size > 1:
                            gathered = _gather_tensor_across_tp(tensor, tp_group, split_dim)
                            cloned = gathered.cpu()
                        else:
                            cloned = tensor.cpu() if tensor.is_cuda else tensor.clone()
                        if self.rank == 0:
                            adapter_state[name] = cloned
                    if self.rank == 0 and adapter_state:
                        sample_keys = list(adapter_state.keys())[:5]
                        logger.info(f"[Rank 0] Sample Megatron keys: {sample_keys}")
                except ImportError:
                    logger.warning(f"[Rank {self.rank}] verl.utils.megatron_peft_utils not available")
                except Exception as e:
                    logger.error(f"[Rank {self.rank}] get_adapter_state_dict failed: {e}")

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
        def _is_mlp_module(param_name: str) -> bool:
            """Check if parameter is from MLP/expert layer."""
            mlp_patterns = ['.mlp.', '.experts.', 'linear_fc1', 'linear_fc2',
                           'gate_proj', 'up_proj', 'down_proj', 'shared_expert']
            name_lower = param_name.lower()
            return any(p in name_lower for p in mlp_patterns)

        # Get number of experts for per-expert expansion (only when use_per_expert_lora=True)
        num_experts = 0
        if model_is_moe and use_per_expert_lora:
            try:
                from transformers import AutoConfig
                hf_config = AutoConfig.from_pretrained(self.base_model, trust_remote_code=True)
                # Different HF configs use different attribute names
                num_experts = getattr(hf_config, "num_experts", None) or getattr(hf_config, "n_routed_experts", 0)
                logger.info(f"[Rank 0] MoE model with {num_experts} experts - expanding shared LoRA to per-expert format")
            except Exception as e:
                logger.warning(f"[Rank 0] Could not get num_experts: {e}, disabling per-expert expansion")
                use_per_expert_lora = False

        if model_is_moe and not use_per_expert_lora:
            logger.info("[Rank 0] MoE model detected - filtering MLP/expert modules (no MLP LoRA trained)")
            logger.info("[Rank 0] Set use_per_expert_lora=True if MLP LoRA was trained")

        lora_state_dict = {}
        mlp_filtered_count = 0
        mlp_expanded_count = 0

        import os

        def _ep_rank_to_expert_range(ep_rank_idx: int, *, ep_size: int, num_experts: int) -> tuple[int, int]:
            """Map an EP rank to its contiguous expert ID range (linear placement).

            Mirrors vLLM's `determine_expert_map(..., expert_placement_strategy='linear')`.
            """
            if ep_size <= 1:
                return 0, num_experts
            base = num_experts // ep_size
            rem = num_experts % ep_size
            start = ep_rank_idx * base + min(ep_rank_idx, rem)
            local = base + 1 if ep_rank_idx < rem else base
            end = min(num_experts, start + local)
            return start, end

        sparse_expert_export = (
            bool(model_is_moe)
            and bool(use_per_expert_lora)
            and num_experts > 0
            and ep_size > 1
            and os.environ.get("MINT_MOE_LORA_SPARSE_EXPERT_EXPORT", "1").strip() != "0"
        )
        expert_representatives = []
        if sparse_expert_export:
            # One representative expert per EP shard (linear placement).
            # The vLLM hot-load patch reconstructs full per-expert tensors at load time.
            base = num_experts // ep_size
            rem = num_experts % ep_size
            expert_representatives = [
                (r * base + min(r, rem))
                for r in range(ep_size)
                if (base + (1 if r < rem else 0)) > 0
            ]

        logger.info(
            f"[Rank 0] Processing {len(adapter_state)} params from adapter_state "
            f"(ep_size={ep_size}, sparse_expert_export={sparse_expert_export}, "
            f"expert_representatives={len(expert_representatives)})"
        )

        # First pass: group EP-indexed keys by base name
        ep_grouped_params = {}  # base_name -> {ep_rank: tensor}
        regular_params = {}  # name -> tensor
        for name, tensor in adapter_state.items():
            if '::EP_RANK::' in name:
                # Parse EP-indexed key: base_name::EP_RANK::X
                base_name, _, ep_rank_str = name.rpartition('::EP_RANK::')
                ep_rank_idx = int(ep_rank_str)
                if base_name not in ep_grouped_params:
                    ep_grouped_params[base_name] = {}
                ep_grouped_params[base_name][ep_rank_idx] = tensor
            else:
                regular_params[name] = tensor

        logger.info(f"[Rank 0] Found {len(ep_grouped_params)} EP-grouped params, {len(regular_params)} regular params")

        # Process EP-grouped params with correct expert assignment
        for base_name, ep_tensors in ep_grouped_params.items():
            # Convert base name to PEFT format
            peft_name = self._convert_megatron_to_peft(base_name)
            if peft_name is None:
                logger.warning(f"Could not convert to PEFT: {base_name}")
                continue

            # Handle MoE MLP/expert modules with EP-aware expansion
            if model_is_moe and _is_mlp_module(peft_name):
                is_shared_expert = '.shared_expert.' in peft_name

                if use_per_expert_lora and num_experts > 0 and not is_shared_expert:
                    # Split fused gate_up_proj if needed, then expand with EP-aware indexing
                    if '.gate_up_proj_fused.' in peft_name:
                        gate_peft_name = peft_name.replace('.gate_up_proj_fused.', '.gate_proj.')
                        up_peft_name = peft_name.replace('.gate_up_proj_fused.', '.up_proj.')

                        # Expand with correct EP rank -> expert index mapping
                        for ep_rank_idx, tensor in ep_tensors.items():
                            expert_start, expert_end = _ep_rank_to_expert_range(
                                ep_rank_idx, ep_size=ep_size, num_experts=num_experts
                            )
                            if expert_start >= expert_end:
                                continue

                            if '.lora_A.' in peft_name:
                                gate_tensor = tensor
                                up_tensor = tensor
                            elif '.lora_B.' in peft_name:
                                # Use TP-aware split for interleaved gate+up layout
                                # For MoE experts, use etp_size; if ETP=1, fallback to simple split
                                gate_tensor, up_tensor = _split_fused_gate_up_lora_b(tensor, etp_size)
                            else:
                                continue

                            if sparse_expert_export:
                                rep_experts = [
                                    e
                                    for e in expert_representatives
                                    if expert_start <= e < expert_end
                                ]
                            else:
                                rep_experts = range(expert_start, expert_end)
                            for expert_idx in rep_experts:
                                gate_expert_name = gate_peft_name.replace(
                                    '.mlp.gate_proj.', f'.mlp.experts.{expert_idx}.gate_proj.'
                                )
                                up_expert_name = up_peft_name.replace(
                                    '.mlp.up_proj.', f'.mlp.experts.{expert_idx}.up_proj.'
                                )
                                lora_state_dict[gate_expert_name] = gate_tensor.clone()
                                lora_state_dict[up_expert_name] = up_tensor.clone()
                                mlp_expanded_count += 2

                        logger.info(
                            f"[Rank 0] EP-expanded fused MLP LoRA for {len(ep_tensors)} EP shards "
                            f"(sparse={sparse_expert_export})"
                        )
                    else:
                        # Non-fused modules with EP-aware expansion
                        for ep_rank_idx, tensor in ep_tensors.items():
                            expert_start, expert_end = _ep_rank_to_expert_range(
                                ep_rank_idx, ep_size=ep_size, num_experts=num_experts
                            )
                            if expert_start >= expert_end:
                                continue

                            if sparse_expert_export:
                                rep_experts = [
                                    e
                                    for e in expert_representatives
                                    if expert_start <= e < expert_end
                                ]
                            else:
                                rep_experts = range(expert_start, expert_end)
                            for expert_idx in rep_experts:
                                # Insert expert index into path
                                if '.mlp.gate_proj.' in peft_name:
                                    expert_peft_name = peft_name.replace('.mlp.gate_proj.', f'.mlp.experts.{expert_idx}.gate_proj.')
                                elif '.mlp.up_proj.' in peft_name:
                                    expert_peft_name = peft_name.replace('.mlp.up_proj.', f'.mlp.experts.{expert_idx}.up_proj.')
                                elif '.mlp.down_proj.' in peft_name:
                                    expert_peft_name = peft_name.replace('.mlp.down_proj.', f'.mlp.experts.{expert_idx}.down_proj.')
                                else:
                                    expert_peft_name = peft_name.replace('.mlp.', f'.mlp.experts.{expert_idx}.')
                                lora_state_dict[expert_peft_name] = tensor.clone()
                                mlp_expanded_count += 1

                        logger.info(
                            f"[Rank 0] EP-expanded MLP LoRA: {base_name} (sparse={sparse_expert_export})"
                        )
                elif is_shared_expert:
                    # Shared expert (EP rank 0 only for shared experts)
                    if 0 in ep_tensors:
                        lora_state_dict[peft_name] = ep_tensors[0]
                        logger.info(f"[Rank 0] Added shared_expert MLP LoRA from EP rank 0: {peft_name}")
                else:
                    mlp_filtered_count += len(ep_tensors)
                continue

            # Non-MLP EP-grouped params (shouldn't happen, but handle gracefully)
            if 0 in ep_tensors:
                lora_state_dict[peft_name] = ep_tensors[0]

        # Process regular (non-EP-indexed) params
        for name, tensor in regular_params.items():
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

            # Handle MoE MLP/expert modules
            # IMPORTANT: shared_expert modules should NOT be expanded to per-expert format
            if model_is_moe and _is_mlp_module(peft_name):
                is_shared_expert = '.shared_expert.' in peft_name

                if use_per_expert_lora and num_experts > 0:
                    # First check if this is a fused gate_up_proj that needs splitting
                    if '.gate_up_proj_fused.' in peft_name:
                        # Split the fused projection before expanding to per-expert
                        gate_peft_name = peft_name.replace('.gate_up_proj_fused.', '.gate_proj.')
                        up_peft_name = peft_name.replace('.gate_up_proj_fused.', '.up_proj.')

                        if '.lora_A.' in peft_name:
                            # lora_A is shared - duplicate for both gate and up
                            gate_tensor = tensor.clone()
                            up_tensor = tensor.clone()
                        elif '.lora_B.' in peft_name:
                            # lora_B uses TP-aware split for interleaved gate+up layout
                            # For MoE/shared experts, use etp_size
                            gate_tensor, up_tensor = _split_fused_gate_up_lora_b(tensor, etp_size)
                        else:
                            logger.warning(f"[Rank 0] Unknown lora type for fused MoE: {peft_name}")
                            continue

                        if is_shared_expert:
                            # Shared expert: add directly without per-expert expansion
                            lora_state_dict[gate_peft_name] = gate_tensor
                            lora_state_dict[up_peft_name] = up_tensor
                            logger.info(f"[Rank 0] Split shared_expert fused MLP LoRA: {gate_peft_name}, {up_peft_name}")
                        else:
                            # Routed experts: expand both gate and up to per-expert format
                            for proj_name, proj_tensor in [(gate_peft_name, gate_tensor), (up_peft_name, up_tensor)]:
                                expanded_names = self._expand_shared_to_per_expert(
                                    proj_name,
                                    num_experts,
                                    expert_indices=expert_representatives if sparse_expert_export else None,
                                )
                                for expanded_name in expanded_names:
                                    lora_state_dict[expanded_name] = proj_tensor.clone()
                                    mlp_expanded_count += 1
                            logger.info(f"[Rank 0] Split and expanded fused MLP LoRA to {len(expanded_names)*2} per-expert weights")
                    else:
                        # Non-fused modules
                        if is_shared_expert:
                            # Shared expert: add directly without expansion
                            lora_state_dict[peft_name] = tensor
                            logger.info(f"[Rank 0] Added shared_expert MLP LoRA: {peft_name}")
                        else:
                            # Routed experts: expand to per-expert
                            expanded_names = self._expand_shared_to_per_expert(
                                peft_name,
                                num_experts,
                                expert_indices=expert_representatives if sparse_expert_export else None,
                            )
                            for expanded_name in expanded_names:
                                lora_state_dict[expanded_name] = tensor.clone()
                                mlp_expanded_count += 1
                else:
                    # Filter out MLP modules when not using per-expert expansion
                    mlp_filtered_count += 1
                continue

            lora_state_dict[peft_name] = tensor

            # Special handling for fused gate+up projection (linear_fc1)
            # vLLM's MergedColumnParallelLinearWithLoRA expects separate gate_proj and up_proj
            # lora_A is duplicated, lora_B is split in half
            if '.gate_up_proj_fused.' in peft_name:
                gate_peft_name = peft_name.replace('.gate_up_proj_fused.', '.gate_proj.')
                up_peft_name = peft_name.replace('.gate_up_proj_fused.', '.up_proj.')

                if '.lora_A.' in peft_name:
                    # lora_A is shared between gate and up projections
                    lora_state_dict[gate_peft_name] = tensor.clone()
                    lora_state_dict[up_peft_name] = tensor.clone()
                    logger.info(f"[Rank 0] Split fused lora_A: {gate_peft_name}, {up_peft_name}")
                elif '.lora_B.' in peft_name:
                    # lora_B uses TP-aware split for interleaved gate+up layout
                    # For dense layers (non-MoE), use tp_size
                    gate_tensor, up_tensor = _split_fused_gate_up_lora_b(tensor, tp_size)
                    lora_state_dict[gate_peft_name] = gate_tensor
                    lora_state_dict[up_peft_name] = up_tensor
                    logger.info(f"[Rank 0] Split fused lora_B: {gate_peft_name} shape {gate_tensor.shape}, {up_peft_name} shape {up_tensor.shape}")
                else:
                    logger.warning(f"[Rank 0] Unknown lora type for fused projection: {peft_name}")

                # Remove the placeholder entry
                del lora_state_dict[peft_name]

        if model_is_moe:
            if use_per_expert_lora:
                logger.info(f"[Rank 0] Expanded {mlp_expanded_count} shared MLP modules to per-expert format")
            else:
                logger.info(f"[Rank 0] Filtered {mlp_filtered_count} MLP/expert modules for MoE model")
        logger.info(f"[Rank 0] Extracted {len(lora_state_dict)} LoRA parameters (PEFT format)")

        # Warn if all weights were filtered (e.g., LoRA targeted only MLP for MoE)
        if not lora_state_dict and mlp_filtered_count > 0:
            logger.warning(
                f"[Rank 0] ALL LoRA weights were filtered ({mlp_filtered_count} MLP modules). "
                "This may indicate LoRA was configured to target only expert layers. "
                "Set use_per_expert_lora=True to expand shared MLP LoRA to per-expert format."
            )

        if lora_state_dict:
            sample_peft_keys = list(lora_state_dict.keys())[:5]
            logger.info(f"[Rank 0] Sample PEFT keys: {sample_peft_keys}")
            # Log shapes for debugging vLLM compatibility
            for key in list(lora_state_dict.keys())[:10]:
                tensor = lora_state_dict[key]
                logger.info(f"[Rank 0] FINAL PEFT {key}: shape={list(tensor.shape)}")

        if model_is_moe and use_per_expert_lora and lora_state_dict:
            import os
            import re

            mode = os.environ.get("MINT_MOE_LORA_SHARED_EXPERT_EXPORT", "1").strip().lower()
            if mode in {"1", "true", "yes"}:
                expert_pat = re.compile(r"\.mlp\.(?:shared_)?experts\.(\d+)\.")
                before = len(lora_state_dict)
                dropped = 0
                filtered = {}
                for k, v in lora_state_dict.items():
                    m = expert_pat.search(k)
                    if m is None:
                        filtered[k] = v
                        continue
                    idx = int(m.group(1))
                    if idx == 0:
                        filtered[k] = v
                    else:
                        dropped += 1
                lora_state_dict = filtered
                logger.info(
                    f"[Rank 0] Shared-expert export (custom extract): kept_expert=0 "
                    f"dropped_keys={dropped} before={before} after={len(lora_state_dict)}"
                )

        return lora_state_dict

    def _convert_megatron_to_peft(self, name: str) -> str | None:
        """Convert Megatron LoRA param name to PEFT format.

        Megatron-Bridge names (with vanilla_mbridge=False):
            Standard attention:
                decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight
                decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight

            MLA attention (DeepSeek V3/Moonlight/K2):
                decoder.layers.0.self_attention.linear_q_down_proj.adapter.linear_in.weight
                decoder.layers.0.self_attention.linear_q_up_proj.adapter.linear_in.weight
                decoder.layers.0.self_attention.linear_kv_down_proj.adapter.linear_in.weight
                decoder.layers.0.self_attention.linear_kv_up_proj.adapter.linear_in.weight
                decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight

            MLP/Expert:
                decoder.layers.0.mlp.experts.linear_fc1.adapter.linear_in.weight
                decoder.layers.0.mlp.experts.linear_fc2.adapter.linear_out.weight

        PEFT names (HuggingFace style):
            Standard attention:
                base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight

            MLA attention:
                base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight
                base_model.model.model.layers.0.self_attn.q_b_proj.lora_A.weight
                base_model.model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_A.weight
                base_model.model.model.layers.0.self_attn.kv_b_proj.lora_A.weight
                base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight
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
        elif 'lora_a' in name.lower():
            lora_type = 'lora_A'
        elif 'lora_b' in name.lower():
            lora_type = 'lora_B'
        else:
            logger.warning(f"Could not determine LoRA type from: {name}")
            return None

        # Check if this is an MLA model based on parameter names
        # MLA models have linear_q_down_proj, linear_kv_down_proj, etc.
        model_is_mla = any(p in name for p in [
            'linear_q_down_proj', 'linear_q_up_proj',
            'linear_kv_down_proj', 'linear_kv_up_proj'
        ])

        # Also check model registry for MLA flag (fallback)
        if not model_is_mla:
            try:
                model_is_mla = get_model_config(self.base_model).is_mla
            except (ValueError, AttributeError):
                pass

        # Determine the target module
        target = None

        # MLA attention projections (DeepSeek V3 / Moonlight / K2)
        # Note: Megatron uses linear_q_proj (single), not linear_q_down_proj/linear_q_up_proj
        if 'linear_q_proj' in name and 'linear_kv' not in name:
            # MLA Q projection - maps to q_proj for vLLM
            # (vLLM's DeepseekV3 model uses q_a_proj/q_b_proj internally but expects q_proj for LoRA)
            target = 'self_attn.q_proj'
        elif 'linear_q_down_proj' in name:
            target = 'self_attn.q_a_proj'
        elif 'linear_q_up_proj' in name:
            target = 'self_attn.q_b_proj'
        elif 'linear_kv_down_proj' in name:
            target = 'self_attn.kv_a_proj_with_mqa'
        elif 'linear_kv_up_proj' in name:
            target = 'self_attn.kv_b_proj'
        # Standard attention projections
        elif 'linear_qkv.adapter' in name or 'linear_qkv.lora_' in name.lower():
            # Fused QKV - map to q_proj (vLLM will handle the fused format)
            target = 'self_attn.q_proj'
        elif 'self_attention.linear_proj' in name:
            target = 'self_attn.o_proj'
        # MLP/Expert projections
        # IMPORTANT: Distinguish shared_experts from routed experts
        # shared_experts: single expert, always active (different intermediate size)
        # experts: routed experts, need per-expert expansion
        elif 'linear_fc1' in name:
            if '.shared_experts.' in name:
                # Shared expert MLP - use separate namespace to avoid overwriting routed experts
                target = 'mlp.shared_expert.gate_up_proj_fused'
            else:
                # Routed expert MLP gate/up fused projection - map to gate_up_proj
                # For vLLM MergedColumnParallelLinear, we need separate gate_proj and up_proj
                # This is handled specially in get_lora_state_dict to split the B matrix
                # Here we mark it as gate_up_proj_fused to trigger special handling
                target = 'mlp.gate_up_proj_fused'
        elif 'linear_fc2' in name:
            if '.shared_experts.' in name:
                # Shared expert down projection
                target = 'mlp.shared_expert.down_proj'
            else:
                # Routed expert down projection
                target = 'mlp.down_proj'

        if target is None:
            logger.warning(f"Unknown module pattern: {name}")
            return None

        peft_name = f"base_model.model.model.layers.{layer_num}.{target}.{lora_type}.weight"
        return peft_name

    def _expand_shared_to_per_expert(
        self, peft_name: str, num_experts: int, *, expert_indices: list[int] | None = None
    ) -> list[str]:
        """Expand shared MLP LoRA to per-expert format for vLLM FusedMoEWithLoRA.

        Megatron-Bridge exports shared LoRA (single adapter for all experts):
            base_model.model.model.layers.X.mlp.gate_proj.lora_A.weight

        vLLM FusedMoEWithLoRA expects per-expert weights:
            base_model.model.model.layers.X.mlp.experts.0.gate_proj.lora_A.weight
            base_model.model.model.layers.X.mlp.experts.1.gate_proj.lora_A.weight
            ...

        This method generates the per-expert weight names by inserting "experts.{i}."
        into the MLP path for each expert.

        Supported naming conventions:
        - Qwen/standard: gate_proj, up_proj, down_proj
        - DeepSeek/alternative: w1, w2, w3, gate_up_proj

        Args:
            peft_name: The shared PEFT name (e.g., ...mlp.gate_proj.lora_A.weight)
            num_experts: Number of experts in the MoE model

        Returns:
            List of per-expert PEFT names
        """
        import re

        # Pattern to match MLP module paths
        # Shared: base_model.model.model.layers.X.mlp.gate_proj.lora_A.weight
        # Per-expert: base_model.model.model.layers.X.mlp.experts.{i}.gate_proj.lora_A.weight
        # Support multiple naming conventions: gate_proj/up_proj/down_proj (Qwen), w1/w2/w3 (DeepSeek), gate_up_proj (fused)
        mlp_patterns = r'(gate_proj|up_proj|down_proj|w1|w2|w3|gate_up_proj)'
        mlp_match = re.search(r'(\.mlp\.)(' + mlp_patterns + r')', peft_name)
        if mlp_match:
            # Insert experts.{i}. after .mlp.
            prefix = peft_name[:mlp_match.end(1)]  # ...mlp.
            suffix = peft_name[mlp_match.end(1):]  # gate_proj.lora_A.weight
            idxs = expert_indices if expert_indices is not None else list(range(num_experts))
            return [f"{prefix}experts.{i}.{suffix}" for i in idxs]

        # If already has experts in path, just return the original
        if '.experts.' in peft_name:
            return [peft_name]

        # Fallback: return original name if pattern not matched
        logger.warning(f"Could not expand to per-expert format: {peft_name}")
        return [peft_name]

    def get_lora_weight_norm(self) -> float:
        """Compute L2 norm of all LoRA weights for debugging."""
        with self.engine.train_mode():
            from verl.utils.megatron_utils import unwrap_model
            model = unwrap_model(self.engine.module)
            while isinstance(model, list):
                model = model[0]
            total_norm_sq = 0.0
            param_count = 0
            for name, param in model.named_parameters():
                name_lower = name.lower()
                if 'lora' not in name_lower and 'adapter' not in name_lower:
                    continue
                if not param.requires_grad:
                    continue
                total_norm_sq += param.data.norm().item() ** 2
                param_count += 1
            total_norm = total_norm_sq ** 0.5
            print(f"[Rank {self.rank}] LoRA weight norm: {total_norm:.6f} ({param_count} params)", flush=True)
            return total_norm

    def get_lora_weight_checksum(self) -> dict:
        """Compute checksum stats for LoRA weights (rank 0 only)."""
        with self.engine.train_mode():
            from verl.utils.megatron_utils import unwrap_model
            model = unwrap_model(self.engine.module)
            while isinstance(model, list):
                model = model[0]
            total_sum = 0.0
            total_abs_sum = 0.0
            param_count = 0
            for name, param in model.named_parameters():
                name_lower = name.lower()
                if 'lora' not in name_lower and 'adapter' not in name_lower:
                    continue
                if not param.requires_grad:
                    continue
                total_sum += float(param.data.sum().item())
                total_abs_sum += float(param.data.abs().sum().item())
                param_count += 1
            return {
                "sum": total_sum,
                "abs_sum": total_abs_sum,
                "count": param_count,
            }

    def get_base_weight_checksum(self, max_params: int = 5) -> dict:
        """Compute checksum stats for a small sample of non-LoRA params (rank 0 only)."""
        with self.engine.train_mode():
            from verl.utils.megatron_utils import unwrap_model
            model = unwrap_model(self.engine.module)
            while isinstance(model, list):
                model = model[0]
            entries = []
            for name, param in model.named_parameters():
                name_lower = name.lower()
                if 'lora' in name_lower or 'adapter' in name_lower:
                    continue
                entries.append((name, param))
            entries.sort(key=lambda x: x[0])
            sample = entries[:max_params]
            total_sum = 0.0
            total_abs_sum = 0.0
            param_count = 0
            sampled_names = []
            for name, param in sample:
                total_sum += float(param.data.sum().item())
                total_abs_sum += float(param.data.abs().sum().item())
                param_count += 1
                sampled_names.append(name)
            return {
                "sum": total_sum,
                "abs_sum": total_abs_sum,
                "count": param_count,
                "names": sampled_names,
            }

    def get_buffer_checksum(self, max_buffers: int = 5) -> dict:
        """Compute checksum stats for a small sample of non-LoRA buffers."""
        with self.engine.train_mode():
            from verl.utils.megatron_utils import unwrap_model
            model = unwrap_model(self.engine.module)
            while isinstance(model, list):
                model = model[0]
            entries = []
            for name, buf in model.named_buffers():
                name_lower = name.lower()
                if 'lora' in name_lower or 'adapter' in name_lower:
                    continue
                entries.append((name, buf))
            entries.sort(key=lambda x: x[0])
            sample = entries[:max_buffers]
            total_sum = 0.0
            total_abs_sum = 0.0
            buf_count = 0
            sampled_names = []
            for name, buf in sample:
                if buf is None:
                    continue
                total_sum += float(buf.float().sum().item())
                total_abs_sum += float(buf.float().abs().sum().item())
                buf_count += 1
                sampled_names.append(name)
            return {
                "sum": total_sum,
                "abs_sum": total_abs_sum,
                "count": buf_count,
                "names": sampled_names,
            }

    def get_optimizer_param_counts(self) -> dict:
        """Count LoRA vs non-LoRA params referenced by optimizer param_groups."""
        optimizer = getattr(self.engine, "optimizer", None)
        if optimizer is None:
            return {"has_optimizer": False}

        # Build id -> name map for current model params (raw + unwrapped)
        from verl.utils.megatron_utils import unwrap_model
        id_to_name: dict[int, str] = {}

        modules = self.engine.module if isinstance(self.engine.module, list) else [self.engine.module]
        for mod in modules:
            for n, p in mod.named_parameters():
                id_to_name[id(p)] = n

        unwrapped = unwrap_model(self.engine.module)
        while isinstance(unwrapped, list):
            unwrapped = unwrapped[0]
        for n, p in unwrapped.named_parameters():
            id_to_name.setdefault(id(p), n)
        lora = 0
        non_lora = 0
        unknown = 0
        non_lora_names = []

        for group in optimizer.param_groups:
            params = group.get("params", [])
            for p in params:
                name = id_to_name.get(id(p))
                if name is None:
                    unknown += 1
                    continue
                name_lower = name.lower()
                if "lora" in name_lower or "adapter" in name_lower:
                    lora += 1
                else:
                    non_lora += 1
                    if len(non_lora_names) < 5:
                        non_lora_names.append(name)

        return {
            "has_optimizer": True,
            "lora": lora,
            "non_lora": non_lora,
            "unknown": unknown,
            "non_lora_names": non_lora_names,
        }

    def reinit_lora_weights(
        self,
        learning_rate: float | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
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
        import torch.nn.init as init
        from megatron.core.optimizer import ChainedOptimizer

        train_attn = True if train_attn is None else bool(train_attn)
        train_mlp = True if train_mlp is None else bool(train_mlp)
        train_unembed = True if train_unembed is None else bool(train_unembed)

        logger.info(
            f"[Rank {self.rank}] reinit_lora_weights: ENTRY (lr={learning_rate}, "
            f"train_attn={train_attn}, train_mlp={train_mlp}, train_unembed={train_unembed})"
        )

        # NOTE: Do NOT clear _session_gradients or _session_optimizer_states here!
        # Those contain saved state for OTHER sessions that must persist.
        # Only the current session's optimizer state needs to be reset (done below).

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

            # Keep only LoRA/adapter params trainable (stable optimizer param ordering across sessions).
            self._freeze_non_lora_params(model)

            # Find and reinitialize trainable LoRA parameters
            skipped: list[str] = []
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
                else:
                    skipped.append(name)

            # Keep disabled LoRA targets at exact zero so they do not affect forward passes.
            zeroed = self._zero_disabled_lora_params(
                model, train_attn=train_attn, train_mlp=train_mlp, train_unembed=train_unembed
            )
            if self.rank == 0 and zeroed:
                print(f"[Rank 0] reinit_lora_weights: zeroed_disabled_lora_params={zeroed}", flush=True)

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
                        if hasattr(state, '_inner_dicts'):
                            # ProxyDict from ChainedOptimizer
                            for inner_dict in state._inner_dicts:
                                inner_dict.clear()
                            logger.info(f"[Rank {self.rank}] Cleared ProxyDict optimizer state ({state_count} entries from {len(state._inner_dicts)} inner dicts)")
                        elif hasattr(state, 'clear'):
                            # Regular dict
                            state.clear()
                            logger.info(f"[Rank {self.rank}] Cleared optimizer state dict ({state_count} entries)")
                        else:
                            # Unknown type - try to clear via iteration
                            keys = list(state.keys()) if hasattr(state, 'keys') else []
                            for key in keys:
                                del state[key]
                            logger.info(f"[Rank {self.rank}] Cleared optimizer state via key deletion ({len(keys)} entries)")
                        opt_state_reset_count = state_count
                    else:
                        logger.warning(f"[Rank {self.rank}] Optimizer {i} has no inner optimizer or it's None")

            # Reset LR scheduler so fresh sessions start from step 0
            self._reset_lr_scheduler()

            # Rebuild optimizer and scheduler AFTER reinit to sync master params
            # with newly initialized LoRA weights. This prevents old master
            # params from overwriting fresh weights on the first step.
            self._rebuild_optimizer_and_scheduler()

        # Update instance learning_rate for future reference
        if learning_rate is not None:
            self.learning_rate = learning_rate
            # Ensure optimizer param_groups reflect requested LR after rebuild
            opt = getattr(self.engine, "optimizer", None)
            if opt is not None and hasattr(opt, "param_groups"):
                for pg in opt.param_groups:
                    pg["lr"] = learning_rate

        logger.info(f"[Rank {self.rank}] Reinitialized {reinit_count} LoRA params, reset {opt_state_reset_count} optimizer states, lr_updated={lr_updated}")
        if self.rank == 0:
            print(
                f"[Rank 0] reinit_lora_weights: reinit_count={reinit_count}, skipped={len(skipped)} "
                f"sample_skipped={skipped[:10]}",
                flush=True,
            )
        return {"status": "ok", "reinit_count": reinit_count, "opt_state_reset": opt_state_reset_count, "lr_updated": lr_updated, "learning_rate": learning_rate}

    @staticmethod
    def _classify_lora_param_target(name_lower: str) -> str:
        # MLP / experts
        if ".mlp." in name_lower or "linear_fc1" in name_lower or "linear_fc2" in name_lower:
            return "mlp"
        # Attention
        if (
            "self_attention" in name_lower
            or "self_attn" in name_lower
            or ".attn." in name_lower
            or "attention" in name_lower
        ):
            return "attn"
        # Unembed / output head
        if "lm_head" in name_lower or "output_layer" in name_lower or "unembed" in name_lower:
            return "unembed"
        return "other"

    def _freeze_non_lora_params(self, model) -> None:
        """Ensure only LoRA/adapter parameters are trainable.

        Keep the optimizer param ordering stable across sessions by leaving all LoRA/adapter
        params trainable. Train-target masking is enforced by projecting disabled LoRA params
        back to zero (see _zero_disabled_lora_params).
        """
        lora_trainable = 0
        non_lora_trainable = 0
        for name, param in model.named_parameters():
            name_lower = name.lower()
            if "lora" in name_lower or "adapter" in name_lower:
                param.requires_grad_(True)
                if param.requires_grad:
                    lora_trainable += 1
            else:
                param.requires_grad_(False)
                if param.requires_grad:
                    non_lora_trainable += 1
        if self.rank == 0:
            print(
                f"[Rank {self.rank}] _freeze_non_lora_params: "
                f"lora_trainable={lora_trainable}, non_lora_trainable={non_lora_trainable}",
                flush=True,
            )

    def _zero_disabled_lora_params(
        self,
        model,
        *,
        train_attn: bool,
        train_mlp: bool,
        train_unembed: bool,
    ) -> int:
        zeroed = 0
        for name, param in model.named_parameters():
            name_lower = name.lower()
            if "lora" not in name_lower and "adapter" not in name_lower:
                continue
            tgt = self._classify_lora_param_target(name_lower)
            if tgt == "attn":
                disable = not train_attn
            elif tgt == "mlp":
                disable = not train_mlp
            elif tgt == "unembed":
                disable = not train_unembed
            else:
                disable = False
            if disable:
                try:
                    param.data.zero_()
                    zeroed += 1
                except Exception:
                    pass
        return zeroed

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

    @staticmethod
    def _infer_target_modules_from_state_dict(
        state_dict: dict,
        fallback_modules: list[str],
    ) -> list[str]:
        import re

        pat = re.compile(
            r"\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_[AB]\.weight$"
        )
        present: set[str] = set()
        for key in state_dict.keys():
            m = pat.search(str(key))
            if m is not None:
                present.add(m.group(1))

        if not present:
            return fallback_modules

        ordered = [name for name in fallback_modules if name in present]
        extras = sorted(present.difference(set(fallback_modules)))
        return ordered + extras

    def save_checkpoint(
        self,
        save_path: str,
        step_count: int = 0,
        actual_rank: int | None = None,
        use_per_expert_lora: bool = False,
    ) -> dict:
        """Save checkpoint: LoRA weights + config + training metadata.

        IMPORTANT: ALL ranks must call this method because get_lora_state_dict()
        uses NCCL collectives. Only rank 0 saves to disk.

        Args:
            save_path: Directory path to save checkpoint files.
            step_count: Current training step (passed from MegatronWorkerGroup).
            actual_rank: Actual LoRA rank for current session (Phase 7).
                         If None, falls back to self.lora_rank (max_lora_rank).
            use_per_expert_lora: If True, expand shared MLP LoRA to per-expert format for MoE.

        Returns:
            Dict with training metadata (rank 0 only, others return empty).
        """
        import json
        import os
        import torch

        from safetensors.torch import save_file
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path

        # ALL ranks must call get_lora_state_dict - it uses NCCL collectives
        # Only rank 0 gets actual data, others get empty dict
        state_dict = self.get_lora_state_dict(use_per_expert_lora=use_per_expert_lora)

        os.makedirs(save_path, exist_ok=True)

        # Save distributed adapter state for training resume (per-rank mp_rank_*_adapter.pt).
        # ALL ranks must participate due to NCCL collectives.
        effective_rank = actual_rank if actual_rank is not None else self.lora_rank
        self.save_adapter_state(
            checkpoint_path=save_path,
            actual_rank=effective_rank,
            trainer_rank=self.lora_rank,
        )

        # Save per-rank optimizer shard (distributed optimizer state lives on each rank).
        rank_path = _get_rank_checkpoint_path(save_path)
        optimizer_file = rank_path + "_optimizer.pt"
        torch.save(self._capture_optimizer_state(), optimizer_file)

        # Only rank 0 saves PEFT-format artifacts used by vLLM.
        if self.rank != 0:
            return {}

        # 1. LoRA weights (PEFT format)
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # 2. LoRA config
        # Use actual session rank (Phase 7) or fall back to max_lora_rank
        try:
            cfg = get_model_config(self.base_model)
            model_is_mla = cfg.is_mla
            model_is_moe = cfg.is_moe
        except ValueError:
            model_is_mla = False
            model_is_moe = False
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ] if not model_is_mla else [
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
        ]
        # MoE: include MLP/expert targets only when exporting per-expert LoRA.
        if (not model_is_moe) or use_per_expert_lora:
            target_modules += ["gate_proj", "up_proj", "down_proj"]
        target_modules = self._infer_target_modules_from_state_dict(
            state_dict=state_dict,
            fallback_modules=target_modules,
        )
        # Include attention modules; add MLP modules only when trained
        config = {
            "r": effective_rank,
            "lora_alpha": effective_rank * 2,
            "target_modules": target_modules,
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

        # Note: state_dict NOT included in return value to avoid OOM on API server
        # when transferring 37k+ MoE LoRA tensors through Ray. vLLM loads from path.
        return meta

    def save_lora_weights(
        self,
        save_path: str,
        step_count: int = 0,
        actual_rank: int | None = None,
        use_per_expert_lora: bool = False,
    ) -> dict:
        """Save LoRA weights in PEFT format for sampling (no optimizer/resume artifacts).

        This is the minimal export required by `save_weights_for_sampler` to hot-load LoRA in vLLM.
        It intentionally avoids saving per-rank adapter shards and optimizer shards, because those
        are not required for sampling and can exceed memory/time budgets on large MoE models.
        """
        import json
        import os

        from safetensors.torch import save_file

        # ALL ranks must call get_lora_state_dict - it uses NCCL collectives.
        state_dict = self.get_lora_state_dict(use_per_expert_lora=use_per_expert_lora)

        os.makedirs(save_path, exist_ok=True)

        if self.rank != 0:
            return {}

        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        effective_rank = actual_rank if actual_rank is not None else self.lora_rank
        try:
            cfg = get_model_config(self.base_model)
            model_is_mla = cfg.is_mla
            model_is_moe = cfg.is_moe
        except ValueError:
            model_is_mla = False
            model_is_moe = False
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ] if not model_is_mla else [
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
        ]
        # MoE: include MLP/expert targets only when exporting per-expert LoRA.
        if (not model_is_moe) or use_per_expert_lora:
            target_modules += ["gate_proj", "up_proj", "down_proj"]
        target_modules = self._infer_target_modules_from_state_dict(
            state_dict=state_dict,
            fallback_modules=target_modules,
        )
        config = {
            "r": effective_rank,
            "lora_alpha": effective_rank * 2,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": self.base_model,
        }
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        meta = {"current_step": step_count, "learning_rate": self.learning_rate}
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info(f"[MegatronRankWorker] Saved LoRA weights to {abs_path} (step={step_count})")
        return meta

    def load_optimizer_state(self, checkpoint_path: str) -> dict:
        """Load per-rank optimizer shard from disk (if present)."""
        import os

        import torch
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path

        rank_path = _get_rank_checkpoint_path(checkpoint_path)
        optimizer_file = rank_path + "_optimizer.pt"
        if not os.path.isfile(optimizer_file):
            raise FileNotFoundError(
                f"Optimizer restore requested, but optimizer shard not found: {optimizer_file}"
            )

        state_dict = torch.load(optimizer_file, map_location="cpu")
        with self.engine.train_mode():
            self._restore_optimizer_state(state_dict)

        if self.rank == 0:
            return {"status": "ok", "optimizer_file": optimizer_file}
        return {}


    # ========================================================================
    # Phase 6: Multi-Session Support Methods
    # ========================================================================

    def load_adapter_state(
        self,
        checkpoint_path: str,
        actual_rank: int | None = None,
        trainer_rank: int | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
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

            train_attn = True if train_attn is None else bool(train_attn)
            train_mlp = True if train_mlp is None else bool(train_mlp)
            train_unembed = True if train_unembed is None else bool(train_unembed)

            # Keep only LoRA/adapter params trainable; then project disabled targets to zero.
            self._freeze_non_lora_params(unwrapped)
            self._zero_disabled_lora_params(
                unwrapped, train_attn=train_attn, train_mlp=train_mlp, train_unembed=train_unembed
            )

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
        import torch
        if self._sticky_train_mode_ctx is not None:
            self._release_sticky_train_mode(reason="shutdown", snapshot_gradients=False)
        # Clear session state
        self._session_gradients.clear()
        self._session_optimizer_states.clear()
        self._current_session_id = None

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()



class MegatronSessionStateManager:
    """Manages session state (LoRA checkpoint paths) for MegatronWorkerGroup.

    Enables multiple sessions to share a single Megatron trainer by tracking
    checkpoint paths per session. When sessions switch, the manager provides
    paths for saving outgoing session state and loading incoming session state.

    Storage layout:
        {base_path}/{session_id}_checkpoint/
            (LoRA weights saved via MegatronWorkerGroup.save_adapter_state)
    """

    def __init__(self, base_path: str | None = None):
        """Initialize the session state manager.

        Args:
            base_path: Root directory for all session checkpoints.
        """
        if base_path is None:
            base_path = os.environ.get("MINT_MEGATRON_SESSIONS_BASE_PATH") or os.path.join(
                PFS_TINKER_PATH,
                "checkpoints",
                "megatron_sessions",
            )

        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self._session_metadata: dict[str, dict] = {}  # session_id -> {step, lr, actual_rank}
        logger.info(f"[MegatronSessionStateManager] Initialized with base_path={base_path}")

    def get_session_path(self, session_id: str) -> str:
        """Get checkpoint directory path for a session."""
        return os.path.join(self.base_path, f"{session_id}_checkpoint")

    def session_exists(self, session_id: str) -> bool:
        """Check if a session has saved state."""
        session_path = self.get_session_path(session_id)
        # Check for mp_rank_*_adapter.pt files (Megatron distributed save format)
        # The save creates files like: mp_rank_00_000_000_adapter.pt
        import glob
        adapter_files = glob.glob(os.path.join(session_path, "mp_rank_*_adapter.pt"))
        return len(adapter_files) > 0

    def save_metadata(self, session_id: str, step: int, lr: float, actual_rank: int | None = None):
        """Save session metadata (step count, learning rate, actual rank)."""
        self._session_metadata[session_id] = {
            "step": step,
            "lr": lr,
            "actual_rank": actual_rank,
        }

    def get_metadata(self, session_id: str) -> dict | None:
        """Get session metadata if exists."""
        return self._session_metadata.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete session checkpoint and metadata."""
        import shutil
        session_path = self.get_session_path(session_id)
        deleted = False
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
            deleted = True
        if session_id in self._session_metadata:
            del self._session_metadata[session_id]
            deleted = True
        if deleted:
            logger.info(f"[MegatronSessionStateManager] Deleted session {session_id}")
        return deleted


@ray.remote(num_gpus=0, num_cpus=0)
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
        init_actor_observability()
        self.base_model = base_model
        self.lora_rank = lora_rank  # This is max_lora_rank for Phase 7
        self.learning_rate = learning_rate
        self.config = distributed_config or DistributedConfig()

        self.workers: list[ray.actor.ActorHandle] = []
        self.placement_group = None
        self._step_count = 0
        self._current_session: str | None = None  # Phase 6: session tracking
        self._actual_rank: int | None = None  # Phase 7: actual LoRA rank for current session
        self._session_manager = MegatronSessionStateManager()  # Issue #44: session state management
        self._master_addr: str | None = None
        self._master_port: int | None = None

        self._initialize()

    def get_rss_bytes(self) -> int:
        with open("/proc/self/statm", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
        rss_pages = int(parts[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return rss_pages * page_size

    def get_master_addr(self) -> str | None:
        return self._master_addr

    def _start_slow_group_watchdog(
        self,
        *,
        op: str,
        session_id: str | None,
        extra: str = "",
    ) -> tuple[threading.Event, threading.Thread] | None:
        timeout_s = _env_float("MINT_MEGATRON_STACK_DUMP_TIMEOUT_S", 0.0)
        if timeout_s <= 0:
            return None
        stack_limit = max(8, _env_int("MINT_MEGATRON_STACK_DUMP_LIMIT", 96))
        stop_event = threading.Event()
        started_at = time.perf_counter()

        def _watch() -> None:
            if stop_event.wait(timeout_s):
                return
            elapsed_s = time.perf_counter() - started_at
            try:
                stack_dump = _collect_python_thread_stacks(limit=stack_limit)
            except Exception as e:
                logger.error(
                    "[MegatronWorkerGroup] slow watchdog failed op=%s session=%s elapsed_s=%.3f: %s: %s",
                    op,
                    session_id,
                    float(elapsed_s),
                    type(e).__name__,
                    e,
                )
                return
            logger.error(
                "[MegatronWorkerGroup] slow watchdog timeout op=%s session=%s elapsed_s=%.3f %s\n%s",
                op,
                session_id,
                float(elapsed_s),
                extra,
                stack_dump,
            )

        thread = threading.Thread(
            target=_watch,
            name=f"mg-group-{op}-watchdog",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    def _stop_slow_group_watchdog(self, token: tuple[threading.Event, threading.Thread] | None) -> None:
        if token is None:
            return
        stop_event, thread = token
        stop_event.set()
        try:
            thread.join(timeout=0.05)
        except Exception:
            pass

    def _initialize(self):
        """Create placement group, spawn workers, then initialize them all together."""
        world_size = self.config.world_size

        # Create placement group with N GPU bundles
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(world_size)]
        from .volc_placement import parse_csv

        allowed_ips = parse_csv(os.environ.get("MINT_MEGATRON_NODE_IPS_CSV", ""))
        if allowed_ips:
            from .volc_placement import build_node_affinity_gpu_bundles, select_free_nodes_from_allowed_ips

            node_ips, gpus_per_node = select_free_nodes_from_allowed_ips(
                allowed_node_ips=allowed_ips,
                required_gpus=world_size,
            )
            bundles = build_node_affinity_gpu_bundles(
                node_ips=node_ips,
                gpus_per_node=gpus_per_node,
                required_gpus=world_size,
                cpu_per_gpu=1,
            )
            logger.info(f"[MegatronWorkerGroup] Volcano placement allowlist nodes={node_ips}")
        else:
            preferred_node_ips = _preferred_worker_node_ips_for_model(self.base_model)
            if preferred_node_ips:
                from .volc_placement import build_node_affinity_gpu_bundles

                gpus_per_node = 8
                nodes_needed = (int(world_size) + int(gpus_per_node) - 1) // int(gpus_per_node)
                if len(preferred_node_ips) < nodes_needed:
                    raise ValueError(
                        f"MINT_MODEL_NODE_IPS_JSON too short for base_model={self.base_model!r}: "
                        f"need {nodes_needed} nodes for world_size={world_size}, got {len(preferred_node_ips)}"
                    )
                node_ips = preferred_node_ips[:nodes_needed]
                bundles = build_node_affinity_gpu_bundles(
                    node_ips=node_ips,
                    gpus_per_node=gpus_per_node,
                    required_gpus=world_size,
                    cpu_per_gpu=1,
                )
                logger.info(f"[MegatronWorkerGroup] Model placement preferred nodes={node_ips}")
        # PACK: try to colocate but allow multi-node for large models (K2: 16+ GPUs)
        # STRICT_PACK would require single node, blocking on 8-GPU nodes
        pg_name = f"{_make_megatron_actor_name(self.base_model)}_pg"
        try:
            self.placement_group = ray.util.get_placement_group(pg_name)
        except Exception:
            self.placement_group = ray.util.placement_group(
                bundles,
                strategy="PACK",
                name=pg_name,
                lifetime="detached",
            )
        ray.get(self.placement_group.ready())

        logger.info(f"[MegatronWorkerGroup] Placement group ready with {world_size} GPUs")

        # Runtime env for workers
        is_mla = False
        disable_nccl_ib = False
        try:
            from .model_registry import get_model_config

            cfg = get_model_config(self.base_model)
            is_mla = bool(cfg.is_mla)
            disable_nccl_ib = bool(getattr(cfg, "train_nccl_ib_disable", False))
        except Exception:
            is_mla = False
            disable_nccl_ib = False

        from ..config import otel_env_vars
        runtime_env = {
            "env_vars": {
                "PYTHONPATH": PFS_PYTHONPATH,
                "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",  # Avoid stale bytecode on PFS
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # Reduce memory fragmentation
                **otel_env_vars(),
                # TransformerEngine debug - see why attention backends are disabled
                "NVTE_DEBUG": "1",
                "NVTE_DEBUG_LEVEL": "2",
                # Allow TE DotProductAttention backends; Megatron flash attention asserts these are 0.
                "NVTE_FUSED_ATTN": "0" if is_mla else "1",
                "NVTE_UNFUSED_ATTN": "0" if is_mla else "1",
                **otel_env_vars(),
            },
        }

        # Forward MoE LoRA export knobs into rank workers.
        for k in (
            "MINT_MOE_LORA_SPARSE_EXPERT_EXPORT",
            "MINT_MOE_LORA_SHARED_EXPERT_EXPORT",
        ):
            v = os.environ.get(k)
            if v is not None:
                runtime_env["env_vars"][k] = v

        # Forward train-mode/diagnostic and DeepEP knobs into rank workers.
        # Rank workers run on GPU nodes; they do NOT inherit the API server's
        # environment unless we explicitly forward selected env vars.
        for k in (
            "MINT_MEGATRON_STICKY_TRAIN_MODE",
            "MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S",
            "MINT_MEGATRON_STICKY_CLOSE_ON_OPTIM",
            "MINT_MEGATRON_STICKY_TIMING_DIAG",
            "MINT_MEGATRON_STACK_DUMP_TIMEOUT_S",
            "MINT_MEGATRON_STACK_DUMP_LIMIT",
            "MINT_MEGATRON_ENABLE_DEEPEP",
            "MINT_MEGATRON_MOE_TOKEN_DISPATCHER_TYPE",
            "MINT_MEGATRON_MOE_FLEX_DISPATCHER_BACKEND",
            "MINT_MEGATRON_MOE_ROUTER_DTYPE",
            "MINT_MEGATRON_MOE_DEEPEP_NUM_SMS",
        ):
            v = os.environ.get(k)
            if v is not None:
                runtime_env["env_vars"][k] = v
        if disable_nccl_ib or os.environ.get("MINT_NCCL_IB_DISABLE", "0") == "1":
            runtime_env["env_vars"]["NCCL_IB_DISABLE"] = "1"
        else:
            # Keep default NCCL transport selection; only expose IB toggle.
            runtime_env["env_vars"]["NCCL_IB_DISABLE"] = "0"

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
        self._master_addr = master_addr
        self._master_port = int(master_port)

        # Spawn workers - __init__ is lightweight, no distributed init yet
        for rank in range(world_size):
            logger.info(f"[MegatronWorkerGroup] Spawning rank {rank}")
            worker = MegatronRankWorker.options(
                num_gpus=1,  # Ray sets CUDA_VISIBLE_DEVICES before process starts
                num_cpus=0,
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
        logger.info("[MegatronWorkerGroup] Calling initialize() on all workers...")
        ray.get([w.initialize.remote() for w in self.workers])

        logger.info(f"[MegatronWorkerGroup] All {world_size} workers initialized and ready")

    def _get_lora_weight_norm(self) -> float:
        """Get LoRA weight norm from rank 0 for debugging."""
        try:
            result = ray.get(self.workers[0].get_lora_weight_norm.remote())
            return result
        except Exception as e:
            logger.warning(f"[MegatronWorkerGroup] Failed to get LoRA norm: {e}")
            return -1.0

    def _get_lora_weight_checksum(self) -> dict:
        """Get LoRA checksum stats from rank 0 for debugging."""
        try:
            result = ray.get(self.workers[0].get_lora_weight_checksum.remote())
            return result
        except Exception as e:
            logger.warning(f"[MegatronWorkerGroup] Failed to get LoRA checksum: {e}")
            return {"sum": 0.0, "abs_sum": 0.0, "count": 0}

    def _get_base_weight_checksum(self) -> dict:
        """Get base weight checksum stats from rank 0 for debugging."""
        try:
            result = ray.get(self.workers[0].get_base_weight_checksum.remote())
            return result
        except Exception as e:
            logger.warning(f"[MegatronWorkerGroup] Failed to get base checksum: {e}")
            return {"sum": 0.0, "abs_sum": 0.0, "count": 0, "names": []}

    def _get_buffer_checksum(self) -> dict:
        """Get buffer checksum stats from rank 0 for debugging."""
        try:
            result = ray.get(self.workers[0].get_buffer_checksum.remote())
            return result
        except Exception as e:
            logger.warning(f"[MegatronWorkerGroup] Failed to get buffer checksum: {e}")
            return {"sum": 0.0, "abs_sum": 0.0, "count": 0, "names": []}

    def _get_optimizer_param_counts(self) -> dict:
        """Get optimizer param composition from rank 0 for debugging."""
        try:
            result = ray.get(self.workers[0].get_optimizer_param_counts.remote())
            return result
        except Exception as e:
            logger.warning(f"[MegatronWorkerGroup] Failed to get optimizer param counts: {e}")
            return {"has_optimizer": False}

    def _swap_session_on_workers(self, new_session_id: str) -> None:
        """Swap session state (gradients + optimizer) on all workers.

        This calls MegatronRankWorker.swap_session_state() on each worker to:
        1. Save outgoing session's gradients and optimizer state to memory
        2. Restore incoming session's gradients and optimizer state (or init fresh)

        Must be called during session switch to ensure optimizer momentum isolation.

        Args:
            new_session_id: Session ID to switch to.
        """
        logger.info(f"[MegatronWorkerGroup] Swapping session state on workers to {new_session_id}")
        futures = [w.swap_session_state.remote(new_session_id) for w in self.workers]
        ray.get(futures)
        logger.info("[MegatronWorkerGroup] Session state swapped on all workers")

    def _ensure_session_loaded(
        self,
        session_id: str | None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> None:
        """Ensure the specified session's state is loaded (LoRA + optimizer + gradients).

        If a different session is currently loaded, this saves its state first,
        then loads the requested session's state.

        State managed:
        - LoRA weights: saved/loaded to disk via save_adapter_state/load_adapter_state
        - Optimizer state: swapped in memory via worker-level swap_session_state
        - Gradients: swapped in memory via worker-level swap_session_state

        This is the MegatronWorkerGroup equivalent of TrainingWorker._ensure_session_loaded().
        Fixes Issue #44: Complete session state now properly saved/loaded on session switch.

        Args:
            session_id: Session ID to load state for. If None, no-op.
        """
        if session_id is None:
            return

        if self._current_session == session_id:
            # Already loaded
            logger.debug(f"[MegatronWorkerGroup] Session {session_id} already loaded")
            return

        train_attn = True if train_attn is None else bool(train_attn)
        train_mlp = True if train_mlp is None else bool(train_mlp)
        train_unembed = True if train_unembed is None else bool(train_unembed)

        timing = _env_flag("MINT_TIMING_DIAG", default=False)
        t0 = time.perf_counter() if timing else 0.0

        logger.info(
            f"[MegatronWorkerGroup] Session switch: {self._current_session} -> {session_id}"
        )

        # DEBUG: Log LoRA norm/checksum before switch
        norm_before = self._get_lora_weight_norm()
        checksum_before = self._get_lora_weight_checksum()
        base_checksum_before = self._get_base_weight_checksum()
        buffer_checksum_before = self._get_buffer_checksum()
        optim_param_counts = self._get_optimizer_param_counts()
        print(
            f"[DEBUG] Session switch {self._current_session} -> {session_id}: "
            f"LoRA norm BEFORE = {norm_before:.6f}, checksum={checksum_before}, "
            f"base_checksum={base_checksum_before}, buffer_checksum={buffer_checksum_before}, "
            f"optim_params={optim_param_counts}",
            flush=True,
        )

        # Save outgoing session's LoRA weights to disk
        t_save0 = time.perf_counter() if timing else 0.0
        if self._current_session is not None:
            old_path = self._session_manager.get_session_path(self._current_session)
            logger.info(f"[MegatronWorkerGroup] Saving outgoing session {self._current_session}")
            self.save_adapter_state(old_path)
            # Save metadata (step count, learning rate, actual rank)
            self._session_manager.save_metadata(
                self._current_session,
                self._step_count,
                self.learning_rate,
                self._actual_rank,
            )
        t_save1 = time.perf_counter() if timing else 0.0

        # Swap session state on workers (gradients + optimizer)
        # This saves outgoing session's gradients/optimizer to CPU memory,
        # and restores incoming session's (or resets for new sessions)
        t_swap0 = time.perf_counter() if timing else 0.0
        self._swap_session_on_workers(session_id)
        t_swap1 = time.perf_counter() if timing else 0.0

        # Load new session's LoRA weights from disk (or reset for new session)
        t_load0 = time.perf_counter() if timing else 0.0
        session_exists = self._session_manager.session_exists(session_id)
        logger.info(f"[MegatronWorkerGroup] session_exists({session_id}) = {session_exists}")
        if session_exists:
            new_path = self._session_manager.get_session_path(session_id)
            meta = self._session_manager.get_metadata(session_id)
            actual_rank = meta.get("actual_rank") if meta else None
            logger.info(f"[MegatronWorkerGroup] Loading session {session_id} from {new_path}")
            self.load_adapter_state(
                new_path,
                actual_rank=actual_rank,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            # Restore metadata
            if meta:
                self._step_count = meta.get("step", 0)
                self.learning_rate = meta.get("lr", self.learning_rate)
                self._actual_rank = meta.get("actual_rank", self.lora_rank)
        else:
            # New session: reinitialize LoRA weights
            logger.info(f"[MegatronWorkerGroup] New session {session_id}, reinitializing LoRA")
            self.reinit_lora_weights(
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            self._step_count = 0
            self._actual_rank = self.lora_rank
        t_load1 = time.perf_counter() if timing else 0.0

        # Reset expert_bias on every session switch to avoid cross-session leakage.
        # expert_bias is not saved/restored with LoRA checkpoints.
        t_bias0 = time.perf_counter() if timing else 0.0
        try:
            self.reset_expert_bias()
        except Exception as e:
            logger.warning(f"[MegatronWorkerGroup] Failed to reset expert_bias for {session_id}: {e}")
        t_bias1 = time.perf_counter() if timing else 0.0

        # DEBUG: Log LoRA norm/checksum after switch
        norm_after = self._get_lora_weight_norm()
        checksum_after = self._get_lora_weight_checksum()
        base_checksum_after = self._get_base_weight_checksum()
        buffer_checksum_after = self._get_buffer_checksum()
        print(
            f"[DEBUG] Session switch {self._current_session} -> {session_id}: "
            f"LoRA norm AFTER = {norm_after:.6f}, checksum={checksum_after}, "
            f"base_checksum={base_checksum_after}, buffer_checksum={buffer_checksum_after}",
            flush=True,
        )

        self._current_session = session_id
        if timing:
            t1 = time.perf_counter()
            logger.info(
                f"[MegatronWorkerGroup] session_switch timing: "
                f"save_s={t_save1 - t_save0:.3f} "
                f"swap_s={t_swap1 - t_swap0:.3f} "
                f"load_s={t_load1 - t_load0:.3f} "
                f"reset_bias_s={t_bias1 - t_bias0:.3f} "
                f"total_s={t1 - t0:.3f} "
                f"session_exists={session_exists}"
            )

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
        rollout_correction_config: dict | None = None,
        session_id: str | None = None,
        reset_bias: bool | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Run forward-backward on all workers.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type.
            loss_fn_config: Optional loss config.
            rollout_correction_config: Optional verl rollout correction config passed to policy_loss.
            session_id: Session ID for multi-tenant gradient isolation.
            reset_bias: If True, reset expert_bias to zero before forward pass.
                If None, uses MINT_RESET_EXPERT_BIAS (default False for training).
                This ensures logprobs match vLLM (which always has bias=0).

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
        loss_fn_config = loss_fn_config or {}

        timing = _env_flag("MINT_TIMING_DIAG", default=False)
        t0 = time.perf_counter() if timing else 0.0

        # Issue #44: Ensure correct session's LoRA weights are loaded before training
        # This saves outgoing session state and loads incoming session state
        effective_session_id = session_id or self._current_session
        watchdog = self._start_slow_group_watchdog(
            op="forward_backward",
            session_id=effective_session_id,
            extra=f"items={len(data_items)} loss_fn={loss_fn}",
        )
        try:
            self._ensure_session_loaded(
                effective_session_id,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            t1 = time.perf_counter() if timing else 0.0

            # Send raw data_items to workers (TensorDict created locally on each worker
            # to avoid Ray serialization issues with nested tensors)
            t2 = time.perf_counter() if timing else 0.0
            futures = [
                w.forward_backward.remote(
                    data_items, loss_fn, loss_fn_config, rollout_correction_config, effective_session_id, reset_bias
                )
                for w in self.workers
            ]
            results = ray.get(futures)
            t3 = time.perf_counter() if timing else 0.0
        finally:
            self._stop_slow_group_watchdog(watchdog)
        if timing:
            logger.info(
                f"[MegatronWorkerGroup] forward_backward timing: "
                f"ensure_session_s={t1 - t0:.3f} "
                f"worker_fb_s={t3 - t2:.3f} "
                f"items={len(data_items)} loss_fn={loss_fn}"
            )

        # Pick the first non-empty result (pipeline last stage).
        rank0_result = next((r for r in results if isinstance(r, dict) and r), {})
        loss_value = rank0_result.get("loss_value", 0.0)
        num_tokens = rank0_result.get("num_tokens", 0)
        valid_count = rank0_result.get("valid_count")
        if valid_count is None:
            valid_count = len(data_items)

        if math.isnan(loss_value) or math.isinf(loss_value):
            raise ValueError(f"non-finite loss_value={loss_value!r}")

        metrics = {
            "loss:mean": float(loss_value),
            "num_samples:sum": float(valid_count),
            "num_tokens:sum": float(num_tokens),
        }
        metrics["routing_replay_enabled:mean"] = float(
            rank0_result.get("routing_replay_enabled", 0.0)
        )
        metrics["routing_replay_items:sum"] = float(
            rank0_result.get("routing_replay_items", 0.0)
        )
        metrics["train_mode_enter_ms:mean"] = float(rank0_result.get("train_mode_enter_ms", 0.0))
        metrics["train_mode_exit_ms:mean"] = float(rank0_result.get("train_mode_exit_ms", 0.0))
        metrics["train_mode_reused:mean"] = float(rank0_result.get("train_mode_reused", 0.0))
        metrics["grad_restore_skipped:mean"] = float(rank0_result.get("grad_restore_skipped", 0.0))
        metrics["forward_backward_batch_ms:mean"] = float(rank0_result.get("forward_backward_batch_ms", 0.0))
        metrics["train_mode_enter_total:sum"] = float(rank0_result.get("train_mode_enter_total", 0.0))
        metrics["train_mode_reuse_total:sum"] = float(rank0_result.get("train_mode_reuse_total", 0.0))
        metrics["train_mode_exit_total:sum"] = float(rank0_result.get("train_mode_exit_total", 0.0))

        # Add PPO metrics if present (now pre-extracted as scalars)
        # importance_sampling uses PPO loss with epsilon=inf, so include it here
        n_ppo = rank0_result.get("n_ppo_results", 0)
        if loss_fn in ("ppo", "importance_sampling") and n_ppo > 0:
            clip_frac_sum = rank0_result.get("clip_frac_sum", 0.0)
            ratio_mean_sum = rank0_result.get("ratio_mean_sum", 0.0)
            metrics["clipfrac:mean"] = float(clip_frac_sum / n_ppo)
            metrics["ratio:mean"] = float(ratio_mean_sum / n_ppo)

        # Add debug metrics if present (precision difference between rollout and training)
        debug_metrics = rank0_result.get("debug_metrics", {})
        if debug_metrics:
            # Filter out None and NaN values to avoid pydantic validation errors
            # (orjson converts NaN to null, which causes pydantic failures)
            filtered_debug_metrics = {
                k: v for k, v in debug_metrics.items()
                if v is not None and isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
            }
            if filtered_debug_metrics:
                metrics.update(filtered_debug_metrics)
                logger.info(f"[MegatronWorkerGroup] Debug metrics: {filtered_debug_metrics}")

        logger.info(f"[MegatronWorkerGroup] forward_backward ({loss_fn}): loss={loss_value:.4f}")

        loss_fn_outputs = rank0_result.get("loss_fn_outputs")
        if not isinstance(loss_fn_outputs, list):
            raise ValueError(f"loss_fn_outputs missing/invalid: {type(loss_fn_outputs)}")
        if len(loss_fn_outputs) != len(data_items):
            raise ValueError(f"loss_fn_outputs length {len(loss_fn_outputs)} != request length {len(data_items)}")
        for i, output in enumerate(loss_fn_outputs):
            if not isinstance(output, dict):
                raise ValueError(f"loss_fn_outputs[{i}] invalid type {type(output)}")
            loss = output.get("loss")
            if not isinstance(loss, dict):
                raise ValueError(f"loss_fn_outputs[{i}].loss missing/invalid: {type(loss)}")
            loss_data = loss.get("data")
            if not (isinstance(loss_data, list) and len(loss_data) == 1):
                raise ValueError(f"loss_fn_outputs[{i}].loss.data invalid: {loss_data!r}")
            v = loss_data[0]
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise ValueError(f"loss_fn_outputs[{i}].loss.data[0] non-finite: {v!r}")

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
        }

    def train_step(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
        rollout_correction_config: dict | None = None,
        learning_rate: float | None = None,
        session_id: str | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Combined forward_backward + optim_step in a single actor call.

        This keeps both operations within the same MegatronWorkerGroup method
        invocation, which is required for some param_offload flows.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type.
            loss_fn_config: Optional loss config.
            rollout_correction_config: Optional verl rollout correction config passed to policy_loss.
            learning_rate: Optional LR override (defaults to current group's LR).
            session_id: Session ID for multi-tenant gradient isolation.

        Returns:
            forward_backward result dict with optim_step metrics merged in.
        """
        effective_session_id = session_id or self._current_session
        if effective_session_id is None:
            raise ValueError("train_step requires session_id (no current session set)")

        fb_result = self.forward_backward(
            data_items=data_items,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            rollout_correction_config=rollout_correction_config,
            session_id=effective_session_id,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

        lr = learning_rate if learning_rate is not None else self.learning_rate
        opt_result = self.optim_step(
            lr,
            session_id=effective_session_id,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

        metrics = dict(fb_result.get("metrics", {}))
        metrics.update(opt_result.get("metrics", {}))
        fb_result["metrics"] = metrics
        return fb_result

    def forward(
        self,
        data_items: list[dict],
        session_id: str | None = None,  # Accepted for API consistency with TrainingWorker
        reset_bias: bool | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Run forward pass only on all workers. Returns per-token logprobs.

        Similar to forward_backward but skips gradient computation.
        Used for computing reference model logprobs in DPO/SL.

        Args:
            data_items: List of Tinker Datum dicts.
            session_id: Session ID for loading correct LoRA weights (Issue #44).
            reset_bias: If True, reset expert_bias to zero before forward pass.
                If None, uses MINT_RESET_EXPERT_BIAS (default True for logprob-only).
                This ensures logprobs match vLLM (which always has bias=0).

        Returns:
            Dict with loss_fn_outputs (including per-token logprobs) and metrics.
        """
        # Issue #44: Ensure correct session's LoRA weights are loaded before forward
        effective_session_id = session_id or self._current_session
        self._ensure_session_loaded(
            effective_session_id,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

        # Send raw data_items to workers (TensorDict created locally on each worker
        # to avoid Ray serialization issues with nested tensors)
        futures = [w.forward.remote(data_items, reset_bias) for w in self.workers]
        results = ray.get(futures)

        # Pick the first non-empty result (pipeline last stage).
        rank0_result = next((r for r in results if isinstance(r, dict) and r), {})
        loss_value = rank0_result.get("loss_value", 0.0)
        num_tokens = rank0_result.get("num_tokens", 0)
        valid_count = rank0_result.get("valid_count")
        if valid_count is None:
            valid_count = len(data_items)

        if math.isnan(loss_value) or math.isinf(loss_value):
            raise ValueError(f"non-finite loss_value={loss_value!r}")

        loss_fn_outputs = rank0_result.get("loss_fn_outputs")
        if not isinstance(loss_fn_outputs, list):
            raise ValueError(f"loss_fn_outputs missing/invalid: {type(loss_fn_outputs)}")
        if len(loss_fn_outputs) != len(data_items):
            raise ValueError(f"loss_fn_outputs length {len(loss_fn_outputs)} != request length {len(data_items)}")
        for i, output in enumerate(loss_fn_outputs):
            if not isinstance(output, dict):
                raise ValueError(f"loss_fn_outputs[{i}] invalid type {type(output)}")
            loss = output.get("loss")
            if not isinstance(loss, dict):
                raise ValueError(f"loss_fn_outputs[{i}].loss missing/invalid: {type(loss)}")
            loss_data = loss.get("data")
            if not (isinstance(loss_data, list) and len(loss_data) == 1):
                raise ValueError(f"loss_fn_outputs[{i}].loss.data invalid: {loss_data!r}")
            v = loss_data[0]
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise ValueError(f"loss_fn_outputs[{i}].loss.data[0] non-finite: {v!r}")
        log_probs = rank0_result.get("log_probs")  # Per-token log_probs tensor

        metrics = {
            "loss:mean": float(loss_value),
            "num_samples:sum": float(valid_count),
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
        session_id: str | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Run optimizer step on all workers.

        Uses per-session gradient storage for multi-tenant isolation.
        Restores session's gradients before applying optimizer step.

        Args:
            learning_rate: Learning rate for this step.
            session_id: Session ID for multi-tenant gradient isolation.

        Returns:
            Dict with metrics including grad_norm from rank 0.
        """
        timing = _env_flag("MINT_TIMING_DIAG", default=False)
        t0 = time.perf_counter() if timing else 0.0

        # Issue #44: Ensure correct session's LoRA weights are loaded before optim step
        effective_session_id = session_id or self._current_session
        self._ensure_session_loaded(
            effective_session_id,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )
        t1 = time.perf_counter() if timing else 0.0

        t2 = time.perf_counter() if timing else 0.0
        futures = [
            w.optim_step.remote(
                learning_rate,
                effective_session_id,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            for w in self.workers
        ]
        results = ray.get(futures)
        t3 = time.perf_counter() if timing else 0.0
        if timing:
            logger.info(
                f"[MegatronWorkerGroup] optim_step timing: "
                f"ensure_session_s={t1 - t0:.3f} "
                f"worker_optim_s={t3 - t2:.3f}"
            )

        # Rank 0 returns the actual result with grad_norm
        rank0_result = results[0]
        grad_norm = rank0_result.get("grad_norm", 0.0)
        lr = rank0_result.get("lr", learning_rate)

        self._step_count += 1

        print(
            f"[MegatronWorkerGroup] optim_step: grad_norm={grad_norm:.4f}, "
            f"lr={lr}, step={self._step_count}",
            flush=True
        )

        return {
            "metrics": {
                "step": self._step_count,
                "grad_norm": grad_norm,
                "lr": lr,
                "train_mode_enter_ms:mean": float(rank0_result.get("train_mode_enter_ms", 0.0)),
                "train_mode_exit_ms:mean": float(rank0_result.get("train_mode_exit_ms", 0.0)),
                "train_mode_reused:mean": float(rank0_result.get("train_mode_reused", 0.0)),
                "grad_restore_skipped:mean": float(rank0_result.get("grad_restore_skipped", 0.0)),
                "optim_step_batch_ms:mean": float(rank0_result.get("optim_step_batch_ms", 0.0)),
                "train_mode_enter_total:sum": float(rank0_result.get("train_mode_enter_total", 0.0)),
                "train_mode_reuse_total:sum": float(rank0_result.get("train_mode_reuse_total", 0.0)),
                "train_mode_exit_total:sum": float(rank0_result.get("train_mode_exit_total", 0.0)),
            }
        }

    def get_lora_state_dict(self, use_per_expert_lora: bool = False) -> dict:
        """Get LoRA state dict from all workers (rank 0 returns data, others empty).

        IMPORTANT: Must call ALL workers in parallel because bridge.export_weights()
        may use NCCL collectives internally. Calling only rank 0 would deadlock.

        Args:
            use_per_expert_lora: If True, expand shared MLP LoRA to per-expert format
                for vLLM FusedMoEWithLoRA compatibility.
        """
        logger.info(f"[MegatronWorkerGroup] get_lora_state_dict: ENTRY (use_per_expert_lora={use_per_expert_lora})")
        logger.info(f"[MegatronWorkerGroup] Calling get_lora_state_dict.remote() on all {len(self.workers)} workers...")

        try:
            # Call ALL workers - bridge.export_weights() may use NCCL allgather
            # Rank 0 returns actual data, other ranks return empty dict
            futures = [w.get_lora_state_dict.remote(use_per_expert_lora) for w in self.workers]
            results = ray.get(futures, timeout=300)  # Increased timeout for large MoE models

            # Rank 0's result has the actual data
            result = results[0]
            logger.info(f"[MegatronWorkerGroup] get_lora_state_dict: returning {len(result)} params from rank 0")
            return result
        except ray.exceptions.GetTimeoutError:
            logger.error("[MegatronWorkerGroup] get_lora_state_dict timed out after 300s")
            raise RuntimeError("get_lora_state_dict timed out - workers not responding (possible NCCL deadlock)")
        except ray.exceptions.RayActorError as e:
            logger.error(f"[MegatronWorkerGroup] get_lora_state_dict failed - worker died: {e}")
            raise RuntimeError(f"get_lora_state_dict failed - worker died: {e}")

    def check_determinism_status(self) -> dict:
        """Check determinism settings on all workers.

        Returns:
            dict with determinism status from rank 0
        """
        logger.info("[MegatronWorkerGroup] check_determinism_status: ENTRY")

        try:
            # Call ALL workers to get status
            futures = [w.check_determinism_status.remote() for w in self.workers]
            results = ray.get(futures, timeout=60)

            # Rank 0's result has the status
            result = results[0]
            logger.info(f"[MegatronWorkerGroup] check_determinism_status: {result}")
            return result
        except ray.exceptions.GetTimeoutError:
            logger.error("[MegatronWorkerGroup] check_determinism_status timed out")
            raise RuntimeError("check_determinism_status timed out")
        except ray.exceptions.RayActorError as e:
            logger.error(f"[MegatronWorkerGroup] check_determinism_status failed: {e}")
            raise RuntimeError(f"check_determinism_status failed: {e}")

    def reset_expert_bias(self) -> dict:
        """Reset expert_bias buffers to zero in all MoE router modules across all workers.

        The expert_bias buffer accumulates during training to balance token distribution
        across experts. However, this buffer is NOT exported with LoRA weights, causing
        train-inference mismatch when comparing Megatron logprobs with vLLM.

        Call this before computing logprobs to ensure consistent behavior with vLLM.

        Returns:
            dict with reset count from rank 0
        """
        logger.info("[MegatronWorkerGroup] reset_expert_bias: ENTRY")

        try:
            # Call ALL workers to ensure distributed consistency
            futures = [w.reset_expert_bias.remote() for w in self.workers]
            results = ray.get(futures, timeout=60)

            # Rank 0's result has the count
            result = results[0]
            logger.info(f"[MegatronWorkerGroup] reset_expert_bias: reset {result.get('reset_count', 0)} buffers")
            return result
        except ray.exceptions.GetTimeoutError:
            logger.error("[MegatronWorkerGroup] reset_expert_bias timed out")
            raise RuntimeError("reset_expert_bias timed out")
        except ray.exceptions.RayActorError as e:
            logger.error(f"[MegatronWorkerGroup] reset_expert_bias failed: {e}")
            raise RuntimeError(f"reset_expert_bias failed: {e}")

    def get_lora_config(self) -> dict:
        """Get LoRA configuration as dictionary.

        Returns:
            PEFT config dict compatible with vLLM's PEFTHelper.

        MLP modules are excluded by default for MoE models unless explicitly enabled.
        """
        try:
            model_is_mla = get_model_config(self.base_model).is_mla
        except ValueError:
            model_is_mla = False
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ] if not model_is_mla else [
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
        ]
        target_modules += ["gate_proj", "up_proj", "down_proj"]

        # Use actual session rank (Phase 7) or fall back to max_lora_rank
        effective_rank = self._actual_rank or self.lora_rank
        return {
            "r": effective_rank,
            "lora_alpha": effective_rank * 2,
            "lora_dropout": 0.0,
            "target_modules": target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "base_model_name_or_path": self.base_model,
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

    def reinit_lora_weights(
        self,
        learning_rate: float | None = None,
        actual_rank: int | None = None,
        new_session_id: str | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Reinitialize LoRA weights to fresh random state on all workers.

        This must be called when reusing an actor for a new session to ensure
        fresh weights instead of inheriting trained weights from previous session.

        Issue #44 fix: Automatically saves the current session's weights before
        reinitializing, so they can be restored later.

        Args:
            learning_rate: New learning rate for the session. If provided,
                updates all optimizer param_groups on all workers.
            actual_rank: Actual LoRA rank for the new session (Phase 7).
                If provided, updates _actual_rank for save_checkpoint.
            new_session_id: Session ID for the new session. If provided,
                saves current session's weights before reinit.

        Returns:
            dict with status and total count of reinitialized parameters.
        """
        # Issue #44: Save current session's weights before reinitializing
        if self._current_session is not None and new_session_id is not None:
            old_path = self._session_manager.get_session_path(self._current_session)
            logger.info(f"[MegatronWorkerGroup] reinit_lora_weights: saving current session {self._current_session} to {old_path}")
            try:
                self.save_adapter_state(old_path)
                self._session_manager.save_metadata(
                    self._current_session,
                    step=self._step_count,
                    lr=self.learning_rate,
                    actual_rank=self._actual_rank,
                )
            except Exception as e:
                logger.warning(f"[MegatronWorkerGroup] reinit_lora_weights: failed to save session {self._current_session}: {e}")

        logger.info(
            f"[MegatronWorkerGroup] reinit_lora_weights: reinitializing on all workers "
            f"(lr={learning_rate}, actual_rank={actual_rank}, train_attn={train_attn}, train_mlp={train_mlp}, train_unembed={train_unembed})"
        )

        # Call reinit on all workers (required for distributed sync)
        futures = [
            w.reinit_lora_weights.remote(
                learning_rate,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            for w in self.workers
        ]
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

        # Update actual_rank for save_checkpoint (Phase 7)
        if actual_rank is not None:
            self._actual_rank = actual_rank

        # NOTE: Do NOT set _current_session here!
        # Session switching must go through _ensure_session_loaded() which calls
        # _swap_session_on_workers() to properly save/restore optimizer state and gradients.
        # Setting _current_session here would make _ensure_session_loaded() think the
        # session is already loaded and skip the critical state swap.

        logger.info(f"[MegatronWorkerGroup] reinit_lora_weights: reinitialized {total_reinit} params, reset {total_opt_state_reset} optimizer states, lr_updated={lr_updated}, actual_rank={self._actual_rank}, new_session={new_session_id}")
        return {"status": "ok", "reinit_count": total_reinit, "opt_state_reset": total_opt_state_reset, "lr_updated": lr_updated, "learning_rate": learning_rate, "actual_rank": self._actual_rank}

    def load_checkpoint(self, load_path: str, load_optimizer: bool = True) -> dict:
        """Load checkpoint from path.

        Delegates to load_adapter_state which handles distributed loading.

        Args:
            load_path: Path to checkpoint directory.
            load_optimizer: Whether to restore optimizer state.

        Returns:
            Dict with load metadata.
        """
        import json
        import os

        logger.info(f"[MegatronWorkerGroup] load_checkpoint: path={load_path}, load_optimizer={load_optimizer}")

        # Check what files exist in checkpoint directory
        if os.path.isdir(load_path):
            files = os.listdir(load_path)
            logger.info(f"[MegatronWorkerGroup] load_checkpoint: found {len(files)} files: {files[:10]}")

            # Look for adapter checkpoint files (mp_rank_*_adapter.pt pattern)
            adapter_files = [f for f in files if f.endswith("_adapter.pt")]
            if adapter_files:
                logger.info(f"[MegatronWorkerGroup] load_checkpoint: found {len(adapter_files)} adapter files")
                # Delegate to load_adapter_state
                result = self.load_adapter_state(load_path, actual_rank=self._actual_rank or self.lora_rank)
                result["load_method"] = "load_adapter_state"

                if load_optimizer:
                    opt_results = ray.get(
                        [w.load_optimizer_state.remote(load_path) for w in self.workers]
                    )
                    optimizer_restored = any(
                        isinstance(r, dict) and r.get("status") == "ok" for r in opt_results
                    )
                    if not optimizer_restored:
                        raise RuntimeError(
                            "Optimizer restore requested, but no rank reported optimizer restored"
                        )
                    result["optimizer_restored"] = True
                else:
                    result["optimizer_restored"] = False

                meta_path = os.path.join(load_path, "training_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    result.update(meta)
                    self._step_count = int(meta.get("current_step", self._step_count) or 0)
                    self.learning_rate = float(meta.get("learning_rate", self.learning_rate) or self.learning_rate)

                return result
            else:
                raise FileNotFoundError(
                    f"Missing distributed adapter shards (mp_rank_*_adapter.pt) in: {load_path}"
                )
        else:
            raise FileNotFoundError(
                f"Checkpoint path does not exist or is not a directory: {load_path}"
            )


    def save_checkpoint(
        self,
        save_path: str,
        use_per_expert_lora: bool = False,
        *,
        session_id: str | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Save checkpoint using all workers (rank 0 saves, others participate in NCCL).

        IMPORTANT: Must call ALL workers because MegatronRankWorker.save_checkpoint
        calls get_lora_state_dict() which uses NCCL collectives internally.
        Calling only rank 0 would deadlock waiting for other ranks.

        Args:
            save_path: Directory path to save checkpoint files.
            use_per_expert_lora: If True, expand shared MLP LoRA to per-expert format for MoE.

        Returns:
            Dict with training metadata (from rank 0).
        """
        effective_session_id = session_id or self._current_session
        if effective_session_id is not None:
            self._ensure_session_loaded(
                effective_session_id,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )

        logger.info(
            f"[MegatronWorkerGroup] save_checkpoint: {save_path} "
            f"(session_id={effective_session_id}, actual_rank={self._actual_rank}, use_per_expert_lora={use_per_expert_lora})"
        )
        # Call ALL workers - get_lora_state_dict uses NCCL allgather
        # Rank 0 saves to disk, other ranks participate in collectives then return empty
        futures = [w.save_checkpoint.remote(save_path, self._step_count, self._actual_rank, use_per_expert_lora) for w in self.workers]
        world_size = len(self.workers)
        if world_size >= 32:
            default_timeout_s = 3600
        elif world_size >= 16:
            default_timeout_s = 1800
        elif world_size >= 4:
            default_timeout_s = 600
        else:
            default_timeout_s = 300
        timeout_s = int(os.environ.get("MINT_MEGATRON_SAVE_CHECKPOINT_TIMEOUT_S", str(default_timeout_s)))
        results = ray.get(futures, timeout=timeout_s)
        result = results[0]  # Only rank 0 returns actual data
        logger.info(f"[MegatronWorkerGroup] save_checkpoint: completed, step={result.get('current_step', 'unknown')}")
        return result

    def save_lora_weights(
        self,
        save_path: str,
        use_per_expert_lora: bool = False,
        *,
        session_id: str | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Save PEFT LoRA weights for sampling (no optimizer/resume artifacts).

        Must call ALL workers because get_lora_state_dict uses NCCL collectives.
        """
        effective_session_id = session_id or self._current_session
        if effective_session_id is not None:
            self._ensure_session_loaded(
                effective_session_id,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )

        logger.info(
            f"[MegatronWorkerGroup] save_lora_weights: {save_path} "
            f"(session_id={effective_session_id}, actual_rank={self._actual_rank}, use_per_expert_lora={use_per_expert_lora})"
        )
        futures = [
            w.save_lora_weights.remote(save_path, self._step_count, self._actual_rank, use_per_expert_lora)
            for w in self.workers
        ]
        world_size = len(self.workers)
        if world_size >= 32:
            default_timeout_s = 3600
        elif world_size >= 16:
            default_timeout_s = 1800
        elif world_size >= 4:
            default_timeout_s = 600
        else:
            default_timeout_s = 300
        timeout_s = int(os.environ.get("MINT_MEGATRON_SAVE_LORA_TIMEOUT_S", str(default_timeout_s)))
        results = ray.get(futures, timeout=timeout_s)
        result = results[0]
        logger.info(f"[MegatronWorkerGroup] save_lora_weights: completed, step={result.get('current_step', 'unknown')}")
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
        self,
        checkpoint_path: str,
        actual_rank: int | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
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
            f"(actual_rank={actual_rank}, trainer_rank={self.lora_rank}, train_attn={train_attn}, train_mlp={train_mlp}, train_unembed={train_unembed})"
        )
        futures = [
            w.load_adapter_state.remote(
                checkpoint_path,
                actual_rank=actual_rank,
                trainer_rank=self.lora_rank,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
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
                ray_kill.kill(w, reason="megatron_worker_group_shutdown", no_restart=True)
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
    session_id: str | None = None,
) -> ray.actor.ActorHandle:
    """Get existing or create new persistent MegatronWorkerGroup for this model.

    Uses detached Ray actor pattern like vLLM for crash resilience.
    Each base_model gets its own Megatron actor (per-model isolation).

    Args:
        base_model: HuggingFace model path.
        lora_rank: LoRA rank.
        learning_rate: Initial learning rate.
        distributed_config: Parallelism config. Defaults to single-GPU.
        session_id: Session ID for Issue #44 session state management.

    Returns:
        Ray actor handle to MegatronWorkerGroup.
    """
    from tinker_server.backend.resource_pool import get_resource_pool, ActorType
    from .model_registry import is_persistent_model

    config = distributed_config or DistributedConfig()
    num_gpus = config.world_size
    is_persistent = is_persistent_model(base_model)

    if not ray.is_initialized():
        init_ray(
            address="auto",
            namespace=PERSISTENT_NAMESPACE,
            ignore_reinit_error=True,
        )

    resource_pool = get_resource_pool()
    actor_name = _make_megatron_actor_name(base_model)

    create_lock = _get_megatron_create_lock(actor_name)
    with create_lock:
        # Try to get existing persistent actor for this model.
        # Must be inside a per-actor lock to avoid concurrent create_lora_training_client races.
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            logger.info(f"Connected to existing Megatron actor: {actor_name}")

            # Verify actor is alive
            try:
                ray.get(actor.get_diagnostics.remote(), timeout=10)
            except ray.exceptions.RayActorError:
                # Actor is dead, kill to free name
                logger.warning(f"Megatron actor {actor_name} is dead, killing to free name")
                try:
                    ray_kill.kill(
                        actor,
                        reason="megatron_actor_dead_free_name",
                        actor_name=actor_name,
                        namespace=PERSISTENT_NAMESPACE,
                        no_restart=True,
                    )
                except Exception:
                    pass
                raise ValueError("Actor dead, will recreate")
            except ray.exceptions.GetTimeoutError:
                # Actor might be busy (queued tasks) rather than dead.
                # Killing on timeout will terminate active training and corrupt in-flight requests.
                logger.warning(
                    f"Megatron actor {actor_name} get_diagnostics timed out; assuming busy and reusing actor"
                )

            # Register with resource pool (reconnection case)
            resource_pool.register(
                actor_name=actor_name,
                actor_type=ActorType.MEGATRON,
                num_gpus=num_gpus,
                actor_handle=actor,
                namespace=PERSISTENT_NAMESPACE,
                base_model=base_model,
                protected=is_persistent,
            )
            # Existing actor is already ready
            resource_pool.mark_ready(actor_name)
            # NOTE: Do NOT reinit weights here for existing actors.
            # Session swapping + reinit happens inside MegatronWorkerGroup._ensure_session_loaded()
            # to avoid clobbering active sessions during create_model.
            return actor
        except ValueError:
            # Actor doesn't exist, create new one
            logger.info(f"Creating new detached Megatron actor: {actor_name} for {base_model}")

        # Heal a common invariant violation: a detached Megatron placement group can outlive the
        # named actor (e.g., crash during initialization). The orphan PG reserves GPUs, which can
        # make ensure_gpus_available() block forever even though nothing is actually running.
        pg_name = f"{actor_name}_pg"
        try:
            orphan_pg = ray.util.get_placement_group(pg_name)
        except ValueError:
            orphan_pg = None
        except Exception as e:
            raise RuntimeError(f"Failed to probe placement group {pg_name!r}: {e}") from e

        if orphan_pg is not None:
            # Race guard: actor creation is not instantaneous. A concurrent request can
            # observe `ray.get_actor(...)` as missing while the named actor is still
            # being registered. Removing the placement group in that window will
            # SIGTERM the in-flight Megatron workers (Ray reports: "placement group was removed").
            import time

            for _ in range(30):
                try:
                    actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
                    logger.info(
                        f"Megatron actor appeared after PG probe; reusing actor={actor_name} pg={pg_name}"
                    )
                    return actor
                except ValueError:
                    time.sleep(1)

            logger.warning(
                f"Found orphan Megatron placement group without actor; removing to unblock recreate: {pg_name}"
            )
            try:
                ray.util.remove_placement_group(orphan_pg)
            except Exception as e:
                raise RuntimeError(f"Failed to remove orphan placement group {pg_name!r}: {e}") from e

        # Check available GPUs and evict LRU actors if necessary.
        # For large, full-cluster Megatron jobs, allow preempting idle protected actors
        # (e.g., inference "always-on" actors) to avoid deadlocking on a fixed-size cluster.
        allow_evict_protected = os.environ.get("MINT_MEGATRON_EVICT_PROTECTED", "0") == "1"
        resource_pool.ensure_gpus_available(num_gpus, allow_evict_protected=allow_evict_protected)

        # Reserve GPUs to prevent race conditions with concurrent requests
        # This must be done AFTER ensure_gpus_available and BEFORE actor creation
        resource_pool.reserve_gpus(num_gpus)

        try:
            from ..config import otel_env_vars

            # Runtime env for PFS code access
            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": PFS_PYTHONPATH,
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # Reduce memory fragmentation
                    **otel_env_vars(),
                }
            }

            # Forward MoE LoRA export knobs into the detached Megatron actor so the
            # on-GPU export path can be switched without code deploys.
            for k in (
                "MINT_MOE_LORA_SPARSE_EXPERT_EXPORT",
                "MINT_MOE_LORA_SHARED_EXPERT_EXPORT",
            ):
                v = os.environ.get(k)
                if v is not None:
                    runtime_env["env_vars"][k] = v
            # Forward sticky/diagnostic knobs into the detached Megatron actor
            # so group-level watchdog and sticky behavior match server settings.
            for k in (
                "MINT_MEGATRON_STICKY_TRAIN_MODE",
                "MINT_MEGATRON_STICKY_IDLE_TIMEOUT_S",
                "MINT_MEGATRON_STICKY_CLOSE_ON_OPTIM",
                "MINT_MEGATRON_STICKY_TIMING_DIAG",
                "MINT_MEGATRON_STACK_DUMP_TIMEOUT_S",
                "MINT_MEGATRON_STACK_DUMP_LIMIT",
            ):
                v = os.environ.get(k)
                if v is not None:
                    runtime_env["env_vars"][k] = v
            explicit_node_ips_csv = os.environ.get("MINT_MEGATRON_NODE_IPS_CSV", "").strip()
            if explicit_node_ips_csv:
                runtime_env["env_vars"]["MINT_MEGATRON_NODE_IPS_CSV"] = explicit_node_ips_csv
            else:
                volc_rq = os.environ.get("MINT_MEGATRON_VOLC_RESOURCE_QUEUE_ID", "").strip()
                if volc_rq:
                    from .volc_placement import list_node_ips_for_resource_queue

                    node_ips = list_node_ips_for_resource_queue(resource_queue_id=volc_rq)
                    if not node_ips:
                        raise RuntimeError(
                            f"no Ray GPU nodes found for MINT_MEGATRON_VOLC_RESOURCE_QUEUE_ID={volc_rq}"
                        )
                    runtime_env["env_vars"]["MINT_MEGATRON_NODE_IPS_CSV"] = ",".join(node_ips)

            timing = os.environ.get("MINT_TIMING_DIAG")
            if timing is not None:
                runtime_env["env_vars"]["MINT_TIMING_DIAG"] = timing
            node_pin_json = os.environ.get("MINT_MODEL_NODE_IPS_JSON")
            if node_pin_json:
                runtime_env["env_vars"]["MINT_MODEL_NODE_IPS_JSON"] = node_pin_json

            # Create detached Ray actor with per-model name
            try:
                actor = MegatronWorkerGroup.options(
                    name=actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    lifetime="detached",
                    runtime_env=runtime_env,
                ).remote(
                    base_model=base_model,
                    lora_rank=lora_rank,
                    learning_rate=learning_rate,
                    distributed_config=config,
                )
            except Exception as e:
                msg = str(e)
                if actor_name in msg and "already exists" in msg:
                    logger.warning(
                        f"Megatron actor create raced (already exists): {actor_name}; reusing existing actor"
                    )
                    actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
                else:
                    raise

            # Register immediately (creating=True) to account for GPU usage and prevent eviction.
            # Actor readiness is awaited in VerlTrainingEngine.create_training_session, which also
            # marks the entry ready (creating=False) after __ray_ready__ completes.
            resource_pool.register(
                actor_name=actor_name,
                actor_type=ActorType.MEGATRON,
                num_gpus=num_gpus,
                actor_handle=actor,
                namespace=PERSISTENT_NAMESPACE,
                base_model=base_model,
                session_id=session_id,
                protected=is_persistent,
            )
            return actor
        finally:
            # Release pending GPU reservation (GPUs now tracked by registered actor or freed on failure)
            resource_pool.release_pending_gpus(num_gpus)


async def async_get_or_create_megatron_worker_group(
    base_model: str,
    lora_rank: int,
    learning_rate: float,
    distributed_config: DistributedConfig | None = None,
    session_id: str | None = None,
) -> ray.actor.ActorHandle:
    """Async version of get_or_create_megatron_worker_group.

    Wraps blocking Ray operations in asyncio.to_thread() to avoid blocking
    the uvicorn event loop during FastAPI request handling.

    Args:
        base_model: HuggingFace model path.
        lora_rank: LoRA rank.
        learning_rate: Initial learning rate.
        distributed_config: Parallelism config. Defaults to single-GPU.
        session_id: Session ID for Issue #44 session state management.

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
        session_id,
    )


def kill_megatron_actor(base_model: str | None = None) -> bool:
    """Kill persistent Megatron actor(s) and release resources.

    Args:
        base_model: If provided, kill actor for this specific model.
                   If None, kill ALL Megatron actors.

    Returns:
        True if any actor was killed, False if none found.
    """
    from tinker_server.backend.resource_pool import get_resource_pool, ActorType

    if not ray.is_initialized():
        init_ray(
            address="auto",
            namespace=PERSISTENT_NAMESPACE,
            ignore_reinit_error=True,
        )

    resource_pool = get_resource_pool()
    killed_any = False

    def _remove_detached_pg(actor_name: str) -> None:
        pg_name = f"{actor_name}_pg"
        try:
            pg = ray.util.get_placement_group(pg_name)
        except ValueError:
            return
        except Exception as e:
            logger.warning(f"Failed to get placement group {pg_name!r}: {e}")
            return
        try:
            ray.util.remove_placement_group(pg)
            logger.info(f"Removed Megatron placement group: {pg_name}")
        except Exception as e:
            logger.warning(f"Failed to remove placement group {pg_name!r}: {e}")

    if base_model:
        # Kill specific model's actor
        actor_name = _make_megatron_actor_name(base_model)
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            try:
                ray.get(actor.shutdown.remote(), timeout=10)
            except Exception:
                pass
            ray_kill.kill(
                actor,
                reason="kill_megatron_actor",
                actor_name=actor_name,
                namespace=PERSISTENT_NAMESPACE,
                no_restart=True,
                base_model=base_model,
            )
            logger.info(f"Killed Megatron actor: {actor_name}")
            resource_pool.unregister(actor_name)
            _remove_detached_pg(actor_name)
            killed_any = True
        except ValueError:
            logger.info(f"No Megatron actor to kill for {base_model}")
            resource_pool.unregister(actor_name)
            _remove_detached_pg(actor_name)
    else:
        # Kill ALL Megatron actors from resource pool
        for entry in resource_pool.iter_entries():
            if entry.actor_type == ActorType.MEGATRON:
                try:
                    actor = ray.get_actor(entry.actor_name, namespace=PERSISTENT_NAMESPACE)
                except ValueError:
                    resource_pool.unregister(entry.actor_name)
                    _remove_detached_pg(entry.actor_name)
                    continue

                try:
                    ray.get(actor.shutdown.remote(), timeout=10)
                except Exception:
                    pass

                ray_kill.kill(
                    actor,
                    reason="kill_megatron_actor",
                    actor_name=entry.actor_name,
                    namespace=PERSISTENT_NAMESPACE,
                    no_restart=True,
                    base_model=entry.base_model,
                )
                logger.info(f"Killed Megatron actor: {entry.actor_name}")
                resource_pool.unregister(entry.actor_name)
                _remove_detached_pg(entry.actor_name)
                killed_any = True

    return killed_any


def is_megatron_actor_running(base_model: str | None = None) -> bool:
    """Check if persistent Megatron actor is running.

    Args:
        base_model: If provided, check for this specific model's actor.
                   If None, check for ANY running Megatron actor.

    Returns:
        True if actor exists and is actually alive (not dead).
    """
    if not ray.is_initialized():
        init_ray(
            address="auto",
            namespace=PERSISTENT_NAMESPACE,
            ignore_reinit_error=True,
        )

    if base_model:
        actor_name = _make_megatron_actor_name(base_model)
        try:
            actor = ray.get_actor(actor_name, namespace=PERSISTENT_NAMESPACE)
            ray.get(actor.get_diagnostics.remote(), timeout=5)
            return True
        except ray.exceptions.GetTimeoutError:
            return False
        except (ValueError, ray.exceptions.RayActorError, Exception):
            return False
    else:
        # Check for any Megatron actor from resource pool
        from tinker_server.backend.resource_pool import get_resource_pool, ActorType
        resource_pool = get_resource_pool()
        for entry in resource_pool.iter_entries():
            if entry.actor_type == ActorType.MEGATRON:
                try:
                    ray.get(entry.actor_handle.get_diagnostics.remote(), timeout=5)
                    return True
                except ray.exceptions.GetTimeoutError:
                    continue
                except Exception:
                    pass
        return False
