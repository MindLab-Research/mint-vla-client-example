"""Distributed Megatron training backend using Ray placement groups.

Defines MegatronWorkerGroup (detached Ray actor) and MegatronRankWorker (per-rank workers)
used by VerlTrainingEngine for MoE model training.

Shared loss functions and Tinker Datum conversion utilities live in megatron_training.py.
"""

from __future__ import annotations  # Allow forward references in type hints

import copy
import os
import json
import math
import hashlib
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
from ..logging_context import (
    get_request_id,
    init_actor_observability,
    restore_trace_id_from_traceparent,
)

logger = logging.getLogger(__name__)

# Import centralized PFS paths from config
from tinker_server.config import PFS_PYTHONPATH, PFS_TINKER_PATH, RAY_NAMESPACE, config as server_config
from tinker_server.backend.model_registry import get_model_config
from tinker_server.ray_utils import init_ray
from tinker_server.model_input_utils import flatten_encoded_text_chunks
from tinker_server.backend.volc_placement import assert_node_ip_capacity, parse_model_node_ip_list
from tinker_server.backend.ray_placement_groups import PlacementGroupMismatchError, get_named_placement_group

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


def _default_megatron_sessions_base_path() -> str:
    return os.environ.get("MINT_MEGATRON_SESSIONS_BASE_PATH") or os.path.join(
        PFS_TINKER_PATH,
        "checkpoints",
        "megatron_sessions",
    )


def _actor_only_snapshot_dir(session_path: str) -> str:
    return os.path.join(session_path, "actor_only_state")


def _actor_only_snapshot_manifest_path(session_path: str) -> str:
    return os.path.join(session_path, "actor_only_state_manifest.json")


def _actor_only_rank_snapshot_path(session_path: str, rank: int) -> str:
    return os.path.join(_actor_only_snapshot_dir(session_path), f"rank_{rank:04d}.pt")


def _estimate_object_bytes(value: object) -> int:
    import sys

    try:
        import torch
    except Exception:
        torch = None

    if value is _GRADIENTS_CONSUMED or value is None:
        return 0
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_estimate_object_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_estimate_object_bytes(v) for v in value)
    if isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return int(sys.getsizeof(value))


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
    model_key = _model_key_from_base_model(base_model)
    lookup_keys = [model_key, model_key.lower(), base_model, base_model.lower()]
    node_ips = parse_model_node_ip_list(
        raw_json=os.environ.get("MINT_MEGATRON_MODEL_NODE_IPS_JSON"),
        lookup_keys=lookup_keys,
        env_var_name="MINT_MEGATRON_MODEL_NODE_IPS_JSON",
        context=f"[MegatronWorkerGroup] node pinning model={model_key}",
    )
    if not node_ips:
        node_ips = parse_model_node_ip_list(
        raw_json=os.environ.get("MINT_MODEL_NODE_IPS_JSON"),
        lookup_keys=[model_key, model_key.lower(), base_model, base_model.lower()],
        env_var_name="MINT_MODEL_NODE_IPS_JSON",
        context=f"[MegatronWorkerGroup] node pinning model={model_key}",
        )
    if not node_ips:
        return []
    logger.info(f"[MegatronWorkerGroup] node pinning for model={model_key}: {node_ips}")
    return node_ips


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


def _bundle_node_ip(bundle: dict[str, float | int]) -> str | None:
    for key, value in bundle.items():
        if isinstance(key, str) and key.startswith("node:") and float(value or 0) > 0:
            return key.split("node:", 1)[1]
    return None


def _node_affinity_resources(node_ip: str | None) -> dict[str, float]:
    if not node_ip:
        return {}
    return {f"node:{node_ip}": 0.001}


def _make_namespace_pg_suffix(namespace: str) -> str:
    raw = str(namespace).strip().lower()
    if not raw:
        return "default"
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    if len(sanitized) <= 24:
        return sanitized or "default"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[:15]}_{digest}"


def _make_megatron_pg_name_from_actor_name(
    actor_name: str,
    *,
    namespace: str = PERSISTENT_NAMESPACE,
) -> str:
    return f"{actor_name}_{_make_namespace_pg_suffix(namespace)}_pg"


def _make_megatron_pg_name(base_model: str, *, namespace: str = PERSISTENT_NAMESPACE) -> str:
    actor_name = _make_megatron_actor_name(base_model)
    return _make_megatron_pg_name_from_actor_name(actor_name, namespace=namespace)


def _get_or_create_megatron_placement_group(*, pg_name: str, bundles: list[dict[str, float | int]]):
    try:
        return get_named_placement_group(
            pg_name,
            namespace=PERSISTENT_NAMESPACE,
            expected_bundles=bundles,
        )
    except PlacementGroupMismatchError as e:
        logger.warning(
            "[MegatronWorkerGroup] Removing incompatible placement group %s: %s",
            pg_name,
            e,
        )
        ray.util.remove_placement_group(e.pg)
    except Exception:
        pass

    return ray.util.placement_group(
        bundles,
        strategy="PACK",
        name=pg_name,
        lifetime="detached",
    )


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


@dataclass
class _HotSessionCacheEntry:
    bytes: int
    last_accessed_s: float
    durable: bool


@dataclass
class _MegatronSessionCacheEntry:
    session_id: str
    session_path: str
    total_bytes: int
    updated_at: float
    age_s: float
    actor_name: str | None
    cold_safe: bool
    skip_reason: str | None


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
        self._session_lr_scheduler_states: dict[str, dict] = {}  # Per-session scheduler state (CPU)
        self._session_hot_cache: dict[str, _HotSessionCacheEntry] = {}
        max_hot_bytes_per_actor = max(0, _env_int("MINT_MEGATRON_MAX_HOT_BYTES_PER_ACTOR", default=0))
        rss_watermark_per_actor = max(
            0,
            _env_int("MINT_MEGATRON_HOT_CACHE_RSS_WATERMARK_BYTES", default=0),
        )
        self._max_hot_sessions = max(0, _env_int("MINT_MEGATRON_MAX_HOT_SESSIONS_PER_ACTOR", default=0))
        self._max_hot_bytes = (
            math.ceil(max_hot_bytes_per_actor / max(1, self.world_size))
            if max_hot_bytes_per_actor > 0
            else 0
        )
        self._rss_watermark_bytes = (
            math.ceil(rss_watermark_per_actor / max(1, self.world_size))
            if rss_watermark_per_actor > 0
            else 0
        )
        self._last_eviction_reason: str | None = None
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
                    # Let _capture_gradients() failures propagate -- the ctx
                    # stays open and bookkeeping is untouched, so the caller
                    # can retry.  Swallowing would silently lose gradients.
                    captured = self._capture_gradients()
                    self._session_gradients[active_session] = captured
                    snap_count = len(captured)

            # -- ctx.__exit__: triggers GPU->CPU model parameter offload --
            # Use try/finally to ensure bookkeeping cleanup happens even if __exit__ fails.
            # This prevents stale ctx handles from being reused after partial failures.
            exit_error = None
            t0 = time.perf_counter()
            try:
                self._sticky_train_mode_ctx.__exit__(None, None, None)
                exit_s = time.perf_counter() - t0
                released = True
            except Exception as e:
                exit_s = time.perf_counter() - t0
                exit_error = e
                logger.error(
                    f"[Rank {self.rank}] sticky_train_mode __exit__ failed "
                    f"(session={active_session}, reason={reason}): {e}",
                    exc_info=True,
                )
            finally:
                # -- Reset all sticky bookkeeping (fail-closed) --
                # CRITICAL: Clear state even if __exit__ failed, to prevent reuse of broken ctx.
                self._sticky_train_mode_ctx = None
                self._sticky_train_mode_session_id = None
                self._sticky_train_mode_last_used_s = 0.0
                self._sticky_train_mode_exit_total += 1

            if self._sticky_train_mode_diag:
                logger.info(
                    f"[Rank {self.rank}] sticky_train_mode release: reason={reason} "
                    f"exit_ms={exit_s * 1000.0:.2f} snapshot_count={snap_count} "
                    f"exit_error={exit_error is not None}"
                )

            # Re-raise __exit__ error after cleanup
            if exit_error is not None:
                raise exit_error

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

    def _release_sticky_for_aux_mode_transition(
        self,
        *,
        reason: str,
        snapshot_gradients: bool = True,
    ) -> None:
        """Release sticky train_mode before entering an auxiliary mode context.

        Auxiliary ops such as export/save/load paths may enter their own
        `eval_mode()` / `train_mode()` contexts. Releasing here avoids nested
        mode transitions while preserving accumulated gradients when possible.
        """
        if self._sticky_train_mode_ctx is None:
            return
        self._release_sticky_train_mode(
            reason=reason,
            snapshot_gradients=snapshot_gradients,
        )

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

    def get_observability_binding(self) -> dict[str, object]:
        import socket

        gpu_indices: list[int] = []
        try:
            for gpu_id in ray.get_gpu_ids():
                if isinstance(gpu_id, (int, float)):
                    gpu_indices.append(int(gpu_id))
                else:
                    gpu_indices.append(int(float(str(gpu_id))))
        except Exception:
            gpu_indices = []
        node_id = None
        try:
            node_id = str(ray.get_runtime_context().get_node_id())
        except Exception:
            node_id = None
        mem: dict[str, int] = {}
        try:
            torch = _get_torch()
            if torch.cuda.is_available():
                allocated = int(torch.cuda.memory_allocated())
                reserved = int(torch.cuda.memory_reserved())
                mem = {
                    "gpu_memory_allocated_bytes": allocated,
                    "gpu_memory_reserved_bytes": reserved,
                    "gpu_memory_fragmentation_bytes": max(0, reserved - allocated),
                }
        except Exception:
            mem = {}
        return {
            "hostname": socket.gethostname(),
            "node_id": node_id,
            "gpu_indices": gpu_indices,
            "rank": int(self.rank),
            **mem,
        }

    def _bind_traceparent(self, traceparent: str | None) -> None:
        if isinstance(traceparent, str) and traceparent:
            restore_trace_id_from_traceparent(traceparent)

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

    def _clear_optimizer_gradients(self, *, session_id: str | None, reason: str) -> None:
        """Clear gradient buffers after optimizer_step in a backend-safe way.

        Fail-loud on unrecoverable clear failures to avoid silent gradient leakage.
        """
        errors: list[str] = []

        if hasattr(self.engine, "optimizer_zero_grad"):
            try:
                self.engine.optimizer_zero_grad()
                logger.debug(
                    f"[Rank {self.rank}] Cleared gradients via optimizer_zero_grad "
                    f"(session={session_id}, reason={reason})"
                )
                return
            except Exception as e:
                errors.append(f"optimizer_zero_grad failed: {type(e).__name__}: {e}")
                logger.warning(
                    f"[Rank {self.rank}] optimizer_zero_grad failed, trying fallback "
                    f"(session={session_id}, reason={reason}): {type(e).__name__}: {e}"
                )

        optimizer = getattr(self.engine, "optimizer", None)
        if optimizer is not None:
            try:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    optimizer.zero_grad()
                logger.debug(
                    f"[Rank {self.rank}] Cleared gradients via optimizer.zero_grad "
                    f"(session={session_id}, reason={reason})"
                )
                return
            except Exception as e:
                errors.append(f"optimizer.zero_grad failed: {type(e).__name__}: {e}")

        if optimizer is None:
            errors.append("optimizer unavailable")

        detail = "; ".join(errors) if errors else "unknown clear failure"
        raise RuntimeError(
            f"[Rank {self.rank}] Failed to clear gradients after optim_step "
            f"(session={session_id}, reason={reason}): {detail}"
        )

    def _capture_optimizer_state(self) -> dict:
        """Capture optimizer state (momentum, variance) to CPU.

        Returns a dict containing optimizer state.
        """
        import torch
        from megatron.core.optimizer import ChainedOptimizer

        def clone_to_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: clone_to_cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clone_to_cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(clone_to_cpu(v) for v in value)
            return copy.deepcopy(value)

        state_dict = {}
        optimizer = self.engine.optimizer

        if optimizer is None:
            return state_dict

        def iter_optimizers(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        for i, _opt in enumerate(iter_optimizers(optimizer)):
            entry = {}
            if hasattr(_opt, "state_dict"):
                try:
                    entry["wrapper_state_dict"] = clone_to_cpu(_opt.state_dict())
                except Exception as e:
                    logger.warning(
                        "[Rank %s] Failed to capture wrapper optimizer state for opt[%s]: %s: %s",
                        self.rank,
                        i,
                        type(e).__name__,
                        e,
                    )

            inner_opt = getattr(_opt, "optimizer", None)
            if inner_opt is not None and hasattr(inner_opt, "state_dict"):
                try:
                    entry["inner_state_dict"] = clone_to_cpu(inner_opt.state_dict())
                except Exception as e:
                    logger.warning(
                        "[Rank %s] Failed to capture inner optimizer state for opt[%s]: %s: %s",
                        self.rank,
                        i,
                        type(e).__name__,
                        e,
                    )

            if entry:
                state_dict[f"optimizer_{i}"] = entry

        logger.debug(f"[Rank {self.rank}] Captured optimizer state for {len(state_dict)} optimizers")
        return state_dict

    def _capture_lr_scheduler_state(self) -> dict:
        """Capture lr scheduler state to a CPU-serializable dict.

        Unlike optimizer state, most scheduler state is already non-tensor metadata.
        We still normalize tensors onto CPU for safety.
        """
        import torch

        lr_scheduler = getattr(self.engine, "lr_scheduler", None)
        if lr_scheduler is None or not hasattr(lr_scheduler, "state_dict"):
            return {}
        try:
            state = lr_scheduler.state_dict()
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to capture lr_scheduler state: {e}")
            return {}

        def _to_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: _to_cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_cpu(v) for v in value)
            return value

        normalized = _to_cpu(state if isinstance(state, dict) else {})
        logger.debug(f"[Rank {self.rank}] Captured lr_scheduler state")
        return normalized if isinstance(normalized, dict) else {}

    def _restore_optimizer_state(self, state_dict: dict) -> None:
        """Restore optimizer state from CPU.

        CRITICAL: Always clears existing state first to prevent contamination
        between sessions. Even if state_dict is empty, we must clear the existing
        optimizer state so the session starts fresh.

        Args:
            state_dict: Dict from _capture_optimizer_state. May be empty for new sessions.
        """
        from megatron.core.optimizer import ChainedOptimizer

        optimizer = self.engine.optimizer
        if optimizer is None:
            return

        def iter_optimizers(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        for i, _opt in enumerate(iter_optimizers(optimizer)):
            entry = state_dict.get(f"optimizer_{i}", {}) if isinstance(state_dict, dict) else {}
            inner_opt = getattr(_opt, "optimizer", None)
            if inner_opt is not None:
                state = getattr(inner_opt, "state", None)
                if hasattr(state, "_inner_dicts"):
                    for inner_dict in state._inner_dicts:
                        inner_dict.clear()
                elif hasattr(state, "clear"):
                    state.clear()
                elif hasattr(state, "keys"):
                    for key in list(state.keys()):
                        del state[key]

            wrapper_state = entry.get("wrapper_state_dict") if isinstance(entry, dict) else None
            if isinstance(wrapper_state, dict) and hasattr(_opt, "load_state_dict"):
                _opt.load_state_dict(copy.deepcopy(wrapper_state))

            inner_state = entry.get("inner_state_dict") if isinstance(entry, dict) else None
            if isinstance(inner_state, dict) and inner_opt is not None and hasattr(inner_opt, "load_state_dict"):
                inner_opt.load_state_dict(copy.deepcopy(inner_state))

        logger.debug(f"[Rank {self.rank}] Restored optimizer state (cleared first)")

    def _restore_lr_scheduler_state(self, state_dict: dict) -> None:
        """Restore lr scheduler state for a session.

        Best-effort: if no scheduler exists, rebuild one when possible. If no state is
        provided, keep the freshly reset/rebuilt scheduler.
        """
        lr_scheduler = getattr(self.engine, "lr_scheduler", None)
        if lr_scheduler is None and hasattr(self.engine, "_build_lr_scheduler"):
            try:
                self.engine.lr_scheduler = self.engine._build_lr_scheduler()
                lr_scheduler = self.engine.lr_scheduler
                for attr_name in ("checkpoint_mananager", "checkpoint_manager"):
                    manager = getattr(self.engine, attr_name, None)
                    if manager is not None and hasattr(manager, "optimizer_scheduler"):
                        manager.optimizer_scheduler = lr_scheduler
            except Exception as e:
                logger.warning(f"[Rank {self.rank}] Failed to rebuild lr_scheduler for restore: {e}")
                return

        if lr_scheduler is None or not hasattr(lr_scheduler, "load_state_dict"):
            return
        if not state_dict:
            return
        try:
            lr_scheduler.load_state_dict(state_dict)
            logger.debug(f"[Rank {self.rank}] Restored lr_scheduler state")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to restore lr_scheduler state: {e}")

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

    def _capture_lr_scheduler_state(self) -> dict:
        """Capture lr_scheduler state to CPU-friendly Python objects."""
        lr_scheduler = getattr(self.engine, "lr_scheduler", None)
        if lr_scheduler is None or not hasattr(lr_scheduler, "state_dict"):
            return {}

        try:
            return copy.deepcopy(lr_scheduler.state_dict())
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to capture lr_scheduler state: {e}")
            return {}

    def _restore_lr_scheduler_state(self, state_dict: dict) -> None:
        """Restore lr_scheduler state for an existing session."""
        lr_scheduler = getattr(self.engine, "lr_scheduler", None)
        if lr_scheduler is None or not hasattr(lr_scheduler, "load_state_dict"):
            return

        try:
            lr_scheduler.load_state_dict(copy.deepcopy(state_dict))
            logger.debug(f"[Rank {self.rank}] Restored lr_scheduler state")
        except Exception as e:
            logger.warning(f"[Rank {self.rank}] Failed to restore lr_scheduler state: {e}")

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

    def _session_path(self, session_id: str) -> str:
        return os.path.join(_default_megatron_sessions_base_path(), f"{session_id}_checkpoint")

    def _actor_only_rank_snapshot_path(self, session_id: str) -> str:
        return _actor_only_rank_snapshot_path(self._session_path(session_id), self.rank)

    def _serialize_gradients_for_snapshot(self, gradients: list[object] | object) -> dict:
        if gradients is _GRADIENTS_CONSUMED:
            return {"kind": "consumed"}
        return {
            "kind": "buffers",
            "buffers": list(gradients) if isinstance(gradients, list) else [],
        }

    def _deserialize_gradients_from_snapshot(self, payload: object) -> list[object] | object:
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"[Rank {self.rank}] Invalid persisted gradients payload type {type(payload).__name__}"
            )
        kind = payload.get("kind")
        if kind == "consumed":
            return _GRADIENTS_CONSUMED
        if kind == "buffers":
            buffers = payload.get("buffers", [])
            if not isinstance(buffers, list):
                raise RuntimeError(
                    f"[Rank {self.rank}] Invalid persisted gradient buffer list type {type(buffers).__name__}"
                )
            return buffers
        raise RuntimeError(f"[Rank {self.rank}] Unsupported persisted gradients kind {kind!r}")

    def _session_hot_bytes(self, session_id: str) -> int:
        return (
            _estimate_object_bytes(self._session_gradients.get(session_id))
            + _estimate_object_bytes(self._session_optimizer_states.get(session_id))
            + _estimate_object_bytes(self._session_lr_scheduler_states.get(session_id))
        )

    def _touch_hot_session(self, session_id: str) -> None:
        entry = self._session_hot_cache.get(session_id)
        if entry is None:
            return
        entry.last_accessed_s = time.time()

    def _drop_hot_session(self, session_id: str) -> None:
        self._session_gradients.pop(session_id, None)
        self._session_optimizer_states.pop(session_id, None)
        self._session_lr_scheduler_states.pop(session_id, None)
        self._session_hot_cache.pop(session_id, None)

    def _current_rss_bytes(self) -> int:
        try:
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().strip().split()
        except OSError:
            return 0
        if len(parts) < 2:
            return 0
        return int(parts[1]) * int(os.sysconf("SC_PAGE_SIZE"))

    def _evict_hot_sessions_if_needed(self) -> None:
        def needs_eviction() -> tuple[bool, str | None]:
            hot_count = len(self._session_hot_cache)
            hot_bytes = sum(entry.bytes for entry in self._session_hot_cache.values())
            if self._max_hot_sessions > 0 and hot_count > self._max_hot_sessions:
                return True, f"max_hot_sessions>{self._max_hot_sessions}"
            if self._max_hot_bytes > 0 and hot_bytes > self._max_hot_bytes:
                return True, f"max_hot_bytes>{self._max_hot_bytes}"
            if self._rss_watermark_bytes > 0 and self._current_rss_bytes() > self._rss_watermark_bytes:
                return True, f"rss>{self._rss_watermark_bytes}"
            return False, None

        while True:
            should_evict, reason = needs_eviction()
            if not should_evict:
                return
            candidates = [
                (session_id, entry)
                for session_id, entry in self._session_hot_cache.items()
                if session_id != self._current_session_id and entry.durable
            ]
            if not candidates:
                logger.warning(
                    "[Rank %s] Hot session cache exceeded budget but no durable session could be evicted",
                    self.rank,
                )
                return
            session_id, entry = min(candidates, key=lambda item: item[1].last_accessed_s)
            self._drop_hot_session(session_id)
            self._last_eviction_reason = reason
            logger.info(
                "[Rank %s] Evicted hot session %s from RAM (bytes=%s, reason=%s)",
                self.rank,
                session_id,
                entry.bytes,
                reason,
            )

    def _persist_actor_only_state(
        self,
        session_id: str,
        gradients: list[object] | object,
        optimizer_state: dict,
        lr_scheduler_state: dict,
    ) -> dict:
        import torch

        session_path = self._session_path(session_id)
        rank_path = self._actor_only_rank_snapshot_path(session_id)
        os.makedirs(os.path.dirname(rank_path), exist_ok=True)
        payload = {
            "version": 1,
            "session_id": session_id,
            "rank": self.rank,
            "saved_at": time.time(),
            "gradients": self._serialize_gradients_for_snapshot(gradients),
            "optimizer_state": optimizer_state,
            "lr_scheduler_state": lr_scheduler_state,
        }
        tmp_path = f"{rank_path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, rank_path)
        return {
            "rank": self.rank,
            "path": rank_path,
            "bytes": int(os.path.getsize(rank_path)),
            "session_path": session_path,
        }

    def _load_persisted_actor_only_state(
        self,
        session_id: str,
        *,
        require: bool,
    ) -> tuple[list[object] | object, dict, dict] | None:
        import torch

        rank_path = self._actor_only_rank_snapshot_path(session_id)
        if not os.path.exists(rank_path):
            if require:
                raise RuntimeError(
                    f"[Rank {self.rank}] Missing persisted actor-only state for session {session_id} at {rank_path}"
                )
            return None
        try:
            payload = torch.load(rank_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"[Rank {self.rank}] Failed to load persisted actor-only state for session {session_id}: "
                f"{type(e).__name__}: {e}"
            ) from e
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"[Rank {self.rank}] Invalid persisted actor-only payload type {type(payload).__name__}"
            )
        gradients = self._deserialize_gradients_from_snapshot(payload.get("gradients", {}))
        optimizer_state = payload.get("optimizer_state", {})
        lr_scheduler_state = payload.get("lr_scheduler_state", {})
        if not isinstance(optimizer_state, dict) or not isinstance(lr_scheduler_state, dict):
            raise RuntimeError(
                f"[Rank {self.rank}] Invalid persisted actor-only optimizer/scheduler payload for session {session_id}"
            )
        return gradients, optimizer_state, lr_scheduler_state

    def _cache_hot_session_state(
        self,
        session_id: str,
        gradients: list[object] | object,
        optimizer_state: dict,
        lr_scheduler_state: dict,
        *,
        durable: bool,
    ) -> None:
        self._session_gradients[session_id] = gradients
        self._session_optimizer_states[session_id] = optimizer_state
        self._session_lr_scheduler_states[session_id] = lr_scheduler_state
        self._session_hot_cache[session_id] = _HotSessionCacheEntry(
            bytes=self._session_hot_bytes(session_id),
            last_accessed_s=time.time(),
            durable=durable,
        )
        self._evict_hot_sessions_if_needed()

    def get_hot_cache_info(self) -> dict:
        return {
            "hot_sessions": sorted(self._session_hot_cache.keys()),
            "hot_session_count": len(self._session_hot_cache),
            "hot_bytes": sum(entry.bytes for entry in self._session_hot_cache.values()),
            "last_eviction_reason": self._last_eviction_reason,
        }

    def swap_session_state(self, new_session_id: str, require_persisted: bool = False) -> dict:
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
            return {"status": "noop", "session_id": new_session_id}

        if self._sticky_train_mode_ctx is not None:
            # Avoid nested train_mode contexts during session switch routines.
            self._release_sticky_train_mode(reason="swap_session_state", snapshot_gradients=True)

        outgoing_persisted = None
        incoming_source = "new"

        # Must use train_mode to access GPU gradient buffers (required for param_offload)
        with self.engine.train_mode():
            if self._current_session_id is not None:
                cached = self._session_gradients.get(self._current_session_id)
                if self._current_session_id not in self._session_gradients:
                    grads = self._capture_gradients()
                elif cached is not None:
                    grads = cached
                else:
                    grads = []
                opt_state = self._capture_optimizer_state()
                lr_scheduler_state = self._capture_lr_scheduler_state()
                outgoing_persisted = self._persist_actor_only_state(
                    self._current_session_id,
                    grads,
                    opt_state,
                    lr_scheduler_state,
                )
                self._cache_hot_session_state(
                    self._current_session_id,
                    grads,
                    opt_state,
                    lr_scheduler_state,
                    durable=True,
                )
                logger.debug(
                    "[Rank %s] Persisted actor-only state for session %s to %s",
                    self.rank,
                    self._current_session_id,
                    outgoing_persisted["path"],
                )

            incoming_gradients = None
            incoming_optimizer_state = None
            incoming_lr_scheduler_state = None
            hot_entry = self._session_hot_cache.get(new_session_id)
            if hot_entry is not None:
                incoming_gradients = self._session_gradients.get(new_session_id)
                incoming_optimizer_state = self._session_optimizer_states.get(new_session_id, {})
                incoming_lr_scheduler_state = self._session_lr_scheduler_states.get(new_session_id, {})
                self._drop_hot_session(new_session_id)
                incoming_source = "hot"
            else:
                session_manager = getattr(self, "_session_manager", None)
                allow_cold_restore = require_persisted or session_manager is not None
                persisted_state = None
                if allow_cold_restore:
                    persisted_state = self._load_persisted_actor_only_state(
                        new_session_id,
                        require=require_persisted,
                    )
                if persisted_state is not None:
                    incoming_gradients, incoming_optimizer_state, incoming_lr_scheduler_state = persisted_state
                    incoming_source = "cold"

            if incoming_gradients is not None and incoming_gradients is not _GRADIENTS_CONSUMED:
                self._restore_gradients(incoming_gradients)
                logger.debug(f"[Rank {self.rank}] Restored gradients for session {new_session_id}")
            else:
                self.engine.optimizer_zero_grad()
                if incoming_gradients is _GRADIENTS_CONSUMED:
                    logger.debug(
                        f"[Rank {self.rank}] Session {new_session_id} gradients were consumed, zeroed gradients"
                    )
                else:
                    logger.debug(f"[Rank {self.rank}] Session {new_session_id} - zeroed gradients")

            if incoming_optimizer_state is not None:
                self._restore_optimizer_state(incoming_optimizer_state)
                self._restore_lr_scheduler_state(incoming_lr_scheduler_state or {})
                logger.debug(
                    "[Rank %s] Restored actor-only state for session %s from %s cache",
                    self.rank,
                    new_session_id,
                    incoming_source,
                )
            else:
                self._reset_optimizer_state()
                logger.info(f"[Rank {self.rank}] New session {new_session_id} - reset optimizer state")

        self._current_session_id = new_session_id
        return {
            "status": "ok",
            "session_id": new_session_id,
            "incoming_source": incoming_source,
            "outgoing_persisted": outgoing_persisted,
            "hot_cache": self.get_hot_cache_info(),
        }

    def clear_session_state(self, session_id: str, traceparent: str | None = None) -> None:
        """Clear saved state for a session (call after session completes).

        Args:
            session_id: Session ID to clear.
        """
        self._bind_traceparent(traceparent)
        self._drop_hot_session(session_id)
        rank_path = self._actor_only_rank_snapshot_path(session_id)
        try:
            os.remove(rank_path)
        except FileNotFoundError:
            pass
        logger.debug(f"[Rank {self.rank}] Cleared state for session {session_id}")

    def has_session_state_cached(self, session_id: str) -> bool:
        """Whether this live worker still has the session's actor-only state in memory."""
        if self._current_session_id == session_id:
            return True
        return session_id in self._session_hot_cache
    def mark_session_loaded(self, session_id: str) -> None:
        """Record that a checkpoint-loaded session is now active on this rank."""
        self._drop_hot_session(session_id)
        self._current_session_id = session_id
        logger.info(f"[Rank {self.rank}] Marked loaded session active: {session_id}")

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
        override_tf_config["gradient_accumulation_fusion"] = False
        if not override_tf_config["gradient_accumulation_fusion"]:
            logger.info(
                f"[Rank {self.rank}] gradient_accumulation_fusion=False "
                f"(available={grad_accum_fusion_available}, lora_rank={self.lora_rank})"
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

    def reset_expert_bias(self, traceparent: str | None = None) -> dict:
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

        self._bind_traceparent(traceparent)

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
        traceparent: str | None = None,
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

        self._bind_traceparent(traceparent)
        reset_bias = self._resolve_reset_bias(reset_bias, default=False)
        output_rank = self._is_output_rank()

        # Reset expert_bias for train-inference consistency
        # expert_bias accumulates during training but isn't exported to vLLM
        # Reset ensures Megatron routing matches vLLM routing for correct KL calculation
        if reset_bias:
            self.reset_expert_bias(traceparent=traceparent)

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
                self._sticky_train_mode_last_used_s = time.perf_counter()
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
                    logger.warning(
                        "[Rank %d] sticky cleanup failed during forward_backward "
                        "error handling: reason=%s session=%s error_type=%s: %s",
                        self.rank,
                        "forward_backward_error",
                        session_id,
                        type(cleanup_error).__name__,
                        cleanup_error,
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
        traceparent: str | None = None,
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

        self._bind_traceparent(traceparent)
        reset_bias = self._resolve_reset_bias(reset_bias, default=True)
        output_rank = self._is_output_rank()

        # Reset expert_bias for train-inference consistency
        # expert_bias accumulates during training but isn't exported to vLLM
        # Reset ensures Megatron routing matches vLLM routing
        if reset_bias:
            self.reset_expert_bias(traceparent=traceparent)

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

    def forward_backward_reverse_kl(
        self,
        data_items: list[dict],
        reference_checkpoint_path: str,
        reference_actual_rank: int | None,
        temperature: float,
        session_id: str | None = None,
        traceparent: str | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
        reference_full_log_prob_chunks: list | None = None,
    ) -> dict:
        """Run reverse-KL distillation loss against a fixed reference adapter checkpoint."""
        import torch
        from tinker_server.backend.megatron_training import (
            create_reverse_kl_loss_fn,
            create_vocab_parallel_logits_extractor_fn,
            tinker_to_tensordict,
        )
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path

        from .mintx_ops import build_scoring_sequence, vocab_parallel_log_probs_from_logits_no_grad

        self._bind_traceparent(traceparent)
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature!r}")

        output_rank = self._is_output_rank()
        completion_lengths: list[int] = []
        completion_weights: list[list[float]] = []
        student_items: list[dict] = []
        reference_items: list[dict] = []
        for item_index, item in enumerate(data_items):
            student_input = item.get("student_input")
            reference_input = item.get("reference_input")
            target_tokens = item.get("target_tokens")
            weights = item.get("weights")
            if not isinstance(student_input, dict) or not isinstance(reference_input, dict):
                raise ValueError(f"Item {item_index}: student_input/reference_input must be dicts")
            student_tokens = flatten_encoded_text_chunks(student_input)
            reference_tokens = flatten_encoded_text_chunks(reference_input)
            if not student_tokens or not reference_tokens:
                raise ValueError(f"Item {item_index}: student/reference input must contain tokens")
            if not isinstance(target_tokens, dict) or not isinstance(weights, dict):
                raise ValueError(f"Item {item_index}: target_tokens and weights must be TensorData-like dicts")
            completion_tokens = target_tokens.get("data", [])
            weight_values = weights.get("data", [])
            if not isinstance(completion_tokens, list) or not completion_tokens:
                raise ValueError(f"Item {item_index}: target_tokens.data must be non-empty")
            if not isinstance(weight_values, list) or len(weight_values) != len(completion_tokens):
                raise ValueError(f"Item {item_index}: weights.data must align with target_tokens.data")
            completion_lengths.append(len(completion_tokens))
            completion_weights.append([float(x) for x in weight_values])

            student_full_input, student_completion_start = build_scoring_sequence(
                [int(x) for x in student_tokens],
                [int(x) for x in completion_tokens],
            )
            reference_full_input, reference_completion_start = build_scoring_sequence(
                [int(x) for x in reference_tokens],
                [int(x) for x in completion_tokens],
            )
            student_full_targets = student_full_input[1:] + [int(completion_tokens[-1])]
            reference_full_targets = reference_full_input[1:] + [int(completion_tokens[-1])]
            student_full_weights = [0.0] * student_completion_start + [float(x) for x in weight_values]
            reference_full_weights = [0.0] * reference_completion_start + [float(x) for x in weight_values]

            student_items.append(
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": student_full_input}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": student_full_targets, "shape": [len(student_full_targets)], "dtype": "int64"},
                        "weights": {"data": student_full_weights, "shape": [len(student_full_weights)], "dtype": "float32"},
                    },
                }
            )
            reference_items.append(
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": reference_full_input}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": reference_full_targets, "shape": [len(reference_full_targets)], "dtype": "int64"},
                        "weights": {"data": reference_full_weights, "shape": [len(reference_full_weights)], "dtype": "float32"},
                    },
                }
            )

        device = torch.cuda.current_device()
        max_token_len = get_model_config(self.base_model).max_model_len
        student_data = tinker_to_tensordict(
            student_items,
            max_token_len_per_gpu=max_token_len,
            device=f"cuda:{device}",
        )
        student_data.set_non_tensor("temperature", float(temperature))
        student_data.set_non_tensor("return_vocab_parallel_logits", True)
        if reference_full_log_prob_chunks is None:
            reference_data = tinker_to_tensordict(
                reference_items,
                max_token_len_per_gpu=max_token_len,
                device=f"cuda:{device}",
            )
            reference_data.set_non_tensor("temperature", float(temperature))
            reference_data.set_non_tensor("return_vocab_parallel_logits", True)

            current_adapter_state = self._capture_adapter_state_dict()
            rank_path = _get_rank_checkpoint_path(reference_checkpoint_path)
            adapter_file = rank_path + "_adapter.pt"
            if not os.path.isfile(adapter_file):
                raise FileNotFoundError(f"Reference adapter checkpoint not found: {adapter_file}")
            checkpoint = torch.load(adapter_file, map_location="cpu", weights_only=False)
            reference_adapter_state = checkpoint.get("adapter_state_dict", {})
            with self.engine.eval_mode():
                try:
                    self._restore_adapter_state_dict(
                        reference_adapter_state,
                        actual_rank=reference_actual_rank,
                        trainer_rank=self.lora_rank,
                        train_attn=train_attn,
                        train_mlp=train_mlp,
                        train_unembed=train_unembed,
                    )
                    extractor = create_vocab_parallel_logits_extractor_fn()
                    with torch.no_grad():
                        reference_result = self.engine.forward_backward_batch(
                            data=reference_data,
                            loss_function=extractor,
                            forward_only=True,
                        )
                    reference_model_output = reference_result.get("model_output", {})
                    reference_local_logits = reference_model_output.get("vocab_parallel_logits")
                    if reference_local_logits is None:
                        raise ValueError("reference forward missing vocab_parallel_logits")
                    if hasattr(reference_local_logits, "values"):
                        reference_local_logits = reference_local_logits.values()
                    reference_log_probs = vocab_parallel_log_probs_from_logits_no_grad(reference_local_logits)
                finally:
                    self._restore_adapter_state_dict(
                        current_adapter_state,
                        trainer_rank=self.lora_rank,
                        train_attn=train_attn,
                        train_mlp=train_mlp,
                        train_unembed=train_unembed,
                    )

            reference_loss_mask = reference_data["loss_mask"]
            if hasattr(reference_loss_mask, "values"):
                reference_loss_mask = reference_loss_mask.values()
            selected_reference_log_probs = reference_log_probs[reference_loss_mask != 0].detach()
            ref_chunks = []
            offset = 0
            for completion_len in completion_lengths:
                ref_chunks.append(
                    selected_reference_log_probs[offset: offset + completion_len].cpu()
                )
                offset += completion_len
        else:
            ref_chunks = [
                chunk.to(dtype=torch.float32, device="cpu")
                if hasattr(chunk, "to")
                else torch.tensor(chunk, dtype=torch.float32, device="cpu")
                for chunk in reference_full_log_prob_chunks
            ]
        # Re-materialize the current adapter weights before the student forward.
        # The 30B SDPO path has repeatedly surfaced zero-storage LoRA tensors in
        # live runs; capturing and restoring the current adapter state forces the
        # parameter storage back through a fresh CPU clone before the batch runs.
        # Keep teacher log-probs outside the TensorDict batch object. The student
        # forward path only needs input_ids/loss_mask; carrying the teacher tensor
        # inside the batch adds a large extra tensor field to every microbatch.
        flat_reference_log_probs = torch.cat(ref_chunks, dim=0)
        reverse_kl_loss_fn = create_reverse_kl_loss_fn(
            temperature=temperature,
            reference_log_probs=flat_reference_log_probs,
        )

        result = None
        def _run_reverse_kl_compute():
            nonlocal result
            watchdog = self._start_slow_op_watchdog(
                op="forward_backward_batch_reverse_kl",
                session_id=session_id,
                extra=f"items={len(data_items)}",
            )
            try:
                result = self.engine.forward_backward_batch(
                    data=student_data,
                    loss_function=reverse_kl_loss_fn,
                    forward_only=False,
                )
            finally:
                self._stop_slow_op_watchdog(watchdog)

        try:
            if self._sticky_enabled_for(session_id):
                sticky = self._ensure_sticky_train_mode(session_id=session_id, reason="forward_backward_reverse_kl")
                train_mode_reused = bool(sticky.get("reused", False))
                cached_grads = self._session_gradients.get(session_id)
                if cached_grads is not None and cached_grads is not _GRADIENTS_CONSUMED and not train_mode_reused:
                    self._restore_gradients(cached_grads)
                try:
                    _run_reverse_kl_compute()
                    self._sticky_train_mode_last_used_s = time.perf_counter()
                except Exception as original_error:
                    try:
                        self._release_sticky_train_mode(
                            reason="forward_backward_reverse_kl_error",
                            snapshot_gradients=False,
                        )
                    except Exception:
                        pass
                    raise original_error
            else:
                with self.engine.train_mode():
                    cached_grads = self._session_gradients.get(session_id) if session_id else None
                    if cached_grads is not None and cached_grads is not _GRADIENTS_CONSUMED:
                        self._restore_gradients(cached_grads)
                    _run_reverse_kl_compute()
                    if session_id is not None:
                        self._session_gradients[session_id] = self._capture_gradients()
        finally:
            pass
        if not output_rank:
            return {}

        result_metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        losses = result.get("loss", []) if isinstance(result, dict) else []
        total_loss = 0.0
        for loss in losses:
            if hasattr(loss, "item"):
                loss = loss.item()
            total_loss += float(loss)

        num_tokens = 0
        for value in result_metrics.get("num_tokens", []):
            if hasattr(value, "item"):
                value = value.item()
            num_tokens += int(value)

        kl_tensors = []
        for value in result_metrics.get("reverse_kl_tokens", []):
            if hasattr(value, "cpu"):
                value = value.cpu()
            kl_tensors.append(value)
        combined_kl = torch.cat(kl_tensors, dim=0) if kl_tensors else None

        outputs = []
        offset = 0
        for completion_len, weights in zip(completion_lengths, completion_weights, strict=True):
            if combined_kl is None:
                item_loss = total_loss / max(len(completion_lengths), 1)
            else:
                sample_kl = combined_kl[offset: offset + completion_len]
                offset += completion_len
                weights_t = torch.tensor(weights, dtype=sample_kl.dtype)
                item_loss = float((sample_kl * weights_t).sum().item())
            outputs.append(
                {
                    "loss": {
                        "data": [float(item_loss)],
                        "shape": [1],
                        "dtype": "float32",
                    }
                }
            )

        avg_loss = total_loss / max(float(num_tokens), 1.0)
        return {
            "outputs": outputs,
            "metrics": {
                "loss:mean": float(avg_loss),
                "reverse_kl:mean": float(avg_loss),
                "num_samples:sum": float(len(outputs)),
                "num_tokens:sum": float(num_tokens),
            },
            "type": "mint_forward_backward_reverse_kl",
        }

    def forward_reference_full_log_probs(
        self,
        data_items: list[dict],
        temperature: float,
        traceparent: str | None = None,
    ) -> dict:
        """Compute full-vocab teacher log-probs on masked completion tokens only."""
        import torch
        from tinker_server.backend.megatron_training import (
            create_vocab_parallel_logits_extractor_fn,
            tinker_to_tensordict,
        )

        from .mintx_ops import vocab_parallel_log_probs_from_logits_no_grad

        self._bind_traceparent(traceparent)
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature!r}")
        completion_lengths: list[int] = []
        for item_index, item in enumerate(data_items):
            weights = item.get("loss_fn_inputs", {}).get("weights", {})
            weight_values = weights.get("data", []) if isinstance(weights, dict) else []
            if not isinstance(weight_values, list):
                raise ValueError(f"Item {item_index}: weights.data must be list")
            completion_lengths.append(sum(1 for w in weight_values if float(w) != 0.0))

        device = torch.cuda.current_device()
        max_token_len = get_model_config(self.base_model).max_model_len
        td = tinker_to_tensordict(data_items, max_token_len_per_gpu=max_token_len, device=f"cuda:{device}")
        td.set_non_tensor("temperature", float(temperature))
        td.set_non_tensor("return_vocab_parallel_logits", True)

        extractor = create_vocab_parallel_logits_extractor_fn()
        with self.engine.eval_mode():
            with torch.no_grad():
                result = self.engine.forward_backward_batch(
                    data=td,
                    loss_function=extractor,
                    forward_only=True,
                )

        model_output = result.get("model_output", {})
        local_logits = model_output.get("vocab_parallel_logits")
        if local_logits is None:
            raise ValueError("reference forward missing vocab_parallel_logits")
        if hasattr(local_logits, "values"):
            local_logits = local_logits.values()
        local_log_probs = vocab_parallel_log_probs_from_logits_no_grad(local_logits)

        loss_mask = td["loss_mask"]
        if hasattr(loss_mask, "values"):
            loss_mask = loss_mask.values()
        selected_local = local_log_probs[loss_mask != 0].contiguous()

        local = selected_local.cpu()
        chunks = []
        offset = 0
        for clen in completion_lengths:
            chunks.append(local[offset:offset + clen].clone().tolist())
            offset += clen
        return {"reference_local_log_probs": chunks}

    def optim_step(
        self,
        learning_rate: float,
        session_id: str | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Run optimizer step (synchronized across ranks).

        Restores session's cached gradients before applying optimizer step.
        Clears cache after - gradients are consumed.

        Note: Session state swap (gradients + optimizer) is handled by
        MegatronWorkerGroup._ensure_session_loaded() calling swap_session_state()
        BEFORE forward_backward/optim_step. This method only needs to restore
        this session's cached gradients from the most recent forward_backward.
        """
        self._bind_traceparent(traceparent)
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

                # Optional diagnostics only. Missing Megatron optimizer modules must not
                # change optim_step semantics.
                if self.rank == 0:
                    try:
                        from megatron.core.optimizer import ChainedOptimizer
                    except ModuleNotFoundError:
                        ChainedOptimizer = None

                    optimizer = self.engine.optimizer
                    if optimizer is not None:
                        def iter_optimizers(opt):
                            if ChainedOptimizer is not None and isinstance(opt, ChainedOptimizer):
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

                # Safety: some optimizer_step implementations do not clear gradients.
                # Explicitly zero here to prevent stale cross-step accumulation when
                # sticky context is reused (e.g. CLOSE_ON_OPTIM=0).
                self._clear_optimizer_gradients(
                    session_id=session_id,
                    reason="post_optim_step",
                )
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
                if self._sticky_train_mode_ctx is not None:
                    self._sticky_train_mode_last_used_s = time.perf_counter()
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
                    logger.warning(
                        "[Rank %d] sticky cleanup failed during optim_step "
                        "error handling: reason=%s session=%s error_type=%s: %s",
                        self.rank,
                        "optim_step_error",
                        session_id,
                        type(cleanup_error).__name__,
                        cleanup_error,
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

    def get_lora_state_dict(
        self,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Get LoRA state dict in PEFT format via Megatron-Bridge only.

        The custom export/conversion path has been removed. All adapter export
        must flow through `bridge.export_adapter_weights()`, which is responsible
        for emitting PEFT-format tensors, including fused-QKV split handling.

        IMPORTANT: ALL ranks must participate because the bridge export can use
        distributed collectives internally. Only rank 0 returns materialized
        tensors; other ranks return an empty dict.
        """
        logger.info(
            "[Rank %s] get_lora_state_dict: ENTRY train_attn=%s train_mlp=%s train_unembed=%s",
            self.rank,
            train_attn,
            train_mlp,
            train_unembed,
        )
        self._release_sticky_for_aux_mode_transition(
            reason="get_lora_state_dict",
            snapshot_gradients=True,
        )
        train_attn = True if train_attn is None else bool(train_attn)
        train_mlp = True if train_mlp is None else bool(train_mlp)
        train_unembed = True if train_unembed is None else bool(train_unembed)

        bridge = getattr(self.engine, "bridge", None)
        if bridge is None or not hasattr(bridge, "export_adapter_weights"):
            raise RuntimeError(
                "Megatron-Bridge export_adapter_weights() is required for LoRA export; "
                "the legacy custom export path has been removed"
            )
        bridge_module = type(bridge).__module__
        use_bridge_internal_patches = bridge_module.startswith("megatron.bridge.")

        def _allow_exported_name(export_name: str) -> bool:
            tgt = self._classify_lora_param_target(export_name.lower())
            if tgt == "attn":
                return train_attn
            if tgt == "mlp":
                return train_mlp
            if tgt == "unembed":
                return train_unembed
            return True

        try:
            from megatron.core import parallel_state as mpu

            pipeline_world_size = int(mpu.get_pipeline_model_parallel_world_size())
        except Exception:
            pipeline_world_size = None

        restore_bridge_patch = None
        if use_bridge_internal_patches and pipeline_world_size == 1:
            from megatron.bridge.models.conversion import model_bridge as bridge_dispatch

            bridge_impl = bridge_dispatch.get_model_bridge(bridge._causal_lm_architecture)
            bridge_cls = type(bridge_impl)
            original_collect = getattr(bridge_cls, "_megatron_global_adapters_info_all_pp_ranks", None)
            if not callable(original_collect):
                raise RuntimeError(
                    "Megatron-Bridge export_adapter_weights() is missing "
                    "_megatron_global_adapters_info_all_pp_ranks; cannot apply single-PP export fix"
                )
            from collections import defaultdict
            import itertools

            from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global
            from megatron.bridge.models.conversion.utils import extract_sort_key, persistent_buffers
            from megatron.bridge.peft.canonical_lora import ModuleDict
            from megatron.bridge.peft.utils import ParallelLinearAdapter, get_adapter_attributes_from_linear
            from megatron.core.utils import get_pg_rank, unwrap_model

            _missing = object()
            previous_attr_values = {
                "_tinker_export_train_attn": getattr(bridge_cls, "_tinker_export_train_attn", _missing),
                "_tinker_export_train_mlp": getattr(bridge_cls, "_tinker_export_train_mlp", _missing),
                "_tinker_export_train_unembed": getattr(bridge_cls, "_tinker_export_train_unembed", _missing),
            }
            setattr(bridge_cls, "_tinker_export_train_attn", train_attn)
            setattr(bridge_cls, "_tinker_export_train_mlp", train_mlp)
            setattr(bridge_cls, "_tinker_export_train_unembed", train_unembed)

            def _allow_target(bridge_self, param_name: str) -> bool:
                allow_attn = bool(getattr(bridge_self, "_tinker_export_train_attn", True))
                allow_mlp = bool(getattr(bridge_self, "_tinker_export_train_mlp", True))
                allow_unembed = bool(getattr(bridge_self, "_tinker_export_train_unembed", True))
                tgt = MegatronRankWorker._classify_lora_param_target(param_name.lower())
                if tgt == "attn":
                    return allow_attn
                if tgt == "mlp":
                    return allow_mlp
                if tgt == "unembed":
                    return allow_unembed
                return True

            def _single_pp_collect(self, megatron_model):
                if hasattr(self, "_cached_param_objects_adapter"):
                    return self._cached_param_objects_adapter

                models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
                pp_group = mpu.get_pipeline_model_parallel_group()
                pp_rank = get_pg_rank(pp_group)
                model_config = unwrap_model(models)[0].config
                global_param_objects = []

                for vp_stage, model in enumerate(models):
                    for local_param_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                        if "_extra_state" in local_param_name:
                            continue
                        local_param_name = self._unwrap_name(local_param_name)
                        global_param_name = _megatron_local_name_to_global(
                            models, model_config, local_param_name, vp_stage
                        )
                        if not self._is_adapter_param_name(global_param_name) or not global_param_name.endswith(
                            ".linear_in.weight"
                        ):
                            continue
                        if not _allow_target(self, global_param_name):
                            continue

                        local_base_prefix = local_param_name.partition(".adapter.")[0]
                        global_base_name = global_param_name[: -len(".linear_in.weight")]
                        adapter, to_wrap = self._get_adapter_wrap_module(local_base_prefix, models, vp_stage)
                        if isinstance(adapter, ModuleDict):
                            adapter_name = local_param_name.removeprefix(local_base_prefix + ".adapter.").split(".")[0]
                            adapter = adapter[adapter_name]
                        if isinstance(adapter, ParallelLinearAdapter):
                            input_is_parallel = adapter.input_is_parallel
                            base_linear_is_parallel = True
                        else:
                            input_is_parallel, _, _, _, _, base_linear_is_parallel = get_adapter_attributes_from_linear(
                                to_wrap
                            )
                        global_param_objects.append(
                            (
                                global_base_name,
                                local_base_prefix,
                                input_is_parallel,
                                base_linear_is_parallel,
                                adapter.alpha,
                                adapter.dim,
                                pp_rank,
                                vp_stage,
                            )
                        )

                gathered_global_param_objects = sorted(
                    list(set(global_param_objects)),
                    key=lambda x: extract_sort_key(x[0]),
                )
                self._cached_param_objects_adapter = gathered_global_param_objects
                return gathered_global_param_objects

            bridge_cls._megatron_global_adapters_info_all_pp_ranks = _single_pp_collect
            logger.info("[Rank %s] Temporarily patched Megatron-Bridge single-PP adapter metadata gather", self.rank)

            def _restore_bridge_patch():
                bridge_cls._megatron_global_adapters_info_all_pp_ranks = original_collect
                for attr_name, previous_value in previous_attr_values.items():
                    if previous_value is _missing:
                        try:
                            delattr(bridge_cls, attr_name)
                        except AttributeError:
                            pass
                    else:
                        setattr(bridge_cls, attr_name, previous_value)

            restore_bridge_patch = _restore_bridge_patch

        if use_bridge_internal_patches:
            import torch
            from megatron.bridge.models.conversion import param_mapping as bridge_param_mapping
            import torch.distributed as dist

            restore_tp_gather = bridge_param_mapping.MegatronParamMapping.gather_from_tp_ranks
            gloo_tp_groups: dict[tuple[int, ...], object] = {}

            def _cpu_gloo_gather_from_tp_ranks(mapping, tensor: torch.Tensor) -> list[torch.Tensor]:
                if mapping.tp_size == 1:
                    return [tensor]

                if not dist.is_gloo_available():
                    raise RuntimeError("Gloo backend is unavailable; cannot run CPU TP gather for adapter export")

                group_ranks = tuple(dist.get_process_group_ranks(mapping.tp_group))
                gloo_group = gloo_tp_groups.get(group_ranks)
                if gloo_group is None:
                    gloo_group = dist.new_group(ranks=list(group_ranks), backend="gloo")
                    gloo_tp_groups[group_ranks] = gloo_group

                try:
                    cpu_tensor = tensor.detach().cpu().contiguous()
                except Exception as exc:
                    raise RuntimeError(
                        "Megatron-Bridge adapter export failed before TP gather could start: "
                        f"hf_param={getattr(mapping, 'hf_param', None)!r} "
                        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
                        f"tp_rank={getattr(mapping, 'tp_rank', None)!r} tp_size={getattr(mapping, 'tp_size', None)!r}"
                    ) from exc
                gathered = [torch.empty_like(cpu_tensor) for _ in range(mapping.tp_size)]
                dist.all_gather(gathered, cpu_tensor, group=gloo_group)
                return gathered

            bridge_param_mapping.MegatronParamMapping.gather_from_tp_ranks = _cpu_gloo_gather_from_tp_ranks
        else:
            bridge_param_mapping = None
            restore_tp_gather = None

        adapter_state: dict[str, torch.Tensor] = {}
        try:
            with self.engine.eval_mode():
                for name, tensor in bridge.export_adapter_weights(
                    self.engine.module,
                    cpu=True,
                    show_progress=False,
                ):
                    if self.rank == 0 and _allow_exported_name(str(name)):
                        adapter_state[str(name)] = tensor.clone()
        finally:
            if bridge_param_mapping is not None and restore_tp_gather is not None:
                bridge_param_mapping.MegatronParamMapping.gather_from_tp_ranks = restore_tp_gather
            if restore_bridge_patch is not None:
                restore_bridge_patch()

        if self.rank != 0:
            logger.info(
                "[Rank %s] get_lora_state_dict: returning empty dict (non-rank-0)",
                self.rank,
            )
            return {}

        if not adapter_state:
            logger.warning("[Rank 0] Megatron-Bridge export_adapter_weights returned no adapter tensors")
            return {}

        sample_keys = list(adapter_state.keys())[:5]
        logger.info(
            "[Rank 0] Megatron-Bridge export_adapter_weights returned %s tensors sample_keys=%s",
            len(adapter_state),
            sample_keys,
        )
        return adapter_state

    

    

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

    def debug_named_parameter(self, needle: str) -> dict:
        """Inspect matching parameters for metadata and basic readback health."""
        with self.engine.train_mode():
            from verl.utils.megatron_utils import unwrap_model

            model = unwrap_model(self.engine.module)
            while isinstance(model, list):
                model = model[0]

            rows = []
            for name, param in model.named_parameters():
                if needle not in name:
                    continue
                row = {
                    "rank": self.rank,
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "device": str(param.device),
                    "requires_grad": bool(param.requires_grad),
                }
                try:
                    row["sum"] = float(param.data.float().sum().item())
                except Exception as e:
                    row["sum_error"] = f"{type(e).__name__}: {e}"
                try:
                    cpu_tensor = param.detach().cpu()
                    row["cpu_shape"] = list(cpu_tensor.shape)
                except Exception as e:
                    row["cpu_error"] = f"{type(e).__name__}: {e}"
                rows.append(row)
            return {"matches": rows}

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
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
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
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Save checkpoint: LoRA weights + config + training metadata.

        IMPORTANT: ALL ranks must call this method because get_lora_state_dict()
        uses NCCL collectives. Only rank 0 saves to disk.

        Args:
            save_path: Directory path to save checkpoint files.
            step_count: Current training step (passed from MegatronWorkerGroup).
            actual_rank: Actual LoRA rank for current session (Phase 7).
                         If None, falls back to self.lora_rank (max_lora_rank).
        Returns:
            Dict with training metadata (rank 0 only, others return empty).
        """
        self._bind_traceparent(traceparent)
        import os
        import torch

        from safetensors.torch import save_file
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path

        # ALL ranks must call get_lora_state_dict - it uses NCCL collectives
        # Only rank 0 gets actual data, others get empty dict
        state_dict = self.get_lora_state_dict(
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

        os.makedirs(save_path, exist_ok=True)

        # Save distributed adapter state for training resume (per-rank mp_rank_*_adapter.pt).
        # ALL ranks must participate due to NCCL collectives.
        effective_rank = actual_rank if actual_rank is not None else self.lora_rank
        self.save_adapter_state(
            checkpoint_path=save_path,
            actual_rank=effective_rank,
            trainer_rank=self.lora_rank,
            traceparent=traceparent,
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
        ] if train_attn else []
        if train_mlp:
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
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
        traceparent: str | None = None,
    ) -> dict:
        """Save LoRA weights in PEFT format for sampling (no optimizer/resume artifacts).

        This is the minimal export required by `save_weights_for_sampler` to hot-load LoRA in vLLM.
        It intentionally avoids saving per-rank adapter shards and optimizer shards, because those
        are not required for sampling and can exceed memory/time budgets on large MoE models.
        """
        self._bind_traceparent(traceparent)
        import os

        from safetensors.torch import save_file

        # ALL ranks must call get_lora_state_dict - it uses NCCL collectives.
        state_dict = self.get_lora_state_dict(
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

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
        ] if train_attn else []
        if train_mlp:
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

    def load_optimizer_state(self, checkpoint_path: str, traceparent: str | None = None) -> dict:
        """Load per-rank optimizer shard from disk (if present)."""
        self._bind_traceparent(traceparent)
        import os

        import torch
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path

        self._release_sticky_for_aux_mode_transition(
            reason="load_optimizer_state",
            snapshot_gradients=True,
        )

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

    def check_optimizer_state_exists(
        self,
        checkpoint_path: str,
        traceparent: str | None = None,
    ) -> dict:
        """Check whether this rank's optimizer shard exists on shared storage."""
        self._bind_traceparent(traceparent)
        import os

        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path

        rank_path = _get_rank_checkpoint_path(checkpoint_path)
        optimizer_file = rank_path + "_optimizer.pt"
        exists = os.path.isfile(optimizer_file)
        return {
            "rank": self.rank,
            "exists": exists,
            "optimizer_file": optimizer_file,
        }


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
        traceparent: str | None = None,
        reload_optimizer_model_params: bool = True,
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
        self._bind_traceparent(traceparent)
        import os

        import torch

        from tinker_server.backend.lora_utils import pad_lora_state_dict
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path
        from verl.utils.megatron_utils import unwrap_model

        self._release_sticky_for_aux_mode_transition(
            reason="load_adapter_state",
            snapshot_gradients=True,
        )

        # Use train_mode context to ensure model is on GPU for loading
        with self.engine.train_mode():
            # Get rank-specific checkpoint path
            rank_path = _get_rank_checkpoint_path(checkpoint_path)
            adapter_file = rank_path + "_adapter.pt"

            if not os.path.isfile(adapter_file):
                raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_file}")

            checkpoint = torch.load(adapter_file, map_location="cpu")
            adapter_state = checkpoint.get("adapter_state_dict", {})
            expert_bias_state = checkpoint.get("expert_bias_state_dict", {})

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

            if isinstance(expert_bias_state, dict):
                named_modules = dict(unwrapped.named_modules())
                for module_name, bias_value in expert_bias_state.items():
                    if not isinstance(module_name, str):
                        continue
                    module = named_modules.get(module_name)
                    if module is None or not hasattr(module, "expert_bias"):
                        continue
                    if isinstance(bias_value, torch.Tensor):
                        module.expert_bias.copy_(bias_value.to(module.expert_bias.device))

            optimizer = getattr(self.engine, "optimizer", None)
            reload_model_params = getattr(optimizer, "reload_model_params", None)
            if reload_optimizer_model_params and callable(reload_model_params):
                reload_model_params(state_dict=adapter_state)

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
        self,
        checkpoint_path: str,
        actual_rank: int | None = None,
        trainer_rank: int | None = None,
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
        import os
        from pathlib import Path

        import torch

        from tinker_server.backend.lora_utils import truncate_lora_state_dict
        from verl.utils.megatron_peft_utils import _get_rank_checkpoint_path, get_adapter_state_dict

        os.makedirs(checkpoint_path, exist_ok=True)
        self._release_sticky_for_aux_mode_transition(
            reason="save_adapter_state",
            snapshot_gradients=True,
        )

        # Use train_mode context to ensure model is on GPU for saving
        with self.engine.train_mode():
            # Get adapter state dict
            adapter_state = get_adapter_state_dict(self.engine.module)
            expert_bias_state = {}
            module_chunks = self.engine.module if isinstance(self.engine.module, list) else [self.engine.module]
            for chunk_idx, module_chunk in enumerate(module_chunks):
                for module_name, module in module_chunk.named_modules():
                    if not module_name or not hasattr(module, "expert_bias") or module.expert_bias is None:
                        continue
                    key = module_name
                    if key in expert_bias_state:
                        key = f"chunk{chunk_idx}.{module_name}"
                    expert_bias_state[key] = module.expert_bias.detach().cpu().clone()

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

            torch.save(
                {
                    "adapter_state_dict": adapter_state,
                    "expert_bias_state_dict": expert_bias_state,
                },
                adapter_file,
            )

        logger.info(f"[Rank {self.rank}] Saved adapter state to {checkpoint_path}")

        if self.rank == 0:
            return {"status": "ok", "path": checkpoint_path, "actual_rank": actual_rank}
        return {}

    def reset_optimizer(
        self,
        learning_rate: float | None = None,
        traceparent: str | None = None,
        *,
        zero_grad_buffers: bool = True,
    ) -> dict:
        """Reset optimizer state for a new session.

        Updates learning rate, zeros gradients, and clears optimizer momentum so
        session-local state cannot leak across non-resume restores.

        Args:
            learning_rate: Optional new learning rate. If None, keeps current.

        Returns:
            Dict with status info.
        """
        self._bind_traceparent(traceparent)
        # Update learning rate if specified
        if learning_rate is not None:
            self.learning_rate = learning_rate
            # Update all param groups
            for group in self.engine.optimizer.param_groups:
                group['lr'] = learning_rate

        if zero_grad_buffers:
            self.engine.optimizer_zero_grad()

        # Clear momentum/variance buffers so future steps start from a clean optimizer.
        self._reset_optimizer_state()

        logger.info(
            f"[Rank {self.rank}] Reset optimizer "
            f"(lr={learning_rate or self.learning_rate}, zero_grad_buffers={zero_grad_buffers}, state cleared)"
        )

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
        try:
            if self._sticky_train_mode_ctx is not None:
                self._release_sticky_train_mode(reason="shutdown", snapshot_gradients=False)
        except Exception as e:
            logger.warning(
                "[Rank %d] sticky release failed during shutdown: %s: %s",
                self.rank, type(e).__name__, e,
            )
        finally:
            # Clear session state and process group regardless of release outcome
            self._session_gradients.clear()
            self._session_optimizer_states.clear()
            self._session_lr_scheduler_states.clear()
            self._session_hot_cache.clear()
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
            base_path = _default_megatron_sessions_base_path()

        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self._session_metadata: dict[str, dict] = {}  # session_id -> {step, lr, actual_rank}
        logger.info(f"[MegatronSessionStateManager] Initialized with base_path={base_path}")

    def get_session_path(self, session_id: str) -> str:
        """Get checkpoint directory path for a session."""
        return os.path.join(self.base_path, f"{session_id}_checkpoint")

    def _actor_only_snapshot_manifest_path(self, session_id: str) -> str:
        return _actor_only_snapshot_manifest_path(self.get_session_path(session_id))

    def _external_checkpoint_marker_path(self, session_id: str) -> str:
        return os.path.join(self.base_path, f"{session_id}_external_checkpoint.json")
    def session_exists(self, session_id: str) -> bool:
        """Check if a session has saved adapter state."""
        session_path = self.get_session_path(session_id)
        import glob
        adapter_files = glob.glob(os.path.join(session_path, "mp_rank_*_adapter.pt"))
        return len(adapter_files) > 0

    def _metadata_path(self, session_id: str) -> str:
        return os.path.join(self.base_path, f"{session_id}_checkpoint.session_metadata.json")

    def _actor_only_state_path(self, session_id: str) -> str:
        return os.path.join(self.base_path, f"{session_id}_checkpoint.actor_only_state.json")

    def _detach_session_path_from_checkpoint(self, session_id: str) -> None:
        session_path = self.get_session_path(session_id)
        if not os.path.islink(session_path):
            return
        os.unlink(session_path)
        os.makedirs(session_path, exist_ok=True)
        logger.info(
            "[MegatronSessionStateManager] Detached session %s from primed checkpoint path for private writes",
            session_id,
        )

    def checkpoint_identity(self, checkpoint_path: str) -> str:
        digest = hashlib.sha256()
        for root, dirnames, filenames in os.walk(checkpoint_path):
            dirnames.sort()
            filenames.sort()
            rel_root = os.path.relpath(root, checkpoint_path)
            digest.update(rel_root.encode("utf-8"))
            for filename in filenames:
                path = os.path.join(root, filename)
                stat = os.stat(path, follow_symlinks=False)
                rel_path = os.path.relpath(path, checkpoint_path)
                digest.update(rel_path.encode("utf-8"))
                digest.update(str(stat.st_size).encode("utf-8"))
                digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        return digest.hexdigest()

    def _replace_session_dir_with_snapshot(self, session_id: str, checkpoint_path: str) -> str:
        import shutil

        session_path = self.get_session_path(session_id)
        if os.path.lexists(session_path):
            if os.path.islink(session_path) or os.path.isfile(session_path):
                os.unlink(session_path)
            else:
                shutil.rmtree(session_path)
        os.makedirs(session_path, exist_ok=True)

        for entry in os.scandir(checkpoint_path):
            src = entry.path
            dst = os.path.join(session_path, entry.name)
            if entry.is_dir(follow_symlinks=False):
                shutil.copytree(src, dst, copy_function=shutil.copy2)
            else:
                shutil.copy2(src, dst)
        return session_path

    def save_metadata(
        self,
        session_id: str,
        step: int,
        lr: float,
        actual_rank: int | None = None,
        *,
        optimizer_restored: bool = True,
        checkpoint_path: str | None = None,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
        checkpoint_identity: str | None = None,
    ):
        """Save session metadata (step count, learning rate, actual rank)."""
        meta = {
            "step": step,
            "lr": lr,
            "actual_rank": actual_rank,
            "optimizer_restored": bool(optimizer_restored),
            "checkpoint_path": os.path.realpath(checkpoint_path or self.get_session_path(session_id)),
            "train_attn": bool(train_attn),
            "train_mlp": bool(train_mlp),
            "train_unembed": bool(train_unembed),
            "checkpoint_identity": checkpoint_identity
            or self.checkpoint_identity(checkpoint_path or self.get_session_path(session_id)),
        }
        self._session_metadata[session_id] = meta
        os.makedirs(self.base_path, exist_ok=True)
        metadata_path = self._metadata_path(session_id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self._maybe_recycle_cache()

    def mark_actor_only_state(
        self,
        session_id: str,
        *,
        reason: str,
        actor_name: str | None = None,
    ) -> None:
        session_path = self.get_session_path(session_id)
        self._detach_session_path_from_checkpoint(session_id)
        os.makedirs(session_path, exist_ok=True)
        marker_path = self._actor_only_state_path(session_id)
        tmp_path = f"{marker_path}.tmp"
        payload = {
            "reason": str(reason),
            "actor_name": None if actor_name is None else str(actor_name),
            "updated_at": time.time(),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, marker_path)

    def has_actor_only_state(self, session_id: str) -> bool:
        marker_path = self._actor_only_state_path(session_id)
        if not os.path.exists(marker_path):
            return False
        self._read_actor_only_state(marker_path)
        return True

    def clear_actor_only_state(self, session_id: str) -> None:
        marker_path = self._actor_only_state_path(session_id)
        try:
            os.remove(marker_path)
        except FileNotFoundError:
            return

    def mark_external_checkpoint(
        self,
        session_id: str,
        *,
        checkpoint_path: str,
        reason: str,
        actor_name: str | None = None,
    ) -> dict:
        session_path = self.get_session_path(session_id)
        os.makedirs(session_path, exist_ok=True)
        marker_path = self._external_checkpoint_marker_path(session_id)
        tmp_path = f"{marker_path}.tmp"
        payload = {
            "checkpoint_path": str(checkpoint_path),
            "reason": str(reason),
            "actor_name": None if actor_name is None else str(actor_name),
            "updated_at": time.time(),
            "is_fresh": True,
            "invalidated_at": None,
            "invalidated_reason": None,
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, marker_path)
        self._maybe_recycle_cache()
        return payload

    def get_external_checkpoint(self, session_id: str) -> dict | None:
        marker_path = self._external_checkpoint_marker_path(session_id)
        if not os.path.exists(marker_path):
            return None
        return self._read_external_checkpoint(marker_path)

    def invalidate_external_checkpoint(self, session_id: str, *, reason: str) -> dict | None:
        marker_path = self._external_checkpoint_marker_path(session_id)
        if not os.path.exists(marker_path):
            return None
        payload = self._read_external_checkpoint(marker_path)
        if not payload.get("is_fresh", False) and payload.get("invalidated_reason") == str(reason):
            return payload
        payload["is_fresh"] = False
        payload["invalidated_at"] = time.time()
        payload["invalidated_reason"] = str(reason)
        tmp_path = f"{marker_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, marker_path)
        self._maybe_recycle_cache()
        return payload

    def clear_external_checkpoint(self, session_id: str) -> None:
        marker_path = self._external_checkpoint_marker_path(session_id)
        try:
            os.remove(marker_path)
        except FileNotFoundError:
            return

    def save_persisted_actor_only_state(
        self,
        session_id: str,
        *,
        actor_name: str,
        worker_entries: list[dict],
    ) -> dict:
        session_path = self.get_session_path(session_id)
        os.makedirs(session_path, exist_ok=True)
        manifest_path = self._actor_only_snapshot_manifest_path(session_id)
        tmp_path = f"{manifest_path}.tmp"
        normalized_entries = []
        total_bytes = 0
        for entry in worker_entries:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Invalid persisted actor-only worker entry type {type(entry).__name__} for session {session_id}"
                )
            rank = entry.get("rank")
            path = entry.get("path")
            byte_count = entry.get("bytes")
            if not isinstance(rank, int) or rank < 0:
                raise RuntimeError(f"Invalid rank in persisted actor-only entry for session {session_id}: {entry!r}")
            if not isinstance(path, str) or not path:
                raise RuntimeError(f"Invalid path in persisted actor-only entry for session {session_id}: {entry!r}")
            if not isinstance(byte_count, int) or byte_count < 0:
                raise RuntimeError(f"Invalid bytes in persisted actor-only entry for session {session_id}: {entry!r}")
            normalized_entries.append({"rank": rank, "path": path, "bytes": byte_count})
            total_bytes += byte_count
        payload = {
            "version": 1,
            "session_id": session_id,
            "actor_name": actor_name,
            "updated_at": time.time(),
            "total_bytes": total_bytes,
            "rank_files": sorted(normalized_entries, key=lambda item: item["rank"]),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, manifest_path)
        self._maybe_recycle_cache()
        return payload

    def _read_actor_only_state(self, marker_path: str) -> dict:
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read actor_only_state marker {marker_path}: {type(e).__name__}: {e}"
            ) from e
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Invalid actor_only_state marker payload type {type(payload).__name__} in {marker_path}"
            )
        marker_actor_name = payload.get("actor_name")
        if not isinstance(marker_actor_name, str) or not marker_actor_name:
            raise RuntimeError(
                f"Invalid actor_only_state marker actor_name={marker_actor_name!r} in {marker_path}"
            )
        return payload

    def _read_external_checkpoint(self, marker_path: str) -> dict:
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read external checkpoint marker {marker_path}: {type(e).__name__}: {e}"
            ) from e
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Invalid external checkpoint marker payload type {type(payload).__name__} in {marker_path}"
            )
        checkpoint_path = payload.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise RuntimeError(
                f"Invalid external checkpoint marker checkpoint_path={checkpoint_path!r} in {marker_path}"
            )
        actor_name = payload.get("actor_name")
        if actor_name is not None and (not isinstance(actor_name, str) or not actor_name):
            raise RuntimeError(
                f"Invalid external checkpoint marker actor_name={actor_name!r} in {marker_path}"
            )
        is_fresh = payload.get("is_fresh", False)
        if not isinstance(is_fresh, bool):
            raise RuntimeError(
                f"Invalid external checkpoint marker is_fresh={is_fresh!r} in {marker_path}"
            )
        invalidated_at = payload.get("invalidated_at")
        if invalidated_at is not None and not isinstance(invalidated_at, (int, float)):
            raise RuntimeError(
                f"Invalid external checkpoint marker invalidated_at={invalidated_at!r} in {marker_path}"
            )
        invalidated_reason = payload.get("invalidated_reason")
        if invalidated_reason is not None and (
            not isinstance(invalidated_reason, str) or not invalidated_reason
        ):
            raise RuntimeError(
                f"Invalid external checkpoint marker invalidated_reason={invalidated_reason!r} in {marker_path}"
            )
        payload["is_fresh"] = is_fresh
        payload["invalidated_at"] = invalidated_at
        payload["invalidated_reason"] = invalidated_reason
        return payload

    def _read_persisted_actor_only_state(self, manifest_path: str) -> dict:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read actor-only snapshot manifest {manifest_path}: {type(e).__name__}: {e}"
            ) from e
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Invalid actor-only snapshot manifest payload type {type(payload).__name__} in {manifest_path}"
            )
        actor_name = payload.get("actor_name")
        total_bytes = payload.get("total_bytes")
        rank_files = payload.get("rank_files")
        if not isinstance(actor_name, str) or not actor_name:
            raise RuntimeError(
                f"Invalid actor-only snapshot manifest actor_name={actor_name!r} in {manifest_path}"
            )
        if not isinstance(total_bytes, int) or total_bytes < 0:
            raise RuntimeError(
                f"Invalid actor-only snapshot manifest total_bytes={total_bytes!r} in {manifest_path}"
            )
        if not isinstance(rank_files, list):
            raise RuntimeError(
                f"Invalid actor-only snapshot manifest rank_files type {type(rank_files).__name__} in {manifest_path}"
            )
        return payload

    def has_persisted_actor_only_state(self, session_id: str) -> bool:
        manifest_path = self._actor_only_snapshot_manifest_path(session_id)
        if not os.path.exists(manifest_path):
            return False
        self._read_persisted_actor_only_state(manifest_path)
        return True

    def get_persisted_actor_only_state(self, session_id: str) -> dict | None:
        manifest_path = self._actor_only_snapshot_manifest_path(session_id)
        if not os.path.exists(manifest_path):
            return None
        return self._read_persisted_actor_only_state(manifest_path)

    def clear_persisted_actor_only_state(self, session_id: str) -> None:
        import shutil

        manifest_path = self._actor_only_snapshot_manifest_path(session_id)
        snapshot_dir = _actor_only_snapshot_dir(self.get_session_path(session_id))
        try:
            os.remove(manifest_path)
        except FileNotFoundError:
            pass
        if os.path.isdir(snapshot_dir):
            shutil.rmtree(snapshot_dir)

    def list_actor_only_state_sessions(self, actor_name: str) -> list[str]:
        import glob

        dirty_sessions: list[str] = []
        pattern = os.path.join(self.base_path, "*_checkpoint.actor_only_state.json")
        for marker_path in glob.glob(pattern):
            payload = self._read_actor_only_state(marker_path)
            marker_actor_name = payload["actor_name"]
            if marker_actor_name != actor_name:
                continue
            marker_name = os.path.basename(marker_path)
            suffix = "_checkpoint.actor_only_state.json"
            if not marker_name.endswith(suffix):
                continue
            dirty_sessions.append(marker_name[: -len(suffix)])
        return sorted(set(dirty_sessions))

    def list_persisted_actor_only_state(self, actor_name: str | None = None) -> dict[str, dict]:
        import glob

        manifests: dict[str, dict] = {}
        pattern = os.path.join(self.base_path, "*_checkpoint", "actor_only_state_manifest.json")
        for manifest_path in glob.glob(pattern):
            payload = self._read_persisted_actor_only_state(manifest_path)
            if actor_name is not None and payload.get("actor_name") != actor_name:
                continue
            session_dir = os.path.basename(os.path.dirname(manifest_path))
            if not session_dir.endswith("_checkpoint"):
                continue
            manifests[session_dir[: -len("_checkpoint")]] = payload
        return manifests

    def prime_session(
        self,
        session_id: str,
        checkpoint_path: str,
        *,
        step: int,
        lr: float,
        actual_rank: int | None = None,
        optimizer_restored: bool = True,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
        checkpoint_identity: str | None = None,
    ) -> str:
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
        session_path = self._replace_session_dir_with_snapshot(session_id, checkpoint_path)
        self.save_metadata(
            session_id,
            step,
            lr,
            actual_rank,
            optimizer_restored=optimizer_restored,
            checkpoint_path=checkpoint_path,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
            checkpoint_identity=checkpoint_identity or self.checkpoint_identity(checkpoint_path),
        )
        return session_path

    def _session_last_updated_at(self, session_path: str) -> float:
        latest = os.path.getmtime(session_path)
        if os.path.isdir(session_path) and not os.path.islink(session_path):
            for root, _dirs, files in os.walk(session_path):
                for name in files:
                    try:
                        latest = max(latest, os.path.getmtime(os.path.join(root, name)))
                    except FileNotFoundError:
                        continue
        return latest

    def _session_total_bytes(self, session_path: str) -> int:
        if os.path.islink(session_path):
            try:
                return int(os.lstat(session_path).st_size)
            except FileNotFoundError:
                return 0
        total = 0
        for root, _dirs, files in os.walk(session_path):
            for name in files:
                try:
                    total += int(os.path.getsize(os.path.join(root, name)))
                except FileNotFoundError:
                    continue
        return total

    def _iter_cache_entries(self) -> list[_MegatronSessionCacheEntry]:
        import glob

        entries: list[_MegatronSessionCacheEntry] = []
        now = time.time()
        pattern = os.path.join(self.base_path, "*_checkpoint")
        for session_path in sorted(glob.glob(pattern)):
            session_dir = os.path.basename(session_path)
            if not session_dir.endswith("_checkpoint"):
                continue
            session_id = session_dir[: -len("_checkpoint")]
            actor_name = None
            cold_safe = False
            skip_reason = None
            if os.path.islink(session_path):
                cold_safe = True
                skip_reason = None
            else:
                external_marker = self.get_external_checkpoint(session_id)
                dirty_marker = self.has_actor_only_state(session_id)
                persisted = self.get_persisted_actor_only_state(session_id)
                if external_marker is not None and bool(external_marker.get("is_fresh", False)):
                    actor_name = external_marker.get("actor_name")
                    cold_safe = True
                elif dirty_marker:
                    actor_name = self._read_actor_only_state(self._actor_only_state_path(session_id)).get("actor_name")
                    skip_reason = "actor_only_state_dirty"
                elif external_marker is not None:
                    actor_name = external_marker.get("actor_name")
                    skip_reason = "stale_external_checkpoint"
                else:
                    skip_reason = "no_external_checkpoint"
                if actor_name is None and isinstance(persisted, dict):
                    actor_name = persisted.get("actor_name")
            updated_at = self._session_last_updated_at(session_path)
            entries.append(
                _MegatronSessionCacheEntry(
                    session_id=session_id,
                    session_path=session_path,
                    total_bytes=self._session_total_bytes(session_path),
                    updated_at=updated_at,
                    age_s=max(0.0, now - updated_at),
                    actor_name=actor_name,
                    cold_safe=cold_safe,
                    skip_reason=skip_reason,
                )
            )
        return entries

    def get_cache_usage(self, *, actor_name: str | None = None) -> dict:
        entries = [
            entry
            for entry in self._iter_cache_entries()
            if actor_name is None or entry.actor_name == actor_name
        ]
        total_bytes = sum(entry.total_bytes for entry in entries)
        oldest_age_s = max((entry.age_s for entry in entries), default=0.0)
        skipped = [entry for entry in entries if not entry.cold_safe]
        evictable = [entry for entry in entries if entry.cold_safe]
        stale = [entry for entry in skipped if entry.skip_reason == "stale_external_checkpoint"]
        dirty = [entry for entry in skipped if entry.skip_reason == "actor_only_state_dirty"]
        missing = [entry for entry in skipped if entry.skip_reason == "no_external_checkpoint"]
        return {
            "base_path": self.base_path,
            "actor_name": actor_name,
            "session_count": len(entries),
            "total_bytes": total_bytes,
            "oldest_entry_age_s": oldest_age_s,
            "skipped_not_cold_safe_count": len(skipped),
            "skipped_not_cold_safe_sessions": [entry.session_id for entry in skipped],
            "stale_external_checkpoint_count": len(stale),
            "stale_external_checkpoint_sessions": [entry.session_id for entry in stale],
            "actor_only_state_dirty_count": len(dirty),
            "actor_only_state_dirty_sessions": [entry.session_id for entry in dirty],
            "no_external_checkpoint_count": len(missing),
            "no_external_checkpoint_sessions": [entry.session_id for entry in missing],
            "evictable_session_count": len(evictable),
            "evictable_bytes": sum(entry.total_bytes for entry in evictable),
        }

    def recycle_cache(
        self,
        *,
        max_total_bytes: int | None = None,
        max_age_s: float | None = None,
        max_bytes_per_actor: int | None = None,
    ) -> dict:
        max_total_bytes = (
            max(0, _env_int("MINT_MEGATRON_SESSION_CACHE_MAX_BYTES", 0))
            if max_total_bytes is None
            else max(0, int(max_total_bytes))
        )
        max_age_s = (
            max(0.0, _env_float("MINT_MEGATRON_SESSION_CACHE_MAX_AGE_S", 0.0))
            if max_age_s is None
            else max(0.0, float(max_age_s))
        )
        max_bytes_per_actor = (
            max(0, _env_int("MINT_MEGATRON_SESSION_CACHE_MAX_BYTES_PER_ACTOR", 0))
            if max_bytes_per_actor is None
            else max(0, int(max_bytes_per_actor))
        )

        before = self.get_cache_usage()
        evicted: list[dict[str, object]] = []

        def _evict_entry(entry: _MegatronSessionCacheEntry, reason: str) -> None:
            if not entry.cold_safe:
                return
            if self.delete_session(entry.session_id):
                evicted.append(
                    {
                        "session_id": entry.session_id,
                        "bytes": entry.total_bytes,
                        "reason": reason,
                        "actor_name": entry.actor_name,
                    }
                )

        entries = self._iter_cache_entries()
        if max_age_s > 0:
            for entry in sorted(entries, key=lambda item: item.updated_at):
                if entry.cold_safe and entry.age_s > max_age_s:
                    _evict_entry(entry, f"max_age_s>{max_age_s}")
            entries = self._iter_cache_entries()

        if max_bytes_per_actor > 0:
            bytes_by_actor: dict[str, int] = {}
            grouped: dict[str, list[_MegatronSessionCacheEntry]] = {}
            for entry in entries:
                if entry.actor_name is None:
                    continue
                bytes_by_actor[entry.actor_name] = bytes_by_actor.get(entry.actor_name, 0) + entry.total_bytes
                grouped.setdefault(entry.actor_name, []).append(entry)
            for actor_key, actor_entries in grouped.items():
                current = bytes_by_actor.get(actor_key, 0)
                for entry in sorted(actor_entries, key=lambda item: item.updated_at):
                    if current <= max_bytes_per_actor:
                        break
                    if not entry.cold_safe:
                        continue
                    _evict_entry(entry, f"max_bytes_per_actor>{max_bytes_per_actor}")
                    current -= entry.total_bytes
            entries = self._iter_cache_entries()

        if max_total_bytes > 0:
            current_total = sum(entry.total_bytes for entry in entries)
            for entry in sorted(entries, key=lambda item: item.updated_at):
                if current_total <= max_total_bytes:
                    break
                if not entry.cold_safe:
                    continue
                _evict_entry(entry, f"max_total_bytes>{max_total_bytes}")
                current_total -= entry.total_bytes

        after = self.get_cache_usage()
        return {
            "base_path": self.base_path,
            "max_total_bytes": max_total_bytes,
            "max_age_s": max_age_s,
            "max_bytes_per_actor": max_bytes_per_actor,
            "evicted_sessions": evicted,
            "evicted_session_count": len(evicted),
            "evicted_bytes": sum(int(item["bytes"]) for item in evicted),
            "before": before,
            "after": after,
        }

    def _maybe_recycle_cache(self) -> dict | None:
        max_total_bytes = max(0, _env_int("MINT_MEGATRON_SESSION_CACHE_MAX_BYTES", 0))
        max_age_s = max(0.0, _env_float("MINT_MEGATRON_SESSION_CACHE_MAX_AGE_S", 0.0))
        max_bytes_per_actor = max(0, _env_int("MINT_MEGATRON_SESSION_CACHE_MAX_BYTES_PER_ACTOR", 0))
        if max_total_bytes <= 0 and max_age_s <= 0 and max_bytes_per_actor <= 0:
            return None
        result = self.recycle_cache(
            max_total_bytes=max_total_bytes,
            max_age_s=max_age_s,
            max_bytes_per_actor=max_bytes_per_actor,
        )
        if result["evicted_session_count"]:
            logger.info(
                "[MegatronSessionStateManager] Recycled %s session(s) from %s (bytes=%s)",
                result["evicted_session_count"],
                self.base_path,
                result["evicted_bytes"],
            )
        return result

    def get_metadata(self, session_id: str) -> dict | None:
        """Get session metadata if exists."""
        metadata_path = self._metadata_path(session_id)
        if not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None
        validated = self._validate_metadata(meta, session_id=session_id)
        if validated is not None:
            self._session_metadata[session_id] = validated
        return validated

    def _validate_metadata(self, meta: object, *, session_id: str) -> dict | None:
        if not isinstance(meta, dict):
            return None
        step = meta.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            return None
        lr_value = meta.get("lr")
        if isinstance(lr_value, bool):
            return None
        try:
            lr = float(lr_value)
        except Exception:
            return None
        if not math.isfinite(lr):
            return None
        actual_rank = meta.get("actual_rank")
        if actual_rank is not None:
            if not isinstance(actual_rank, int) or isinstance(actual_rank, bool) or actual_rank <= 0:
                return None
        optimizer_restored = meta.get("optimizer_restored", True)
        if not isinstance(optimizer_restored, bool):
            return None
        checkpoint_path = meta.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            return None
        checkpoint_identity = meta.get("checkpoint_identity")
        if not isinstance(checkpoint_identity, str) or not checkpoint_identity:
            return None
        train_attn = meta.get("train_attn", True)
        train_mlp = meta.get("train_mlp", True)
        train_unembed = meta.get("train_unembed", True)
        if not isinstance(train_attn, bool) or not isinstance(train_mlp, bool) or not isinstance(train_unembed, bool):
            return None
        return {
            "step": step,
            "lr": lr,
            "actual_rank": actual_rank,
            "optimizer_restored": optimizer_restored,
            "checkpoint_path": checkpoint_path,
            "checkpoint_identity": checkpoint_identity,
            "train_attn": train_attn,
            "train_mlp": train_mlp,
            "train_unembed": train_unembed,
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete session checkpoint and metadata."""
        import shutil

        session_path = self.get_session_path(session_id)
        deleted = False
        if os.path.lexists(session_path):
            if os.path.islink(session_path) or os.path.isfile(session_path):
                os.unlink(session_path)
            else:
                shutil.rmtree(session_path)
            deleted = True
        self.clear_external_checkpoint(session_id)
        if session_id in self._session_metadata:
            del self._session_metadata[session_id]
            deleted = True
        for sidecar_path in (self._metadata_path(session_id), self._actor_only_state_path(session_id)):
            try:
                os.remove(sidecar_path)
                deleted = True
            except FileNotFoundError:
                pass
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
        observability_base_model: str | None = None,
    ):
        init_actor_observability()
        self.base_model = base_model
        self.observability_base_model = str(observability_base_model or base_model or "unknown")
        self.lora_rank = lora_rank  # This is max_lora_rank for Phase 7
        self.learning_rate = learning_rate
        self.config = distributed_config or DistributedConfig()

        self.workers: list[ray.actor.ActorHandle] = []
        self.placement_group = None
        self._step_count = 0
        self._current_session: str | None = None  # Phase 6: session tracking
        self._session_unknown_due_to_partial_swap = False
        self._actual_rank: int | None = None  # Phase 7: actual LoRA rank for current session
        self._last_session_switch_stats: dict[str, object] | None = None
        self._session_manager = MegatronSessionStateManager()  # Issue #44: session state management
        self._master_addr: str | None = None
        self._master_port: int | None = None
        self._placement_bundle_node_ips: list[str | None] = []
        self._placement_requested_node_ips: list[str] = []

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

    def _bind_traceparent(self, traceparent: str | None) -> None:
        if isinstance(traceparent, str) and traceparent:
            restore_trace_id_from_traceparent(traceparent)

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

    def _training_remote_call_timeout_s(self, op: str) -> float | None:
        configured = server_config.training_remote_call_timeout_s
        if configured is not None:
            configured = float(configured)
            return configured if configured > 0 else None

        if op == "train_step":
            return 3600.0
        return 1800.0

    def _ray_get_group_results(
        self,
        futures: list[ray.ObjectRef],
        *,
        op: str,
        session_id: str | None,
        timeout_s: float | None = None,
    ) -> list:
        if timeout_s is None:
            timeout_s = self._training_remote_call_timeout_s(op)
        try:
            if timeout_s is not None and timeout_s > 0:
                return ray.get(futures, timeout=timeout_s)
            return ray.get(futures)
        except TypeError as e:
            if "not an ray.ObjectRef" in str(e):
                return list(futures)
            raise
        except ray.exceptions.GetTimeoutError as e:
            raise RuntimeError(
                f"Megatron worker group {op} timed out after {timeout_s}s "
                f"session_id={session_id!r} workers={len(self.workers)}"
            ) from e

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
                required_by_node_ip: dict[str, int] = {}
                for i in range(world_size):
                    node_ip = node_ips[i // int(gpus_per_node)]
                    required_by_node_ip[node_ip] = required_by_node_ip.get(node_ip, 0) + 1
                assert_node_ip_capacity(
                    required_gpus_by_node_ip=required_by_node_ip,
                    context=f"[MegatronWorkerGroup] node pinning base_model={self.base_model}",
                )
                bundles = build_node_affinity_gpu_bundles(
                    node_ips=node_ips,
                    gpus_per_node=gpus_per_node,
                    required_gpus=world_size,
                    cpu_per_gpu=1,
                )
                logger.info(f"[MegatronWorkerGroup] Model placement preferred nodes={node_ips}")
        self._placement_bundle_node_ips = [_bundle_node_ip(bundle) for bundle in bundles]
        self._placement_requested_node_ips = [ip for ip in self._placement_bundle_node_ips if ip is not None]
        logger.info(
            f"[MegatronWorkerGroup] Placement bundle node IPs={self._placement_bundle_node_ips}"
        )

        # PACK: try to colocate but allow multi-node for large models (K2: 16+ GPUs)
        # STRICT_PACK would require single node, blocking on 8-GPU nodes
        pg_name = _make_megatron_pg_name(self.base_model)
        self.placement_group = _get_or_create_megatron_placement_group(
            pg_name=pg_name,
            bundles=bundles,
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

        from ..config import actor_runtime_env_vars, otel_env_vars
        runtime_env = {
            "env_vars": actor_runtime_env_vars(
                pythonpath=PFS_PYTHONPATH,
                extra={
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
            ),
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
                resources=_node_affinity_resources(self._placement_bundle_node_ips[0]),
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
                resources=_node_affinity_resources(self._placement_bundle_node_ips[rank]),
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

    def debug_named_parameter(self, needle: str) -> list[dict]:
        """Inspect a parameter across all Megatron ranks."""
        return ray.get([w.debug_named_parameter.remote(needle) for w in self.workers])

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

    def _swap_session_on_workers(
        self,
        new_session_id: str,
        *,
        require_persisted_actor_only_state: bool = False,
    ) -> list[dict]:
        """Swap session state (gradients + optimizer) on all workers.

        This calls MegatronRankWorker.swap_session_state() on each worker to:
        1. Save outgoing session's gradients and optimizer state to PFS + RAM hot cache
        2. Restore incoming session's gradients and optimizer state from RAM or PFS

        Must be called during session switch to ensure optimizer momentum isolation.

        If any worker fails, sets ``_current_session = None`` to invalidate the
        group-level session cache and prevent the "already loaded" early-return
        from masking a split-state condition across ranks.

        Args:
            new_session_id: Session ID to switch to.
            require_persisted_actor_only_state: If True, workers must fail-loud when
                a persisted actor-only snapshot is expected but missing.

        Raises:
            Exception: Re-raises the first worker error after invalidating
                ``_current_session``.
        """
        logger.info(f"[MegatronWorkerGroup] Swapping session state on workers to {new_session_id}")
        futures = []
        for w in self.workers:
            if not hasattr(w, "swap_session_state"):
                continue
            remote = w.swap_session_state.remote
            if require_persisted_actor_only_state:
                try:
                    futures.append(
                        remote(
                            new_session_id,
                            require_persisted=require_persisted_actor_only_state,
                        )
                    )
                    continue
                except TypeError:
                    pass
            futures.append(remote(new_session_id))
        if not futures:
            return []
        try:
            results = ray.get(futures)
        except Exception as e:
            # Some workers may have swapped while others failed.
            # Invalidate _current_session so the next request cannot hit the
            # "already loaded" early-return (line 4974) and must re-evaluate.
            logger.error(
                "[MegatronWorkerGroup] Partial failure during session swap to %s "
                "(old_session=%s, workers=%d, error_type=%s, error=%r). "
                "Setting _current_session=None to force re-evaluation on next request.",
                new_session_id,
                self._current_session,
                len(self.workers),
                type(e).__name__,
                e,
                exc_info=True,
            )
            from .runtime_observability import runtime_observability

            runtime_observability.record_megatron_session_switch_failure(
                base_model=str(
                    getattr(self, "observability_base_model", getattr(self, "base_model", "unknown") or "unknown")
                ),
                reason="partial_swap",
            )
            self._current_session = None
            self._session_unknown_due_to_partial_swap = True
            raise
        logger.info("[MegatronWorkerGroup] Session state swapped on all workers")
        return results

    def _session_state_cached_on_workers(self, session_id: str) -> bool:
        """Whether every live worker still has actor-only state for this session in memory."""
        if not self.workers:
            return False
        try:
            states = ray.get([w.has_session_state_cached.remote(session_id) for w in self.workers])
        except Exception as e:
            logger.warning(
                "[MegatronWorkerGroup] Failed to query cached session state for %s: %s: %s",
                session_id,
                type(e).__name__,
                e,
            )
            return False
        return all(bool(state) for state in states)
    def _ensure_session_loaded(
        self,
        session_id: str | None,
        traceparent: str | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
        reload_optimizer_model_params: bool = True,
    ) -> dict[str, object]:
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
            self._last_session_switch_stats = None
            return {"switched": False}
        self._bind_traceparent(traceparent)

        if self._current_session == session_id:
            session_exists = self._session_manager.session_exists(session_id)
            if session_exists:
                has_actor_only_state = getattr(
                    self._session_manager,
                    "has_actor_only_state",
                    lambda _session_id: False,
                )(session_id)
                if has_actor_only_state and not self._session_state_cached_on_workers(session_id):
                    raise RuntimeError(
                        f"Session cache for {session_id} still has actor-only training state; "
                        "reload it from an explicit checkpoint before continuing."
                    )
                meta = getattr(self._session_manager, "get_metadata", lambda _session_id: None)(session_id)
                if not isinstance(meta, dict):
                    raise RuntimeError(
                        f"Session cache for {session_id} is missing session_metadata.json; "
                        "reload from an explicit checkpoint before continuing."
                    )
            logger.debug(f"[MegatronWorkerGroup] Session {session_id} already loaded")
            self._last_session_switch_stats = None
            return {"switched": False}

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

        has_actor_only_state = getattr(
            self._session_manager,
            "has_actor_only_state",
            lambda _session_id: False,
        )(session_id)
        has_persisted_actor_only_state = getattr(
            self._session_manager,
            "has_persisted_actor_only_state",
            lambda _session_id: False,
        )(session_id)
        session_exists = self._session_manager.session_exists(session_id)
        prevalidated_meta = None
        logger.info(f"[MegatronWorkerGroup] session_exists({session_id}) = {session_exists}")
        if self._current_session == session_id:
            if session_exists:
                meta = getattr(self._session_manager, "get_metadata", lambda _session_id: None)(session_id)
                if not isinstance(meta, dict):
                    raise RuntimeError(
                        f"Session cache for {session_id} is missing session_metadata.json; "
                        "reload from an explicit checkpoint before continuing."
                    )
            logger.debug(f"[MegatronWorkerGroup] Session {session_id} already loaded")
            return
        if has_actor_only_state:
            if not self._session_state_cached_on_workers(session_id):
                raise RuntimeError(
                    f"Session cache for {session_id} still has actor-only training state; "
                    "reload it from an explicit checkpoint before continuing."
                )
            logger.info(
                "[MegatronWorkerGroup] Session %s has actor-only marker but state is still cached in-memory on all workers; continuing with live session swap",
                session_id,
            )
        if session_exists:
            prevalidated_meta = getattr(self._session_manager, "get_metadata", lambda _session_id: None)(session_id)
            if not isinstance(prevalidated_meta, dict):
                raise RuntimeError(
                    f"Session cache for {session_id} is missing session_metadata.json; "
                    "reload from an explicit checkpoint before continuing."
                )

        outgoing_session_id = self._current_session
        # Save outgoing session's LoRA weights to disk
        t_save0 = time.perf_counter() if timing else 0.0
        if outgoing_session_id is not None:
            old_path = self._session_manager.get_session_path(outgoing_session_id)
            logger.info(f"[MegatronWorkerGroup] Saving outgoing session {outgoing_session_id}")
            self.save_adapter_state(old_path, traceparent=traceparent)
            # Save metadata (step count, learning rate, actual rank)
            self._session_manager.save_metadata(
                outgoing_session_id,
                self._step_count,
                self.learning_rate,
                self._actual_rank,
            )
        t_save1 = time.perf_counter() if timing else 0.0

        # Swap session state on workers (gradients + optimizer)
        # This saves outgoing session's gradients/optimizer to RAM + PFS,
        # and restores incoming session's state from RAM or PFS.
        t_swap0 = time.perf_counter() if timing else 0.0
        try:
            swap_results = self._swap_session_on_workers(
                session_id,
                require_persisted_actor_only_state=has_persisted_actor_only_state,
            )
        except TypeError:
            swap_results = self._swap_session_on_workers(session_id)
        if outgoing_session_id is not None:
            swap_results = [] if swap_results is None else swap_results
            if swap_results:
                persisted_entries = [
                    result.get("outgoing_persisted")
                    for result in swap_results
                    if isinstance(result, dict) and result.get("outgoing_persisted")
                ]
                if len(persisted_entries) != len(swap_results):
                    raise RuntimeError(
                        f"Failed to persist actor-only state for outgoing session {outgoing_session_id}: "
                        f"expected {len(swap_results)} rank snapshots, got {len(persisted_entries)}"
                    )
                save_persisted_actor_only_state = getattr(
                    self._session_manager,
                    "save_persisted_actor_only_state",
                    None,
                )
                if save_persisted_actor_only_state is not None:
                    save_persisted_actor_only_state(
                        outgoing_session_id,
                        actor_name=_make_megatron_actor_name(self.base_model),
                        worker_entries=persisted_entries,
                    )
                clear_actor_only_state = getattr(self._session_manager, "clear_actor_only_state", None)
                if clear_actor_only_state is not None:
                    clear_actor_only_state(outgoing_session_id)
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
                traceparent=traceparent,
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
                traceparent=traceparent,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
            )
            self._step_count = 0
            self._actual_rank = self.lora_rank
        t_load1 = time.perf_counter() if timing else 0.0

        # Reset expert_bias only for fresh sessions. Existing sessions may have
        # checkpointed expert_bias that should survive the switch.
        t_bias0 = time.perf_counter() if timing else 0.0
        if not session_exists:
            try:
                self.reset_expert_bias(traceparent=traceparent)
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
        self._session_unknown_due_to_partial_swap = False
        total_s = (time.perf_counter() - t0) if timing else max(0.0, (t_save1 - t_save0) + (t_swap1 - t_swap0) + (t_load1 - t_load0) + (t_bias1 - t_bias0))
        switch_stats = {
            "switched": True,
            "session_state": "existing" if session_exists else "new",
            "save_s": float(max(0.0, t_save1 - t_save0)),
            "swap_s": float(max(0.0, t_swap1 - t_swap0)),
            "load_s": float(max(0.0, t_load1 - t_load0)),
            "reset_bias_s": float(max(0.0, t_bias1 - t_bias0)),
            "total_s": float(max(0.0, total_s)),
        }
        self._last_session_switch_stats = switch_stats
        if timing:
            logger.info(
                f"[MegatronWorkerGroup] session_switch timing: "
                f"save_s={switch_stats['save_s']:.3f} "
                f"swap_s={switch_stats['swap_s']:.3f} "
                f"load_s={switch_stats['load_s']:.3f} "
                f"reset_bias_s={switch_stats['reset_bias_s']:.3f} "
                f"total_s={switch_stats['total_s']:.3f} "
                f"session_exists={session_exists}"
            )
        return switch_stats

    def _prepare_session_for_explicit_load(
        self,
        session_id: str | None,
        traceparent: str | None = None,
    ) -> None:
        """Prepare the actor for an explicit checkpoint load without trusting target cache."""
        if session_id is None:
            return
        self._bind_traceparent(traceparent)
        if self._current_session == session_id:
            return
        session_manager = getattr(self, "_session_manager", None)
        current_is_dirty = False
        target_exists = False
        target_has_actor_only_state = False
        if session_manager is not None:
            has_actor_only_state = getattr(session_manager, "has_actor_only_state", None)
            if callable(has_actor_only_state) and self._current_session is not None:
                current_is_dirty = bool(has_actor_only_state(self._current_session))
                target_has_actor_only_state = bool(has_actor_only_state(session_id))
            session_exists = getattr(session_manager, "session_exists", None)
            if callable(session_exists):
                target_exists = bool(session_exists(session_id))

        if self._current_session is not None and session_manager is not None and current_is_dirty:
            old_path = session_manager.get_session_path(self._current_session)
            logger.info(f"[MegatronWorkerGroup] Saving outgoing session {self._current_session}")
            self.save_adapter_state(old_path, traceparent=traceparent)
            save_metadata = getattr(session_manager, "save_metadata", None)
            if save_metadata is not None:
                save_metadata(
                    self._current_session,
                    self._step_count,
                    self.learning_rate,
                    self._actual_rank,
                )
        clear_refs = [
            w.clear_session_state.remote(session_id, traceparent=traceparent)
            for w in self.workers
            if hasattr(w, "clear_session_state")
        ]
        if clear_refs:
            ray.get(clear_refs)
        if session_manager is not None:
            clear_persisted_actor_only_state = getattr(
                session_manager,
                "clear_persisted_actor_only_state",
                None,
            )
            if clear_persisted_actor_only_state is not None:
                clear_persisted_actor_only_state(session_id)
            clear_actor_only_state = getattr(session_manager, "clear_actor_only_state", None)
            if clear_actor_only_state is not None:
                clear_actor_only_state(session_id)
        if target_exists and not target_has_actor_only_state:
            mark_refs = [
                w.mark_session_loaded.remote(session_id)
                for w in self.workers
                if hasattr(w, "mark_session_loaded")
            ]
            if mark_refs:
                ray.get(mark_refs)
        else:
            self._swap_session_on_workers(session_id)
        self._current_session = session_id
        self._session_unknown_due_to_partial_swap = False

    def _resolve_required_session_id(self, session_id: str | None, *, op: str) -> str:
        """Resolve session_id with fail-closed behavior for unknown group state."""
        if session_id is not None and session_id.strip() == "":
            raise ValueError(f"session_id must be non-empty when provided (op={op})")
        effective_session_id = session_id if session_id is not None else self._current_session
        if effective_session_id is not None and effective_session_id.strip() == "":
            raise ValueError(
                f"resolved session_id is empty; refusing to run (op={op})"
            )
        if effective_session_id is None:
            if getattr(self, "_session_unknown_due_to_partial_swap", False):
                raise RuntimeError(
                    "session state unknown after swap failure; explicit session_id required "
                    f"(op={op})"
                )
            raise ValueError(
                f"no session loaded; explicit session_id required (op={op})"
            )
        return effective_session_id

    def _invalidate_session_durability(
        self,
        session_id: str | None,
        *,
        reason: str,
        preserve_existing_reason: bool = False,
    ) -> dict | None:
        if session_id is None:
            return None
        session_manager = getattr(self, "_session_manager", None)
        if session_manager is None:
            return None
        if preserve_existing_reason:
            get_external_checkpoint = getattr(session_manager, "get_external_checkpoint", None)
            if get_external_checkpoint is not None:
                marker = get_external_checkpoint(session_id)
                if isinstance(marker, dict) and not bool(marker.get("is_fresh", False)):
                    return marker
        invalidate_external_checkpoint = getattr(
            session_manager,
            "invalidate_external_checkpoint",
            None,
        )
        if invalidate_external_checkpoint is None:
            return None
        return invalidate_external_checkpoint(session_id, reason=reason)

    def _ensure_session_for_request(
        self,
        *,
        op: str,
        session_id: str | None,
        traceparent: str | None,
        train_attn: bool | None,
        train_mlp: bool | None,
        train_unembed: bool | None,
        reload_optimizer_model_params: bool = True,
    ) -> tuple[str, dict[str, object]]:
        """Resolve and restore session state for forward/backward/step style requests."""
        effective_session_id = self._resolve_required_session_id(session_id, op=op)
        ensure_kwargs = {
            "traceparent": traceparent,
            "train_attn": train_attn,
            "train_mlp": train_mlp,
            "train_unembed": train_unembed,
        }
        if not reload_optimizer_model_params:
            ensure_kwargs["reload_optimizer_model_params"] = False
        switch_stats = self._ensure_session_loaded(
            effective_session_id,
            **ensure_kwargs,
        )
        if not isinstance(switch_stats, dict):
            switch_stats = dict(getattr(self, "_last_session_switch_stats", None) or {})
        return effective_session_id, switch_stats

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
        rollout_correction_config: dict | None = None,
        session_id: str | None = None,
        reset_bias: bool | None = None,
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
        loss_fn_config = loss_fn_config or {}

        timing = _env_flag("MINT_TIMING_DIAG", default=False)
        t0 = time.perf_counter() if timing else 0.0

        # Resolve once for watchdog context, then use the shared recovery path.
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="forward_backward",
        )
        watchdog = self._start_slow_group_watchdog(
            op="forward_backward",
            session_id=effective_session_id,
            extra=f"items={len(data_items)} loss_fn={loss_fn}",
        )
        try:
            effective_session_id, switch_stats = self._ensure_session_for_request(
                op="forward_backward",
                session_id=effective_session_id,
                traceparent=traceparent,
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
                    data_items,
                    loss_fn,
                    loss_fn_config,
                    rollout_correction_config,
                    effective_session_id,
                    reset_bias,
                    traceparent=traceparent,
                )
                for w in self.workers
            ]
            results = self._ray_get_group_results(
                futures,
                op="forward_backward",
                session_id=effective_session_id,
            )
            t3 = time.perf_counter() if timing else 0.0
        finally:
            self._stop_slow_group_watchdog(watchdog)
        self._session_manager.mark_actor_only_state(
            effective_session_id,
            reason="forward_backward",
            actor_name=_make_megatron_actor_name(self.base_model),
        )
        invalidate_external_checkpoint = getattr(
            self._session_manager,
            "invalidate_external_checkpoint",
            None,
        )
        if invalidate_external_checkpoint is not None:
            invalidate_external_checkpoint(
                effective_session_id,
                reason="forward_backward",
            )
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
        metrics["session_switch_total:sum"] = 1.0 if switch_stats.get("switched") else 0.0
        metrics["session_switch_save_s:sum"] = float(switch_stats.get("save_s", 0.0))
        metrics["session_switch_swap_s:sum"] = float(switch_stats.get("swap_s", 0.0))
        metrics["session_switch_load_s:sum"] = float(switch_stats.get("load_s", 0.0))
        metrics["session_switch_reset_bias_s:sum"] = float(switch_stats.get("reset_bias_s", 0.0))
        metrics["session_switch_total_s:sum"] = float(switch_stats.get("total_s", 0.0))
        metrics["session_switch_existing_session:mean"] = 1.0 if switch_stats.get("session_state") == "existing" else 0.0

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
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="train_step",
        )

        fb_result = self.forward_backward(
            data_items=data_items,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            rollout_correction_config=rollout_correction_config,
            session_id=effective_session_id,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

        lr = learning_rate if learning_rate is not None else self.learning_rate
        opt_result = self.optim_step(
            lr,
            session_id=effective_session_id,
            traceparent=traceparent,
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
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
        effective_session_id, switch_stats = self._ensure_session_for_request(
            op="forward",
            session_id=session_id,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
            reload_optimizer_model_params=False,
        )

        # Send raw data_items to workers (TensorDict created locally on each worker
        # to avoid Ray serialization issues with nested tensors)
        futures = [w.forward.remote(data_items, reset_bias, traceparent=traceparent) for w in self.workers]
        results = self._ray_get_group_results(
            futures,
            op="forward",
            session_id=effective_session_id,
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
            "session_switch_total:sum": 1.0 if switch_stats.get("switched") else 0.0,
            "session_switch_save_s:sum": float(switch_stats.get("save_s", 0.0)),
            "session_switch_swap_s:sum": float(switch_stats.get("swap_s", 0.0)),
            "session_switch_load_s:sum": float(switch_stats.get("load_s", 0.0)),
            "session_switch_reset_bias_s:sum": float(switch_stats.get("reset_bias_s", 0.0)),
            "session_switch_total_s:sum": float(switch_stats.get("total_s", 0.0)),
            "session_switch_existing_session:mean": 1.0 if switch_stats.get("session_state") == "existing" else 0.0,
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

    def forward_backward_reverse_kl(
        self,
        data_items: list[dict],
        reference_checkpoint_path: str | None,
        temperature: float,
        session_id: str | None = None,
        traceparent: str | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
        reference_full_log_prob_chunks: list | None = None,
    ) -> dict:
        """Run Mint reverse-KL forward/backward on all workers."""
        self._bind_traceparent(traceparent)
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="forward_backward_reverse_kl",
        )
        self._ensure_session_loaded(
            effective_session_id,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
            reload_optimizer_model_params=False,
        )

        futures = [
            w.forward_backward_reverse_kl.remote(
                data_items,
                reference_checkpoint_path,
                None,
                temperature,
                effective_session_id,
                traceparent=traceparent,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
                reference_full_log_prob_chunks=(
                    reference_full_log_prob_chunks[idx]
                    if isinstance(reference_full_log_prob_chunks, list)
                    and len(reference_full_log_prob_chunks) == len(self.workers)
                    else reference_full_log_prob_chunks
                ),
            )
            for idx, w in enumerate(self.workers)
        ]
        results = ray.get(futures)
        rank0_result = next((r for r in results if isinstance(r, dict) and r), {})
        if not rank0_result:
            raise ValueError("reverse_kl produced no rank0 result")
        return rank0_result

    def forward_reference_full_log_probs(
        self,
        data_items: list[dict],
        temperature: float,
        session_id: str | None = None,
        traceparent: str | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> list:
        self._bind_traceparent(traceparent)
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="forward_reference_full_log_probs",
        )
        self._ensure_session_loaded(
            effective_session_id,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
            reload_optimizer_model_params=False,
        )
        futures = [
            w.forward_reference_full_log_probs.remote(
                data_items,
                temperature,
                traceparent=traceparent,
            )
            for w in self.workers
        ]
        results = ray.get(futures)
        chunks_by_rank = []
        for idx, result in enumerate(results):
            chunks = result.get("reference_local_log_probs") if isinstance(result, dict) else None
            if not isinstance(chunks, list):
                raise ValueError(f"reference_local_log_probs missing from worker index {idx}")
            chunks_by_rank.append(chunks)
        return chunks_by_rank

    def debug_lora_storage(
        self,
        session_id: str | None = None,
        traceparent: str | None = None,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        self._bind_traceparent(traceparent)
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="debug_lora_storage",
        )
        self._ensure_session_loaded(
            effective_session_id,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )
        results = ray.get([w.debug_lora_storage.remote(traceparent=traceparent) for w in self.workers])
        return {"results": results}

    def optim_step(
        self,
        learning_rate: float,
        session_id: str | None = None,
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
        timing = _env_flag("MINT_TIMING_DIAG", default=False)
        t0 = time.perf_counter() if timing else 0.0

        # optim_step must re-enter the ordinary session-load path.
        # If another session interleaved after forward_backward, restore both
        # adapter weights and in-memory optimizer/gradient state first.
        effective_session_id, switch_stats = self._ensure_session_for_request(
            op="optim_step",
            session_id=session_id,
            traceparent=traceparent,
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
                traceparent=traceparent,
            )
            for w in self.workers
        ]
        results = self._ray_get_group_results(
            futures,
            op="optim_step",
            session_id=effective_session_id,
        )
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
        self._session_manager.mark_actor_only_state(
            effective_session_id,
            reason="optim_step",
            actor_name=_make_megatron_actor_name(self.base_model),
        )
        invalidate_external_checkpoint = getattr(
            self._session_manager,
            "invalidate_external_checkpoint",
            None,
        )
        if invalidate_external_checkpoint is not None:
            invalidate_external_checkpoint(
                effective_session_id,
                reason="optim_step",
            )

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
                "session_switch_total:sum": 1.0 if switch_stats.get("switched") else 0.0,
                "session_switch_save_s:sum": float(switch_stats.get("save_s", 0.0)),
                "session_switch_swap_s:sum": float(switch_stats.get("swap_s", 0.0)),
                "session_switch_load_s:sum": float(switch_stats.get("load_s", 0.0)),
                "session_switch_reset_bias_s:sum": float(switch_stats.get("reset_bias_s", 0.0)),
                "session_switch_total_s:sum": float(switch_stats.get("total_s", 0.0)),
                "session_switch_existing_session:mean": 1.0 if switch_stats.get("session_state") == "existing" else 0.0,
            }
        }

    def get_lora_state_dict(
        self,
        *,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Get LoRA state dict from all workers (rank 0 returns data, others empty).

        IMPORTANT: Must call ALL workers in parallel because bridge.export_weights()
        may use NCCL collectives internally. Calling only rank 0 would deadlock.
        """
        logger.info(
            f"[MegatronWorkerGroup] get_lora_state_dict: ENTRY "
            f"(train_attn={train_attn}, train_mlp={train_mlp}, train_unembed={train_unembed})"
        )
        logger.info(f"[MegatronWorkerGroup] Calling get_lora_state_dict.remote() on all {len(self.workers)} workers...")

        try:
            # Call ALL workers - bridge.export_weights() may use NCCL allgather
            # Rank 0 returns actual data, other ranks return empty dict
            futures = [
                w.get_lora_state_dict.remote(
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                )
                for w in self.workers
            ]
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

    def reset_expert_bias(self, traceparent: str | None = None) -> dict:
        """Reset expert_bias buffers to zero in all MoE router modules across all workers.

        The expert_bias buffer accumulates during training to balance token distribution
        across experts. However, this buffer is NOT exported with LoRA weights, causing
        train-inference mismatch when comparing Megatron logprobs with vLLM.

        Call this before computing logprobs to ensure consistent behavior with vLLM.

        Returns:
            dict with reset count from rank 0
        """
        self._bind_traceparent(traceparent)
        logger.info("[MegatronWorkerGroup] reset_expert_bias: ENTRY")

        try:
            # Call ALL workers to ensure distributed consistency
            futures = [w.reset_expert_bias.remote(traceparent=traceparent) for w in self.workers]
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
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
        # Issue #44: Save current session's weights before reinitializing
        if self._current_session is not None and new_session_id is not None:
            old_path = self._session_manager.get_session_path(self._current_session)
            logger.info(f"[MegatronWorkerGroup] reinit_lora_weights: saving current session {self._current_session} to {old_path}")
            try:
                self.save_adapter_state(old_path, traceparent=traceparent)
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
        self._invalidate_session_durability(
            self._current_session,
            reason="reinit_lora_weights",
            preserve_existing_reason=True,
        )

        # Call reinit on all workers (required for distributed sync)
        futures = [
            w.reinit_lora_weights.remote(
                learning_rate,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
                traceparent=traceparent,
            )
            for w in self.workers
        ]
        results = self._ray_get_group_results(
            futures,
            op="reinit_lora_weights",
            session_id=self._current_session,
        )

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

    def load_checkpoint(
        self,
        load_path: str,
        load_optimizer: bool = True,
        traceparent: str | None = None,
        *,
        session_id: str | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Load checkpoint from path.

        Delegates to load_adapter_state which handles distributed loading.

        Args:
            load_path: Path to checkpoint directory.
            load_optimizer: Whether to restore optimizer state.
            session_id: Target session to materialize before loading.

        Returns:
            Dict with load metadata.
        """
        self._bind_traceparent(traceparent)
        import os

        # load_checkpoint resolves session id with the same fail-closed rules,
        # but materializes state via _prepare_session_for_explicit_load() later.
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="load_checkpoint",
        )
        if not os.path.isdir(load_path):
            raise FileNotFoundError(
                f"Checkpoint path does not exist or is not a directory: {load_path}"
            )
        files = os.listdir(load_path)
        logger.info(f"[MegatronWorkerGroup] load_checkpoint: found {len(files)} files: {files[:10]}")
        adapter_files = [f for f in files if f.endswith("_adapter.pt")]
        if not adapter_files:
            raise FileNotFoundError(
                f"Missing distributed adapter shards (mp_rank_*_adapter.pt) in: {load_path}"
            )
        if load_optimizer:
            opt_presence = ray.get(
                [w.check_optimizer_state_exists.remote(load_path, traceparent=traceparent) for w in self.workers]
            )
            missing = [
                item.get("optimizer_file", "<unknown>")
                for item in opt_presence
                if not isinstance(item, dict) or not bool(item.get("exists"))
            ]
            if missing:
                raise FileNotFoundError(
                    "Optimizer restore requested, but optimizer shard(s) not found: "
                    + ", ".join(missing)
                )
        self._prepare_session_for_explicit_load(
            effective_session_id,
            traceparent=traceparent,
        )

        logger.info(f"[MegatronWorkerGroup] load_checkpoint: path={load_path}, load_optimizer={load_optimizer}")
        logger.info(f"[MegatronWorkerGroup] load_checkpoint: found {len(adapter_files)} adapter files")

        adapter_config_path = os.path.join(load_path, "adapter_config.json")
        if not os.path.isfile(adapter_config_path):
            raise FileNotFoundError(
                f"Missing adapter_config.json required to recover actual LoRA rank: {adapter_config_path}"
            )
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        if not isinstance(adapter_config, dict):
            raise RuntimeError(
                f"Invalid adapter_config.json type {type(adapter_config).__name__} in {adapter_config_path}"
            )
        checkpoint_rank = adapter_config.get("r")
        if not isinstance(checkpoint_rank, int) or isinstance(checkpoint_rank, bool) or checkpoint_rank <= 0:
            raise RuntimeError(
                f"Invalid adapter rank in {adapter_config_path}: expected positive int, got {checkpoint_rank!r}"
            )

        if not load_optimizer:
            self._invalidate_session_durability(
                effective_session_id,
                reason="load_checkpoint_without_optimizer",
            )
        # Delegate to load_adapter_state
        result = self.load_adapter_state(
            load_path,
            actual_rank=checkpoint_rank,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
            reload_optimizer_model_params=load_optimizer,
        )
        result["load_method"] = "load_adapter_state"

        if load_optimizer:
            opt_results = ray.get(
                [w.load_optimizer_state.remote(load_path, traceparent=traceparent) for w in self.workers]
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

        result["checkpoint_path"] = load_path
        result["train_attn"] = True if train_attn is None else bool(train_attn)
        result["train_mlp"] = True if train_mlp is None else bool(train_mlp)
        result["train_unembed"] = True if train_unembed is None else bool(train_unembed)

        meta_path = os.path.join(load_path, "training_meta.json")
        checkpoint_lr = self.learning_rate
        checkpoint_step = self._step_count
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                loaded_meta = json.load(f)
            if isinstance(loaded_meta, dict):
                meta = loaded_meta
                result.update(meta)
                if "current_step" in meta:
                    meta_step = meta["current_step"]
                    if isinstance(meta_step, int) and not isinstance(meta_step, bool):
                        checkpoint_step = meta_step
                    else:
                        logger.warning(
                            "[MegatronWorkerGroup] Invalid current_step type=%s value=%r in %s; "
                            "preserving step=%s",
                            type(meta_step).__name__,
                            meta_step,
                            meta_path,
                            checkpoint_step,
                        )
                checkpoint_lr_value = meta.get("learning_rate", checkpoint_lr)
                try:
                    checkpoint_lr = float(checkpoint_lr_value)
                except Exception:
                    logger.warning(
                        "[MegatronWorkerGroup] Invalid learning_rate value=%r in %s; "
                        "preserving lr=%s",
                        checkpoint_lr_value,
                        meta_path,
                        checkpoint_lr,
                    )
            else:
                logger.warning(
                    "[MegatronWorkerGroup] Invalid checkpoint metadata type %s in %s; "
                    "preserving step/lr state",
                    type(loaded_meta).__name__,
                    meta_path,
                )

        session_manager = getattr(self, "_session_manager", None)
        if not load_optimizer:
            result["actor_only_state_dirty"] = False
            clear_refs = [
                w.clear_session_state.remote(effective_session_id, traceparent=traceparent)
                for w in self.workers
                if hasattr(w, "clear_session_state")
            ]
            if clear_refs:
                ray.get(clear_refs)
            if session_manager is not None:
                clear_persisted_actor_only_state = getattr(
                    session_manager,
                    "clear_persisted_actor_only_state",
                    None,
                )
                if clear_persisted_actor_only_state is not None:
                    clear_persisted_actor_only_state(effective_session_id)
                clear_actor_only_state = getattr(session_manager, "clear_actor_only_state", None)
                if clear_actor_only_state is not None:
                    clear_actor_only_state(effective_session_id)
            self.reset_optimizer(checkpoint_lr, traceparent=traceparent, zero_grad_buffers=False)
            result["optimizer_reset"] = True
        else:
            result["optimizer_reset"] = False
            result["actor_only_state_dirty"] = True
            if session_manager is not None:
                clear_persisted_actor_only_state = getattr(
                    session_manager,
                    "clear_persisted_actor_only_state",
                    None,
                )
                if clear_persisted_actor_only_state is not None:
                    clear_persisted_actor_only_state(effective_session_id)
                session_manager.mark_actor_only_state(
                    effective_session_id,
                    reason="load_weights",
                    actor_name=_make_megatron_actor_name(self.base_model),
                )

        self._step_count = checkpoint_step
        self.learning_rate = checkpoint_lr
        if load_optimizer:
            mark_external_checkpoint = getattr(
                session_manager,
                "mark_external_checkpoint",
                None,
            )
            if mark_external_checkpoint is not None:
                mark_external_checkpoint(
                    effective_session_id,
                    checkpoint_path=load_path,
                    reason="load_checkpoint",
                    actor_name=_make_megatron_actor_name(self.base_model),
                )

        return result


    def save_checkpoint(
        self,
        save_path: str,
        traceparent: str | None = None,
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
        Returns:
            Dict with training metadata (from rank 0).
        """
        self._bind_traceparent(traceparent)
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="save_checkpoint",
        )
        self._ensure_session_loaded(
            effective_session_id,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

        logger.info(
            f"[MegatronWorkerGroup] save_checkpoint: {save_path} "
            f"(session_id={effective_session_id}, actual_rank={self._actual_rank})"
        )
        # Call ALL workers - get_lora_state_dict uses NCCL allgather
        # Rank 0 saves to disk, other ranks participate in collectives then return empty
        futures = [
            w.save_checkpoint.remote(
                save_path,
                self._step_count,
                self._actual_rank,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
                traceparent=traceparent,
            )
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
        timeout_s = int(os.environ.get("MINT_MEGATRON_SAVE_CHECKPOINT_TIMEOUT_S", str(default_timeout_s)))
        results = self._ray_get_group_results(
            futures,
            op="save_checkpoint",
            session_id=effective_session_id,
            timeout_s=timeout_s,
        )
        result = results[0]  # Only rank 0 returns actual data
        session_manager = getattr(self, "_session_manager", None)
        mark_external_checkpoint = getattr(
            session_manager,
            "mark_external_checkpoint",
            None,
        )
        if mark_external_checkpoint is not None:
            mark_external_checkpoint(
                effective_session_id,
                checkpoint_path=save_path,
                reason="save_checkpoint",
                actor_name=_make_megatron_actor_name(self.base_model),
            )
        if session_manager is not None:
            prime_session = getattr(session_manager, "prime_session", None)
            if prime_session is not None:
                prime_session(
                    effective_session_id,
                    save_path,
                    step=self._step_count,
                    lr=self.learning_rate,
                    actual_rank=self._actual_rank,
                    optimizer_restored=True,
                    train_attn=True if train_attn is None else bool(train_attn),
                    train_mlp=True if train_mlp is None else bool(train_mlp),
                    train_unembed=True if train_unembed is None else bool(train_unembed),
                )
            clear_actor_only_state = getattr(session_manager, "clear_actor_only_state", None)
            if clear_actor_only_state is not None:
                clear_actor_only_state(effective_session_id)
        logger.info(f"[MegatronWorkerGroup] save_checkpoint: completed, step={result.get('current_step', 'unknown')}")
        return result

    def save_lora_weights(
        self,
        save_path: str,
        traceparent: str | None = None,
        *,
        session_id: str | None = None,
        train_attn: bool | None = None,
        train_mlp: bool | None = None,
        train_unembed: bool | None = None,
    ) -> dict:
        """Save PEFT LoRA weights for sampling (no optimizer/resume artifacts).

        Must call ALL workers because get_lora_state_dict uses NCCL collectives.
        """
        self._bind_traceparent(traceparent)
        effective_session_id = self._resolve_required_session_id(
            session_id,
            op="save_lora_weights",
        )
        self._ensure_session_loaded(
            effective_session_id,
            traceparent=traceparent,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )

        logger.info(
            f"[MegatronWorkerGroup] save_lora_weights: {save_path} "
            f"(session_id={effective_session_id}, actual_rank={self._actual_rank})"
        )
        futures = [
            w.save_lora_weights.remote(
                save_path,
                self._step_count,
                self._actual_rank,
                train_attn=train_attn,
                train_mlp=train_mlp,
                train_unembed=train_unembed,
                traceparent=traceparent,
            )
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
        results = self._ray_get_group_results(
            futures,
            op="save_lora_weights",
            session_id=effective_session_id,
            timeout_s=timeout_s,
        )
        result = results[0]
        logger.info(f"[MegatronWorkerGroup] save_lora_weights: completed, step={result.get('current_step', 'unknown')}")
        return result

    def _get_session_cache_store_diagnostics(self) -> dict:
        actor_name = _make_megatron_actor_name(self.base_model)
        get_cache_usage = getattr(self._session_manager, "get_cache_usage", None)
        if get_cache_usage is None:
            return {
                "global": {},
                "actor": {},
            }
        return {
            "global": get_cache_usage(),
            "actor": get_cache_usage(actor_name=actor_name),
        }

    def _get_session_cache_diagnostics(self) -> dict:
        hot_infos: list[dict] = []
        if self.workers:
            try:
                hot_infos = ray.get([w.get_hot_cache_info.remote() for w in self.workers])
            except Exception as e:
                logger.warning(
                    "[MegatronWorkerGroup] Failed to query hot cache diagnostics: %s: %s",
                    type(e).__name__,
                    e,
                )
                hot_infos = []
        worker_count = len(hot_infos)
        hot_membership_counts: dict[str, int] = {}
        for info in hot_infos:
            if not isinstance(info, dict):
                continue
            for session_id in info.get("hot_sessions", []):
                hot_membership_counts[session_id] = hot_membership_counts.get(session_id, 0) + 1
        fully_hot_session_ids = {
            session_id
            for session_id, count in hot_membership_counts.items()
            if worker_count > 0 and count == worker_count
        }
        partially_hot_session_ids = {
            session_id
            for session_id, count in hot_membership_counts.items()
            if 0 < count < worker_count
        }
        hot_bytes = sum(int(info.get("hot_bytes", 0)) for info in hot_infos if isinstance(info, dict))
        last_eviction_reason = next(
            (
                info.get("last_eviction_reason")
                for info in hot_infos
                if isinstance(info, dict) and info.get("last_eviction_reason")
            ),
            None,
        )
        actor_name = _make_megatron_actor_name(self.base_model)
        persisted = getattr(
            self._session_manager,
            "list_persisted_actor_only_state",
            lambda _actor_name=None: {},
        )(actor_name)
        current_session = self._current_session
        cold_session_ids = [
            session_id
            for session_id in sorted(persisted)
            if session_id not in hot_membership_counts and session_id != current_session
        ]
        cold_bytes = sum(int(persisted[session_id].get("total_bytes", 0)) for session_id in cold_session_ids)
        return {
            "hot_session_count": len(fully_hot_session_ids),
            "partially_hot_session_count": len(partially_hot_session_ids),
            "cold_session_count": len(cold_session_ids),
            "hot_bytes": hot_bytes,
            "cold_bytes": cold_bytes,
            "last_eviction_reason": last_eviction_reason,
            "hot_sessions": sorted(fully_hot_session_ids),
            "partially_hot_sessions": sorted(partially_hot_session_ids),
            "cold_sessions": cold_session_ids,
        }

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
            "session_cache": self._get_session_cache_diagnostics(),
            "session_cache_store": self._get_session_cache_store_diagnostics(),
            "placement_bundle_node_ips": list(self._placement_bundle_node_ips),
            "placement_requested_node_ips": list(dict.fromkeys(self._placement_requested_node_ips)),
        }

    def get_observability_binding(self) -> dict[str, object]:
        bindings: list[dict[str, object]] = []
        allocated_bytes = 0
        reserved_bytes = 0
        fragmentation_bytes = 0
        saw_memory = False
        for worker in self.workers:
            try:
                binding = ray.get(worker.get_observability_binding.remote(), timeout=5)
            except Exception:
                continue
            if not isinstance(binding, dict):
                continue
            gpu_indices = binding.get("gpu_indices")
            if isinstance(gpu_indices, list):
                for gpu_index in gpu_indices:
                    bindings.append(
                        {
                            "hostname": binding.get("hostname"),
                            "node_id": binding.get("node_id"),
                            "gpu_index": int(gpu_index),
                            "rank": binding.get("rank"),
                        }
                    )
            for field_name, total_name in (
                ("gpu_memory_allocated_bytes", "allocated"),
                ("gpu_memory_reserved_bytes", "reserved"),
                ("gpu_memory_fragmentation_bytes", "fragmentation"),
            ):
                value = binding.get(field_name)
                if not isinstance(value, (int, float)):
                    continue
                saw_memory = True
                if total_name == "allocated":
                    allocated_bytes += max(0, int(value))
                elif total_name == "reserved":
                    reserved_bytes += max(0, int(value))
                else:
                    fragmentation_bytes += max(0, int(value))
        out: dict[str, object] = {
            "gpu_bindings": bindings,
            "active_sessions": int(self._current_session is not None),
            "session_unknown": int(bool(self._session_unknown_due_to_partial_swap)),
            "session_step": max(0, int(self._step_count)),
            "learning_rate": max(0.0, float(self.learning_rate)),
        }
        if saw_memory:
            out.update(
                {
                    "gpu_memory_allocated_bytes": allocated_bytes,
                    "gpu_memory_reserved_bytes": reserved_bytes,
                    "gpu_memory_fragmentation_bytes": fragmentation_bytes,
                }
            )
        return out


    # ========================================================================
    # Phase 6: Multi-Session Support Methods
    # ========================================================================

    def load_adapter_state(
        self,
        checkpoint_path: str,
        actual_rank: int | None = None,
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
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
                traceparent=traceparent,
            )
            for w in self.workers
        ]
        results = ray.get(futures)
        result = results[0]  # Rank 0 result
        self._actual_rank = actual_rank if actual_rank is not None else self.lora_rank
        logger.info(f"[MegatronWorkerGroup] Adapter state loaded: {result}")
        return result

    def save_adapter_state(
        self,
        checkpoint_path: str,
        actual_rank: int | None = None,
        traceparent: str | None = None,
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
        self._bind_traceparent(traceparent)
        effective_rank = actual_rank or self._actual_rank or self.lora_rank
        logger.info(
            f"[MegatronWorkerGroup] Saving adapter state to {checkpoint_path} "
            f"(actual_rank={effective_rank}, trainer_rank={self.lora_rank})"
        )
        futures = [
            w.save_adapter_state.remote(
                checkpoint_path,
                actual_rank=effective_rank,
                trainer_rank=self.lora_rank,
                traceparent=traceparent,
            )
            for w in self.workers
        ]
        results = ray.get(futures)
        result = results[0]  # Rank 0 result
        logger.info(f"[MegatronWorkerGroup] Adapter state saved: {result}")
        return result

    def reset_optimizer(
        self,
        learning_rate: float | None = None,
        traceparent: str | None = None,
        *,
        zero_grad_buffers: bool = True,
    ) -> dict:
        """Reset optimizer state on all workers.

        Used for new sessions to start fresh without prior momentum.

        Args:
            learning_rate: Optional new learning rate.

        Returns:
            Dict with status info from rank 0.
        """
        self._bind_traceparent(traceparent)
        logger.info(f"[MegatronWorkerGroup] Resetting optimizer (lr={learning_rate})")
        self._invalidate_session_durability(
            self._current_session,
            reason="reset_optimizer",
            preserve_existing_reason=True,
        )
        futures = [
            w.reset_optimizer.remote(
                learning_rate,
                traceparent=traceparent,
                zero_grad_buffers=zero_grad_buffers,
            )
            for w in self.workers
        ]
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
        self._session_unknown_due_to_partial_swap = False
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

    def mark_session_loaded(
        self,
        session_id: str,
        *,
        step_count: int,
        learning_rate: float,
        actual_rank: int | None = None,
        actor_only_state_dirty: bool = False,
        checkpoint_path: str | None = None,
        optimizer_restored: bool = True,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = True,
    ) -> dict:
        """Record that a checkpoint-loaded session is the current active session."""
        ray.get([w.mark_session_loaded.remote(session_id) for w in self.workers])
        self._current_session = session_id
        self._step_count = int(step_count)
        self.learning_rate = float(learning_rate)
        self._actual_rank = actual_rank if actual_rank is not None else self.lora_rank
        session_manager = getattr(self, "_session_manager", None)
        if session_manager is not None:
            if checkpoint_path is not None:
                prime_session = getattr(session_manager, "prime_session", None)
                if prime_session is not None:
                    prime_session(
                        session_id,
                        checkpoint_path,
                        step=self._step_count,
                        lr=self.learning_rate,
                        actual_rank=self._actual_rank,
                        optimizer_restored=optimizer_restored,
                        train_attn=train_attn,
                        train_mlp=train_mlp,
                        train_unembed=train_unembed,
                    )
                else:
                    session_manager.save_metadata(
                        session_id,
                        self._step_count,
                        self.learning_rate,
                        self._actual_rank,
                    )
            else:
                session_manager.save_metadata(
                    session_id,
                    self._step_count,
                    self.learning_rate,
                    self._actual_rank,
                )
            clear_persisted_actor_only_state = getattr(
                session_manager,
                "clear_persisted_actor_only_state",
                None,
            )
            if clear_persisted_actor_only_state is not None:
                clear_persisted_actor_only_state(session_id)
            if actor_only_state_dirty:
                session_manager.mark_actor_only_state(
                    session_id,
                    reason="load_weights",
                    actor_name=_make_megatron_actor_name(self.base_model),
                )
            else:
                clear_actor_only_state = getattr(session_manager, "clear_actor_only_state", None)
                if clear_actor_only_state is not None:
                    clear_actor_only_state(session_id)
        logger.info(
            f"[MegatronWorkerGroup] Marked loaded session active: {session_id} "
            f"(step={self._step_count}, actual_rank={self._actual_rank})"
        )
        return {"status": "ok", "session_id": session_id}

    def prime_session_checkpoint(
        self,
        session_id: str,
        checkpoint_path: str,
        *,
        step_count: int,
        learning_rate: float,
        actual_rank: int | None = None,
        optimizer_restored: bool = True,
    ) -> dict:
        session_path = self._session_manager.prime_session(
            session_id,
            checkpoint_path,
            step=int(step_count),
            lr=float(learning_rate),
            actual_rank=actual_rank,
            optimizer_restored=optimizer_restored,
        )
        logger.info(
            f"[MegatronWorkerGroup] Primed session {session_id} from {checkpoint_path} "
            f"into {session_path} (actual_rank={actual_rank})"
        )
        return {
            "status": "ok",
            "session_id": session_id,
            "session_path": session_path,
            "actual_rank": actual_rank,
        }

    def delete_session(
        self,
        session_id: str,
        *,
        traceparent: str | None = None,
    ) -> dict:
        self._bind_traceparent(traceparent)
        ray.get([w.clear_session_state.remote(session_id, traceparent=traceparent) for w in self.workers])
        deleted = self._session_manager.delete_session(session_id)
        if self._current_session == session_id:
            self._current_session = None
            self._session_unknown_due_to_partial_swap = False
        return {"status": "ok", "session_id": session_id, "deleted": bool(deleted)}

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
            "session_cache": self._get_session_cache_diagnostics(),
            "session_cache_store": self._get_session_cache_store_diagnostics(),
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
    observability_base_model: str | None = None,
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
    from tinker_server.backend.resource_pool import (
        ActorType,
        actor_observability_metadata,
        get_resource_pool,
    )
    from .model_registry import is_persistent_model

    config = distributed_config or DistributedConfig()
    num_gpus = config.world_size
    is_persistent = is_persistent_model(base_model)
    observability_model = str(observability_base_model or base_model or "unknown")

    if not ray.is_initialized():
        init_ray(
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
                from .runtime_observability import runtime_observability

                runtime_observability.record_megatron_actor_lifecycle(
                    base_model=observability_model,
                    event="recreate",
                )
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
                base_model=observability_model,
                protected=is_persistent,
                metadata=dict(actor_observability_metadata(actor) or {}),
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
        pg_name = _make_megatron_pg_name(base_model)
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
            from ..config import actor_runtime_env_vars, otel_env_vars

            # Runtime env for PFS code access
            runtime_env = {
                "env_vars": actor_runtime_env_vars(
                    pythonpath=PFS_PYTHONPATH,
                    extra={
                    "HF_HOME": "/vePFS-Mindverse/share/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # Reduce memory fragmentation
                    **otel_env_vars(),
                    },
                )
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
            for k in (
                "CUDA_LAUNCH_BLOCKING",
                "TORCH_DISTRIBUTED_DEBUG",
                "NCCL_DEBUG",
                "NCCL_DEBUG_SUBSYS",
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
            megatron_node_pin_json = os.environ.get("MINT_MEGATRON_MODEL_NODE_IPS_JSON")
            if explicit_node_ips_csv:
                runtime_env["env_vars"]["MINT_MEGATRON_NODE_IPS_CSV"] = explicit_node_ips_csv
            elif not megatron_node_pin_json:
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
            if megatron_node_pin_json:
                runtime_env["env_vars"]["MINT_MEGATRON_MODEL_NODE_IPS_JSON"] = megatron_node_pin_json

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
                    observability_base_model=observability_model,
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
                base_model=observability_model,
                session_id=session_id,
                protected=is_persistent,
                metadata=dict(actor_observability_metadata(actor) or {}),
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
    observability_base_model: str | None = None,
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
        observability_base_model,
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
            namespace=PERSISTENT_NAMESPACE,
            ignore_reinit_error=True,
        )

    resource_pool = get_resource_pool()
    killed_any = False

    def _remove_detached_pg(actor_name: str) -> None:
        pg_name = _make_megatron_pg_name_from_actor_name(actor_name)
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
                verify_absent=True,
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
                    verify_absent=True,
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
