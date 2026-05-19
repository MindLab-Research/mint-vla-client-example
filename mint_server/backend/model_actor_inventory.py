"""Process-local model actor inventory owned by ModelActorSupervisor.

All GPU-using runtime actors publish inventory, inflight counts, protection
flags, and session bindings here. The inventory is intentionally process-local:
the durable scheduling state lives in TaskStateStore/ModelWorkScheduler, while
actor desired-state reconciliation belongs to ModelActorSupervisor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import ray

from ..config import config as server_config
from . import ray_kill
from .async_ray_control import async_get_ray_ref

logger = logging.getLogger(__name__)
ActorHandle = Any


class ModelActorSupervisorStaleError(RuntimeError):
    """ModelActorInventory inventory/state disagrees with Ray named-actor registry."""


class ActorType(Enum):
    MEGATRON = "megatron"  # MoE training (8 GPUs)
    DENSE = "dense"        # Dense training (1 GPU)
    OPENPI = "openpi"      # OpenPI shared training (1 GPU)
    VLLM = "vllm"          # Inference (1-4 GPUs)


@dataclass
class ActorEntry:
    actor_name: str
    actor_type: ActorType
    num_gpus: int
    actor_handle: ActorHandle | None = None
    namespace: str = "mint"
    base_model: str = ""
    current_session: str | None = None
    node_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    inflight_count: int = 0
    creating: bool = True
    protected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_sample_time: float | None = None
    metadata_sample_source: str | None = None
    rss_bytes: int | None = None
    rss_sample_time: float | None = None
    rss_sample_source: str | None = None

    def touch(self) -> None:
        self.last_accessed = time.time()

    def mark_ready(self) -> None:
        self.creating = False
        self.touch()

    def is_idle(self, session_idle_timeout: float = 300) -> bool:
        if self.creating:
            return False
        if self.inflight_count > 0:
            return False
        if self.actor_type == ActorType.VLLM:
            return self.idle_time() > session_idle_timeout
        if self.current_session is None:
            return True
        return self.idle_time() > session_idle_timeout

    def age(self) -> float:
        return time.time() - self.created_at

    def idle_time(self) -> float:
        return time.time() - self.last_accessed


class _ModelActorInventoryState:
    """Authoritative ModelActorInventory state machine.

    This object stores only serializable control-plane metadata. Actor handles
    intentionally stay worker-local.
    """

    def __init__(self, *, session_idle_timeout: int) -> None:
        self.entries: dict[str, ActorEntry] = {}
        self.session_idle_timeout = int(session_idle_timeout)
        self.lifecycle_metrics: dict[tuple[str, str], int] = {}

    def register(
        self,
        *,
        actor_name: str,
        actor_type: ActorType,
        num_gpus: int,
        namespace: str = "mint",
        base_model: str = "",
        session_id: str | None = None,
        node_id: str | None = None,
        protected: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ActorEntry:
        entry = self.entries.get(actor_name)
        if entry is None:
            has_metadata = bool(metadata)
            entry = ActorEntry(
                actor_name=actor_name,
                actor_type=actor_type,
                num_gpus=int(num_gpus),
                namespace=namespace,
                base_model=base_model,
                current_session=session_id,
                node_id=node_id,
                protected=bool(protected),
                metadata=dict(metadata or {}),
                metadata_sample_time=(time.time() if has_metadata else None),
                metadata_sample_source=("register" if has_metadata else None),
            )
            self.entries[actor_name] = entry
            logger.info(
                "[ModelActorInventory] Registered %s actor=%s num_gpus=%s base_model=%s node_id=%s",
                actor_type.value,
                actor_name,
                num_gpus,
                base_model,
                node_id,
            )
            return entry

        entry.touch()
        entry.actor_type = actor_type
        entry.num_gpus = int(num_gpus)
        entry.namespace = namespace
        entry.base_model = base_model
        if session_id is not None:
            entry.current_session = session_id
        if node_id is not None:
            entry.node_id = node_id
        if protected:
            entry.protected = True
        if metadata:
            entry.metadata.update(dict(metadata))
            entry.metadata_sample_time = time.time()
            entry.metadata_sample_source = "register"
        return entry

    def unregister(self, actor_name: str) -> bool:
        removed = self.entries.pop(actor_name, None) is not None
        if removed:
            logger.info("[ModelActorInventory] Unregistered actor=%s", actor_name)
        return removed

    def get(self, actor_name: str, *, touch: bool) -> ActorEntry | None:
        entry = self.entries.get(actor_name)
        if entry is not None and touch:
            entry.touch()
        return entry

    def set_session(self, actor_name: str, session_id: str | None) -> bool:
        entry = self.entries.get(actor_name)
        if entry is None:
            return False
        entry.current_session = session_id
        if session_id is not None:
            entry.touch()
        return True

    def set_protected(self, actor_name: str, protected: bool = True) -> bool:
        entry = self.entries.get(actor_name)
        if entry is None:
            return False
        entry.protected = bool(protected)
        return True

    def is_protected(self, actor_name: str) -> bool:
        entry = self.entries.get(actor_name)
        return bool(entry and entry.protected)

    def touch(self, actor_name: str) -> bool:
        entry = self.entries.get(actor_name)
        if entry is None:
            return False
        entry.touch()
        return True

    def mark_inflight(self, actor_name: str, delta: int) -> bool:
        entry = self.entries.get(actor_name)
        if entry is None:
            return False
        if int(delta) == 0:
            return True
        entry.inflight_count = max(0, int(entry.inflight_count) + int(delta))
        if int(delta) > 0:
            entry.touch()
        return True

    def mark_ready(self, actor_name: str) -> bool:
        entry = self.entries.get(actor_name)
        if entry is None:
            return False
        entry.mark_ready()
        return True

    def update_metadata(
        self,
        actor_name: str,
        *,
        metadata: dict[str, Any],
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        entry = self.entries.get(actor_name)
        if entry is None:
            return False
        entry.metadata = dict(metadata)
        entry.metadata_sample_time = time.time() if sample_time is None else float(sample_time)
        entry.metadata_sample_source = None if sample_source is None else str(sample_source)
        return True

    def record_lifecycle_event(self, *, base_model: str, event: str) -> None:
        key = (str(base_model or "unknown"), str(event or "unknown"))
        self.lifecycle_metrics[key] = int(self.lifecycle_metrics.get(key, 0)) + 1

    def lifecycle_metrics_snapshot(self) -> list[dict[str, int | str]]:
        return [
            {
                "base_model": base_model,
                "event": event,
                "count": int(count),
            }
            for (base_model, event), count in sorted(self.lifecycle_metrics.items())
        ]

    def _kill_actor(self, entry: ActorEntry) -> bool:
        try:
            actor = ray.get_actor(entry.actor_name, namespace=entry.namespace)
        except ValueError:
            logger.warning("[ModelActorInventory] Actor not found during eviction: %s", entry.actor_name)
            return False
        except Exception as e:
            logger.warning("[ModelActorInventory] Actor lookup failed during eviction actor=%s err=%s", entry.actor_name, e)
            return False

        try:
            try:
                if hasattr(actor, "shutdown"):
                    ray.get(actor.shutdown.remote(), timeout=10)
            except Exception:
                pass

            ray_kill.kill(
                actor,
                reason="model_actor_inventory_evict",
                actor_name=entry.actor_name,
                namespace=entry.namespace,
                actor_type=entry.actor_type.value,
                num_gpus=entry.num_gpus,
                base_model=entry.base_model,
                current_session=entry.current_session,
                creating=entry.creating,
                idle_time=f"{entry.idle_time():.1f}",
                age=f"{entry.age():.1f}",
                session_idle_timeout=self.session_idle_timeout,
            )
            logger.info("[ModelActorInventory] Killed actor=%s", entry.actor_name)
            return True
        except Exception as e:
            logger.warning("[ModelActorInventory] Error killing actor %s: %s", entry.actor_name, e)
            return False

    def iter_entries(self) -> list[ActorEntry]:
        return list(self.entries.values())

    def prune_stale(self) -> int:
        stale: list[str] = []
        for name, entry in self.entries.items():
            try:
                ray.get_actor(entry.actor_name, namespace=entry.namespace)
            except ValueError:
                stale.append(name)
            except Exception as e:
                logger.warning("[ModelActorInventory] Error checking actor %s: %s", name, e)
        for name in stale:
            self.entries.pop(name, None)
        return len(stale)

    def clear_session(self, session_id: str, *, actor_type: ActorType | None = None) -> int:
        cleared = 0
        for entry in self.entries.values():
            if actor_type is not None and entry.actor_type != actor_type:
                continue
            if entry.current_session != session_id:
                continue
            entry.current_session = None
            entry.touch()
            cleared += 1
        if cleared:
            logger.info(
                "[ModelActorInventory] Cleared current_session=%s actor_type=%s count=%s",
                session_id,
                actor_type.value if actor_type is not None else "any",
                cleared,
            )
        return cleared

    def total_gpus_used(self) -> int:
        return sum(int(entry.num_gpus) for entry in self.entries.values())

    def clear(self, *, kill_actors: bool = True) -> int:
        count = len(self.entries)
        if kill_actors:
            for entry in list(self.entries.values()):
                self._kill_actor(entry)
        self.entries.clear()
        return count


def _entry_to_record(entry: ActorEntry) -> dict[str, Any]:
    return {
        "actor_name": entry.actor_name,
        "actor_type": entry.actor_type.value,
        "num_gpus": int(entry.num_gpus),
        "namespace": entry.namespace,
        "base_model": entry.base_model,
        "current_session": entry.current_session,
        "node_id": entry.node_id,
        "created_at": float(entry.created_at),
        "last_accessed": float(entry.last_accessed),
        "inflight_count": int(entry.inflight_count),
        "creating": bool(entry.creating),
        "protected": bool(entry.protected),
        "metadata": dict(entry.metadata or {}),
        "metadata_sample_time": entry.metadata_sample_time,
        "metadata_sample_source": entry.metadata_sample_source,
        "rss_bytes": entry.rss_bytes,
        "rss_sample_time": entry.rss_sample_time,
        "rss_sample_source": entry.rss_sample_source,
    }


def _record_to_entry(record: dict[str, Any], *, actor_handle: ActorHandle | None = None) -> ActorEntry:
    return ActorEntry(
        actor_name=str(record.get("actor_name") or ""),
        actor_type=ActorType(str(record.get("actor_type") or ActorType.DENSE.value)),
        num_gpus=int(record.get("num_gpus") or 0),
        actor_handle=actor_handle,
            namespace=str(record.get("namespace") or "mint"),
        base_model=str(record.get("base_model") or ""),
        current_session=record.get("current_session"),
        node_id=record.get("node_id"),
        created_at=float(record.get("created_at") or time.time()),
        last_accessed=float(record.get("last_accessed") or time.time()),
        inflight_count=int(record.get("inflight_count") or 0),
        creating=bool(record.get("creating", True)),
        protected=bool(record.get("protected", False)),
        metadata=dict(record.get("metadata") or {}),
        metadata_sample_time=(
            None if record.get("metadata_sample_time") is None else float(record.get("metadata_sample_time"))
        ),
        metadata_sample_source=(
            None
            if record.get("metadata_sample_source") is None
            else str(record.get("metadata_sample_source"))
        ),
        rss_bytes=(None if record.get("rss_bytes") is None else int(record.get("rss_bytes"))),
        rss_sample_time=(
            None if record.get("rss_sample_time") is None else float(record.get("rss_sample_time"))
        ),
        rss_sample_source=(
            None if record.get("rss_sample_source") is None else str(record.get("rss_sample_source"))
        ),
    )


def _backend_for_entry(entry: ActorEntry) -> str:
    if entry.actor_type == ActorType.DENSE:
        return "peft"
    if entry.actor_type == ActorType.OPENPI:
        return "openpi"
    if entry.actor_type == ActorType.MEGATRON:
        return "megatron"
    return "vllm"


def _role_for_entry(entry: ActorEntry) -> str:
    return "inference" if entry.actor_type == ActorType.VLLM else "trainer"


def _exclude_actor_types(values: tuple[ActorType, ...]) -> list[str]:
    return [value.value if isinstance(value, ActorType) else str(value) for value in values]


def _restore_actor_types(values: list[str] | tuple[str, ...] | None) -> tuple[ActorType, ...]:
    if not values:
        return ()
    return tuple(ActorType(str(value)) for value in values)


def _normalize_actor_observability_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    out: dict[str, Any] = {}
    hostname = payload.get("hostname")
    if isinstance(hostname, str) and hostname.strip():
        out["hostname"] = hostname.strip()
    node_id = payload.get("node_id")
    if isinstance(node_id, str) and node_id.strip():
        out["node_id"] = node_id.strip()
    gpu_indices = payload.get("gpu_indices")
    if isinstance(gpu_indices, list):
        out["gpu_indices"] = [int(idx) for idx in gpu_indices if isinstance(idx, (int, float, str)) and str(idx).strip()]
    gpu_bindings = payload.get("gpu_bindings")
    if isinstance(gpu_bindings, list):
        clean_bindings = []
        for binding in gpu_bindings:
            if not isinstance(binding, dict):
                continue
            gpu_index = binding.get("gpu_index")
            gpu_uuid = binding.get("gpu_uuid")
            if gpu_index is None and not (isinstance(gpu_uuid, str) and gpu_uuid.strip()):
                continue
            clean_binding = {
                "hostname": binding.get("hostname"),
                "node_id": binding.get("node_id"),
                "rank": binding.get("rank"),
            }
            ray_gpu_id = binding.get("ray_gpu_id")
            if isinstance(ray_gpu_id, str) and ray_gpu_id.strip():
                clean_binding["ray_gpu_id"] = ray_gpu_id.strip()
            if gpu_index is not None:
                clean_binding["gpu_index"] = int(gpu_index)
            if isinstance(gpu_uuid, str) and gpu_uuid.strip():
                clean_binding["gpu_uuid"] = gpu_uuid.strip()
            clean_bindings.append(clean_binding)
        if clean_bindings:
            out["gpu_bindings"] = clean_bindings
    int_fields = {
        "scheduler_waiting_requests",
        "scheduler_running_requests",
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "preemptions_total",
        "active_sessions",
        "session_unknown",
        "session_step",
        "gpu_memory_allocated_bytes",
        "gpu_memory_reserved_bytes",
        "gpu_memory_fragmentation_bytes",
        "max_lora_rank",
        "actual_rank",
    }
    float_fields = {
        "scheduler_kv_cache_usage_ratio",
        "prefix_cache_hit_ratio",
        "learning_rate",
    }
    skip_fields = {"hostname", "node_id", "gpu_indices", "gpu_bindings", "rank"}
    for src, value in payload.items():
        if src in skip_fields:
            continue
        if not isinstance(value, (int, float, str)) or not str(value).strip():
            continue
        try:
            if src in int_fields or src.endswith(("_count", "_bytes")):
                out[src] = max(0, int(value))
            elif src in float_fields or src.endswith(("_ratio", "_total", "_max", "_p50_recent", "_p95_recent")):
                out[src] = max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    return out or None


def actor_observability_metadata(actor_handle: ActorHandle | None, *, timeout_s: float = 5.0) -> dict[str, Any] | None:
    if actor_handle is None:
        return None
    getter = getattr(actor_handle, "get_observability_binding", None)
    if not callable(getter):
        return None
    try:
        payload = ray.get(getter.remote(), timeout=float(timeout_s))
    except Exception as e:
        logger.debug("[ModelActorInventory] get_observability_binding failed: %s", e)
        return None
    return _normalize_actor_observability_payload(payload)


async def async_actor_observability_metadata(
    actor_handle: ActorHandle | None,
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any] | None:
    if actor_handle is None:
        return None
    getter = getattr(actor_handle, "get_observability_binding", None)
    if not callable(getter):
        return None
    try:
        payload = await async_get_ray_ref(getter.remote(), timeout_s=float(timeout_s))
    except Exception as e:
        logger.debug("[ModelActorInventory] async get_observability_binding failed: %s", e)
        return None
    return _normalize_actor_observability_payload(payload)


class ModelActorInventory:
    """Process-local registry for GPU-using actors."""

    _instance: "ModelActorInventory | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        session_idle_timeout = int(server_config.model_actor_inventory_session_idle_timeout_s)
        self.SESSION_IDLE_TIMEOUT = session_idle_timeout
        self._local_state = _ModelActorInventoryState(
            session_idle_timeout=session_idle_timeout,
        )
        self._local_lock = threading.Lock()
        self._handle_cache: dict[str, ActorHandle] = {}
        self.RSS_TTL_S = float(os.environ.get("MINT_MODEL_ACTOR_INVENTORY_RSS_TTL_S", "60.0"))
        self.METADATA_TTL_S = float(os.environ.get("MINT_MODEL_ACTOR_INVENTORY_OBSERVABILITY_TTL_S", "30.0"))
        self.METADATA_TIMEOUT_S = float(os.environ.get("MINT_MODEL_ACTOR_INVENTORY_OBSERVABILITY_TIMEOUT_S", "1.0"))
        self.METADATA_REFRESH_CONCURRENCY = max(
            1,
            int(os.environ.get("MINT_MODEL_ACTOR_INVENTORY_OBSERVABILITY_REFRESH_CONCURRENCY", "8")),
        )
        self._metadata_metrics_lock = threading.Lock()
        self._metadata_metrics: dict[str, dict[str, int]] = {}
        # Backward-compatible aliases used by observability tests.
        self._pool_lock = self._local_lock
        self._entries = self._local_state.entries
        self._initialized = True
        logger.info(
            "[ModelActorInventory] Initialized SESSION_IDLE_TIMEOUT=%s",
            self.SESSION_IDLE_TIMEOUT,
        )

    def _clear_cached_handle(self, actor_name: str) -> None:
        with self._local_lock:
            self._handle_cache.pop(actor_name, None)

    def _remember_handle(self, actor_name: str, actor_handle: ActorHandle | None) -> None:
        if actor_handle is None:
            return
        with self._local_lock:
            self._handle_cache[actor_name] = actor_handle

    def _lookup_handle(self, actor_name: str, namespace: str) -> ActorHandle | None:
        with self._local_lock:
            actor = self._handle_cache.get(actor_name)
        if actor is not None:
            return actor
        try:
            if not ray.is_initialized():
                return None
        except Exception:
            return None
        try:
            actor = ray.get_actor(actor_name, namespace=namespace)
        except Exception:
            return None
        self._remember_handle(actor_name, actor)
        return actor

    async def _lookup_handle_async(self, actor_name: str, namespace: str) -> ActorHandle | None:
        with self._local_lock:
            actor = self._handle_cache.get(actor_name)
        if actor is not None:
            return actor
        try:
            if not ray.is_initialized():
                return None
        except Exception:
            return None
        return await asyncio.to_thread(self._lookup_handle, actor_name, namespace)

    def _local(self, fn, *args, **kwargs):
        with self._local_lock:
            return fn(*args, **kwargs)

    def register(
        self,
        actor_name: str,
        actor_type: ActorType,
        num_gpus: int,
        actor_handle: ActorHandle | None = None,
        namespace: str = "mint",
        base_model: str = "",
        session_id: str | None = None,
        node_id: str | None = None,
        protected: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ActorEntry:
        self._remember_handle(actor_name, actor_handle)
        return self._local(
            self._local_state.register,
            actor_name=actor_name,
            actor_type=actor_type,
            num_gpus=num_gpus,
            namespace=namespace,
            base_model=base_model,
            session_id=session_id,
            node_id=node_id,
            protected=protected,
            metadata=metadata,
        )

    def unregister(self, actor_name: str) -> bool:
        self._clear_cached_handle(actor_name)
        return bool(self._local(self._local_state.unregister, actor_name))

    def get(self, actor_name: str) -> ActorEntry | None:
        return self._local(self._local_state.get, actor_name, touch=True)

    def set_session(self, actor_name: str, session_id: str | None) -> None:
        self._local(self._local_state.set_session, actor_name, session_id)

    async def async_set_session(self, actor_name: str, session_id: str | None) -> None:
        await asyncio.to_thread(self._local, self._local_state.set_session, actor_name, session_id)

    def set_protected(self, actor_name: str, protected: bool = True) -> bool:
        return bool(self._local(self._local_state.set_protected, actor_name, protected))

    def is_protected(self, actor_name: str) -> bool:
        return bool(self._local(self._local_state.is_protected, actor_name))

    def touch(self, actor_name: str) -> bool:
        return bool(self._local(self._local_state.touch, actor_name))

    async def async_touch(self, actor_name: str) -> bool:
        return bool(await asyncio.to_thread(self._local, self._local_state.touch, actor_name))

    def mark_inflight(self, actor_name: str, delta: int) -> None:
        self._local(self._local_state.mark_inflight, actor_name, delta)

    def mark_ready(self, actor_name: str) -> None:
        self._local(self._local_state.mark_ready, actor_name)

    def update_metadata(
        self,
        actor_name: str,
        *,
        metadata: dict[str, Any],
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        return bool(
            self._local(
                self._local_state.update_metadata,
                actor_name,
                metadata=dict(metadata or {}),
                sample_time=sample_time,
                sample_source=sample_source,
            )
        )

    async def async_update_metadata(
        self,
        actor_name: str,
        *,
        metadata: dict[str, Any],
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        return bool(
            self._local(
                self._local_state.update_metadata,
                actor_name,
                metadata=dict(metadata or {}),
                sample_time=sample_time,
                sample_source=sample_source,
            )
        )

    def list_actors(
        self,
        *,
        refresh_metadata: bool = False,
        actor_type: ActorType | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        entries = self.iter_entries(prune_stale=True)
        if actor_type is not None:
            entries = [entry for entry in entries if entry.actor_type == actor_type]
        if model_name is not None:
            entries = [entry for entry in entries if entry.base_model == model_name]
        now = time.time()
        if refresh_metadata:
            for entry in entries:
                self._refresh_entry_metadata(entry, now=now, sample_source="list_actors")
        return [self._actor_inventory_record(entry, now=now) for entry in entries]

    async def async_list_actors(
        self,
        *,
        refresh_metadata: bool = False,
        actor_type: ActorType | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        entries = await self.async_iter_entries(prune_stale=True)
        if actor_type is not None:
            entries = [entry for entry in entries if entry.actor_type == actor_type]
        if model_name is not None:
            entries = [entry for entry in entries if entry.base_model == model_name]
        now = time.time()
        if refresh_metadata:
            await self._refresh_entries_metadata_async(entries, now=now, sample_source="list_actors")
        return [self._actor_inventory_record(entry, now=now) for entry in entries]

    def _actor_inventory_record(self, entry: ActorEntry, *, now: float) -> dict[str, Any]:
        return {
            "actor_name": entry.actor_name,
            "actor_type": entry.actor_type.value,
            "backend": _backend_for_entry(entry),
            "role": _role_for_entry(entry),
            "num_gpus": entry.num_gpus,
            "base_model": entry.base_model,
            "current_session": entry.current_session,
            "node_id": entry.node_id,
            "creating": entry.creating,
            "protected": entry.protected,
            **self._metadata_snapshot_fields(entry, now=now),
            "idle": entry.is_idle(self.SESSION_IDLE_TIMEOUT),
            "idle_time": entry.idle_time(),
            "age": entry.age(),
        }

    def _metadata_is_fresh(self, entry: ActorEntry, *, now: float) -> bool:
        if entry.metadata_sample_time is None:
            return False
        return max(0.0, float(now) - float(entry.metadata_sample_time)) <= float(self.METADATA_TTL_S)

    def _metadata_snapshot_fields(self, entry: ActorEntry, *, now: float) -> dict[str, Any]:
        fields: dict[str, Any] = {"metadata": dict(entry.metadata or {})}
        if entry.metadata_sample_time is None:
            fields["metadata_cache_state"] = "unknown"
            return fields

        sample_age = max(0.0, float(now) - float(entry.metadata_sample_time))
        fields["metadata_sample_age_s"] = sample_age
        if entry.metadata_sample_source is not None:
            fields["metadata_sample_source"] = str(entry.metadata_sample_source)
        fields["metadata_cache_state"] = "fresh" if sample_age <= float(self.METADATA_TTL_S) else "stale"
        return fields

    def _record_metadata_metric(self, actor_type: ActorType, key: str) -> None:
        actor_key = actor_type.value
        with self._metadata_metrics_lock:
            bucket = self._metadata_metrics.setdefault(
                actor_key,
                {
                    "cache_hits_total": 0,
                    "cache_stale_total": 0,
                    "refresh_success_total": 0,
                    "refresh_failures_total": 0,
                },
            )
            bucket[key] = int(bucket.get(key, 0)) + 1

    def metadata_cache_metrics_snapshot(self) -> list[dict[str, int | str]]:
        with self._metadata_metrics_lock:
            items = sorted(self._metadata_metrics.items())
        return [
            {
                "actor_type": actor_type,
                "cache_hits_total": int(values.get("cache_hits_total", 0)),
                "cache_stale_total": int(values.get("cache_stale_total", 0)),
                "refresh_success_total": int(values.get("refresh_success_total", 0)),
                "refresh_failures_total": int(values.get("refresh_failures_total", 0)),
            }
            for actor_type, values in items
        ]

    def lifecycle_metrics_snapshot(self) -> list[dict[str, int | str]]:
        return self._local(self._local_state.lifecycle_metrics_snapshot)

    def _refresh_entry_metadata(
        self,
        entry: ActorEntry,
        *,
        now: float,
        sample_source: str = "cached_snapshot",
    ) -> None:
        if entry.actor_type not in {ActorType.VLLM, ActorType.MEGATRON}:
            return
        if self._metadata_is_fresh(entry, now=now):
            self._record_metadata_metric(entry.actor_type, "cache_hits_total")
            return
        self._record_metadata_metric(entry.actor_type, "cache_stale_total")
        handle = entry.actor_handle or self._lookup_handle(entry.actor_name, entry.namespace)
        if handle is None:
            self._record_metadata_metric(entry.actor_type, "refresh_failures_total")
            return
        metadata = actor_observability_metadata(handle, timeout_s=self.METADATA_TIMEOUT_S)
        if metadata is None:
            self._record_metadata_metric(entry.actor_type, "refresh_failures_total")
            return
        sample_time = float(now)
        if self.update_metadata(
            entry.actor_name,
            metadata=metadata,
            sample_time=sample_time,
            sample_source=sample_source,
        ):
            entry.metadata = dict(metadata)
            entry.metadata_sample_time = sample_time
            entry.metadata_sample_source = sample_source
            self._record_metadata_metric(entry.actor_type, "refresh_success_total")
            return
        self._record_metadata_metric(entry.actor_type, "refresh_failures_total")

    async def _refresh_entry_metadata_async(
        self,
        entry: ActorEntry,
        *,
        now: float,
        sample_source: str = "cached_snapshot",
    ) -> None:
        if entry.actor_type not in {ActorType.VLLM, ActorType.MEGATRON}:
            return
        if self._metadata_is_fresh(entry, now=now):
            self._record_metadata_metric(entry.actor_type, "cache_hits_total")
            return
        self._record_metadata_metric(entry.actor_type, "cache_stale_total")
        handle = entry.actor_handle or await self._lookup_handle_async(entry.actor_name, entry.namespace)
        if handle is None:
            self._record_metadata_metric(entry.actor_type, "refresh_failures_total")
            return
        metadata = await async_actor_observability_metadata(handle, timeout_s=self.METADATA_TIMEOUT_S)
        if metadata is None:
            self._record_metadata_metric(entry.actor_type, "refresh_failures_total")
            return
        sample_time = float(now)
        if await self.async_update_metadata(
            entry.actor_name,
            metadata=metadata,
            sample_time=sample_time,
            sample_source=sample_source,
        ):
            entry.metadata = dict(metadata)
            entry.metadata_sample_time = sample_time
            entry.metadata_sample_source = sample_source
            self._record_metadata_metric(entry.actor_type, "refresh_success_total")
            return
        self._record_metadata_metric(entry.actor_type, "refresh_failures_total")

    async def _refresh_entries_metadata_async(
        self,
        entries: list[ActorEntry],
        *,
        now: float,
        sample_source: str,
    ) -> None:
        concurrency = max(1, int(self.METADATA_REFRESH_CONCURRENCY))
        if concurrency == 1 or len(entries) <= 1:
            for entry in entries:
                await self._refresh_entry_metadata_async(entry, now=now, sample_source=sample_source)
            return

        for start in range(0, len(entries), concurrency):
            batch = entries[start : start + concurrency]
            results = await asyncio.gather(
                *(
                    self._refresh_entry_metadata_async(entry, now=now, sample_source=sample_source)
                    for entry in batch
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    def _cached_snapshot_record(self, entry: ActorEntry, *, now: float) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "actor_name": entry.actor_name,
            "actor_type": entry.actor_type.value,
            "num_gpus": entry.num_gpus,
            "base_model": entry.base_model,
            "current_session": entry.current_session,
            "node_id": entry.node_id,
            "creating": entry.creating,
            "protected": entry.protected,
            **self._metadata_snapshot_fields(entry, now=now),
            "idle": entry.is_idle(self.SESSION_IDLE_TIMEOUT),
            "idle_time": entry.idle_time(),
            "age": entry.age(),
        }
        if entry.rss_bytes is None or entry.rss_sample_time is None:
            rec["rss_cache_state"] = "unknown"
            return rec

        sample_age = max(0.0, float(now) - float(entry.rss_sample_time))
        rec["rss_sample_age_s"] = sample_age
        if entry.rss_sample_source is not None:
            rec["rss_sample_source"] = str(entry.rss_sample_source)
        if sample_age <= float(self.RSS_TTL_S):
            rec["rss_cache_state"] = "fresh"
            rec["rss_bytes"] = int(entry.rss_bytes)
        else:
            rec["rss_cache_state"] = "stale"
        return rec

    def cached_snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._pool_lock:
            entries = list(self._entries.values())
        for entry in entries:
            self._refresh_entry_metadata(entry, now=now)
        return [self._cached_snapshot_record(entry, now=now) for entry in entries]

    def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
        out: list[dict] = []
        now = time.time()
        for entry in self.iter_entries():
            rec = {
                "actor_name": entry.actor_name,
                "actor_type": entry.actor_type.value,
                "num_gpus": entry.num_gpus,
                "base_model": entry.base_model,
                "current_session": entry.current_session,
                "node_id": entry.node_id,
                **self._metadata_snapshot_fields(entry, now=now),
                "idle": entry.is_idle(self.SESSION_IDLE_TIMEOUT),
                "idle_time": entry.idle_time(),
                "age": entry.age(),
            }
            handle = entry.actor_handle or self._lookup_handle(entry.actor_name, entry.namespace)
            if handle is None:
                rec["rss_bytes"] = 0
                rec["error"] = "missing actor_handle"
                out.append(rec)
                continue
            try:
                rss = ray.get(handle.get_rss_bytes.remote(), timeout=float(timeout_s))
                rec["rss_bytes"] = int(cast(Any, rss))
            except Exception as ex:
                rec["rss_bytes"] = 0
                rec["error"] = f"{type(ex).__name__}: {ex}"
            out.append(rec)
        return out

    def iter_entries(self, *, prune_stale: bool = False) -> list[ActorEntry]:
        if prune_stale and ray.is_initialized():
            self._local(self._local_state.prune_stale)
        with self._local_lock:
            return list(self._local_state.iter_entries())

    async def async_iter_entries(self, *, prune_stale: bool = False) -> list[ActorEntry]:
        if prune_stale and ray.is_initialized():
            await asyncio.to_thread(self._local, self._local_state.prune_stale)
        with self._local_lock:
            return list(self._local_state.iter_entries())

    def clear_session(self, session_id: str, *, actor_type: ActorType | None = None) -> int:
        return int(self._local(self._local_state.clear_session, session_id, actor_type=actor_type))

    def total_gpus_used(self) -> int:
        return int(self._local(self._local_state.total_gpus_used))

    async def async_total_gpus_used(self) -> int:
        return int(self._local(self._local_state.total_gpus_used))

    def gpus_used_by_node(self) -> dict[str, int]:
        usage: dict[str, int] = {}
        for entry in self.iter_entries():
            node_id = entry.node_id
            if not node_id and entry.actor_handle is not None:
                node_id = self._get_actor_node_id(entry.actor_handle)
            if not node_id:
                continue
            usage[node_id] = usage.get(node_id, 0) + int(entry.num_gpus)
        return usage

    def _get_actor_node_id(self, actor_handle: ActorHandle) -> str | None:
        try:
            actor_id = actor_handle._actor_id
            actor_id_hex = actor_id.hex()
            from ray._private.state import actors as state_actors

            actor_info = state_actors(actor_id_hex)
            if actor_info:
                address = actor_info.get("Address", {})
                return address.get("NodeID")
        except Exception as e:
            logger.debug("[ModelActorInventory] Could not get node_id: %s", e)
        return None

    def clear(self, kill_actors: bool = True) -> int:
        with self._local_lock:
            self._handle_cache.clear()
        return int(self._local(self._local_state.clear, kill_actors=kill_actors))


# Global singleton accessor

def get_model_actor_inventory() -> ModelActorInventory:
    return ModelActorInventory()
