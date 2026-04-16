"""Unified ResourcePool with detached control-plane state.

All GPU-using actors share one admission and eviction control plane. In
multi-worker API deployments the authoritative state lives in a detached Ray
actor so inventory, pending GPU reservations, LRU, inflight counts, protection
flags, and session bindings stay consistent across workers.

Process-local state is limited to actor-handle caching. When Ray is not
initialized, the module falls back to an in-process state object so unit tests
can still use the same API.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import ray

from ..config import config as server_config, otel_env_vars
from ..ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator
from . import ray_kill

logger = logging.getLogger(__name__)
ActorHandle = Any


class ResourcePoolStaleError(RuntimeError):
    """ResourcePool inventory/state disagrees with Ray named-actor registry."""


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
    namespace: str = "tinker"
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


class _ResourcePoolState:
    """Authoritative ResourcePool state machine.

    This object stores only serializable control-plane metadata. Actor handles
    intentionally stay worker-local.
    """

    def __init__(self, *, min_actor_age: int, session_idle_timeout: int) -> None:
        self.entries: dict[str, ActorEntry] = {}
        self.pending_gpus: int = 0
        self.min_actor_age = int(min_actor_age)
        self.session_idle_timeout = int(session_idle_timeout)
        self.lifecycle_metrics: dict[tuple[str, str], int] = {}

    def register(
        self,
        *,
        actor_name: str,
        actor_type: ActorType,
        num_gpus: int,
        namespace: str = "tinker",
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
                "[ResourcePool] Registered %s actor=%s num_gpus=%s base_model=%s node_id=%s",
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
            logger.info("[ResourcePool] Unregistered actor=%s", actor_name)
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

    def reserve_gpus(self, num_gpus: int) -> bool:
        self.pending_gpus += max(0, int(num_gpus))
        return True

    def release_pending_gpus(self, num_gpus: int) -> int:
        self.pending_gpus = max(0, self.pending_gpus - max(0, int(num_gpus)))
        return self.pending_gpus

    def get_effective_available_gpus(self, *, ray_available: int | None = None) -> int:
        available = int(ray_available) if ray_available is not None else int(ray.available_resources().get("GPU", 0))
        return max(0, available - int(self.pending_gpus))

    def _get_evictable_actors_lru(
        self,
        *,
        allow_evict_protected: bool,
        exclude_actor_types: tuple[ActorType, ...] = (),
    ) -> list[ActorEntry]:
        evictable = [
            entry
            for entry in self.entries.values()
            if entry.actor_type not in exclude_actor_types
            if allow_evict_protected or not entry.protected
            if entry.is_idle(self.session_idle_timeout)
            if entry.idle_time() > self.min_actor_age
        ]
        return sorted(evictable, key=lambda entry: (entry.protected, entry.last_accessed))

    def _kill_actor(self, entry: ActorEntry) -> bool:
        try:
            actor = ray.get_actor(entry.actor_name, namespace=entry.namespace)
        except ValueError:
            logger.warning("[ResourcePool] Actor not found during eviction: %s", entry.actor_name)
            return False
        except Exception as e:
            logger.warning("[ResourcePool] Actor lookup failed during eviction actor=%s err=%s", entry.actor_name, e)
            return False

        try:
            try:
                if hasattr(actor, "shutdown"):
                    ray.get(actor.shutdown.remote(), timeout=10)
            except Exception:
                pass

            ray_kill.kill(
                actor,
                reason="resource_pool_evict",
                actor_name=entry.actor_name,
                namespace=entry.namespace,
                actor_type=entry.actor_type.value,
                num_gpus=entry.num_gpus,
                base_model=entry.base_model,
                current_session=entry.current_session,
                creating=entry.creating,
                idle_time=f"{entry.idle_time():.1f}",
                age=f"{entry.age():.1f}",
                min_actor_age=self.min_actor_age,
                session_idle_timeout=self.session_idle_timeout,
            )
            if entry.actor_type == ActorType.MEGATRON:
                self.record_lifecycle_event(base_model=entry.base_model, event="evicted")
            logger.info("[ResourcePool] Evicted actor=%s", entry.actor_name)
            return True
        except Exception as e:
            logger.warning("[ResourcePool] Error killing actor %s: %s", entry.actor_name, e)
            return False

    def evict_for_gpus(
        self,
        needed_gpus: int,
        *,
        allow_evict_protected: bool,
        exclude_actor_types: tuple[ActorType, ...] = (),
    ) -> int:
        freed_gpus = 0
        victims = self._get_evictable_actors_lru(
            allow_evict_protected=allow_evict_protected,
            exclude_actor_types=exclude_actor_types,
        )
        for entry in victims:
            if freed_gpus >= int(needed_gpus):
                break
            if self._kill_actor(entry):
                freed_gpus += int(entry.num_gpus)
                self.entries.pop(entry.actor_name, None)
        return freed_gpus

    def ensure_gpus_available(
        self,
        needed_gpus: int,
        timeout: float = 600.0,
        *,
        allow_evict_protected: bool = False,
        exclude_actor_types: tuple[ActorType, ...] = (),
    ) -> bool:
        import time as time_module

        start_time = time_module.time()
        poll_interval = 5.0
        iteration = 0

        while True:
            iteration += 1
            available = self.get_effective_available_gpus()
            if available >= int(needed_gpus):
                return True

            need_to_free = int(needed_gpus) - int(available)
            evictable = self._get_evictable_actors_lru(
                allow_evict_protected=allow_evict_protected,
                exclude_actor_types=exclude_actor_types,
            )
            logger.info(
                "[ResourcePool] ensure_gpus_available iter=%s need=%s available=%s pending=%s "
                "need_to_free=%s evictable=%s allow_evict_protected=%s exclude_actor_types=%s",
                iteration,
                needed_gpus,
                available,
                self.pending_gpus,
                need_to_free,
                len(evictable),
                allow_evict_protected,
                [actor_type.value for actor_type in exclude_actor_types],
            )

            freed = self.evict_for_gpus(
                need_to_free,
                allow_evict_protected=allow_evict_protected,
                exclude_actor_types=exclude_actor_types,
            )
            if freed > 0:
                time_module.sleep(2.0)
                if self.get_effective_available_gpus() >= int(needed_gpus):
                    return True

            elapsed = time_module.time() - start_time
            if elapsed >= float(timeout):
                raise ValueError(
                    f"Insufficient GPUs: need {needed_gpus}, available {self.get_effective_available_gpus()} "
                    f"after eviction. Freed {freed} GPUs but resources did not become available within "
                    f"{timeout}s timeout. Other actors may be in use. Check cluster status with 'ray status'."
                )
            time_module.sleep(min(poll_interval, float(timeout) - elapsed))

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
                logger.warning("[ResourcePool] Error checking actor %s: %s", name, e)
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
                "[ResourcePool] Cleared current_session=%s actor_type=%s count=%s",
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
        self.pending_gpus = 0
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
        namespace=str(record.get("namespace") or "tinker"),
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


def _ray_namespace() -> str:
    env_ns = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _actor_name() -> str:
    return os.environ.get("MINT_RESOURCE_POOL_ACTOR_NAME", "tinker_resource_pool")


def _detached_enabled() -> bool:
    if os.environ.get("MINT_RESOURCE_POOL_LOCAL_ONLY", "0") == "1":
        return False
    try:
        return bool(ray.is_initialized())
    except Exception:
        return False


async def _await_ray_ref(ref: Any) -> Any:
    if hasattr(ref, "__await__"):
        return await cast(Any, ref)

    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, asyncio.Future):
            return await fut
        if isinstance(fut, concurrent.futures.Future):
            return await asyncio.wrap_future(fut)
        if hasattr(fut, "__await__"):
            return await cast(Any, fut)

    raise TypeError(f"Ray ref is not awaitable: {type(ref)}")


_RESOURCE_POOL_ACTOR_HANDLE = None

def _reset_cached_actor_handle() -> None:
    global _RESOURCE_POOL_ACTOR_HANDLE
    _RESOURCE_POOL_ACTOR_HANDLE = None

_register_ray_reconnect_invalidator(_reset_cached_actor_handle)


def _get_or_create_actor_sync() -> Any:
    global _RESOURCE_POOL_ACTOR_HANDLE

    if not _detached_enabled():
        raise RuntimeError("Ray not initialized")

    name = _actor_name()
    namespace = _ray_namespace()
    try:
        _RESOURCE_POOL_ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _RESOURCE_POOL_ACTOR_HANDLE
    except ValueError:
        pass

    min_actor_age = int(server_config.resource_pool_min_actor_age_s)
    session_idle_timeout = int(server_config.resource_pool_session_idle_timeout_s)

    @ray.remote(num_cpus=0)
    class _ResourcePoolActor:
        def __init__(self, *, min_actor_age: int, session_idle_timeout: int) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._state = _ResourcePoolState(
                min_actor_age=min_actor_age,
                session_idle_timeout=session_idle_timeout,
            )

        def register(self, info: dict[str, Any]) -> dict[str, Any]:
            entry = self._state.register(
                actor_name=str(info.get("actor_name") or ""),
                actor_type=ActorType(str(info.get("actor_type") or ActorType.DENSE.value)),
                num_gpus=int(info.get("num_gpus") or 0),
                namespace=str(info.get("namespace") or "tinker"),
                base_model=str(info.get("base_model") or ""),
                session_id=info.get("session_id"),
                node_id=info.get("node_id"),
                protected=bool(info.get("protected", False)),
                metadata=dict(info.get("metadata") or {}),
            )
            return _entry_to_record(entry)

        def unregister(self, actor_name: str) -> bool:
            return self._state.unregister(actor_name)

        def get(self, actor_name: str, touch: bool = True) -> dict[str, Any] | None:
            entry = self._state.get(actor_name, touch=bool(touch))
            return None if entry is None else _entry_to_record(entry)

        def set_session(self, actor_name: str, session_id: str | None) -> bool:
            return self._state.set_session(actor_name, session_id)

        def set_protected(self, actor_name: str, protected: bool = True) -> bool:
            return self._state.set_protected(actor_name, protected)

        def is_protected(self, actor_name: str) -> bool:
            return self._state.is_protected(actor_name)

        def touch(self, actor_name: str) -> bool:
            return self._state.touch(actor_name)

        def mark_inflight(self, actor_name: str, delta: int) -> bool:
            return self._state.mark_inflight(actor_name, delta)

        def mark_ready(self, actor_name: str) -> bool:
            return self._state.mark_ready(actor_name)

        def update_metadata(
            self,
            actor_name: str,
            metadata: dict[str, Any],
            sample_time: float | None = None,
            sample_source: str | None = None,
        ) -> bool:
            return self._state.update_metadata(
                actor_name,
                metadata=dict(metadata or {}),
                sample_time=sample_time,
                sample_source=sample_source,
            )

        def lifecycle_metrics_snapshot(self) -> list[dict[str, int | str]]:
            return self._state.lifecycle_metrics_snapshot()

        def reserve_gpus(self, num_gpus: int) -> bool:
            return self._state.reserve_gpus(num_gpus)

        def release_pending_gpus(self, num_gpus: int) -> int:
            return self._state.release_pending_gpus(num_gpus)

        def get_effective_available_gpus(self) -> int:
            return self._state.get_effective_available_gpus()

        def ensure_gpus_available(
            self,
            needed_gpus: int,
            timeout: float = 600.0,
            allow_evict_protected: bool = False,
            exclude_actor_types: list[str] | None = None,
        ) -> bool:
            return self._state.ensure_gpus_available(
                needed_gpus,
                timeout=timeout,
                allow_evict_protected=allow_evict_protected,
                exclude_actor_types=_restore_actor_types(exclude_actor_types),
            )

        def list_entries(self, prune_stale: bool = False) -> list[dict[str, Any]]:
            if prune_stale:
                try:
                    self._state.prune_stale()
                except Exception as e:
                    logger.warning("[ResourcePool] prune_stale failed: %s", e)
            return [_entry_to_record(entry) for entry in self._state.iter_entries()]

        def clear_session(self, session_id: str, actor_type: str | None = None) -> int:
            parsed_actor_type = None if actor_type is None else ActorType(str(actor_type))
            return self._state.clear_session(session_id, actor_type=parsed_actor_type)

        def total_gpus_used(self) -> int:
            return self._state.total_gpus_used()

        def clear(self, kill_actors: bool = True) -> int:
            return self._state.clear(kill_actors=kill_actors)

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }

    from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources

    apply_detached_actor_resources(options, ray)
    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=otel_env_vars(),
    )

    try:
        _RESOURCE_POOL_ACTOR_HANDLE = _ResourcePoolActor.options(**options).remote(
            min_actor_age=min_actor_age,
            session_idle_timeout=session_idle_timeout,
        )
        return _RESOURCE_POOL_ACTOR_HANDLE
    except Exception:
        _RESOURCE_POOL_ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _RESOURCE_POOL_ACTOR_HANDLE


def _call_actor_sync(method_name: str, *args, retry_on_actor_restart: bool = False, **kwargs) -> Any:
    global _RESOURCE_POOL_ACTOR_HANDLE

    actor = _get_or_create_actor_sync()
    remote_method = getattr(actor, method_name)
    try:
        return ray.get(remote_method.remote(*args, **kwargs))
    except Exception:
        if not retry_on_actor_restart:
            raise
        _RESOURCE_POOL_ACTOR_HANDLE = None
        actor = _get_or_create_actor_sync()
        remote_method = getattr(actor, method_name)
        return ray.get(remote_method.remote(*args, **kwargs))


async def _call_actor_async(method_name: str, *args, retry_on_actor_restart: bool = False, **kwargs) -> Any:
    global _RESOURCE_POOL_ACTOR_HANDLE

    actor = await asyncio.to_thread(_get_or_create_actor_sync)
    remote_method = getattr(actor, method_name)
    try:
        return await _await_ray_ref(remote_method.remote(*args, **kwargs))
    except Exception:
        if not retry_on_actor_restart:
            raise
        _RESOURCE_POOL_ACTOR_HANDLE = None
        actor = await asyncio.to_thread(_get_or_create_actor_sync)
        remote_method = getattr(actor, method_name)
        return await _await_ray_ref(remote_method.remote(*args, **kwargs))


def actor_observability_metadata(actor_handle: ActorHandle | None, *, timeout_s: float = 5.0) -> dict[str, Any] | None:
    if actor_handle is None:
        return None
    getter = getattr(actor_handle, "get_observability_binding", None)
    if not callable(getter):
        return None
    try:
        payload = ray.get(getter.remote(), timeout=float(timeout_s))
    except Exception as e:
        logger.debug("[ResourcePool] get_observability_binding failed: %s", e)
        return None
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
            if gpu_index is None:
                continue
            clean_bindings.append(
                {
                    "hostname": binding.get("hostname"),
                    "node_id": binding.get("node_id"),
                    "gpu_index": int(gpu_index),
                    "rank": binding.get("rank"),
                }
            )
        if clean_bindings:
            out["gpu_bindings"] = clean_bindings
    int_fields = (
        "scheduler_waiting_requests",
        "scheduler_running_requests",
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "preemptions_total",
        "queue_time_s_count",
        "prefill_time_s_count",
        "decode_time_s_count",
        "time_per_output_token_s_count",
        "active_sessions",
        "session_unknown",
        "session_step",
        "gpu_memory_allocated_bytes",
        "gpu_memory_reserved_bytes",
        "gpu_memory_fragmentation_bytes",
    )
    float_fields = (
        "scheduler_kv_cache_usage_ratio",
        "prefix_cache_hit_ratio",
        "queue_time_s_total",
        "queue_time_s_max",
        "prefill_time_s_total",
        "prefill_time_s_max",
        "decode_time_s_total",
        "decode_time_s_max",
        "time_per_output_token_s_total",
        "time_per_output_token_s_max",
        "learning_rate",
    )
    for src in int_fields:
        value = payload.get(src)
        if isinstance(value, (int, float, str)) and str(value).strip():
            try:
                out[src] = max(0, int(value))
            except (TypeError, ValueError):
                pass
    for src in float_fields:
        value = payload.get(src)
        if isinstance(value, (int, float, str)) and str(value).strip():
            try:
                out[src] = max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    return out or None


class ResourcePool:
    """Unified pool managing all GPU-using actors with detached control plane."""

    _instance: "ResourcePool | None" = None
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
        min_actor_age = int(server_config.resource_pool_min_actor_age_s)
        session_idle_timeout = int(server_config.resource_pool_session_idle_timeout_s)
        self.MIN_ACTOR_AGE = min_actor_age
        self.SESSION_IDLE_TIMEOUT = session_idle_timeout
        self._local_state = _ResourcePoolState(
            min_actor_age=min_actor_age,
            session_idle_timeout=session_idle_timeout,
        )
        self._local_lock = threading.Lock()
        self._handle_cache: dict[str, ActorHandle] = {}
        self.RSS_TTL_S = float(os.environ.get("MINT_RESOURCE_POOL_RSS_TTL_S", "60.0"))
        self.METADATA_TTL_S = float(os.environ.get("MINT_RESOURCE_POOL_OBSERVABILITY_TTL_S", "30.0"))
        self.METADATA_TIMEOUT_S = float(os.environ.get("MINT_RESOURCE_POOL_OBSERVABILITY_TIMEOUT_S", "1.0"))
        self._metadata_metrics_lock = threading.Lock()
        self._metadata_metrics: dict[str, dict[str, int]] = {}
        # Backward-compatible aliases used by observability tests.
        self._pool_lock = self._local_lock
        self._entries = self._local_state.entries
        self._initialized = True
        logger.info(
            "[ResourcePool] Initialized MIN_ACTOR_AGE=%s SESSION_IDLE_TIMEOUT=%s detached=%s",
            self.MIN_ACTOR_AGE,
            self.SESSION_IDLE_TIMEOUT,
            _detached_enabled(),
        )

    def _use_detached(self) -> bool:
        return _detached_enabled()

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
        if not self._use_detached():
            return None
        try:
            actor = ray.get_actor(actor_name, namespace=namespace)
        except Exception:
            return None
        self._remember_handle(actor_name, actor)
        return actor

    def _local(self, fn, *args, **kwargs):
        with self._local_lock:
            return fn(*args, **kwargs)

    def register(
        self,
        actor_name: str,
        actor_type: ActorType,
        num_gpus: int,
        actor_handle: ActorHandle | None = None,
        namespace: str = "tinker",
        base_model: str = "",
        session_id: str | None = None,
        node_id: str | None = None,
        protected: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ActorEntry:
        self._remember_handle(actor_name, actor_handle)
        if not self._use_detached():
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
        record = _call_actor_sync(
            "register",
            {
                "actor_name": actor_name,
                "actor_type": actor_type.value,
                "num_gpus": int(num_gpus),
                "namespace": namespace,
                "base_model": base_model,
                "session_id": session_id,
                "node_id": node_id,
                "protected": bool(protected),
                "metadata": dict(metadata or {}),
            },
        )
        return _record_to_entry(record, actor_handle=self._lookup_handle(actor_name, namespace))

    def unregister(self, actor_name: str) -> bool:
        self._clear_cached_handle(actor_name)
        if not self._use_detached():
            return bool(self._local(self._local_state.unregister, actor_name))
        try:
            return bool(_call_actor_sync("unregister", actor_name))
        except ray.exceptions.GetTimeoutError:
            logger.warning("[ResourcePool] unregister timed out for actor=%s", actor_name)
            return False

    def get(self, actor_name: str) -> ActorEntry | None:
        if not self._use_detached():
            return self._local(self._local_state.get, actor_name, touch=True)
        record = _call_actor_sync("get", actor_name, True, retry_on_actor_restart=True)
        if not isinstance(record, dict):
            return None
        return _record_to_entry(
            record,
            actor_handle=self._lookup_handle(actor_name, str(record.get("namespace") or "tinker")),
        )

    def set_session(self, actor_name: str, session_id: str | None) -> None:
        if not self._use_detached():
            self._local(self._local_state.set_session, actor_name, session_id)
            return
        _call_actor_sync("set_session", actor_name, session_id)

    async def async_set_session(self, actor_name: str, session_id: str | None) -> None:
        if not self._use_detached():
            await asyncio.to_thread(self._local, self._local_state.set_session, actor_name, session_id)
            return
        await _call_actor_async("set_session", actor_name, session_id)

    def set_protected(self, actor_name: str, protected: bool = True) -> bool:
        if not self._use_detached():
            return bool(self._local(self._local_state.set_protected, actor_name, protected))
        return bool(_call_actor_sync("set_protected", actor_name, protected))

    def is_protected(self, actor_name: str) -> bool:
        if not self._use_detached():
            return bool(self._local(self._local_state.is_protected, actor_name))
        return bool(_call_actor_sync("is_protected", actor_name, retry_on_actor_restart=True))

    def touch(self, actor_name: str) -> bool:
        if not self._use_detached():
            return bool(self._local(self._local_state.touch, actor_name))
        return bool(_call_actor_sync("touch", actor_name))

    async def async_touch(self, actor_name: str) -> bool:
        if not self._use_detached():
            return bool(await asyncio.to_thread(self._local, self._local_state.touch, actor_name))
        return bool(await _call_actor_async("touch", actor_name))

    def mark_inflight(self, actor_name: str, delta: int) -> None:
        if not self._use_detached():
            self._local(self._local_state.mark_inflight, actor_name, delta)
            return
        _call_actor_sync("mark_inflight", actor_name, int(delta))

    def mark_ready(self, actor_name: str) -> None:
        if not self._use_detached():
            self._local(self._local_state.mark_ready, actor_name)
            return
        _call_actor_sync("mark_ready", actor_name)

    def update_metadata(
        self,
        actor_name: str,
        *,
        metadata: dict[str, Any],
        sample_time: float | None = None,
        sample_source: str | None = None,
    ) -> bool:
        if not self._use_detached():
            return bool(
                self._local(
                    self._local_state.update_metadata,
                    actor_name,
                    metadata=dict(metadata or {}),
                    sample_time=sample_time,
                    sample_source=sample_source,
                )
            )
        return bool(
            _call_actor_sync(
                "update_metadata",
                actor_name,
                dict(metadata or {}),
                sample_time,
                sample_source,
                retry_on_actor_restart=True,
            )
        )

    def reserve_gpus(self, num_gpus: int) -> bool:
        if not self._use_detached():
            return bool(self._local(self._local_state.reserve_gpus, num_gpus))
        return bool(_call_actor_sync("reserve_gpus", int(num_gpus)))

    def release_pending_gpus(self, num_gpus: int) -> None:
        if not self._use_detached():
            self._local(self._local_state.release_pending_gpus, num_gpus)
            return
        _call_actor_sync("release_pending_gpus", int(num_gpus))

    def get_effective_available_gpus(self) -> int:
        if not self._use_detached():
            return int(self._local(self._local_state.get_effective_available_gpus))
        return int(_call_actor_sync("get_effective_available_gpus", retry_on_actor_restart=True))

    def ensure_gpus_available(
        self,
        needed_gpus: int,
        timeout: float = 600,
        *,
        allow_evict_protected: bool = False,
        exclude_actor_types: tuple[ActorType, ...] = (),
    ) -> bool:
        if not self._use_detached():
            return bool(
                self._local(
                    self._local_state.ensure_gpus_available,
                    int(needed_gpus),
                    timeout=float(timeout),
                    allow_evict_protected=allow_evict_protected,
                    exclude_actor_types=exclude_actor_types,
                )
            )
        return bool(
            _call_actor_sync(
                "ensure_gpus_available",
                int(needed_gpus),
                float(timeout),
                bool(allow_evict_protected),
                _exclude_actor_types(exclude_actor_types),
            )
        )

    def list_actors(self) -> list[dict]:
        entries = self.iter_entries(prune_stale=True)
        return [
            {
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
                "metadata": dict(entry.metadata or {}),
                "idle": entry.is_idle(self.SESSION_IDLE_TIMEOUT),
                "idle_time": entry.idle_time(),
                "age": entry.age(),
            }
            for entry in entries
        ]

    def _metadata_is_fresh(self, entry: ActorEntry, *, now: float) -> bool:
        if entry.metadata_sample_time is None:
            return False
        return max(0.0, float(now) - float(entry.metadata_sample_time)) <= float(self.METADATA_TTL_S)

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
        if not self._use_detached():
            return self._local(self._local_state.lifecycle_metrics_snapshot)
        rows = _call_actor_sync("lifecycle_metrics_snapshot", retry_on_actor_restart=True)
        return rows if isinstance(rows, list) else []

    def _refresh_entry_metadata(self, entry: ActorEntry, *, now: float) -> None:
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
            sample_source="cached_snapshot",
        ):
            entry.metadata = dict(metadata)
            entry.metadata_sample_time = sample_time
            entry.metadata_sample_source = "cached_snapshot"
            self._record_metadata_metric(entry.actor_type, "refresh_success_total")
            return
        self._record_metadata_metric(entry.actor_type, "refresh_failures_total")

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
            "metadata": dict(entry.metadata or {}),
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
        if not self._use_detached():
            with self._pool_lock:
                entries = list(self._entries.values())
        else:
            entries = self.iter_entries(prune_stale=True)
        for entry in entries:
            self._refresh_entry_metadata(entry, now=now)
        return [self._cached_snapshot_record(entry, now=now) for entry in entries]

    def rss_snapshot(self, *, timeout_s: float = 10.0) -> list[dict]:
        out: list[dict] = []
        for entry in self.iter_entries():
            rec = {
                "actor_name": entry.actor_name,
                "actor_type": entry.actor_type.value,
                "num_gpus": entry.num_gpus,
                "base_model": entry.base_model,
                "current_session": entry.current_session,
                "node_id": entry.node_id,
                "idle": entry.is_idle(self.SESSION_IDLE_TIMEOUT),
                "idle_time": entry.idle_time(),
                "age": entry.age(),
            }
            handle = entry.actor_handle or self._lookup_handle(entry.actor_name, entry.namespace)
            if handle is None:
                rec["error"] = "missing actor_handle"
                out.append(rec)
                continue
            try:
                rss = ray.get(handle.get_rss_bytes.remote(), timeout=float(timeout_s))
                rec["rss_bytes"] = int(cast(Any, rss))
            except Exception as ex:
                rec["error"] = f"{type(ex).__name__}: {ex}"
            out.append(rec)
        return out

    def iter_entries(self, *, prune_stale: bool = False) -> list[ActorEntry]:
        if not self._use_detached():
            if prune_stale and ray.is_initialized():
                self._local(self._local_state.prune_stale)
            with self._local_lock:
                return list(self._local_state.iter_entries())

        records = _call_actor_sync("list_entries", bool(prune_stale), retry_on_actor_restart=True)
        out: list[ActorEntry] = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            namespace = str(record.get("namespace") or "tinker")
            out.append(
                _record_to_entry(
                    record,
                    actor_handle=self._lookup_handle(str(record.get("actor_name") or ""), namespace),
                )
            )
        return out

    def clear_session(self, session_id: str, *, actor_type: ActorType | None = None) -> int:
        if not self._use_detached():
            return int(self._local(self._local_state.clear_session, session_id, actor_type=actor_type))
        arg = None if actor_type is None else actor_type.value
        return int(_call_actor_sync("clear_session", session_id, arg))

    def total_gpus_used(self) -> int:
        if not self._use_detached():
            return int(self._local(self._local_state.total_gpus_used))
        return int(_call_actor_sync("total_gpus_used", retry_on_actor_restart=True))

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
            logger.debug("[ResourcePool] Could not get node_id: %s", e)
        return None

    def clear(self, kill_actors: bool = True) -> int:
        with self._local_lock:
            self._handle_cache.clear()
        if not self._use_detached():
            return int(self._local(self._local_state.clear, kill_actors=kill_actors))
        return int(_call_actor_sync("clear", bool(kill_actors)))


# Global singleton accessor

def get_resource_pool() -> ResourcePool:
    return ResourcePool()
