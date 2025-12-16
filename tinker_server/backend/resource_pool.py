"""Unified Resource Pool with LRU eviction across all actor types.

All GPU-using actors (MoE training, dense training, vLLM inference) share
a single resource pool. When GPUs are needed, the least recently used
idle actors are evicted regardless of type.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import ray

logger = logging.getLogger(__name__)


class ActorType(Enum):
    MEGATRON = "megatron"  # MoE training (8 GPUs)
    DENSE = "dense"        # Dense training (1 GPU)
    VLLM = "vllm"          # Inference (1-4 GPUs)


@dataclass
class ActorEntry:
    """Entry in the unified resource pool."""

    actor_name: str
    actor_type: ActorType
    num_gpus: int
    actor_handle: ray.actor.ActorHandle | None = None
    namespace: str = "tinker"
    # For training actors: model path; for vLLM: model path
    base_model: str = ""
    # Session tracking (training actors only)
    current_session: str | None = None
    # LRU tracking
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        """Update last_accessed timestamp."""
        self.last_accessed = time.time()

    def is_idle(self) -> bool:
        """Check if actor can be evicted.

        Training actors: idle when no active session.
        vLLM: always considered evictable (will be recreated on demand).
        """
        if self.actor_type == ActorType.VLLM:
            return True  # vLLM can always be evicted
        return self.current_session is None

    def age(self) -> float:
        """Seconds since actor was created."""
        return time.time() - self.created_at

    def idle_time(self) -> float:
        """Seconds since last access."""
        return time.time() - self.last_accessed


class ResourcePool:
    """Unified pool managing all GPU-using actors with LRU eviction.

    Thread-safe singleton that tracks:
    - MegatronWorkerGroup actors (MoE training)
    - TrainingWorker actors (dense training)
    - vLLM inference actors

    When GPUs are needed, evicts LRU idle actors regardless of type.
    """

    _instance: "ResourcePool | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._entries: dict[str, ActorEntry] = {}  # key: actor_name
        self._pool_lock = threading.Lock()
        # Read MIN_ACTOR_AGE at init time (not class definition time)
        # Set MINT_MIN_ACTOR_AGE=0 for testing to allow immediate eviction
        self.MIN_ACTOR_AGE = int(os.environ.get("MINT_MIN_ACTOR_AGE", "300"))
        logger.info(f"[ResourcePool] Initialized with MIN_ACTOR_AGE={self.MIN_ACTOR_AGE}")
        self._initialized = True

    def register(
        self,
        actor_name: str,
        actor_type: ActorType,
        num_gpus: int,
        actor_handle: ray.actor.ActorHandle | None = None,
        namespace: str = "tinker",
        base_model: str = "",
        session_id: str | None = None,
    ) -> ActorEntry:
        """Register an actor with the pool.

        Args:
            actor_name: Unique name of the Ray actor.
            actor_type: Type of actor (MEGATRON, DENSE, VLLM).
            num_gpus: Number of GPUs used by this actor.
            actor_handle: Ray actor handle (optional, can be looked up later).
            namespace: Ray namespace.
            base_model: Model path being used.
            session_id: Active session ID (for training actors).

        Returns:
            The registered ActorEntry.
        """
        with self._pool_lock:
            if actor_name in self._entries:
                # Update existing entry
                entry = self._entries[actor_name]
                entry.touch()
                if session_id:
                    entry.current_session = session_id
                if actor_handle:
                    entry.actor_handle = actor_handle
                logger.info(f"[ResourcePool] Updated existing entry: {actor_name}")
            else:
                entry = ActorEntry(
                    actor_name=actor_name,
                    actor_type=actor_type,
                    num_gpus=num_gpus,
                    actor_handle=actor_handle,
                    namespace=namespace,
                    base_model=base_model,
                    current_session=session_id,
                )
                self._entries[actor_name] = entry
                logger.info(
                    f"[ResourcePool] Registered {actor_type.value} actor: {actor_name} "
                    f"({num_gpus} GPUs, model={base_model})"
                )
            return entry

    def unregister(self, actor_name: str) -> bool:
        """Remove an actor from the pool.

        Returns:
            True if actor was found and removed.
        """
        with self._pool_lock:
            if actor_name in self._entries:
                del self._entries[actor_name]
                logger.info(f"[ResourcePool] Unregistered actor: {actor_name}")
                return True
            return False

    def get(self, actor_name: str) -> ActorEntry | None:
        """Get an actor entry and update its LRU timestamp."""
        with self._pool_lock:
            entry = self._entries.get(actor_name)
            if entry:
                entry.touch()
            return entry

    def set_session(self, actor_name: str, session_id: str | None):
        """Update the active session for a training actor."""
        with self._pool_lock:
            entry = self._entries.get(actor_name)
            if entry:
                entry.current_session = session_id
                entry.touch()

    def _get_evictable_actors_lru(self) -> list[ActorEntry]:
        """Get evictable actors sorted by last access time (LRU first).

        Must be called with lock held.

        An actor is evictable if:
        1. It is idle (no active session)
        2. It has been idle longer than MIN_ACTOR_AGE

        Note: We use idle_time() (time since last access) rather than age()
        (time since creation) to protect recently-active actors from eviction.
        """
        evictable = [
            e for e in self._entries.values()
            if e.is_idle() and e.idle_time() > self.MIN_ACTOR_AGE
        ]
        return sorted(evictable, key=lambda e: e.last_accessed)

    def _kill_actor(self, entry: ActorEntry) -> bool:
        """Kill a Ray actor.

        Returns:
            True if actor was killed successfully.
        """
        try:
            # Get actor handle if we don't have it
            actor = entry.actor_handle
            if actor is None:
                try:
                    actor = ray.get_actor(entry.actor_name, namespace=entry.namespace)
                except ValueError:
                    logger.warning(f"[ResourcePool] Actor not found: {entry.actor_name}")
                    return False

            # Try graceful shutdown first
            try:
                if hasattr(actor, 'shutdown'):
                    ray.get(actor.shutdown.remote(), timeout=10)
            except Exception:
                pass

            # Force kill
            ray.kill(actor)
            logger.info(f"[ResourcePool] Killed actor: {entry.actor_name}")
            return True

        except Exception as e:
            logger.warning(f"[ResourcePool] Error killing actor {entry.actor_name}: {e}")
            return False

    def evict_for_gpus(self, needed_gpus: int) -> int:
        """Evict LRU idle actors to free GPUs.

        Args:
            needed_gpus: Number of GPUs to free.

        Returns:
            Number of GPUs freed.
        """
        freed_gpus = 0

        with self._pool_lock:
            evictable = self._get_evictable_actors_lru()

            for entry in evictable:
                if freed_gpus >= needed_gpus:
                    break

                logger.info(
                    f"[ResourcePool] Evicting {entry.actor_type.value} actor: {entry.actor_name} "
                    f"(idle {entry.idle_time():.1f}s, frees {entry.num_gpus} GPUs)"
                )

                if self._kill_actor(entry):
                    freed_gpus += entry.num_gpus
                    del self._entries[entry.actor_name]

        return freed_gpus

    def ensure_gpus_available(self, needed_gpus: int) -> bool:
        """Ensure at least needed_gpus are available, evicting if necessary.

        Args:
            needed_gpus: Number of GPUs needed.

        Returns:
            True if enough GPUs are available.

        Raises:
            ValueError: If unable to free enough GPUs.
        """
        available = int(ray.available_resources().get("GPU", 0))

        if available >= needed_gpus:
            logger.debug(f"[ResourcePool] Sufficient GPUs: {available} >= {needed_gpus}")
            return True

        need_to_free = needed_gpus - available
        logger.info(
            f"[ResourcePool] Need {needed_gpus} GPUs, only {available} available. "
            f"Evicting LRU actors to free {need_to_free} GPUs."
        )

        freed = self.evict_for_gpus(need_to_free)

        # Wait for Ray to reclaim resources
        time.sleep(2)
        available = int(ray.available_resources().get("GPU", 0))

        if available >= needed_gpus:
            logger.info(f"[ResourcePool] After eviction: {available} GPUs available")
            return True

        raise ValueError(
            f"Insufficient GPUs: need {needed_gpus}, available {available} after eviction. "
            f"Freed {freed} GPUs but may have resource fragmentation. "
            f"Check cluster status with 'ray status'."
        )

    def list_actors(self) -> list[dict]:
        """List all tracked actors (validates liveness)."""
        with self._pool_lock:
            # Validate actors still exist in Ray, remove stale entries
            # Only check if Ray is initialized
            stale = []
            if ray.is_initialized():
                for name, e in self._entries.items():
                    try:
                        ray.get_actor(e.actor_name, namespace=e.namespace)
                    except ValueError as ex:
                        logger.warning(f"[ResourcePool] Actor {name} not found in Ray: {ex}")
                        stale.append(name)
                    except Exception as ex:
                        logger.warning(f"[ResourcePool] Error checking actor {name}: {ex}")
                for name in stale:
                    logger.info(f"[ResourcePool] Removing stale actor: {name}")
                    del self._entries[name]

            return [
                {
                    "actor_name": e.actor_name,
                    "actor_type": e.actor_type.value,
                    "num_gpus": e.num_gpus,
                    "base_model": e.base_model,
                    "current_session": e.current_session,
                    "idle": e.is_idle(),
                    "idle_time": e.idle_time(),
                    "age": e.age(),
                }
                for e in self._entries.values()
            ]

    def total_gpus_used(self) -> int:
        """Total GPUs used by all tracked actors."""
        with self._pool_lock:
            return sum(e.num_gpus for e in self._entries.values())

    def clear(self, kill_actors: bool = True) -> int:
        """Remove all actors from the pool.

        Args:
            kill_actors: If True, also kill the Ray actors.

        Returns:
            Number of actors removed.
        """
        with self._pool_lock:
            count = len(self._entries)
            if kill_actors:
                for entry in list(self._entries.values()):
                    self._kill_actor(entry)
            self._entries.clear()
            logger.info(f"[ResourcePool] Cleared {count} actors")
            return count


# Global singleton accessor
def get_resource_pool() -> ResourcePool:
    """Get the global ResourcePool instance."""
    return ResourcePool()
