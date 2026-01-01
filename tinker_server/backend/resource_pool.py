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
    # Node tracking for scheduling
    node_id: str | None = None
    # LRU tracking
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    # Protection flag: actor being created/initialized is not evictable
    # Set to False after initialization completes
    creating: bool = True

    def touch(self):
        """Update last_accessed timestamp."""
        self.last_accessed = time.time()

    def mark_ready(self):
        """Mark actor as ready (no longer creating)."""
        self.creating = False
        self.touch()

    def is_idle(self, session_idle_timeout: float = 300) -> bool:
        """Check if actor can be evicted.

        An actor is idle if ALL of the following:
        1. It's not currently being created/initialized
        2. AND one of:
           a. It has no current_session (training actors only)
           b. It hasn't been accessed in session_idle_timeout seconds

        Note: Tinker clients don't explicitly end sessions - they just stop
        sending requests. We use time-based idle detection to handle this.
        """
        # Actors being created are NEVER idle
        if self.creating:
            return False
        if self.actor_type == ActorType.VLLM:
            # vLLM uses same idle timeout as other actors to prevent eviction during active use
            return self.idle_time() > session_idle_timeout
        if self.current_session is None:
            return True  # No session loaded
        # Time-based idle detection for sessions
        return self.idle_time() > session_idle_timeout

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
        # Pending GPU reservations - GPUs reserved but not yet allocated to actors
        # This prevents race conditions where multiple concurrent requests
        # both think they have enough GPUs available
        self._pending_gpus: int = 0
        # Read MIN_ACTOR_AGE at init time (not class definition time)
        # Set MINT_MIN_ACTOR_AGE=0 for testing to allow immediate eviction
        self.MIN_ACTOR_AGE = int(os.environ.get("MINT_MIN_ACTOR_AGE", "300"))
        # Session idle timeout: after this period of inactivity, session is considered stale
        # Set MINT_SESSION_IDLE_TIMEOUT=0 for testing to allow immediate eviction
        self.SESSION_IDLE_TIMEOUT = int(os.environ.get("MINT_SESSION_IDLE_TIMEOUT", "300"))
        logger.info(f"[ResourcePool] Initialized with MIN_ACTOR_AGE={self.MIN_ACTOR_AGE}, SESSION_IDLE_TIMEOUT={self.SESSION_IDLE_TIMEOUT}")
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
        node_id: str | None = None,
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
            node_id: Ray node ID where actor is running.

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
                if node_id:
                    entry.node_id = node_id
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
                    node_id=node_id,
                )
                self._entries[actor_name] = entry
                logger.info(
                    f"[ResourcePool] Registered {actor_type.value} actor: {actor_name} "
                    f"({num_gpus} GPUs, model={base_model}, node={node_id[:8] if node_id else 'unknown'})"
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

    def touch(self, actor_name: str) -> bool:
        """Update last_accessed timestamp to mark actor as recently used.

        Called during training operations to prevent eviction of active actors.
        Without this, actors are evicted based on creation time rather than
        actual usage, causing unexpected termination of active training.

        Returns:
            True if actor was found and touched.
        """
        with self._pool_lock:
            entry = self._entries.get(actor_name)
            if entry:
                entry.touch()
                return True
            return False

    def mark_ready(self, actor_name: str):
        """Mark an actor as ready (no longer creating).

        Call this after actor initialization completes to allow LRU eviction.
        """
        with self._pool_lock:
            entry = self._entries.get(actor_name)
            if entry:
                entry.mark_ready()
                logger.info(f"[ResourcePool] Actor {actor_name} marked ready")

    def reserve_gpus(self, num_gpus: int) -> bool:
        """Reserve GPUs before actor creation.

        This prevents race conditions where multiple concurrent requests
        both pass availability checks before either creates an actor.

        Args:
            num_gpus: Number of GPUs to reserve.

        Returns:
            True if reservation was made (caller should release after actor creation).
        """
        with self._pool_lock:
            self._pending_gpus += num_gpus
            logger.info(f"[ResourcePool] Reserved {num_gpus} GPUs (pending total: {self._pending_gpus})")
            return True

    def release_pending_gpus(self, num_gpus: int):
        """Release pending GPU reservation.

        Call this after actor creation completes (success or failure).
        On success, the GPUs are now tracked by the registered actor.
        On failure, the reservation is simply released.

        Args:
            num_gpus: Number of GPUs to release from pending.
        """
        with self._pool_lock:
            self._pending_gpus = max(0, self._pending_gpus - num_gpus)
            logger.info(f"[ResourcePool] Released {num_gpus} pending GPUs (pending total: {self._pending_gpus})")

    def get_effective_available_gpus(self) -> int:
        """Get available GPUs minus pending reservations.

        This is the correct number to check when deciding if there are
        enough GPUs for a new actor.

        Returns:
            Number of GPUs available for allocation.
        """
        ray_available = int(ray.available_resources().get("GPU", 0))
        with self._pool_lock:
            effective = ray_available - self._pending_gpus
        return max(0, effective)

    def _get_evictable_actors_lru(self) -> list[ActorEntry]:
        """Get evictable actors sorted by last access time (LRU first).

        Must be called with lock held.

        An actor is evictable if:
        1. It is idle (no active session OR session inactive > SESSION_IDLE_TIMEOUT)
        2. It has been idle longer than MIN_ACTOR_AGE

        Note: We use idle_time() (time since last access) rather than age()
        (time since creation) to protect recently-active actors from eviction.
        """
        evictable = [
            e for e in self._entries.values()
            if e.is_idle(self.SESSION_IDLE_TIMEOUT) and e.idle_time() > self.MIN_ACTOR_AGE
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

    def ensure_gpus_available(self, needed_gpus: int, timeout: float = 600) -> bool:
        """Ensure at least needed_gpus are available, evicting and waiting if necessary.

        Uses get_effective_available_gpus() which accounts for pending reservations
        from other concurrent requests that haven't yet created their actors.

        Args:
            needed_gpus: Number of GPUs needed.
            timeout: Maximum time to wait for resources (seconds). Default 10 minutes.

        Returns:
            True if enough GPUs are available.

        Raises:
            ValueError: If unable to free enough GPUs within timeout.
        """
        import time as time_module

        start_time = time_module.time()
        poll_interval = 5  # seconds between checks
        iteration = 0

        logger.info(f"[ResourcePool] ensure_gpus_available: need {needed_gpus} GPUs, timeout={timeout}s")

        while True:
            iteration += 1
            # Use effective available GPUs (accounts for pending reservations)
            available = self.get_effective_available_gpus()

            if available >= needed_gpus:
                logger.info(f"[ResourcePool] Sufficient GPUs: {available} >= {needed_gpus}")
                return True

            need_to_free = needed_gpus - available

            # Log actor states for debugging
            evictable_list = self._get_evictable_actors_lru()
            with self._pool_lock:
                all_actors = [(e.actor_name, e.is_idle(self.SESSION_IDLE_TIMEOUT), e.idle_time(), e.creating)
                              for e in self._entries.values()]
                pending = self._pending_gpus
            logger.info(
                f"[ResourcePool] Iteration {iteration}: need {needed_gpus} GPUs, available {available}, "
                f"pending {pending}, need_to_free {need_to_free}, evictable={len(evictable_list)}, "
                f"all_actors={all_actors}"
            )

            if evictable_list:
                logger.info(
                    f"[ResourcePool] Evicting {len(evictable_list)} actors: "
                    f"{[(e.actor_name, e.num_gpus) for e in evictable_list]}"
                )

            freed = self.evict_for_gpus(need_to_free)

            if freed > 0:
                # Wait for Ray to reclaim resources
                time_module.sleep(2)
                available = self.get_effective_available_gpus()

                if available >= needed_gpus:
                    logger.info(f"[ResourcePool] After eviction: {available} GPUs available")
                    return True

            # Check timeout
            elapsed = time_module.time() - start_time
            if elapsed >= timeout:
                logger.error(
                    f"[ResourcePool] TIMEOUT: need {needed_gpus} GPUs, available {available}, "
                    f"elapsed {elapsed:.1f}s >= timeout {timeout}s"
                )
                raise ValueError(
                    f"Insufficient GPUs: need {needed_gpus}, available {available} after eviction. "
                    f"Freed {freed} GPUs but resources did not become available within {timeout}s timeout. "
                    f"Other actors may be in use. Check cluster status with 'ray status'."
                )

            # Wait before retrying - resources may become available when other actors finish
            remaining = timeout - elapsed
            wait_time = min(poll_interval, remaining)
            if iteration % 10 == 0:  # Log every 10 iterations (~50s)
                logger.info(
                    f"[ResourcePool] Waiting for resources... "
                    f"(iteration={iteration}, available={available}, needed={needed_gpus}, "
                    f"elapsed={elapsed:.1f}s, timeout={timeout}s)"
                )
            time_module.sleep(wait_time)

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
                    "node_id": e.node_id,
                    "creating": e.creating,
                    "idle": e.is_idle(self.SESSION_IDLE_TIMEOUT),
                    "idle_time": e.idle_time(),
                    "age": e.age(),
                }
                for e in self._entries.values()
            ]

    def iter_entries(self) -> list[ActorEntry]:
        """Return list of ActorEntry objects (for internal use).

        Unlike list_actors() which returns dicts, this returns the actual
        ActorEntry objects for operations that need the full object.
        """
        with self._pool_lock:
            return list(self._entries.values())

    def total_gpus_used(self) -> int:
        """Total GPUs used by all tracked actors."""
        with self._pool_lock:
            return sum(e.num_gpus for e in self._entries.values())

    def gpus_used_by_node(self) -> dict[str, int]:
        """Get GPU usage per node from tracked actors.

        For actors without node_id, tries to look it up via Ray state API.

        Returns:
            Dict mapping node_id -> number of GPUs used by actors on that node.
            Actors with unknown node_id are excluded.
        """
        with self._pool_lock:
            usage: dict[str, int] = {}
            for e in self._entries.values():
                node_id = e.node_id
                # Try to look up node_id if missing
                if not node_id and e.actor_handle:
                    node_id = self._get_actor_node_id(e.actor_handle)
                    if node_id:
                        e.node_id = node_id  # Cache it for future calls
                        logger.debug(f"[ResourcePool] Resolved node_id for {e.actor_name}: {node_id[:8]}")
                if node_id:
                    usage[node_id] = usage.get(node_id, 0) + e.num_gpus
            return usage

    def _get_actor_node_id(self, actor_handle: ray.actor.ActorHandle) -> str | None:
        """Get node_id where an actor is running via Ray state API."""
        try:
            actor_id = actor_handle._actor_id
            # Convert ActorID to hex string for the state API
            actor_id_hex = actor_id.hex()
            from ray._private.state import actors as state_actors
            actor_info = state_actors(actor_id_hex)
            if actor_info:
                # NodeID is nested under Address
                address = actor_info.get("Address", {})
                return address.get("NodeID")
        except Exception as e:
            logger.debug(f"[ResourcePool] Could not get node_id: {e}")
        return None

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
