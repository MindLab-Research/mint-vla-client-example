"""Storage for async operation results.

Maps request_id to results for the async polling pattern:
1. Client sends request, gets request_id
2. Server processes in background
3. Client polls with request_id until result ready
"""

import os
import time
import uuid
from enum import Enum
from typing import Any


class FutureStatus(Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class FutureStore:
    """Thread-safe storage for async operation results."""

    def __init__(self):
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._pending: set[str] = set()

    def create(self) -> str:
        """Create a new pending future and return its request_id."""
        request_id = str(uuid.uuid4())
        self._pending.add(request_id)
        return request_id

    def resolve(self, request_id: str, result: Any) -> None:
        """Mark a future as completed with the given result."""
        self._pending.discard(request_id)
        self._results[request_id] = result

    def fail(self, request_id: str, error: str) -> None:
        """Mark a future as failed with the given error message."""
        self._pending.discard(request_id)
        self._errors[request_id] = error

    def get_status(self, request_id: str) -> FutureStatus:
        """Get the status of a future.

        Raises KeyError if request_id is unknown.
        """
        if request_id in self._results:
            return FutureStatus.DONE
        if request_id in self._errors:
            return FutureStatus.FAILED
        if request_id in self._pending:
            return FutureStatus.PENDING
        raise KeyError(f"Unknown request_id: {request_id}")

    def get_result(self, request_id: str) -> Any:
        """Get the result of a completed future, or None if not done."""
        return self._results.get(request_id)

    def get_error(self, request_id: str) -> str | None:
        """Get the error message of a failed future, or None if not failed."""
        return self._errors.get(request_id)

    def cleanup(self, request_id: str) -> None:
        """Remove a future from storage after retrieval."""
        self._pending.discard(request_id)
        self._results.pop(request_id, None)
        self._errors.pop(request_id, None)


class _RayFutureStoreActor:
    """Ray-backed future store shared across API server processes/replicas.

    Stored in a named, detached actor in the configured Ray namespace.
    """

    def __init__(self, ttl_s: float | None):
        self._ttl_s = ttl_s if ttl_s and ttl_s > 0 else None
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._pending: set[str] = set()
        self._created_at: dict[str, float] = {}

    def _gc(self) -> None:
        if self._ttl_s is None:
            return
        if not self._created_at:
            return
        now = time.time()
        dead = [rid for rid, ts in self._created_at.items() if now - ts > self._ttl_s]
        for rid in dead:
            self._created_at.pop(rid, None)
            self._pending.discard(rid)
            self._results.pop(rid, None)
            self._errors.pop(rid, None)

    def create(self) -> str:
        rid = str(uuid.uuid4())
        self._pending.add(rid)
        self._created_at[rid] = time.time()
        return rid

    def resolve(self, request_id: str, result: Any) -> None:
        self._pending.discard(request_id)
        self._results[request_id] = result

    def fail(self, request_id: str, error: str) -> None:
        self._pending.discard(request_id)
        self._errors[request_id] = error

    def get_status(self, request_id: str) -> str:
        self._gc()
        if request_id in self._results:
            return FutureStatus.DONE.value
        if request_id in self._errors:
            return FutureStatus.FAILED.value
        if request_id in self._pending:
            return FutureStatus.PENDING.value
        raise KeyError(f"Unknown request_id: {request_id}")

    def get_result(self, request_id: str) -> Any:
        self._gc()
        return self._results.get(request_id)

    def get_error(self, request_id: str) -> str | None:
        self._gc()
        return self._errors.get(request_id)

    def cleanup(self, request_id: str) -> None:
        self._created_at.pop(request_id, None)
        self._pending.discard(request_id)
        self._results.pop(request_id, None)
        self._errors.pop(request_id, None)


class DistributedFutureStore:
    """FutureStore interface backed by a shared Ray named actor.

    Falls back to process-local FutureStore if Ray is unavailable.
    """

    def __init__(self):
        self._local = FutureStore()
        self._actor = None

    def _get_actor(self):
        if self._actor is not None:
            return self._actor
        ttl_s = float(os.environ.get("TINKER_FUTURE_TTL_S", "900"))
        name = os.environ.get("TINKER_FUTURE_ACTOR_NAME", "tinker_future_store")

        try:
            import ray
            from ..config import RAY_NAMESPACE
            from ..ray_utils import init_ray

            if not ray.is_initialized():
                init_ray(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True)

            try:
                self._actor = ray.get_actor(name, namespace=RAY_NAMESPACE)
            except Exception:
                Actor = ray.remote(_RayFutureStoreActor)
                self._actor = Actor.options(
                    name=name,
                    namespace=RAY_NAMESPACE,
                    lifetime="detached",
                    num_cpus=0,
                ).remote(ttl_s=ttl_s if ttl_s > 0 else None)
            return self._actor
        except Exception:
            return None

    def create(self) -> str:
        actor = self._get_actor()
        if actor is None:
            return self._local.create()
        import ray

        return str(ray.get(actor.create.remote()))

    def resolve(self, request_id: str, result: Any) -> None:
        actor = self._get_actor()
        if actor is None:
            return self._local.resolve(request_id, result)
        import ray

        ray.get(actor.resolve.remote(request_id, result))

    def fail(self, request_id: str, error: str) -> None:
        actor = self._get_actor()
        if actor is None:
            return self._local.fail(request_id, error)
        import ray

        ray.get(actor.fail.remote(request_id, error))

    def get_status(self, request_id: str) -> FutureStatus:
        actor = self._get_actor()
        if actor is None:
            return self._local.get_status(request_id)
        import ray

        v = ray.get(actor.get_status.remote(request_id))
        return FutureStatus(str(v))

    def get_result(self, request_id: str) -> Any:
        actor = self._get_actor()
        if actor is None:
            return self._local.get_result(request_id)
        import ray

        return ray.get(actor.get_result.remote(request_id))

    def get_error(self, request_id: str) -> str | None:
        actor = self._get_actor()
        if actor is None:
            return self._local.get_error(request_id)
        import ray

        return ray.get(actor.get_error.remote(request_id))

    def cleanup(self, request_id: str) -> None:
        actor = self._get_actor()
        if actor is None:
            return self._local.cleanup(request_id)
        import ray

        ray.get(actor.cleanup.remote(request_id))


# Global instance
future_store = DistributedFutureStore()
