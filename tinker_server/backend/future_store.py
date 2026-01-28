"""Storage for async operation results.

Maps request_id to results for the async polling pattern:
1. Client sends request, gets request_id
2. Server processes in background
3. Client polls with request_id until result ready

Issue #85: process-local in-memory futures can produce 404 "Unknown request_id"
if the create and retrieve requests land on different server processes/replicas.
We default to a detached Ray actor backend when Ray is available.
"""

from __future__ import annotations

import os
import time
import uuid
from enum import Enum
from typing import Any


class FutureStatus(Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class _InMemoryFutureStore:
    def __init__(self) -> None:
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._pending: set[str] = set()

    def add_pending(self, request_id: str) -> None:
        self._pending.add(request_id)

    def resolve(self, request_id: str, result: Any) -> None:
        self._pending.discard(request_id)
        self._results[request_id] = result

    def fail(self, request_id: str, error: str) -> None:
        self._pending.discard(request_id)
        self._errors[request_id] = error

    def get_status(self, request_id: str) -> FutureStatus:
        if request_id in self._results:
            return FutureStatus.DONE
        if request_id in self._errors:
            return FutureStatus.FAILED
        if request_id in self._pending:
            return FutureStatus.PENDING
        raise KeyError(f"Unknown request_id: {request_id}")

    def get_result(self, request_id: str) -> Any:
        return self._results.get(request_id)

    def get_error(self, request_id: str) -> str | None:
        return self._errors.get(request_id)

    def cleanup(self, request_id: str) -> None:
        self._pending.discard(request_id)
        self._results.pop(request_id, None)
        self._errors.pop(request_id, None)


def _ray_namespace() -> str:
    return os.environ.get("MINT_RAY_NAMESPACE", "tinker")


def _ray_future_store_actor_name() -> str:
    return os.environ.get("MINT_FUTURE_STORE_ACTOR_NAME", "tinker_future_store")


def _ray_future_ttl_s() -> float:
    # Safety net for leaked futures when clients never retrieve.
    return float(os.environ.get("MINT_FUTURE_TTL_S", "86400"))


def _ray_future_done_ttl_s() -> float:
    # Keep DONE/FAILED entries briefly for idempotent retries.
    return float(os.environ.get("MINT_FUTURE_DONE_TTL_S", "300"))


def _get_or_create_ray_actor():
    import ray

    actor_name = _ray_future_store_actor_name()
    namespace = _ray_namespace()
    ttl_s = _ray_future_ttl_s()
    done_ttl_s = _ray_future_done_ttl_s()

    try:
        return ray.get_actor(actor_name, namespace=namespace)
    except ValueError:
        pass

    @ray.remote
    class _RayFutureStoreActor:
        def __init__(self, ttl_s: float, done_ttl_s: float) -> None:
            self._pending: set[str] = set()
            self._results: dict[str, Any] = {}
            self._errors: dict[str, str] = {}
            self._refs: dict[str, Any] = {}
            self._meta: dict[str, dict[str, Any]] = {}
            self._created_at: dict[str, float] = {}
            self._done_at: dict[str, float] = {}
            self._ttl_s = float(ttl_s)
            self._done_ttl_s = float(done_ttl_s)

        def _prune(self) -> int:
            now = time.time()
            removed = 0

            if self._ttl_s > 0:
                expired_pending = [
                    rid for rid, ts in self._created_at.items()
                    if rid in self._pending and (now - ts) > self._ttl_s
                ]
                for rid in expired_pending:
                    self._pending.discard(rid)
                    self._refs.pop(rid, None)
                    self._meta.pop(rid, None)
                    self._created_at.pop(rid, None)
                    removed += 1

            if self._done_ttl_s > 0:
                expired_done = [
                    rid for rid, ts in self._done_at.items()
                    if (now - ts) > self._done_ttl_s
                ]
                for rid in expired_done:
                    self._results.pop(rid, None)
                    self._errors.pop(rid, None)
                    self._refs.pop(rid, None)
                    self._meta.pop(rid, None)
                    self._created_at.pop(rid, None)
                    self._done_at.pop(rid, None)
                    removed += 1

            return removed

        def add_pending(self, request_id: str) -> None:
            self._prune()
            self._pending.add(request_id)
            self._created_at[request_id] = time.time()

        def attach_ref(self, request_id: str, ref: Any, meta: dict[str, Any] | None = None) -> None:
            self._prune()
            self._pending.add(request_id)
            self._created_at[request_id] = time.time()
            self._refs[request_id] = ref
            if meta is not None:
                self._meta[request_id] = dict(meta)

        def submit(
            self,
            request_id: str,
            target_actor: Any,
            method_name: str,
            args: list[Any] | dict[str, Any],
            meta: dict[str, Any] | None = None,
        ) -> None:
            self._prune()
            self._pending.add(request_id)
            self._created_at[request_id] = time.time()
            if meta is not None:
                self._meta[request_id] = dict(meta)
            method = getattr(target_actor, method_name)
            if isinstance(args, dict):
                self._refs[request_id] = method.remote(**args)
            else:
                self._refs[request_id] = method.remote(*args)

        def resolve(self, request_id: str, result: Any) -> None:
            self._prune()
            self._pending.discard(request_id)
            self._refs.pop(request_id, None)
            self._meta.pop(request_id, None)
            self._results[request_id] = result
            self._done_at[request_id] = time.time()

        def fail(self, request_id: str, error: str) -> None:
            self._prune()
            self._pending.discard(request_id)
            self._refs.pop(request_id, None)
            self._meta.pop(request_id, None)
            self._errors[request_id] = error
            self._done_at[request_id] = time.time()

        def get_status(self, request_id: str) -> str:
            self._prune()
            if request_id in self._results:
                return FutureStatus.DONE.value
            if request_id in self._errors:
                return FutureStatus.FAILED.value
            if request_id in self._refs:
                import ray

                ref = self._refs[request_id]
                ready, _ = ray.wait([ref], timeout=0)
                if ready:
                    meta = self._meta.get(request_id)
                    self._pending.discard(request_id)
                    try:
                        result = ray.get(ref)
                        if isinstance(meta, dict) and meta.get("op") == "optim_step":
                            model_id = meta.get("model_id")
                            if model_id and isinstance(result, dict):
                                try:
                                    from .training_session_store import _get_or_create_actor  # type: ignore

                                    store = _get_or_create_actor()
                                    step = ray.get(store.bump_step.remote(str(model_id)))
                                    metrics = result.get("metrics")
                                    if not isinstance(metrics, dict):
                                        metrics = {}
                                        result["metrics"] = metrics
                                    metrics["step"] = int(step)
                                except Exception:
                                    pass

                        self._results[request_id] = result
                        self._done_at[request_id] = time.time()
                        self._refs.pop(request_id, None)
                        self._meta.pop(request_id, None)
                        return FutureStatus.DONE.value
                    except Exception as e:
                        self._errors[request_id] = str(e)
                        self._done_at[request_id] = time.time()
                        self._refs.pop(request_id, None)
                        self._meta.pop(request_id, None)
                        return FutureStatus.FAILED.value
            if request_id in self._pending:
                return FutureStatus.PENDING.value
            raise KeyError(f"Unknown request_id: {request_id}")

        def get_result(self, request_id: str) -> Any:
            self._prune()
            if request_id in self._refs:
                self.get_status(request_id)
            return self._results.get(request_id)

        def get_error(self, request_id: str) -> str | None:
            self._prune()
            if request_id in self._refs:
                self.get_status(request_id)
            return self._errors.get(request_id)

        def get_meta(self, request_id: str) -> dict[str, Any] | None:
            self._prune()
            return self._meta.get(request_id)

        def cleanup(self, request_id: str) -> None:
            self._pending.discard(request_id)
            self._results.pop(request_id, None)
            self._errors.pop(request_id, None)
            self._refs.pop(request_id, None)
            self._meta.pop(request_id, None)
            self._created_at.pop(request_id, None)
            self._done_at.pop(request_id, None)

    try:
        return _RayFutureStoreActor.options(
            name=actor_name,
            namespace=namespace,
            lifetime="detached",
        ).remote(ttl_s, done_ttl_s)
    except Exception:
        # Race: another process may have created the actor between get_actor and create.
        return ray.get_actor(actor_name, namespace=namespace)


class FutureStore:
    """FutureStore with a detached Ray actor backend when available."""

    def __init__(self) -> None:
        self._local = _InMemoryFutureStore()
        self._ray_actor = None

    def _get_ray_actor(self):
        try:
            import ray
        except Exception:
            return None

        if not ray.is_initialized():
            return None

        if self._ray_actor is None:
            self._ray_actor = _get_or_create_ray_actor()
        return self._ray_actor

    def create(self) -> str:
        request_id = str(uuid.uuid4())
        actor = self._get_ray_actor()
        if actor is None:
            self._local.add_pending(request_id)
            return request_id

        import ray

        ray.get(actor.add_pending.remote(request_id))
        return request_id

    def resolve(self, request_id: str, result: Any) -> None:
        actor = self._get_ray_actor()
        if actor is None:
            self._local.resolve(request_id, result)
            return
        actor.resolve.remote(request_id, result)

    def fail(self, request_id: str, error: str) -> None:
        actor = self._get_ray_actor()
        if actor is None:
            self._local.fail(request_id, error)
            return
        actor.fail.remote(request_id, error)

    def get_status(self, request_id: str) -> FutureStatus:
        actor = self._get_ray_actor()
        if actor is None:
            return self._local.get_status(request_id)

        import ray

        status = ray.get(actor.get_status.remote(request_id))
        return FutureStatus(status)

    def get_result(self, request_id: str) -> Any:
        actor = self._get_ray_actor()
        if actor is None:
            return self._local.get_result(request_id)

        import ray

        return ray.get(actor.get_result.remote(request_id))

    def get_error(self, request_id: str) -> str | None:
        actor = self._get_ray_actor()
        if actor is None:
            return self._local.get_error(request_id)

        import ray

        return ray.get(actor.get_error.remote(request_id))

    def attach_ref(self, request_id: str, ref: Any, meta: dict[str, Any] | None = None) -> None:
        actor = self._get_ray_actor()
        if actor is None:
            raise RuntimeError("Ray not initialized: FutureStore.attach_ref requires Ray")

        import ray

        ray.get(actor.attach_ref.remote(request_id, ref, meta))

    def submit(
        self,
        request_id: str,
        target_actor: Any,
        method_name: str,
        args: list[Any] | dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> None:
        actor = self._get_ray_actor()
        if actor is None:
            raise RuntimeError("Ray not initialized: FutureStore.submit requires Ray")

        import ray

        ray.get(actor.submit.remote(request_id, target_actor, method_name, args, meta))

    def get_meta(self, request_id: str) -> dict[str, Any] | None:
        actor = self._get_ray_actor()
        if actor is None:
            return None

        import ray

        return ray.get(actor.get_meta.remote(request_id))

    def cleanup(self, request_id: str) -> None:
        actor = self._get_ray_actor()
        if actor is None:
            self._local.cleanup(request_id)
            return
        actor.cleanup.remote(request_id)


future_store = FutureStore()
