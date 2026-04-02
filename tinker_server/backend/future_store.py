"""Storage for async operation results.

Maps request_id to results for the async polling pattern:
1. Client sends request, gets request_id
2. Server processes in background
3. Client polls with request_id until result ready

Ray is a hard requirement: futures are stored in a detached Ray actor so they
survive multi-worker deployments without per-process state loss.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
from enum import Enum
from typing import Any

from .queue_execution_context import get_current_queue_generation_id
from ..config import config as server_config, otel_env_vars


class FutureStoreUnavailableError(RuntimeError):
    pass


class FutureStatus(Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRIEVED = "retrieved"


def _ray_namespace() -> str:
    v = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _ray_future_store_actor_name() -> str:
    return str(server_config.future_store_actor_name)


def _ray_future_ttl_s() -> float:
    # Execution timeout (seconds). Queue time does not count.
    return float(server_config.future_store_ttl_s)


def _ray_future_queue_ttl_s() -> float:
    # Maximum time (seconds) a request may remain QUEUED (not RUNNING) before being FAILED.
    return float(server_config.future_store_queue_ttl_s)


def _ray_future_done_ttl_s() -> float:
    # Retain DONE/FAILED results for this long, then transition to EXPIRED tombstone.
    return float(server_config.future_store_done_ttl_s)


def _ray_future_tombstone_ttl_s() -> float:
    # Keep EXPIRED/RETRIEVED tombstones briefly before forgetting request_id.
    return float(getattr(server_config, "future_store_tombstone_ttl_s", 300.0))


def _require_ray_address() -> str:
    from ..ray_utils import require_ray_address

    return require_ray_address()


def _is_training_step_op(op: Any) -> bool:
    return str(op or "") in {"training.optim_step", "training.train_step"}


def _extract_training_step(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None
    step = metrics.get("step")
    if isinstance(step, bool):
        return None
    if isinstance(step, int):
        return int(step)
    if isinstance(step, float) and step.is_integer():
        return int(step)
    return None


def _sync_training_session_step(meta: dict[str, Any] | None, result: Any) -> Any:
    if not isinstance(meta, dict) or not _is_training_step_op(meta.get("op")):
        return result
    model_id = meta.get("model_id")
    if not model_id:
        return result

    try:
        from .training_session_store import (
            bump_training_session_step_best_effort,
            set_training_session_step_best_effort,
        )

        step = _extract_training_step(result)
        if step is None:
            bump_training_session_step_best_effort(str(model_id))
            return result

        set_training_session_step_best_effort(str(model_id), int(step))
        if isinstance(result, dict):
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                metrics["step"] = int(step)
        return result
    except Exception:
        return result


async def _await_ray_ref(ref: Any) -> Any:
    """Await a Ray ObjectRef using native async integration."""
    if hasattr(ref, "__await__"):
        return await ref

    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, asyncio.Future):
            return await fut
        if isinstance(fut, concurrent.futures.Future):
            return await asyncio.wrap_future(fut)
        if hasattr(fut, "__await__"):
            return await fut

    raise TypeError(f"Ray ref is not awaitable: {type(ref)}")


def _get_or_create_ray_actor():
    import ray

    actor_name = _ray_future_store_actor_name()
    namespace = _ray_namespace()
    ttl_s = _ray_future_ttl_s()
    queue_ttl_s = _ray_future_queue_ttl_s()
    done_ttl_s = _ray_future_done_ttl_s()
    tombstone_ttl_s = _ray_future_tombstone_ttl_s()

    try:
        return ray.get_actor(actor_name, namespace=namespace)
    except ValueError:
        pass

    @ray.remote(num_cpus=0)
    class _RayFutureStoreActor:
        def __init__(
            self, ttl_s: float, queue_ttl_s: float, done_ttl_s: float, tombstone_ttl_s: float
        ) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._pending: set[str] = set()
            self._result_refs: dict[str, Any] = {}
            self._errors: dict[str, str] = {}
            self._refs: dict[str, Any] = {}
            self._meta: dict[str, dict[str, Any]] = {}
            self._op_by_request: dict[str, str] = {}

            self._created_at: dict[str, float] = {}
            self._queued_at: dict[str, float] = {}
            self._running_at: dict[str, float] = {}
            self._done_at: dict[str, float] = {}
            self._expired_at: dict[str, float] = {}
            self._retrieved_at: dict[str, float] = {}

            self._execution_timeout_s = float(ttl_s)
            self._queue_timeout_s = float(queue_ttl_s)
            self._result_ttl_s = float(done_ttl_s)
            self._tombstone_ttl_s = float(tombstone_ttl_s)
            self._timeout_counts: dict[str, int] = {"queue": 0, "execution": 0}
            self._timeout_counts_by_op: dict[str, dict[str, int]] = {}

        def _op_from_meta(self, meta: dict[str, Any] | None) -> str | None:
            if not isinstance(meta, dict):
                return None
            op = meta.get("op")
            if not isinstance(op, str):
                return None
            op = op.strip()
            return op or None

        def _update_op_from_meta(self, request_id: str, meta: dict[str, Any] | None) -> None:
            op = self._op_from_meta(meta)
            if op is not None:
                self._op_by_request[request_id] = op

        def _request_op(self, request_id: str) -> str:
            op = self._op_by_request.get(request_id)
            if isinstance(op, str) and op:
                return op
            op_from_meta = self._op_from_meta(self._meta.get(request_id))
            if op_from_meta is not None:
                self._op_by_request[request_id] = op_from_meta
                return op_from_meta
            return "unknown"

        def _stats_by_op(self) -> dict[str, dict[str, int]]:
            by_op: dict[str, dict[str, int]] = {}

            def _bump(op: str, key: str) -> None:
                bucket = by_op.setdefault(op, {"pending": 0, "results": 0, "errors": 0})
                bucket[key] = int(bucket.get(key, 0)) + 1

            for rid in self._pending:
                _bump(self._request_op(rid), "pending")
            for rid in self._result_refs:
                _bump(self._request_op(rid), "results")
            for rid in self._errors:
                _bump(self._request_op(rid), "errors")

            return by_op

        def _age_stats(self) -> dict[str, float]:
            now = time.time()
            pending_ages: list[float] = []
            for rid in self._pending:
                ts = self._created_at.get(rid)
                if ts is not None:
                    pending_ages.append(max(0.0, now - float(ts)))

            done_ids = set(self._result_refs.keys()) | set(self._errors.keys())
            done_ages: list[float] = []
            for rid in done_ids:
                ts = self._done_at.get(rid)
                if ts is None:
                    ts = self._created_at.get(rid)
                if ts is not None:
                    done_ages.append(max(0.0, now - float(ts)))

            return {
                "oldest_pending_s": max(pending_ages) if pending_ages else 0.0,
                "oldest_done_s": max(done_ages) if done_ages else 0.0,
                "avg_pending_s": (sum(pending_ages) / len(pending_ages)) if pending_ages else 0.0,
                "avg_done_s": (sum(done_ages) / len(done_ages)) if done_ages else 0.0,
            }

        def _payload_stats(self) -> dict[str, int]:
            return {
                "result_refs_count": len(self._result_refs),
                "errors_count": len(self._errors),
                "refs_count": len(self._refs),
            }

        def _record_timeout(self, request_id: str, *, kind: str) -> None:
            from ..logging_context import record_future_store_timeout_metric

            timeout_kind = str(kind).strip() or "unknown"
            self._timeout_counts[timeout_kind] = int(self._timeout_counts.get(timeout_kind, 0)) + 1
            op = self._request_op(request_id)
            bucket = self._timeout_counts_by_op.setdefault(op, {"queue": 0, "execution": 0})
            bucket[timeout_kind] = int(bucket.get(timeout_kind, 0)) + 1
            record_future_store_timeout_metric(kind=timeout_kind, op=op)

        def _timeout_stats(self) -> dict[str, Any]:
            queue_count = int(self._timeout_counts.get("queue", 0))
            execution_count = int(self._timeout_counts.get("execution", 0))
            by_op: dict[str, dict[str, int]] = {}
            for op, bucket in sorted(self._timeout_counts_by_op.items()):
                if not isinstance(bucket, dict):
                    continue
                queue_op = int(bucket.get("queue", 0))
                execution_op = int(bucket.get("execution", 0))
                by_op[op] = {
                    "queue": queue_op,
                    "execution": execution_op,
                    "total": queue_op + execution_op,
                }
            return {
                "queue": queue_count,
                "execution": execution_count,
                "total": queue_count + execution_count,
                "by_op": by_op,
            }

        def get_rss_bytes(self) -> int:
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) < 2:
                raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
            rss_pages = int(parts[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return rss_pages * page_size

        def stats(self) -> dict[str, Any]:
            self._prune()
            return {
                "pending": len(self._pending),
                "results": len(self._result_refs),
                "errors": len(self._errors),
                "refs": len(self._refs),
                "meta": len(self._meta),
                "expired": len(self._expired_at),
                "retrieved": len(self._retrieved_at),
                "execution_timeout_s": float(self._execution_timeout_s),
                "queue_timeout_s": float(self._queue_timeout_s),
                "result_ttl_s": float(self._result_ttl_s),
                "tombstone_ttl_s": float(self._tombstone_ttl_s),
                "by_op": self._stats_by_op(),
                "age_stats": self._age_stats(),
                "payload_stats": self._payload_stats(),
                "timeout_counts": self._timeout_stats(),
            }

        def _prune(self) -> dict[str, list[str]]:
            now = time.time()
            expired: list[str] = []
            timed_out: list[str] = []

            # Queue timeout applies only after QUEUED, before RUNNING.
            if self._queue_timeout_s > 0:
                for rid, ts in list(self._queued_at.items()):
                    if (
                        rid in self._pending
                        and rid not in self._running_at
                        and (now - ts) > self._queue_timeout_s
                    ):
                        self._pending.discard(rid)
                        self._refs.pop(rid, None)
                        self._result_refs.pop(rid, None)
                        self._errors[rid] = "queue timeout"
                        self._done_at[rid] = now
                        self._queued_at.pop(rid, None)
                        self._running_at.pop(rid, None)
                        self._record_timeout(rid, kind="queue")
                        timed_out.append(rid)

            # Execution timeout applies only once RUNNING begins.
            if self._execution_timeout_s > 0:
                for rid, ts in list(self._running_at.items()):
                    if rid in self._pending and (now - ts) > self._execution_timeout_s:
                        self._pending.discard(rid)
                        self._refs.pop(rid, None)
                        self._result_refs.pop(rid, None)
                        self._errors[rid] = "execution timeout"
                        self._done_at[rid] = now
                        self._running_at.pop(rid, None)
                        self._record_timeout(rid, kind="execution")
                        timed_out.append(rid)

            # Result retention TTL: DONE/FAILED become EXPIRED tombstones.
            if self._result_ttl_s > 0:
                for rid, ts in list(self._done_at.items()):
                    if rid in self._expired_at or rid in self._retrieved_at:
                        continue
                    if (now - ts) > self._result_ttl_s:
                        self._result_refs.pop(rid, None)
                        self._errors.pop(rid, None)
                        self._expired_at[rid] = now
                        expired.append(rid)

            # Tombstone TTL: forget request_id after explicit EXPIRED/RETRIEVED window.
            if self._tombstone_ttl_s > 0:
                for rid, ts in list(self._expired_at.items()):
                    if (now - ts) > self._tombstone_ttl_s:
                        self._forget(rid)
                for rid, ts in list(self._retrieved_at.items()):
                    if (now - ts) > self._tombstone_ttl_s:
                        self._forget(rid)

            return {"expired": expired, "timed_out": timed_out}

        def _forget(self, request_id: str) -> None:
            self._pending.discard(request_id)
            self._result_refs.pop(request_id, None)
            self._errors.pop(request_id, None)
            self._refs.pop(request_id, None)
            self._meta.pop(request_id, None)
            self._op_by_request.pop(request_id, None)
            self._created_at.pop(request_id, None)
            self._queued_at.pop(request_id, None)
            self._running_at.pop(request_id, None)
            self._done_at.pop(request_id, None)
            self._expired_at.pop(request_id, None)
            self._retrieved_at.pop(request_id, None)

        def add_pending(self, request_id: str) -> None:
            self._prune()
            self._pending.add(request_id)
            self._created_at[request_id] = time.time()

        def ensure_pending(self, request_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
            self._prune()
            exists = (
                request_id in self._pending
                or request_id in self._refs
                or request_id in self._result_refs
                or request_id in self._errors
                or request_id in self._expired_at
                or request_id in self._retrieved_at
            )
            existing_meta = self._meta.get(request_id)
            if exists:
                return {"created": False, "meta": existing_meta}
            self._pending.add(request_id)
            self._created_at[request_id] = time.time()
            if meta is not None:
                self._meta[request_id] = dict(meta)
            return {"created": True, "meta": existing_meta}

        def mark_queued(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
            self._prune()
            now = time.time()
            if request_id in self._pending:
                self._queued_at[request_id] = now
            m = self._meta.get(request_id) or {}
            if meta is not None:
                m.update(dict(meta))
            if "queue_state" not in m:
                m["queue_state"] = "queued"
            if "stage" not in m:
                m["stage"] = "queued"
            if not isinstance(m.get("queued_at"), (int, float)):
                m["queued_at"] = now
            self._meta[request_id] = m
            self._update_op_from_meta(request_id, m)

        def mark_running(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
            self._prune()
            if request_id in self._pending:
                self._running_at[request_id] = time.time()
            if meta is not None:
                m = self._meta.get(request_id) or {}
                m.update(dict(meta))
                self._meta[request_id] = m
                self._update_op_from_meta(request_id, m)

        def update_meta(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
            self._prune()
            if meta is None:
                return
            exists = (
                request_id in self._pending
                or request_id in self._refs
                or request_id in self._result_refs
                or request_id in self._errors
                or request_id in self._expired_at
                or request_id in self._retrieved_at
            )
            if not exists:
                return
            m = self._meta.get(request_id) or {}
            m.update(dict(meta))
            self._meta[request_id] = m
            self._update_op_from_meta(request_id, m)

        def attach_ref(self, request_id: str, ref: Any, meta: dict[str, Any] | None = None) -> None:
            self._prune()
            self._pending.add(request_id)
            self._created_at[request_id] = time.time()
            self._refs[request_id] = ref
            if meta is not None:
                self._meta[request_id] = dict(meta)
                self._update_op_from_meta(request_id, self._meta[request_id])

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
                self._update_op_from_meta(request_id, self._meta[request_id])
            method = getattr(target_actor, method_name)
            if isinstance(args, dict):
                self._refs[request_id] = method.remote(**args)
            else:
                self._refs[request_id] = method.remote(*args)

        def resolve(self, request_id: str, result: Any) -> None:
            self._prune()
            if (
                request_id in self._result_refs
                or request_id in self._errors
                or request_id in self._expired_at
                or request_id in self._retrieved_at
            ):
                return
            self._pending.discard(request_id)
            self._refs.pop(request_id, None)
            self._update_op_from_meta(request_id, self._meta.get(request_id))
            self._meta.pop(request_id, None)
            import ray

            self._result_refs[request_id] = ray.put(result)
            self._done_at[request_id] = time.time()

        def resolve_ref(self, request_id: str, ref: Any) -> None:
            self._prune()
            if (
                request_id in self._result_refs
                or request_id in self._errors
                or request_id in self._expired_at
                or request_id in self._retrieved_at
            ):
                return
            self._pending.discard(request_id)
            self._refs.pop(request_id, None)
            self._result_refs[request_id] = ref
            self._done_at[request_id] = time.time()

        def fail(self, request_id: str, error: str) -> None:
            self._prune()
            if (
                request_id in self._result_refs
                or request_id in self._errors
                or request_id in self._expired_at
                or request_id in self._retrieved_at
            ):
                return
            self._pending.discard(request_id)
            self._refs.pop(request_id, None)
            self._update_op_from_meta(request_id, self._meta.get(request_id))
            self._meta.pop(request_id, None)
            self._errors[request_id] = error
            self._done_at[request_id] = time.time()

        def get_status(self, request_id: str) -> str:
            self._prune()
            if request_id in self._retrieved_at:
                return FutureStatus.RETRIEVED.value
            if request_id in self._expired_at:
                return FutureStatus.EXPIRED.value
            if request_id in self._result_refs:
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
                    self._update_op_from_meta(request_id, meta)
                    try:
                        result = ray.get(ref)
                        result = _sync_training_session_step(meta, result)
                        self._result_refs[request_id] = ray.put(result)
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
            return self._result_refs.get(request_id)

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
            self._refs.pop(request_id, None)
            self._meta.pop(request_id, None)
            self._created_at.pop(request_id, None)
            self._queued_at.pop(request_id, None)
            self._running_at.pop(request_id, None)
            self._expired_at.pop(request_id, None)
            self._retrieved_at[request_id] = time.time()

        def fail_stale_running_requests(self, active_consumer_job_id: str, error: str) -> list[str]:
            self._prune()
            now = time.time()
            active = str(active_consumer_job_id)
            message = str(error)
            failed: list[str] = []
            for request_id in list(self._pending):
                meta = self._meta.get(request_id)
                if not isinstance(meta, dict):
                    continue
                if str(meta.get("queue_state") or "") != "running":
                    continue
                owner = str(meta.get("consumer_job_id") or "").strip()
                if not owner or owner == active:
                    continue
                self._pending.discard(request_id)
                self._refs.pop(request_id, None)
                self._update_op_from_meta(request_id, meta)
                self._meta.pop(request_id, None)
                self._errors[request_id] = message
                self._done_at[request_id] = now
                failed.append(str(request_id))
            return failed

        def fail_training_requests_for_model(self, model_id: str, error: str) -> list[str]:
            self._prune()
            target_model_id = str(model_id).strip()
            if not target_model_id:
                return []
            now = time.time()
            message = str(error)
            failed: list[str] = []
            for request_id in list(self._pending):
                meta = self._meta.get(request_id)
                if not isinstance(meta, dict):
                    continue
                if str(meta.get("model_id") or "").strip() != target_model_id:
                    continue
                op = self._op_from_meta(meta) or ""
                if not op.startswith("training."):
                    continue
                self._pending.discard(request_id)
                self._refs.pop(request_id, None)
                self._update_op_from_meta(request_id, meta)
                self._meta.pop(request_id, None)
                self._errors[request_id] = message
                self._done_at[request_id] = now
                failed.append(str(request_id))
            return failed

        def forget(self, request_id: str) -> None:
            self._forget(request_id)

        def reap(self) -> dict[str, list[str]]:
            # Return request_ids that transitioned to terminal tombstones and must
            # release any external reservations.
            return self._prune()

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    try:
        if "node:__internal_head__" in ray.cluster_resources():
            options["resources"] = {"node:__internal_head__": 0.001}
    except Exception:
        pass
    actor_otel_env = otel_env_vars()
    from ..config import PFS_PYTHONPATH, actor_runtime_env
    options["runtime_env"] = actor_runtime_env(
        pythonpath=PFS_PYTHONPATH,
        extra=actor_otel_env,
    )

    try:
        return _RayFutureStoreActor.options(
            **options
        ).remote(ttl_s, queue_ttl_s, done_ttl_s, tombstone_ttl_s)
    except Exception:
        # Race: another process may have created the actor between get_actor and create.
        return ray.get_actor(actor_name, namespace=namespace)


class FutureStore:
    """FutureStore backed by a detached Ray actor (hard requirement)."""

    def __init__(self) -> None:
        self._ray_actor = None

    def _get_cached_ray_actor_for_async_request_path(self):
        try:
            import ray
        except Exception as e:
            raise FutureStoreUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            raise FutureStoreUnavailableError("Ray not initialized")

        if self._ray_actor is None:
            raise FutureStoreUnavailableError(
                "Detached Ray FutureStore actor is not ready on this API server"
            )
        return self._ray_actor
    async def async_ensure_ready(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        """Async variant of ensure_ready for request/control-plane paths."""
        actor = await self._get_ray_actor_async()
        import ray

        try:
            out = await asyncio.wait_for(_await_ray_ref(actor.stats.remote()), timeout=float(timeout_s))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        if not isinstance(out, dict):
            raise TypeError(f"FutureStore.stats returned non-dict: {type(out)}")
        return out
    async def async_rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        try:
            v = await asyncio.wait_for(_await_ray_ref(actor.get_rss_bytes.remote()), timeout=float(timeout_s))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        return int(v)
    async def _get_ray_actor_async(self):
        try:
            import ray
        except Exception as e:
            raise FutureStoreUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            try:
                from ..ray_utils import init_ray

                init_ray(namespace=_ray_namespace(), ignore_reinit_error=True)
            except Exception as e:
                raise FutureStoreUnavailableError("Ray not initialized (init_ray failed)") from e

        if not ray.is_initialized():
            raise FutureStoreUnavailableError("Ray not initialized")

        actor = self._ray_actor
        if actor is not None:
            try:
                await asyncio.wait_for(_await_ray_ref(actor.stats.remote()), timeout=1.0)
                return actor
            except Exception:
                self._ray_actor = None

        try:
            self._ray_actor = _get_or_create_ray_actor()
        except Exception as e:
            raise FutureStoreUnavailableError("Failed to get/create detached Ray FutureStore actor") from e
        return self._ray_actor
    async def async_debug_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ray_namespace": _ray_namespace(),
            "ray_actor_name": _ray_future_store_actor_name(),
            "future_ttl_s": _ray_future_ttl_s(),
            "future_done_ttl_s": _ray_future_done_ttl_s(),
        }

        try:
            import ray  # type: ignore

            out["ray_initialized"] = bool(ray.is_initialized())
            if not ray.is_initialized():
                out["ray_address"] = _require_ray_address()
                return out

            actor = self._ray_actor
            if actor is None:
                out["ray_actor_get_error"] = "actor_handle_not_cached"
                return out

            try:
                out["ray_actor_stats"] = await asyncio.wait_for(
                    _await_ray_ref(actor.stats.remote()),
                    timeout=float(timeout_s),
                )
            except Exception as e:
                out["ray_actor_stats_error"] = f"{type(e).__name__}: {e}"
            return out
        except Exception as e:
            out["ray_import_error"] = f"{type(e).__name__}: {e}"
            return out
    async def async_create_with_id(self, request_id: str) -> str:
        actor = self._get_cached_ray_actor_for_async_request_path()

        import ray

        try:
            await _await_ray_ref(actor.add_pending.remote(request_id=str(request_id)))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        return str(request_id)
    async def async_ensure_pending(self, request_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        payload = None if meta is None else dict(meta)
        try:
            out = await _await_ray_ref(actor.ensure_pending.remote(request_id=str(request_id), meta=payload))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        if not isinstance(out, dict):
            raise TypeError(f"FutureStore.ensure_pending returned non-dict: {type(out)}")
        return out
    async def async_mark_queued(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        payload = None if meta is None else dict(meta)
        try:
            await _await_ray_ref(actor.mark_queued.remote(request_id=request_id, meta=payload))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
    async def async_mark_running(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        payload = None if meta is None else dict(meta)
        try:
            await _await_ray_ref(actor.mark_running.remote(request_id=request_id, meta=payload))
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from None
    async def async_update_meta(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        payload = None if meta is None else dict(meta)
        try:
            await _await_ray_ref(actor.update_meta.remote(request_id=request_id, meta=payload))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e

    async def _async_stale_generation_finalize_guard(self) -> tuple[bool, str | None]:
        generation_id = get_current_queue_generation_id()
        if generation_id is None:
            return False, None
        try:
            from .queue_supervisor import queue_supervisor

            if await queue_supervisor.async_is_generation_current(generation_id=int(generation_id)):
                return False, None
            return True, f"stale generation finalize rejected (generation_id={generation_id})"
        except Exception as e:
            return True, f"stale generation finalize check failed: {type(e).__name__}: {e}"

    async def async_resolve(self, request_id: str, result: Any) -> None:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        stale, message = await self._async_stale_generation_finalize_guard()
        try:
            if stale:
                await _await_ray_ref(actor.fail.remote(request_id=request_id, error=str(message)))
                return
            meta = await _await_ray_ref(actor.get_meta.remote(request_id=request_id))
            result = _sync_training_session_step(meta, result)
            ref = ray.put(result)
            await _await_ray_ref(actor.resolve_ref.remote(request_id=request_id, ref=ref))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
    async def async_fail(self, request_id: str, error: str) -> None:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        stale, message = await self._async_stale_generation_finalize_guard()
        try:
            await _await_ray_ref(actor.fail.remote(request_id=request_id, error=str(message if stale and message else error)))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        try:
            from .capacity_manager import capacity_manager

            await capacity_manager.async_release_object_store(request_id)
        except Exception:
            pass
    async def async_fail_training_requests_for_model(self, model_id: str, error: str) -> list[str]:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        try:
            failed = await _await_ray_ref(
                actor.fail_training_requests_for_model.remote(
                    model_id=str(model_id),
                    error=str(error),
                )
            )
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e

        if not isinstance(failed, list):
            raise TypeError("FutureStore.fail_training_requests_for_model returned non-list")

        failed_ids = [str(request_id) for request_id in failed]
        if not failed_ids:
            return []

        try:
            from .capacity_manager import capacity_manager

            for request_id in failed_ids:
                await capacity_manager.async_release_all(request_id)
        except Exception:
            pass

        return failed_ids
    async def async_get_status(self, request_id: str) -> FutureStatus:
        actor = self._get_cached_ray_actor_for_async_request_path()

        import ray

        try:
            status = await _await_ray_ref(actor.get_status.remote(request_id=request_id))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        except ray.exceptions.RayTaskError as e:
            msg = str(e)
            cause = getattr(e, "cause", None) or getattr(e, "__cause__", None)
            is_unknown = "Unknown request_id:" in msg or isinstance(cause, KeyError)
            if is_unknown:
                raise KeyError(f"Unknown request_id: {request_id}") from None
            raise
        return FutureStatus(status)
    async def async_get_result(self, request_id: str) -> Any:
        actor = self._get_cached_ray_actor_for_async_request_path()

        import ray

        try:
            return await _await_ray_ref(actor.get_result.remote(request_id=request_id))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        except ray.exceptions.RayTaskError as e:
            msg = str(e)
            cause = getattr(e, "cause", None) or getattr(e, "__cause__", None)
            is_unknown = "Unknown request_id:" in msg or isinstance(cause, KeyError)
            if is_unknown:
                raise KeyError(f"Unknown request_id: {request_id}") from None
            raise
    async def async_reap(self) -> dict[str, list[str]]:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        try:
            out = await _await_ray_ref(actor.reap.remote())
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        if not isinstance(out, dict):
            raise TypeError(f"FutureStore.reap returned non-dict: {type(out)}")
        expired = out.get("expired") or []
        timed_out = out.get("timed_out") or []
        if not isinstance(expired, list) or not isinstance(timed_out, list):
            raise TypeError("FutureStore.reap returned invalid payload")
        return {"expired": [str(x) for x in expired], "timed_out": [str(x) for x in timed_out]}
    async def async_get_error(self, request_id: str) -> str | None:
        actor = self._get_cached_ray_actor_for_async_request_path()

        import ray

        try:
            out = await _await_ray_ref(actor.get_error.remote(request_id=request_id))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        except ray.exceptions.RayTaskError as e:
            msg = str(e)
            cause = getattr(e, "cause", None) or getattr(e, "__cause__", None)
            is_unknown = "Unknown request_id:" in msg or isinstance(cause, KeyError)
            if is_unknown:
                raise KeyError(f"Unknown request_id: {request_id}") from None
            raise
        if out is None:
            return None
        return str(out)
    async def async_get_meta(self, request_id: str) -> dict[str, Any] | None:
        actor = self._get_cached_ray_actor_for_async_request_path()

        import ray

        try:
            out = await _await_ray_ref(actor.get_meta.remote(request_id=request_id))
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from None

        if out is None:
            return None
        if not isinstance(out, dict):
            raise TypeError(f"FutureStore.get_meta returned non-dict: {type(out)}")
        return out

    async def async_forget(self, request_id: str) -> None:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        try:
            await _await_ray_ref(actor.forget.remote(request_id=request_id))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
    async def async_cleanup(self, request_id: str) -> None:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        try:
            await _await_ray_ref(actor.cleanup.remote(request_id=request_id))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
    async def async_fail_stale_running_requests(self, active_consumer_job_id: str, error: str) -> list[str]:
        actor = self._get_cached_ray_actor_for_async_request_path()
        import ray

        try:
            out = await _await_ray_ref(
                actor.fail_stale_running_requests.remote(
                    active_consumer_job_id=str(active_consumer_job_id),
                    error=str(error),
                )
            )
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        if not isinstance(out, list):
            raise TypeError(f"FutureStore.fail_stale_running_requests returned non-list: {type(out)}")
        return [str(x) for x in out]


future_store = FutureStore()
