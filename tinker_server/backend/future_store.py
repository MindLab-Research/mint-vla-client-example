"""Storage for async operation results.

Maps request_id to results for the async polling pattern:
1. Client sends request, gets request_id
2. Server processes in background
3. Client polls with request_id until result ready

Ray is a hard requirement: futures are stored in a detached Ray actor so they
survive multi-worker deployments without per-process state loss.
"""

from __future__ import annotations

import os
import time
import uuid
from enum import Enum
from typing import Any

from ..config import config as server_config


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


def _infer_ray_address() -> str | None:
    """Infer Ray GCS address for remote clusters.

    Prefer explicit RAY_ADDRESS. Fall back to ray_head_ip.txt on shared storage
    (Volcano writes the canonical head IP there).
    """
    addr = (os.environ.get("RAY_ADDRESS") or "").strip()
    if addr:
        return addr

    candidates: list[str] = []
    pfs_tinker_path = (os.environ.get("PFS_TINKER_PATH") or "").strip()
    if pfs_tinker_path:
        candidates.append(os.path.join(pfs_tinker_path, "ray_head_ip.txt"))
    candidates.extend(
        [
            # Common prod/dev code roots on PFS
            "/vePFS-Mindverse/share/code/tinker-server-auth/ray_head_ip.txt",
            "/vePFS-Mindverse/share/code/tinker-server/ray_head_ip.txt",
            # Local repo fallback (useful for workstation tunnels)
            os.path.join(os.getcwd(), "ray_head_ip.txt"),
        ]
    )

    for p in candidates:
        try:
            ip = open(p, "r", encoding="utf-8").read().strip()
        except OSError:
            continue
        if ip:
            return f"{ip}:6379"
    return None


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

    @ray.remote
    class _RayFutureStoreActor:
        def __init__(
            self, ttl_s: float, queue_ttl_s: float, done_ttl_s: float, tombstone_ttl_s: float
        ) -> None:
            self._pending: set[str] = set()
            self._result_refs: dict[str, Any] = {}
            self._errors: dict[str, str] = {}
            self._refs: dict[str, Any] = {}
            self._meta: dict[str, dict[str, Any]] = {}

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

        def mark_queued(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
            self._prune()
            if request_id in self._pending:
                self._queued_at[request_id] = time.time()
            if meta is not None:
                m = self._meta.get(request_id) or {}
                m.update(dict(meta))
                self._meta[request_id] = m

        def mark_running(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
            self._prune()
            if request_id in self._pending:
                self._running_at[request_id] = time.time()
            if meta is not None:
                m = self._meta.get(request_id) or {}
                m.update(dict(meta))
                self._meta[request_id] = m

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
            import ray

            self._result_refs[request_id] = ray.put(result)
            self._done_at[request_id] = time.time()

        def resolve_ref(self, request_id: str, ref: Any) -> None:
            self._prune()
            self._pending.discard(request_id)
            self._refs.pop(request_id, None)
            self._result_refs[request_id] = ref
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
            self._result_refs.pop(request_id, None)
            self._errors.pop(request_id, None)
            self._refs.pop(request_id, None)
            self._meta.pop(request_id, None)
            self._created_at.pop(request_id, None)
            self._queued_at.pop(request_id, None)
            self._running_at.pop(request_id, None)
            self._done_at.pop(request_id, None)
            self._expired_at.pop(request_id, None)
            self._retrieved_at[request_id] = time.time()

        def reap(self) -> dict[str, list[str]]:
            # Return request_ids that transitioned to terminal tombstones and must
            # release any external reservations.
            return self._prune()

    try:
        return _RayFutureStoreActor.options(
            name=actor_name,
            namespace=namespace,
            lifetime="detached",
        ).remote(ttl_s, queue_ttl_s, done_ttl_s, tombstone_ttl_s)
    except Exception:
        # Race: another process may have created the actor between get_actor and create.
        return ray.get_actor(actor_name, namespace=namespace)


class FutureStore:
    """FutureStore backed by a detached Ray actor (hard requirement)."""

    def __init__(self) -> None:
        self._ray_actor = None

    def ensure_ready(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        """Fail fast if Ray or the detached FutureStore actor is unavailable."""
        actor = self._get_ray_actor()
        import ray

        return ray.get(actor.stats.remote(), timeout=float(timeout_s))

    def rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        actor = self._get_ray_actor()
        import ray

        try:
            v = ray.get(actor.get_rss_bytes.remote(), timeout=float(timeout_s))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        return int(v)

    def _get_ray_actor(self):
        try:
            import ray
        except Exception as e:
            raise FutureStoreUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            try:
                from ..ray_utils import init_ray

                addr = _infer_ray_address()
                init_ray(address=addr or "auto", namespace=_ray_namespace(), ignore_reinit_error=True)
            except Exception as e:
                raise FutureStoreUnavailableError("Ray not initialized (init_ray failed)") from e

        if not ray.is_initialized():
            raise FutureStoreUnavailableError("Ray not initialized")

        if self._ray_actor is None:
            try:
                self._ray_actor = _get_or_create_ray_actor()
            except Exception as e:
                raise FutureStoreUnavailableError("Failed to get/create detached Ray FutureStore actor") from e
        return self._ray_actor

    def debug_snapshot(self) -> dict[str, Any]:
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
                out["ray_address_inferred"] = _infer_ray_address()
                return out

            try:
                actor = ray.get_actor(_ray_future_store_actor_name(), namespace=_ray_namespace())
            except Exception as e:
                out["ray_actor_get_error"] = f"{type(e).__name__}: {e}"
                return out

            try:
                out["ray_actor_stats"] = ray.get(actor.stats.remote())
            except Exception as e:
                out["ray_actor_stats_error"] = f"{type(e).__name__}: {e}"
            return out
        except Exception as e:
            out["ray_import_error"] = f"{type(e).__name__}: {e}"
            return out

    def create(self) -> str:
        request_id = str(uuid.uuid4())
        return self.create_with_id(request_id)

    def create_with_id(self, request_id: str) -> str:
        actor = self._get_ray_actor()

        import ray

        try:
            ray.get(actor.add_pending.remote(request_id=str(request_id)))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        return str(request_id)

    def mark_queued(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        actor = self._get_ray_actor()
        import ray

        payload = None if meta is None else dict(meta)
        try:
            actor.mark_queued.remote(request_id=request_id, meta=payload)
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None
            actor = self._get_ray_actor()
            actor.mark_queued.remote(request_id=request_id, meta=payload)

    def mark_running(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        actor = self._get_ray_actor()
        import ray

        try:
            actor.mark_running.remote(request_id=request_id, meta=None if meta is None else dict(meta))
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None
            actor = self._get_ray_actor()
            actor.mark_running.remote(request_id=request_id, meta=None if meta is None else dict(meta))

    def resolve(self, request_id: str, result: Any) -> None:
        actor = self._get_ray_actor()
        import ray

        try:
            ref = ray.put(result)
            actor.resolve_ref.remote(request_id=request_id, ref=ref)
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e

    def fail(self, request_id: str, error: str) -> None:
        actor = self._get_ray_actor()
        import ray

        try:
            actor.fail.remote(request_id=request_id, error=str(error))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e
        try:
            from .capacity_manager import capacity_manager

            capacity_manager.release_object_store(request_id)
        except Exception:
            pass

    def get_status(self, request_id: str) -> FutureStatus:
        actor = self._get_ray_actor()

        import ray

        try:
            status = ray.get(actor.get_status.remote(request_id=request_id))
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

    def get_result(self, request_id: str) -> Any:
        actor = self._get_ray_actor()

        import ray

        try:
            # Ray auto-dereferences ObjectRef return values, so actor.get_result
            # yields the actual payload (or None), not an ObjectRef.
            return ray.get(actor.get_result.remote(request_id=request_id))
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

    def reap(self) -> dict[str, list[str]]:
        actor = self._get_ray_actor()
        import ray

        try:
            out = ray.get(actor.reap.remote())
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

    def get_error(self, request_id: str) -> str | None:
        actor = self._get_ray_actor()

        import ray

        try:
            return ray.get(actor.get_error.remote(request_id=request_id))
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

    def attach_ref(self, request_id: str, ref: Any, meta: dict[str, Any] | None = None) -> None:
        actor = self._get_ray_actor()

        import ray

        try:
            ray.get(actor.attach_ref.remote(request_id=request_id, ref=ref, meta=meta))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise FutureStoreUnavailableError("Detached Ray FutureStore actor died") from e

    def submit(
        self,
        request_id: str,
        target_actor: Any,
        method_name: str,
        args: list[Any] | dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> None:
        actor = self._get_ray_actor()

        import ray

        try:
            ray.get(
                actor.submit.remote(
                    request_id=request_id,
                    target_actor=target_actor,
                    method_name=method_name,
                    args=args,
                    meta=meta,
                )
            )
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None
            actor = self._get_ray_actor()
            ray.get(
                actor.submit.remote(
                    request_id=request_id,
                    target_actor=target_actor,
                    method_name=method_name,
                    args=args,
                    meta=meta,
                )
            )

    def get_meta(self, request_id: str) -> dict[str, Any] | None:
        actor = self._get_ray_actor()

        import ray

        try:
            return ray.get(actor.get_meta.remote(request_id=request_id))
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None
            actor = self._get_ray_actor()
            return ray.get(actor.get_meta.remote(request_id=request_id))

    def cleanup(self, request_id: str) -> None:
        actor = self._get_ray_actor()
        import ray

        try:
            actor.cleanup.remote(request_id=request_id)
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None
            actor = self._get_ray_actor()
            actor.cleanup.remote(request_id=request_id)


future_store = FutureStore()
