from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..config import config as server_config, otel_env_vars
from .async_ray_control import _await_with_ray_get_timeout


class CapacityManagerUnavailableError(RuntimeError):
    pass


def _ray_namespace() -> str:
    v = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _ray_capacity_manager_actor_name() -> str:
    return str(getattr(server_config, "capacity_manager_actor_name", "tinker_capacity_manager"))


@dataclass(frozen=True)
class CapacitySnapshot:
    queue_bytes_budget: int
    queue_bytes_reserved: int
    object_store_bytes_reserved: int
    object_store_free_bytes: int | None
    rejects_total: int
    reserves_total: int


def _require_int(name: str, v: Any) -> int:
    if not isinstance(v, int):
        raise TypeError(f"{name} must be int, got {type(v)}")
    return v


def _object_store_free_bytes() -> int:
    import ray

    free = ray.available_resources().get("object_store_memory")
    if free is None:
        raise CapacityManagerUnavailableError(
            "Ray does not expose available_resources()['object_store_memory']; cannot enforce object store admission"
        )
    # Ray reports as float; treat as bytes.
    return int(free)


async def _await_ray_ref(ref: Any) -> Any:
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

    actor_name = _ray_capacity_manager_actor_name()
    try:
        return ray.get_actor(actor_name, namespace=_ray_namespace())
    except ValueError:
        pass

    queue_bytes_budget = int(getattr(server_config, "capacity_queue_bytes_budget", 512 * 1024 * 1024))

    @ray.remote(num_cpus=0)
    class _RayCapacityManagerActor:
        def __init__(self, *, queue_bytes_budget: int) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._queue_bytes_budget = int(queue_bytes_budget)
            self._queue_bytes_reserved = 0
            self._object_store_bytes_reserved = 0
            self._reserves_total = 0
            self._rejects_total = 0
            self._reservations: dict[str, dict[str, int]] = {}
            self._queue_released: set[str] = set()
            self._object_store_released: set[str] = set()
            self._created_at: dict[str, float] = {}

        def get_rss_bytes(self) -> int:
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) < 2:
                raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
            rss_pages = int(parts[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return rss_pages * page_size

        def snapshot(self) -> dict[str, Any]:
            free = None
            try:
                free = _object_store_free_bytes()
            except Exception:
                free = None
            return {
                "queue_bytes_budget": int(self._queue_bytes_budget),
                "queue_bytes_reserved": int(self._queue_bytes_reserved),
                "object_store_bytes_reserved": int(self._object_store_bytes_reserved),
                "object_store_free_bytes": None if free is None else int(free),
                "rejects_total": int(self._rejects_total),
                "reserves_total": int(self._reserves_total),
                "reservations": len(self._reservations),
            }

        def try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict[str, Any]:
            if request_id in self._reservations:
                self._rejects_total += 1
                return {"ok": False, "reason": "duplicate_request_id"}

            queue_bytes = int(queue_bytes)
            object_store_bytes = int(object_store_bytes)
            if queue_bytes < 0 or object_store_bytes < 0:
                self._rejects_total += 1
                return {"ok": False, "reason": "negative_bytes"}

            if self._queue_bytes_reserved + queue_bytes > self._queue_bytes_budget:
                self._rejects_total += 1
                return {
                    "ok": False,
                    "reason": "queue_bytes_budget_exceeded",
                    "queue_bytes_budget": int(self._queue_bytes_budget),
                    "queue_bytes_reserved": int(self._queue_bytes_reserved),
                }

            try:
                free = _object_store_free_bytes()
            except Exception as e:
                self._rejects_total += 1
                return {"ok": False, "reason": "object_store_signal_unavailable", "error": str(e)}

            # Fail closed: do not overcommit beyond Ray's reported free bytes.
            if self._object_store_bytes_reserved + object_store_bytes > free:
                self._rejects_total += 1
                return {
                    "ok": False,
                    "reason": "object_store_budget_exceeded",
                    "object_store_free_bytes": int(free),
                    "object_store_bytes_reserved": int(self._object_store_bytes_reserved),
                }

            self._queue_bytes_reserved += queue_bytes
            self._object_store_bytes_reserved += object_store_bytes
            self._reservations[request_id] = {
                "queue_bytes": queue_bytes,
                "object_store_bytes": object_store_bytes,
            }
            self._created_at[request_id] = time.time()
            self._reserves_total += 1
            return {"ok": True}

        def release_queue(self, request_id: str) -> dict[str, Any]:
            if request_id in self._queue_released:
                return {"ok": True, "already": True}
            res = self._reservations.get(request_id)
            if not isinstance(res, dict):
                return {"ok": False, "reason": "unknown_request_id"}
            qb = int(res.get("queue_bytes", 0))
            self._queue_bytes_reserved -= qb
            self._queue_released.add(request_id)
            return {"ok": True}

        def release_object_store(self, request_id: str) -> dict[str, Any]:
            if request_id in self._object_store_released:
                return {"ok": True, "already": True}
            res = self._reservations.get(request_id)
            if not isinstance(res, dict):
                return {"ok": False, "reason": "unknown_request_id"}
            ob = int(res.get("object_store_bytes", 0))
            self._object_store_bytes_reserved -= ob
            self._object_store_released.add(request_id)
            return {"ok": True}

        def release_all(self, request_id: str) -> dict[str, Any]:
            self.release_queue(request_id)
            self.release_object_store(request_id)
            self._reservations.pop(request_id, None)
            self._created_at.pop(request_id, None)
            self._queue_released.discard(request_id)
            self._object_store_released.discard(request_id)
            return {"ok": True}

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "max_restarts": -1,
        "max_task_retries": -1,
    }
    actor_otel_env = otel_env_vars()
    from ..config import PFS_PYTHONPATH, actor_runtime_env_vars, apply_detached_actor_resources
    apply_detached_actor_resources(options, ray)
    options["runtime_env"] = {
        "env_vars": actor_runtime_env_vars(
            pythonpath=PFS_PYTHONPATH,
            extra=actor_otel_env,
        )
    }

    try:
        return _RayCapacityManagerActor.options(  # type: ignore[attr-defined]
            **options
        ).remote(queue_bytes_budget=queue_bytes_budget)
    except Exception:
        # Race: another request may have created the detached actor first.
        return ray.get_actor(actor_name, namespace=_ray_namespace())


class CapacityManager:
    def __init__(self) -> None:
        self._ray_actor = None
        self._ray_actor_lock = threading.Lock()
        from ..ray_utils import register_ray_reconnect_invalidator

        register_ray_reconnect_invalidator(self._reset_ray_actor)

    def _reset_ray_actor(self, actor: Any | None = None) -> None:
        with self._ray_actor_lock:
            if actor is None or self._ray_actor is actor:
                self._ray_actor = None

    def _get_cached_ray_actor_for_async_request_path(self):
        try:
            import ray
        except Exception as e:
            raise CapacityManagerUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            raise CapacityManagerUnavailableError("Ray not initialized")

        actor = self._ray_actor
        if actor is None:
            with self._ray_actor_lock:
                if self._ray_actor is None:
                    try:
                        self._ray_actor = _get_or_create_ray_actor()
                    except Exception as e:
                        raise CapacityManagerUnavailableError(
                            "Failed to get/create detached Ray CapacityManager actor"
                        ) from e
                actor = self._ray_actor
        return actor

    async def _get_ray_actor_async(self):
        try:
            import ray
        except Exception as e:
            raise CapacityManagerUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            raise CapacityManagerUnavailableError("Ray not initialized")

        actor = self._ray_actor
        if actor is not None:
            return actor

        with self._ray_actor_lock:
            if self._ray_actor is None:
                try:
                    self._ray_actor = _get_or_create_ray_actor()
                except Exception as e:
                    raise CapacityManagerUnavailableError("Failed to get/create detached Ray CapacityManager actor") from e
            return self._ray_actor


    async def _async_with_actor_retry(self, fn, *, err_msg: str):
        import ray

        actor_died_errors = tuple(
            err
            for err in (
                getattr(ray.exceptions, "ActorDiedError", None),
                getattr(ray.exceptions, "RayActorError", None),
            )
            if isinstance(err, type) and issubclass(err, BaseException)
        )

        try:
            actor = self._get_cached_ray_actor_for_async_request_path()
        except CapacityManagerUnavailableError:
            actor = await self._get_ray_actor_async()

        try:
            return await fn(actor)
        except actor_died_errors as e:
            self._reset_ray_actor(actor)
            raise CapacityManagerUnavailableError(err_msg) from e

    def _run_coro_sync_best_effort(self, coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        loop.create_task(coro)
        return None

    def release_queue(self, request_id: str) -> None:
        self._run_coro_sync_best_effort(self.async_release_queue(request_id))

    def release_object_store(self, request_id: str) -> None:
        self._run_coro_sync_best_effort(self.async_release_object_store(request_id))

    def release_all(self, request_id: str) -> None:
        self._run_coro_sync_best_effort(self.async_release_all(request_id))

    async def async_ensure_ready(self, *, timeout_s: float = 10.0) -> CapacitySnapshot:
        actor = await self._get_ray_actor_async()
        d = await _await_with_ray_get_timeout(
            _await_ray_ref(actor.snapshot.remote()),
            timeout_s=float(timeout_s),
        )
        if not isinstance(d, dict):
            raise TypeError(f"CapacityManager.snapshot returned non-dict: {type(d)}")
        return CapacitySnapshot(
            queue_bytes_budget=_require_int("queue_bytes_budget", d.get("queue_bytes_budget")),
            queue_bytes_reserved=_require_int("queue_bytes_reserved", d.get("queue_bytes_reserved")),
            object_store_bytes_reserved=_require_int("object_store_bytes_reserved", d.get("object_store_bytes_reserved")),
            object_store_free_bytes=None if d.get("object_store_free_bytes") is None else int(d["object_store_free_bytes"]),
            rejects_total=_require_int("rejects_total", d.get("rejects_total")),
            reserves_total=_require_int("reserves_total", d.get("reserves_total")),
        )

    async def async_try_reserve(
        self,
        request_id: str,
        *,
        queue_bytes: int,
        object_store_bytes: int,
    ) -> dict[str, Any]:
        out = await self._async_with_actor_retry(
            lambda actor: _await_ray_ref(
                actor.try_reserve.remote(
                    request_id,
                    queue_bytes=int(queue_bytes),
                    object_store_bytes=int(object_store_bytes),
                )
            ),
            err_msg="Detached Ray CapacityManager actor died",
        )
        if not isinstance(out, dict):
            raise TypeError(f"CapacityManager.try_reserve returned non-dict: {type(out)}")
        return out

    async def async_release_queue(self, request_id: str) -> None:
        try:
            await self._async_with_actor_retry(
                lambda actor: _await_ray_ref(actor.release_queue.remote(request_id)),
                err_msg="Detached Ray CapacityManager actor died",
            )
        except CapacityManagerUnavailableError:
            self._reset_ray_actor()

    async def async_release_object_store(self, request_id: str) -> None:
        try:
            await self._async_with_actor_retry(
                lambda actor: _await_ray_ref(actor.release_object_store.remote(request_id)),
                err_msg="Detached Ray CapacityManager actor died",
            )
        except CapacityManagerUnavailableError:
            self._reset_ray_actor()

    async def async_release_all(self, request_id: str) -> None:
        try:
            await self._async_with_actor_retry(
                lambda actor: _await_ray_ref(actor.release_all.remote(request_id)),
                err_msg="Detached Ray CapacityManager actor died",
            )
        except CapacityManagerUnavailableError:
            self._reset_ray_actor()

    async def async_snapshot(self, *, timeout_s: float = 10.0) -> CapacitySnapshot:
        d = await self._async_with_actor_retry(
            lambda actor: _await_with_ray_get_timeout(
                _await_ray_ref(actor.snapshot.remote()),
                timeout_s=float(timeout_s),
            ),
            err_msg="Detached Ray CapacityManager actor died",
        )
        if not isinstance(d, dict):
            raise TypeError(f"CapacityManager.snapshot returned non-dict: {type(d)}")
        return CapacitySnapshot(
            queue_bytes_budget=_require_int("queue_bytes_budget", d.get("queue_bytes_budget")),
            queue_bytes_reserved=_require_int("queue_bytes_reserved", d.get("queue_bytes_reserved")),
            object_store_bytes_reserved=_require_int("object_store_bytes_reserved", d.get("object_store_bytes_reserved")),
            object_store_free_bytes=None if d.get("object_store_free_bytes") is None else int(d["object_store_free_bytes"]),
            rejects_total=_require_int("rejects_total", d.get("rejects_total")),
            reserves_total=_require_int("reserves_total", d.get("reserves_total")),
        )

    async def async_rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        v = await self._async_with_actor_retry(
            lambda actor: _await_with_ray_get_timeout(
                _await_ray_ref(actor.get_rss_bytes.remote()),
                timeout_s=float(timeout_s),
            ),
            err_msg="Detached Ray CapacityManager actor died",
        )
        return int(v)


capacity_manager = CapacityManager()
