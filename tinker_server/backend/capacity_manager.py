from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from ..config import config as server_config


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


def _get_or_create_ray_actor():
    import ray

    actor_name = _ray_capacity_manager_actor_name()
    try:
        return ray.get_actor(actor_name, namespace=_ray_namespace())
    except ValueError:
        pass

    queue_bytes_budget = int(getattr(server_config, "capacity_queue_bytes_budget", 512 * 1024 * 1024))

    @ray.remote
    class _RayCapacityManagerActor:
        def __init__(self, *, queue_bytes_budget: int) -> None:
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

    return _RayCapacityManagerActor.options(  # type: ignore[attr-defined]
        name=actor_name, namespace=_ray_namespace(), lifetime="detached", get_if_exists=True
    ).remote(queue_bytes_budget=queue_bytes_budget)


class CapacityManager:
    def __init__(self) -> None:
        self._ray_actor = None

    def _get_ray_actor(self):
        try:
            import ray
        except Exception as e:
            raise CapacityManagerUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            try:
                from ..ray_utils import init_ray
                from .future_store import _infer_ray_address  # type: ignore

                addr = _infer_ray_address()
                init_ray(address=addr or "auto", namespace=_ray_namespace(), ignore_reinit_error=True)
            except Exception as e:
                raise CapacityManagerUnavailableError("Ray not initialized (init_ray failed)") from e

        if not ray.is_initialized():
            raise CapacityManagerUnavailableError("Ray not initialized")

        if self._ray_actor is None:
            try:
                self._ray_actor = _get_or_create_ray_actor()
            except Exception as e:
                raise CapacityManagerUnavailableError("Failed to get/create detached Ray CapacityManager actor") from e
        return self._ray_actor

    def try_reserve(self, request_id: str, *, queue_bytes: int, object_store_bytes: int) -> dict[str, Any]:
        actor = self._get_ray_actor()
        import ray

        try:
            return ray.get(actor.try_reserve.remote(request_id, queue_bytes=int(queue_bytes), object_store_bytes=int(object_store_bytes)))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise CapacityManagerUnavailableError("Detached Ray CapacityManager actor died") from e

    def release_queue(self, request_id: str) -> None:
        actor = self._get_ray_actor()
        import ray

        try:
            actor.release_queue.remote(request_id)
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None

    def release_object_store(self, request_id: str) -> None:
        actor = self._get_ray_actor()
        import ray

        try:
            actor.release_object_store.remote(request_id)
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None

    def release_all(self, request_id: str) -> None:
        actor = self._get_ray_actor()
        import ray

        try:
            actor.release_all.remote(request_id)
        except ray.exceptions.ActorDiedError:
            self._ray_actor = None

    def snapshot(self, *, timeout_s: float = 10.0) -> CapacitySnapshot:
        actor = self._get_ray_actor()
        import ray

        try:
            d = ray.get(actor.snapshot.remote(), timeout=float(timeout_s))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise CapacityManagerUnavailableError("Detached Ray CapacityManager actor died") from e
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

    def rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        actor = self._get_ray_actor()
        import ray

        try:
            v = ray.get(actor.get_rss_bytes.remote(), timeout=float(timeout_s))
        except ray.exceptions.ActorDiedError as e:
            self._ray_actor = None
            raise CapacityManagerUnavailableError("Detached Ray CapacityManager actor died") from e
        return int(v)


capacity_manager = CapacityManager()
