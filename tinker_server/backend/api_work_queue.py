from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..config import config as server_config


class ApiWorkQueueUnavailableError(RuntimeError):
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


def _ray_api_work_queue_actor_name() -> str:
    return str(getattr(server_config, "api_work_queue_actor_name", "tinker_api_work_queue"))


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    op: str
    request_json: bytes
    user_id: str | None
    webhook_url: str | None
    extra: dict[str, Any]
    created_at: float


def _get_or_create_ray_actor():
    import ray

    actor_name = _ray_api_work_queue_actor_name()
    try:
        return ray.get_actor(actor_name, namespace=_ray_namespace())
    except ValueError:
        pass

    @ray.remote
    class _RayApiWorkQueueActor:
        def __init__(self) -> None:
            import asyncio
            from collections import deque

            self._items = deque()
            self._cv = asyncio.Condition()
            self._enqueued = 0
            self._dequeued = 0

        def get_rss_bytes(self) -> int:
            with open("/proc/self/statm", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) < 2:
                raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
            rss_pages = int(parts[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return rss_pages * page_size

        async def enqueue(self, item: dict[str, Any]) -> None:
            async with self._cv:
                self._items.append(dict(item))
                self._enqueued += 1
                self._cv.notify(1)

        async def dequeue(self) -> dict[str, Any]:
            async with self._cv:
                while not self._items:
                    await self._cv.wait()
                self._dequeued += 1
                return self._items.popleft()

        def stats(self) -> dict[str, Any]:
            return {"depth": len(self._items), "enqueued": int(self._enqueued), "dequeued": int(self._dequeued)}

    return _RayApiWorkQueueActor.options(  # type: ignore[attr-defined]
        name=actor_name, namespace=_ray_namespace(), lifetime="detached", get_if_exists=True
    ).remote()


Executor = Callable[[WorkItem], Awaitable[None]]


class ApiWorkQueueClient:
    def __init__(self) -> None:
        self._ray_actor = None
        self._executors: dict[str, Executor] = {}
        self._worker_tasks: list[Any] = []
        self._running = False
        self._dequeue_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def _get_ray_actor(self):
        try:
            import ray
        except Exception as e:
            raise ApiWorkQueueUnavailableError("Ray import failed") from e

        if not ray.is_initialized():
            try:
                from ..ray_utils import init_ray
                from .future_store import _infer_ray_address  # type: ignore

                addr = _infer_ray_address()
                init_ray(address=addr or "auto", namespace=_ray_namespace(), ignore_reinit_error=True)
            except Exception as e:
                raise ApiWorkQueueUnavailableError("Ray not initialized (init_ray failed)") from e

        if not ray.is_initialized():
            raise ApiWorkQueueUnavailableError("Ray not initialized")

        if self._ray_actor is None:
            try:
                self._ray_actor = _get_or_create_ray_actor()
            except Exception as e:
                raise ApiWorkQueueUnavailableError("Failed to get/create detached Ray ApiWorkQueue actor") from e
        return self._ray_actor

    def set_executor(self, op: str, executor: Executor) -> None:
        self._executors[str(op)] = executor

    async def enqueue(
        self,
        *,
        request_id: str,
        op: str,
        request_json: bytes,
        user_id: str | None,
        webhook_url: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        import ray

        actor = self._get_ray_actor()
        item = {
            "request_id": str(request_id),
            "op": str(op),
            "request_json": bytes(request_json),
            "user_id": None if user_id is None else str(user_id),
            "webhook_url": None if webhook_url is None else str(webhook_url),
            "extra": {} if extra is None else dict(extra),
            "created_at": time.time(),
        }
        # Ensure enqueue succeeds, otherwise the request can remain pending forever
        # while capacity stays reserved.
        ref = actor.enqueue.remote(item)
        ray.get(ref, timeout=10.0)

    async def _dequeue(self) -> WorkItem:
        import asyncio
        import ray

        if self._dequeue_executor is None:
            raise RuntimeError("ApiWorkQueueClient not started (dequeue executor missing)")

        actor = self._get_ray_actor()
        ref = actor.dequeue.remote()
        loop = asyncio.get_running_loop()
        item = await loop.run_in_executor(self._dequeue_executor, ray.get, ref)
        if not isinstance(item, dict):
            raise TypeError(f"ApiWorkQueue.dequeue returned non-dict: {type(item)}")
        return WorkItem(
            request_id=str(item["request_id"]),
            op=str(item["op"]),
            request_json=bytes(item["request_json"]),
            user_id=None if item.get("user_id") is None else str(item["user_id"]),
            webhook_url=None if item.get("webhook_url") is None else str(item["webhook_url"]),
            extra=dict(item.get("extra") or {}),
            created_at=float(item.get("created_at", 0.0)),
        )

    async def start_workers(self, *, num_workers: int) -> None:
        import asyncio

        if self._running:
            return
        self._running = True
        n = int(num_workers)
        if n < 1:
            n = 1
        self._dequeue_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=n,
            thread_name_prefix="api_work_queue_dequeue",
        )
        self._worker_tasks = [asyncio.create_task(self._worker_loop(i)) for i in range(n)]

    async def shutdown(self) -> None:
        import asyncio

        self._running = False
        for t in self._worker_tasks:
            t.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []
        if self._dequeue_executor is not None:
            self._dequeue_executor.shutdown(wait=False, cancel_futures=True)
            self._dequeue_executor = None

    async def _worker_loop(self, worker_idx: int) -> None:
        from .capacity_manager import capacity_manager
        from .future_store import FutureStatus, future_store

        while self._running:
            item = await self._dequeue()
            try:
                capacity_manager.release_queue(item.request_id)
            except Exception:
                # Do not fail open: the reservation leak will force 429 and surface via stats.
                pass

            # If the future has already transitioned to a terminal state (for example due to
            # queue-timeout), do not run the executor. This prevents a timed-out future from
            # later being overwritten by a "successful" resolve.
            try:
                status = future_store.get_status(item.request_id)
            except KeyError:
                try:
                    capacity_manager.release_all(item.request_id)
                except Exception:
                    pass
                continue
            except Exception:
                status = None

            if status is not None and status != FutureStatus.PENDING:
                try:
                    capacity_manager.release_all(item.request_id)
                except Exception:
                    pass
                continue

            try:
                future_store.mark_running(item.request_id, meta={"worker_idx": int(worker_idx), "op": item.op})
            except Exception:
                pass

            ex = self._executors.get(item.op)
            if ex is None:
                try:
                    future_store.fail(item.request_id, f"unknown op: {item.op!r}")
                except Exception:
                    pass
                try:
                    capacity_manager.release_object_store(item.request_id)
                except Exception:
                    pass
                continue

            try:
                await ex(item)
            except Exception as e:
                # Ensure the future does not remain pending forever.
                try:
                    future_store.fail(item.request_id, f"executor failed: {e}")
                except Exception:
                    pass
                try:
                    capacity_manager.release_object_store(item.request_id)
                except Exception:
                    pass

    async def stats(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        import ray

        actor = self._get_ray_actor()
        ref = actor.stats.remote()
        return ray.get(ref, timeout=float(timeout_s))

    async def rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        import ray

        actor = self._get_ray_actor()
        ref = actor.get_rss_bytes.remote()
        v = ray.get(ref, timeout=float(timeout_s))
        return int(v)


api_work_queue = ApiWorkQueueClient()
