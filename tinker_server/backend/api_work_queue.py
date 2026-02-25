from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..config import config as server_config

logger = logging.getLogger(__name__)


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
        actor = ray.get_actor(actor_name, namespace=_ray_namespace())
        try:
            # Ensure the handle is actually usable. A dead named actor can still
            # be discoverable via `ray.get_actor`, but any call will raise
            # ActorDiedError and enqueue will fail with 503.
            ray.get(actor.stats.remote(), timeout=1.0)
            return actor
        except Exception:
            pass
    except ValueError:
        pass

    max_concurrency = int(os.environ.get("MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY", "128"))

    @ray.remote(max_concurrency=max_concurrency)
    class _RayApiWorkQueueActor:
        def __init__(self) -> None:
            import asyncio
            from collections import deque

            self._items = deque()
            self._cv = asyncio.Condition()
            self._enqueued = 0
            self._dequeued = 0
            self._recent_dequeues = deque(maxlen=int(os.environ.get("MINT_API_WORK_QUEUE_DEBUG_MAX", "50")))
            self._recent_enqueues = deque(maxlen=int(os.environ.get("MINT_API_WORK_QUEUE_DEBUG_MAX", "50")))

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
                try:
                    import ray

                    ctx = ray.get_runtime_context()
                    self._recent_enqueues.append(
                        {
                            "ts": time.time(),
                            "job_id": str(ctx.get_job_id()),
                            "task_id": str(ctx.get_task_id()),
                            "request_id": str(item.get("request_id")),
                            "op": str(item.get("op")),
                        }
                    )
                except Exception:
                    pass
                self._cv.notify(1)

        async def dequeue(self) -> dict[str, Any]:
            async with self._cv:
                while not self._items:
                    await self._cv.wait()
                self._dequeued += 1
                item = self._items.popleft()
                try:
                    import ray

                    ctx = ray.get_runtime_context()
                    self._recent_dequeues.append(
                        {
                            "ts": time.time(),
                            "job_id": str(ctx.get_job_id()),
                            "task_id": str(ctx.get_task_id()),
                            "request_id": str(item.get("request_id")),
                            "op": str(item.get("op")),
                        }
                    )
                except Exception:
                    pass
                return item

        def stats(self) -> dict[str, Any]:
            return {"depth": len(self._items), "enqueued": int(self._enqueued), "dequeued": int(self._dequeued)}

        def debug_state(self) -> dict[str, Any]:
            return {
                "stats": self.stats(),
                "recent_enqueues": list(self._recent_enqueues),
                "recent_dequeues": list(self._recent_dequeues),
            }

    # Keep the detached queue actor on the head node when possible. Losing this
    # actor drops all queued items (in-memory queue), which can leave futures
    # pending forever.
    resources = None
    try:
        if "node:__internal_head__" in ray.cluster_resources():
            resources = {"node:__internal_head__": 0.001}
    except Exception:
        resources = None

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "max_restarts": -1,
        "max_task_retries": -1,
    }
    if resources is not None:
        options["resources"] = resources

    return _RayApiWorkQueueActor.options(  # type: ignore[attr-defined]
        **options
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

        if self._ray_actor is not None:
            try:
                ray.get(self._ray_actor.stats.remote(), timeout=1.0)
            except Exception:
                self._ray_actor = None

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
        import asyncio
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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: ray.get(ref, timeout=10.0))

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
        import asyncio

        from .capacity_manager import capacity_manager
        from .future_store import FutureStatus, future_store

        while self._running:
            try:
                item = await self._dequeue()
            except Exception as e:
                # Never let a dequeue failure permanently kill the background workers.
                # If the detached Ray queue actor dies (or Ray connectivity blips),
                # keep the server alive and retry.
                try:
                    import ray

                    if isinstance(e, (ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)):
                        self._ray_actor = None
                except Exception:
                    logger.error(
                        "[api_work_queue] failed to classify dequeue exception as Ray error (worker_idx=%s): %s: %s",
                        int(worker_idx),
                        type(e).__name__,
                        e,
                    )

                logger.error(
                    "[api_work_queue] dequeue failed (worker_idx=%s): %s: %s",
                    int(worker_idx),
                    type(e).__name__,
                    e,
                )
                await asyncio.sleep(1.0)
                continue
            if str(item.op) == "training.create_model":
                try:
                    age_s = max(0.0, time.time() - float(item.created_at))
                except Exception:
                    age_s = -1.0
                logger.info(
                    "[api_work_queue] dequeued request_id=%s op=%s worker_idx=%s age_s=%.3f",
                    str(item.request_id),
                    str(item.op),
                    int(worker_idx),
                    float(age_s),
                )
            try:
                capacity_manager.release_queue(item.request_id)
            except Exception as e:
                # Do not fail open: the reservation leak will force 429 and surface via stats.
                logger.error(
                    "[api_work_queue] release_queue failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                    int(worker_idx),
                    str(item.request_id),
                    str(item.op),
                    type(e).__name__,
                    e,
                )

            # If the future has already transitioned to a terminal state (for example due to
            # queue-timeout), do not run the executor. This prevents a timed-out future from
            # later being overwritten by a "successful" resolve.
            try:
                status = future_store.get_status(item.request_id)
            except KeyError:
                try:
                    capacity_manager.release_all(item.request_id)
                except Exception as e:
                    logger.error(
                        "[api_work_queue] release_all failed after unknown future (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e).__name__,
                        e,
                    )
                continue
            except Exception as e:
                logger.error(
                    "[api_work_queue] get_status failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                    int(worker_idx),
                    str(item.request_id),
                    str(item.op),
                    type(e).__name__,
                    e,
                )
                try:
                    future_store.fail(item.request_id, f"internal error: future_store.get_status failed: {type(e).__name__}: {e}")
                except Exception as e2:
                    logger.error(
                        "[api_work_queue] future_store.fail failed after get_status error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e2).__name__,
                        e2,
                    )
                try:
                    capacity_manager.release_all(item.request_id)
                except Exception as e2:
                    logger.error(
                        "[api_work_queue] release_all failed after get_status error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e2).__name__,
                        e2,
                    )
                continue

            if status is not None and status != FutureStatus.PENDING:
                if str(item.op) == "training.create_model":
                    logger.info(
                        "[api_work_queue] skip_non_pending request_id=%s op=%s worker_idx=%s status=%s",
                        str(item.request_id),
                        str(item.op),
                        int(worker_idx),
                        str(status),
                    )
                try:
                    capacity_manager.release_all(item.request_id)
                except Exception as e:
                    logger.error(
                        "[api_work_queue] release_all failed after skip_non_pending (worker_idx=%s, request_id=%s, op=%s, status=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        str(status),
                        type(e).__name__,
                        e,
                    )
                continue

            try:
                future_store.mark_running(item.request_id, meta={"worker_idx": int(worker_idx), "op": item.op})
            except Exception as e:
                logger.error(
                    "[api_work_queue] mark_running failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                    int(worker_idx),
                    str(item.request_id),
                    str(item.op),
                    type(e).__name__,
                    e,
                )
                try:
                    future_store.fail(item.request_id, f"internal error: future_store.mark_running failed: {type(e).__name__}: {e}")
                except Exception as e2:
                    logger.error(
                        "[api_work_queue] future_store.fail failed after mark_running error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e2).__name__,
                        e2,
                    )
                try:
                    capacity_manager.release_all(item.request_id)
                except Exception as e2:
                    logger.error(
                        "[api_work_queue] release_all failed after mark_running error (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e2).__name__,
                        e2,
                    )
                continue
            if str(item.op) == "training.create_model":
                logger.info(
                    "[api_work_queue] running request_id=%s op=%s worker_idx=%s",
                    str(item.request_id),
                    str(item.op),
                    int(worker_idx),
                )

            ex = self._executors.get(item.op)
            if ex is None:
                logger.error(
                    "[api_work_queue] unknown op request_id=%s op=%s worker_idx=%s",
                    str(item.request_id),
                    str(item.op),
                    int(worker_idx),
                )
                try:
                    future_store.fail(item.request_id, f"unknown op: {item.op!r}")
                except Exception as e:
                    logger.error(
                        "[api_work_queue] future_store.fail failed for unknown op (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e).__name__,
                        e,
                    )
                try:
                    capacity_manager.release_object_store(item.request_id)
                except Exception as e:
                    logger.error(
                        "[api_work_queue] release_object_store failed for unknown op (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e).__name__,
                        e,
                    )
                continue

            try:
                await ex(item)
                if str(item.op) == "training.create_model":
                    logger.info(
                        "[api_work_queue] done request_id=%s op=%s worker_idx=%s",
                        str(item.request_id),
                        str(item.op),
                        int(worker_idx),
                    )
            except Exception as e:
                logger.error(
                    "[api_work_queue] executor failed (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                    int(worker_idx),
                    str(item.request_id),
                    str(item.op),
                    type(e).__name__,
                    e,
                )
                # Ensure the future does not remain pending forever.
                try:
                    future_store.fail(item.request_id, f"executor failed: {e}")
                except Exception as e2:
                    logger.error(
                        "[api_work_queue] future_store.fail failed after executor exception (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e2).__name__,
                        e2,
                    )
                try:
                    capacity_manager.release_object_store(item.request_id)
                except Exception as e2:
                    logger.error(
                        "[api_work_queue] release_object_store failed after executor exception (worker_idx=%s, request_id=%s, op=%s): %s: %s",
                        int(worker_idx),
                        str(item.request_id),
                        str(item.op),
                        type(e2).__name__,
                        e2,
                    )

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
