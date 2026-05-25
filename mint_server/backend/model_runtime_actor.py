from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from ..config import PFS_PYTHONPATH, actor_runtime_env_vars, apply_detached_actor_resources, otel_env_vars
from ..runtime_env import env_nonempty
from ..logging_context import (
    classify_failure_reason,
    extract_trace_id_from_traceparent,
    get_otel_tracer,
    record_scheduler_decision_otel,
    set_request_id,
    set_trace_id,
)
from .async_ray_control import sync_get_ray_ref
from .model_actor_supervisor import consumer_id_for_replica, queue_id_for_replica
from .model_work_scheduler import ModelWorkSchedulerClient, model_work_scheduler
from .model_work_execution_context import ModelWorkFinalizeBuffer, model_work_execution_context
from .task_payload_store import TaskPayloadStore
from .task_state_store import FutureStatus, task_futures, task_state_store

logger = logging.getLogger(__name__)

ModelWorkExecutor = Callable[[dict[str, Any]], Awaitable[None]]
_EXECUTION_BINDINGS: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelRuntimeActorConfig:
    domain_key: str
    replica_id: str
    actor_name: str
    actor_generation: int
    base_model: str | None = None
    poll_interval_s: float = 0.2
    lease_ttl_s: float = 30.0
    max_claim: int = 1
    token_budget: int | None = None

    @property
    def consumer_id(self) -> str:
        return consumer_id_for_replica(self.domain_key, self.replica_id, self.actor_generation)

    @property
    def queue_id(self) -> str:
        return queue_id_for_replica(self.domain_key, self.replica_id)


def _sanitize_actor_name_part(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return out or "unknown"


def default_model_runtime_actor_name(domain_key: str, replica_id: str) -> str:
    return f"mint_model_runtime_{_sanitize_actor_name_part(domain_key).lower()}_{_sanitize_actor_name_part(replica_id).lower()}"


def _ray_namespace() -> str:
    v = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def get_or_create_model_runtime_actor(
    *,
    domain_key: str,
    replica_id: str = "replica-0",
    actor_name: str | None = None,
    actor_generation: int = 0,
    base_model: str | None = None,
    poll_interval_s: float | None = None,
    lease_ttl_s: float | None = None,
    max_claim: int = 1,
    token_budget: int | None = None,
    runtime_env_extra: dict[str, str] | None = None,
) -> Any:
    import ray

    name = str(actor_name or default_model_runtime_actor_name(domain_key, replica_id))
    try:
        existing = ray.get_actor(name, namespace=_ray_namespace())
        health = sync_get_ray_ref(existing.health_snapshot.remote(), timeout_s=5.0)
        if (
            isinstance(health, dict)
            and str(health.get("domain_key")) == str(domain_key)
            and str(health.get("replica_id")) == str(replica_id)
            and int(health.get("actor_generation") or -1) == int(actor_generation)
        ):
            return existing
        logger.warning(
            "[model_runtime] killing stale detached actor name=%s expected_domain=%s expected_replica=%s expected_generation=%s health=%s",
            name,
            domain_key,
            replica_id,
            actor_generation,
            health,
        )
        ray.kill(existing, no_restart=True)
    except ValueError:
        pass
    except Exception as e:
        logger.warning(
            "[model_runtime] existing actor health check failed name=%s error_type=%s error=%s; recreating",
            name,
            type(e).__name__,
            e,
        )
        try:
            existing = ray.get_actor(name, namespace=_ray_namespace())
            ray.kill(existing, no_restart=True)
        except Exception:
            pass

    remote_cls = ray.remote(num_cpus=0, max_concurrency=64, max_restarts=-1)(ModelRuntimeActor)
    options: dict[str, Any] = {
        "name": name,
        "namespace": _ray_namespace(),
        "lifetime": "detached",
        "get_if_exists": True,
        "max_task_retries": -1,
        "runtime_env": {
            "env_vars": actor_runtime_env_vars(
                pythonpath=PFS_PYTHONPATH,
                extra={
                    **otel_env_vars(),
                    **(runtime_env_extra or {}),
                },
            )
        },
    }
    apply_detached_actor_resources(options, ray)
    return remote_cls.options(**options).remote(
        domain_key=str(domain_key),
        replica_id=str(replica_id),
        actor_name=name,
        actor_generation=int(actor_generation),
        base_model=base_model,
        poll_interval_s=poll_interval_s,
        lease_ttl_s=lease_ttl_s,
        max_claim=max_claim,
        token_budget=token_budget,
    )


async def _default_executor(lease: dict[str, Any]) -> None:
    await _ensure_execution_bindings()
    item = lease.get("item")
    if not isinstance(item, dict):
        raise RuntimeError(f"model work lease missing item: {lease!r}")
    op = str(item.get("op") or "")
    from .model_work_dispatch import execute_model_work_item

    await execute_model_work_item(
        SimpleNamespace(
            request_id=str(item["request_id"]),
            op=op,
            request_json=bytes(item.get("request_json") or b""),
            user_id=None if item.get("user_id") is None else str(item.get("user_id")),
            apikey_id=None if item.get("apikey_id") is None else str(item.get("apikey_id")),
            throttle_principal=(
                None
                if item.get("throttle_principal") is None
                else str(item.get("throttle_principal"))
            ),
            webhook_url=None if item.get("webhook_url") is None else str(item.get("webhook_url")),
            extra=dict(item.get("extra") or {}),
            created_at=float(item.get("created_at") or time.time()),
        ),
        component="model_runtime_actor",
    )


async def _ensure_execution_bindings() -> dict[str, Any]:
    global _EXECUTION_BINDINGS
    if _EXECUTION_BINDINGS is not None:
        return _EXECUTION_BINDINGS
    from .execution_bindings import initialize_execution_bindings

    _EXECUTION_BINDINGS = await initialize_execution_bindings()
    return _EXECUTION_BINDINGS


class ModelRuntimeActor:
    """Model-owned pull executor bound to one scheduler ReplicaQueue."""

    def __init__(
        self,
        *,
        domain_key: str,
        replica_id: str = "replica-0",
        actor_name: str | None = None,
        actor_generation: int = 0,
        base_model: str | None = None,
        poll_interval_s: float | None = None,
        lease_ttl_s: float | None = None,
        max_claim: int = 1,
        token_budget: int | None = None,
        scheduler_client: ModelWorkSchedulerClient | None = None,
        task_futures_client: Any | None = None,
        task_state_store_client: Any | None = None,
        payload_store: TaskPayloadStore | None = None,
        executor: ModelWorkExecutor | None = None,
    ) -> None:
        domain = str(domain_key).strip()
        replica = str(replica_id).strip()
        if not domain:
            raise ValueError("domain_key is required")
        if not replica:
            raise ValueError("replica_id is required")
        self._config = ModelRuntimeActorConfig(
            domain_key=domain,
            replica_id=replica,
            actor_name=str(actor_name or default_model_runtime_actor_name(domain, replica)),
            actor_generation=int(actor_generation),
            base_model=None if base_model is None else str(base_model),
            poll_interval_s=max(
                0.01,
                float(
                    poll_interval_s
                    if poll_interval_s is not None
                    else os.environ.get("MINT_MODEL_RUNTIME_POLL_INTERVAL_S", "0.2")
                ),
            ),
            lease_ttl_s=max(
                0.1,
                float(
                    lease_ttl_s
                    if lease_ttl_s is not None
                    else os.environ.get("MINT_MODEL_RUNTIME_LEASE_TTL_S", "30.0")
                ),
            ),
            max_claim=max(1, int(max_claim)),
            token_budget=None if token_budget is None else int(token_budget),
        )
        self._scheduler = scheduler_client if scheduler_client is not None else model_work_scheduler
        self._task_futures = task_futures_client if task_futures_client is not None else task_futures
        self._task_state_store = (
            task_state_store_client if task_state_store_client is not None else task_state_store
        )
        self._payload_store = payload_store if payload_store is not None else TaskPayloadStore()
        self._executor = executor if executor is not None else _default_executor

        self._running = False
        self._draining = False
        self._loop_task: asyncio.Task | None = None
        self._active_request_id: str | None = None
        self._active_lease_id: str | None = None
        self._started_at = time.time()
        self._last_claimed_at: float | None = None
        self._last_completed_at: float | None = None
        self._last_renewed_at: float | None = None
        self._last_error: str | None = None
        self._last_error_traceback: str | None = None
        self._processed_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._requeued_total = 0
        self._renewed_total = 0
        self._empty_polls_total = 0

    async def _task_state_future_call(self, method: str, **kwargs: Any) -> Any:
        async_method = getattr(self._task_state_store, f"async_future_{method}", None)
        if callable(async_method):
            return await async_method(**kwargs)
        async_method = getattr(self._task_state_store, f"async_{method}", None)
        if callable(async_method):
            return await async_method(**kwargs)
        sync_method = getattr(self._task_state_store, method)
        return sync_method(**kwargs)

    @property
    def domain_key(self) -> str:
        return self._config.domain_key

    @property
    def replica_id(self) -> str:
        return self._config.replica_id

    @property
    def actor_name(self) -> str:
        return self._config.actor_name

    @property
    def actor_generation(self) -> int:
        return self._config.actor_generation

    @property
    def consumer_id(self) -> str:
        return self._config.consumer_id

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "actor_name": self._config.actor_name,
            "domain_key": self._config.domain_key,
            "replica_id": self._config.replica_id,
            "queue_id": self._config.queue_id,
            "consumer_id": self._config.consumer_id,
            "base_model": self._config.base_model,
            "actor_generation": int(self._config.actor_generation),
            "running": bool(self._running),
            "draining": bool(self._draining),
            "active_request_id": self._active_request_id,
            "active_lease_id": self._active_lease_id,
            "started_at": float(self._started_at),
            "last_claimed_at": self._last_claimed_at,
            "last_completed_at": self._last_completed_at,
            "last_renewed_at": self._last_renewed_at,
            "last_error": self._last_error,
            "last_error_traceback": self._last_error_traceback,
            "processed_total": int(self._processed_total),
            "completed_total": int(self._completed_total),
            "failed_total": int(self._failed_total),
            "requeued_total": int(self._requeued_total),
            "renewed_total": int(self._renewed_total),
            "empty_polls_total": int(self._empty_polls_total),
            "poll_interval_s": float(self._config.poll_interval_s),
            "lease_ttl_s": float(self._config.lease_ttl_s),
            "max_claim": int(self._config.max_claim),
            "token_budget": self._config.token_budget,
        }

    async def start(self) -> dict[str, Any]:
        self._draining = False
        if self._running and self._loop_task is not None and not self._loop_task.done():
            return self.health_snapshot()
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        return self.health_snapshot()

    async def drain(self) -> dict[str, Any]:
        self._draining = True
        return self.health_snapshot()

    async def shutdown(self) -> dict[str, Any]:
        self._running = False
        self._draining = True
        task = self._loop_task
        self._loop_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.health_snapshot()

    async def _run_loop(self) -> None:
        while self._running:
            try:
                if self._draining:
                    await asyncio.sleep(self._config.poll_interval_s)
                    continue
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._record_error(e)
                logger.error(
                    "[model_runtime] loop failed actor=%s domain=%s replica=%s error_type=%s failure_reason=%s",
                    self._config.actor_name,
                    self._config.domain_key,
                    self._config.replica_id,
                    type(e).__name__,
                    classify_failure_reason(e),
                )
            await asyncio.sleep(self._config.poll_interval_s)

    async def run_once(self) -> dict[str, Any]:
        if self._draining:
            return {"claimed": 0, "executed": 0, "draining": True}
        claimed = await self._scheduler.claim_from_replica_queue(
            domain_key=self._config.domain_key,
            replica_id=self._config.replica_id,
            consumer_id=self._config.consumer_id,
            consumer_generation=self._config.actor_generation,
            max_items=self._config.max_claim,
            token_budget=self._config.token_budget,
            lease_ttl_s=self._config.lease_ttl_s,
        )
        leases = claimed.get("leases") if isinstance(claimed, dict) else None
        if not leases:
            self._empty_polls_total += 1
            self._clear_transient_scheduler_error()
            return {"claimed": 0, "executed": 0}

        self._last_claimed_at = time.time()
        executed = 0
        for lease in leases:
            if not isinstance(lease, dict):
                continue
            await self._execute_lease(lease)
            executed += 1
        return {"claimed": len(leases), "executed": executed}

    def _restore_item_context(self, lease: dict[str, Any]) -> None:
        item = lease.get("item") if isinstance(lease, dict) else {}
        extra = item.get("extra") if isinstance(item, dict) else {}
        extra = extra if isinstance(extra, dict) else {}
        trace_id = extra.get("_trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            traceparent = extra.get("_traceparent")
            trace_id = extract_trace_id_from_traceparent(traceparent) if isinstance(traceparent, str) else None
        set_trace_id(trace_id if isinstance(trace_id, str) else None)
        request_id = item.get("request_id") if isinstance(item, dict) else None
        if request_id is not None:
            set_request_id(str(request_id))

    def _record_error(self, e: BaseException) -> None:
        self._last_error = f"{type(e).__name__}: {e}"
        self._last_error_traceback = traceback.format_exc()

    def _clear_transient_scheduler_error(self) -> None:
        error = str(self._last_error or "").lower()
        if not error:
            return
        if "modelworkschedulerconflicterror" in error or "consumer_id mismatch" in error:
            self._last_error = None
            self._last_error_traceback = None

    async def _status_is_pending(self, lease: dict[str, Any]) -> bool:
        item = lease["item"]
        request_id = str(item["request_id"])
        lease_id = str(lease["lease_id"])
        try:
            status = await self._task_futures.async_get_status(request_id)
        except KeyError:
            await self._scheduler.complete_lease(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
            )
            return False
        if status is not None and status != FutureStatus.PENDING:
            await self._scheduler.complete_lease(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
            )
            return False
        return True

    async def _fail_lost_lease_if_still_pending(
        self,
        request_id: str,
        error: str,
        *,
        attempt_id: str | None = None,
    ) -> bool:
        expected_meta = {"model_work_attempt_id": attempt_id} if attempt_id else None
        out = await self._task_futures.async_fail_if_pending_meta_matches(
            request_id,
            error,
            expected_meta=expected_meta,
        )
        return bool(out.get("failed")) if isinstance(out, dict) else False

    def _task_state_finalize_enabled(self, lease: dict[str, Any]) -> bool:
        return lease.get("scheduler_epoch") is not None and bool(lease.get("attempt_id"))

    def _require_task_state_finalize(self, lease: dict[str, Any]) -> None:
        if self._task_state_finalize_enabled(lease):
            return
        raise RuntimeError(
            "model work lease missing TaskStateStore finalize metadata "
            "(attempt_id and scheduler_epoch are required)"
        )

    def _lease_attempt_id(self, lease: dict[str, Any]) -> str | None:
        attempt_id = str(lease.get("attempt_id") or "") or None
        if attempt_id:
            return attempt_id
        item = lease.get("item") if isinstance(lease, dict) else {}
        extra = item.get("extra") if isinstance(item, dict) and isinstance(item.get("extra"), dict) else {}
        return str(extra.get("model_work_attempt_id") or "") or None

    def _payload_attempt_id_for_lease(self, lease: dict[str, Any]) -> str:
        attempt_id = str(lease["attempt_id"])
        lease_id = str(lease["lease_id"])
        return f"{attempt_id}__{lease_id}"

    def _staged_payload_path_for_lease(self, lease: dict[str, Any]) -> str:
        item = lease["item"]
        return str(
            self._payload_store.payload_path(
                request_id=str(item["request_id"]),
                attempt_id=self._payload_attempt_id_for_lease(lease),
            )
        )

    async def _commit_task_state_success(
        self,
        lease: dict[str, Any],
        *,
        payload: Any,
        billing_observations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._require_task_state_finalize(lease)
        item = lease["item"]
        request_id = str(item["request_id"])
        attempt_id = str(lease["attempt_id"])
        payload_meta = await asyncio.to_thread(
            self._payload_store.write_json_payload,
            request_id=request_id,
            attempt_id=self._payload_attempt_id_for_lease(lease),
            payload=payload,
        )
        await self._task_state_future_call(
            "commit_finalize_success",
            request_id=request_id,
            lease_id=str(lease["lease_id"]),
            attempt_id=attempt_id,
            scheduler_epoch=int(lease["scheduler_epoch"]),
            runtime_generation=int(self._config.actor_generation),
            result_path=str(payload_meta["path"]),
            result_checksum=str(payload_meta["checksum"]),
            result_size_bytes=int(payload_meta["size_bytes"]),
            billing_observations=billing_observations,
        )

    async def _commit_task_state_failure(
        self,
        lease: dict[str, Any],
        *,
        error: str,
    ) -> None:
        self._require_task_state_finalize(lease)
        item = lease["item"]
        await self._task_state_future_call(
            "commit_finalize_failure",
            request_id=str(item["request_id"]),
            lease_id=str(lease["lease_id"]),
            attempt_id=str(lease["attempt_id"]),
            scheduler_epoch=int(lease["scheduler_epoch"]),
            runtime_generation=int(self._config.actor_generation),
            error=str(error),
        )

    async def _mark_running(self, lease: dict[str, Any]) -> None:
        item = lease["item"]
        extra = item.get("extra") if isinstance(item, dict) else {}
        extra = extra if isinstance(extra, dict) else {}
        running_at = time.time()
        queued_at = extra.get("queued_at")
        queue_wait_s = None
        if isinstance(queued_at, (int, float)):
            queue_wait_s = max(0.0, running_at - float(queued_at))
        op = str(item.get("op") or "unknown")
        await self._task_futures.async_mark_running(
            str(item["request_id"]),
            meta={
                "consumer_id": self._config.consumer_id,
                "actor_name": self._config.actor_name,
                "actor_generation": int(self._config.actor_generation),
                "domain_key": self._config.domain_key,
                "replica_id": self._config.replica_id,
                "queue_id": self._config.queue_id,
                "lease_id": str(lease["lease_id"]),
                "op": op,
                "queue_state": "running",
                "stage": "prefill",
                "queued_at": queued_at,
                "dequeue_at": running_at,
                "running_at": running_at,
                "executor_started_at": running_at,
                "queue_wait_s": queue_wait_s,
            },
        )
        if queue_wait_s is not None:
            record_scheduler_decision_otel(
                op=op,
                backend="model_work",
                queue_kind="model_work_scheduler",
                reason="lease_claimed",
                queue_wait_s=queue_wait_s,
                switched=False,
            )

    async def _renew_until_done(self, lease: dict[str, Any], task: asyncio.Task) -> None:
        interval_s = max(0.1, min(float(self._config.lease_ttl_s) / 3.0, 10.0))
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=interval_s)
                return
            except asyncio.TimeoutError:
                result = await self._scheduler.renew_lease(
                    lease_id=str(lease["lease_id"]),
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
                    lease_ttl_s=self._config.lease_ttl_s,
                )
                if isinstance(result, dict) and bool(result.get("ok")):
                    self._last_renewed_at = time.time()
                    self._renewed_total += 1
                    continue
                raise RuntimeError(f"model work lease renew failed: {result!r}")

    async def _run_executor(self, lease: dict[str, Any]) -> None:
        item = lease.get("item") if isinstance(lease, dict) else {}
        tracer = get_otel_tracer()
        if tracer is None:
            await self._executor(lease)
            return
        try:
            from opentelemetry.propagate import extract
            from opentelemetry.trace import SpanKind, Status, StatusCode
        except Exception:
            await self._executor(lease)
            return

        span_context = None
        extra = item.get("extra") if isinstance(item, dict) else {}
        traceparent = extra.get("_traceparent") if isinstance(extra, dict) else None
        if isinstance(traceparent, str) and traceparent:
            try:
                span_context = extract({"traceparent": traceparent})
            except Exception:
                span_context = None
        with tracer.start_as_current_span(
            "model_runtime.execute",
            kind=SpanKind.INTERNAL,
            context=span_context,
        ) as span:
            span.set_attribute("component", "model_runtime_actor")
            span.set_attribute("op", str(item.get("op") if isinstance(item, dict) else "unknown"))
            span.set_attribute("request_id", str(item.get("request_id") if isinstance(item, dict) else "unknown"))
            span.set_attribute("domain_key", self._config.domain_key)
            span.set_attribute("replica_id", self._config.replica_id)
            span.set_attribute("actor_name", self._config.actor_name)
            try:
                await self._executor(lease)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    async def _execute_lease(self, lease: dict[str, Any]) -> None:
        item = lease["item"]
        request_id = str(item["request_id"])
        lease_id = str(lease["lease_id"])
        attempt_id = self._lease_attempt_id(lease)
        self._active_request_id = request_id
        self._active_lease_id = lease_id
        self._restore_item_context(lease)
        if not await self._status_is_pending(lease):
            self._active_request_id = None
            self._active_lease_id = None
            return

        try:
            await self._mark_running(lease)
        except Exception as e:
            self._record_error(e)
            try:
                await self._scheduler.fail_lease(
                    lease_id=lease_id,
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
                    reason="mark_running_failed",
                    requeue=True,
                )
                self._requeued_total += 1
            except Exception:
                pass
            self._active_request_id = None
            self._active_lease_id = None
            return

        task: asyncio.Task | None = None
        try:
            finalize_buffer = ModelWorkFinalizeBuffer()
            executor_started_at = time.time()
            with model_work_execution_context(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                finalize_buffer=finalize_buffer,
            ):
                task = asyncio.create_task(self._run_executor(lease))
                await self._renew_until_done(lease, task)
                await task
            executor_done_at = time.time()
            try:
                await self._task_futures.async_update_meta(
                    request_id,
                    {
                        "executor_done_at": executor_done_at,
                        "executor_exec_s": max(0.0, executor_done_at - executor_started_at),
                    },
                )
            except Exception:
                pass
            finalization = finalize_buffer.finalization
            if finalization is None:
                raise RuntimeError("model work executor finished without resolving or failing future")
            if finalization.request_id != request_id:
                raise RuntimeError(
                    f"model work executor finalized wrong request_id: {finalization.request_id!r}"
                )
            if finalization.kind not in {"resolve", "fail"}:
                raise RuntimeError(f"unknown model work finalization kind: {finalization.kind!r}")
            begin_finalize = await self._scheduler.begin_finalize_lease(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                finalize_ttl_s=self._config.lease_ttl_s,
                staged_payload_path=(
                    self._staged_payload_path_for_lease(lease)
                    if finalization.kind == "resolve"
                    else None
                ),
            )
            if not isinstance(begin_finalize, dict) or not bool(begin_finalize.get("ok")):
                logger.warning(
                    "[model_runtime] lease finalize rejected actor=%s request_id=%s result=%s",
                    self._config.actor_name,
                    request_id,
                    begin_finalize,
                )
                try:
                    failed = await self._fail_lost_lease_if_still_pending(
                        request_id,
                        "model work scheduler lost active lease; request must be retried",
                        attempt_id=attempt_id,
                    )
                    if failed:
                        self._failed_total += 1
                except Exception as e:
                    logger.error(
                        "[model_runtime] lost-lease task_futures.fail failed actor=%s request_id=%s error_type=%s error=%s",
                        self._config.actor_name,
                        request_id,
                        type(e).__name__,
                        e,
                    )
                self._requeued_total += 1
                return
            task_state_committed = False
            try:
                if finalization.kind == "resolve":
                    await self._commit_task_state_success(
                        lease,
                        payload=finalization.payload,
                        billing_observations=finalization.billing_observations,
                    )
                    task_state_committed = True
                else:
                    await self._commit_task_state_failure(lease, error=str(finalization.payload))
                    task_state_committed = True
            except Exception as e:
                if task_state_committed:
                    logger.error(
                        "[model_runtime] task_state finalize failed after commit actor=%s request_id=%s error_type=%s error=%s",
                        self._config.actor_name,
                        request_id,
                        type(e).__name__,
                        e,
                    )
                else:
                    try:
                        await self._scheduler.fail_lease(
                            lease_id=lease_id,
                            consumer_id=self._config.consumer_id,
                            consumer_generation=self._config.actor_generation,
                            reason="task_state_finalize_failed",
                            requeue=True,
                        )
                        self._requeued_total += 1
                    except Exception:
                        pass
                    return
            try:
                if finalization.kind == "resolve":
                    completed = await self._scheduler.complete_lease(
                        lease_id=lease_id,
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
                    )
                    if not isinstance(completed, dict) or not bool(completed.get("ok")):
                        logger.warning(
                            "[model_runtime] lease complete rejected after future resolve actor=%s request_id=%s result=%s",
                            self._config.actor_name,
                            request_id,
                            completed,
                        )
                    self._processed_total += 1
                    self._completed_total += 1
                    self._last_completed_at = time.time()
                    self._last_error = None
                    self._last_error_traceback = None
                    return
            except Exception:
                try:
                    await self._scheduler.fail_lease(
                        lease_id=lease_id,
                        consumer_id=self._config.consumer_id,
                        consumer_generation=self._config.actor_generation,
                        reason="scheduler_complete_failed",
                        requeue=True,
                    )
                    self._requeued_total += 1
                except Exception:
                    pass
                return

            failed = await self._scheduler.fail_lease(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                reason="future_failed",
                requeue=False,
            )
            if not isinstance(failed, dict) or not bool(failed.get("ok")):
                logger.warning(
                    "[model_runtime] lease fail rejected after future fail actor=%s request_id=%s result=%s",
                    self._config.actor_name,
                    request_id,
                    failed,
                )
            self._processed_total += 1
            self._failed_total += 1
            self._last_error = f"future failed: {finalization.payload}"
            self._last_error_traceback = None
            self._last_completed_at = time.time()
        except asyncio.CancelledError:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            try:
                await self._scheduler.fail_lease(
                    lease_id=lease_id,
                    consumer_id=self._config.consumer_id,
                    consumer_generation=self._config.actor_generation,
                    reason="model_runtime_cancelled",
                    requeue=True,
                )
                self._requeued_total += 1
            except Exception:
                pass
            raise
        except Exception as e:
            self._record_error(e)
            logger.error(
                "[model_runtime] executor failed actor=%s request_id=%s op=%s error_type=%s failure_reason=%s",
                self._config.actor_name,
                request_id,
                item.get("op"),
                type(e).__name__,
                classify_failure_reason(e),
            )
            begin_finalize = await self._scheduler.begin_finalize_lease(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                finalize_ttl_s=self._config.lease_ttl_s,
            )
            if not isinstance(begin_finalize, dict) or not bool(begin_finalize.get("ok")):
                logger.warning(
                    "[model_runtime] failure lease finalize rejected actor=%s request_id=%s result=%s",
                    self._config.actor_name,
                    request_id,
                    begin_finalize,
                )
                try:
                    failed = await self._fail_lost_lease_if_still_pending(
                        request_id,
                        "model work scheduler lost active lease after executor failure; request must be retried",
                        attempt_id=attempt_id,
                    )
                    if failed:
                        self._failed_total += 1
                except Exception as e2:
                    logger.error(
                        "[model_runtime] lost-failure-lease task_futures.fail failed actor=%s request_id=%s error_type=%s error=%s",
                        self._config.actor_name,
                        request_id,
                        type(e2).__name__,
                        e2,
                    )
                self._requeued_total += 1
                return
            task_state_committed = False
            try:
                await self._commit_task_state_failure(lease, error=f"executor failed: {e}")
                task_state_committed = True
            except Exception as e2:
                logger.error(
                    "[model_runtime] task_state failure finalize failed actor=%s request_id=%s error_type=%s error=%s",
                    self._config.actor_name,
                    request_id,
                    type(e2).__name__,
                    e2,
                )
                if not task_state_committed:
                    try:
                        await self._scheduler.fail_lease(
                            lease_id=lease_id,
                            consumer_id=self._config.consumer_id,
                            consumer_generation=self._config.actor_generation,
                            reason="task_state_finalize_failed",
                            requeue=True,
                        )
                        self._requeued_total += 1
                    except Exception:
                        pass
                    return
            await self._scheduler.fail_lease(
                lease_id=lease_id,
                consumer_id=self._config.consumer_id,
                consumer_generation=self._config.actor_generation,
                reason="executor_failed",
                requeue=False,
            )
            self._processed_total += 1
            self._failed_total += 1
        finally:
            self._active_request_id = None
            self._active_lease_id = None
