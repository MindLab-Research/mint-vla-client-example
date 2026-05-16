"""Internal API routes for usage tracking, health checks, and checkpoint management.

Checkpoint endpoints follow /internal/v1/checkpoints spec:
- GET /checkpoints: List all user's checkpoints
- GET /checkpoints/{checkpoint_id}/archive: Download checkpoint as tar.gz
"""

import logging
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from ..auth_identity import can_bypass_ownership
from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..checkpoint_index import (
    checkpoint_index_enabled,
    get_catalog_checkpoint,
    list_catalog_checkpoints,
)
from ..health_checks import deep_healthz_response
from ..logging_context import get_otel_tracer
from ..queue_priority import merge_queue_priority_extra
from ..ray_cluster_health import get_ray_cluster_health_snapshot
from ..ray_gcs_metrics import get_ray_gcs_metrics_snapshot
from ..usage_store import get_usage_store

router = APIRouter()
logger = logging.getLogger(__name__)


async def _enqueue_internal_request_with_trace(
    *,
    route_start_s: float,
    request_id: str,
    op: str,
    enqueue_coro,
) -> None:
    tracer = get_otel_tracer()
    future_ready_elapsed_ms = (time.perf_counter() - route_start_s) * 1000.0
    if tracer is None:
        await enqueue_coro
        return

    with tracer.start_as_current_span(f"{op}.enqueue") as span:
        span.set_attribute("component", "routes.internal")
        span.set_attribute("op", str(op))
        span.set_attribute("request_id", str(request_id))
        span.add_event(
            "task_state_futures_ready",
            {
                "elapsed_ms": round(future_ready_elapsed_ms, 3),
                "route_elapsed_ms": round(future_ready_elapsed_ms, 3),
            },
        )
        enqueue_start_s = time.perf_counter()
        await enqueue_coro
        span.add_event(
            "enqueue_done",
            {
                "elapsed_ms": round((time.perf_counter() - enqueue_start_s) * 1000.0, 3),
                "route_elapsed_ms": round((time.perf_counter() - route_start_s) * 1000.0, 3),
            },
        )


def _get_account_id(request: Request) -> str | None:
    """Extract account_id from request state (set by gateway auth middleware)."""
    user_data = _request_user_data(request)
    if user_data:
        account_id = user_data.get("account_id")
        if account_id:
            return account_id
        user_id = user_data.get("user_id")
        if user_id and str(user_id) != "admin":
            return str(user_id)
    return None


def _get_user_id(request: Request) -> str | None:
    return _request_user_id(request)


class UsageLogEntry(BaseModel):
    """Single usage log entry."""

    source_index: int
    event_time: str
    account_id: str
    apikey_id: str
    charge_item: str
    quantity: int
    request_id: str
    label: str


class UsageLogsResponse(BaseModel):
    """Response for usage logs query."""

    logs: list[UsageLogEntry]
    count: int
    has_more: bool
    next_offset: int | None


class UsageSummaryResponse(BaseModel):
    """Response for usage summary."""

    total_quantity: int
    charge_item_totals: dict[str, int]


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str
    database: str
    timestamp: str


@router.get("/usage_logs", response_model=UsageLogsResponse)
async def get_usage_logs(
    request: Request,
    since: Annotated[str | None, Query(description="ISO 8601 timestamp")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Query usage logs for the authenticated user.

    Logs are automatically filtered by the user_id extracted from the sk- token.

    Args:
        since: Only return logs after this timestamp (ISO 8601)
        limit: Maximum number of logs to return (1-1000)
        offset: Number of logs to skip for pagination
    """
    # Get account_id from gateway forwarded auth headers
    account_id = _get_account_id(request)
    if account_id is None and not can_bypass_ownership(request):
        raise HTTPException(status_code=403, detail="Access denied")

    # Parse 'since' timestamp if provided
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            since_dt = None

    usage_store = await get_usage_store()
    logs, total_count, has_more = await usage_store.query_logs(
        since=since_dt,
        account_id=account_id,
        limit=limit,
        offset=offset,
    )

    return UsageLogsResponse(
        logs=[UsageLogEntry(**log) for log in logs],
        count=total_count,
        has_more=has_more,
        next_offset=offset + limit if has_more else None,
    )


@router.get("/usage_summary/{account_id}", response_model=UsageSummaryResponse)
async def get_usage_summary(account_id: str, request: Request):
    """Get usage summary for a specific account.

    Args:
        account_id: The account ID to get summary for
    """
    request_account_id = _get_account_id(request)
    if not can_bypass_ownership(request) and request_account_id is None:
        raise HTTPException(status_code=403, detail="Access denied")
    if not can_bypass_ownership(request) and account_id != request_account_id:
        raise HTTPException(status_code=403, detail="Access denied")

    usage_store = await get_usage_store()
    summary = await usage_store.get_account_summary(account_id)

    return UsageSummaryResponse(
        total_quantity=summary["total_quantity"],
        charge_item_totals=summary["charge_item_totals"],
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for internal monitoring."""
    usage_store = await get_usage_store()
    db_status = "ok" if await usage_store.health_check() else "error"

    return HealthResponse(
        status="ok",
        database=db_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/healthz/deep", response_model=None)
async def deep_health_check():
    """Costly internal health endpoint with active Ray diagnostics."""
    return await deep_healthz_response()


def _self_rss_bytes() -> int:
    with open("/proc/self/statm", encoding="utf-8") as f:
        parts = f.read().strip().split()
    if len(parts) < 2:
        raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
    rss_pages = int(parts[1])
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    return rss_pages * page_size


def _model_actor_registry_local_snapshot() -> list[dict]:
    from ..backend.model_actor_registry import get_model_actor_registry

    pool = get_model_actor_registry()
    return pool.cached_snapshot()


@router.get("/admission_stats")
async def admission_stats(*, include_actor_rss: bool = True) -> dict:
    from ..backend.task_state_store import task_state_futures
    from ..backend.model_actor_supervisor import model_actor_supervisor
    from ..backend.model_work_scheduler import model_work_scheduler
    from ..backend.maintenance_cron_actor import maintenance_cron_actor
    from ..backend.session_heartbeat_store import session_heartbeat_store
    from ..routes import sampling as sampling_route
    from ..routes import service as service_route

    timeout_s = 10.0

    model_scheduler = None
    try:
        model_scheduler = await model_work_scheduler.stats(timeout_s=timeout_s)
        if not isinstance(model_scheduler, dict):
            model_scheduler = {"error": f"model_work_scheduler snapshot returned non-dict: {type(model_scheduler)}"}
    except Exception as e:
        model_scheduler = {"error": f"{type(e).__name__}: {e}"}

    model_supervisor = None
    try:
        model_supervisor = await model_actor_supervisor.async_snapshot(timeout_s=timeout_s)
        if not isinstance(model_supervisor, dict):
            model_supervisor = {"error": f"model_actor_supervisor snapshot returned non-dict: {type(model_supervisor)}"}
    except Exception as e:
        model_supervisor = {"error": f"{type(e).__name__}: {e}"}

    fs = None
    try:
        if not include_actor_rss and hasattr(task_state_futures, "metrics_snapshot"):
            fs = task_state_futures.metrics_snapshot()
        elif hasattr(task_state_futures, "async_ensure_ready"):
            fs = await task_state_futures.async_ensure_ready(timeout_s=timeout_s)
        else:
            fs = task_state_futures.ensure_ready(timeout_s=timeout_s)
    except Exception as e:
        fs = {"error": f"{type(e).__name__}: {e}"}

    actors: dict = {}
    if include_actor_rss:
        try:
            actors["task_state_futures"] = {
                "rss_bytes": int(await task_state_futures.async_rss_bytes(timeout_s=timeout_s))
            }
        except Exception as e:
            actors["task_state_futures"] = {"error": f"{type(e).__name__}: {e}"}

        try:
            from ..backend.model_actor_registry import get_model_actor_registry

            pool = get_model_actor_registry()
            actors["model_actor_registry"] = pool.rss_snapshot(timeout_s=timeout_s)
            actors["model_actor_registry_metadata_cache"] = pool.metadata_cache_metrics_snapshot()
            actors["model_actor_registry_lifecycle"] = pool.lifecycle_metrics_snapshot()
        except Exception as e:
            actors["model_actor_registry"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        # Metrics scrapes must stay cheap. A single hung actor in rss_snapshot()
        # can otherwise block the API thread and stall unrelated routes.
        try:
            from ..backend.model_actor_registry import get_model_actor_registry

            pool = get_model_actor_registry()
            actors["model_actor_registry"] = pool.cached_snapshot()
            actors["model_actor_registry_metadata_cache"] = pool.metadata_cache_metrics_snapshot()
            actors["model_actor_registry_lifecycle"] = pool.lifecycle_metrics_snapshot()
        except Exception as e:
            actors["model_actor_registry"] = {"error": f"{type(e).__name__}: {e}"}

    proc = {"pid": int(os.getpid())}
    try:
        proc["rss_bytes"] = int(_self_rss_bytes())
    except Exception as e:
        proc["rss_error"] = f"{type(e).__name__}: {e}"

    try:
        session_heartbeat_entries = int(await session_heartbeat_store.async_size())
    except Exception as e:
        session_heartbeat_entries = 0
        actors["session_heartbeat_store"] = {"error": f"{type(e).__name__}: {e}"}

    driver_state: dict = {
        "sdk_sessions_fallback": 0,
        "session_heartbeat_entries": session_heartbeat_entries,
    }
    try:
        driver_state["lora_load_locks"] = int(await sampling_route._lora_load_lock_count())
    except Exception as e:
        driver_state["lora_load_locks_error"] = f"{type(e).__name__}: {e}"

    manager = service_route.session_manager
    if manager is not None:
        try:
            driver_state.update(manager.observability_snapshot())
        except Exception as e:
            driver_state["sampling_sessions_error"] = f"{type(e).__name__}: {e}"

    try:
        from ..backend.dense_session_state import collect_dense_session_state_stats

        driver_state.update(collect_dense_session_state_stats())
    except Exception as e:
        driver_state["dense_session_state_error"] = f"{type(e).__name__}: {e}"

    ray_cluster = None
    try:
        ray_cluster = get_ray_cluster_health_snapshot()
    except Exception as e:
        ray_cluster = {"error": f"{type(e).__name__}: {e}"}

    ray_gcs_metrics = None
    try:
        ray_gcs_metrics = get_ray_gcs_metrics_snapshot()
    except Exception as e:
        ray_gcs_metrics = {"error": f"{type(e).__name__}: {e}"}

    maintenance_cron = None
    try:
        maintenance_cron = await maintenance_cron_actor.async_health_snapshot(timeout_s=timeout_s)
    except Exception as e:
        maintenance_cron = {"error": f"{type(e).__name__}: {e}"}

    return {
        "model_work_scheduler": model_scheduler,
        "model_actor_supervisor": model_supervisor,
        "task_state_futures": fs,
        "actors": actors,
        "process": proc,
        "driver_state": driver_state,
        "ray_cluster": ray_cluster,
        "ray_gcs_metrics": ray_gcs_metrics,
        "maintenance_cron_actor": maintenance_cron,
    }


@router.get("/maintenance_cron_actor")
async def maintenance_cron_actor_health() -> dict:
    from ..backend.maintenance_cron_actor import maintenance_cron_actor

    return await maintenance_cron_actor.async_health_snapshot(timeout_s=10.0)


@router.get("/model_work_scheduler")
async def model_work_scheduler_health() -> dict:
    from ..backend.model_work_scheduler import model_work_scheduler

    return await model_work_scheduler.stats(timeout_s=10.0)


@router.get("/model_actor_supervisor")
async def model_actor_supervisor_health() -> dict:
    from ..backend.model_actor_supervisor import model_actor_supervisor

    return await model_actor_supervisor.async_snapshot(timeout_s=10.0)


@router.get("/ray_cluster_health")
async def ray_cluster_health() -> dict:
    return get_ray_cluster_health_snapshot()


@router.get("/ray_gcs_metrics")
async def ray_gcs_metrics() -> dict:
    return get_ray_gcs_metrics_snapshot()


def _prom_sanitize_name(v: str) -> str:
    out = []
    for ch in str(v):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    if not s:
        s = "value"
    if s[0].isdigit():
        s = f"_{s}"
    return s


def _prom_escape_label_value(v: object) -> str:
    return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _prom_number(v: object) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        fv = float(v)
        if math.isfinite(fv):
            return fv
    return None


def _prom_format_number(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.12g}"


def _append_metric(
    lines: list[str],
    metric_name: str,
    value: object,
    labels: dict[str, object] | None = None,
) -> None:
    num = _prom_number(value)
    if num is None:
        return

    name = _prom_sanitize_name(metric_name)
    if labels:
        label_parts = []
        for k, raw_v in sorted(labels.items()):
            if raw_v is None:
                continue
            key = _prom_sanitize_name(k)
            label_parts.append(f'{key}="{_prom_escape_label_value(raw_v)}"')
        if label_parts:
            lines.append(f"{name}{{{','.join(label_parts)}}} {_prom_format_number(num)}")
            return
    lines.append(f"{name} {_prom_format_number(num)}")


def _append_raw_prom_sample(
    lines: list[str],
    metric_name: str,
    value: object,
    labels: dict[str, object] | None = None,
) -> None:
    num = _prom_number(value)
    if num is None:
        return

    if labels:
        label_parts = []
        for key, raw_v in sorted(labels.items()):
            if raw_v is None:
                continue
            label_parts.append(f'{key}="{_prom_escape_label_value(raw_v)}"')
        if label_parts:
            lines.append(f"{metric_name}{{{','.join(label_parts)}}} {_prom_format_number(num)}")
            return
    lines.append(f"{metric_name} {_prom_format_number(num)}")


def _scheduler_domain_base_model(scheduler_domain: object) -> str | None:
    domain = str(scheduler_domain or "").strip()
    if not domain or ":" not in domain:
        return None
    _backend, domain_key = domain.split(":", 1)
    domain_key = domain_key.strip()
    if not domain_key:
        return None
    if "::replica::" in domain_key:
        domain_key = domain_key.split("::replica::", 1)[0].strip()
    return domain_key or None


def _actor_workload(actor_type: object) -> str:
    return "sample" if str(actor_type or "").strip().lower() == "vllm" else "train"


def _model_actor_registry_gpu_bindings(rec: dict[str, object]) -> list[dict[str, str]]:
    metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    actor_name = str(rec.get("actor_name") or "unknown")
    actor_type = str(rec.get("actor_type") or "unknown")
    workload = _actor_workload(actor_type)

    bindings = metadata.get("gpu_bindings") if isinstance(metadata, dict) else None
    if isinstance(bindings, list):
        out: list[dict[str, str]] = []
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            gpu_index = binding.get("gpu_index")
            gpu_uuid = binding.get("gpu_uuid")
            if gpu_index is None and not (isinstance(gpu_uuid, str) and gpu_uuid.strip()):
                continue
            labels = {
                "actor_name": actor_name,
                "workload": workload,
                "hostname": str(binding.get("hostname") or metadata.get("hostname") or "unknown"),
            }
            if gpu_index is not None:
                labels["gpu_index"] = str(gpu_index)
            if isinstance(gpu_uuid, str) and gpu_uuid.strip():
                labels["gpu_uuid"] = gpu_uuid.strip()
            out.append(labels)
        if out:
            return out

    gpu_indices = metadata.get("gpu_indices") if isinstance(metadata, dict) else None
    if not isinstance(gpu_indices, list):
        return []
    hostname = str(metadata.get("hostname") or "unknown") if isinstance(metadata, dict) else "unknown"
    out = []
    for gpu_index in gpu_indices:
        out.append(
            {
                "actor_name": actor_name,
                "workload": workload,
                "hostname": hostname,
                "gpu_index": str(gpu_index),
            }
        )
    return out


@router.get("/metrics")
async def metrics() -> Response:
    stats = await admission_stats(include_actor_rss=False)
    lines: list[str] = []
    megatron_actor_lifecycle_counts: dict[tuple[str, str], float] = {}

    model_scheduler = stats.get("model_work_scheduler")
    if isinstance(model_scheduler, dict):
        _append_metric(lines, "mint_model_work_scheduler_depth", model_scheduler.get("depth"))
        _append_metric(lines, "mint_model_work_scheduler_backlog_depth", model_scheduler.get("backlog_depth"))
        sample_model_load: dict[str, dict[str, float]] = {}
        counters = model_scheduler.get("counters")
        if isinstance(counters, dict):
            for key in ("appended", "assigned", "claimed", "completed", "failed", "requeued"):
                _append_metric(lines, f"mint_model_work_scheduler_{key}_total", counters.get(key))
        backlog_by_domain = model_scheduler.get("backlog_depth_by_domain")
        if isinstance(backlog_by_domain, dict):
            for domain_key, depth in backlog_by_domain.items():
                _append_metric(
                    lines,
                    "mint_model_work_scheduler_domain_backlog_depth",
                    depth,
                    labels={"domain_key": domain_key},
                )
        replica_queues = model_scheduler.get("replica_queues")
        if isinstance(replica_queues, dict):
            for queue_id, rec in replica_queues.items():
                if not isinstance(rec, dict):
                    continue
                labels = {
                    "domain_key": rec.get("domain_key") or "unknown",
                    "replica_id": rec.get("replica_id") or "unknown",
                    "queue_id": queue_id,
                    "status": rec.get("status") or "unknown",
                }
                _append_metric(lines, "mint_model_work_scheduler_replica_queue_depth", rec.get("depth"), labels=labels)
                domain_key = str(rec.get("domain_key") or "")
                if domain_key.startswith("vllm:"):
                    base_model = _scheduler_domain_base_model(domain_key)
                    if base_model:
                        bucket = sample_model_load.setdefault(
                            base_model,
                            {"pending_requests": 0.0, "inflight_workers": 0.0, "capacity_workers": 0.0},
                        )
                        bucket["pending_requests"] += float(_prom_number(rec.get("depth")) or 0.0)
                        if str(rec.get("status") or "").lower() in {"healthy", "ready"}:
                            bucket["capacity_workers"] += 1.0
        leases = model_scheduler.get("leases")
        if isinstance(leases, list):
            _append_metric(lines, "mint_model_work_scheduler_leases", len(leases))
            for lease in leases:
                if not isinstance(lease, dict):
                    continue
                item = lease.get("item") if isinstance(lease.get("item"), dict) else {}
                domain_key = str(item.get("domain_key") or lease.get("domain_key") or "")
                if not domain_key.startswith("vllm:"):
                    continue
                base_model = _scheduler_domain_base_model(domain_key)
                if not base_model:
                    continue
                bucket = sample_model_load.setdefault(
                    base_model,
                    {"pending_requests": 0.0, "inflight_workers": 0.0, "capacity_workers": 0.0},
                )
                bucket["inflight_workers"] += 1.0
        for base_model, agg in sorted(sample_model_load.items()):
            labels = {"base_model": base_model, "workload": "sample"}
            capacity_workers = float(agg.get("capacity_workers", 0.0))
            load_pct = 0.0 if capacity_workers <= 0 else (
                100.0 * float(agg.get("inflight_workers", 0.0)) / capacity_workers
            )
            _append_metric(lines, "mint_model_load_pct", load_pct, labels=labels)
            _append_metric(lines, "mint_model_pending_requests", agg.get("pending_requests"), labels=labels)
            _append_metric(lines, "mint_model_inflight_workers", agg.get("inflight_workers"), labels=labels)
            _append_metric(lines, "mint_model_capacity_workers", agg.get("capacity_workers"), labels=labels)

    model_supervisor = stats.get("model_actor_supervisor")
    if isinstance(model_supervisor, dict):
        for key in (
            "desired_total",
            "managed_total",
            "domain_total",
            "reconcile_total",
            "created_total",
            "restarted_total",
            "blocked_total",
            "busy_recycle_skipped_total",
            "scheduler_sync_failures_total",
        ):
            _append_metric(lines, f"mint_model_actor_supervisor_{key}", model_supervisor.get(key))
        domains = model_supervisor.get("domains")
        if isinstance(domains, dict):
            for domain_key, rec in domains.items():
                if not isinstance(rec, dict):
                    continue
                labels = {"domain_key": domain_key}
                _append_metric(lines, "mint_model_actor_supervisor_domain_replicas", rec.get("replicas"), labels=labels)
                _append_metric(lines, "mint_model_actor_supervisor_domain_healthy", rec.get("healthy"), labels=labels)
                _append_metric(lines, "mint_model_actor_supervisor_domain_unhealthy", rec.get("unhealthy"), labels=labels)
        replicas = model_supervisor.get("replicas")
        if isinstance(replicas, dict):
            for replica_key, rec in replicas.items():
                if not isinstance(rec, dict):
                    continue
                labels = {
                    "domain_key": rec.get("domain_key") or "unknown",
                    "replica_id": rec.get("replica_id") or "unknown",
                    "actor_name": rec.get("actor_name") or "unknown",
                    "state": rec.get("state") or "unknown",
                }
                _append_metric(lines, "mint_model_actor_supervisor_replica_state", 1, labels=labels)
                _append_metric(lines, "mint_model_actor_supervisor_replica_generation", rec.get("generation"), labels=labels)

    fs = stats.get("task_state_futures")
    if isinstance(fs, dict):
        for key in (
            "pending",
            "results",
            "errors",
            "refs",
            "meta",
            "expired",
            "retrieved",
            "execution_timeout_s",
            "queue_timeout_s",
            "result_ttl_s",
            "tombstone_ttl_s",
        ):
            _append_metric(lines, f"mint_task_state_futures_{key}", fs.get(key))

        by_op = fs.get("by_op")
        if isinstance(by_op, dict):
            for op, counters in by_op.items():
                if not isinstance(counters, dict):
                    continue
                _append_metric(lines, "mint_task_state_futures_pending", counters.get("pending"), labels={"op": op})
                _append_metric(lines, "mint_task_state_futures_results", counters.get("results"), labels={"op": op})
                _append_metric(lines, "mint_task_state_futures_errors", counters.get("errors"), labels={"op": op})

        age_stats = fs.get("age_stats")
        if isinstance(age_stats, dict):
            _append_metric(lines, "mint_task_state_futures_oldest_pending_s", age_stats.get("oldest_pending_s"))
            _append_metric(lines, "mint_task_state_futures_oldest_done_s", age_stats.get("oldest_done_s"))
            _append_metric(lines, "mint_task_state_futures_avg_pending_s", age_stats.get("avg_pending_s"))
            _append_metric(lines, "mint_task_state_futures_avg_done_s", age_stats.get("avg_done_s"))

        payload_stats = fs.get("payload_stats")
        if isinstance(payload_stats, dict):
            _append_metric(
                lines, "mint_task_state_futures_result_refs_count", payload_stats.get("result_refs_count")
            )
            _append_metric(lines, "mint_task_state_futures_errors_count", payload_stats.get("errors_count"))
            _append_metric(lines, "mint_task_state_futures_refs_count", payload_stats.get("refs_count"))

        timeout_counts = fs.get("timeout_counts")
        if isinstance(timeout_counts, dict):
            for kind in ("queue", "execution", "total"):
                _append_metric(
                    lines,
                    "mint_task_state_futures_timeouts_total",
                    timeout_counts.get(kind),
                    labels={"kind": kind},
                )
            by_op = timeout_counts.get("by_op")
            if isinstance(by_op, dict):
                for op, rec in by_op.items():
                    if not isinstance(rec, dict):
                        continue
                    for kind in ("queue", "execution", "total"):
                        _append_metric(
                            lines,
                            "mint_task_state_futures_timeouts_total",
                            rec.get(kind),
                            labels={"op": op, "kind": kind},
                        )

    actors = stats.get("actors")
    if isinstance(actors, dict):
        for actor_key in ("task_state_futures",):
            rec = actors.get(actor_key)
            if isinstance(rec, dict):
                _append_metric(
                    lines,
                    "mint_actor_rss_bytes",
                    rec.get("rss_bytes"),
                    labels={"actor": actor_key},
                )

        model_actor_registry = actors.get("model_actor_registry")
        if isinstance(model_actor_registry, list):
            grouped: dict[tuple[str, str], dict[str, float]] = {}
            dense_poisoned_grouped: dict[tuple[str, str], float] = {}
            for rec in model_actor_registry:
                if not isinstance(rec, dict):
                    continue
                actor_type = str(rec.get("actor_type") or "unknown")
                model = str(rec.get("base_model") or "unknown")
                actor_name = str(rec.get("actor_name") or "unknown")
                labels = {"actor_type": actor_type, "model": model, "actor_name": actor_name}
                metadata = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                _append_metric(lines, "mint_model_actor_registry_actor_idle_time_s", rec.get("idle_time"), labels=labels)
                _append_metric(lines, "mint_model_actor_registry_actor_age_s", rec.get("age"), labels=labels)
                _append_metric(lines, "mint_model_actor_registry_actor_rss_bytes", rec.get("rss_bytes"), labels=labels)
                _append_metric(lines, "mint_model_actor_registry_actor_rss_sample_age_s", rec.get("rss_sample_age_s"), labels=labels)
                if actor_type.strip().lower() == "vllm":
                    vllm_labels = {"actor_name": actor_name, "base_model": model}
                    _append_metric(
                        lines,
                        "mint_vllm_scheduler_waiting_requests",
                        metadata.get("scheduler_waiting_requests"),
                        labels=vllm_labels,
                    )
                    _append_metric(
                        lines,
                        "mint_vllm_scheduler_running_requests",
                        metadata.get("scheduler_running_requests"),
                        labels=vllm_labels,
                    )
                    _append_metric(
                        lines,
                        "mint_vllm_scheduler_kv_cache_usage_ratio",
                        metadata.get("scheduler_kv_cache_usage_ratio"),
                        labels=vllm_labels,
                    )
                    _append_metric(
                        lines,
                        "mint_vllm_prefix_cache_queries_total",
                        metadata.get("prefix_cache_queries_total"),
                        labels=vllm_labels,
                    )
                    _append_metric(
                        lines,
                        "mint_vllm_prefix_cache_hits_total",
                        metadata.get("prefix_cache_hits_total"),
                        labels=vllm_labels,
                    )
                    _append_metric(
                        lines,
                        "mint_vllm_prefix_cache_hit_ratio",
                        metadata.get("prefix_cache_hit_ratio"),
                        labels=vllm_labels,
                    )
                    _append_metric(
                        lines,
                        "mint_vllm_preemptions_total",
                        metadata.get("preemptions_total"),
                        labels=vllm_labels,
                    )
                    for stem in (
                        "queue_time_s",
                        "prefill_time_s",
                        "decode_time_s",
                        "time_per_output_token_s",
                        "scheduled_tokens_iter",
                        "scheduled_new_requests_iter",
                        "scheduled_cached_requests_iter",
                        "prefill_requests_iter",
                        "decode_requests_iter",
                        "prompt_tokens_iter",
                        "generation_tokens_iter",
                        "time_to_first_token_s",
                        "inter_token_latency_s",
                        "executor_execute_model_s",
                        "worker_execute_model_s",
                        "seq_slot_wait_s",
                        "generate_lock_wait_s",
                        "engine_read_lock_wait_s",
                        "add_request_wait_s",
                        "add_request_exec_s",
                        "first_token_observed_s",
                    ):
                        _append_metric(lines, f"mint_vllm_{stem}_sum", metadata.get(f"{stem}_total"), labels=vllm_labels)
                        _append_metric(lines, f"mint_vllm_{stem}_count", metadata.get(f"{stem}_count"), labels=vllm_labels)
                        _append_metric(lines, f"mint_vllm_{stem}_max", metadata.get(f"{stem}_max"), labels=vllm_labels)
                        _append_metric(
                            lines,
                            f"mint_vllm_{stem}_p50_recent",
                            metadata.get(f"{stem}_p50_recent"),
                            labels=vllm_labels,
                        )
                        _append_metric(
                            lines,
                            f"mint_vllm_{stem}_p95_recent",
                            metadata.get(f"{stem}_p95_recent"),
                            labels=vllm_labels,
                        )
                elif actor_type.strip().lower() == "megatron":
                    megatron_labels = {"actor_name": actor_name, "base_model": model}
                    for stem in (
                        "active_sessions",
                        "session_unknown",
                        "session_step",
                        "learning_rate",
                        "gpu_memory_allocated_bytes",
                        "gpu_memory_reserved_bytes",
                        "gpu_memory_fragmentation_bytes",
                    ):
                        _append_metric(lines, f"mint_megatron_{stem}", metadata.get(stem), labels=megatron_labels)
                elif actor_type.strip().lower() == "dense" and bool(metadata.get("poisoned")):
                    last_fatal_op = str(metadata.get("last_fatal_op") or "unknown")
                    poisoned_labels = {
                        "actor_name": actor_name,
                        "base_model": model,
                        "last_fatal_op": last_fatal_op,
                    }
                    _append_metric(lines, "mint_dense_actor_poisoned", 1, labels=poisoned_labels)
                    poisoned_at = _prom_number(metadata.get("poisoned_at"))
                    if poisoned_at is not None:
                        _append_metric(
                            lines,
                            "mint_dense_actor_poisoned_age_s",
                            max(0.0, time.time() - poisoned_at),
                            labels=poisoned_labels,
                        )
                    key = (model, last_fatal_op)
                    dense_poisoned_grouped[key] = float(dense_poisoned_grouped.get(key, 0.0)) + 1.0
                for binding in _model_actor_registry_gpu_bindings(rec):
                    _append_metric(lines, "mint_model_actor_registry_actor_gpu_binding", 1, labels=binding)

                rss_state = str(rec.get("rss_cache_state") or "").strip().lower()
                if rss_state not in {"fresh", "stale", "unknown"}:
                    if _prom_number(rec.get("rss_bytes")) is not None:
                        rss_state = "fresh"
                    elif rec.get("rss_sample_age_s") is not None or rec.get("rss_sample_source") is not None:
                        rss_state = "stale"
                    else:
                        rss_state = "unknown"
                _append_metric(
                    lines,
                    "mint_model_actor_registry_actor_rss_cache_state",
                    1,
                    labels={**labels, "state": rss_state},
                )

                bucket = grouped.setdefault(
                    (actor_type, model),
                    {
                        "count": 0.0,
                        "rss_sum": 0.0,
                        "rss_count": 0.0,
                        "rss_fresh": 0.0,
                        "rss_stale": 0.0,
                        "rss_unknown": 0.0,
                        "max_idle": 0.0,
                        "max_age": 0.0,
                    },
                )
                bucket["count"] += 1.0
                bucket[f"rss_{rss_state}"] += 1.0
                idle = _prom_number(rec.get("idle_time"))
                if idle is not None and idle > bucket["max_idle"]:
                    bucket["max_idle"] = idle
                age = _prom_number(rec.get("age"))
                if age is not None and age > bucket["max_age"]:
                    bucket["max_age"] = age
                rss = _prom_number(rec.get("rss_bytes"))
                if rss is not None:
                    bucket["rss_sum"] += rss
                    bucket["rss_count"] += 1.0

            for (actor_type, model), agg in grouped.items():
                labels = {"actor_type": actor_type, "model": model}
                _append_metric(lines, "mint_model_actor_registry_actors", agg["count"], labels=labels)
                _append_metric(lines, "mint_model_actor_registry_group_oldest_idle_time_s", agg["max_idle"], labels=labels)
                _append_metric(lines, "mint_model_actor_registry_group_oldest_age_s", agg["max_age"], labels=labels)
                if agg.get("rss_count", 0.0) > 0.0:
                    _append_metric(lines, "mint_model_actor_registry_group_rss_bytes", agg["rss_sum"], labels=labels)
                for state in ("fresh", "stale", "unknown"):
                    key = f"rss_{state}"
                    if agg.get(key, 0.0) > 0.0:
                        _append_metric(
                            lines,
                            "mint_model_actor_registry_group_rss_cache_samples",
                            agg[key],
                            labels={**labels, "state": state},
                        )

            for (base_model, last_fatal_op), count in sorted(dense_poisoned_grouped.items()):
                _append_metric(
                    lines,
                    "mint_dense_poisoned_actors",
                    count,
                    labels={"base_model": base_model, "last_fatal_op": last_fatal_op},
                )

        model_actor_registry_metadata_cache = actors.get("model_actor_registry_metadata_cache")
        if isinstance(model_actor_registry_metadata_cache, list):
            for row in model_actor_registry_metadata_cache:
                if not isinstance(row, dict):
                    continue
                labels = {"actor_type": row.get("actor_type") or "unknown"}
                _append_metric(
                    lines,
                    "mint_model_actor_registry_observability_cache_hits_total",
                    row.get("cache_hits_total"),
                    labels=labels,
                )
                _append_metric(
                    lines,
                    "mint_model_actor_registry_observability_cache_stale_total",
                    row.get("cache_stale_total"),
                    labels=labels,
                )
                _append_metric(
                    lines,
                    "mint_model_actor_registry_observability_refresh_success_total",
                    row.get("refresh_success_total"),
                    labels=labels,
                )
                _append_metric(
                    lines,
                    "mint_model_actor_registry_observability_refresh_failures_total",
                    row.get("refresh_failures_total"),
                    labels=labels,
                )

        model_actor_registry_lifecycle = actors.get("model_actor_registry_lifecycle")
        if isinstance(model_actor_registry_lifecycle, list):
            for row in model_actor_registry_lifecycle:
                if not isinstance(row, dict):
                    continue
                key = (str(row.get("base_model") or "unknown"), str(row.get("event") or "unknown"))
                megatron_actor_lifecycle_counts[key] = float(megatron_actor_lifecycle_counts.get(key, 0.0)) + float(
                    row.get("count") or 0.0
                )

    proc = stats.get("process")
    if isinstance(proc, dict):
        _append_metric(lines, "mint_api_server_process_rss_bytes", proc.get("rss_bytes"))
        _append_metric(lines, "mint_api_server_process_pid", proc.get("pid"))
        _append_metric(lines, "mint_driver_process_rss_bytes", proc.get("rss_bytes"))

    driver_state = stats.get("driver_state")
    if isinstance(driver_state, dict):
        _append_metric(lines, "mint_driver_sdk_sessions_fallback", driver_state.get("sdk_sessions_fallback"))
        _append_metric(lines, "mint_driver_session_heartbeat_entries", driver_state.get("session_heartbeat_entries"))
        _append_metric(lines, "mint_driver_lora_load_locks", driver_state.get("lora_load_locks"))
        _append_metric(lines, "mint_driver_sampling_sessions_total", driver_state.get("sampling_sessions_total"))
        _append_metric(lines, "mint_dense_session_state_bytes", driver_state.get("dense_session_state_bytes"))
        _append_metric(lines, "mint_dense_session_state_dirs", driver_state.get("dense_session_state_dirs"))
        _append_metric(
            lines,
            "mint_dense_session_state_oldest_age_s",
            driver_state.get("dense_session_state_oldest_age_s"),
        )
        _append_metric(
            lines,
            "mint_driver_sampling_sessions_multi_lora",
            driver_state.get("sampling_sessions_multi_lora"),
        )
        _append_metric(
            lines,
            "mint_driver_sampling_sessions_base_model",
            driver_state.get("sampling_sessions_base_model"),
        )
        _append_metric(
            lines,
            "mint_driver_sampling_sessions_lora_loaded",
            driver_state.get("sampling_sessions_lora_loaded"),
        )
        _append_metric(
            lines,
            "mint_driver_sampling_sessions_inflight",
            driver_state.get("sampling_sessions_inflight"),
        )

    for (base_model, event), count in sorted(megatron_actor_lifecycle_counts.items()):
        _append_metric(
            lines,
            "mint_megatron_actor_lifecycle_events_total",
            count,
            labels={"base_model": base_model, "event": event},
        )

    ray_cluster = stats.get("ray_cluster")
    if isinstance(ray_cluster, dict):
        _append_metric(lines, "mint_ray_cluster_up", ray_cluster.get("up"))
        _append_metric(lines, "mint_ray_cluster_warning_count", ray_cluster.get("warning_count"))
        _append_metric(lines, "mint_ray_cluster_probe_error_count", ray_cluster.get("probe_error_count"))
        _append_metric(lines, "mint_ray_cluster_slow_probe_count", ray_cluster.get("slow_probe_count"))
        _append_metric(lines, "mint_ray_cluster_total_probe_latency_ms", ray_cluster.get("total_probe_latency_ms"))
        _append_metric(lines, "mint_ray_cluster_cache_age_s", ray_cluster.get("cache_age_s"))

        nodes = ray_cluster.get("nodes")
        if isinstance(nodes, dict):
            _append_metric(lines, "mint_ray_cluster_nodes", nodes.get("alive"), labels={"state": "alive"})
            _append_metric(lines, "mint_ray_cluster_nodes", nodes.get("dead"), labels={"state": "dead"})
            _append_metric(
                lines,
                "mint_ray_cluster_dead_nodes_missing_heartbeats",
                nodes.get("dead_missing_heartbeats"),
            )

        resources = ray_cluster.get("resources")
        if isinstance(resources, dict):
            for key in (
                "cpu_total",
                "cpu_available",
                "gpu_total",
                "gpu_available",
                "memory_total",
                "memory_available",
                "object_store_memory_total",
                "object_store_memory_available",
            ):
                _append_metric(lines, f"mint_ray_cluster_{key}", resources.get(key))

        placement_groups = ray_cluster.get("placement_groups")
        if isinstance(placement_groups, dict):
            for key in ("total", "created", "removed", "pending", "pending_gpu"):
                _append_metric(lines, f"mint_ray_cluster_placement_groups_{key}", placement_groups.get(key))

        named_actors = ray_cluster.get("named_actors")
        if isinstance(named_actors, dict):
            _append_metric(lines, "mint_ray_cluster_named_actors_total", named_actors.get("total"))
            _append_metric(
                lines,
                "mint_ray_cluster_named_actors_namespace",
                named_actors.get("namespace"),
            )

        _append_metric(lines, "mint_ray_cluster_last_success_unixtime", ray_cluster.get("last_success_unixtime"))
        _append_metric(lines, "mint_ray_cluster_last_success_age_s", ray_cluster.get("last_success_age_s"))

        probes = ray_cluster.get("probes")
        if isinstance(probes, dict):
            for probe_name, probe in probes.items():
                if not isinstance(probe, dict):
                    continue
                labels = {"probe": probe_name}
                _append_metric(lines, "mint_ray_cluster_probe_success", probe.get("ok"), labels=labels)
                _append_metric(lines, "mint_ray_cluster_probe_latency_ms", probe.get("latency_ms"), labels=labels)

    ray_gcs_metrics = stats.get("ray_gcs_metrics")
    if isinstance(ray_gcs_metrics, dict):
        _append_metric(lines, "mint_ray_gcs_metrics_bridge_up", ray_gcs_metrics.get("up"))
        _append_metric(
            lines,
            "mint_ray_gcs_metrics_bridge_scrape_error_count",
            ray_gcs_metrics.get("scrape_error_count"),
        )
        _append_metric(
            lines,
            "mint_ray_gcs_metrics_bridge_sample_count",
            ray_gcs_metrics.get("sample_count"),
        )
        _append_metric(
            lines,
            "mint_ray_gcs_metrics_bridge_scrape_latency_ms",
            ray_gcs_metrics.get("scrape_latency_ms"),
        )
        _append_metric(
            lines,
            "mint_ray_gcs_metrics_bridge_cache_age_s",
            ray_gcs_metrics.get("cache_age_s"),
        )
        _append_metric(
            lines,
            "mint_ray_gcs_metrics_bridge_last_success_unixtime",
            ray_gcs_metrics.get("last_success_unixtime"),
        )
        _append_metric(
            lines,
            "mint_ray_gcs_metrics_bridge_last_success_age_s",
            ray_gcs_metrics.get("last_success_age_s"),
        )

        derived = ray_gcs_metrics.get("derived")
        if isinstance(derived, dict):
            for key, value in derived.items():
                _append_metric(lines, f"mint_ray_gcs_{key}", value)

        samples = ray_gcs_metrics.get("samples")
        if isinstance(samples, list):
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                metric_name = sample.get("name")
                if not isinstance(metric_name, str) or not metric_name:
                    continue
                labels = sample.get("labels") if isinstance(sample.get("labels"), dict) else None
                _append_raw_prom_sample(lines, metric_name, sample.get("value"), labels=labels)

    if not lines:
        lines.append("mint_metrics_up 0")
    else:
        lines.append("mint_metrics_up 1")
    payload = "\n".join(lines) + "\n"
    return Response(content=payload, media_type="text/plain; version=0.0.4")


@router.post("/work_queue/noop")
async def work_queue_noop(http_request: Request) -> dict:
    from ..backend.task_state_store import task_state_futures
    from ..backend.model_actor_supervisor import domain_key_for_internal_control
    from ..backend.model_work_admission import enqueue_model_work

    route_start_s = time.perf_counter()
    request_id = uuid.uuid4().hex
    request_json = b"{}"
    try:
        await _enqueue_internal_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="internal.noop",
            enqueue_coro=enqueue_model_work(
                request_id=request_id,
                op="internal.noop",
                request_json=request_json,
                user_id=None,
                webhook_url=None,
                domain_key=domain_key_for_internal_control(),
                affinity_group="internal:no-op",
                ordering_key=None,
                token_cost=1,
                extra=merge_queue_priority_extra({"ts": float(time.time())}, request=http_request),
                queued_meta={
                    "op": "internal.noop",
                    "queue_state": "queued",
                    "stage": "queued",
                    "queued_at": time.time(),
                },
                task_state_futures_client=task_state_futures,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to enqueue internal.noop: {e}")

    return {"request_id": request_id}


@router.get("/work_queue/debug_state")
async def work_queue_debug_state() -> dict:
    from ..backend.model_work_scheduler import model_work_scheduler

    return await model_work_scheduler.stats(timeout_s=10.0)


@router.get("/debug/scheduler_decisions")
async def scheduler_decisions_debug(
    limit: int = Query(default=100, ge=1, le=5000),
    scheduler_domain: str | None = None,
    reason: str | None = None,
    since_seq: int | None = Query(default=None, ge=0),
) -> dict:
    from ..backend.model_work_scheduler import model_work_scheduler

    stats = await model_work_scheduler.stats(timeout_s=10.0)
    domain_filter = scheduler_domain.strip() if isinstance(scheduler_domain, str) else None
    if not domain_filter and reason is None and since_seq is None:
        return stats
    out = dict(stats)
    if domain_filter:
        out["replica_queues"] = {
            queue_id: rec
            for queue_id, rec in (stats.get("replica_queues") or {}).items()
            if isinstance(rec, dict) and rec.get("domain_key") == domain_filter
        }
        out["backlog_depth_by_domain"] = {
            key: value
            for key, value in (stats.get("backlog_depth_by_domain") or {}).items()
            if key == domain_filter
        }
        out["leases"] = [
            lease
            for lease in (stats.get("leases") or [])
            if isinstance(lease, dict)
            and (
                lease.get("domain_key") == domain_filter
                or (
                    isinstance(lease.get("item"), dict)
                    and lease["item"].get("domain_key") == domain_filter
                )
            )
        ]
    out["decision_log_removed"] = True
    return out


# =============================================================================
# Checkpoint API (per spec: .claude/skills/architecture-design/references/checkpoint-download-api.md)
# =============================================================================


class CheckpointInfo(BaseModel):
    """Checkpoint information per spec."""

    checkpoint_id: str  # Unique ID (format: {model_id}_{checkpoint_name})
    model_name: str  # Base model (e.g., "Qwen/Qwen2.5-7B-Instruct")
    created_at: str  # ISO 8601 timestamp
    type: str  # "training" or "inference"
    size_bytes: int  # Total directory size


class CheckpointsListResponse(BaseModel):
    """Response for listing checkpoints."""

    checkpoints: list[CheckpointInfo]


def _catalog_timestamp_text(row: dict[str, Any]) -> str:
    for key in ("checkpoint_created_at", "published_at", "updated_at"):
        value = row.get(key)
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, str) and value:
            return value
    return datetime.now(timezone.utc).isoformat()


def _catalog_checkpoint_path(row: dict[str, Any]) -> str | None:
    storage_root = row.get("storage_root")
    owner_id = row.get("owner_id")
    model_id = row.get("model_id")
    raw_checkpoint_id = row.get("raw_checkpoint_id")
    checkpoint_type = row.get("checkpoint_type")

    if not isinstance(storage_root, str) or not storage_root:
        return None
    if not isinstance(model_id, str) or not model_id:
        return None
    if not isinstance(raw_checkpoint_id, str) or not raw_checkpoint_id:
        return None
    if checkpoint_type not in ("training", "sampler"):
        return None

    owner = str(owner_id or "anonymous").strip() or "anonymous"

    def _valid_segment(value: str) -> bool:
        if not value or value in (".", ".."):
            return False
        if "/" in value or "\\" in value:
            return False
        return True

    if not _valid_segment(owner):
        return None
    if not _valid_segment(model_id):
        return None
    if not _valid_segment(raw_checkpoint_id):
        return None

    root_real = os.path.realpath(storage_root)
    candidate = os.path.join(storage_root, owner, model_id, raw_checkpoint_id, checkpoint_type)
    candidate_real = os.path.realpath(candidate)
    if not (candidate_real == root_real or candidate_real.startswith(root_real + os.sep)):
        return None
    return candidate_real


async def _scan_checkpoints_from_catalog(
    user_id: str | None,
    *,
    is_admin: bool = False,
) -> list[CheckpointInfo]:
    rows = await list_catalog_checkpoints(owner_id=user_id, is_admin=is_admin)
    checkpoints: list[CheckpointInfo] = []
    for row in rows:
        raw_ckpt_id = row.get("ckpt_id")
        # asyncpg returns PostgreSQL UUID columns as uuid.UUID, not str.
        if isinstance(raw_ckpt_id, uuid.UUID):
            ckpt_id = str(raw_ckpt_id)
        elif isinstance(raw_ckpt_id, str):
            ckpt_id = raw_ckpt_id.strip()
        else:
            continue
        if not ckpt_id:
            continue
        model_name = row.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            model_name = "unknown"
        ckpt_type = row.get("checkpoint_type")
        if ckpt_type not in ("training", "sampler"):
            ckpt_type = "training"
        size_bytes = row.get("size_bytes")
        try:
            size = int(size_bytes)
        except Exception:
            size = 0

        checkpoints.append(
            CheckpointInfo(
                checkpoint_id=ckpt_id,
                model_name=model_name,
                created_at=_catalog_timestamp_text(row),
                type=ckpt_type,
                size_bytes=max(0, size),
            )
        )

    checkpoints.sort(key=lambda x: x.created_at, reverse=True)
    return checkpoints


async def _resolve_catalog_checkpoint_entry(
    checkpoint_id: str,
    *,
    user_id: str | None,
    is_admin: bool,
) -> tuple[str, dict[str, Any]] | None:
    row = await get_catalog_checkpoint(checkpoint_id, owner_id=user_id, is_admin=is_admin)
    if row is None:
        return None

    ckpt_path = _catalog_checkpoint_path(row)
    if ckpt_path is None:
        return None

    metadata = {
        "checkpoint_id": row.get("raw_checkpoint_id") or checkpoint_id,
        "owner_id": row.get("owner_id"),
        "model_id": row.get("model_id"),
        "model_name": row.get("model_name"),
        "checkpoint_type": row.get("checkpoint_type"),
        "created_at": _catalog_timestamp_text(row),
    }
    return ckpt_path, metadata


@router.get("/v1/checkpoints", response_model=CheckpointsListResponse)
async def list_checkpoints(request: Request):
    """List all checkpoints for the authenticated user.

    Returns checkpoints owned by the user (via metadata.json owner_id).
    Admin users see all checkpoints.
    """
    user_id = _get_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    is_admin = can_bypass_ownership(request)

    if not checkpoint_index_enabled():
        raise HTTPException(status_code=503, detail="Checkpoint catalog unavailable")

    try:
        # Catalog-backed list avoids recursively sizing uncataloged
        # filesystem checkpoints, which is too slow for prod list calls.
        catalog_checkpoints = await _scan_checkpoints_from_catalog(
            user_id,
            is_admin=is_admin,
        )
    except Exception:
        logger.exception(
            "[internal.list_checkpoints] catalog query failed; filesystem fallback disabled"
        )
        raise HTTPException(status_code=503, detail="Checkpoint catalog unavailable")
    return CheckpointsListResponse(checkpoints=catalog_checkpoints)


@router.get("/v1/checkpoints/{checkpoint_id}/archive")
async def download_checkpoint(checkpoint_id: str, request: Request):
    """Download checkpoint as tar.gz archive.

    Streams the checkpoint directory as a gzipped tarball.
    Uses subprocess for true streaming without loading into memory.
    """
    user_id = _get_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    is_admin = can_bypass_ownership(request)

    if not checkpoint_index_enabled():
        raise HTTPException(status_code=503, detail="Checkpoint catalog unavailable")

    try:
        resolved = await _resolve_catalog_checkpoint_entry(
            checkpoint_id,
            user_id=user_id,
            is_admin=is_admin,
        )
    except Exception:
        logger.exception(
            "[internal.download_checkpoint] catalog lookup failed checkpoint_id=%s",
            checkpoint_id,
        )
        raise HTTPException(status_code=503, detail="Checkpoint catalog unavailable")

    if resolved is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    ckpt_path, metadata = resolved

    if not is_admin and metadata.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isdir(ckpt_path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    def stream_tar_gz():
        """Stream tar.gz via subprocess to avoid memory explosion."""
        parent_dir = os.path.dirname(ckpt_path)
        dir_name = os.path.basename(ckpt_path)
        proc = subprocess.Popen(
            ["tar", "czf", "-", dir_name],
            cwd=parent_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            while chunk := proc.stdout.read(65536):
                yield chunk
        finally:
            proc.stdout.close()
            proc.wait()

    archive_name = metadata.get("checkpoint_id") if isinstance(metadata.get("checkpoint_id"), str) else checkpoint_id
    filename = f"{archive_name}.tar.gz"
    return StreamingResponse(
        stream_tar_gz(),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
