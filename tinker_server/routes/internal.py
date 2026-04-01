"""Internal API routes for usage tracking, health checks, and checkpoint management.

Checkpoint endpoints follow /internal/v1/checkpoints spec:
- GET /checkpoints: List all user's checkpoints
- GET /checkpoints/{checkpoint_id}/archive: Download checkpoint as tar.gz
"""

import json
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..auth_identity import is_admin_request
from ..checkpoints import _iter_metadata_paths, get_persistent_search_roots, get_resolution_roots
from ..config import config as server_config
from ..health_checks import deep_healthz_response
from ..logging_context import get_otel_tracer
from ..ray_cluster_health import get_ray_cluster_health_snapshot
from ..ray_gcs_metrics import get_ray_gcs_metrics_snapshot
from ..usage_store import get_usage_store

# Checkpoint directory (shared filesystem)
CHECKPOINTS_DIR = server_config.checkpoint_dir

router = APIRouter()


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
            "future_store_ready",
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
    if account_id is None and not is_admin_request(request):
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
    if not is_admin_request(request) and request_account_id is None:
        raise HTTPException(status_code=403, detail="Access denied")
    if not is_admin_request(request) and account_id != request_account_id:
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


@router.get("/admission_stats")
async def admission_stats() -> dict:
    from dataclasses import asdict

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.future_store import future_store
    from ..backend.owner_runtime_supervisor import owner_runtime_supervisor
    from ..backend.queue_execution_runtime import queue_execution_runtime
    from ..backend.queue_supervisor import queue_supervisor
    from ..backend.resource_pool import get_resource_pool
    from ..backend.session_heartbeat_store import session_heartbeat_store
    from ..routes import sampling as sampling_route
    from ..routes import service as service_route

    def _self_rss_bytes() -> int:
        with open("/proc/self/statm", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if len(parts) < 2:
            raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
        rss_pages = int(parts[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return rss_pages * page_size

    timeout_s = 10.0

    cap = None
    try:
        cap = asdict(await capacity_manager.async_snapshot(timeout_s=timeout_s))
    except Exception as e:
        cap = {"error": f"{type(e).__name__}: {e}"}

    q = None
    try:
        q = await api_work_queue.stats(timeout_s=timeout_s)
        if not isinstance(q, dict):
            q = {"error": f"api_work_queue.stats returned non-dict: {type(q)}"}
    except Exception as e:
        q = {"error": f"{type(e).__name__}: {e}"}

    fs = None
    try:
        fs = await future_store.async_ensure_ready(timeout_s=timeout_s)
    except Exception as e:
        fs = {"error": f"{type(e).__name__}: {e}"}

    actors: dict = {}
    try:
        actors["capacity_manager"] = {"rss_bytes": int(await capacity_manager.async_rss_bytes(timeout_s=timeout_s))}
    except Exception as e:
        actors["capacity_manager"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        actors["api_work_queue"] = {"rss_bytes": int(await api_work_queue.rss_bytes(timeout_s=timeout_s))}
    except Exception as e:
        actors["api_work_queue"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        actors["future_store"] = {"rss_bytes": int(await future_store.async_rss_bytes(timeout_s=timeout_s))}
    except Exception as e:
        actors["future_store"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        pool = get_resource_pool()
        actors["resource_pool"] = pool.rss_snapshot(timeout_s=timeout_s)
    except Exception as e:
        actors["resource_pool"] = {"error": f"{type(e).__name__}: {e}"}

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

    owner_runtime = None
    try:
        owner_runtime = await owner_runtime_supervisor.async_health_snapshot(timeout_s=timeout_s)
    except Exception as e:
        owner_runtime = {"error": f"{type(e).__name__}: {e}"}

    queue_runtime = None
    try:
        queue_runtime = await queue_supervisor.async_snapshot(timeout_s=timeout_s)
    except Exception as e:
        queue_runtime = {"error": f"{type(e).__name__}: {e}"}

    queue_execution = None
    try:
        queue_execution = await queue_execution_runtime.async_health_snapshot(timeout_s=timeout_s)
    except Exception as e:
        queue_execution = {"error": f"{type(e).__name__}: {e}"}

    return {
        "capacity": cap,
        "work_queue": q,
        "future_store": fs,
        "actors": actors,
        "process": proc,
        "driver_state": driver_state,
        "ray_cluster": ray_cluster,
        "ray_gcs_metrics": ray_gcs_metrics,
        "owner_runtime_supervisor": owner_runtime,
        "queue_supervisor": queue_runtime,
        "queue_execution_runtime": queue_execution,
    }


@router.get("/owner_runtime_supervisor")
async def owner_runtime_supervisor_health() -> dict:
    from ..backend.owner_runtime_supervisor import owner_runtime_supervisor

    return await owner_runtime_supervisor.async_health_snapshot(timeout_s=10.0)


@router.get("/queue_supervisor")
async def queue_supervisor_health() -> dict:
    from ..backend.queue_supervisor import queue_supervisor

    return await queue_supervisor.async_snapshot(timeout_s=10.0)


@router.get("/queue_execution_runtime")
async def queue_execution_runtime_health() -> dict:
    from ..backend.queue_execution_runtime import queue_execution_runtime

    return await queue_execution_runtime.async_health_snapshot(timeout_s=10.0)


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


@router.get("/metrics")
async def metrics() -> Response:
    stats = await admission_stats()
    lines: list[str] = []

    cap = stats.get("capacity")
    if isinstance(cap, dict):
        for key, value in cap.items():
            _append_metric(lines, f"mint_capacity_{key}", value)

    wq = stats.get("work_queue")
    if isinstance(wq, dict):
        # Existing queue counters.
        _append_metric(lines, "mint_work_queue_depth", wq.get("depth"))
        _append_metric(lines, "mint_work_queue_depth_legacy", wq.get("depth_legacy"))
        _append_metric(lines, "mint_work_queue_depth_scheduled", wq.get("depth_scheduled"))
        _append_metric(lines, "mint_work_queue_enqueued", wq.get("enqueued"))
        _append_metric(lines, "mint_work_queue_dequeued", wq.get("dequeued"))
        _append_metric(lines, "mint_work_queue_scheduler_enabled", wq.get("scheduler_enabled"))
        _append_metric(lines, "mint_work_queue_scheduler_picks_total", wq.get("scheduler_picks_total"))
        _append_metric(lines, "mint_work_queue_scheduler_switches_total", wq.get("scheduler_switches_total"))
        _append_metric(
            lines,
            "mint_work_queue_scheduler_starvation_picks_total",
            wq.get("scheduler_starvation_picks_total"),
        )
        _append_metric(lines, "mint_work_queue_scheduler_wait_s_sum", wq.get("scheduler_wait_s_sum"))
        _append_metric(lines, "mint_work_queue_scheduler_domains_total", wq.get("scheduler_domains_total"))

        # Phase 2: grouped depth by executor/op.
        by_executor = wq.get("by_executor")
        if isinstance(by_executor, dict):
            for executor, depth in by_executor.items():
                _append_metric(
                    lines,
                    "mint_work_queue_depth",
                    depth,
                    labels={"executor": executor},
                )

        # Phase 2: queued age stats.
        age_stats = wq.get("age_stats")
        if isinstance(age_stats, dict):
            _append_metric(lines, "mint_work_queue_oldest_queued_s", age_stats.get("oldest_queued_s"))
            _append_metric(lines, "mint_work_queue_avg_queued_s", age_stats.get("avg_queued_s"))

        execution_time_s_by_op = wq.get("execution_time_s_by_op")
        if isinstance(execution_time_s_by_op, dict):
            for op, rec in execution_time_s_by_op.items():
                if not isinstance(rec, dict):
                    continue
                labels = {"op": op}
                _append_metric(lines, "mint_work_queue_execution_last_s", rec.get("last"), labels=labels)
                _append_metric(lines, "mint_work_queue_execution_ema_s", rec.get("ema"), labels=labels)
                _append_metric(lines, "mint_work_queue_execution_sum_s", rec.get("sum"), labels=labels)
                _append_metric(lines, "mint_work_queue_execution_count", rec.get("count"), labels=labels)
                _append_metric(lines, "mint_work_queue_execution_max_s", rec.get("max"), labels=labels)

    fs = stats.get("future_store")
    if isinstance(fs, dict):
        # Existing FutureStore counters/settings from /internal/admission_stats.
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
            _append_metric(lines, f"mint_future_store_{key}", fs.get(key))

        # Phase 2: grouped counters by operation.
        by_op = fs.get("by_op")
        if isinstance(by_op, dict):
            for op, counters in by_op.items():
                if not isinstance(counters, dict):
                    continue
                _append_metric(lines, "mint_future_store_pending", counters.get("pending"), labels={"op": op})
                _append_metric(lines, "mint_future_store_results", counters.get("results"), labels={"op": op})
                _append_metric(lines, "mint_future_store_errors", counters.get("errors"), labels={"op": op})

        age_stats = fs.get("age_stats")
        if isinstance(age_stats, dict):
            _append_metric(lines, "mint_future_store_oldest_pending_s", age_stats.get("oldest_pending_s"))
            _append_metric(lines, "mint_future_store_oldest_done_s", age_stats.get("oldest_done_s"))
            _append_metric(lines, "mint_future_store_avg_pending_s", age_stats.get("avg_pending_s"))
            _append_metric(lines, "mint_future_store_avg_done_s", age_stats.get("avg_done_s"))

        payload_stats = fs.get("payload_stats")
        if isinstance(payload_stats, dict):
            _append_metric(lines, "mint_future_store_result_refs_count", payload_stats.get("result_refs_count"))
            _append_metric(lines, "mint_future_store_errors_count", payload_stats.get("errors_count"))
            _append_metric(lines, "mint_future_store_refs_count", payload_stats.get("refs_count"))

        timeout_counts = fs.get("timeout_counts")
        if isinstance(timeout_counts, dict):
            for kind in ("queue", "execution", "total"):
                _append_metric(
                    lines,
                    "mint_future_store_timeouts_total",
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
                            "mint_future_store_timeouts_total",
                            rec.get(kind),
                            labels={"op": op, "kind": kind},
                        )

    actors = stats.get("actors")
    if isinstance(actors, dict):
        for actor_key in ("capacity_manager", "api_work_queue", "future_store"):
            rec = actors.get(actor_key)
            if isinstance(rec, dict):
                _append_metric(
                    lines,
                    "mint_actor_rss_bytes",
                    rec.get("rss_bytes"),
                    labels={"actor": actor_key},
                )

        resource_pool = actors.get("resource_pool")
        if isinstance(resource_pool, list):
            grouped: dict[tuple[str, str], dict[str, float]] = {}
            for rec in resource_pool:
                if not isinstance(rec, dict):
                    continue
                actor_type = str(rec.get("actor_type") or "unknown")
                model = str(rec.get("base_model") or "unknown")
                actor_name = str(rec.get("actor_name") or "unknown")
                labels = {"actor_type": actor_type, "model": model, "actor_name": actor_name}
                _append_metric(lines, "mint_resource_pool_actor_idle_time_s", rec.get("idle_time"), labels=labels)
                _append_metric(lines, "mint_resource_pool_actor_age_s", rec.get("age"), labels=labels)
                _append_metric(lines, "mint_resource_pool_actor_rss_bytes", rec.get("rss_bytes"), labels=labels)

                bucket = grouped.setdefault((actor_type, model), {"count": 0.0, "rss_sum": 0.0, "max_idle": 0.0, "max_age": 0.0})
                bucket["count"] += 1.0
                idle = _prom_number(rec.get("idle_time"))
                if idle is not None and idle > bucket["max_idle"]:
                    bucket["max_idle"] = idle
                age = _prom_number(rec.get("age"))
                if age is not None and age > bucket["max_age"]:
                    bucket["max_age"] = age
                rss = _prom_number(rec.get("rss_bytes"))
                if rss is not None:
                    bucket["rss_sum"] += rss

            for (actor_type, model), agg in grouped.items():
                labels = {"actor_type": actor_type, "model": model}
                _append_metric(lines, "mint_resource_pool_actors", agg["count"], labels=labels)
                _append_metric(lines, "mint_resource_pool_group_rss_bytes", agg["rss_sum"], labels=labels)
                _append_metric(lines, "mint_resource_pool_group_oldest_idle_time_s", agg["max_idle"], labels=labels)
                _append_metric(lines, "mint_resource_pool_group_oldest_age_s", agg["max_age"], labels=labels)

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
async def work_queue_noop() -> dict:
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.future_store import future_store
    from ..backend.result_size_estimator import estimate_small_result_bytes

    route_start_s = time.perf_counter()
    request_id = uuid.uuid4().hex
    request_json = b"{}"
    reserve = await capacity_manager.async_try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(request_id, meta={"op": "internal.noop"})
        await _enqueue_internal_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="internal.noop",
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="internal.noop",
                request_json=request_json,
                user_id=None,
                webhook_url=None,
                extra={"ts": float(time.time())},
            ),
        )
    except Exception as e:
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue internal.noop: {e}")

    return {"request_id": request_id}


@router.get("/work_queue/debug_state")
async def work_queue_debug_state() -> dict:
    from ..backend.api_work_queue import api_work_queue

    return await api_work_queue.debug_state(timeout_s=10.0)


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


def _get_dir_size(path: str) -> int:
    """Calculate total size of directory."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _checkpoint_rank(storage_tier: str | None) -> int:
    if storage_tier == "persistent_tos":
        return 2
    if storage_tier == "persistent_cache":
        return 1
    return 0


def _public_checkpoint_id(metadata: dict, *, is_admin: bool) -> str | None:
    raw_checkpoint_id = metadata.get("checkpoint_id")
    if not isinstance(raw_checkpoint_id, str) or not raw_checkpoint_id:
        return None

    model_id = metadata.get("model_id")
    public_id = raw_checkpoint_id
    if isinstance(model_id, str) and model_id:
        public_id = f"{model_id}_{raw_checkpoint_id}"

    if is_admin:
        owner_id = metadata.get("owner_id")
        owner = str(owner_id or "anonymous").strip() or "anonymous"
        public_id = f"{owner}:{public_id}"

    return public_id


def _iter_checkpoint_entries(user_id: str | None, *, is_admin: bool = False):
    for metadata_path in _iter_metadata_paths(
        get_resolution_roots(primary_root=CHECKPOINTS_DIR),
        user_id=user_id,
        is_admin=is_admin,
    ):
        ckpt_path = os.path.dirname(metadata_path)
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        owner_id = metadata.get("owner_id")
        if not is_admin and owner_id != user_id:
            continue

        public_id = _public_checkpoint_id(metadata, is_admin=is_admin)
        if public_id is None:
            continue

        yield metadata, ckpt_path, public_id


def _looks_like_legacy_checkpoint_dir(path: str) -> bool:
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    for entry in entries:
        if os.path.isfile(os.path.join(path, entry)):
            return True
    return False



def _iter_legacy_checkpoint_entries(user_id: str | None, *, is_admin: bool = False):
    """Yield metadata-less legacy checkpoints from the old /owner/checkpoint layout."""
    seen: set[str] = set()
    roots = get_persistent_search_roots(primary_root=CHECKPOINTS_DIR)

    if is_admin:
        top_level_dirs: list[tuple[str, str]] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            top_level_dirs.extend(
                (root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
            )
    else:
        top_level_dirs = []
        if user_id is None:
            return
        for root in roots:
            user_dir = os.path.join(root, user_id)
            if os.path.isdir(user_dir):
                top_level_dirs.append((root, user_id))

    for root, owner_dir in top_level_dirs:
        top_path = os.path.join(root, owner_dir)
        for sub_dir in os.listdir(top_path):
            sub_path = os.path.join(top_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            if os.path.exists(os.path.join(sub_path, "metadata.json")):
                continue
            if not _looks_like_legacy_checkpoint_dir(sub_path):
                continue

            real = os.path.realpath(sub_path)
            if real in seen:
                continue
            seen.add(real)

            metadata = {
                "checkpoint_id": f"{owner_dir}_{sub_dir}",
                "owner_id": owner_dir,
                "type": "training",
            }
            yield metadata, sub_path, metadata["checkpoint_id"]


def _scan_checkpoints(user_id: str | None, *, is_admin: bool = False) -> list[CheckpointInfo]:
    """Scan metadata-backed checkpoints and return the newest visible entry per checkpoint_id."""
    checkpoints_by_id: dict[str, tuple[int, CheckpointInfo]] = {}

    for metadata, ckpt_path, public_id in list(_iter_checkpoint_entries(user_id, is_admin=is_admin)) + list(
        _iter_legacy_checkpoint_entries(user_id, is_admin=is_admin)
    ):
        model_name = metadata.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            adapter_config_path = os.path.join(ckpt_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path) as f:
                        adapter_config = json.load(f)
                    model_name = adapter_config.get("base_model_name_or_path") or "unknown"
                except (json.JSONDecodeError, OSError):
                    model_name = "unknown"
            else:
                model_name = "unknown"

        created_at = metadata.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            try:
                mtime = os.path.getmtime(ckpt_path)
                created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                created_at = datetime.now(timezone.utc).isoformat()

        ckpt_type = metadata.get("checkpoint_type") or metadata.get("type") or "training"
        size_bytes = _get_dir_size(ckpt_path)
        info = CheckpointInfo(
            checkpoint_id=public_id,
            model_name=model_name,
            created_at=created_at,
            type=ckpt_type,
            size_bytes=size_bytes,
        )
        rank = _checkpoint_rank(metadata.get("storage_tier"))
        current = checkpoints_by_id.get(public_id)
        if current is None or rank >= current[0]:
            checkpoints_by_id[public_id] = (rank, info)

    checkpoints = [entry[1] for entry in checkpoints_by_id.values()]
    checkpoints.sort(key=lambda x: x.created_at, reverse=True)
    return checkpoints


def _resolve_checkpoint_entry(
    checkpoint_id: str,
    *,
    user_id: str | None,
    is_admin: bool,
) -> tuple[str, dict] | None:
    def _collect(scope_user_id: str | None, scope_is_admin: bool, *, allow_raw: bool) -> list[tuple[int, str, str, dict]]:
        matches: list[tuple[int, str, str, dict]] = []
        for metadata, ckpt_path, public_id in _iter_checkpoint_entries(scope_user_id, is_admin=scope_is_admin):
            raw_checkpoint_id = metadata.get("checkpoint_id")
            user_public_id = _public_checkpoint_id(metadata, is_admin=False)
            if public_id == checkpoint_id or user_public_id == checkpoint_id:
                matches.append((_checkpoint_rank(metadata.get("storage_tier")), public_id, ckpt_path, metadata))
                continue
            if allow_raw and isinstance(raw_checkpoint_id, str) and raw_checkpoint_id == checkpoint_id:
                matches.append((_checkpoint_rank(metadata.get("storage_tier")), public_id, ckpt_path, metadata))
        return matches

    matches = _collect(user_id, is_admin, allow_raw=not is_admin)
    if not matches:
        for metadata, ckpt_path, public_id in _iter_legacy_checkpoint_entries(user_id, is_admin=is_admin):
            if public_id == checkpoint_id:
                matches.append((0, public_id, ckpt_path, metadata))

    if not matches and not is_admin:
        matches = _collect(None, True, allow_raw=False)
        if not matches:
            for metadata, ckpt_path, public_id in _iter_legacy_checkpoint_entries(None, is_admin=True):
                if public_id == checkpoint_id:
                    matches.append((0, public_id, ckpt_path, metadata))
    if not matches:
        return None

    public_ids = {match[1] for match in matches}
    if len(public_ids) > 1:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    _, _, ckpt_path, metadata = matches[0]
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

    checkpoints = _scan_checkpoints(user_id, is_admin=is_admin_request(request))
    return CheckpointsListResponse(checkpoints=checkpoints)


@router.get("/v1/checkpoints/{checkpoint_id}/archive")
async def download_checkpoint(checkpoint_id: str, request: Request):
    """Download checkpoint as tar.gz archive.

    Streams the checkpoint directory as a gzipped tarball.
    Uses subprocess for true streaming without loading into memory.
    """
    user_id = _get_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Resolve checkpoint path via metadata-backed checkpoint ids.
    resolved = _resolve_checkpoint_entry(
        checkpoint_id,
        user_id=user_id,
        is_admin=is_admin_request(request),
    )
    if resolved is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    ckpt_path, metadata = resolved

    if not is_admin_request(request) and metadata.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

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
