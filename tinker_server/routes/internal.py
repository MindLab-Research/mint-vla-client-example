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
from ..checkpoints import get_persistent_search_roots
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

    driver_state: dict = {
        "sdk_sessions_fallback": int(len(service_route.sessions)),
        "session_heartbeat_entries": int(session_heartbeat_store.size()),
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

    return {
        "capacity": cap,
        "work_queue": q,
        "future_store": fs,
        "actors": actors,
        "process": proc,
        "driver_state": driver_state,
        "ray_cluster": ray_cluster,
        "ray_gcs_metrics": ray_gcs_metrics,
    }


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


def _scan_checkpoints(user_id: str | None, *, is_admin: bool = False) -> list[CheckpointInfo]:
    """Scan checkpoint directories and return those owned by user.

    Storage schema: /checkpoints/{user_id}/{checkpoint_id}/

    Admin sees all checkpoints, regular users only see their own directory.
    """
    checkpoints = []

    # Determine which directories to scan
    if is_admin:
        # Admin sees all - scan all top-level directories
        top_level_dirs: list[tuple[str, str]] = []
        for root in get_persistent_search_roots():
            if not os.path.isdir(root):
                continue
            top_level_dirs.extend(
                (root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
            )
    else:
        # Regular user - only scan their own directory
        top_level_dirs = []
        for root in get_persistent_search_roots():
            user_dir = os.path.join(root, user_id)
            if os.path.isdir(user_dir):
                top_level_dirs.append((root, user_id))

    for root, top_level in top_level_dirs:
        top_path = os.path.join(root, top_level)

        for sub_dir in os.listdir(top_path):
            sub_path = os.path.join(top_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue

            # Read metadata.json
            metadata_path = os.path.join(sub_path, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            # Get checkpoint_id from metadata or construct from path
            checkpoint_id = metadata.get("checkpoint_id", f"{top_level}_{sub_dir}")

            # Get model_name from metadata or infer from adapter_config.json
            model_name = metadata.get("model_name")
            if not model_name:
                adapter_config_path = os.path.join(sub_path, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    try:
                        with open(adapter_config_path) as f:
                            adapter_config = json.load(f)
                            model_name = adapter_config.get("base_model_name_or_path", "unknown")
                    except (json.JSONDecodeError, OSError):
                        model_name = "unknown"
                else:
                    model_name = "unknown"

            # Get created_at from metadata or file mtime
            created_at = metadata.get("created_at")
            if not created_at:
                try:
                    mtime = os.path.getmtime(sub_path)
                    created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                except OSError:
                    created_at = datetime.now(timezone.utc).isoformat()

            # Get type from metadata (default to "training")
            ckpt_type = metadata.get("type", "training")

            # Calculate size
            size_bytes = _get_dir_size(sub_path)

            checkpoints.append(CheckpointInfo(
                checkpoint_id=checkpoint_id,
                model_name=model_name,
                created_at=created_at,
                type=ckpt_type,
                size_bytes=size_bytes,
            ))

    # Sort by created_at descending
    checkpoints.sort(key=lambda x: x.created_at, reverse=True)
    return checkpoints


def _resolve_checkpoint_path(checkpoint_id: str) -> str | None:
    """Resolve checkpoint_id to filesystem path.

    Supports two schemas:
    1. New: checkpoint_id is stored in metadata, path is /checkpoints/{user_id}/{checkpoint_id}/
    2. Legacy: checkpoint_id format is {model_id}_{checkpoint_name}

    Scans all directories to find matching checkpoint_id.
    """
    # Search all checkpoint directories
    for root in get_persistent_search_roots():
        if not os.path.isdir(root):
            continue
        for top_level in os.listdir(root):
            top_path = os.path.join(root, top_level)
            if not os.path.isdir(top_path):
                continue

            for sub_dir in os.listdir(top_path):
                sub_path = os.path.join(top_path, sub_dir)
                if not os.path.isdir(sub_path):
                    continue

                # Check if this checkpoint matches
                metadata_path = os.path.join(sub_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        if metadata.get("checkpoint_id") == checkpoint_id:
                            return sub_path
                    except (json.JSONDecodeError, OSError):
                        pass

                # Check legacy format
                legacy_id = f"{top_level}_{sub_dir}"
                if legacy_id == checkpoint_id:
                    return sub_path

    return None


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

    # Resolve checkpoint path
    ckpt_path = _resolve_checkpoint_path(checkpoint_id)
    if ckpt_path is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Check ownership via metadata.json (admin can access all, others only their own)
    if not is_admin_request(request):
        metadata_path = os.path.join(ckpt_path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                owner_id = metadata.get("owner_id")
                if owner_id != user_id:
                    raise HTTPException(status_code=403, detail="Access denied")
            except (json.JSONDecodeError, OSError):
                # No valid metadata = no owner = deny access for non-admin
                raise HTTPException(status_code=403, detail="Access denied")
        else:
            # No metadata.json = legacy checkpoint = deny access for non-admin
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

    filename = f"{checkpoint_id}.tar.gz"
    return StreamingResponse(
        stream_tar_gz(),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
