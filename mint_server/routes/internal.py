"""Internal API routes for usage tracking, health checks, and checkpoint management.

Checkpoint endpoints follow /internal/v1/checkpoints spec:
- GET /checkpoints: List all user's checkpoints
- GET /checkpoints/{checkpoint_id}/archive: Download checkpoint as tar.gz
"""

import structlog
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
from ..logging_context import get_otel_tracer
from ..queue_priority import merge_queue_priority_extra
from ..ray_cluster_health import get_ray_cluster_health_snapshot
from ..ray_gcs_metrics import get_ray_gcs_metrics_snapshot
from ..usage_store import get_usage_store
from mint_server.backend.ops.actor_admin import KillActorsRequest
from mint_server.backend.stores.task_state_store import task_futures

router = APIRouter()
logger = structlog.get_logger(__name__)


def _internal_prometheus_metrics_enabled() -> bool:
    return os.environ.get("MINT_INTERNAL_PROMETHEUS_METRICS_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
            "task_futures_ready",
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


@router.get("/usage_logs", response_model=UsageLogsResponse)
async def get_usage_logs(
    request: Request,
    since: Annotated[str | None, Query(description="ISO 8601 timestamp")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Query usage logs for the authenticated user.

    Logs are automatically filtered by the account id from platform-forwarded
    identity headers.

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


@router.get("/actors")
async def list_actors(
    request: Request,
    actor_type: str | None = Query(None, alias="type"),
    model_name: str | None = None,
    refresh_metadata: bool = Query(
        True,
        description="Refresh VLLM/Megatron observability metadata before returning actors.",
    ),
) -> dict:
    """List model actor inventory. Internal/admin endpoint."""
    from mint_server.backend.ops.actor_admin import ActorListRequest, list_actor_inventory, require_admin

    require_admin(request)
    return await list_actor_inventory(
        ActorListRequest(
            actor_type=actor_type,
            model_name=model_name,
            refresh_metadata=refresh_metadata,
        )
    )


@router.post("/actors/kill")
async def kill_actors(request: Request, body: "KillActorsRequest") -> dict:
    """Kill model actors by type or exact name. Internal/admin endpoint."""
    from mint_server.backend.ops.actor_admin import kill_actors as kill_actor_inventory
    from mint_server.backend.ops.actor_admin import require_admin

    require_admin(request)
    return await kill_actor_inventory(request, body)


def _self_rss_bytes() -> int:
    with open("/proc/self/statm", encoding="utf-8") as f:
        parts = f.read().strip().split()
    if len(parts) < 2:
        raise ValueError(f"unexpected /proc/self/statm format: {parts!r}")
    rss_pages = int(parts[1])
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    return rss_pages * page_size


def _model_actor_inventory_local_snapshot() -> list[dict]:
    from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

    pool = get_model_actor_supervisor()
    return pool.cached_snapshot()


@router.get("/admission_stats")
async def admission_stats(*, include_actor_rss: bool = True) -> dict:
    from mint_server.backend.stores.task_state_store import task_futures
    from mint_server.backend.actors.model_actor_supervisor import model_actor_supervisor
    from mint_server.backend.scheduling.model_work_scheduler import model_work_scheduler
    from mint_server.backend.ops.maintenance_cron_actor import maintenance_cron_actor
    from mint_server.backend.stores.session_heartbeat_store import session_heartbeat_store
    from ..routes import sampling as sampling_route
    from ..routes import service as service_route

    timeout_s = 10.0

    model_scheduler = None
    try:
        model_scheduler = await model_work_scheduler.stats(timeout_s=timeout_s, create_if_missing=False)
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
        if hasattr(task_futures, "async_stats"):
            fs = await task_futures.async_stats()
        elif hasattr(task_futures, "async_ensure_ready"):
            fs = await task_futures.async_ensure_ready(timeout_s=timeout_s, create_if_missing=False)
        elif not include_actor_rss and hasattr(task_futures, "metrics_snapshot"):
            fs = task_futures.metrics_snapshot()
        else:
            fs = task_futures.ping(timeout_s=timeout_s)
    except Exception as e:
        fs = {"error": f"{type(e).__name__}: {e}"}

    actors: dict = {}
    if include_actor_rss:
        try:
            await task_futures.async_ping(timeout_s=timeout_s)
            actors["task_futures"] = {"rss_bytes": int(await task_futures.async_rss_bytes(timeout_s=timeout_s))}
        except Exception as e:
            actors["task_futures"] = {"error": f"{type(e).__name__}: {e}"}

        try:
            from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

            pool = get_model_actor_supervisor()
            actors["model_actor_inventory"] = pool.rss_snapshot(timeout_s=timeout_s)
            actors["model_actor_inventory_metadata_cache"] = pool.metadata_cache_metrics_snapshot()
            actors["model_actor_inventory_lifecycle"] = pool.lifecycle_metrics_snapshot()
        except Exception as e:
            actors["model_actor_inventory"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        # Metrics scrapes must stay cheap. A single hung actor in rss_snapshot()
        # can otherwise block the API thread and stall unrelated routes.
        try:
            from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

            pool = get_model_actor_supervisor()
            actors["model_actor_inventory"] = pool.cached_snapshot()
            actors["model_actor_inventory_metadata_cache"] = pool.metadata_cache_metrics_snapshot()
            actors["model_actor_inventory_lifecycle"] = pool.lifecycle_metrics_snapshot()
        except Exception as e:
            actors["model_actor_inventory"] = {"error": f"{type(e).__name__}: {e}"}

    proc = {"pid": int(os.getpid())}
    try:
        proc["rss_bytes"] = int(_self_rss_bytes())
    except Exception as e:
        proc["rss_error"] = f"{type(e).__name__}: {e}"

    try:
        session_heartbeat_entries = int(await session_heartbeat_store.async_size(create_if_missing=False))
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
        from mint_server.backend.training.dense.dense_session_state import collect_dense_session_state_stats

        driver_state.update(collect_dense_session_state_stats())
    except Exception as e:
        driver_state["dense_session_state_error"] = f"{type(e).__name__}: {e}"

    ray_cluster = None
    ray_gcs_metrics = None
    if include_actor_rss:
        try:
            ray_cluster = get_ray_cluster_health_snapshot()
        except Exception as e:
            ray_cluster = {"error": f"{type(e).__name__}: {e}"}

        try:
            ray_gcs_metrics = get_ray_gcs_metrics_snapshot()
        except Exception as e:
            ray_gcs_metrics = {"error": f"{type(e).__name__}: {e}"}

    maintenance_cron = None
    try:
        maintenance_cron = await maintenance_cron_actor.async_health_snapshot(
            timeout_s=timeout_s,
            create_if_missing=False,
        )
    except Exception as e:
        maintenance_cron = {"error": f"{type(e).__name__}: {e}"}

    return {
        "model_work_scheduler": model_scheduler,
        "model_actor_supervisor": model_supervisor,
        "task_futures": fs,
        "actors": actors,
        "process": proc,
        "driver_state": driver_state,
        "ray_cluster": ray_cluster,
        "ray_gcs_metrics": ray_gcs_metrics,
        "maintenance_cron_actor": maintenance_cron,
    }


@router.get("/maintenance_cron_actor")
async def maintenance_cron_actor_health() -> dict:
    from mint_server.backend.ops.maintenance_cron_actor import maintenance_cron_actor

    return await maintenance_cron_actor.async_health_snapshot(timeout_s=10.0, create_if_missing=False)


@router.get("/model_work_scheduler")
async def model_work_scheduler_health() -> dict:
    from mint_server.backend.scheduling.model_work_scheduler import model_work_scheduler

    return await model_work_scheduler.stats(timeout_s=10.0, create_if_missing=False)


@router.get("/model_actor_supervisor")
async def model_actor_supervisor_health() -> dict:
    from mint_server.backend.actors.model_actor_supervisor import model_actor_supervisor

    return await model_actor_supervisor.async_snapshot(timeout_s=10.0)


@router.get("/ray_cluster_health")
async def ray_cluster_health() -> dict:
    return get_ray_cluster_health_snapshot()


@router.get("/ray_gcs_metrics")
async def ray_gcs_metrics() -> dict:
    return get_ray_gcs_metrics_snapshot()


@router.get("/metrics")
async def metrics() -> Response:
    if not _internal_prometheus_metrics_enabled():
        raise HTTPException(status_code=404, detail="metrics endpoint disabled")
    payload = (
        "# HELP mint_metrics_up Internal Prometheus debug endpoint status. "
        "Production MinT metrics are exported through OpenTelemetry push.\n"
        "# TYPE mint_metrics_up gauge\n"
        "mint_metrics_up 1\n"
    )
    return Response(content=payload, media_type="text/plain; version=0.0.4")


@router.post("/model_work_scheduler/noop")
async def model_work_scheduler_noop(http_request: Request) -> dict:
    from mint_server.backend.actors.model_actor_supervisor import domain_key_for_internal_runtime
    from mint_server.backend.scheduling.model_work_admission import enqueue_model_work

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
                domain_key=domain_key_for_internal_runtime(),
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
                future_service_client=task_futures,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to enqueue internal.noop: {e}")

    return {"request_id": request_id}


@router.get("/model_work_scheduler/debug_state")
async def model_work_scheduler_debug_state() -> dict:
    from mint_server.backend.scheduling.model_work_scheduler import model_work_scheduler

    return await model_work_scheduler.stats(timeout_s=10.0, create_if_missing=False)


@router.get("/debug/scheduler_decisions")
async def scheduler_decisions_debug(
    limit: int = Query(default=100, ge=1, le=5000),
    scheduler_domain: str | None = None,
    reason: str | None = None,
    since_seq: int | None = Query(default=None, ge=0),
) -> dict:
    from mint_server.backend.scheduling.model_work_scheduler import model_work_scheduler

    stats = await model_work_scheduler.stats(timeout_s=10.0, create_if_missing=False)
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
