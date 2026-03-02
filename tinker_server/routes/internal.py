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

from ..usage_logger import get_usage_logger
from ..config import config as server_config

# Checkpoint directory (shared filesystem)
CHECKPOINTS_DIR = server_config.checkpoint_dir

router = APIRouter()


def _get_user_id(request: Request) -> str | None:
    """Extract user_id from request state (set by auth middleware)."""
    user_data = getattr(request.state, "user_data", None)
    if user_data:
        return user_data.get("user_id")
    return None


class UsageLogEntry(BaseModel):
    """Single usage log entry."""

    user_id: str
    operation_type: str
    model_name: str
    token_count: int
    session_id: str
    request_id: str
    timestamp: str


class UsageLogsResponse(BaseModel):
    """Response for usage logs query."""

    logs: list[UsageLogEntry]
    count: int
    has_more: bool
    next_offset: int | None


class UsageSummaryResponse(BaseModel):
    """Response for usage summary."""

    total_tokens: int
    operation_counts: dict[str, int]  # token totals by operation type


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
    # Get user_id from authenticated token
    user_id = _get_user_id(request)

    # Parse 'since' timestamp if provided
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            since_dt = None

    logger = get_usage_logger()
    logs, total_count, has_more = logger.query_logs(
        since=since_dt,
        user_id=user_id,  # Filter by authenticated user
        limit=limit,
        offset=offset,
    )

    return UsageLogsResponse(
        logs=[UsageLogEntry(**log) for log in logs],
        count=total_count,
        has_more=has_more,
        next_offset=offset + limit if has_more else None,
    )


@router.get("/usage_summary/{user_id}", response_model=UsageSummaryResponse)
async def get_usage_summary(user_id: str):
    """Get usage summary for a specific user.

    Args:
        user_id: The user ID to get summary for
    """
    logger = get_usage_logger()
    summary = logger.get_user_summary(user_id)

    return UsageSummaryResponse(
        total_tokens=summary["total_tokens"],
        operation_counts=summary["operation_counts"],
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for internal monitoring."""
    # Check if usage log directory is accessible
    logger = get_usage_logger()
    db_status = "ok" if logger.log_dir.exists() else "error"

    return HealthResponse(
        status="ok",
        database=db_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/admission_stats")
async def admission_stats() -> dict:
    from dataclasses import asdict

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.future_store import future_store
    from ..backend.resource_pool import get_resource_pool

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
        cap = asdict(capacity_manager.snapshot(timeout_s=timeout_s))
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
        fs = future_store.ensure_ready(timeout_s=timeout_s)
    except Exception as e:
        fs = {"error": f"{type(e).__name__}: {e}"}

    actors: dict = {}
    try:
        actors["capacity_manager"] = {"rss_bytes": int(capacity_manager.rss_bytes(timeout_s=timeout_s))}
    except Exception as e:
        actors["capacity_manager"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        actors["api_work_queue"] = {"rss_bytes": int(await api_work_queue.rss_bytes(timeout_s=timeout_s))}
    except Exception as e:
        actors["api_work_queue"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        actors["future_store"] = {"rss_bytes": int(future_store.rss_bytes(timeout_s=timeout_s))}
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

    return {"capacity": cap, "work_queue": q, "future_store": fs, "actors": actors, "process": proc}


def _prom_sanitize_name(v: str) -> str:
    out = []
    for ch in str(v):
        if ch.isalnum() or ch == "_":
            out.append(ch.lower())
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


@router.get("/metrics")
async def metrics() -> Response:
    stats = await admission_stats()
    lines: list[str] = []

    cap = stats.get("capacity")
    if isinstance(cap, dict):
        for key, value in cap.items():
            _append_metric(lines, f"tinker_capacity_{key}", value)

    wq = stats.get("work_queue")
    if isinstance(wq, dict):
        # Existing queue counters.
        _append_metric(lines, "tinker_work_queue_depth", wq.get("depth"))
        _append_metric(lines, "tinker_work_queue_enqueued", wq.get("enqueued"))
        _append_metric(lines, "tinker_work_queue_dequeued", wq.get("dequeued"))

        # Phase 2: grouped depth by executor/op.
        by_executor = wq.get("by_executor")
        if isinstance(by_executor, dict):
            for executor, depth in by_executor.items():
                _append_metric(
                    lines,
                    "tinker_work_queue_depth",
                    depth,
                    labels={"executor": executor},
                )

        # Phase 2: queued age stats.
        age_stats = wq.get("age_stats")
        if isinstance(age_stats, dict):
            _append_metric(lines, "tinker_work_queue_oldest_queued_s", age_stats.get("oldest_queued_s"))
            _append_metric(lines, "tinker_work_queue_avg_queued_s", age_stats.get("avg_queued_s"))

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
            _append_metric(lines, f"tinker_future_store_{key}", fs.get(key))

        # Phase 2: grouped counters by operation.
        by_op = fs.get("by_op")
        if isinstance(by_op, dict):
            for op, counters in by_op.items():
                if not isinstance(counters, dict):
                    continue
                _append_metric(lines, "tinker_future_store_pending", counters.get("pending"), labels={"op": op})
                _append_metric(lines, "tinker_future_store_results", counters.get("results"), labels={"op": op})
                _append_metric(lines, "tinker_future_store_errors", counters.get("errors"), labels={"op": op})

        age_stats = fs.get("age_stats")
        if isinstance(age_stats, dict):
            _append_metric(lines, "tinker_future_store_oldest_pending_s", age_stats.get("oldest_pending_s"))
            _append_metric(lines, "tinker_future_store_oldest_done_s", age_stats.get("oldest_done_s"))
            _append_metric(lines, "tinker_future_store_avg_pending_s", age_stats.get("avg_pending_s"))
            _append_metric(lines, "tinker_future_store_avg_done_s", age_stats.get("avg_done_s"))

        payload_stats = fs.get("payload_stats")
        if isinstance(payload_stats, dict):
            _append_metric(lines, "tinker_future_store_result_refs_count", payload_stats.get("result_refs_count"))
            _append_metric(lines, "tinker_future_store_errors_count", payload_stats.get("errors_count"))
            _append_metric(lines, "tinker_future_store_refs_count", payload_stats.get("refs_count"))

    actors = stats.get("actors")
    if isinstance(actors, dict):
        for actor_key in ("capacity_manager", "api_work_queue", "future_store"):
            rec = actors.get(actor_key)
            if isinstance(rec, dict):
                _append_metric(
                    lines,
                    "tinker_actor_rss_bytes",
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
                _append_metric(lines, "tinker_resource_pool_actor_idle_time_s", rec.get("idle_time"), labels=labels)
                _append_metric(lines, "tinker_resource_pool_actor_age_s", rec.get("age"), labels=labels)
                _append_metric(lines, "tinker_resource_pool_actor_rss_bytes", rec.get("rss_bytes"), labels=labels)

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
                _append_metric(lines, "tinker_resource_pool_actors", agg["count"], labels=labels)
                _append_metric(lines, "tinker_resource_pool_group_rss_bytes", agg["rss_sum"], labels=labels)
                _append_metric(lines, "tinker_resource_pool_group_oldest_idle_time_s", agg["max_idle"], labels=labels)
                _append_metric(lines, "tinker_resource_pool_group_oldest_age_s", agg["max_age"], labels=labels)

    proc = stats.get("process")
    if isinstance(proc, dict):
        _append_metric(lines, "tinker_api_server_process_rss_bytes", proc.get("rss_bytes"))
        _append_metric(lines, "tinker_api_server_process_pid", proc.get("pid"))

    if not lines:
        lines.append("tinker_metrics_up 0")
    else:
        lines.append("tinker_metrics_up 1")
    payload = "\n".join(lines) + "\n"
    return Response(content=payload, media_type="text/plain; version=0.0.4")


@router.post("/work_queue/noop")
async def work_queue_noop() -> dict:
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.future_store import future_store
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_id = uuid.uuid4().hex
    request_json = b"{}"
    reserve = capacity_manager.try_reserve(
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
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(request_id, meta={"op": "internal.noop"})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="internal.noop",
            request_json=request_json,
            user_id=None,
            webhook_url=None,
            extra={"ts": float(time.time())},
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
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


def _scan_checkpoints(user_id: str | None) -> list[CheckpointInfo]:
    """Scan checkpoint directories and return those owned by user.

    Storage schema: /checkpoints/{user_id}/{checkpoint_id}/

    Admin sees all checkpoints, regular users only see their own directory.
    """
    checkpoints = []

    if not os.path.exists(CHECKPOINTS_DIR):
        return checkpoints

    # Determine which directories to scan
    if user_id == "admin":
        # Admin sees all - scan all top-level directories
        top_level_dirs = [
            d for d in os.listdir(CHECKPOINTS_DIR)
            if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))
        ]
    else:
        # Regular user - only scan their own directory
        user_dir = os.path.join(CHECKPOINTS_DIR, user_id)
        if not os.path.exists(user_dir):
            return checkpoints
        top_level_dirs = [user_id]

    for top_level in top_level_dirs:
        top_path = os.path.join(CHECKPOINTS_DIR, top_level)

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
    if not os.path.exists(CHECKPOINTS_DIR):
        return None

    # Search all checkpoint directories
    for top_level in os.listdir(CHECKPOINTS_DIR):
        top_path = os.path.join(CHECKPOINTS_DIR, top_level)
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

    checkpoints = _scan_checkpoints(user_id)
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
    if user_id != "admin":
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
