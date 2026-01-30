"""Internal API routes for usage tracking, health checks, and checkpoint management.

Checkpoint endpoints follow /internal/v1/checkpoints spec:
- GET /checkpoints: List all user's checkpoints
- GET /checkpoints/{checkpoint_id}/archive: Download checkpoint as tar.gz
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
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


# =============================================================================
# Checkpoint API (per spec: docs/checkpoint-download-api.md)
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
