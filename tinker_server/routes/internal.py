"""Internal API routes for usage tracking and health checks."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from ..usage_logger import get_usage_logger

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
    operation_counts: dict[str, int]


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
