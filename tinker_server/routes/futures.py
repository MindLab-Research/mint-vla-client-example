"""Futures routes for async operation polling.

Endpoints:
- POST /retrieve_future: Get result of an async operation
"""

from fastapi import APIRouter, HTTPException, Request, Response

from ..backend.future_store import FutureStatus, future_store
from ..models.types import FutureRetrieveRequest

router = APIRouter()

GENERIC_ERROR_MESSAGE = "Operation failed. Contact administrator if issue persists."


def _is_privileged(request: Request) -> bool:
    """Check if request is from privileged user (admin API key)."""
    from ..config import config as server_config
    if not server_config.auth_enabled:
        return True
    user_data = getattr(request.state, "user_data", None)
    return user_data is not None and user_data.get("user_id") == "admin"


@router.post("/retrieve_future")
async def retrieve_future(
    body: FutureRetrieveRequest, http_request: Request, response: Response
) -> dict:
    """Retrieve the result of an async operation.

    Tinker polling protocol:
        - HTTP 408: operation pending, client should retry
        - HTTP 200 with {"error": "..."}: operation failed
        - HTTP 200 with result: operation completed

    Error details are only exposed to privileged users (admin API key).
    Regular users receive a generic error message.
    """
    try:
        status = future_store.get_status(body.request_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown request_id: {body.request_id}",
        )

    if status == FutureStatus.PENDING:
        # Tinker client expects HTTP 408 for pending
        response.status_code = 408
        return {"queue_state": "active"}
    elif status == FutureStatus.FAILED:
        error = future_store.get_error(body.request_id)
        # Only expose full error details to privileged users
        if _is_privileged(http_request):
            return {"error": error, "category": "system"}
        else:
            return {"error": GENERIC_ERROR_MESSAGE, "category": "system"}
    else:
        # DONE - return the result
        result = future_store.get_result(body.request_id)
        return result
