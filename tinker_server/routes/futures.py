"""Futures routes for async operation polling.

Endpoints:
- POST /retrieve_future: Get result of an async operation
"""

from fastapi import APIRouter, HTTPException, Response

from ..backend.future_store import FutureStatus, future_store
from ..models.types import FutureRetrieveRequest

router = APIRouter()


@router.post("/retrieve_future")
async def retrieve_future(request: FutureRetrieveRequest, response: Response) -> dict:
    """Retrieve the result of an async operation.

    Tinker polling protocol:
        - HTTP 408: operation pending, client should retry
        - HTTP 200 with {"error": "..."}: operation failed
        - HTTP 200 with result: operation completed
    """
    try:
        status = future_store.get_status(request.request_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown request_id: {request.request_id}",
        )

    if status == FutureStatus.PENDING:
        # Tinker client expects HTTP 408 for pending
        response.status_code = 408
        return {"queue_state": "active"}
    elif status == FutureStatus.FAILED:
        error = future_store.get_error(request.request_id)
        # Return error in response body
        return {"error": error, "category": "system"}
    else:
        # DONE - return the result
        result = future_store.get_result(request.request_id)
        return result
