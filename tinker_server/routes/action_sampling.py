"""Action inference routes.

Endpoints:
- POST /act: Async action inference request (returns future)
- DELETE /action_sessions/{action_session_id}: Release an action session
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException

from ..backend.future_store import FutureStoreUnavailableError, future_store
from ..models.types import ActRequest, UntypedAPIFuture

logger = logging.getLogger(__name__)

router = APIRouter()

# Global action session manager reference (set by app lifespan).
action_session_manager: object | None = None


async def _do_act(request_id: str, request: ActRequest) -> None:
    try:
        if action_session_manager is None:
            raise RuntimeError("Action session manager not initialized")

        out = await action_session_manager.act(  # type: ignore[attr-defined]
            action_session_id=request.action_session_id,
            observation=request.observation,
            extra_inputs=request.extra_inputs,
        )
        payload = dict(out)
        payload["type"] = "act"
        future_store.resolve(request_id, payload)
    except Exception as e:
        try:
            future_store.fail(request_id, f"{type(e).__name__}: {e}")
        except Exception:
            logger.exception("[act] failed to mark future failed: request_id=%s", request_id)
        logger.exception("[act] background failed: request_id=%s", request_id)


@router.post("/act")
async def act(request: ActRequest) -> UntypedAPIFuture:
    # Action inference requires a state tensor input at minimum.
    if "state" not in request.extra_inputs:
        raise HTTPException(status_code=400, detail="extra_inputs.state is required")

    request_id = f"act_{uuid.uuid4().hex}"
    try:
        future_store.create_with_id(request_id)
        future_store.mark_queued(request_id, meta={"op": "act"})
    except FutureStoreUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    asyncio.create_task(_do_act(request_id, request))
    return UntypedAPIFuture(request_id=request_id)


@router.delete("/action_sessions/{action_session_id}")
async def delete_action_session(action_session_id: str) -> dict[str, str]:
    if action_session_manager is None:
        raise HTTPException(status_code=503, detail="Action session manager not initialized")

    await action_session_manager.shutdown_session(action_session_id)  # type: ignore[attr-defined]
    return {
        "action_session_id": action_session_id,
        "status": "deleted",
    }
