"""Internal action inference helpers for MintX action routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..backend.future_store import future_store
from ..models.types import ActRequest

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
