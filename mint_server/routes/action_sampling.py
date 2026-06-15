"""Internal action inference helpers for MintX action routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from mint_server.backend.stores.task_state_store import billing_observations_from_input, task_futures
from ..models.types import ActRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# Execution-runtime action session manager reference (left unbound in API workers).
action_session_manager: object | None = None


def _current_action_session_manager() -> object | None:
    try:
        from mint_server.backend.core.execution_context import current_execution_context

        context = current_execution_context()
        if context is not None:
            return context.action_manager
    except Exception:
        pass
    return action_session_manager


async def _do_act(
    request_id: str,
    request: ActRequest,
    billing_observations: list[dict] | None = None,
    gateway_auth: dict | None = None,
    billing_observation_input: dict | None = None,
) -> None:
    try:
        manager = _current_action_session_manager()
        if manager is None:
            raise RuntimeError("Action session manager not initialized")

        out = await manager.act(  # type: ignore[attr-defined]
            action_session_id=request.action_session_id,
            observation=request.observation,
            extra_inputs=request.extra_inputs,
            temperature=request.temperature,
        )
        payload = dict(out)
        payload["type"] = "act"
        async_resolve = getattr(task_futures, "async_resolve", None)
        if callable(async_resolve):
            await async_resolve(
                request_id,
                payload,
                billing_observations=(
                    billing_observations
                    if billing_observations is not None
                    else billing_observations_from_input(
                        gateway_auth=gateway_auth,
                        request_id=request_id,
                        billing_input=billing_observation_input,
                    )
                ),
            )
        else:
            task_futures.resolve(request_id, payload)
    except Exception as e:
        try:
            async_fail = getattr(task_futures, "async_fail", None)
            if callable(async_fail):
                await async_fail(request_id, f"{type(e).__name__}: {e}")
            else:
                task_futures.fail(request_id, f"{type(e).__name__}: {e}")
        except Exception:
            logger.exception("[act] failed to mark future failed: request_id=%s", request_id)
        logger.exception("[act] background failed: request_id=%s", request_id)
