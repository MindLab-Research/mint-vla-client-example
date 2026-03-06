"""Futures routes for async operation polling.

Endpoints:
- POST /retrieve_future: Get result of an async operation
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from ..backend.future_store import FutureStatus, FutureStoreUnavailableError, future_store
from ..futures_utils import pending_future_http_response
from ..models.types import FutureRetrieveRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def _retrieve_grace_s() -> float:
    v = (
        os.environ.get("MINT_RETRIEVE_FUTURE_GRACE_S")
        or os.environ.get("TINKER_RETRIEVE_FUTURE_GRACE_S")
        or "120"
    ).strip()
    try:
        return max(0.0, float(v))
    except Exception:
        return 120.0


def _retrieve_pending_min_poll_s() -> float:
    v = (
        os.environ.get("MINT_RETRIEVE_FUTURE_MIN_POLL_S")
        or os.environ.get("TINKER_RETRIEVE_FUTURE_MIN_POLL_S")
        or "1.0"
    ).strip()
    try:
        return max(0.0, float(v))
    except Exception:
        return 1.0


_RECENT_MAX = int(os.environ.get("MINT_RETRIEVE_FUTURE_RECENT_MAX", "2048"))
_RECENT: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
_PENDING_HINTS_MAX = int(os.environ.get("MINT_RETRIEVE_FUTURE_PENDING_MAX", "8192"))
_PENDING_HINTS: "OrderedDict[str, float]" = OrderedDict()


def _recent_put(request_id: str, payload: Any) -> None:
    now = time.time()
    _RECENT[request_id] = (now, payload)
    _RECENT.move_to_end(request_id)
    # Evict oldest first.
    while len(_RECENT) > _RECENT_MAX:
        _RECENT.popitem(last=False)


def _recent_get(request_id: str) -> Any | None:
    grace_s = _retrieve_grace_s()
    if grace_s <= 0:
        return None
    now = time.time()
    v = _RECENT.get(request_id)
    if v is None:
        return None
    ts, payload = v
    if (now - ts) > grace_s:
        _RECENT.pop(request_id, None)
        return None
    _RECENT.move_to_end(request_id)
    return payload


def _pending_hint_maybe_throttle(request_id: str) -> Any | None:
    now = time.time()
    next_probe_at = _PENDING_HINTS.get(request_id)
    if next_probe_at is None:
        return None
    if now >= next_probe_at:
        _PENDING_HINTS.move_to_end(request_id)
        return None
    retry_after_s = max(1, int(next_probe_at - now + 0.999))
    return pending_future_http_response(retry_after_s=retry_after_s, throttled=True)


def _pending_hint_note_pending(request_id: str) -> None:
    min_poll_s = _retrieve_pending_min_poll_s()
    if min_poll_s <= 0:
        _PENDING_HINTS.pop(request_id, None)
        return
    _PENDING_HINTS[request_id] = time.time() + min_poll_s
    _PENDING_HINTS.move_to_end(request_id)
    while len(_PENDING_HINTS) > _PENDING_HINTS_MAX:
        _PENDING_HINTS.popitem(last=False)


def _pending_hint_clear(request_id: str) -> None:
    _PENDING_HINTS.pop(request_id, None)


GENERIC_ERROR_MESSAGE = "Operation failed. Contact administrator if issue persists."

_SAFE_ERROR_PREFIXES = (
    "Access denied",
    "Checkpoint not found:",
    "Future expired",
    "Future already retrieved",
)


def _public_error(error: str | None) -> str:
    if not isinstance(error, str) or not error:
        return GENERIC_ERROR_MESSAGE
    for p in _SAFE_ERROR_PREFIXES:
        if error.startswith(p):
            return error
    return GENERIC_ERROR_MESSAGE


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

    Error details are only exposed to privileged users (admin API key), except for
    a small allowlist of safe, user-actionable errors (e.g. permission/ownership).
    Regular users receive a generic error message for other failures.
    """
    from ..gateway import decode_request_id, forward_json, upstream_for_alias

    throttled_pending = _pending_hint_maybe_throttle(body.request_id)
    if throttled_pending is not None:
        response.status_code = throttled_pending.status_code
        response.headers.update(throttled_pending.headers)
        return throttled_pending.body

    decoded = decode_request_id(body.request_id)
    if decoded is not None:
        upstream_alias, upstream_request_id = decoded
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        try:
            upstream_resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/retrieve_future",
                incoming_headers=dict(http_request.headers),
                json_body={"request_id": upstream_request_id},
                timeout_s=30.0,
            )
        except Exception:
            raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} retrieve_future failed")

        response.status_code = upstream_resp.status_code
        for k, v in upstream_resp.headers.items():
            lk = k.lower()
            if lk.startswith("x-") or lk == "retry-after":
                response.headers[k] = v
        try:
            payload = upstream_resp.json()
        except Exception:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream {upstream_alias!r} returned non-JSON retrieve_future payload",
            )

        if (
            upstream_resp.status_code == 404
            and isinstance(payload, dict)
            and isinstance(payload.get("detail"), str)
            and "Unknown request_id:" in payload["detail"]
        ):
            _pending_hint_clear(body.request_id)
            detail: object = GENERIC_ERROR_MESSAGE
            if _is_privileged(http_request):
                detail = {
                    "error": "Lost future (upstream Unknown request_id)",
                    "upstream_alias": upstream_alias,
                    "upstream_request_id": upstream_request_id,
                    "upstream_detail": payload.get("detail"),
                }
            raise HTTPException(status_code=503, detail=detail)

        # If this future corresponds to an ephemeral save_weights_for_sampler on an upstream,
        # register the returned sampling_session_id so subsequent /asample routes correctly.
        from ..gateway import maybe_register_sampling_session_from_retrieve_future

        maybe_register_sampling_session_from_retrieve_future(
            upstream_alias=upstream_alias,
            upstream_request_id=upstream_request_id,
            payload=payload,
        )

        if upstream_resp.status_code == 408:
            _pending_hint_note_pending(body.request_id)
        else:
            _pending_hint_clear(body.request_id)

        # If the gateway uses an upstream credential (static_api_key), the upstream may treat
        # the request as privileged. Preserve local error-masking semantics based on the caller.
        if (
            upstream_resp.status_code == 200
            and isinstance(payload, dict)
            and "error" in payload
            and not _is_privileged(http_request)
        ):
            payload = dict(payload)
            payload["error"] = _public_error(payload.get("error"))
        if (
            upstream_resp.status_code != 200
            and isinstance(payload, dict)
            and "detail" in payload
            and not _is_privileged(http_request)
        ):
            payload = dict(payload)
            payload["detail"] = GENERIC_ERROR_MESSAGE
        return payload

    try:
        status = future_store.get_status(body.request_id)
    except FutureStoreUnavailableError:
        raise HTTPException(status_code=503, detail="Ray unavailable: FutureStore requires Ray")
    except KeyError:
        _pending_hint_clear(body.request_id)
        logger.info("[retrieve_future] request_id=%s status=unknown", body.request_id)
        detail: object = f"Unknown request_id: {body.request_id}"
        if _is_privileged(http_request):
            detail = {
                "error": detail,
                "request_id": body.request_id,
                "future_store": future_store.debug_snapshot(),
            }
        raise HTTPException(status_code=404, detail=detail)

    if status == FutureStatus.PENDING:
        meta = None
        try:
            meta = future_store.get_meta(body.request_id)
        except Exception:
            meta = None
        if isinstance(meta, dict):
            actor_name = meta.get("actor_name")
            session_id = meta.get("model_id")
            if actor_name:
                try:
                    from ..backend.resource_pool import get_resource_pool

                    rp = get_resource_pool()
                    rp.touch(actor_name)
                    if session_id:
                        rp.set_session(actor_name, session_id)
                except Exception:
                    pass

        # Tinker client expects HTTP 408 for pending
        _pending_hint_note_pending(body.request_id)
        pending = pending_future_http_response()
        response.status_code = pending.status_code
        response.headers.update(pending.headers)
        return pending.body
    elif status == FutureStatus.EXPIRED:
        _pending_hint_clear(body.request_id)
        logger.info("[retrieve_future] request_id=%s status=expired", body.request_id)
        return {"error": "Future expired", "category": "system"}
    elif status == FutureStatus.RETRIEVED:
        _pending_hint_clear(body.request_id)
        cached = _recent_get(body.request_id)
        if cached is not None:
            logger.info("[retrieve_future] request_id=%s status=retrieved served=cached", body.request_id)
            return cached
        logger.info("[retrieve_future] request_id=%s status=retrieved served=error", body.request_id)
        return {"error": "Future already retrieved", "category": "system"}
    elif status == FutureStatus.FAILED:
        _pending_hint_clear(body.request_id)
        error = future_store.get_error(body.request_id)
        # Only expose full error details to privileged users
        if _is_privileged(http_request):
            payload = {"error": error, "category": "system"}
        else:
            payload = {"error": _public_error(error), "category": "system"}
        _recent_put(body.request_id, payload)
        logger.info("[retrieve_future] request_id=%s status=failed", body.request_id)
        try:
            from ..backend.capacity_manager import capacity_manager

            import ray

            if ray.is_initialized():
                capacity_manager.release_all(body.request_id)
        except Exception:
            pass
        try:
            future_store.cleanup(body.request_id)
        except Exception:
            pass
        return payload
    else:
        _pending_hint_clear(body.request_id)
        # DONE - return the result
        result = future_store.get_result(body.request_id)
        _recent_put(body.request_id, result)
        logger.info("[retrieve_future] request_id=%s status=done", body.request_id)
        try:
            from ..backend.capacity_manager import capacity_manager

            import ray

            if ray.is_initialized():
                capacity_manager.release_all(body.request_id)
        except Exception:
            pass
        try:
            future_store.cleanup(body.request_id)
        except Exception:
            pass
        return result
