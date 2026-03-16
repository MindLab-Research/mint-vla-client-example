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

from ..auth_identity import is_admin_request
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


def _cached_response(status_code: int, headers: dict[str, str], body: Any) -> dict[str, Any]:
    return {
        "__cached_status_code__": int(status_code),
        "__cached_headers__": dict(headers),
        "__cached_body__": body,
    }


def _apply_cached_response(cached: Any, response: Response) -> Any:
    if not isinstance(cached, dict) or "__cached_body__" not in cached:
        return cached
    response.status_code = int(cached.get("__cached_status_code__", 200))
    headers = cached.get("__cached_headers__", {})
    if isinstance(headers, dict):
        response.headers.update(headers)
    return cached["__cached_body__"]


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
    return is_admin_request(request)


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
        # Check cache first for gateway-routed futures
        cached = _recent_get(body.request_id)
        if cached is not None:
            logger.info("[retrieve_future] request_id=%s gateway_cache_hit=true", body.request_id)
            return _apply_cached_response(cached, response)

        upstream_alias, upstream_request_id = decoded
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(
                status_code=500,
                detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
            )

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
        if isinstance(payload, dict) and "request_id" in payload:
            payload = dict(payload)
            payload["request_id"] = body.request_id
        if upstream_resp.status_code != 408:
            _recent_put(
                body.request_id,
                _cached_response(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=payload,
                ),
            )
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

        now = time.time()
        extra_body: dict[str, Any] = {}
        extra_headers: dict[str, str] = {}
        queue_state = None
        queue_state_reason = None
        stage = None
        queued_at = None
        running_at = None
        progress = None
        max_tokens = None
        last_progress_at = None
        op = None
        if isinstance(meta, dict):
            queue_state = meta.get("queue_state")
            queue_state_reason = meta.get("queue_state_reason")
            stage = meta.get("stage")
            queued_at = meta.get("queued_at")
            running_at = meta.get("running_at")
            progress = meta.get("progress")
            max_tokens = meta.get("max_tokens")
            last_progress_at = meta.get("last_progress_at")
            op = meta.get("op")
        if not isinstance(queue_state_reason, str) or not queue_state_reason.strip():
            queue_state_reason = None

        status_field = None
        is_sampling = isinstance(op, str) and op.startswith("sampling.")
        if is_sampling:
            if stage in ("prefill", "decode"):
                status_field = stage
            elif queue_state == "queued":
                status_field = "queued"
            elif queue_state == "running":
                status_field = "prefill"
        else:
            if queue_state in ("queued", "running"):
                status_field = str(queue_state)

        queue_position = None
        queue_depth = None
        estimated_wait_s = None
        from ..backend.api_work_queue import ApiWorkQueueUnavailableError, api_work_queue
        scheduler_enabled = str(os.environ.get("MINT_SCHEDULER_ENABLE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        try:
            pos = await api_work_queue.find_position(body.request_id)
        except ApiWorkQueueUnavailableError as e:
            raise HTTPException(status_code=503, detail=f"ApiWorkQueue unavailable: {e}") from e
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ApiWorkQueue position lookup failed: {type(e).__name__}: {e}") from e
        if not scheduler_enabled and isinstance(pos, dict):
            queue_depth = pos.get("depth")
            if status_field == "queued":
                queue_position = pos.get("position")
        if isinstance(queue_depth, (int, float)):
            queue_depth = int(queue_depth)
        else:
            queue_depth = None
        if isinstance(queue_position, (int, float)):
            queue_position = int(queue_position)
        else:
            queue_position = None
        if scheduler_enabled and status_field == "queued":
            queue_state_reason = "scheduler_enabled"
            queue_depth = None
            queue_position = None
            estimated_wait_s = None
        else:
            if queue_state_reason is None and status_field == "queued":
                if isinstance(queue_depth, int) and queue_depth > 0:
                    queue_state_reason = "queue_backlog"
                elif queue_position is None:
                    queue_state_reason = "queue_position_unknown"
            if status_field == "queued":
                try:
                    from ..config import config as server_config

                    eta_state = await api_work_queue.get_eta_state(op if isinstance(op, str) else None)
                    ema_exec_s = None
                    if isinstance(eta_state, dict):
                        ema_exec_s = eta_state.get("ema_exec_s")
                    worker_count = int(server_config.api_work_queue_num_workers)
                    if worker_count > 0 and isinstance(ema_exec_s, (int, float)) and queue_position is not None:
                        estimated_wait_s = (float(queue_position) + 1.0) * float(ema_exec_s) / float(worker_count)
                except ApiWorkQueueUnavailableError as e:
                    raise HTTPException(status_code=503, detail=f"ApiWorkQueue unavailable: {e}") from e
                except Exception as e:
                    raise HTTPException(status_code=503, detail=f"ApiWorkQueue ETA lookup failed: {type(e).__name__}: {e}") from e

        progress_payload = None
        if isinstance(progress, dict):
            tg = progress.get("tokens_generated")
            mx = progress.get("max_tokens")
            if isinstance(tg, (int, float)) and isinstance(mx, (int, float)):
                progress_payload = {
                    "tokens_generated": int(tg),
                    "max_tokens": int(mx),
                }
            if last_progress_at is None:
                lp = progress.get("last_progress_at")
                if isinstance(lp, (int, float)):
                    last_progress_at = lp
        if progress_payload is None and isinstance(max_tokens, (int, float)) and status_field == "decode":
            progress_payload = {"tokens_generated": 0, "max_tokens": int(max_tokens)}

        queued_for_s = None
        running_for_s = None
        last_progress_s = None
        if isinstance(queued_at, (int, float)):
            queued_for_s = max(0.0, now - float(queued_at))
        if isinstance(running_at, (int, float)):
            running_for_s = max(0.0, now - float(running_at))
        if isinstance(last_progress_at, (int, float)):
            last_progress_s = max(0.0, now - float(last_progress_at))

        def _compute_retry_after_s() -> int:
            if isinstance(estimated_wait_s, (int, float)) and float(estimated_wait_s) > 0:
                v = float(estimated_wait_s) / 4.0
                return int(max(1.0, min(30.0, v)))
            if isinstance(queue_depth, int):
                return int(max(1, min(10, queue_depth)))
            return 1

        retry_after_s = _compute_retry_after_s()

        extra_body.update(
            {
                "request_id": body.request_id,
                "type": "try_again",
                "status": status_field,
                "queue_state_reason": queue_state_reason,
                "queue_position": queue_position,
                "queue_depth": queue_depth,
                "estimated_wait_s": estimated_wait_s,
                "progress": progress_payload,
                "queued_for_s": queued_for_s,
                "running_for_s": running_for_s,
                "last_progress_s": last_progress_s,
            }
        )

        if status_field is not None:
            extra_headers["X-Queue-Status"] = str(status_field)
        if queue_position is not None:
            extra_headers["X-Queue-Position"] = str(int(queue_position))
        if queue_depth is not None:
            extra_headers["X-Queue-Depth"] = str(int(queue_depth))
        if estimated_wait_s is not None:
            extra_headers["X-Queue-ETA-S"] = f"{float(estimated_wait_s):.3f}"
        if isinstance(progress_payload, dict):
            extra_headers["X-Queue-Tokens-Generated"] = str(int(progress_payload.get("tokens_generated", 0)))
            extra_headers["X-Queue-Max-Tokens"] = str(int(progress_payload.get("max_tokens", 0)))

        # Tinker client expects HTTP 408 for pending
        _pending_hint_note_pending(body.request_id)
        pending = pending_future_http_response(
            retry_after_s=retry_after_s,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
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
            return _apply_cached_response(cached, response)
        result = future_store.get_result(body.request_id)
        if result is not None:
            _recent_put(body.request_id, result)
            logger.info("[retrieve_future] request_id=%s status=retrieved served=result", body.request_id)
            return result
        error = future_store.get_error(body.request_id)
        if error is not None:
            if _is_privileged(http_request):
                payload = {"error": error, "category": "system"}
            else:
                payload = {"error": _public_error(error), "category": "system"}
            _recent_put(body.request_id, payload)
            logger.info("[retrieve_future] request_id=%s status=retrieved served=error_payload", body.request_id)
            return payload
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
