"""Futures routes for async operation polling.

Endpoints:
- POST /retrieve_future: Get result of an async operation
"""

from fastapi import APIRouter, HTTPException, Request, Response

from ..backend.future_store import FutureStatus, future_store
from ..futures_utils import pending_future_http_response
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
        from ..gateway import decode_request_id, forward_json, upstream_for_alias

        decoded = decode_request_id(body.request_id)
        if decoded is None:
            raise HTTPException(status_code=404, detail=f"Unknown request_id: {body.request_id}")

        upstream_alias, upstream_request_id = decoded
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        upstream_resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/retrieve_future",
            incoming_headers=dict(http_request.headers),
            json_body={"request_id": upstream_request_id},
            timeout_s=30.0,
        )

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

        # If the gateway uses an upstream credential (static_api_key), the upstream may treat
        # the request as privileged. Preserve local error-masking semantics based on the caller.
        if (
            upstream_resp.status_code == 200
            and isinstance(payload, dict)
            and "error" in payload
            and not _is_privileged(http_request)
        ):
            payload = dict(payload)
            payload["error"] = GENERIC_ERROR_MESSAGE
        return payload

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
        pending = pending_future_http_response()
        response.status_code = pending.status_code
        response.headers.update(pending.headers)
        return pending.body
    elif status == FutureStatus.FAILED:
        error = future_store.get_error(body.request_id)
        # Only expose full error details to privileged users
        if _is_privileged(http_request):
            payload = {"error": error, "category": "system"}
        else:
            payload = {"error": GENERIC_ERROR_MESSAGE, "category": "system"}
        future_store.cleanup(body.request_id)
        return payload
    else:
        # DONE - return the result
        result = future_store.get_result(body.request_id)
        future_store.cleanup(body.request_id)
        return result
