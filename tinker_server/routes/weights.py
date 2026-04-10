"""Weight and state routes for saving/loading model weights and checkpoints.

Endpoints:
- POST /save_weights: Save model weights (currently LoRA-only; legacy)
- POST /save_state: Save checkpoint for training resume (LoRA-only, includes optimizer state)
- POST /load_state: Load checkpoint to resume training
- GET /training_runs/{model_id}/checkpoints: List checkpoints for model
- DELETE /training_runs/{model_id}/checkpoints/{checkpoint_id}: Delete checkpoint
- GET /training_runs/{model_id}/checkpoints/{checkpoint_id}/archive: Download as tar.gz
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
import json
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from starlette.background import BackgroundTask

from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..auth_identity import is_admin_request
from ..backend.future_store import future_store
from ..checkpoints import (
    CHECKPOINTS_DIR,
    MIRROR_STATUS_PENDING,
    begin_async_checkpoint_mirror,
    build_gateway_proxy_archive_path,
    build_persistent_cache_dir,
    checkpoint_has_optimizer_state,
    async_create_checkpoint_archive,
    ensure_checkpoint_path_allowed,
    get_persistent_cache_dir,
    get_persistent_checkpoints_dir,
    get_persistent_search_roots,
    materialize_persistent_checkpoint,
    resolve_checkpoint_path,
    safe_extract_checkpoint_archive,
    validate_checkpoint_dir,
    validate_sampler_checkpoint_for_sampling,
    write_checkpoint_metadata,
)
from ..models.types import (
    CheckpointInfo,
    CheckpointUploadResponse,
    CheckpointsListResponse,
    LoadStateRequest,
    SaveStateRequest,
    UntypedAPIFuture,
)
from ..logging_context import classify_failure_reason, get_otel_tracer, run_async_with_otel_span, set_request_id
from ..model_access_control import can_access_model, get_access_denied_error
from ..queue_priority import merge_queue_priority_extra
from ..webhook import EventType, send_task_event

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager
    from ..backend.training_session_manager import TrainingSessionManager
    from ..backend.verl_training import VerlTrainingEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Global references (set by app lifespan)
training_manager: TrainingSessionManager | None = None
training_engine: VerlTrainingEngine | None = None
inference_manager: SessionManager | None = None  # For multi-LoRA sampling registration


def _mark_training_inflight(model_id: str, delta: int) -> None:
    if training_manager is None:
        return
    mark = getattr(training_manager, "mark_inflight", None)
    if callable(mark):
        mark(model_id, delta)


async def _fail_future(request_id: str, error: str) -> None:
    async_fail = getattr(future_store, "async_fail", None)
    if callable(async_fail):
        await async_fail(request_id, error)
        return
    fail = getattr(future_store, "fail", None)
    if callable(fail):
        fail(request_id, error)
        return
    raise AttributeError("future_store has neither async_fail nor fail")


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state."""
    return _request_user_data(request)


def _get_user_id(request: Request) -> str | None:
    """Extract user_id from request state."""
    return _request_user_id(request)


def _get_webhook_url(request: Request) -> str | None:
    """Extract webhook_url from request state."""
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("webhook_url")
    return None

def _build_execution_serial_extra(*, model_id: str, extra: dict | None = None) -> dict:
    payload = {} if extra is None else dict(extra)
    payload["execution_serial_key"] = f"training_session:{model_id}"
    return payload


def _build_weights_future_meta(
    *,
    op: str,
    model_id: str,
    store_info: dict | None = None,
    seq_id: int | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "op": str(op),
        "model_id": str(model_id),
        "queue_state": "queued",
        "stage": "queued",
        "queued_at": time.time(),
    }
    if isinstance(store_info, dict):
        session_id = store_info.get("session_id")
        base_model = store_info.get("base_model")
        backend = store_info.get("backend")
        if session_id:
            meta["session_id"] = str(session_id)
        if base_model:
            meta["base_model"] = str(base_model)
        if backend:
            meta["backend"] = str(backend)
    if seq_id is not None:
        try:
            meta["seq_id"] = int(seq_id)
        except Exception:
            meta["seq_id"] = None
    return meta


async def _get_route_training_store_info(model_id: str) -> dict | None:
    from ..routes.training import _get_training_route_session_info

    info = await _get_training_route_session_info(model_id)
    if isinstance(info, dict):
        return info
    if training_manager is not None:
        return None

    try:
        from ..backend.training_session_store import async_get_training_session_info

        store_info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e
    return store_info if isinstance(store_info, dict) else None


async def _protect_training_session_enqueue_window(session_info: dict) -> None:
    from ..routes.training import _protect_training_session_enqueue_window as _training_protect

    await _training_protect(session_info)


async def _enqueue_weights_request_with_trace(
    *,
    route_start_s: float,
    request_id: str,
    op: str,
    enqueue_coro,
    model_id: str | None = None,
) -> None:
    tracer = get_otel_tracer()
    future_ready_elapsed_ms = (time.perf_counter() - route_start_s) * 1000.0
    if tracer is None:
        await enqueue_coro
        return

    with tracer.start_as_current_span(f"{op}.enqueue") as span:
        span.set_attribute("component", "routes.weights")
        span.set_attribute("op", str(op))
        span.set_attribute("request_id", str(request_id))
        if model_id:
            span.set_attribute("model_id", str(model_id))
        span.add_event(
            "future_store_ready",
            {
                "elapsed_ms": round(future_ready_elapsed_ms, 3),
                "route_elapsed_ms": round(future_ready_elapsed_ms, 3),
            },
        )
        enqueue_start_s = time.perf_counter()
        await enqueue_coro
        span.add_event(
            "enqueue_done",
            {
                "elapsed_ms": round((time.perf_counter() - enqueue_start_s) * 1000.0, 3),
                "route_elapsed_ms": round((time.perf_counter() - route_start_s) * 1000.0, 3),
            },
        )


def _resolve_mint_path(mint_uri: str, *, user_id: str | None, is_admin: bool = False) -> str:
    """Convert path identifier to filesystem path.

    Args:
        mint_uri: One of:
            - checkpoint_id (ckpt_xxx): Search in all checkpoint directories
            - tinker://{model_id}/{weights|sampler_weights}/{name}
            - mint://{model_id}/{weights|sampler_weights}/{name}
            - file:///path: Strip prefix
            - Absolute path: Return as-is

    Returns:
        Filesystem path.
    """
    try:
        resolved = resolve_checkpoint_path(mint_uri, user_id=user_id, is_admin=is_admin)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    ensure_checkpoint_path_allowed(resolved, user_id=user_id, is_admin=is_admin)
    return materialize_persistent_checkpoint(resolved)


def _checkpoint_rank(storage_tier: str | None) -> int:
    if storage_tier == "persistent_tos":
        return 2
    if storage_tier == "persistent_cache":
        return 1
    return 0


def _require_checkpoint_owner(*, request_user_id: str | None, owner_id: str | None, is_admin: bool = False) -> None:
    if is_admin:
        return
    if request_user_id is None:
        if owner_id is None:
            return
        raise HTTPException(status_code=403, detail="Access denied")
    if owner_id == request_user_id:
        return
    raise HTTPException(status_code=403, detail="Access denied")


def _persistent_owner_root(user_id: str | None) -> str:
    return os.path.join(get_persistent_checkpoints_dir(), user_id or "anonymous")


def _persistent_candidate_paths(
    *,
    model_id: str,
    checkpoint_name: str,
    user_id: str | None,
    is_admin: bool = False,
    checkpoint_type: str | None = None,
) -> list[str]:
    candidates: list[str] = []
    type_suffix = checkpoint_type if checkpoint_type in ("training", "sampler") else None

    def _extend(base_path: str) -> None:
        if type_suffix is not None:
            candidates.append(os.path.join(base_path, type_suffix))
        candidates.append(base_path)

    for root in get_persistent_search_roots(primary_root=CHECKPOINTS_DIR):
        owner_dir = user_id or "anonymous"
        _extend(os.path.join(root, owner_dir, model_id, checkpoint_name))
        _extend(os.path.join(root, model_id, checkpoint_name))
        _extend(os.path.join(root, owner_dir, checkpoint_name))
        if is_admin and os.path.isdir(root):
            try:
                for owner in os.listdir(root):
                    _extend(os.path.join(root, owner, model_id, checkpoint_name))
                    _extend(os.path.join(root, owner, checkpoint_name))
            except OSError:
                pass
    # Preserve order while dropping duplicates.
    out: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        out.append(path)
    return out


def _iter_checkpoint_views(root_path: str) -> list[tuple[str, str]]:
    views: list[tuple[str, str]] = []
    seen: set[str] = set()
    for checkpoint_type in ("training", "sampler"):
        typed_path = os.path.join(root_path, checkpoint_type)
        metadata_path = os.path.join(typed_path, "metadata.json")
        if not os.path.exists(metadata_path):
            continue
        real = os.path.realpath(typed_path)
        if real in seen:
            continue
        seen.add(real)
        views.append((checkpoint_type, typed_path))

    metadata_path = os.path.join(root_path, "metadata.json")
    if os.path.exists(metadata_path):
        real = os.path.realpath(root_path)
        if real not in seen:
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                checkpoint_type = metadata.get("checkpoint_type")
            except Exception:
                checkpoint_type = None
            if checkpoint_type in ("training", "sampler"):
                views.append((checkpoint_type, root_path))
    return views


def _build_sdk_archive_redirect_response(
    *,
    request: Request,
    user_id: str | None,
    model_id: str,
    checkpoint_id: str,
) -> RedirectResponse:
    from ..config import config
    from ..download_tokens import make_archive_download_token
    from starlette.datastructures import URL

    secret = (config.token_secret_key or config.api_key or "").strip()

    def _first_forwarded(value: str | None) -> str | None:
        if not value:
            return None
        return value.split(",")[0].strip() or None

    xf_proto = _first_forwarded(request.headers.get("x-forwarded-proto"))
    xf_host = _first_forwarded(request.headers.get("x-forwarded-host"))
    xf_port = _first_forwarded(request.headers.get("x-forwarded-port"))

    scheme = xf_proto or request.url.scheme
    host = xf_host or request.headers.get("host") or request.url.netloc
    if xf_port and host and ":" not in host:
        host = f"{host}:{xf_port}"

    base = URL(f"{scheme}://{host}")
    direct_url_obj = base.replace(path=request.url.path).include_query_params(direct="1")
    if secret:
        token, exp = make_archive_download_token(
            secret=secret,
            user_id=user_id,
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            ttl_s=15 * 60,
        )
        direct_url_obj = direct_url_obj.include_query_params(download_token=token)
        expires = datetime.fromtimestamp(exp, tz=timezone.utc)
    else:
        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
    expires_header = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
    return RedirectResponse(
        url=str(direct_url_obj),
        status_code=302,
        headers={"Expires": expires_header},
    )


async def _close_upstream_response(response, client) -> None:
    try:
        await response.aclose()
    finally:
        await client.aclose()


async def _forward_remote_checkpoint_route(*, model_id: str, request: Request):
    from ..gateway import async_remote_training_model_info, forward_json, upstream_for_alias

    remote = await async_remote_training_model_info(model_id)
    if remote is None:
        return None

    upstream_alias = str(remote.get("upstream_alias") or "")
    base_model = str(remote.get("base_model") or "")
    owner_id = remote.get("owner_id")
    upstream = upstream_for_alias(upstream_alias)
    if upstream is None:
        raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

    user_data = _get_user_data(request)
    if not can_access_model(base_model, user_data):
        raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))
    _require_checkpoint_owner(
        request_user_id=_get_user_id(request),
        owner_id=owner_id,
        is_admin=is_admin_request(request),
    )

    try:
        resp = await forward_json(
            upstream=upstream,
            method="GET",
            path=request.url.path,
            incoming_headers=dict(request.headers),
            json_body=None,
            timeout_s=30.0,
        )
    except Exception:
        logger.exception("Upstream checkpoint list failed: %s", upstream_alias)
        raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} checkpoint list failed")

    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            detail = resp.text
        else:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return CheckpointsListResponse.model_validate(resp.json())


async def _forward_remote_checkpoint_archive(*, model_id: str, checkpoint_id: str, request: Request, direct: bool):
    from ..gateway import async_remote_training_model_info, forward_request, upstream_for_alias

    remote = await async_remote_training_model_info(model_id)
    if remote is None:
        return None

    upstream_alias = str(remote.get("upstream_alias") or "")
    base_model = str(remote.get("base_model") or "")
    owner_id = remote.get("owner_id")
    upstream = upstream_for_alias(upstream_alias)
    if upstream is None:
        raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

    user_data = _get_user_data(request)
    if not can_access_model(base_model, user_data):
        raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))
    _require_checkpoint_owner(
        request_user_id=_get_user_id(request),
        owner_id=owner_id,
        is_admin=is_admin_request(request),
    )

    params = dict(request.query_params)
    if direct:
        params["direct"] = "1"

    try:
        client, resp = await forward_request(
            upstream=upstream,
            method="GET",
            path=request.url.path,
            incoming_headers=dict(request.headers),
            params=params,
            timeout_s=600.0,
            stream=True,
        )
    except Exception:
        logger.exception("Upstream checkpoint archive failed: %s", upstream_alias)
        raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} checkpoint archive failed")

    if resp.status_code >= 400:
        text = await resp.aread()
        await _close_upstream_response(resp, client)
        decoded = text.decode("utf-8", errors="replace")
        try:
            payload = json.loads(decoded)
        except Exception:
            detail = decoded
        else:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise HTTPException(status_code=resp.status_code, detail=detail)

    if resp.status_code in (301, 302, 303, 307, 308):
        headers = {}
        expires = resp.headers.get("expires")
        if expires:
            headers["Expires"] = expires
        await _close_upstream_response(resp, client)
        redirect = _build_sdk_archive_redirect_response(
            request=request,
            user_id=_get_user_id(request),
            model_id=model_id,
            checkpoint_id=checkpoint_id,
        )
        redirect.status_code = resp.status_code
        redirect.headers.update(headers)
        return redirect

    response_headers = {}
    for name in ("Content-Disposition", "Content-Length", "Expires"):
        value = resp.headers.get(name)
        if value:
            response_headers[name] = value

    return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
        headers=response_headers,
        background=BackgroundTask(_close_upstream_response, resp, client),
    )


# =============================================================================
# POST /save_weights - async (legacy)
# =============================================================================


@router.post("/save_weights", response_model=UntypedAPIFuture)
async def save_weights(
    request: SaveStateRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save training checkpoint (Tinker TrainingClient.save_state).

    Tinker SDK calls POST /api/v1/save_weights for TrainingClient.save_state(...).
    This must produce a training checkpoint (weights + optimizer state).
    """
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    store_info = await _get_route_training_store_info(request.model_id)
    if not isinstance(store_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path=http_request.url.path,
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=30.0,
                )
            except Exception:
                logger.exception("Upstream save_weights failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} save_weights failed")

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(status_code=502, detail="Upstream save_weights returned invalid request_id")
            return UntypedAPIFuture(
                request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
            )

    if not isinstance(store_info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    await _protect_training_session_enqueue_window(store_info)
    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)
    from ..client_compat import prefer_tinker_uri

    prefer_tinker = prefer_tinker_uri(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex

    reserve = await capacity_manager.async_try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    inflight_marked = False
    try:
        if training_manager is not None:
            _mark_training_inflight(request.model_id, +1)
            inflight_marked = True
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta=_build_weights_future_meta(
                op="weights.save_weights",
                model_id=request.model_id,
                store_info=store_info,
                seq_id=getattr(request, "seq_id", None),
            ),
        )
        await _enqueue_weights_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="weights.save_weights",
            model_id=request.model_id,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="weights.save_weights",
                request_json=request_json,
                user_id=user_id,
                webhook_url=webhook_url,
                extra=merge_queue_priority_extra(
                    _build_execution_serial_extra(
                        model_id=request.model_id,
                        extra={"prefer_tinker": bool(prefer_tinker)},
                    ),
                    request=http_request,
                ),
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue save_weights request: {e}")

    return UntypedAPIFuture(request_id=request_id)


# =============================================================================
# POST /save_state - async (training checkpoint: includes optimizer state)
# =============================================================================


@router.post("/save_state", response_model=UntypedAPIFuture)
async def save_state(
    request: SaveStateRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save model state to checkpoint.

    This endpoint produces a training checkpoint intended for resume, including optimizer state.
    """
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    store_info = await _get_route_training_store_info(request.model_id)
    if not isinstance(store_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/save_state",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=30.0,
                )
            except Exception:
                logger.exception("Upstream save_state failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} save_state failed")

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(status_code=502, detail="Upstream save_state returned invalid request_id")
            return UntypedAPIFuture(
                request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
            )

    if not isinstance(store_info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    await _protect_training_session_enqueue_window(store_info)
    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)
    from ..client_compat import prefer_tinker_uri

    prefer_tinker = prefer_tinker_uri(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = await capacity_manager.async_try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    inflight_marked = False
    try:
        if training_manager is not None:
            _mark_training_inflight(request.model_id, +1)
            inflight_marked = True
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta=_build_weights_future_meta(
                op="weights.save_state",
                model_id=request.model_id,
                store_info=store_info,
                seq_id=getattr(request, "seq_id", None),
            ),
        )
        await _enqueue_weights_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="weights.save_state",
            model_id=request.model_id,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="weights.save_state",
                request_json=request_json,
                user_id=user_id,
                webhook_url=webhook_url,
                extra=merge_queue_priority_extra(
                    _build_execution_serial_extra(
                        model_id=request.model_id,
                        extra={"prefer_tinker": bool(prefer_tinker)},
                    ),
                    request=http_request,
                ),
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue save_state request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_save_state(
    request_id: str,
    request: SaveStateRequest,
    user_id: str | None = None,
    webhook_url: str | None = None,
    prefer_tinker: bool = False,
) -> None:
    """Background task to save state.

    Storage schema: /checkpoints/{owner_id}/{model_id}/{checkpoint_name}/
    Also registers the model for sampling via multi-LoRA engine.
    """
    session = None
    inflight_marked = False

    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        checkpoint_name = request.path.strip() if request.path is not None else ""
        if checkpoint_name:
            if checkpoint_name in (".", "..") or "/" in checkpoint_name or "\\" in checkpoint_name:
                raise ValueError(f"Invalid checkpoint name: {request.path!r}")
        else:
            checkpoint_name = f"ckpt_{uuid.uuid4().hex[:12]}"

        save_path = build_persistent_cache_dir(
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type="training",
        )

        logger.info(f"[{session.model_id}] Saving state to: {save_path}")

        # Save training checkpoint on worker, returns path
        await run_async_with_otel_span(
            "weights.save_state.execute",
            lambda: training_engine.save_weights(session, save_path),
            component="routes.weights",
            op="weights.save_state",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(getattr(session, "backend", "unknown")),
                "checkpoint_type": "training",
                "checkpoint_name": str(checkpoint_name),
            },
        )

        # Save ownership metadata (for user-scoped checkpoint API)
        # Note: Directory is created by Ray Worker on GPU node, but shared filesystem
        # sync may not be complete yet. Create directory on API server to ensure it exists.
        os.makedirs(save_path, exist_ok=True)

        optimizer_present = bool(checkpoint_has_optimizer_state(save_path))
        if not optimizer_present:
            raise RuntimeError(
                f"save_state must produce optimizer artifacts, but none found under: {save_path}"
            )
        try:
            validate_checkpoint_dir(save_path, checkpoint_type="training")
        except ValueError as e:
            raise RuntimeError(
                f"save_state produced an invalid training checkpoint at {save_path}: {e}"
            ) from e

        metadata = {
            "checkpoint_id": checkpoint_name,
            "owner_id": user_id,
            "model_id": session.model_id,
            "model_name": session.base_model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "step": session.current_step,
            "checkpoint_type": "training",
            "optimizer_present": optimizer_present,
            "backend": getattr(session, "backend", "unknown"),
            "type": "training",
            "storage_tier": "persistent_cache",
            "ttl_seconds": request.ttl_seconds,
        }
        write_checkpoint_metadata(save_path, metadata)

        persistent_path = begin_async_checkpoint_mirror(
            save_path,
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
        )

        # Sampling engines load checkpoints on demand via checkpoint_uri/create_sampling_session.
        # Do not block save_state completion on vLLM engine creation or LoRA hot-load.
        sampling_registered = False

        from ..client_compat import checkpoint_uri

        mint_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=False,
            checkpoint_type="training",
        )
        tinker_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=True,
            checkpoint_type="training",
        )
        selected_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=prefer_tinker,
            checkpoint_type="training",
        )

        await future_store.async_resolve(request_id, {
            "checkpoint_id": checkpoint_name,
            "path": selected_path,
            "mint_path": mint_path,
            "tinker_path": tinker_path,
            "filesystem_path": save_path,
            "persistent_filesystem_path": persistent_path,
            "mirror_status": MIRROR_STATUS_PENDING,
            "storage_tier": "persistent_cache",
            "mirror_error": None,
            "type": "save_weights",
            "sampling_registered": sampling_registered,
            "checkpoint_type": "training",
        })

        # 发送 completed 状态 - 训练完成（权重已保存）
        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_COMPLETED,
                user_id=user_id,
                session_id=session.model_id,
                task_name=f"Training {session.base_model}",
                task_type="training",
                model_name=session.base_model,
                result={"checkpoint_id": checkpoint_name, "step": session.current_step},
            )

    except Exception as e:
        logger.exception(
            "[weights.save_state] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_checkpoint_path",
        )
        await _fail_future(request_id, str(e))

        # 发送 failed 状态
        if webhook_url and user_id:
            failed_session_id = session.model_id if session is not None else request.model_id
            failed_model_name = session.base_model if session is not None else None
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_FAILED,
                user_id=user_id,
                session_id=failed_session_id,
                task_name=f"Training {failed_model_name or request.model_id}",
                task_type="training",
                model_name=failed_model_name,
                error=str(e),
            )
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


async def _do_save_weights(
    request_id: str,
    request: SaveStateRequest,
    user_id: str | None = None,
    webhook_url: str | None = None,
    prefer_tinker: bool = False,
) -> None:
    """Background task to save sampler-only LoRA weights (legacy /save_weights).

    Storage schema: /checkpoints/{owner_id}/{model_id}/{checkpoint_name}/
    Also registers the model for sampling via multi-LoRA engine.
    """
    session = None
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        checkpoint_name = request.path.strip() if request.path is not None else ""
        if checkpoint_name:
            if checkpoint_name in (".", "..") or "/" in checkpoint_name or "\\\\" in checkpoint_name:
                raise ValueError(f"Invalid checkpoint name: {request.path!r}")
        else:
            checkpoint_name = f"ckpt_{uuid.uuid4().hex[:12]}"

        save_path = build_persistent_cache_dir(
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type="sampler",
        )

        logger.info(f"[{session.model_id}] Saving sampler weights to: {save_path}")

        await run_async_with_otel_span(
            "weights.save_weights.execute",
            lambda: training_engine.save_weights_for_sampler(
                session=session,
                checkpoint_name=checkpoint_name,
                checkpoint_base_dir=os.path.dirname(os.path.dirname(os.path.dirname(save_path))),
                checkpoint_type="sampler",
                use_per_expert_lora=False,
            ),
            component="routes.weights",
            op="weights.save_weights",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(getattr(session, "backend", "unknown")),
                "checkpoint_type": "sampler",
                "checkpoint_name": str(checkpoint_name),
            },
        )

        os.makedirs(save_path, exist_ok=True)

        if checkpoint_has_optimizer_state(save_path):
            raise RuntimeError(
                f"save_weights must not produce optimizer artifacts, but found some under: {save_path}"
            )

        metadata = {
            "checkpoint_id": checkpoint_name,
            "owner_id": user_id,
            "model_id": session.model_id,
            "model_name": session.base_model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "step": session.current_step,
            "checkpoint_type": "sampler",
            "optimizer_present": False,
            "backend": getattr(session, "backend", "unknown"),
            "type": "sampler",
            "storage_tier": "persistent_cache",
            "ttl_seconds": request.ttl_seconds,
        }
        write_checkpoint_metadata(save_path, metadata)

        persistent_path = begin_async_checkpoint_mirror(
            save_path,
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
        )

        # Keep save_weights completion scoped to checkpoint export + metadata publication.
        # Sampler clients can load the returned checkpoint path on demand.
        sampling_registered = False

        from ..client_compat import checkpoint_uri

        mint_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=False,
            checkpoint_type="sampler",
        )
        tinker_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=True,
            checkpoint_type="sampler",
        )
        selected_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=prefer_tinker,
            checkpoint_type="sampler",
        )

        await future_store.async_resolve(
            request_id,
            {
                "checkpoint_id": checkpoint_name,
                "path": selected_path,
                "mint_path": mint_path,
                "tinker_path": tinker_path,
                "filesystem_path": save_path,
                "persistent_filesystem_path": persistent_path,
                "mirror_status": MIRROR_STATUS_PENDING,
                "storage_tier": "persistent_cache",
                "mirror_error": None,
                "type": "save_weights",
                "sampling_registered": sampling_registered,
                "checkpoint_type": "sampler",
            },
        )

        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_COMPLETED,
                user_id=user_id,
                session_id=session.model_id,
                task_name=f"Training {session.base_model}",
                task_type="training",
                model_name=session.base_model,
                result={"checkpoint_id": checkpoint_name, "step": session.current_step},
            )

    except Exception as e:
        logger.exception(
            "[weights.save_weights] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_sampler_checkpoint_export",
        )
        await _fail_future(request_id, str(e))

        if webhook_url and user_id:
            failed_session_id = session.model_id if session is not None else request.model_id
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_FAILED,
                user_id=user_id,
                session_id=failed_session_id,
                task_name="Save weights",
                task_type="training",
                model_name=None,
                error=str(e),
            )
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


# =============================================================================
# POST /load_state - async
# =============================================================================


@router.post("/load_state", response_model=UntypedAPIFuture)
@router.post("/load_weights", response_model=UntypedAPIFuture)  # SDK alias
async def load_state(
    request: LoadStateRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Load model state from checkpoint."""
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_file,
        forward_json,
        upstream_for_alias,
    )

    store_info = await _get_route_training_store_info(request.model_id)
    remote = None if isinstance(store_info, dict) else await async_remote_training_model(request.model_id)
    if remote is not None:
        upstream_alias, base_model = remote
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        user_data = _get_user_data(http_request)
        if not can_access_model(base_model, user_data):
            raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

        user_id = _get_user_id(http_request)
        incoming_headers = dict(http_request.headers)
        json_body = request.model_dump()
        if request.path.startswith(("tinker://", "mint://", "ckpt_")):
            try:
                local_path = resolve_checkpoint_path(
                    request.path, user_id=user_id, is_admin=is_admin_request(http_request)
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            try:
                ensure_checkpoint_path_allowed(local_path, user_id=user_id, is_admin=is_admin_request(http_request))
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e)) from e
            if os.path.isdir(local_path):
                proxy_timeout_s = float(os.environ.get("MINT_GATEWAY_CHECKPOINT_PROXY_TIMEOUT_S", "600"))
                tmp_archive = build_gateway_proxy_archive_path()
                try:
                    await async_create_checkpoint_archive(local_path, tmp_archive, timeout_s=proxy_timeout_s)
                    upload_resp = await forward_file(
                        upstream=upstream,
                        path="/api/v1/checkpoints/upload",
                        incoming_headers=incoming_headers,
                        file_path=tmp_archive,
                        timeout_s=proxy_timeout_s,
                    )
                finally:
                    try:
                        os.unlink(tmp_archive)
                    except OSError:
                        pass
                if upload_resp.status_code >= 400:
                    raise HTTPException(status_code=upload_resp.status_code, detail=upload_resp.text)
                payload = upload_resp.json()
                ckpt_id = payload.get("checkpoint_id")
                if not isinstance(ckpt_id, str) or not ckpt_id:
                    raise HTTPException(
                        status_code=502,
                        detail="Upstream checkpoints/upload returned invalid checkpoint_id",
                    )
                json_body["path"] = ckpt_id

        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path=http_request.url.path,
                incoming_headers=incoming_headers,
                json_body=json_body,
                timeout_s=30.0,
            )
        except Exception:
            logger.exception("Upstream load_state failed: %s", upstream_alias)
            raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} load_state failed")

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream load_state returned invalid request_id")
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
        )

    if not isinstance(store_info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    await _protect_training_session_enqueue_window(store_info)
    user_id = _get_user_id(http_request)
    load_path = _resolve_mint_path(request.path, user_id=user_id, is_admin=is_admin_request(http_request))
    request = request.model_copy(update={"path": load_path})
    if request.optimizer:
        try:
            from ..checkpoints import validate_checkpoint_load_contract

            validate_checkpoint_load_contract(load_path, load_optimizer=True)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_checkpoint_for_optimizer_restore",
                    "error": str(e),
                    "path": request.path,
                },
            ) from e

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = await capacity_manager.async_try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    inflight_marked = False
    try:
        if training_manager is not None:
            _mark_training_inflight(request.model_id, +1)
            inflight_marked = True
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta=_build_weights_future_meta(
                op="weights.load_state",
                model_id=request.model_id,
                store_info=store_info,
                seq_id=getattr(request, "seq_id", None),
            ),
        )
        await _enqueue_weights_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="weights.load_state",
            model_id=request.model_id,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="weights.load_state",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=merge_queue_priority_extra(
                    _build_execution_serial_extra(model_id=request.model_id),
                    request=http_request,
                ),
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue load_state request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_load_state(
    request_id: str, request: LoadStateRequest, user_id: str | None
) -> None:
    """Background task to load state."""
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        load_path = request.path

        logger.info(f"[{session.model_id}] Loading state from: {load_path}")

        if request.optimizer:
            from ..checkpoints import validate_checkpoint_load_contract

            validate_checkpoint_load_contract(load_path, load_optimizer=True)

        # Call training engine to load checkpoint
        await run_async_with_otel_span(
            "weights.load_state.execute",
            lambda: training_engine.load_weights(session, load_path, load_optimizer=request.optimizer),
            component="routes.weights",
            op="weights.load_state",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(getattr(session, "backend", "unknown")),
                "load_optimizer": bool(request.optimizer),
            },
        )

        await future_store.async_resolve(request_id, {
            "path": request.path,
            "type": "load_weights",
        })

    except Exception as e:
        logger.exception(
            "[weights.load_state] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_checkpoint_contract_and_permissions",
        )
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


# =============================================================================
# POST /checkpoints/upload - upload tar.gz archive for resume
# =============================================================================


@router.post("/checkpoints/upload", response_model=CheckpointUploadResponse)
async def upload_checkpoint_archive(
    http_request: Request,
    file: UploadFile = File(...),
) -> CheckpointUploadResponse:
    """Upload a tar.gz checkpoint archive and register it for resume.

    Stores extracted checkpoint under /checkpoints/{owner}/{checkpoint_id}/ with metadata.json.
    Returns a checkpoint identifier usable by load_state/create_model_from_state.
    """
    import json
    import tempfile

    user_id = _get_user_id(http_request)
    owner_dir = user_id or "anonymous"

    checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
    parent_dir = os.path.join(get_persistent_search_roots(primary_root=CHECKPOINTS_DIR)[0], owner_dir)
    final_dir = os.path.join(parent_dir, checkpoint_id)
    tmp_dir = final_dir + ".tmp"
    tmp_archive: str | None = None

    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(final_dir):
        raise HTTPException(status_code=409, detail="Checkpoint already exists")

    try:
        os.makedirs(tmp_dir, exist_ok=False)

        fd, tmp_archive = tempfile.mkstemp(
            dir=parent_dir,
            prefix=f"upload_{checkpoint_id}_",
            suffix=".tar.gz",
        )
        os.close(fd)

        chunk_size = 8 * 1024 * 1024
        with open(tmp_archive, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)

        safe_extract_checkpoint_archive(tmp_archive, tmp_dir)

        # Determine checkpoint class (training vs sampler).
        # Rules:
        # - If metadata.json declares a checkpoint_type, it must match the artifacts.
        # - Otherwise, infer from presence of optimizer artifacts.
        from ..checkpoints import checkpoint_has_optimizer_state

        inferred_type = "training" if checkpoint_has_optimizer_state(tmp_dir) else "sampler"

        existing_meta: dict | None = None
        existing_meta_path = os.path.join(tmp_dir, "metadata.json")
        if os.path.exists(existing_meta_path):
            try:
                with open(existing_meta_path) as f:
                    existing_meta = json.load(f)
            except Exception:
                existing_meta = None

        declared_type = None
        if isinstance(existing_meta, dict):
            declared_type = existing_meta.get("checkpoint_type") or existing_meta.get("type")
        if declared_type is not None and declared_type not in ("training", "sampler"):
            raise HTTPException(status_code=400, detail=f"Invalid checkpoint_type in metadata.json: {declared_type!r}")

        checkpoint_type = declared_type or inferred_type
        if declared_type is not None and declared_type != inferred_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Checkpoint metadata declares checkpoint_type={declared_type!r}, "
                    f"but artifacts look like {inferred_type!r}"
                ),
            )

        validate_checkpoint_dir(tmp_dir, checkpoint_type=checkpoint_type)

        # Optional metadata extraction from checkpoint files
        model_id = None
        model_name = None
        step = None
        try:
            meta_path = os.path.join(tmp_dir, "training_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                step = meta.get("current_step", step)
        except Exception:
            pass
        try:
            adapter_cfg_path = os.path.join(tmp_dir, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                with open(adapter_cfg_path) as f:
                    cfg = json.load(f)
                model_name = cfg.get("base_model_name_or_path") or cfg.get("base_model") or model_name
        except Exception:
            pass
        if isinstance(existing_meta, dict):
            model_id = existing_meta.get("model_id") or model_id
            model_name = existing_meta.get("model_name") or model_name
            step = existing_meta.get("step") if step is None else step

        metadata = {
            "checkpoint_id": checkpoint_id,
            "owner_id": user_id,
            "model_id": model_id,
            "model_name": model_name,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "step": step,
            "checkpoint_type": checkpoint_type,
            "optimizer_present": checkpoint_type == "training",
            "backend": None,
            "type": checkpoint_type,
            "storage_tier": "persistent_tos",
        }
        write_checkpoint_metadata(tmp_dir, metadata)

        os.rename(tmp_dir, final_dir)
        return CheckpointUploadResponse(checkpoint_id=checkpoint_id, path=checkpoint_id)
    except ValueError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        if tmp_archive is not None:
            try:
                os.unlink(tmp_archive)
            except OSError:
                pass


# =============================================================================
# GET /training_runs/{model_id}/checkpoints
# =============================================================================


@router.get("/training_runs/{model_id}/checkpoints", response_model=CheckpointsListResponse)
async def list_checkpoints(model_id: str, request: Request) -> CheckpointsListResponse:
    """List all checkpoints for a model.

    Works for both active training sessions and saved checkpoints.
    Ownership verified via metadata.json (admin can access all).
    """
    remote_response = await _forward_remote_checkpoint_route(model_id=model_id, request=request)
    if remote_response is not None:
        return remote_response

    user_id = _get_user_id(request)
    owner_dir = user_id or "anonymous"
    from ..client_compat import checkpoint_uri, prefer_tinker_uri

    prefer_tinker = prefer_tinker_uri(request)

    persistent_roots = get_persistent_search_roots(primary_root=CHECKPOINTS_DIR)
    cache_root = get_persistent_cache_dir()

    candidate_paths: list[str] = []
    for root in [*persistent_roots, cache_root]:
        candidate_paths.extend(
            [
                os.path.join(root, owner_dir, model_id),
                os.path.join(root, model_id),
            ]
        )
        if is_admin_request(request) and os.path.isdir(root):
            try:
                for owner in os.listdir(root):
                    candidate_paths.append(os.path.join(root, owner, model_id))
            except OSError:
                pass

    if not any(os.path.exists(p) for p in candidate_paths):
        raise HTTPException(
            status_code=404, detail=f"No checkpoints found for model '{model_id}'"
        )

    checkpoints_by_id: dict[str, tuple[int, CheckpointInfo]] = {}

    def _store_checkpoint(info: CheckpointInfo) -> None:
        rank = _checkpoint_rank(info.storage_tier)
        current = checkpoints_by_id.get(info.checkpoint_id)
        if current is None or rank >= current[0]:
            checkpoints_by_id[info.checkpoint_id] = (rank, info)

    seen: set[str] = set()
    for checkpoints_path in candidate_paths:
        if not os.path.isdir(checkpoints_path):
            continue
        for name in os.listdir(checkpoints_path):
            ckpt_root = os.path.join(checkpoints_path, name)
            if not os.path.isdir(ckpt_root):
                continue
            for checkpoint_type, ckpt_path in _iter_checkpoint_views(ckpt_root):
                key = os.path.realpath(ckpt_path)
                if key in seen:
                    continue
                seen.add(key)

                metadata_path = os.path.join(ckpt_path, "metadata.json")
                if not os.path.exists(metadata_path):
                    continue
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                except Exception:
                    continue

                if metadata.get("model_id") != model_id:
                    continue
                if not is_admin_request(request) and metadata.get("owner_id") != user_id:
                    continue

                step = None
                if name.startswith("checkpoint-"):
                    try:
                        step = int(name.split("-")[1])
                    except (IndexError, ValueError):
                        pass

                created_at = datetime.fromtimestamp(os.path.getctime(ckpt_path)).isoformat()
                created_at = metadata.get("created_at") or created_at
                try:
                    created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except Exception:
                    created_time = datetime.fromtimestamp(os.path.getctime(ckpt_path))

                checkpoint_type = metadata.get("checkpoint_type")
                if checkpoint_type not in ("training", "sampler"):
                    continue
                try:
                    if checkpoint_type == "sampler":
                        validate_sampler_checkpoint_for_sampling(ckpt_path)
                    else:
                        validate_checkpoint_dir(ckpt_path, checkpoint_type=checkpoint_type)
                except ValueError:
                    continue
                storage_tier = metadata.get("storage_tier")
                mirror_status = metadata.get("mirror_status")
                mirror_error = metadata.get("mirror_error")
                checkpoint_id = (
                    f"weights/{name}" if checkpoint_type == "training" else f"sampler_weights/{name}"
                )
                tinker_path = checkpoint_uri(
                    model_id,
                    name,
                    prefer_tinker=True,
                    checkpoint_type=checkpoint_type,
                )
                path_uri = checkpoint_uri(
                    model_id,
                    name,
                    prefer_tinker=prefer_tinker,
                    checkpoint_type=checkpoint_type,
                )

                _store_checkpoint(
                    CheckpointInfo(
                        checkpoint_id=checkpoint_id,
                        checkpoint_type=checkpoint_type,
                        time=created_time,
                        tinker_path=tinker_path,
                        path=path_uri,
                        step=step,
                        created_at=created_at,
                        storage_tier=storage_tier,
                        mirror_status=mirror_status,
                        mirror_error=mirror_error,
                    )
                )

    # Also include uploaded checkpoints stored as /checkpoints/{owner}/{checkpoint_id}/ if metadata.model_id matches.
    owner_roots: list[str]
    if is_admin_request(request):
        try:
            owner_roots = []
            for root in [*persistent_roots, cache_root]:
                if not os.path.isdir(root):
                    continue
                owner_roots.extend(
                    os.path.join(root, d)
                    for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))
                )
        except OSError:
            owner_roots = []
    else:
        owner_roots = [os.path.join(root, owner_dir) for root in [*persistent_roots, cache_root]]

    for root in owner_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if not name.startswith("ckpt_"):
                continue
            ckpt_path = os.path.join(root, name)
            if not os.path.isdir(ckpt_path):
                continue
            metadata_path = os.path.join(ckpt_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                continue
            if metadata.get("model_id") != model_id:
                continue
            if not is_admin_request(request) and metadata.get("owner_id") != user_id:
                continue
            created_at = datetime.fromtimestamp(os.path.getctime(ckpt_path)).isoformat()
            checkpoint_type = metadata.get("checkpoint_type")
            if checkpoint_type not in ("training", "sampler"):
                continue
            try:
                if checkpoint_type == "sampler":
                    validate_sampler_checkpoint_for_sampling(ckpt_path)
                else:
                    validate_checkpoint_dir(ckpt_path, checkpoint_type=checkpoint_type)
            except ValueError:
                continue
            storage_tier = metadata.get("storage_tier")
            mirror_status = metadata.get("mirror_status")
            mirror_error = metadata.get("mirror_error")

            created_at = metadata.get("created_at") or created_at
            try:
                created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_time = datetime.fromtimestamp(os.path.getctime(ckpt_path))

            checkpoint_id = (
                f"weights/{name}" if checkpoint_type == "training" else f"sampler_weights/{name}"
            )
            tinker_path = checkpoint_uri(
                model_id,
                name,
                prefer_tinker=True,
                checkpoint_type=checkpoint_type,
            )
            path_uri = checkpoint_uri(
                model_id,
                name,
                prefer_tinker=prefer_tinker,
                checkpoint_type=checkpoint_type,
            )
            _store_checkpoint(
                CheckpointInfo(
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=checkpoint_type,
                    time=created_time,
                    tinker_path=tinker_path,
                    path=path_uri,
                    step=None,
                    created_at=created_at,
                    storage_tier=storage_tier,
                    mirror_status=mirror_status,
                    mirror_error=mirror_error,
                )
            )

    # Sort by step (descending)
    checkpoints = [item for _, item in checkpoints_by_id.values()]
    checkpoints.sort(key=lambda x: x.step or 0, reverse=True)

    return CheckpointsListResponse(
        model_id=model_id,
        checkpoints=checkpoints,
    )


# =============================================================================
# DELETE /training_runs/{model_id}/checkpoints/{checkpoint_id}
# =============================================================================


def _split_tinker_checkpoint_id(checkpoint_id: str) -> tuple[str, str | None]:
    # Tinker canonical checkpoint IDs include an explicit kind prefix:
    # - weights/<name> -> training
    # - sampler_weights/<name> -> sampler
    parts = checkpoint_id.split("/")
    if len(parts) == 2 and parts[0] in ("weights", "sampler_weights") and parts[1]:
        return parts[1], ("training" if parts[0] == "weights" else "sampler")
    return checkpoint_id, None


@router.delete("/training_runs/{model_id}/checkpoints/{checkpoint_id:path}")
async def delete_checkpoint(model_id: str, checkpoint_id: str, request: Request):
    """Delete a specific checkpoint.

    Ownership verified via metadata.json (admin can delete all).
    """
    user_id = _get_user_id(request)

    checkpoint_name, expected_type = _split_tinker_checkpoint_id(checkpoint_id)
    candidates = _persistent_candidate_paths(
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        user_id=user_id,
        is_admin=is_admin_request(request),
        checkpoint_type=expected_type,
    )

    ckpt_path = next((p for p in candidates if os.path.isdir(p)), None)
    if ckpt_path is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    metadata_path = os.path.join(ckpt_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        import json

        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception:
        raise HTTPException(status_code=403, detail="Access denied")

    if metadata.get("model_id") != model_id:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
    if not is_admin_request(request) and metadata.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if expected_type is not None and metadata.get("checkpoint_type") != expected_type:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    shutil.rmtree(ckpt_path)

    logger.info(f"[{model_id}] Deleted checkpoint: {checkpoint_id}")

    return {"status": "deleted", "checkpoint_id": checkpoint_id}


# =============================================================================
# GET /training_runs/{model_id}/checkpoints/{checkpoint_id}/archive
# =============================================================================


@router.get("/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/archive")
async def download_checkpoint_archive(
    model_id: str,
    checkpoint_id: str,
    request: Request,
    direct: bool = False,
):
    """Download checkpoint as tar.gz archive.

    Uses subprocess tar+gzip for true streaming without loading into memory.
    Essential for large checkpoints (7GB+).
    Ownership verified via metadata.json (admin can download all).
    """
    import subprocess

    remote_response = await _forward_remote_checkpoint_archive(
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        request=request,
        direct=direct,
    )
    if remote_response is not None:
        return remote_response
    from ..config import config
    from ..download_tokens import verify_download_token

    user_id = _get_user_id(request)
    if user_id is None:
        download_token = request.query_params.get("download_token")
        secret = (config.token_secret_key or config.api_key or "").strip()
        payload = verify_download_token(download_token or "", secret=secret)
        if (
            isinstance(payload, dict)
            and payload.get("model_id") == model_id
            and payload.get("checkpoint_id") == checkpoint_id
        ):
            user_id = payload.get("user_id")

    checkpoint_name, expected_type = _split_tinker_checkpoint_id(checkpoint_id)
    candidates = _persistent_candidate_paths(
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        user_id=user_id,
        is_admin=is_admin_request(request),
        checkpoint_type=expected_type,
    )

    # Prefer a candidate whose metadata matches the requested type.
    # This avoids false 404s when both "training" and "sampler" checkpoints share the same name.
    import json

    existing = [p for p in candidates if os.path.isdir(p)]
    if not existing:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    ckpt_path: str | None = None
    metadata: dict | None = None
    saw_unowned = False
    saw_unreadable_metadata = False
    for p in existing:
        metadata_path = os.path.join(p, "metadata.json")
        if not os.path.exists(metadata_path):
            saw_unreadable_metadata = True
            continue
        try:
            with open(metadata_path) as f:
                md = json.load(f)
        except Exception:
            saw_unreadable_metadata = True
            continue
        if md.get("model_id") != model_id:
            continue
        if expected_type is not None and md.get("checkpoint_type") != expected_type:
            continue
        if not is_admin_request(request) and md.get("owner_id") != user_id:
            saw_unowned = True
            continue
        ckpt_path = p
        metadata = md
        break

    if ckpt_path is None or metadata is None:
        if saw_unowned or saw_unreadable_metadata:
            raise HTTPException(status_code=403, detail="Access denied")
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    # Tinker SDK expects this endpoint to respond with 302 + Location.
    # It does not follow redirects automatically; it treats Location as a signed URL.
    if not direct:
        from ..client_compat import is_tinker_sdk_user_agent

        if is_tinker_sdk_user_agent(request.headers.get("user-agent")):
            return _build_sdk_archive_redirect_response(
                request=request,
                user_id=user_id,
                model_id=model_id,
                checkpoint_id=checkpoint_id,
            )

    def stream_tar_gz():
        """Stream tar.gz via subprocess to avoid memory explosion."""
        parent_dir = os.path.dirname(ckpt_path)
        dir_name = os.path.basename(ckpt_path)
        proc = subprocess.Popen(
            ["tar", "czf", "-", dir_name],
            cwd=parent_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            while chunk := proc.stdout.read(65536):
                yield chunk
        finally:
            proc.stdout.close()
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"tar exited with code {returncode}")

    safe_checkpoint_id = checkpoint_id.replace("/", "_")
    filename = f"{model_id}_{safe_checkpoint_id}.tar.gz"
    logger.info(f"[{model_id}] Streaming checkpoint archive: {checkpoint_id}")

    return StreamingResponse(
        stream_tar_gz(),
        media_type="application/gzip",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )
