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
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from starlette.background import BackgroundTask

from ..auth_identity import can_bypass_ownership
from ..auth_identity import can_manage_system
from ..auth_identity import can_write
from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..backend.future_store import future_store
from ..checkpoint_index import (
    CheckpointAlreadyExistsError,
    CheckpointAlreadyFailedError,
    CheckpointAlreadyUploadingError,
    claim_checkpoint_publication,
    mark_checkpoint_failed,
)
from ..checkpoints import (
    CHECKPOINTS_DIR,
    MIRROR_STATUS_PENDING,
    begin_async_checkpoint_mirror,
    build_gateway_proxy_archive_path,
    build_persistent_cache_dir,
    checkpoint_has_openpi_training_state,
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
    WeightsInfoRequest,
    WeightsInfoResponse,
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


def _loaded_training_session_lora_payload(lora_config: Any) -> dict[str, Any] | None:
    if lora_config is None:
        return None
    model_dump = getattr(lora_config, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())
    if isinstance(lora_config, dict):
        return dict(lora_config)
    if hasattr(lora_config, "__dict__"):
        return dict(lora_config.__dict__)
    raise TypeError(f"Unsupported lora_config type: {type(lora_config).__name__}")


def _next_loaded_training_session_metadata_version(session: Any) -> int:
    from ..backend.training_session_manager import TRAINING_SESSION_METADATA_VERSION

    current = max(
        int(getattr(session, "metadata_version")),
        TRAINING_SESSION_METADATA_VERSION - 1,
    )
    next_version = current + 1
    session.metadata_version = next_version
    return next_version


async def _persist_loaded_training_session(session: Any, *, request_user_id: str | None) -> None:
    from ..backend.training_session_manager import MATERIALIZATION_STATE_READY
    from ..backend.training_session_store import async_upsert_training_session
    from ..config import RAY_NAMESPACE

    model_id = str(getattr(session, "model_id", "") or "")
    session_id = str(getattr(session, "session_id", "") or "")
    base_model = str(getattr(session, "base_model", "") or "")
    if not model_id or not session_id or not base_model:
        raise RuntimeError("Cannot persist loaded training session without model_id, session_id, and base_model")

    actor_name = None
    if training_engine is not None:
        actor_name = getattr(training_engine, "_resource_pool_actor_names", {}).get(model_id)
    if actor_name is not None:
        session.actor_name = str(actor_name or "") or None
    if not getattr(session, "namespace", None):
        session.namespace = RAY_NAMESPACE

    metadata_version = _next_loaded_training_session_metadata_version(session)
    materialization_state = str(
        getattr(session, "materialization_state", MATERIALIZATION_STATE_READY) or MATERIALIZATION_STATE_READY
    )
    await async_upsert_training_session(
        {
            "model_id": model_id,
            "session_id": session_id,
            "model_seq_id": int(getattr(session, "model_seq_id")),
            "base_model": base_model,
            "lora_config": _loaded_training_session_lora_payload(getattr(session, "lora_config", None)),
            "rollout_correction_config": getattr(session, "rollout_correction_config", None),
            "user_metadata": dict(getattr(session, "user_metadata", {}) or {}),
            "learning_rate": float(getattr(session, "learning_rate")),
            "current_step": int(getattr(session, "current_step")),
            "backend": str(getattr(session, "backend")),
            "actor_name": getattr(session, "actor_name", None),
            "namespace": str(getattr(session, "namespace", RAY_NAMESPACE) or RAY_NAMESPACE),
            "user_id": getattr(session, "user_id", None) or request_user_id,
            "created_at": getattr(session, "created_at", None),
            "last_activity": float(getattr(session, "last_activity")),
            "metadata_version": metadata_version,
            "materialization_state": materialization_state,
            "tokenizer_info": getattr(session, "tokenizer_info", None),
            "tokenizer_identity": getattr(session, "tokenizer_identity", None),
            "tokenizer_source_path": getattr(session, "tokenizer_source_path", None),
        }
    )
    if training_manager is not None:
        mark_persisted = getattr(training_manager, "mark_persisted", None)
        if callable(mark_persisted):
            mark_persisted(model_id)


async def _claim_checkpoint_or_raise(
    *,
    owner_id: str | None,
    model_id: str,
    raw_checkpoint_id: str,
    checkpoint_type: str,
    model_name: str | None,
    checkpoint_created_at: str,
    retry: bool,
) -> str | None:
    try:
        return await claim_checkpoint_publication(
            owner_id=owner_id,
            model_id=model_id,
            raw_checkpoint_id=raw_checkpoint_id,
            checkpoint_type=checkpoint_type,
            storage_root=get_persistent_cache_dir(),
            model_name=model_name,
            checkpoint_created_at=checkpoint_created_at,
            retry=retry,
        )
    except (CheckpointAlreadyUploadingError, CheckpointAlreadyExistsError, CheckpointAlreadyFailedError) as e:
        raise RuntimeError(str(e)) from e


async def _mark_checkpoint_failed_safe(ckpt_id: str | None, *, fail_reason: str) -> None:
    try:
        await mark_checkpoint_failed(ckpt_id, fail_reason=fail_reason)
    except Exception:
        logger.exception(
            "[weights.checkpoint_index] mark_failed failed ckpt_id=%s fail_reason=%s",
            ckpt_id,
            fail_reason,
        )


def _require_write_access(request: Request) -> None:
    if not can_write(request):
        raise HTTPException(status_code=403, detail="Write access required")


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
            - mint://{model_id}/{name}: Legacy format -> /checkpoints/{model_id}/{name}
            - file:///path: Strip prefix
            - Absolute path: Return as-is

    Returns:
        Filesystem path.
    """
    resolved = resolve_checkpoint_path(mint_uri, user_id=user_id, is_admin=is_admin)
    ensure_checkpoint_path_allowed(resolved, user_id=user_id, is_admin=is_admin)
    return materialize_persistent_checkpoint(resolved)


def _to_mint_path(model_id: str, checkpoint_name: str) -> str:
    """Convert to mint://{model_id}/ URI (legacy format)."""
    return f"mint://{model_id}/{checkpoint_name}"


def _infer_train_flags_from_target_modules(target_modules: object) -> tuple[bool, bool, bool]:
    if not isinstance(target_modules, list) or not all(isinstance(name, str) for name in target_modules):
        raise ValueError("Checkpoint adapter_config.json target_modules must be a list of strings")
    names = set(target_modules)
    attn_names = {
        "linear_qkv",
        "linear_proj",
        "linear_q_proj",
        "linear_kv_down_proj",
        "linear_kv_up_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "wq_b",
        "wk",
        "weights_proj",
    }
    mlp_names = {"linear_fc1", "linear_fc2", "gate_proj", "up_proj", "down_proj", "gate"}
    unembed_names = {"lm_head", "output_layer", "unembed"}
    return bool(names & attn_names), bool(names & mlp_names), bool(names & unembed_names)


def _read_checkpoint_json_object(path: str, *, label: str) -> dict[str, object]:
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Malformed {label}: {e.msg}") from e
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Unable to read {label}: {e}") from e
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"Invalid {label}: expected JSON object")
    return payload


def _checkpoint_can_recreate_training_client(path: str) -> bool:
    if checkpoint_has_openpi_training_state(path):
        return True
    try:
        names = os.listdir(path)
    except OSError:
        return False
    has_rank_adapter = any(name.startswith("mp_rank_") and name.endswith("_adapter.pt") for name in names)
    if has_rank_adapter:
        return True
    has_adapter_model = "adapter_model.safetensors" in names
    return has_adapter_model and checkpoint_has_optimizer_state(path)


@router.post("/weights_info", response_model=WeightsInfoResponse)
async def weights_info(
    request: WeightsInfoRequest,
    http_request: Request,
) -> WeightsInfoResponse:
    user_id = _get_user_id(http_request)
    try:
        path = _resolve_mint_path(
            request.tinker_path,
            user_id=user_id,
            is_admin=can_bypass_ownership(http_request),
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    metadata_path = os.path.join(path, "metadata.json")
    adapter_cfg_path = os.path.join(path, "adapter_config.json")
    adapter_model_path = os.path.join(path, "adapter_model.safetensors")

    metadata: dict[str, object] = {}
    if os.path.exists(metadata_path):
        metadata = _read_checkpoint_json_object(metadata_path, label="metadata.json")

    declared_type = metadata.get("checkpoint_type", metadata.get("type"))
    if declared_type == "sampler" or "/sampler_weights/" in request.tinker_path:
        raise HTTPException(status_code=400, detail="Sampler checkpoint cannot recreate a training client")
    if not _checkpoint_can_recreate_training_client(path):
        raise HTTPException(status_code=400, detail="Missing training weights in checkpoint")

    adapter_cfg: dict[str, object] = {}
    if os.path.exists(adapter_cfg_path):
        adapter_cfg = _read_checkpoint_json_object(adapter_cfg_path, label="adapter_config.json")

    base_model = metadata.get("model_name")
    if not isinstance(base_model, str) or not base_model:
        base_model = adapter_cfg.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise HTTPException(status_code=400, detail="Checkpoint metadata missing base model")

    is_lora = bool(
        os.path.exists(adapter_cfg_path)
        or os.path.exists(adapter_model_path)
        or any(name.endswith("_adapter.pt") for name in os.listdir(path))
    )
    lora_rank = adapter_cfg.get("r")
    if not isinstance(lora_rank, int) or isinstance(lora_rank, bool) or lora_rank <= 0:
        if is_lora:
            raise HTTPException(status_code=400, detail="Invalid or missing LoRA rank in adapter_config.json")
        lora_rank = None

    train_attn = train_mlp = train_unembed = None
    if is_lora:
        try:
            train_attn, train_mlp, train_unembed = _infer_train_flags_from_target_modules(
                adapter_cfg.get("target_modules")
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return WeightsInfoResponse(
        base_model=base_model,
        is_lora=is_lora,
        lora_rank=lora_rank,
        train_unembed=train_unembed,
        train_mlp=train_mlp,
        train_attn=train_attn,
    )


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
) -> list[str]:
    candidates: list[str] = []
    for root in get_persistent_search_roots(primary_root=CHECKPOINTS_DIR):
        owner_dir = user_id or "anonymous"
        candidates.extend(
            [
                os.path.join(root, owner_dir, model_id, checkpoint_name),
                os.path.join(root, model_id, checkpoint_name),
                os.path.join(root, owner_dir, checkpoint_name),
            ]
        )
        if is_admin and os.path.isdir(root):
            try:
                for owner in os.listdir(root):
                    candidates.append(os.path.join(root, owner, model_id, checkpoint_name))
                    candidates.append(os.path.join(root, owner, checkpoint_name))
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
        is_admin=can_bypass_ownership(request),
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
        is_admin=can_bypass_ownership(request),
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
    _require_write_access(http_request)
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
        await future_store.async_mark_queued(request_id, meta={"op": "weights.save_weights", "model_id": request.model_id})
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
    _require_write_access(http_request)
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
        await future_store.async_mark_queued(request_id, meta={"op": "weights.save_state", "model_id": request.model_id})
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
    claimed_ckpt_id: str | None = None
    mirror_started = False

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

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        claimed_ckpt_id = await _claim_checkpoint_or_raise(
            owner_id=user_id,
            model_id=session.model_id,
            raw_checkpoint_id=checkpoint_name,
            checkpoint_type="training",
            model_name=session.base_model,
            checkpoint_created_at=created_at,
            retry=bool(request.retry),
        )

        save_path = build_persistent_cache_dir(
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type="training",
        )

        logger.info(f"[{session.model_id}] Saving state to: {save_path}")

        # Save training checkpoint on worker, returns path
        async def _save_state_once() -> None:
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

        try:
            await _save_state_once()
        except Exception as save_exc:
            text = str(save_exc)
            recoverable = (
                "missing worker for backend" in text
                or "non-create_session request before initialization" in text
                or "OpenPI FAST runtime session is not initialized" in text
            )
            if not recoverable:
                raise
            logger.warning(
                "[weights.save_state] rematerializing training session after checkpoint export failure: model_id=%s error=%s",
                request.model_id,
                save_exc,
            )
            await training_engine.create_training_session(session)
            await _save_state_once()

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
            "created_at": created_at,
            "step": session.current_step,
            "checkpoint_type": "training",
            "optimizer_present": optimizer_present,
            "backend": getattr(session, "backend", "unknown"),
            "type": "training",
            "storage_tier": "persistent_cache",
            "ttl_seconds": request.ttl_seconds,
            "ckpt_id": claimed_ckpt_id,
        }
        write_checkpoint_metadata(save_path, metadata)

        persistent_path = begin_async_checkpoint_mirror(
            save_path,
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type="training",
        )
        mirror_started = True

        # Sampling engines load checkpoints on demand via checkpoint_uri/create_sampling_session.
        # Do not block save_state completion on vLLM engine creation or LoRA hot-load.
        sampling_registered = False

        from ..client_compat import checkpoint_uri

        mint_path = _to_mint_path(session.model_id, checkpoint_name)
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
            "checkpoint_record_id": claimed_ckpt_id,
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
        if not mirror_started:
            await _mark_checkpoint_failed_safe(claimed_ckpt_id, fail_reason="upload_error")
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
    claimed_ckpt_id: str | None = None
    mirror_started = False
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

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        claimed_ckpt_id = await _claim_checkpoint_or_raise(
            owner_id=user_id,
            model_id=session.model_id,
            raw_checkpoint_id=checkpoint_name,
            checkpoint_type="sampler",
            model_name=session.base_model,
            checkpoint_created_at=created_at,
            retry=bool(request.retry),
        )

        save_path = build_persistent_cache_dir(
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type="sampler",
        )

        logger.info(f"[{session.model_id}] Saving sampler weights to: {save_path}")

        async def _save_weights_once() -> None:
            await run_async_with_otel_span(
                "weights.save_weights.execute",
                lambda: training_engine.save_weights_for_sampler(
                    session=session,
                    checkpoint_name=checkpoint_name,
                    checkpoint_base_dir=os.path.dirname(os.path.dirname(os.path.dirname(save_path))),
                    checkpoint_type="sampler",
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

        try:
            await _save_weights_once()
        except Exception as save_exc:
            text = str(save_exc)
            recoverable = (
                "missing worker for backend" in text
                or "non-create_session request before initialization" in text
                or "OpenPI FAST runtime session is not initialized" in text
            )
            if not recoverable:
                raise
            logger.warning(
                "[weights.save_weights] rematerializing training session after sampler export failure: model_id=%s error=%s",
                request.model_id,
                save_exc,
            )
            await training_engine.create_training_session(session)
            await _save_weights_once()

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
            "created_at": created_at,
            "step": session.current_step,
            "checkpoint_type": "sampler",
            "optimizer_present": False,
            "backend": getattr(session, "backend", "unknown"),
            "type": "sampler",
            "storage_tier": "persistent_cache",
            "ttl_seconds": request.ttl_seconds,
            "ckpt_id": claimed_ckpt_id,
        }
        write_checkpoint_metadata(save_path, metadata)

        persistent_path = begin_async_checkpoint_mirror(
            save_path,
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type="sampler",
        )
        mirror_started = True

        # Keep save_weights completion scoped to checkpoint export + metadata publication.
        # Sampler clients can load the returned checkpoint path on demand.
        sampling_registered = False

        from ..client_compat import checkpoint_uri

        mint_path = _to_mint_path(session.model_id, checkpoint_name)
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
                "checkpoint_record_id": claimed_ckpt_id,
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
        if not mirror_started:
            await _mark_checkpoint_failed_safe(claimed_ckpt_id, fail_reason="upload_error")
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
    _require_write_access(http_request)
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
        can_system = can_manage_system(http_request)
        if request.path.startswith(("tinker://", "mint://", "ckpt_")):
            local_path = resolve_checkpoint_path(request.path, user_id=user_id, is_admin=can_system)
            try:
                ensure_checkpoint_path_allowed(local_path, user_id=user_id, is_admin=can_system)
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
    load_path = _resolve_mint_path(request.path, user_id=user_id, is_admin=can_manage_system(http_request))
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
        await future_store.async_mark_queued(request_id, meta={"op": "weights.load_state", "model_id": request.model_id})
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
        async def _load_state_once() -> None:
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

        try:
            await _load_state_once()
        except Exception as load_exc:
            text = str(load_exc)
            recoverable = (
                "missing worker for backend" in text
                or "non-create_session request before initialization" in text
                or "OpenPI FAST runtime session is not initialized" in text
            )
            if not recoverable:
                raise
            logger.warning(
                "[weights.load_state] rematerializing training session after load failure: model_id=%s error=%s",
                request.model_id,
                load_exc,
            )
            await training_engine.create_training_session(session)
            await _load_state_once()

        metadata_persisted = True
        metadata_persist_error = None
        try:
            await _persist_loaded_training_session(session, request_user_id=user_id)
        except Exception as persist_exc:
            metadata_persisted = False
            metadata_persist_error = f"{type(persist_exc).__name__}: {persist_exc}"
            logger.exception(
                "[weights.load_state] detached metadata persistence failed after actor load succeeded: "
                "request_id=%s model_id=%s error_type=%s",
                str(request_id),
                str(request.model_id),
                type(persist_exc).__name__,
            )
        payload = {
            "path": request.path,
            "type": "load_weights",
        }
        if not metadata_persisted:
            payload["metadata_persisted"] = False
            payload["metadata_persist_error"] = metadata_persist_error
        await future_store.async_resolve(request_id, payload)

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
    _require_write_access(http_request)
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
        if can_bypass_ownership(request) and os.path.isdir(root):
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
            ckpt_path = os.path.join(checkpoints_path, name)
            if not os.path.isdir(ckpt_path):
                continue
            key = f"{checkpoints_path}:{name}"
            if key in seen:
                continue
            seen.add(key)

            metadata_path = os.path.join(ckpt_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue  # refuse unauthenticated legacy dirs
            try:
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                continue

            if metadata.get("model_id") != model_id:
                continue
            if not can_bypass_ownership(request) and metadata.get("owner_id") != user_id:
                continue

            # Try to parse step from directory name
            step = None
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                except (IndexError, ValueError):
                    pass

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
                    step=step,
                    created_at=created_at,
                    storage_tier=storage_tier,
                    mirror_status=mirror_status,
                    mirror_error=mirror_error,
                )
            )

    # Also include uploaded checkpoints stored as /checkpoints/{owner}/{checkpoint_id}/ if metadata.model_id matches.
    owner_roots: list[str]
    if can_bypass_ownership(request):
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
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                continue
            if metadata.get("model_id") != model_id:
                continue
            if not can_bypass_ownership(request) and metadata.get("owner_id") != user_id:
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
    _require_write_access(request)
    user_id = _get_user_id(request)

    checkpoint_name, expected_type = _split_tinker_checkpoint_id(checkpoint_id)
    candidates = _persistent_candidate_paths(
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        user_id=user_id,
        is_admin=can_bypass_ownership(request),
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
    if not can_bypass_ownership(request) and metadata.get("owner_id") != user_id:
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
        is_admin=can_bypass_ownership(request),
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
        if not can_bypass_ownership(request) and md.get("owner_id") != user_id:
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
        # Run tar in parent directory, archive the checkpoint_id folder
        parent_dir = os.path.dirname(ckpt_path)
        proc = subprocess.Popen(
            ["tar", "czf", "-", checkpoint_name],
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
