"""Service routes for session management.

Endpoints:
- GET /healthz: Health check
- POST /create_session: Create a new session
- POST /create_sampling_session: Create a sampling session with dedicated engine
- GET /sessions: List sessions
- GET /sessions/{session_id}: Get session details
- GET /samplers/{sampler_id}: Get sampler details
- POST /session_heartbeat: Keep session alive
- POST /telemetry: Accept telemetry data (discarded)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..auth_identity import is_admin_request, is_admin_user_data
from ..backend.async_ray_control import (
    _await_ray_ref,
    async_kill_named_actor,
    async_lookup_actor_handle,
    async_placement_group_table,
    is_actor_lookup_not_found,
)
from ..backend.session_heartbeat_store import session_heartbeat_store
from ..health_checks import public_healthz_response
from ..model_access_control import can_access_model, get_access_denied_error
from ..models.types import (
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    GetSamplerResponse,
    GetSessionResponse,
    ListSessionsResponse,
    SessionHeartbeatRequest,
    SessionHeartbeatResponse,
    TelemetryRequest,
    TelemetryResponse,
)
from ..server_info import get_server_info

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Global session manager reference (set only in execution runtimes or tests)
session_manager: SessionManager | None = None


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state (set by auth middleware)."""
    return _request_user_data(request)


def _get_user_id(request: Request) -> str | None:
    return _request_user_id(request)


def _user_visible(request_user_data: dict | None, owner: str | None) -> bool:
    request_user_id = str(request_user_data.get("user_id")) if request_user_data and request_user_data.get("user_id") else None
    if request_user_id is None:
        return True
    if is_admin_user_data(request_user_data):
        return True
    return bool(owner) and owner == request_user_id


def _local_sampling_config(session_id: str) -> tuple[str | None, str | None, int | None]:
    if session_manager is None:
        return None, None, None
    get_base_model = getattr(session_manager, "get_session_base_model", None)
    get_adapter_path = getattr(session_manager, "get_session_adapter_path", None)
    get_lora_rank = getattr(session_manager, "get_session_lora_rank", None)
    return (
        get_base_model(session_id) if callable(get_base_model) else None,
        get_adapter_path(session_id) if callable(get_adapter_path) else None,
        get_lora_rank(session_id) if callable(get_lora_rank) else None,
    )


def _parse_checkpoint_path(model_path: str) -> tuple[str, str] | None:
    if model_path.startswith("tinker://"):
        path_part = model_path[len("tinker://") :]
    elif model_path.startswith("mint://"):
        path_part = model_path[len("mint://") :]
    else:
        return None

    parts = [p for p in path_part.split("/") if p]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 3 and parts[1] in ("weights", "sampler_weights"):
        return parts[0], parts[2]
    return None


@router.get("/healthz", response_model=None)
async def healthz() -> dict:
    """Public health endpoint for cheap API-worker readiness only."""
    return public_healthz_response()


@router.get("/get_server_capabilities")
async def get_server_capabilities(http_request: Request) -> dict:
    """Return server capabilities for tinker client."""
    from ..backend.model_registry import get_model_config, list_supported_models
    from ..gateway import get_gateway_config, get_upstream_capabilities, upstream_for_model

    supported_local = list_supported_models()
    cfg = get_gateway_config()

    if cfg is None or not cfg.model_to_upstream:
        supported = supported_local
        models = [
            {
                "model_name": m,
                "max_context_length": get_model_config(m).max_model_len,
                "num_parameters": get_model_config(m).num_parameters,
            }
            for m in supported
        ]
        models.sort(key=lambda x: x["num_parameters"])
        return {
            "supported_models": models,
        }

    incoming_headers = dict(http_request.headers)
    remote_models = list(cfg.model_to_upstream.keys())

    # Fetch capabilities once per upstream alias that has at least one routed model.
    alias_to_caps: dict[str, dict[str, int]] = {}
    unavailable_aliases: set[str] = set()
    gateway_errors: list[dict[str, str]] = []
    for alias in set(cfg.model_to_upstream.values()):
        upstream = cfg.upstreams.get(alias)
        if upstream is None:
            unavailable_aliases.add(alias)
            gateway_errors.append(
                {"type": "gateway_misconfig", "alias": alias, "error": "unknown upstream alias"}
            )
            continue
        try:
            alias_to_caps[alias] = await get_upstream_capabilities(
                upstream=upstream, incoming_headers=incoming_headers
            )
        except Exception as e:
            logger.exception("Upstream capabilities unavailable: %s", alias)
            unavailable_aliases.add(alias)
            gateway_errors.append(
                {"type": "upstream_unavailable", "alias": alias, "error": f"{type(e).__name__}: {e}"}
            )

    merged: list[dict] = []
    seen: set[str] = set()
    for m in supported_local + remote_models:
        if m in seen:
            continue
        seen.add(m)

        if m in cfg.model_to_upstream:
            upstream = upstream_for_model(m)
            if upstream is None:
                gateway_errors.append(
                    {"type": "gateway_misconfig", "model": m, "error": "no upstream resolved for routed model"}
                )
                continue
            if upstream.alias in unavailable_aliases:
                gateway_errors.append(
                    {"type": "upstream_unavailable", "model": m, "alias": upstream.alias, "error": "capabilities unavailable"}
                )
                continue
            caps = alias_to_caps.get(upstream.alias, {})
            config = None
            try:
                config = get_model_config(m)
            except (ValueError, KeyError):
                config = None

            if m in caps:
                max_len = int(caps[m])
            else:
                gateway_errors.append(
                    {
                        "type": "gateway_misconfig",
                        "model": m,
                        "alias": upstream.alias,
                        "error": "model not present in upstream capabilities",
                    }
                )
                continue
            num_params = config.num_parameters if config is not None else None
        else:
            config = get_model_config(m)
            max_len = int(config.max_model_len)
            num_params = config.num_parameters

        entry = {"model_name": m, "max_context_length": max_len}
        if num_params is not None:
            entry["num_parameters"] = num_params
        merged.append(entry)

    # Sort by num_parameters (models without num_parameters go last)
    merged.sort(key=lambda x: (x.get("num_parameters") is None, x.get("num_parameters", float("inf"))))

    return {
        "supported_models": merged,
        "status": "degraded" if gateway_errors else "ready",
        "gateway_errors": gateway_errors,
    }


@router.get("/server_info")
async def server_info() -> dict:
    return get_server_info()


@router.post("/create_session")
async def create_session(request: CreateSessionRequest, http_request: Request) -> CreateSessionResponse:
    """Create a new session.

    Sessions are used to group related operations together.
    """
    session_id = str(uuid.uuid4())
    user_id = _get_user_id(http_request)
    created_at = datetime.now().isoformat()
    try:
        from ..backend.session_index_store import upsert_session_index

        upsert_session_index(
            {
                "session_id": session_id,
                "training_run_ids": [],
                "sampler_ids": [],
                "user_id": user_id,
                "created_at": created_at,
            }
        )
    except Exception as e:
        logger.warning("[create_session] session index write failed: %s", e)
    return CreateSessionResponse(session_id=session_id)


async def _create_sampling_session_impl(
    request: CreateSamplingSessionRequest,
    http_request: Request,
) -> CreateSamplingSessionResponse:
    """Create a sampling session using the shared multi-LoRA engine.

    Uses the shared multi-LoRA engine for efficient session management:
    - Without model_path: Uses base model (no LoRA)
    - With model_path: Loads LoRA adapter into shared engine

    First call lazily initializes the multi-LoRA engine (~60s).
    Subsequent calls register sessions instantly (<1s).
    """
    user_id = _get_user_id(http_request)
    created_at = datetime.now().isoformat()
    # Determine base_model from request or infer from model_path.
    base_model, adapter_path = _resolve_base_model_for_sampling_request(
        base_model=request.base_model,
        model_path=request.model_path,
        user_id=user_id,
        http_request=http_request,
    )

    if not base_model:
        raise HTTPException(
            status_code=422,
            detail="base_model is required. Provide base_model or model_path with adapter_config.json containing base_model_name_or_path.",
        )

    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(base_model=base_model, http_request=http_request)

    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(base_model, user_data):
        raise HTTPException(
            status_code=403,
            detail=get_access_denied_error(base_model)
        )

    # Gateway forwarding: if base_model is configured as remote, proxy to upstream and
    # return upstream sampling_session_id (tracking it for subsequent asample routing).
    from ..gateway import async_register_remote_sampling_session, forward_json, upstream_for_model

    upstream = upstream_for_model(base_model)
    if upstream is not None:
        payload = request.model_dump()
        payload["base_model"] = base_model
        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/create_sampling_session",
                incoming_headers=dict(http_request.headers),
                json_body=payload,
                timeout_s=90.0,
            )
        except Exception:
            raise HTTPException(status_code=503, detail=f"Upstream {upstream.alias!r} create_sampling_session failed")

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        sampling_session_id_remote = data.get("sampling_session_id")
        if not isinstance(sampling_session_id_remote, str) or not sampling_session_id_remote:
            raise HTTPException(
                status_code=502, detail="Upstream create_sampling_session returned invalid sampling_session_id"
            )

        await async_register_remote_sampling_session(
            sampling_session_id=sampling_session_id_remote,
            upstream_alias=upstream.alias,
            base_model=base_model,
        )
        return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id_remote)

    if request.model_path:
        # Resolve adapter directory (file://, mint://, absolute path).
        if adapter_path is None:
            adapter_path = _resolve_model_path(
                request.model_path, user_id=user_id, http_request=http_request
            )

        # Fast validation: ensure weights exist; loading happens on first /asample.
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            raise HTTPException(status_code=400, detail=f"Adapter weights not found: {weights_path}")

        # The MinT SDK `create_sampling_client(model_path=...)` does not expose lora_rank.
        # Use adapter_config.json (if present) as the source of truth to avoid
        # registering a mismatched rank (e.g., default 32 vs adapter r=64).
        lora_rank = request.lora_rank
        config_path = os.path.join(adapter_path, "adapter_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    adapter_config = json.load(f)
                inferred_rank = adapter_config.get("r")
                if isinstance(inferred_rank, int) and inferred_rank > 0:
                    if int(lora_rank) != inferred_rank:
                        logger.info(
                            f"[create_sampling_session] overriding lora_rank={lora_rank} "
                            f"with adapter_config.r={inferred_rank} for {adapter_path}"
                        )
                    lora_rank = inferred_rank
        except Exception as e:
            logger.warning(
                f"[create_sampling_session] failed to infer lora_rank from {config_path}: "
                f"{type(e).__name__}: {e}"
            )
    else:
        lora_rank = 0

    def _write_sampler_index(sampler_id: str) -> None:
        try:
            from ..backend.session_index_store import add_sampler_to_session, upsert_sampler_index

            add_sampler_to_session(
                session_id=request.session_id,
                sampler_id=sampler_id,
                user_id=user_id,
                created_at=created_at,
            )

            sampler_info: dict = {
                "sampler_id": sampler_id,
                "session_id": request.session_id,
                "base_model": base_model,
                "user_id": user_id,
                "created_at": created_at,
            }

            if request.model_path:
                parsed = _parse_checkpoint_path(request.model_path)
                if parsed:
                    model_id, checkpoint_name = parsed
                    sampler_info.update(
                        {
                            "source_type": "checkpoint",
                            "model_id": model_id,
                            "checkpoint_name": checkpoint_name,
                            "model_path_raw": request.model_path,
                        }
                    )
                else:
                    sampler_info.update(
                        {
                            "source_type": "raw_model_path",
                            "model_path_raw": request.model_path,
                        }
                    )
            else:
                sampler_info.update({"source_type": "base_model"})

            upsert_sampler_index(sampler_info)
        except Exception as e:
            logger.warning("[create_sampling_session] sampler index write failed: %s", e)

    if request.sampling_session_seq_id is not None:
        sampling_session_id = f"{request.session_id}:sample:{request.sampling_session_seq_id}"
        existing_info = None
        try:
            from ..backend.sampling_session_store import async_get_sampling_session_info

            existing_info = await async_get_sampling_session_info(sampling_session_id)
        except Exception:
            existing_info = None
        if isinstance(existing_info, dict):
            existing_base = str(existing_info.get("base_model") or "")
            existing_adapter = existing_info.get("adapter_path")
            existing_rank = int(existing_info.get("lora_rank") or 0)
        else:
            existing_base, existing_adapter, existing_rank = _local_sampling_config(sampling_session_id)
        if existing_base is not None:
            expected_adapter = adapter_path if request.model_path else None
            expected_rank = int(lora_rank)
            if existing_base != base_model or existing_adapter != expected_adapter or int(existing_rank or 0) != expected_rank:
                raise HTTPException(
                    status_code=409,
                    detail="Sampling session already exists with different configuration",
                )
            _write_sampler_index(sampling_session_id)
            return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)
    else:
        sampling_session_id = str(uuid.uuid4())

    try:
        from ..backend.sampling_session_store import upsert_sampling_session

        upsert_sampling_session(
            {
                "session_id": sampling_session_id,
                "base_model": base_model,
                "lora_rank": int(lora_rank),
                "adapter_path": adapter_path,
                "lora_loaded": False,
                "lora_int_id": None,
                "uses_base_model": not bool(request.model_path),
                "last_activity": time.time(),
                "inflight_requests": 0,
                "metadata_version": 1,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail="Sampling session store unavailable") from e

    if session_manager is not None:
        if request.model_path:
            session_manager.register_multi_lora_session(
                session_id=sampling_session_id,
                base_model=base_model,
                lora_rank=lora_rank,
                adapter_path=adapter_path,
                lora_loaded=False,
            )
        else:
            session_manager.register_base_model_session(sampling_session_id, base_model=base_model)

    _write_sampler_index(sampling_session_id)

    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


@router.post("/create_sampling_session")
async def create_sampling_session(
    request: CreateSamplingSessionRequest,
    http_request: Request,
) -> CreateSamplingSessionResponse:
    return await _create_sampling_session_impl(request, http_request)


async def ensure_sampling_session(
    *,
    model_path: str,
    http_request: Request,
    parent_session_id: str | None = None,
) -> tuple[str, str]:
    """Ensure a sampling session exists for an OpenAI-compatible request."""
    from ..gateway import async_remote_sampling_session

    request_kwargs: dict[str, str] = {
        "session_id": parent_session_id or str(uuid.uuid4()),
    }
    if model_path.startswith(("tinker://", "mint://", "ckpt_", "file://", "/")):
        request_kwargs["model_path"] = model_path
    else:
        request_kwargs["base_model"] = model_path

    sampling_request = CreateSamplingSessionRequest(**request_kwargs)
    response = await create_sampling_session(sampling_request, http_request)
    sampling_session_id = response.sampling_session_id
    base_model = None
    try:
        from ..backend.sampling_session_store import async_get_sampling_session_info

        info = await async_get_sampling_session_info(sampling_session_id)
        if isinstance(info, dict):
            base_model = info.get("base_model")
    except Exception:
        base_model = None
    if base_model is None and session_manager is not None:
        base_model, _adapter_path, _rank = _local_sampling_config(sampling_session_id)
    if base_model is None:
        remote = await async_remote_sampling_session(sampling_session_id)
        if remote is not None:
            _, base_model = remote
    if base_model is None:
        base_model = request_kwargs.get("base_model")

    if not base_model:
        raise HTTPException(
            status_code=500,
            detail=f"Sampling session {sampling_session_id!r} missing base_model after creation",
        )
    return sampling_session_id, str(base_model)


@router.get("/sessions/{session_id}", response_model=GetSessionResponse)
async def get_session(session_id: str, http_request: Request) -> GetSessionResponse:
    request_user_data = _get_user_data(http_request)
    info = None
    try:
        from ..backend.session_index_store import async_get_session_index

        info = await async_get_session_index(session_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Session index store unavailable") from e

    if isinstance(info, dict):
        if not _user_visible(request_user_data, info.get("user_id")):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return GetSessionResponse(
            training_run_ids=list(info.get("training_run_ids") or []),
            sampler_ids=list(info.get("sampler_ids") or []),
        )

    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.get("/sessions", response_model=ListSessionsResponse)
async def list_sessions(limit: int = 20, offset: int = 0, http_request: Request = None) -> ListSessionsResponse:
    request_user_data = _get_user_data(http_request) if http_request else None
    entries: list[dict] = []

    try:
        from ..backend.session_index_store import async_list_session_index

        infos = await async_list_session_index()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Session index store unavailable") from e

    for info in infos or []:
        sid = info.get("session_id")
        if not isinstance(sid, str) or not sid:
            continue
        if not _user_visible(request_user_data, info.get("user_id")):
            continue
        entries.append({"session_id": sid, "created_at": info.get("created_at")})

    entries.sort(key=lambda x: str(x.get("session_id") or ""))
    entries.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)

    if offset < 0:
        offset = 0
    if limit < 0:
        limit = 0
    page = entries[offset : offset + limit]
    return ListSessionsResponse(sessions=[e["session_id"] for e in page])


@router.get("/samplers/{sampler_id}", response_model=GetSamplerResponse)
async def get_sampler(sampler_id: str, http_request: Request) -> GetSamplerResponse:
    request_user_data = _get_user_data(http_request)
    info = None
    try:
        from ..backend.session_index_store import async_get_sampler_index

        info = await async_get_sampler_index(sampler_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Session index store unavailable") from e

    if isinstance(info, dict):
        if not _user_visible(request_user_data, info.get("user_id")):
            raise HTTPException(status_code=404, detail=f"Sampler '{sampler_id}' not found")
        base_model = info.get("base_model")
        if not base_model:
            try:
                from ..backend.sampling_session_store import async_get_sampling_session_info

                detached = await async_get_sampling_session_info(sampler_id)
            except Exception:
                detached = None
            if isinstance(detached, dict):
                base_model = detached.get("base_model")
            elif session_manager is not None:
                base_model, _adapter_path, _rank = _local_sampling_config(sampler_id)

        from ..client_compat import checkpoint_uri, prefer_tinker_uri

        model_path = None
        source_type = info.get("source_type")
        if source_type == "checkpoint":
            model_id = info.get("model_id")
            checkpoint_name = info.get("checkpoint_name")
            if model_id and checkpoint_name:
                model_path = checkpoint_uri(
                    str(model_id),
                    str(checkpoint_name),
                    prefer_tinker=prefer_tinker_uri(http_request),
                    checkpoint_type="sampler",
                )
            else:
                model_path = info.get("model_path_raw")
        elif source_type == "raw_model_path":
            model_path = info.get("model_path_raw")
        elif source_type == "base_model":
            model_path = None
        else:
            model_path = info.get("model_path_raw")

        if not base_model:
            raise HTTPException(status_code=404, detail=f"Sampler '{sampler_id}' not found")

        return GetSamplerResponse(
            sampler_id=sampler_id,
            base_model=str(base_model),
            model_path=model_path,
        )

    try:
        from ..backend.sampling_session_store import async_get_sampling_session_info

        detached = await async_get_sampling_session_info(sampler_id)
    except Exception:
        detached = None
    if isinstance(detached, dict):
        base_model = detached.get("base_model")
        if base_model:
            return GetSamplerResponse(
                sampler_id=sampler_id,
                base_model=str(base_model),
                model_path=None,
            )
    if session_manager is not None:
        base_model, _adapter_path, _rank = _local_sampling_config(sampler_id)
        if base_model:
            return GetSamplerResponse(
                sampler_id=sampler_id,
                base_model=str(base_model),
                model_path=None,
            )

    raise HTTPException(status_code=404, detail=f"Sampler '{sampler_id}' not found")


def _resolve_model_path(
    model_path: str, *, user_id: str | None, http_request: Request
) -> str:
    """Resolve model_path URI to filesystem path.

    Args:
        model_path: URI like file:///path, mint://{uuid}/..., or absolute path.

    Returns:
        Absolute filesystem path to adapter directory.
    """
    from ..checkpoints import (
        ensure_checkpoint_path_allowed,
        materialize_persistent_checkpoint,
        resolve_checkpoint_uri,
    )

    is_admin = is_admin_request(http_request)
    if not is_admin and not model_path.startswith(("tinker://", "mint://", "ckpt_")):
        raise HTTPException(status_code=403, detail="Access denied")

    resolved = resolve_checkpoint_uri(model_path, "", user_id=user_id, is_admin=is_admin)
    if model_path.startswith("ckpt_") and resolved == model_path:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    try:
        ensure_checkpoint_path_allowed(resolved, user_id=user_id, is_admin=is_admin)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return materialize_persistent_checkpoint(resolved)


def _resolve_base_model_for_sampling_request(
    *,
    base_model: str | None,
    model_path: str | None,
    user_id: str | None,
    http_request: Request,
) -> tuple[str | None, str | None]:
    """Return the effective base_model and resolved adapter path for a sampling request."""
    adapter_path: str | None = None
    if not base_model and model_path:
        adapter_path = _resolve_model_path(
            model_path,
            user_id=user_id,
            http_request=http_request,
        )
        base_model = _infer_base_model_from_adapter(adapter_path)
    return base_model, adapter_path


def _infer_base_model_from_adapter(adapter_path: str) -> str | None:
    """Infer base_model from adapter_config.json if present.

    Args:
        adapter_path: Filesystem path to adapter directory.

    Returns:
        base_model name if found, None otherwise.
    """
    import json
    import os

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            return config.get("base_model_name_or_path") or config.get("base_model")
    return None


def _load_adapter_from_path(adapter_path: str, lora_rank: int) -> tuple[dict, dict]:
    """Load LoRA adapter weights and config from disk.

    Args:
        adapter_path: Filesystem path to adapter directory.
        lora_rank: LoRA rank for config.

    Returns:
        (state_dict, peft_config) tuple.
    """
    import json
    import os

    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cannot load adapter_model.safetensors on this API host (missing torch-backed safetensors). "
                "Load adapters via a GPU worker environment."
            ),
        ) from e

    # Load weights
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=400,
            detail=f"Adapter weights not found: {weights_path}",
        )
    state_dict = load_file(weights_path)

    # Load config
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            peft_config = json.load(f)
    else:
        # Construct minimal config if missing
        # Include MLP modules for dense models (vLLM supports them).
        # Note: MoE expert LoRA is NOT supported by vLLM's FusedMoE kernel.
        peft_config = {
            "r": lora_rank,
            "lora_alpha": lora_rank,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }

    return state_dict, peft_config


@router.post("/session_heartbeat")
async def session_heartbeat(
    request: SessionHeartbeatRequest,
) -> SessionHeartbeatResponse:
    """Keep session alive.

    Accepts heartbeat and returns acknowledgment. Session validation not implemented.
    """
    await session_heartbeat_store.async_update(request.session_id)
    try:
        from ..backend.sampling_session_store import async_set_sampling_session_last_activity

        await async_set_sampling_session_last_activity(request.session_id, time.time())
    except Exception as e:
        raise HTTPException(status_code=503, detail="Sampling session store unavailable") from e
    return SessionHeartbeatResponse()


@router.post("/telemetry")
async def send_telemetry(request: TelemetryRequest) -> TelemetryResponse:
    """Accept telemetry data from tinker client.

    Silently accepts and discards telemetry data.
    """
    return TelemetryResponse(status="accepted")


# =============================================================================
# Admin endpoints for actor management
# =============================================================================
def _require_admin(request: Request) -> None:
    """Raise 403 if not admin user."""
    from ..config import config as server_config
    if not server_config.auth_enabled:
        return
    user_data = getattr(request.state, "user_data", None)
    if not is_admin_user_data(user_data):
        raise HTTPException(status_code=403, detail="Admin access required")


async def _augment_with_placement_groups(actors: list[dict]) -> None:
    try:
        import ray

        if not ray.is_initialized():
            return

        # Offload PG inspection into a Ray task so we never block the API event loop
        # with synchronous control-plane calls.
        timeout_s = float(os.environ.get("MINT_ACTORS_PG_TABLE_TIMEOUT_S", "2.0"))
        try:
            tbl = await async_placement_group_table(timeout_s=timeout_s)
        except asyncio.TimeoutError:
            return
        except Exception:
            return

        by_name: dict[str, dict] = {}
        for info in tbl.values():
            if isinstance(info, dict):
                name = info.get("name")
                if isinstance(name, str) and name:
                    by_name[name] = info

        for a in actors:
            name = a.get("actor_name")
            if not isinstance(name, str) or not name:
                continue
            pg_name = f"{name}_pg"
            try:
                info = by_name.get(pg_name)
                if not isinstance(info, dict):
                    continue
                bundles = info.get("bundles") or {}
                if not isinstance(bundles, dict):
                    continue
                total_gpu = 0
                for bundle in bundles.values():
                    if isinstance(bundle, dict):
                        total_gpu += int(bundle.get("GPU", 0) or 0)
                a["pg_name"] = pg_name
                a["pg_bundle_count"] = len(bundles)
                a["pg_total_gpus"] = int(total_gpu)
            except Exception:
                continue
    except Exception:
        return


@router.get("/actors")
async def list_actors(
    request: Request,
    actor_type: str | None = Query(None, alias="type"),
    model_name: str | None = None,
) -> dict:
    """List actors in the unified ResourcePool. Admin only when auth enabled.

    Query params:
        type: Optional filter ("vllm" | "megatron" | "dense")
        model_name: Optional filter on ActorEntry.base_model
    """
    _require_admin(request)
    from ..backend.resource_pool import ActorType, get_resource_pool

    pool = get_resource_pool()
    actors = pool.list_actors()

    if actor_type is not None:
        t = actor_type.strip().lower()
        allowed = {x.value for x in ActorType}
        if t not in allowed:
            raise HTTPException(status_code=422, detail=f"Invalid type {actor_type!r}; expected one of {sorted(allowed)}")
        actors = [a for a in actors if a.get("actor_type") == t]

    if model_name is not None:
        actors = [a for a in actors if a.get("base_model") == model_name]

    await _augment_with_placement_groups(actors)
    return {"actors": actors, "total_gpus_used": pool.total_gpus_used()}


class KillActorsRequest(BaseModel):
    """Request to kill actor(s)."""

    actor_type: str  # "vllm" | "megatron" | "dense" | "all"
    model_name: str | None = None  # optional per-type model filter
    actor_name: str | None = None  # optional exact actor target
    force: bool = False  # allow killing actors with in-flight work
    reason: str | None = None  # optional operator-provided audit reason


def _entry_actor_type_name(entry: object) -> str:
    raw = getattr(entry, "actor_type", None)
    value = getattr(raw, "value", raw)
    return str(value or "").strip().lower()


def _entry_matches_kill_request(
    entry: object,
    *,
    actor_type: str,
    model_name: str | None,
    actor_name: str | None,
) -> bool:
    entry_name = str(getattr(entry, "actor_name", "") or "")
    if actor_name is not None and entry_name != actor_name:
        return False

    entry_type = _entry_actor_type_name(entry)
    if actor_type != "all" and entry_type != actor_type:
        return False

    if model_name is not None and str(getattr(entry, "base_model", "") or "") != model_name:
        return False

    return True


def _collect_kill_target_entries(
    *,
    actor_type: str,
    model_name: str | None,
    actor_name: str | None,
) -> list[object]:
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    return [
        entry
        for entry in pool.iter_entries()
        if _entry_matches_kill_request(
            entry,
            actor_type=actor_type,
            model_name=model_name,
            actor_name=actor_name,
        )
    ]


def _kill_target_snapshot(entries: list[object]) -> list[dict[str, object]]:
    return [
        {
            "actor_name": str(getattr(entry, "actor_name", "") or ""),
            "actor_type": _entry_actor_type_name(entry),
            "base_model": str(getattr(entry, "base_model", "") or ""),
            "current_session": getattr(entry, "current_session", None),
            "inflight_count": int(getattr(entry, "inflight_count", 0) or 0),
            "creating": bool(getattr(entry, "creating", False)),
            "protected": bool(getattr(entry, "protected", False)),
        }
        for entry in entries
    ]


def _request_audit_fields(request: Request) -> dict[str, object]:
    user_data = _get_user_data(request)
    client = getattr(request, "client", None)
    return {
        "client_host": getattr(client, "host", None),
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "user_agent": request.headers.get("user-agent"),
        "origin": request.headers.get("origin"),
        "referer": request.headers.get("referer"),
        "user_id": user_data.get("user_id") if isinstance(user_data, dict) else None,
        "is_admin": bool(is_admin_user_data(user_data)),
    }


def _log_kill_request(
    request: Request,
    body: KillActorsRequest,
    *,
    stage: str,
    targets: list[dict[str, object]],
    detail: str | None = None,
    result: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "stage": stage,
        "actor_type": body.actor_type,
        "model_name": body.model_name,
        "actor_name": body.actor_name,
        "force": body.force,
        "reason": body.reason,
        "targets": targets,
    }
    payload.update(_request_audit_fields(request))
    if detail is not None:
        payload["detail"] = detail
    if result is not None:
        payload["result"] = result
    logger.info("[actors.kill] %s", payload)


def _raise_if_busy_kill_targets(
    *,
    request: Request,
    body: KillActorsRequest,
    targets: list[dict[str, object]],
) -> None:
    if body.force:
        return
    busy = [target for target in targets if int(target.get("inflight_count", 0) or 0) > 0]
    if not busy:
        return
    actor_list = ", ".join(str(target.get("actor_name") or "<unknown>") for target in busy)
    detail = (
        f"Refusing to kill busy actor(s): {actor_list}. "
        "Pass force=true to override."
    )
    _log_kill_request(request, body, stage="blocked_busy", targets=targets, detail=detail)
    raise HTTPException(status_code=409, detail=detail)


def _remove_actor_pg(actor_name: str) -> None:
    try:
        import ray

        pg = ray.util.get_placement_group(f"{actor_name}_pg")
        ray.util.remove_placement_group(pg)
    except Exception:
        pass


async def _kill_exact_vllm_actor(*, actor_name: str) -> int:
    from ..backend.multi_lora_engine import PERSISTENT_NAMESPACE
    from ..backend.resource_pool import ActorType, ResourcePoolStaleError, get_resource_pool

    pool = get_resource_pool()
    entry = pool.get(actor_name)
    if entry is not None and entry.actor_type != ActorType.VLLM:
        return 0

    namespace = entry.namespace if entry is not None else PERSISTENT_NAMESPACE
    try:
        actor = await async_lookup_actor_handle(actor_name, namespace)
    except Exception as exc:
        if not is_actor_lookup_not_found(exc):
            raise
        pool.unregister(actor_name)
        _remove_actor_pg(actor_name)
        return 0

    try:
        await async_kill_named_actor(
            actor_name,
            namespace,
            actor_handle=actor,
            base_model=entry.base_model if entry is not None else None,
            reason="vllm_kill_by_actor_name",
        )
    except ResourcePoolStaleError:
        raise
    pool.unregister(actor_name)
    _remove_actor_pg(actor_name)
    return 1


async def _kill_exact_megatron_actor(*, actor_name: str) -> int:
    from ..backend.megatron_distributed import PERSISTENT_NAMESPACE
    from ..backend.resource_pool import ActorType, get_resource_pool

    pool = get_resource_pool()
    entry = pool.get(actor_name)
    if entry is not None and entry.actor_type != ActorType.MEGATRON:
        return 0

    namespace = entry.namespace if entry is not None else PERSISTENT_NAMESPACE
    try:
        actor = await async_lookup_actor_handle(actor_name, namespace)
    except Exception as exc:
        if not is_actor_lookup_not_found(exc):
            raise
        pool.unregister(actor_name)
        _remove_actor_pg(actor_name)
        return 0

    try:
        try:
            await asyncio.wait_for(_await_ray_ref(actor.shutdown.remote()), timeout=10.0)
        except Exception:
            pass
        await async_kill_named_actor(
            actor_name,
            namespace,
            actor_handle=actor,
            base_model=entry.base_model if entry is not None else None,
            reason="kill_megatron_actor_by_name",
            verify_absent=True,
        )
    finally:
        pool.unregister(actor_name)
        _remove_actor_pg(actor_name)
    return 1


async def _kill_exact_dense_actor(*, actor_name: str) -> int:
    from ..backend.resource_pool import ActorType, get_resource_pool

    pool = get_resource_pool()
    entry = pool.get(actor_name)
    if entry is not None and entry.actor_type != ActorType.DENSE:
        return 0
    if entry is None:
        return 0

    try:
        try:
            await async_lookup_actor_handle(entry.actor_name, entry.namespace)
            await async_kill_named_actor(
                entry.actor_name,
                entry.namespace,
                actor_handle=entry.actor_handle if entry.actor_handle is not None else None,
                base_model=entry.base_model,
                reason="dense_kill_by_actor_name",
            )
        except Exception:
            pass
    finally:
        pool.unregister(entry.actor_name)
        _remove_actor_pg(entry.actor_name)
    return 1


async def _kill_dense_actors(base_model: str | None) -> int:
    from ..backend.resource_pool import ActorType, get_resource_pool

    pool = get_resource_pool()
    targets = [
        e
        for e in pool.iter_entries()
        if e.actor_type == ActorType.DENSE and (base_model is None or e.base_model == base_model)
    ]

    killed = 0
    for e in targets:
        try:
            await async_kill_named_actor(
                e.actor_name,
                e.namespace,
                actor_handle=e.actor_handle if e.actor_handle is not None else None,
                base_model=e.base_model,
                reason="dense_kill_by_api",
            )
        except Exception:
            pass
        pool.unregister(e.actor_name)
        _remove_actor_pg(e.actor_name)
        killed += 1
    return killed


@router.post("/actors/kill")
async def kill_actors(request: Request, body: KillActorsRequest) -> dict:
    """Kill actor(s) by type. Admin only when auth enabled."""
    _require_admin(request)

    t = body.actor_type.strip().lower()
    model_name = body.model_name
    actor_name = body.actor_name.strip() if body.actor_name else None

    targets = _kill_target_snapshot(
        _collect_kill_target_entries(
            actor_type=t,
            model_name=model_name,
            actor_name=actor_name,
        )
    )
    _log_kill_request(request, body, stage="received", targets=targets)
    _raise_if_busy_kill_targets(request=request, body=body, targets=targets)

    killed_by_type: dict[str, int] = {"vllm": 0, "megatron": 0, "dense": 0}

    if actor_name:
        if t == "all":
            raise HTTPException(status_code=422, detail="actor_name cannot be combined with actor_type=all")
        if t == "vllm":
            from ..backend.resource_pool import ResourcePoolStaleError

            try:
                killed_by_type["vllm"] = await _kill_exact_vllm_actor(actor_name=actor_name)
            except ResourcePoolStaleError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e
        elif t == "megatron":
            killed_by_type["megatron"] = await _kill_exact_megatron_actor(actor_name=actor_name)
        elif t == "dense":
            killed_by_type["dense"] = await _kill_exact_dense_actor(actor_name=actor_name)
        else:
            raise HTTPException(status_code=422, detail="actor_type must be one of: vllm, megatron, dense, all")
        result = {
            "killed": int(sum(killed_by_type.values())),
            "killed_by_type": killed_by_type,
        }
        _log_kill_request(request, body, stage="completed", targets=targets, result=result)
        return result

    if t in ("vllm", "all"):
        from ..backend.multi_lora_engine import kill_persistent_vllm_actor
        from ..backend.resource_pool import ResourcePoolStaleError

        try:
            if t == "vllm":
                killed_by_type["vllm"] = 1 if kill_persistent_vllm_actor(model_name) else 0
            else:
                killed_by_type["vllm"] = 1 if kill_persistent_vllm_actor(None) else 0
        except ResourcePoolStaleError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    if t in ("megatron", "all"):
        from ..backend.megatron_distributed import kill_megatron_actor

        if t == "megatron":
            killed_by_type["megatron"] = 1 if kill_megatron_actor(model_name) else 0
        else:
            killed_by_type["megatron"] = 1 if kill_megatron_actor(None) else 0

    if t in ("dense", "all"):
        killed_by_type["dense"] = await _kill_dense_actors(model_name if t == "dense" else None)

    if t not in ("vllm", "megatron", "dense", "all"):
        raise HTTPException(status_code=422, detail="actor_type must be one of: vllm, megatron, dense, all")

    result = {
        "killed": int(sum(killed_by_type.values())),
        "killed_by_type": killed_by_type,
    }
    _log_kill_request(request, body, stage="completed", targets=targets, result=result)
    return result
