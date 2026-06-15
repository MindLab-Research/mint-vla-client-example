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

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from ..auth_identity import can_bypass_ownership_user_data
from ..auth_identity import can_manage_system
from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from mint_server.backend.stores.session_heartbeat_store import session_heartbeat_store
from ..health_checks import internal_lightweight_healthz_response, public_business_healthz_response
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
    from mint_server.backend.sessions.session_manager import SessionManager

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
    if can_bypass_ownership_user_data(request_user_data):
        return True
    return bool(owner) and owner == request_user_id


def _local_sampling_config(session_id: str) -> tuple[str | None, str | None, int | None]:
    """Runtime/test helper; HTTP routes must prefer detached sampling metadata."""
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
    if model_path.startswith("mint://"):
        path_part = model_path[len("mint://") :]
    else:
        return None

    parts = [p for p in path_part.split("/") if p]
    if len(parts) == 3 and parts[1] in ("weights", "sampler_weights"):
        return parts[0], parts[2]
    return None


def _infer_base_model_from_checkpoint(
    model_path: str,
    *,
    user_id: str | None,
    is_admin: bool = False,
) -> str | None:
    from ..checkpoints import get_checkpoints_dir, read_checkpoint_metadata, resolve_checkpoint_uri

    resolved = resolve_checkpoint_uri(
        model_path,
        get_checkpoints_dir(),
        user_id=user_id,
        is_admin=is_admin,
    )
    if not resolved or not os.path.isdir(resolved):
        return None
    try:
        metadata = read_checkpoint_metadata(resolved)
    except Exception:
        return None
    model_name = metadata.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name
    return None


@router.get("/healthz", response_model=None)
async def healthz() -> dict:
    """Public business health endpoint for client readiness."""
    return await public_business_healthz_response()


@router.get("/internal/healthz", response_model=None)
async def internal_healthz() -> dict:
    """Lightweight internal operational health for gateway-mounted path."""
    return await internal_lightweight_healthz_response()


@router.get("/get_server_capabilities")
async def get_server_capabilities(http_request: Request) -> dict:
    """Return server capabilities for tinker client."""
    from mint_server.backend.core.model_registry import get_model_config, list_supported_models
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
    from mint_server.backend.stores.session_index_store import upsert_session_index

    try:
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
        raise HTTPException(status_code=503, detail="Session index store unavailable") from e
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
        owner_id=request.owner_id,
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
        # Resolve adapter directory (file://, typed checkpoint URI, absolute path).
        if adapter_path is None:
            adapter_path = _resolve_model_path(
                request.model_path,
                user_id=user_id,
                owner_id=request.owner_id,
                http_request=http_request,
            )

        from mint_server.backend.training.bumblebee.bumblebee_lora import prepare_lora_adapter_for_vllm

        try:
            adapter_path = prepare_lora_adapter_for_vllm(adapter_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to prepare LoRA adapter for sampling: {exc}",
            ) from exc

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
        from mint_server.backend.stores.session_index_store import add_sampler_to_session, upsert_sampler_index

        # Generic create_sampling_session children stay out of root heartbeat fanout.
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

    if request.sampling_session_seq_id is not None:
        sampling_session_id = f"{request.session_id}:sample:{request.sampling_session_seq_id}"
        existing_info = None
        try:
            from mint_server.backend.stores.sampling_session_store import async_get_sampling_session_info

            existing_info = await async_get_sampling_session_info(sampling_session_id)
        except Exception:
            existing_info = None
        if isinstance(existing_info, dict):
            existing_base = str(existing_info.get("base_model") or "")
            existing_adapter = existing_info.get("adapter_path")
            existing_rank = int(existing_info.get("lora_rank") or 0)
        if isinstance(existing_info, dict):
            expected_adapter = adapter_path if request.model_path else None
            expected_rank = int(lora_rank)
            if existing_base != base_model or existing_adapter != expected_adapter or int(existing_rank or 0) != expected_rank:
                raise HTTPException(
                    status_code=409,
                    detail="Sampling session already exists with different configuration",
                )
            try:
                _write_sampler_index(sampling_session_id)
            except Exception as e:
                raise HTTPException(status_code=503, detail="Session index store unavailable") from e
            return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)
    else:
        sampling_session_id = str(uuid.uuid4())

    try:
        from mint_server.backend.stores.sampling_session_store import upsert_sampling_session

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

    try:
        _write_sampler_index(sampling_session_id)
    except Exception as e:
        try:
            from mint_server.backend.stores.sampling_session_store import delete_sampling_session

            delete_sampling_session(sampling_session_id)
        except Exception:
            logger.warning(
                "[create_sampling_session] cleanup failed after sampler index write error session_id=%s",
                sampling_session_id,
            )
        raise HTTPException(status_code=503, detail="Session index store unavailable") from e

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
    if model_path.startswith(("mint://", "ckpt_", "file://", "/")):
        request_kwargs["model_path"] = model_path
    else:
        request_kwargs["base_model"] = model_path

    sampling_request = CreateSamplingSessionRequest(**request_kwargs)
    response = await create_sampling_session(sampling_request, http_request)
    sampling_session_id = response.sampling_session_id
    base_model = None
    try:
        from mint_server.backend.stores.sampling_session_store import async_get_sampling_session_info

        info = await async_get_sampling_session_info(sampling_session_id)
        if isinstance(info, dict):
            base_model = info.get("base_model")
    except Exception:
        base_model = None
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
        from mint_server.backend.stores.session_index_store import async_get_session_index

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
        from mint_server.backend.stores.session_index_store import async_list_session_index

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
        from mint_server.backend.stores.session_index_store import async_get_sampler_index

        info = await async_get_sampler_index(sampler_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Session index store unavailable") from e

    if isinstance(info, dict):
        if not _user_visible(request_user_data, info.get("user_id")):
            raise HTTPException(status_code=404, detail=f"Sampler '{sampler_id}' not found")
        base_model = info.get("base_model")
        if not base_model:
            try:
                from mint_server.backend.stores.sampling_session_store import async_get_sampling_session_info

                persisted = await async_get_sampling_session_info(sampler_id)
            except Exception:
                persisted = None
            if isinstance(persisted, dict):
                base_model = persisted.get("base_model")

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
        from mint_server.backend.stores.sampling_session_store import async_get_sampling_session_info

        persisted = await async_get_sampling_session_info(sampler_id)
    except Exception:
        persisted = None
    if isinstance(persisted, dict):
        base_model = persisted.get("base_model")
        if base_model:
            return GetSamplerResponse(
                sampler_id=sampler_id,
                base_model=str(base_model),
                model_path=None,
            )
    raise HTTPException(status_code=404, detail=f"Sampler '{sampler_id}' not found")


def _resolve_model_path(
    model_path: str,
    *,
    user_id: str | None,
    owner_id: str | None = None,
    http_request: Request,
) -> str:
    """Resolve model_path URI to filesystem path.

    Args:
        model_path: URI like file:///path, mint://{run_id}/{kind}/{name}, or absolute path.

    Returns:
        Absolute filesystem path to adapter directory.
    """
    from ..checkpoints import (
        ensure_checkpoint_path_allowed,
        materialize_persistent_checkpoint,
        resolve_checkpoint_uri,
    )

    can_system = can_manage_system(http_request)
    if not can_system and not model_path.startswith(("mint://", "ckpt_")):
        raise HTTPException(status_code=403, detail="Access denied")

    owner_scope = owner_id if can_system else user_id
    try:
        resolved = resolve_checkpoint_uri(model_path, "", user_id=owner_scope, is_admin=can_system)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if model_path.startswith("ckpt_") and resolved == model_path:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    try:
        ensure_checkpoint_path_allowed(resolved, user_id=owner_scope, is_admin=can_system)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return materialize_persistent_checkpoint(resolved)


def _resolve_base_model_for_sampling_request(
    *,
    base_model: str | None,
    model_path: str | None,
    user_id: str | None,
    owner_id: str | None,
    http_request: Request,
) -> tuple[str | None, str | None]:
    """Return the effective base_model and resolved adapter path for a sampling request."""
    adapter_path: str | None = None
    if not base_model and model_path:
        adapter_path = _resolve_model_path(
            model_path,
            user_id=user_id,
            owner_id=owner_id,
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


async def _child_sampler_ids_for_heartbeat(
    root_session_id: str,
    request_user_data: dict | None,
) -> list[str]:
    try:
        from mint_server.backend.stores.session_index_store import async_get_sampler_index, async_get_session_index

        info = await async_get_session_index(root_session_id)
    except Exception as e:
        logger.warning("[session_heartbeat] session index lookup failed for %s: %s", root_session_id, e)
        return []

    if not isinstance(info, dict):
        return []
    if not _user_visible(request_user_data, info.get("user_id")):
        logger.warning("[session_heartbeat] child sampler propagation denied for %s", root_session_id)
        return []

    seen: set[str] = set()
    direct = info.get("heartbeat_sampler_ids") or []
    if direct:
        out: list[str] = []
        for sampler_id in direct:
            if not isinstance(sampler_id, str) or not sampler_id or sampler_id in seen:
                continue
            seen.add(sampler_id)
            out.append(sampler_id)
        return out

    training_run_ids = {
        training_run_id
        for training_run_id in info.get("training_run_ids") or []
        if isinstance(training_run_id, str) and training_run_id
    }
    out: list[str] = []
    for sampler_id in info.get("sampler_ids") or []:
        if not isinstance(sampler_id, str) or not sampler_id or sampler_id in seen:
            continue
        seen.add(sampler_id)
        try:
            sampler_info = await async_get_sampler_index(sampler_id)
        except Exception as e:
            logger.warning("[session_heartbeat] sampler index lookup failed for %s: %s", sampler_id, e)
            continue
        if not isinstance(sampler_info, dict):
            continue
        if sampler_info.get("source_type") != "checkpoint":
            continue
        model_id = sampler_info.get("model_id")
        if isinstance(model_id, str) and model_id in training_run_ids:
            out.append(sampler_id)
    return out


async def _touch_child_sampler_sessions(root_session_id: str, request_user_data: dict | None) -> None:
    from mint_server.backend.stores.sampling_session_store import async_set_sampling_session_last_activity

    for sampler_id in await _child_sampler_ids_for_heartbeat(root_session_id, request_user_data):
        try:
            await async_set_sampling_session_last_activity(sampler_id, time.time())
        except Exception as e:
            logger.warning("[session_heartbeat] child sampler activity update failed for %s: %s", sampler_id, e)


async def _update_session_heartbeat_store(session_id: str) -> None:
    async_update = getattr(session_heartbeat_store, "async_update", None)
    if callable(async_update):
        await async_update(session_id)
        return
    update = getattr(session_heartbeat_store, "update", None)
    if callable(update):
        update(session_id)
        return
    raise AttributeError("session_heartbeat_store has neither async_update nor update")


@router.post("/session_heartbeat")
async def session_heartbeat(
    request: SessionHeartbeatRequest,
    http_request: Request,
) -> SessionHeartbeatResponse:
    """Keep session alive.

    Accepts heartbeat and returns acknowledgment. Session validation not implemented.
    """
    await _update_session_heartbeat_store(request.session_id)
    try:
        from mint_server.backend.stores.sampling_session_store import async_set_sampling_session_last_activity

        await async_set_sampling_session_last_activity(request.session_id, time.time())
    except Exception as e:
        logger.warning("[session_heartbeat] sampling session activity update failed for %s: %s", request.session_id, e)
    await _touch_child_sampler_sessions(request.session_id, _get_user_data(http_request))
    return SessionHeartbeatResponse()


@router.post("/telemetry")
async def send_telemetry(request: TelemetryRequest) -> TelemetryResponse:
    """Accept telemetry data from tinker client.

    Silently accepts and discards telemetry data.
    """
    return TelemetryResponse(status="accepted")
