"""Service routes for session management.

Endpoints:
- GET /healthz: Health check
- POST /create_session: Create a new session
- POST /create_sampling_session: Create a sampling session with dedicated engine
- POST /session_heartbeat: Keep session alive
- POST /telemetry: Accept telemetry data (discarded)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..backend.session_heartbeat_store import session_heartbeat_store
from ..model_access_control import can_access_model, get_access_denied_error
from ..models.types import (
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
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

# In-memory session storage
sessions: dict[str, dict] = {}
sampling_sessions: dict[str, str] = {}  # sampling_session_id -> base_model

# Global session manager reference (set by app lifespan)
session_manager: SessionManager | None = None


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state (set by auth middleware)."""
    return getattr(request.state, "user_data", None)


def _get_user_id(request: Request) -> str | None:
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("user_id")
    return None


@router.get("/healthz")
async def healthz() -> dict:
    """Health check endpoint."""
    return {"status": "ready"}


@router.get("/get_server_capabilities")
async def get_server_capabilities(http_request: Request) -> dict:
    """Return server capabilities for tinker client."""
    from ..backend.model_registry import get_model_config, list_supported_models
    from ..gateway import get_gateway_config, get_upstream_capabilities, upstream_for_model

    supported_local = list_supported_models()
    cfg = get_gateway_config()

    if cfg is None or not cfg.model_to_upstream:
        supported = supported_local
        return {
            "supported_models": [
                {
                    "model_name": m,
                    "max_context_length": get_model_config(m).max_model_len,
                }
                for m in supported
            ],
        }

    incoming_headers = dict(http_request.headers)
    remote_models = list(cfg.model_to_upstream.keys())

    # Fetch capabilities once per upstream alias that has at least one routed model.
    alias_to_caps: dict[str, dict[str, int]] = {}
    unavailable_aliases: set[str] = set()
    for alias in set(cfg.model_to_upstream.values()):
        upstream = cfg.upstreams.get(alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {alias!r}")
        try:
            alias_to_caps[alias] = await get_upstream_capabilities(
                upstream=upstream, incoming_headers=incoming_headers
            )
        except Exception:
            logger.exception("Upstream capabilities unavailable: %s", alias)
            unavailable_aliases.add(alias)

    merged: list[dict] = []
    seen: set[str] = set()
    for m in supported_local + remote_models:
        if m in seen:
            continue
        seen.add(m)

        if m in cfg.model_to_upstream:
            upstream = upstream_for_model(m)
            if upstream is None:
                continue
            if upstream.alias in unavailable_aliases:
                continue
            caps = alias_to_caps.get(upstream.alias, {})
            if m not in caps:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: model {m!r} not present in upstream {upstream.alias!r} capabilities",
                )
            max_len = int(caps[m])
        else:
            max_len = int(get_model_config(m).max_model_len)

        merged.append({"model_name": m, "max_context_length": max_len})

    return {
        "supported_models": merged,
    }


@router.get("/server_info")
async def server_info() -> dict:
    return get_server_info()


@router.post("/create_session")
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new session.

    Sessions are used to group related operations together.
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "tags": request.tags,
        "metadata": request.user_metadata,
    }
    return CreateSessionResponse(session_id=session_id)


@router.post("/create_sampling_session")
async def create_sampling_session(
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
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    sampling_session_id = str(uuid.uuid4())
    user_id = _get_user_id(http_request)

    # Determine base_model from request or infer from model_path
    base_model = request.base_model
    if not base_model and request.model_path:
        # Try to infer base_model from adapter_config.json
        adapter_path = _resolve_model_path(request.model_path, user_id=user_id)
        base_model = _infer_base_model_from_adapter(adapter_path)

    if not base_model:
        raise HTTPException(
            status_code=422,
            detail="base_model is required. Provide base_model or model_path with adapter_config.json containing base_model_name_or_path.",
        )

    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(base_model, user_data):
        raise HTTPException(
            status_code=403,
            detail=get_access_denied_error(base_model)
        )

    # Gateway forwarding: if base_model is configured as remote, proxy to upstream and
    # return upstream sampling_session_id (tracking it for subsequent asample routing).
    from ..gateway import forward_json, register_remote_sampling_session, upstream_for_model

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

        register_remote_sampling_session(
            sampling_session_id=sampling_session_id_remote,
            upstream_alias=upstream.alias,
            base_model=base_model,
        )
        return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id_remote)

    # Get or create engine for this model (dynamically creates vLLM actor if needed)
    # Do not block on vLLM cold-start here (can exceed reverse-proxy timeouts).
    # Warm vLLM in the background; /asample work will await readiness.
    async def _warm_engine() -> None:
        try:
            await session_manager.get_engine_for_model(base_model)
        except Exception as e:
            logger.warning(f"[create_sampling_session] warm engine failed: model={base_model} err={e}")

    asyncio.create_task(_warm_engine())

    if request.model_path:
        # Resolve adapter directory (file://, mint://, absolute path).
        adapter_path = _resolve_model_path(request.model_path, user_id=user_id)

        # Fast validation: ensure weights exist; loading happens on first /asample.
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            raise HTTPException(status_code=400, detail=f"Adapter weights not found: {weights_path}")

        # Register session now; LoRA will be loaded lazily on first /asample.
        session_manager.register_multi_lora_session(
            session_id=sampling_session_id,
            base_model=base_model,
            lora_rank=request.lora_rank,
            adapter_path=adapter_path,
            lora_loaded=False,
        )
    else:
        # Base model (no LoRA): register session directly
        session_manager.register_base_model_session(sampling_session_id, base_model=base_model)

    # Store metadata
    sampling_sessions[sampling_session_id] = base_model

    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


def _resolve_model_path(model_path: str, *, user_id: str | None) -> str:
    """Resolve model_path URI to filesystem path.

    Args:
        model_path: URI like file:///path, mint://{uuid}/..., or absolute path.

    Returns:
        Absolute filesystem path to adapter directory.
    """
    from ..checkpoints import get_checkpoints_dir, resolve_checkpoint_uri

    checkpoint_dir = get_checkpoints_dir()
    resolved = resolve_checkpoint_uri(model_path, checkpoint_dir, user_id=user_id)
    if user_id and user_id != "admin":
        resolved_real = os.path.realpath(resolved)
        checkpoints_real = os.path.realpath(checkpoint_dir)
        allowed_real = os.path.realpath(os.path.join(checkpoint_dir, user_id))
        if resolved_real.startswith(checkpoints_real + os.sep) and not resolved_real.startswith(
            allowed_real + os.sep
        ):
            raise HTTPException(status_code=403, detail="Access denied")
    return resolved


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
    session_heartbeat_store.update(request.session_id)
    return SessionHeartbeatResponse()


@router.post("/telemetry")
async def send_telemetry(request: TelemetryRequest) -> TelemetryResponse:
    """Accept telemetry data from tinker client.

    Silently accepts and discards telemetry data.
    """
    return TelemetryResponse(status="accepted")


# =============================================================================
# Admin endpoints for vLLM management
# =============================================================================



def _require_admin(request: Request) -> None:
    """Raise 403 if not admin user."""
    from ..config import config as server_config
    if not server_config.auth_enabled:
        return
    user_data = getattr(request.state, "user_data", None)
    if not user_data or user_data.get("user_id") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


class KillVllmRequest(BaseModel):
    """Request to kill vLLM actor(s)."""

    model_name: str | None = None  # Kill specific model's actor, or all if None


@router.post("/kill_vllm")
async def kill_vllm(request: Request, body: KillVllmRequest | None = None) -> dict:
    """Kill vLLM inference actor(s). Admin only.

    Args:
        model_name: If provided, kill actor for this specific model.
                   If None/omitted, kill ALL vLLM actors.

    Use this to force a full restart of the vLLM engine.
    The next request that needs vLLM will create a new actor (~80s init).
    """
    _require_admin(request)
    from ..backend.multi_lora_engine import kill_persistent_vllm_actor

    model_name = body.model_name if body else None
    killed = kill_persistent_vllm_actor(model_name)

    if model_name:
        msg = f"Killed vLLM actor for {model_name}" if killed else f"No vLLM actor found for {model_name}"
    else:
        msg = "All vLLM actors killed" if killed else "No vLLM actors found"

    return {"killed": killed, "message": msg}


@router.get("/vllm_status")
async def vllm_status(request: Request, model_name: str | None = None) -> dict:
    """Check if vLLM actor(s) exist. Admin only.

    Args:
        model_name: If provided, check for this specific model's actor.
                   If None, check for ANY running vLLM actor.

    Returns:
        alive: True if matching actor exists and is alive
        actors: List of running vLLM actors from resource pool
    """
    _require_admin(request)
    from ..backend.multi_lora_engine import check_persistent_vllm_actor, list_vllm_actors

    alive = check_persistent_vllm_actor(model_name)
    actors = list_vllm_actors()

    # Augment actor entries with Ray placement group info (when available).
    #
    # Issue #82 requires verifying GPU reservation at Ray scheduling level. Resource pool
    # tracking is not sufficient evidence if it diverges from placement-group bundles.
    try:
        import ray
        from ..config import RAY_NAMESPACE
        from ..ray_utils import init_ray

        if not ray.is_initialized():
            init_ray(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True)

        for a in actors:
            if not isinstance(a, dict):
                continue
            name = a.get("name")
            if not isinstance(name, str) or not name:
                continue
            pg_name = f"{name}_pg"
            try:
                pg = ray.util.get_placement_group(pg_name)
                bundles = getattr(pg, "bundle_specs", None)
                if isinstance(bundles, list):
                    a["pg_name"] = pg_name
                    a["pg_bundle_count"] = len(bundles)
                    a["pg_total_gpus"] = sum(int(b.get("GPU", 0) or 0) for b in bundles if isinstance(b, dict))
            except Exception:
                continue
    except Exception:
        pass

    return {"alive": alive, "actors": actors, "query_model_name": model_name}


class KillMegatronRequest(BaseModel):
    """Request to kill Megatron actor(s)."""

    base_model: str | None = None  # Kill specific model's actor, or all if None


@router.post("/kill_megatron")
async def kill_megatron(request: Request, body: KillMegatronRequest | None = None) -> dict:
    """Kill Megatron training actor(s). Admin only.

    Args:
        base_model: If provided, kill actor for this specific model.
                   If None/omitted, kill ALL Megatron actors.

    Use this to force a full restart of the Megatron worker group.
    The next training request will create a new actor.
    """
    _require_admin(request)
    from ..backend.megatron_distributed import kill_megatron_actor

    base_model = body.base_model if body else None
    killed = kill_megatron_actor(base_model)

    if base_model:
        msg = f"Killed Megatron actor for {base_model}" if killed else f"No Megatron actor found for {base_model}"
    else:
        msg = "All Megatron actors killed" if killed else "No Megatron actors found"

    return {"killed": killed, "message": msg}



@router.get("/megatron_status")
async def megatron_status(request: Request, base_model: str | None = None) -> dict:
    """Check if Megatron actor(s) exist. Admin only.

    Args:
        base_model: If provided, check for this specific model's actor.
                   If None, check for ANY running Megatron actor.

    Returns:
        alive: True if matching actor exists and is alive
        actors: List of running Megatron actors from resource pool
    """
    _require_admin(request)
    from ..backend.megatron_distributed import is_megatron_actor_running
    from ..backend.resource_pool import ActorType, get_resource_pool

    alive = is_megatron_actor_running(base_model)

    # Get list of all Megatron actors
    resource_pool = get_resource_pool()
    actors = [
        {"name": e["actor_name"], "gpus": e["num_gpus"], "base_model": e["base_model"]}
        for e in resource_pool.list_actors()
        if e.get("actor_type") == ActorType.MEGATRON.value
    ]

    return {"alive": alive, "actors": actors, "query_base_model": base_model}



@router.get("/resource_pool")
async def resource_pool_status(request: Request) -> dict:
    """Get unified resource pool status. Admin only.

    Returns:
        actors: List of all tracked actors with LRU info
        total_gpus: Total GPUs used
        min_actor_age: Minimum actor age before eviction eligible
    """
    _require_admin(request)
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    return {
        "actors": pool.list_actors(),
        "total_gpus_used": pool.total_gpus_used(),
        "min_actor_age": pool.MIN_ACTOR_AGE,
    }


@router.post("/clear_resource_pool")
async def clear_resource_pool(request: Request) -> dict:
    """Clear all entries from the resource pool. Admin only.

    Used for debugging when pool has stale entries after actors are killed externally.
    Does NOT kill actors - just clears the tracking entries.
    """
    _require_admin(request)
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    count = pool.clear(kill_actors=False)
    return {"cleared": count, "message": f"Cleared {count} entries from resource pool"}


@router.post("/kill_all_actors")
async def kill_all_actors(request: Request) -> dict:
    """Kill all actors and clear the resource pool. Admin only.

    Use this to free all GPUs when actors are stuck or not being evicted properly.
    """
    _require_admin(request)
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    count = pool.clear(kill_actors=True)
    return {"killed": count, "message": f"Killed {count} actors and freed GPUs"}
