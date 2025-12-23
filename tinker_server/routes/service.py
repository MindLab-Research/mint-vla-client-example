"""Service routes for session management.

Endpoints:
- GET /healthz: Health check
- POST /create_session: Create a new session
- POST /create_sampling_session: Create a sampling session with dedicated engine
- POST /session_heartbeat: Keep session alive
- POST /telemetry: Accept telemetry data (discarded)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from safetensors.torch import load_file

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

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager

router = APIRouter()

# In-memory session storage
sessions: dict[str, dict] = {}
sampling_sessions: dict[str, str] = {}  # sampling_session_id -> base_model

# Global session manager reference (set by app lifespan)
session_manager: SessionManager | None = None


@router.get("/healthz")
async def healthz() -> dict:
    """Health check endpoint."""
    return {"status": "ready"}


@router.get("/get_server_capabilities")
async def get_server_capabilities() -> dict:
    """Return server capabilities for tinker client."""
    from ..backend.model_registry import list_supported_models

    supported = list_supported_models()
    return {
        "supported_models": [{"model_name": m} for m in supported],
    }


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

    # Determine base_model from request or infer from model_path
    base_model = request.base_model
    if not base_model and request.model_path:
        # Try to infer base_model from adapter_config.json
        adapter_path = _resolve_model_path(request.model_path)
        base_model = _infer_base_model_from_adapter(adapter_path)

    if not base_model:
        raise HTTPException(
            status_code=422,
            detail="base_model is required. Provide base_model or model_path with adapter_config.json containing base_model_name_or_path.",
        )

    # Get or create engine for this model (dynamically creates vLLM actor if needed)
    multi_lora_engine = await session_manager.get_engine_for_model(base_model)

    if request.model_path:
        # Load LoRA weights from path into multi-LoRA engine
        # Resolve path (file://, tinker://localhost, or absolute path)
        adapter_path = _resolve_model_path(request.model_path)

        # Load adapter weights and config from disk
        state_dict, peft_config = _load_adapter_from_path(adapter_path, request.lora_rank)

        # Add LoRA to engine and register session
        await multi_lora_engine.add_lora_for_session(
            sampling_session_id=sampling_session_id,
            state_dict=state_dict,
            peft_config=peft_config,
        )
        session_manager.register_multi_lora_session(
            session_id=sampling_session_id,
            base_model=base_model,
            lora_rank=request.lora_rank,
        )
    else:
        # Base model (no LoRA): register session directly
        session_manager.register_base_model_session(sampling_session_id, base_model=base_model)

    # Store metadata
    sampling_sessions[sampling_session_id] = base_model

    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


def _resolve_model_path(model_path: str) -> str:
    """Resolve model_path URI to filesystem path.

    Args:
        model_path: URI like file:///path, tinker://localhost/path, or absolute path.

    Returns:
        Absolute filesystem path to adapter directory.
    """
    if model_path.startswith("file://"):
        return model_path[7:]  # Strip file:// prefix
    elif model_path.startswith("tinker://localhost"):
        # Local server tinker:// format: tinker://localhost/<absolute_path>
        return model_path[len("tinker://localhost"):]
    elif model_path.startswith("tinker://"):
        # Cloud tinker:// paths not supported locally
        raise HTTPException(
            status_code=400,
            detail=f"Cloud tinker:// paths not supported locally: {model_path}",
        )
    else:
        # Assume absolute path
        return model_path


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

    from safetensors.torch import load_file

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


class KillVLLMRequest(BaseModel):
    """Request to kill a vLLM actor."""

    model_name: str | None = None  # If None, kills default actor; if set, kills that model's actor


@router.post("/kill_vllm")
async def kill_vllm(request: KillVLLMRequest | None = None) -> dict:
    """Kill a vLLM actor.

    Use this to force a full restart of the vLLM engine.
    The next request that needs vLLM will create a new actor (~80s init).

    Args:
        model_name: If provided, kills the vLLM actor for that specific model.
                   If None, kills the default actor.

    This is useful when:
    - You need to reload the base model
    - vLLM is in a bad state
    - You want to free GPU memory
    """
    model_name = request.model_name if request else None

    if model_name:
        # Kill specific model's vLLM actor
        if session_manager is None:
            return {"killed": False, "message": "Session manager not initialized"}

        multi_model_manager = session_manager.get_multi_model_manager()
        if multi_model_manager is None:
            return {"killed": False, "message": "Multi-model manager not initialized"}

        engine = multi_model_manager.get_engine_if_exists(model_name)
        if engine is None:
            return {"killed": False, "message": f"No vLLM engine found for model: {model_name}"}

        try:
            await engine.shutdown(kill_actor=True)
            # Remove from manager's cache
            if model_name in multi_model_manager._engines:
                del multi_model_manager._engines[model_name]
            return {"killed": True, "message": f"vLLM actor for {model_name} killed"}
        except Exception as e:
            return {"killed": False, "message": f"Error killing actor: {e}"}
    else:
        # Kill default actor
        from ..backend.multi_lora_engine import kill_persistent_vllm_actor

        killed = kill_persistent_vllm_actor()
        return {"killed": killed, "message": "vLLM actor killed" if killed else "No vLLM actor found"}


@router.get("/vllm_status")
async def vllm_status() -> dict:
    """Check if persistent vLLM actor exists.

    Returns:
        alive: True if actor exists and is alive
        actor_name: The well-known actor name
    """
    from ..backend.multi_lora_engine import (
        PERSISTENT_VLLM_ACTOR_NAME,
        check_persistent_vllm_actor,
    )

    alive = check_persistent_vllm_actor()
    return {"alive": alive, "actor_name": PERSISTENT_VLLM_ACTOR_NAME}


class KillMegatronRequest(BaseModel):
    """Request to kill Megatron actor(s)."""

    base_model: str | None = None  # Kill specific model's actor, or all if None


@router.post("/kill_megatron")
async def kill_megatron(request: KillMegatronRequest | None = None) -> dict:
    """Kill Megatron training actor(s).

    Args:
        base_model: If provided, kill actor for this specific model.
                   If None/omitted, kill ALL Megatron actors.

    Use this to force a full restart of the Megatron worker group.
    The next training request will create a new actor.
    """
    from ..backend.megatron_distributed import kill_megatron_actor

    base_model = request.base_model if request else None
    killed = kill_megatron_actor(base_model)

    if base_model:
        msg = f"Killed Megatron actor for {base_model}" if killed else f"No Megatron actor found for {base_model}"
    else:
        msg = "All Megatron actors killed" if killed else "No Megatron actors found"

    return {"killed": killed, "message": msg}



@router.get("/megatron_status")
async def megatron_status(base_model: str | None = None) -> dict:
    """Check if Megatron actor(s) exist.

    Args:
        base_model: If provided, check for this specific model's actor.
                   If None, check for ANY running Megatron actor.

    Returns:
        alive: True if matching actor exists and is alive
        actors: List of running Megatron actors from resource pool
    """
    from ..backend.megatron_distributed import is_megatron_actor_running
    from ..backend.resource_pool import get_resource_pool, ActorType

    alive = is_megatron_actor_running(base_model)

    # Get list of all Megatron actors
    resource_pool = get_resource_pool()
    actors = [
        {"name": e.actor_name, "gpus": e.num_gpus, "base_model": e.base_model}
        for e in resource_pool.list_actors()
        if e.actor_type == ActorType.MEGATRON
    ]

    return {"alive": alive, "actors": actors, "query_base_model": base_model}


@router.get("/resource_pool")
async def resource_pool_status() -> dict:
    """Get unified resource pool status.

    Returns:
        actors: List of all tracked actors with LRU info
        total_gpus: Total GPUs used
        min_actor_age: Minimum actor age before eviction eligible
    """
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    return {
        "actors": pool.list_actors(),
        "total_gpus_used": pool.total_gpus_used(),
        "min_actor_age": pool.MIN_ACTOR_AGE,
    }


@router.post("/clear_resource_pool")
async def clear_resource_pool() -> dict:
    """Clear all entries from the resource pool.

    Used for debugging when pool has stale entries after actors are killed externally.
    Does NOT kill actors - just clears the tracking entries.
    """
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    count = pool.clear(kill_actors=False)
    return {"cleared": count, "message": f"Cleared {count} entries from resource pool"}


@router.post("/kill_all_actors")
async def kill_all_actors() -> dict:
    """Kill all actors and clear the resource pool.

    Use this to free all GPUs when actors are stuck or not being evicted properly.
    """
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    count = pool.clear(kill_actors=True)
    return {"killed": count, "message": f"Killed {count} actors and freed GPUs"}
