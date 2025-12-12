"""Service routes for session management.

Endpoints:
- GET /healthz: Health check
- POST /create_session: Create a new session
- POST /create_sampling_session: Create a sampling session with dedicated engine
- POST /session_heartbeat: Keep session alive
- POST /telemetry: Accept telemetry data (discarded)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

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
    import os

    model_path = os.environ.get("TINKER_MODEL_PATH", "")
    model_name = model_path.split("/")[-1] if model_path else "unknown"

    return {
        "supported_models": [{"model_name": model_name}],
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

    # Ensure multi-LoRA engine is initialized (lazy init on first call)
    multi_lora_engine = await session_manager.ensure_multi_lora_engine()

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
            lora_rank=request.lora_rank,
        )
    else:
        # Base model (no LoRA): register session directly
        session_manager.register_base_model_session(sampling_session_id)

    # Store metadata
    sampling_sessions[sampling_session_id] = (
        request.base_model or "Qwen/Qwen2.5-7B-Instruct"
    )

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
        peft_config = {
            "r": lora_rank,
            "lora_alpha": lora_rank,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
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


@router.post("/kill_vllm")
async def kill_vllm() -> dict:
    """Kill the persistent vLLM actor.

    Use this to force a full restart of the vLLM engine.
    The next request that needs vLLM will create a new actor (~80s init).

    This is useful when:
    - You need to reload the base model
    - vLLM is in a bad state
    - You want to free GPU memory
    """
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


@router.post("/kill_megatron")
async def kill_megatron() -> dict:
    """Kill the persistent Megatron training actor.

    Use this to force a full restart of the Megatron worker group.
    The next training request will create a new actor.

    This is useful when:
    - Workers are in a bad state after a crash
    - You want to free GPU memory
    - You need to reload code changes
    """
    from ..backend.megatron_distributed import kill_megatron_actor

    killed = kill_megatron_actor()
    return {"killed": killed, "message": "Megatron actor killed" if killed else "No Megatron actor found"}


@router.get("/megatron_status")
async def megatron_status() -> dict:
    """Check if persistent Megatron actor exists.

    Returns:
        alive: True if actor exists and is alive
        actor_name: The well-known actor name
    """
    from ..backend.megatron_distributed import (
        PERSISTENT_MEGATRON_ACTOR_NAME,
        is_megatron_actor_running,
    )

    alive = is_megatron_actor_running()
    return {"alive": alive, "actor_name": PERSISTENT_MEGATRON_ACTOR_NAME}
