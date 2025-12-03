"""Service routes for session management.

Endpoints:
- GET /healthz: Health check
- POST /create_session: Create a new session
- POST /create_sampling_session: Create a sampling session with dedicated engine
- POST /telemetry/send: Accept telemetry data
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
    """Create a sampling session with dedicated inference engine.

    Each sampling session spawns a new VerlInferenceEngine with its own
    LoRA adapter, enabling session isolation.
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    sampling_session_id = str(uuid.uuid4())

    # Spawn dedicated engine for this session
    await session_manager.create_session(
        session_id=sampling_session_id,
        lora_rank=request.lora_rank,
    )

    # Store metadata
    sampling_sessions[sampling_session_id] = (
        request.base_model or "Qwen/Qwen2.5-7B-Instruct"
    )

    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


@router.post("/telemetry/send")
async def send_telemetry(request: TelemetryRequest) -> TelemetryResponse:
    """Accept telemetry data from tinker client.

    Silently accepts and discards telemetry data.
    """
    return TelemetryResponse(status="accepted")
