"""Service routes for session management.

Endpoints:
- GET /healthz: Health check
- POST /create_session: Create a new session
- POST /create_sampling_session: Create a sampling session
"""

import uuid

from fastapi import APIRouter

from ..models.types import (
    CreateSamplingSessionRequest,
    CreateSamplingSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
)

router = APIRouter()

# In-memory session storage
sessions: dict[str, dict] = {}
sampling_sessions: dict[str, str] = {}  # sampling_session_id -> base_model


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
    """Create a sampling session for text generation.

    For MVP, we use a pre-loaded model; the requested model is stored
    for future multi-model support.
    """
    sampling_session_id = str(uuid.uuid4())
    # Store the requested model for future use
    sampling_sessions[sampling_session_id] = (
        request.base_model or "Qwen/Qwen2.5-7B-Instruct"
    )
    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)
