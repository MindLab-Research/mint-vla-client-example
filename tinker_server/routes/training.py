"""Training routes for model training.

Endpoints:
- POST /create_model: Create a training model
- POST /forward_backward: Forward + backward pass
- POST /optim_step: Optimizer update
- GET /models: List training models
- DELETE /models/{model_id}: Delete a model
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..backend.future_store import future_store
from ..models.types import (
    CreateModelRequest,
    CreateModelResponse,
    ForwardBackwardRequest,
    OptimStepRequest,
    UntypedAPIFuture,
)

if TYPE_CHECKING:
    from ..backend.training_session_manager import TrainingSessionManager
    from ..backend.verl_training import VerlTrainingEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Global references (set by app lifespan)
training_manager: TrainingSessionManager | None = None
training_engine: VerlTrainingEngine | None = None


def _generate_model_id(session_id: str, model_seq_id: int) -> str:
    """Generate unique model_id from session_id and model_seq_id."""
    return f"{session_id}_{model_seq_id}"


# =============================================================================
# create_model - async
# =============================================================================


@router.post("/create_model", response_model=UntypedAPIFuture)
async def create_model(
    request: CreateModelRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Create a new training model with LoRA."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    request_id = future_store.create()
    background_tasks.add_task(_do_create_model, request_id, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_create_model(request_id: str, request: CreateModelRequest) -> None:
    """Background task to create training model."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        model_id = _generate_model_id(request.session_id, request.model_seq_id)

        # Check if model already exists (from failed previous attempt)
        existing = training_manager.get_session(model_id)
        if existing is not None:
            # Clean up stale session and retry
            logger.warning(f"[{model_id}] Cleaning up stale session from previous attempt")
            await training_engine.shutdown_session(existing)
            training_manager.delete_session(model_id)

        # Create session metadata first
        session = training_manager.create_session(
            model_id=model_id,
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            user_metadata=request.user_metadata,
        )

        # Create Ray actor - if this fails, session will be cleaned up in except block
        await training_engine.create_training_session(session)

        response = CreateModelResponse(
            request_id=request_id,
            model_id=model_id,
            type="create_model",
        )
        future_store.resolve(request_id, response.model_dump())

    except Exception as e:
        logger.exception(f"[create_model] Failed: {e}")
        # Clean up session if it was created
        model_id = _generate_model_id(request.session_id, request.model_seq_id)
        if training_manager and training_manager.get_session(model_id):
            training_manager.delete_session(model_id)
        future_store.fail(request_id, str(e))


# =============================================================================
# forward_backward - async
# =============================================================================


@router.post("/forward_backward", response_model=UntypedAPIFuture)
async def forward_backward(
    request: ForwardBackwardRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Perform forward + backward pass on training data."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    request_id = future_store.create()
    background_tasks.add_task(_do_forward_backward, request_id, session, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_forward_backward(
    request_id: str, session, request: ForwardBackwardRequest
) -> None:
    """Background task for forward_backward."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        result = await training_engine.forward_backward(session, request)
        future_store.resolve(request_id, result)

    except Exception as e:
        logger.exception(f"[forward_backward] Failed: {e}")
        future_store.fail(request_id, str(e))


# =============================================================================
# optim_step - async
# =============================================================================


@router.post("/optim_step", response_model=UntypedAPIFuture)
async def optim_step(
    request: OptimStepRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Perform optimizer step to update weights."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    request_id = future_store.create()
    background_tasks.add_task(_do_optim_step, request_id, session, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_optim_step(request_id: str, session, request: OptimStepRequest) -> None:
    """Background task for optim_step."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        result = await training_engine.optim_step(session, request)
        future_store.resolve(request_id, result)

    except Exception as e:
        logger.exception(f"[optim_step] Failed: {e}")
        future_store.fail(request_id, str(e))


# =============================================================================
# Model info endpoints
# =============================================================================


@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a training model."""
    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training manager not initialized")

    session = training_manager.get_session(model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found"
        )

    return {
        "model_id": session.model_id,
        "session_id": session.session_id,
        "model_seq_id": session.model_seq_id,
        "base_model": session.base_model,
        "lora_config": session.lora_config.model_dump() if session.lora_config else None,
        "user_metadata": session.user_metadata,
        "created_at": session.created_at,
        "current_step": session.current_step,
        "is_active": session.is_active,
    }


@router.get("/models")
async def list_models():
    """List all training models."""
    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training manager not initialized")

    sessions = training_manager.list_sessions()
    return {
        "models": [
            {
                "model_id": s.model_id,
                "session_id": s.session_id,
                "model_seq_id": s.model_seq_id,
                "base_model": s.base_model,
                "created_at": s.created_at,
                "current_step": s.current_step,
                "is_active": s.is_active,
            }
            for s in sessions
        ],
        "total": len(sessions),
    }


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a training model and release resources."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found"
        )

    await training_engine.shutdown_session(session)
    training_manager.delete_session(model_id)

    return {"model_id": model_id, "status": "deleted"}
