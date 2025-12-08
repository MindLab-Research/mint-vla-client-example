"""State routes for saving/loading model state (checkpointing).

Endpoints:
- POST /save_weights: Save full checkpoint (LoRA + optimizer + metadata)
- POST /load_weights: Load checkpoint
- GET /training_runs/{model_id}/checkpoints: List checkpoints for model
- DELETE /training_runs/{model_id}/checkpoints/{checkpoint_id}: Delete checkpoint
- GET /training_runs/{model_id}/checkpoints/{checkpoint_id}/archive: Download (501)
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..backend.future_store import future_store
from ..models.types import (
    CheckpointInfo,
    CheckpointsListResponse,
    LoadStateRequest,
    SaveStateRequest,
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

# Checkpoint directory (shared filesystem required for distributed deployments)
CHECKPOINTS_DIR = os.environ.get("TINKER_CHECKPOINT_DIR", "./checkpoints")


def _resolve_tinker_path(tinker_uri: str) -> str:
    """Convert tinker://local/model_id/name to filesystem path.

    Args:
        tinker_uri: URI like tinker://local/..., file://..., or absolute path.

    Returns:
        Filesystem path.
    """
    if tinker_uri.startswith("tinker://local/"):
        # tinker://local/model_id/checkpoint-100 -> ./checkpoints/model_id/checkpoint-100
        path_part = tinker_uri[len("tinker://local/"):]
        return os.path.join(CHECKPOINTS_DIR, path_part)
    elif tinker_uri.startswith("tinker://localhost"):
        # tinker://localhost/abs/path -> /abs/path
        return tinker_uri[len("tinker://localhost"):]
    elif tinker_uri.startswith("file://"):
        return tinker_uri[7:]
    return tinker_uri


def _to_tinker_path(model_id: str, checkpoint_name: str) -> str:
    """Convert to tinker://local/ URI."""
    return f"tinker://local/{model_id}/{checkpoint_name}"


# =============================================================================
# POST /save_weights - async
# =============================================================================


@router.post("/save_weights", response_model=UntypedAPIFuture)
async def save_weights(
    request: SaveStateRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Save model state to checkpoint (LoRA + optimizer + metadata)."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    request_id = future_store.create()
    background_tasks.add_task(_do_save_state, request_id, session, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_save_state(request_id: str, session, request: SaveStateRequest) -> None:
    """Background task to save state."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        # Build save path
        checkpoint_name = request.path or f"checkpoint-{session.step}"
        save_path = os.path.join(CHECKPOINTS_DIR, session.model_id, checkpoint_name)

        logger.info(f"[{session.model_id}] Saving state to: {save_path}")

        # Call training engine to save full checkpoint
        await training_engine.save_weights(session, save_path)

        # Build tinker:// path for response
        tinker_path = _to_tinker_path(session.model_id, checkpoint_name)

        future_store.resolve(request_id, {
            "path": tinker_path,
            "type": "save_weights",
        })

    except Exception as e:
        logger.error(f"[save_state] Failed: {e}", exc_info=True)
        future_store.fail(request_id, str(e))


# =============================================================================
# POST /load_weights - async
# =============================================================================


@router.post("/load_weights", response_model=UntypedAPIFuture)
async def load_weights(
    request: LoadStateRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Load model state from checkpoint."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    request_id = future_store.create()
    background_tasks.add_task(_do_load_state, request_id, session, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_load_state(request_id: str, session, request: LoadStateRequest) -> None:
    """Background task to load state."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        # Resolve path
        load_path = _resolve_tinker_path(request.path)

        logger.info(f"[{session.model_id}] Loading state from: {load_path}")

        # Call training engine to load checkpoint
        await training_engine.load_weights(session, load_path, load_optimizer=request.optimizer)

        future_store.resolve(request_id, {
            "path": request.path,
            "type": "load_weights",
        })

    except Exception as e:
        logger.error(f"[load_state] Failed: {e}", exc_info=True)
        future_store.fail(request_id, str(e))


# =============================================================================
# GET /training_runs/{model_id}/checkpoints
# =============================================================================


@router.get("/training_runs/{model_id}/checkpoints", response_model=CheckpointsListResponse)
async def list_checkpoints(model_id: str) -> CheckpointsListResponse:
    """List all checkpoints for a model."""
    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    checkpoints_path = os.path.join(CHECKPOINTS_DIR, model_id)
    checkpoints = []

    if os.path.exists(checkpoints_path):
        for name in os.listdir(checkpoints_path):
            ckpt_path = os.path.join(checkpoints_path, name)
            if os.path.isdir(ckpt_path):
                # Try to parse step from directory name
                step = None
                if name.startswith("checkpoint-"):
                    try:
                        step = int(name.split("-")[1])
                    except (IndexError, ValueError):
                        pass

                # Get creation time
                created_at = datetime.fromtimestamp(
                    os.path.getctime(ckpt_path)
                ).isoformat()

                checkpoints.append(CheckpointInfo(
                    checkpoint_id=name,
                    path=_to_tinker_path(model_id, name),
                    step=step,
                    created_at=created_at,
                ))

    # Sort by step (descending)
    checkpoints.sort(key=lambda x: x.step or 0, reverse=True)

    return CheckpointsListResponse(
        model_id=model_id,
        checkpoints=checkpoints,
    )


# =============================================================================
# DELETE /training_runs/{model_id}/checkpoints/{checkpoint_id}
# =============================================================================


@router.delete("/training_runs/{model_id}/checkpoints/{checkpoint_id}")
async def delete_checkpoint(model_id: str, checkpoint_id: str):
    """Delete a specific checkpoint."""
    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, model_id, checkpoint_id)

    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    shutil.rmtree(ckpt_path)

    logger.info(f"[{model_id}] Deleted checkpoint: {checkpoint_id}")

    return {"status": "deleted", "checkpoint_id": checkpoint_id}


# =============================================================================
# GET /training_runs/{model_id}/checkpoints/{checkpoint_id}/archive
# =============================================================================


@router.get("/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive")
async def download_checkpoint_archive(model_id: str, checkpoint_id: str):
    """Download checkpoint as archive.

    Not implemented - returns 501.
    """
    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, model_id, checkpoint_id)

    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    # TODO: Implement archive download
    # Options: 1) Create tar.gz 2) Upload to object storage 3) Return signed URL
    raise HTTPException(
        status_code=501,
        detail="Checkpoint archive download not implemented"
    )
