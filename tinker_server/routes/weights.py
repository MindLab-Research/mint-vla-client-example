"""Weight and state routes for saving/loading model weights and checkpoints.

Endpoints:
- POST /save_weights: Save LoRA weights for sampling (Tinker SDK: save_weights_for_sampler)
- POST /save_state: Save full checkpoint (LoRA + optimizer + metadata) for training resume
- POST /load_state: Load checkpoint to resume training
- GET /training_runs/{model_id}/checkpoints: List checkpoints for model
- DELETE /training_runs/{model_id}/checkpoints/{checkpoint_id}: Delete checkpoint
- GET /training_runs/{model_id}/checkpoints/{checkpoint_id}/archive: Download as tar.gz
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..backend.future_store import future_store
from ..models.types import (
    CheckpointInfo,
    CheckpointsListResponse,
    LoadStateRequest,
    SaveStateRequest,
    UntypedAPIFuture,
)
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

# Checkpoint directory (shared filesystem required for distributed deployments)
CHECKPOINTS_DIR = os.environ.get("TINKER_CHECKPOINT_DIR", "./checkpoints")


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state."""
    return getattr(request.state, "user_data", None)


def _get_user_id(request: Request) -> str | None:
    """Extract user_id from request state."""
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("user_id")
    return None


def _get_webhook_url(request: Request) -> str | None:
    """Extract webhook_url from request state."""
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("webhook_url")
    return None


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
# POST /save_weights - async (for sampling, Tinker SDK: save_weights_for_sampler)
# =============================================================================


@router.post("/save_weights", response_model=UntypedAPIFuture)
async def save_weights(
    request: SaveStateRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save LoRA weights for sampling.

    This endpoint saves weights and registers them for multi-LoRA sampling.
    Maps to Tinker SDK: save_weights_for_sampler() / save_weights_and_get_sampling_client()
    """
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)
    # Reuse _do_save_state - both endpoints save weights and register for sampling
    background_tasks.add_task(_do_save_state, request_id, session, request, user_id, webhook_url)
    return UntypedAPIFuture(request_id=request_id)


# =============================================================================
# POST /save_state - async (full checkpoint for training resume)
# =============================================================================


@router.post("/save_state", response_model=UntypedAPIFuture)
async def save_state(
    request: SaveStateRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save full model state to checkpoint (LoRA + optimizer + metadata).

    This endpoint saves weights AND optimizer state for resuming training.
    Maps to Tinker SDK: save_state()
    """
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)
    background_tasks.add_task(_do_save_state, request_id, session, request, user_id, webhook_url)
    return UntypedAPIFuture(request_id=request_id)


async def _do_save_state(
    request_id: str,
    session,
    request: SaveStateRequest,
    user_id: str | None = None,
    webhook_url: str | None = None,
) -> None:
    """Background task to save state.

    Also registers the model for sampling via multi-LoRA engine, enabling
    subsequent asample calls with the same model_id.
    """
    import json

    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        # Build save path
        checkpoint_name = request.path or f"checkpoint-{session.current_step}"
        save_path = os.path.join(CHECKPOINTS_DIR, session.model_id, checkpoint_name)

        logger.info(f"[{session.model_id}] Saving state to: {save_path}")

        # Call training engine to save full checkpoint
        # Returns dict with path, state_dict, and peft_config
        result = await training_engine.save_weights(session, save_path)

        # Save ownership metadata (for user-scoped checkpoint API)
        metadata = {
            "owner_id": user_id,
            "model_id": session.model_id,
            "model_name": session.base_model,
            "checkpoint_name": checkpoint_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "step": session.current_step,
            "type": "training",  # vs "inference" for sampler-only weights
        }
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Register model for sampling via multi-LoRA engine (Tinker SDK compatibility)
        # This allows asample to work with model_id after save_weights
        state_dict = result.get("state_dict")
        peft_config = result.get("peft_config")

        # Try to register model for sampling via multi-LoRA engine (Tinker SDK compatibility)
        # This allows asample to work with model_id after save_weights
        # Note: Registration may fail if resources unavailable (e.g., MoE needs separate GPUs)
        sampling_registered = False
        if inference_manager is not None and state_dict is not None and peft_config is not None:
            base_model = session.base_model
            if base_model is None:
                logger.warning(f"[{session.model_id}] Cannot register for sampling: base_model not set")
            else:
                try:
                    # Get or create engine for this model (dynamically creates vLLM actor if needed)
                    multi_lora_engine = await inference_manager.get_engine_for_model(base_model)

                    # Check if already registered (update existing)
                    existing_lora_id = await multi_lora_engine.registry.get_lora_id(session.model_id)
                    if existing_lora_id is not None:
                        await multi_lora_engine.remove_session(session.model_id)

                    # Register with model_id as session_id for SDK compatibility
                    await multi_lora_engine.add_lora_for_session(
                        sampling_session_id=session.model_id,
                        state_dict=state_dict,
                        peft_config=peft_config,
                    )
                    try:
                        inference_manager.register_multi_lora_session(
                            session.model_id, base_model=base_model
                        )
                    except ValueError:
                        pass  # Session already registered
                    sampling_registered = True
                    logger.info(f"[{session.model_id}] Registered for multi-LoRA sampling (model={base_model})")
                except Exception as reg_err:
                    # Log but don't fail save_weights - sampling registration is optional
                    logger.warning(f"[{session.model_id}] Could not register for sampling: {reg_err}")
        else:
            logger.warning(f"[{session.model_id}] Cannot register for sampling: "
                          f"inference_manager={inference_manager is not None}, "
                          f"state_dict={state_dict is not None}, "
                          f"peft_config={peft_config is not None}")

        # Build tinker:// path for response
        tinker_path = _to_tinker_path(session.model_id, checkpoint_name)

        # Include state_dict metadata in response for verification (e.g., checking MLP modules)
        # Keys are JSON-serializable, tensors are not
        state_dict_keys = list(state_dict.keys()) if state_dict else []

        future_store.resolve(request_id, {
            "path": tinker_path,
            "type": "save_weights",
            "state_dict_keys": state_dict_keys,  # List of parameter names for verification
            "peft_config": peft_config,
            "sampling_registered": sampling_registered,
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
                result={"checkpoint_path": tinker_path, "step": session.current_step},
            )

    except Exception as e:
        logger.error(f"[save_state] Failed: {e}", exc_info=True)
        future_store.fail(request_id, str(e))

        # 发送 failed 状态
        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_FAILED,
                user_id=user_id,
                session_id=session.model_id,
                task_name=f"Training {session.base_model}",
                task_type="training",
                model_name=session.base_model,
                error=str(e),
            )


# =============================================================================
# POST /load_state - async
# =============================================================================


@router.post("/load_state", response_model=UntypedAPIFuture)
async def load_state(
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
    """Download checkpoint as tar.gz archive.

    Uses subprocess tar+gzip for true streaming without loading into memory.
    Essential for large checkpoints (7GB+).
    """
    import subprocess

    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    ckpt_path = os.path.join(CHECKPOINTS_DIR, model_id, checkpoint_id)

    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    def stream_tar_gz():
        """Stream tar.gz via subprocess to avoid memory explosion."""
        # Run tar in parent directory, archive the checkpoint_id folder
        parent_dir = os.path.dirname(ckpt_path)
        proc = subprocess.Popen(
            ["tar", "czf", "-", checkpoint_id],
            cwd=parent_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            while chunk := proc.stdout.read(65536):
                yield chunk
        finally:
            proc.stdout.close()
            proc.wait()

    filename = f"{model_id}_{checkpoint_id}.tar.gz"
    logger.info(f"[{model_id}] Streaming checkpoint archive: {checkpoint_id}")

    return StreamingResponse(
        stream_tar_gz(),
        media_type="application/gzip",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )
