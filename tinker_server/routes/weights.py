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
import uuid
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


def _resolve_mint_path(mint_uri: str) -> str:
    """Convert path identifier to filesystem path.

    Args:
        mint_uri: One of:
            - checkpoint_id (ckpt_xxx): Search in all checkpoint directories
            - mint://{model_id}/{name}: Legacy format -> /checkpoints/{model_id}/{name}
            - file:///path: Strip prefix
            - Absolute path: Return as-is

    Returns:
        Filesystem path.
    """
    import json

    # New format: checkpoint_id (ckpt_xxx)
    if mint_uri.startswith("ckpt_"):
        # Search for checkpoint by ID in metadata
        for top_level in os.listdir(CHECKPOINTS_DIR):
            top_path = os.path.join(CHECKPOINTS_DIR, top_level)
            if not os.path.isdir(top_path):
                continue
            for sub_dir in os.listdir(top_path):
                sub_path = os.path.join(top_path, sub_dir)
                if not os.path.isdir(sub_path):
                    continue
                metadata_path = os.path.join(sub_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        if metadata.get("checkpoint_id") == mint_uri:
                            return sub_path
                    except (json.JSONDecodeError, OSError):
                        pass
        # Not found - return as-is (will fail later)
        return mint_uri

    # tinker:// or mint:// format: {scheme}://{model_id}/checkpoint-100
    if mint_uri.startswith("tinker://"):
        path_part = mint_uri[len("tinker://"):]
        return os.path.join(CHECKPOINTS_DIR, path_part)
    if mint_uri.startswith("mint://"):
        path_part = mint_uri[len("mint://"):]
        return os.path.join(CHECKPOINTS_DIR, path_part)

    # File URI
    if mint_uri.startswith("file://"):
        return mint_uri[7:]

    # Absolute path
    return mint_uri


def _to_mint_path(model_id: str, checkpoint_name: str) -> str:
    """Convert to mint://{model_id}/ URI (legacy format)."""
    return f"mint://{model_id}/{checkpoint_name}"


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

    Uses new storage schema: /checkpoints/{user_id}/{checkpoint_id}/
    Also registers the model for sampling via multi-LoRA engine.
    """
    import json

    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        # Generate unique checkpoint_id (spec format: ckpt_xxx)
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"

        # Use user-based directory (fallback to "anonymous" if no user)
        owner_dir = user_id or "anonymous"
        save_path = os.path.join(CHECKPOINTS_DIR, owner_dir, checkpoint_id)

        logger.info(f"[{session.model_id}] Saving state to: {save_path}")

        # Call training engine to save full checkpoint
        # Returns dict with path, state_dict, and peft_config
        result = await training_engine.save_weights(session, save_path)

        # Save ownership metadata (for user-scoped checkpoint API)
        # Note: Directory is created by Ray Worker on GPU node, but shared filesystem
        # sync may not be complete yet. Create directory on API server to ensure it exists.
        os.makedirs(save_path, exist_ok=True)

        metadata = {
            "checkpoint_id": checkpoint_id,
            "owner_id": user_id,
            "model_id": session.model_id,
            "model_name": session.base_model,
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

        # Include state_dict metadata in response for verification (e.g., checking MLP modules)
        # Keys are JSON-serializable, tensors are not
        state_dict_keys = list(state_dict.keys()) if state_dict else []

        future_store.resolve(request_id, {
            "checkpoint_id": checkpoint_id,
            "path": save_path,  # Filesystem path for debugging
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
                result={"checkpoint_id": checkpoint_id, "step": session.current_step},
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
        load_path = _resolve_mint_path(request.path)

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
async def list_checkpoints(model_id: str, request: Request) -> CheckpointsListResponse:
    """List all checkpoints for a model.

    Works for both active training sessions and saved checkpoints.
    Ownership verified via metadata.json (admin can access all).
    """
    user_id = _get_user_id(request)
    checkpoints_path = os.path.join(CHECKPOINTS_DIR, model_id)

    if not os.path.exists(checkpoints_path):
        raise HTTPException(status_code=404, detail=f"No checkpoints found for model '{model_id}'")

    checkpoints = []
    for name in os.listdir(checkpoints_path):
        ckpt_path = os.path.join(checkpoints_path, name)
        if os.path.isdir(ckpt_path):
            # Check ownership via metadata.json (admin can access all, others only their own)
            if user_id != "admin":
                metadata_path = os.path.join(ckpt_path, "metadata.json")
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    if metadata.get("owner_id") != user_id:
                        continue  # Skip checkpoints not owned by user
                else:
                    continue  # Skip checkpoints without metadata (legacy)

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
                path=_to_mint_path(model_id, name),
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
async def delete_checkpoint(model_id: str, checkpoint_id: str, request: Request):
    """Delete a specific checkpoint.

    Ownership verified via metadata.json (admin can delete all).
    """
    user_id = _get_user_id(request)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, model_id, checkpoint_id)

    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    # Check ownership (admin can delete all, others only their own)
    if user_id != "admin":
        metadata_path = os.path.join(ckpt_path, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            if metadata.get("owner_id") != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        else:
            # No metadata.json = legacy checkpoint = deny access for non-admin
            raise HTTPException(status_code=403, detail="Access denied")

    shutil.rmtree(ckpt_path)

    logger.info(f"[{model_id}] Deleted checkpoint: {checkpoint_id}")

    return {"status": "deleted", "checkpoint_id": checkpoint_id}


# =============================================================================
# GET /training_runs/{model_id}/checkpoints/{checkpoint_id}/archive
# =============================================================================


@router.get("/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive")
async def download_checkpoint_archive(model_id: str, checkpoint_id: str, request: Request):
    """Download checkpoint as tar.gz archive.

    Uses subprocess tar+gzip for true streaming without loading into memory.
    Essential for large checkpoints (7GB+).
    Ownership verified via metadata.json (admin can download all).
    """
    import subprocess

    user_id = _get_user_id(request)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, model_id, checkpoint_id)

    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    # Check ownership (admin can download all, others only their own)
    if user_id != "admin":
        metadata_path = os.path.join(ckpt_path, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            if metadata.get("owner_id") != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        else:
            # No metadata.json = legacy checkpoint = deny access for non-admin
            raise HTTPException(status_code=403, detail="Access denied")

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
