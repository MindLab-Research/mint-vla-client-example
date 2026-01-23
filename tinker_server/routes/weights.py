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
from ..checkpoints import CHECKPOINTS_DIR, resolve_checkpoint_path
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


def _to_mint_path(model_id: str, checkpoint_name: str) -> str:
    """Convert to mint://{model_id}/ URI (legacy format)."""
    return f"mint://{model_id}/{checkpoint_name}"


# =============================================================================
# POST /save_weights, /save_weights_for_sampler - async (SDK compatibility)
# =============================================================================


@router.post("/save_weights", response_model=UntypedAPIFuture)
@router.post("/save_weights_for_sampler", response_model=UntypedAPIFuture)  # SDK alias
async def save_weights(
    request: SaveStateRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save LoRA weights for sampling.

    Saves LoRA weights and registers them for multi-LoRA sampling.

    SDK compatibility:
    - /save_weights: Called by SDK save_state() - should save optimizer but doesn't (TODO)
    - /save_weights_for_sampler: Called by SDK save_weights_for_sampler()

    TEMPORARY WORKAROUND: Both endpoints currently save only LoRA weights, not optimizer
    state. True training resume with preserved optimizer momentum is not supported.
    See GitHub issue #67 for optimizer state saving implementation.
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
# POST /save_state - async (full checkpoint - optimizer saving NOT IMPLEMENTED)
# =============================================================================


@router.post("/save_state", response_model=UntypedAPIFuture)
async def save_state(
    request: SaveStateRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save model state to checkpoint.

    CURRENT BEHAVIOR: Saves LoRA weights only (same as /save_weights).

    INTENDED BEHAVIOR (not implemented): Should save weights + optimizer state
    for true training resume with preserved momentum. See GitHub issue #67.
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

        # Save checkpoint on worker, returns path
        abs_path = await training_engine.save_weights(session, save_path)

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

        # Register for sampling via path-based loading (avoids tensor transfer OOM)
        sampling_registered = False
        base_model = session.base_model
        if inference_manager is not None and base_model is not None:
            try:
                multi_lora_engine = await inference_manager.get_engine_for_model(base_model)

                # Remove existing registration if any
                existing_lora_id = await multi_lora_engine.registry.get_lora_id(session.model_id)
                if existing_lora_id is not None:
                    await multi_lora_engine.remove_session(session.model_id)

                # Path-based loading - vLLM loads directly from shared filesystem
                await multi_lora_engine.add_lora_for_session_from_path(
                    sampling_session_id=session.model_id,
                    lora_path=abs_path,
                )
                try:
                    inference_manager.register_multi_lora_session(
                        session.model_id, base_model=base_model
                    )
                except ValueError:
                    pass  # Already registered
                sampling_registered = True
                logger.info(f"[{session.model_id}] Registered for sampling (path={abs_path})")
            except Exception as reg_err:
                logger.warning(f"[{session.model_id}] Could not register for sampling: {reg_err}")

        future_store.resolve(request_id, {
            "checkpoint_id": checkpoint_id,
            "path": save_path,  # Filesystem path for debugging
            "type": "save_weights",
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
@router.post("/load_weights", response_model=UntypedAPIFuture)  # SDK alias
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
    # DEBUG: Log entry
    with open("/vePFS-Mindverse/share/code/load_adapter_debug.log", "a") as dbg:
        dbg.write(f"[_do_load_state] ENTRY: request_id={request_id}, model_id={session.model_id}, path={request.path}\n")
        dbg.flush()

    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        # Resolve path
        load_path = resolve_checkpoint_path(request.path)

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
