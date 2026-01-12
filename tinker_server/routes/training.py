"""Training routes for model training.

Endpoints:
- POST /create_model: Create a training model
- POST /create_model_from_state: Create model and load checkpoint (resume training)
- POST /forward_backward: Forward + backward pass
- POST /forward: Forward pass only (no backward), returns logprobs
- POST /optim_step: Optimizer update
- POST /save_weights_for_sampler: Save weights for inference
- POST /get_info: Get model info (tinker client compatible)
- GET /models: List training models
- GET /models/{model_id}: Get model info
- GET /models/{model_id}/tokenizer: Get tokenizer config
- DELETE /models/{model_id}: Delete a model
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..backend.future_store import future_store
from ..models.types import (
    CreateModelFromStateRequest,
    CreateModelFromStateResponse,
    CreateModelRequest,
    CreateModelResponse,
    ForwardBackwardRequest,
    ForwardRequest,
    GetInfoRequest,
    GetInfoResponse,
    ModelData,
    OptimStepRequest,
    ResetExpertBiasRequest,
    ResetExpertBiasResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    UntypedAPIFuture,
)

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager
    from ..backend.training_session_manager import TrainingSessionManager
    from ..backend.verl_training import VerlTrainingEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Global references (set by app lifespan)
training_manager: TrainingSessionManager | None = None
training_engine: VerlTrainingEngine | None = None
inference_manager: SessionManager | None = None  # For ephemeral flow


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
            backend=session.backend,  # "megatron" or "peft"
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
# create_model_from_state - async (composes create_model + load_state)
# =============================================================================

# Checkpoint directory (shared filesystem required for distributed deployments)
CHECKPOINTS_DIR = os.environ.get("TINKER_CHECKPOINT_DIR", "./checkpoints")


def _resolve_state_path(state_uri: str) -> str:
    """Convert tinker://local/model_id/name or file:// to filesystem path.

    Args:
        state_uri: URI like tinker://local/..., tinker://localhost/...,
                   file://..., or absolute path.

    Returns:
        Filesystem path.
    """
    if state_uri.startswith("tinker://local/"):
        # tinker://local/model_id/checkpoint-100 -> ./checkpoints/model_id/checkpoint-100
        path_part = state_uri[len("tinker://local/"):]
        return os.path.join(CHECKPOINTS_DIR, path_part)
    elif state_uri.startswith("tinker://localhost"):
        # tinker://localhost/abs/path -> /abs/path
        return state_uri[len("tinker://localhost"):]
    elif state_uri.startswith("file://"):
        return state_uri[7:]
    return state_uri


@router.post("/create_model_from_state", response_model=UntypedAPIFuture)
async def create_model_from_state(
    request: CreateModelFromStateRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Create a training model and load existing checkpoint.

    Composes create_model + load_state into single operation.
    Useful for resuming training from a saved checkpoint.
    """
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    request_id = future_store.create()
    background_tasks.add_task(_do_create_model_from_state, request_id, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_create_model_from_state(
    request_id: str, request: CreateModelFromStateRequest
) -> None:
    """Background task to create model and load checkpoint."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        model_id = _generate_model_id(request.session_id, request.model_seq_id)

        # Check if model already exists (from failed previous attempt)
        existing = training_manager.get_session(model_id)
        if existing is not None:
            logger.warning(f"[{model_id}] Cleaning up stale session from previous attempt")
            await training_engine.shutdown_session(existing)
            training_manager.delete_session(model_id)

        # Create session metadata
        session = training_manager.create_session(
            model_id=model_id,
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            user_metadata=request.user_metadata,
        )

        # Create Ray actor
        await training_engine.create_training_session(session)

        # Resolve state path
        load_path = _resolve_state_path(request.state_path)

        # Load checkpoint into the newly created model
        await training_engine.load_weights(
            session=session,
            load_path=load_path,
            load_optimizer=request.load_optimizer,
        )

        logger.info(
            f"[{model_id}] Created model from state: {request.state_path} "
            f"(step={session.current_step})"
        )

        response = CreateModelFromStateResponse(
            request_id=request_id,
            model_id=model_id,
            type="create_model_from_state",
        )
        future_store.resolve(request_id, response.model_dump())

    except Exception as e:
        logger.exception(f"[create_model_from_state] Failed: {e}")
        # Clean up session if it was created
        model_id = _generate_model_id(request.session_id, request.model_seq_id)
        if training_manager and training_manager.get_session(model_id):
            try:
                session = training_manager.get_session(model_id)
                if session:
                    await training_engine.shutdown_session(session)
            except Exception:
                pass  # Ignore cleanup errors
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
# forward - async (forward only, no backward)
# =============================================================================


@router.post("/forward", response_model=UntypedAPIFuture)
async def forward(
    request: ForwardRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Perform forward pass only (no backward). Returns logprobs.

    Uses ForwardRequest with forward_input field (not forward_backward_input)
    to match tinker client API.
    """
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    request_id = future_store.create()
    background_tasks.add_task(_do_forward, request_id, session, request)
    return UntypedAPIFuture(request_id=request_id)


async def _do_forward(
    request_id: str, session, request: ForwardRequest
) -> None:
    """Background task for forward."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        result = await training_engine.forward(session, request)
        future_store.resolve(request_id, result)

    except Exception as e:
        logger.exception(f"[forward] Failed: {e}")
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
# reset_expert_bias - sync (fast operation)
# =============================================================================


@router.post("/reset_expert_bias", response_model=ResetExpertBiasResponse)
async def reset_expert_bias(
    request: ResetExpertBiasRequest,
) -> ResetExpertBiasResponse:
    """Reset expert_bias buffers in MoE router modules.

    This ensures consistent behavior between Megatron (training) and vLLM
    (inference), as expert_bias accumulates during training but is not
    exported with LoRA weights.

    Call this before computing logprobs to ensure consistent routing with vLLM.
    """
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    try:
        result = await training_engine.reset_expert_bias(session)
        return ResetExpertBiasResponse(
            model_id=request.model_id,
            modules_reset=result.get("modules_reset", 0),
            status="success" if result.get("modules_reset", 0) > 0 else "not_applicable",
        )
    except Exception as e:
        logger.exception(f"[reset_expert_bias] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# save_weights_for_sampler - async
# =============================================================================


@router.post("/save_weights_for_sampler", response_model=UntypedAPIFuture)
async def save_weights_for_sampler(
    request: SaveWeightsForSamplerRequest,
    background_tasks: BackgroundTasks,
) -> UntypedAPIFuture:
    """Save model weights for inference use."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    request_id = future_store.create()
    background_tasks.add_task(
        _do_save_weights_for_sampler, request_id, session, request
    )
    return UntypedAPIFuture(request_id=request_id)


async def _do_save_weights_for_sampler(
    request_id: str, session, request: SaveWeightsForSamplerRequest
) -> None:
    """Background task for save_weights_for_sampler.

    Two flows:
    - Named (path is not None): Save to persistent location, return path
    - Ephemeral (path is None): Use per-session inference engine for isolated concurrent access
    """
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        # Get checkpoint directory from environment or default
        checkpoint_dir = os.environ.get(
            "TINKER_CHECKPOINT_DIR",
            os.path.join(os.getcwd(), "checkpoints")
        )

        # Determine checkpoint name
        if request.path is not None:
            # Named save - use provided path
            checkpoint_name = request.path
        else:
            # Ephemeral save - generate unique temp name
            checkpoint_name = f"_ephemeral_{uuid.uuid4().hex[:8]}"

        # Save weights
        save_path = await training_engine.save_weights_for_sampler(
            session=session,
            checkpoint_name=checkpoint_name,
            checkpoint_base_dir=checkpoint_dir,
        )

        # Use tinker:// URI format for SDK compatibility
        # Format: tinker://localhost/<absolute_path>
        path_uri = f"tinker://localhost{save_path}"

        if request.path is not None:
            # Named flow: Return path, caller creates session separately
            response = SaveWeightsForSamplerResponse(
                path=path_uri,
                sampling_session_id=None,
            )
        else:
            # Ephemeral flow: Use multi-LoRA engine for frozen per-session weights
            # Each sampling session gets unique lora_int_id with frozen weights.
            # Matches Tinker SDK semantics where each save creates isolated snapshot.
            if inference_manager is None:
                raise RuntimeError("Inference manager not initialized")

            import json
            import time

            from safetensors.torch import load_file

            sampling_session_id = str(uuid.uuid4())
            lora_rank = session.lora_config.rank if session.lora_config else 32
            base_model = session.base_model

            # Get or create engine for this model (dynamically creates vLLM actor if needed)
            multi_lora_engine = await inference_manager.get_engine_for_model(base_model)

            if multi_lora_engine is not None:
                # Multi-LoRA mode: Each sampling session gets frozen weights
                # Transfer tensors via Ray to support distributed deployments
                # where API server and worker have different filesystems.
                # Worker saves locally and creates file-based LoRARequest,
                # which supports vLLM's GPU/CPU swapping.
                start_time = time.time()

                # Load tensors from saved checkpoint on API server
                weights_path = os.path.join(save_path, "adapter_model.safetensors")
                config_path = os.path.join(save_path, "adapter_config.json")
                state_dict = load_file(weights_path)
                with open(config_path, "r") as f:
                    peft_config = json.load(f)

                # Add LoRA with unique ID for this sampling session
                # Tensors transferred via Ray, worker saves locally
                lora_id = await multi_lora_engine.add_lora_for_session(
                    sampling_session_id=sampling_session_id,
                    state_dict=state_dict,
                    peft_config=peft_config,
                )

                # Register in session manager with base_model for multi-model routing
                inference_manager.register_multi_lora_session(
                    session_id=sampling_session_id,
                    base_model=base_model,
                    lora_rank=lora_rank,
                )

                load_time = time.time() - start_time
                logger.info(
                    f"[save_weights_for_sampler] Multi-LoRA: added lora_id={lora_id} "
                    f"for session {sampling_session_id} (model={base_model}) in {load_time:.3f}s"
                )
            else:
                # Fallback: Per-session engine mode (legacy)
                # This path is used when multi-LoRA engine creation fails
                from ..backend.verl_inference import VerlInferenceEngine
                from ..backend.multi_lora_engine import _resolve_model_path

                if session.inference_engine is None:
                    # First call: Create dedicated inference engine
                    logger.info(
                        f"[save_weights_for_sampler] Creating per-session inference engine "
                        f"for {session.model_id}"
                    )
                    start_time = time.time()

                    # Resolve model path and use session's base_model
                    resolved_model_path = _resolve_model_path(base_model)

                    engine = VerlInferenceEngine(
                        model_path=resolved_model_path,
                        tensor_parallel_size=inference_manager.tensor_parallel_size,
                        data_parallel_size=inference_manager.data_parallel_size,
                        gpu_memory_utilization=inference_manager.gpu_memory_utilization,
                        max_model_len=inference_manager.max_model_len,
                        lora_rank=lora_rank,
                        lora_adapter_path=save_path,
                    )
                    await engine.initialize()
                    session.inference_engine = engine

                    init_time = time.time() - start_time
                    logger.info(
                        f"[save_weights_for_sampler] Per-session engine initialized "
                        f"in {init_time:.2f}s for {session.model_id}"
                    )
                else:
                    # Subsequent calls: Hot-reload LoRA on existing engine
                    start_time = time.time()
                    await session.inference_engine.load_lora_from_path(save_path)
                    reload_time = time.time() - start_time
                    logger.info(
                        f"[save_weights_for_sampler] Hot-reloaded LoRA "
                        f"in {reload_time:.3f}s for {session.model_id}"
                    )

                # Register sampling session pointing to per-session engine
                inference_manager.create_session_with_engine(
                    session_id=sampling_session_id,
                    engine=session.inference_engine,
                    lora_rank=lora_rank,
                )
                logger.info(
                    f"[save_weights_for_sampler] Ephemeral (legacy): session "
                    f"{sampling_session_id} using per-session engine for {session.model_id}"
                )

            response = SaveWeightsForSamplerResponse(
                path=None,  # Ephemeral - no path returned
                sampling_session_id=sampling_session_id,
            )

        future_store.resolve(request_id, response.model_dump())

    except Exception as e:
        logger.exception(f"[save_weights_for_sampler] Failed: {e}")
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


@router.post("/get_info", response_model=GetInfoResponse)
async def get_info(request: GetInfoRequest) -> GetInfoResponse:
    """Get model info (tinker client compatible endpoint).

    Returns model architecture, tokenizer, and LoRA configuration.
    """
    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training manager not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    # Build response matching tinker client expectations
    lora_rank = session.lora_config.rank if session.lora_config else None
    is_lora = session.lora_config is not None

    return GetInfoResponse(
        model_id=session.model_id,
        model_data=ModelData(
            arch="transformer",  # Generic architecture identifier
            model_name=session.base_model,
            tokenizer_id=session.base_model,  # Use base model as tokenizer ID
        ),
        model_name=session.base_model,
        is_lora=is_lora,
        lora_rank=lora_rank,
    )


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


@router.get("/models/{model_id}/tokenizer")
async def get_tokenizer(model_id: str):
    """Get tokenizer configuration for a training model.

    Returns tokenizer info (vocab_size, special tokens, etc.)
    for client-side tokenization.
    """
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found"
        )

    tokenizer_info = await training_engine.get_tokenizer_info(session)
    return {
        "model_id": model_id,
        "tokenizer": tokenizer_info,
    }
