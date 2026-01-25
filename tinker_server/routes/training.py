"""Training routes for model training.

Endpoints:
- POST /create_model: Create a training model
- POST /create_model_from_state: Create model and load checkpoint (resume training)
- POST /forward_backward: Forward + backward pass
- POST /train_step: Combined forward_backward + optim_step
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
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from ..backend.future_store import future_store
from ..checkpoints import CHECKPOINTS_DIR, resolve_checkpoint_path
from ..model_access_control import can_access_model, get_access_denied_error
from ..models.types import (
    CreateModelFromStateRequest,
    CreateModelFromStateResponse,
    CreateModelRequest,
    CreateModelResponse,
    Datum,
    ForwardBackwardRequest,
    ForwardRequest,
    GetInfoRequest,
    GetInfoResponse,
    LoRAConfig,
    ModelData,
    OptimStepRequest,
    ResetExpertBiasRequest,
    ResetExpertBiasResponse,
    SaveWeightsForSamplerRequest,
    SaveWeightsForSamplerResponse,
    TrainStepRequest,
    UntypedAPIFuture,
)
from ..usage_logger import get_usage_logger
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
inference_manager: SessionManager | None = None  # For ephemeral flow


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state (set by auth middleware)."""
    return getattr(request.state, "user_data", None)


def _get_user_id(request: Request) -> str | None:
    """Extract user_id from request state (set by auth middleware)."""
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("user_id")
    return None


def _get_webhook_url(request: Request) -> str | None:
    """Extract webhook_url from request state (set by auth middleware)."""
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("webhook_url")
    return None


def _restore_training_session(model_id: str):
    """Best-effort restore of a training session after API process restart."""
    if training_engine is None or training_manager is None:
        return None
    try:
        import ray
        from ..backend.training_session_store import get_training_session_info

        info = get_training_session_info(model_id)
        if not isinstance(info, dict):
            return None

        lora_cfg = None
        if info.get("lora_config"):
            lora_cfg = LoRAConfig(**info["lora_config"])

        session = training_manager.get_session(model_id)
        if session is None:
            session = training_manager.create_session(
                model_id=model_id,
                session_id=str(info.get("session_id", "")),
                model_seq_id=int(info.get("model_seq_id", 0)),
                base_model=str(info.get("base_model", "")),
                lora_config=lora_cfg,
                user_metadata=info.get("user_metadata") or {},
                learning_rate=float(info.get("learning_rate", 1e-4)),
            )

        session.backend = str(info.get("backend", session.backend))
        try:
            session.current_step = int(info.get("current_step", session.current_step))
        except Exception:
            pass
        session.is_active = True

        actor_name = info.get("actor_name")
        if actor_name:
            namespace = str(info.get("namespace") or os.environ.get("MINT_RAY_NAMESPACE", "tinker"))
            worker = ray.get_actor(actor_name, namespace=namespace)
            getattr(training_engine, "_workers", {})[model_id] = worker
            getattr(training_engine, "_resource_pool_actor_names", {})[model_id] = actor_name

        return session
    except Exception as e:
        logger.exception(f"[{model_id}] restore_training_session failed: {e}")
        return None


def _generate_model_id(session_id: str, model_seq_id: int) -> str:
    """Generate unique model_id from session_id and model_seq_id."""
    return f"{session_id}_{model_seq_id}"

def _compute_token_stats(data: list[Datum]) -> tuple[int, int]:
    """Compute (total_tokens, max_seq_len) without materializing token-id lists."""
    total_tokens = 0
    max_seq_len = 0
    for datum in data:
        seq_len = 0
        for chunk in datum.model_input.chunks:
            if chunk.type == "encoded_text":
                seq_len += len(chunk.tokens)
        total_tokens += seq_len
        if seq_len > max_seq_len:
            max_seq_len = seq_len
    return total_tokens, max_seq_len

def _get_max_model_len(base_model: str | None) -> int | None:
    """Return the configured max_model_len for a supported model name, else None."""
    if not base_model:
        return None
    try:
        from ..backend.model_registry import get_model_config, normalize_model_name

        model_name = normalize_model_name(base_model)
        return int(get_model_config(model_name).max_model_len)
    except Exception:
        return None


# =============================================================================
# create_model - async
# =============================================================================


@router.post("/create_model", response_model=UntypedAPIFuture)
async def create_model(
    request: CreateModelRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Create a new training model with LoRA."""
    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(request.base_model, user_data):
        raise HTTPException(
            status_code=403,
            detail=get_access_denied_error(request.base_model)
        )

    # Gateway forwarding: if base_model is configured as remote, proxy to upstream and
    # return a gateway-encoded request_id so /retrieve_future can route it.
    from ..gateway import (
        encode_request_id,
        forward_json,
        register_remote_training_model,
        upstream_for_model,
    )

    upstream = upstream_for_model(request.base_model)
    if upstream is not None:
        resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/create_model",
            incoming_headers=dict(http_request.headers),
            json_body=request.model_dump(),
            timeout_s=30.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream create_model returned invalid request_id")

        model_id = _generate_model_id(request.session_id, request.model_seq_id)
        register_remote_training_model(
            model_id=model_id,
            upstream_alias=upstream.alias,
            base_model=request.base_model,
        )
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream.alias, upstream_request_id=upstream_request_id)
        )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)

    # 1. 发送 pending 状态 - 任务已创建，等待执行
    model_id = _generate_model_id(request.session_id, request.model_seq_id)
    if webhook_url and user_id:
        send_task_event(
            webhook_url=webhook_url,
            event_type=EventType.TASK_CREATED,  # pending
            user_id=user_id,
            session_id=model_id,
            task_name=f"Training {request.base_model}",
            task_type="training",
            model_name=request.base_model,
            config={"lora_rank": request.lora_config.rank if request.lora_config else None},
        )

    background_tasks.add_task(_do_create_model, request_id, request, user_id, webhook_url)
    return UntypedAPIFuture(request_id=request_id)


async def _do_create_model(
    request_id: str,
    request: CreateModelRequest,
    user_id: str | None,
    webhook_url: str | None,
) -> None:
    """Background task to create training model."""
    model_id = _generate_model_id(request.session_id, request.model_seq_id)
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

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

        try:
            from ..backend.training_session_store import upsert_training_session

            actor_name = getattr(training_engine, "_resource_pool_actor_names", {}).get(model_id)
            upsert_training_session({
                "model_id": model_id,
                "session_id": request.session_id,
                "model_seq_id": request.model_seq_id,
                "base_model": request.base_model,
                "lora_config": request.lora_config.model_dump() if request.lora_config else None,
                "user_metadata": request.user_metadata or {},
                "learning_rate": session.learning_rate,
                "backend": session.backend,
                "actor_name": actor_name,
                "namespace": os.environ.get("MINT_RAY_NAMESPACE", "tinker"),
            })
        except Exception:
            pass

        response = CreateModelResponse(
            request_id=request_id,
            model_id=model_id,
            type="create_model",
            backend=session.backend,  # "megatron" or "peft"
        )
        future_store.resolve(request_id, response.model_dump())

        # 2. 发送 running 状态 - 模型创建成功，训练就绪
        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_STARTED,  # running
                user_id=user_id,
                session_id=model_id,
                task_name=f"Training {request.base_model}",
                task_type="training",
                model_name=request.base_model,
            )

    except Exception as e:
        logger.exception(f"[create_model] Failed: {e}")
        # Clean up session if it was created
        if training_manager and training_manager.get_session(model_id):
            training_manager.delete_session(model_id)
        # If session tracking was updated in ResourcePool during a partially-failed
        # create_training_session, clear it to avoid pinning actors as non-idle.
        try:
            from ..backend.resource_pool import get_resource_pool

            get_resource_pool().clear_session(model_id)
        except Exception:
            pass
        future_store.fail(request_id, str(e))

        # 发送 failed 状态
        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_FAILED,
                user_id=user_id,
                session_id=model_id,
                task_name=f"Training {request.base_model}",
                task_type="training",
                model_name=request.base_model,
                error=str(e),
            )


# =============================================================================
# create_model_from_state - async (composes create_model + load_state)
# =============================================================================

def _resolve_state_path(state_uri: str, *, user_id: str | None) -> str:
    return resolve_checkpoint_path(state_uri, user_id=user_id)
@router.post("/create_model_from_state", response_model=UntypedAPIFuture)
async def create_model_from_state(
    request: CreateModelFromStateRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Create a training model and load existing checkpoint.

    Composes create_model + load_state into single operation.
    Useful for resuming training from a saved checkpoint.
    """
    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(request.base_model, user_data):
        raise HTTPException(
            status_code=403,
            detail=get_access_denied_error(request.base_model)
        )

    from ..gateway import (
        encode_request_id,
        forward_json,
        register_remote_training_model,
        upstream_for_model,
    )

    upstream = upstream_for_model(request.base_model)
    if upstream is not None:
        resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/create_model_from_state",
            incoming_headers=dict(http_request.headers),
            json_body=request.model_dump(),
            timeout_s=30.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream create_model_from_state returned invalid request_id")

        model_id = _generate_model_id(request.session_id, request.model_seq_id)
        register_remote_training_model(
            model_id=model_id,
            upstream_alias=upstream.alias,
            base_model=request.base_model,
        )
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream.alias, upstream_request_id=upstream_request_id)
        )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    background_tasks.add_task(_do_create_model_from_state, request_id, request, user_id)
    return UntypedAPIFuture(request_id=request_id)


async def _do_create_model_from_state(
    request_id: str, request: CreateModelFromStateRequest, user_id: str | None
) -> None:
    """Background task to create model and load checkpoint."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        model_id = _generate_model_id(request.session_id, request.model_seq_id)

        # Resolve state path (before creating a session/actor)
        load_path = _resolve_state_path(request.state_path, user_id=user_id)
        if user_id and user_id != "admin":
            load_real = os.path.realpath(load_path)
            checkpoints_real = os.path.realpath(CHECKPOINTS_DIR)
            allowed_real = os.path.realpath(os.path.join(CHECKPOINTS_DIR, user_id))
            if load_real.startswith(checkpoints_real + os.sep) and not load_real.startswith(
                allowed_real + os.sep
            ):
                raise PermissionError("Access denied")
        if request.state_path.startswith(("tinker://", "mint://", "ckpt_")) and not os.path.isdir(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {request.state_path}")

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

        # Load checkpoint into the newly created model
        await training_engine.load_weights(
            session=session,
            load_path=load_path,
            load_optimizer=request.load_optimizer,
        )

        try:
            from ..backend.training_session_store import upsert_training_session

            actor_name = getattr(training_engine, "_resource_pool_actor_names", {}).get(model_id)
            upsert_training_session({
                "model_id": model_id,
                "session_id": request.session_id,
                "model_seq_id": request.model_seq_id,
                "base_model": request.base_model,
                "lora_config": request.lora_config.model_dump() if request.lora_config else None,
                "user_metadata": request.user_metadata or {},
                "learning_rate": session.learning_rate,
                "backend": session.backend,
                "actor_name": actor_name,
                "namespace": os.environ.get("MINT_RAY_NAMESPACE", "tinker"),
            })
        except Exception:
            pass

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
        try:
            from ..backend.training_session_store import delete_training_session

            delete_training_session(model_id)
        except Exception:
            pass
        future_store.fail(request_id, str(e))


# =============================================================================
# forward_backward - async
# =============================================================================


@router.post("/forward_backward", response_model=UntypedAPIFuture)
async def forward_backward(
    request: ForwardBackwardRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform forward + backward pass on training data."""
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    remote = remote_training_model(request.model_id)
    if remote is not None:
        upstream_alias, base_model = remote
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")
        user_data = _get_user_data(http_request)
        if not can_access_model(base_model, user_data):
            raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

        resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/forward_backward",
            incoming_headers=dict(http_request.headers),
            json_body=request.model_dump(),
            timeout_s=300.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream forward_backward returned invalid request_id")
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
        )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        session = _restore_training_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    max_model_len = _get_max_model_len(session.base_model)
    if max_model_len is not None:
        _, max_seq_len = _compute_token_stats(request.forward_backward_input.data)
        if max_seq_len > max_model_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                    f"for model {session.base_model}"
                ),
            )

    request_id = future_store.create()
    user_id = _get_user_id(http_request)

    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        worker = getattr(training_engine, "_workers", {}).get(session.model_id)
        if worker is None:
            raise RuntimeError(f"Training worker not found for model_id={session.model_id}")

        batch = request.forward_backward_input.data
        token_count, max_seq_len = _compute_token_stats(batch)
        msg = (
            f"[{session.model_id}] forward_backward submit request_id={request_id} "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len} "
            f"loss_fn={request.forward_backward_input.loss_fn}"
        )
        print(msg, flush=True)
        logger.info(msg)

        data_items = [item.model_dump() for item in request.forward_backward_input.data]
        loss_fn = request.forward_backward_input.loss_fn
        loss_fn_config = request.forward_backward_input.loss_fn_config or {}

        actor_name = getattr(training_engine, "_resource_pool_actor_names", {}).get(session.model_id)
        future_store.submit(
            request_id,
            worker,
            "forward_backward",
            [data_items, loss_fn, loss_fn_config, session.model_id],
            meta={"actor_name": actor_name, "model_id": session.model_id, "op": "forward_backward"},
        )

        if user_id:
            get_usage_logger().log(
                user_id=user_id,
                operation_type="forward_backward",
                model_name=session.base_model,
                token_count=token_count,
                session_id=session.model_id,
                request_id=request_id,
            )
    except Exception as e:
        logger.exception(f"[forward_backward] submit failed: {e}")
        future_store.fail(request_id, str(e))

    return UntypedAPIFuture(request_id=request_id)


async def _do_forward_backward(
    request_id: str, session, request: ForwardBackwardRequest, user_id: str | None
) -> None:
    """Background task for forward_backward."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        batch = request.forward_backward_input.data
        token_count, max_seq_len = _compute_token_stats(batch)
        t0 = time.time()
        msg = (
            f"[{session.model_id}] forward_backward start request_id={request_id} "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len} "
            f"loss_fn={request.forward_backward_input.loss_fn}"
        )
        print(msg, flush=True)
        logger.info(msg)
        result = await training_engine.forward_backward(session, request)
        elapsed_s = time.time() - t0
        msg = f"[{session.model_id}] forward_backward done request_id={request_id} elapsed_s={elapsed_s:.3f}"
        print(msg, flush=True)
        logger.info(msg)
        future_store.resolve(request_id, result)

        # Log usage
        if user_id:
            get_usage_logger().log(
                user_id=user_id,
                operation_type="forward_backward",
                model_name=session.base_model,
                token_count=token_count,
                session_id=session.model_id,
                request_id=request_id,
            )

    except Exception as e:
        logger.exception(f"[forward_backward] Failed: {e}")
        future_store.fail(request_id, str(e))


# =============================================================================
# train_step - async (forward_backward + optim_step)
# =============================================================================


@router.post("/train_step", response_model=UntypedAPIFuture)
async def train_step(
    request: TrainStepRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform a combined forward_backward + optim_step."""
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        session = _restore_training_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    max_model_len = _get_max_model_len(session.base_model)
    if max_model_len is not None:
        _, max_seq_len = _compute_token_stats(request.forward_backward_input.data)
        if max_seq_len > max_model_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                    f"for model {session.base_model}"
                ),
            )

    request_id = future_store.create()
    user_id = _get_user_id(http_request)
    background_tasks.add_task(_do_train_step, request_id, session, request, user_id)
    return UntypedAPIFuture(request_id=request_id)


async def _do_train_step(
    request_id: str, session, request: TrainStepRequest, user_id: str | None
) -> None:
    """Background task for train_step."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        batch = request.forward_backward_input.data
        token_count, max_seq_len = _compute_token_stats(batch)
        t0 = time.time()
        msg = (
            f"[{session.model_id}] train_step start request_id={request_id} "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len}"
        )
        print(msg, flush=True)
        logger.info(msg)
        result = await training_engine.train_step(session, request)
        elapsed_s = time.time() - t0
        msg = f"[{session.model_id}] train_step done request_id={request_id} elapsed_s={elapsed_s:.3f}"
        print(msg, flush=True)
        logger.info(msg)
        future_store.resolve(request_id, result)

        # Log usage
        if user_id:
            get_usage_logger().log(
                user_id=user_id,
                operation_type="train_step",
                model_name=session.base_model,
                token_count=token_count,
                session_id=session.model_id,
                request_id=request_id,
            )

    except Exception as e:
        logger.exception(f"[train_step] Failed: {e}")
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
        session = _restore_training_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    max_model_len = _get_max_model_len(session.base_model)
    if max_model_len is not None:
        _, max_seq_len = _compute_token_stats(request.forward_input.data)
        if max_seq_len > max_model_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                    f"for model {session.base_model}"
                ),
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
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform optimizer step to update weights."""
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    remote = remote_training_model(request.model_id)
    if remote is not None:
        upstream_alias, base_model = remote
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        user_data = _get_user_data(http_request)
        if not can_access_model(base_model, user_data):
            raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

        resp = await forward_json(
            upstream=upstream,
            method="POST",
            path="/api/v1/optim_step",
            incoming_headers=dict(http_request.headers),
            json_body=request.model_dump(),
            timeout_s=300.0,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream optim_step returned invalid request_id")
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
        )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        session = _restore_training_session(request.model_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    request_id = future_store.create()

    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        worker = getattr(training_engine, "_workers", {}).get(session.model_id)
        if worker is None:
            raise RuntimeError(f"Training worker not found for model_id={session.model_id}")

        lr = request.adam_params.learning_rate if request.adam_params else None
        msg = f"[{session.model_id}] optim_step submit request_id={request_id} lr={lr}"
        print(msg, flush=True)
        logger.info(msg)

        actor_name = getattr(training_engine, "_resource_pool_actor_names", {}).get(session.model_id)
        future_store.submit(
            request_id,
            worker,
            "optim_step",
            [lr, session.model_id],
            meta={"actor_name": actor_name, "model_id": session.model_id, "op": "optim_step"},
        )
    except Exception as e:
        logger.exception(f"[optim_step] submit failed: {e}")
        future_store.fail(request_id, str(e))

    return UntypedAPIFuture(request_id=request_id)


async def _do_optim_step(request_id: str, session, request: OptimStepRequest) -> None:
    """Background task for optim_step."""
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        lr = request.adam_params.learning_rate if request.adam_params else None
        t0 = time.time()
        msg = f"[{session.model_id}] optim_step start request_id={request_id} lr={lr}"
        print(msg, flush=True)
        logger.info(msg)
        result = await training_engine.optim_step(session, request)
        elapsed_s = time.time() - t0
        msg = f"[{session.model_id}] optim_step done request_id={request_id} elapsed_s={elapsed_s:.3f}"
        print(msg, flush=True)
        logger.info(msg)
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
        session = _restore_training_session(request.model_id)
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
        session = _restore_training_session(request.model_id)
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
    print(f"[DEBUG _do_save_weights_for_sampler] ENTRY request_id={request_id}", flush=True)
    try:
        if training_engine is None:
            raise RuntimeError("Training engine not initialized")

        from ..checkpoints import get_checkpoints_dir

        checkpoint_dir = get_checkpoints_dir()

        # Determine checkpoint name
        if request.path is not None:
            # Named save - use provided path
            checkpoint_name = request.path
        else:
            # Ephemeral save - generate unique temp name
            checkpoint_name = f"_ephemeral_{uuid.uuid4().hex[:8]}"

        use_per_expert_lora = bool(request.use_per_expert_lora)
        if (
            session.backend == "megatron"
            and not use_per_expert_lora
            and "use_per_expert_lora" not in getattr(request, "model_fields_set", set())
        ):
            # Default behavior for MoE: if the session trains MLP LoRA, export in
            # per-expert format so vLLM can consume it.
            if getattr(getattr(session, "lora_config", None), "train_mlp", False):
                use_per_expert_lora = True

        print(f"[DEBUG _do_save_weights_for_sampler] calling save_weights_for_sampler", flush=True)
        # Save weights
        save_path = await training_engine.save_weights_for_sampler(
            session=session,
            checkpoint_name=checkpoint_name,
            checkpoint_base_dir=checkpoint_dir,
            use_per_expert_lora=use_per_expert_lora,
        )
        print(f"[DEBUG _do_save_weights_for_sampler] save_path={save_path}", flush=True)

        # Use tinker:// URI format (matches client SDK expectation)
        # Format: tinker://{model_id}/{checkpoint_name}
        path_uri = f"tinker://{session.model_id}/{checkpoint_name}"

        if request.path is not None:
            # Named flow: Return path, caller creates session separately
            print(f"[DEBUG _do_save_weights_for_sampler] Named flow", flush=True)
            response = SaveWeightsForSamplerResponse(
                path=path_uri,
                sampling_session_id=None,
            )
        else:
            # Ephemeral flow: Use multi-LoRA engine for frozen per-session weights
            # Each sampling session gets unique lora_int_id with frozen weights.
            # Matches Tinker SDK semantics where each save creates isolated snapshot.
            print(f"[DEBUG _do_save_weights_for_sampler] Ephemeral flow", flush=True)
            if inference_manager is None:
                raise RuntimeError("Inference manager not initialized")

            import json
            import time

            from safetensors.torch import load_file

            sampling_session_id = str(uuid.uuid4())
            lora_rank = session.lora_config.rank if session.lora_config else 32
            base_model = session.base_model
            print(f"[DEBUG _do_save_weights_for_sampler] base_model={base_model}", flush=True)

            # Get or create engine for this model (dynamically creates vLLM actor if needed)
            print(f"[DEBUG _do_save_weights_for_sampler] getting engine for model", flush=True)
            multi_lora_engine = await inference_manager.get_engine_for_model(base_model)
            print(f"[DEBUG _do_save_weights_for_sampler] got engine: {multi_lora_engine is not None}", flush=True)

            if multi_lora_engine is not None:
                # Multi-LoRA mode: Each sampling session gets frozen weights
                # Use path-based loading for MoE models (avoids 30k+ tensor Ray transfer)
                # vLLM worker loads directly from shared PFS
                start_time = time.time()

                # Add LoRA from path - vLLM worker loads directly from PFS
                # This avoids serializing 37k+ tensors through Ray object store
                print(f"[DEBUG _do_save_weights_for_sampler] calling add_lora_for_session_from_path with {save_path}", flush=True)
                lora_id = await multi_lora_engine.add_lora_for_session_from_path(
                    sampling_session_id=sampling_session_id,
                    lora_path=save_path,
                )
                print(f"[DEBUG _do_save_weights_for_sampler] add_lora_for_session_from_path returned lora_id={lora_id}", flush=True)

                # Register in session manager with base_model for multi-model routing
                inference_manager.register_multi_lora_session(
                    session_id=sampling_session_id,
                    base_model=base_model,
                    lora_rank=lora_rank,
                    adapter_path=save_path,
                )
                print(f"[DEBUG _do_save_weights_for_sampler] registered session", flush=True)

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
        session = _restore_training_session(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

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
        session = _restore_training_session(request.model_id)
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
    try:
        from ..backend.training_session_store import delete_training_session

        delete_training_session(model_id)
    except Exception:
        pass
    # Clear ResourcePool session tracking even if shutdown_session couldn't find a worker
    # (e.g., deletion races with create_training_session still in-flight).
    try:
        from ..backend.resource_pool import get_resource_pool

        get_resource_pool().clear_session(model_id)
    except Exception:
        pass

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
