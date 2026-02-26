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
- GET /training_runs: List training runs (RestClient)
- GET /training_runs/{training_run_id}: Get training run (RestClient)
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
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from ..backend.future_store import future_store
from ..checkpoints import CHECKPOINTS_DIR, create_checkpoint_archive, resolve_checkpoint_path
from ..config import RAY_NAMESPACE
from ..model_access_control import can_access_model, get_access_denied_error
from ..models.types import (
    CreateModelFromStateRequest,
    CreateModelFromStateResponse,
    CreateModelRequest,
    CreateModelResponse,
    Cursor,
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
    TrainingRun,
    TrainingRunsResponse,
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
                user_id=info.get("user_id"),
                learning_rate=float(info.get("learning_rate", 1e-4)),
            )

        session.backend = str(info.get("backend", session.backend))
        created_at = info.get("created_at")
        if isinstance(created_at, str) and created_at:
            session.created_at = created_at
        try:
            session.current_step = int(info.get("current_step", session.current_step))
        except Exception:
            pass
        session.is_active = True

        actor_name = info.get("actor_name")
        if actor_name:
            namespace = str(info.get("namespace") or RAY_NAMESPACE)
            worker = ray.get_actor(actor_name, namespace=namespace)
            getattr(training_engine, "_workers", {})[model_id] = worker
            getattr(training_engine, "_resource_pool_actor_names", {})[model_id] = actor_name

        return session
    except Exception as e:
        logger.exception(f"[{model_id}] restore_training_session failed: {e}")
        return None


def _raise_if_local_model_id_exists(model_id: str) -> None:
    if training_engine is None or training_manager is None:
        return
    if training_manager.get_session(model_id) is not None:
        raise HTTPException(status_code=409, detail=f"Model_id conflict: local model already exists: {model_id!r}")
    try:
        from ..backend.training_session_store import get_training_session_info

        info = get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e
    if isinstance(info, dict):
        raise HTTPException(status_code=409, detail=f"Model_id conflict: local model already exists: {model_id!r}")


def _generate_model_id(session_id: str, model_seq_id: int) -> str:
    """Generate unique model_id from session_id and model_seq_id."""
    return f"{session_id}_{model_seq_id}"


def _session_info_from_live(session) -> dict:
    return {
        "model_id": session.model_id,
        "session_id": session.session_id,
        "model_seq_id": session.model_seq_id,
        "base_model": session.base_model,
        "lora_config": session.lora_config.model_dump() if session.lora_config else None,
        "user_metadata": session.user_metadata,
        "learning_rate": session.learning_rate,
        "current_step": session.current_step,
        "is_active": session.is_active,
        "created_at": session.created_at,
        "backend": session.backend,
        "user_id": session.user_id,
    }


def _training_run_from_info(info: dict) -> TrainingRun:
    lora_cfg = info.get("lora_config")
    lora_rank = None
    is_lora = False
    if isinstance(lora_cfg, dict):
        lora_rank = lora_cfg.get("rank")
        is_lora = lora_rank is not None
    elif lora_cfg is not None:
        try:
            lora_rank = int(getattr(lora_cfg, "rank", None))
            is_lora = lora_rank is not None
        except Exception:
            pass

    return TrainingRun(
        training_run_id=str(info.get("model_id", "")),
        base_model=str(info.get("base_model", "")),
        model_owner=str(info.get("user_id") or ""),
        is_lora=bool(is_lora),
        corrupted=False,
        lora_rank=lora_rank,
        last_request_time=(
            info.get("last_request_time")
            or info.get("created_at")
            or datetime.now().isoformat()
        ),
        last_checkpoint=None,
        last_sampler_checkpoint=None,
        user_metadata=info.get("user_metadata") or {},
    )

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

def _get_max_model_len(base_model: str | None) -> int:
    """Return the configured max_model_len for a supported model name.

    If the server cannot determine the model's max_model_len, fail fast rather
    than silently skipping the length gate.
    """
    if not base_model:
        raise HTTPException(status_code=500, detail="Training session missing base_model")
    from ..backend.model_registry import get_model_config

    try:
        return int(get_model_config(base_model).max_model_len)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Cannot determine max_model_len for base_model {base_model!r}: "
                f"{type(e).__name__}: {e}"
            ),
        )


# =============================================================================
# create_model - async
# =============================================================================


@router.post("/create_model", response_model=UntypedAPIFuture)
async def create_model(
    request: CreateModelRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Create a new training model with LoRA."""
    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(base_model=request.base_model, http_request=http_request)
    request = request.model_copy(update={"base_model": base_model})

    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(request.base_model, user_data):
        raise HTTPException(
            status_code=403,
            detail=get_access_denied_error(request.base_model)
        )

    model_id = _generate_model_id(request.session_id, request.model_seq_id)

    # Gateway forwarding: if base_model is configured as remote, proxy to upstream and
    # return a gateway-encoded request_id so /retrieve_future can route it.
    from ..gateway import (
        encode_request_id,
        forward_json,
        get_gateway_config,
        register_remote_training_model,
        remote_training_model,
        upstream_for_model,
    )

    upstream = upstream_for_model(request.base_model)
    if upstream is not None:
        _raise_if_local_model_id_exists(model_id)
        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/create_model",
                incoming_headers=dict(http_request.headers),
                json_body=request.model_dump(),
                timeout_s=30.0,
            )
        except Exception:
            logger.exception("Upstream create_model failed: %s", upstream.alias)
            raise HTTPException(status_code=503, detail=f"Upstream {upstream.alias!r} create_model failed")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream create_model returned invalid request_id")

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

    cfg = get_gateway_config()
    if cfg is not None and cfg.model_to_upstream:
        remote = remote_training_model(model_id)
        if remote is not None:
            upstream_alias, _ = remote
            raise HTTPException(
                status_code=409,
                detail=f"Model_id conflict: {model_id!r} is registered as remote via upstream {upstream_alias!r}",
            )

    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)

    # 1. 发送 pending 状态 - 任务已创建，等待执行
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

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = capacity_manager.try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(request_id, meta={"op": "create_model"})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="training.create_model",
            request_json=request_json,
            user_id=user_id,
            webhook_url=webhook_url,
        )
        logger.info(
            "[create_model] enqueued request_id=%s op=%s model_id=%s base_model=%s bytes=%s",
            str(request_id),
            "training.create_model",
            str(model_id),
            str(request.base_model),
            int(len(request_json)),
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue create_model request: {e}")

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
            user_id=user_id,
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
                "namespace": RAY_NAMESPACE,
                "user_id": user_id,
                "created_at": session.created_at,
            })
        except Exception:
            pass

        try:
            from ..backend.session_index_store import add_training_run_to_session

            add_training_run_to_session(
                session_id=request.session_id,
                training_run_id=model_id,
                user_id=user_id,
                created_at=session.created_at,
            )
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
    http_request: Request,
) -> UntypedAPIFuture:
    """Create a training model and load existing checkpoint.

    Composes create_model + load_state into single operation.
    Useful for resuming training from a saved checkpoint.
    """
    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(base_model=request.base_model, http_request=http_request)
    request = request.model_copy(update={"base_model": base_model})

    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(request.base_model, user_data):
        raise HTTPException(
            status_code=403,
            detail=get_access_denied_error(request.base_model)
        )

    model_id = _generate_model_id(request.session_id, request.model_seq_id)

    from ..gateway import (
        encode_request_id,
        forward_file,
        forward_json,
        get_gateway_config,
        register_remote_training_model,
        remote_training_model,
        upstream_for_model,
    )

    upstream = upstream_for_model(request.base_model)
    if upstream is not None:
        _raise_if_local_model_id_exists(model_id)
        user_id = _get_user_id(http_request)
        incoming_headers = dict(http_request.headers)
        if request.state_path.startswith(("tinker://", "mint://", "ckpt_")):
            local_path = resolve_checkpoint_path(request.state_path, user_id=user_id)
            if user_id and user_id != "admin":
                load_real = os.path.realpath(local_path)
                checkpoints_real = os.path.realpath(CHECKPOINTS_DIR)
                allowed_real = os.path.realpath(os.path.join(CHECKPOINTS_DIR, user_id))
                if load_real.startswith(checkpoints_real + os.sep) and not (
                    load_real == allowed_real or load_real.startswith(allowed_real + os.sep)
                ):
                    raise HTTPException(status_code=403, detail="Access denied")
            if os.path.isdir(local_path):
                import asyncio
                import tempfile

                proxy_timeout_s = float(os.environ.get("MINT_GATEWAY_CHECKPOINT_PROXY_TIMEOUT_S", "600"))
                fd, tmp_archive = tempfile.mkstemp(prefix="gateway_ckpt_proxy_", suffix=".tar.gz")
                os.close(fd)
                try:
                    await asyncio.to_thread(create_checkpoint_archive, local_path, tmp_archive)
                    upload_resp = await forward_file(
                        upstream=upstream,
                        path="/api/v1/checkpoints/upload",
                        incoming_headers=incoming_headers,
                        file_path=tmp_archive,
                        timeout_s=proxy_timeout_s,
                    )
                finally:
                    try:
                        os.unlink(tmp_archive)
                    except OSError:
                        pass
                if upload_resp.status_code >= 400:
                    raise HTTPException(status_code=upload_resp.status_code, detail=upload_resp.text)
                payload = upload_resp.json()
                ckpt_id = payload.get("checkpoint_id")
                if not isinstance(ckpt_id, str) or not ckpt_id:
                    raise HTTPException(
                        status_code=502,
                        detail="Upstream checkpoints/upload returned invalid checkpoint_id",
                    )
                request = request.model_copy(update={"state_path": ckpt_id})
        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/create_model_from_state",
                incoming_headers=incoming_headers,
                json_body=request.model_dump(),
                timeout_s=30.0,
            )
        except Exception:
            logger.exception("Upstream create_model_from_state failed: %s", upstream.alias)
            raise HTTPException(status_code=503, detail=f"Upstream {upstream.alias!r} create_model_from_state failed")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream create_model_from_state returned invalid request_id")

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

    cfg = get_gateway_config()
    if cfg is not None and cfg.model_to_upstream:
        remote = remote_training_model(model_id)
        if remote is not None:
            upstream_alias, _ = remote
            raise HTTPException(
                status_code=409,
                detail=f"Model_id conflict: {model_id!r} is registered as remote via upstream {upstream_alias!r}",
            )

    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_forward_backward_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = capacity_manager.try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_forward_backward_result_bytes(request),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(request_id, meta={"op": "create_model_from_state"})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="training.create_model_from_state",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue create_model_from_state request: {e}"
        )

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
            user_id=user_id,
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
                "namespace": RAY_NAMESPACE,
                "user_id": user_id,
                "created_at": session.created_at,
            })
        except Exception:
            pass

        try:
            from ..backend.session_index_store import add_training_run_to_session

            add_training_run_to_session(
                session_id=request.session_id,
                training_run_id=model_id,
                user_id=user_id,
                created_at=session.created_at,
            )
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
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform forward + backward pass on training data."""
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    session = None
    if training_manager is not None:
        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)

    if session is None:
        remote = remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/forward_backward",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=300.0,
                )
            except Exception:
                logger.exception("Upstream forward_backward failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} forward_backward failed")
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

    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    max_model_len = _get_max_model_len(session.base_model)
    _, max_seq_len = _compute_token_stats(request.forward_backward_input.data)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {session.base_model}"
            ),
        )

    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_forward_backward_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = capacity_manager.try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_forward_backward_result_bytes(request),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(request_id, meta={"op": "forward_backward", "model_id": request.model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="training.forward_backward",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue forward_backward request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_forward_backward(request_id: str, request: ForwardBackwardRequest, user_id: str | None) -> None:
    """Background task for forward_backward."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        max_model_len = _get_max_model_len(session.base_model)
        _, max_seq_len = _compute_token_stats(request.forward_backward_input.data)
        if max_seq_len > max_model_len:
            raise RuntimeError(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {session.base_model}"
            )

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
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform a combined forward_backward + optim_step."""
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    session = None
    if training_manager is not None:
        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)

    if session is None:
        remote = remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/train_step",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=300.0,
                )
            except Exception:
                logger.exception("Upstream train_step failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} train_step failed")

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(status_code=502, detail="Upstream train_step returned invalid request_id")
            return UntypedAPIFuture(
                request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
            )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    max_model_len = _get_max_model_len(session.base_model)
    _, max_seq_len = _compute_token_stats(request.forward_backward_input.data)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {session.base_model}"
            ),
        )

    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = capacity_manager.try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(request_id, meta={"op": "train_step", "model_id": request.model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="training.train_step",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue train_step request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_train_step(
    request_id: str, request: TrainStepRequest, user_id: str | None
) -> None:
    """Background task for train_step."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

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
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform forward pass only (no backward). Returns logprobs.

    Uses ForwardRequest with forward_input field (not forward_backward_input)
    to match tinker client API.
    """
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    session = None
    if training_manager is not None:
        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)

    if session is None:
        remote = remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/forward",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=300.0,
                )
            except Exception:
                logger.exception("Upstream forward failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} forward failed")

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(status_code=502, detail="Upstream forward returned invalid request_id")
            return UntypedAPIFuture(
                request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
            )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    max_model_len = _get_max_model_len(session.base_model)
    _, max_seq_len = _compute_token_stats(request.forward_input.data)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {session.base_model}"
            ),
        )

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = capacity_manager.try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(request_id, meta={"op": "forward", "model_id": request.model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="training.forward",
            request_json=request_json,
            user_id=None,
            webhook_url=None,
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue forward request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_forward(
    request_id: str, request: ForwardRequest
) -> None:
    """Background task for forward."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

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
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform optimizer step to update weights."""
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    session = None
    if training_manager is not None:
        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)

    if session is None:
        remote = remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/optim_step",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=300.0,
                )
            except Exception:
                logger.exception("Upstream optim_step failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} optim_step failed")
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

    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = capacity_manager.try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(request_id, meta={"op": "optim_step", "model_id": request.model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="training.optim_step",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue optim_step request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_optim_step(request_id: str, request: OptimStepRequest, user_id: str | None) -> None:
    """Background task for optim_step."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

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
    http_request: Request,
) -> ResetExpertBiasResponse:
    """Reset expert_bias buffers in MoE router modules.

    This ensures consistent behavior between Megatron (training) and vLLM
    (inference), as expert_bias accumulates during training but is not
    exported with LoRA weights.

    Call this before computing logprobs to ensure consistent routing with vLLM.
    """
    from ..gateway import forward_json, remote_training_model, upstream_for_alias

    session = None
    if training_manager is not None:
        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)

    if session is None:
        remote = remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/reset_expert_bias",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=30.0,
                )
            except Exception:
                logger.exception("Upstream reset_expert_bias failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} reset_expert_bias failed")
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return ResetExpertBiasResponse.model_validate(resp.json())

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

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
    http_request: Request,
) -> UntypedAPIFuture:
    """Save model weights for inference use."""
    from ..gateway import (
        encode_request_id,
        forward_json,
        register_pending_save_weights_for_sampler_future,
        remote_training_model,
        upstream_for_alias,
    )

    session = None
    if training_manager is not None:
        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)

    if session is None:
        remote = remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/save_weights_for_sampler",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=30.0,
                )
            except Exception:
                logger.exception("Upstream save_weights_for_sampler failed: %s", upstream_alias)
                raise HTTPException(
                    status_code=503, detail=f"Upstream {upstream_alias!r} save_weights_for_sampler failed"
                )
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(status_code=502, detail="Upstream save_weights_for_sampler returned invalid request_id")
            if request.path is None:
                register_pending_save_weights_for_sampler_future(
                    upstream_alias=upstream_alias, upstream_request_id=upstream_request_id, base_model=base_model
                )
            return UntypedAPIFuture(
                request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
            )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    user_id = _get_user_id(http_request)
    from ..client_compat import prefer_tinker_uri

    prefer_tinker = prefer_tinker_uri(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = capacity_manager.try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    try:
        future_store.create_with_id(request_id)
        created = True
        future_store.mark_queued(
            request_id, meta={"op": "save_weights_for_sampler", "model_id": request.model_id}
        )
        await api_work_queue.enqueue(
            request_id=request_id,
            op="training.save_weights_for_sampler",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
            extra={"prefer_tinker": bool(prefer_tinker)},
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue save_weights_for_sampler request: {e}"
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_save_weights_for_sampler(
    request_id: str,
    request: SaveWeightsForSamplerRequest,
    user_id: str | None,
    prefer_tinker: bool,
) -> None:
    """Background task for save_weights_for_sampler.

    Two flows:
    - Named (path is not None): Save to persistent location, return path
    - Ephemeral (path is None): Use per-session inference engine for isolated concurrent access
    """
    print(f"[DEBUG _do_save_weights_for_sampler] ENTRY request_id={request_id}", flush=True)
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        from ..checkpoints import get_checkpoints_dir

        checkpoints_root = get_checkpoints_dir()
        owner_dir = None if user_id == "admin" else (user_id or "anonymous")
        checkpoint_dir = checkpoints_root if owner_dir is None else os.path.join(checkpoints_root, owner_dir)

        # Determine checkpoint name
        if request.path is not None:
            # Named save - use provided path
            checkpoint_name = request.path
        else:
            # Ephemeral save - generate unique temp name
            checkpoint_name = f"_ephemeral_{uuid.uuid4().hex[:8]}"

        use_per_expert_lora = bool(request.use_per_expert_lora)
        train_mlp = bool(getattr(getattr(session, "lora_config", None), "train_mlp", False))
        if session.backend == "megatron":
            # vLLM MoE LoRA expects per-expert MLP weights. If MLP LoRA was trained, we must
            # export in per-expert format regardless of the caller-provided flag.
            if train_mlp:
                if not use_per_expert_lora:
                    logger.info(
                        "[save_weights_for_sampler] forcing use_per_expert_lora=True because train_mlp=True"
                    )
                use_per_expert_lora = True
            else:
                if use_per_expert_lora:
                    logger.info(
                        "[save_weights_for_sampler] forcing use_per_expert_lora=False because train_mlp=False"
                    )
                use_per_expert_lora = False

        print(f"[DEBUG _do_save_weights_for_sampler] calling save_weights_for_sampler", flush=True)
        # Save weights
        save_path = await training_engine.save_weights_for_sampler(
            session=session,
            checkpoint_name=checkpoint_name,
            checkpoint_base_dir=checkpoint_dir,
            use_per_expert_lora=use_per_expert_lora,
        )
        print(f"[DEBUG _do_save_weights_for_sampler] save_path={save_path}", flush=True)

        tinker_uri = f"tinker://{session.model_id}/{checkpoint_name}"
        mint_uri = f"mint://{session.model_id}/{checkpoint_name}"
        path_uri = tinker_uri if prefer_tinker else mint_uri

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

            import time

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

            try:
                from ..backend.session_index_store import add_sampler_to_session, upsert_sampler_index

                created_at = datetime.now().isoformat()
                add_sampler_to_session(
                    session_id=session.session_id,
                    sampler_id=sampling_session_id,
                    user_id=user_id,
                    created_at=created_at,
                )

                upsert_sampler_index(
                    {
                        "sampler_id": sampling_session_id,
                        "session_id": session.session_id,
                        "base_model": base_model,
                        "user_id": user_id,
                        "created_at": created_at,
                        "source_type": "checkpoint",
                        "model_id": session.model_id,
                        "checkpoint_name": checkpoint_name,
                        "model_path_raw": tinker_uri,
                    }
                )
            except Exception:
                pass

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


def _owner_visible(request_user_id: str | None, owner: str | None) -> bool:
    if request_user_id is None:
        return True
    if request_user_id == "admin":
        return True
    return bool(owner) and owner == request_user_id


@router.get("/training_runs/{training_run_id}", response_model=TrainingRun)
async def get_training_run(training_run_id: str, http_request: Request) -> TrainingRun:
    request_user_id = _get_user_id(http_request)
    info = None

    if training_manager is not None:
        session = training_manager.get_session(training_run_id)
        if session is None:
            session = _restore_training_session(training_run_id)
        if session is not None:
            info = _session_info_from_live(session)

    if info is None:
        try:
            from ..backend.training_session_store import get_training_session_info

            info = get_training_session_info(training_run_id)
        except Exception:
            info = None

    if not isinstance(info, dict):
        raise HTTPException(status_code=404, detail=f"Training run '{training_run_id}' not found")

    if not _owner_visible(request_user_id, info.get("user_id")):
        raise HTTPException(status_code=404, detail=f"Training run '{training_run_id}' not found")

    if "model_id" not in info:
        info = dict(info)
        info["model_id"] = training_run_id

    return _training_run_from_info(info)


@router.get("/training_runs", response_model=TrainingRunsResponse)
async def list_training_runs(limit: int = 20, offset: int = 0, http_request: Request = None) -> TrainingRunsResponse:
    request_user_id = _get_user_id(http_request) if http_request else None
    infos_by_id: dict[str, dict] = {}

    if training_manager is not None:
        for session in training_manager.list_sessions():
            infos_by_id[session.model_id] = _session_info_from_live(session)

    try:
        from ..backend.training_session_store import list_training_sessions

        for info in list_training_sessions():
            model_id = info.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                continue
            if model_id in infos_by_id:
                existing = infos_by_id[model_id]
                if not existing.get("user_id") and info.get("user_id"):
                    existing["user_id"] = info.get("user_id")
                if not existing.get("created_at") and info.get("created_at"):
                    existing["created_at"] = info.get("created_at")
                if not existing.get("user_metadata") and info.get("user_metadata"):
                    existing["user_metadata"] = info.get("user_metadata")
                continue
            infos_by_id[model_id] = info
    except Exception:
        pass

    infos = [
        info for info in infos_by_id.values() if _owner_visible(request_user_id, info.get("user_id"))
    ]

    infos.sort(key=lambda x: str(x.get("model_id") or ""))
    infos.sort(
        key=lambda x: (
            str(x.get("created_at") or ""),
            int(x.get("model_seq_id") or 0),
        ),
        reverse=True,
    )

    total_count = len(infos)
    if offset < 0:
        offset = 0
    if limit < 0:
        limit = 0
    page = infos[offset : offset + limit]

    return TrainingRunsResponse(
        training_runs=[_training_run_from_info(info) for info in page],
        cursor=Cursor(offset=offset, limit=limit, total_count=total_count),
    )


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
async def get_info(request: GetInfoRequest, http_request: Request) -> GetInfoResponse:
    """Get model info (tinker client compatible endpoint).

    Returns model architecture, tokenizer, and LoRA configuration.
    """
    from ..gateway import forward_json, remote_training_model, upstream_for_alias

    session = None
    if training_manager is not None:
        session = training_manager.get_session(request.model_id)
        if session is None:
            session = _restore_training_session(request.model_id)

    if session is None:
        remote = remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}"
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/get_info",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=30.0,
                )
            except Exception:
                logger.exception("Upstream get_info failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} get_info failed")
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return GetInfoResponse.model_validate(resp.json())

    if training_manager is None:
        raise HTTPException(status_code=503, detail="Training manager not initialized")

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
