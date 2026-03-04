"""Weight and state routes for saving/loading model weights and checkpoints.

Endpoints:
- POST /save_weights: Save model weights (currently LoRA-only; legacy)
- POST /save_state: Save checkpoint for training resume (LoRA-only, includes optimizer state)
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
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse

from ..backend.future_store import future_store
from ..checkpoints import (
    CHECKPOINTS_DIR,
    create_checkpoint_archive,
    resolve_checkpoint_path,
    safe_extract_checkpoint_archive,
    validate_checkpoint_dir,
)
from ..models.types import (
    CheckpointInfo,
    CheckpointUploadResponse,
    CheckpointsListResponse,
    LoadStateRequest,
    SaveStateRequest,
    UntypedAPIFuture,
)
from ..model_access_control import can_access_model, get_access_denied_error
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


def _resolve_mint_path(mint_uri: str, *, user_id: str | None) -> str:
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
    return resolve_checkpoint_path(mint_uri, user_id=user_id)


def _to_mint_path(model_id: str, checkpoint_name: str) -> str:
    """Convert to mint://{model_id}/ URI (legacy format)."""
    return f"mint://{model_id}/{checkpoint_name}"


# =============================================================================
# POST /save_weights - async (legacy)
# =============================================================================


@router.post("/save_weights", response_model=UntypedAPIFuture)
async def save_weights(
    request: SaveStateRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save training checkpoint (Tinker TrainingClient.save_state).

    Tinker SDK calls POST /api/v1/save_weights for TrainingClient.save_state(...).
    This must produce a training checkpoint (weights + optimizer state).
    """
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    session = training_manager.get_session(request.model_id) if training_manager is not None else None

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
                    path=http_request.url.path,
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=30.0,
                )
            except Exception:
                logger.exception("Upstream save_weights failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} save_weights failed")

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(status_code=502, detail="Upstream save_weights returned invalid request_id")
            return UntypedAPIFuture(
                request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
            )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)
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
        future_store.mark_queued(request_id, meta={"op": "weights.save_weights", "model_id": request.model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="weights.save_weights",
            request_json=request_json,
            user_id=user_id,
            webhook_url=webhook_url,
            extra={"prefer_tinker": bool(prefer_tinker)},
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue save_weights request: {e}")

    return UntypedAPIFuture(request_id=request_id)


# =============================================================================
# POST /save_state - async (training checkpoint: includes optimizer state)
# =============================================================================


@router.post("/save_state", response_model=UntypedAPIFuture)
async def save_state(
    request: SaveStateRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save model state to checkpoint.

    This endpoint produces a training checkpoint intended for resume, including optimizer state.
    """
    from ..gateway import (
        encode_request_id,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    session = training_manager.get_session(request.model_id) if training_manager is not None else None
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
                    path="/api/v1/save_state",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=30.0,
                )
            except Exception:
                logger.exception("Upstream save_state failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} save_state failed")

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(status_code=502, detail="Upstream save_state returned invalid request_id")
            return UntypedAPIFuture(
                request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
            )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)
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
        future_store.mark_queued(request_id, meta={"op": "weights.save_state", "model_id": request.model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="weights.save_state",
            request_json=request_json,
            user_id=user_id,
            webhook_url=webhook_url,
            extra={"prefer_tinker": bool(prefer_tinker)},
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue save_state request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_save_state(
    request_id: str,
    request: SaveStateRequest,
    user_id: str | None = None,
    webhook_url: str | None = None,
    prefer_tinker: bool = False,
) -> None:
    """Background task to save state.

    Storage schema: /checkpoints/{owner_id}/{model_id}/{checkpoint_name}/
    Also registers the model for sampling via multi-LoRA engine.
    """
    import json

    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        checkpoint_name = request.path.strip() if request.path is not None else ""
        if checkpoint_name:
            if checkpoint_name in (".", "..") or "/" in checkpoint_name or "\\" in checkpoint_name:
                raise ValueError(f"Invalid checkpoint name: {request.path!r}")
        else:
            checkpoint_name = f"ckpt_{uuid.uuid4().hex[:12]}"

        owner_dir = user_id or "anonymous"
        save_path = os.path.join(CHECKPOINTS_DIR, owner_dir, session.model_id, checkpoint_name)

        logger.info(f"[{session.model_id}] Saving state to: {save_path}")

        # Save training checkpoint on worker, returns path
        abs_path = await training_engine.save_weights(session, save_path)

        # Save ownership metadata (for user-scoped checkpoint API)
        # Note: Directory is created by Ray Worker on GPU node, but shared filesystem
        # sync may not be complete yet. Create directory on API server to ensure it exists.
        os.makedirs(save_path, exist_ok=True)

        from ..checkpoints import checkpoint_has_optimizer_state, write_checkpoint_metadata

        optimizer_present = bool(checkpoint_has_optimizer_state(save_path))
        if not optimizer_present:
            raise RuntimeError(
                f"save_state must produce optimizer artifacts, but none found under: {save_path}"
            )

        metadata = {
            "checkpoint_id": checkpoint_name,
            "owner_id": user_id,
            "model_id": session.model_id,
            "model_name": session.base_model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "step": session.current_step,
            "checkpoint_type": "training",
            "optimizer_present": optimizer_present,
            "backend": session.backend,
            "type": "training",
        }
        write_checkpoint_metadata(save_path, metadata)

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

        from ..client_compat import checkpoint_uri

        mint_path = _to_mint_path(session.model_id, checkpoint_name)
        tinker_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=True,
            checkpoint_type="training",
        )
        selected_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=prefer_tinker,
            checkpoint_type="training",
        )

        # Include state_dict metadata in response for verification (e.g., checking MLP modules)
        # Keys are JSON-serializable, tensors are not
        state_dict_keys = []  # state_dict not available in path-based flow

        future_store.resolve(request_id, {
            "checkpoint_id": checkpoint_name,
            "path": selected_path,
            "mint_path": mint_path,
            "tinker_path": tinker_path,
            "filesystem_path": save_path,
            "type": "save_weights",
            "sampling_registered": sampling_registered,
            "checkpoint_type": "training",
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
                result={"checkpoint_id": checkpoint_name, "step": session.current_step},
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


async def _do_save_weights(
    request_id: str,
    request: SaveStateRequest,
    user_id: str | None = None,
    webhook_url: str | None = None,
    prefer_tinker: bool = False,
) -> None:
    """Background task to save sampler-only LoRA weights (legacy /save_weights).

    Storage schema: /checkpoints/{owner_id}/{model_id}/{checkpoint_name}/
    Also registers the model for sampling via multi-LoRA engine.
    """
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        checkpoint_name = request.path.strip() if request.path is not None else ""
        if checkpoint_name:
            if checkpoint_name in (".", "..") or "/" in checkpoint_name or "\\\\" in checkpoint_name:
                raise ValueError(f"Invalid checkpoint name: {request.path!r}")
        else:
            checkpoint_name = f"ckpt_{uuid.uuid4().hex[:12]}"

        owner_dir = user_id or "anonymous"
        checkpoint_base_dir = os.path.join(CHECKPOINTS_DIR, owner_dir)
        save_path = os.path.join(checkpoint_base_dir, session.model_id, checkpoint_name)

        logger.info(f"[{session.model_id}] Saving sampler weights to: {save_path}")

        abs_path = await training_engine.save_weights_for_sampler(
            session=session,
            checkpoint_name=checkpoint_name,
            checkpoint_base_dir=checkpoint_base_dir,
            use_per_expert_lora=False,
        )

        os.makedirs(save_path, exist_ok=True)

        from ..checkpoints import checkpoint_has_optimizer_state, write_checkpoint_metadata

        if checkpoint_has_optimizer_state(save_path):
            raise RuntimeError(
                f"save_weights must not produce optimizer artifacts, but found some under: {save_path}"
            )

        metadata = {
            "checkpoint_id": checkpoint_name,
            "owner_id": user_id,
            "model_id": session.model_id,
            "model_name": session.base_model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "step": session.current_step,
            "checkpoint_type": "sampler",
            "optimizer_present": False,
            "backend": session.backend,
            "type": "sampler",
        }
        write_checkpoint_metadata(save_path, metadata)

        sampling_registered = False
        base_model = session.base_model
        if inference_manager is not None and base_model is not None:
            try:
                multi_lora_engine = await inference_manager.get_engine_for_model(base_model)

                existing_lora_id = await multi_lora_engine.registry.get_lora_id(session.model_id)
                if existing_lora_id is not None:
                    await multi_lora_engine.remove_session(session.model_id)

                await multi_lora_engine.add_lora_for_session_from_path(
                    sampling_session_id=session.model_id,
                    lora_path=abs_path,
                )
                try:
                    inference_manager.register_multi_lora_session(
                        session.model_id, base_model=base_model
                    )
                except ValueError:
                    pass
                sampling_registered = True
                logger.info(f"[{session.model_id}] Registered for sampling (path={abs_path})")
            except Exception as reg_err:
                logger.warning(f"[{session.model_id}] Could not register for sampling: {reg_err}")

        from ..client_compat import checkpoint_uri

        mint_path = _to_mint_path(session.model_id, checkpoint_name)
        tinker_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=True,
            checkpoint_type="sampler",
        )
        selected_path = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=prefer_tinker,
            checkpoint_type="sampler",
        )

        future_store.resolve(
            request_id,
            {
                "checkpoint_id": checkpoint_name,
                "path": selected_path,
                "mint_path": mint_path,
                "tinker_path": tinker_path,
                "filesystem_path": save_path,
                "type": "save_weights",
                "sampling_registered": sampling_registered,
                "checkpoint_type": "sampler",
            },
        )

        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_COMPLETED,
                user_id=user_id,
                session_id=session.model_id,
                task_name=f"Training {session.base_model}",
                task_type="training",
                model_name=session.base_model,
                result={"checkpoint_id": checkpoint_name, "step": session.current_step},
            )

    except Exception as e:
        logger.error(f"[save_weights] Failed: {e}", exc_info=True)
        future_store.fail(request_id, str(e))

        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_FAILED,
                user_id=user_id,
                session_id=request.model_id,
                task_name="Save weights",
                task_type="training",
                model_name=None,
                error=str(e),
            )


# =============================================================================
# POST /load_state - async
# =============================================================================


@router.post("/load_state", response_model=UntypedAPIFuture)
@router.post("/load_weights", response_model=UntypedAPIFuture)  # SDK alias
async def load_state(
    request: LoadStateRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Load model state from checkpoint."""
    from ..gateway import (
        encode_request_id,
        forward_file,
        forward_json,
        remote_training_model,
        upstream_for_alias,
    )

    session = training_manager.get_session(request.model_id) if training_manager is not None else None
    remote = None if session is not None else remote_training_model(request.model_id)
    if remote is not None:
        upstream_alias, base_model = remote
        upstream = upstream_for_alias(upstream_alias)
        if upstream is None:
            raise HTTPException(status_code=500, detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}")

        user_data = _get_user_data(http_request)
        if not can_access_model(base_model, user_data):
            raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

        user_id = _get_user_id(http_request)
        incoming_headers = dict(http_request.headers)
        json_body = request.model_dump()
        if request.path.startswith(("tinker://", "mint://", "ckpt_")):
            local_path = resolve_checkpoint_path(request.path, user_id=user_id)
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
                json_body["path"] = ckpt_id

        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path=http_request.url.path,
                incoming_headers=incoming_headers,
                json_body=json_body,
                timeout_s=30.0,
            )
        except Exception:
            logger.exception("Upstream load_state failed: %s", upstream_alias)
            raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} load_state failed")

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream load_state returned invalid request_id")
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream_alias, upstream_request_id=upstream_request_id)
        )

    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    user_id = _get_user_id(http_request)
    if request.optimizer:
        try:
            from ..checkpoints import validate_checkpoint_load_contract

            load_path = _resolve_mint_path(request.path, user_id=user_id)
            validate_checkpoint_load_contract(load_path, load_optimizer=True)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_checkpoint_for_optimizer_restore",
                    "error": str(e),
                    "path": request.path,
                },
            ) from e

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
        future_store.mark_queued(request_id, meta={"op": "weights.load_state", "model_id": request.model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op="weights.load_state",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
        )
    except Exception as e:
        capacity_manager.release_all(request_id)
        if created:
            future_store.cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue load_state request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_load_state(
    request_id: str, request: LoadStateRequest, user_id: str | None
) -> None:
    """Background task to load state."""
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        session = training_manager.get_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        # Resolve path
        load_path = _resolve_mint_path(request.path, user_id=user_id)
        if user_id and user_id != "admin":
            load_real = os.path.realpath(load_path)
            checkpoints_real = os.path.realpath(CHECKPOINTS_DIR)
            allowed_real = os.path.realpath(os.path.join(CHECKPOINTS_DIR, user_id))
            if load_real.startswith(checkpoints_real + os.sep) and not load_real.startswith(
                allowed_real + os.sep
            ):
                raise PermissionError("Access denied")

        logger.info(f"[{session.model_id}] Loading state from: {load_path}")

        if request.optimizer:
            from ..checkpoints import validate_checkpoint_load_contract

            validate_checkpoint_load_contract(load_path, load_optimizer=True)

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
# POST /checkpoints/upload - upload tar.gz archive for resume
# =============================================================================


@router.post("/checkpoints/upload", response_model=CheckpointUploadResponse)
async def upload_checkpoint_archive(
    http_request: Request,
    file: UploadFile = File(...),
) -> CheckpointUploadResponse:
    """Upload a tar.gz checkpoint archive and register it for resume.

    Stores extracted checkpoint under /checkpoints/{owner}/{checkpoint_id}/ with metadata.json.
    Returns a checkpoint identifier usable by load_state/create_model_from_state.
    """
    import json
    import tempfile

    user_id = _get_user_id(http_request)
    owner_dir = user_id or "anonymous"

    checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
    parent_dir = os.path.join(CHECKPOINTS_DIR, owner_dir)
    final_dir = os.path.join(parent_dir, checkpoint_id)
    tmp_dir = final_dir + ".tmp"
    tmp_archive: str | None = None

    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(final_dir):
        raise HTTPException(status_code=409, detail="Checkpoint already exists")

    try:
        os.makedirs(tmp_dir, exist_ok=False)

        fd, tmp_archive = tempfile.mkstemp(
            dir=parent_dir,
            prefix=f"upload_{checkpoint_id}_",
            suffix=".tar.gz",
        )
        os.close(fd)

        chunk_size = 8 * 1024 * 1024
        with open(tmp_archive, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)

        safe_extract_checkpoint_archive(tmp_archive, tmp_dir)

        # Determine checkpoint class (training vs sampler).
        # Rules:
        # - If metadata.json declares a checkpoint_type, it must match the artifacts.
        # - Otherwise, infer from presence of optimizer artifacts.
        from ..checkpoints import checkpoint_has_optimizer_state

        inferred_type = "training" if checkpoint_has_optimizer_state(tmp_dir) else "sampler"

        existing_meta: dict | None = None
        existing_meta_path = os.path.join(tmp_dir, "metadata.json")
        if os.path.exists(existing_meta_path):
            try:
                with open(existing_meta_path) as f:
                    existing_meta = json.load(f)
            except Exception:
                existing_meta = None

        declared_type = None
        if isinstance(existing_meta, dict):
            declared_type = existing_meta.get("checkpoint_type") or existing_meta.get("type")
        if declared_type is not None and declared_type not in ("training", "sampler"):
            raise HTTPException(status_code=400, detail=f"Invalid checkpoint_type in metadata.json: {declared_type!r}")

        checkpoint_type = declared_type or inferred_type
        if declared_type is not None and declared_type != inferred_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Checkpoint metadata declares checkpoint_type={declared_type!r}, "
                    f"but artifacts look like {inferred_type!r}"
                ),
            )

        validate_checkpoint_dir(tmp_dir, checkpoint_type=checkpoint_type)

        # Optional metadata extraction from checkpoint files
        model_id = None
        model_name = None
        step = None
        try:
            meta_path = os.path.join(tmp_dir, "training_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                step = meta.get("current_step", step)
        except Exception:
            pass
        try:
            adapter_cfg_path = os.path.join(tmp_dir, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                with open(adapter_cfg_path) as f:
                    cfg = json.load(f)
                model_name = cfg.get("base_model_name_or_path") or cfg.get("base_model") or model_name
        except Exception:
            pass
        if isinstance(existing_meta, dict):
            model_id = existing_meta.get("model_id") or model_id
            model_name = existing_meta.get("model_name") or model_name
            step = existing_meta.get("step") if step is None else step

        metadata = {
            "checkpoint_id": checkpoint_id,
            "owner_id": user_id,
            "model_id": model_id,
            "model_name": model_name,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "step": step,
            "checkpoint_type": checkpoint_type,
            "optimizer_present": checkpoint_type == "training",
            "backend": None,
            "type": checkpoint_type,
        }
        with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        os.rename(tmp_dir, final_dir)
        return CheckpointUploadResponse(checkpoint_id=checkpoint_id, path=checkpoint_id)
    except ValueError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        if tmp_archive is not None:
            try:
                os.unlink(tmp_archive)
            except OSError:
                pass


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
    owner_dir = user_id or "anonymous"
    from ..client_compat import checkpoint_uri, prefer_tinker_uri

    prefer_tinker = prefer_tinker_uri(request)

    candidate_paths: list[str] = [
        os.path.join(CHECKPOINTS_DIR, owner_dir, model_id),
        os.path.join(CHECKPOINTS_DIR, model_id),  # legacy (pre user-scoping)
    ]
    if user_id == "admin":
        candidate_paths = [os.path.join(CHECKPOINTS_DIR, model_id)]
        try:
            for owner in os.listdir(CHECKPOINTS_DIR):
                candidate_paths.append(os.path.join(CHECKPOINTS_DIR, owner, model_id))
        except OSError:
            pass

    if not any(os.path.exists(p) for p in candidate_paths):
        raise HTTPException(
            status_code=404, detail=f"No checkpoints found for model '{model_id}'"
        )

    checkpoints = []
    seen: set[str] = set()
    for checkpoints_path in candidate_paths:
        if not os.path.isdir(checkpoints_path):
            continue
        for name in os.listdir(checkpoints_path):
            ckpt_path = os.path.join(checkpoints_path, name)
            if not os.path.isdir(ckpt_path):
                continue
            key = f"{checkpoints_path}:{name}"
            if key in seen:
                continue
            seen.add(key)

            metadata_path = os.path.join(ckpt_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue  # refuse unauthenticated legacy dirs
            try:
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                continue

            if metadata.get("model_id") != model_id:
                continue
            if user_id != "admin" and metadata.get("owner_id") != user_id:
                continue

            # Try to parse step from directory name
            step = None
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                except (IndexError, ValueError):
                    pass

            created_at = datetime.fromtimestamp(os.path.getctime(ckpt_path)).isoformat()
            checkpoint_type = metadata.get("checkpoint_type")
            if checkpoint_type not in ("training", "sampler"):
                continue

            created_at = metadata.get("created_at") or created_at
            try:
                created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_time = datetime.fromtimestamp(os.path.getctime(ckpt_path))

            checkpoint_id = (
                f"weights/{name}" if checkpoint_type == "training" else f"sampler_weights/{name}"
            )
            tinker_path = checkpoint_uri(
                model_id,
                name,
                prefer_tinker=True,
                checkpoint_type=checkpoint_type,
            )
            path_uri = checkpoint_uri(
                model_id,
                name,
                prefer_tinker=prefer_tinker,
                checkpoint_type=checkpoint_type,
            )

            checkpoints.append(
                CheckpointInfo(
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=checkpoint_type,
                    time=created_time,
                    tinker_path=tinker_path,
                    path=path_uri,
                    step=step,
                    created_at=created_at,
                )
            )

    # Also include uploaded checkpoints stored as /checkpoints/{owner}/{checkpoint_id}/ if metadata.model_id matches.
    owner_roots: list[str]
    if user_id == "admin":
        try:
            owner_roots = [
                os.path.join(CHECKPOINTS_DIR, d)
                for d in os.listdir(CHECKPOINTS_DIR)
                if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))
            ]
        except OSError:
            owner_roots = []
    else:
        owner_roots = [os.path.join(CHECKPOINTS_DIR, owner_dir)]

    for root in owner_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if not name.startswith("ckpt_"):
                continue
            ckpt_path = os.path.join(root, name)
            if not os.path.isdir(ckpt_path):
                continue
            metadata_path = os.path.join(ckpt_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            try:
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                continue
            if metadata.get("model_id") != model_id:
                continue
            if user_id != "admin" and metadata.get("owner_id") != user_id:
                continue
            created_at = datetime.fromtimestamp(os.path.getctime(ckpt_path)).isoformat()
            checkpoint_type = metadata.get("checkpoint_type")
            if checkpoint_type not in ("training", "sampler"):
                continue

            created_at = metadata.get("created_at") or created_at
            try:
                created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_time = datetime.fromtimestamp(os.path.getctime(ckpt_path))

            checkpoint_id = (
                f"weights/{name}" if checkpoint_type == "training" else f"sampler_weights/{name}"
            )
            tinker_path = checkpoint_uri(
                model_id,
                name,
                prefer_tinker=True,
                checkpoint_type=checkpoint_type,
            )
            path_uri = checkpoint_uri(
                model_id,
                name,
                prefer_tinker=prefer_tinker,
                checkpoint_type=checkpoint_type,
            )
            checkpoints.append(
                CheckpointInfo(
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=checkpoint_type,
                    time=created_time,
                    tinker_path=tinker_path,
                    path=path_uri,
                    step=None,
                    created_at=created_at,
                )
            )

    # Sort by step (descending)
    checkpoints.sort(key=lambda x: x.step or 0, reverse=True)

    return CheckpointsListResponse(
        model_id=model_id,
        checkpoints=checkpoints,
    )


# =============================================================================
# DELETE /training_runs/{model_id}/checkpoints/{checkpoint_id}
# =============================================================================


def _split_tinker_checkpoint_id(checkpoint_id: str) -> tuple[str, str | None]:
    # Tinker canonical checkpoint IDs include an explicit kind prefix:
    # - weights/<name> -> training
    # - sampler_weights/<name> -> sampler
    parts = checkpoint_id.split("/")
    if len(parts) == 2 and parts[0] in ("weights", "sampler_weights") and parts[1]:
        return parts[1], ("training" if parts[0] == "weights" else "sampler")
    return checkpoint_id, None


@router.delete("/training_runs/{model_id}/checkpoints/{checkpoint_id:path}")
async def delete_checkpoint(model_id: str, checkpoint_id: str, request: Request):
    """Delete a specific checkpoint.

    Ownership verified via metadata.json (admin can delete all).
    """
    user_id = _get_user_id(request)
    owner_dir = user_id or "anonymous"
    checkpoint_name, expected_type = _split_tinker_checkpoint_id(checkpoint_id)
    candidates = [
        os.path.join(CHECKPOINTS_DIR, owner_dir, model_id, checkpoint_name),
        os.path.join(CHECKPOINTS_DIR, model_id, checkpoint_name),  # legacy
        os.path.join(
            CHECKPOINTS_DIR, owner_dir, checkpoint_name
        ),  # uploaded /checkpoints/{owner}/{ckpt_id}
    ]
    if user_id == "admin":
        candidates.append(os.path.join(CHECKPOINTS_DIR, checkpoint_name))
        try:
            for owner in os.listdir(CHECKPOINTS_DIR):
                candidates.append(os.path.join(CHECKPOINTS_DIR, owner, model_id, checkpoint_name))
                candidates.append(os.path.join(CHECKPOINTS_DIR, owner, checkpoint_name))
        except OSError:
            pass

    ckpt_path = next((p for p in candidates if os.path.isdir(p)), None)
    if ckpt_path is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    metadata_path = os.path.join(ckpt_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        import json

        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception:
        raise HTTPException(status_code=403, detail="Access denied")

    if metadata.get("model_id") != model_id:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
    if user_id != "admin" and metadata.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if expected_type is not None and metadata.get("checkpoint_type") != expected_type:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    shutil.rmtree(ckpt_path)

    logger.info(f"[{model_id}] Deleted checkpoint: {checkpoint_id}")

    return {"status": "deleted", "checkpoint_id": checkpoint_id}


# =============================================================================
# GET /training_runs/{model_id}/checkpoints/{checkpoint_id}/archive
# =============================================================================


@router.get("/training_runs/{model_id}/checkpoints/{checkpoint_id:path}/archive")
async def download_checkpoint_archive(
    model_id: str,
    checkpoint_id: str,
    request: Request,
    direct: bool = False,
):
    """Download checkpoint as tar.gz archive.

    Uses subprocess tar+gzip for true streaming without loading into memory.
    Essential for large checkpoints (7GB+).
    Ownership verified via metadata.json (admin can download all).
    """
    import subprocess

    user_id = _get_user_id(request)
    owner_dir = user_id or "anonymous"
    checkpoint_name, expected_type = _split_tinker_checkpoint_id(checkpoint_id)
    candidates = [
        os.path.join(CHECKPOINTS_DIR, owner_dir, model_id, checkpoint_name),
        os.path.join(CHECKPOINTS_DIR, model_id, checkpoint_name),  # legacy
        os.path.join(
            CHECKPOINTS_DIR, owner_dir, checkpoint_name
        ),  # uploaded /checkpoints/{owner}/{ckpt_id}
    ]
    if user_id == "admin":
        candidates.append(os.path.join(CHECKPOINTS_DIR, checkpoint_name))
        try:
            for owner in os.listdir(CHECKPOINTS_DIR):
                candidates.append(os.path.join(CHECKPOINTS_DIR, owner, model_id, checkpoint_name))
                candidates.append(os.path.join(CHECKPOINTS_DIR, owner, checkpoint_name))
        except OSError:
            pass

    ckpt_path = next((p for p in candidates if os.path.isdir(p)), None)
    if ckpt_path is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    metadata_path = os.path.join(ckpt_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        import json

        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception:
        raise HTTPException(status_code=403, detail="Access denied")

    if metadata.get("model_id") != model_id:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")
    if user_id != "admin" and metadata.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if expected_type is not None and metadata.get("checkpoint_type") != expected_type:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{checkpoint_id}' not found")

    # Tinker SDK expects this endpoint to respond with 302 + Location.
    # It does not follow redirects automatically; it treats Location as a signed URL.
    if not direct:
        from ..client_compat import is_tinker_sdk_user_agent

        if is_tinker_sdk_user_agent(request.headers.get("user-agent")):
            from ..config import config
            from ..download_tokens import make_archive_download_token

            secret = (config.token_secret_key or config.api_key or "").strip()
            direct_url_obj = request.url.include_query_params(direct="1")
            if secret:
                token, exp = make_archive_download_token(
                    secret=secret,
                    user_id=user_id,
                    model_id=model_id,
                    checkpoint_id=checkpoint_id,
                    ttl_s=15 * 60,
                )
                direct_url_obj = direct_url_obj.include_query_params(download_token=token)
                expires = datetime.fromtimestamp(exp, tz=timezone.utc)
            else:
                expires = datetime.now(timezone.utc) + timedelta(minutes=15)
            expires_header = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
            return RedirectResponse(
                url=str(direct_url_obj),
                status_code=302,
                headers={"Expires": expires_header},
            )

    def stream_tar_gz():
        """Stream tar.gz via subprocess to avoid memory explosion."""
        # Run tar in parent directory, archive the checkpoint_id folder
        parent_dir = os.path.dirname(ckpt_path)
        proc = subprocess.Popen(
            ["tar", "czf", "-", checkpoint_name],
            cwd=parent_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            while chunk := proc.stdout.read(65536):
                yield chunk
        finally:
            proc.stdout.close()
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"tar exited with code {returncode}")

    safe_checkpoint_id = checkpoint_id.replace("/", "_")
    filename = f"{model_id}_{safe_checkpoint_id}.tar.gz"
    logger.info(f"[{model_id}] Streaming checkpoint archive: {checkpoint_id}")

    return StreamingResponse(
        stream_tar_gz(),
        media_type="application/gzip",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )
