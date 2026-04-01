from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..auth_identity import is_admin_request
from ..backend.future_store import future_store
from ..backend.mintx_ops import interpolate_checkpoints_to_dir
from ..checkpoints import (
    MIRROR_STATUS_PENDING,
    begin_async_checkpoint_mirror,
    build_persistent_cache_dir,
    ensure_checkpoint_path_allowed,
    materialize_persistent_checkpoint,
    resolve_checkpoint_path,
)
from ..client_compat import checkpoint_uri
from ..logging_context import classify_failure_reason, set_request_id
from ..model_access_control import can_access_model, get_access_denied_error
from ..models.mint_types import (
    ForwardBackwardReverseKLRequest,
    InterpolateCheckpointsRequest,
    MintActRequest,
    MintCreateActionSessionRequest,
    MintCreateActionSessionResponse,
    MintDeleteActionSessionResponse,
)
from ..models.types import ActRequest, UntypedAPIFuture
from .service import _infer_base_model_from_checkpoint

logger = logging.getLogger(__name__)
router = APIRouter()

training_manager = None
training_engine = None
action_session_manager = None


def _get_user_data(request: Request) -> dict | None:
    return _request_user_data(request)


def _get_user_id(request: Request) -> str | None:
    return _request_user_id(request)


def _resolve_checkpoint_for_user(path: str, *, user_id: str | None, is_admin: bool) -> str:
    resolved = resolve_checkpoint_path(path, user_id=user_id, is_admin=is_admin)
    ensure_checkpoint_path_allowed(resolved, user_id=user_id, is_admin=is_admin)
    return materialize_persistent_checkpoint(resolved)


def _reverse_kl_token_stats(data: list) -> tuple[int, int]:
    from ..model_input_utils import flatten_encoded_text_chunks

    total_tokens = 0
    max_seq_len = 0
    for item in data:
        student_tokens = flatten_encoded_text_chunks(item.student_input.model_dump())
        ref_tokens = flatten_encoded_text_chunks(item.reference_input.model_dump())
        if not student_tokens:
            raise ValueError("student_input must contain at least one token")
        if not ref_tokens:
            raise ValueError("reference_input must contain at least one token")
        target_shape = list(item.target_tokens.shape)
        if len(target_shape) != 1:
            raise ValueError(f"target_tokens must be rank-1, got shape={target_shape}")
        weight_shape = list(item.weights.shape)
        if len(weight_shape) != 1 or weight_shape[0] != target_shape[0]:
            raise ValueError(
                f"weights shape {weight_shape} incompatible with target_tokens length {target_shape[0]}"
            )
        total_tokens += int(target_shape[0])
        max_seq_len = max(
            max_seq_len,
            len(student_tokens) + int(target_shape[0]) - 1,
            len(ref_tokens) + int(target_shape[0]) - 1,
        )
    return total_tokens, max_seq_len


@router.post("/action_sessions", response_model=MintCreateActionSessionResponse)
async def create_action_session(
    request: MintCreateActionSessionRequest,
    http_request: Request,
) -> MintCreateActionSessionResponse:
    if action_session_manager is None:
        raise HTTPException(status_code=503, detail="Action session manager not initialized")

    user_id = _get_user_id(http_request)
    is_admin = is_admin_request(http_request)
    base_model = request.base_model
    if not base_model and request.model_path:
        base_model = _infer_base_model_from_checkpoint(
            request.model_path,
            user_id=user_id,
            is_admin=is_admin,
        )
    if not base_model:
        raise HTTPException(status_code=422, detail="base_model is required")

    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(base_model=base_model, http_request=http_request)

    user_data = _get_user_data(http_request)
    if not can_access_model(base_model, user_data):
        raise HTTPException(status_code=403, detail=get_access_denied_error(base_model))

    model_path = request.model_path
    if model_path:
        model_path = _resolve_checkpoint_for_user(
            model_path,
            user_id=user_id,
            is_admin=is_admin,
        )

    action_session_id = await action_session_manager.create_session(  # type: ignore[attr-defined]
        session_id=request.session_id,
        action_session_seq_id=request.action_session_seq_id,
        base_model=base_model,
        model_path=model_path,
        user_id=user_id,
    )
    return MintCreateActionSessionResponse(action_session_id=str(action_session_id))


@router.post("/action_sessions/{action_session_id}/act", response_model=UntypedAPIFuture)
async def act(
    action_session_id: str,
    request: MintActRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    if "state" not in request.extra_inputs:
        raise HTTPException(status_code=400, detail="extra_inputs.state is required")
    if action_session_manager is None:
        raise HTTPException(status_code=503, detail="Action session manager not initialized")

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    queued_request = ActRequest(
        action_session_id=action_session_id,
        seq_id=request.seq_id,
        observation=request.observation,
        extra_inputs=request.extra_inputs,
    )
    request_json = queued_request.model_dump_json().encode("utf-8")
    request_id = f"act_{uuid.uuid4().hex}"
    reserve = await capacity_manager.async_try_reserve(
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
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta={"op": "mint.action.act", "action_session_id": action_session_id},
        )
        await api_work_queue.enqueue(
            request_id=request_id,
            op="mint.action.act",
            request_json=request_json,
            user_id=_get_user_id(http_request),
            webhook_url=None,
        )
    except Exception as e:
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue action act request: {e}")

    return UntypedAPIFuture(request_id=request_id)


@router.delete("/action_sessions/{action_session_id}", response_model=MintDeleteActionSessionResponse)
async def delete_action_session(action_session_id: str) -> MintDeleteActionSessionResponse:
    if action_session_manager is None:
        raise HTTPException(status_code=503, detail="Action session manager not initialized")

    await action_session_manager.shutdown_session(action_session_id)  # type: ignore[attr-defined]
    return MintDeleteActionSessionResponse(action_session_id=action_session_id)


@router.post("/checkpoints/interpolate", response_model=UntypedAPIFuture)
async def interpolate_checkpoints(
    request: InterpolateCheckpointsRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = await capacity_manager.async_try_reserve(
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
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta={"op": "mint.interpolate_checkpoints"},
        )
        await api_work_queue.enqueue(
            request_id=request_id,
            op="mint.interpolate_checkpoints",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
        )
    except Exception as e:
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue interpolate_checkpoints request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_interpolate_checkpoints(
    request_id: str,
    request: InterpolateCheckpointsRequest,
    user_id: str | None,
    is_admin: bool = False,
) -> None:
    set_request_id(request_id)
    try:
        resolved_sources = [
            _resolve_checkpoint_for_user(path, user_id=user_id, is_admin=is_admin)
            for path in request.source_paths
        ]
        # validate metadata/model lineage before choosing output location
        from ..backend.mintx_ops import _validate_source_metadata

        model_id, _model_name, _backend, _first_meta = _validate_source_metadata(resolved_sources)
        checkpoint_name = request.output_path.strip() if request.output_path else f"mintx-{uuid.uuid4().hex[:12]}"
        if checkpoint_name in (".", "..") or "/" in checkpoint_name or "\\" in checkpoint_name:
            raise ValueError(f"Invalid output_path: {request.output_path!r}")

        save_path = build_persistent_cache_dir(
            user_id=user_id,
            model_id=model_id,
            checkpoint_name=checkpoint_name,
        )
        artifacts = interpolate_checkpoints_to_dir(
            source_paths=resolved_sources,
            coefficients=request.coefficients,
            output_dir=save_path,
            checkpoint_name=checkpoint_name,
            user_id=user_id,
            output_checkpoint_type=request.output_checkpoint_type,
        )
        persistent_path = begin_async_checkpoint_mirror(
            save_path,
            user_id=user_id,
            model_id=model_id,
            checkpoint_name=checkpoint_name,
        )
        path_uri = checkpoint_uri(
            model_id,
            checkpoint_name,
            prefer_tinker=False,
            checkpoint_type="sampler",
        )
        future_store.resolve(
            request_id,
            {
                "path": path_uri,
                "checkpoint_type": artifacts.output_checkpoint_type,
                "source_paths": request.source_paths,
                "coefficients": [float(c) for c in request.coefficients],
                "has_rank_shards": artifacts.has_rank_shards,
                "filesystem_path": save_path,
                "persistent_filesystem_path": persistent_path,
                "mirror_status": MIRROR_STATUS_PENDING,
                "type": "mint_interpolate_checkpoints",
            },
        )
    except Exception as e:
        logger.exception(
            "[mint.interpolate_checkpoints] failed request_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_source_checkpoints_and_coefficients",
        )
        await future_store.async_fail(request_id, str(e))


@router.post("/forward_backward_reverse_kl", response_model=UntypedAPIFuture)
async def forward_backward_reverse_kl(
    request: ForwardBackwardReverseKLRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    if training_engine is None or training_manager is None:
        raise HTTPException(status_code=503, detail="Training engine not initialized")

    session = training_manager.get_session(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    try:
        _token_count, max_seq_len = _reverse_kl_token_stats(request.data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    from ..routes.training import _get_max_model_len

    max_model_len = _get_max_model_len(session.base_model)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {session.base_model}"
            ),
        )

    user_id = _get_user_id(http_request)
    is_admin = is_admin_request(http_request)
    resolved_reference_path = _resolve_checkpoint_for_user(
        request.reference_model_path,
        user_id=user_id,
        is_admin=is_admin,
    )
    request = request.model_copy(update={"reference_model_path": resolved_reference_path})

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve = await capacity_manager.async_try_reserve(
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
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta={"op": "mint.forward_backward_reverse_kl", "model_id": request.model_id},
        )
        await api_work_queue.enqueue(
            request_id=request_id,
            op="mint.forward_backward_reverse_kl",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
        )
    except Exception as e:
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue forward_backward_reverse_kl request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_forward_backward_reverse_kl(
    request_id: str,
    request: ForwardBackwardReverseKLRequest,
    user_id: str | None,
) -> None:
    set_request_id(request_id)
    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        session = training_manager.get_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        token_count, max_seq_len = _reverse_kl_token_stats(request.data)
        from ..routes.training import _get_max_model_len

        max_model_len = _get_max_model_len(session.base_model)
        if max_seq_len > max_model_len:
            raise RuntimeError(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} for model {session.base_model}"
            )

        t0 = time.time()
        logger.info(
            "[%s] forward_backward_reverse_kl start: backend=%s batch=%s tokens=%s max_len=%s reference=%s",
            session.model_id,
            session.backend,
            len(request.data),
            token_count,
            max_seq_len,
            request.reference_model_path,
        )
        result = await training_engine.forward_backward_reverse_kl(session, request)
        logger.info(
            "[%s] forward_backward_reverse_kl done: elapsed_s=%.3f",
            session.model_id,
            time.time() - t0,
        )
        future_store.resolve(request_id, result)
    except Exception as e:
        logger.exception(
            "[mint.forward_backward_reverse_kl] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_reference_checkpoint_and_reverse_kl_batch_shape",
        )
        await future_store.async_fail(request_id, str(e))
