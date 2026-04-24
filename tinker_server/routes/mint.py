from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from ..auth_identity import can_bypass_ownership
from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..backend.future_store import future_store
from ..backend.mintx_ops import interpolate_checkpoints_to_dir
from ..checkpoints import (
    MIRROR_STATUS_PENDING,
    begin_async_checkpoint_mirror,
    build_persistent_cache_dir,
    ensure_checkpoint_path_allowed,
    get_persistent_cache_dir,
    materialize_persistent_checkpoint,
    read_checkpoint_metadata,
    resolve_checkpoint_path,
    write_checkpoint_metadata,
)
from ..client_compat import checkpoint_uri
from ..checkpoint_index import (
    CheckpointAlreadyExistsError,
    CheckpointAlreadyFailedError,
    CheckpointAlreadyUploadingError,
    claim_checkpoint_publication,
    mark_checkpoint_failed,
)
from ..logging_context import classify_failure_reason, set_request_id
from ..model_access_control import can_access_model, get_access_denied_error
from ..queue_priority import merge_queue_priority_extra
from ..models.mint_types import (
    ForwardBackwardReverseKLRequest,
    InterpolateCheckpointsRequest,
    MintCreateActionSessionRequest,
    MintCreateActionSessionResponse,
    MintDeleteActionSessionResponse,
    VLAActRequest,
    VLADatum,
    VLATrainStepRequest,
)
from ..models.types import (
    ActRequest,
    Datum,
    ForwardBackwardInput,
    TrainStepRequest,
    UntypedAPIFuture,
)
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


def _resolve_checkpoint_for_user(
    path: str,
    *,
    user_id: str | None,
    is_admin: bool,
    owner_id: str | None = None,
) -> str:
    owner_scope = owner_id if is_admin else user_id
    try:
        resolved = resolve_checkpoint_path(path, user_id=owner_scope, is_admin=is_admin)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if path.startswith("ckpt_") and resolved == path:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    try:
        ensure_checkpoint_path_allowed(resolved, user_id=owner_scope, is_admin=is_admin)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return materialize_persistent_checkpoint(resolved)


def _can_bypass_checkpoint_ownership(request: Request) -> bool:
    return can_bypass_ownership(request)


def _resolve_checkpoint_for_request(path: str, request: Request, *, owner_id: str | None = None) -> str:
    return _resolve_checkpoint_for_user(
        path,
        user_id=_get_user_id(request),
        is_admin=_can_bypass_checkpoint_ownership(request),
        owner_id=owner_id,
    )


def _infer_base_model_from_checkpoint_for_request(
    model_path: str,
    request: Request,
    *,
    owner_id: str | None = None,
) -> str:
    try:
        return _infer_base_model_from_checkpoint(
            model_path,
            user_id=(owner_id if _can_bypass_checkpoint_ownership(request) else _get_user_id(request)),
            is_admin=_can_bypass_checkpoint_ownership(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _session_field(session: object, key: str, default=None):
    if isinstance(session, dict):
        return session.get(key, default)
    return getattr(session, key, default)


async def _claim_sampler_checkpoint_or_raise(
    *,
    owner_id: str | None,
    model_id: str,
    raw_checkpoint_id: str,
    model_name: str | None,
    checkpoint_created_at: str,
    retry: bool,
) -> str | None:
    try:
        return await claim_checkpoint_publication(
            owner_id=owner_id,
            model_id=model_id,
            raw_checkpoint_id=raw_checkpoint_id,
            checkpoint_type="sampler",
            storage_root=get_persistent_cache_dir(),
            model_name=model_name,
            checkpoint_created_at=checkpoint_created_at,
            retry=retry,
        )
    except (CheckpointAlreadyUploadingError, CheckpointAlreadyExistsError, CheckpointAlreadyFailedError) as e:
        raise RuntimeError(str(e)) from e


async def _mark_checkpoint_failed_safe(ckpt_id: str | None, *, fail_reason: str) -> None:
    try:
        await mark_checkpoint_failed(ckpt_id, fail_reason=fail_reason)
    except Exception:
        logger.exception(
            "[mint.checkpoint_index] mark_failed failed ckpt_id=%s fail_reason=%s",
            ckpt_id,
            fail_reason,
        )


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

def _vla_token_stats(data: list[VLADatum]) -> tuple[int, int]:
    total_tokens = 0
    max_seq_len = 0
    for item in data:
        seq_len = 0
        for chunk in item.observation.model_input.chunks:
            if chunk.type == "encoded_text":
                seq_len += len(chunk.tokens)
        target_tokens = item.supervision.get("target_tokens")
        if target_tokens is not None:
            target_shape = list(target_tokens.shape)
            if len(target_shape) != 1:
                raise ValueError(f"target_tokens must be rank-1, got shape={target_shape}")
            seq_len += int(target_shape[0])
        total_tokens += seq_len
        if seq_len > max_seq_len:
            max_seq_len = seq_len
    return total_tokens, max_seq_len


def _lower_vla_datum(item: VLADatum) -> Datum:
    if "state" in item.supervision:
        raise ValueError("VLADatum.supervision must not contain 'state'; use observation.state")
    return Datum(
        model_input=item.observation.model_input,
        loss_fn_inputs={
            "state": item.observation.state,
            **item.supervision,
        },
    )


def _lower_vla_train_step_request(request: VLATrainStepRequest) -> TrainStepRequest:
    return TrainStepRequest(
        model_id=request.model_id,
        seq_id=request.seq_id,
        adam_params=request.adam_params,
        forward_backward_input=ForwardBackwardInput(
            data=[_lower_vla_datum(item) for item in request.data],
            loss_fn=request.loss_fn,
            loss_fn_config=request.loss_fn_config,
        ),
    )
@router.post("/action_sessions", response_model=MintCreateActionSessionResponse)
async def create_action_session(
    request: MintCreateActionSessionRequest,
    http_request: Request,
) -> MintCreateActionSessionResponse:
    if action_session_manager is None:
        raise HTTPException(status_code=503, detail="Action session manager not initialized")

    user_id = _get_user_id(http_request)
    base_model = request.base_model
    if not base_model and request.model_path:
        base_model = _infer_base_model_from_checkpoint_for_request(
            request.model_path,
            http_request,
            owner_id=request.owner_id,
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
        model_path = _resolve_checkpoint_for_request(model_path, http_request, owner_id=request.owner_id)

    try:
        action_session_id = await action_session_manager.create_session(  # type: ignore[attr-defined]
            session_id=request.session_id,
            action_session_seq_id=request.action_session_seq_id,
            base_model=base_model,
            model_path=model_path,
            user_id=user_id,
        )
    except RuntimeError as exc:
        message = str(exc)
        if 'pinned node capacity check failed' in message:
            raise HTTPException(status_code=503, detail=message) from exc
        raise
    return MintCreateActionSessionResponse(action_session_id=str(action_session_id))


@router.post("/action_sessions/{action_session_id}/act", response_model=UntypedAPIFuture)
async def act(
    action_session_id: str,
    request: VLAActRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    if action_session_manager is None:
        raise HTTPException(status_code=503, detail="Action session manager not initialized")

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    queued_request = ActRequest(
        action_session_id=action_session_id,
        seq_id=request.seq_id,
        observation=request.observation.model_input,
        extra_inputs={"state": request.observation.state},
        temperature=request.temperature,
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


@router.post("/vla/train_step", response_model=UntypedAPIFuture)
async def vla_train_step(
    request: VLATrainStepRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    route_start_s = time.perf_counter()
    from . import training as training_routes

    session = None
    if training_manager is not None:
        get_session = getattr(training_manager, "get_session", None)
        if callable(get_session):
            session = get_session(request.model_id)
    if session is None:
        session = await _get_route_training_store_info(request.model_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    try:
        _, max_seq_len = _vla_token_stats(request.data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    base_model = str(_session_field(session, "base_model", "") or "")
    backend = str(_session_field(session, "backend", "unknown") or "unknown")
    max_model_len = training_routes._get_max_model_len(base_model)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {base_model}"
            ),
        )

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
    inflight_marked = False
    try:
        if training_manager is not None:
            training_manager.mark_inflight(request.model_id, +1)
            inflight_marked = True
        scheduler_extra = training_routes._build_training_scheduler_extra(
            session=session,
            model_id=request.model_id,
            training_op="train_step",
            seq_id=request.seq_id,
        )
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta={"op": "mint.vla.train_step", "model_id": request.model_id},
        )
        await training_routes._enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="mint.vla.train_step",
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="mint.vla.train_step",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=scheduler_extra,
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            training_manager.mark_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue VLA train_step request: {e}")

    return UntypedAPIFuture(request_id=request_id)


@router.delete("/action_sessions/{action_session_id}", response_model=MintDeleteActionSessionResponse)
async def delete_action_session(action_session_id: str) -> MintDeleteActionSessionResponse:
    if action_session_manager is None:
        raise HTTPException(status_code=503, detail="Action session manager not initialized")

    await action_session_manager.shutdown_session(action_session_id)  # type: ignore[attr-defined]
    return MintDeleteActionSessionResponse(action_session_id=action_session_id)

async def _get_route_training_store_info(model_id: str) -> dict | None:
    from ..routes.training import _get_training_route_session_info

    info = await _get_training_route_session_info(model_id)
    if isinstance(info, dict):
        return info
    if training_manager is not None:
        return None

    try:
        from ..backend.training_session_store import async_get_training_session_info

        store_info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e
    return store_info if isinstance(store_info, dict) else None


async def _protect_training_session_enqueue_window(session_info: dict) -> None:
    from ..routes.training import _protect_training_session_enqueue_window as _training_protect

    await _training_protect(session_info)


def _build_mint_future_meta(
    *,
    op: str,
    model_id: str | None = None,
    session_info: dict | None = None,
    extra: dict | None = None,
) -> dict[str, object]:
    meta: dict[str, object] = {
        "op": str(op),
        "queue_state": "queued",
        "stage": "queued",
        "queued_at": time.time(),
    }
    if model_id:
        meta["model_id"] = str(model_id)
    if isinstance(session_info, dict):
        session_id = session_info.get("session_id")
        base_model = session_info.get("base_model")
        backend = session_info.get("backend")
        if session_id:
            meta["session_id"] = str(session_id)
        if base_model:
            meta["base_model"] = str(base_model)
        if backend:
            meta["backend"] = str(backend)
    if isinstance(extra, dict):
        meta.update(extra)
    return meta


@router.post("/checkpoints/interpolate", response_model=UntypedAPIFuture)
async def interpolate_checkpoints(
    request: InterpolateCheckpointsRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    user_id = _get_user_id(http_request)
    request = request.model_copy(
        update={
            "source_paths": [
                _resolve_checkpoint_for_request(path, http_request, owner_id=request.owner_id)
                for path in request.source_paths
            ]
        }
    )
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
            meta=_build_mint_future_meta(
                op="mint.interpolate_checkpoints",
                extra={
                    "checkpoint_count": len(request.source_paths),
                    "output_path": request.output_path,
                },
            ),
        )
        await api_work_queue.enqueue(
            request_id=request_id,
            op="mint.interpolate_checkpoints",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
            extra=merge_queue_priority_extra(request=http_request),
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
) -> None:
    claimed_ckpt_id: str | None = None
    mirror_started = False
    set_request_id(request_id)
    try:
        resolved_sources = list(request.source_paths)
        output_checkpoint_type = str(request.output_checkpoint_type or "sampler")
        if output_checkpoint_type != "sampler":
            raise ValueError(
                f"output_checkpoint_type={output_checkpoint_type!r} is not supported; only 'sampler' is allowed"
            )

        # validate metadata/model lineage before choosing output location
        from ..backend.mintx_ops import _validate_source_metadata

        model_id, model_name, _backend, _first_meta = _validate_source_metadata(resolved_sources)
        checkpoint_name = request.output_path.strip() if request.output_path else f"mintx-{uuid.uuid4().hex[:12]}"
        if checkpoint_name in (".", "..") or "/" in checkpoint_name or "\\" in checkpoint_name:
            raise ValueError(f"Invalid output_path: {request.output_path!r}")

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        claimed_ckpt_id = await _claim_sampler_checkpoint_or_raise(
            owner_id=user_id,
            model_id=model_id,
            raw_checkpoint_id=checkpoint_name,
            model_name=model_name,
            checkpoint_created_at=created_at,
            retry=bool(request.retry),
        )

        save_path = build_persistent_cache_dir(
            user_id=user_id,
            model_id=model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type=output_checkpoint_type,
        )
        artifacts = interpolate_checkpoints_to_dir(
            source_paths=resolved_sources,
            coefficients=request.coefficients,
            output_dir=save_path,
            checkpoint_name=checkpoint_name,
            user_id=user_id,
            output_checkpoint_type=output_checkpoint_type,
        )

        metadata = read_checkpoint_metadata(save_path)
        metadata["created_at"] = created_at
        metadata["ckpt_id"] = claimed_ckpt_id
        write_checkpoint_metadata(save_path, metadata)

        persistent_path = begin_async_checkpoint_mirror(
            save_path,
            user_id=user_id,
            model_id=model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type=output_checkpoint_type,
        )
        mirror_started = True
        path_uri = checkpoint_uri(
            model_id,
            checkpoint_name,
            prefer_tinker=False,
            checkpoint_type=output_checkpoint_type,
        )
        await future_store.async_resolve(
            request_id,
            {
                "checkpoint_id": checkpoint_name,
                "checkpoint_record_id": claimed_ckpt_id,
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
        if not mirror_started:
            await _mark_checkpoint_failed_safe(claimed_ckpt_id, fail_reason="upload_error")
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
    info = await _get_route_training_store_info(request.model_id)
    if not isinstance(info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    await _protect_training_session_enqueue_window(info)
    try:
        _token_count, max_seq_len = _reverse_kl_token_stats(request.data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    from ..routes.training import _get_max_model_len

    base_model = str(info.get("base_model") or "")
    max_model_len = _get_max_model_len(base_model)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {base_model}"
            ),
        )

    user_id = _get_user_id(http_request)
    resolved_reference_path = _resolve_checkpoint_for_request(
        request.reference_model_path,
        http_request,
        owner_id=request.owner_id,
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
            meta=_build_mint_future_meta(
                op="mint.forward_backward_reverse_kl",
                model_id=request.model_id,
                session_info=info,
            ),
        )
        await api_work_queue.enqueue(
            request_id=request_id,
            op="mint.forward_backward_reverse_kl",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
            extra=merge_queue_priority_extra(request=http_request),
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
        await future_store.async_resolve(request_id, result)
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


async def _do_vla_train_step(
    request_id: str,
    request: VLATrainStepRequest,
    user_id: str | None,
) -> None:
    from . import training as training_routes

    try:
        internal_request = _lower_vla_train_step_request(request)
    except Exception as e:
        logger.exception(
            "[mint.vla.train_step] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_vla_observation_and_supervision_shapes",
        )
        await future_store.async_fail(request_id, str(e))
        if training_manager is not None:
            training_manager.mark_inflight(request.model_id, -1)
        return

    await training_routes._do_train_step(request_id, internal_request, user_id)
