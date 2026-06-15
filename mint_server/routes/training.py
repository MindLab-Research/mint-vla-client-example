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
- GET /models/{model_id}/session_guard_state: Get contamination/block guard state
- DELETE /models/{model_id}: Delete a model
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request

from ..auth_identity import can_bypass_ownership_user_data
from ..auth_identity import can_manage_system
from ..auth_identity import can_write
from ..auth_identity import get_apikey_id as _request_apikey_id
from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..backend.async_ray_control import async_get_ray_ref, async_lookup_actor_handle
from ..gateway_auth import GatewayAuthContext, build_billing_auth_context
from ..logging_context import (
    classify_failure_reason,
    get_current_traceparent,
    get_otel_tracer,
    run_async_with_otel_span,
    set_request_id,
    start_as_current_span,
    start_as_current_span_from_traceparent,
)

from ..backend.task_state_store import (
    FutureStatus,
    billing_observations_from_auth,
    billing_observations_from_input,
    task_futures,
)
from ..checkpoint_index import (
    CheckpointAlreadyExistsError,
    CheckpointAlreadyFailedError,
    CheckpointAlreadyUploadingError,
    claim_checkpoint_publication,
    mark_checkpoint_failed,
)
from ..checkpoints import (
    MIRROR_STATUS_PENDING,
    begin_async_checkpoint_mirror,
    build_ephemeral_checkpoint_dir,
    build_gateway_proxy_archive_path,
    build_persistent_cache_dir,
    checkpoint_has_optimizer_state,
    async_create_checkpoint_archive,
    ensure_checkpoint_path_allowed,
    get_ephemeral_checkpoints_dir,
    get_persistent_cache_dir,
    get_persistent_checkpoints_dir,
    materialize_persistent_checkpoint,
    resolve_checkpoint_path,
    validate_sampler_checkpoint_for_sampling,
    write_checkpoint_metadata,
)
from ..config import RAY_NAMESPACE, config as server_config
from ..model_access_control import can_access_model, get_access_denied_error
from ..queue_priority import merge_queue_priority_extra
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
from ..webhook import EventType, send_task_event

if TYPE_CHECKING:
    from ..backend.session_manager import SessionManager
    from ..backend.training_session_manager import TrainingSessionManager
    from ..backend.verl_training import VerlTrainingEngine

logger = logging.getLogger(__name__)
router = APIRouter()


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
    except (
        CheckpointAlreadyUploadingError,
        CheckpointAlreadyExistsError,
        CheckpointAlreadyFailedError,
    ) as e:
        raise RuntimeError(str(e)) from e


async def _mark_checkpoint_failed_safe(
    ckpt_id: str | None, *, fail_reason: str
) -> None:
    try:
        await mark_checkpoint_failed(ckpt_id, fail_reason=fail_reason)
    except Exception:
        logger.exception(
            "[training.checkpoint_index] mark_failed failed ckpt_id=%s fail_reason=%s",
            ckpt_id,
            fail_reason,
        )


# Execution-runtime references (left unbound in API workers).
training_manager: TrainingSessionManager | None = None
training_engine: VerlTrainingEngine | None = None
inference_manager: SessionManager | None = None  # For ephemeral flow


def _current_training_manager():
    try:
        from ..backend.execution_context import current_execution_context

        context = current_execution_context()
        if context is not None:
            return context.train_manager
    except Exception:
        pass
    return training_manager


def _current_training_engine():
    try:
        from ..backend.execution_context import current_execution_context

        context = current_execution_context()
        if context is not None:
            return context.train_engine
    except Exception:
        pass
    return training_engine


def _current_inference_manager():
    try:
        from ..backend.execution_context import current_execution_context

        context = current_execution_context()
        if context is not None:
            return context.inference_manager
    except Exception:
        pass
    return inference_manager


def _require_write_access(request: Request) -> None:
    if not can_write(request):
        raise HTTPException(status_code=403, detail="Write access required")


async def _mark_training_inflight(model_id: str, delta: int) -> None:
    from ..backend.training_session_store import async_mark_training_session_inflight

    await async_mark_training_session_inflight(model_id, delta)


async def _fail_future(request_id: str, error: str) -> None:
    async_fail = getattr(task_futures, "async_fail", None)
    if callable(async_fail):
        result = async_fail(request_id, error)
        if inspect.isawaitable(result):
            await result
        return
    fail = getattr(task_futures, "fail", None)
    if callable(fail):
        fail(request_id, error)
        return
    raise AttributeError("task_futures has neither async_fail nor fail")


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state (set by auth middleware)."""
    return _request_user_data(request)


def _get_user_id(request: Request) -> str | None:
    """Extract user_id from request state (set by auth middleware)."""
    return _request_user_id(request)


def _get_apikey_id(
    request: Request,
    *,
    gateway_auth: GatewayAuthContext | None = None,
) -> str | None:
    if gateway_auth is not None and gateway_auth.apikey_id:
        return str(gateway_auth.apikey_id)
    return _request_apikey_id(request)


def _build_training_usage_label(*, model: str, route: str) -> str:
    return f"model={model},route={route},dimension=train"


def _build_training_billing_observations(
    *,
    gateway_auth: dict | None,
    request_id: str,
    model: str,
    route: str,
    token_count: int,
) -> list[dict]:
    auth_ctx = GatewayAuthContext(**gateway_auth) if gateway_auth else None
    return billing_observations_from_auth(
        auth_ctx=auth_ctx,
        request_id=request_id,
        charge_item="training",
        quantity=int(token_count),
        unit="tokens",
        route=route,
        dimension="train",
        model=model,
    )


def _cleanup_generated_checkpoint_dir(path: str | None) -> None:
    if not path:
        return
    try:
        real = os.path.realpath(path)
    except Exception:
        return
    managed_roots = [
        os.path.realpath(get_ephemeral_checkpoints_dir()),
        os.path.realpath(get_persistent_cache_dir()),
        os.path.realpath(get_persistent_checkpoints_dir()),
    ]
    if not any(
        real == root or real.startswith(root + os.sep) for root in managed_roots
    ):
        logger.warning("Refusing to cleanup checkpoint outside managed roots: %s", path)
        return
    if os.path.isdir(real):
        shutil.rmtree(real, ignore_errors=True)


def _training_heartbeat_stale_timeout_s() -> float:
    raw = os.environ.get("MINT_TRAINING_HEARTBEAT_STALE_S", "300")
    try:
        return max(0.0, float(raw))
    except Exception:
        logger.warning(
            "Invalid MINT_TRAINING_HEARTBEAT_STALE_S=%r; defaulting to 300s", raw
        )
        return 300.0


async def _enqueue_training_request_with_trace(
    *,
    route_start_s: float,
    request_id: str,
    op: str,
    enqueue_coro,
    model_id: str | None = None,
    base_model: str | None = None,
    backend: str | None = None,
) -> Any:
    tracer = get_otel_tracer()
    future_ready_elapsed_ms = (time.perf_counter() - route_start_s) * 1000.0
    if tracer is None:
        return await enqueue_coro

    with tracer.start_as_current_span(f"{op}.enqueue") as span:
        span.set_attribute("component", "routes.training")
        span.set_attribute("op", str(op))
        span.set_attribute("request_id", str(request_id))
        if model_id:
            span.set_attribute("model_id", str(model_id))
        if base_model:
            span.set_attribute("base_model", str(base_model))
        if backend:
            span.set_attribute("backend", str(backend))
        span.add_event(
            "task_futures_ready",
            {
                "elapsed_ms": round(future_ready_elapsed_ms, 3),
                "route_elapsed_ms": round(future_ready_elapsed_ms, 3),
            },
        )
        enqueue_start_s = time.perf_counter()
        out = await enqueue_coro
        span.add_event(
            "enqueue_done",
            {
                "elapsed_ms": round(
                    (time.perf_counter() - enqueue_start_s) * 1000.0, 3
                ),
                "route_elapsed_ms": round(
                    (time.perf_counter() - route_start_s) * 1000.0, 3
                ),
            },
        )
        return out


async def _safe_update_training_meta(request_id: str, meta: dict[str, object]) -> None:
    try:
        await task_futures.async_update_meta(request_id, meta)
    except Exception:
        pass


def _training_model_work_domain_key(
    *, backend: str, base_model: str, model_id: str
) -> str:
    backend_value = str(backend or "").strip()
    base = str(base_model or "").strip()
    if backend_value in {"bumblebee", "megatron"} and base:
        from ..backend.model_actor_supervisor import domain_key_for_training_base_model

        return domain_key_for_training_base_model(base, backend=backend_value)
    if base:
        return f"training:{base}"
    return f"training_session:{model_id}"


async def _enqueue_training_model_work_route(
    *,
    route_start_s: float,
    request_id: str,
    op: str,
    request_json: bytes,
    user_id: str | None,
    apikey_id: str | None = None,
    webhook_url: str | None = None,
    extra: dict[str, Any],
    model_id: str,
    base_model: str,
    backend: str,
    queued_meta: dict[str, Any],
) -> dict[str, Any]:
    from ..backend.model_work_admission import enqueue_model_work
    from ..backend.model_work_scheduler import model_work_scheduler

    domain_key = _training_model_work_domain_key(
        backend=backend, base_model=base_model, model_id=model_id
    )
    affinity_group = f"training_session:{model_id}"
    result = await enqueue_model_work(
        request_id=request_id,
        op=op,
        request_json=request_json,
        user_id=user_id,
        apikey_id=apikey_id,
        throttle_principal=None,
        webhook_url=webhook_url,
        domain_key=domain_key,
        affinity_group=affinity_group,
        ordering_key=affinity_group,
        token_cost=1,
        assign=True,
        assign_max_items=1,
        extra={
            **dict(extra),
            "model_work_scheduler": True,
            "domain_key": domain_key,
            "affinity_group": affinity_group,
            "model_work_attempt_id": str(
                extra.get("model_work_attempt_id") or uuid.uuid4().hex
            ),
        },
        queued_meta=queued_meta,
        scheduler_client=model_work_scheduler,
        future_service_client=task_futures,
        trace_enqueue=_enqueue_training_request_with_trace,
        trace_kwargs={
            "route_start_s": route_start_s,
            "model_id": model_id,
            "base_model": base_model,
            "backend": backend,
        },
    )
    return result.scheduler_result.to_wire()


def _get_webhook_url(request: Request) -> str | None:
    """Extract webhook_url from request state (set by auth middleware)."""
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("webhook_url")
    return None


def _find_actor_handle(actor_name: str, namespace: str):
    from ..backend.model_actor_supervisor import get_model_actor_supervisor

    pool = get_model_actor_supervisor()
    for entry in pool.iter_entries():
        if (
            entry.actor_name == actor_name
            and entry.namespace == namespace
            and entry.actor_handle
        ):
            return entry.actor_handle
    return None


def _snapshot_from_training_session(model_id: str):
    manager = _current_training_manager()
    if manager is None:
        return None
    get_session = getattr(manager, "get_session", None)
    if not callable(get_session):
        return None
    session = get_session(model_id)
    if session is None:
        return None
    try:
        from ..backend.training_session_manager import TrainingSessionSnapshot

        lora_config = getattr(session, "lora_config", None)
        if lora_config is not None and hasattr(lora_config, "model_dump"):
            lora_config = lora_config.model_dump()
        return TrainingSessionSnapshot(
            model_id=str(getattr(session, "model_id", model_id) or model_id),
            session_id=str(getattr(session, "session_id", "") or ""),
            model_seq_id=int(getattr(session, "model_seq_id", 0) or 0),
            base_model=str(getattr(session, "base_model", "") or ""),
            backend=str(getattr(session, "backend", "peft") or "peft"),
            current_step=int(getattr(session, "current_step", 0) or 0),
            lora_config=lora_config,
            rollout_correction_config=getattr(
                session, "rollout_correction_config", None
            ),
            user_metadata=dict(getattr(session, "user_metadata", {}) or {}),
            learning_rate=float(getattr(session, "learning_rate", 1e-4) or 1e-4),
            metadata_version=max(1, int(getattr(session, "metadata_version", 1) or 1)),
        )
    except Exception:
        return None


def _get_training_snapshot(model_id: str):
    manager = _current_training_manager()
    if manager is None:
        return None
    get_snapshot = getattr(manager, "get_training_session_snapshot", None)
    if callable(get_snapshot):
        snapshot = get_snapshot(model_id)
        if snapshot is not None:
            return snapshot
    return _snapshot_from_training_session(model_id)


def _drop_local_training_session(model_id: str) -> None:
    manager = _current_training_manager()
    engine = _current_training_engine()
    if manager is not None:
        delete_session = getattr(manager, "delete_session", None)
        if callable(delete_session):
            try:
                delete_session(model_id)
            except Exception as e:
                logger.warning(
                    "Failed to delete stale local training session %s: %s", model_id, e
                )
    if engine is not None:
        getattr(engine, "_workers", {}).pop(model_id, None)
        getattr(engine, "_model_actor_supervisor_actor_names", {}).pop(model_id, None)


def _refresh_training_session_from_info_if_needed(
    model_id: str, info: dict, snapshot=None
):
    manager = _current_training_manager()
    engine = _current_training_engine()
    if manager is None or not isinstance(info, dict):
        return snapshot
    snap = snapshot or _get_training_snapshot(model_id)
    if snap is None:
        _restore_training_session_info_compat(info)
        return _get_training_snapshot(model_id)

    incoming_version = max(1, int(info.get("metadata_version") or 1))
    incoming_step = max(0, int(info.get("current_step") or 0))
    incoming_actor_name = str(info.get("actor_name") or "") or None
    incoming_namespace = str(info.get("namespace") or "") or None
    current_version = max(1, int(getattr(snap, "metadata_version", 1) or 1))
    current_step = max(0, int(getattr(snap, "current_step", 0) or 0))
    current_session = None
    if manager is not None:
        get_local_session = getattr(manager, "get_local_session", None)
        current_session = (
            get_local_session(model_id) if callable(get_local_session) else None
        )
    current_actor_name = str(getattr(current_session, "actor_name", "") or "") or None
    current_namespace = str(getattr(current_session, "namespace", "") or "") or None
    actor_binding_changed = (
        incoming_actor_name is not None and incoming_actor_name != current_actor_name
    ) or (incoming_namespace is not None and incoming_namespace != current_namespace)
    if actor_binding_changed and engine is not None:
        getattr(engine, "_workers", {}).pop(model_id, None)
        if incoming_actor_name is not None:
            getattr(engine, "_model_actor_supervisor_actor_names", {})[model_id] = (
                incoming_actor_name
            )
        else:
            getattr(engine, "_model_actor_supervisor_actor_names", {}).pop(
                model_id, None
            )
    if (
        incoming_version <= current_version
        and incoming_step <= current_step
        and not actor_binding_changed
    ):
        return snap
    _restore_training_session_info_compat(info)
    return _get_training_snapshot(model_id)


def _restore_training_session_info_compat(info: dict):
    manager = _current_training_manager()
    if manager is None:
        return None
    restore = getattr(manager, "restore_training_session_info", None)
    if callable(restore):
        return restore(info)

    get_session = getattr(manager, "get_session", None)
    create_session = getattr(manager, "create_session", None)
    if not callable(get_session) or not callable(create_session):
        return None

    model_id = str(info.get("model_id") or "")
    session_id = str(info.get("session_id") or "")
    base_model = str(info.get("base_model") or "")
    if not model_id or not session_id or not base_model:
        return None

    session = cast(Any, get_session(model_id))
    if session is None:
        session = cast(
            Any,
            create_session(
                model_id=model_id,
                session_id=session_id,
                model_seq_id=int(info.get("model_seq_id", 0) or 0),
                base_model=base_model,
                lora_config=None,
                rollout_correction_config=info.get("rollout_correction_config"),
                user_metadata=info.get("user_metadata") or {},
                user_id=info.get("user_id"),
                learning_rate=float(info.get("learning_rate", 1e-4) or 1e-4),
                metadata_version=max(1, int(info.get("metadata_version", 1) or 1)),
                materialization_state=info.get("materialization_state"),
                tokenizer_info=info.get("tokenizer_info")
                if isinstance(info.get("tokenizer_info"), dict)
                else None,
                tokenizer_identity=info.get("tokenizer_identity"),
                tokenizer_source_path=info.get("tokenizer_source_path"),
                actor_name=info.get("actor_name"),
                namespace=info.get("namespace"),
            ),
        )

    session.session_id = session_id
    session.model_seq_id = int(
        info.get("model_seq_id", getattr(session, "model_seq_id", 0)) or 0
    )
    session.base_model = base_model
    session.rollout_correction_config = info.get("rollout_correction_config")
    session.user_metadata = info.get("user_metadata") or {}
    session.user_id = info.get("user_id")
    session.learning_rate = float(
        info.get("learning_rate", getattr(session, "learning_rate", 1e-4)) or 1e-4
    )
    session.backend = str(
        info.get("backend", getattr(session, "backend", "peft")) or "peft"
    )
    session.current_step = int(
        info.get("current_step", getattr(session, "current_step", 0)) or 0
    )
    session.metadata_version = max(
        1,
        int(info.get("metadata_version", getattr(session, "metadata_version", 1)) or 1),
    )
    if hasattr(session, "materialization_state"):
        session.materialization_state = str(
            info.get("materialization_state")
            or getattr(session, "materialization_state", "ready")
        )
    if isinstance(info.get("tokenizer_info"), dict):
        session.tokenizer_info = dict(info.get("tokenizer_info") or {})
    if info.get("tokenizer_identity") is not None:
        session.tokenizer_identity = str(info.get("tokenizer_identity") or "") or None
    if info.get("tokenizer_source_path") is not None:
        session.tokenizer_source_path = (
            str(info.get("tokenizer_source_path") or "") or None
        )
    if info.get("actor_name") is not None:
        session.actor_name = str(info.get("actor_name") or "") or None
    if info.get("namespace") is not None:
        session.namespace = str(info.get("namespace") or "") or None
    return session


async def _restore_training_session(model_id: str):
    """Best-effort restore of a training session after API process restart."""
    engine = _current_training_engine()
    manager = _current_training_manager()
    if engine is None or manager is None:
        return None
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
        if not isinstance(info, dict):
            return None

        lora_cfg = None
        if info.get("lora_config"):
            lora_cfg = LoRAConfig(**info["lora_config"])

        get_local_session = getattr(manager, "get_local_session", None)
        session = (
            get_local_session(model_id)
            if callable(get_local_session)
            else manager.get_session(model_id)
        )
        created_session = False
        original_session_state = None
        if session is None:
            session = manager.create_session(
                model_id=model_id,
                session_id=str(info.get("session_id", "")),
                model_seq_id=int(info.get("model_seq_id", 0)),
                base_model=str(info.get("base_model", "")),
                lora_config=lora_cfg,
                rollout_correction_config=info.get("rollout_correction_config"),
                user_metadata=info.get("user_metadata") or {},
                user_id=info.get("user_id"),
                learning_rate=float(info.get("learning_rate", 1e-4)),
                metadata_version=max(1, int(info.get("metadata_version") or 1)),
                materialization_state=info.get("materialization_state"),
                tokenizer_info=info.get("tokenizer_info")
                if isinstance(info.get("tokenizer_info"), dict)
                else None,
                tokenizer_identity=info.get("tokenizer_identity"),
                tokenizer_source_path=info.get("tokenizer_source_path"),
                actor_name=info.get("actor_name"),
                namespace=info.get("namespace"),
            )
            created_session = True
        else:
            original_session_state = {
                "backend": session.backend,
                "created_at": session.created_at,
                "current_step": session.current_step,
                "is_active": session.is_active,
                "metadata_version": getattr(session, "metadata_version", 1),
                "materialization_state": getattr(
                    session, "materialization_state", "ready"
                ),
                "tokenizer_info": getattr(session, "tokenizer_info", None),
                "tokenizer_identity": getattr(session, "tokenizer_identity", None),
                "tokenizer_source_path": getattr(
                    session, "tokenizer_source_path", None
                ),
                "actor_name": getattr(session, "actor_name", None),
                "namespace": getattr(session, "namespace", None),
            }

        session.backend = str(info.get("backend", session.backend))
        last_activity_set = False
        try:
            raw_last_activity = info.get("last_activity")
            if raw_last_activity is not None:
                session.last_activity = float(raw_last_activity)
                last_activity_set = True
        except (TypeError, ValueError, OverflowError):
            last_activity_set = False
        created_at = info.get("created_at")
        if isinstance(created_at, str) and created_at:
            session.created_at = created_at
            if not last_activity_set:
                # Fall back to created_at for older store entries that predate
                # persisted activity timestamps. This preserves fail-closed
                # cleanup semantics without granting a fresh idle window.
                try:
                    session.last_activity = datetime.fromisoformat(
                        created_at
                    ).timestamp()
                    last_activity_set = True
                except (ValueError, OSError):
                    pass
        if not last_activity_set:
            session.last_activity = 0.0
        try:
            session.current_step = int(info.get("current_step", session.current_step))
        except Exception:
            pass
        if hasattr(session, "materialization_state"):
            session.materialization_state = str(
                info.get("materialization_state")
                or getattr(session, "materialization_state", "ready")
            )
        if isinstance(info.get("tokenizer_info"), dict):
            session.tokenizer_info = dict(info.get("tokenizer_info") or {})
        if info.get("tokenizer_identity") is not None:
            session.tokenizer_identity = (
                str(info.get("tokenizer_identity") or "") or None
            )
        if info.get("tokenizer_source_path") is not None:
            session.tokenizer_source_path = (
                str(info.get("tokenizer_source_path") or "") or None
            )
        if info.get("actor_name") is not None:
            session.actor_name = str(info.get("actor_name") or "") or None
        if info.get("namespace") is not None:
            session.namespace = str(info.get("namespace") or "") or None
        session.is_active = True

        actor_name = info.get("actor_name")
        if actor_name:
            namespace = str(info.get("namespace") or RAY_NAMESPACE)
            worker = _find_actor_handle(actor_name, namespace)
            if worker is None:
                try:
                    worker = await async_lookup_actor_handle(actor_name, namespace)
                except Exception:
                    worker = None
            if worker is None:
                if created_session:
                    manager.delete_session(model_id)
                elif original_session_state is not None:
                    session.backend = original_session_state["backend"]
                    session.created_at = original_session_state["created_at"]
                    session.current_step = original_session_state["current_step"]
                    session.is_active = original_session_state["is_active"]
                    session.metadata_version = original_session_state[
                        "metadata_version"
                    ]
                    if hasattr(session, "materialization_state"):
                        session.materialization_state = original_session_state[
                            "materialization_state"
                        ]
                        session.tokenizer_info = original_session_state[
                            "tokenizer_info"
                        ]
                        session.tokenizer_identity = original_session_state[
                            "tokenizer_identity"
                        ]
                        session.tokenizer_source_path = original_session_state[
                            "tokenizer_source_path"
                        ]
                        session.actor_name = original_session_state["actor_name"]
                        session.namespace = original_session_state["namespace"]
                return None
            getattr(engine, "_workers", {})[model_id] = worker
            getattr(engine, "_model_actor_supervisor_actor_names", {})[model_id] = (
                actor_name
            )

        return session
    except Exception as e:
        logger.exception(f"[{model_id}] restore_training_session failed: {e}")


async def _raise_if_local_model_id_exists(model_id: str) -> None:
    engine = _current_training_engine()
    manager = _current_training_manager()
    if engine is None or manager is None:
        return
    get_local_session = getattr(manager, "get_local_session", None)
    local_session = (
        get_local_session(model_id)
        if callable(get_local_session)
        else manager.get_session(model_id)
    )
    if local_session is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Model_id conflict: local model already exists: {model_id!r}",
        )
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(
            status_code=503, detail="Training session store unavailable"
        ) from e
    if isinstance(info, dict):
        raise HTTPException(
            status_code=409,
            detail=f"Model_id conflict: local model already exists: {model_id!r}",
        )


def _generate_model_id(session_id: str, model_seq_id: int) -> str:
    """Generate unique model_id from session_id and model_seq_id."""
    return f"{session_id}_{model_seq_id}"


def _session_info_from_live(session) -> dict:
    return {
        "model_id": session.model_id,
        "session_id": session.session_id,
        "model_seq_id": session.model_seq_id,
        "base_model": session.base_model,
        "lora_config": session.lora_config.model_dump()
        if session.lora_config
        else None,
        "user_metadata": session.user_metadata,
        "learning_rate": session.learning_rate,
        "current_step": session.current_step,
        "is_active": session.is_active,
        "created_at": session.created_at,
        "last_activity": getattr(session, "last_activity", None),
        "backend": session.backend,
        "user_id": session.user_id,
        "actor_name": getattr(session, "actor_name", None),
        "namespace": getattr(session, "namespace", None),
    }


async def _best_effort_delete_training_session(
    model_id: str,
    *,
    reason: str,
    allow_actor_shutdown: bool,
) -> bool:
    if training_engine is None or training_manager is None:
        return False

    try:
        fail_pending = getattr(
            task_futures, "async_fail_training_requests_for_model", None
        )
        if callable(fail_pending):
            failed_request_ids = await fail_pending(
                model_id,
                f"Training session terminated due to {reason}",
            )
            if failed_request_ids:
                logger.warning(
                    "[%s] failed pending training futures during stale cleanup (%s): request_ids=%s",
                    model_id,
                    reason,
                    failed_request_ids,
                )
    except Exception as e:
        logger.warning(
            "[%s] stale training cleanup aborted because pending future fail failed (%s): %s: %s",
            model_id,
            reason,
            type(e).__name__,
            e,
        )
        return False

    session = training_manager.get_session(model_id)
    restored = False
    if session is None:
        session = await _restore_training_session(model_id)
        restored = session is not None

    shutdown_attempted = False
    cleanup_ok = True
    if session is not None:
        if allow_actor_shutdown:
            try:
                shutdown_attempted = True
                await training_engine.shutdown_session(session)
            except Exception as e:
                cleanup_ok = False
                logger.warning(
                    "[%s] best-effort stale training cleanup shutdown failed (%s): %s: %s",
                    model_id,
                    reason,
                    type(e).__name__,
                    e,
                )
        else:
            logger.warning(
                "[%s] skipping actor shutdown during stale training cleanup (%s); "
                "restored=%s allow_actor_shutdown=%s",
                model_id,
                reason,
                restored,
                allow_actor_shutdown,
            )
            actor_name = str(getattr(session, "actor_name", "") or "")
            other_users = [
                mid
                for mid, bound_actor in getattr(
                    training_engine, "_model_actor_supervisor_actor_names", {}
                ).items()
                if bound_actor == actor_name and mid != model_id
            ]
            replacement_session = other_users[0] if other_users else None
            worker = getattr(training_engine, "_workers", {}).get(model_id)
            if worker is None:
                actor_name = str(getattr(session, "actor_name", "") or "")
                namespace = str(getattr(session, "namespace", "") or RAY_NAMESPACE)
                if actor_name:
                    try:
                        import ray

                        worker = await asyncio.to_thread(
                            ray.get_actor, actor_name, namespace=namespace
                        )
                    except Exception as e:
                        logger.warning(
                            "[%s] best-effort stale training cleanup rebind failed (%s): %s: %s",
                            model_id,
                            reason,
                            type(e).__name__,
                            e,
                        )
            delete_session = (
                getattr(worker, "delete_session", None) if worker is not None else None
            )
            delete_ok = worker is None and not str(
                getattr(session, "actor_name", "") or ""
            )
            if delete_session is not None:
                try:
                    await async_get_ray_ref(
                        delete_session.remote(model_id), timeout_s=30
                    )
                    delete_ok = True
                except Exception as e:
                    logger.warning(
                        "[%s] best-effort stale training cleanup remote delete failed (%s): %s: %s",
                        model_id,
                        reason,
                        type(e).__name__,
                        e,
                    )
            cleanup_ok = cleanup_ok and delete_ok
            getattr(training_engine, "_model_actor_supervisor_actor_names", {}).pop(
                model_id, None
            )
            getattr(training_engine, "_workers", {}).pop(model_id, None)
            session.is_active = False
            if actor_name:
                try:
                    from ..backend.model_actor_supervisor import (
                        get_model_actor_supervisor,
                    )

                    get_model_actor_supervisor().set_session(
                        actor_name, replacement_session
                    )
                except Exception:
                    pass

    if not cleanup_ok:
        return False

    try:
        training_manager.delete_session(model_id)
    except Exception:
        pass

    try:
        from ..backend.training_session_store import delete_training_session

        delete_training_session(model_id)
    except Exception as e:
        logger.warning(
            "[%s] best-effort stale training cleanup store delete failed (%s): %s: %s",
            model_id,
            reason,
            type(e).__name__,
            e,
        )

    try:
        from ..backend.model_actor_supervisor import get_model_actor_supervisor

        get_model_actor_supervisor().clear_session(model_id)
    except Exception:
        pass

    return session is not None or shutdown_attempted


async def cleanup_stale_training_sessions_once(
    *, stale_after_s: float | None = None
) -> list[str]:
    if stale_after_s is None:
        stale_after_s = _training_heartbeat_stale_timeout_s()
    if stale_after_s <= 0:
        return []
    if training_engine is None or training_manager is None:
        return []

    from ..backend.session_heartbeat_store import session_heartbeat_store

    try:
        from ..backend.training_session_store import async_list_training_sessions

        infos = await async_list_training_sessions()
    except Exception as e:
        logger.warning(
            "stale training cleanup skipped: failed to list TaskStateStore-backed training sessions: %s: %s",
            type(e).__name__,
            e,
        )
        return []

    cleaned: list[str] = []
    actor_refcounts: dict[str, int] = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        actor_name = str(info.get("actor_name") or "").strip()
        if actor_name:
            actor_refcounts[actor_name] = actor_refcounts.get(actor_name, 0) + 1

    for info in infos:
        if not isinstance(info, dict):
            continue
        model_id = str(info.get("model_id") or "").strip()
        session_id = str(info.get("session_id") or "").strip()
        actor_name = str(info.get("actor_name") or "").strip()
        if not model_id or not session_id:
            continue
        if not session_heartbeat_store.is_stale(session_id, float(stale_after_s)):
            continue
        try:
            allow_actor_shutdown = (
                bool(actor_name) and actor_refcounts.get(actor_name, 0) <= 1
            )
            deleted = await _best_effort_delete_training_session(
                model_id,
                reason=f"stale heartbeat (> {float(stale_after_s):.1f}s)",
                allow_actor_shutdown=allow_actor_shutdown,
            )
            if not deleted:
                continue
            cleaned.append(model_id)
            logger.warning(
                "[%s] auto-terminated stale training session: session_id=%s stale_after_s=%.1f "
                "allow_actor_shutdown=%s actor_name=%s actor_refcount=%s",
                model_id,
                session_id,
                float(stale_after_s),
                allow_actor_shutdown,
                actor_name or "<unknown>",
                actor_refcounts.get(actor_name, 0) if actor_name else 0,
            )
        except Exception as e:
            logger.warning(
                "[%s] stale training cleanup failed for session_id=%s: %s: %s",
                model_id,
                session_id,
                type(e).__name__,
                e,
            )
    return cleaned


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

    last_activity = info.get("last_activity")
    idle_for_s = info.get("idle_for_s")
    if idle_for_s is None and last_activity is not None:
        try:
            idle_for_s = max(0.0, time.time() - float(last_activity))
        except Exception:
            idle_for_s = None

    return TrainingRun(
        training_run_id=str(info.get("model_id", "")),
        base_model=str(info.get("base_model", "")),
        model_owner=str(info.get("user_id") or "anonymous"),
        is_lora=bool(is_lora),
        corrupted=False,
        lora_rank=lora_rank,
        last_request_time=str(
            info.get("last_request_time")
            or info.get("created_at")
            or datetime.now().isoformat()
        ),
        last_activity=float(last_activity) if last_activity is not None else None,
        idle_for_s=float(idle_for_s) if idle_for_s is not None else None,
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


def _validate_training_batch_has_explicit_loss_masks_or_422(data: list[Datum]) -> None:
    for item_index, datum in enumerate(data):
        loss_fn_inputs = datum.loss_fn_inputs or {}
        if any(key in loss_fn_inputs for key in ("loss_mask", "mask", "weights")):
            continue
        raise HTTPException(
            status_code=422, detail=f"Item {item_index} missing loss_mask/mask/weights"
        )


def _field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def _get_training_route_session_info(model_id: str) -> dict[str, Any] | None:
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(
            status_code=503, detail="Training session store unavailable"
        ) from e
    return info if isinstance(info, dict) else None


def _session_view_from_info(model_id: str, info: dict[str, Any]) -> Any:
    return SimpleNamespace(
        model_id=str(info.get("model_id") or model_id),
        session_id=str(info.get("session_id") or ""),
        base_model=str(info.get("base_model") or ""),
        backend=str(info.get("backend") or "peft"),
        user_id=info.get("user_id"),
        lora_config=None,
    )


def _has_training_worker_binding(model_id: str) -> bool:
    if training_engine is None:
        return False
    if model_id in getattr(training_engine, "_workers", {}):
        return True
    return model_id in getattr(
        training_engine, "_model_actor_supervisor_actor_names", {}
    )


async def _get_training_session_for_request(model_id: str):
    snapshot = _get_training_snapshot(model_id)
    info = await _get_training_route_session_info(model_id)
    if not isinstance(info, dict):
        _drop_local_training_session(model_id)
        return None, None

    snapshot = _refresh_training_session_from_info_if_needed(
        model_id, info, snapshot=snapshot
    )
    session = (
        training_manager.get_session(model_id) if training_manager is not None else None
    )
    if session is not None and _has_training_worker_binding(model_id):
        return session, snapshot

    restored = await _restore_training_session(model_id)
    if restored is not None:
        return restored, _get_training_snapshot(model_id) or snapshot
    return session, snapshot


async def _resolve_training_route_session(
    model_id: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    info = await _get_training_route_session_info(model_id)
    if isinstance(info, dict):
        return _session_view_from_info(model_id, info), info
    return None, None


def _ensure_route_session_info(
    *,
    model_id: str,
    session: Any | None,
    route_session_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(route_session_info, dict):
        return route_session_info
    if session is not None:
        return _session_info_from_live(session)
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


async def _protect_training_session_enqueue_window(
    session_info: dict[str, Any] | None,
) -> None:
    if not isinstance(session_info, dict):
        return
    session_id = str(session_info.get("session_id") or "").strip()
    if not session_id:
        return
    try:
        from ..backend.session_heartbeat_store import session_heartbeat_store

        await session_heartbeat_store.async_update(
            session_id=session_id, now=time.time()
        )
    except Exception as e:
        raise HTTPException(
            status_code=503, detail="Training heartbeat store unavailable"
        ) from e


def _normalize_megatron_scheduler_domain_key(base_model: str) -> str:
    hf_cache_pattern = r"models--([^/]+)--([^/]+)/snapshots"
    match = re.search(hf_cache_pattern, base_model)
    if match:
        _org, model = match.groups()
        model_name = model.lower().replace("-", "_").replace(".", "_")
    else:
        model_name = (
            base_model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        )
    return f"mint_megatron_{model_name}"


_TOKENIZER_METADATA_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def _infer_training_backend_for_base_model(base_model: str) -> str:
    from ..backend.model_registry import get_model_config

    cfg = get_model_config(base_model)
    training_backend = str(getattr(cfg, "training_backend", "mint_text") or "mint_text")
    if training_backend == "openpi_fast":
        return "openpi_fast"
    if training_backend == "openpi_pi05":
        return "openpi_pi05"
    from ..backend.verl_training import _select_moe_training_backend, _uses_distributed_training_backend

    if _uses_distributed_training_backend(base_model):
        return _select_moe_training_backend(base_model)
    return "peft"


def _supports_control_plane_tokenizer_metadata(backend: str) -> bool:
    return str(backend) in {"bumblebee", "megatron", "peft"}


def _resolve_local_tokenizer_source_path(base_model: str) -> str:
    candidate = str(base_model or "").strip()
    if not candidate:
        raise RuntimeError("base_model is empty")
    if os.path.exists(candidate):
        return os.path.realpath(candidate)
    resolver = getattr(training_engine, "_resolve_hf_model_path", None)
    if callable(resolver):
        resolved = resolver(candidate)
        resolved_path = (
            os.fspath(resolved) if isinstance(resolved, os.PathLike) else resolved
        )
        if (
            isinstance(resolved_path, str)
            and resolved_path
            and os.path.exists(resolved_path)
        ):
            return os.path.realpath(resolved_path)
    raise RuntimeError(
        f"Tokenizer source is not available on this API host for base_model {candidate!r}"
    )


def _public_load_metadata(meta: object) -> dict[str, object] | None:
    if not isinstance(meta, dict):
        return None
    allowed_keys = {
        "migration_mode",
        "migration_source_backend",
        "migration_target_backend",
        "optimizer_restored",
        "optimizer_reset",
        "requested_optimizer_restore",
    }
    public = {key: meta[key] for key in allowed_keys if key in meta}
    return public or None


def _tokenizer_identity_from_source_path(source_path: str) -> str:
    path = os.path.realpath(str(source_path))
    if os.path.isdir(path):
        candidates = []
        for name in _TOKENIZER_METADATA_FILES:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                candidates.append(candidate)
        if not candidates:
            candidates.append(path)
    else:
        candidates = [path]

    payload = []
    for candidate in candidates:
        stat = os.stat(candidate)
        payload.append(
            {
                "name": os.path.basename(candidate),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{path}#{digest}"


@lru_cache(maxsize=256)
def _load_tokenizer_info_from_local_source(
    source_path: str,
    tokenizer_identity: str,
    backend: str,
) -> dict[str, Any]:
    del tokenizer_identity
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source_path, trust_remote_code=True, local_files_only=True
    )
    info = {
        "vocab_size": getattr(tokenizer, "vocab_size", len(tokenizer)),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
    }
    if str(backend) in {"bumblebee", "megatron"}:
        info["vocab_size"] = len(tokenizer)
    return info


def _build_local_tokenizer_metadata(base_model: str, backend: str) -> dict[str, Any]:
    source_path = _resolve_local_tokenizer_source_path(base_model)
    tokenizer_identity = _tokenizer_identity_from_source_path(source_path)
    tokenizer_info = _load_tokenizer_info_from_local_source(
        source_path, tokenizer_identity, backend
    )
    return {
        "tokenizer_source_path": source_path,
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_info": dict(tokenizer_info),
    }


async def _collect_control_plane_tokenizer_metadata(session: Any) -> dict[str, Any]:
    engine = _current_training_engine()
    if engine is None:
        raise RuntimeError("Training engine not initialized")
    tokenizer_metadata: dict[str, Any] = {
        "tokenizer_info": dict(await engine.get_tokenizer_info(session))
    }
    try:
        local_metadata = await asyncio.to_thread(
            _build_local_tokenizer_metadata,
            str(getattr(session, "base_model", "") or ""),
            str(getattr(session, "backend", "") or ""),
        )
        tokenizer_metadata["tokenizer_identity"] = local_metadata.get(
            "tokenizer_identity"
        )
        tokenizer_metadata["tokenizer_source_path"] = local_metadata.get(
            "tokenizer_source_path"
        )
    except Exception as e:
        logger.info(
            "[tokenizer-metadata] local identity unavailable model_id=%s error_type=%s error=%s",
            str(getattr(session, "model_id", "")),
            type(e).__name__,
            e,
        )
    return tokenizer_metadata


def _next_training_session_metadata_version(session: Any) -> int:
    from ..backend.training_session_manager import TRAINING_SESSION_METADATA_VERSION

    current = max(
        int(
            getattr(session, "metadata_version", TRAINING_SESSION_METADATA_VERSION - 1)
            or 0
        ),
        TRAINING_SESSION_METADATA_VERSION - 1,
    )
    next_version = current + 1
    session.metadata_version = next_version
    return next_version


async def _best_effort_local_tokenizer_metadata_for_session(
    session: Any,
) -> dict[str, Any] | None:
    if not _supports_control_plane_tokenizer_metadata(
        str(getattr(session, "backend", "") or "")
    ):
        return None
    try:
        return await asyncio.to_thread(
            _build_local_tokenizer_metadata,
            str(getattr(session, "base_model", "") or ""),
            str(getattr(session, "backend", "") or ""),
        )
    except Exception as e:
        logger.info(
            "[tokenizer-metadata] local metadata unavailable model_id=%s error_type=%s error=%s",
            str(getattr(session, "model_id", "")),
            type(e).__name__,
            e,
        )
        return None


def _build_training_session_store_payload(
    *,
    session: Any,
    user_id: str | None,
    lora_config: Any,
    rollout_correction_config: dict[str, Any] | None,
    user_metadata: dict[str, Any] | None,
    actor_name: str | None,
    materialization_state: str,
    tokenizer_metadata: dict[str, Any] | None,
    metadata_version: int | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ..backend.training_session_manager import TRAINING_SESSION_METADATA_VERSION

    if lora_config is None:
        lora_payload = None
    elif hasattr(lora_config, "model_dump") and callable(
        getattr(lora_config, "model_dump")
    ):
        lora_payload = lora_config.model_dump()
    elif isinstance(lora_config, dict):
        lora_payload = dict(lora_config)
    else:
        lora_payload = dict(getattr(lora_config, "__dict__", {}))

    payload = {
        "model_id": str(getattr(session, "model_id", "") or ""),
        "session_id": str(getattr(session, "session_id", "") or ""),
        "model_seq_id": int(getattr(session, "model_seq_id", 0) or 0),
        "base_model": str(getattr(session, "base_model", "") or ""),
        "lora_config": lora_payload,
        "rollout_correction_config": rollout_correction_config,
        "user_metadata": dict(user_metadata or {}),
        "learning_rate": float(getattr(session, "learning_rate", 1e-4) or 1e-4),
        "current_step": int(getattr(session, "current_step", 0) or 0),
        "backend": str(getattr(session, "backend", "peft") or "peft"),
        "actor_name": actor_name,
        "namespace": str(getattr(session, "namespace", None) or RAY_NAMESPACE),
        "user_id": user_id,
        "created_at": getattr(session, "created_at", None),
        "last_activity": float(
            getattr(session, "last_activity", time.time()) or time.time()
        ),
        "metadata_version": max(
            TRAINING_SESSION_METADATA_VERSION,
            int(
                metadata_version
                if metadata_version is not None
                else getattr(
                    session, "metadata_version", TRAINING_SESSION_METADATA_VERSION
                )
            ),
        ),
        "materialization_state": materialization_state,
    }
    if isinstance(tokenizer_metadata, dict):
        payload["tokenizer_info"] = dict(tokenizer_metadata.get("tokenizer_info") or {})
        payload["tokenizer_identity"] = tokenizer_metadata.get("tokenizer_identity")
        payload["tokenizer_source_path"] = tokenizer_metadata.get(
            "tokenizer_source_path"
        )
    if isinstance(extra_fields, dict):
        payload.update(extra_fields)
    return payload


async def _materialize_training_session_for_stateful_use(session: Any) -> Any:
    from ..backend.training_session_manager import (
        MATERIALIZATION_STATE_FAILED,
        MATERIALIZATION_STATE_MATERIALIZING,
        MATERIALIZATION_STATE_READY,
    )
    from ..backend.training_session_store import async_upsert_training_session

    engine = _current_training_engine()
    if engine is None:
        raise RuntimeError("Training engine not initialized")

    state = str(
        getattr(session, "materialization_state", MATERIALIZATION_STATE_READY)
        or MATERIALIZATION_STATE_READY
    )
    if (
        bool(getattr(session, "is_active", False))
        and state == MATERIALIZATION_STATE_READY
    ):
        return session

    session.materialization_state = MATERIALIZATION_STATE_MATERIALIZING
    materializing_version = _next_training_session_metadata_version(session)
    await async_upsert_training_session(
        _build_training_session_store_payload(
            session=session,
            user_id=getattr(session, "user_id", None),
            lora_config=getattr(session, "lora_config", None),
            rollout_correction_config=getattr(
                session, "rollout_correction_config", None
            ),
            user_metadata=getattr(session, "user_metadata", None),
            actor_name=None,
            materialization_state=MATERIALIZATION_STATE_MATERIALIZING,
            tokenizer_metadata={
                "tokenizer_info": getattr(session, "tokenizer_info", None),
                "tokenizer_identity": getattr(session, "tokenizer_identity", None),
                "tokenizer_source_path": getattr(
                    session, "tokenizer_source_path", None
                ),
            },
            metadata_version=materializing_version,
        )
    )

    try:
        await engine.create_training_session(session)
        tokenizer_metadata = None
        if _supports_control_plane_tokenizer_metadata(
            str(getattr(session, "backend", "") or "")
        ):
            tokenizer_metadata = await _collect_control_plane_tokenizer_metadata(
                session
            )
            session.tokenizer_info = dict(
                tokenizer_metadata.get("tokenizer_info") or {}
            )
            session.tokenizer_identity = (
                str(tokenizer_metadata.get("tokenizer_identity") or "") or None
            )
            session.tokenizer_source_path = (
                str(tokenizer_metadata.get("tokenizer_source_path") or "") or None
            )
        actor_name = getattr(engine, "_model_actor_supervisor_actor_names", {}).get(
            session.model_id
        )
        session.actor_name = str(actor_name or "") or None
        session.namespace = RAY_NAMESPACE
        session.materialization_state = MATERIALIZATION_STATE_READY
        ready_version = _next_training_session_metadata_version(session)
        await async_upsert_training_session(
            _build_training_session_store_payload(
                session=session,
                user_id=getattr(session, "user_id", None),
                lora_config=getattr(session, "lora_config", None),
                rollout_correction_config=getattr(
                    session, "rollout_correction_config", None
                ),
                user_metadata=getattr(session, "user_metadata", None),
                actor_name=actor_name,
                materialization_state=MATERIALIZATION_STATE_READY,
                tokenizer_metadata=tokenizer_metadata,
                metadata_version=ready_version,
            )
        )
        manager = _current_training_manager()
        if manager is not None:
            manager.mark_persisted(session.model_id)
        return session
    except Exception as e:
        session.materialization_state = MATERIALIZATION_STATE_FAILED
        failed_version = _next_training_session_metadata_version(session)
        try:
            await async_upsert_training_session(
                _build_training_session_store_payload(
                    session=session,
                    user_id=getattr(session, "user_id", None),
                    lora_config=getattr(session, "lora_config", None),
                    rollout_correction_config=getattr(
                        session, "rollout_correction_config", None
                    ),
                    user_metadata=getattr(session, "user_metadata", None),
                    actor_name=None,
                    materialization_state=MATERIALIZATION_STATE_FAILED,
                    tokenizer_metadata={
                        "tokenizer_info": getattr(session, "tokenizer_info", None),
                        "tokenizer_identity": getattr(
                            session, "tokenizer_identity", None
                        ),
                        "tokenizer_source_path": getattr(
                            session, "tokenizer_source_path", None
                        ),
                    },
                    metadata_version=failed_version,
                    extra_fields={"materialization_error": f"{type(e).__name__}: {e}"},
                )
            )
        except Exception:
            pass
        raise


def _get_max_model_len(base_model: str | None) -> int:
    """Return the configured max_model_len for a supported model name.

    If the server cannot determine the model's max_model_len, fail fast rather
    than silently skipping the length gate.
    """
    if not base_model:
        raise HTTPException(
            status_code=500, detail="Training session missing base_model"
        )
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


def _validate_rollout_correction_config_or_400(
    *,
    base_model: str,
    rollout_correction_config: Any,
) -> None:
    if rollout_correction_config is None:
        return

    from ..backend.model_registry import get_model_config

    if not bool(get_model_config(base_model).is_moe):
        raise HTTPException(
            status_code=400,
            detail=(
                "rollout_correction_config is only supported on Megatron backend (MoE models); "
                f"base_model={base_model!r} is not configured as MoE"
            ),
        )

    try:
        cfg = rollout_correction_config.model_dump(exclude_none=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="rollout_correction_config must be a pydantic model compatible with .model_dump()",
        )

    try:
        import importlib

        verl_cfg_module = importlib.import_module("verl.trainer.config")
        VerlRolloutCorrectionConfig = getattr(
            verl_cfg_module, "RolloutCorrectionConfig"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "rollout_correction_config requires verl to be installed on the API server "
                f"(import failed: {type(e).__name__}: {e})"
            ),
        )

    try:
        VerlRolloutCorrectionConfig(**cfg)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rollout_correction_config for verl: {type(e).__name__}: {e}",
        )


def _validate_lora_rank_or_400(lora_config: LoRAConfig | None) -> None:
    if lora_config is None:
        return
    requested_rank = int(lora_config.rank)
    max_rank = int(server_config.max_lora_rank)
    if requested_rank > max_rank:
        raise HTTPException(
            status_code=400,
            detail=f"Requested LoRA rank {requested_rank} exceeds server max_lora_rank={max_rank}",
        )


def _build_training_scheduler_extra(
    *,
    session: Any,
    model_id: str,
    training_op: str,
    seq_id: int | None = None,
) -> dict[str, Any]:
    from ..backend.model_actor_supervisor import domain_key_for_training_base_model

    enabled = str(os.environ.get("MINT_SCHEDULER_ENABLE", "1")).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    backend = str(_field(session, "backend", "") or "unknown")
    base_model = str(_field(session, "base_model", "") or "")
    openpi_train_step = training_op == "train_step" and backend in {
        "openpi_fast",
        "openpi_pi05",
    }
    if openpi_train_step:
        enabled = True
    if backend in {"bumblebee", "megatron"} and base_model:
        domain_key = domain_key_for_training_base_model(base_model, backend=backend)
    else:
        domain_key = (
            domain_key_for_training_base_model(base_model)
            if base_model
            else f"training_session:{model_id}"
        )
    extra: dict[str, Any] = {
        "scheduler_enabled": bool(enabled),
        "scheduler_domain": domain_key,
        # Scheduler session key is model_id (server-side training session identity),
        # not the user-provided create_model session_id string.
        "scheduler_session_key": str(model_id),
        # Always serialize model-bound training ops by server-side training session identity,
        # regardless of whether the scheduler feature flag is enabled.
        "execution_serial_key": f"training_session:{model_id}",
        "training_op": str(training_op),
    }
    if openpi_train_step:
        extra["scheduler_fairness"] = "rr"
        extra["scheduler_max_consecutive"] = 1
    if seq_id is not None:
        try:
            extra["seq_id"] = int(seq_id)
        except Exception:
            extra["seq_id"] = None
    return extra


def _build_training_queued_meta(
    *,
    op: str,
    model_id: str,
    session: Any | None = None,
    seq_id: int | None = None,
) -> dict[str, Any]:
    op_name = str(op)
    if not op_name.startswith("training."):
        op_name = f"training.{op_name}"
    meta: dict[str, Any] = {
        "op": op_name,
        "model_id": str(model_id),
        "queue_state": "queued",
        "stage": "queued",
        "queued_at": time.time(),
    }
    if session is not None:
        meta["session_id"] = str(_field(session, "session_id", "") or model_id)
        base_model = str(_field(session, "base_model", "") or "")
        if base_model:
            meta["base_model"] = base_model
        backend = str(_field(session, "backend", "") or "")
        if backend:
            meta["backend"] = backend
    if seq_id is not None:
        try:
            meta["seq_id"] = int(seq_id)
        except Exception:
            meta["seq_id"] = seq_id
    return meta


def _build_create_scheduler_extra(
    *,
    base_model: str,
    model_id: str,
    training_op: str,
) -> dict[str, Any]:
    from ..backend.model_actor_supervisor import domain_key_for_training_base_model

    scheduler_domain = domain_key_for_training_base_model(base_model)
    return {
        "scheduler_enabled": str(os.environ.get("MINT_SCHEDULER_ENABLE", "1"))
        .strip()
        .lower()
        in ("1", "true", "yes", "y", "on"),
        "scheduler_domain": scheduler_domain,
        "scheduler_session_key": str(model_id),
        "execution_serial_key": f"training_session:{model_id}",
        "training_op": str(training_op),
    }


def _sync_route_wait_timeout_s() -> float:
    try:
        return max(
            1.0,
            float(
                str(os.environ.get("MINT_SYNC_ROUTE_WAIT_TIMEOUT_S", "3600")).strip()
            ),
        )
    except Exception:
        return 3600.0


def _sync_route_wait_poll_interval_s() -> float:
    try:
        return max(
            0.01,
            float(
                str(
                    os.environ.get("MINT_SYNC_ROUTE_WAIT_POLL_INTERVAL_S", "0.2")
                ).strip()
            ),
        )
    except Exception:
        return 0.2


async def _wait_internal_future_result(request_id: str) -> Any:
    deadline = time.perf_counter() + _sync_route_wait_timeout_s()
    poll_interval_s = _sync_route_wait_poll_interval_s()
    try:
        while True:
            status = await task_futures.async_get_status(request_id)
            if status == FutureStatus.PENDING:
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for internal future request_id={request_id}"
                    )
                await asyncio.sleep(poll_interval_s)
                continue
            if status == FutureStatus.DONE:
                return await task_futures.async_get_result(request_id)
            if status == FutureStatus.FAILED:
                err = await task_futures.async_get_error(request_id)
                raise RuntimeError(
                    str(err or f"internal queued op failed request_id={request_id}")
                )
            raise RuntimeError(
                f"internal future reached unexpected terminal state={status.value} request_id={request_id}"
            )
    finally:
        try:
            await task_futures.async_cleanup(request_id)
        except Exception:
            pass


async def _enqueue_internal_serialized_model_op(
    *,
    model_id: str,
    op: str,
    request_json: bytes,
    extra: dict[str, Any],
    user_id: str | None = None,
) -> str:
    from ..backend.model_work_admission import enqueue_model_work
    from ..backend.model_work_scheduler import model_work_scheduler

    request_id = uuid.uuid4().hex
    inflight_marked = False
    try:
        await _mark_training_inflight(model_id, +1)
        inflight_marked = True
        await enqueue_model_work(
            request_id=request_id,
            op=op,
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
            domain_key=str(
                extra.get("scheduler_domain") or f"training_session:{model_id}"
            ),
            affinity_group=f"training_session:{model_id}",
            ordering_key=f"training_session:{model_id}",
            extra=dict(extra),
            queued_meta=_build_training_queued_meta(op=op, model_id=model_id),
            scheduler_client=model_work_scheduler,
            future_service_client=task_futures,
        )
    except Exception as e:
        if inflight_marked:
            await _mark_training_inflight(model_id, -1)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue {op} request: {e}"
        ) from e
    return request_id


# =============================================================================
# create_model - async
# =============================================================================


@router.post("/create_model", response_model=UntypedAPIFuture)
async def create_model(
    request: CreateModelRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Create a new training model with LoRA."""
    _require_write_access(http_request)
    route_start_s = time.perf_counter()
    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(
        base_model=request.base_model, http_request=http_request
    )
    request = request.model_copy(update={"base_model": base_model})

    _validate_rollout_correction_config_or_400(
        base_model=request.base_model,
        rollout_correction_config=request.rollout_correction_config,
    )
    _validate_lora_rank_or_400(request.lora_config)
    try:
        from ..backend.openpi_fast_training import validate_openpi_fast_create_request
        from ..backend.openpi_pi05_training import validate_openpi_pi05_create_request

        validate_openpi_fast_create_request(request)
        validate_openpi_pi05_create_request(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(request.base_model, user_data):
        raise HTTPException(
            status_code=403, detail=get_access_denied_error(request.base_model)
        )

    user_id = _get_user_id(http_request)
    model_id = _generate_model_id(request.session_id, request.model_seq_id)

    # Gateway forwarding: if base_model is configured as remote, proxy to upstream and
    # return a gateway-encoded request_id so /retrieve_future can route it.
    from ..gateway import (
        async_register_remote_training_model,
        async_remote_training_model,
        encode_request_id,
        forward_json,
        get_gateway_config,
        upstream_for_model,
    )

    upstream = upstream_for_model(request.base_model)
    if upstream is not None:
        await _raise_if_local_model_id_exists(model_id)
        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/create_model",
                incoming_headers=dict(http_request.headers),
                json_body=request.model_dump(),
                timeout_s=120.0,
            )
        except Exception:
            logger.exception("Upstream create_model failed: %s", upstream.alias)
            raise HTTPException(
                status_code=503,
                detail=f"Upstream {upstream.alias!r} create_model failed",
            )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(
                status_code=502,
                detail="Upstream create_model returned invalid request_id",
            )

        await async_register_remote_training_model(
            model_id=model_id,
            upstream_alias=upstream.alias,
            base_model=request.base_model,
            owner_id=user_id,
        )
        return UntypedAPIFuture(
            request_id=encode_request_id(
                upstream_alias=upstream.alias, upstream_request_id=upstream_request_id
            )
        )

    cfg = get_gateway_config()
    if cfg is not None and cfg.model_to_upstream:
        remote = await async_remote_training_model(model_id)
        if remote is not None:
            upstream_alias, _ = remote
            raise HTTPException(
                status_code=409,
                detail=f"Model_id conflict: {model_id!r} is registered as remote via upstream {upstream_alias!r}",
            )

    user_id = _get_user_id(http_request)
    webhook_url = _get_webhook_url(http_request)
    scheduler_extra = _build_create_scheduler_extra(
        base_model=request.base_model,
        model_id=model_id,
        training_op="create_model",
    )

    from ..backend.model_work_admission import enqueue_model_work
    from ..backend.model_work_scheduler import model_work_scheduler

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex

    try:
        await enqueue_model_work(
            request_id=request_id,
            op="training.create_model",
            request_json=request_json,
            user_id=user_id,
            webhook_url=webhook_url,
            domain_key=str(scheduler_extra["scheduler_domain"]),
            affinity_group=f"training_session:{model_id}",
            ordering_key=f"training_session:{model_id}",
            extra=scheduler_extra,
            queued_meta=_build_training_queued_meta(
                op="create_model", model_id=model_id
            ),
            scheduler_client=model_work_scheduler,
            future_service_client=task_futures,
            trace_enqueue=_enqueue_training_request_with_trace,
            trace_kwargs={
                "route_start_s": route_start_s,
                "model_id": model_id,
                "base_model": request.base_model,
            },
        )
        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_CREATED,  # pending
                user_id=user_id,
                session_id=model_id,
                task_name=f"Training {request.base_model}",
                task_type="training",
                model_name=request.base_model,
                config={
                    "lora_rank": request.lora_config.rank
                    if request.lora_config
                    else None
                },
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
        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_FAILED,
                user_id=user_id,
                session_id=model_id,
                task_name=f"Training {request.base_model}",
                task_type="training",
                model_name=request.base_model,
                error=f"enqueue_failed: {type(e).__name__}: {e}",
                config={
                    "lora_rank": request.lora_config.rank
                    if request.lora_config
                    else None
                },
            )
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue create_model request: {e}"
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_create_model(
    request_id: str,
    request: CreateModelRequest,
    user_id: str | None,
    webhook_url: str | None,
) -> None:
    """Background task to create training model."""
    from ..backend.training_session_manager import (
        MATERIALIZATION_STATE_UNMATERIALIZED,
        TRAINING_SESSION_METADATA_VERSION,
    )

    model_id = _generate_model_id(request.session_id, request.model_seq_id)
    inflight_marked = False
    session_created = False
    planned_backend = _infer_training_backend_for_base_model(request.base_model)
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        get_local_session = getattr(manager, "get_local_session", None)
        existing = (
            get_local_session(model_id)
            if callable(get_local_session)
            else manager.get_session(model_id)
        )
        if existing is not None:
            if bool(getattr(existing, "is_active", False)):
                raise RuntimeError(f"Model '{model_id}' already exists")
            logger.warning(
                f"[{model_id}] Cleaning up stale inactive session from previous attempt"
            )
            await engine.shutdown_session(existing)
            manager.delete_session(model_id)

        rollout_correction_config = (
            request.rollout_correction_config.model_dump(exclude_none=True)
            if request.rollout_correction_config
            else None
        )
        session = manager.create_session(
            model_id=model_id,
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            rollout_correction_config=rollout_correction_config,
            user_metadata=request.user_metadata,
            user_id=user_id,
            backend=planned_backend,
            metadata_version=TRAINING_SESSION_METADATA_VERSION,
            materialization_state=MATERIALIZATION_STATE_UNMATERIALIZED,
        )
        session_created = True

        manager.mark_inflight(model_id, +1)
        inflight_marked = True

        tokenizer_metadata = await _best_effort_local_tokenizer_metadata_for_session(
            session
        )
        if isinstance(tokenizer_metadata, dict):
            session.tokenizer_info = dict(
                tokenizer_metadata.get("tokenizer_info") or {}
            )
            session.tokenizer_identity = (
                str(tokenizer_metadata.get("tokenizer_identity") or "") or None
            )
            session.tokenizer_source_path = (
                str(tokenizer_metadata.get("tokenizer_source_path") or "") or None
            )
        session.namespace = RAY_NAMESPACE

        from ..backend.training_session_store import async_upsert_training_session

        await async_upsert_training_session(
            _build_training_session_store_payload(
                session=session,
                user_id=user_id,
                lora_config=request.lora_config,
                rollout_correction_config=rollout_correction_config,
                user_metadata=request.user_metadata,
                actor_name=None,
                materialization_state=MATERIALIZATION_STATE_UNMATERIALIZED,
                tokenizer_metadata=tokenizer_metadata,
            )
        )
        manager.mark_persisted(model_id)

        try:
            from ..backend.session_index_store import add_training_run_to_session

            add_training_run_to_session(
                session_id=request.session_id,
                training_run_id=model_id,
                user_id=user_id,
                created_at=session.created_at,
            )
        except Exception as e:
            logger.warning("[create_model] session index write failed: %s", e)

        response = CreateModelResponse(
            request_id=request_id,
            model_id=model_id,
            type="create_model",
            backend=planned_backend,
        )
        await task_futures.async_resolve(request_id, response.model_dump())

        if webhook_url and user_id:
            send_task_event(
                webhook_url=webhook_url,
                event_type=EventType.TASK_STARTED,
                user_id=user_id,
                session_id=model_id,
                task_name=f"Training {request.base_model}",
                task_type="training",
                model_name=request.base_model,
            )

    except Exception as e:
        logger.exception(
            "[create_model] failed request_id=%s model_id=%s base_model=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(model_id),
            str(request.base_model),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_actor",
        )
        manager = _current_training_manager()
        if session_created and manager is not None:
            get_local_session = getattr(manager, "get_local_session", None)
            session = (
                get_local_session(model_id)
                if callable(get_local_session)
                else manager.get_session(model_id)
            )
            if session is not None:
                manager.delete_session(model_id)
        if session_created:
            try:
                from ..backend.training_session_store import delete_training_session

                delete_training_session(model_id)
            except Exception:
                pass
        try:
            from ..backend.model_actor_supervisor import get_model_actor_supervisor

            get_model_actor_supervisor().clear_session(model_id)
        except Exception:
            pass
        await task_futures.async_fail(request_id, str(e))

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
    finally:
        if inflight_marked:
            manager = _current_training_manager()
            if manager is not None:
                manager.mark_inflight(model_id, -1)


# =============================================================================
# create_model_from_state - async (composes create_model + load_state)
# =============================================================================


def _resolve_state_path(
    state_uri: str,
    *,
    user_id: str | None,
    is_admin: bool = False,
    owner_id: str | None = None,
) -> str:
    if not is_admin and not state_uri.startswith(("mint://", "ckpt_")):
        raise HTTPException(status_code=403, detail="Access denied")

    owner_scope = owner_id if is_admin else user_id
    try:
        resolved = resolve_checkpoint_path(
            state_uri, user_id=owner_scope, is_admin=is_admin
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if state_uri.startswith("ckpt_") and resolved == state_uri:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    try:
        ensure_checkpoint_path_allowed(resolved, user_id=owner_scope, is_admin=is_admin)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return materialize_persistent_checkpoint(resolved)


@router.post("/create_model_from_state", response_model=UntypedAPIFuture)
async def create_model_from_state(
    request: CreateModelFromStateRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Create a training model and load existing checkpoint.

    Composes create_model + load_state into single operation.
    Useful for resuming training from a saved checkpoint.
    """
    _require_write_access(http_request)
    route_start_s = time.perf_counter()
    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(
        base_model=request.base_model, http_request=http_request
    )
    request = request.model_copy(update={"base_model": base_model})

    _validate_rollout_correction_config_or_400(
        base_model=request.base_model,
        rollout_correction_config=request.rollout_correction_config,
    )
    _validate_lora_rank_or_400(request.lora_config)
    try:
        from ..backend.openpi_fast_training import validate_openpi_fast_create_request
        from ..backend.openpi_pi05_training import validate_openpi_pi05_create_request

        validate_openpi_fast_create_request(request)
        validate_openpi_pi05_create_request(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check model access permissions
    user_data = _get_user_data(http_request)
    if not can_access_model(request.base_model, user_data):
        raise HTTPException(
            status_code=403, detail=get_access_denied_error(request.base_model)
        )

    model_id = _generate_model_id(request.session_id, request.model_seq_id)
    user_id = _get_user_id(http_request)

    # Fail fast: sampler checkpoints are not eligible for optimizer restore.
    if bool(request.load_optimizer):
        try:
            from ..checkpoints import validate_checkpoint_load_contract

            local_path = _resolve_state_path(
                request.state_path,
                user_id=user_id,
                is_admin=can_manage_system(http_request),
                owner_id=request.owner_id,
            )
            if os.path.isdir(local_path) and os.path.exists(
                os.path.join(local_path, "metadata.json")
            ):
                validate_checkpoint_load_contract(local_path, load_optimizer=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    from ..gateway import (
        async_register_remote_training_model,
        async_remote_training_model,
        encode_request_id,
        forward_file,
        forward_json,
        get_gateway_config,
        upstream_for_model,
    )

    upstream = upstream_for_model(request.base_model)
    if upstream is not None:
        await _raise_if_local_model_id_exists(model_id)
        incoming_headers = dict(http_request.headers)
        if request.state_path.startswith(("mint://", "ckpt_")):
            local_path = _resolve_state_path(
                request.state_path,
                user_id=user_id,
                is_admin=can_manage_system(http_request),
                owner_id=request.owner_id,
            )
            if os.path.isdir(local_path):
                proxy_timeout_s = float(
                    os.environ.get("MINT_GATEWAY_CHECKPOINT_PROXY_TIMEOUT_S", "600")
                )
                tmp_archive = build_gateway_proxy_archive_path()
                try:
                    await async_create_checkpoint_archive(
                        local_path, tmp_archive, timeout_s=proxy_timeout_s
                    )
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
                    raise HTTPException(
                        status_code=upload_resp.status_code, detail=upload_resp.text
                    )
                payload = upload_resp.json()
                ckpt_id = payload.get("checkpoint_id")
                if not isinstance(ckpt_id, str) or not ckpt_id:
                    raise HTTPException(
                        status_code=502,
                        detail="Upstream checkpoints/upload returned invalid checkpoint_id",
                    )
                owner_scope = (
                    request.owner_id if can_manage_system(http_request) else user_id
                )
                request = request.model_copy(
                    update={"state_path": ckpt_id, "owner_id": owner_scope}
                )
        try:
            resp = await forward_json(
                upstream=upstream,
                method="POST",
                path="/api/v1/create_model_from_state",
                incoming_headers=incoming_headers,
                json_body=request.model_dump(),
                timeout_s=120.0,
            )
        except Exception:
            logger.exception(
                "Upstream create_model_from_state failed: %s", upstream.alias
            )
            raise HTTPException(
                status_code=503,
                detail=f"Upstream {upstream.alias!r} create_model_from_state failed",
            )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(
                status_code=502,
                detail="Upstream create_model_from_state returned invalid request_id",
            )

        await async_register_remote_training_model(
            model_id=model_id,
            upstream_alias=upstream.alias,
            base_model=request.base_model,
            owner_id=user_id,
        )
        return UntypedAPIFuture(
            request_id=encode_request_id(
                upstream_alias=upstream.alias, upstream_request_id=upstream_request_id
            )
        )

    cfg = get_gateway_config()
    if cfg is not None and cfg.model_to_upstream:
        remote = await async_remote_training_model(model_id)
        if remote is not None:
            upstream_alias, _ = remote
            raise HTTPException(
                status_code=409,
                detail=f"Model_id conflict: {model_id!r} is registered as remote via upstream {upstream_alias!r}",
            )

    from ..backend.model_work_admission import enqueue_model_work
    from ..backend.model_work_scheduler import model_work_scheduler

    resolved_state_path = _resolve_state_path(
        request.state_path,
        user_id=user_id,
        is_admin=can_manage_system(http_request),
        owner_id=request.owner_id,
    )
    if request.state_path.startswith(("mint://", "ckpt_")) and not os.path.isdir(
        resolved_state_path
    ):
        raise HTTPException(
            status_code=404, detail=f"Checkpoint not found: {request.state_path}"
        )
    request = request.model_copy(update={"state_path": resolved_state_path})

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    scheduler_extra = _build_create_scheduler_extra(
        base_model=request.base_model,
        model_id=model_id,
        training_op="create_model_from_state",
    )

    try:
        await enqueue_model_work(
            request_id=request_id,
            op="training.create_model_from_state",
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
            domain_key=str(scheduler_extra["scheduler_domain"]),
            affinity_group=f"training_session:{model_id}",
            ordering_key=f"training_session:{model_id}",
            extra=scheduler_extra,
            queued_meta=_build_training_queued_meta(
                op="create_model_from_state", model_id=model_id
            ),
            scheduler_client=model_work_scheduler,
            future_service_client=task_futures,
            trace_enqueue=_enqueue_training_request_with_trace,
            trace_kwargs={
                "route_start_s": route_start_s,
                "model_id": model_id,
                "base_model": request.base_model,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue create_model_from_state request: {e}",
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_create_model_from_state(
    request_id: str, request: CreateModelFromStateRequest, user_id: str | None
) -> None:
    """Background task to create model and load checkpoint."""
    from ..backend.training_session_manager import (
        MATERIALIZATION_STATE_READY,
        TRAINING_SESSION_METADATA_VERSION,
    )

    model_id = _generate_model_id(request.session_id, request.model_seq_id)
    inflight_marked = False
    session_created = False
    session_materialized = False
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        load_path = request.state_path

        get_local_session = getattr(manager, "get_local_session", None)
        existing = (
            get_local_session(model_id)
            if callable(get_local_session)
            else manager.get_session(model_id)
        )
        if existing is not None:
            if bool(getattr(existing, "is_active", False)):
                raise RuntimeError(f"Model '{model_id}' already exists")
            logger.warning(
                f"[{model_id}] Cleaning up stale inactive session from previous attempt"
            )
            await engine.shutdown_session(existing)
            manager.delete_session(model_id)

        rollout_correction_config = (
            request.rollout_correction_config.model_dump(exclude_none=True)
            if request.rollout_correction_config
            else None
        )
        planned_backend = _infer_training_backend_for_base_model(request.base_model)
        tokenizer_metadata = None

        session = manager.create_session(
            model_id=model_id,
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            rollout_correction_config=rollout_correction_config,
            user_metadata=request.user_metadata,
            user_id=user_id,
            backend=planned_backend,
            metadata_version=TRAINING_SESSION_METADATA_VERSION,
            materialization_state=MATERIALIZATION_STATE_READY,
        )
        session_created = True

        manager.mark_inflight(model_id, +1)
        inflight_marked = True

        async def _create_and_restore_model():
            await engine.create_training_session(session)
            return await engine.load_weights(
                session=session,
                load_path=load_path,
                load_optimizer=request.load_optimizer,
            )

        load_metadata = await run_async_with_otel_span(
            "training.create_model_from_state.execute",
            _create_and_restore_model,
            component="routes.training",
            op="training.create_model_from_state",
            request_id=str(request_id),
            attributes={
                "model_id": str(model_id),
                "base_model": str(request.base_model),
                "backend": str(session.backend),
                "load_optimizer": bool(request.load_optimizer),
                "lora_enabled": bool(request.lora_config is not None),
                "lora_rank": int(request.lora_config.rank)
                if request.lora_config is not None
                else None,
            },
        )
        session_materialized = True
        if _supports_control_plane_tokenizer_metadata(
            str(getattr(session, "backend", "") or "")
        ):
            tokenizer_metadata = await _collect_control_plane_tokenizer_metadata(
                session
            )
            session.tokenizer_info = dict(
                tokenizer_metadata.get("tokenizer_info") or {}
            )
            session.tokenizer_identity = (
                str(tokenizer_metadata.get("tokenizer_identity") or "") or None
            )
            session.tokenizer_source_path = (
                str(tokenizer_metadata.get("tokenizer_source_path") or "") or None
            )

        from ..backend.training_session_store import async_upsert_training_session

        actor_name = getattr(engine, "_model_actor_supervisor_actor_names", {}).get(
            model_id
        )
        session.actor_name = str(actor_name or "") or None
        session.namespace = RAY_NAMESPACE
        await async_upsert_training_session(
            _build_training_session_store_payload(
                session=session,
                user_id=user_id,
                lora_config=session.lora_config,
                rollout_correction_config=rollout_correction_config,
                user_metadata=request.user_metadata,
                actor_name=actor_name,
                materialization_state=MATERIALIZATION_STATE_READY,
                tokenizer_metadata=tokenizer_metadata,
            )
        )
        manager.mark_persisted(model_id)

        try:
            from ..backend.session_index_store import add_training_run_to_session

            add_training_run_to_session(
                session_id=request.session_id,
                training_run_id=model_id,
                user_id=user_id,
                created_at=session.created_at,
            )
        except Exception as e:
            logger.warning(
                "[create_model_from_state] session index write failed: %s", e
            )

        logger.info(
            f"[{model_id}] Created model from state: {request.state_path} "
            f"(step={session.current_step})"
        )

        response = CreateModelFromStateResponse(
            request_id=request_id,
            model_id=model_id,
            type="create_model_from_state",
            load_metadata=_public_load_metadata(load_metadata),
        )
        await task_futures.async_resolve(
            request_id, response.model_dump(exclude_none=True)
        )

    except Exception as e:
        logger.exception(
            "[create_model_from_state] failed request_id=%s model_id=%s base_model=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(_generate_model_id(request.session_id, request.model_seq_id)),
            str(request.base_model),
            classify_failure_reason(e),
            type(e).__name__,
            "check_checkpoint_path_and_training_actor",
        )
        manager = _current_training_manager()
        engine = _current_training_engine()
        if session_created and manager is not None:
            get_local_session = getattr(manager, "get_local_session", None)
            session = (
                get_local_session(model_id)
                if callable(get_local_session)
                else manager.get_session(model_id)
            )
            if session is not None and session_materialized and engine is not None:
                try:
                    await engine.shutdown_session(session)
                except Exception as shutdown_error:
                    logger.warning(
                        "[create_model_from_state] cleanup shutdown failed model_id=%s error_type=%s error=%s",
                        str(model_id),
                        type(shutdown_error).__name__,
                        shutdown_error,
                    )
            if session is not None:
                manager.delete_session(model_id)
        if session_created:
            try:
                from ..backend.training_session_store import delete_training_session

                delete_training_session(model_id)
            except Exception:
                pass
        await task_futures.async_fail(request_id, str(e))
    finally:
        if inflight_marked:
            manager = _current_training_manager()
            if manager is not None:
                manager.mark_inflight(model_id, -1)


# =============================================================================
# forward_backward - async
# =============================================================================


@router.post("/forward_backward", response_model=UntypedAPIFuture)
async def forward_backward(
    request: ForwardBackwardRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform forward + backward pass on training data."""
    _require_write_access(http_request)
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    _validate_training_batch_has_explicit_loss_masks_or_422(
        request.forward_backward_input.data
    )

    session, route_session_info = await _resolve_training_route_session(
        request.model_id
    )

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(
                    status_code=403, detail=get_access_denied_error(base_model)
                )

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
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream {upstream_alias!r} forward_backward failed",
                )
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream forward_backward returned invalid request_id",
                )
            return UntypedAPIFuture(
                request_id=encode_request_id(
                    upstream_alias=upstream_alias,
                    upstream_request_id=upstream_request_id,
                )
            )

    if route_session_info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    route_session_info = _ensure_route_session_info(
        model_id=request.model_id,
        session=session,
        route_session_info=route_session_info,
    )
    base_model = str(route_session_info.get("base_model") or "")
    backend = str(route_session_info.get("backend") or "unknown")
    max_model_len = _get_max_model_len(base_model)
    _, max_seq_len = _compute_token_stats(request.forward_backward_input.data)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {base_model}"
            ),
        )

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    user_id = _get_user_id(http_request)
    gateway_auth = build_billing_auth_context(
        http_request, fallback_request_id=request_id
    )

    set_request_id(request_id)
    logger.info(f"forward_backward request received: model_id={request.model_id}")

    inflight_marked = False
    try:
        await _protect_training_session_enqueue_window(route_session_info)
        await _mark_training_inflight(request.model_id, +1)
        inflight_marked = True
        scheduler_extra = merge_queue_priority_extra(
            _build_training_scheduler_extra(
                session=route_session_info,
                model_id=request.model_id,
                training_op="forward_backward",
                seq_id=request.seq_id,
            ),
            request=http_request,
        )
        if gateway_auth is not None:
            scheduler_extra["gateway_auth"] = gateway_auth.__dict__
        await _enqueue_training_model_work_route(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.forward_backward",
            request_json=request_json,
            user_id=user_id,
            apikey_id=_get_apikey_id(http_request, gateway_auth=gateway_auth),
            extra=scheduler_extra,
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            queued_meta=_build_training_queued_meta(
                op="forward_backward",
                model_id=request.model_id,
                session=route_session_info,
                seq_id=request.seq_id,
            ),
        )
    except Exception as e:
        if inflight_marked:
            await _mark_training_inflight(request.model_id, -1)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue forward_backward request: {e}"
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_forward_backward(
    request_id: str,
    request: ForwardBackwardRequest,
    user_id: str | None,
    gateway_auth: dict | None = None,
) -> None:
    """Background task for forward_backward."""
    # Restore request_id context for logging
    set_request_id(request_id)
    try:
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        session = manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        session = await _materialize_training_session_for_stateful_use(session)
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
        logger.info(
            f"[{session.model_id}] forward_backward start: "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len} "
            f"loss_fn={request.forward_backward_input.loss_fn}"
        )
        result = await run_async_with_otel_span(
            "training.forward_backward.execute",
            lambda: engine.forward_backward(session, request),
            component="routes.training",
            op="training.forward_backward",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(session.backend),
                "batch_size": int(len(batch)),
                "token_count": int(token_count),
                "max_seq_len": int(max_seq_len),
                "loss_fn": str(request.forward_backward_input.loss_fn),
            },
        )
        elapsed_s = time.time() - t0
        logger.info(
            f"[{session.model_id}] forward_backward done: elapsed_s={elapsed_s:.3f}"
        )
        await task_futures.async_resolve(
            request_id,
            result,
            billing_observations=_build_training_billing_observations(
                gateway_auth=gateway_auth,
                request_id=request_id,
                model=session.base_model,
                route="training.forward_backward",
                token_count=token_count,
            ),
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception(
            "[forward_backward] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_batch_shape",
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(request.model_id, -1)


# =============================================================================
# train_step - async (forward_backward + optim_step)
# =============================================================================


@router.post("/train_step", response_model=UntypedAPIFuture)
async def train_step(
    request: TrainStepRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform a combined forward_backward + optim_step."""
    _require_write_access(http_request)
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    _validate_training_batch_has_explicit_loss_masks_or_422(
        request.forward_backward_input.data
    )

    session, route_session_info = await _resolve_training_route_session(
        request.model_id
    )

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(
                    status_code=403, detail=get_access_denied_error(base_model)
                )

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
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream {upstream_alias!r} train_step failed",
                )

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream train_step returned invalid request_id",
                )
            return UntypedAPIFuture(
                request_id=encode_request_id(
                    upstream_alias=upstream_alias,
                    upstream_request_id=upstream_request_id,
                )
            )

    if route_session_info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    route_session_info = _ensure_route_session_info(
        model_id=request.model_id,
        session=session,
        route_session_info=route_session_info,
    )
    base_model = str(route_session_info.get("base_model") or "")
    backend = str(route_session_info.get("backend") or "unknown")
    max_model_len = _get_max_model_len(base_model)
    _, max_seq_len = _compute_token_stats(request.forward_backward_input.data)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {base_model}"
            ),
        )

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    user_id = _get_user_id(http_request)
    gateway_auth = build_billing_auth_context(
        http_request, fallback_request_id=request_id
    )

    inflight_marked = False
    try:
        await _protect_training_session_enqueue_window(route_session_info)
        await _mark_training_inflight(request.model_id, +1)
        inflight_marked = True
        scheduler_extra = merge_queue_priority_extra(
            _build_training_scheduler_extra(
                session=route_session_info,
                model_id=request.model_id,
                training_op="train_step",
                seq_id=request.seq_id,
            ),
            request=http_request,
        )
        if gateway_auth is not None:
            scheduler_extra["gateway_auth"] = gateway_auth.__dict__
        await _enqueue_training_model_work_route(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.train_step",
            request_json=request_json,
            user_id=user_id,
            apikey_id=_get_apikey_id(http_request, gateway_auth=gateway_auth),
            extra=scheduler_extra,
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            queued_meta=_build_training_queued_meta(
                op="train_step",
                model_id=request.model_id,
                session=route_session_info,
                seq_id=request.seq_id,
            ),
        )
    except Exception as e:
        if inflight_marked:
            await _mark_training_inflight(request.model_id, -1)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue train_step request: {e}"
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_train_step(
    request_id: str,
    request: TrainStepRequest,
    user_id: str | None,
    gateway_auth: dict | None = None,
    billing_observations: list[dict] | None = None,
    billing_observation_input: dict | None = None,
) -> None:
    """Background task for train_step."""
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        session = manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        session = await _materialize_training_session_for_stateful_use(session)
        batch = request.forward_backward_input.data
        token_count, max_seq_len = _compute_token_stats(batch)
        t0 = time.time()
        msg = (
            f"[{session.model_id}] train_step start request_id={request_id} "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len}"
        )
        logger.info(msg)
        result = await run_async_with_otel_span(
            "training.train_step.execute",
            lambda: engine.train_step(session, request),
            component="routes.training",
            op="training.train_step",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(session.backend),
                "batch_size": int(len(batch)),
                "token_count": int(token_count),
                "max_seq_len": int(max_seq_len),
                "seq_id": int(request.seq_id) if request.seq_id is not None else None,
                "loss_fn": str(request.forward_backward_input.loss_fn),
            },
        )
        elapsed_s = time.time() - t0
        msg = f"[{session.model_id}] train_step done request_id={request_id} elapsed_s={elapsed_s:.3f}"
        logger.info(msg)
        await task_futures.async_resolve(
            request_id,
            result,
            billing_observations=(
                billing_observations
                if billing_observations is not None
                else billing_observations_from_input(
                    gateway_auth=gateway_auth,
                    request_id=request_id,
                    billing_input=billing_observation_input,
                )
                if billing_observation_input is not None
                else _build_training_billing_observations(
                    gateway_auth=gateway_auth,
                    request_id=request_id,
                    model=session.base_model,
                    route="training.train_step",
                    token_count=token_count,
                )
            ),
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception(
            "[train_step] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_actor",
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(request.model_id, -1)


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
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    session, route_session_info = await _resolve_training_route_session(
        request.model_id
    )

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(
                    status_code=403, detail=get_access_denied_error(base_model)
                )

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
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream {upstream_alias!r} forward failed",
                )

            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream forward returned invalid request_id",
                )
            return UntypedAPIFuture(
                request_id=encode_request_id(
                    upstream_alias=upstream_alias,
                    upstream_request_id=upstream_request_id,
                )
            )

    if route_session_info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    route_session_info = _ensure_route_session_info(
        model_id=request.model_id,
        session=session,
        route_session_info=route_session_info,
    )
    base_model = str(route_session_info.get("base_model") or "")
    backend = str(route_session_info.get("backend") or "unknown")
    max_model_len = _get_max_model_len(base_model)
    _, max_seq_len = _compute_token_stats(request.forward_input.data)
    if max_seq_len > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input sequence length {max_seq_len} exceeds max_model_len {max_model_len} "
                f"for model {base_model}"
            ),
        )

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    gateway_auth = build_billing_auth_context(
        http_request, fallback_request_id=request_id
    )
    user_id = _get_user_id(http_request)

    inflight_marked = False
    try:
        await _protect_training_session_enqueue_window(route_session_info)
        await _mark_training_inflight(request.model_id, +1)
        inflight_marked = True
        scheduler_extra = merge_queue_priority_extra(
            _build_training_scheduler_extra(
                session=route_session_info,
                model_id=request.model_id,
                training_op="forward",
                seq_id=request.seq_id,
            ),
            request=http_request,
        )
        if gateway_auth is not None:
            scheduler_extra["gateway_auth"] = gateway_auth.__dict__
        await _enqueue_training_model_work_route(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.forward",
            request_json=request_json,
            user_id=user_id,
            apikey_id=_get_apikey_id(http_request, gateway_auth=gateway_auth),
            extra=scheduler_extra,
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            queued_meta=_build_training_queued_meta(
                op="forward",
                model_id=request.model_id,
                session=route_session_info,
                seq_id=request.seq_id,
            ),
        )
    except Exception as e:
        if inflight_marked:
            await _mark_training_inflight(request.model_id, -1)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue forward request: {e}"
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_forward(
    request_id: str,
    request: ForwardRequest,
    gateway_auth: dict | None = None,
) -> None:
    """Background task for forward."""
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        session = manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        session = await _materialize_training_session_for_stateful_use(session)
        batch = request.forward_input.data
        token_count, max_seq_len = _compute_token_stats(batch)
        t0 = time.time()
        logger.info(
            f"[{session.model_id}] forward start: "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len}"
        )
        result = await run_async_with_otel_span(
            "training.forward.execute",
            lambda: engine.forward(session, request),
            component="routes.training",
            op="training.forward",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(session.backend),
                "batch_size": int(len(batch)),
                "token_count": int(token_count),
                "max_seq_len": int(max_seq_len),
                "seq_id": int(request.seq_id) if request.seq_id is not None else None,
            },
        )
        elapsed_s = time.time() - t0
        logger.info(f"[{session.model_id}] forward done: elapsed_s={elapsed_s:.3f}")
        await task_futures.async_resolve(
            request_id,
            result,
            billing_observations=_build_training_billing_observations(
                gateway_auth=gateway_auth,
                request_id=request_id,
                model=session.base_model,
                route="training.forward",
                token_count=token_count,
            ),
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception(
            "[forward] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_input_tokens",
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(request.model_id, -1)


# =============================================================================
# optim_step - async
# =============================================================================


@router.post("/optim_step", response_model=UntypedAPIFuture)
async def optim_step(
    request: OptimStepRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform optimizer step to update weights."""
    _require_write_access(http_request)
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    session, route_session_info = await _resolve_training_route_session(
        request.model_id
    )

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
                )

            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(
                    status_code=403, detail=get_access_denied_error(base_model)
                )

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
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream {upstream_alias!r} optim_step failed",
                )
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream optim_step returned invalid request_id",
                )
            return UntypedAPIFuture(
                request_id=encode_request_id(
                    upstream_alias=upstream_alias,
                    upstream_request_id=upstream_request_id,
                )
            )

    if route_session_info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    route_session_info = _ensure_route_session_info(
        model_id=request.model_id,
        session=session,
        route_session_info=route_session_info,
    )
    base_model = str(route_session_info.get("base_model") or "")
    backend = str(route_session_info.get("backend") or "unknown")

    user_id = _get_user_id(http_request)

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex

    inflight_marked = False
    try:
        await _protect_training_session_enqueue_window(route_session_info)
        await _mark_training_inflight(request.model_id, +1)
        inflight_marked = True
        scheduler_extra = merge_queue_priority_extra(
            _build_training_scheduler_extra(
                session=route_session_info,
                model_id=request.model_id,
                training_op="optim_step",
                seq_id=request.seq_id,
            ),
            request=http_request,
        )
        await _enqueue_training_model_work_route(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.optim_step",
            request_json=request_json,
            user_id=user_id,
            extra=scheduler_extra,
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            queued_meta=_build_training_queued_meta(
                op="optim_step",
                model_id=request.model_id,
                session=route_session_info,
                seq_id=request.seq_id,
            ),
        )
    except Exception as e:
        if inflight_marked:
            await _mark_training_inflight(request.model_id, -1)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue optim_step request: {e}"
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_optim_step(
    request_id: str, request: OptimStepRequest, user_id: str | None
) -> None:
    """Background task for optim_step."""
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        session = manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        session = await _materialize_training_session_for_stateful_use(session)
        lr = request.adam_params.learning_rate if request.adam_params else None
        t0 = time.time()
        msg = f"[{session.model_id}] optim_step start request_id={request_id} lr={lr}"
        logger.info(msg)
        result = await run_async_with_otel_span(
            "training.optim_step.execute",
            lambda: engine.optim_step(session, request),
            component="routes.training",
            op="training.optim_step",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(session.backend),
                "learning_rate": float(lr) if lr is not None else None,
                "seq_id": int(request.seq_id) if request.seq_id is not None else None,
            },
        )
        elapsed_s = time.time() - t0
        msg = f"[{session.model_id}] optim_step done request_id={request_id} elapsed_s={elapsed_s:.3f}"
        logger.info(msg)
        await task_futures.async_resolve(request_id, result)

    except Exception as e:
        logger.exception(
            "[optim_step] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_optimizer_state",
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(request.model_id, -1)


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
    _require_write_access(http_request)
    from ..gateway import async_remote_training_model, forward_json, upstream_for_alias

    session, route_session_info = await _resolve_training_route_session(
        request.model_id
    )

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(
                    status_code=403, detail=get_access_denied_error(base_model)
                )

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/reset_expert_bias",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=120.0,
                )
            except Exception:
                logger.exception(
                    "Upstream reset_expert_bias failed: %s", upstream_alias
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream {upstream_alias!r} reset_expert_bias failed",
                )
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return ResetExpertBiasResponse.model_validate(resp.json())

    if route_session_info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    try:
        request_id = await _enqueue_internal_serialized_model_op(
            model_id=request.model_id,
            op="training.reset_expert_bias",
            request_json=request.model_dump_json().encode("utf-8"),
            extra=_build_training_scheduler_extra(
                session=route_session_info,
                model_id=request.model_id,
                training_op="reset_expert_bias",
            ),
            user_id=_get_user_id(http_request),
        )
        payload = await _wait_internal_future_result(request_id)
        return ResetExpertBiasResponse.model_validate(payload)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[reset_expert_bias] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _do_reset_expert_bias(
    request_id: str,
    request: ResetExpertBiasRequest,
) -> None:
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        session = manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        result = await engine.reset_expert_bias(session)
        modules_reset = int(result.get("modules_reset", 0) or 0)
        await task_futures.async_resolve(
            request_id,
            ResetExpertBiasResponse(
                model_id=request.model_id,
                modules_reset=modules_reset,
                status="success" if modules_reset > 0 else "not_applicable",
            ).model_dump(),
        )
    except Exception as e:
        logger.exception(
            "[training.reset_expert_bias] failed request_id=%s model_id=%s error_type=%s error=%s",
            str(request_id),
            str(request.model_id),
            type(e).__name__,
            e,
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(request.model_id, -1)


# =============================================================================
# save_weights_for_sampler - async
# =============================================================================


@router.post("/save_weights_for_sampler", response_model=UntypedAPIFuture)
async def save_weights_for_sampler(
    request: SaveWeightsForSamplerRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save model weights for inference use."""
    _require_write_access(http_request)
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        register_pending_save_weights_for_sampler_future,
        upstream_for_alias,
    )

    session, route_session_info = await _resolve_training_route_session(
        request.model_id
    )

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(
                    status_code=403, detail=get_access_denied_error(base_model)
                )

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/save_weights_for_sampler",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=300.0,
                )
            except Exception:
                logger.exception(
                    "Upstream save_weights_for_sampler failed: %s", upstream_alias
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream {upstream_alias!r} save_weights_for_sampler failed",
                )
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            payload = resp.json()
            upstream_request_id = payload.get("request_id")
            if not isinstance(upstream_request_id, str) or not upstream_request_id:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream save_weights_for_sampler returned invalid request_id",
                )
            if request.path is None:
                register_pending_save_weights_for_sampler_future(
                    upstream_alias=upstream_alias,
                    upstream_request_id=upstream_request_id,
                    base_model=base_model,
                )
            return UntypedAPIFuture(
                request_id=encode_request_id(
                    upstream_alias=upstream_alias,
                    upstream_request_id=upstream_request_id,
                )
            )

    if route_session_info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    route_session_info = _ensure_route_session_info(
        model_id=request.model_id,
        session=session,
        route_session_info=route_session_info,
    )
    base_model = str(route_session_info.get("base_model") or "")
    backend = str(route_session_info.get("backend") or "unknown")

    user_id = _get_user_id(http_request)
    from ..client_compat import prefer_tinker_uri

    prefer_tinker = prefer_tinker_uri(http_request)

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex

    inflight_marked = False
    try:
        await _protect_training_session_enqueue_window(route_session_info)
        await _mark_training_inflight(request.model_id, +1)
        inflight_marked = True
        scheduler_extra = merge_queue_priority_extra(
            _build_training_scheduler_extra(
                session=route_session_info,
                model_id=request.model_id,
                training_op="save_weights_for_sampler",
                seq_id=request.seq_id,
            ),
            request=http_request,
        )
        scheduler_extra["prefer_tinker"] = bool(prefer_tinker)
        scheduler_extra["is_admin"] = can_manage_system(http_request)
        await _enqueue_training_model_work_route(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.save_weights_for_sampler",
            request_json=request_json,
            user_id=user_id,
            extra=scheduler_extra,
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            queued_meta=_build_training_queued_meta(
                op="save_weights_for_sampler",
                model_id=request.model_id,
                session=route_session_info,
                seq_id=request.seq_id,
            ),
        )
    except Exception as e:
        if inflight_marked:
            await _mark_training_inflight(request.model_id, -1)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue save_weights_for_sampler request: {e}",
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_save_weights_for_sampler(
    request_id: str,
    request: SaveWeightsForSamplerRequest,
    user_id: str | None,
    prefer_tinker: bool,
    is_admin: bool = False,
) -> None:
    """Background task for save_weights_for_sampler.

    Two flows:
    - Named (path is not None): Save to persistent location, return path
    - Ephemeral (path is None): Use per-session inference engine for isolated concurrent access
    """
    claimed_ckpt_id: str | None = None
    mirror_started = False
    save_path: str | None = None
    persistent_path: str | None = None
    sampling_session_id: str | None = None
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        session = manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        session = await _materialize_training_session_for_stateful_use(session)
        # Determine checkpoint name
        if request.path is not None:
            # Named save - use provided path
            checkpoint_name = request.path.strip()
            if (
                not checkpoint_name
                or checkpoint_name in (".", "..")
                or "/" in checkpoint_name
                or "\\" in checkpoint_name
            ):
                raise ValueError(f"Invalid checkpoint name: {request.path!r}")
            created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            claimed_ckpt_id = await _claim_sampler_checkpoint_or_raise(
                owner_id=None if is_admin else user_id,
                model_id=session.model_id,
                raw_checkpoint_id=checkpoint_name,
                model_name=session.base_model,
                checkpoint_created_at=created_at,
                retry=bool(request.retry),
            )
            save_path = build_persistent_cache_dir(
                user_id=None if is_admin else user_id,
                model_id=session.model_id,
                checkpoint_name=checkpoint_name,
                checkpoint_type="sampler",
            )
        else:
            # Ephemeral save - generate unique temp name
            checkpoint_name = f"_ephemeral_{uuid.uuid4().hex[:8]}"
            created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            save_path = build_ephemeral_checkpoint_dir(
                user_id=None if is_admin else user_id,
                model_id=session.model_id,
                checkpoint_name=checkpoint_name,
                checkpoint_type="sampler",
            )

        train_mlp = bool(
            getattr(getattr(session, "lora_config", None), "train_mlp", False)
        )

        # Save weights
        checkpoint_export_t0 = time.perf_counter()
        await _safe_update_training_meta(
            request_id,
            {
                "stage": "checkpoint_export",
                "checkpoint_export_started_at": time.time(),
            },
        )
        save_path = await run_async_with_otel_span(
            "training.save_weights_for_sampler.execute",
            lambda: engine.save_weights_for_sampler(
                session=session,
                checkpoint_name=checkpoint_name,
                checkpoint_base_dir=os.path.dirname(
                    os.path.dirname(os.path.dirname(save_path))
                ),
                checkpoint_type="sampler",
            ),
            component="routes.training",
            op="training.save_weights_for_sampler",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "base_model": str(session.base_model),
                "backend": str(session.backend),
                "save_mode": "named" if request.path is not None else "ephemeral",
                "train_mlp": bool(train_mlp),
            },
        )
        await _safe_update_training_meta(
            request_id,
            {
                "stage": "validate_checkpoint",
                "checkpoint_export_s": max(
                    0.0, time.perf_counter() - checkpoint_export_t0
                ),
            },
        )

        validate_checkpoint_t0 = time.perf_counter()
        with start_as_current_span(
            "training.save_weights_for_sampler.validate_checkpoint",
            component="routes.training",
            op="training.save_weights_for_sampler.validate_checkpoint",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "save_path": str(save_path),
                "save_mode": "named" if request.path is not None else "ephemeral",
            },
        ):
            if checkpoint_has_optimizer_state(save_path):
                raise RuntimeError(
                    f"save_weights_for_sampler must not produce optimizer artifacts, but found some under: {save_path}"
                )
            try:
                validate_sampler_checkpoint_for_sampling(save_path)
            except ValueError as e:
                raise RuntimeError(
                    f"save_weights_for_sampler produced an invalid sampler checkpoint at {save_path}: {e}"
                ) from e

        await _safe_update_training_meta(
            request_id,
            {
                "stage": "write_checkpoint_metadata",
                "validate_checkpoint_s": max(
                    0.0, time.perf_counter() - validate_checkpoint_t0
                ),
            },
        )
        ttl_seconds = request.ttl_seconds
        if request.path is None and ttl_seconds is None:
            ttl_seconds = None
        write_metadata_t0 = time.perf_counter()
        with start_as_current_span(
            "training.save_weights_for_sampler.write_checkpoint_metadata",
            component="routes.training",
            op="training.save_weights_for_sampler.write_checkpoint_metadata",
            request_id=str(request_id),
            attributes={
                "model_id": str(request.model_id),
                "checkpoint_name": str(checkpoint_name),
                "storage_tier": "ephemeral_pfs"
                if request.path is None
                else "persistent_cache",
                "ttl_seconds": None
                if request.ttl_seconds is None
                else int(request.ttl_seconds),
            },
        ):
            write_checkpoint_metadata(
                save_path,
                {
                    "checkpoint_id": checkpoint_name,
                    "owner_id": None if is_admin else user_id,
                    "model_id": session.model_id,
                    "model_name": session.base_model,
                    "created_at": created_at,
                    "step": session.current_step,
                    "checkpoint_type": "sampler",
                    "optimizer_present": False,
                    "backend": session.backend,
                    "type": "sampler",
                    "storage_tier": "ephemeral_pfs"
                    if request.path is None
                    else "persistent_cache",
                    "ttl_seconds": request.ttl_seconds,
                    "ckpt_id": claimed_ckpt_id,
                },
            )
        await _safe_update_training_meta(
            request_id,
            {
                "stage": "checkpoint_ready",
                "write_checkpoint_metadata_s": max(
                    0.0, time.perf_counter() - write_metadata_t0
                ),
            },
        )

        persistent_path = None
        if request.path is not None:
            mirror_t0 = time.perf_counter()
            await _safe_update_training_meta(
                request_id,
                {
                    "stage": "begin_async_checkpoint_mirror",
                    "mirror_started_at": time.time(),
                },
            )
            with start_as_current_span(
                "training.save_weights_for_sampler.begin_async_checkpoint_mirror",
                component="routes.training",
                op="training.save_weights_for_sampler.begin_async_checkpoint_mirror",
                request_id=str(request_id),
                attributes={
                    "model_id": str(request.model_id),
                    "checkpoint_name": str(checkpoint_name),
                    "save_path": str(save_path),
                },
            ):
                persistent_path = begin_async_checkpoint_mirror(
                    save_path,
                    user_id=None if is_admin else user_id,
                    model_id=session.model_id,
                    checkpoint_name=checkpoint_name,
                    checkpoint_type="sampler",
                )
                mirror_started = True
            await _safe_update_training_meta(
                request_id,
                {
                    "stage": "checkpoint_ready",
                    "begin_async_checkpoint_mirror_s": max(
                        0.0, time.perf_counter() - mirror_t0
                    ),
                },
            )

        from ..client_compat import checkpoint_uri

        tinker_uri = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=True,
            checkpoint_type="sampler",
        )
        mint_uri = checkpoint_uri(
            session.model_id,
            checkpoint_name,
            prefer_tinker=False,
            checkpoint_type="sampler",
        )
        path_uri = tinker_uri if prefer_tinker else mint_uri
        checkpoint_owner_id = (None if is_admin else user_id) or "anonymous"

        if request.path is not None:
            # Named flow: Return path, caller creates session separately
            response = SaveWeightsForSamplerResponse(
                path=path_uri,
                sampling_session_id=None,
                owner_id=checkpoint_owner_id,
            ).model_dump()
            response.update(
                checkpoint_record_id=claimed_ckpt_id,
                filesystem_path=save_path,
                persistent_filesystem_path=persistent_path,
                mirror_status=MIRROR_STATUS_PENDING,
                storage_tier="persistent_cache",
                mirror_error=None,
            )
        else:
            # Ephemeral flow: Use multi-LoRA engine for frozen per-session weights
            # Each sampling session gets unique lora_int_id with frozen weights.
            # Matches Tinker SDK semantics where each save creates isolated snapshot.
            inf_mgr = _current_inference_manager()
            if inf_mgr is None:
                raise RuntimeError("Inference manager not initialized")

            sampling_session_id = str(uuid.uuid4())
            lora_rank = session.lora_config.rank if session.lora_config else 32
            base_model = session.base_model

            # Do not block save-time on vLLM actor cold-start. Kick off engine
            # warm in the background, but still fail fast if the warm task trips
            # an immediate configuration or capacity error before we return a
            # sampling_session_id. Allow one short grace window so async failures
            # that happen right after the first await still surface on save.
            warm_traceparent = get_current_traceparent()

            async def _warm_engine() -> None:
                with start_as_current_span_from_traceparent(
                    "training.save_weights_for_sampler.background_engine_warm",
                    traceparent=warm_traceparent,
                    component="routes.training",
                    op="training.save_weights_for_sampler.background_engine_warm",
                    request_id=str(request_id),
                    attributes={
                        "model_id": str(request.model_id),
                        "sampling_session_id": str(sampling_session_id),
                        "base_model": str(base_model),
                        "save_mode": "ephemeral",
                    },
                ):
                    await inf_mgr.get_engine_for_model(base_model)

            pending_warms = getattr(inf_mgr, "_background_engine_warm_tasks", None)
            if not isinstance(pending_warms, dict):
                pending_warms = {}
                setattr(inf_mgr, "_background_engine_warm_tasks", pending_warms)

            existing_warm = pending_warms.get(base_model)
            warm_schedule_t0 = time.perf_counter()
            await _safe_update_training_meta(
                request_id,
                {
                    "stage": "schedule_background_engine_warm",
                    "engine_warm_started_at": time.time(),
                },
            )
            with start_as_current_span(
                "training.save_weights_for_sampler.schedule_background_engine_warm",
                component="routes.training",
                op="training.save_weights_for_sampler.schedule_background_engine_warm",
                request_id=str(request_id),
                attributes={
                    "model_id": str(request.model_id),
                    "sampling_session_id": str(sampling_session_id),
                    "base_model": str(base_model),
                    "warm_timeout_s": 0.05,
                    "reused_existing_task": bool(
                        isinstance(existing_warm, asyncio.Task)
                        and not existing_warm.done()
                    ),
                },
            ):
                if isinstance(existing_warm, asyncio.Task) and not existing_warm.done():
                    warm_task = existing_warm
                else:
                    warm_task = asyncio.create_task(_warm_engine())
                    pending_warms[base_model] = warm_task

                    def _log_warm_failure(task: asyncio.Task[object]) -> None:
                        if pending_warms.get(base_model) is task:
                            pending_warms.pop(base_model, None)
                        if task.cancelled():
                            return
                        try:
                            exc = task.exception()
                        except Exception:
                            return
                        if exc is not None:
                            logger.warning(
                                "[save_weights_for_sampler] background engine warm failed: "
                                "model=%s err=%s",
                                base_model,
                                exc,
                            )

                    warm_task.add_done_callback(_log_warm_failure)

                done, _pending = await asyncio.wait({warm_task}, timeout=0.05)
            warm_completed_inline = warm_task in done
            warm_reused = bool(
                isinstance(existing_warm, asyncio.Task) and not existing_warm.done()
            )

            if warm_task in done:
                await warm_task

            # Multi-LoRA mode: register the sampling session immediately and let
            # /asample load the adapter lazily on first use. This matches
            # create_sampling_session() semantics and avoids blocking save-time
            # on engine cold-start or vLLM's exclusive-engine gate while another
            # generate is active.
            with start_as_current_span(
                "training.save_weights_for_sampler.register_sampling_session",
                component="routes.training",
                op="training.save_weights_for_sampler.register_sampling_session",
                request_id=str(request_id),
                attributes={
                    "model_id": str(request.model_id),
                    "sampling_session_id": str(sampling_session_id),
                    "base_model": str(base_model),
                    "lora_rank": int(lora_rank),
                    "warm_reused": bool(warm_reused),
                    "warm_completed_inline": bool(warm_completed_inline),
                },
            ):
                inf_mgr.register_multi_lora_session(
                    session_id=sampling_session_id,
                    base_model=base_model,
                    lora_rank=lora_rank,
                    adapter_path=save_path,
                    lora_loaded=False,
                )
            await _safe_update_training_meta(
                request_id,
                {
                    "stage": "session_index_write",
                    "engine_warm_schedule_s": max(
                        0.0, time.perf_counter() - warm_schedule_t0
                    ),
                    "sampling_session_id": str(sampling_session_id),
                },
            )

            logger.info(
                f"[save_weights_for_sampler] Multi-LoRA: registered lazy-load session "
                f"{sampling_session_id} (model={base_model}, path={save_path})"
            )

            try:
                from ..backend.session_index_store import (
                    add_heartbeat_sampler_to_session,
                    upsert_sampler_index,
                )

                created_at = datetime.now().isoformat()
                session_index_t0 = time.perf_counter()
                with start_as_current_span(
                    "training.save_weights_for_sampler.session_index_write",
                    component="routes.training",
                    op="training.save_weights_for_sampler.session_index_write",
                    request_id=str(request_id),
                    attributes={
                        "model_id": str(request.model_id),
                        "sampling_session_id": str(sampling_session_id),
                        "session_id": str(session.session_id),
                        "checkpoint_name": str(checkpoint_name),
                    },
                ):
                    add_heartbeat_sampler_to_session(
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
                await _safe_update_training_meta(
                    request_id,
                    {
                        "stage": "ready",
                        "session_index_write_s": max(
                            0.0, time.perf_counter() - session_index_t0
                        ),
                    },
                )
            except Exception as e:
                logger.warning(
                    "[save_weights_for_sampler] session index write failed: %s", e
                )

            response = SaveWeightsForSamplerResponse(
                path=None,  # Ephemeral - no path returned
                sampling_session_id=sampling_session_id,
                owner_id=checkpoint_owner_id,
            ).model_dump()

        await task_futures.async_resolve(request_id, response)

    except Exception as e:
        if not mirror_started:
            await _mark_checkpoint_failed_safe(
                claimed_ckpt_id, fail_reason="upload_error"
            )
        inf_mgr = _current_inference_manager()
        if sampling_session_id is not None and inf_mgr is not None:
            try:
                await inf_mgr.end_session(sampling_session_id)
            except Exception as cleanup_error:
                logger.warning(
                    "[save_weights_for_sampler] failed to cleanup sampling session %s: %s: %s",
                    sampling_session_id,
                    type(cleanup_error).__name__,
                    cleanup_error,
                )
        if not mirror_started:
            for candidate in (persistent_path, save_path):
                try:
                    _cleanup_generated_checkpoint_dir(candidate)
                except Exception as cleanup_error:
                    logger.warning(
                        "[save_weights_for_sampler] failed to cleanup checkpoint path %s: %s: %s",
                        candidate,
                        type(cleanup_error).__name__,
                        cleanup_error,
                    )
        logger.exception(
            "[save_weights_for_sampler] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_checkpoint_export_and_inference_registration",
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(request.model_id, -1)


# =============================================================================
# Model info endpoints
# =============================================================================


def _owner_visible(request_user_data: dict | None, owner: str | None) -> bool:
    request_user_id = (
        str(request_user_data.get("user_id"))
        if request_user_data and request_user_data.get("user_id")
        else None
    )
    if request_user_id is None:
        return True
    if can_bypass_ownership_user_data(request_user_data):
        return True
    return bool(owner) and owner == request_user_id


async def _get_authoritative_training_info(model_id: str) -> dict[str, Any] | None:
    info = await _get_training_route_session_info(model_id)
    if isinstance(info, dict):
        _refresh_training_session_from_info_if_needed(model_id, info)
        return info
    _drop_local_training_session(model_id)
    return None


async def _list_authoritative_training_infos() -> dict[str, dict[str, Any]]:
    from ..backend.training_session_store import async_list_training_sessions

    infos_by_id: dict[str, dict[str, Any]] = {}
    for info in await async_list_training_sessions():
        model_id = info.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        info = dict(info)
        infos_by_id[model_id] = info
        _refresh_training_session_from_info_if_needed(model_id, info)

    if training_manager is not None:
        for model_id, session in list(
            getattr(training_manager, "_sessions", {}).items()
        ):
            if model_id in infos_by_id:
                continue
            if not bool(getattr(session, "pending_persist", False)):
                _drop_local_training_session(model_id)
    return infos_by_id


@router.get("/training_runs/{training_run_id}", response_model=TrainingRun)
async def get_training_run(training_run_id: str, http_request: Request) -> TrainingRun:
    request_user_data = _get_user_data(http_request)
    try:
        info = await _get_authoritative_training_info(training_run_id)
    except Exception as e:
        raise HTTPException(
            status_code=503, detail="Training session store unavailable"
        ) from e

    if not isinstance(info, dict):
        raise HTTPException(
            status_code=404, detail=f"Training run '{training_run_id}' not found"
        )

    if not _owner_visible(request_user_data, info.get("user_id")):
        raise HTTPException(
            status_code=404, detail=f"Training run '{training_run_id}' not found"
        )

    if "model_id" not in info:
        info = dict(info)
        info["model_id"] = training_run_id

    return _training_run_from_info(info)


@router.get("/training_runs", response_model=TrainingRunsResponse)
async def list_training_runs(
    limit: int = 20,
    offset: int = 0,
    http_request: Request = cast(Request, None),
) -> TrainingRunsResponse:
    request_user_data = _get_user_data(http_request) if http_request else None
    try:
        infos_by_id = await _list_authoritative_training_infos()
    except Exception as e:
        raise HTTPException(
            status_code=503, detail="Training session store unavailable"
        ) from e

    infos = [
        info
        for info in infos_by_id.values()
        if _owner_visible(request_user_data, info.get("user_id"))
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
    try:
        info = await _get_authoritative_training_info(model_id)
    except Exception as e:
        raise HTTPException(
            status_code=503, detail="Training session store unavailable"
        ) from e
    if not isinstance(info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return {
        "model_id": str(info.get("model_id") or model_id),
        "session_id": str(info.get("session_id") or ""),
        "model_seq_id": int(info.get("model_seq_id") or 0),
        "base_model": str(info.get("base_model") or ""),
        "lora_config": info.get("lora_config"),
        "user_metadata": info.get("user_metadata") or {},
        "learning_rate": float(info.get("learning_rate") or 1e-4),
        "created_at": info.get("created_at"),
        "current_step": int(info.get("current_step") or 0),
        "is_active": bool(info.get("is_active", True)),
        "backend": str(info.get("backend") or "peft"),
        "user_id": info.get("user_id"),
        "last_activity": info.get("last_activity"),
        "idle_for_s": max(0.0, time.time() - float(info.get("last_activity") or 0.0))
        if info.get("last_activity") is not None
        else None,
    }


@router.get("/models/{model_id}/session_guard_state")
async def get_session_guard_state(model_id: str):
    """Get megatron contamination/block guard state for one training model."""
    _session, route_session_info = await _resolve_training_route_session(model_id)
    if route_session_info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    request_id = await _enqueue_internal_serialized_model_op(
        model_id=model_id,
        op="training.get_session_guard_state",
        request_json=json.dumps({"model_id": model_id}).encode("utf-8"),
        extra=_build_training_scheduler_extra(
            session=route_session_info,
            model_id=model_id,
            training_op="get_session_guard_state",
        ),
        user_id=str(route_session_info.get("user_id") or ""),
    )
    try:
        guard_state = await _wait_internal_future_result(request_id)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(f"Failed to query session guard state: {type(e).__name__}: {e}"),
        )
    return {
        "model_id": model_id,
        "backend": str(route_session_info.get("backend") or "peft"),
        "guard_state": guard_state,
    }


@router.post("/get_info", response_model=GetInfoResponse)
async def get_info(request: GetInfoRequest, http_request: Request) -> GetInfoResponse:
    """Get model info (tinker client compatible endpoint).

    Returns model architecture, tokenizer, and LoRA configuration.
    """
    from ..gateway import async_remote_training_model, forward_json, upstream_for_alias

    info = None
    try:
        info = await _get_authoritative_training_info(request.model_id)
    except Exception:
        info = None

    if info is None:
        remote = await async_remote_training_model(request.model_id)
        if remote is not None:
            upstream_alias, base_model = remote
            upstream = upstream_for_alias(upstream_alias)
            if upstream is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gateway misconfig: unknown upstream alias {upstream_alias!r}",
                )
            user_data = _get_user_data(http_request)
            if not can_access_model(base_model, user_data):
                raise HTTPException(
                    status_code=403, detail=get_access_denied_error(base_model)
                )

            try:
                resp = await forward_json(
                    upstream=upstream,
                    method="POST",
                    path="/api/v1/get_info",
                    incoming_headers=dict(http_request.headers),
                    json_body=request.model_dump(),
                    timeout_s=120.0,
                )
            except Exception:
                logger.exception("Upstream get_info failed: %s", upstream_alias)
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream {upstream_alias!r} get_info failed",
                )
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return GetInfoResponse.model_validate(resp.json())

    if info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    lora_cfg = (
        info.get("lora_config") if isinstance(info.get("lora_config"), dict) else None
    )
    lora_rank = lora_cfg.get("rank") if isinstance(lora_cfg, dict) else None
    is_lora = lora_cfg is not None
    base_model = str(info.get("base_model") or "")
    model_id = str(info.get("model_id") or request.model_id)

    return GetInfoResponse(
        model_id=model_id,
        model_data=ModelData(
            arch="transformer",
            model_name=base_model,
            tokenizer_id=base_model,
        ),
        model_name=base_model,
        is_lora=is_lora,
        lora_rank=lora_rank,
    )


@router.get("/models")
async def list_models():
    """List all training models."""
    try:
        infos_by_id = await _list_authoritative_training_infos()
    except Exception as e:
        raise HTTPException(
            status_code=503, detail="Training session store unavailable"
        ) from e

    infos = list(infos_by_id.values())
    infos.sort(key=lambda info: str(info.get("model_id") or ""))
    return {
        "models": [
            {
                "model_id": str(info.get("model_id") or ""),
                "session_id": str(info.get("session_id") or ""),
                "model_seq_id": int(info.get("model_seq_id") or 0),
                "base_model": str(info.get("base_model") or ""),
                "created_at": info.get("created_at"),
                "current_step": int(info.get("current_step") or 0),
                "is_active": bool(info.get("is_active", True)),
                "last_activity": info.get("last_activity"),
                "idle_for_s": max(
                    0.0, time.time() - float(info.get("last_activity") or 0.0)
                )
                if info.get("last_activity") is not None
                else None,
                "backend": str(info.get("backend") or "peft"),
            }
            for info in infos
        ],
        "total": len(infos),
    }


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, http_request: Request):
    """Delete a training model and release resources."""
    _require_write_access(http_request)

    _session, route_session_info = await _resolve_training_route_session(model_id)
    if route_session_info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    request_id = await _enqueue_internal_serialized_model_op(
        model_id=model_id,
        op="training.delete_model",
        request_json=json.dumps({"model_id": model_id}).encode("utf-8"),
        extra=_build_training_scheduler_extra(
            session=route_session_info,
            model_id=model_id,
            training_op="delete_model",
        ),
        user_id=str(route_session_info.get("user_id") or ""),
    )
    try:
        return await _wait_internal_future_result(request_id)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "[training.delete_model] Failed model_id=%s error=%s", str(model_id), e
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _do_delete_model(request_id: str, model_id: str) -> None:
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")

        session = manager.get_session(model_id)
        if session is not None:
            await engine.shutdown_session(session)
            manager.delete_session(model_id)

        try:
            from ..backend.training_session_store import delete_training_session

            delete_training_session(model_id)
        except Exception:
            pass
        try:
            from ..backend.model_actor_supervisor import get_model_actor_supervisor

            get_model_actor_supervisor().clear_session(model_id)
        except Exception:
            pass

        await task_futures.async_resolve(
            request_id, {"model_id": model_id, "status": "deleted"}
        )
    except Exception as e:
        logger.exception(
            "[training.delete_model] failed request_id=%s model_id=%s error_type=%s error=%s",
            str(request_id),
            str(model_id),
            type(e).__name__,
            e,
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(model_id, -1)


async def _do_get_session_guard_state(request_id: str, model_id: str) -> None:
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")
        session = manager.get_session(model_id)
        if session is None:
            session = await _restore_training_session(model_id)
        if session is None:
            raise RuntimeError(f"Model '{model_id}' not found")
        guard_state = await engine.get_session_guard_state(session)
        await task_futures.async_resolve(request_id, guard_state)
    except Exception as e:
        logger.exception(
            "[training.get_session_guard_state] failed request_id=%s model_id=%s error_type=%s error=%s",
            str(request_id),
            str(model_id),
            type(e).__name__,
            e,
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(model_id, -1)


async def _get_control_plane_tokenizer_info(
    model_id: str, info: dict[str, Any]
) -> dict[str, Any]:
    from ..backend.training_session_manager import (
        MATERIALIZATION_STATE_READY,
        TRAINING_SESSION_METADATA_VERSION,
    )

    backend = str(
        info.get("backend")
        or _infer_training_backend_for_base_model(str(info.get("base_model") or ""))
    )
    tokenizer_info = info.get("tokenizer_info")
    if isinstance(tokenizer_info, dict):
        return dict(tokenizer_info)

    tokenizer_metadata = await asyncio.to_thread(
        _build_local_tokenizer_metadata,
        str(info.get("base_model") or ""),
        backend,
    )

    backfill_payload = {
        "model_id": str(model_id),
        "metadata_version": max(
            int(info.get("metadata_version") or 1) + 1,
            TRAINING_SESSION_METADATA_VERSION,
        ),
        "materialization_state": str(
            info.get("materialization_state") or MATERIALIZATION_STATE_READY
        ),
        "tokenizer_info": dict(tokenizer_metadata.get("tokenizer_info") or {}),
    }
    if tokenizer_metadata.get("tokenizer_identity") is not None:
        backfill_payload["tokenizer_identity"] = tokenizer_metadata.get(
            "tokenizer_identity"
        )
    if tokenizer_metadata.get("tokenizer_source_path") is not None:
        backfill_payload["tokenizer_source_path"] = tokenizer_metadata.get(
            "tokenizer_source_path"
        )
    try:
        from ..backend.training_session_store import async_upsert_training_session

        await async_upsert_training_session(backfill_payload)
        merged_info = dict(info)
        merged_info.update(backfill_payload)
        _refresh_training_session_from_info_if_needed(model_id, merged_info)
    except Exception as e:
        logger.warning(
            "[get_tokenizer] detached metadata backfill failed model_id=%s error_type=%s error=%s",
            str(model_id),
            type(e).__name__,
            e,
        )
    return dict(tokenizer_metadata["tokenizer_info"])


async def _get_runtime_tokenizer_info(
    model_id: str, info: dict[str, Any]
) -> dict[str, Any]:
    request_id = await _enqueue_internal_serialized_model_op(
        model_id=model_id,
        op="training.get_tokenizer_info",
        request_json=json.dumps({"model_id": model_id}).encode("utf-8"),
        extra=_build_training_scheduler_extra(
            session=info,
            model_id=model_id,
            training_op="get_tokenizer_info",
        ),
        user_id=str(info.get("user_id") or ""),
    )
    payload = await _wait_internal_future_result(request_id)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"training.get_tokenizer_info returned non-dict payload: {type(payload).__name__}"
        )
    return dict(payload)


async def _do_get_tokenizer_info(request_id: str, model_id: str) -> None:
    try:
        set_request_id(request_id)
        engine = _current_training_engine()
        manager = _current_training_manager()
        if engine is None or manager is None:
            raise RuntimeError("Training engine not initialized")
        session = manager.get_session(model_id)
        if session is None:
            session = await _restore_training_session(model_id)
        if session is None:
            raise RuntimeError(f"Model '{model_id}' not found")
        tokenizer_info = await engine.get_tokenizer_info(session)
        await task_futures.async_resolve(request_id, dict(tokenizer_info))
    except Exception as e:
        logger.exception(
            "[training.get_tokenizer_info] failed request_id=%s model_id=%s error_type=%s error=%s",
            str(request_id),
            str(model_id),
            type(e).__name__,
            e,
        )
        await task_futures.async_fail(request_id, str(e))
    finally:
        await _mark_training_inflight(model_id, -1)


@router.get("/models/{model_id}/tokenizer")
async def get_tokenizer(model_id: str):
    """Get tokenizer configuration for a training model.

    Returns tokenizer info (vocab_size, special tokens, etc.)
    for client-side tokenization.
    """
    info = await _get_training_route_session_info(model_id)
    if not isinstance(info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    _refresh_training_session_from_info_if_needed(model_id, info)

    backend = str(
        info.get("backend")
        or _infer_training_backend_for_base_model(str(info.get("base_model") or ""))
    )
    try:
        if _supports_control_plane_tokenizer_metadata(backend):
            tokenizer_info = await _get_control_plane_tokenizer_info(model_id, info)
        else:
            tokenizer_info = await _get_runtime_tokenizer_info(model_id, info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load tokenizer for model {model_id!r}: {type(e).__name__}: {e}",
        ) from e

    return {
        "model_id": model_id,
        "tokenizer": tokenizer_info,
    }
