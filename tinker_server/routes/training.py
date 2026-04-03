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

import asyncio
import inspect
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request

from ..auth_identity import get_user_data as _request_user_data
from ..auth_identity import get_user_id as _request_user_id
from ..auth_identity import is_admin_request, is_admin_user_data
from ..backend.async_ray_control import async_lookup_actor_handle
from ..gateway_auth import GatewayAuthContext, build_billing_auth_context
from ..logging_context import (
    classify_failure_reason,
    get_otel_tracer,
    run_async_with_otel_span,
    set_request_id,
)

from ..backend.future_store import FutureStatus, future_store
from ..checkpoints import (
    MIRROR_STATUS_PENDING,
    begin_async_checkpoint_mirror,
    build_ephemeral_checkpoint_dir,
    build_gateway_proxy_archive_path,
    build_persistent_cache_dir,
    checkpoint_has_optimizer_state,
    async_create_checkpoint_archive,
    ensure_checkpoint_path_allowed,
    materialize_persistent_checkpoint,
    resolve_checkpoint_path,
    validate_sampler_checkpoint_for_sampling,
    write_checkpoint_metadata,
)
from ..config import RAY_NAMESPACE
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
from ..usage_store import UsageEvent, get_usage_store
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


def _mark_training_inflight(model_id: str, delta: int) -> None:
    if training_manager is None:
        return
    mark = getattr(training_manager, "mark_inflight", None)
    if callable(mark):
        mark(model_id, delta)


async def _fail_future(request_id: str, error: str) -> None:
    async_fail = getattr(future_store, "async_fail", None)
    if callable(async_fail):
        await async_fail(request_id, error)
        return
    fail = getattr(future_store, "fail", None)
    if callable(fail):
        fail(request_id, error)
        return
    raise AttributeError("future_store has neither async_fail nor fail")


def _get_user_data(request: Request) -> dict | None:
    """Extract full user_data from request state (set by auth middleware)."""
    return _request_user_data(request)


def _get_user_id(request: Request) -> str | None:
    """Extract user_id from request state (set by auth middleware)."""
    return _request_user_id(request)


def _build_training_usage_label(*, model: str, route: str) -> str:
    return f"model={model},route={route},dimension=train"


def _training_heartbeat_stale_timeout_s() -> float:
    raw = os.environ.get("MINT_TRAINING_HEARTBEAT_STALE_S", "300")
    try:
        return max(0.0, float(raw))
    except Exception:
        logger.warning("Invalid MINT_TRAINING_HEARTBEAT_STALE_S=%r; defaulting to 300s", raw)
        return 300.0


async def _persist_usage_events(*, events: list[UsageEvent]) -> None:
    usage_store = await get_usage_store()
    await usage_store.write_events(events)


async def _enqueue_training_request_with_trace(
    *,
    route_start_s: float,
    request_id: str,
    op: str,
    enqueue_coro,
    model_id: str | None = None,
    base_model: str | None = None,
    backend: str | None = None,
) -> None:
    tracer = get_otel_tracer()
    future_ready_elapsed_ms = (time.perf_counter() - route_start_s) * 1000.0
    if tracer is None:
        await enqueue_coro
        return

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
            "future_store_ready",
            {
                "elapsed_ms": round(future_ready_elapsed_ms, 3),
                "route_elapsed_ms": round(future_ready_elapsed_ms, 3),
            },
        )
        enqueue_start_s = time.perf_counter()
        await enqueue_coro
        span.add_event(
            "enqueue_done",
            {
                "elapsed_ms": round((time.perf_counter() - enqueue_start_s) * 1000.0, 3),
                "route_elapsed_ms": round((time.perf_counter() - route_start_s) * 1000.0, 3),
            },
        )


def _get_webhook_url(request: Request) -> str | None:
    """Extract webhook_url from request state (set by auth middleware)."""
    user_data = _get_user_data(request)
    if user_data:
        return user_data.get("webhook_url")
    return None



def _find_actor_handle(actor_name: str, namespace: str):
    from ..backend.resource_pool import get_resource_pool

    pool = get_resource_pool()
    for entry in pool.iter_entries():
        if entry.actor_name == actor_name and entry.namespace == namespace and entry.actor_handle:
            return entry.actor_handle
    return None


def _snapshot_from_training_session(model_id: str):
    if training_manager is None:
        return None
    get_session = getattr(training_manager, "get_session", None)
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
            rollout_correction_config=getattr(session, "rollout_correction_config", None),
            user_metadata=dict(getattr(session, "user_metadata", {}) or {}),
            learning_rate=float(getattr(session, "learning_rate", 1e-4) or 1e-4),
            metadata_version=max(1, int(getattr(session, "metadata_version", 1) or 1)),
        )
    except Exception:
        return None


def _get_training_snapshot(model_id: str):
    if training_manager is None:
        return None
    get_snapshot = getattr(training_manager, "get_training_session_snapshot", None)
    if callable(get_snapshot):
        snapshot = get_snapshot(model_id)
        if snapshot is not None:
            return snapshot
    return _snapshot_from_training_session(model_id)


def _drop_local_training_session(model_id: str) -> None:
    if training_manager is not None:
        delete_session = getattr(training_manager, "delete_session", None)
        if callable(delete_session):
            try:
                delete_session(model_id)
            except Exception as e:
                logger.warning("Failed to delete stale local training session %s: %s", model_id, e)
    if training_engine is not None:
        getattr(training_engine, "_workers", {}).pop(model_id, None)
        getattr(training_engine, "_resource_pool_actor_names", {}).pop(model_id, None)


def _refresh_training_session_from_info_if_needed(model_id: str, info: dict, snapshot=None):
    if training_manager is None or not isinstance(info, dict):
        return snapshot
    snap = snapshot or _get_training_snapshot(model_id)
    if snap is None:
        _restore_training_session_info_compat(info)
        return _get_training_snapshot(model_id)

    incoming_version = max(1, int(info.get("metadata_version") or 1))
    incoming_step = max(0, int(info.get("current_step") or 0))
    current_version = max(1, int(getattr(snap, "metadata_version", 1) or 1))
    current_step = max(0, int(getattr(snap, "current_step", 0) or 0))
    if incoming_version <= current_version and incoming_step <= current_step:
        return snap
    _restore_training_session_info_compat(info)
    return _get_training_snapshot(model_id)


def _restore_training_session_info_compat(info: dict):
    if training_manager is None:
        return None
    restore = getattr(training_manager, "restore_training_session_info", None)
    if callable(restore):
        return restore(info)

    get_session = getattr(training_manager, "get_session", None)
    create_session = getattr(training_manager, "create_session", None)
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
            ),
        )

    session.session_id = session_id
    session.model_seq_id = int(info.get("model_seq_id", getattr(session, "model_seq_id", 0)) or 0)
    session.base_model = base_model
    session.rollout_correction_config = info.get("rollout_correction_config")
    session.user_metadata = info.get("user_metadata") or {}
    session.user_id = info.get("user_id")
    session.learning_rate = float(info.get("learning_rate", getattr(session, "learning_rate", 1e-4)) or 1e-4)
    session.backend = str(info.get("backend", getattr(session, "backend", "peft")) or "peft")
    session.current_step = int(info.get("current_step", getattr(session, "current_step", 0)) or 0)
    session.metadata_version = max(1, int(info.get("metadata_version", getattr(session, "metadata_version", 1)) or 1))
    return session


async def _restore_training_session(model_id: str):
    """Best-effort restore of a training session after API process restart."""
    if training_engine is None or training_manager is None:
        return None
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
        if not isinstance(info, dict):
            return None

        session = training_manager.get_session(model_id)
        created_session = False
        original_session_state = None
        if session is None:
            session = _restore_training_session_info_compat(info)
            created_session = session is not None
        else:
            original_session_state = {
                "backend": getattr(session, "backend", "peft"),
                "created_at": getattr(session, "created_at", ""),
                "current_step": getattr(session, "current_step", 0),
                "is_active": getattr(session, "is_active", False),
                "metadata_version": getattr(session, "metadata_version", 1),
            }
            session = _restore_training_session_info_compat(info) or session

        if session is None:
            return None

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
                    session.last_activity = datetime.fromisoformat(created_at).timestamp()
                    last_activity_set = True
                except (ValueError, OSError):
                    pass
        if not last_activity_set:
            session.last_activity = 0.0
        try:
            session.current_step = int(info.get("current_step", session.current_step))
        except Exception:
            pass
        session.is_active = True

        actor_name = info.get("actor_name")
        if actor_name:
            namespace = str(info.get("namespace") or RAY_NAMESPACE)
            worker = _find_actor_handle(actor_name, namespace)
            if worker is None:
                try:
                    worker = await async_lookup_actor_handle(actor_name, namespace)
                except Exception as lookup_error:
                    logger.warning(
                        "[training restore] async actor lookup failed model_id=%s actor_name=%s namespace=%s error_type=%s error=%s",
                        model_id,
                        actor_name,
                        namespace,
                        type(lookup_error).__name__,
                        lookup_error,
                    )
                    worker = None
            if worker is None:
                logger.warning(
                    "[training restore] missing worker handle model_id=%s actor_name=%s namespace=%s created_session=%s",
                    model_id,
                    actor_name,
                    namespace,
                    created_session,
                )
                if created_session:
                    training_manager.delete_session(model_id)
                elif original_session_state is not None:
                    session.backend = original_session_state["backend"]
                    session.created_at = original_session_state["created_at"]
                    session.current_step = original_session_state["current_step"]
                    session.is_active = original_session_state["is_active"]
                    session.metadata_version = original_session_state["metadata_version"]
                return None
            getattr(training_engine, "_workers", {})[model_id] = worker
            getattr(training_engine, "_resource_pool_actor_names", {})[model_id] = actor_name
        else:
            logger.warning("[training restore] store entry missing actor_name model_id=%s info=%s", model_id, info)

        logger.info(
            "[training restore] restored detached training session model_id=%s actor_name=%s backend=%s step=%s",
            model_id,
            info.get("actor_name"),
            getattr(session, "backend", None),
            getattr(session, "current_step", None),
        )
        return session
    except Exception as e:
        logger.exception(f"[{model_id}] restore_training_session failed: {e}")
        return None


async def _refresh_training_snapshot_if_needed(model_id: str, snapshot):
    if training_manager is None:
        return snapshot
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception:
        return snapshot
    if not isinstance(info, dict):
        _drop_local_training_session(model_id)
        return None
    incoming_version = max(1, int(info.get("metadata_version") or 1))
    incoming_step = max(0, int(info.get("current_step") or 0))
    try:
        current_version = max(1, int(getattr(snapshot, "metadata_version", 1) or 1))
    except Exception:
        current_version = 1
    try:
        current_step = max(0, int(getattr(snapshot, "current_step", 0) or 0))
    except Exception:
        current_step = 0
    if incoming_version <= current_version and incoming_step <= current_step:
        return snapshot
    try:
        _restore_training_session_info_compat(info)
    except Exception:
        return snapshot
    refreshed = _get_training_snapshot(model_id)
    return refreshed or snapshot


def _has_training_worker_binding(model_id: str) -> bool:
    if training_engine is None:
        return False
    workers = getattr(training_engine, "_workers", {})
    return model_id in workers and workers.get(model_id) is not None


async def _async_get_training_store_info(model_id: str) -> dict[str, Any] | None:
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e
    return info if isinstance(info, dict) else None


async def _get_training_session_for_request(model_id: str):
    if training_manager is None:
        return None, None
    snapshot = _get_training_snapshot(model_id)
    if snapshot is None:
        session = await _restore_training_session(model_id)
        snapshot = _get_training_snapshot(model_id) if session is not None else None
        return session, snapshot
    snapshot = await _refresh_training_snapshot_if_needed(model_id, snapshot)
    session = training_manager.get_session(model_id)
    if session is not None and not _has_training_worker_binding(model_id):
        restored = await _restore_training_session(model_id)
        if restored is not None:
            session = restored
            snapshot = _get_training_snapshot(model_id) or snapshot
    return session, snapshot


async def _raise_if_local_model_id_exists(model_id: str) -> None:
    if training_engine is None or training_manager is None:
        return
    if training_manager.get_session(model_id) is not None:
        raise HTTPException(status_code=409, detail=f"Model_id conflict: local model already exists: {model_id!r}")
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e
    if isinstance(info, dict):
        raise HTTPException(status_code=409, detail=f"Model_id conflict: local model already exists: {model_id!r}")


def _generate_model_id(session_id: str, model_seq_id: int) -> str:
    """Generate unique model_id from session_id and model_seq_id."""
    return f"{session_id}_{model_seq_id}"


def _field(source: Any, key: str, default=None):
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


async def _get_training_route_session_info(model_id: str) -> dict[str, Any] | None:
    """Resolve route-time session metadata from detached store only."""
    return await _async_get_training_store_info(model_id)


async def _protect_training_session_enqueue_window(session_info: dict[str, Any]) -> None:
    """Write detached heartbeat at enqueue-time so stale cleanup cannot race queued work."""
    session_id = str(session_info.get("session_id") or "")
    if not session_id:
        return
    try:
        from ..backend.session_heartbeat_store import session_heartbeat_store

        await session_heartbeat_store.async_update(session_id=session_id, now=time.time())
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training heartbeat store unavailable") from e


async def _best_effort_delete_training_session(
    model_id: str,
    *,
    reason: str,
    allow_actor_shutdown: bool,
) -> bool:
    if training_engine is None or training_manager is None:
        return False

    try:
        failed_request_ids = await future_store.async_fail_training_requests_for_model(
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
        restore_result = _restore_training_session(model_id)
        if inspect.isawaitable(restore_result):
            session = await restore_result
        else:
            session = restore_result
        restored = session is not None

    shutdown_attempted = False
    if session is not None:
        if allow_actor_shutdown:
            try:
                shutdown_attempted = True
                await training_engine.shutdown_session(session)
            except Exception as e:
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
            worker = getattr(training_engine, "_workers", {}).get(model_id)
            delete_session = getattr(worker, "delete_session", None) if worker is not None else None
            if delete_session is not None:
                try:
                    import ray

                    await asyncio.to_thread(ray.get, delete_session.remote(model_id), timeout=30)
                except Exception as e:
                    logger.warning(
                        "[%s] best-effort stale training cleanup remote delete failed (%s): %s: %s",
                        model_id,
                        reason,
                        type(e).__name__,
                        e,
                    )
            getattr(training_engine, "_resource_pool_actor_names", {}).pop(model_id, None)
            getattr(training_engine, "_workers", {}).pop(model_id, None)
            session.is_active = False

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
        from ..backend.resource_pool import get_resource_pool

        get_resource_pool().clear_session(model_id)
    except Exception:
        pass

    return session is not None or shutdown_attempted


async def cleanup_stale_training_sessions_once(*, stale_after_s: float | None = None) -> list[str]:
    from ..backend.training_cleanup_executor import training_cleanup_executor

    return await training_cleanup_executor.async_cleanup_stale_sessions_once(stale_after_s=stale_after_s)


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
        model_owner=str(info.get("user_id") or "anonymous"),
        is_lora=bool(is_lora),
        corrupted=False,
        lora_rank=lora_rank,
        last_request_time=str(
            info.get("last_request_time") or info.get("created_at") or datetime.now().isoformat()
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


def _normalize_megatron_scheduler_domain_key(base_model: str) -> str:
    hf_cache_pattern = r"models--([^/]+)--([^/]+)/snapshots"
    match = re.search(hf_cache_pattern, base_model)
    if match:
        _org, model = match.groups()
        model_name = model.lower().replace("-", "_").replace(".", "_")
    else:
        model_name = base_model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    return f"megatron_{model_name}"

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
        from verl.trainer.config import RolloutCorrectionConfig as VerlRolloutCorrectionConfig
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


def _build_training_scheduler_extra(
    *,
    session: Any,
    model_id: str,
    training_op: str,
    seq_id: int | None = None,
) -> dict[str, Any]:
    enabled = str(os.environ.get("MINT_SCHEDULER_ENABLE", "1")).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    backend = str(_field(session, "backend", "") or "unknown")
    base_model = str(_field(session, "base_model", "") or "")
    openpi_train_step = training_op == "train_step" and backend in {"openpi_fast", "openpi_pi05"}
    if openpi_train_step:
        enabled = True
    if backend == "megatron" and base_model:
        domain_key = _normalize_megatron_scheduler_domain_key(base_model)
    else:
        domain_key = base_model if base_model else str(model_id)
    extra: dict[str, Any] = {
        "scheduler_enabled": bool(enabled),
        "scheduler_domain": f"{backend}:{domain_key}",
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

def _build_create_scheduler_extra(
    *,
    base_model: str,
    model_id: str,
    training_op: str,
) -> dict[str, Any]:
    from ..backend.model_registry import get_model_config

    if bool(get_model_config(base_model).is_moe):
        backend = "megatron"
        domain_key = _normalize_megatron_scheduler_domain_key(base_model)
    else:
        backend = "peft"
        domain_key = base_model
    return {
        "scheduler_enabled": str(os.environ.get("MINT_SCHEDULER_ENABLE", "1")).strip().lower()
        in ("1", "true", "yes", "y", "on"),
        "scheduler_domain": f"{backend}:{domain_key}",
        "scheduler_session_key": str(model_id),
        "execution_serial_key": f"training_session:{model_id}",
        "training_op": str(training_op),
    }


def _sync_route_wait_timeout_s() -> float:
    try:
        return max(1.0, float(str(os.environ.get("MINT_SYNC_ROUTE_WAIT_TIMEOUT_S", "3600")).strip()))
    except Exception:
        return 3600.0


def _sync_route_wait_poll_interval_s() -> float:
    try:
        return max(0.01, float(str(os.environ.get("MINT_SYNC_ROUTE_WAIT_POLL_INTERVAL_S", "0.2")).strip()))
    except Exception:
        return 0.2


async def _wait_internal_future_result(request_id: str) -> Any:
    deadline = time.perf_counter() + _sync_route_wait_timeout_s()
    poll_interval_s = _sync_route_wait_poll_interval_s()
    try:
        while True:
            status = await future_store.async_get_status(request_id)
            if status == FutureStatus.PENDING:
                if time.perf_counter() >= deadline:
                    raise TimeoutError(f"Timed out waiting for internal future request_id={request_id}")
                await asyncio.sleep(poll_interval_s)
                continue
            if status == FutureStatus.DONE:
                try:
                    from ..backend.capacity_manager import capacity_manager

                    await capacity_manager.async_release_all(request_id)
                except Exception:
                    pass
                return await future_store.async_get_result(request_id)
            if status == FutureStatus.FAILED:
                err = await future_store.async_get_error(request_id)
                try:
                    from ..backend.capacity_manager import capacity_manager

                    await capacity_manager.async_release_all(request_id)
                except Exception:
                    pass
                raise RuntimeError(str(err or f"internal queued op failed request_id={request_id}"))
            try:
                from ..backend.capacity_manager import capacity_manager

                await capacity_manager.async_release_all(request_id)
            except Exception:
                pass
            raise RuntimeError(f"internal future reached unexpected terminal state={status.value} request_id={request_id}")
    finally:
        try:
            await future_store.async_cleanup(request_id)
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
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

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
            _mark_training_inflight(model_id, +1)
            inflight_marked = True
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(request_id, meta={"op": op, "model_id": model_id})
        await api_work_queue.enqueue(
            request_id=request_id,
            op=op,
            request_json=request_json,
            user_id=user_id,
            webhook_url=None,
            extra=dict(extra),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue {op} request: {e}") from e
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
    route_start_s = time.perf_counter()
    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(base_model=request.base_model, http_request=http_request)
    request = request.model_copy(update={"base_model": base_model})

    _validate_rollout_correction_config_or_400(
        base_model=request.base_model,
        rollout_correction_config=request.rollout_correction_config,
    )
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
            status_code=403,
            detail=get_access_denied_error(request.base_model)
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
            raise HTTPException(status_code=503, detail=f"Upstream {upstream.alias!r} create_model failed")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        payload = resp.json()
        upstream_request_id = payload.get("request_id")
        if not isinstance(upstream_request_id, str) or not upstream_request_id:
            raise HTTPException(status_code=502, detail="Upstream create_model returned invalid request_id")

        await async_register_remote_training_model(
            model_id=model_id,
            upstream_alias=upstream.alias,
            base_model=request.base_model,
            owner_id=user_id,
        )
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream.alias, upstream_request_id=upstream_request_id)
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
    scheduler_extra = merge_queue_priority_extra(
        _build_create_scheduler_extra(
            base_model=request.base_model,
            model_id=model_id,
            training_op="create_model",
        ),
        request=http_request,
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
        await future_store.async_mark_queued(request_id, meta={"op": "training.create_model", "model_id": model_id})
        await _enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.create_model",
            model_id=model_id,
            base_model=request.base_model,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="training.create_model",
                request_json=request_json,
                user_id=user_id,
                webhook_url=webhook_url,
                extra=scheduler_extra,
            ),
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
                config={"lora_rank": request.lora_config.rank if request.lora_config else None},
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
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
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
                config={"lora_rank": request.lora_config.rank if request.lora_config else None},
            )
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
    inflight_marked = False
    session_created = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        # Check if model already exists (from failed previous attempt)
        existing = training_manager.get_session(model_id)
        if existing is not None:
            if bool(getattr(existing, "is_active", False)):
                raise RuntimeError(f"Model '{model_id}' already exists")
            logger.warning(f"[{model_id}] Cleaning up stale inactive session from previous attempt")
            await training_engine.shutdown_session(existing)
            training_manager.delete_session(model_id)

        # Create session metadata first
        session = training_manager.create_session(
            model_id=model_id,
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            rollout_correction_config=request.rollout_correction_config.model_dump(exclude_none=True)
            if request.rollout_correction_config
            else None,
            user_metadata=request.user_metadata,
            user_id=user_id,
        )
        session_created = True

        # Mark inflight immediately so the idle cleanup loop does not evict
        # the session during the potentially slow actor creation below.
        _mark_training_inflight(model_id, +1)
        inflight_marked = True

        # Create Ray actor - if this fails, session will be cleaned up in except block
        await run_async_with_otel_span(
            "training.create_model.execute",
            lambda: training_engine.create_training_session(session),
            component="routes.training",
            op="training.create_model",
            request_id=str(request_id),
            attributes={
                "model_id": str(model_id),
                "base_model": str(request.base_model),
                "backend": str(session.backend),
                "lora_enabled": bool(request.lora_config is not None),
                "lora_rank": int(request.lora_config.rank) if request.lora_config is not None else None,
            },
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
                "rollout_correction_config": request.rollout_correction_config.model_dump(exclude_none=True)
                if request.rollout_correction_config
                else None,
                "user_metadata": request.user_metadata or {},
                "learning_rate": session.learning_rate,
                "current_step": session.current_step,
                "backend": session.backend,
                "actor_name": actor_name,
                "namespace": RAY_NAMESPACE,
                "user_id": user_id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "metadata_version": getattr(session, "metadata_version", 1),
            })
            training_manager.mark_persisted(model_id)
        except Exception as e:
            logger.warning("[create_model] training session store write failed: %s", e)

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
            backend=session.backend,  # "megatron" or "peft"
        )
        await future_store.async_resolve(request_id, response.model_dump())

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
        logger.exception(
            "[create_model] failed request_id=%s model_id=%s base_model=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(model_id),
            str(request.base_model),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_actor",
        )
        # Clean up session if it was created
        if session_created and training_manager and training_manager.get_session(model_id):
            training_manager.delete_session(model_id)
        # If session tracking was updated in ResourcePool during a partially-failed
        # create_training_session, clear it to avoid pinning actors as non-idle.
        try:
            from ..backend.resource_pool import get_resource_pool

            get_resource_pool().clear_session(model_id)
        except Exception:
            pass
        await _fail_future(request_id, str(e))

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
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(model_id, -1)


# =============================================================================
# create_model_from_state - async (composes create_model + load_state)
# =============================================================================

def _resolve_state_path(state_uri: str, *, user_id: str | None, is_admin: bool = False) -> str:
    if not is_admin and not state_uri.startswith(("tinker://", "mint://", "ckpt_")):
        raise HTTPException(status_code=403, detail="Access denied")

    resolved = resolve_checkpoint_path(state_uri, user_id=user_id, is_admin=is_admin)
    if state_uri.startswith("ckpt_") and resolved == state_uri:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    try:
        ensure_checkpoint_path_allowed(resolved, user_id=user_id, is_admin=is_admin)
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
    route_start_s = time.perf_counter()
    from ..supported_models_gate import enforce_base_model_allowed

    base_model = await enforce_base_model_allowed(base_model=request.base_model, http_request=http_request)
    request = request.model_copy(update={"base_model": base_model})

    _validate_rollout_correction_config_or_400(
        base_model=request.base_model,
        rollout_correction_config=request.rollout_correction_config,
    )
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
            status_code=403,
            detail=get_access_denied_error(request.base_model)
        )

    model_id = _generate_model_id(request.session_id, request.model_seq_id)
    user_id = _get_user_id(http_request)

    # Fail fast: sampler checkpoints are not eligible for optimizer restore.
    if bool(request.load_optimizer):
        try:
            from ..checkpoints import validate_checkpoint_load_contract

            local_path = _resolve_state_path(request.state_path, user_id=user_id, is_admin=is_admin_request(http_request))
            if os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "metadata.json")):
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
        if request.state_path.startswith(("tinker://", "mint://", "ckpt_")):
            local_path = _resolve_state_path(request.state_path, user_id=user_id, is_admin=is_admin_request(http_request))
            if os.path.isdir(local_path):
                proxy_timeout_s = float(os.environ.get("MINT_GATEWAY_CHECKPOINT_PROXY_TIMEOUT_S", "600"))
                tmp_archive = build_gateway_proxy_archive_path()
                try:
                    await async_create_checkpoint_archive(local_path, tmp_archive, timeout_s=proxy_timeout_s)
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
                timeout_s=120.0,
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

        await async_register_remote_training_model(
            model_id=model_id,
            upstream_alias=upstream.alias,
            base_model=request.base_model,
            owner_id=user_id,
        )
        return UntypedAPIFuture(
            request_id=encode_request_id(upstream_alias=upstream.alias, upstream_request_id=upstream_request_id)
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

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    resolved_state_path = _resolve_state_path(
        request.state_path,
        user_id=user_id,
        is_admin=is_admin_request(http_request),
    )
    if request.state_path.startswith(("tinker://", "mint://", "ckpt_")) and not os.path.isdir(resolved_state_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.state_path}")
    request = request.model_copy(update={"state_path": resolved_state_path})

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    scheduler_extra = merge_queue_priority_extra(
        _build_create_scheduler_extra(
            base_model=request.base_model,
            model_id=model_id,
            training_op="create_model_from_state",
        ),
        request=http_request,
    )
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
            meta={"op": "training.create_model_from_state", "model_id": model_id},
        )
        await _enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.create_model_from_state",
            model_id=model_id,
            base_model=request.base_model,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="training.create_model_from_state",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=scheduler_extra,
            ),
        )
    except Exception as e:
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue create_model_from_state request: {e}"
        )

    return UntypedAPIFuture(request_id=request_id)


async def _do_create_model_from_state(
    request_id: str, request: CreateModelFromStateRequest, user_id: str | None
) -> None:
    """Background task to create model and load checkpoint."""
    model_id = _generate_model_id(request.session_id, request.model_seq_id)
    inflight_marked = False
    session_created = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")

        # Queue-time validation hands the background worker a concrete local path.
        load_path = request.state_path

        # Check if model already exists (from failed previous attempt)
        existing = training_manager.get_session(model_id)
        if existing is not None:
            if bool(getattr(existing, "is_active", False)):
                raise RuntimeError(f"Model '{model_id}' already exists")
            logger.warning(f"[{model_id}] Cleaning up stale inactive session from previous attempt")
            await training_engine.shutdown_session(existing)
            training_manager.delete_session(model_id)

        # Create session metadata
        session = training_manager.create_session(
            model_id=model_id,
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            rollout_correction_config=request.rollout_correction_config.model_dump(exclude_none=True)
            if request.rollout_correction_config
            else None,
            user_metadata=request.user_metadata,
            user_id=user_id,
        )
        session_created = True

        # Mark inflight immediately (same rationale as _do_create_model).
        _mark_training_inflight(model_id, +1)
        inflight_marked = True

        async def _create_and_restore_model():
            await training_engine.create_training_session(session)
            await training_engine.load_weights(
                session=session,
                load_path=load_path,
                load_optimizer=request.load_optimizer,
            )

        await run_async_with_otel_span(
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
                "lora_rank": int(request.lora_config.rank) if request.lora_config is not None else None,
            },
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
                "rollout_correction_config": request.rollout_correction_config.model_dump(exclude_none=True)
                if request.rollout_correction_config
                else None,
                "user_metadata": request.user_metadata or {},
                "learning_rate": session.learning_rate,
                "current_step": session.current_step,
                "backend": session.backend,
                "actor_name": actor_name,
                "namespace": RAY_NAMESPACE,
                "user_id": user_id,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "metadata_version": getattr(session, "metadata_version", 1),
            })
            training_manager.mark_persisted(model_id)
        except Exception as e:
            logger.warning("[create_model_from_state] training session store write failed: %s", e)

        try:
            from ..backend.session_index_store import add_training_run_to_session

            add_training_run_to_session(
                session_id=request.session_id,
                training_run_id=model_id,
                user_id=user_id,
                created_at=session.created_at,
            )
        except Exception as e:
            logger.warning("[create_model_from_state] session index write failed: %s", e)

        logger.info(
            f"[{model_id}] Created model from state: {request.state_path} "
            f"(step={session.current_step})"
        )

        response = CreateModelFromStateResponse(
            request_id=request_id,
            model_id=model_id,
            type="create_model_from_state",
        )
        await future_store.async_resolve(request_id, response.model_dump())

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
        # Clean up session if it was created
        if session_created and training_manager and training_manager.get_session(model_id):
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
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(model_id, -1)


# =============================================================================
# forward_backward - async
# =============================================================================


@router.post("/forward_backward", response_model=UntypedAPIFuture)
async def forward_backward(
    request: ForwardBackwardRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform forward + backward pass on training data."""
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    route_session_info = await _get_training_route_session_info(request.model_id)

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
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

    if not isinstance(route_session_info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    await _protect_training_session_enqueue_window(route_session_info)
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

    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_forward_backward_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    gateway_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)

    # Set request_id in context for logging
    set_request_id(request_id)
    logger.info(f"forward_backward request received: model_id={request.model_id}")

    reserve = await capacity_manager.async_try_reserve(
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
    inflight_marked = False
    try:
        _mark_training_inflight(request.model_id, +1)
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
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta={"op": "training.forward_backward", "model_id": request.model_id},
        )
        await _enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.forward_backward",
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="training.forward_backward",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=scheduler_extra,
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue forward_backward request: {e}")

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
    inflight_marked = False

    try:
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
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
        logger.info(
            f"[{session.model_id}] forward_backward start: "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len} "
            f"loss_fn={request.forward_backward_input.loss_fn}"
        )
        result = await run_async_with_otel_span(
            "training.forward_backward.execute",
            lambda: training_engine.forward_backward(session, request),
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
        if gateway_auth:
            auth_ctx = GatewayAuthContext(**gateway_auth)
            await _persist_usage_events(
                events=[
                    UsageEvent(
                        account_id=auth_ctx.account_id,
                        apikey_id=auth_ctx.apikey_id,
                        charge_item="training",
                        quantity=token_count,
                        request_id=auth_ctx.request_id,
                        label=_build_training_usage_label(
                            model=session.base_model,
                            route="training.forward_backward",
                        ),
                    )
                ]
            )
        await future_store.async_resolve(request_id, result)

    except Exception as e:
        logger.exception(
            "[forward_backward] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_batch_shape",
        )
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


# =============================================================================
# train_step - async (forward_backward + optim_step)
# =============================================================================


@router.post("/train_step", response_model=UntypedAPIFuture)
async def train_step(
    request: TrainStepRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform a combined forward_backward + optim_step."""
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    route_session_info = await _get_training_route_session_info(request.model_id)

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
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

    if not isinstance(route_session_info, dict):
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    await _protect_training_session_enqueue_window(route_session_info)
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

    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    gateway_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)
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
        _mark_training_inflight(request.model_id, +1)
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
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(request_id, meta={"op": "training.train_step", "model_id": request.model_id})
        await _enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.train_step",
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="training.train_step",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=scheduler_extra,
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue train_step request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_train_step(
    request_id: str,
    request: TrainStepRequest,
    user_id: str | None,
    gateway_auth: dict | None = None,
) -> None:
    """Background task for train_step."""
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
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
            lambda: training_engine.train_step(session, request),
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
        if gateway_auth:
            auth_ctx = GatewayAuthContext(**gateway_auth)
            await _persist_usage_events(
                events=[
                    UsageEvent(
                        account_id=auth_ctx.account_id,
                        apikey_id=auth_ctx.apikey_id,
                        charge_item="training",
                        quantity=token_count,
                        request_id=auth_ctx.request_id,
                        label=_build_training_usage_label(
                            model=session.base_model,
                            route="training.train_step",
                        ),
                    )
                ]
            )
        await future_store.async_resolve(request_id, result)

    except Exception as e:
        logger.exception(
            "[train_step] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_actor",
        )
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


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

    route_session_info = await _get_training_route_session_info(request.model_id)

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
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

    if not isinstance(route_session_info, dict):
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    await _protect_training_session_enqueue_window(route_session_info)
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

    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    gateway_auth = build_billing_auth_context(http_request, fallback_request_id=request_id)
    user_id = _get_user_id(http_request)
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
        _mark_training_inflight(request.model_id, +1)
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
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(request_id, meta={"op": "training.forward", "model_id": request.model_id})
        await _enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.forward",
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="training.forward",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=scheduler_extra,
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue forward request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_forward(
    request_id: str,
    request: ForwardRequest,
    gateway_auth: dict | None = None,
) -> None:
    """Background task for forward."""
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        batch = request.forward_input.data
        token_count, max_seq_len = _compute_token_stats(batch)
        t0 = time.time()
        logger.info(
            f"[{session.model_id}] forward start: "
            f"backend={session.backend} batch={len(batch)} tokens={token_count} max_len={max_seq_len}"
        )
        result = await run_async_with_otel_span(
            "training.forward.execute",
            lambda: training_engine.forward(session, request),
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
        if gateway_auth:
            auth_ctx = GatewayAuthContext(**gateway_auth)
            await _persist_usage_events(
                events=[
                    UsageEvent(
                        account_id=auth_ctx.account_id,
                        apikey_id=auth_ctx.apikey_id,
                        charge_item="training",
                        quantity=token_count,
                        request_id=auth_ctx.request_id,
                        label=_build_training_usage_label(
                            model=session.base_model,
                            route="training.forward",
                        ),
                    )
                ]
            )
        await future_store.async_resolve(request_id, result)

    except Exception as e:
        logger.exception(
            "[forward] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_input_tokens",
        )
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


# =============================================================================
# optim_step - async
# =============================================================================


@router.post("/optim_step", response_model=UntypedAPIFuture)
async def optim_step(
    request: OptimStepRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Perform optimizer step to update weights."""
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        upstream_for_alias,
    )

    restore_start_s = time.perf_counter()
    route_session_info = await _get_training_route_session_info(request.model_id)
    logger.info(
        "[optim_step route] model_id=%s stage=resolve_session_info elapsed_ms=%.3f resolved=%s",
        str(request.model_id),
        (time.perf_counter() - restore_start_s) * 1000.0,
        bool(isinstance(route_session_info, dict)),
    )

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
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

    if not isinstance(route_session_info, dict):
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )
    await _protect_training_session_enqueue_window(route_session_info)
    base_model = str(route_session_info.get("base_model") or "")
    backend = str(route_session_info.get("backend") or "unknown")

    user_id = _get_user_id(http_request)
    from ..backend.api_work_queue import api_work_queue
    from ..backend.capacity_manager import capacity_manager
    from ..backend.result_size_estimator import estimate_small_result_bytes

    request_json = request.model_dump_json().encode("utf-8")
    request_id = uuid.uuid4().hex
    reserve_start_s = time.perf_counter()
    reserve = await capacity_manager.async_try_reserve(
        request_id,
        queue_bytes=len(request_json),
        object_store_bytes=estimate_small_result_bytes(),
    )
    logger.info(
        "[optim_step route] request_id=%s model_id=%s stage=capacity_reserve elapsed_ms=%.3f ok=%s",
        str(request_id),
        str(request.model_id),
        (time.perf_counter() - reserve_start_s) * 1000.0,
        bool(reserve.get("ok")),
    )
    if not bool(reserve.get("ok")):
        raise HTTPException(
            status_code=429,
            detail={"code": "tinker_overloaded", **{k: v for k, v in reserve.items() if k != "ok"}},
        )

    created = False
    inflight_marked = False
    try:
        _mark_training_inflight(request.model_id, +1)
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
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(request_id, meta={"op": "training.optim_step", "model_id": request.model_id})
        await _enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.optim_step",
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="training.optim_step",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=scheduler_extra,
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(status_code=503, detail=f"Failed to enqueue optim_step request: {e}")

    return UntypedAPIFuture(request_id=request_id)


async def _do_optim_step(request_id: str, request: OptimStepRequest, user_id: str | None) -> None:
    """Background task for optim_step."""
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        lr = request.adam_params.learning_rate if request.adam_params else None
        t0 = time.time()
        msg = f"[{session.model_id}] optim_step start request_id={request_id} lr={lr}"
        logger.info(msg)
        result = await run_async_with_otel_span(
            "training.optim_step.execute",
            lambda: training_engine.optim_step(session, request),
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
        await future_store.async_resolve(request_id, result)

    except Exception as e:
        logger.exception(
            "[optim_step] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_training_session_and_optimizer_state",
        )
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


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
    from ..gateway import async_remote_training_model, forward_json, upstream_for_alias

    route_session_info = await _get_training_route_session_info(request.model_id)

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
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
                    timeout_s=120.0,
                )
            except Exception:
                logger.exception("Upstream reset_expert_bias failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} reset_expert_bias failed")
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return ResetExpertBiasResponse.model_validate(resp.json())

    if not isinstance(route_session_info, dict):
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    await _protect_training_session_enqueue_window(route_session_info)
    try:
        request_id = await _enqueue_internal_serialized_model_op(
            model_id=request.model_id,
            op="training.reset_expert_bias",
            request_json=request.model_dump_json().encode("utf-8"),
            extra=merge_queue_priority_extra(
                _build_training_scheduler_extra(
                    session=route_session_info,
                    model_id=request.model_id,
                    training_op="reset_expert_bias",
                ),
                request=http_request,
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
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")

        result = await training_engine.reset_expert_bias(session)
        modules_reset = int(result.get("modules_reset", 0) or 0)
        await future_store.async_resolve(
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
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


# =============================================================================
# save_weights_for_sampler - async
# =============================================================================


@router.post("/save_weights_for_sampler", response_model=UntypedAPIFuture)
async def save_weights_for_sampler(
    request: SaveWeightsForSamplerRequest,
    http_request: Request,
) -> UntypedAPIFuture:
    """Save model weights for inference use."""
    route_start_s = time.perf_counter()
    from ..gateway import (
        async_remote_training_model,
        encode_request_id,
        forward_json,
        register_pending_save_weights_for_sampler_future,
        upstream_for_alias,
    )

    route_session_info = await _get_training_route_session_info(request.model_id)

    if not isinstance(route_session_info, dict):
        remote = await async_remote_training_model(request.model_id)
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
                    timeout_s=300.0,
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

    if not isinstance(route_session_info, dict):
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )
    await _protect_training_session_enqueue_window(route_session_info)
    base_model = str(route_session_info.get("base_model") or "")
    backend = str(route_session_info.get("backend") or "unknown")

    user_id = _get_user_id(http_request)
    from ..client_compat import prefer_tinker_uri

    prefer_tinker = prefer_tinker_uri(http_request)
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
        _mark_training_inflight(request.model_id, +1)
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
        scheduler_extra["is_admin"] = is_admin_request(http_request)
        await future_store.async_create_with_id(request_id)
        created = True
        await future_store.async_mark_queued(
            request_id,
            meta={"op": "training.save_weights_for_sampler", "model_id": request.model_id},
        )
        await _enqueue_training_request_with_trace(
            route_start_s=route_start_s,
            request_id=request_id,
            op="training.save_weights_for_sampler",
            model_id=request.model_id,
            base_model=base_model,
            backend=backend,
            enqueue_coro=api_work_queue.enqueue(
                request_id=request_id,
                op="training.save_weights_for_sampler",
                request_json=request_json,
                user_id=user_id,
                webhook_url=None,
                extra=scheduler_extra,
            ),
        )
    except Exception as e:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)
        await capacity_manager.async_release_all(request_id)
        if created:
            await future_store.async_cleanup(request_id)
        raise HTTPException(
            status_code=503, detail=f"Failed to enqueue save_weights_for_sampler request: {e}"
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
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(request.model_id)
        if session is None:
            session = await _restore_training_session(request.model_id)
        if session is None:
            raise RuntimeError(f"Model '{request.model_id}' not found")
        # Determine checkpoint name
        if request.path is not None:
            # Named save - use provided path
            checkpoint_name = request.path
            save_path = build_persistent_cache_dir(
                user_id=None if is_admin else user_id,
                model_id=session.model_id,
                checkpoint_name=checkpoint_name,
            )
        else:
            # Ephemeral save - generate unique temp name
            checkpoint_name = f"_ephemeral_{uuid.uuid4().hex[:8]}"
            save_path = build_ephemeral_checkpoint_dir(
                user_id=None if is_admin else user_id,
                model_id=session.model_id,
                checkpoint_name=checkpoint_name,
            )

        train_mlp = bool(getattr(getattr(session, "lora_config", None), "train_mlp", False))

        # Save weights
        save_path = await run_async_with_otel_span(
            "training.save_weights_for_sampler.execute",
            lambda: training_engine.save_weights_for_sampler(
                session=session,
                checkpoint_name=checkpoint_name,
                checkpoint_base_dir=os.path.dirname(os.path.dirname(save_path)),
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

        ttl_seconds = request.ttl_seconds
        if request.path is None and ttl_seconds is None:
            ttl_seconds = None
        write_checkpoint_metadata(
            save_path,
            {
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
                "storage_tier": "ephemeral_pfs" if request.path is None else "persistent_cache",
                "ttl_seconds": request.ttl_seconds,
            },
        )

        persistent_path = None
        if request.path is not None:
            persistent_path = begin_async_checkpoint_mirror(
                save_path,
                user_id=None if is_admin else user_id,
                model_id=session.model_id,
                checkpoint_name=checkpoint_name,
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

        if request.path is not None:
            # Named flow: Return path, caller creates session separately
            response = SaveWeightsForSamplerResponse(
                path=path_uri,
                sampling_session_id=None,
            ).model_dump()
            response.update(
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
            if inference_manager is None:
                raise RuntimeError("Inference manager not initialized")

            import time

            sampling_session_id = str(uuid.uuid4())
            lora_rank = session.lora_config.rank if session.lora_config else 32
            base_model = session.base_model

            # Do not block save-time on vLLM actor cold-start. Kick off engine
            # warm in the background, but still fail fast if the warm task trips
            # an immediate configuration or capacity error before we return a
            # sampling_session_id. Allow one short grace window so async failures
            # that happen right after the first await still surface on save.
            async def _warm_engine() -> None:
                await inference_manager.get_engine_for_model(base_model)

            pending_warms = getattr(inference_manager, "_background_engine_warm_tasks", None)
            if not isinstance(pending_warms, dict):
                pending_warms = {}
                setattr(inference_manager, "_background_engine_warm_tasks", pending_warms)

            existing_warm = pending_warms.get(base_model)
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
            if warm_task in done:
                await warm_task

            # Multi-LoRA mode: register the sampling session immediately and let
            # /asample load the adapter lazily on first use. This matches
            # create_sampling_session() semantics and avoids blocking save-time
            # on engine cold-start or vLLM's exclusive-engine gate while another
            # generate is active.
            inference_manager.register_multi_lora_session(
                session_id=sampling_session_id,
                base_model=base_model,
                lora_rank=lora_rank,
                adapter_path=save_path,
                lora_loaded=False,
            )

            logger.info(
                f"[save_weights_for_sampler] Multi-LoRA: registered lazy-load session "
                f"{sampling_session_id} (model={base_model}, path={save_path})"
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
            except Exception as e:
                logger.warning("[save_weights_for_sampler] session index write failed: %s", e)

            response = SaveWeightsForSamplerResponse(
                path=None,  # Ephemeral - no path returned
                sampling_session_id=sampling_session_id,
            ).model_dump()

        await future_store.async_resolve(request_id, response)

    except Exception as e:
        logger.exception(
            "[save_weights_for_sampler] failed request_id=%s model_id=%s failure_reason=%s error_type=%s next_action=%s",
            str(request_id),
            str(request.model_id),
            classify_failure_reason(e),
            type(e).__name__,
            "check_checkpoint_export_and_inference_registration",
        )
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(request.model_id, -1)


# =============================================================================
# Model info endpoints
# =============================================================================


def _owner_visible(request_user_data: dict | None, owner: str | None) -> bool:
    request_user_id = str(request_user_data.get("user_id")) if request_user_data and request_user_data.get("user_id") else None
    if request_user_id is None:
        return True
    if is_admin_user_data(request_user_data):
        return True
    return bool(owner) and owner == request_user_id


@router.get("/training_runs/{training_run_id}", response_model=TrainingRun)
async def get_training_run(training_run_id: str, http_request: Request) -> TrainingRun:
    request_user_data = _get_user_data(http_request)
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(training_run_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e

    if not isinstance(info, dict):
        raise HTTPException(status_code=404, detail=f"Training run '{training_run_id}' not found")

    if not _owner_visible(request_user_data, info.get("user_id")):
        raise HTTPException(status_code=404, detail=f"Training run '{training_run_id}' not found")

    if "model_id" not in info:
        info = dict(info)
        info["model_id"] = training_run_id

    return _training_run_from_info(info)


@router.get("/training_runs", response_model=TrainingRunsResponse)
async def list_training_runs(limit: int = 20, offset: int = 0, http_request: Request = None) -> TrainingRunsResponse:
    request_user_data = _get_user_data(http_request) if http_request else None
    infos_by_id: dict[str, dict] = {}

    try:
        from ..backend.training_session_store import async_list_training_sessions

        for info in await async_list_training_sessions():
            model_id = info.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                continue
            infos_by_id[model_id] = info
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e

    if training_manager is not None:
        for session in training_manager.list_sessions():
            model_id = str(getattr(session, "model_id", "") or "")
            if model_id and model_id not in infos_by_id:
                _drop_local_training_session(model_id)

    infos = [
        info for info in infos_by_id.values() if _owner_visible(request_user_data, info.get("user_id"))
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
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e
    if not isinstance(info, dict):
        _drop_local_training_session(model_id)
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return {
        "model_id": str(info.get("model_id") or model_id),
        "session_id": info.get("session_id"),
        "model_seq_id": info.get("model_seq_id"),
        "base_model": info.get("base_model"),
        "lora_config": info.get("lora_config"),
        "user_metadata": info.get("user_metadata") or {},
        "learning_rate": info.get("learning_rate"),
        "created_at": info.get("created_at"),
        "current_step": info.get("current_step", 0),
        "is_active": info.get("is_active", False),
        "backend": info.get("backend"),
        "user_id": info.get("user_id"),
    }


@router.post("/get_info", response_model=GetInfoResponse)
async def get_info(request: GetInfoRequest, http_request: Request) -> GetInfoResponse:
    """Get model info (tinker client compatible endpoint).

    Returns model architecture, tokenizer, and LoRA configuration.
    """
    from ..gateway import async_remote_training_model, forward_json, upstream_for_alias

    try:
        from ..backend.training_session_store import async_get_training_session_info

        store_info = await async_get_training_session_info(request.model_id)
    except Exception:
        store_info = None

    if not isinstance(store_info, dict):
        remote = await async_remote_training_model(request.model_id)
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
                    timeout_s=120.0,
                )
            except Exception:
                logger.exception("Upstream get_info failed: %s", upstream_alias)
                raise HTTPException(status_code=503, detail=f"Upstream {upstream_alias!r} get_info failed")
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return GetInfoResponse.model_validate(resp.json())

    if not isinstance(store_info, dict):
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model_id}' not found"
        )

    # Build response matching tinker client expectations
    lora_cfg = store_info.get("lora_config") if isinstance(store_info, dict) else None
    lora_rank = lora_cfg.get("rank") if isinstance(lora_cfg, dict) else None
    is_lora = isinstance(lora_cfg, dict)

    base_model = str(store_info.get("base_model") or "")
    return GetInfoResponse(
        model_id=str(store_info.get("model_id") or request.model_id),
        model_data=ModelData(
            arch="transformer",  # Generic architecture identifier
            model_name=base_model,
            tokenizer_id=base_model,  # Use base model as tokenizer ID
        ),
        model_name=base_model,
        is_lora=is_lora,
        lora_rank=lora_rank,
    )


@router.get("/models")
async def list_models():
    """List all training models."""
    try:
        from ..backend.training_session_store import async_list_training_sessions

        store_infos = await async_list_training_sessions()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e

    models = []
    for info in store_infos:
        model_id = str(info.get("model_id") or "")
        if not model_id:
            continue
        models.append(
            {
                "model_id": info.get("model_id"),
                "session_id": info.get("session_id"),
                "model_seq_id": info.get("model_seq_id"),
                "base_model": info.get("base_model"),
                "created_at": info.get("created_at"),
                "current_step": info.get("current_step", 0),
                "is_active": info.get("is_active", False),
            }
        )

    return {"models": models, "total": len(models)}


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a training model and release resources."""
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception:
        info = None

    if not isinstance(info, dict) and training_manager is not None:
        get_session = getattr(training_manager, "get_session", None)
        if callable(get_session):
            session = get_session(model_id)
            if session is not None:
                info = {
                    "model_id": getattr(session, "model_id", model_id),
                    "base_model": getattr(session, "base_model", None),
                    "backend": getattr(session, "backend", None),
                    "user_id": getattr(session, "user_id", None),
                }

    if not isinstance(info, dict):
        raise HTTPException(
            status_code=404, detail=f"Model '{model_id}' not found"
        )

    request_id = await _enqueue_internal_serialized_model_op(
        model_id=model_id,
        op="training.delete_model",
        request_json=json.dumps({"model_id": model_id}).encode("utf-8"),
        extra=_build_training_scheduler_extra(
            session=info,
            model_id=model_id,
            training_op="delete_model",
        ),
        user_id=info.get("user_id"),
    )
    try:
        return await _wait_internal_future_result(request_id)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[training.delete_model] Failed model_id=%s error=%s", str(model_id), e)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _do_delete_model(request_id: str, model_id: str) -> None:
    inflight_marked = False
    try:
        set_request_id(request_id)
        if training_engine is None or training_manager is None:
            raise RuntimeError("Training engine not initialized")
        inflight_marked = True

        session = training_manager.get_session(model_id)
        if session is not None:
            await training_engine.shutdown_session(session)
            training_manager.delete_session(model_id)

        try:
            from ..backend.training_session_store import delete_training_session

            delete_training_session(model_id)
        except Exception:
            pass
        try:
            from ..backend.resource_pool import get_resource_pool

            get_resource_pool().clear_session(model_id)
        except Exception:
            pass

        await future_store.async_resolve(request_id, {"model_id": model_id, "status": "deleted"})
    except Exception as e:
        logger.exception(
            "[training.delete_model] failed request_id=%s model_id=%s error_type=%s error=%s",
            str(request_id),
            str(model_id),
            type(e).__name__,
            e,
        )
        await _fail_future(request_id, str(e))
    finally:
        if inflight_marked and training_manager is not None:
            _mark_training_inflight(model_id, -1)


@router.get("/models/{model_id}/tokenizer")
async def get_tokenizer(model_id: str):
    """Get tokenizer configuration for a training model.

    Returns tokenizer info (vocab_size, special tokens, etc.)
    for client-side tokenization.
    """
    try:
        from ..backend.training_session_store import async_get_training_session_info

        info = await async_get_training_session_info(model_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Training session store unavailable") from e
    if not isinstance(info, dict):
        _drop_local_training_session(model_id)
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    try:
        from ..backend.queue_execution_runtime import queue_execution_runtime

        tokenizer_info = await queue_execution_runtime.async_get_tokenizer_info(model_id=model_id)
    except Exception as e:
        if training_engine is None or training_manager is None:
            raise HTTPException(status_code=503, detail=f"Training runtime unavailable: {type(e).__name__}: {e}") from e
        _refresh_training_session_from_info_if_needed(model_id, info)
        session, _snapshot = await _get_training_session_for_request(model_id)
        if session is None:
            session = _restore_training_session_info_compat(info)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        tokenizer_info = await training_engine.get_tokenizer_info(session)
    return {
        "model_id": model_id,
        "tokenizer": tokenizer_info,
    }
