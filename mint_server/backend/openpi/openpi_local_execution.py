"""In-process, Ray-free execution path for OpenPI (pi0.5) — step 2 of Separate.

This module is the self-contained replacement for the Ray request path
(enqueue_model_work -> model_work_scheduler actor -> ModelEngineHost actor ->
engine) for OpenPI backends only. It owns:

  * a process-local training engine (OpenPIPi05TrainingEngine + local runtime
    factory -> in-process OpenPIDirectWorkerClient, NO Ray actor),
  * a process-local training-session registry (plain dict; no detached actor,
    no task_state_store),
  * a process-local action-session manager (local action runtime factory),
  * a process-local future store (dict) that mirrors the Tinker future contract
    used by /retrieve_future.

Why a private future store: the shared `task_futures` sits on
`TaskStateStoreClient`, which is a detached Ray actor proxy. Reusing it would
keep OpenPI on Ray. OpenPI ops here run *inline* (synchronously) inside the API
request handler, so the result is ready the instant we store it — a plain dict
is sufficient and Ray-free.

Routes call the handle_* coroutines when the model/session is an OpenPI backend;
everything else keeps the existing Ray-scheduled path untouched.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# OpenPI training backends this local path serves.
OPENPI_LOCAL_BACKENDS = {"openpi_pi05"}


def is_openpi_local_base_model(base_model: str | None) -> bool:
    if not base_model:
        return False
    try:
        from mint_server.backend.core.model_registry import get_model_config

        return str(get_model_config(base_model).training_backend) in OPENPI_LOCAL_BACKENDS
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Process-local singletons (lazy)                                             #
# --------------------------------------------------------------------------- #
_engine: Any = None
_train_manager: Any = None
_action_manager: Any = None
_engine_lock = asyncio.Lock()
# Serialize all stateful engine ops (single-tenant A: one training loop at a time).
_exec_lock = asyncio.Lock()

# Process-local sessions and futures.
_sessions: dict[str, Any] = {}
_futures: dict[str, dict[str, Any]] = {}


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        from mint_server.backend.openpi.openpi_pi05_local_runtime import (
            make_local_pi05_runtime,
        )
        from mint_server.backend.openpi.openpi_pi05_training import (
            OpenPIPi05TrainingEngine,
        )

        _engine = OpenPIPi05TrainingEngine(runtime_factory=make_local_pi05_runtime)
    return _engine


def _get_train_manager() -> Any:
    global _train_manager
    if _train_manager is None:
        from mint_server.backend.training.training_session_manager import (
            TrainingSessionManager,
        )

        _train_manager = TrainingSessionManager()
    return _train_manager


def _get_action_manager() -> Any:
    global _action_manager
    if _action_manager is None:
        from mint_server.backend.openpi.action_session_manager import (
            OpenPIPi05ActionSessionManager,
        )
        from mint_server.backend.openpi.openpi_pi05_local_runtime import (
            make_local_pi05_action_runtime,
        )

        _action_manager = OpenPIPi05ActionSessionManager(
            runtime_factory=make_local_pi05_action_runtime,
        )
    return _action_manager


def get_local_training_session(model_id: str) -> Any | None:
    return _sessions.get(model_id)


def has_local_training_session(model_id: str) -> bool:
    return model_id in _sessions


# --------------------------------------------------------------------------- #
# In-process future store (mirrors the Tinker future contract, Ray-free)      #
# --------------------------------------------------------------------------- #
def has_future(request_id: str) -> bool:
    return request_id in _futures


def _put_done(request_id: str, result: dict[str, Any]) -> None:
    _futures[request_id] = {"status": "done", "result": result, "ts": time.time()}
    _evict_futures_if_needed()


def _put_failed(request_id: str, error: str) -> None:
    _futures[request_id] = {"status": "failed", "error": error, "ts": time.time()}
    _evict_futures_if_needed()


_FUTURES_MAX = 4096


def get_future(request_id: str) -> dict[str, Any] | None:
    """Return the terminal future record (non-destructive; safe for retries).

    OpenPI ops run inline, so a future is always terminal (done/failed) by the
    time the client polls. Returns None if this request_id is not ours.
    """
    return _futures.get(request_id)


def _evict_futures_if_needed() -> None:
    while len(_futures) > _FUTURES_MAX:
        # dicts preserve insertion order; drop oldest.
        oldest = next(iter(_futures))
        _futures.pop(oldest, None)


async def _run_inline(request_id: str, coro_factory) -> str:
    """Execute an OpenPI op inline (serialized), store terminal result, return id."""
    async with _exec_lock:
        try:
            result = await coro_factory()
            _put_done(request_id, result)
        except Exception as e:
            logger.exception("[openpi-local] request_id=%s failed", request_id)
            _put_failed(request_id, f"{type(e).__name__}: {e}")
    return request_id


# --------------------------------------------------------------------------- #
# Training handlers                                                            #
# --------------------------------------------------------------------------- #
async def handle_create_model(request: Any, user_id: str | None) -> str:
    """Create an OpenPI training session in the process-local registry."""
    from mint_server.backend.training.training_session_manager import (
        MATERIALIZATION_STATE_UNMATERIALIZED,
        TRAINING_SESSION_METADATA_VERSION,
    )
    from mint_server.models.types import CreateModelResponse

    request_id = uuid.uuid4().hex
    model_id = f"{request.session_id}_{request.model_seq_id}"

    async def _do() -> dict[str, Any]:
        if model_id in _sessions:
            raise RuntimeError(f"Model '{model_id}' already exists")
        manager = _get_train_manager()
        session = manager.create_session(
            model_id=model_id,
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            rollout_correction_config=None,
            user_metadata=request.user_metadata,
            user_id=user_id,
            backend="openpi_pi05",
            metadata_version=TRAINING_SESSION_METADATA_VERSION,
            materialization_state=MATERIALIZATION_STATE_UNMATERIALIZED,
        )
        _sessions[model_id] = session
        return CreateModelResponse(
            request_id=request_id,
            model_id=model_id,
            type="create_model",
            backend="openpi_pi05",
        ).model_dump()

    await _run_inline(request_id, _do)
    return request_id


async def _ensure_runtime(session: Any) -> None:
    """Create the in-process worker runtime for this session if not present."""
    engine = _get_engine()
    if session.model_id in getattr(engine, "_runtime_clients", {}):
        return
    async with _engine_lock:
        if session.model_id in getattr(engine, "_runtime_clients", {}):
            return
        await engine.create_training_session(session)


async def handle_train_step(request: Any) -> str:
    """forward_backward + optim_step for an OpenPI session (VLA train_step)."""
    from mint_server.models.types import AdamParams

    request_id = uuid.uuid4().hex
    model_id = request.model_id

    async def _do() -> dict[str, Any]:
        session = _sessions.get(model_id)
        if session is None:
            raise RuntimeError(f"Model '{model_id}' not found")
        await _ensure_runtime(session)
        engine = _get_engine()

        fb_out = await engine.forward_backward(session, request)
        optim_req = type(
            "_OptimReq", (), {"adam_params": getattr(request, "adam_params", None) or AdamParams()}
        )()
        optim_out = await engine.optim_step(session, optim_req)
        metrics = dict(fb_out.get("metrics") or {})
        metrics.update(dict(optim_out.get("metrics") or {}))
        result = dict(fb_out)
        result["metrics"] = metrics
        result.setdefault("type", "train_step")
        return result

    await _run_inline(request_id, _do)
    return request_id


async def handle_save_weights_for_sampler(request: Any, user_id: str | None) -> str:
    """Export sampler weights to persistent cache; return a mint:// URI.

    Mirrors the openpi-named branch of routes.training._do_save_weights_for_sampler
    but Ray-free: no checkpoint-index claim (openpi skips it), no async mirror.
    """
    import os

    from mint_server.checkpoints import (
        build_persistent_cache_dir,
        validate_sampler_checkpoint_for_sampling,
        write_checkpoint_metadata,
    )
    from mint_server.models.types import SaveWeightsForSamplerResponse
    from mint_server.utils.client_compat import checkpoint_uri

    request_id = uuid.uuid4().hex
    model_id = request.model_id

    async def _do() -> dict[str, Any]:
        from datetime import datetime, timezone

        session = _sessions.get(model_id)
        if session is None:
            raise RuntimeError(f"Model '{model_id}' not found")
        await _ensure_runtime(session)
        engine = _get_engine()

        if request.path is None:
            raise ValueError(
                "OpenPI local save_weights_for_sampler requires an explicit path "
                "(named checkpoint); ephemeral sampler flow is not supported off-Ray"
            )
        checkpoint_name = request.path.strip()
        if (
            not checkpoint_name
            or checkpoint_name in (".", "..")
            or "/" in checkpoint_name
            or "\\" in checkpoint_name
        ):
            raise ValueError(f"Invalid checkpoint name: {request.path!r}")

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        save_path = build_persistent_cache_dir(
            user_id=user_id,
            model_id=session.model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type="sampler",
        )
        # engine expects the base dir three levels up from the leaf sampler dir.
        checkpoint_base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(save_path))
        )
        exported = await engine.save_weights_for_sampler(
            session=session,
            checkpoint_name=checkpoint_name,
            checkpoint_base_dir=checkpoint_base_dir,
            checkpoint_type="sampler",
        )
        validate_sampler_checkpoint_for_sampling(exported)
        write_checkpoint_metadata(
            exported,
            {
                "checkpoint_id": checkpoint_name,
                "owner_id": user_id,
                "model_id": session.model_id,
                "model_name": session.base_model,
                "created_at": created_at,
                "step": session.current_step,
                "checkpoint_type": "sampler",
                "optimizer_present": False,
                "backend": session.backend,
                "type": "sampler",
                "storage_tier": "persistent_cache",
            },
        )
        mint_uri = checkpoint_uri(
            session.model_id, checkpoint_name, prefer_tinker=False, checkpoint_type="sampler"
        )
        response = SaveWeightsForSamplerResponse(
            path=mint_uri,
            sampling_session_id=None,
            owner_id=user_id or "anonymous",
        ).model_dump()
        response.update(filesystem_path=exported, storage_tier="persistent_cache")
        return response

    await _run_inline(request_id, _do)
    return request_id


def delete_local_training_session(model_id: str) -> bool:
    """Drop a process-local openpi training session + its runtime. Best-effort."""
    session = _sessions.pop(model_id, None)
    if session is None:
        return False
    engine = _get_engine()
    clients = getattr(engine, "_runtime_clients", {})
    if model_id in clients:
        # Fire-and-forget shutdown; the worker is in-process.
        async def _shutdown() -> None:
            try:
                await engine.shutdown_session(session)
            except Exception:
                logger.warning("[openpi-local] shutdown_session failed for %s", model_id)

        try:
            asyncio.get_running_loop().create_task(_shutdown())
        except RuntimeError:
            pass
    try:
        _get_train_manager().delete_session(model_id)
    except Exception:
        pass
    return True


# --------------------------------------------------------------------------- #
# Action / inference handlers                                                  #
# --------------------------------------------------------------------------- #
async def handle_create_action_session(
    *,
    session_id: str,
    action_session_seq_id: int | None,
    base_model: str,
    model_path: str | None,
    user_id: str | None,
) -> str:
    """Create an in-process openpi action session; returns action_session_id."""
    manager = _get_action_manager()
    async with _exec_lock:
        return await manager.create_session(
            session_id=session_id,
            action_session_seq_id=action_session_seq_id,
            base_model=base_model,
            model_path=model_path,
            user_id=user_id,
        )


def has_local_action_session(action_session_id: str) -> bool:
    manager = _get_action_manager()
    return action_session_id in getattr(manager, "_runtime_clients", {})


async def handle_act(
    *,
    action_session_id: str,
    observation: Any,
    extra_inputs: dict[str, Any],
    temperature: float | None,
    return_rollout_trace: bool | None,
    rollout_trace_config: dict[str, Any] | None,
) -> str:
    """Run inference inline; store the act result in the process-local future."""
    request_id = f"act_{uuid.uuid4().hex}"
    manager = _get_action_manager()

    async def _do() -> dict[str, Any]:
        out = await manager.act(
            action_session_id=action_session_id,
            observation=observation,
            extra_inputs=extra_inputs,
            temperature=temperature,
            return_rollout_trace=return_rollout_trace,
            rollout_trace_config=rollout_trace_config,
        )
        payload = dict(out)
        payload["type"] = "act"
        return payload

    await _run_inline(request_id, _do)
    return request_id


async def handle_act_batch(
    *,
    action_session_id: str,
    observations: list[Any],
    temperature: float | None,
) -> str:
    """Run batched inference inline; see action_session_manager.act_batch and
    openpi_pi05_action_worker.OpenPIPi05ActionSession.act_batch for why this
    exists (multi-GPU data-sharded inference instead of one un-jitted,
    un-batched call per frame -- see ExperimentLog_MultiGPU.md / the batch
    inference experiment docs for the ~100x+ latency gap this closes)."""
    request_id = f"act_batch_{uuid.uuid4().hex}"
    manager = _get_action_manager()

    async def _do() -> dict[str, Any]:
        out = await manager.act_batch(
            action_session_id=action_session_id,
            observations=observations,
            temperature=temperature,
        )
        payload = dict(out)
        payload["type"] = "act_batch"
        return payload

    await _run_inline(request_id, _do)
    return request_id


async def handle_shutdown_action_session(action_session_id: str) -> None:
    manager = _get_action_manager()
    try:
        await manager.shutdown_session(action_session_id)
    except Exception:
        logger.warning("[openpi-local] action shutdown failed for %s", action_session_id)
