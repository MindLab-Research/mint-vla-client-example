"""FastAPI application for tinker-server."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .auth_identity import get_apikey_id as get_request_apikey_id
from .auth_identity import get_request_observability_context
from .backend.task_state_store import TaskStateStoreUnavailableError
from .backend.session_manager import SessionManager
from .config import config
from .gateway import close_http_clients
from .health_state import (
    clear_runtime_degraded_state,
    clear_startup_degraded_state,
    set_runtime_degraded_state,
    set_startup_degraded_state,
)
from .gateway_auth import extract_gateway_auth_context, has_gateway_auth_headers
from .logging_context import (
    classify_failure_reason,
    bind_request_trace_context,
    ensure_trace_id,
    extract_trace_id_from_traceparent,
    get_trace_id,
    get_otel_tracer,
    record_http_server_metrics,
    set_trace_id,
)
from .ray_utils import init_ray, ray_connection_epoch, ray_reconnect_poll_s
from .routes import action_sampling, futures, internal, openai_compat, sampling, service, training, weights
from .server_info import _git_sha
from .token_encryptor import TokenEncryptor

if TYPE_CHECKING:
    from .backend.multi_lora_engine import MultiModelInferenceManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
_STARTUP_LEASE_ROLE = os.environ.get("MINT_STARTUP_LEASE_ROLE", "mint_api_startup_owner")
_DISABLE_MINT_ROUTE = os.environ.get("MINT_DISABLE_MINT_ROUTE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if _DISABLE_MINT_ROUTE:
    mint = None
else:
    from .routes import mint


def _maintenance_cron_snapshot_details(
    snapshot: dict[str, object],
    *,
    loop_name: object | None = None,
    loop_state: dict[str, object] | None = None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "actor_name": snapshot.get("actor_name"),
        "namespace": snapshot.get("namespace"),
        "epoch_id": snapshot.get("epoch_id"),
        "code_identity": snapshot.get("code_identity"),
    }
    if loop_name is not None:
        details["loop"] = str(loop_name)
    if loop_state is not None:
        for key in (
            "last_error_type",
            "last_error_at",
            "last_success_at",
            "error_count",
            "success_count",
        ):
            if key in loop_state:
                details[key] = loop_state.get(key)
    return {key: value for key, value in details.items() if value is not None}


def _http_route_label(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return request.url.path


def _should_preload_openai_tokenizers() -> bool:
    raw = os.environ.get("MINT_OAI_PRELOAD_TOKENIZERS", "auto").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    try:
        workers = max(1, int(os.environ.get("MINT_UVICORN_WORKERS", "1")))
    except Exception:
        workers = 1
    return workers <= 1


async def _cleanup_stale_actors() -> None:
    try:
        from .backend.actor_reconciliation import cleanup_stale_actors_once

        await cleanup_stale_actors_once()
    except Exception as e:
        set_startup_degraded_state(
            reason="startup_actor_cleanup_failed",
            error=f"{type(e).__name__}: {e}",
        )
        logger.error(f"Actor cleanup failed; healthz will be degraded: {type(e).__name__}: {e}")


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _shutdown_local_inference_runtime(inference_manager: SessionManager) -> None:
    await _cancel_task(getattr(inference_manager, "_cleanup_task", None))
    if hasattr(inference_manager, "_cleanup_task"):
        inference_manager._cleanup_task = None

    sessions = dict(getattr(inference_manager, "_sessions", {}))
    getattr(inference_manager, "_sessions", {}).clear()
    for session_id, info in sessions.items():
        engine = getattr(info, "engine", None)
        if engine is None or bool(getattr(info, "is_shared", False)):
            continue
        try:
            await engine.shutdown()
            logger.info("Locally shutdown inference engine for session %s", session_id)
        except Exception as e:
            logger.warning("Local inference runtime shutdown failed session=%s: %s", session_id, e)

    shared_engine = getattr(inference_manager, "_shared_engine", None)
    if shared_engine is not None:
        try:
            await shared_engine.shutdown()
            logger.info("Locally shutdown shared inference engine")
        except Exception as e:
            logger.warning("Local shared inference engine shutdown failed: %s", e)
        inference_manager._shared_engine = None


async def _shutdown_local_training_runtime(train_manager) -> None:
    await _cancel_task(getattr(train_manager, "_cleanup_task", None))
    if hasattr(train_manager, "_cleanup_task"):
        train_manager._cleanup_task = None

    sessions = dict(getattr(train_manager, "_sessions", {}))
    getattr(train_manager, "_sessions", {}).clear()
    for model_id, session in sessions.items():
        inference_engine = getattr(session, "inference_engine", None)
        if inference_engine is None:
            continue
        try:
            await inference_engine.shutdown()
            logger.info("Locally shutdown training-side inference engine for model %s", model_id)
        except Exception as e:
            logger.warning("Local training runtime shutdown failed model=%s: %s", model_id, e)


def _clear_local_execution_route_globals() -> None:
    from .routes import sampling, service, training, weights

    service.session_manager = None
    sampling.session_manager = None
    training.training_manager = None
    training.training_engine = None
    training.inference_manager = None
    if mint is not None:
        mint.training_manager = None
        mint.training_engine = None
    weights.training_manager = None
    weights.training_engine = None
    weights.inference_manager = None

async def _restore_sampling_sessions(inference_manager: SessionManager) -> int:
    """Restore detached sampling-session metadata into SessionManager."""
    from .backend.sampling_session_store import async_list_sampling_sessions

    restored = 0
    for info in await async_list_sampling_sessions():
        try:
            if inference_manager.restore_sampling_session(info):
                restored += 1
        except Exception as e:
            logger.warning(
                "Failed to restore sampling session %r from detached store: %s: %s",
                info.get("session_id"),
                type(e).__name__,
                e,
            )
    if restored:
        logger.info("Restored %s sampling session(s) from detached store", restored)
    return restored


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Initializes both inference SessionManager and training components
    on startup, shuts down all sessions on application exit.
    """
    # ==========================================================================
    # Ray driver: startup invariant
    # ==========================================================================
    clear_startup_degraded_state()
    clear_runtime_degraded_state()
    from .backend.task_state_store import task_state_futures
    from .backend.config_actor import async_ensure_started as async_ensure_config_actor_started
    from .backend.gateway_session_store import ensure_ready as ensure_gateway_session_store_ready
    from .backend.maintenance_cron_actor import maintenance_cron_actor
    from .backend.sampling_session_store import ensure_ready as ensure_sampling_session_store_ready
    from .backend.session_heartbeat_store import session_heartbeat_store
    from .backend.session_index_store import ensure_ready as ensure_session_index_store_ready
    from .backend.startup_lease import acquire_startup_lease
    from .backend.training_session_store import ensure_ready as ensure_training_session_store_ready
    from .config import RAY_NAMESPACE

    init_ray(namespace=RAY_NAMESPACE, ignore_reinit_error=True)

    startup_lease = await acquire_startup_lease(_STARTUP_LEASE_ROLE)
    startup_owner = bool(startup_lease.is_owner)
    startup_lease_task: asyncio.Task | None = None
    if startup_owner and not startup_lease.local_only:
        startup_lease_task = asyncio.create_task(startup_lease.heartbeat_loop())
    logger.info(
        "startup lease role=%s is_owner=%s local_only=%s owner_id=%s",
        _STARTUP_LEASE_ROLE,
        startup_owner,
        startup_lease.local_only,
        startup_lease.owner_id,
    )

    maintenance_cron_local_only = os.environ.get("MINT_MAINTENANCE_CRON_LOCAL_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}

    from .usage_store import get_usage_store

    usage_store = await get_usage_store()
    if not await usage_store.health_check():
        raise RuntimeError("usage billing postgres health check failed")
    if startup_owner:
        await async_ensure_config_actor_started()
        await task_state_futures.async_ensure_started()
        ensure_gateway_session_store_ready()
        ensure_sampling_session_store_ready()
        session_heartbeat_store.ensure_ready()
        ensure_session_index_store_ready()
        ensure_training_session_store_ready()
        from .backend.future_replay import ensure_future_replay_sweeper

        ensure_future_replay_sweeper()
    if maintenance_cron_local_only:
        maintenance_cron = {"actor_name": "local_maintenance_cron_actor", "epoch_id": "local"}
    else:
        maintenance_cron = await maintenance_cron_actor.async_ensure_started()

    from .backend.action_session_manager import ActionSessionRouter

    action_manager = ActionSessionRouter()
    action_sampling.action_session_manager = action_manager
    if mint is not None:
        mint.action_session_manager = action_manager
    try:
        from .backend.dense_session_state import cleanup_legacy_dense_session_state_once
        from .backend.training_session_store import async_list_training_sessions

        active_model_ids = {
            str(info.get("model_id"))
            for info in await async_list_training_sessions()
            if isinstance(info, dict) and str(info.get("model_id") or "").strip()
        }
        dense_cleanup = await asyncio.to_thread(
            cleanup_legacy_dense_session_state_once,
            active_session_ids=active_model_ids,
        )
        migrated_items = dense_cleanup.get("migrated", [])
        deleted_items = dense_cleanup.get("deleted", [])
        skipped_items = dense_cleanup.get("skipped", [])
        errors = dense_cleanup.get("errors", [])
        migrated = len(migrated_items) if isinstance(migrated_items, list) else 0
        deleted = len(deleted_items) if isinstance(deleted_items, list) else 0
        skipped = len(skipped_items) if isinstance(skipped_items, list) else 0
        if migrated or deleted or skipped or errors:
            logger.info(
                "dense session-state startup cleanup target_root=%s migrated=%s deleted=%s skipped=%s errors=%s",
                dense_cleanup.get("target_root"),
                migrated,
                deleted,
                skipped,
                len(errors) if isinstance(errors, list) else 0,
            )
            if errors:
                logger.warning("dense session-state startup cleanup errors: %s", errors)
    except Exception:
        logger.exception("dense session-state startup cleanup failed")

    app_module_git_sha = _git_sha()
    logger.info(
        "maintenance cron actor ready actor=%s epoch=%s",
        maintenance_cron.get("actor_name"),
        maintenance_cron.get("epoch_id"),
    )

    def _maintenance_cron_health_error(snapshot: dict[str, object]) -> tuple[str, str, dict[str, object]] | None:
        code_identity = snapshot.get("code_identity")
        if code_identity != app_module_git_sha:
            return (
                "maintenance_cron_actor_code_mismatch",
                f"expected code_identity={app_module_git_sha!r} actual={code_identity!r}",
                _maintenance_cron_snapshot_details(snapshot),
            )
        loops = snapshot.get("loops")
        if isinstance(loops, dict):
            for loop_name, raw in loops.items():
                if not isinstance(raw, dict):
                    continue
                last_error = raw.get("last_error")
                last_error_at = raw.get("last_error_at")
                last_success_at = raw.get("last_success_at")
                if last_error and last_error_at is not None and (
                    last_success_at is None or float(last_error_at) >= float(last_success_at)
                ):
                    return (
                        "maintenance_cron_actor_loop_error",
                        f"loop={loop_name} last_error={last_error}",
                        _maintenance_cron_snapshot_details(snapshot, loop_name=loop_name, loop_state=raw),
                    )
        return None

    async def _maintenance_cron_health_loop() -> None:
        while True:
            try:
                snapshot = await maintenance_cron_actor.async_health_snapshot(timeout_s=10.0)
                err = _maintenance_cron_health_error(snapshot)
                if err is None:
                    clear_runtime_degraded_state()
                else:
                    reason, error, details = err
                    set_runtime_degraded_state(reason=reason, error=error, details=details)
            except Exception as e:
                set_runtime_degraded_state(
                    reason="maintenance_cron_actor_unavailable",
                    error=f"{type(e).__name__}: {e}",
                    details={},
                )
            await asyncio.sleep(5.0)

    if maintenance_cron_local_only:
        clear_runtime_degraded_state()
        maintenance_cron_health_task = None
    else:
        maintenance_cron_health_task = asyncio.create_task(_maintenance_cron_health_loop())
    ray_reconnect_watch_task: asyncio.Task | None = None
    model_actor_supervisor_task: asyncio.Task | None = None
    last_ray_connection_epoch = ray_connection_epoch()

    inference_manager = None
    train_manager = None
    multi_model_manager = None
    stale_training_heartbeat_task = None

    try:
        # ==========================================================================
        # Cleanup: Kill stale actors from previous server runs
        # ==========================================================================
        if startup_owner:
            if maintenance_cron_local_only:
                await _cleanup_stale_actors()
            else:
                await maintenance_cron_actor.async_run_once("actor_reconciliation", timeout_s=60.0)
        else:
            logger.info("Skipping stale-actor cleanup on follower worker")

        # ==========================================================================
        # Action route layer: process-local router can recover detached runtimes
        # from ModelActorSupervisorInventory metadata after API or worker restarts.
        # ==========================================================================
        action_sampling.action_session_manager = action_manager
        if mint is not None:
            mint.action_session_manager = action_manager

        # ==========================================================================
        # Inference route layer: stateless API path uses detached stores only
        # ==========================================================================
        inference_manager = None
        service.session_manager = None
        sampling.session_manager = None
        multi_model_manager: MultiModelInferenceManager | None = None

        # ==========================================================================
        # Training route layer: stateless API path uses detached stores only
        # ==========================================================================
        training.training_manager = None
        training.training_engine = None
        training.inference_manager = None
        if mint is not None:
            mint.training_manager = None
            mint.training_engine = None
        weights.training_manager = None
        weights.training_engine = None
        weights.inference_manager = None
        logger.info(
            "Training route globals left unbound in API process; detached queue runtime owns training execution state"
        )

        # ==========================================================================
        # Persistent prewarm runs inside the execution runtime that owns training state.
        # ==========================================================================
        if not startup_owner:
            logger.info("Skipping execution-runtime prewarm on follower worker")

        # ==========================================================================
        # OpenAI compat: preload tokenizers only for single-worker startup.
        # Multi-worker preloading duplicates large tokenizer state in every API process
        # and can destabilize worker startup; lazy loading remains available per request.
        # ==========================================================================
        if _should_preload_openai_tokenizers():
            try:
                preload_failures = openai_compat.preload_supported_tokenizers()
                if preload_failures:
                    logger.warning(
                        "OpenAI-compatible tokenizer preload incomplete: %s",
                        preload_failures,
                    )
                else:
                    logger.info("OpenAI-compatible tokenizers preloaded")
            except Exception as e:
                logger.exception("OpenAI-compatible tokenizer preload failed: %s", e)
        else:
            logger.info("Skipping OpenAI-compatible tokenizer preload for multi-worker startup")

        # ==========================================================================
        # Model scheduler + runtime supervisors
        # ==========================================================================
        from .backend.model_actor_supervisor import model_actor_supervisor
        from .backend.model_work_scheduler import model_work_scheduler

        logger.info("startup stage=before_model_work_scheduler_started")
        await model_work_scheduler.stats(timeout_s=10.0)
        logger.info("startup stage=after_model_work_scheduler_started")

        if startup_owner and model_actor_supervisor.snapshot().get("desired_total", 0):
            logger.info("startup stage=before_model_actor_supervisor_reconcile")
            await model_actor_supervisor.reconcile_once()
            logger.info("startup stage=after_model_actor_supervisor_reconcile")

        async def _model_actor_supervisor_loop() -> None:
            interval_s = float(os.environ.get("MINT_MODEL_ACTOR_SUPERVISOR_RECONCILE_INTERVAL_S", "5.0"))
            while True:
                await asyncio.sleep(max(1.0, interval_s))
                try:
                    if model_actor_supervisor.snapshot().get("desired_total", 0):
                        await model_actor_supervisor.reconcile_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("model actor supervisor reconcile failed")

        if startup_owner and model_actor_supervisor.snapshot().get("desired_total", 0):
            model_actor_supervisor_task = asyncio.create_task(_model_actor_supervisor_loop())

        async def _ray_reconnect_watch_loop() -> None:
            nonlocal last_ray_connection_epoch
            poll_s = ray_reconnect_poll_s()
            while True:
                await asyncio.sleep(poll_s)
                try:
                    init_ray(namespace=RAY_NAMESPACE, ignore_reinit_error=True)
                    current_epoch = ray_connection_epoch()
                    if current_epoch == last_ray_connection_epoch:
                        continue
                    last_ray_connection_epoch = current_epoch
                    logger.warning(
                        "Ray connection epoch advanced to %s; refreshing detached control-plane handles",
                        current_epoch,
                    )
                    await async_ensure_config_actor_started()
                    await task_state_futures.async_ensure_started()
                    ensure_gateway_session_store_ready()
                    ensure_sampling_session_store_ready()
                    session_heartbeat_store.ensure_ready()
                    ensure_session_index_store_ready()
                    ensure_training_session_store_ready()
                    await model_work_scheduler.stats(timeout_s=10.0)
                    if model_actor_supervisor.snapshot().get("desired_total", 0):
                        await model_actor_supervisor.reconcile_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Ray reconnect watch failed")

        ray_reconnect_watch_task = asyncio.create_task(_ray_reconnect_watch_loop())

        stale_training_heartbeat_task = None

    except Exception:
        await _cancel_task(model_actor_supervisor_task)
        await _cancel_task(ray_reconnect_watch_task)
        await _cancel_task(startup_lease_task)
        await startup_lease.release()
        if train_manager is not None:
            await _shutdown_local_training_runtime(train_manager)
        if inference_manager is not None:
            await _shutdown_local_inference_runtime(inference_manager)
        if multi_model_manager is not None:
            await multi_model_manager.shutdown_all()
        _clear_local_execution_route_globals()
        raise

    yield

    # ==========================================================================
    # Shutdown
    # ==========================================================================
    await _cancel_task(ray_reconnect_watch_task)
    await _cancel_task(model_actor_supervisor_task)
    await _cancel_task(maintenance_cron_health_task)
    await _cancel_task(stale_training_heartbeat_task)
    await _cancel_task(startup_lease_task)
    await startup_lease.release()
    logger.info("Shutting down local runtime state")

    # Do not let an arbitrary API worker exit delete shared metadata or global actors.
    if train_manager is not None:
        await _shutdown_local_training_runtime(train_manager)
    if inference_manager is not None:
        await _shutdown_local_inference_runtime(inference_manager)

    # Shutdown multi-model inference manager
    if multi_model_manager is not None:
        await multi_model_manager.shutdown_all()
        logger.info("Multi-model inference manager shutdown")
    _clear_local_execution_route_globals()

    openai_compat.shutdown_tokenizer_executor()

    from .usage_store import close_usage_store

    await close_usage_store()


    await close_http_clients()


app = FastAPI(
    lifespan=lifespan,
    title="MinT",
    description="Mind Lab Toolkit - Training API for LLMs",
    version="0.1.0",
    docs_url=None,  # Disable built-in Swagger UI
)

@app.exception_handler(TaskStateStoreUnavailableError)
async def task_state_store_unavailable_handler(_: Request, __: TaskStateStoreUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "TaskStateStore unavailable"},
    )


# Paths that don't require authentication
UNAUTHENTICATED_PATHS = {"/api/v1/healthz", "/"}

# Paths excluded from OTel span creation (high-frequency polling endpoints).
# Set MINT_OTEL_EXCLUDE_NONE=1 to disable exclusions and trace everything.
_OTEL_EXCLUDE_NONE = os.environ.get("MINT_OTEL_EXCLUDE_NONE", "").strip().lower() in ("1", "true", "yes")
_OTEL_EXCLUDED_PATHS: set[str] = set() if _OTEL_EXCLUDE_NONE else {
    "/api/v1/retrieve_future",
    "/api/v1/healthz",
    "/api/v1/telemetry",
    "/api/v1/session_heartbeat",
    "/api/v1/internal/admission_stats",
    "/internal/admission_stats",
    "/api/v1/internal/metrics",
    "/internal/metrics",
}

# Token encryptor for sk- token validation (initialized lazily)
_token_encryptor: TokenEncryptor | None = None


def get_token_encryptor() -> TokenEncryptor | None:
    """Get or create token encryptor if secret key is configured."""
    global _token_encryptor
    if _token_encryptor is None and config.token_secret_key:
        _token_encryptor = TokenEncryptor(config.token_secret_key)
    return _token_encryptor


@app.middleware("http")
async def otel_trace_metrics_middleware(request: Request, call_next):
    """Manual OTel instrumentation for HTTP server traces and metrics."""
    tracer = get_otel_tracer()
    method = request.method
    route = _http_route_label(request)
    start_s = time.perf_counter()
    status_code = 500
    failure_error: Exception | None = None

    def _request_obs() -> dict[str, str]:
        return get_request_observability_context(request)

    def _apply_http_identity_to_span(span) -> None:
        for key, value in _request_obs().items():
            span.set_attribute(f"mint.{key}", value)

    def _log_request_observation(elapsed_ms: float) -> None:
        obs = _request_obs()
        user_id = obs.get("user_id", "-")
        user_role = obs.get("user_role", "-")
        account_id = obs.get("account_id", "-")
        apikey_id = obs.get("apikey_id", "-")
        gateway_request_id = obs.get("gateway_request_id", "-")
        gateway_session_id = obs.get("gateway_session_id", "-")
        if status_code >= 500:
            reason = classify_failure_reason(failure_error or RuntimeError(f"http_{status_code}"))
            logger.error(
                "[http.request] failed method=%s route=%s status_code=%s elapsed_ms=%.3f "
                "user_id=%s user_role=%s account_id=%s apikey_id=%s gateway_request_id=%s gateway_session_id=%s "
                "failure_reason=%s error_type=%s next_action=%s",
                method,
                route,
                int(status_code),
                float(elapsed_ms),
                user_id,
                user_role,
                account_id,
                apikey_id,
                gateway_request_id,
                gateway_session_id,
                reason,
                type(failure_error).__name__ if failure_error is not None else "HTTPStatusError",
                "check_logs_and_trace",
            )
            return
        if status_code >= 400:
            logger.warning(
                "[http.request] client_error method=%s route=%s status_code=%s elapsed_ms=%.3f "
                "user_id=%s user_role=%s account_id=%s apikey_id=%s gateway_request_id=%s gateway_session_id=%s",
                method,
                route,
                int(status_code),
                float(elapsed_ms),
                user_id,
                user_role,
                account_id,
                apikey_id,
                gateway_request_id,
                gateway_session_id,
            )
            return
        logger.info(
            "[http.request] completed method=%s route=%s status_code=%s elapsed_ms=%.3f "
            "user_id=%s user_role=%s account_id=%s apikey_id=%s gateway_request_id=%s gateway_session_id=%s",
            method,
            route,
            int(status_code),
            float(elapsed_ms),
            user_id,
            user_role,
            account_id,
            apikey_id,
            gateway_request_id,
            gateway_session_id,
        )

    # Skip OTel span and request logging for high-frequency polling endpoints.
    # Metrics are still recorded; only traces and per-request log lines are suppressed.
    _skip_otel = route in _OTEL_EXCLUDED_PATHS
    if tracer is None or _skip_otel:
        try:
            response = await call_next(request)
            status_code = int(getattr(response, "status_code", 500))
            return response
        except Exception as e:
            failure_error = e
            raise
        finally:
            route = _http_route_label(request)
            elapsed_ms = (time.perf_counter() - start_s) * 1000.0
            record_http_server_metrics(
                method=method,
                route=route,
                status_code=status_code,
                duration_ms=elapsed_ms,
            )
            if not _skip_otel:
                _log_request_observation(elapsed_ms)

    try:
        from opentelemetry.propagate import extract
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except Exception:
        try:
            response = await call_next(request)
            status_code = int(getattr(response, "status_code", 500))
            return response
        except Exception as e:
            failure_error = e
            raise
        finally:
            route = _http_route_label(request)
            elapsed_ms = (time.perf_counter() - start_s) * 1000.0
            record_http_server_metrics(
                method=method,
                route=route,
                status_code=status_code,
                duration_ms=elapsed_ms,
            )
            _log_request_observation(elapsed_ms)

    context = extract(dict(request.headers))
    span_name = f"{method} {route}"
    with tracer.start_as_current_span(span_name, context=context, kind=SpanKind.SERVER) as span:
        span_ctx = span.get_span_context()
        if span_ctx and getattr(span_ctx, "trace_id", 0):
            trace_id = f"{int(span_ctx.trace_id):032x}"
            set_trace_id(trace_id)
            request.state.trace_id = trace_id
        span.set_attribute("http.method", method)
        span.set_attribute("http.route", route)
        _apply_http_identity_to_span(span)
        error_recorded = False

        def _record_server_error(error: Exception, *, escaped: bool) -> None:
            nonlocal error_recorded
            span.record_exception(error, attributes={"exception.escaped": bool(escaped)})
            error_recorded = True

        try:
            response = await call_next(request)
            route = _http_route_label(request)
            status_code = int(getattr(response, "status_code", 500))
            try:
                span.update_name(f"{method} {route}")
            except Exception:
                pass
            _apply_http_identity_to_span(span)
            span.set_attribute("http.status_code", status_code)
            span.set_attribute("http.route", route)
            if status_code >= 500:
                # FastAPI may convert errors into HTTP 5xx responses before they
                # propagate here. Record a synthetic error so traces still include
                # an explicit error record for server failures.
                if not error_recorded:
                    _record_server_error(
                        RuntimeError(f"HTTP {status_code} response for {method} {route}"),
                        escaped=False,
                    )
                span.set_status(Status(StatusCode.ERROR, f"http.status_code={status_code}"))
            return response
        except Exception as e:
            failure_error = e
            if isinstance(e, HTTPException):
                try:
                    status_code = int(e.status_code)
                except Exception:
                    status_code = 500
            else:
                status_code = 500
            route = _http_route_label(request)
            try:
                span.update_name(f"{method} {route}")
            except Exception:
                pass
            _apply_http_identity_to_span(span)
            span.set_attribute("http.status_code", status_code)
            span.set_attribute("http.route", route)
            if status_code >= 500:
                _record_server_error(e, escaped=True)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            route = _http_route_label(request)
            elapsed_ms = (time.perf_counter() - start_s) * 1000.0
            record_http_server_metrics(
                method=method,
                route=route,
                status_code=status_code,
                duration_ms=elapsed_ms,
            )
            _log_request_observation(elapsed_ms)


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Validate gateway-forwarded auth headers (preferred) with legacy fallback."""
    path = request.url.path
    traceparent_trace_id = extract_trace_id_from_traceparent(request.headers.get("traceparent"))
    incoming_trace_id = traceparent_trace_id
    if incoming_trace_id is None and get_trace_id() is None:
        incoming_trace_id = request.headers.get("X-Trace-Id")
    trace_id = ensure_trace_id(incoming_trace_id)
    request.state.trace_id = trace_id

    def _with_trace(response):
        final_trace_id = ensure_trace_id(
            getattr(request.state, "trace_id", None) or get_trace_id() or trace_id
        )
        request.state.trace_id = final_trace_id
        response.headers["X-Trace-Id"] = final_trace_id
        response.headers["X-MinT-Server-Pid"] = str(os.getpid())
        apikey_id = get_request_apikey_id(request)
        if apikey_id:
            response.headers["X-MinT-Apikey-Id"] = apikey_id
        return response

    async def _next_with_trace():
        return _with_trace(await call_next(request))

    # Skip auth for specific paths.
    if path in UNAUTHENTICATED_PATHS:
        return await _next_with_trace()

    # Special-case: allow unauthenticated checkpoint archive downloads when a valid,
    # short-lived signed token is provided in the URL (Tinker SDK expects a signed URL).
    if path.startswith("/api/v1/training_runs/") and path.endswith("/archive"):
        direct = request.query_params.get("direct")
        download_token = request.query_params.get("download_token")
        if direct and download_token:
            try:
                from .download_tokens import verify_download_token

                # Prefer token_secret_key (if configured), otherwise api_key.
                secret = config.token_secret_key or config.api_key or ""
                payload = verify_download_token(str(download_token), secret=secret)
                if payload is None:
                    raise ValueError("invalid token")

                prefix = "/api/v1/training_runs/"
                mid_and_rest = path[len(prefix) :]
                model_id, rest = mid_and_rest.split("/checkpoints/", 1)
                checkpoint_id = rest[: -len("/archive")]

                if payload.get("model_id") != model_id or payload.get("checkpoint_id") != checkpoint_id:
                    raise ValueError("token does not match request path")

                request.state.user_data = {"user_id": payload.get("user_id")}
                return await _next_with_trace()
            except Exception:
                return _with_trace(JSONResponse(status_code=401, content={"error": "Invalid download token"}))

    if path.startswith(("/api/v1/", "/internal/")):
        if has_gateway_auth_headers(dict(request.headers)):
            try:
                auth_ctx = extract_gateway_auth_context(
                    request,
                    internal_api_token=config.internal_api_token,
                )
            except HTTPException as exc:
                return _with_trace(JSONResponse(status_code=exc.status_code, content={"error": exc.detail}))
            request.state.gateway_auth = auth_ctx
            request.state.user_data = {
                "user_id": auth_ctx.user_id,
                "user_role": auth_ctx.user_role,
                "is_admin": auth_ctx.user_role == "admin",
                "account_id": auth_ctx.account_id,
                "apikey_id": auth_ctx.apikey_id,
                "request_id": auth_ctx.request_id,
                "session_id": auth_ctx.session_id,
                "cap_write": auth_ctx.cap_write,
                "cap_view_internal_errors": auth_ctx.cap_view_internal_errors,
                "cap_bypass_ownership": auth_ctx.cap_bypass_ownership,
                "cap_manage_system": auth_ctx.cap_manage_system,
                "caps_from_headers": auth_ctx.caps_from_headers,
            }
            with bind_request_trace_context(
                request_id=auth_ctx.request_id,
                trace_id=trace_id,
                user_id=auth_ctx.user_id,
                user_role=auth_ctx.user_role,
                account_id=auth_ctx.account_id,
                apikey_id=auth_ctx.apikey_id,
                gateway_request_id=auth_ctx.request_id,
                gateway_session_id=auth_ctx.session_id,
            ):
                return await _next_with_trace()

        # Legacy auth disabled => dev mode pass-through with explicit write caps.
        if not config.auth_enabled:
            existing_user_data = getattr(request.state, "user_data", None)
            if not isinstance(existing_user_data, dict):
                request.state.user_data = {
                    "user_id": "000000000000000000000001",
                    "user_role": "admin",
                    "is_admin": True,
                    "account_id": "000000000000000000000001",
                    "apikey_id": "000000000000000000000002",
                    "cap_write": True,
                    "cap_view_internal_errors": True,
                    "cap_bypass_ownership": True,
                    "cap_manage_system": True,
                    "caps_from_headers": True,
                }
            obs = get_request_observability_context(request)
            with bind_request_trace_context(
                trace_id=trace_id,
                user_id=obs.get("user_id"),
                user_role=obs.get("user_role"),
                account_id=obs.get("account_id"),
                apikey_id=obs.get("apikey_id"),
                gateway_request_id=obs.get("gateway_request_id"),
                gateway_session_id=obs.get("gateway_session_id"),
            ):
                return await _next_with_trace()

        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]
            elif auth_header.startswith("sk-"):
                api_key = auth_header

        if not api_key:
            return _with_trace(JSONResponse(status_code=401, content={"error": "Missing API key"}))

        if config.validate_api_key(api_key):
            request.state.user_data = {"user_id": "admin", "user_role": "admin", "is_admin": True}
            with bind_request_trace_context(
                trace_id=trace_id,
                user_id="admin",
                user_role="admin",
            ):
                return await _next_with_trace()

        if api_key.startswith("sk-") and config.token_secret_key:
            encryptor = get_token_encryptor()
            if encryptor:
                user_data = encryptor.decrypt_token(api_key)
                if user_data is not None:
                    if "user_role" not in user_data:
                        user_data["user_role"] = "admin" if user_data.get("user_id") == "admin" else "user"
                    if "is_admin" not in user_data:
                        user_data["is_admin"] = user_data.get("user_role") == "admin"
                    request.state.user_data = user_data
                    obs = get_request_observability_context(request)
                    with bind_request_trace_context(
                        request_id=obs.get("gateway_request_id"),
                        trace_id=trace_id,
                        user_id=obs.get("user_id"),
                        user_role=obs.get("user_role"),
                        account_id=obs.get("account_id"),
                        apikey_id=obs.get("apikey_id"),
                        gateway_request_id=obs.get("gateway_request_id"),
                        gateway_session_id=obs.get("gateway_session_id"),
                    ):
                        return await _next_with_trace()
        return _with_trace(JSONResponse(status_code=401, content={"error": "Invalid API key or token"}))
    with bind_request_trace_context(trace_id=trace_id):
        return await _next_with_trace()


# Register routes with API prefix
app.include_router(service.router, prefix="/api/v1", tags=["service"])
app.include_router(action_sampling.router, prefix="/api/v1", tags=["action_sampling"])
app.include_router(sampling.router, prefix="/api/v1", tags=["sampling"])
app.include_router(futures.router, prefix="/api/v1", tags=["futures"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(weights.router, prefix="/api/v1", tags=["weights"])
if mint is not None:
    app.include_router(mint.router, prefix="/api/v1/mint", tags=["mint"])
app.include_router(openai_compat.router, prefix="/oai/api/v1", tags=["openai-compat"])
app.include_router(internal.router, prefix="/internal", tags=["internal"])

@app.get("/")
async def root():
    return {"status": "ready", "healthz": "/api/v1/healthz"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
