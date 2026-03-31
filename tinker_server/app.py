"""FastAPI application for tinker-server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .auth_identity import get_apikey_id as get_request_apikey_id
from .auth_identity import get_request_observability_context
from .backend.api_work_queue import ApiWorkQueueUnavailableError
from .backend.capacity_manager import CapacityManagerUnavailableError
from .backend.future_store import FutureStoreUnavailableError
from .backend.session_manager import DEFAULT_INACTIVITY_TIMEOUT, SessionManager
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
    run_async_with_otel_span,
    set_trace_id,
)
from .ray_utils import init_ray
from .routes import futures, internal, mint, openai_compat, sampling, service, training, weights
from .server_info import _git_sha
from .token_encryptor import TokenEncryptor

if TYPE_CHECKING:
    from .backend.multi_lora_engine import MultiModelInferenceManager
    from .backend.verl_training import VerlTrainingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
_STARTUP_LEASE_ROLE = os.environ.get("MINT_STARTUP_LEASE_ROLE", "mint_api_startup_owner")


def _http_route_label(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return request.url.path


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

async def _prewarm_persistent_models(
    train_engine: VerlTrainingEngine,
    multi_model_manager: MultiModelInferenceManager | None,
) -> None:
    """Pre-create and protect persistent actors at server startup.

    Controlled by:
      - MINT_PERSISTENT_MODELS: comma-separated HF model names
      - MINT_PERSISTENT_TRAIN_LORA_RANK (default: 16)
      - MINT_PERSISTENT_TRAIN_LR (default: 5e-5)

    When enabled, creates:
      - Training actors (pooled PEFT trainers and MegatronWorkerGroup)
      - vLLM inference actors (MultiModelInferenceManager)

    and marks them as ResourcePool protected to prevent LRU eviction.
    """
    failures: list[str] = []

    def _record_failure(stage: str, model_name: str, exc: Exception) -> None:
        failures.append(f"{stage} failed model={model_name}: {type(exc).__name__}: {exc}")

    def _raise_if_failures() -> None:
        if failures:
            raise RuntimeError("persistent prewarm failed:\n" + "\n".join(failures))

    models_csv = (config.prewarm_persistent_models_csv or "").strip()
    if not models_csv:
        logger.info("No persistent models configured (MINT_PERSISTENT_MODELS empty); skipping prewarm")
        return

    models = [m.strip() for m in models_csv.split(",") if m.strip()]
    if not models:
        logger.info("No persistent models configured (MINT_PERSISTENT_MODELS parsed empty); skipping prewarm")
        return

    lora_rank = int(config.prewarm_train_lora_rank)
    learning_rate = float(config.prewarm_train_lr)
    megatron_ready_timeout_s = float(config.prewarm_megatron_ready_timeout_s)
    prewarm_training = bool(config.prewarm_enable_training)
    prewarm_inference = bool(config.prewarm_enable_inference)

    from tinker_server.backend.model_registry import (
        get_model_config,
        get_training_parallelism,
        normalize_model_name,
        requires_fp8,
    )
    from tinker_server.backend.resource_pool import get_resource_pool

    resource_pool = get_resource_pool()
    import ray

    pinned_vllm_node_ip: dict[str, str] = {}
    pinned_vllm_node_ip_json = os.environ.get("MINT_VLLM_PINNED_NODE_IP_JSON")
    if pinned_vllm_node_ip_json:
        try:
            parsed = json.loads(pinned_vllm_node_ip_json)
            if isinstance(parsed, dict):
                pinned_vllm_node_ip = {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pinned_vllm_node_ip = {}

    def _preferred_pg_node_id(pg_name: str, model_name: str) -> str | None:
        preferred_ips: list[str] = []
        for env_name in ("MINT_MEGATRON_MODEL_NODE_IPS_JSON", "MINT_MODEL_NODE_IPS_JSON"):
            raw = os.environ.get(env_name, "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            selected = None
            for key in (model_name, model_name.lower()):
                selected = data.get(key)
                if selected is not None:
                    break
            if isinstance(selected, list):
                preferred_ips.extend(str(ip).strip() for ip in selected if str(ip).strip())
            if preferred_ips:
                break

        node_ip_by_id = {
            str(n.get("NodeID") or ""): str(n.get("NodeManagerAddress") or "").strip()
            for n in ray.nodes()
            if n.get("Alive")
        }
        candidate_node_ids: list[str] = []
        for info in ray.util.placement_group_table().values():
            if info.get("state") != "CREATED" or info.get("name") != pg_name:
                continue
            bundles_to_node_id = info.get("bundles_to_node_id") or {}
            for node_id in bundles_to_node_id.values():
                node_id_str = str(node_id or "").strip()
                if node_id_str:
                    candidate_node_ids.append(node_id_str)
        if not candidate_node_ids:
            return None
        if preferred_ips:
            for node_id in candidate_node_ids:
                if node_ip_by_id.get(node_id) in preferred_ips:
                    return node_id
            return None
        return candidate_node_ids[0]

    logger.info(
        f"[prewarm] persistent models={models} train_lora_rank={lora_rank} train_lr={learning_rate} "
        f"megatron_ready_timeout_s={megatron_ready_timeout_s} "
        f"prewarm_training={prewarm_training} prewarm_inference={prewarm_inference}"
    )

    # Order by descending GPU footprint to avoid fragmenting the cluster before
    # large multi-node actors (e.g., 235B vLLM TP=16) are created.
    ordered: list[tuple[int, str, str]] = []
    for model in models:
        try:
            model_name = normalize_model_name(model)
        except Exception:
            model_name = model
        try:
            cfg = get_model_config(model_name)
            gpus = max(cfg.train_gpus, cfg.total_gpus)
        except Exception:
            gpus = 0
        ordered.append((gpus, model_name, model))
    ordered.sort(key=lambda x: (-x[0], x[1]))

    deferred_dense_training: list[tuple[str, str]] = []

    for _, model_name, _raw_model in ordered:
        try:
            cfg = get_model_config(model_name)
        except Exception as e:
            logger.exception(f"[prewarm] unknown model in MINT_PERSISTENT_MODELS: {model_name}: {e}")
            continue

        if prewarm_training:
            # -------------------------
            # Training actor prewarm
            # -------------------------
            try:
                base_model = model_name
                if model_name and not model_name.startswith("/"):
                    base_model = train_engine._resolve_hf_model_path(model_name)
                    if not base_model:
                        raise RuntimeError(f"HF cache path not found for {model_name}")

                if cfg.is_moe:
                    from tinker_server.backend.megatron_distributed import (
                        DistributedConfig,
                        _make_megatron_actor_name,
                        async_get_or_create_megatron_worker_group,
                    )

                    train_tp, train_pp, train_ep, train_cp, train_etp = get_training_parallelism(model_name)
                    use_fp8 = requires_fp8(model_name)
                    distributed_config = DistributedConfig(
                        tensor_parallel_size=train_tp,
                        pipeline_parallel_size=train_pp,
                        expert_parallel_size=train_ep,
                        context_parallel_size=train_cp,
                        expert_tensor_parallel_size=train_etp,
                        use_fp8=use_fp8,
                    )

                    logger.info(
                        f"[prewarm] training create start model={model_name} backend=megatron "
                        f"TP={train_tp} PP={train_pp} EP={train_ep} CP={train_cp} ETP={train_etp} world_size={distributed_config.world_size}"
                    )
                    actor = await async_get_or_create_megatron_worker_group(
                        base_model=base_model,
                        lora_rank=lora_rank,
                        learning_rate=learning_rate,
                        distributed_config=distributed_config,
                        session_id=None,
                    )
                    actor_name = _make_megatron_actor_name(base_model or model_name)
                    # Protect as soon as the actor is registered, so readiness timeouts don't leave it evictable.
                    resource_pool.set_protected(actor_name, True)
                    logger.info(f"[prewarm] training __ray_ready__ scheduled model={model_name} actor={actor_name}")

                    if (
                        cfg.vllm_engine == "async"
                        and cfg.vllm_distributed_executor_backend == "mp"
                        and (cfg.train_gpus + cfg.total_gpus) <= 8
                    ):
                        try:
                            pg_name = f"{actor_name}_pg"
                            node_id = None
                            deadline = time.monotonic() + 30.0
                            while time.monotonic() < deadline and not node_id:
                                node_id = _preferred_pg_node_id(pg_name, model_name)
                                if not node_id:
                                    await asyncio.sleep(0.5)
                            if node_id:
                                for n in ray.nodes():
                                    if n.get("NodeID") == node_id:
                                        ip = n.get("NodeManagerAddress")
                                        if isinstance(ip, str) and ip.strip():
                                            pinned_vllm_node_ip[model_name] = ip.strip()
                                            logger.info(
                                                f"[prewarm] pin_infer model={model_name} pg={pg_name} node_ip={ip.strip()}"
                                            )
                                        break
                        except Exception as pin_err:
                            logger.warning(
                                f"[prewarm] training pin_infer_to_pg_node failed model={model_name}: {pin_err}"
                            )

                    try:
                        await asyncio.to_thread(
                            ray.get,
                            actor.__ray_ready__.remote(),
                            timeout=megatron_ready_timeout_s,
                        )
                        resource_pool.mark_ready(actor_name)
                        logger.info(f"[prewarm] training ready+protected model={model_name} actor={actor_name}")
                    except SystemExit as ready_err:
                        if getattr(ready_err, "code", None) == 15:
                            raise
                        raise RuntimeError(
                            f"[prewarm] training __ray_ready__ SystemExit model={model_name} actor={actor_name}: {ready_err}"
                        ) from ready_err
                    except Exception as ready_err:
                        raise RuntimeError(
                            f"[prewarm] training __ray_ready__ failed/timeout model={model_name} actor={actor_name}: {ready_err}"
                        ) from ready_err
                else:
                    # Defer dense pool creation until after multi-node vLLM inference is initialized,
                    # to avoid fragmenting the remaining 8-GPU nodes into 1-2 free GPUs each.
                    deferred_dense_training.append((model_name, base_model))
                    logger.info(f"[prewarm] training deferred model={model_name} backend=dense_pool")
            except Exception as e:
                _record_failure("training", model_name, e)
                logger.exception(f"[prewarm] training failed model={model_name}: {e}")
        else:
            logger.info(f"[prewarm] training skipped model={model_name} (MINT_PERSISTENT_PREWARM_TRAINING=0)")

        # -------------------------
        # Inference (vLLM) prewarm
        # -------------------------
        if multi_model_manager is None:
            logger.warning(f"[prewarm] inference skipped (multi-LoRA disabled) model={model_name}")
            continue

        # NOTE: prewarm inference is scheduled after training loop, ordered to avoid
        # multi-node vLLM initialization fragmenting the cluster before 4-GPU single-node vLLM
        # actors (e.g., Qwen3-30B TP=4) can be placed.

    if not prewarm_inference:
        _raise_if_failures()
        return

    if multi_model_manager is None:
        _raise_if_failures()
        return

    def _infer_gpus(model_name: str) -> int:
        try:
            cfg = get_model_config(model_name)
            return int(cfg.total_gpus)
        except Exception:
            return 0

    def _infer_is_moe(model_name: str) -> bool:
        try:
            cfg = get_model_config(model_name)
            return bool(cfg.is_moe)
        except Exception:
            return False

    # Order inference:
    # - Single-node MoE first (e.g., Qwen3-30B TP=4) to ensure a 4-GPU slot exists.
    # - Multi-node next (e.g., Qwen3-235B TP=16) while 8-GPU nodes are still mostly free.
    # - Dense models last (0.6B/4B) to avoid consuming 1 GPU on every free node.
    infer_models = [m for _, m, _ in ordered]
    infer_moe_single = [m for m in infer_models if _infer_is_moe(m) and _infer_gpus(m) <= 8]
    infer_multi = [m for m in infer_models if _infer_gpus(m) > 8]
    infer_multi.sort(key=lambda m: (-_infer_gpus(m), m))
    infer_dense = [m for m in infer_models if not _infer_is_moe(m) and _infer_gpus(m) <= 8]
    infer_moe_single.sort(key=lambda m: (-_infer_gpus(m), m))
    infer_dense.sort(key=lambda m: (-_infer_gpus(m), m))

    # For 2-node clusters (e.g., 16 GPUs as 2x 8-GPU nodes), avoid fragmenting nodes by
    # pinning dense models to a separate node from single-node MoE models.
    try:
        gpu_node_ips = sorted(
            {
                str(n.get("NodeManagerAddress") or "").strip()
                for n in ray.nodes()
                if n.get("Alive")
                and float((n.get("Resources") or {}).get("GPU", 0) or 0) > 0
                and str(n.get("NodeManagerAddress") or "").strip()
            }
        )
        if len(gpu_node_ips) == 2:
            moe_ip, dense_ip = gpu_node_ips[0], gpu_node_ips[1]
            for m in infer_moe_single:
                pinned_vllm_node_ip.setdefault(m, moe_ip)
            for m in infer_dense:
                pinned_vllm_node_ip.setdefault(m, dense_ip)
    except Exception:
        pass

    if pinned_vllm_node_ip:
        os.environ["MINT_VLLM_PINNED_NODE_IP_JSON"] = json.dumps(pinned_vllm_node_ip)
        logger.info(f"[prewarm] MINT_VLLM_PINNED_NODE_IP_JSON={os.environ['MINT_VLLM_PINNED_NODE_IP_JSON']}")

    timeout_s = float(os.environ.get("MINT_PERSISTENT_INFER_TIMEOUT_S", "1800"))

    for model_name in infer_moe_single + infer_multi + infer_dense:
        try:
            logger.info(f"[prewarm] inference create start model={model_name} timeout_s={timeout_s}")
            engine = await asyncio.wait_for(multi_model_manager.get_engine(model_name), timeout=timeout_s)
            actor_name = getattr(engine, "actor_name", None)
            if not actor_name:
                raise RuntimeError("engine has no actor_name")
            ok = resource_pool.set_protected(actor_name, True)
            if not ok:
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    ok = resource_pool.set_protected(actor_name, True)
                    if ok:
                        break
            if ok:
                logger.info(f"[prewarm] inference ready+protected model={model_name} actor={actor_name}")
            else:
                logger.warning(f"[prewarm] inference ready (but not in ResourcePool) model={model_name} actor={actor_name}")
        except SystemExit as e:
            if getattr(e, "code", None) == 15:
                raise
            _record_failure("inference", model_name, e)
            logger.exception(f"[prewarm] inference SystemExit model={model_name}: {e}")
        except Exception as e:
            _record_failure("inference", model_name, e)
            logger.exception(f"[prewarm] inference failed model={model_name}: {e}")

    # -------------------------
    # Dense training pools (deferred)
    # -------------------------
    if prewarm_training and deferred_dense_training:
        from tinker_server.backend.dense_trainer import get_or_create_dense_trainer
        from tinker_server.backend.verl_training import TrainingWorker

        for model_name, base_model in deferred_dense_training:
            try:
                logger.info(f"[prewarm] training create start model={model_name} backend=peft_trainer")
                dense = await asyncio.to_thread(
                    get_or_create_dense_trainer,
                    training_worker_cls=TrainingWorker,
                    base_model=base_model,
                    model_key=model_name,
                    lora_rank=lora_rank,
                    learning_rate=learning_rate,
                    session_id=None,
                )
                actor_name = dense.actor_name
                resource_pool.set_protected(actor_name, True)
                logger.info(f"[prewarm] training ready+protected model={model_name} actor={actor_name}")
            except Exception as e:
                _record_failure("training", model_name, e)
                logger.exception(f"[prewarm] training failed model={model_name} backend=peft_trainer: {e}")

    _raise_if_failures()


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
    # Ray: hard requirement (fail fast)
    # ==========================================================================
    clear_startup_degraded_state()
    clear_runtime_degraded_state()
    from .backend.future_store import future_store
    from .backend.gateway_session_store import ensure_ready as ensure_gateway_session_store_ready
    from .backend.owner_runtime_supervisor import owner_runtime_supervisor
    from .backend.sampling_session_store import ensure_ready as ensure_sampling_session_store_ready
    from .backend.session_heartbeat_store import session_heartbeat_store
    from .backend.session_index_store import ensure_ready as ensure_session_index_store_ready
    from .backend.startup_lease import acquire_startup_lease
    from .backend.training_session_store import ensure_ready as ensure_training_session_store_ready

    future_store.ensure_ready()
    ensure_gateway_session_store_ready()
    ensure_sampling_session_store_ready()
    session_heartbeat_store.ensure_ready()
    ensure_session_index_store_ready()
    ensure_training_session_store_ready()

    try:
        from .backend.dense_session_state import cleanup_legacy_dense_session_state_once
        from .backend.training_session_store import list_training_sessions

        active_model_ids = {
            str(info.get("model_id"))
            for info in await asyncio.to_thread(list_training_sessions)
            if isinstance(info, dict) and str(info.get("model_id") or "").strip()
        }
        dense_cleanup = await asyncio.to_thread(
            cleanup_legacy_dense_session_state_once,
            active_session_ids=active_model_ids,
        )
        migrated = len(dense_cleanup.get("migrated", []))
        deleted = len(dense_cleanup.get("deleted", []))
        skipped = len(dense_cleanup.get("skipped", []))
        errors = dense_cleanup.get("errors", [])
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
    owner_runtime = await owner_runtime_supervisor.async_ensure_started()
    logger.info(
        "owner runtime supervisor ready actor=%s epoch=%s",
        owner_runtime.get("actor_name"),
        owner_runtime.get("epoch_id"),
    )

    def _owner_runtime_health_error(snapshot: dict[str, object]) -> tuple[str, str, dict[str, object]] | None:
        code_identity = snapshot.get("code_identity")
        if code_identity != app_module_git_sha:
            return (
                "owner_runtime_supervisor_code_mismatch",
                f"expected code_identity={app_module_git_sha!r} actual={code_identity!r}",
                {"snapshot": snapshot},
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
                        "owner_runtime_supervisor_loop_error",
                        f"loop={loop_name} last_error={last_error}",
                        {"snapshot": snapshot},
                    )
        return None

    async def _owner_runtime_health_loop() -> None:
        while True:
            try:
                snapshot = await owner_runtime_supervisor.async_health_snapshot(timeout_s=10.0)
                err = _owner_runtime_health_error(snapshot)
                if err is None:
                    clear_runtime_degraded_state()
                else:
                    reason, error, details = err
                    set_runtime_degraded_state(reason=reason, error=error, details=details)
            except Exception as e:
                set_runtime_degraded_state(
                    reason="owner_runtime_supervisor_unavailable",
                    error=f"{type(e).__name__}: {e}",
                    details={},
                )
            await asyncio.sleep(5.0)

    owner_runtime_health_task = asyncio.create_task(_owner_runtime_health_loop())

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

    inference_manager = None
    train_manager = None
    multi_model_manager = None
    stale_training_heartbeat_task = None

    try:
        # ==========================================================================
        # Cleanup: Kill stale actors from previous server runs
        # ==========================================================================
        if startup_owner:
            await owner_runtime_supervisor.async_run_once("actor_reconciliation", timeout_s=60.0)
        else:
            logger.info("Skipping stale-actor cleanup on follower worker")

        # ==========================================================================
        # Inference route layer: stateless API path uses detached stores only
        # ==========================================================================
        inference_manager = None
        service.session_manager = None
        sampling.session_manager = None
        multi_model_manager: MultiModelInferenceManager | None = None

        # ==========================================================================
        # Training: Initialize TrainingSessionManager and VerlTrainingEngine
        # ==========================================================================
        logger.info("Initializing training components")

        from .backend.training_session_manager import TrainingSessionManager
        from .backend.verl_training import VerlTrainingEngine

        train_manager = TrainingSessionManager(
            inactivity_timeout=config.training_inactivity_timeout_s,
        )
        train_engine = VerlTrainingEngine()
        await train_engine.initialize()

        # Make training components available to routes
        training.training_manager = train_manager
        training.training_engine = train_engine
        training.inference_manager = None  # Queue execution runtime owns inference-side execution state
        mint.training_manager = train_manager
        mint.training_engine = train_engine

        # Weights router also needs training components and inference manager
        weights.training_manager = train_manager
        weights.training_engine = train_engine
        weights.inference_manager = None  # Queue execution runtime owns inference-side execution state

        logger.info("Training components initialized")

        # ==========================================================================
        # Persistent actors: pre-create and protect at startup
        # ==========================================================================
        if startup_owner:
            await _prewarm_persistent_models(train_engine, multi_model_manager)
        else:
            logger.info("Skipping persistent prewarm on follower worker")

        # ==========================================================================
        # OpenAI compat: preload tokenizers so request paths stay non-blocking
        # ==========================================================================
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

        # ==========================================================================
        # Issue #84: Admission control + API work queue workers + future reaper
        # ==========================================================================
        from .backend.api_work_queue import api_work_queue
        from .backend.capacity_manager import capacity_manager
        from .backend.queue_execution_runtime import queue_execution_runtime

        capacity_manager.ensure_ready()
        api_work_queue.ensure_ready()
        await queue_execution_runtime.async_ensure_started(num_workers=int(config.api_work_queue_num_workers))

        stale_training_heartbeat_task = None

    except Exception:
        await _cancel_task(startup_lease_task)
        await startup_lease.release()
        if train_manager is not None:
            await _shutdown_local_training_runtime(train_manager)
        if inference_manager is not None:
            await _shutdown_local_inference_runtime(inference_manager)
        if multi_model_manager is not None:
            await multi_model_manager.shutdown_all()
        raise

    yield

    # ==========================================================================
    # Shutdown
    # ==========================================================================
    await _cancel_task(owner_runtime_health_task)
    await _cancel_task(stale_training_heartbeat_task)
    await _cancel_task(startup_lease_task)
    await api_work_queue.shutdown()
    await startup_lease.release()
    logger.info("Shutting down local runtime state")

    # Do not let an arbitrary API worker exit delete shared metadata or global actors.
    await _shutdown_local_training_runtime(train_manager)
    await _shutdown_local_inference_runtime(inference_manager)

    # Shutdown multi-model inference manager
    if multi_model_manager is not None:
        await multi_model_manager.shutdown_all()
        logger.info("Multi-model inference manager shutdown")

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

@app.exception_handler(FutureStoreUnavailableError)
async def future_store_unavailable_handler(_: Request, __: FutureStoreUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Ray unavailable: FutureStore requires Ray"},
    )


@app.exception_handler(ApiWorkQueueUnavailableError)
async def api_work_queue_unavailable_handler(_: Request, __: ApiWorkQueueUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Ray unavailable: ApiWorkQueue requires Ray"},
    )


@app.exception_handler(CapacityManagerUnavailableError)
async def capacity_manager_unavailable_handler(_: Request, __: CapacityManagerUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Ray unavailable: CapacityManager requires Ray"},
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

        # Legacy auth disabled => dev mode pass-through.
        if not config.auth_enabled:
            with bind_request_trace_context(trace_id=trace_id):
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
app.include_router(sampling.router, prefix="/api/v1", tags=["sampling"])
app.include_router(futures.router, prefix="/api/v1", tags=["futures"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(weights.router, prefix="/api/v1", tags=["weights"])
app.include_router(mint.router, prefix="/api/v1/mint", tags=["mint"])
app.include_router(openai_compat.router, prefix="/oai/api/v1", tags=["openai-compat"])
app.include_router(internal.router, prefix="/internal", tags=["internal"])

@app.get("/")
async def root():
    return {"status": "ready", "healthz": "/api/v1/healthz"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
