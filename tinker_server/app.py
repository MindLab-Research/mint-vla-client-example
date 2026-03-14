"""FastAPI application for tinker-server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .backend.api_work_queue import ApiWorkQueueUnavailableError
from .backend.capacity_manager import CapacityManagerUnavailableError
from .backend.future_store import FutureStoreUnavailableError
from .backend.session_manager import DEFAULT_INACTIVITY_TIMEOUT, SessionManager
from .config import config
from .gateway import close_http_clients
from .health_state import clear_startup_degraded_state, set_startup_degraded_state
from .gateway_auth import extract_gateway_auth_context, has_gateway_auth_headers
from .logging_context import (
    classify_failure_reason,
    ensure_trace_id,
    extract_trace_id_from_traceparent,
    get_trace_id,
    get_otel_tracer,
    record_http_server_metrics,
    run_async_with_otel_span,
    set_trace_id,
)
from .ray_utils import init_ray
from .routes import futures, internal, sampling, service, training, weights
from .token_encryptor import TokenEncryptor

if TYPE_CHECKING:
    from .backend.multi_lora_engine import MultiModelInferenceManager
    from .backend.training_session_manager import TrainingSessionManager
    from .backend.verl_training import VerlTrainingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _http_route_label(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return request.url.path


async def _cleanup_stale_actors() -> None:
    """Clean up stale Ray actors and register alive ones with ResourcePool.

    Detached actors survive server restarts and can block resources.
    This function:
    1. Kills dead/unresponsive actors in the configured Ray namespace
    2. Registers alive actors with ResourcePool for proper GPU tracking
    """
    # Skip cleanup if disabled (useful for debugging)
    if config.skip_actor_cleanup:
        logger.info("Skipping actor cleanup (MINT_SKIP_ACTOR_CLEANUP=1)")
        return

    try:
        import ray
        from .backend import ray_kill
        from .backend.multi_lora_engine import PERSISTENT_NAMESPACE
        from .backend.resource_pool import get_resource_pool, ActorType

        if not ray.is_initialized():
            init_ray(
                address="auto",
                namespace=PERSISTENT_NAMESPACE,
                ignore_reinit_error=True,
            )

        def _normalize_model_part(s: str) -> str:
            return s.lower().replace("-", "_").replace(".", "_")

        def _lookup_model_config(model_part: str):
            try:
                from tinker_server.backend.model_registry import MODEL_CONFIGS
            except Exception:
                return "", None

            needle = _normalize_model_part(model_part)
            for model_name, cfg in MODEL_CONFIGS.items():
                if _normalize_model_part(model_name.split("/")[-1]) == needle:
                    return model_name, cfg
            return "", None

        # Get all named actors in the configured namespace
        actors = ray.util.list_named_actors(all_namespaces=True)
        tinker_actors = [a for a in actors if a.get("namespace") == PERSISTENT_NAMESPACE]

        if not tinker_actors:
            logger.info(f"No actors found in namespace {PERSISTENT_NAMESPACE}")
            return

        logger.info(f"Found {len(tinker_actors)} actors in namespace {PERSISTENT_NAMESPACE}, checking status...")

        resource_pool = get_resource_pool()
        cleaned = 0
        registered = 0

        for actor_info in tinker_actors:
            name = actor_info["name"]
            try:
                actor = ray.get_actor(name, namespace=PERSISTENT_NAMESPACE)

                # Hard break: legacy dense trainer actor names are no longer supported.
                # Kill them proactively to avoid consuming GPUs indefinitely.
                if name.startswith("dense_trainer_pool_"):
                    try:
                        ray_kill.kill(
                            actor,
                            reason="legacy_dense_trainer_prefix",
                            actor_name=name,
                            namespace=PERSISTENT_NAMESPACE,
                            no_restart=True,
                        )
                        cleaned += 1
                    except Exception as kill_err:
                        logger.warning(f"Failed to kill legacy dense trainer actor {name}: {kill_err}")
                    try:
                        resource_pool.unregister(name)
                    except Exception:
                        pass
                    continue

                # Check if actor is alive.
                # WARNING: __ray_ready__ is a normal actor task and can time out if the actor is busy.
                # Do not treat GetTimeoutError as death; killing busy detached actors breaks in-flight work.
                try:
                    ray.get(actor.__ray_ready__.remote(), timeout=2)

                    # Actor is alive - register it with ResourcePool
                    # Determine actor type and GPU count from name/diagnostics
                    def _pg_total_gpus(actor_name: str) -> int | None:
                        try:
                            pg = ray.util.get_placement_group(f"{actor_name}_pg")
                            info = ray.util.placement_group_table(pg)
                        except Exception:
                            return None
                        bundles = info.get("bundles") or {}
                        total = sum(
                            int(b.get("GPU", 0) or 0)
                            for b in bundles.values()
                            if isinstance(b, dict)
                        )
                        return total or None

                    if name.startswith("tinker_vllm_") or name.startswith("multinode_vllm_"):
                        actor_type = ActorType.VLLM
                        base_model = ""
                        num_gpus = 1  # Fallback for unknown models
                        if name.startswith("tinker_vllm_"):
                            model_part = name[len("tinker_vllm_"):]
                        else:
                            model_part = name[len("multinode_vllm_"):]
                        model_name, cfg = _lookup_model_config(model_part)
                        if cfg is not None:
                            base_model = model_name
                            num_gpus = cfg.total_gpus
                        num_gpus = _pg_total_gpus(name) or num_gpus
                    elif name.startswith("peft_trainer_"):
                        actor_type = ActorType.DENSE
                        num_gpus = 1
                        base_model = ""
                    elif name.startswith("megatron_"):
                        # MegatronWorkerGroup actors: megatron_{model_name}
                        actor_type = ActorType.MEGATRON
                        base_model = ""
                        model_part = name[len("megatron_"):]
                        model_name, cfg = _lookup_model_config(model_part)
                        if cfg is not None:
                            base_model = model_name
                            num_gpus = cfg.train_gpus
                        else:
                            num_gpus = 8  # Fallback for unknown models

                        # Prefer real world_size when actor is responsive.
                        try:
                            diag = ray.get(actor.get_diagnostics.remote(), timeout=10)
                            num_gpus = int(diag.get("world_size", num_gpus))
                            base_model = diag.get("base_model", "") or base_model
                        except Exception:
                            pass
                        num_gpus = _pg_total_gpus(name) or num_gpus
                    else:
                        logger.debug(f"Unknown actor type for {name}, skipping registration")
                        continue

                    from tinker_server.backend.model_registry import is_persistent_model

                    resource_pool.register(
                        actor_name=name,
                        actor_type=actor_type,
                        num_gpus=num_gpus,
                        actor_handle=actor,
                        namespace=PERSISTENT_NAMESPACE,
                        base_model=base_model,
                        protected=bool(base_model and is_persistent_model(base_model)),
                    )
                    # Mark as ready since the actor passed health check
                    resource_pool.mark_ready(name)
                    registered += 1
                    logger.info(f"Registered existing actor: {name} ({actor_type.value}, {num_gpus} GPUs)")

                except ray.exceptions.RayActorError:
                    # Actor is dead
                    logger.info(f"Cleaning up dead actor: {name}")
                    try:
                        ray_kill.kill(
                            actor,
                            reason="startup_cleanup_dead_actor",
                            actor_name=name,
                            namespace=PERSISTENT_NAMESPACE,
                            no_restart=True,
                        )
                        cleaned += 1
                    except Exception as kill_err:
                        logger.warning(f"Failed to kill actor {name}: {kill_err}")
                except ray.exceptions.GetTimeoutError:
                    # Actor might be busy; do not treat a timeout as readiness.
                    # Register it as "creating" so operators can see reconciliation uncertainty.
                    logger.warning(
                        f"Actor {name} __ray_ready__ timed out; registering without marking ready"
                    )
                    try:
                        def _pg_total_gpus(actor_name: str) -> int | None:
                            try:
                                pg = ray.util.get_placement_group(f"{actor_name}_pg")
                                info = ray.util.placement_group_table(pg)
                            except Exception:
                                return None
                            bundles = info.get("bundles") or {}
                            total = sum(
                                int(b.get("GPU", 0) or 0)
                                for b in bundles.values()
                                if isinstance(b, dict)
                            )
                            return total or None

                        if name.startswith("tinker_vllm_") or name.startswith("multinode_vllm_"):
                            actor_type = ActorType.VLLM
                            num_gpus = 1
                            base_model = ""
                            if name.startswith("tinker_vllm_"):
                                model_part = name[len("tinker_vllm_"):]
                            else:
                                model_part = name[len("multinode_vllm_"):]
                            model_name, cfg = _lookup_model_config(model_part)
                            if cfg is not None:
                                base_model = model_name
                                num_gpus = cfg.total_gpus
                            num_gpus = _pg_total_gpus(name) or num_gpus
                        elif name.startswith("peft_trainer_"):
                            actor_type = ActorType.DENSE
                            num_gpus = 1
                            base_model = ""
                        elif name.startswith("megatron_"):
                            actor_type = ActorType.MEGATRON
                            base_model = ""
                            model_part = name[len("megatron_"):]
                            model_name, cfg = _lookup_model_config(model_part)
                            if cfg is not None:
                                base_model = model_name
                                num_gpus = cfg.train_gpus
                            else:
                                num_gpus = 8
                            num_gpus = _pg_total_gpus(name) or num_gpus
                        else:
                            logger.debug(f"Unknown actor type for {name}, skipping registration")
                            continue

                        from tinker_server.backend.model_registry import is_persistent_model

                        resource_pool.register(
                            actor_name=name,
                            actor_type=actor_type,
                            num_gpus=num_gpus,
                            actor_handle=actor,
                            namespace=PERSISTENT_NAMESPACE,
                            base_model=base_model,
                            protected=bool(base_model and is_persistent_model(base_model)),
                            metadata={"startup_reconcile": "__ray_ready__timeout"},
                        )
                        registered += 1
                        logger.info(
                            f"Registered busy actor (not ready): {name} ({actor_type.value}, {num_gpus} GPUs)"
                        )
                    except Exception as reg_err:
                        logger.warning(f"Failed to register busy actor {name}: {reg_err}")

            except ValueError:
                # Actor name registered but no actor exists
                logger.debug(f"Actor {name} not found (name registered but no actor)")
                try:
                    resource_pool.unregister(name)
                except Exception:
                    pass
                try:
                    pg_name = f"{name}_pg"
                    pg = ray.util.get_placement_group(pg_name)
                    ray.util.remove_placement_group(pg)
                    logger.warning(f"Removed orphan placement_group={pg_name}")
                except Exception:
                    pass

        logger.info(f"Actor cleanup complete: {cleaned} cleaned, {registered} registered")

    except Exception as e:
        # Surface reconciliation failures via degraded health rather than silently continuing.
        set_startup_degraded_state(
            reason="startup_actor_cleanup_failed",
            error=f"{type(e).__name__}: {e}",
        )
        logger.error(f"Actor cleanup failed; healthz will be degraded: {type(e).__name__}: {e}")

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
                                pg_table = ray.util.placement_group_table()
                                for info in pg_table.values():
                                    if info.get("state") != "CREATED":
                                        continue
                                    if info.get("name") != pg_name:
                                        continue
                                    node_id = (info.get("bundles_to_node_id") or {}).get(0)
                                    if node_id:
                                        break
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

                    async def _await_ready(
                        actor=actor,
                        actor_name=actor_name,
                        model_name=model_name,
                    ) -> None:
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
                            logger.warning(
                                f"[prewarm] training __ray_ready__ SystemExit model={model_name} actor={actor_name}: {ready_err}"
                            )
                        except Exception as ready_err:
                            logger.warning(
                                f"[prewarm] training __ray_ready__ failed/timeout model={model_name} actor={actor_name}: {ready_err}"
                            )

                    asyncio.create_task(_await_ready())
                else:
                    # Defer dense pool creation until after multi-node vLLM inference is initialized,
                    # to avoid fragmenting the remaining 8-GPU nodes into 1-2 free GPUs each.
                    deferred_dense_training.append((model_name, base_model))
                    logger.info(f"[prewarm] training deferred model={model_name} backend=dense_pool")
            except Exception as e:
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
        return

    if multi_model_manager is None:
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
            logger.exception(f"[prewarm] inference SystemExit model={model_name}: {e}")
        except Exception as e:
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
                logger.exception(f"[prewarm] training failed model={model_name} backend=peft_trainer: {e}")


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
    from .backend.future_store import future_store
    from .checkpoints import get_checkpoint_reap_interval_s, reap_runtime_checkpoints

    future_store.ensure_ready()

    # ==========================================================================
    # Cleanup: Kill stale actors from previous server runs
    # ==========================================================================
    await _cleanup_stale_actors()

    # ==========================================================================
    # Inference: Initialize SessionManager
    # ==========================================================================
    logger.info("Initializing inference session manager")

    inference_manager = SessionManager(
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        inactivity_timeout=config.session_inactivity_timeout_s
        if config.session_inactivity_timeout_s is not None
        else DEFAULT_INACTIVITY_TIMEOUT,
    )

    # Make session manager available to routes
    service.session_manager = inference_manager
    sampling.session_manager = inference_manager

    # Start background cleanup task
    await inference_manager.start_cleanup_task()

    logger.info("Inference session manager initialized")

    # ==========================================================================
    # Multi-Model Inference: Initialize manager for dynamic engine creation
    # ==========================================================================
    multi_model_manager: MultiModelInferenceManager | None = None

    if config.enable_multi_lora:
        from .backend.multi_lora_engine import MultiModelInferenceManager

        logger.info(
            f"Initializing Multi-Model Inference Manager: max_loras={config.max_loras}, "
            f"max_cpu_loras={config.max_cpu_loras}, max_lora_rank={config.max_lora_rank}"
        )

        # Create manager - engines are created lazily per model
        multi_model_manager = MultiModelInferenceManager(
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_loras=config.max_loras,
            max_cpu_loras=config.max_cpu_loras,
            max_lora_rank=config.max_lora_rank,
        )

        # Register with session manager
        inference_manager.set_multi_model_manager(multi_model_manager)
        logger.info("Multi-model inference manager initialized (engines created on-demand)")
    else:
        logger.info("Multi-LoRA disabled, using per-session engines")

    # ==========================================================================
    # Training: Initialize TrainingSessionManager and VerlTrainingEngine
    # ==========================================================================
    logger.info("Initializing training components")

    from .backend.training_session_manager import TrainingSessionManager
    from .backend.verl_training import VerlTrainingEngine

    train_manager = TrainingSessionManager()
    train_engine = VerlTrainingEngine()
    await train_engine.initialize()

    # Make training components available to routes
    training.training_manager = train_manager
    training.training_engine = train_engine
    training.inference_manager = inference_manager  # For ephemeral save flow

    # Weights router also needs training components and inference manager
    weights.training_manager = train_manager
    weights.training_engine = train_engine
    weights.inference_manager = inference_manager  # For multi-LoRA sampling registration

    logger.info("Training components initialized")

    # ==========================================================================
    # Issue #84: Admission control + API work queue workers + future reaper
    # ==========================================================================
    from .backend.api_work_queue import api_work_queue
    from .backend.capacity_manager import capacity_manager
    from .models.types import (
        ComputeLogprobsRequest,
        CreateModelFromStateRequest,
        CreateModelRequest,
        ForwardRequest,
        ForwardBackwardRequest,
        LoadStateRequest,
        OptimStepRequest,
        SampleRequest,
        SaveStateRequest,
        SaveWeightsForSamplerRequest,
        TrainStepRequest,
    )

    async def _exec_sampling_asample(item):
        async def _run():
            logger.info(
                "[api_work_queue] sampling.asample request_id=%s stage=before_model_validate",
                str(item.request_id),
            )
            req = SampleRequest.model_validate_json(item.request_json)
            logger.info(
                "[api_work_queue] sampling.asample request_id=%s stage=after_model_validate",
                str(item.request_id),
            )
            await sampling._do_sample(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        await run_async_with_otel_span(
            "queue.stage.sampling.asample",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.sampling.asample"},
        )

    async def _exec_sampling_compute_logprobs(item):
        async def _run():
            req = ComputeLogprobsRequest.model_validate_json(item.request_json)
            await sampling._do_compute_logprobs(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        await run_async_with_otel_span(
            "queue.stage.sampling.compute_logprobs",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.sampling.compute_logprobs"},
        )

    async def _exec_training_create_model(item):
        async def _run():
            req = CreateModelRequest.model_validate_json(item.request_json)
            await training._do_create_model(item.request_id, req, item.user_id, item.webhook_url)

        await run_async_with_otel_span(
            "queue.stage.training.create_model",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.create_model"},
        )

    async def _exec_training_create_model_from_state(item):
        async def _run():
            req = CreateModelFromStateRequest.model_validate_json(item.request_json)
            await training._do_create_model_from_state(item.request_id, req, item.user_id)

        await run_async_with_otel_span(
            "queue.stage.training.create_model_from_state",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.create_model_from_state"},
        )

    async def _exec_training_train_step(item):
        async def _run():
            req = TrainStepRequest.model_validate_json(item.request_json)
            await training._do_train_step(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        await run_async_with_otel_span(
            "queue.stage.training.train_step",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.train_step"},
        )

    async def _exec_training_forward(item):
        async def _run():
            req = ForwardRequest.model_validate_json(item.request_json)
            await training._do_forward(
                item.request_id,
                req,
                (item.extra or {}).get("gateway_auth"),
            )

        await run_async_with_otel_span(
            "queue.stage.training.forward",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.forward"},
        )

    async def _exec_training_forward_backward(item):
        async def _run():
            req = ForwardBackwardRequest.model_validate_json(item.request_json)
            await training._do_forward_backward(
                item.request_id,
                req,
                item.user_id,
                (item.extra or {}).get("gateway_auth"),
            )

        await run_async_with_otel_span(
            "queue.stage.training.forward_backward",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.forward_backward"},
        )

    async def _exec_training_save_weights_for_sampler(item):
        async def _run():
            req = SaveWeightsForSamplerRequest.model_validate_json(item.request_json)
            prefer_tinker = bool((item.extra or {}).get("prefer_tinker"))
            is_admin = bool((item.extra or {}).get("is_admin"))
            await training._do_save_weights_for_sampler(
                item.request_id,
                req,
                item.user_id,
                prefer_tinker,
                is_admin,
            )

        await run_async_with_otel_span(
            "queue.stage.training.save_weights_for_sampler",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.save_weights_for_sampler"},
        )

    async def _exec_training_optim_step(item):
        async def _run():
            req = OptimStepRequest.model_validate_json(item.request_json)
            await training._do_optim_step(item.request_id, req, item.user_id)

        await run_async_with_otel_span(
            "queue.stage.training.optim_step",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.training.optim_step"},
        )

    async def _exec_weights_save_weights(item):
        async def _run():
            req = SaveStateRequest.model_validate_json(item.request_json)
            prefer_tinker = bool((item.extra or {}).get("prefer_tinker"))
            # Tinker SDK calls POST /api/v1/save_weights for TrainingClient.save_state(...).
            # This must produce a training checkpoint (weights + optimizer state).
            await weights._do_save_state(
                item.request_id,
                req,
                user_id=item.user_id,
                webhook_url=item.webhook_url,
                prefer_tinker=prefer_tinker,
            )

        await run_async_with_otel_span(
            "queue.stage.weights.save_weights",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.weights.save_weights"},
        )

    async def _exec_weights_save_state(item):
        async def _run():
            req = SaveStateRequest.model_validate_json(item.request_json)
            prefer_tinker = bool((item.extra or {}).get("prefer_tinker"))
            await weights._do_save_state(
                item.request_id,
                req,
                user_id=item.user_id,
                webhook_url=item.webhook_url,
                prefer_tinker=prefer_tinker,
            )

        await run_async_with_otel_span(
            "queue.stage.weights.save_state",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.weights.save_state"},
        )

    async def _exec_weights_load_state(item):
        async def _run():
            req = LoadStateRequest.model_validate_json(item.request_json)
            await weights._do_load_state(item.request_id, req, item.user_id)

        await run_async_with_otel_span(
            "queue.stage.weights.load_state",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.weights.load_state"},
        )

    async def _exec_internal_noop(item):
        async def _run():
            from .backend.future_store import future_store

            future_store.resolve(
                str(item.request_id),
                {"ok": True, "op": "internal.noop", "ts": time.time()},
            )

        await run_async_with_otel_span(
            "queue.stage.internal.noop",
            _run,
            component="api_work_queue",
            op=str(item.op),
            request_id=str(item.request_id),
            attributes={"queue.stage": "queue.stage.internal.noop"},
        )

    api_work_queue.set_executor("sampling.asample", _exec_sampling_asample)
    api_work_queue.set_executor("sampling.compute_logprobs", _exec_sampling_compute_logprobs)
    api_work_queue.set_executor("training.create_model", _exec_training_create_model)
    api_work_queue.set_executor("training.create_model_from_state", _exec_training_create_model_from_state)
    api_work_queue.set_executor("training.train_step", _exec_training_train_step)
    api_work_queue.set_executor("training.forward", _exec_training_forward)
    api_work_queue.set_executor("training.forward_backward", _exec_training_forward_backward)
    api_work_queue.set_executor("training.save_weights_for_sampler", _exec_training_save_weights_for_sampler)
    api_work_queue.set_executor("training.optim_step", _exec_training_optim_step)
    api_work_queue.set_executor("weights.save_weights", _exec_weights_save_weights)
    api_work_queue.set_executor("weights.save_state", _exec_weights_save_state)
    api_work_queue.set_executor("weights.load_state", _exec_weights_load_state)
    api_work_queue.set_executor("internal.noop", _exec_internal_noop)

    await api_work_queue.start_workers(num_workers=int(config.api_work_queue_num_workers))

    async def _future_reaper_loop() -> None:
        while True:
            await asyncio.sleep(float(config.api_work_queue_reap_interval_s))
            try:
                reaped = future_store.reap()
                for rid in list(reaped.get("expired", [])) + list(reaped.get("timed_out", [])):
                    capacity_manager.release_all(str(rid))
            except Exception:
                pass

    future_reaper_task = asyncio.create_task(_future_reaper_loop())

    async def _checkpoint_reaper_loop() -> None:
        while True:
            await asyncio.sleep(float(get_checkpoint_reap_interval_s()))
            try:
                reaped = reap_runtime_checkpoints()
                total = len(reaped["ephemeral"]) + len(reaped["persistent_cache"]) + len(reaped["persistent"])
                if total:
                    logger.info(
                        "checkpoint reaper removed ephemeral=%s persistent_cache=%s persistent=%s",
                        len(reaped["ephemeral"]),
                        len(reaped["persistent_cache"]),
                        len(reaped["persistent"]),
                    )
            except Exception:
                logger.exception("checkpoint reaper failed")

    checkpoint_reaper_task = asyncio.create_task(_checkpoint_reaper_loop())

    # ==========================================================================
    # Persistent actors: pre-create and protect at startup
    # ==========================================================================
    asyncio.create_task(_prewarm_persistent_models(train_engine, multi_model_manager))

    yield

    # ==========================================================================
    # Shutdown
    # ==========================================================================
    future_reaper_task.cancel()
    checkpoint_reaper_task.cancel()
    await asyncio.gather(future_reaper_task, checkpoint_reaper_task, return_exceptions=True)
    await api_work_queue.shutdown()
    logger.info("Shutting down all sessions")

    # Shutdown training sessions
    await train_manager.shutdown_all(train_engine)

    # Shutdown inference sessions
    await inference_manager.shutdown_all()

    # Shutdown multi-model inference manager
    if multi_model_manager is not None:
        await multi_model_manager.shutdown_all()
        logger.info("Multi-model inference manager shutdown")

    from .usage_store import close_usage_store

    await close_usage_store()

    from .gateway import close_http_clients

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

    def _log_request_observation(elapsed_ms: float) -> None:
        if status_code >= 500:
            reason = classify_failure_reason(failure_error or RuntimeError(f"http_{status_code}"))
            logger.error(
                "[http.request] failed method=%s route=%s status_code=%s elapsed_ms=%.3f failure_reason=%s error_type=%s next_action=%s",
                method,
                route,
                int(status_code),
                float(elapsed_ms),
                reason,
                type(failure_error).__name__ if failure_error is not None else "HTTPStatusError",
                "check_logs_and_trace",
            )
            return
        if status_code >= 400:
            logger.warning(
                "[http.request] client_error method=%s route=%s status_code=%s elapsed_ms=%.3f",
                method,
                route,
                int(status_code),
                float(elapsed_ms),
            )
            return
        logger.info(
            "[http.request] completed method=%s route=%s status_code=%s elapsed_ms=%.3f",
            method,
            route,
            int(status_code),
            float(elapsed_ms),
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
            }
            return await _next_with_trace()

        # Legacy auth disabled => dev mode pass-through.
        if not config.auth_enabled:
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
                    return await _next_with_trace()
        return _with_trace(JSONResponse(status_code=401, content={"error": "Invalid API key or token"}))
    return await _next_with_trace()


# Register routes with API prefix
app.include_router(service.router, prefix="/api/v1", tags=["service"])
app.include_router(sampling.router, prefix="/api/v1", tags=["sampling"])
app.include_router(futures.router, prefix="/api/v1", tags=["futures"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(weights.router, prefix="/api/v1", tags=["weights"])
app.include_router(internal.router, prefix="/internal", tags=["internal"])

@app.get("/")
async def root():
    return {"status": "ready", "healthz": "/api/v1/healthz"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)
