from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from ..config import config
from .async_ray_control import async_get_ray_ref

if TYPE_CHECKING:
    from .multi_lora_engine import MultiModelInferenceManager
    from .verl_training import VerlTrainingEngine

logger = logging.getLogger(__name__)


async def prewarm_persistent_models(
    train_engine: VerlTrainingEngine | None,
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

    and marks them as ModelActorSupervisorInventory protected to prevent LRU eviction.
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
    prewarm_training_requested = bool(config.prewarm_enable_training)
    prewarm_training = prewarm_training_requested and train_engine is not None
    prewarm_inference = bool(config.prewarm_enable_inference)
    if prewarm_training_requested and train_engine is None:
        raise RuntimeError(
            "persistent prewarm training configured but unavailable in the execution runtime"
        )

    from tinker_server.backend.model_registry import (
        get_model_config,
        get_training_parallelism,
        normalize_model_name,
        requires_fp8,
    )
    from tinker_server.backend.model_actor_supervisor import get_model_actor_supervisor

    model_actor_supervisor_inventory = get_model_actor_supervisor()

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
                        observability_base_model=model_name,
                    )
                    actor_name = _make_megatron_actor_name(base_model or model_name)
                    # Protect as soon as the actor is registered, so readiness timeouts don't leave it evictable.
                    model_actor_supervisor_inventory.set_protected(actor_name, True)
                    logger.info(f"[prewarm] training __ray_ready__ scheduled model={model_name} actor={actor_name}")

                    try:
                        await async_get_ray_ref(actor.__ray_ready__.remote(), timeout_s=megatron_ready_timeout_s)
                        model_actor_supervisor_inventory.mark_ready(actor_name)
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

    timeout_s = float(os.environ.get("MINT_PERSISTENT_INFER_TIMEOUT_S", "1800"))

    for model_name in infer_moe_single + infer_multi + infer_dense:
        try:
            logger.info(f"[prewarm] inference create start model={model_name} timeout_s={timeout_s}")
            engine = await asyncio.wait_for(multi_model_manager.get_engine(model_name), timeout=timeout_s)
            actor_name = getattr(engine, "actor_name", None)
            if not actor_name:
                raise RuntimeError("engine has no actor_name")
            ok = model_actor_supervisor_inventory.set_protected(actor_name, True)
            if not ok:
                for _ in range(50):
                    await asyncio.sleep(0.1)
                    ok = model_actor_supervisor_inventory.set_protected(actor_name, True)
                    if ok:
                        break
            if ok:
                logger.info(f"[prewarm] inference ready+protected model={model_name} actor={actor_name}")
            else:
                logger.warning(f"[prewarm] inference ready (but not in ModelActorSupervisorInventory) model={model_name} actor={actor_name}")
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
                model_actor_supervisor_inventory.set_protected(actor_name, True)
                logger.info(f"[prewarm] training ready+protected model={model_name} actor={actor_name}")
            except Exception as e:
                _record_failure("training", model_name, e)
                logger.exception(f"[prewarm] training failed model={model_name} backend=peft_trainer: {e}")

    _raise_if_failures()
