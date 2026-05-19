from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


async def _restore_sampling_sessions_for_worker(inference_manager: Any) -> int:
    from .sampling_session_store import async_list_sampling_sessions

    restored = 0
    for info in await async_list_sampling_sessions():
        try:
            if inference_manager.restore_sampling_session(info):
                restored += 1
        except Exception as e:
            logger.warning(
                "execution bindings sampling-session restore skipped session=%r error_type=%s error=%s",
                info.get("session_id") if isinstance(info, dict) else None,
                type(e).__name__,
                e,
            )
    return restored


async def initialize_execution_bindings() -> dict[str, Any]:
    from ..config import config
    from ..routes import action_sampling, sampling, service, training, weights
    from .action_session_manager import ActionSessionRouter
    from .sampling_session_store import ensure_ready as ensure_sampling_session_store_ready
    from .session_manager import DEFAULT_INACTIVITY_TIMEOUT, SessionManager
    from .training_engine_router import TrainingEngineRouter
    from .training_session_manager import TrainingSessionManager

    disable_mint_route = os.environ.get("MINT_DISABLE_MINT_ROUTE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mint = None
    if not disable_mint_route:
        from ..routes import mint

    inference_manager = SessionManager(
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        inactivity_timeout=config.session_inactivity_timeout_s
        if config.session_inactivity_timeout_s is not None
        else DEFAULT_INACTIVITY_TIMEOUT,
    )
    service.session_manager = inference_manager
    sampling.session_manager = inference_manager

    restored_sampling_sessions = 0
    try:
        await asyncio.to_thread(ensure_sampling_session_store_ready)
        restored_sampling_sessions = await _restore_sampling_sessions_for_worker(inference_manager)
    except Exception as e:
        logger.warning(
            "execution bindings sampling-session restore skipped: %s: %s",
            type(e).__name__,
            e,
        )

    multi_model_manager = None
    if config.enable_multi_lora:
        from .multi_lora_engine import MultiModelInferenceManager

        multi_model_manager = MultiModelInferenceManager(
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_loras=config.max_loras,
            max_cpu_loras=config.max_cpu_loras,
            max_lora_rank=config.max_lora_rank,
        )
        inference_manager.set_multi_model_manager(multi_model_manager)

    train_manager = TrainingSessionManager(
        inactivity_timeout=config.training_inactivity_timeout_s,
    )
    train_engine = TrainingEngineRouter()
    action_manager = ActionSessionRouter()
    await train_engine.initialize()

    action_sampling.action_session_manager = action_manager
    training.training_manager = train_manager
    training.training_engine = train_engine
    training.inference_manager = inference_manager
    if mint is not None:
        mint.training_manager = train_manager
        mint.training_engine = train_engine
        mint.action_session_manager = action_manager
    weights.training_manager = train_manager
    weights.training_engine = train_engine
    weights.inference_manager = inference_manager

    return {
        "inference_manager": inference_manager,
        "train_manager": train_manager,
        "train_engine": train_engine,
        "multi_model_manager": multi_model_manager,
        "restored_sampling_sessions": int(restored_sampling_sessions),
        "multi_model_enabled": bool(config.enable_multi_lora),
    }
