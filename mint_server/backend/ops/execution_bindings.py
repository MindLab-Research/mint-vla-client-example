from __future__ import annotations

import structlog
from typing import Any

logger = structlog.get_logger(__name__)


async def _restore_sampling_sessions_for_worker(inference_manager: Any) -> int:
    from mint_server.backend.stores.sampling_session_store import async_list_sampling_sessions

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
    from mint_server.config import config
    from mint_server.backend.openpi.action_session_manager import ActionSessionRouter
    from mint_server.backend.core.execution_context import ExecutionContext
    from mint_server.backend.sessions.session_manager import DEFAULT_INACTIVITY_TIMEOUT, SessionManager
    from mint_server.backend.training.training_engine_router import TrainingEngineRouter
    from mint_server.backend.training.training_session_manager import TrainingSessionManager

    inference_manager = SessionManager(
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        inactivity_timeout=config.session_inactivity_timeout_s
        if config.session_inactivity_timeout_s is not None
        else DEFAULT_INACTIVITY_TIMEOUT,
    )

    restored_sampling_sessions = 0
    try:
        restored_sampling_sessions = await _restore_sampling_sessions_for_worker(inference_manager)
    except Exception as e:
        logger.warning(
            "execution bindings sampling-session restore skipped: %s: %s",
            type(e).__name__,
            e,
        )

    multi_model_manager = None
    if config.enable_multi_lora:
        from mint_server.backend.inference.multi_lora_engine import MultiModelInferenceManager

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

    return ExecutionContext(
        inference_manager=inference_manager,
        train_manager=train_manager,
        train_engine=train_engine,
        action_manager=action_manager,
        multi_model_manager=multi_model_manager,
        restored_sampling_sessions=int(restored_sampling_sessions),
        multi_model_enabled=bool(config.enable_multi_lora),
    ).as_dict()
