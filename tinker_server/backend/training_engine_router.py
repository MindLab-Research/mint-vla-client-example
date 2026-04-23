from __future__ import annotations

from typing import Any

from .model_registry import get_model_config
from .openpi_fast_training import OpenPIFastTrainingEngine, OPENPI_FAST_TRAINING_BACKEND
from .openpi_pi05_training import (
    OpenPIPi05TrainingEngine,
    OPENPI_PI05_TRAINING_BACKEND,
)


class TrainingEngineRouter:
    def __init__(
        self,
        *,
        text_engine: Any | None = None,
        openpi_fast_engine: Any | None = None,
        openpi_pi05_engine: Any | None = None,
    ) -> None:
        if text_engine is None:
            from .verl_training import VerlTrainingEngine

            text_engine = VerlTrainingEngine()
        self._text_engine = text_engine
        self._openpi_fast_engine = (
            openpi_fast_engine if openpi_fast_engine is not None else OpenPIFastTrainingEngine()
        )
        self._openpi_pi05_engine = (
            openpi_pi05_engine
            if openpi_pi05_engine is not None
            else OpenPIPi05TrainingEngine()
        )

    async def initialize(self) -> None:
        await self._text_engine.initialize()
        # Keep OpenPI startup on the request path. Sampling-only app startup should
        # not require Ray/OpenPI env just because the router is present.

    def _resolve_hf_model_path(self, model_name: str) -> str | None:
        resolver = getattr(self._text_engine, "_resolve_hf_model_path", None)
        if not callable(resolver):
            raise AttributeError("text training engine does not expose _resolve_hf_model_path")
        return resolver(model_name)

    def _engine_for_base_model(self, base_model: str) -> Any:
        training_backend = get_model_config(base_model).training_backend
        if training_backend == OPENPI_FAST_TRAINING_BACKEND:
            return self._openpi_fast_engine
        if training_backend == OPENPI_PI05_TRAINING_BACKEND:
            return self._openpi_pi05_engine
        return self._text_engine

    def _engine_for_session(self, session: Any) -> Any:
        return self._engine_for_base_model(session.base_model)

    async def create_training_session(self, session: Any) -> Any:
        return await self._engine_for_session(session).create_training_session(session)

    async def forward_backward(self, session: Any, request: Any) -> Any:
        return await self._engine_for_session(session).forward_backward(session, request)

    async def forward(self, session: Any, request: Any) -> Any:
        return await self._engine_for_session(session).forward(session, request)

    async def forward_backward_reverse_kl(self, session: Any, request: Any) -> Any:
        return await self._text_engine.forward_backward_reverse_kl(session, request)

    async def get_tokenizer_info(self, session: Any) -> Any:
        return await self._engine_for_session(session).get_tokenizer_info(session)

    async def optim_step(self, session: Any, request: Any) -> Any:
        return await self._engine_for_session(session).optim_step(session, request)

    async def train_step(self, session: Any, request: Any) -> Any:
        return await self._engine_for_session(session).train_step(session, request)

    async def reset_expert_bias(self, session: Any) -> Any:
        return await self._engine_for_session(session).reset_expert_bias(session)

    async def save_weights_for_sampler(
        self,
        *,
        session: Any,
        checkpoint_name: str,
        checkpoint_base_dir: str,
    ) -> Any:
        return await self._engine_for_session(session).save_weights_for_sampler(
            session=session,
            checkpoint_name=checkpoint_name,
            checkpoint_base_dir=checkpoint_base_dir,
        )

    async def save_weights(
        self,
        session: Any,
        save_path: str,
    ) -> Any:
        return await self._engine_for_session(session).save_weights(
            session,
            save_path,
        )

    async def load_weights(
        self,
        session: Any,
        load_path: str,
        load_optimizer: bool = True,
    ) -> Any:
        return await self._engine_for_session(session).load_weights(
            session,
            load_path,
            load_optimizer=load_optimizer,
        )

    async def shutdown_session(self, session: Any) -> Any:
        return await self._engine_for_session(session).shutdown_session(session)
