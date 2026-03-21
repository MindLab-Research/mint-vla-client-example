from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from ..checkpoints import get_checkpoints_dir, resolve_checkpoint_uri
from ..models.types import ActRequest, ModelInput, TensorData
from .model_registry import get_model_config
from .openpi_fast_action_runtime import OpenPIFastActionWorkerClient
from .openpi_fast_training import (
    OPENPI_FAST_TRAINING_BACKEND,
    get_openpi_fast_config_name,
)


def _is_openpi_fast_model(base_model: str) -> bool:
    try:
        return get_model_config(base_model).training_backend == OPENPI_FAST_TRAINING_BACKEND
    except Exception:
        return False


async def _default_runtime_factory(
    *,
    action_session_id: str,
    base_model: str,
    checkpoint_path: str,
    model_config: Any,
    config_name: str,
) -> Any:
    del action_session_id, base_model, checkpoint_path, model_config, config_name
    return await OpenPIFastActionWorkerClient.start()


class OpenPIFastActionSessionManager:
    def __init__(
        self,
        *,
        runtime_factory: Callable[..., Awaitable[Any]] | None = None,
        checkpoints_dir: str | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._checkpoints_dir = checkpoints_dir or get_checkpoints_dir()
        self._runtime_clients: dict[str, Any] = {}

    def _action_session_id(self, session_id: str, action_session_seq_id: int | None) -> str:
        if action_session_seq_id is None:
            return str(uuid.uuid4())
        return f"{session_id}:action:{int(action_session_seq_id)}"

    def _resolve_model_path(self, model_path: str, user_id: str | None) -> str:
        resolved = resolve_checkpoint_uri(model_path, self._checkpoints_dir, user_id=user_id)
        if not resolved:
            raise FileNotFoundError(f"Action checkpoint path did not resolve: {model_path}")
        return resolved

    async def create_session(
        self,
        *,
        session_id: str,
        action_session_seq_id: int | None,
        base_model: str,
        model_path: str | None,
        user_id: str | None,
    ) -> str:
        if not _is_openpi_fast_model(base_model):
            raise ValueError(f"OpenPI FAST action inference does not support {base_model!r}")
        if not model_path:
            raise ValueError("OpenPI FAST action inference requires model_path")

        model_config = get_model_config(base_model)
        checkpoint_path = self._resolve_model_path(model_path, user_id)
        action_session_id = self._action_session_id(session_id, action_session_seq_id)
        config_name = get_openpi_fast_config_name(base_model)
        client = await self._runtime_factory(
            action_session_id=action_session_id,
            base_model=base_model,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            config_name=config_name,
        )
        try:
            await client.request(
                "create_session",
                {
                    "action_session_id": action_session_id,
                    "base_model": base_model,
                    "checkpoint_path": checkpoint_path,
                    "config_name": config_name,
                    "action_dim": int(model_config.action_dim or 0),
                    "action_horizon": int(model_config.action_horizon or 0),
                    "max_token_len": int(model_config.max_model_len),
                    "camera_layout": list(model_config.camera_layout),
                },
            )
        except Exception:
            close = getattr(client, "close", None)
            if callable(close):
                await close()
            raise

        self._runtime_clients[action_session_id] = client
        return action_session_id

    async def act(
        self,
        *,
        action_session_id: str,
        observation: ModelInput,
        extra_inputs: dict[str, TensorData],
    ) -> dict[str, Any]:
        try:
            runtime = self._runtime_clients[action_session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown action_session_id: {action_session_id}") from exc
        request = ActRequest(
            action_session_id=action_session_id,
            observation=observation,
            extra_inputs=extra_inputs,
        )
        return await runtime.request("act", request.model_dump(mode="json"))

    async def shutdown_session(self, action_session_id: str) -> None:
        runtime = self._runtime_clients.pop(action_session_id, None)
        if runtime is None:
            return
        try:
            await runtime.request("shutdown", {"action_session_id": action_session_id})
        finally:
            close = getattr(runtime, "close", None)
            if callable(close):
                await close()

    async def shutdown_all(self) -> None:
        for action_session_id in list(self._runtime_clients.keys()):
            await self.shutdown_session(action_session_id)
