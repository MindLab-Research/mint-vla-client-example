from __future__ import annotations

import dataclasses
import os
import structlog
import uuid
from typing import Any, Awaitable, Callable

from mint_server.checkpoints.checkpoints import get_checkpoints_dir, resolve_checkpoint_uri
from mint_server.models.types import ActRequest, ModelInput, TensorData
from mint_server.backend.core.model_registry import get_model_config
from mint_server.backend.openpi.openpi_direct_runtime import OpenPIDirectWorkerClient
from mint_server.backend.openpi.openpi_fast_action_runtime import (
    OPENPI_FAST_ACTION_WORKER_MODULE,
    OpenPIFastActionRuntimeSpec,
)
from mint_server.backend.openpi.openpi_fast_training import (
    OPENPI_FAST_TRAINING_BACKEND,
    get_openpi_fast_config_name,
)
from mint_server.backend.openpi.openpi_pi05_training import (
    OPENPI_PI05_TRAINING_BACKEND,
    OPENPI_PI05_ACTION_WORKER_MODULE,
    get_openpi_pi05_config_name,
)
from mint_server.backend.openpi.pi05_profiles import profile_from_model_config

logger = structlog.get_logger(__name__)


def _is_openpi_fast_model(base_model: str) -> bool:
    try:
        return get_model_config(base_model).training_backend == OPENPI_FAST_TRAINING_BACKEND
    except Exception:
        return False


def _is_openpi_pi05_model(base_model: str) -> bool:
    try:
        return get_model_config(base_model).training_backend == OPENPI_PI05_TRAINING_BACKEND
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
    # Ray removed: openpi_fast action inference runs in-process via direct client.
    del action_session_id, base_model, checkpoint_path, model_config, config_name
    spec = OpenPIFastActionRuntimeSpec(worker_module=OPENPI_FAST_ACTION_WORKER_MODULE)
    return await OpenPIDirectWorkerClient.start(spec)


async def _default_pi05_runtime_factory(
    *,
    action_session_id: str,
    base_model: str,
    checkpoint_path: str,
    model_config: Any,
    config_name: str,
) -> Any:
    # Ray removed: pi0.5 action inference runs in-process via direct client.
    from mint_server.backend.openpi.openpi_pi05_local_runtime import (
        make_local_pi05_action_runtime,
    )

    return await make_local_pi05_action_runtime(
        action_session_id=action_session_id,
        base_model=base_model,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        config_name=config_name,
    )


def _runtime_spec_for_worker_module(worker_module: str | None) -> OpenPIFastActionRuntimeSpec:
    spec = OpenPIFastActionRuntimeSpec.from_env()
    if worker_module:
        return dataclasses.replace(spec, worker_module=str(worker_module))
    return spec


def _recover_detached_action_runtime_client(
    *,
    action_session_id: str,
    supports_base_model: Callable[[str], bool],
    supports_worker_module: Callable[[str], bool],
) -> Any | None:
    # Ray removed: openpi action runtimes are process-local (held in
    # self._runtime_clients). There are no detached Ray actors to recover from,
    # so recovery is a no-op. A missing session simply means it is not local.
    del action_session_id, supports_base_model, supports_worker_module
    return None


def _action_billing_metadata(base_model: str, model_config: Any) -> dict[str, Any]:
    policy_family = str(getattr(model_config, "policy_family", "") or "")
    action_token_budget = max(0, int(getattr(model_config, "action_token_budget", 0) or 0))
    action_output_tokens = action_token_budget if policy_family == "ar_action_tokens" else 0
    return {
        "base_model": str(base_model or ""),
        "policy_family": policy_family,
        "action_dim": max(0, int(getattr(model_config, "action_dim", 0) or 0)),
        "action_horizon": max(0, int(getattr(model_config, "action_horizon", 0) or 0)),
        "action_token_budget": action_token_budget,
        "action_output_tokens": action_output_tokens,
    }


def _action_billing_metadata_for_base_model(base_model: str) -> dict[str, Any]:
    try:
        return _action_billing_metadata(base_model, get_model_config(base_model))
    except Exception:
        return {"base_model": str(base_model or "")}


def _is_retryable_openpi_runtime_error(exc: BaseException | None) -> bool:
    if exc is None:
        return False

    # Ray removed: match by name/message only (no ray.exceptions types).
    cause = getattr(exc, "__cause__", None) or getattr(exc, "cause", None)
    if cause is not None and cause is not exc and _is_retryable_openpi_runtime_error(cause):
        return True

    messages = [type(exc).__name__, str(exc)]
    messages.extend(str(note) for note in list(getattr(exc, "__notes__", ()) or ()))
    return any(
        marker in message
        for message in messages
        for marker in (
            "ActorDiedError",
            "RayActorError",
            "ActorUnavailableError",
            "killed by `ray.kill`",
            "The actor died unexpectedly",
        )
    )


async def _close_runtime_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        await close()


async def _create_openpi_action_runtime_client(
    *,
    runtime_factory: Callable[..., Awaitable[Any]],
    action_session_id: str,
    base_model: str,
    checkpoint_path: str,
    model_config: Any,
    config_name: str,
    create_payload: dict[str, Any],
    log_label: str,
    max_attempts: int = 2,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            client = await runtime_factory(
                action_session_id=action_session_id,
                base_model=base_model,
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                config_name=config_name,
            )
        except Exception as exc:
            if attempt < max_attempts and _is_retryable_openpi_runtime_error(exc):
                logger.warning(
                    "[%s] runtime_factory hit retryable actor error; retrying action-session bootstrap "
                    "attempt=%s/%s action_session_id=%s base_model=%s checkpoint_path=%s error_type=%s error=%s",
                    log_label,
                    attempt,
                    max_attempts,
                    action_session_id,
                    base_model,
                    checkpoint_path,
                    type(exc).__name__,
                    exc,
                )
                continue
            logger.exception(
                "[%s] runtime_factory failed: action_session_id=%s base_model=%s checkpoint_path=%s",
                log_label,
                action_session_id,
                base_model,
                checkpoint_path,
            )
            raise

        try:
            await client.request("create_session", create_payload)
            return client
        except Exception as exc:
            await _close_runtime_client(client)
            if attempt < max_attempts and _is_retryable_openpi_runtime_error(exc):
                logger.warning(
                    "[%s] create_session hit retryable actor error; retrying action-session bootstrap "
                    "attempt=%s/%s action_session_id=%s base_model=%s checkpoint_path=%s error_type=%s error=%s",
                    log_label,
                    attempt,
                    max_attempts,
                    action_session_id,
                    base_model,
                    checkpoint_path,
                    type(exc).__name__,
                    exc,
                )
                continue
            logger.exception(
                "[%s] create_session failed: action_session_id=%s base_model=%s checkpoint_path=%s",
                log_label,
                action_session_id,
                base_model,
                checkpoint_path,
            )
            raise

    raise RuntimeError(
        f"[{log_label}] action-session bootstrap exhausted retry budget for {action_session_id!r}"
    )


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
        self._billing_metadata: dict[str, dict[str, Any]] = {}

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
        client = await _create_openpi_action_runtime_client(
            runtime_factory=self._runtime_factory,
            action_session_id=action_session_id,
            base_model=base_model,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            config_name=config_name,
            create_payload={
                "action_session_id": action_session_id,
                "base_model": base_model,
                "checkpoint_path": checkpoint_path,
                "config_name": config_name,
                "action_dim": int(model_config.action_dim or 0),
                "action_horizon": int(model_config.action_horizon or 0),
                "action_token_budget": int(model_config.action_token_budget or 0),
                "max_token_len": int(model_config.max_model_len),
                "camera_layout": list(model_config.camera_layout),
            },
            log_label="openpi_fast_action",
        )

        self._runtime_clients[action_session_id] = client
        self._billing_metadata[action_session_id] = _action_billing_metadata(base_model, model_config)
        return action_session_id

    def get_billing_metadata(self, action_session_id: str) -> dict[str, Any] | None:
        metadata = self._billing_metadata.get(action_session_id)
        return None if metadata is None else dict(metadata)

    async def act(
        self,
        *,
        action_session_id: str,
        observation: ModelInput,
        extra_inputs: dict[str, TensorData],
        temperature: float | None = None,
        return_rollout_trace: bool | None = None,
        rollout_trace_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = self._runtime_clients.get(action_session_id)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_fast_model,
                supports_worker_module=lambda worker_module: worker_module == OPENPI_FAST_ACTION_WORKER_MODULE,
            )
            if runtime is not None:
                self._runtime_clients[action_session_id] = runtime
        if runtime is None:
            raise KeyError(f"Unknown action_session_id: {action_session_id}")
        request = ActRequest(
            action_session_id=action_session_id,
            observation=observation,
            extra_inputs=extra_inputs,
            temperature=temperature,
            return_rollout_trace=return_rollout_trace,
            rollout_trace_config=rollout_trace_config,
        )
        return await runtime.request("act", request.model_dump(mode="json", exclude_none=True))

    async def shutdown_session(self, action_session_id: str) -> None:
        runtime = self._runtime_clients.pop(action_session_id, None)
        self._billing_metadata.pop(action_session_id, None)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_fast_model,
                supports_worker_module=lambda worker_module: worker_module == OPENPI_FAST_ACTION_WORKER_MODULE,
            )
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


class OpenPIPi05ActionSessionManager:
    def __init__(
        self,
        *,
        runtime_factory: Callable[..., Awaitable[Any]] | None = None,
        checkpoints_dir: str | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or _default_pi05_runtime_factory
        self._checkpoints_dir = checkpoints_dir or get_checkpoints_dir()
        self._runtime_clients: dict[str, Any] = {}
        self._billing_metadata: dict[str, dict[str, Any]] = {}

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
        if not _is_openpi_pi05_model(base_model):
            raise ValueError(f"OpenPI pi0.5 action inference does not support {base_model!r}")
        if not model_path:
            raise ValueError("OpenPI pi0.5 action inference requires model_path")

        model_config = get_model_config(base_model)
        checkpoint_path = self._resolve_model_path(model_path, user_id)
        action_session_id = self._action_session_id(session_id, action_session_seq_id)
        config_name = get_openpi_pi05_config_name(base_model)
        create_payload = {
            "action_session_id": action_session_id,
            "base_model": base_model,
            "checkpoint_path": checkpoint_path,
            "config_name": config_name,
            "action_dim": int(model_config.action_dim or 0),
            "action_horizon": int(model_config.action_horizon or 0),
            "max_token_len": int(model_config.max_model_len),
            "camera_layout": list(model_config.camera_layout),
        }
        if model_config.profile:
            create_payload["profile"] = profile_from_model_config(model_config).checkpoint_manifest()
        client = await _create_openpi_action_runtime_client(
            runtime_factory=self._runtime_factory,
            action_session_id=action_session_id,
            base_model=base_model,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            config_name=config_name,
            create_payload=create_payload,
            log_label="openpi_pi05_action",
        )

        self._runtime_clients[action_session_id] = client
        self._billing_metadata[action_session_id] = _action_billing_metadata(base_model, model_config)
        return action_session_id

    def get_billing_metadata(self, action_session_id: str) -> dict[str, Any] | None:
        metadata = self._billing_metadata.get(action_session_id)
        return None if metadata is None else dict(metadata)

    async def act(
        self,
        *,
        action_session_id: str,
        observation: ModelInput,
        extra_inputs: dict[str, TensorData],
        temperature: float | None = None,
        return_rollout_trace: bool | None = None,
        rollout_trace_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = self._runtime_clients.get(action_session_id)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_pi05_model,
                supports_worker_module=lambda worker_module: worker_module == OPENPI_PI05_ACTION_WORKER_MODULE,
            )
            if runtime is not None:
                self._runtime_clients[action_session_id] = runtime
        if runtime is None:
            raise KeyError(f"Unknown action_session_id: {action_session_id}")
        request = ActRequest(
            action_session_id=action_session_id,
            observation=observation,
            extra_inputs=extra_inputs,
            temperature=temperature,
            return_rollout_trace=return_rollout_trace,
            rollout_trace_config=rollout_trace_config,
        )
        payload = request.model_dump(mode="json", exclude_none=True)
        return await runtime.request("act", payload)

    async def act_batch(
        self,
        *,
        action_session_id: str,
        observations: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Batched act(): one HTTP round-trip infers `observations` (list of
        {"chunks": [...]} model_input dicts already carrying `extra_inputs.state`
        merged in by the caller -- see routes/mint.py's act_batch endpoint) in a
        single jitted, data-sharded worker call. See
        `openpi_pi05_action_worker.OpenPIPi05ActionSession.act_batch` for the
        sharding mechanics; this method is pure plumbing.
        """
        runtime = self._runtime_clients.get(action_session_id)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_pi05_model,
                supports_worker_module=lambda worker_module: worker_module == OPENPI_PI05_ACTION_WORKER_MODULE,
            )
            if runtime is not None:
                self._runtime_clients[action_session_id] = runtime
        if runtime is None:
            raise KeyError(f"Unknown action_session_id: {action_session_id}")
        payload = {"observations": observations, "temperature": temperature}
        return await runtime.request("act_batch", payload)

    async def shutdown_session(self, action_session_id: str) -> None:
        runtime = self._runtime_clients.pop(action_session_id, None)
        self._billing_metadata.pop(action_session_id, None)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_pi05_model,
                supports_worker_module=lambda worker_module: worker_module == OPENPI_PI05_ACTION_WORKER_MODULE,
            )
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


class ActionSessionRouter:
    def __init__(
        self,
        *,
        openpi_fast_manager: OpenPIFastActionSessionManager | None = None,
        openpi_pi05_manager: OpenPIPi05ActionSessionManager | None = None,
    ) -> None:
        self._openpi_fast = openpi_fast_manager or OpenPIFastActionSessionManager()
        self._openpi_pi05 = openpi_pi05_manager or OpenPIPi05ActionSessionManager()
        self._manager_for_session: dict[str, object] = {}
        self._billing_metadata: dict[str, dict[str, Any]] = {}

    def _manager_for_model(self, base_model: str) -> object:
        if _is_openpi_fast_model(base_model):
            return self._openpi_fast
        if _is_openpi_pi05_model(base_model):
            return self._openpi_pi05
        raise ValueError(f"Action inference does not support {base_model!r}")

    def _recover_manager_for_session(self, action_session_id: str) -> object | None:
        # Ray removed: action sessions are process-local (self._manager_for_session).
        # There are no detached Ray actors to recover from after a restart.
        del action_session_id
        return None

    async def create_session(
        self,
        *,
        session_id: str,
        action_session_seq_id: int | None,
        base_model: str,
        model_path: str | None,
        user_id: str | None,
    ) -> str:
        manager = self._manager_for_model(base_model)
        action_session_id = await manager.create_session(  # type: ignore[attr-defined]
            session_id=session_id,
            action_session_seq_id=action_session_seq_id,
            base_model=base_model,
            model_path=model_path,
            user_id=user_id,
        )
        self._manager_for_session[action_session_id] = manager
        self._billing_metadata[action_session_id] = _action_billing_metadata_for_base_model(base_model)
        return action_session_id

    def get_billing_metadata(self, action_session_id: str) -> dict[str, Any] | None:
        metadata = self._billing_metadata.get(action_session_id)
        if metadata is not None:
            return dict(metadata)
        manager = self._manager_for_session.get(action_session_id)
        if manager is None:
            manager = self._recover_manager_for_session(action_session_id)
        getter = getattr(manager, "get_billing_metadata", None)
        if callable(getter):
            metadata = getter(action_session_id)
            if isinstance(metadata, dict):
                self._billing_metadata[action_session_id] = dict(metadata)
                return dict(metadata)
        return None

    async def act(
        self,
        *,
        action_session_id: str,
        observation: ModelInput,
        extra_inputs: dict[str, TensorData],
        temperature: float | None = None,
        return_rollout_trace: bool | None = None,
        rollout_trace_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manager = self._manager_for_session.get(action_session_id)
        if manager is None:
            manager = self._recover_manager_for_session(action_session_id)
        if manager is None:
            raise KeyError(f"Unknown action_session_id: {action_session_id}")
        act_kwargs: dict[str, Any] = dict(
            action_session_id=action_session_id,
            observation=observation,
            extra_inputs=extra_inputs,
            temperature=temperature,
            return_rollout_trace=return_rollout_trace,
            rollout_trace_config=rollout_trace_config,
        )
        return await manager.act(**act_kwargs)  # type: ignore[attr-defined]

    async def act_batch(
        self,
        *,
        action_session_id: str,
        observations: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        manager = self._manager_for_session.get(action_session_id)
        if manager is None:
            manager = self._recover_manager_for_session(action_session_id)
        if manager is None:
            raise KeyError(f"Unknown action_session_id: {action_session_id}")
        act_batch_fn = getattr(manager, "act_batch", None)
        if not callable(act_batch_fn):
            raise NotImplementedError(
                f"{type(manager).__name__} does not support act_batch (openpi_fast backend not covered yet)"
            )
        return await act_batch_fn(
            action_session_id=action_session_id,
            observations=observations,
            temperature=temperature,
        )

    async def shutdown_session(self, action_session_id: str) -> None:
        manager = self._manager_for_session.pop(action_session_id, None)
        self._billing_metadata.pop(action_session_id, None)
        if manager is None:
            manager = self._recover_manager_for_session(action_session_id)
        if manager is None:
            await self._openpi_fast.shutdown_session(action_session_id)
            await self._openpi_pi05.shutdown_session(action_session_id)
            return
        self._manager_for_session.pop(action_session_id, None)
        await manager.shutdown_session(action_session_id)  # type: ignore[attr-defined]

    async def shutdown_all(self) -> None:
        self._manager_for_session.clear()
        await self._openpi_fast.shutdown_all()
        await self._openpi_pi05.shutdown_all()
