from __future__ import annotations

import dataclasses
import logging
import uuid
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import ray

from ..checkpoints import get_checkpoints_dir, resolve_checkpoint_uri
from ..models.types import ActRequest, ModelInput, TensorData
from .model_registry import get_model_config
from .openpi_action_ray_runtime import (
    OpenPIActionRayRuntimeClient,
    _actor_ready_timeout_s,
    start_openpi_action_ray_runtime,
)
from .openpi_fast_action_runtime import OpenPIFastActionRuntimeSpec
from .openpi_fast_training import (
    OPENPI_FAST_TRAINING_BACKEND,
    get_openpi_fast_config_name,
)
from .openpi_shared_ray_runtime import start_openpi_shared_ray_runtime
from .openpi_shared_ray_runtime import OpenPISharedRayRuntimeClient
from .openpi_pi05_training import (
    OPENPI_PI05_TRAINING_BACKEND,
    get_openpi_pi05_config_name,
)
from .resource_pool import ActorType, get_resource_pool

logger = logging.getLogger(__name__)


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
    del checkpoint_path
    return await start_openpi_shared_ray_runtime(
        session=SimpleNamespace(model_id=action_session_id, base_model=base_model),
        spec=OpenPIFastActionRuntimeSpec.from_env(),
        config_name=config_name,
        model_config=model_config,
        template_reusable=False,
    )


async def _default_pi05_runtime_factory(
    *,
    action_session_id: str,
    base_model: str,
    checkpoint_path: str,
    model_config: Any,
    config_name: str,
) -> Any:
    del checkpoint_path
    spec = dataclasses.replace(
        OpenPIFastActionRuntimeSpec.from_env(),
        worker_module="tinker_server.backend.openpi_pi05_action_worker",
    )
    return await start_openpi_shared_ray_runtime(
        session=SimpleNamespace(model_id=action_session_id, base_model=base_model),
        spec=spec,
        config_name=config_name,
        model_config=model_config,
        template_reusable=False,
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
) -> OpenPIActionRayRuntimeClient | OpenPISharedRayRuntimeClient | None:
    pool = get_resource_pool()
    shared_candidates: list[tuple[Any, OpenPIFastActionRuntimeSpec]] = []
    for entry in pool.iter_entries(prune_stale=True):
        metadata = dict(entry.metadata or {})
        if entry.actor_type != ActorType.OPENPI:
            continue
        base_model = str(entry.base_model or "")
        if not supports_base_model(base_model):
            continue
        worker_module = str(metadata.get("worker_module") or "")
        if not worker_module or not supports_worker_module(worker_module):
            continue
        current = pool.get(entry.actor_name)
        actor_entry = current or entry
        actor_handle = actor_entry.actor_handle
        if actor_handle is None:
            continue
        spec = _runtime_spec_for_worker_module(worker_module)
        is_shared = str(entry.actor_name).startswith("openpi_shared_runtime_") or "pool_key" in metadata
        if is_shared:
            shared_candidates.append((actor_entry, spec))
        recovered = str(metadata.get("action_session_id") or entry.current_session or "") == action_session_id
        if not recovered and str(entry.actor_name or "").startswith("openpi_shared_runtime_"):
            try:
                recovered = action_session_id in _shared_actor_known_sessions(actor_handle)
            except Exception:
                recovered = False
        if not recovered:
            continue
        if is_shared:
            return OpenPISharedRayRuntimeClient(
                actor=actor_handle,
                actor_name=actor_entry.actor_name,
                spec=spec,
                session_id=action_session_id,
                ready_timeout_s=_actor_ready_timeout_s(spec),
            )
        return OpenPIActionRayRuntimeClient(
            actor=actor_handle,
            actor_name=actor_entry.actor_name,
            spec=spec,
            action_session_id=action_session_id,
            ready_timeout_s=_actor_ready_timeout_s(spec),
        )
    if len(shared_candidates) == 1:
        actor_entry, spec = shared_candidates[0]
        logger.warning(
            "[action_session_recover] inferring shared action actor without exact session membership: "
            "action_session_id=%s actor_name=%s base_model=%s worker_module=%s",
            action_session_id,
            actor_entry.actor_name,
            actor_entry.base_model,
            spec.worker_module,
        )
        return OpenPISharedRayRuntimeClient(
            actor=actor_entry.actor_handle,
            actor_name=actor_entry.actor_name,
            spec=spec,
            session_id=action_session_id,
            ready_timeout_s=_actor_ready_timeout_s(spec),
        )
    return None


def _shared_actor_known_sessions(actor_handle: Any) -> set[str]:
    describe = getattr(actor_handle, "describe", None)
    remote = getattr(describe, "remote", None)
    if not callable(remote):
        return set()
    payload = ray.get(remote(), timeout=5.0)
    if not isinstance(payload, dict):
        return set()
    return {str(session_id) for session_id in list(payload.get("known_session_ids") or []) if session_id}


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
        try:
            client = await self._runtime_factory(
                action_session_id=action_session_id,
                base_model=base_model,
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                config_name=config_name,
            )
        except Exception:
            logger.exception(
                "[openpi_fast_action] runtime_factory failed: action_session_id=%s base_model=%s checkpoint_path=%s",
                action_session_id,
                base_model,
                checkpoint_path,
            )
            raise
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
                    "action_token_budget": int(model_config.action_token_budget or 0),
                    "max_token_len": int(model_config.max_model_len),
                    "camera_layout": list(model_config.camera_layout),
                },
            )
        except Exception:
            logger.exception(
                "[openpi_fast_action] create_session failed: action_session_id=%s base_model=%s checkpoint_path=%s",
                action_session_id,
                base_model,
                checkpoint_path,
            )
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
        temperature: float | None = None,
    ) -> dict[str, Any]:
        runtime = self._runtime_clients.get(action_session_id)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_fast_model,
                supports_worker_module=lambda worker_module: worker_module
                == "tinker_server.backend.openpi_fast_action_worker",
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
        )
        return await runtime.request("act", request.model_dump(mode="json"))

    async def shutdown_session(self, action_session_id: str) -> None:
        runtime = self._runtime_clients.pop(action_session_id, None)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_fast_model,
                supports_worker_module=lambda worker_module: worker_module
                == "tinker_server.backend.openpi_fast_action_worker",
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
        temperature: float | None = None,
    ) -> dict[str, Any]:
        runtime = self._runtime_clients.get(action_session_id)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_pi05_model,
                supports_worker_module=lambda worker_module: worker_module
                == "tinker_server.backend.openpi_pi05_action_worker",
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
        )
        return await runtime.request("act", request.model_dump(mode="json"))

    async def shutdown_session(self, action_session_id: str) -> None:
        runtime = self._runtime_clients.pop(action_session_id, None)
        if runtime is None:
            runtime = _recover_detached_action_runtime_client(
                action_session_id=action_session_id,
                supports_base_model=_is_openpi_pi05_model,
                supports_worker_module=lambda worker_module: worker_module
                == "tinker_server.backend.openpi_pi05_action_worker",
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

    def _manager_for_model(self, base_model: str) -> object:
        if _is_openpi_fast_model(base_model):
            return self._openpi_fast
        if _is_openpi_pi05_model(base_model):
            return self._openpi_pi05
        raise ValueError(f"Action inference does not support {base_model!r}")

    def _recover_manager_for_session(self, action_session_id: str) -> object | None:
        pool = get_resource_pool()
        candidate_managers: dict[int, object] = {}
        for entry in pool.iter_entries(prune_stale=True):
            metadata = dict(entry.metadata or {})
            if entry.actor_type != ActorType.OPENPI:
                continue
            worker_module = str(metadata.get("worker_module") or "")
            if not worker_module.endswith("_action_worker"):
                continue
            recovered = str(metadata.get("action_session_id") or entry.current_session or "") == action_session_id
            if not recovered and str(entry.actor_name or "").startswith("openpi_shared_runtime_"):
                actor_handle = entry.actor_handle
                if actor_handle is not None:
                    try:
                        recovered = action_session_id in _shared_actor_known_sessions(actor_handle)
                    except Exception:
                        recovered = False
            if not recovered:
                continue
            try:
                manager = self._manager_for_model(str(entry.base_model or ""))
            except ValueError:
                continue
            candidate_managers.setdefault(id(manager), manager)
            if not recovered:
                continue
            self._manager_for_session[action_session_id] = manager
            return manager
        if len(candidate_managers) == 1:
            manager = next(iter(candidate_managers.values()))
            logger.warning(
                "[action_session_router] inferring action manager without exact session membership: action_session_id=%s",
                action_session_id,
            )
            self._manager_for_session[action_session_id] = manager
            return manager
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
        return action_session_id

    async def act(
        self,
        *,
        action_session_id: str,
        observation: ModelInput,
        extra_inputs: dict[str, TensorData],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        manager = self._manager_for_session.get(action_session_id)
        if manager is None:
            manager = self._recover_manager_for_session(action_session_id)
        if manager is None:
            raise KeyError(f"Unknown action_session_id: {action_session_id}")
        return await manager.act(  # type: ignore[attr-defined]
            action_session_id=action_session_id,
            observation=observation,
            extra_inputs=extra_inputs,
            temperature=temperature,
        )

    async def shutdown_session(self, action_session_id: str) -> None:
        manager = self._manager_for_session.pop(action_session_id, None)
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
