from __future__ import annotations

import asyncio
import structlog
import os
from pathlib import Path
from typing import Any

import ray

from mint_server.config import PFS_PYTHONPATH, RAY_NAMESPACE, actor_runtime_env_vars
from mint_server.ray.ray_utils import init_ray
from mint_server.ray.runtime_env import env_nonempty
from mint_server.backend.ray_cluster.async_ray_control import async_get_ray_ref
from mint_server.backend.openpi.openpi_direct_runtime import OpenPIDirectWorkerClient
from mint_server.backend.openpi.openpi_fast_runtime import (
    OpenPIFastRuntimeSpec,
    OpenPIFastWorkerError,
    OpenPIFastWorkerProtocolError,
)


logger = structlog.get_logger(__name__)


def _openpi_runtime_env_vars() -> dict[str, str]:
    extra = {"PYTHONDONTWRITEBYTECODE": "1"}
    for key, value in os.environ.items():
        if key.startswith("MINT_OPENPI_"):
            extra[key] = value
    xla_flags = os.environ.get("MINT_OPENPI_XLA_FLAGS", "").strip()
    if xla_flags:
        extra["XLA_FLAGS"] = xla_flags
    for key in ("HF_HOME", "HF_HUB_OFFLINE", "OPENPI_DATA_HOME"):
        value = os.environ.get(key, "").strip()
        if value:
            extra[key] = value
    return actor_runtime_env_vars(
        pythonpath=PFS_PYTHONPATH,
        extra=extra,
        include_ray_attach_hints=False,
    )


def _action_session_state_root(actor_name: str) -> str:
    mint_code_root = str(os.environ.get("MINT_CODE_ROOT") or "").strip()
    if not mint_code_root:
        raise RuntimeError("OpenPI action session state root requires MINT_CODE_ROOT")
    namespace = str(
        env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
        or RAY_NAMESPACE
        or ""
    ).strip()
    if not namespace:
        raise RuntimeError("OpenPI action session state root requires a Ray namespace")
    namespace_dir = namespace.replace("/", "_")
    return str(
        (
            Path(mint_code_root).resolve()
            / "checkpoints"
            / "openpi_action_session_state"
            / namespace_dir
            / actor_name
        ).resolve()
    )


def ensure_openpi_ray_initialized() -> None:
    if not ray.is_initialized():
        init_ray(address="auto", namespace=RAY_NAMESPACE, ignore_reinit_error=True)
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized for OpenPI training runtime")


def _ray_timeout(timeout_s: float | None, *, extra_s: float = 5.0) -> float | None:
    if timeout_s is None:
        return None
    return max(float(timeout_s), 0.0) + extra_s


def _actor_ready_timeout_s(spec: OpenPIFastRuntimeSpec) -> float:
    override = (os.environ.get("MINT_OPENPI_RAY_ACTOR_READY_TIMEOUT_S") or "").strip()
    if override:
        return float(override)
    return max(
        float(spec.startup_timeout_s),
        float(spec.create_session_timeout_s),
        300.0,
    )


@ray.remote(num_gpus=1, max_concurrency=1)
class OpenPIRayRuntimeActor:
    def __init__(
        self,
        *,
        model_id: str,
        training_session_id: str,
        spec: OpenPIFastRuntimeSpec,
    ) -> None:
        self._model_id = model_id
        self._training_session_id = training_session_id
        self._spec = spec
        self._runtime: OpenPIDirectWorkerClient | None = None

    async def _ensure_runtime(self) -> OpenPIDirectWorkerClient:
        if self._runtime is None:
            self._runtime = await OpenPIDirectWorkerClient.start(self._spec)
        return self._runtime

    async def ready_metadata(self) -> dict[str, Any]:
        await self._ensure_runtime()
        return self.describe()

    def describe(self) -> dict[str, Any]:
        context = ray.get_runtime_context()
        get_actor_id = getattr(context, "get_actor_id", None)
        get_node_id = getattr(context, "get_node_id", None)
        return {
            "model_id": self._model_id,
            "session_id": self._training_session_id,
            "worker_module": self._spec.worker_module,
            "actor_id": str(get_actor_id()) if callable(get_actor_id) else None,
            "node_id": str(get_node_id()) if callable(get_node_id) else None,
            "node_ip": ray.util.get_node_ip_address(),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    async def request(
        self,
        op: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        runtime = await self._ensure_runtime()
        return await runtime.request(op, payload or {}, timeout_s=timeout_s)

    async def shutdown(self) -> dict[str, Any]:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            await runtime.close()
        return {"stopped": True, **self.describe()}


class OpenPIRayRuntimeClient:
    def __init__(
        self,
        *,
        actor: Any,
        spec: OpenPIFastRuntimeSpec,
        session_id: str,
        ready_timeout_s: float,
    ) -> None:
        self._actor = actor
        self._spec = spec
        self._session_id = session_id
        self._ready_timeout_s = float(ready_timeout_s)
        self._closed = False
        self._metadata: dict[str, Any] | None = None

    @property
    def metadata(self) -> dict[str, Any] | None:
        if self._metadata is None:
            return None
        return dict(self._metadata)

    def timeout_for(self, op: str) -> float:
        if op == "create_session":
            return self._spec.create_session_timeout_s
        if op == "save_weights":
            return self._spec.save_weights_timeout_s
        if op == "load_weights":
            return self._spec.load_weights_timeout_s
        return self._spec.request_timeout_s

    async def _ray_get(self, ref: Any, *, timeout_s: float | None) -> Any:
        try:
            return await async_get_ray_ref(ref, timeout_s=_ray_timeout(timeout_s))
        except ray.exceptions.GetTimeoutError as exc:
            raise OpenPIFastWorkerProtocolError(
                f"Ray runtime timed out for session {self._session_id!r} after {timeout_s}s"
            ) from exc
        except ray.exceptions.RayTaskError as exc:
            converted = exc.as_instanceof_cause()
            if isinstance(converted, OpenPIFastWorkerError):
                raise converted from None
            cause = getattr(exc, "cause", None) or getattr(exc, "__cause__", None)
            if isinstance(cause, OpenPIFastWorkerError):
                raise cause from None
            raise

    async def ready(self) -> dict[str, Any]:
        metadata = await self._ray_get(
            self._actor.ready_metadata.remote(),
            timeout_s=self._ready_timeout_s,
        )
        if not isinstance(metadata, dict):
            raise TypeError(f"OpenPI Ray runtime ready payload must be dict, got {type(metadata)}")
        self._metadata = metadata
        return metadata

    async def request(
        self,
        op: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise OpenPIFastWorkerProtocolError("OpenPI Ray runtime client is closed")
        result = await self._ray_get(
            self._actor.request.remote(op, payload or {}, timeout_s=timeout_s),
            timeout_s=timeout_s,
        )
        if not isinstance(result, dict):
            raise TypeError(f"OpenPI Ray runtime request returned non-dict payload: {type(result)}")
        return result

    async def describe(self) -> dict[str, Any]:
        metadata = await self._ray_get(self._actor.describe.remote(), timeout_s=5.0)
        if not isinstance(metadata, dict):
            raise TypeError(f"OpenPI Ray runtime metadata must be dict, got {type(metadata)}")
        self._metadata = metadata
        return metadata

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ray_get(self._actor.shutdown.remote(), timeout_s=5.0)
        except Exception as exc:
            logger.warning(
                "OpenPI Ray runtime shutdown failed for session %s: %s: %s",
                self._session_id,
                type(exc).__name__,
                exc,
            )
        try:
            await asyncio.to_thread(ray.kill, self._actor, no_restart=True)
        except Exception as exc:
            logger.warning(
                "OpenPI Ray runtime kill failed for session %s: %s: %s",
                self._session_id,
                type(exc).__name__,
                exc,
            )


async def start_openpi_ray_runtime(
    *,
    session: Any,
    spec: OpenPIFastRuntimeSpec,
) -> OpenPIRayRuntimeClient:
    ensure_openpi_ray_initialized()
    actor = OpenPIRayRuntimeActor.options(
        runtime_env={"env_vars": _openpi_runtime_env_vars()},
    ).remote(
        model_id=str(session.model_id),
        training_session_id=str(session.session_id),
        spec=spec,
    )
    client = OpenPIRayRuntimeClient(
        actor=actor,
        spec=spec,
        session_id=str(session.model_id),
        ready_timeout_s=_actor_ready_timeout_s(spec),
    )
    try:
        await client.ready()
    except Exception:
        await client.close()
        raise
    return client
