from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any

import ray

from ..config import RAY_NAMESPACE
from .openpi_fast_runtime import (
    OpenPIFastRuntimeSpec,
    OpenPIFastWorkerClient,
    OpenPIFastWorkerError,
    OpenPIFastWorkerProtocolError,
)
from .openpi_ray_runtime import (
    _actor_ready_timeout_s,
    _openpi_runtime_env_vars,
    _ray_timeout,
    ensure_openpi_ray_initialized,
)
from .resource_pool import ActorType, get_resource_pool


logger = logging.getLogger(__name__)


def _action_actor_name(
    *,
    action_session_id: str,
    base_model: str,
    worker_module: str,
) -> str:
    payload = json.dumps(
        {
            "action_session_id": action_session_id,
            "base_model": base_model,
            "worker_module": worker_module,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"openpi_action_runtime_{hashlib.sha1(payload).hexdigest()[:12]}"


@ray.remote(num_gpus=1, max_concurrency=1)
class OpenPIActionRayRuntimeActor:
    def __init__(
        self,
        *,
        action_session_id: str,
        base_model: str,
        spec: OpenPIFastRuntimeSpec,
    ) -> None:
        self._action_session_id = action_session_id
        self._base_model = base_model
        self._spec = spec
        self._runtime: OpenPIFastWorkerClient | None = None

    async def _ensure_runtime(self) -> OpenPIFastWorkerClient:
        if self._runtime is None:
            self._runtime = await OpenPIFastWorkerClient.start(self._spec)
        return self._runtime

    async def ready_metadata(self) -> dict[str, Any]:
        await self._ensure_runtime()
        return self.describe()

    def describe(self) -> dict[str, Any]:
        context = ray.get_runtime_context()
        get_actor_id = getattr(context, "get_actor_id", None)
        get_node_id = getattr(context, "get_node_id", None)
        return {
            "action_session_id": self._action_session_id,
            "base_model": self._base_model,
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


class OpenPIActionRayRuntimeClient:
    def __init__(
        self,
        *,
        actor: Any,
        actor_name: str,
        spec: OpenPIFastRuntimeSpec,
        action_session_id: str,
        ready_timeout_s: float,
    ) -> None:
        self._actor = actor
        self._actor_name = actor_name
        self._spec = spec
        self._action_session_id = action_session_id
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
            return await asyncio.to_thread(ray.get, ref, timeout=_ray_timeout(timeout_s))
        except ray.exceptions.GetTimeoutError as exc:
            raise OpenPIFastWorkerProtocolError(
                "OpenPI action Ray runtime timed out for action_session_id "
                f"{self._action_session_id!r} after {timeout_s}s"
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
            raise TypeError(
                f"OpenPI action Ray runtime ready payload must be dict, got {type(metadata)}"
            )
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
            raise OpenPIFastWorkerProtocolError("OpenPI action Ray runtime client is closed")

        pool = get_resource_pool()
        pool.mark_inflight(self._actor_name, +1)
        try:
            result = await self._ray_get(
                self._actor.request.remote(op, payload or {}, timeout_s=timeout_s),
                timeout_s=timeout_s,
            )
        finally:
            pool.mark_inflight(self._actor_name, -1)

        if not isinstance(result, dict):
            raise TypeError(
                f"OpenPI action Ray runtime request returned non-dict payload: {type(result)}"
            )
        if op == "shutdown":
            pool.set_session(self._actor_name, None)
        else:
            pool.set_session(self._actor_name, self._action_session_id)
        pool.touch(self._actor_name)
        return result

    async def describe(self) -> dict[str, Any]:
        metadata = await self._ray_get(self._actor.describe.remote(), timeout_s=5.0)
        if not isinstance(metadata, dict):
            raise TypeError(
                f"OpenPI action Ray runtime metadata must be dict, got {type(metadata)}"
            )
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
                "OpenPI action Ray runtime shutdown failed for %s: %s: %s",
                self._action_session_id,
                type(exc).__name__,
                exc,
            )
        try:
            await asyncio.to_thread(ray.kill, self._actor, no_restart=True)
        except Exception as exc:
            logger.warning(
                "OpenPI action Ray runtime kill failed for %s: %s: %s",
                self._action_session_id,
                type(exc).__name__,
                exc,
            )
        get_resource_pool().unregister(self._actor_name)


async def start_openpi_action_ray_runtime(
    *,
    action_session_id: str,
    base_model: str,
    spec: OpenPIFastRuntimeSpec,
) -> OpenPIActionRayRuntimeClient:
    ensure_openpi_ray_initialized()

    actor_name = _action_actor_name(
        action_session_id=action_session_id,
        base_model=base_model,
        worker_module=spec.worker_module,
    )
    actor = OpenPIActionRayRuntimeActor.options(
        name=actor_name,
        namespace=RAY_NAMESPACE,
        runtime_env={"env_vars": _openpi_runtime_env_vars()},
    ).remote(
        action_session_id=action_session_id,
        base_model=base_model,
        spec=spec,
    )
    client = OpenPIActionRayRuntimeClient(
        actor=actor,
        actor_name=actor_name,
        spec=spec,
        action_session_id=action_session_id,
        ready_timeout_s=_actor_ready_timeout_s(spec),
    )
    try:
        metadata = await client.ready()
    except Exception:
        await client.close()
        raise

    pool = get_resource_pool()
    pool.register(
        actor_name=actor_name,
        actor_type=ActorType.OPENPI,
        num_gpus=1,
        actor_handle=actor,
        namespace=RAY_NAMESPACE,
        base_model=base_model,
        session_id=action_session_id,
        node_id=metadata.get("node_id"),
        metadata={
            "worker_module": spec.worker_module,
            "action_session_id": action_session_id,
            "actor_id": metadata.get("actor_id"),
            "node_ip": metadata.get("node_ip"),
            "pid": metadata.get("pid"),
            "cuda_visible_devices": metadata.get("cuda_visible_devices"),
        },
    )
    pool.mark_ready(actor_name)
    pool.touch(actor_name)
    return client
