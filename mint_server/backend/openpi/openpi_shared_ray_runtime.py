from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import structlog
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from mint_server.config import RAY_NAMESPACE
from mint_server.backend.ray_cluster.async_ray_control import async_get_ray_ref
from mint_server.backend.openpi.openpi_fast_runtime import (
    OpenPIFastRuntimeSpec,
    OpenPIFastWorkerClient,
    OpenPIFastWorkerError,
    OpenPIFastWorkerProtocolError,
)
from mint_server.backend.openpi.openpi_ray_runtime import (
    _action_session_state_root,
    _actor_ready_timeout_s,
    _openpi_runtime_env_vars,
    _ray_timeout,
    ensure_openpi_ray_initialized,
)
from mint_server.backend.actors.model_actor_supervisor import ActorType, get_model_actor_supervisor
from mint_server.backend.actors.node_placement import (
    assert_node_ip_capacity,
    parse_model_gpu_placement,
)


logger = structlog.get_logger(__name__)

OPENPI_SHARED_TEMPLATE_SESSION_ID = "__mint_initial__"

_SHARED_ACTOR_PREFIX = "mint_openpi_shared_"
_SHARED_POOL_LOCK = threading.Lock()


@dataclass
class _SharedActorEntry:
    actor_name: str
    actor: Any
    pool_key: dict[str, Any]
    metadata: dict[str, Any] | None = None


_SHARED_ACTORS: dict[str, _SharedActorEntry] = {}


def _normalize_pool_key(
    *,
    spec: OpenPIFastRuntimeSpec,
    session: Any,
    config_name: str,
    model_config: Any,
) -> dict[str, Any]:
    return {
        "base_model": str(getattr(session, "base_model", "") or ""),
        "worker_module": str(spec.worker_module),
        "config_name": str(config_name),
        "action_dim": int(getattr(model_config, "action_dim", 0) or 0),
        "action_horizon": int(getattr(model_config, "action_horizon", 0) or 0),
        "max_model_len": int(getattr(model_config, "max_model_len", 0) or 0),
        "startup_timeout_s": float(spec.startup_timeout_s),
        "request_timeout_s": float(spec.request_timeout_s),
        "create_session_timeout_s": float(spec.create_session_timeout_s),
        "save_weights_timeout_s": float(spec.save_weights_timeout_s),
        "load_weights_timeout_s": float(spec.load_weights_timeout_s),
    }


def _shared_actor_name(pool_key: dict[str, Any]) -> str:
    payload = json.dumps(pool_key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{_SHARED_ACTOR_PREFIX}{hashlib.sha1(payload).hexdigest()[:12]}"


def _preferred_openpi_node_ip(base_model: str, actor_name: str) -> str | None:
    lookup_keys = [
        str(base_model).strip(),
        str(base_model).strip().lower(),
        str(actor_name).strip(),
        str(actor_name).strip().lower(),
    ]
    context = f"[OpenPISharedRuntime] node pinning model={base_model!r} actor={actor_name!r}"
    placement = parse_model_gpu_placement(
        raw_json=os.environ.get("MINT_MODEL_PLACEMENT_JSON"),
        lookup_keys=lookup_keys,
        env_var_name="MINT_MODEL_PLACEMENT_JSON",
        context=context,
        replica=0,
    )
    if placement is None:
        return None
    if len(placement.slices) != 1:
        raise RuntimeError(
            f"[OpenPISharedRuntime] node pinning model={base_model!r} actor={actor_name!r}: "
            f"expected exactly 1 placement slice for single-GPU actor, got {len(placement.slices)}"
        )
    if placement.total_gpus != 1:
        raise RuntimeError(
            f"[OpenPISharedRuntime] node pinning model={base_model!r} actor={actor_name!r}: "
            f"expected exactly 1 GPU, got {placement.total_gpus}"
        )
    assert_node_ip_capacity(
        required_gpus_by_node_ip={placement.slices[0].node_ip: 1},
        context=f"[OpenPISharedRuntime] node pinning model={base_model!r} actor={actor_name!r}",
    )
    return placement.slices[0].node_ip


def _single_node_actor_options(*, base_model: str, actor_name: str) -> dict[str, Any]:
    preferred_ip = _preferred_openpi_node_ip(base_model, actor_name)
    if not preferred_ip:
        return {}
    node_map = {
        str(node.get("NodeManagerAddress") or ""): str(node.get("NodeID") or "")
        for node in ray.nodes()
        if node.get("Alive")
    }
    node_id = node_map.get(preferred_ip)
    if not node_id:
        raise RuntimeError(
            f"[OpenPISharedRuntime] pinned node_ip={preferred_ip} for actor={actor_name!r} "
            "is not an alive Ray node"
        )
    logger.info(
        "[OpenPISharedRuntime] pin model=%s actor=%s node_ip=%s node_id=%s",
        base_model,
        actor_name,
        preferred_ip,
        node_id,
    )
    return {
        "resources": {f"node:{preferred_ip}": 0.001},
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        ),
    }


def _template_session_id(actor_metadata: dict[str, Any] | None) -> str:
    metadata = dict(actor_metadata or {})
    scope = str(metadata.get("actor_name") or "").strip()
    if not scope:
        scope = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()[:12]
    return f"{OPENPI_SHARED_TEMPLATE_SESSION_ID}:{digest}"


def clear_openpi_shared_runtime_pool() -> None:
    with _SHARED_POOL_LOCK:
        entries = list(_SHARED_ACTORS.values())
        _SHARED_ACTORS.clear()

    pool = get_model_actor_supervisor()
    for entry in entries:
        pool.unregister(entry.actor_name)
        if not ray.is_initialized():
            continue
        try:
            ray.get(entry.actor.shutdown.remote(), timeout=5.0)
        except Exception:
            pass
        try:
            ray.kill(entry.actor, no_restart=True)
        except Exception:
            pass


def _drop_shared_actor_entry(actor_name: str, *, actor: Any | None = None) -> _SharedActorEntry | None:
    with _SHARED_POOL_LOCK:
        entry = _SHARED_ACTORS.get(actor_name)
        if entry is None:
            return None
        if actor is not None and entry.actor is not actor:
            return None
        return _SHARED_ACTORS.pop(actor_name, None)


async def _cleanup_failed_shared_actor_start(*, actor_name: str, actor: Any) -> list[str]:
    errors: list[str] = []
    await _pool_call("unregister", actor_name)
    if not ray.is_initialized():
        return errors

    try:
        await async_get_ray_ref(actor.shutdown.remote(), timeout_s=5.0)
    except Exception as exc:
        errors.append(
            f"OpenPI shared actor shutdown failed for {actor_name}: {type(exc).__name__}: {exc}"
        )
    try:
        await asyncio.to_thread(ray.kill, actor, no_restart=True)
    except Exception as exc:
        errors.append(
            f"OpenPI shared actor kill failed for {actor_name}: {type(exc).__name__}: {exc}"
        )
    return errors


async def _pool_call(method_name: str, *args: Any, **kwargs: Any) -> Any:
    pool = get_model_actor_supervisor()
    method = getattr(pool, method_name)
    return await asyncio.to_thread(method, *args, **kwargs)


class OpenPISharedRuntimeCore:
    def __init__(
        self,
        *,
        spec: OpenPIFastRuntimeSpec,
        runtime_factory: Callable[[OpenPIFastRuntimeSpec], Any] | None = None,
        actor_metadata: dict[str, Any] | None = None,
        template_reusable: bool = True,
    ) -> None:
        self._spec = spec
        self._runtime_factory = runtime_factory or OpenPIFastWorkerClient.start
        self._actor_metadata = dict(actor_metadata or {})
        self._template_session_id = _template_session_id(self._actor_metadata)
        self._template_reusable = bool(template_reusable)
        self._runtime: OpenPIFastWorkerClient | Any | None = None
        self._session_payloads: dict[str, dict[str, Any]] = {}
        self._initialized_sessions: set[str] = set()
        self._current_session_id: str | None = None
        self._create_session_response: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    async def _ensure_runtime(self) -> Any:
        if self._runtime is None:
            runtime_or_awaitable = self._runtime_factory(self._spec)
            if inspect.isawaitable(runtime_or_awaitable):
                self._runtime = await runtime_or_awaitable
            else:
                self._runtime = runtime_or_awaitable
        return self._runtime

    async def register_session(
        self,
        session_id: str,
        create_payload: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._lock:
            payload = dict(create_payload)
            if self._create_session_response is not None:
                self._session_payloads[session_id] = payload
                if not self._template_reusable:
                    self._initialized_sessions.discard(session_id)
                return dict(self._create_session_response)

            runtime = await self._ensure_runtime()
            try:
                response = await runtime.request(
                    "create_session",
                    payload,
                    timeout_s=self._spec.create_session_timeout_s,
                )
                if not isinstance(response, dict):
                    raise TypeError(
                        "OpenPI shared runtime create_session returned non-dict payload: "
                        f"{type(response)}"
                    )

                await runtime.request(
                    "save_session_state",
                    {"session_id": self._template_session_id},
                    timeout_s=self._spec.request_timeout_s,
                )
                await runtime.request(
                    "save_session_state",
                    {"session_id": session_id},
                    timeout_s=self._spec.request_timeout_s,
                )
            except Exception:
                self._runtime = None
                try:
                    await runtime.close()
                except Exception as cleanup_exc:
                    logger.warning(
                        "OpenPI shared runtime cleanup failed after create_session error for %s: %s: %s",
                        session_id,
                        type(cleanup_exc).__name__,
                        cleanup_exc,
                    )
                raise

            self._session_payloads[session_id] = payload
            self._initialized_sessions.update({self._template_session_id, session_id})
            self._current_session_id = session_id
            self._create_session_response = dict(response)
            return dict(response)

    async def _ensure_session_loaded(
        self,
        session_id: str,
        *,
        timeout_s: float | None,
    ) -> None:
        if session_id not in self._session_payloads:
            raise ValueError(
                f"OpenPI shared runtime session is not registered for {session_id!r}"
            )
        if self._current_session_id == session_id:
            return

        runtime = await self._ensure_runtime()
        effective_timeout = self._spec.request_timeout_s if timeout_s is None else float(timeout_s)
        if self._current_session_id is not None:
            await runtime.request(
                "save_session_state",
                {"session_id": self._current_session_id},
                timeout_s=effective_timeout,
            )
            self._initialized_sessions.add(self._current_session_id)

        if session_id in self._initialized_sessions:
            load_session_id = session_id
            await runtime.request(
                "load_session_state",
                {"session_id": load_session_id},
                timeout_s=effective_timeout,
            )
            self._current_session_id = session_id
            return

        if self._template_reusable:
            await runtime.request(
                "load_session_state",
                {"session_id": self._template_session_id},
                timeout_s=effective_timeout,
            )
            self._current_session_id = session_id
            return

        payload = dict(self._session_payloads[session_id])
        response = await runtime.request(
            "create_session",
            payload,
            timeout_s=self._spec.create_session_timeout_s,
        )
        if not isinstance(response, dict):
            raise TypeError(
                "OpenPI shared runtime create_session returned non-dict payload: "
                f"{type(response)}"
            )
        await runtime.request(
            "save_session_state",
            {"session_id": session_id},
            timeout_s=effective_timeout,
        )
        self._initialized_sessions.add(session_id)
        self._current_session_id = session_id

    async def request_for_session(
        self,
        session_id: str,
        op: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if op == "create_session":
            return await self.register_session(session_id, payload or {})

        async with self._lock:
            runtime = await self._ensure_runtime()
            effective_timeout = self._spec.request_timeout_s if timeout_s is None else float(timeout_s)

            if op == "shutdown":
                if session_id in self._session_payloads and self._current_session_id == session_id:
                    await runtime.request(
                        "save_session_state",
                        {"session_id": session_id},
                        timeout_s=effective_timeout,
                    )
                    self._initialized_sessions.add(session_id)
                    self._current_session_id = None
                self._session_payloads.pop(session_id, None)
                self._initialized_sessions.discard(session_id)
                return {"stopped": True, **self.describe()}

            if self._create_session_response is None:
                raise RuntimeError(
                    "OpenPI shared runtime received a non-create_session request before initialization"
                )

            await self._ensure_session_loaded(session_id, timeout_s=effective_timeout)
            response = await runtime.request(
                op,
                payload or {},
                timeout_s=timeout_s,
            )
            if not isinstance(response, dict):
                raise TypeError(
                    f"OpenPI shared runtime request returned non-dict payload: {type(response)}"
                )
            return response

    def describe(self) -> dict[str, Any]:
        return {
            **self._actor_metadata,
            "worker_module": self._spec.worker_module,
            "current_session_id": self._current_session_id,
            "known_session_ids": sorted(self._session_payloads),
        }

    async def shutdown(self) -> None:
        async with self._lock:
            runtime = self._runtime
            self._runtime = None
            self._session_payloads.clear()
            self._initialized_sessions.clear()
            self._current_session_id = None
            if runtime is not None:
                await runtime.close()


@ray.remote(num_gpus=1, max_concurrency=1)
class OpenPISharedRayRuntimeActor:
    def __init__(
        self,
        *,
        actor_name: str,
        pool_key: dict[str, Any],
        spec: OpenPIFastRuntimeSpec,
        template_reusable: bool = True,
    ) -> None:
        self._actor_name = actor_name
        self._pool_key = dict(pool_key)
        self._core = OpenPISharedRuntimeCore(
            spec=spec,
            actor_metadata={
                "actor_name": actor_name,
                "pool_key": dict(pool_key),
            },
            template_reusable=template_reusable,
        )

    async def ready_metadata(self) -> dict[str, Any]:
        return self.describe()

    def describe(self) -> dict[str, Any]:
        context = ray.get_runtime_context()
        get_actor_id = getattr(context, "get_actor_id", None)
        get_node_id = getattr(context, "get_node_id", None)
        return {
            **self._core.describe(),
            "actor_id": str(get_actor_id()) if callable(get_actor_id) else None,
            "node_id": str(get_node_id()) if callable(get_node_id) else None,
            "node_ip": ray.util.get_node_ip_address(),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    async def register_session(
        self,
        session_id: str,
        create_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._core.register_session(session_id, create_payload)

    async def request_for_session(
        self,
        session_id: str,
        op: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await self._core.request_for_session(
            session_id,
            op,
            payload or {},
            timeout_s=timeout_s,
        )

    async def shutdown(self) -> dict[str, Any]:
        await self._core.shutdown()
        return {"stopped": True, **self.describe()}


class OpenPISharedRayRuntimeClient:
    def __init__(
        self,
        *,
        actor: Any,
        actor_name: str,
        spec: OpenPIFastRuntimeSpec,
        session_id: str,
        ready_timeout_s: float,
        owns_started_actor: bool = False,
    ) -> None:
        self._actor = actor
        self._actor_name = actor_name
        self._spec = spec
        self._session_id = session_id
        self._ready_timeout_s = float(ready_timeout_s)
        self._closed = False
        self._metadata: dict[str, Any] | None = None
        self._owns_started_actor = bool(owns_started_actor)
        self._bootstrap_session_pending = bool(owns_started_actor)
        self._cleanup_attempted = False

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

    def _refresh_actor_handle(self) -> None:
        if not ray.is_initialized():
            return
        try:
            actor = ray.get_actor(self._actor_name, namespace=RAY_NAMESPACE)
        except ValueError:
            return
        self._actor = actor

    async def _ray_get(self, ref: Any, *, timeout_s: float | None) -> Any:
        try:
            return await async_get_ray_ref(ref, timeout_s=_ray_timeout(timeout_s))
        except ray.exceptions.GetTimeoutError as exc:
            raise OpenPIFastWorkerProtocolError(
                f"OpenPI shared Ray runtime timed out for session {self._session_id!r} after {timeout_s}s"
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
        self._refresh_actor_handle()
        metadata = await self._ray_get(
            self._actor.ready_metadata.remote(),
            timeout_s=self._ready_timeout_s,
        )
        if not isinstance(metadata, dict):
            raise TypeError(
                f"OpenPI shared Ray runtime ready payload must be dict, got {type(metadata)}"
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
            raise OpenPIFastWorkerProtocolError(
                f"OpenPI shared Ray runtime client is closed for session {self._session_id!r}"
            )

        self._refresh_actor_handle()
        await _pool_call("mark_inflight", self._actor_name, +1)
        try:
            if op == "create_session":
                ref = self._actor.register_session.remote(self._session_id, payload or {})
            else:
                ref = self._actor.request_for_session.remote(
                    self._session_id,
                    op,
                    payload or {},
                    timeout_s=timeout_s,
                )
            result = await self._ray_get(ref, timeout_s=timeout_s)
        finally:
            await _pool_call("mark_inflight", self._actor_name, -1)

        if not isinstance(result, dict):
            raise TypeError(
                f"OpenPI shared Ray runtime request returned non-dict payload: {type(result)}"
            )
        if op == "create_session":
            self._bootstrap_session_pending = False
        if op == "shutdown":
            await _pool_call("set_session", self._actor_name, None)
            if not list(result.get("known_session_ids") or []):
                _drop_shared_actor_entry(self._actor_name, actor=self._actor)
                cleanup_errors = await _cleanup_failed_shared_actor_start(
                    actor_name=self._actor_name,
                    actor=self._actor,
                )
                for note in cleanup_errors:
                    logger.warning("s")
        else:
            await _pool_call("set_session", self._actor_name, self._session_id)
        await _pool_call("touch", self._actor_name)
        return result

    async def describe(self) -> dict[str, Any]:
        self._refresh_actor_handle()
        metadata = await self._ray_get(self._actor.describe.remote(), timeout_s=5.0)
        if not isinstance(metadata, dict):
            raise TypeError(
                f"OpenPI shared Ray runtime metadata must be dict, got {type(metadata)}"
            )
        self._metadata = metadata
        return metadata

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._owns_started_actor or not self._bootstrap_session_pending or self._cleanup_attempted:
            return

        self._cleanup_attempted = True
        dropped = _drop_shared_actor_entry(self._actor_name, actor=self._actor)
        if dropped is None:
            return

        cleanup_errors = await _cleanup_failed_shared_actor_start(
            actor_name=self._actor_name,
            actor=self._actor,
        )
        for note in cleanup_errors:
            logger.warning("s")


async def start_openpi_shared_ray_runtime(
    *,
    session: Any,
    spec: OpenPIFastRuntimeSpec,
    config_name: str,
    model_config: Any,
    template_reusable: bool = True,
) -> OpenPISharedRayRuntimeClient:
    ensure_openpi_ray_initialized()

    pool_key = _normalize_pool_key(
        spec=spec,
        session=session,
        config_name=config_name,
        model_config=model_config,
    )
    actor_name = _shared_actor_name(pool_key)
    owns_started_actor = False

    with _SHARED_POOL_LOCK:
        entry = _SHARED_ACTORS.get(actor_name)
        actor = entry.actor if entry is not None else None
        runtime_env_vars = {
            **_openpi_runtime_env_vars(),
            "MINT_OPENPI_FAST_ACTION_SESSION_STATE_ROOT": _action_session_state_root(actor_name),
        }
        if ray.is_initialized():
            try:
                actor = ray.get_actor(actor_name, namespace=RAY_NAMESPACE)
            except ValueError:
                actor = None
        if actor is None:
            owns_started_actor = True
            actor = OpenPISharedRayRuntimeActor.options(
                name=actor_name,
                namespace=RAY_NAMESPACE,
                lifetime="detached",
                runtime_env={"env_vars": runtime_env_vars},
                **_single_node_actor_options(
                    base_model=str(getattr(session, "base_model", "") or ""),
                    actor_name=actor_name,
                ),
            ).remote(
                actor_name=actor_name,
                pool_key=pool_key,
                spec=spec,
                template_reusable=template_reusable,
            )
        if entry is None:
            entry = _SharedActorEntry(
                actor_name=actor_name,
                actor=actor,
                pool_key=dict(pool_key),
            )
            _SHARED_ACTORS[actor_name] = entry
        else:
            entry.actor = actor
            entry.pool_key = dict(pool_key)

    client = OpenPISharedRayRuntimeClient(
        actor=entry.actor,
        actor_name=actor_name,
        spec=spec,
        session_id=str(session.model_id),
        ready_timeout_s=_actor_ready_timeout_s(spec),
        owns_started_actor=owns_started_actor,
    )
    try:
        metadata = await client.ready()
    except Exception as exc:
        _drop_shared_actor_entry(actor_name)
        cleanup_errors = await _cleanup_failed_shared_actor_start(
            actor_name=actor_name,
            actor=entry.actor,
        )
        for note in cleanup_errors:
            exc.add_note(note)
        raise

    with _SHARED_POOL_LOCK:
        current = _SHARED_ACTORS.get(actor_name)
        if current is not None:
            current.metadata = dict(metadata)

    await _pool_call(
        "register",
        actor_name=actor_name,
        actor_type=ActorType.OPENPI,
        num_gpus=1,
        actor_handle=entry.actor,
        namespace=RAY_NAMESPACE,
        base_model=str(getattr(session, "base_model", "") or ""),
        session_id=metadata.get("current_session_id"),
        node_id=metadata.get("node_id"),
        metadata={
            "pool_key": dict(pool_key),
            "worker_module": spec.worker_module,
            "actor_id": metadata.get("actor_id"),
            "node_ip": metadata.get("node_ip"),
            "pid": metadata.get("pid"),
            "cuda_visible_devices": metadata.get("cuda_visible_devices"),
        },
    )
    await _pool_call("mark_ready", actor_name)
    await _pool_call("touch", actor_name)
    return client
