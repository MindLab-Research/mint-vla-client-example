from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from typing import Any

from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, config as server_config, otel_env_vars

logger = logging.getLogger(__name__)
_ACTOR_HANDLE = None

def _reset_cached_actor_handle() -> None:
    global _ACTOR_HANDLE
    _ACTOR_HANDLE = None

from ..ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator
_register_ray_reconnect_invalidator(_reset_cached_actor_handle)


def _runtime_env_overrides() -> dict[str, str]:
    out: dict[str, str] = {}

    direct_keys = (
        "MINT_QUEUE_EXECUTION_RUNTIME_ACTOR_NAME",
        "MINT_QUEUE_SUPERVISOR_ACTOR_NAME",
        "TINKER_API_WORK_QUEUE_ACTOR_NAME",
        "MINT_API_WORK_QUEUE_ACTOR_NAME",
        "MINT_API_WORK_QUEUE_PINNED_NODE_IP",
        "MINT_FUTURE_STORE_ACTOR_NAME",
        "MINT_GATEWAY_SESSION_STORE_ACTOR_NAME",
        "MINT_SAMPLING_SESSION_STORE_ACTOR_NAME",
        "MINT_TRAINING_SESSION_STORE_ACTOR_NAME",
        "MINT_SESSION_HEARTBEAT_ACTOR_NAME",
        "MINT_SESSION_INDEX_ACTOR_NAME",
        "MINT_RESOURCE_POOL_ACTOR_NAME",
        "MINT_OWNER_RUNTIME_SUPERVISOR_ACTOR_NAME",
        "MINT_TRAINING_CLEANUP_EXECUTOR_ACTOR_NAME",
        "MINT_SAMPLING_CLEANUP_EXECUTOR_ACTOR_NAME",
        "MINT_API_WORK_QUEUE_ACTOR_MAX_CONCURRENCY",
        "MINT_SUPPORTED_MODELS",
        "TINKER_SUPPORTED_MODELS",
        "ALLOW_UNSUPPORTED_MODELS",
        "TINKER_ENABLE_MULTI_LORA",
        "MINT_MODEL_NODE_IPS_JSON",
        "MINT_DENSE_MODEL_NODE_IPS_JSON",
        "MINT_MEGATRON_MODEL_NODE_IPS_JSON",
        "MINT_VLLM_MODEL_NODE_IPS_JSON",
        "MINT_VLLM_PINNED_NODE_IP_JSON",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "OPENPI_DATA_HOME",
        "XDG_CACHE_HOME",
        "TMPDIR",
        "TINKER_RAY_NAMESPACE",
        "MINT_RAY_NAMESPACE",
        "MINT_K2_INFER_VOLC_RESOURCE_QUEUE_ID",
        "MINT_VLLM_PINNED_NODE_IP_JSON",
        "MINT_VLLM_VOLC_RESOURCE_QUEUE_ID",
        "MINT_DENSE_MODEL_NODE_IPS_JSON",
        "MINT_MODEL_NODE_IPS_JSON",
        "MINT_VLLM_MODEL_NODE_IPS_JSON",
        "MINT_MEGATRON_MODEL_NODE_IPS_JSON",
        "MINT_MEGATRON_NODE_IPS_CSV",
        "MINT_MEGATRON_VOLC_RESOURCE_QUEUE_ID",
        "MINT_SUPPORTED_MODELS",
        "MINT_RUNTIME_OBSERVABILITY_FLUSH_S",
    )
    for key in direct_keys:
        value = os.environ.get(key, "").strip()
        if value:
            out[key] = value
    for key, value in os.environ.items():
        if key.startswith("MINT_OPENPI_") and value.strip():
            out[key] = value.strip()

    # Canonicalize legacy actor-name envs before detached runtime actors inherit them.
    compat_keys = {
        "MINT_API_WORK_QUEUE_ACTOR_NAME": "TINKER_API_WORK_QUEUE_ACTOR_NAME",
        "MINT_CAPACITY_MANAGER_ACTOR_NAME": "TINKER_CAPACITY_MANAGER_ACTOR_NAME",
    }
    for canonical, legacy in compat_keys.items():
        value = os.environ.get(canonical, "").strip() or os.environ.get(legacy, "").strip()
        if value:
            out[canonical] = value

    out.setdefault(
        "MINT_API_WORK_QUEUE_ACTOR_NAME",
        str(getattr(server_config, "api_work_queue_actor_name", "tinker_api_work_queue")),
    )
    out.setdefault(
        "MINT_CAPACITY_MANAGER_ACTOR_NAME",
        str(getattr(server_config, "capacity_manager_actor_name", "tinker_capacity_manager")),
    )
    return out


def _ray_namespace() -> str:
    env_ns = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _actor_name() -> str:
    return os.environ.get("MINT_QUEUE_EXECUTION_RUNTIME_ACTOR_NAME", "tinker_queue_execution_runtime")


async def _await_ray_ref(ref: Any) -> Any:
    if hasattr(ref, "__await__"):
        return await ref
    to_future = getattr(ref, "future", None)
    if callable(to_future):
        fut = to_future()
        if isinstance(fut, asyncio.Future):
            return await fut
        if isinstance(fut, concurrent.futures.Future):
            return await asyncio.wrap_future(fut)
        if hasattr(fut, "__await__"):
            return await fut
    raise TypeError(f"Ray ref is not awaitable: {type(ref)}")


async def _restore_sampling_sessions_for_worker(inference_manager) -> int:
    from .sampling_session_store import async_list_sampling_sessions

    restored = 0
    for info in await async_list_sampling_sessions():
        try:
            if inference_manager.restore_sampling_session(info):
                restored += 1
        except Exception as e:
            logger.warning(
                "queue execution runtime failed to restore sampling session %r: %s: %s",
                info.get("session_id") if isinstance(info, dict) else None,
                type(e).__name__,
                e,
            )
    return restored


async def _initialize_execution_bindings() -> dict[str, Any]:
    from ..config import config
    from ..routes import action_sampling, mint, sampling, service, training, weights
    from .action_session_manager import ActionSessionRouter
    from .session_manager import DEFAULT_INACTIVITY_TIMEOUT, SessionManager
    from .training_engine_router import TrainingEngineRouter
    from .training_session_manager import TrainingSessionManager

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
    restored_sampling_sessions = await _restore_sampling_sessions_for_worker(inference_manager)

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
    mint.training_manager = train_manager
    mint.training_engine = train_engine
    mint.action_session_manager = action_manager
    weights.training_manager = train_manager
    weights.training_engine = train_engine
    weights.inference_manager = inference_manager

    return {
        "restored_sampling_sessions": restored_sampling_sessions,
        "multi_model_enabled": bool(config.enable_multi_lora),
    }


def _get_or_create_actor():
    import ray

    global _ACTOR_HANDLE
    name = _actor_name()
    namespace = _ray_namespace()
    try:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except ValueError:
        pass

    @ray.remote(num_cpus=0, max_concurrency=32)
    class _QueueExecutionRuntimeActor:
        def __init__(self) -> None:
            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._started_at = time.time()
            self._runtime_initialized = False
            self._desired_workers = 0
            self._last_error = None
            self._last_started_at = None
            self._observability_flush_task: asyncio.Task | None = None
            self._observability_flush_interval_s = max(
                5.0,
                float(os.environ.get("MINT_RUNTIME_OBSERVABILITY_FLUSH_S", "15.0")),
            )
            self._lock = asyncio.Lock()

        async def _ensure_observability_flush_task(self) -> None:
            if self._observability_flush_task is not None and not self._observability_flush_task.done():
                return
            self._observability_flush_task = asyncio.create_task(self._observability_flush_loop())

        async def _observability_flush_loop(self) -> None:
            from .runtime_observability import runtime_observability

            while True:
                try:
                    runtime_observability.flush_otel()
                except Exception as e:
                    logger.debug("queue_execution_runtime observability flush failed: %s", e)
                await asyncio.sleep(self._observability_flush_interval_s)

        async def ensure_started(self, *, num_workers: int) -> dict[str, Any]:
            from .api_work_queue import api_work_queue
            from .api_work_queue_dispatch import register_api_work_queue_executors
            from .capacity_manager import capacity_manager
            from .future_store import future_store

            async with self._lock:
                try:
                    if not self._runtime_initialized:
                        await _initialize_execution_bindings()
                        self._runtime_initialized = True
                    await self._ensure_observability_flush_task()
                    await capacity_manager.async_ensure_ready()
                    await future_store.async_ensure_started()
                    await api_work_queue.async_ensure_started()
                    register_api_work_queue_executors(api_work_queue)
                    self._desired_workers = max(1, int(num_workers))
                    await api_work_queue.start_workers(num_workers=self._desired_workers)
                    await api_work_queue.wait_until_execution_ready(timeout_s=120.0)
                    self._last_started_at = time.time()
                    self._last_error = None
                except Exception as e:
                    self._last_error = f"{type(e).__name__}: {e}"
                    raise
            return await self.health_snapshot()

        async def get_tokenizer_info(self, *, model_id: str) -> dict[str, Any]:
            from ..routes import training

            if not self._runtime_initialized:
                await _initialize_execution_bindings()
                self._runtime_initialized = True
            session, _snapshot = await training._get_training_session_for_request(str(model_id))
            if session is None:
                raise RuntimeError(f"Model {model_id!r} not found")
            return await training.training_engine.get_tokenizer_info(session)

        async def health_snapshot(self) -> dict[str, Any]:
            from ..routes import service, training
            from .api_work_queue import api_work_queue
            from .runtime_observability import runtime_observability

            sampling_sessions = None
            manager = getattr(service, "session_manager", None)
            if manager is not None:
                snapshot = getattr(manager, "observability_snapshot", None)
                if callable(snapshot):
                    try:
                        sampling_sessions = snapshot()
                    except Exception as e:
                        sampling_sessions = {"error": f"{type(e).__name__}: {e}"}

            training_sessions = None
            train_manager = getattr(training, "training_manager", None)
            if train_manager is not None:
                snapshot = getattr(train_manager, "observability_snapshot", None)
                if callable(snapshot):
                    try:
                        training_sessions = snapshot()
                    except Exception as e:
                        training_sessions = {"error": f"{type(e).__name__}: {e}"}

            return {
                "actor_name": _actor_name(),
                "namespace": _ray_namespace(),
                "started_at": float(self._started_at),
                "runtime_initialized": bool(self._runtime_initialized),
                "desired_workers": int(self._desired_workers),
                "last_started_at": self._last_started_at,
                "last_error": self._last_error,
                "running": bool(getattr(api_work_queue, "_running", False)),
                "consumer_job_id": getattr(api_work_queue, "_consumer_job_id", None),
                "consumer_generation_id": getattr(api_work_queue, "_consumer_generation_id", None),
                "execution_ready": bool(
                    getattr(getattr(api_work_queue, "_execution_ready_event", None), "is_set", lambda: False)()
                ),
                "execution_ready_generation_id": getattr(api_work_queue, "_execution_ready_generation_id", None),
                "execution_ready_at": getattr(api_work_queue, "_execution_ready_at", None),
                "local_worker_tasks": len([t for t in getattr(api_work_queue, "_worker_tasks", []) if not t.done()]),
                "sampling_sessions": sampling_sessions,
                "training_sessions": training_sessions,
                "runtime_observability": runtime_observability.snapshot(),
            }

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    apply_detached_actor_resources(options, ray)
    env = otel_env_vars()
    env.update(_runtime_env_overrides())
    options["runtime_env"] = actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=env)

    try:
        created = _QueueExecutionRuntimeActor.options(**options).remote()
        try:
            ray.get(created.health_snapshot.remote())
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


class QueueExecutionRuntime:
    def __init__(self) -> None:
        self._ray_actor = None

    def _get_ray_actor(self):
        import ray

        global _ACTOR_HANDLE
        if self._ray_actor is not None:
            return self._ray_actor
        if _ACTOR_HANDLE is not None:
            self._ray_actor = _ACTOR_HANDLE
            return self._ray_actor
        if not ray.is_initialized():
            raise RuntimeError("Ray not initialized")
        self._ray_actor = _get_or_create_actor()
        return self._ray_actor

    async def async_ensure_started(self, *, num_workers: int, timeout_s: float = 120.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        out = await asyncio.wait_for(
            _await_ray_ref(actor.ensure_started.remote(num_workers=int(num_workers))),
            timeout=float(timeout_s),
        )
        if not isinstance(out, dict):
            raise TypeError(f"QueueExecutionRuntime.ensure_started returned non-dict: {type(out)}")
        return out

    async def async_get_tokenizer_info(self, *, model_id: str, timeout_s: float = 60.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        out = await asyncio.wait_for(
            _await_ray_ref(actor.get_tokenizer_info.remote(model_id=str(model_id))),
            timeout=float(timeout_s),
        )
        if not isinstance(out, dict):
            raise TypeError(f"QueueExecutionRuntime.get_tokenizer_info returned non-dict: {type(out)}")
        return out

    async def async_health_snapshot(self, *, timeout_s: float = 30.0) -> dict[str, Any]:
        actor = self._get_ray_actor()
        out = await asyncio.wait_for(_await_ray_ref(actor.health_snapshot.remote()), timeout=float(timeout_s))
        if not isinstance(out, dict):
            raise TypeError(f"QueueExecutionRuntime.health_snapshot returned non-dict: {type(out)}")
        return out


queue_execution_runtime = QueueExecutionRuntime()
