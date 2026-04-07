from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import shutil
import time
from typing import Any

from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, otel_env_vars

logger = logging.getLogger(__name__)
_ACTOR_HANDLE = None

def _reset_cached_actor_handle() -> None:
    global _ACTOR_HANDLE
    _ACTOR_HANDLE = None

from ..ray_utils import register_ray_reconnect_invalidator as _register_ray_reconnect_invalidator
_register_ray_reconnect_invalidator(_reset_cached_actor_handle)


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
    return os.environ.get("MINT_SAMPLING_CLEANUP_EXECUTOR_ACTOR_NAME", "tinker_sampling_cleanup_executor")


def _sampling_inactivity_timeout_s() -> float:
    raw = os.environ.get("TINKER_SESSION_INACTIVITY_TIMEOUT_S") or os.environ.get(
        "TINKER_INACTIVITY_TIMEOUT_S",
        "1800",
    )
    try:
        return max(0.0, float(raw))
    except Exception:
        logger.warning("Invalid sampling inactivity timeout=%r; defaulting to 1800s", raw)
        return 1800.0


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


def _cleanup_sampler_indices(sampler_id: str) -> None:
    try:
        from .session_index_store import delete_sampler_index, get_sampler_index, remove_sampler_from_session

        sampler_info = get_sampler_index(sampler_id)
        parent_session_id = None
        if isinstance(sampler_info, dict):
            raw_session_id = sampler_info.get("session_id")
            if isinstance(raw_session_id, str) and raw_session_id:
                parent_session_id = raw_session_id

        delete_sampler_index(sampler_id)
        if parent_session_id is not None:
            remove_sampler_from_session(parent_session_id, sampler_id)
    except Exception as e:
        logger.debug("Failed to cleanup sampler index %s: %s", sampler_id, e)


async def _remove_loaded_lora_if_last_reference(*, base_model: str, lora_int_id: int) -> None:
    import ray

    from .multi_lora_engine import PERSISTENT_NAMESPACE, _model_to_actor_name

    actor_name = _model_to_actor_name(base_model)
    try:
        actor = await asyncio.to_thread(ray.get_actor, actor_name, namespace=PERSISTENT_NAMESPACE)
    except Exception:
        return
    await asyncio.to_thread(ray.get, actor.remove_lora.remote(int(lora_int_id)), timeout=30)


async def cleanup_stale_sampling_sessions_once_impl(*, stale_after_s: float | None = None) -> list[str]:
    if stale_after_s is None:
        stale_after_s = _sampling_inactivity_timeout_s()
    stale_after_s = float(stale_after_s)
    if stale_after_s <= 0:
        return []

    from .future_store import future_store
    from .sampling_session_store import delete_sampling_session, list_sampling_sessions

    try:
        infos = await asyncio.to_thread(list_sampling_sessions)
    except Exception as e:
        logger.warning(
            "sampling cleanup executor skipped: failed to list detached sampling sessions: %s: %s",
            type(e).__name__,
            e,
        )
        return []

    now = time.time()
    adapter_refcounts: dict[str, int] = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        if bool(info.get("uses_base_model")):
            continue
        adapter_path = str(info.get("adapter_path") or "").strip()
        if adapter_path:
            adapter_refcounts[adapter_path] = adapter_refcounts.get(adapter_path, 0) + 1

    cleaned: list[str] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        session_id = str(info.get("session_id") or "").strip()
        base_model = str(info.get("base_model") or "").strip()
        adapter_path = str(info.get("adapter_path") or "").strip()
        uses_base_model = bool(info.get("uses_base_model"))
        inflight_requests = int(info.get("inflight_requests") or 0)
        last_activity = float(info.get("last_activity") or 0.0)
        lora_loaded = bool(info.get("lora_loaded"))
        lora_int_id = info.get("lora_int_id")
        if not session_id or not base_model:
            continue
        if uses_base_model:
            continue
        if inflight_requests > 0:
            continue
        if (now - last_activity) <= stale_after_s:
            continue

        reason = f"sampling inactivity (> {stale_after_s:.1f}s)"
        try:
            failed_request_ids = future_store.fail_sampling_requests_for_session(
                session_id,
                f"Sampling session terminated due to {reason}",
            )
            if failed_request_ids:
                logger.warning(
                    "[%s] failed pending sampling futures during detached cleanup (%s): request_ids=%s",
                    session_id,
                    reason,
                    failed_request_ids,
                )
        except Exception as e:
            logger.warning(
                "[%s] detached sampling cleanup aborted because future fail failed (%s): %s: %s",
                session_id,
                reason,
                type(e).__name__,
                e,
            )
            continue

        current_adapter_refcount = adapter_refcounts.get(adapter_path, 0) if adapter_path else 0
        should_unload = bool(adapter_path) and current_adapter_refcount <= 1
        if should_unload and lora_loaded and lora_int_id is not None:
            try:
                await _remove_loaded_lora_if_last_reference(base_model=base_model, lora_int_id=int(lora_int_id))
            except Exception as e:
                logger.warning(
                    "[%s] detached sampling cleanup actor actuation failed (%s): %s: %s",
                    session_id,
                    reason,
                    type(e).__name__,
                    e,
                )

        try:
            delete_sampling_session(session_id)
        except Exception as e:
            logger.warning(
                "[%s] detached sampling cleanup store delete failed (%s): %s: %s",
                session_id,
                reason,
                type(e).__name__,
                e,
            )

        _cleanup_sampler_indices(session_id)

        if adapter_path:
            try:
                if os.path.isdir(adapter_path) and os.path.basename(adapter_path).startswith("_ephemeral_"):
                    await asyncio.to_thread(shutil.rmtree, adapter_path)
            except Exception as e:
                logger.warning(
                    "[%s] detached sampling cleanup adapter delete failed (%s): %s: %s",
                    session_id,
                    reason,
                    type(e).__name__,
                    e,
                )

        if adapter_path and current_adapter_refcount > 0:
            adapter_refcounts[adapter_path] = current_adapter_refcount - 1

        cleaned.append(session_id)
        logger.warning(
            "[%s] detached sampling cleanup removed stale session: base_model=%s should_unload=%s adapter_refcount=%s",
            session_id,
            base_model,
            should_unload,
            current_adapter_refcount,
        )

    return cleaned


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

    @ray.remote(num_cpus=0, max_concurrency=16)
    class _SamplingCleanupExecutorActor:
        async def cleanup_stale_sessions_once(self, stale_after_s: float | None = None) -> dict[str, Any]:
            cleaned = await cleanup_stale_sampling_sessions_once_impl(stale_after_s=stale_after_s)
            return {"cleaned": list(cleaned)}

        def health_snapshot(self) -> dict[str, Any]:
            return {
                "actor_name": _actor_name(),
                "namespace": _ray_namespace(),
            }

    options: dict[str, Any] = {
        "name": name,
        "namespace": namespace,
        "lifetime": "detached",
    }
    apply_detached_actor_resources(options, ray)
    options["runtime_env"] = actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=otel_env_vars())

    try:
        created = _SamplingCleanupExecutorActor.options(**options).remote()
        try:
            ray.get(created.health_snapshot.remote())
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


class SamplingCleanupExecutor:
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

    def ensure_ready(self) -> dict[str, Any]:
        import ray

        actor = self._get_ray_actor()
        return ray.get(actor.health_snapshot.remote())

    async def async_cleanup_stale_sessions_once(self, *, stale_after_s: float | None = None) -> list[str]:
        actor = self._get_ray_actor()
        out = await _await_ray_ref(actor.cleanup_stale_sessions_once.remote(stale_after_s=stale_after_s))
        if not isinstance(out, dict):
            raise TypeError(f"SamplingCleanupExecutor returned non-dict: {type(out)}")
        cleaned = out.get("cleaned") or []
        return [str(session_id) for session_id in cleaned]


sampling_cleanup_executor = SamplingCleanupExecutor()
