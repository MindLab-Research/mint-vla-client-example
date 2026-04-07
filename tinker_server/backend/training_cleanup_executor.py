from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
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
    return os.environ.get("MINT_TRAINING_CLEANUP_EXECUTOR_ACTOR_NAME", "tinker_training_cleanup_executor")


def _training_heartbeat_stale_timeout_s() -> float:
    raw = os.environ.get("MINT_TRAINING_HEARTBEAT_STALE_S", "300")
    try:
        return max(0.0, float(raw))
    except Exception:
        logger.warning("Invalid MINT_TRAINING_HEARTBEAT_STALE_S=%r; defaulting to 300s", raw)
        return 300.0


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


async def _delete_shared_worker_session(*, actor_name: str, namespace: str, model_id: str) -> None:
    import ray

    try:
        actor = await asyncio.to_thread(ray.get_actor, actor_name, namespace=namespace)
    except Exception:
        return

    delete_session = getattr(actor, "delete_session", None)
    if delete_session is None:
        return

    await asyncio.to_thread(ray.get, delete_session.remote(model_id), timeout=30)


async def _kill_training_actor(*, actor_name: str, namespace: str, model_id: str) -> None:
    import ray

    from . import ray_kill

    try:
        actor = await asyncio.to_thread(ray.get_actor, actor_name, namespace=namespace)
    except Exception:
        return

    await asyncio.to_thread(
        ray_kill.kill,
        actor,
        reason="training_cleanup_executor",
        actor_name=actor_name,
        namespace=namespace,
        no_restart=True,
        model_id=model_id,
    )


async def cleanup_stale_training_sessions_once_impl(*, stale_after_s: float | None = None) -> list[str]:
    if stale_after_s is None:
        stale_after_s = _training_heartbeat_stale_timeout_s()
    stale_after_s = float(stale_after_s)
    if stale_after_s <= 0:
        return []

    from .future_store import future_store
    from .resource_pool import get_resource_pool
    from .session_heartbeat_store import session_heartbeat_store
    from .training_session_store import delete_training_session, list_training_sessions

    try:
        infos = await asyncio.to_thread(list_training_sessions)
    except Exception as e:
        logger.warning(
            "training cleanup executor skipped: failed to list detached training sessions: %s: %s",
            type(e).__name__,
            e,
        )
        return []

    actor_refcounts: dict[str, int] = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        actor_name = str(info.get("actor_name") or "").strip()
        if actor_name:
            actor_refcounts[actor_name] = actor_refcounts.get(actor_name, 0) + 1

    cleaned: list[str] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        model_id = str(info.get("model_id") or "").strip()
        session_id = str(info.get("session_id") or "").strip()
        actor_name = str(info.get("actor_name") or "").strip()
        namespace = str(info.get("namespace") or _ray_namespace()).strip() or _ray_namespace()
        if not model_id or not session_id:
            continue
        if not await session_heartbeat_store.async_is_stale(session_id, stale_after_s):
            continue

        reason = f"stale heartbeat (> {stale_after_s:.1f}s)"
        try:
            failed_request_ids = future_store.fail_training_requests_for_model(
                model_id,
                f"Training session terminated due to {reason}",
            )
            if failed_request_ids:
                logger.warning(
                    "[%s] failed pending training futures during detached cleanup (%s): request_ids=%s",
                    model_id,
                    reason,
                    failed_request_ids,
                )
        except Exception as e:
            logger.warning(
                "[%s] detached training cleanup aborted because future fail failed (%s): %s: %s",
                model_id,
                reason,
                type(e).__name__,
                e,
            )
            continue

        allow_actor_shutdown = bool(actor_name) and actor_refcounts.get(actor_name, 0) <= 1
        try:
            if actor_name:
                if allow_actor_shutdown:
                    await _kill_training_actor(actor_name=actor_name, namespace=namespace, model_id=model_id)
                else:
                    await _delete_shared_worker_session(actor_name=actor_name, namespace=namespace, model_id=model_id)
        except Exception as e:
            logger.warning(
                "[%s] detached training cleanup actor actuation failed (%s): %s: %s",
                model_id,
                reason,
                type(e).__name__,
                e,
            )

        try:
            delete_training_session(model_id)
        except Exception as e:
            logger.warning(
                "[%s] detached training cleanup store delete failed (%s): %s: %s",
                model_id,
                reason,
                type(e).__name__,
                e,
            )

        try:
            delete_heartbeat = getattr(session_heartbeat_store, "delete", None)
            if callable(delete_heartbeat):
                delete_heartbeat(session_id)
        except Exception:
            pass

        try:
            get_resource_pool().clear_session(model_id)
        except Exception:
            pass

        cleaned.append(model_id)
        logger.warning(
            "[%s] detached training cleanup removed stale session: session_id=%s actor_name=%s allow_actor_shutdown=%s actor_refcount=%s",
            model_id,
            session_id,
            actor_name or "<unknown>",
            allow_actor_shutdown,
            actor_refcounts.get(actor_name, 0) if actor_name else 0,
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
    class _TrainingCleanupExecutorActor:
        async def cleanup_stale_sessions_once(self, stale_after_s: float | None = None) -> dict[str, Any]:
            cleaned = await cleanup_stale_training_sessions_once_impl(stale_after_s=stale_after_s)
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
        created = _TrainingCleanupExecutorActor.options(**options).remote()
        try:
            ray.get(created.health_snapshot.remote())
            _ACTOR_HANDLE = created
        except Exception:
            _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE
    except Exception:
        _ACTOR_HANDLE = ray.get_actor(name, namespace=namespace)
        return _ACTOR_HANDLE


class TrainingCleanupExecutor:
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
            raise TypeError(f"TrainingCleanupExecutor returned non-dict: {type(out)}")
        cleaned = out.get("cleaned") or []
        return [str(model_id) for model_id in cleaned]


training_cleanup_executor = TrainingCleanupExecutor()
