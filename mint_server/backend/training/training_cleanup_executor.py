from __future__ import annotations

import asyncio
import structlog
import os

from mint_server.runtime_env import env_nonempty
from mint_server.backend.ray_cluster.async_ray_control import async_get_ray_ref

logger = structlog.get_logger(__name__)


def _ray_namespace() -> str:
    env_ns = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from mint_server.config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _training_heartbeat_stale_timeout_s() -> float:
    raw = os.environ.get("MINT_TRAINING_HEARTBEAT_STALE_S", "300")
    try:
        return max(0.0, float(raw))
    except Exception:
        logger.warning("invalid_training_heartbeat_stale_s", raw=raw, default_s=300)
        return 300.0


async def _delete_shared_worker_session(*, actor_name: str, namespace: str, model_id: str) -> None:
    import ray

    try:
        actor = await asyncio.to_thread(ray.get_actor, actor_name, namespace=namespace)
    except Exception:
        return

    delete_session = getattr(actor, "delete_session", None)
    if delete_session is None:
        return

    await async_get_ray_ref(delete_session.remote(model_id), timeout_s=30)


async def _kill_training_actor(*, actor_name: str, namespace: str, model_id: str) -> None:
    import ray

    import mint_server.backend.ray_cluster.ray_kill as ray_kill

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

    from mint_server.backend.stores.task_state_store import task_futures
    from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor
    from mint_server.backend.stores.session_heartbeat_store import session_heartbeat_store
    from mint_server.backend.stores.training_session_store import async_list_training_sessions, delete_training_session

    try:
        infos = await async_list_training_sessions()
    except Exception as e:
        logger.warning(
            "training cleanup executor skipped: failed to list TaskStateStore-backed training sessions: %s: %s",
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
            failed_request_ids = await task_futures.async_fail_training_requests_for_model(
                model_id,
                f"Training session terminated due to {reason}",
            )
            if failed_request_ids:
                logger.warning(
                    "[%s] failed pending training futures during TaskStateStore-backed cleanup (%s): request_ids=%s",
                    model_id,
                    reason,
                    failed_request_ids,
                )
        except Exception as e:
            logger.warning(
                "[%s] TaskStateStore-backed training cleanup aborted because future fail failed (%s): %s: %s",
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
                "[%s] TaskStateStore-backed training cleanup actor actuation failed (%s): %s: %s",
                model_id,
                reason,
                type(e).__name__,
                e,
            )

        try:
            delete_training_session(model_id)
        except Exception as e:
            logger.warning(
                "[%s] TaskStateStore-backed training cleanup store delete failed (%s): %s: %s",
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
            get_model_actor_supervisor().clear_session(model_id)
        except Exception:
            pass

        cleaned.append(model_id)
        logger.warning(
            "[%s] TaskStateStore-backed training cleanup removed stale session: session_id=%s actor_name=%s allow_actor_shutdown=%s actor_refcount=%s",
            model_id,
            session_id,
            actor_name or "<unknown>",
            allow_actor_shutdown,
            actor_refcounts.get(actor_name, 0) if actor_name else 0,
        )

    return cleaned


class TrainingCleanupExecutor:
    async def async_cleanup_stale_sessions_once(self, *, stale_after_s: float | None = None) -> list[str]:
        return await cleanup_stale_training_sessions_once_impl(stale_after_s=stale_after_s)


training_cleanup_executor = TrainingCleanupExecutor()
