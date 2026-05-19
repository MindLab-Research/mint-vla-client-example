from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from typing import Any

from ..runtime_env import env_nonempty
from .async_ray_control import async_get_ray_ref

logger = logging.getLogger(__name__)


def _ray_namespace() -> str:
    env_ns = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if env_ns:
        return env_ns
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _sampling_inactivity_timeout_s() -> float:
    raw = env_nonempty(os.environ, "MINT_SESSION_INACTIVITY_TIMEOUT_S") or env_nonempty(
        os.environ,
        "MINT_INACTIVITY_TIMEOUT_S",
    ) or "1800"
    try:
        return max(0.0, float(raw))
    except Exception:
        logger.warning("Invalid sampling inactivity timeout=%r; defaulting to 1800s", raw)
        return 1800.0


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
    await async_get_ray_ref(actor.remove_lora.remote(int(lora_int_id)), timeout_s=30)


async def cleanup_stale_sampling_sessions_once_impl(*, stale_after_s: float | None = None) -> list[str]:
    if stale_after_s is None:
        stale_after_s = _sampling_inactivity_timeout_s()
    stale_after_s = float(stale_after_s)
    if stale_after_s <= 0:
        return []

    from .task_state_store import task_futures
    from .sampling_session_store import async_list_sampling_sessions, delete_sampling_session

    try:
        infos = await async_list_sampling_sessions()
    except Exception as e:
        logger.warning(
            "sampling cleanup executor skipped: failed to list TaskStateStore-backed sampling sessions: %s: %s",
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
            failed_request_ids = await task_futures.async_fail_sampling_requests_for_session(
                session_id,
                f"Sampling session terminated due to {reason}",
            )
            if failed_request_ids:
                logger.warning(
                    "[%s] failed pending sampling futures during TaskStateStore-backed cleanup (%s): request_ids=%s",
                    session_id,
                    reason,
                    failed_request_ids,
                )
        except Exception as e:
            logger.warning(
                "[%s] TaskStateStore-backed sampling cleanup aborted because future fail failed (%s): %s: %s",
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
                    "[%s] TaskStateStore-backed sampling cleanup actor actuation failed (%s): %s: %s",
                    session_id,
                    reason,
                    type(e).__name__,
                    e,
                )

        try:
            delete_sampling_session(session_id)
        except Exception as e:
            logger.warning(
                "[%s] TaskStateStore-backed sampling cleanup store delete failed (%s): %s: %s",
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
                    "[%s] TaskStateStore-backed sampling cleanup adapter delete failed (%s): %s: %s",
                    session_id,
                    reason,
                    type(e).__name__,
                    e,
                )

        if adapter_path and current_adapter_refcount > 0:
            adapter_refcounts[adapter_path] = current_adapter_refcount - 1

        cleaned.append(session_id)
        logger.warning(
            "[%s] TaskStateStore-backed sampling cleanup removed stale session: base_model=%s should_unload=%s adapter_refcount=%s",
            session_id,
            base_model,
            should_unload,
            current_adapter_refcount,
        )

    return cleaned


class SamplingCleanupExecutor:
    async def async_cleanup_stale_sessions_once(self, *, stale_after_s: float | None = None) -> list[str]:
        return await cleanup_stale_sampling_sessions_once_impl(stale_after_s=stale_after_s)


sampling_cleanup_executor = SamplingCleanupExecutor()
