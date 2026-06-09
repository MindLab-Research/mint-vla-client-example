"""Training-session metadata facade backed by TaskStateStore."""

from __future__ import annotations

import logging
import time
from typing import Any

from .task_state_store import TaskStateStoreUnavailableError, task_state_store


logger = logging.getLogger(__name__)


def upsert_training_session(info: dict[str, Any]) -> None:
    payload = dict(info)
    model_id = str(payload.get("model_id") or "")
    if not model_id:
        raise RuntimeError("Training session store write failed: upsert: missing model_id")
    payload.setdefault("current_step", 0)
    payload.setdefault("last_activity", time.time())
    payload["metadata_version"] = max(1, int(payload.get("metadata_version") or 1))
    task_state_store.upsert_training_session(model_id=model_id, info=payload)


async def async_upsert_training_session(info: dict[str, Any]) -> None:
    payload = dict(info)
    model_id = str(payload.get("model_id") or "")
    if not model_id:
        raise RuntimeError("Training session store write failed: upsert: missing model_id")
    payload.setdefault("current_step", 0)
    payload.setdefault("last_activity", time.time())
    payload["metadata_version"] = max(1, int(payload.get("metadata_version") or 1))
    await task_state_store.async_upsert_training_session(model_id=model_id, info=payload)


def delete_training_session(model_id: str) -> None:
    try:
        task_state_store.delete_training_session(model_id=str(model_id))
    except TaskStateStoreUnavailableError:
        logger.warning("Training session store write skipped: TaskStateStore unavailable")
    except Exception as e:
        logger.warning("Training session store write failed: delete: %s", e)


def set_training_session_last_activity(model_id: str, last_activity: float) -> None:
    try:
        task_state_store.set_training_session_last_activity(model_id=str(model_id), last_activity=float(last_activity))
    except TaskStateStoreUnavailableError:
        return
    except Exception as e:
        logger.debug("Training session store write failed: last_activity: %s", e)


async def async_set_training_session_last_activity(model_id: str, last_activity: float) -> float | None:
    return await task_state_store.async_set_training_session_last_activity(
        model_id=str(model_id),
        last_activity=float(last_activity),
    )


async def async_mark_training_session_inflight(model_id: str, delta: int) -> int | None:
    return await task_state_store.async_mark_training_session_inflight(
        model_id=str(model_id),
        delta=int(delta),
    )


def get_training_session_info(model_id: str) -> dict[str, Any] | None:
    return task_state_store.get_training_session(model_id=str(model_id))


def bump_training_session_step(model_id: str) -> int:
    return task_state_store.bump_training_session_step(model_id=str(model_id))


def set_training_session_step(model_id: str, step: int) -> int:
    return task_state_store.set_training_session_step(model_id=str(model_id), step=int(step))


async def async_get_training_session_info(model_id: str) -> dict[str, Any] | None:
    return await task_state_store.async_get_training_session(model_id=str(model_id))


def set_training_session_step_best_effort(model_id: str, step: int) -> None:
    try:
        task_state_store.set_training_session_step_best_effort(model_id=str(model_id), step=int(step))
    except TaskStateStoreUnavailableError:
        return


def bump_training_session_step_best_effort(model_id: str) -> None:
    try:
        task_state_store.bump_training_session_step_best_effort(model_id=str(model_id))
    except TaskStateStoreUnavailableError:
        return


def list_training_sessions() -> list[dict[str, Any]]:
    return task_state_store.list_training_sessions()


async def async_list_training_sessions() -> list[dict[str, Any]]:
    return await task_state_store.async_list_training_sessions()
