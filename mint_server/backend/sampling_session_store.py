"""Sampling-session metadata facade backed by TaskStateStore."""

from __future__ import annotations

import logging
import time
from typing import Any

from .task_state_store import TaskStateStoreUnavailableError, task_state_store


logger = logging.getLogger(__name__)


def upsert_sampling_session(info: dict[str, Any]) -> None:
    payload = dict(info)
    session_id = str(payload.get("session_id", ""))
    if not session_id:
        raise RuntimeError("Sampling session store write failed: upsert: missing session_id")
    payload.setdefault("last_activity", time.time())
    payload["metadata_version"] = max(1, int(payload.get("metadata_version") or 1))
    task_state_store.upsert_sampling_session(session_id=session_id, info=payload)


def delete_sampling_session(session_id: str) -> None:
    try:
        task_state_store.delete_sampling_session(session_id=str(session_id))
    except TaskStateStoreUnavailableError:
        logger.warning("Sampling session store write skipped: TaskStateStore unavailable")
    except Exception as e:
        logger.warning("Sampling session store write failed: delete: %s", e)


def set_sampling_session_last_activity(session_id: str, last_activity: float) -> None:
    try:
        task_state_store.set_sampling_session_last_activity(
            session_id=str(session_id),
            last_activity=float(last_activity),
        )
    except TaskStateStoreUnavailableError:
        return
    except Exception as e:
        logger.debug("Sampling session store write failed: last_activity: %s", e)


async def async_set_sampling_session_last_activity(session_id: str, last_activity: float) -> float | None:
    return await task_state_store.async_set_sampling_session_last_activity(
        session_id=str(session_id),
        last_activity=float(last_activity),
    )


def get_sampling_session_info(session_id: str) -> dict[str, Any] | None:
    return task_state_store.get_sampling_session(session_id=str(session_id))


async def async_get_sampling_session_info(session_id: str) -> dict[str, Any] | None:
    return await task_state_store.async_get_sampling_session(session_id=str(session_id))


def list_sampling_sessions() -> list[dict[str, Any]]:
    return task_state_store.list_sampling_sessions()


async def async_list_sampling_sessions() -> list[dict[str, Any]]:
    return await task_state_store.async_list_sampling_sessions()
