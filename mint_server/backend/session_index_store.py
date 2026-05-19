"""Session and sampler index facade backed by TaskStateStore."""

from __future__ import annotations

import logging
from typing import Any

from .task_state_store import TaskStateStoreUnavailableError, task_state_store


logger = logging.getLogger(__name__)


def upsert_session_index(info: dict[str, Any]) -> None:
    session_id = str(info.get("session_id") or "")
    if not session_id:
        return
    task_state_store.upsert_session_index(session_id=session_id, info=dict(info))


def add_training_run_to_session(
    session_id: str,
    training_run_id: str,
    *,
    user_id: str | None = None,
    created_at: str | None = None,
) -> None:
    if not session_id or not training_run_id:
        return
    task_state_store.add_training_run_to_session_index(
        session_id=str(session_id),
        training_run_id=str(training_run_id),
        user_id=user_id,
        created_at=created_at,
    )


def add_sampler_to_session(
    session_id: str,
    sampler_id: str,
    *,
    user_id: str | None = None,
    created_at: str | None = None,
) -> None:
    if not session_id or not sampler_id:
        return
    task_state_store.add_sampler_to_session_index(
        session_id=str(session_id),
        sampler_id=str(sampler_id),
        user_id=user_id,
        created_at=created_at,
    )


def remove_sampler_from_session(session_id: str, sampler_id: str) -> None:
    if not session_id or not sampler_id:
        return
    try:
        task_state_store.remove_sampler_from_session_index(
            session_id=str(session_id),
            sampler_id=str(sampler_id),
        )
    except TaskStateStoreUnavailableError:
        logger.warning("Session index store write skipped: TaskStateStore unavailable")
    except Exception as e:
        logger.warning("Session index store write failed: remove_sampler: %s", e)


def add_heartbeat_sampler_to_session(
    session_id: str,
    sampler_id: str,
    *,
    user_id: str | None = None,
    created_at: str | None = None,
) -> None:
    if not session_id or not sampler_id:
        return
    task_state_store.add_heartbeat_sampler_to_session_index(
        session_id=str(session_id),
        sampler_id=str(sampler_id),
        user_id=user_id,
        created_at=created_at,
    )


def get_session_index(session_id: str) -> dict[str, Any] | None:
    return task_state_store.get_session_index(session_id=str(session_id))


async def async_get_session_index(session_id: str) -> dict[str, Any] | None:
    return await task_state_store.async_get_session_index(session_id=str(session_id))


def list_session_index() -> list[dict[str, Any]]:
    return task_state_store.list_session_index()


async def async_list_session_index() -> list[dict[str, Any]]:
    return await task_state_store.async_list_session_index()


def upsert_sampler_index(info: dict[str, Any]) -> None:
    sampler_id = str(info.get("sampler_id") or "")
    if not sampler_id:
        return
    task_state_store.upsert_sampler_index(sampler_id=sampler_id, info=dict(info))


def delete_sampler_index(sampler_id: str) -> None:
    if not sampler_id:
        return
    try:
        task_state_store.delete_sampler_index(sampler_id=str(sampler_id))
    except TaskStateStoreUnavailableError:
        logger.warning("Session index store write skipped: TaskStateStore unavailable")
    except Exception as e:
        logger.warning("Session index store write failed: delete_sampler: %s", e)


def get_sampler_index(sampler_id: str) -> dict[str, Any] | None:
    return task_state_store.get_sampler_index(sampler_id=str(sampler_id))


async def async_get_sampler_index(sampler_id: str) -> dict[str, Any] | None:
    return await task_state_store.async_get_sampler_index(sampler_id=str(sampler_id))


def list_sampler_index() -> list[dict[str, Any]]:
    return task_state_store.list_sampler_index()


async def async_list_sampler_index() -> list[dict[str, Any]]:
    return await task_state_store.async_list_sampler_index()
