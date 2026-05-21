from __future__ import annotations

from .task_state_store import task_state_store


class SessionHeartbeatStore:
    def update(self, session_id: str, now: float | None = None) -> None:
        task_state_store.update_session_heartbeat(session_id=str(session_id), now=now)

    async def async_update(self, session_id: str, now: float | None = None) -> None:
        await task_state_store.async_update_session_heartbeat(session_id=str(session_id), now=now)

    def last_seen(self, session_id: str) -> float | None:
        return task_state_store.get_session_heartbeat(session_id=str(session_id))

    def delete(self, session_id: str) -> bool:
        return task_state_store.delete_session_heartbeat(session_id=str(session_id))

    def size(self) -> int:
        return task_state_store.session_heartbeat_size()

    async def async_size(self, *, create_if_missing: bool = False) -> int:
        return await task_state_store.async_session_heartbeat_size(create_if_missing=create_if_missing)

    def is_stale(self, session_id: str, ttl_s: float) -> bool:
        return task_state_store.is_session_heartbeat_stale(session_id=str(session_id), ttl_s=float(ttl_s))

    async def async_is_stale(self, session_id: str, ttl_s: float) -> bool:
        return await task_state_store.async_is_session_heartbeat_stale(
            session_id=str(session_id),
            ttl_s=float(ttl_s),
        )

    def prune(self, max_age_s: float) -> int:
        return task_state_store.prune_session_heartbeats(max_age_s=float(max_age_s))


session_heartbeat_store = SessionHeartbeatStore()
