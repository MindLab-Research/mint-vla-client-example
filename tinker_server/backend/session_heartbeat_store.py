import threading
import time


class SessionHeartbeatStore:
    """In-memory session heartbeat timestamps (best-effort).

    Tracks last heartbeat time for SDK sessions created via /create_session and
    kept alive via /session_heartbeat.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}

    def update(self, session_id: str, now: float | None = None) -> None:
        if not session_id:
            return
        ts = time.time() if now is None else now
        with self._lock:
            self._last_seen[session_id] = ts

    def last_seen(self, session_id: str) -> float | None:
        with self._lock:
            return self._last_seen.get(session_id)

    def is_stale(self, session_id: str, ttl_s: float) -> bool:
        if ttl_s <= 0:
            return False
        if not session_id:
            return False
        now = time.time()
        with self._lock:
            last = self._last_seen.get(session_id)
        if last is None:
            # Unknown session: treat as live to avoid canceling clients that don't heartbeat.
            return False
        return (now - last) > ttl_s

    def prune(self, max_age_s: float) -> int:
        if max_age_s <= 0:
            return 0
        now = time.time()
        with self._lock:
            to_delete = [sid for sid, ts in self._last_seen.items() if (now - ts) > max_age_s]
            for sid in to_delete:
                del self._last_seen[sid]
        return len(to_delete)


session_heartbeat_store = SessionHeartbeatStore()

