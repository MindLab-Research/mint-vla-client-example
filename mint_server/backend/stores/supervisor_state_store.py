from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SUPERVISOR_STATE_SCHEMA_VERSION = 1
DEFAULT_SUPERVISOR_STATE_EVENT_LIMIT = 1000


class SupervisorStateStoreError(RuntimeError):
    pass


class SupervisorStateOwnerConflictError(SupervisorStateStoreError):
    pass


def _now(now: float | None = None) -> float:
    return time.time() if now is None else float(now)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _json_loads(value: str | bytes | None) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return None
    return json.loads(value)


def normalize_supervisor_state_backend(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if raw in {"", "memory", "memory-only", "in-memory"}:
        return "memory"
    if raw in {"sqlite", "sqlite3"}:
        return "sqlite"
    raise ValueError(f"unsupported supervisor state backend: {value!r}")


class SupervisorMemoryStateStore:
    backend = "memory"
    db_path: str | None = None

    def __init__(self, *, event_limit: int = DEFAULT_SUPERVISOR_STATE_EVENT_LIMIT) -> None:
        self._lock = threading.RLock()
        self._kv: dict[str, tuple[Any, float]] = {}
        self._owner: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._event_limit = max(1, int(event_limit))
        self._next_event_id = 1

    def close(self) -> None:
        return None

    def acquire_owner(
        self,
        *,
        name: str,
        owner_id: str,
        ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        lease_until = ts + float(ttl_s)
        with self._lock:
            current = self._owner.get(str(name))
            if (
                current is not None
                and str(current.get("owner_id")) != str(owner_id)
                and float(current.get("lease_until") or 0.0) > ts
            ):
                raise SupervisorStateOwnerConflictError(
                    f"supervisor state owner active: name={name!r} owner_id={current.get('owner_id')!r}"
                )
            epoch = 1 if current is None else int(current.get("epoch") or 0)
            if current is not None and str(current.get("owner_id")) != str(owner_id):
                epoch += 1
            started_at = ts if current is None or str(current.get("owner_id")) != str(owner_id) else float(current["started_at"])
            record = {
                "name": str(name),
                "owner_id": str(owner_id),
                "epoch": int(epoch),
                "started_at": float(started_at),
                "last_heartbeat_at": ts,
                "lease_until": lease_until,
                "schema_version": SUPERVISOR_STATE_SCHEMA_VERSION,
            }
            self._owner[str(name)] = record
            return dict(record)

    def heartbeat_owner(
        self,
        *,
        name: str,
        owner_id: str,
        epoch: int,
        ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._lock:
            current = self._owner.get(str(name))
            if current is None or str(current.get("owner_id")) != str(owner_id) or int(current.get("epoch") or 0) != int(epoch):
                raise SupervisorStateOwnerConflictError("stale supervisor state owner")
            current["last_heartbeat_at"] = ts
            current["lease_until"] = ts + float(ttl_s)
            return dict(current)

    def owner_snapshot(self, *, name: str) -> dict[str, Any] | None:
        with self._lock:
            current = self._owner.get(str(name))
            return None if current is None else dict(current)

    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._lock:
            item = self._kv.get(str(key))
            return default if item is None else item[0]

    def set_kv(self, key: str, value: Any, *, now: float | None = None) -> None:
        with self._lock:
            self._kv[str(key)] = (value, _now(now))

    def delete_kv(self, key: str) -> None:
        with self._lock:
            self._kv.pop(str(key), None)

    def reserve_generation(self, key: str, *, floor: int = 0, now: float | None = None) -> int:
        with self._lock:
            current = self.get_kv(key, 0)
            try:
                current_int = int(current or 0)
            except Exception:
                current_int = 0
            generation = max(int(floor), current_int + 1)
            self._kv[str(key)] = (generation, _now(now))
            return generation

    def append_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        owner: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._lock:
            event = {
                "id": self._next_event_id,
                "event_type": str(event_type),
                "payload": dict(payload or {}),
                "owner_id": None if owner is None else owner.get("owner_id"),
                "epoch": None if owner is None else owner.get("epoch"),
                "created_at": ts,
            }
            self._next_event_id += 1
            self._events.append(event)
            if len(self._events) > self._event_limit:
                self._events = self._events[-self._event_limit :]
            return dict(event)

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events[-max(0, int(limit)) :]]


class SupervisorSQLiteStateStore:
    backend = "sqlite"

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        event_limit: int = DEFAULT_SUPERVISOR_STATE_EVENT_LIMIT,
    ) -> None:
        self._db_path = str(db_path)
        self._event_limit = max(1, int(event_limit))
        self._lock = threading.RLock()
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._apply_schema()

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _configure_connection(self) -> None:
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = TRUNCATE")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                if self._conn.in_transaction:
                    self._conn.execute("COMMIT")

    def _apply_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    owner_id TEXT,
                    epoch INTEGER
                );

                CREATE TABLE IF NOT EXISTS owner (
                    name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    started_at REAL NOT NULL,
                    last_heartbeat_at REAL NOT NULL,
                    lease_until REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    owner_id TEXT,
                    epoch INTEGER,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_supervisor_events_created
                    ON events(created_at);
                """
            )

    def acquire_owner(
        self,
        *,
        name: str,
        owner_id: str,
        ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        lease_until = ts + float(ttl_s)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT owner_id, epoch, started_at, lease_until, schema_version FROM owner WHERE name = ?",
                (str(name),),
            ).fetchone()
            if (
                row is not None
                and str(row["owner_id"]) != str(owner_id)
                and float(row["lease_until"]) > ts
            ):
                raise SupervisorStateOwnerConflictError(
                    f"supervisor state owner active: name={name!r} owner_id={row['owner_id']!r}"
                )
            epoch = 1 if row is None else int(row["epoch"])
            if row is not None and str(row["owner_id"]) != str(owner_id):
                epoch += 1
            started_at = ts if row is None or str(row["owner_id"]) != str(owner_id) else float(row["started_at"])
            conn.execute(
                """
                INSERT INTO owner(name, owner_id, epoch, started_at, last_heartbeat_at, lease_until, schema_version)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    epoch = excluded.epoch,
                    started_at = excluded.started_at,
                    last_heartbeat_at = excluded.last_heartbeat_at,
                    lease_until = excluded.lease_until,
                    schema_version = excluded.schema_version
                """,
                (
                    str(name),
                    str(owner_id),
                    int(epoch),
                    float(started_at),
                    ts,
                    lease_until,
                    SUPERVISOR_STATE_SCHEMA_VERSION,
                ),
            )
            return {
                "name": str(name),
                "owner_id": str(owner_id),
                "epoch": int(epoch),
                "started_at": float(started_at),
                "last_heartbeat_at": ts,
                "lease_until": lease_until,
                "schema_version": SUPERVISOR_STATE_SCHEMA_VERSION,
            }

    def heartbeat_owner(
        self,
        *,
        name: str,
        owner_id: str,
        epoch: int,
        ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        lease_until = ts + float(ttl_s)
        with self._transaction() as conn:
            cur = conn.execute(
                """
                UPDATE owner
                SET last_heartbeat_at = ?, lease_until = ?
                WHERE name = ? AND owner_id = ? AND epoch = ?
                """,
                (ts, lease_until, str(name), str(owner_id), int(epoch)),
            )
            if cur.rowcount != 1:
                raise SupervisorStateOwnerConflictError("stale supervisor state owner")
            row = conn.execute("SELECT * FROM owner WHERE name = ?", (str(name),)).fetchone()
            return dict(row)

    def owner_snapshot(self, *, name: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM owner WHERE name = ?", (str(name),)).fetchone()
            return None if row is None else dict(row)

    def get_kv(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value_json FROM kv WHERE key = ?", (str(key),)).fetchone()
            return default if row is None else _json_loads(row["value_json"])

    def set_kv(
        self,
        key: str,
        value: Any,
        *,
        now: float | None = None,
        owner: dict[str, Any] | None = None,
    ) -> None:
        ts = _now(now)
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO kv(key, value_json, updated_at, owner_id, epoch)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at,
                    owner_id = excluded.owner_id,
                    epoch = excluded.epoch
                """,
                (
                    str(key),
                    _json_dumps(value),
                    ts,
                    None if owner is None else owner.get("owner_id"),
                    None if owner is None else owner.get("epoch"),
                ),
            )

    def delete_kv(self, key: str) -> None:
        with self._transaction() as conn:
            conn.execute("DELETE FROM kv WHERE key = ?", (str(key),))

    def reserve_generation(self, key: str, *, floor: int = 0, now: float | None = None) -> int:
        ts = _now(now)
        with self._transaction() as conn:
            row = conn.execute("SELECT value_json FROM kv WHERE key = ?", (str(key),)).fetchone()
            try:
                current = int(_json_loads(row["value_json"]) if row is not None else 0)
            except Exception:
                current = 0
            generation = max(int(floor), current + 1)
            conn.execute(
                """
                INSERT INTO kv(key, value_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (str(key), _json_dumps(generation), ts),
            )
            return generation

    def append_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        owner: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO events(event_type, payload_json, owner_id, epoch, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    str(event_type),
                    _json_dumps(dict(payload or {})),
                    None if owner is None else owner.get("owner_id"),
                    None if owner is None else owner.get("epoch"),
                    ts,
                ),
            )
            conn.execute(
                """
                DELETE FROM events
                WHERE id NOT IN (
                    SELECT id FROM events ORDER BY id DESC LIMIT ?
                )
                """,
                (self._event_limit,),
            )
            return {
                "id": int(cur.lastrowid),
                "event_type": str(event_type),
                "payload": dict(payload or {}),
                "owner_id": None if owner is None else owner.get("owner_id"),
                "epoch": None if owner is None else owner.get("epoch"),
                "created_at": ts,
            }

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (max(0, int(limit)),),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for row in reversed(rows):
                item = dict(row)
                item["payload"] = _json_loads(item.pop("payload_json")) or {}
                out.append(item)
            return out


def create_supervisor_state_store(
    *,
    backend: str | None,
    db_path: str | os.PathLike[str] | None = None,
    event_limit: int = DEFAULT_SUPERVISOR_STATE_EVENT_LIMIT,
) -> SupervisorMemoryStateStore | SupervisorSQLiteStateStore:
    normalized = normalize_supervisor_state_backend(backend)
    if normalized == "memory":
        return SupervisorMemoryStateStore(event_limit=event_limit)
    if not db_path:
        raise ValueError("supervisor SQLite state backend requires db_path")
    return SupervisorSQLiteStateStore(db_path, event_limit=event_limit)
