from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ACTIVE_TASK_STATUSES = frozenset({"pending", "assigned", "leased", "finalizing"})
TERMINAL_TASK_STATUSES = frozenset({"done", "failed", "cancelled", "expired"})


class TaskStateStoreError(RuntimeError):
    pass


class TaskStateConflictError(TaskStateStoreError):
    pass


class TaskStateNotFoundError(TaskStateStoreError, KeyError):
    pass


def _now(now: float | None = None) -> float:
    return time.time() if now is None else float(now)


def _json_dumps(value: dict[str, Any] | None) -> str:
    return json.dumps({} if value is None else dict(value), ensure_ascii=True, sort_keys=True)


def _json_loads(value: str | bytes | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return {}
    out = json.loads(value)
    if not isinstance(out, dict):
        raise TaskStateStoreError(f"expected JSON object, got {type(out).__name__}")
    return out


class TaskStateStore:
    """SQLite-backed task state machine.

    The class is intentionally synchronous. V1 deployment should wrap it in a
    single-writer actor/service so API workers, schedulers, and runtimes do not
    open the SQLite file directly.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self._db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._apply_schema()

    @classmethod
    def in_memory(cls) -> "TaskStateStore":
        return cls(":memory:")

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _configure_connection(self) -> None:
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self._db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
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
            conn = self._conn
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduler_owner (
                    name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    renewed_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    fencing_token TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    request_id TEXT PRIMARY KEY,
                    op TEXT NOT NULL,
                    status TEXT NOT NULL,
                    domain_key TEXT NOT NULL,
                    subqueue_id TEXT,
                    lease_id TEXT,
                    attempt_id TEXT,
                    scheduler_epoch INTEGER,
                    runtime_generation INTEGER,
                    consumer_id TEXT,
                    request_json BLOB NOT NULL,
                    payload_hash TEXT,
                    result_path TEXT,
                    result_checksum TEXT,
                    result_size_bytes INTEGER,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    assigned_at REAL,
                    leased_at REAL,
                    lease_expires_at REAL,
                    finalizing_until REAL
                );

                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES tasks(request_id)
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status_created
                    ON tasks(status, created_at);

                CREATE INDEX IF NOT EXISTS idx_tasks_domain_status_created
                    ON tasks(domain_key, status, created_at);

                CREATE INDEX IF NOT EXISTS idx_tasks_subqueue_status_created
                    ON tasks(subqueue_id, status, created_at);

                CREATE INDEX IF NOT EXISTS idx_tasks_lease_expires
                    ON tasks(lease_expires_at);

                CREATE INDEX IF NOT EXISTS idx_tasks_finalizing_until
                    ON tasks(finalizing_until);
                """
            )

    def integrity_check(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row is not None else ""

    def _record_event(
        self,
        conn: sqlite3.Connection,
        *,
        request_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        now: float,
    ) -> None:
        version_row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM task_events WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        version = int(version_row[0]) if version_row is not None else 1
        conn.execute(
            """
            INSERT INTO task_events(request_id, version, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (request_id, version, event_type, _json_dumps(payload), now),
        )

    def acquire_scheduler_owner(
        self,
        *,
        owner_id: str,
        ttl_s: float,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        expires_at = ts + max(1.0, float(ttl_s))
        owner_id = str(owner_id)
        name = str(name)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT owner_id, epoch, expires_at FROM scheduler_owner WHERE name = ?",
                (name,),
            ).fetchone()
            if row is not None and str(row["owner_id"]) != owner_id and float(row["expires_at"]) > ts:
                return {
                    "ok": False,
                    "reason": "owner_active",
                    "owner_id": str(row["owner_id"]),
                    "epoch": int(row["epoch"]),
                    "expires_at": float(row["expires_at"]),
                }
            epoch = 1 if row is None else int(row["epoch"]) + (0 if str(row["owner_id"]) == owner_id else 1)
            fencing_token = f"{name}:{epoch}:{owner_id}"
            conn.execute(
                """
                INSERT INTO scheduler_owner(name, owner_id, epoch, renewed_at, expires_at, fencing_token)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    epoch = excluded.epoch,
                    renewed_at = excluded.renewed_at,
                    expires_at = excluded.expires_at,
                    fencing_token = excluded.fencing_token
                """,
                (name, owner_id, epoch, ts, expires_at, fencing_token),
            )
            return {
                "ok": True,
                "owner_id": owner_id,
                "epoch": epoch,
                "expires_at": expires_at,
                "fencing_token": fencing_token,
            }

    def renew_scheduler_owner(
        self,
        *,
        owner_id: str,
        epoch: int,
        ttl_s: float,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        expires_at = ts + max(1.0, float(ttl_s))
        with self._transaction() as conn:
            cur = conn.execute(
                """
                UPDATE scheduler_owner
                SET renewed_at = ?, expires_at = ?
                WHERE name = ? AND owner_id = ? AND epoch = ?
                """,
                (ts, expires_at, str(name), str(owner_id), int(epoch)),
            )
            if cur.rowcount != 1:
                return {"ok": False, "reason": "stale_owner"}
            return {"ok": True, "owner_id": str(owner_id), "epoch": int(epoch), "expires_at": expires_at}

    def assert_scheduler_owner(
        self,
        conn: sqlite3.Connection,
        *,
        scheduler_epoch: int,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> None:
        ts = _now(now)
        row = conn.execute(
            "SELECT epoch, expires_at FROM scheduler_owner WHERE name = ?",
            (str(name),),
        ).fetchone()
        if row is None or int(row["epoch"]) != int(scheduler_epoch) or float(row["expires_at"]) <= ts:
            raise TaskStateConflictError("stale scheduler owner epoch")

    def create_task(
        self,
        *,
        request_id: str,
        op: str,
        domain_key: str,
        request_json: bytes,
        payload_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT payload_hash FROM tasks WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
            if existing is not None:
                existing_hash = existing["payload_hash"]
                if payload_hash is not None and existing_hash not in (None, payload_hash):
                    raise TaskStateConflictError("duplicate request_id with different payload hash")
                return {"ok": True, "created": False, "record": self._row_to_record(self._get_row(conn, request_id))}
            conn.execute(
                """
                INSERT INTO tasks(
                    request_id, op, status, domain_key, request_json, payload_hash,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(request_id),
                    str(op),
                    str(domain_key),
                    bytes(request_json),
                    payload_hash,
                    _json_dumps(metadata),
                    ts,
                    ts,
                ),
            )
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type="task_created",
                payload={"status": "pending"},
                now=ts,
            )
            return {"ok": True, "created": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def assign_task(
        self,
        *,
        request_id: str,
        subqueue_id: str,
        scheduler_epoch: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            self.assert_scheduler_owner(conn, scheduler_epoch=scheduler_epoch, now=ts)
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = 'assigned',
                    subqueue_id = ?,
                    scheduler_epoch = ?,
                    assigned_at = ?,
                    updated_at = ?
                WHERE request_id = ? AND status = 'pending'
                """,
                (str(subqueue_id), int(scheduler_epoch), ts, ts, str(request_id)),
            )
            if cur.rowcount != 1:
                self._raise_task_transition_error(conn, request_id, "assign from pending")
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type="task_assigned",
                payload={"subqueue_id": str(subqueue_id), "scheduler_epoch": int(scheduler_epoch)},
                now=ts,
            )
            return {"ok": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def claim_task(
        self,
        *,
        request_id: str,
        subqueue_id: str,
        lease_id: str,
        attempt_id: str,
        consumer_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        lease_ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        expires_at = ts + max(1.0, float(lease_ttl_s))
        with self._transaction() as conn:
            self.assert_scheduler_owner(conn, scheduler_epoch=scheduler_epoch, now=ts)
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = 'leased',
                    lease_id = ?,
                    attempt_id = ?,
                    consumer_id = ?,
                    scheduler_epoch = ?,
                    runtime_generation = ?,
                    leased_at = ?,
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE request_id = ?
                  AND status = 'assigned'
                  AND subqueue_id = ?
                  AND scheduler_epoch = ?
                """,
                (
                    str(lease_id),
                    str(attempt_id),
                    str(consumer_id),
                    int(scheduler_epoch),
                    int(runtime_generation),
                    ts,
                    expires_at,
                    ts,
                    str(request_id),
                    str(subqueue_id),
                    int(scheduler_epoch),
                ),
            )
            if cur.rowcount != 1:
                self._raise_task_transition_error(conn, request_id, "claim assigned task")
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type="lease_claimed",
                payload={
                    "lease_id": str(lease_id),
                    "attempt_id": str(attempt_id),
                    "consumer_id": str(consumer_id),
                    "scheduler_epoch": int(scheduler_epoch),
                    "runtime_generation": int(runtime_generation),
                    "lease_expires_at": expires_at,
                },
                now=ts,
            )
            return {"ok": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def begin_finalize(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        finalize_ttl_s: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        finalizing_until = ts + max(1.0, float(finalize_ttl_s))
        with self._transaction() as conn:
            self.assert_scheduler_owner(conn, scheduler_epoch=scheduler_epoch, now=ts)
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = 'finalizing',
                    finalizing_until = ?,
                    lease_expires_at = MAX(COALESCE(lease_expires_at, 0), ?),
                    updated_at = ?
                WHERE request_id = ?
                  AND status = 'leased'
                  AND lease_id = ?
                  AND attempt_id = ?
                  AND scheduler_epoch = ?
                  AND runtime_generation = ?
                """,
                (
                    finalizing_until,
                    finalizing_until,
                    ts,
                    str(request_id),
                    str(lease_id),
                    str(attempt_id),
                    int(scheduler_epoch),
                    int(runtime_generation),
                ),
            )
            if cur.rowcount != 1:
                self._raise_task_transition_error(conn, request_id, "begin finalize")
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type="lease_finalizing",
                payload={"finalizing_until": finalizing_until},
                now=ts,
            )
            return {"ok": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def commit_finalize_success(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        result_path: str,
        result_checksum: str,
        result_size_bytes: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        return self._commit_finalize(
            request_id=request_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=scheduler_epoch,
            runtime_generation=runtime_generation,
            status="done",
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            error=None,
            now=now,
        )

    def commit_finalize_failure(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        return self._commit_finalize(
            request_id=request_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=scheduler_epoch,
            runtime_generation=runtime_generation,
            status="failed",
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            error=str(error),
            now=now,
        )

    def _commit_finalize(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        status: str,
        result_path: str | None,
        result_checksum: str | None,
        result_size_bytes: int | None,
        error: str | None,
        now: float | None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            row = self._get_row(conn, request_id)
            if str(row["status"]) in TERMINAL_TASK_STATUSES:
                if (
                    str(row["status"]) == status
                    and str(row["lease_id"]) == str(lease_id)
                    and str(row["attempt_id"]) == str(attempt_id)
                    and (row["result_path"] == result_path)
                    and (row["result_checksum"] == result_checksum)
                    and (row["result_size_bytes"] == result_size_bytes)
                    and (row["error"] == error)
                ):
                    return {"ok": True, "idempotent": True, "record": self._row_to_record(row)}
                raise TaskStateConflictError("terminal task commit payload mismatch")
            self.assert_scheduler_owner(conn, scheduler_epoch=scheduler_epoch, now=ts)
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    result_path = ?,
                    result_checksum = ?,
                    result_size_bytes = ?,
                    error = ?,
                    finalizing_until = NULL,
                    updated_at = ?
                WHERE request_id = ?
                  AND status = 'finalizing'
                  AND lease_id = ?
                  AND attempt_id = ?
                  AND scheduler_epoch = ?
                  AND runtime_generation = ?
                """,
                (
                    status,
                    result_path,
                    result_checksum,
                    result_size_bytes,
                    error,
                    ts,
                    str(request_id),
                    str(lease_id),
                    str(attempt_id),
                    int(scheduler_epoch),
                    int(runtime_generation),
                ),
            )
            if cur.rowcount != 1:
                self._raise_task_transition_error(conn, request_id, f"commit finalize {status}")
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type=f"task_{status}",
                payload={
                    "result_path": result_path,
                    "result_checksum": result_checksum,
                    "result_size_bytes": result_size_bytes,
                    "error": error,
                },
                now=ts,
            )
            return {"ok": True, "idempotent": False, "record": self._row_to_record(self._get_row(conn, request_id))}

    def list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT * FROM tasks
            WHERE status IN ('pending', 'assigned', 'leased', 'finalizing')
            ORDER BY created_at, request_id
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (max(0, int(limit)),)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_expired_leases(self, *, now: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        ts = _now(now)
        sql = """
            SELECT * FROM tasks
            WHERE status IN ('leased', 'finalizing')
              AND COALESCE(finalizing_until, lease_expires_at, 0) <= ?
            ORDER BY COALESCE(finalizing_until, lease_expires_at, 0), created_at
        """
        params: tuple[Any, ...] = (ts,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (ts, max(0, int(limit)))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_task(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._get_row(self._conn, request_id)
        return self._row_to_record(row)

    def _get_row(self, conn: sqlite3.Connection, request_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM tasks WHERE request_id = ?", (str(request_id),)).fetchone()
        if row is None:
            raise TaskStateNotFoundError(str(request_id))
        return row

    def _raise_task_transition_error(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        action: str,
    ) -> None:
        row = conn.execute(
            "SELECT status FROM tasks WHERE request_id = ?",
            (str(request_id),),
        ).fetchone()
        if row is None:
            raise TaskStateNotFoundError(str(request_id))
        raise TaskStateConflictError(f"cannot {action}; current status={row['status']!r}")

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "request_id": row["request_id"],
            "op": row["op"],
            "status": row["status"],
            "domain_key": row["domain_key"],
            "subqueue_id": row["subqueue_id"],
            "lease_id": row["lease_id"],
            "attempt_id": row["attempt_id"],
            "scheduler_epoch": row["scheduler_epoch"],
            "runtime_generation": row["runtime_generation"],
            "consumer_id": row["consumer_id"],
            "request_json": bytes(row["request_json"]),
            "payload_hash": row["payload_hash"],
            "result_path": row["result_path"],
            "result_checksum": row["result_checksum"],
            "result_size_bytes": row["result_size_bytes"],
            "error": row["error"],
            "metadata": _json_loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "assigned_at": row["assigned_at"],
            "leased_at": row["leased_at"],
            "lease_expires_at": row["lease_expires_at"],
            "finalizing_until": row["finalizing_until"],
        }
