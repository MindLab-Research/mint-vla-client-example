from __future__ import annotations

import json
import os
import sqlite3
import asyncio
import threading
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from ..config import PFS_PYTHONPATH, actor_runtime_env, config as server_config, otel_env_vars
from .async_ray_control import async_get_ray_ref, sync_get_ray_ref
from .queue_execution_context import ModelWorkFinalize, get_current_model_work_finalize_buffer


ACTIVE_TASK_STATUSES = frozenset({"pending", "queued", "running", "assigned", "leased", "finalizing"})
TERMINAL_TASK_STATUSES = frozenset({"done", "failed", "cancelled", "expired", "retrieved"})


class TaskStateStoreError(RuntimeError):
    pass


class TaskStateConflictError(TaskStateStoreError):
    pass


class TaskStateNotFoundError(TaskStateStoreError, KeyError):
    pass


class TaskStateStoreUnavailableError(TaskStateStoreError):
    pass


class FutureStatus(Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRIEVED = "retrieved"


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


def _status_from_task_record(record: dict[str, Any]) -> FutureStatus:
    status = str(record.get("status") or "")
    if status in {"pending", "queued", "assigned", "leased", "running", "finalizing"}:
        return FutureStatus.PENDING
    if status == "done":
        return FutureStatus.DONE
    if status in {"failed", "cancelled"}:
        return FutureStatus.FAILED
    if status == "expired":
        return FutureStatus.EXPIRED
    if status == "retrieved":
        return FutureStatus.RETRIEVED
    raise KeyError(f"Unknown task status for request_id={record.get('request_id')!r}: {status!r}")


def _is_training_step_op(op: Any) -> bool:
    return str(op or "") in {"training.optim_step", "training.train_step"}


def _extract_training_step(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None
    step = metrics.get("step")
    if isinstance(step, bool):
        return None
    if isinstance(step, int):
        return int(step)
    if isinstance(step, float) and step.is_integer():
        return int(step)
    return None


def _sync_training_session_step(meta: dict[str, Any] | None, result: Any) -> Any:
    if not isinstance(meta, dict) or not _is_training_step_op(meta.get("op")):
        return result
    model_id = meta.get("model_id")
    if not model_id:
        return result

    try:
        from .training_session_store import (
            bump_training_session_step_best_effort,
            set_training_session_step_best_effort,
        )

        step = _extract_training_step(result)
        if step is None:
            bump_training_session_step_best_effort(str(model_id))
            return result

        set_training_session_step_best_effort(str(model_id), int(step))
        if isinstance(result, dict):
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                metrics["step"] = int(step)
        return result
    except Exception:
        return result


def _meta_with_request_op(meta: dict[str, Any] | None, request_op: Any) -> dict[str, Any]:
    out = dict(meta or {})
    op = out.get("op")
    if isinstance(op, str) and op.strip():
        return out
    op = str(request_op or "").strip()
    if op:
        out["op"] = op
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
                "SELECT * FROM tasks WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
            if existing is not None:
                existing_hash = existing["payload_hash"]
                if payload_hash is not None and existing_hash not in (None, payload_hash):
                    raise TaskStateConflictError("duplicate request_id with different payload hash")
                if str(existing["op"]) != str(op) or str(existing["domain_key"]) != str(domain_key):
                    raise TaskStateConflictError("duplicate request_id with different task identity")
                merged = {**_json_loads(existing["metadata_json"]), **dict(metadata or {})}
                if existing_hash is None and payload_hash is not None:
                    conn.execute(
                        """
                        UPDATE tasks
                        SET request_json = ?, payload_hash = ?, metadata_json = ?, updated_at = ?
                        WHERE request_id = ?
                        """,
                        (bytes(request_json), payload_hash, _json_dumps(merged), ts, str(request_id)),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE tasks
                        SET request_json = ?, metadata_json = ?, updated_at = ?
                        WHERE request_id = ?
                        """,
                        (bytes(request_json), _json_dumps(merged), ts, str(request_id)),
                    )
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

    def ensure_task(
        self,
        *,
        request_id: str,
        op: str = "unknown",
        domain_key: str = "future:default",
        request_json: bytes = b"{}",
        payload_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "pending",
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
            if row is not None:
                merged = {**_json_loads(row["metadata_json"]), **dict(metadata or {})}
                conn.execute(
                    """
                    UPDATE tasks
                    SET metadata_json = ?, updated_at = ?
                    WHERE request_id = ?
                    """,
                    (_json_dumps(merged), ts, str(request_id)),
                )
                return {"ok": True, "created": False, "record": self._row_to_record(self._get_row(conn, request_id))}
            conn.execute(
                """
                INSERT INTO tasks(
                    request_id, op, status, domain_key, request_json, payload_hash,
                    metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(request_id),
                    str(op),
                    str(status),
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
                payload={"status": str(status)},
                now=ts,
            )
            return {"ok": True, "created": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def update_task_metadata(
        self,
        *,
        request_id: str,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            row = self._get_row(conn, request_id)
            merged = {**_json_loads(row["metadata_json"]), **dict(metadata or {})}
            if status is None:
                conn.execute(
                    """
                    UPDATE tasks
                    SET metadata_json = ?, updated_at = ?
                    WHERE request_id = ?
                    """,
                    (_json_dumps(merged), ts, str(request_id)),
                )
            else:
                conn.execute(
                    """
                    UPDATE tasks
                    SET metadata_json = ?, status = ?, updated_at = ?
                    WHERE request_id = ?
                    """,
                    (_json_dumps(merged), str(status), ts, str(request_id)),
                )
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type="task_metadata_updated",
                payload={"status": status, "metadata": dict(metadata or {})},
                now=ts,
            )
            return {"ok": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def complete_task_success(
        self,
        *,
        request_id: str,
        result_path: str,
        result_checksum: str,
        result_size_bytes: int,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        return self._complete_task_direct(
            request_id=request_id,
            status="done",
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            error=None,
            metadata=metadata,
            now=now,
        )

    def complete_task_failure(
        self,
        *,
        request_id: str,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        return self._complete_task_direct(
            request_id=request_id,
            status="failed",
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            error=str(error),
            metadata=metadata,
            now=now,
        )

    def mark_task_retrieved(self, *, request_id: str, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            row = self._get_row(conn, request_id)
            if str(row["status"]) == "retrieved":
                return {"ok": True, "record": self._row_to_record(row)}
            if str(row["status"]) not in {"done", "failed", "expired", "cancelled"}:
                raise TaskStateConflictError(f"cannot mark retrieved; current status={row['status']!r}")
            metadata = _json_loads(row["metadata_json"])
            metadata.setdefault("terminal_status", str(row["status"]))
            conn.execute(
                """
                UPDATE tasks
                SET status = 'retrieved', metadata_json = ?, updated_at = ?
                WHERE request_id = ?
                """,
                (_json_dumps(metadata), ts, str(request_id)),
            )
            self._record_event(conn, request_id=str(request_id), event_type="task_retrieved", payload={}, now=ts)
            return {"ok": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def forget_task(self, *, request_id: str) -> dict[str, Any]:
        with self._transaction() as conn:
            conn.execute("DELETE FROM task_events WHERE request_id = ?", (str(request_id),))
            cur = conn.execute("DELETE FROM tasks WHERE request_id = ?", (str(request_id),))
            return {"ok": True, "deleted": cur.rowcount > 0}

    def list_tasks_by_metadata(
        self,
        *,
        filters: dict[str, Any] | None = None,
        statuses: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        status_values = list(statuses or [])
        params: list[Any] = []
        sql = "SELECT * FROM tasks"
        if status_values:
            placeholders = ", ".join("?" for _ in status_values)
            sql += f" WHERE status IN ({placeholders})"
            params.extend(status_values)
        sql += " ORDER BY created_at, request_id"
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        normalized_filters = dict(filters or {})
        out: list[dict[str, Any]] = []
        for row in rows:
            record = self._row_to_record(row)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            if all(metadata.get(key) == value for key, value in normalized_filters.items()):
                out.append(record)
                if len(out) >= int(limit):
                    break
        return out

    def _complete_task_direct(
        self,
        *,
        request_id: str,
        status: str,
        result_path: str | None,
        result_checksum: str | None,
        result_size_bytes: int | None,
        error: str | None,
        metadata: dict[str, Any] | None,
        now: float | None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            row = self._get_row(conn, request_id)
            if str(row["status"]) in TERMINAL_TASK_STATUSES:
                if (
                    str(row["status"]) == status
                    and row["result_path"] == result_path
                    and row["result_checksum"] == result_checksum
                    and row["result_size_bytes"] == result_size_bytes
                    and row["error"] == error
                ):
                    return {"ok": True, "idempotent": True, "record": self._row_to_record(row)}
                raise TaskStateConflictError("terminal task commit payload mismatch")
            merged = {**_json_loads(row["metadata_json"]), **dict(metadata or {})}
            conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    result_path = ?,
                    result_checksum = ?,
                    result_size_bytes = ?,
                    error = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE request_id = ?
                """,
                (
                    str(status),
                    result_path,
                    result_checksum,
                    result_size_bytes,
                    error,
                    _json_dumps(merged),
                    ts,
                    str(request_id),
                ),
            )
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
                  AND status IN ('leased', 'running')
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

    def requeue_task(
        self,
        *,
        request_id: str,
        scheduler_epoch: int,
        reason: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            self.assert_scheduler_owner(conn, scheduler_epoch=scheduler_epoch, now=ts)
            row = self._get_row(conn, request_id)
            if str(row["status"]) in TERMINAL_TASK_STATUSES:
                return {"ok": False, "reason": "terminal", "record": self._row_to_record(row)}
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = 'pending',
                    subqueue_id = NULL,
                    lease_id = NULL,
                    attempt_id = NULL,
                    scheduler_epoch = NULL,
                    runtime_generation = NULL,
                    consumer_id = NULL,
                    assigned_at = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    finalizing_until = NULL,
                    updated_at = ?
                WHERE request_id = ?
                  AND status IN ('pending', 'assigned', 'leased', 'running', 'finalizing')
                """,
                (ts, str(request_id)),
            )
            if cur.rowcount != 1:
                self._raise_task_transition_error(conn, request_id, "requeue active task")
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type="task_requeued",
                payload={"reason": str(reason), "scheduler_epoch": int(scheduler_epoch)},
                now=ts,
            )
            return {"ok": True, "record": self._row_to_record(self._get_row(conn, request_id))}

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


def _ray_namespace() -> str:
    v = os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "tinker"


def _ray_task_state_store_actor_name() -> str:
    return str(
        os.environ.get("MINT_TASK_STATE_STORE_ACTOR_NAME")
        or getattr(server_config, "task_state_store_actor_name", "mint_task_state_store")
    )


def _task_state_store_db_path() -> str:
    return str(
        os.environ.get("MINT_TASK_STATE_STORE_DB_PATH")
        or getattr(
            server_config,
            "task_state_store_db_path",
            "/vePFS-Mindverse/share/mint-prod-dev/task-state/task_state.sqlite3",
        )
    )


class _TaskStateStoreActor:
    def __init__(self, db_path: str | None = None) -> None:
        self._started_at = time.time()
        self._store = TaskStateStore(db_path or _task_state_store_db_path())

    def close(self) -> None:
        self._store.close()

    def stats(self) -> dict[str, Any]:
        active = self._store.list_active_tasks()
        by_status: dict[str, int] = {}
        for record in active:
            status = str(record.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "actor_name": _ray_task_state_store_actor_name(),
            "namespace": _ray_namespace(),
            "db_path": self._store.db_path,
            "started_at": self._started_at,
            "active_tasks": len(active),
            "active_by_status": by_status,
        }

    def integrity_check(self) -> str:
        return self._store.integrity_check()

    def acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.acquire_scheduler_owner(**kwargs)

    def renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.renew_scheduler_owner(**kwargs)

    def create_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.create_task(**kwargs)

    def ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.ensure_task(**kwargs)

    def update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.update_task_metadata(**kwargs)

    def complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.complete_task_success(**kwargs)

    def complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.complete_task_failure(**kwargs)

    def mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.mark_task_retrieved(**kwargs)

    def forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.forget_task(**kwargs)

    def list_tasks_by_metadata(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_tasks_by_metadata(**kwargs)

    def assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.assign_task(**kwargs)

    def claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.claim_task(**kwargs)

    def begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.begin_finalize(**kwargs)

    def commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.commit_finalize_success(**kwargs)

    def commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.commit_finalize_failure(**kwargs)

    def requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.requeue_task(**kwargs)

    def list_active_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_active_tasks(**kwargs)

    def list_expired_leases(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_expired_leases(**kwargs)

    def get_task(self, request_id: str) -> dict[str, Any]:
        return self._store.get_task(request_id)


def _create_ray_actor(*, require_ready: bool = True):
    try:
        import ray
    except Exception as e:
        raise TaskStateStoreUnavailableError("Ray import failed") from e

    actor_name = _ray_task_state_store_actor_name()
    namespace = _ray_namespace()
    db_path = _task_state_store_db_path()
    max_concurrency = int(os.environ.get("MINT_TASK_STATE_STORE_ACTOR_MAX_CONCURRENCY", "128"))

    @ray.remote(num_cpus=0, max_concurrency=max_concurrency, max_restarts=0)
    class _RayTaskStateStoreActor(_TaskStateStoreActor):
        pass

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=otel_env_vars()),
    }
    actor = _RayTaskStateStoreActor.options(**options).remote(db_path)
    if require_ready:
        out = sync_get_ray_ref(actor.stats.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.stats returned non-dict: {type(out)}")
    return actor


class TaskStateStoreClient:
    def __init__(self) -> None:
        self._ray_actor = None

    def _reset_ray_actor(self) -> None:
        self._ray_actor = None

    async def _get_ray_actor_async(self, *, require_ready: bool = True):
        try:
            import ray
        except Exception as e:
            raise TaskStateStoreUnavailableError("Ray import failed") from e
        if not ray.is_initialized():
            raise TaskStateStoreUnavailableError("Ray not initialized")
        if self._ray_actor is not None:
            if not require_ready:
                return self._ray_actor
            try:
                out = await async_get_ray_ref(self._ray_actor.stats.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"TaskStateStore.stats returned non-dict: {type(out)}")
                return self._ray_actor
            except Exception:
                self._reset_ray_actor()
        import ray

        actor_name = _ray_task_state_store_actor_name()
        try:
            self._ray_actor = await asyncio.to_thread(
                ray.get_actor,
                actor_name,
                namespace=_ray_namespace(),
            )
        except Exception:
            try:
                self._ray_actor = _create_ray_actor(require_ready=require_ready)
            except Exception as e:
                raise TaskStateStoreUnavailableError(
                    "Failed to get/create detached Ray TaskStateStore actor"
                ) from e
        return self._ray_actor

    async def _call(self, method: str, **kwargs: Any) -> Any:
        actor = await self._get_ray_actor_async()
        remote = getattr(actor, method).remote
        return await async_get_ray_ref(remote(**kwargs))

    async def async_ensure_started(self) -> None:
        await self._get_ray_actor_async(require_ready=False)

    async def async_stats(self) -> dict[str, Any]:
        out = await self._call("stats")
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.stats returned non-dict: {type(out)}")
        return out

    async def async_integrity_check(self) -> str:
        return str(await self._call("integrity_check"))

    async def async_acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("acquire_scheduler_owner", **kwargs)

    async def async_renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("renew_scheduler_owner", **kwargs)

    async def async_create_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("create_task", **kwargs)

    async def async_ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("ensure_task", **kwargs)

    async def async_update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("update_task_metadata", **kwargs)

    async def async_complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("complete_task_success", **kwargs)

    async def async_complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("complete_task_failure", **kwargs)

    async def async_mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_task_retrieved", **kwargs)

    async def async_forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("forget_task", **kwargs)

    async def async_assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("assign_task", **kwargs)

    async def async_claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("claim_task", **kwargs)

    async def async_begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("begin_finalize", **kwargs)

    async def async_commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("commit_finalize_success", **kwargs)

    async def async_commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("commit_finalize_failure", **kwargs)

    async def async_requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("requeue_task", **kwargs)

    async def async_get_task(self, request_id: str) -> dict[str, Any]:
        return await self._dict_call("get_task", request_id=str(request_id))

    async def async_list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        out = await self._call("list_active_tasks", limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_active_tasks returned non-list: {type(out)}")
        return out

    async def async_list_expired_leases(
        self,
        *,
        now: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        out = await self._call("list_expired_leases", now=now, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_expired_leases returned non-list: {type(out)}")
        return out

    async def async_list_tasks_by_metadata(
        self,
        *,
        filters: dict[str, Any] | None = None,
        statuses: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        out = await self._call("list_tasks_by_metadata", filters=filters, statuses=statuses, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_tasks_by_metadata returned non-list: {type(out)}")
        return out

    async def _dict_call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        out = await self._call(method, **kwargs)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.{method} returned non-dict: {type(out)}")
        return out


class TaskStateFutureStore:
    """Future polling facade backed by TaskStateStore and TaskPayloadStore.

    This owns the external Tinker future lifecycle while the durable state lives
    in TaskStateStore. It intentionally mirrors the old async future methods so
    routes can migrate without changing public polling semantics.
    """

    def __init__(
        self,
        *,
        task_state_client: TaskStateStoreClient | None = None,
        payload_store: Any | None = None,
    ) -> None:
        self._task_state = task_state_client if task_state_client is not None else task_state_store
        self._payload_store = payload_store

    @property
    def _payloads(self) -> Any:
        if self._payload_store is None:
            from .task_payload_store import TaskPayloadStore

            self._payload_store = TaskPayloadStore()
        return self._payload_store

    async def async_create_with_id(self, request_id: str) -> str:
        await self._task_state.async_ensure_task(request_id=str(request_id), status="pending")
        return str(request_id)

    async def async_create_model_work_with_id(
        self,
        request_id: str,
        *,
        op: str,
        domain_key: str,
        request_json: bytes,
        meta: dict[str, Any] | None = None,
        payload_hash: str | None = None,
    ) -> str:
        await self._task_state.async_ensure_task(
            request_id=str(request_id),
            op=str(op),
            domain_key=str(domain_key),
            request_json=bytes(request_json),
            payload_hash=payload_hash,
            metadata=dict(meta or {}),
            status="pending",
        )
        return str(request_id)

    async def async_ensure_pending(self, request_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        out = await self._task_state.async_ensure_task(
            request_id=str(request_id),
            op=str((meta or {}).get("op") or "unknown"),
            domain_key=str((meta or {}).get("domain_key") or "future:default"),
            metadata=dict(meta or {}),
            status="pending",
        )
        return {"created": bool(out.get("created")), "meta": out.get("record", {}).get("metadata") or {}}

    async def async_mark_queued(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        await self._task_state.async_ensure_task(
            request_id=str(request_id),
            op=str((meta or {}).get("op") or "unknown"),
            domain_key=str((meta or {}).get("domain_key") or "future:default"),
            metadata={**dict(meta or {}), "queue_state": "queued"},
            status="queued",
        )

    async def async_mark_running(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        await self._task_state.async_update_task_metadata(
            request_id=str(request_id),
            metadata={**dict(meta or {}), "queue_state": "running"},
            status="running",
        )

    async def async_update_meta(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        await self._task_state.async_update_task_metadata(
            request_id=str(request_id),
            metadata=dict(meta or {}),
        )

    async def async_resolve(self, request_id: str, result: Any) -> None:
        if self._buffer_model_work_finalize(kind="resolve", request_id=request_id, payload=result):
            return
        meta = await self.async_get_meta(request_id)
        result = _sync_training_session_step(meta, result)
        payload = await asyncio.to_thread(
            self._payloads.write_json_payload,
            request_id=str(request_id),
            attempt_id=str((meta or {}).get("model_work_attempt_id") or "future"),
            payload=result,
        )
        await self._task_state.async_complete_task_success(
            request_id=str(request_id),
            result_path=str(payload["path"]),
            result_checksum=str(payload["checksum"]),
            result_size_bytes=int(payload["size_bytes"]),
            metadata={"done_at": time.time(), "final_status": FutureStatus.DONE.value},
        )

    async def async_fail(self, request_id: str, error: str) -> None:
        if self._buffer_model_work_finalize(kind="fail", request_id=request_id, payload=str(error)):
            return
        try:
            await self._task_state.async_complete_task_failure(
                request_id=str(request_id),
                error=str(error),
                metadata={"failed_at": time.time(), "done_at": time.time(), "final_status": FutureStatus.FAILED.value},
            )
        except (KeyError, TaskStateNotFoundError):
            await self._task_state.async_ensure_task(
                request_id=str(request_id),
                status="pending",
                metadata={"failed_at": time.time()},
            )
            await self._task_state.async_complete_task_failure(
                request_id=str(request_id),
                error=str(error),
                metadata={"failed_at": time.time(), "done_at": time.time(), "final_status": FutureStatus.FAILED.value},
            )

    async def async_fail_if_pending_meta_matches(
        self,
        request_id: str,
        error: str,
        *,
        expected_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            record = await self._task_state.async_get_task(str(request_id))
        except (KeyError, TaskStateNotFoundError):
            return {"failed": False, "reason": "unknown"}
        if _status_from_task_record(record) != FutureStatus.PENDING:
            return {"failed": False, "reason": "not_pending"}
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key, value in dict(expected_meta or {}).items():
            if metadata.get(key) != value:
                return {"failed": False, "reason": "meta_mismatch"}
        await self.async_fail(str(request_id), str(error))
        return {"failed": True}

    async def async_get_status(self, request_id: str) -> FutureStatus:
        try:
            record = await self._task_state.async_get_task(str(request_id))
        except (KeyError, TaskStateNotFoundError):
            raise KeyError(f"Unknown request_id: {request_id}") from None
        return _status_from_task_record(record)

    async def async_get_result(self, request_id: str) -> Any:
        record = await self._task_state.async_get_task(str(request_id))
        if str(record.get("status")) == "retrieved":
            raise KeyError(f"Future already retrieved: {request_id}")
        if str(record.get("status")) != "done":
            raise KeyError(f"Future is not done: {request_id}")
        result_path = record.get("result_path")
        if not isinstance(result_path, str) or not result_path:
            raise KeyError(f"Future result payload missing: {request_id}")
        payload = await asyncio.to_thread(
            self._payloads.read_json_payload,
            path=result_path,
            expected_checksum=record.get("result_checksum"),
        )
        await self._task_state.async_mark_task_retrieved(request_id=str(request_id))
        return payload

    async def async_get_error(self, request_id: str) -> str | None:
        record = await self._task_state.async_get_task(str(request_id))
        return None if record.get("error") is None else str(record.get("error"))

    async def async_get_meta(self, request_id: str) -> dict[str, Any] | None:
        record = await self._task_state.async_get_task(str(request_id))
        metadata = record.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else None

    async def async_list_pending_by_meta(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        records = await self._task_state.async_list_tasks_by_metadata(
            filters=dict(filters or {}),
            statuses=["pending", "queued", "assigned", "leased", "running", "finalizing"],
            limit=int(limit),
        )
        return [
            {
                "request_id": record["request_id"],
                "status": record["status"],
                "meta": record.get("metadata") or {},
            }
            for record in records
        ]

    async def async_fail_training_requests_for_model(self, model_id: str, error: str) -> list[str]:
        return await self._fail_requests_by_metadata(
            filters={"model_id": str(model_id)},
            op_prefix="training.",
            error=error,
        )

    async def async_fail_sampling_requests_for_session(self, sampling_session_id: str, error: str) -> list[str]:
        return await self._fail_requests_by_metadata(
            filters={"sampling_session_id": str(sampling_session_id)},
            op_prefix="sampling.",
            error=error,
        )

    async def _fail_requests_by_metadata(
        self,
        *,
        filters: dict[str, Any],
        op_prefix: str,
        error: str,
    ) -> list[str]:
        records = await self._task_state.async_list_tasks_by_metadata(
            filters=dict(filters),
            statuses=["pending", "queued", "assigned", "leased", "running", "finalizing"],
            limit=10000,
        )
        failed: list[str] = []
        for record in records:
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            op = metadata.get("op") or record.get("op")
            if not str(op or "").startswith(op_prefix):
                continue
            await self.async_fail(str(record["request_id"]), str(error))
            failed.append(str(record["request_id"]))
        return failed

    async def async_forget(self, request_id: str) -> None:
        await self._task_state.async_forget_task(request_id=str(request_id))

    async def async_cleanup(self, request_id: str) -> None:
        try:
            await self._task_state.async_mark_task_retrieved(request_id=str(request_id))
        except Exception:
            await self._task_state.async_forget_task(request_id=str(request_id))

    async def async_reap(self) -> dict[str, list[str]]:
        return {"expired": [], "timed_out": []}

    async def async_fail_stale_running_requests(self, active_consumer_job_id: str, error: str) -> list[str]:
        records = await self._task_state.async_list_tasks_by_metadata(
            filters={"consumer_job_id": str(active_consumer_job_id)},
            statuses=["running"],
            limit=10000,
        )
        failed: list[str] = []
        for record in records:
            await self.async_fail(str(record["request_id"]), str(error))
            failed.append(str(record["request_id"]))
        return failed

    async def async_ensure_started(self) -> None:
        await self._task_state.async_ensure_started()

    async def async_ensure_ready(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        _ = timeout_s
        return await self._task_state.async_stats()

    def metrics_snapshot(self) -> dict[str, Any]:
        return {"backend": "task_state_store"}

    async def async_rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        _ = timeout_s
        return 0

    async def async_debug_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        _ = timeout_s
        try:
            return {"backend": "task_state_store", "task_state_store": await self._task_state.async_stats()}
        except Exception as e:
            return {"backend": "task_state_store", "error": f"{type(e).__name__}: {e}"}

    def _buffer_model_work_finalize(self, *, kind: str, request_id: str, payload: Any) -> bool:
        buffer = get_current_model_work_finalize_buffer()
        if buffer is None:
            return False
        buffer.finalization = ModelWorkFinalize(kind=kind, request_id=str(request_id), payload=payload)
        return True


task_state_store = TaskStateStoreClient()
task_state_futures = TaskStateFutureStore()
