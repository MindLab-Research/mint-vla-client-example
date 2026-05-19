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
from ..runtime_env import env_nonempty
from .async_ray_control import async_get_ray_ref, sync_get_ray_ref
from .model_work_execution_context import ModelWorkFinalize, get_current_model_work_finalize_buffer


ACTIVE_TASK_STATUSES = frozenset({"pending", "queued", "running", "assigned", "leased", "finalizing"})
TERMINAL_TASK_STATUSES = frozenset({"done", "failed", "cancelled", "expired", "retrieved"})
_REAPER_METRICS: dict[str, float] = {
    "expire_pending": 0.0,
    "evict_payload": 0.0,
    "delete_tombstone": 0.0,
}
_REAPER_PAYLOAD_EVICT_ERRORS_TOTAL = 0.0


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


def _inc_reaper_rows(action: str, count: int) -> None:
    if int(count) <= 0:
        return
    key = str(action)
    _REAPER_METRICS[key] = float(_REAPER_METRICS.get(key, 0.0)) + float(count)
    try:
        from ..logging_context import record_task_future_reaper_rows_metric

        record_task_future_reaper_rows_metric(action=key, count=int(count))
    except Exception:
        pass


def _inc_payload_evict_errors(count: int = 1) -> None:
    global _REAPER_PAYLOAD_EVICT_ERRORS_TOTAL
    if int(count) <= 0:
        return
    _REAPER_PAYLOAD_EVICT_ERRORS_TOTAL += float(count)
    try:
        from ..logging_context import record_task_future_payload_evict_error_metric

        record_task_future_payload_evict_error_metric(count=int(count))
    except Exception:
        pass


def task_future_reaper_metrics_snapshot() -> dict[str, Any]:
    return {
        "rows_total": dict(_REAPER_METRICS),
        "payload_evict_errors_total": float(_REAPER_PAYLOAD_EVICT_ERRORS_TOTAL),
    }


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
        self._sampling_sessions: dict[str, dict[str, Any]] = {}
        self._session_index: dict[str, dict[str, Any]] = {}
        self._sampler_index: dict[str, dict[str, Any]] = {}
        self._session_heartbeats: dict[str, float] = {}
        self._training_sessions: dict[str, dict[str, Any]] = {}
        self._gateway_sampling_sessions: dict[str, dict[str, str]] = {}
        self._gateway_training_models: dict[str, dict[str, str | None]] = {}
        self._session_heartbeat_max_age_s = float(
            os.environ.get("MINT_SESSION_HEARTBEAT_MAX_AGE_S", str(7 * 24 * 3600))
        )
        self._session_heartbeat_prune_every = max(
            1,
            int(os.environ.get("MINT_SESSION_HEARTBEAT_PRUNE_EVERY", "256")),
        )
        self._session_heartbeat_updates_since_prune = 0
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

    def ping(self) -> dict[str, Any]:
        with self._lock:
            self._conn.execute("SELECT 1").fetchone()
        return {"ok": True}

    def upsert_sampling_session(self, *, session_id: str, info: dict[str, Any]) -> None:
        session_id = str(session_id)
        if not session_id:
            raise ValueError("session_id is required")
        incoming = dict(info)
        incoming["session_id"] = session_id
        incoming.setdefault("last_activity", time.time())
        incoming.setdefault("lora_loaded", False)
        incoming.setdefault("uses_base_model", False)
        incoming.setdefault("inflight_requests", 0)
        incoming_version = int(incoming.get("metadata_version") or 0)
        with self._lock:
            existing = self._sampling_sessions.get(session_id)
            if existing is not None:
                existing_version = int(existing.get("metadata_version") or 0)
                if incoming_version and incoming_version < existing_version:
                    if "last_activity" in incoming:
                        existing["last_activity"] = max(
                            float(existing.get("last_activity") or 0.0),
                            float(incoming.get("last_activity") or 0.0),
                        )
                    return
                merged = {**existing, **incoming}
                merged["metadata_version"] = max(existing_version + 1, incoming_version, 1)
                self._sampling_sessions[session_id] = merged
                return
            incoming["metadata_version"] = max(incoming_version, 1)
            self._sampling_sessions[session_id] = incoming

    def delete_sampling_session(self, *, session_id: str) -> None:
        with self._lock:
            self._sampling_sessions.pop(str(session_id), None)

    def set_sampling_session_last_activity(self, *, session_id: str, last_activity: float) -> float | None:
        ts = float(last_activity)
        with self._lock:
            existing = self._sampling_sessions.get(str(session_id))
            if existing is None:
                return None
            existing["last_activity"] = ts
            existing["metadata_version"] = int(existing.get("metadata_version") or 0) + 1
            return ts

    def get_sampling_session(self, *, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            existing = self._sampling_sessions.get(str(session_id))
            return dict(existing) if existing is not None else None

    def list_sampling_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(value) for value in self._sampling_sessions.values()]

    def upsert_training_session(self, *, model_id: str, info: dict[str, Any]) -> None:
        model_id = str(model_id)
        if not model_id:
            raise ValueError("model_id is required")
        incoming = dict(info)
        incoming["model_id"] = model_id
        incoming.setdefault("current_step", 0)
        incoming.setdefault("last_activity", time.time())
        incoming_version = max(1, int(incoming.get("metadata_version") or 1))
        with self._lock:
            current = dict(self._training_sessions.get(model_id, {}))
            current_version = max(1, int(current.get("metadata_version") or 1))
            if incoming_version < current_version:
                incoming_last_activity = float(incoming.get("last_activity", 0.0) or 0.0)
                current_last_activity = float(current.get("last_activity", 0.0) or 0.0)
                current["last_activity"] = max(current_last_activity, incoming_last_activity)
                try:
                    incoming_step = int(incoming.get("current_step", 0))
                    current_step = int(current.get("current_step", 0))
                    current["current_step"] = max(current_step, incoming_step)
                except Exception:
                    pass
                self._training_sessions[model_id] = current
                return
            merged = {**current, **incoming}
            merged.setdefault("current_step", int(current.get("current_step", 0)))
            merged.setdefault("last_activity", time.time())
            merged["metadata_version"] = incoming_version
            self._training_sessions[model_id] = merged

    def delete_training_session(self, *, model_id: str) -> None:
        with self._lock:
            self._training_sessions.pop(str(model_id), None)

    def set_training_session_last_activity(self, *, model_id: str, last_activity: float) -> float | None:
        with self._lock:
            info = self._training_sessions.get(str(model_id))
            if info is None:
                return None
            info["last_activity"] = float(last_activity)
            return float(info["last_activity"])

    def get_training_session(self, *, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            info = self._training_sessions.get(str(model_id))
            return dict(info) if info is not None else None

    def bump_training_session_step(self, *, model_id: str) -> int:
        with self._lock:
            info = self._training_sessions.get(str(model_id))
            if info is None:
                return 0
            info["current_step"] = int(info.get("current_step", 0)) + 1
            return int(info["current_step"])

    def set_training_session_step(self, *, model_id: str, step: int) -> int:
        with self._lock:
            info = self._training_sessions.get(str(model_id))
            if info is None:
                return int(step)
            info["current_step"] = max(int(info.get("current_step", 0)), int(step))
            return int(info["current_step"])

    def list_training_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(value) for value in self._training_sessions.values()]

    def upsert_gateway_sampling_session(
        self,
        *,
        sampling_session_id: str,
        upstream_alias: str,
        base_model: str,
    ) -> None:
        with self._lock:
            self._gateway_sampling_sessions[str(sampling_session_id)] = {
                "upstream_alias": str(upstream_alias),
                "base_model": str(base_model),
            }

    def get_gateway_sampling_session(self, *, sampling_session_id: str) -> dict[str, str] | None:
        with self._lock:
            info = self._gateway_sampling_sessions.get(str(sampling_session_id))
            return dict(info) if info is not None else None

    def delete_gateway_sampling_session(self, *, sampling_session_id: str) -> None:
        with self._lock:
            self._gateway_sampling_sessions.pop(str(sampling_session_id), None)

    def upsert_gateway_training_model(
        self,
        *,
        model_id: str,
        upstream_alias: str,
        base_model: str,
        owner_id: str | None = None,
    ) -> None:
        with self._lock:
            self._gateway_training_models[str(model_id)] = {
                "upstream_alias": str(upstream_alias),
                "base_model": str(base_model),
                "owner_id": None if owner_id is None else str(owner_id),
            }

    def get_gateway_training_model(self, *, model_id: str) -> dict[str, str | None] | None:
        with self._lock:
            info = self._gateway_training_models.get(str(model_id))
            return dict(info) if info is not None else None

    def delete_gateway_training_model(self, *, model_id: str) -> None:
        with self._lock:
            self._gateway_training_models.pop(str(model_id), None)

    def list_gateway_routes(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sampling_sessions": dict(self._gateway_sampling_sessions),
                "training_models": dict(self._gateway_training_models),
            }

    def upsert_session_index(self, *, session_id: str, info: dict[str, Any]) -> None:
        session_id = str(session_id)
        if not session_id:
            raise ValueError("session_id is required")
        incoming = dict(info)
        incoming["session_id"] = session_id
        with self._lock:
            existing = self._session_index.get(session_id, {})
            merged = {**existing, **incoming}
            merged.setdefault("training_run_ids", list(existing.get("training_run_ids") or []))
            merged.setdefault("sampler_ids", list(existing.get("sampler_ids") or []))
            merged.setdefault("heartbeat_sampler_ids", list(existing.get("heartbeat_sampler_ids") or []))
            self._session_index[session_id] = merged

    def add_training_run_to_session_index(
        self,
        *,
        session_id: str,
        training_run_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        session_id = str(session_id)
        training_run_id = str(training_run_id)
        with self._lock:
            item = dict(self._session_index.get(session_id) or {"session_id": session_id})
            runs = list(item.get("training_run_ids") or [])
            if training_run_id not in runs:
                runs.append(training_run_id)
            item["training_run_ids"] = runs
            item.setdefault("sampler_ids", [])
            item.setdefault("heartbeat_sampler_ids", [])
            if user_id is not None:
                item.setdefault("user_id", str(user_id))
            if created_at is not None:
                item.setdefault("created_at", created_at)
            self._session_index[session_id] = item

    def add_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        session_id = str(session_id)
        sampler_id = str(sampler_id)
        with self._lock:
            item = dict(self._session_index.get(session_id) or {"session_id": session_id})
            samplers = list(item.get("sampler_ids") or [])
            if sampler_id not in samplers:
                samplers.append(sampler_id)
            item["sampler_ids"] = samplers
            item.setdefault("training_run_ids", [])
            item.setdefault("heartbeat_sampler_ids", list(item.get("heartbeat_sampler_ids") or []))
            if user_id is not None:
                item.setdefault("user_id", str(user_id))
            if created_at is not None:
                item.setdefault("created_at", created_at)
            self._session_index[session_id] = item

    def add_heartbeat_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        session_id = str(session_id)
        sampler_id = str(sampler_id)
        with self._lock:
            item = dict(self._session_index.get(session_id) or {"session_id": session_id})
            samplers = list(item.get("sampler_ids") or [])
            if sampler_id not in samplers:
                samplers.append(sampler_id)
            heartbeat_samplers = list(item.get("heartbeat_sampler_ids") or [])
            if sampler_id not in heartbeat_samplers:
                heartbeat_samplers.append(sampler_id)
            item["sampler_ids"] = samplers
            item["heartbeat_sampler_ids"] = heartbeat_samplers
            item.setdefault("training_run_ids", [])
            if user_id is not None:
                item.setdefault("user_id", str(user_id))
            if created_at is not None:
                item.setdefault("created_at", created_at)
            self._session_index[session_id] = item

    def remove_sampler_from_session_index(self, *, session_id: str, sampler_id: str) -> None:
        with self._lock:
            item = self._session_index.get(str(session_id))
            if item is None:
                return
            sid = str(sampler_id)
            item["sampler_ids"] = [x for x in list(item.get("sampler_ids") or []) if str(x) != sid]
            item["heartbeat_sampler_ids"] = [
                x for x in list(item.get("heartbeat_sampler_ids") or []) if str(x) != sid
            ]

    def get_session_index(self, *, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            existing = self._session_index.get(str(session_id))
            return dict(existing) if existing is not None else None

    def list_session_index(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(value) for value in self._session_index.values()]

    def upsert_sampler_index(self, *, sampler_id: str, info: dict[str, Any]) -> None:
        sampler_id = str(sampler_id)
        if not sampler_id:
            raise ValueError("sampler_id is required")
        incoming = dict(info)
        incoming["sampler_id"] = sampler_id
        with self._lock:
            self._sampler_index[sampler_id] = {**self._sampler_index.get(sampler_id, {}), **incoming}

    def delete_sampler_index(self, *, sampler_id: str) -> None:
        with self._lock:
            self._sampler_index.pop(str(sampler_id), None)

    def get_sampler_index(self, *, sampler_id: str) -> dict[str, Any] | None:
        with self._lock:
            existing = self._sampler_index.get(str(sampler_id))
            return dict(existing) if existing is not None else None

    def list_sampler_index(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(value) for value in self._sampler_index.values()]

    def update_session_heartbeat(self, *, session_id: str, now: float | None = None) -> None:
        session_id = str(session_id)
        if not session_id:
            return
        ts = _now(now)
        with self._lock:
            self._session_heartbeats[session_id] = ts
            self._session_heartbeat_updates_since_prune += 1
            if self._session_heartbeat_updates_since_prune >= self._session_heartbeat_prune_every:
                self._session_heartbeat_updates_since_prune = 0
                self._prune_session_heartbeats_locked(now=ts, max_age_s=self._session_heartbeat_max_age_s)

    def get_session_heartbeat(self, *, session_id: str) -> float | None:
        with self._lock:
            value = self._session_heartbeats.get(str(session_id))
            return None if value is None else float(value)

    def delete_session_heartbeat(self, *, session_id: str) -> bool:
        with self._lock:
            return self._session_heartbeats.pop(str(session_id), None) is not None

    def session_heartbeat_size(self) -> int:
        with self._lock:
            return len(self._session_heartbeats)

    def is_session_heartbeat_stale(self, *, session_id: str, ttl_s: float, now: float | None = None) -> bool:
        ttl = float(ttl_s)
        if ttl <= 0:
            return False
        session_id = str(session_id)
        if not session_id:
            return False
        ts = _now(now)
        with self._lock:
            last = self._session_heartbeats.get(session_id)
        if last is None:
            return False
        return (ts - float(last)) > ttl

    def prune_session_heartbeats(self, *, max_age_s: float, now: float | None = None) -> int:
        with self._lock:
            return self._prune_session_heartbeats_locked(now=_now(now), max_age_s=float(max_age_s))

    def _prune_session_heartbeats_locked(self, *, now: float, max_age_s: float) -> int:
        if max_age_s <= 0:
            return 0
        cutoff = float(now) - float(max_age_s)
        stale = [sid for sid, seen in self._session_heartbeats.items() if float(seen) < cutoff]
        for sid in stale:
            self._session_heartbeats.pop(sid, None)
        return len(stale)

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

                CREATE INDEX IF NOT EXISTS idx_tasks_result_path
                    ON tasks(result_path);
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

    def expire_active_tasks(
        self,
        *,
        older_than_s: float,
        now: float | None = None,
        limit: int = 1000,
    ) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        ts = _now(now)
        cutoff = ts - ttl_s
        batch_limit = max(0, int(limit))
        if batch_limit <= 0:
            return []
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE status IN ('pending', 'queued', 'assigned')
                  AND created_at <= ?
                ORDER BY created_at, request_id
                LIMIT ?
                """,
                (cutoff, batch_limit),
            ).fetchall()
            expired: list[str] = []
            for row in rows:
                request_id = str(row["request_id"])
                metadata = _json_loads(row["metadata_json"])
                metadata.setdefault("terminal_status", "expired")
                metadata.setdefault("expired_at", ts)
                metadata.setdefault("failed_at", ts)
                cur = conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'expired',
                        error = COALESCE(error, ?),
                        metadata_json = ?,
                        updated_at = ?
                    WHERE request_id = ?
                      AND status IN ('pending', 'queued', 'assigned')
                    """,
                    ("Future expired", _json_dumps(metadata), ts, request_id),
                )
                if cur.rowcount == 1:
                    self._record_event(
                        conn,
                        request_id=request_id,
                        event_type="task_expired",
                        payload={"reason": "pending_ttl", "ttl_s": ttl_s},
                        now=ts,
                    )
                    expired.append(request_id)
            _inc_reaper_rows("expire_pending", len(expired))
            return expired

    def list_terminal_payloads_for_eviction(
        self,
        *,
        older_than_s: float,
        now: float | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        sql = """
            SELECT * FROM tasks
            WHERE status IN ('done', 'failed', 'cancelled', 'expired', 'retrieved')
              AND result_path IS NOT NULL
              AND result_path != ''
            ORDER BY updated_at, request_id
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, (max(0, int(limit)),)).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            record = self._row_to_record(row)
            if self._terminal_completed_at(record) <= cutoff:
                out.append(record)
        return out

    def mark_payload_evicted(
        self,
        *,
        request_id: str,
        expected_result_path: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._transaction() as conn:
            row = self._get_row(conn, request_id)
            if str(row["status"]) not in TERMINAL_TASK_STATUSES:
                raise TaskStateConflictError(f"cannot evict payload; current status={row['status']!r}")
            if str(row["result_path"] or "") != str(expected_result_path):
                return {"ok": False, "reason": "payload_changed", "record": self._row_to_record(row)}
            metadata = _json_loads(row["metadata_json"])
            metadata.setdefault("terminal_status", str(row["status"]))
            metadata["payload_evicted_at"] = ts
            metadata.setdefault("evicted_result_size_bytes", row["result_size_bytes"])
            cur = conn.execute(
                """
                UPDATE tasks
                SET result_path = NULL,
                    result_checksum = NULL,
                    result_size_bytes = NULL,
                    metadata_json = ?,
                    updated_at = ?
                WHERE request_id = ?
                  AND result_path = ?
                """,
                (_json_dumps(metadata), ts, str(request_id), str(expected_result_path)),
            )
            if cur.rowcount != 1:
                return {"ok": False, "reason": "payload_changed", "record": self._row_to_record(self._get_row(conn, request_id))}
            _inc_reaper_rows("evict_payload", 1)
            self._record_event(
                conn,
                request_id=str(request_id),
                event_type="task_payload_evicted",
                payload={"result_path": str(expected_result_path)},
                now=ts,
            )
            return {"ok": True, "record": self._row_to_record(self._get_row(conn, request_id))}

    def delete_expired_tombstones(
        self,
        *,
        older_than_s: float,
        now: float | None = None,
        limit: int = 1000,
    ) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        batch_limit = max(0, int(limit))
        if batch_limit <= 0:
            return []
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tasks
                WHERE status IN ('done', 'failed', 'cancelled', 'expired', 'retrieved')
                  AND (result_path IS NULL OR result_path = '')
                ORDER BY updated_at, request_id
                LIMIT ?
                """,
                (batch_limit,),
            ).fetchall()
            deleted: list[str] = []
            for row in rows:
                record = self._row_to_record(row)
                if self._terminal_completed_at(record) > cutoff:
                    continue
                request_id = str(row["request_id"])
                conn.execute("DELETE FROM task_events WHERE request_id = ?", (request_id,))
                cur = conn.execute("DELETE FROM tasks WHERE request_id = ?", (request_id,))
                if cur.rowcount == 1:
                    deleted.append(request_id)
            _inc_reaper_rows("delete_tombstone", len(deleted))
            return deleted

    def record_payload_evict_error(self, *, count: int = 1) -> dict[str, Any]:
        _inc_payload_evict_errors(int(count))
        return {"ok": True, "metrics": task_future_reaper_metrics_snapshot()}

    def _terminal_completed_at(self, record: dict[str, Any]) -> float:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key in ("done_at", "failed_at"):
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            try:
                if value is not None:
                    return float(value)
            except Exception:
                pass
        return float(record.get("updated_at") or 0.0)

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
    v = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from ..config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


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
            "/vePFS-Mindverse/share/mint/dev/data/task-state/task_state.sqlite3",
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
            "task_future_reaper": task_future_reaper_metrics_snapshot(),
        }

    def ping(self) -> dict[str, Any]:
        out = self._store.ping()
        return {
            "ok": bool(out.get("ok")),
            "actor_name": _ray_task_state_store_actor_name(),
            "namespace": _ray_namespace(),
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

    def expire_active_tasks(self, **kwargs: Any) -> list[str]:
        return self._store.expire_active_tasks(**kwargs)

    def list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_terminal_payloads_for_eviction(**kwargs)

    def mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.mark_payload_evicted(**kwargs)

    def delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        return self._store.delete_expired_tombstones(**kwargs)

    def record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.record_payload_evict_error(**kwargs)

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

    def upsert_sampling_session(self, **kwargs: Any) -> None:
        return self._store.upsert_sampling_session(**kwargs)

    def delete_sampling_session(self, **kwargs: Any) -> None:
        return self._store.delete_sampling_session(**kwargs)

    def set_sampling_session_last_activity(self, **kwargs: Any) -> float | None:
        return self._store.set_sampling_session_last_activity(**kwargs)

    def get_sampling_session(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_sampling_session(**kwargs)

    def list_sampling_sessions(self) -> list[dict[str, Any]]:
        return self._store.list_sampling_sessions()

    def upsert_training_session(self, **kwargs: Any) -> None:
        return self._store.upsert_training_session(**kwargs)

    def delete_training_session(self, **kwargs: Any) -> None:
        return self._store.delete_training_session(**kwargs)

    def set_training_session_last_activity(self, **kwargs: Any) -> float | None:
        return self._store.set_training_session_last_activity(**kwargs)

    def get_training_session(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_training_session(**kwargs)

    def bump_training_session_step(self, **kwargs: Any) -> int:
        return self._store.bump_training_session_step(**kwargs)

    def set_training_session_step(self, **kwargs: Any) -> int:
        return self._store.set_training_session_step(**kwargs)

    def list_training_sessions(self) -> list[dict[str, Any]]:
        return self._store.list_training_sessions()

    def upsert_gateway_sampling_session(self, **kwargs: Any) -> None:
        return self._store.upsert_gateway_sampling_session(**kwargs)

    def get_gateway_sampling_session(self, **kwargs: Any) -> dict[str, str] | None:
        return self._store.get_gateway_sampling_session(**kwargs)

    def delete_gateway_sampling_session(self, **kwargs: Any) -> None:
        return self._store.delete_gateway_sampling_session(**kwargs)

    def upsert_gateway_training_model(self, **kwargs: Any) -> None:
        return self._store.upsert_gateway_training_model(**kwargs)

    def get_gateway_training_model(self, **kwargs: Any) -> dict[str, str | None] | None:
        return self._store.get_gateway_training_model(**kwargs)

    def delete_gateway_training_model(self, **kwargs: Any) -> None:
        return self._store.delete_gateway_training_model(**kwargs)

    def list_gateway_routes(self) -> dict[str, Any]:
        return self._store.list_gateway_routes()

    def upsert_session_index(self, **kwargs: Any) -> None:
        return self._store.upsert_session_index(**kwargs)

    def add_training_run_to_session_index(self, **kwargs: Any) -> None:
        return self._store.add_training_run_to_session_index(**kwargs)

    def add_sampler_to_session_index(self, **kwargs: Any) -> None:
        return self._store.add_sampler_to_session_index(**kwargs)

    def add_heartbeat_sampler_to_session_index(self, **kwargs: Any) -> None:
        return self._store.add_heartbeat_sampler_to_session_index(**kwargs)

    def remove_sampler_from_session_index(self, **kwargs: Any) -> None:
        return self._store.remove_sampler_from_session_index(**kwargs)

    def get_session_index(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_session_index(**kwargs)

    def list_session_index(self) -> list[dict[str, Any]]:
        return self._store.list_session_index()

    def upsert_sampler_index(self, **kwargs: Any) -> None:
        return self._store.upsert_sampler_index(**kwargs)

    def delete_sampler_index(self, **kwargs: Any) -> None:
        return self._store.delete_sampler_index(**kwargs)

    def get_sampler_index(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_sampler_index(**kwargs)

    def list_sampler_index(self) -> list[dict[str, Any]]:
        return self._store.list_sampler_index()

    def update_session_heartbeat(self, **kwargs: Any) -> None:
        return self._store.update_session_heartbeat(**kwargs)

    def get_session_heartbeat(self, **kwargs: Any) -> float | None:
        return self._store.get_session_heartbeat(**kwargs)

    def delete_session_heartbeat(self, **kwargs: Any) -> bool:
        return self._store.delete_session_heartbeat(**kwargs)

    def session_heartbeat_size(self) -> int:
        return self._store.session_heartbeat_size()

    def is_session_heartbeat_stale(self, **kwargs: Any) -> bool:
        return self._store.is_session_heartbeat_stale(**kwargs)

    def prune_session_heartbeats(self, **kwargs: Any) -> int:
        return self._store.prune_session_heartbeats(**kwargs)


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

    def _get_ray_actor_sync(self, *, require_ready: bool = True, create_if_missing: bool = True):
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
                out = sync_get_ray_ref(self._ray_actor.stats.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"TaskStateStore.stats returned non-dict: {type(out)}")
                return self._ray_actor
            except Exception:
                self._reset_ray_actor()
        actor_name = _ray_task_state_store_actor_name()
        try:
            self._ray_actor = ray.get_actor(actor_name, namespace=_ray_namespace())
        except Exception:
            if not create_if_missing:
                raise TaskStateStoreUnavailableError(
                    f"Detached Ray TaskStateStore actor unavailable actor_name={actor_name!r}"
                )
            try:
                self._ray_actor = _create_ray_actor(require_ready=require_ready)
            except Exception as e:
                raise TaskStateStoreUnavailableError(
                    "Failed to get/create detached Ray TaskStateStore actor"
                ) from e
        return self._ray_actor

    async def _get_ray_actor_async(self, *, require_ready: bool = True, create_if_missing: bool = True):
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
            if not create_if_missing:
                raise TaskStateStoreUnavailableError(
                    f"Detached Ray TaskStateStore actor unavailable actor_name={actor_name!r}"
                )
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

    def _call_sync(self, method: str, **kwargs: Any) -> Any:
        actor = self._get_ray_actor_sync()
        remote = getattr(actor, method).remote
        return sync_get_ray_ref(remote(**kwargs))

    def ensure_ready(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = self._get_ray_actor_sync()
        out = sync_get_ray_ref(actor.stats.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.stats returned non-dict: {type(out)}")
        return out

    def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = self._get_ray_actor_sync(require_ready=False, create_if_missing=False)
        try:
            out = sync_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise TaskStateStoreUnavailableError(f"TaskStateStore ping failed: {out!r}")
        return out

    async def async_ensure_started(self) -> None:
        await self._get_ray_actor_async(require_ready=False)

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=False)
        try:
            out = await async_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise TaskStateStoreUnavailableError(f"TaskStateStore ping failed: {out!r}")
        return out

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

    async def async_expire_active_tasks(self, **kwargs: Any) -> list[str]:
        out = await self._call("expire_active_tasks", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.expire_active_tasks returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("list_terminal_payloads_for_eviction", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_terminal_payloads_for_eviction returned non-list: {type(out)}")
        return out

    async def async_mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_payload_evicted", **kwargs)

    async def async_delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        out = await self._call("delete_expired_tombstones", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.delete_expired_tombstones returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("record_payload_evict_error", **kwargs)

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

    def upsert_sampling_session(self, *, session_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_sampling_session", session_id=str(session_id), info=dict(info))

    async def async_upsert_sampling_session(self, *, session_id: str, info: dict[str, Any]) -> None:
        await self._call("upsert_sampling_session", session_id=str(session_id), info=dict(info))

    def delete_sampling_session(self, *, session_id: str) -> None:
        self._call_sync("delete_sampling_session", session_id=str(session_id))

    async def async_delete_sampling_session(self, *, session_id: str) -> None:
        await self._call("delete_sampling_session", session_id=str(session_id))

    def set_sampling_session_last_activity(self, *, session_id: str, last_activity: float) -> float | None:
        out = self._call_sync(
            "set_sampling_session_last_activity",
            session_id=str(session_id),
            last_activity=float(last_activity),
        )
        return None if out is None else float(out)

    async def async_set_sampling_session_last_activity(self, *, session_id: str, last_activity: float) -> float | None:
        out = await self._call(
            "set_sampling_session_last_activity",
            session_id=str(session_id),
            last_activity=float(last_activity),
        )
        return None if out is None else float(out)

    def get_sampling_session(self, *, session_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_sampling_session", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_sampling_session(self, *, session_id: str) -> dict[str, Any] | None:
        out = await self._call("get_sampling_session", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    def list_sampling_sessions(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_sampling_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampling_sessions returned non-list: {type(out)}")
        return out

    async def async_list_sampling_sessions(self) -> list[dict[str, Any]]:
        out = await self._call("list_sampling_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampling_sessions returned non-list: {type(out)}")
        return out

    def upsert_training_session(self, *, model_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_training_session", model_id=str(model_id), info=dict(info))

    async def async_upsert_training_session(self, *, model_id: str, info: dict[str, Any]) -> None:
        await self._call("upsert_training_session", model_id=str(model_id), info=dict(info))

    def delete_training_session(self, *, model_id: str) -> None:
        self._call_sync("delete_training_session", model_id=str(model_id))

    def set_training_session_last_activity(self, *, model_id: str, last_activity: float) -> float | None:
        out = self._call_sync("set_training_session_last_activity", model_id=str(model_id), last_activity=float(last_activity))
        return None if out is None else float(out)

    def get_training_session(self, *, model_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_training_session", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_training_session(self, *, model_id: str) -> dict[str, Any] | None:
        out = await self._call("get_training_session", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    def bump_training_session_step(self, *, model_id: str) -> int:
        return int(self._call_sync("bump_training_session_step", model_id=str(model_id)))

    def set_training_session_step(self, *, model_id: str, step: int) -> int:
        return int(self._call_sync("set_training_session_step", model_id=str(model_id), step=int(step)))

    def set_training_session_step_best_effort(self, *, model_id: str, step: int) -> None:
        self._call_sync("set_training_session_step", model_id=str(model_id), step=int(step))

    def bump_training_session_step_best_effort(self, *, model_id: str) -> None:
        self._call_sync("bump_training_session_step", model_id=str(model_id))

    def list_training_sessions(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_training_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_training_sessions returned non-list: {type(out)}")
        return out

    async def async_list_training_sessions(self) -> list[dict[str, Any]]:
        out = await self._call("list_training_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_training_sessions returned non-list: {type(out)}")
        return out

    def upsert_gateway_sampling_session(self, *, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
        self._call_sync(
            "upsert_gateway_sampling_session",
            sampling_session_id=str(sampling_session_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
        )

    async def async_upsert_gateway_sampling_session(self, *, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
        await self._call(
            "upsert_gateway_sampling_session",
            sampling_session_id=str(sampling_session_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
        )

    def get_gateway_sampling_session(self, *, sampling_session_id: str) -> dict[str, str] | None:
        out = self._call_sync("get_gateway_sampling_session", sampling_session_id=str(sampling_session_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_gateway_sampling_session(self, *, sampling_session_id: str) -> dict[str, str] | None:
        out = await self._call("get_gateway_sampling_session", sampling_session_id=str(sampling_session_id))
        return dict(out) if isinstance(out, dict) else None

    def delete_gateway_sampling_session(self, *, sampling_session_id: str) -> None:
        self._call_sync("delete_gateway_sampling_session", sampling_session_id=str(sampling_session_id))

    async def async_delete_gateway_sampling_session(self, *, sampling_session_id: str) -> None:
        await self._call("delete_gateway_sampling_session", sampling_session_id=str(sampling_session_id))

    def upsert_gateway_training_model(
        self,
        *,
        model_id: str,
        upstream_alias: str,
        base_model: str,
        owner_id: str | None = None,
    ) -> None:
        self._call_sync(
            "upsert_gateway_training_model",
            model_id=str(model_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
            owner_id=owner_id,
        )

    async def async_upsert_gateway_training_model(
        self,
        *,
        model_id: str,
        upstream_alias: str,
        base_model: str,
        owner_id: str | None = None,
    ) -> None:
        await self._call(
            "upsert_gateway_training_model",
            model_id=str(model_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
            owner_id=owner_id,
        )

    def get_gateway_training_model(self, *, model_id: str) -> dict[str, str | None] | None:
        out = self._call_sync("get_gateway_training_model", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_gateway_training_model(self, *, model_id: str) -> dict[str, str | None] | None:
        out = await self._call("get_gateway_training_model", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    def delete_gateway_training_model(self, *, model_id: str) -> None:
        self._call_sync("delete_gateway_training_model", model_id=str(model_id))

    async def async_delete_gateway_training_model(self, *, model_id: str) -> None:
        await self._call("delete_gateway_training_model", model_id=str(model_id))

    def list_gateway_routes(self) -> dict[str, Any]:
        out = self._call_sync("list_gateway_routes")
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.list_gateway_routes returned non-dict: {type(out)}")
        return out

    def upsert_session_index(self, *, session_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_session_index", session_id=str(session_id), info=dict(info))

    async def async_upsert_session_index(self, *, session_id: str, info: dict[str, Any]) -> None:
        await self._call("upsert_session_index", session_id=str(session_id), info=dict(info))

    def add_training_run_to_session_index(
        self,
        *,
        session_id: str,
        training_run_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        self._call_sync(
            "add_training_run_to_session_index",
            session_id=str(session_id),
            training_run_id=str(training_run_id),
            user_id=user_id,
            created_at=created_at,
        )

    def add_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        self._call_sync(
            "add_sampler_to_session_index",
            session_id=str(session_id),
            sampler_id=str(sampler_id),
            user_id=user_id,
            created_at=created_at,
        )

    def add_heartbeat_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        self._call_sync(
            "add_heartbeat_sampler_to_session_index",
            session_id=str(session_id),
            sampler_id=str(sampler_id),
            user_id=user_id,
            created_at=created_at,
        )

    def remove_sampler_from_session_index(self, *, session_id: str, sampler_id: str) -> None:
        self._call_sync("remove_sampler_from_session_index", session_id=str(session_id), sampler_id=str(sampler_id))

    def get_session_index(self, *, session_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_session_index", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_session_index(self, *, session_id: str) -> dict[str, Any] | None:
        out = await self._call("get_session_index", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    def list_session_index(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_session_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_session_index returned non-list: {type(out)}")
        return out

    async def async_list_session_index(self) -> list[dict[str, Any]]:
        out = await self._call("list_session_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_session_index returned non-list: {type(out)}")
        return out

    def upsert_sampler_index(self, *, sampler_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_sampler_index", sampler_id=str(sampler_id), info=dict(info))

    def delete_sampler_index(self, *, sampler_id: str) -> None:
        self._call_sync("delete_sampler_index", sampler_id=str(sampler_id))

    def get_sampler_index(self, *, sampler_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_sampler_index", sampler_id=str(sampler_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_sampler_index(self, *, sampler_id: str) -> dict[str, Any] | None:
        out = await self._call("get_sampler_index", sampler_id=str(sampler_id))
        return dict(out) if isinstance(out, dict) else None

    def list_sampler_index(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_sampler_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampler_index returned non-list: {type(out)}")
        return out

    async def async_list_sampler_index(self) -> list[dict[str, Any]]:
        out = await self._call("list_sampler_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampler_index returned non-list: {type(out)}")
        return out

    def update_session_heartbeat(self, *, session_id: str, now: float | None = None) -> None:
        self._call_sync("update_session_heartbeat", session_id=str(session_id), now=now)

    async def async_update_session_heartbeat(self, *, session_id: str, now: float | None = None) -> None:
        await self._call("update_session_heartbeat", session_id=str(session_id), now=now)

    def get_session_heartbeat(self, *, session_id: str) -> float | None:
        out = self._call_sync("get_session_heartbeat", session_id=str(session_id))
        return None if out is None else float(out)

    def delete_session_heartbeat(self, *, session_id: str) -> bool:
        return bool(self._call_sync("delete_session_heartbeat", session_id=str(session_id)))

    def session_heartbeat_size(self) -> int:
        return int(self._call_sync("session_heartbeat_size"))

    async def async_session_heartbeat_size(self, *, create_if_missing: bool = True) -> int:
        if create_if_missing:
            return int(await self._call("session_heartbeat_size"))
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=False)
        out = await async_get_ray_ref(actor.session_heartbeat_size.remote())
        return int(out)

    def is_session_heartbeat_stale(self, *, session_id: str, ttl_s: float) -> bool:
        return bool(self._call_sync("is_session_heartbeat_stale", session_id=str(session_id), ttl_s=float(ttl_s)))

    async def async_is_session_heartbeat_stale(self, *, session_id: str, ttl_s: float) -> bool:
        return bool(await self._call("is_session_heartbeat_stale", session_id=str(session_id), ttl_s=float(ttl_s)))

    def prune_session_heartbeats(self, *, max_age_s: float) -> int:
        return int(self._call_sync("prune_session_heartbeats", max_age_s=float(max_age_s)))

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


class TaskFutureService:
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
        status = str(record.get("status"))
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if status == "retrieved" and str(metadata.get("terminal_status") or "done") != "done":
            raise KeyError(f"Future already retrieved without result: {request_id}")
        if status not in {"done", "retrieved"}:
            raise KeyError(f"Future is not done: {request_id}")
        result_path = record.get("result_path")
        if not isinstance(result_path, str) or not result_path:
            raise KeyError(f"Future result payload missing: {request_id}")
        payload = await asyncio.to_thread(
            self._payloads.read_json_payload,
            path=result_path,
            expected_checksum=record.get("result_checksum"),
        )
        if status != "retrieved":
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

    async def async_reap(self) -> dict[str, Any]:
        from ..config import config as cfg

        now = time.time()
        batch_size = max(1, int(os.environ.get("MINT_TASK_FUTURE_REAPER_BATCH_SIZE", "1000")))

        expired = await self._task_state.async_expire_active_tasks(
            older_than_s=float(getattr(cfg, "task_pending_ttl_s", 86400.0)),
            now=now,
            limit=batch_size,
        )

        evicted: list[str] = []
        payload_evict_errors: list[dict[str, str]] = []
        payload_candidates = await self._task_state.async_list_terminal_payloads_for_eviction(
            older_than_s=float(getattr(cfg, "task_result_ttl_s", 86400.0)),
            now=now,
            limit=batch_size,
        )
        for record in payload_candidates:
            request_id = str(record.get("request_id") or "")
            result_path = record.get("result_path")
            if not request_id or not isinstance(result_path, str) or not result_path:
                continue
            try:
                await self._payloads.async_delete_json_payload(path=result_path)
            except Exception as e:
                await self._task_state.async_record_payload_evict_error(count=1)
                payload_evict_errors.append(
                    {
                        "request_id": request_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            marked = await self._task_state.async_mark_payload_evicted(
                request_id=request_id,
                expected_result_path=result_path,
                now=now,
            )
            if bool(marked.get("ok")):
                evicted.append(request_id)

        deleted_tombstones = await self._task_state.async_delete_expired_tombstones(
            older_than_s=float(getattr(cfg, "task_tombstone_ttl_s", 604800.0)),
            now=now,
            limit=batch_size,
        )

        return {
            "expired": expired,
            "timed_out": [],
            "payload_evicted": evicted,
            "tombstones_deleted": deleted_tombstones,
            "payload_evict_errors": payload_evict_errors,
            "metrics": task_future_reaper_metrics_snapshot(),
        }

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

    async def async_ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = True,
    ) -> dict[str, Any]:
        actor = await self._task_state._get_ray_actor_async(
            require_ready=False,
            create_if_missing=create_if_missing,
        )
        out = await async_get_ray_ref(actor.stats.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.stats returned non-dict: {type(out)}")
        return out

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        return await self._task_state.async_ping(timeout_s=timeout_s)

    def metrics_snapshot(self) -> dict[str, Any]:
        return {"backend": "task_state_store", "task_future_reaper": task_future_reaper_metrics_snapshot()}

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
task_futures = TaskFutureService()
