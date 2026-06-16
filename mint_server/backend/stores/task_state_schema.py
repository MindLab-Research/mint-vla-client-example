from __future__ import annotations

import sqlite3
from pathlib import Path


class TaskStateSchema:
    def __init__(self, *, db_path: str) -> None:
        self._db_path = str(db_path)

    def configure_connection(self, conn: sqlite3.Connection) -> None:
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        if self._db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

    def apply_schema(self, conn: sqlite3.Connection) -> None:
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
                staged_payload_path TEXT,
                staged_payload_checksum TEXT,
                staged_payload_size_bytes INTEGER,
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

            CREATE TABLE IF NOT EXISTS billing_outbox (
                outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_json TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                claim_id TEXT,
                claimed_until REAL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
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

            CREATE INDEX IF NOT EXISTS idx_billing_outbox_status_claimed
                ON billing_outbox(status, claimed_until, created_at);
            """
        )
        self.ensure_tasks_columns(conn)
        self.ensure_billing_outbox_columns(conn)

    @staticmethod
    def ensure_tasks_columns(conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        required = {
            "staged_payload_path": "TEXT",
            "staged_payload_checksum": "TEXT",
            "staged_payload_size_bytes": "INTEGER",
        }
        for name, type_sql in required.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {type_sql}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_staged_payload_path ON tasks(staged_payload_path)"
        )

    @staticmethod
    def ensure_billing_outbox_columns(conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(billing_outbox)").fetchall()
        }
        required = {
            "event_hash": "TEXT",
            "claim_id": "TEXT",
            "claimed_until": "REAL",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "last_error": "TEXT",
        }
        for name, type_sql in required.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE billing_outbox ADD COLUMN {name} {type_sql}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_billing_outbox_status_claimed ON billing_outbox(status, claimed_until, created_at)"
        )
