from __future__ import annotations

import json
import sqlite3
from typing import Any


def _json_loads(value: str | bytes | None) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class TaskQueries:
    """Read-only task-store queries.

    The facade owns the SQLite connection and lock. This collaborator only
    executes SELECT statements on a caller-provided connection.
    """

    def __init__(self, *, not_found_error: type[Exception], conflict_error: type[Exception]) -> None:
        self._not_found_error = not_found_error
        self._conflict_error = conflict_error

    def get_row(self, conn: sqlite3.Connection, request_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM tasks WHERE request_id = ?", (str(request_id),)).fetchone()
        if row is None:
            raise self._not_found_error(str(request_id))
        return row

    def raise_transition_error(self, conn: sqlite3.Connection, request_id: str, action: str) -> None:
        row = conn.execute(
            "SELECT status FROM tasks WHERE request_id = ?",
            (str(request_id),),
        ).fetchone()
        if row is None:
            raise self._not_found_error(str(request_id))
        raise self._conflict_error(f"cannot {action}; current status={row['status']!r}")

    def list_active_tasks(self, conn: sqlite3.Connection, *, limit: int | None = None) -> list[dict[str, Any]]:
        return [self.row_to_record(row) for row in self.list_active_task_rows(conn, limit=limit)]

    def list_active_task_rows(self, conn: sqlite3.Connection, *, limit: int | None = None) -> list[sqlite3.Row]:
        sql = """
            SELECT * FROM tasks
            WHERE status IN ('pending', 'assigned', 'leased', 'finalizing')
            ORDER BY created_at, request_id
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (max(0, int(limit)),)
        return conn.execute(sql, params).fetchall()

    def list_expired_leases(
        self,
        conn: sqlite3.Connection,
        *,
        now: float,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return [self.row_to_record(row) for row in self.list_expired_lease_rows(conn, now=now, limit=limit)]

    def list_expired_lease_rows(
        self,
        conn: sqlite3.Connection,
        *,
        now: float,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        sql = """
            SELECT * FROM tasks
            WHERE status IN ('leased', 'finalizing')
              AND COALESCE(finalizing_until, lease_expires_at, 0) <= ?
            ORDER BY COALESCE(finalizing_until, lease_expires_at, 0), created_at
        """
        params: tuple[Any, ...] = (float(now),)
        if limit is not None:
            sql += " LIMIT ?"
            params = (float(now), max(0, int(limit)))
        return conn.execute(sql, params).fetchall()

    def list_tasks_by_metadata(
        self,
        conn: sqlite3.Connection,
        *,
        filters: dict[str, Any] | None = None,
        statuses: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        rows = self.list_tasks_by_metadata_rows(conn, statuses=statuses)
        return self.filter_tasks_by_metadata_rows(rows, filters=filters, limit=limit)

    def list_tasks_by_metadata_rows(
        self,
        conn: sqlite3.Connection,
        *,
        statuses: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        status_values = list(statuses or [])
        params: list[Any] = []
        sql = "SELECT * FROM tasks"
        if status_values:
            placeholders = ", ".join("?" for _ in status_values)
            sql += f" WHERE status IN ({placeholders})"
            params.extend(status_values)
        sql += " ORDER BY created_at, request_id"
        return conn.execute(sql, tuple(params)).fetchall()

    def filter_tasks_by_metadata_rows(
        self,
        rows: list[sqlite3.Row],
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        normalized_filters = dict(filters or {})
        out: list[dict[str, Any]] = []
        for row in rows:
            record = self.row_to_record(row)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            assert metadata is not None
            if all(metadata.get(key) == value for key, value in normalized_filters.items()):
                out.append(record)
                if len(out) >= int(limit):
                    break
        return out

    def list_terminal_payloads_for_eviction(
        self,
        conn: sqlite3.Connection,
        *,
        older_than_s: float,
        now: float,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        rows = self.list_terminal_payload_rows_for_eviction(conn, limit=limit)
        return self.filter_terminal_payload_rows_for_eviction(
            rows,
            older_than_s=older_than_s,
            now=now,
        )

    def list_terminal_payload_rows_for_eviction(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int = 1000,
    ) -> list[sqlite3.Row]:
        batch_limit = max(0, int(limit))
        if batch_limit <= 0:
            return []
        return conn.execute(
            """
            SELECT * FROM tasks
            WHERE status IN ('done', 'failed', 'cancelled', 'expired', 'retrieved')
              AND result_path IS NOT NULL
              AND result_path != ''
            ORDER BY updated_at, request_id
            LIMIT ?
            """,
            (batch_limit,),
        ).fetchall()

    def filter_terminal_payload_rows_for_eviction(
        self,
        rows: list[sqlite3.Row],
        *,
        older_than_s: float,
        now: float,
    ) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = float(now) - ttl_s
        out: list[dict[str, Any]] = []
        for row in rows:
            record = self.row_to_record(row)
            if self.terminal_completed_at(record) <= cutoff:
                out.append(record)
        return out

    def list_staged_payloads_for_gc(
        self,
        conn: sqlite3.Connection,
        *,
        older_than_s: float,
        now: float,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        rows = self.list_staged_payload_rows_for_gc(conn, limit=limit)
        return self.filter_staged_payload_rows_for_gc(
            rows,
            older_than_s=older_than_s,
            now=now,
            limit=limit,
        )

    def list_staged_payload_rows_for_gc(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int = 1000,
    ) -> list[sqlite3.Row]:
        batch_limit = max(0, int(limit))
        if batch_limit <= 0:
            return []
        return conn.execute(
            """
            SELECT * FROM tasks
            WHERE staged_payload_path IS NOT NULL
               OR metadata_json LIKE '%abandoned_staged_payload_paths%'
            ORDER BY updated_at, request_id
            LIMIT ?
            """,
            (batch_limit,),
        ).fetchall()

    def filter_staged_payload_rows_for_gc(
        self,
        rows: list[sqlite3.Row],
        *,
        older_than_s: float,
        now: float,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = float(now) - ttl_s
        batch_limit = max(0, int(limit))
        if batch_limit <= 0:
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            record = self.row_to_record(row)
            request_id = str(record["request_id"])
            active_path = record.get("staged_payload_path")
            status = str(record.get("status") or "")
            if isinstance(active_path, str) and active_path:
                if status == "finalizing":
                    finalizing_until = record.get("finalizing_until")
                    expired_at = float(finalizing_until or record.get("updated_at") or 0.0)
                    eligible = expired_at <= cutoff
                else:
                    eligible = float(record.get("updated_at") or 0.0) <= cutoff
                if eligible:
                    out.append(
                        {
                            "request_id": request_id,
                            "path": active_path,
                            "kind": "active",
                            "status": status,
                        }
                    )
                    if len(out) >= batch_limit:
                        return out

            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            assert metadata is not None
            abandoned = metadata.get("abandoned_staged_payload_paths")
            if not isinstance(abandoned, list):
                continue
            if float(record.get("updated_at") or 0.0) > cutoff:
                continue
            for path in abandoned:
                if not isinstance(path, str) or not path:
                    continue
                out.append(
                    {
                        "request_id": request_id,
                        "path": path,
                        "kind": "abandoned",
                        "status": status,
                    }
                )
                if len(out) >= batch_limit:
                    return out
        return out

    def future_metrics_stats(
        self,
        conn: sqlite3.Connection,
        *,
        now: float,
        active_statuses: tuple[str, ...],
        terminal_statuses: tuple[str, ...],
        execution_timeout_s: float,
        queue_timeout_s: float,
        result_ttl_s: float,
        tombstone_ttl_s: float,
        timeout_counts: dict[str, Any],
    ) -> dict[str, Any]:
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
        ).fetchall()
        op_rows = conn.execute(
            "SELECT op, status, COUNT(*) AS count FROM tasks GROUP BY op, status"
        ).fetchall()
        scalar_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN result_path IS NOT NULL AND result_path != '' THEN 1 ELSE 0 END) AS refs,
                SUM(CASE WHEN metadata_json IS NOT NULL AND metadata_json != '{}' THEN 1 ELSE 0 END) AS meta
            FROM tasks
            """
        ).fetchone()
        pending_age_row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS count,
                MAX(? - created_at) AS oldest_s,
                AVG(? - created_at) AS avg_s
            FROM tasks
            WHERE status IN ({",".join("?" for _ in active_statuses)})
            """,
            (now, now, *active_statuses),
        ).fetchone()
        done_age_row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS count,
                MAX(? - updated_at) AS oldest_s,
                AVG(? - updated_at) AS avg_s
            FROM tasks
            WHERE status IN ({",".join("?" for _ in terminal_statuses)})
            """,
            (now, now, *terminal_statuses),
        ).fetchone()

        by_status = {str(row["status"]): int(row["count"] or 0) for row in status_rows}
        pending_statuses = set(active_statuses)
        result_statuses = {"done", "retrieved"}
        error_statuses = {"failed"}

        by_op: dict[str, dict[str, int]] = {}
        for row in op_rows:
            op = str(row["op"] or "unknown")
            status = str(row["status"] or "unknown")
            count = int(row["count"] or 0)
            bucket = by_op.setdefault(op, {"pending": 0, "results": 0, "errors": 0})
            if status in pending_statuses:
                bucket["pending"] += count
            elif status in result_statuses:
                bucket["results"] += count
            elif status in error_statuses:
                bucket["errors"] += count

        pending = sum(by_status.get(status, 0) for status in pending_statuses)
        results = sum(by_status.get(status, 0) for status in result_statuses)
        errors = sum(by_status.get(status, 0) for status in error_statuses)
        refs = int(scalar_row["refs"] or 0) if scalar_row is not None else 0
        meta = int(scalar_row["meta"] or 0) if scalar_row is not None else 0
        oldest_pending_s = float(pending_age_row["oldest_s"] or 0.0) if pending_age_row is not None else 0.0
        oldest_done_s = float(done_age_row["oldest_s"] or 0.0) if done_age_row is not None else 0.0
        avg_pending_s = float(pending_age_row["avg_s"] or 0.0) if pending_age_row is not None else 0.0
        avg_done_s = float(done_age_row["avg_s"] or 0.0) if done_age_row is not None else 0.0

        return {
            "pending": int(pending),
            "results": int(results),
            "errors": int(errors),
            "refs": refs,
            "meta": meta,
            "expired": int(by_status.get("expired", 0)),
            "retrieved": int(by_status.get("retrieved", 0)),
            "execution_timeout_s": float(execution_timeout_s),
            "queue_timeout_s": float(queue_timeout_s),
            "result_ttl_s": float(result_ttl_s),
            "tombstone_ttl_s": float(tombstone_ttl_s),
            "by_op": by_op,
            "age_stats": {
                "oldest_pending_s": oldest_pending_s,
                "oldest_done_s": oldest_done_s,
                "avg_pending_s": avg_pending_s,
                "avg_done_s": avg_done_s,
            },
            "payload_stats": {
                "result_refs_count": refs,
                "errors_count": int(errors),
                "refs_count": refs,
            },
            "timeout_counts": timeout_counts,
        }

    @staticmethod
    def terminal_completed_at(record: dict[str, Any]) -> float:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key in ("done_at", "failed_at"):
            assert metadata is not None
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            try:
                if value is not None:
                    return float(value)
            except Exception:
                pass
        return float(record.get("updated_at") or 0.0)

    @staticmethod
    def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
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
            "staged_payload_path": row["staged_payload_path"],
            "staged_payload_checksum": row["staged_payload_checksum"],
            "staged_payload_size_bytes": row["staged_payload_size_bytes"],
            "error": row["error"],
            "metadata": _json_loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "assigned_at": row["assigned_at"],
            "leased_at": row["leased_at"],
            "lease_expires_at": row["lease_expires_at"],
            "finalizing_until": row["finalizing_until"],
        }
