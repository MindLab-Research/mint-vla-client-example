"""Async usage storage backend for billing usage_event."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import sqlite3
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from .config import config

logger = logging.getLogger(__name__)

ChargeItem = Literal["sampling", "inference", "training", "checkpoint_storage"]
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OUTBOX_BATCH_SIZE = 128


def _import_asyncpg():
    import asyncpg

    return asyncpg


def _default_jsonl_usage_path() -> Path:
    usage_log_dir = str(config.usage_log_dir or "").strip()
    if usage_log_dir:
        return Path(usage_log_dir) / "usage_event.jsonl"
    return Path(config.checkpoint_dir) / ".billing" / "usage_event.jsonl"


def _usage_pg_dsn() -> str:
    return str(config.usage_pg_dsn or "").strip()


@dataclass(frozen=True)
class UsageEvent:
    account_id: str
    apikey_id: str
    charge_item: ChargeItem
    quantity: int
    request_id: str
    label: str = ""
    event_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UsageStore(Protocol):
    async def write_event(self, event: UsageEvent) -> None: ...

    async def write_events(self, events: list[UsageEvent]) -> None: ...

    async def flush_outbox(self, limit: int = _OUTBOX_BATCH_SIZE) -> int: ...

    async def query_logs(
        self,
        since: datetime | None = None,
        account_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int, bool]: ...

    async def get_account_summary(self, account_id: str) -> dict: ...

    async def health_check(self) -> bool: ...

    async def close(self) -> None: ...


@asynccontextmanager
async def _null_async_context():
    yield


class PostgresUsageStore:
    def __init__(
        self,
        *,
        dsn: str,
        pool_min: int = 1,
        pool_max: int = 10,
        write_timeout_ms: int = 2000,
        table: str = "billing.usage_event",
        outbox_path: str | None = None,
        outbox_flush_interval_s: float = 5.0,
    ):
        if not dsn:
            raise ValueError("Postgres DSN is required for usage backend")
        self._dsn = dsn
        self._pool_min = max(1, int(pool_min))
        self._pool_max = max(self._pool_min, int(pool_max))
        self._write_timeout_s = max(0.1, float(write_timeout_ms) / 1000.0)
        self._pool = None
        self._pool_lock = asyncio.Lock()
        self._table = str(table or "billing.usage_event").strip()
        self._sequence = self._build_sequence_name(self._table)
        self._dedupe_index = self._build_dedupe_index_name(self._table)
        default_outbox = Path(config.checkpoint_dir) / ".billing" / "usage_outbox.sqlite3"
        self._outbox_path = Path(outbox_path) if outbox_path else default_outbox
        self._outbox_lock = asyncio.Lock()
        self._outbox_ready = False
        self._flush_interval_s = max(0.5, float(outbox_flush_interval_s))
        self._close_event = asyncio.Event()
        self._flush_task: asyncio.Task[None] | None = None

    @staticmethod
    def _parse_qualified_name(value: str) -> tuple[str, str]:
        parts = [part.strip() for part in value.split(".", 1)]
        if len(parts) == 1:
            schema, name = "public", parts[0]
        else:
            schema, name = parts
        if not _SQL_IDENT_RE.match(schema) or not _SQL_IDENT_RE.match(name):
            raise ValueError(f"Unsupported SQL identifier: {value!r}")
        return schema, name

    @classmethod
    def _build_sequence_name(cls, table: str) -> str:
        schema, name = cls._parse_qualified_name(table)
        return f"{schema}.{name}_source_index_seq"

    @classmethod
    def _build_dedupe_index_name(cls, table: str) -> str:
        _, name = cls._parse_qualified_name(table)
        return f"idx_{name}_request_charge_label_uniq"

    @staticmethod
    def _normalize_event_time(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    @staticmethod
    def _to_iso8601(ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _event_identity(event: UsageEvent) -> tuple[str, str, str]:
        return (str(event.request_id), str(event.charge_item), str(event.label or ""))

    def _validate_events(self, events: list[UsageEvent]) -> list[UsageEvent]:
        normalized = list(events)
        if not normalized:
            raise ValueError("write_events requires at least one event")
        seen: set[tuple[str, str, str]] = set()
        for event in normalized:
            request_id = str(event.request_id or "").strip()
            if not request_id:
                raise ValueError("usage_event request_id must be non-empty")
            key = self._event_identity(event)
            if key in seen:
                raise ValueError(f"duplicate usage_event identity in batch: {key!r}")
            seen.add(key)
        return normalized

    async def _ensure_pool(self, *, start_flush_task: bool = True):
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is not None:
                return self._pool
            try:
                asyncpg = _import_asyncpg()
            except Exception as e:
                raise RuntimeError("asyncpg is required for postgres usage backend") from e

            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                command_timeout=max(5.0, self._write_timeout_s),
                server_settings={"application_name": "tinker_server_usage"},
            )
            await self._ensure_pg_schema()
            await self._ensure_outbox()
            if start_flush_task:
                self._ensure_flush_task()
            return self._pool

    async def _ensure_pg_schema(self) -> None:
        pool = self._pool
        if pool is None:
            raise RuntimeError("usage pool is not initialized")
        async with pool.acquire() as conn:
            await conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {self._sequence} AS BIGINT")
            await conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {self._dedupe_index} "
                f"ON {self._table} (request_id, charge_item, label)"
            )

    async def _ensure_outbox(self) -> None:
        if self._outbox_ready:
            return
        async with self._outbox_lock:
            if self._outbox_ready:
                return
            await asyncio.to_thread(self._init_outbox_db)
            self._outbox_ready = True

    def _init_outbox_db(self) -> None:
        self._outbox_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._outbox_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_usage_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    charge_item TEXT NOT NULL,
                    label TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    apikey_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    event_time TEXT NOT NULL,
                    UNIQUE(request_id, charge_item, label)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_usage_event_id ON pending_usage_event(id)"
            )
            conn.commit()
        finally:
            conn.close()

    def _ensure_flush_task(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_outbox_loop())

    def _outbox_upsert(self, events: list[UsageEvent]) -> int:
        conn = sqlite3.connect(self._outbox_path)
        try:
            conn.executemany(
                """
                INSERT INTO pending_usage_event
                    (request_id, charge_item, label, account_id, apikey_id, quantity, event_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id, charge_item, label) DO NOTHING
                """,
                [
                    (
                        event.request_id,
                        event.charge_item,
                        event.label or "",
                        event.account_id,
                        event.apikey_id,
                        int(event.quantity),
                        self._to_iso8601(self._normalize_event_time(event.event_time)),
                    )
                    for event in events
                ],
            )
            conn.commit()
            return int(conn.total_changes)
        finally:
            conn.close()

    def _outbox_fetch_batch(self, limit: int) -> list[dict]:
        conn = sqlite3.connect(self._outbox_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT id, request_id, charge_item, label, account_id, apikey_id, quantity, event_time
                FROM pending_usage_event
                ORDER BY id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _outbox_delete_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        conn = sqlite3.connect(self._outbox_path)
        try:
            conn.executemany("DELETE FROM pending_usage_event WHERE id = ?", [(int(i),) for i in ids])
            conn.commit()
        finally:
            conn.close()

    async def _enqueue_outbox(self, events: list[UsageEvent], error: Exception) -> None:
        await self._ensure_outbox()
        changed = await asyncio.to_thread(self._outbox_upsert, events)
        logger.warning(
            "usage_event write diverted to outbox: request_ids=%s enqueued=%s err=%s",
            [event.request_id for event in events],
            changed,
            error,
        )
        self._ensure_flush_task()

    @asynccontextmanager
    async def _maybe_transaction(self, conn):
        tx_factory = getattr(conn, "transaction", None)
        if callable(tx_factory):
            async with tx_factory():
                yield
            return
        async with _null_async_context():
            yield

    async def _write_events_to_pg(self, events: list[UsageEvent]) -> None:
        pool = await self._ensure_pool()
        sql = f"""
        INSERT INTO {self._table}
            (source_index, event_time, account_id, apikey_id, charge_item, quantity, request_id, label)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (request_id, charge_item, label) DO NOTHING
        """

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with pool.acquire() as conn:
                    async with self._maybe_transaction(conn):
                        for event in events:
                            source_index = int(await conn.fetchval(f"SELECT nextval('{self._sequence}')"))
                            await asyncio.wait_for(
                                conn.execute(
                                    sql,
                                    source_index,
                                    self._normalize_event_time(event.event_time),
                                    event.account_id,
                                    event.apikey_id,
                                    event.charge_item,
                                    max(0, int(event.quantity)),
                                    event.request_id,
                                    event.label or "",
                                ),
                                timeout=self._write_timeout_s,
                            )
                return
            except Exception as e:
                last_error = e
                if attempt < 2:
                    backoff_s = float(2**attempt)
                    logger.warning(
                        "usage_event pg write failed, retrying: attempt=%s request_ids=%s err=%s",
                        attempt + 1,
                        [event.request_id for event in events],
                        e,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                break

        if last_error is not None:
            raise last_error
        raise RuntimeError("usage_event write failed with unknown error")

    async def write_event(self, event: UsageEvent) -> None:
        await self.write_events([event])

    async def write_events(self, events: list[UsageEvent]) -> None:
        normalized = self._validate_events(events)
        try:
            await self._write_events_to_pg(normalized)
        except Exception as pg_error:
            try:
                await self._enqueue_outbox(normalized, pg_error)
            except Exception as outbox_error:
                logger.error(
                    "usage_event persistence failed: request_ids=%s pg_err=%s outbox_err=%s",
                    [event.request_id for event in normalized],
                    pg_error,
                    outbox_error,
                )
                raise RuntimeError("usage_event persistence failed") from outbox_error

    async def flush_outbox(self, limit: int = _OUTBOX_BATCH_SIZE) -> int:
        await self._ensure_outbox()
        rows = await asyncio.to_thread(self._outbox_fetch_batch, max(1, int(limit)))
        if not rows:
            return 0
        events = [
            UsageEvent(
                account_id=str(row["account_id"]),
                apikey_id=str(row["apikey_id"]),
                charge_item=str(row["charge_item"]),
                quantity=int(row["quantity"]),
                request_id=str(row["request_id"]),
                label=str(row["label"] or ""),
                event_time=datetime.fromisoformat(str(row["event_time"]).replace("Z", "+00:00")),
            )
            for row in rows
        ]
        await self._write_events_to_pg(events)
        await asyncio.to_thread(self._outbox_delete_ids, [int(row["id"]) for row in rows])
        return len(rows)

    async def _flush_outbox_loop(self) -> None:
        while not self._close_event.is_set():
            wait_s = self._flush_interval_s
            try:
                flushed = await self.flush_outbox(limit=_OUTBOX_BATCH_SIZE)
                if flushed > 0:
                    wait_s = 0.2
            except Exception as e:
                logger.warning("usage outbox flush failed: %s", e)
            try:
                await asyncio.wait_for(self._close_event.wait(), timeout=wait_s)
            except asyncio.TimeoutError:
                continue

    def _build_where(
        self,
        *,
        since: datetime | None,
        account_id: str | None,
        args: list,
    ) -> str:
        clauses: list[str] = []
        if account_id is not None:
            clauses.append(f"account_id = ${len(args) + 1}")
            args.append(account_id)
        if since is not None:
            clauses.append(f"event_time >= ${len(args) + 1}")
            args.append(self._normalize_event_time(since))
        if not clauses:
            return ""
        return " WHERE " + " AND ".join(clauses)

    async def query_logs(
        self,
        since: datetime | None = None,
        account_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int, bool]:
        pool = await self._ensure_pool()
        args: list = []
        where_sql = self._build_where(since=since, account_id=account_id, args=args)

        count_sql = f"SELECT COUNT(*) FROM {self._table}{where_sql}"
        query_sql = (
            "SELECT source_index, event_time, account_id, apikey_id, charge_item, quantity, request_id, label "
            f"FROM {self._table}{where_sql} "
            f"ORDER BY event_time DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
        )

        async with pool.acquire() as conn:
            total_count = int(await conn.fetchval(count_sql, *args))
            rows = await conn.fetch(query_sql, *args, int(limit), int(offset))

        logs = [
            {
                "source_index": int(row["source_index"]),
                "event_time": self._to_iso8601(row["event_time"]),
                "account_id": row["account_id"],
                "apikey_id": row["apikey_id"],
                "charge_item": row["charge_item"],
                "quantity": int(row["quantity"]),
                "request_id": row["request_id"],
                "label": row["label"] or "",
            }
            for row in rows
        ]
        has_more = offset + limit < total_count
        return logs, total_count, has_more

    async def get_account_summary(self, account_id: str) -> dict:
        pool = await self._ensure_pool()
        total_sql = f"SELECT COALESCE(SUM(quantity), 0) FROM {self._table} WHERE account_id = $1"
        grouped_sql = (
            f"SELECT charge_item, COALESCE(SUM(quantity), 0) AS total_quantity "
            f"FROM {self._table} WHERE account_id = $1 GROUP BY charge_item"
        )
        async with pool.acquire() as conn:
            total_quantity = int(await conn.fetchval(total_sql, account_id))
            rows = await conn.fetch(grouped_sql, account_id)
        charge_item_totals = {str(row["charge_item"]): int(row["total_quantity"]) for row in rows}
        return {"total_quantity": total_quantity, "charge_item_totals": charge_item_totals}

    async def health_check(self) -> bool:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        return int(val) == 1

    async def close(self) -> None:
        self._close_event.set()
        if self._flush_task is not None:
            self._flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None
        if self._outbox_ready:
            if self._pool is None:
                try:
                    await self._ensure_pool(start_flush_task=False)
                except Exception as e:
                    logger.warning("usage outbox final drain could not initialize pool during shutdown: %s", e)
            if self._pool is not None:
                try:
                    while await self.flush_outbox(limit=_OUTBOX_BATCH_SIZE):
                        pass
                except Exception as e:
                    logger.warning("usage outbox final drain failed during shutdown: %s", e)
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


class JsonlUsageStore:
    def __init__(self, *, path: str | Path):
        self._path = Path(path)
        self._lock = asyncio.Lock()
        self._loaded = False
        self._records: list[dict] = []
        self._known_identities: set[tuple[str, str, str]] = set()
        self._next_source_index = 1
        self._stat_size = 0
        self._stat_mtime_ns = 0

    @staticmethod
    def _normalize_event_time(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    @staticmethod
    def _to_iso8601(ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _event_identity(event: UsageEvent) -> tuple[str, str, str]:
        return (str(event.request_id), str(event.charge_item), str(event.label or ""))

    def _validate_events(self, events: list[UsageEvent]) -> list[UsageEvent]:
        normalized = list(events)
        if not normalized:
            raise ValueError("write_events requires at least one event")
        seen: set[tuple[str, str, str]] = set()
        for event in normalized:
            request_id = str(event.request_id or "").strip()
            if not request_id:
                raise ValueError("usage_event request_id must be non-empty")
            key = self._event_identity(event)
            if key in seen:
                raise ValueError(f"duplicate usage_event identity in batch: {key!r}")
            seen.add(key)
        return normalized

    def _coerce_record(self, payload: dict, *, source_name: str, line_no: int) -> dict:
        try:
            source_index = int(payload["source_index"])
            event_time = self._to_iso8601(
                datetime.fromisoformat(str(payload["event_time"]).replace("Z", "+00:00"))
            )
            record = {
                "source_index": source_index,
                "event_time": event_time,
                "account_id": str(payload["account_id"]),
                "apikey_id": str(payload["apikey_id"]),
                "charge_item": str(payload["charge_item"]),
                "quantity": int(payload["quantity"]),
                "request_id": str(payload["request_id"]),
                "label": str(payload.get("label") or ""),
            }
        except Exception as e:
            raise ValueError(f"invalid usage_event record at {source_name}:{line_no}") from e
        return record

    def _load_from_stream_locked(self, stream) -> None:
        records: list[dict] = []
        known_identities: set[tuple[str, str, str]] = set()
        next_source_index = 1
        stream.seek(0)
        for line_no, raw_line in enumerate(stream, start=1):
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                record = self._coerce_record(payload, source_name=str(self._path), line_no=line_no)
            except Exception as e:
                logger.warning("skipping malformed usage_event JSONL row: %s", e)
                continue
            identity = (
                record["request_id"],
                record["charge_item"],
                record["label"],
            )
            if identity in known_identities:
                logger.warning(
                    "skipping duplicate usage_event JSONL row: request_id=%s charge_item=%s label=%s",
                    record["request_id"],
                    record["charge_item"],
                    record["label"],
                )
                continue
            records.append(record)
            known_identities.add(identity)
            next_source_index = max(next_source_index, int(record["source_index"]) + 1)
        stat = os.fstat(stream.fileno())
        self._records = records
        self._known_identities = known_identities
        self._next_source_index = next_source_index
        self._stat_size = int(stat.st_size)
        self._stat_mtime_ns = int(stat.st_mtime_ns)
        self._loaded = True

    def _reload_from_disk_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                self._load_from_stream_locked(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _disk_state_changed_locked(self) -> bool:
        if not self._loaded:
            return True
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return self._stat_size != 0 or self._stat_mtime_ns != 0
        return int(stat.st_size) != self._stat_size or int(stat.st_mtime_ns) != self._stat_mtime_ns

    async def _ensure_loaded(self) -> None:
        async with self._lock:
            if self._disk_state_changed_locked():
                self._reload_from_disk_locked()

    def _append_records_to_stream_locked(self, stream, records: list[dict]) -> None:
        payload = "".join(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n" for record in records
        ).encode("utf-8")
        stream.seek(0, os.SEEK_END)
        start_offset = stream.tell()
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            with suppress(Exception):
                stream.seek(start_offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
            raise

    @staticmethod
    def _record_matches_since(record: dict, since: datetime | None) -> bool:
        if since is None:
            return True
        record_dt = datetime.fromisoformat(str(record["event_time"]).replace("Z", "+00:00"))
        if record_dt.tzinfo is None:
            record_dt = record_dt.replace(tzinfo=timezone.utc)
        return record_dt >= (
            since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        )

    async def write_event(self, event: UsageEvent) -> None:
        await self.write_events([event])

    async def write_events(self, events: list[UsageEvent]) -> None:
        normalized = self._validate_events(events)
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a+b") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    self._load_from_stream_locked(f)
                    fresh_records: list[dict] = []
                    next_source_index = self._next_source_index
                    for event in normalized:
                        identity = self._event_identity(event)
                        if identity in self._known_identities:
                            continue
                        record = {
                            "source_index": next_source_index,
                            "event_time": self._to_iso8601(self._normalize_event_time(event.event_time)),
                            "account_id": str(event.account_id),
                            "apikey_id": str(event.apikey_id),
                            "charge_item": str(event.charge_item),
                            "quantity": max(0, int(event.quantity)),
                            "request_id": str(event.request_id),
                            "label": str(event.label or ""),
                        }
                        next_source_index += 1
                        fresh_records.append(record)
                    if fresh_records:
                        self._append_records_to_stream_locked(f, fresh_records)
                        self._load_from_stream_locked(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def flush_outbox(self, limit: int = _OUTBOX_BATCH_SIZE) -> int:
        return 0

    async def query_logs(
        self,
        since: datetime | None = None,
        account_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int, bool]:
        await self._ensure_loaded()
        async with self._lock:
            filtered = [
                dict(record)
                for record in self._records
                if (account_id is None or record["account_id"] == account_id)
                and self._record_matches_since(record, since)
            ]
        filtered.sort(key=lambda record: (record["event_time"], int(record["source_index"])), reverse=True)
        total_count = len(filtered)
        page = filtered[int(offset) : int(offset) + int(limit)]
        has_more = int(offset) + int(limit) < total_count
        return page, total_count, has_more

    async def get_account_summary(self, account_id: str) -> dict:
        await self._ensure_loaded()
        async with self._lock:
            records = [record for record in self._records if record["account_id"] == account_id]
        charge_item_totals: dict[str, int] = {}
        total_quantity = 0
        for record in records:
            quantity = int(record["quantity"])
            total_quantity += quantity
            charge_item = str(record["charge_item"])
            charge_item_totals[charge_item] = charge_item_totals.get(charge_item, 0) + quantity
        return {"total_quantity": total_quantity, "charge_item_totals": charge_item_totals}

    async def health_check(self) -> bool:
        try:
            await self._ensure_loaded()
            async with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.touch(exist_ok=True)
            return True
        except Exception as e:
            logger.warning("usage_event JSONL health check failed: %s", e)
            return False

    async def close(self) -> None:
        return None


_usage_store: UsageStore | None = None
_usage_store_guard = asyncio.Lock()


def _build_usage_store() -> UsageStore:
    path = _default_jsonl_usage_path()
    backend = str(config.usage_backend or "postgres").strip().lower()
    dsn = _usage_pg_dsn()
    default_tmp_path = Path("/tmp/tinker_usage/usage_event.jsonl")
    if backend and backend != "postgres":
        raise ValueError(f"Unsupported usage backend {backend!r}; only 'postgres' is accepted")
    if dsn:
        logger.warning(
            "usage PG config is deprecated and ignored; JSONL usage_event store remains active at %s",
            path,
        )
    else:
        logger.info("usage JSONL store active at %s", path)
    if path == default_tmp_path and dsn:
        logger.warning(
            "usage JSONL store is using default tmp path %s; set TINKER_USAGE_LOG_DIR explicitly for a shared pull target",
            path,
        )
    return JsonlUsageStore(path=path)


async def get_usage_store() -> UsageStore:
    global _usage_store
    if _usage_store is not None:
        return _usage_store
    async with _usage_store_guard:
        if _usage_store is None:
            _usage_store = _build_usage_store()
    return _usage_store


async def close_usage_store() -> None:
    global _usage_store
    async with _usage_store_guard:
        if _usage_store is not None:
            await _usage_store.close()
            _usage_store = None
