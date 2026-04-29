"""PostgreSQL usage storage backend for billing usage_event."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Literal, Protocol

from .config import config

logger = logging.getLogger(__name__)

ChargeItem = Literal["sampling", "inference", "training", "checkpoint_storage"]
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EVENT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "mindlab.mint.billing.usage_event.v1")
_PENDING_WRITE_TASKS: set[asyncio.Task] = set()
_MAX_PENDING_WRITE_TASKS = max(1, int(os.environ.get("MINT_USAGE_MAX_PENDING_WRITE_TASKS", "1024")))
_USAGE_STORE_CLOSING = False


def _import_asyncpg():
    import asyncpg

    return asyncpg


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
    event_id: str = ""
    event_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UsageStore(Protocol):
    async def write_event(self, event: UsageEvent) -> list[str]: ...

    async def write_events(self, events: list[UsageEvent]) -> list[str]: ...

    async def flush_outbox(self, limit: int = 0) -> int: ...

    async def delete_events(self, events: list[UsageEvent]) -> None: ...

    async def delete_event_ids(self, event_ids: list[str]) -> None: ...

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
        table: str = "usage_event",
        outbox_path: str | None = None,
        outbox_flush_interval_s: float = 0.0,
    ):
        if not dsn:
            raise ValueError("Postgres DSN is required for usage backend")
        _ = outbox_path, outbox_flush_interval_s
        self._dsn = dsn
        self._pool_min = max(1, int(pool_min))
        self._pool_max = max(self._pool_min, int(pool_max))
        self._write_timeout_s = max(0.1, float(write_timeout_ms) / 1000.0)
        self._pool = None
        self._pool_lock = asyncio.Lock()
        self._schema, self._name = self._parse_qualified_name(str(table or "usage_event").strip())
        self._table = f"{self._schema}.{self._name}"
        self._sequence = f"{self._schema}.{self._name}_source_index_seq"
        self._dedupe_index = f"idx_{self._name}_event_id_uniq"

    @staticmethod
    def _parse_qualified_name(value: str) -> tuple[str, str]:
        parts = [part.strip() for part in value.split(".", 1)]
        if len(parts) == 1:
            schema, name = "public", parts[0]
        else:
            schema, name = parts
        if not _SQL_IDENT_RE.match(schema) or not _SQL_IDENT_RE.match(name):
            raise ValueError(f"Unsupported SQL identifier: {value!r}")
        return schema.lower(), name.lower()

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
    def build_event_id(event: UsageEvent) -> str:
        key = "|".join(
            [
                str(event.account_id).strip(),
                str(event.apikey_id).strip(),
                str(event.request_id).strip(),
                str(event.charge_item).strip(),
                str(event.label or "").strip(),
            ]
        )
        return uuid.uuid5(_EVENT_ID_NAMESPACE, key).hex

    def _normalize_events(self, events: list[UsageEvent]) -> list[UsageEvent]:
        raw = list(events)
        if not raw:
            raise ValueError("write_events requires at least one event")
        normalized: list[UsageEvent] = []
        seen: set[str] = set()
        for event in raw:
            account_id = str(event.account_id or "").strip()
            apikey_id = str(event.apikey_id or "").strip()
            request_id = str(event.request_id or "").strip()
            charge_item = str(event.charge_item or "").strip()
            label = str(event.label or "").strip()
            if not request_id:
                raise ValueError("usage_event request_id must be non-empty")
            quantity = int(event.quantity)
            if quantity < 0:
                raise ValueError("usage_event quantity must be non-negative")
            event_id = str(event.event_id or "").strip() or self.build_event_id(
                replace(
                    event,
                    account_id=account_id,
                    apikey_id=apikey_id,
                    request_id=request_id,
                    charge_item=charge_item,
                    label=label,
                )
            )
            if event_id in seen:
                raise ValueError(f"duplicate usage_event event_id in batch: {event_id!r}")
            seen.add(event_id)
            normalized.append(
                replace(
                    event,
                    account_id=account_id,
                    apikey_id=apikey_id,
                    request_id=request_id,
                    charge_item=charge_item,
                    event_id=event_id,
                    quantity=quantity,
                    label=label,
                )
            )
        return normalized

    def _event_ids_for_delete(self, events: list[UsageEvent]) -> list[str]:
        raw = list(events)
        if not raw:
            return []
        event_ids: list[str] = []
        seen: set[str] = set()
        for event in raw:
            event_id = str(event.event_id or "").strip()
            if not event_id:
                request_id = str(event.request_id or "").strip()
                if not request_id:
                    raise ValueError("usage_event request_id must be non-empty")
                event_id = self.build_event_id(event)
            if event_id in seen:
                continue
            seen.add(event_id)
            event_ids.append(event_id)
        return event_ids

    async def _ensure_pool(self):
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
            return self._pool

    async def _ensure_pg_schema(self) -> None:
        pool = self._pool
        if pool is None:
            raise RuntimeError("usage pool is not initialized")
        async with pool.acquire() as conn:
            if self._schema != "public":
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
            await conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {self._sequence} AS BIGINT")
            await conn.execute(f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS event_id TEXT")
            null_event_ids = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table} WHERE event_id IS NULL")
            if int(null_event_ids or 0) > 0:
                raise RuntimeError(
                    f"{self._table} contains usage_event rows with NULL event_id; "
                    "run backend/db/migrations/004_direct_pg_usage_event_id.sql before starting MinT direct-PG billing"
                )
            await conn.execute(f"ALTER TABLE {self._table} ALTER COLUMN event_id SET NOT NULL")
            await conn.execute(
                f"ALTER TABLE {self._table} ALTER COLUMN source_index SET DEFAULT nextval('{self._sequence}'::regclass)"
            )
            await conn.execute(
                f"""
                SELECT setval(
                    '{self._sequence}',
                    GREATEST(
                        (SELECT COALESCE(MAX(source_index), 0) FROM {self._table}),
                        (SELECT last_value FROM {self._sequence})
                    ),
                    true
                )
                """
            )
            await conn.execute(
                f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS {self._dedupe_index} ON {self._table} (event_id)"
            )
            await self._assert_event_id_unique_index(conn)

    async def _assert_event_id_unique_index(self, conn) -> None:
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_index i
                JOIN pg_class t ON t.oid = i.indrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                JOIN pg_attribute a
                  ON a.attrelid = t.oid
                 AND a.attnum = (i.indkey::smallint[])[0]
                WHERE n.nspname = $1
                  AND t.relname = $2
                  AND i.indisunique
                  AND i.indisvalid
                  AND i.indpred IS NULL
                  AND COALESCE((to_jsonb(i)->>'indnkeyatts')::int, i.indnatts) = 1
                  AND a.attname = 'event_id'
            )
            """,
            self._schema,
            self._name,
        )
        if not exists:
            raise RuntimeError(f"{self._table} requires a full-table unique index on event_id")

    @asynccontextmanager
    async def _maybe_transaction(self, conn):
        tx_factory = getattr(conn, "transaction", None)
        if callable(tx_factory):
            async with tx_factory():
                yield
            return
        async with _null_async_context():
            yield

    async def _write_events_to_pg(self, events: list[UsageEvent]) -> list[str]:
        pool = await self._ensure_pool()
        sql = f"""
        INSERT INTO {self._table}
            (event_id, event_time, account_id, apikey_id, charge_item, quantity, request_id, label)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (event_id) DO NOTHING
        """

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                inserted_event_ids: list[str] = []
                async with pool.acquire() as conn:
                    async with self._maybe_transaction(conn):
                        for event in events:
                            result = await asyncio.wait_for(
                                conn.execute(
                                    sql,
                                    event.event_id,
                                    self._normalize_event_time(event.event_time),
                                    event.account_id,
                                    event.apikey_id,
                                    event.charge_item,
                                    event.quantity,
                                    event.request_id,
                                    event.label or "",
                                ),
                                timeout=self._write_timeout_s,
                            )
                            if str(result).strip().endswith(" 1"):
                                inserted_event_ids.append(event.event_id)
                return inserted_event_ids
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

    async def write_event(self, event: UsageEvent) -> list[str]:
        return await self.write_events([event])

    async def write_events(self, events: list[UsageEvent]) -> list[str]:
        return await self._write_events_to_pg(self._normalize_events(events))

    async def flush_outbox(self, limit: int = 0) -> int:
        _ = limit
        return 0

    async def delete_events(self, events: list[UsageEvent]) -> None:
        await self.delete_event_ids(self._event_ids_for_delete(events))

    async def delete_event_ids(self, event_ids: list[str]) -> None:
        normalized = [str(event_id).strip() for event_id in event_ids if str(event_id).strip()]
        if not normalized:
            return
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await asyncio.wait_for(
                conn.execute(f"DELETE FROM {self._table} WHERE event_id = ANY($1::text[])", normalized),
                timeout=self._write_timeout_s,
            )

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
            f"ORDER BY event_time DESC, source_index DESC LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"
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
        if self._pool is not None:
            with suppress(Exception):
                await self._pool.close()
            self._pool = None


_usage_store: UsageStore | None = None
_usage_store_guard = asyncio.Lock()


def _build_usage_store() -> UsageStore:
    backend = str(config.usage_backend or "postgres").strip().lower()
    if backend != "postgres":
        raise ValueError(f"Unsupported usage backend {backend!r}; only 'postgres' is accepted")
    dsn = _usage_pg_dsn()
    if not dsn:
        raise ValueError("TINKER_USAGE_PG_DSN or TINKER_USAGE_PG_HOST is required for postgres usage backend")
    return PostgresUsageStore(
        dsn=dsn,
        pool_min=config.usage_pg_pool_min,
        pool_max=config.usage_pg_pool_max,
        write_timeout_ms=config.usage_write_timeout_ms,
        table=config.usage_pg_table,
    )


async def get_usage_store() -> UsageStore:
    global _usage_store
    if _usage_store is not None:
        return _usage_store
    async with _usage_store_guard:
        if _usage_store is None:
            _usage_store = _build_usage_store()
    return _usage_store


async def _write_usage_events_safely(events: list[UsageEvent]) -> None:
    try:
        usage_store = await get_usage_store()
        await usage_store.write_events(events)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "usage_event async persistence failed after user result completed: request_ids=%s",
            [event.request_id for event in events],
        )


def schedule_usage_events(events: list[UsageEvent]) -> None:
    normalized = list(events)
    if not normalized:
        return
    if _USAGE_STORE_CLOSING:
        logger.error(
            "usage_event async persistence dropped because usage store is closing: request_ids=%s",
            [event.request_id for event in normalized],
        )
        return
    if len(_PENDING_WRITE_TASKS) >= _MAX_PENDING_WRITE_TASKS:
        logger.error(
            "usage_event async persistence dropped because pending task limit is full: pending=%s limit=%s request_ids=%s",
            len(_PENDING_WRITE_TASKS),
            _MAX_PENDING_WRITE_TASKS,
            [event.request_id for event in normalized],
        )
        return
    task = asyncio.create_task(_write_usage_events_safely(normalized))
    _PENDING_WRITE_TASKS.add(task)
    task.add_done_callback(_PENDING_WRITE_TASKS.discard)


async def close_usage_store() -> None:
    global _usage_store, _USAGE_STORE_CLOSING
    _USAGE_STORE_CLOSING = True
    for task in list(_PENDING_WRITE_TASKS):
        task.cancel()
    if _PENDING_WRITE_TASKS:
        await asyncio.gather(*list(_PENDING_WRITE_TASKS), return_exceptions=True)
        _PENDING_WRITE_TASKS.clear()
    async with _usage_store_guard:
        if _usage_store is not None:
            await _usage_store.close()
            _usage_store = None
