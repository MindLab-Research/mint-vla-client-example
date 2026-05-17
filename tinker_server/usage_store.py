"""PostgreSQL usage storage backend for billing usage_event."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from .config import config

logger = logging.getLogger(__name__)

ChargeItem = Literal["sampling", "inference", "training", "checkpoint_storage"]
_ALLOWED_CHARGE_ITEMS = {"sampling", "inference", "training", "checkpoint_storage"}
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EVENT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "mindlab.mint.billing.usage_event.v1")
_PENDING_WRITE_TASKS: set[asyncio.Task] = set()
_USAGE_STORE_CLOSING = False


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


_MAX_PENDING_WRITE_TASKS = _env_int("MINT_USAGE_MAX_PENDING_WRITE_TASKS", 1024, minimum=1)
_SHUTDOWN_FLUSH_TIMEOUT_S = _env_float("MINT_USAGE_SHUTDOWN_FLUSH_TIMEOUT_S", 5.0, minimum=0.0)


def _import_asyncpg():
    import asyncpg

    return asyncpg


def _usage_pg_dsn() -> str:
    return str(config.usage_pg_dsn or "").strip()


def _exception_detail(exc: BaseException) -> dict[str, Any]:
    diag = getattr(exc, "diag", None)
    return {
        "error_type": type(exc).__name__,
        "sqlstate": getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None),
        "constraint_name": getattr(exc, "constraint_name", None) or getattr(diag, "constraint_name", None),
        "table_name": getattr(exc, "table_name", None) or getattr(diag, "table_name", None),
        "schema_name": getattr(exc, "schema_name", None) or getattr(diag, "schema_name", None),
    }


def _is_permanent_pg_write_error(exc: BaseException) -> bool:
    sqlstate = str(getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None) or "")
    # SQLSTATE class 23 is integrity constraint violation. Retrying cannot fix
    # permanent identity/schema issues such as missing apikey FK rows.
    return sqlstate.startswith("23")


def _event_detail(events: list["UsageEvent"]) -> dict[str, Any]:
    return {
        "event_count": len(events),
        "event_ids": [event.event_id for event in events],
        "request_ids": [event.request_id for event in events],
        "charge_items": sorted({str(event.charge_item) for event in events}),
        "account_ids": sorted({str(event.account_id) for event in events}),
        "apikey_ids": sorted({str(event.apikey_id) for event in events}),
    }


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
        if outbox_path or float(outbox_flush_interval_s or 0.0) != 0.0:
            logger.warning("PostgresUsageStore ignores outbox configuration; direct PostgreSQL usage writes are required")
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
            if charge_item not in _ALLOWED_CHARGE_ITEMS:
                raise ValueError(f"unsupported usage_event charge_item: {charge_item!r}")
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
                account_id = str(event.account_id or "").strip()
                apikey_id = str(event.apikey_id or "").strip()
                request_id = str(event.request_id or "").strip()
                charge_item = str(event.charge_item or "").strip()
                label = str(event.label or "").strip()
                if not account_id or not apikey_id or not request_id:
                    raise ValueError("usage_event delete requires event_id or complete identity fields")
                if charge_item not in _ALLOWED_CHARGE_ITEMS:
                    raise ValueError(f"unsupported usage_event charge_item: {charge_item!r}")
                event_id = self.build_event_id(
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

            pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                command_timeout=max(5.0, self._write_timeout_s),
                server_settings={"application_name": "tinker_server_usage"},
                statement_cache_size=0,
            )
            try:
                await self._ensure_pg_schema(pool)
            except Exception:
                await pool.close()
                raise
            self._pool = pool
            return pool

    async def _ensure_pg_schema(self, pool) -> None:
        if pool is None:
            raise RuntimeError("usage pool is not initialized")
        async with pool.acquire() as conn:
            await self._assert_pg_schema_ready(conn)
            await self._assert_event_id_unique_index(conn)

    async def _assert_pg_schema_ready(self, conn) -> None:
        required_columns = [
            "source_index",
            "event_id",
            "event_time",
            "account_id",
            "apikey_id",
            "charge_item",
            "quantity",
            "request_id",
            "label",
            "created_at",
        ]
        required_not_null = [
            "source_index",
            "event_id",
            "event_time",
            "account_id",
            "apikey_id",
            "charge_item",
            "quantity",
            "request_id",
            "label",
            "created_at",
        ]
        migration_hint = (
            "run the platform direct-PG billing migration, or scripts/tools/init_usage_pg.sql "
            "for a standalone MinT usage DB, before starting MinT direct-PG billing"
        )
        table_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = $1
                  AND table_name = $2
            )
            """,
            self._schema,
            self._name,
        )
        if not table_exists:
            raise RuntimeError(f"{self._table} is missing; {migration_hint}")

        column_count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = $2
              AND column_name = ANY($3::text[])
            """,
            self._schema,
            self._name,
            required_columns,
        )
        if int(column_count or 0) != len(required_columns):
            raise RuntimeError(f"{self._table} schema is missing direct-PG usage_event columns; {migration_hint}")

        not_null_count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = $2
              AND column_name = ANY($3::text[])
              AND is_nullable = 'NO'
            """,
            self._schema,
            self._name,
            required_not_null,
        )
        if int(not_null_count or 0) != len(required_not_null):
            raise RuntimeError(f"{self._table} schema has nullable direct-PG usage_event columns; {migration_hint}")

        source_index_default = await conn.fetchval(
            """
            SELECT column_default
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = $2
              AND column_name = 'source_index'
            """,
            self._schema,
            self._name,
        )
        expected_sequence = self._sequence.split(".")[-1]
        if "nextval(" not in str(source_index_default or "") or expected_sequence not in str(source_index_default or ""):
            raise RuntimeError(
                f"{self._table}.source_index is missing the expected sequence default {self._sequence}; {migration_hint}"
            )

        has_null_event_id = await conn.fetchval(f"SELECT EXISTS(SELECT 1 FROM {self._table} WHERE event_id IS NULL)")
        if bool(has_null_event_id):
            raise RuntimeError(f"{self._table} contains usage_event rows with NULL event_id; {migration_hint}")

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
            raise RuntimeError(f"{self._table} requires a full-table unique index on event_id, expected {self._dedupe_index}")

    async def _write_events_to_pg(self, events: list[UsageEvent]) -> list[str]:
        pool = await self._ensure_pool()
        sql = f"""
        INSERT INTO {self._table}
            (event_id, event_time, account_id, apikey_id, charge_item, quantity, request_id, label)
        SELECT *
        FROM UNNEST(
            $1::text[],
            $2::timestamptz[],
            $3::text[],
            $4::text[],
            $5::text[],
            $6::bigint[],
            $7::text[],
            $8::text[]
        )
        ON CONFLICT (event_id) DO NOTHING
        RETURNING event_id
        """

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with pool.acquire() as conn:
                    rows = await asyncio.wait_for(
                        conn.fetch(
                            sql,
                            [event.event_id for event in events],
                            [self._normalize_event_time(event.event_time) for event in events],
                            [event.account_id for event in events],
                            [event.apikey_id for event in events],
                            [event.charge_item for event in events],
                            [event.quantity for event in events],
                            [event.request_id for event in events],
                            [event.label or "" for event in events],
                        ),
                        timeout=self._write_timeout_s,
                    )
                return [str(row["event_id"]) for row in rows]
            except Exception as e:
                last_error = e
                permanent_error = _is_permanent_pg_write_error(e)
                if permanent_error:
                    break
                if attempt < 2:
                    backoff_s = float(2**attempt)
                    event_detail = _event_detail(events)
                    error_detail = _exception_detail(e)
                    logger.warning(
                        "usage_event pg write failed, retrying: attempt=%s event_ids=%s request_ids=%s charge_items=%s error_type=%s sqlstate=%s constraint_name=%s table_name=%s err=%s",
                        attempt + 1,
                        event_detail["event_ids"],
                        event_detail["request_ids"],
                        event_detail["charge_items"],
                        error_detail["error_type"],
                        error_detail["sqlstate"],
                        error_detail["constraint_name"],
                        error_detail["table_name"],
                        e,
                    )
                    await asyncio.sleep(backoff_s)
                    continue
                break

        if last_error is not None:
            event_detail = _event_detail(events)
            error_detail = _exception_detail(last_error)
            logger.error(
                "usage_event pg write exhausted: event_ids=%s request_ids=%s charge_items=%s event_count=%s error_type=%s sqlstate=%s constraint_name=%s table_name=%s err=%s",
                event_detail["event_ids"],
                event_detail["request_ids"],
                event_detail["charge_items"],
                event_detail["event_count"],
                error_detail["error_type"],
                error_detail["sqlstate"],
                error_detail["constraint_name"],
                error_detail["table_name"],
                last_error,
            )
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
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                val = await conn.fetchval("SELECT 1")
            return int(val) == 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("usage_event postgres health check failed", exc_info=True)
            return False

    async def close(self) -> None:
        if self._pool is not None:
            with suppress(Exception):
                await self._pool.close()
            self._pool = None


class DisabledUsageStore:
    async def write_event(self, event: UsageEvent) -> list[str]:
        return await self.write_events([event])

    async def write_events(self, events: list[UsageEvent]) -> list[str]:
        return [
            str(event.event_id or "").strip() or PostgresUsageStore.build_event_id(event)
            for event in events
        ]

    async def flush_outbox(self, limit: int = 0) -> int:
        return 0

    async def delete_events(self, events: list[UsageEvent]) -> None:
        return None

    async def delete_event_ids(self, event_ids: list[str]) -> None:
        return None

    async def query_logs(
        self,
        since: datetime | None = None,
        account_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int, bool]:
        return [], 0, False

    async def get_account_summary(self, account_id: str) -> dict:
        return {"total_quantity": 0, "charge_item_totals": {}}

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


_usage_store: UsageStore | None = None
_usage_store_guard = asyncio.Lock()


def _build_usage_store() -> UsageStore:
    backend = str(config.usage_backend or "postgres").strip().lower()
    if backend in {"disabled", "noop"}:
        return DisabledUsageStore()
    if backend != "postgres":
        raise ValueError(
            f"Unsupported usage backend {backend!r}; expected one of 'postgres', 'disabled', or 'noop'"
        )
    dsn = _usage_pg_dsn()
    if not dsn:
        raise ValueError("MINT_USAGE_PG_DSN or MINT_USAGE_PG_HOST is required for postgres usage backend")
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
    except Exception as e:
        event_detail = _event_detail(events)
        error_detail = _exception_detail(e)
        logger.exception(
            "usage_event async persistence failed after user result completed: event_ids=%s request_ids=%s charge_items=%s event_count=%s error_type=%s sqlstate=%s constraint_name=%s table_name=%s",
            event_detail["event_ids"],
            event_detail["request_ids"],
            event_detail["charge_items"],
            event_detail["event_count"],
            error_detail["error_type"],
            error_detail["sqlstate"],
            error_detail["constraint_name"],
            error_detail["table_name"],
        )


async def persist_usage_events(events: list[UsageEvent]) -> None:
    normalized = list(events)
    if normalized:
        await _write_usage_events_safely(normalized)


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
    pending = list(_PENDING_WRITE_TASKS)
    if pending:
        done, pending_set = await asyncio.wait(pending, timeout=_SHUTDOWN_FLUSH_TIMEOUT_S)
        for task in done:
            with suppress(Exception, asyncio.CancelledError):
                task.result()
        for task in pending_set:
            task.cancel()
        if pending_set:
            await asyncio.gather(*pending_set, return_exceptions=True)
        _PENDING_WRITE_TASKS.clear()
    async with _usage_store_guard:
        if _usage_store is not None:
            await _usage_store.close()
            _usage_store = None
