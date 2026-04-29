import asyncio
import sys
from types import SimpleNamespace

import pytest

from tinker_server.usage_store import PostgresUsageStore, UsageEvent


class _FakeConn:
    def __init__(self, state: dict):
        self._state = state

    async def execute(self, sql: str, *args):
        text = sql.strip()
        if text.startswith("DELETE FROM"):
            event_ids = set(args[0])
            self._state["rows"] = [row for row in self._state["rows"] if row["event_id"] not in event_ids]
            self._state["seen_event_ids"] -= event_ids
            return "DELETE"
        if not text.startswith("INSERT INTO"):
            self._state["schema_statements"].append(text)
            return "OK"

        if self._state.get("fail_always"):
            raise RuntimeError("simulated persistent pg error")
        if self._state["fail_once"]:
            self._state["fail_once"] = False
            raise RuntimeError("simulated transient pg error")

        self._state["nextval"] += 1
        row = {
            "source_index": self._state["nextval"],
            "event_id": str(args[0]),
            "event_time": args[1],
            "account_id": str(args[2]),
            "apikey_id": str(args[3]),
            "charge_item": str(args[4]),
            "quantity": int(args[5]),
            "request_id": str(args[6]),
            "label": str(args[7]),
        }
        if row["event_id"] in self._state["seen_event_ids"]:
            return "INSERT 0 0"
        self._state["seen_event_ids"].add(row["event_id"])
        self._state["rows"].append(row)
        return "INSERT 0 1"

    async def fetchval(self, sql: str, *args):
        text = sql.strip()
        if text == "SELECT 1":
            return 1
        if "FROM pg_index" in text:
            return bool(self._state.get("event_id_unique_index", True))
        if "WHERE event_id IS NULL" in text:
            return int(self._state.get("null_event_id_count", 0))
        if sql.startswith("SELECT COUNT(*)"):
            return len(self._filter_rows(sql, args))
        if sql.startswith("SELECT COALESCE(SUM(quantity), 0)"):
            account_id = str(args[0])
            return sum(r["quantity"] for r in self._state["rows"] if r["account_id"] == account_id)
        raise AssertionError(f"unexpected SQL for fetchval: {sql}")

    async def fetch(self, sql: str, *args):
        if sql.startswith("SELECT charge_item, COALESCE(SUM(quantity), 0) AS total_quantity"):
            account_id = str(args[0])
            grouped: dict[str, int] = {}
            for row in self._state["rows"]:
                if row["account_id"] != account_id:
                    continue
                item = str(row["charge_item"])
                grouped[item] = grouped.get(item, 0) + int(row["quantity"])
            return [{"charge_item": k, "total_quantity": v} for k, v in grouped.items()]

        if sql.startswith(
            "SELECT source_index, event_time, account_id, apikey_id, charge_item, quantity, request_id, label"
        ):
            rows = self._filter_rows(sql, args)
            limit = int(args[-2])
            offset = int(args[-1])
            ordered = sorted(rows, key=lambda r: (r["event_time"], r["source_index"]), reverse=True)
            return ordered[offset : offset + limit]

        raise AssertionError(f"unexpected SQL for fetch: {sql}")

    def _filter_rows(self, sql: str, args):
        rows = list(self._state["rows"])
        if "WHERE account_id = $1" in sql:
            account_id = str(args[0])
            rows = [r for r in rows if r["account_id"] == account_id]
        return rows


class _FakeAcquireCtx:
    def __init__(self, state: dict):
        self._conn = _FakeConn(state)

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakePool:
    def __init__(self, state: dict):
        self._state = state
        self.closed = False

    def acquire(self):
        return _FakeAcquireCtx(self._state)

    async def close(self):
        self.closed = True


def _state(**overrides):
    state = {
        "rows": [],
        "fail_once": False,
        "fail_always": False,
        "nextval": 1000,
        "seen_event_ids": set(),
        "schema_statements": [],
    }
    state.update(overrides)
    return state


def _install_fake_asyncpg(monkeypatch, state: dict):
    async def _create_pool(**kwargs):
        state["pool_kwargs"] = kwargs
        return _FakePool(state)

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))


async def _no_sleep(_: float):
    return None


def test_postgres_usage_store_dedupes_by_event_id(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake", pool_min=1, pool_max=2, write_timeout_ms=2000)
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="sampling",
        quantity=11,
        request_id="req-same",
        label="model=x,route=sampling.asample,dimension=prefill",
    )

    async def _run():
        await store.write_event(event)
        await store.write_event(event)
        logs, count, has_more = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        summary = await store.get_account_summary("aaaaaaaaaaaaaaaaaaaaaaaa")
        await store.close()

        assert count == 1
        assert has_more is False
        assert len(logs) == 1
        assert logs[0]["request_id"] == "req-same"
        assert summary == {"total_quantity": 11, "charge_item_totals": {"sampling": 11}}
        assert len(state["seen_event_ids"]) == 1

    asyncio.run(_run())


def test_postgres_usage_store_respects_explicit_event_id(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    first = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=9,
        request_id="req-a",
        label="route=training.forward",
        event_id="explicit-event-id",
    )
    second = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=99,
        request_id="req-b",
        label="route=training.forward",
        event_id="explicit-event-id",
    )

    async def _run():
        await store.write_event(first)
        await store.write_event(second)
        logs, count, _ = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        await store.close()

        assert count == 1
        assert logs[0]["quantity"] == 9
        assert state["rows"][0]["event_id"] == "explicit-event-id"

    asyncio.run(_run())


def test_postgres_usage_store_retry_does_not_use_local_source_index(monkeypatch):
    state = _state(fail_once=True)
    _install_fake_asyncpg(monkeypatch, state)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    store = PostgresUsageStore(dsn="postgresql://fake", pool_min=1, pool_max=2, write_timeout_ms=2000)
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="inference",
        quantity=77,
        request_id="req-retry",
        label="route=sampling.compute_logprobs",
    )

    async def _run():
        await store.write_event(event)
        await store.close()

        assert len(state["rows"]) == 1
        assert state["rows"][0]["source_index"] == 1001
        assert any("ALTER COLUMN source_index SET DEFAULT nextval" in stmt for stmt in state["schema_statements"])
        assert not any(stmt.startswith("CREATE TABLE") for stmt in state["schema_statements"])

    asyncio.run(_run())


def test_postgres_usage_store_persistent_pg_failure_fails_request(monkeypatch):
    state = _state(fail_always=True)
    _install_fake_asyncpg(monkeypatch, state)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=12,
        request_id="req-fail",
        label="route=training.train_step",
    )

    async def _run():
        with pytest.raises(RuntimeError, match="simulated persistent pg error"):
            await store.write_event(event)
        await store.close()
        assert state["rows"] == []

    asyncio.run(_run())


def test_postgres_usage_store_delete_events_removes_by_event_id(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=12,
        request_id="req-delete",
        label="route=training.train_step",
    )

    async def _run():
        await store.write_event(event)
        logs, count, _ = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        assert count == 1
        assert logs[0]["request_id"] == "req-delete"

        await store.delete_events([event])
        logs, count, _ = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        await store.close()

        assert count == 0
        assert logs == []

    asyncio.run(_run())


def test_postgres_usage_store_rejects_duplicate_event_id_in_one_batch(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=12,
        request_id="req-fail",
        label="route=training.train_step",
        event_id="same",
    )

    async def _run():
        with pytest.raises(ValueError, match="duplicate usage_event event_id"):
            await store.write_events([event, event])
        await store.close()

    asyncio.run(_run())


def test_postgres_usage_store_rejects_negative_quantity(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=-1,
        request_id="req-negative",
        label="route=training.train_step",
    )

    async def _run():
        with pytest.raises(ValueError, match="quantity must be non-negative"):
            await store.write_event(event)
        await store.close()
        assert state["rows"] == []

    asyncio.run(_run())


def test_postgres_usage_store_fails_when_event_id_unique_index_missing(monkeypatch):
    state = _state(event_id_unique_index=False)
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=1,
        request_id="req-no-index",
        label="route=training.train_step",
    )

    async def _run():
        with pytest.raises(RuntimeError, match="requires a full-table unique index on event_id"):
            await store.write_event(event)
        await store.close()

    asyncio.run(_run())


def test_postgres_usage_store_delete_events_allows_negative_quantity(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=1,
        request_id="req-delete-negative",
        label="route=training.train_step",
    )
    deletion = UsageEvent(
        account_id=event.account_id,
        apikey_id=event.apikey_id,
        charge_item=event.charge_item,
        quantity=-1,
        request_id=event.request_id,
        label=event.label,
    )

    async def _run():
        await store.write_event(event)
        assert len(state["rows"]) == 1
        await store.delete_events([deletion])
        await store.close()
        assert state["rows"] == []

    asyncio.run(_run())


def test_postgres_usage_store_normalizes_unquoted_identifier_case() -> None:
    store = PostgresUsageStore(dsn="postgresql://fake", table="Billing.UsageEvent")
    assert store._schema == "billing"
    assert store._name == "usageevent"
    assert store._table == "billing.usageevent"


def test_postgres_usage_store_delete_events_accepts_explicit_event_id_without_request_id(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=1,
        request_id="req-delete-by-id",
        label="route=training.train_step",
        event_id="delete-by-id",
    )
    deletion = UsageEvent(
        account_id="",
        apikey_id="",
        charge_item="training",
        quantity=-1,
        request_id="",
        label="",
        event_id="delete-by-id",
    )

    async def _run():
        await store.write_event(event)
        assert len(state["rows"]) == 1
        await store.delete_events([deletion])
        await store.close()
        assert state["rows"] == []

    asyncio.run(_run())


def test_postgres_usage_store_normalizes_event_id_fields_for_storage(monkeypatch):
    state = _state()
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="  aaaaaaaaaaaaaaaaaaaaaaaa  ",
        apikey_id="  bbbbbbbbbbbbbbbbbbbbbbbb  ",
        charge_item="  training  ",
        quantity=1,
        request_id="  req-normalized  ",
        label="  route=training.train_step  ",
    )

    async def _run():
        await store.write_event(event)
        await store.close()
        row = state["rows"][0]
        assert row["account_id"] == "aaaaaaaaaaaaaaaaaaaaaaaa"
        assert row["apikey_id"] == "bbbbbbbbbbbbbbbbbbbbbbbb"
        assert row["charge_item"] == "training"
        assert row["request_id"] == "req-normalized"
        assert row["label"] == "route=training.train_step"
        assert row["event_id"] == PostgresUsageStore.build_event_id(
            UsageEvent(
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                charge_item="training",
                quantity=1,
                request_id="req-normalized",
                label="route=training.train_step",
            )
        )

    asyncio.run(_run())


def test_postgres_usage_store_requires_migration_when_event_id_nulls_exist(monkeypatch):
    state = _state(null_event_id_count=1)
    _install_fake_asyncpg(monkeypatch, state)

    store = PostgresUsageStore(dsn="postgresql://fake")
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=1,
        request_id="req-null-event-id",
        label="route=training.train_step",
    )

    async def _run():
        with pytest.raises(RuntimeError, match="004_direct_pg_usage_event_id"):
            await store.write_event(event)
        await store.close()

    asyncio.run(_run())


def test_schedule_usage_events_rejects_new_tasks_after_close_started(monkeypatch):
    import tinker_server.usage_store as usage_store_module

    async def _never_write(events):
        await asyncio.sleep(60)

    async def _run():
        usage_store_module._USAGE_STORE_CLOSING = False
        usage_store_module._PENDING_WRITE_TASKS.clear()
        monkeypatch.setattr(usage_store_module, "_write_usage_events_safely", _never_write)
        event = UsageEvent(
            account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            charge_item="training",
            quantity=1,
            request_id="req-close",
            label="route=training.train_step",
        )
        usage_store_module.schedule_usage_events([event])
        assert len(usage_store_module._PENDING_WRITE_TASKS) == 1
        await usage_store_module.close_usage_store()
        assert usage_store_module._USAGE_STORE_CLOSING is True
        usage_store_module.schedule_usage_events([event])
        assert len(usage_store_module._PENDING_WRITE_TASKS) == 0
        usage_store_module._USAGE_STORE_CLOSING = False

    asyncio.run(_run())
