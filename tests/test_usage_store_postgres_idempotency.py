import asyncio
import sys
from types import SimpleNamespace

from tinker_server.usage_store import PostgresUsageStore, UsageEvent


class _FakeConn:
    def __init__(self, state: dict):
        self._state = state

    async def execute(self, sql: str, *args):
        if sql.startswith("CREATE SEQUENCE IF NOT EXISTS"):
            self._state["sequence_ensured"] += 1
            return "CREATE SEQUENCE"
        if sql.startswith("CREATE UNIQUE INDEX IF NOT EXISTS"):
            self._state["index_ensured"] += 1
            return "CREATE INDEX"
        if "INSERT INTO" not in sql:
            raise AssertionError(f"unexpected SQL for execute: {sql}")
        source_index = int(args[0])
        self._state["source_indexes"].append(source_index)

        if self._state.get("fail_always"):
            raise RuntimeError("simulated persistent pg error")
        if self._state["fail_once"]:
            self._state["fail_once"] = False
            raise RuntimeError("simulated transient pg error")

        row = {
            "source_index": source_index,
            "event_time": args[1],
            "account_id": str(args[2]),
            "apikey_id": str(args[3]),
            "charge_item": str(args[4]),
            "quantity": int(args[5]),
            "request_id": str(args[6]),
            "label": str(args[7]),
        }
        key = (row["request_id"], row["charge_item"], row["label"])
        if key in self._state["seen_keys"]:
            return "INSERT 0 0"
        self._state["seen_keys"].add(key)
        self._state["rows"].append(row)
        return "INSERT 0 1"

    async def fetchval(self, sql: str, *args):
        if sql.strip() == "SELECT 1":
            return 1
        if sql.startswith("SELECT nextval("):
            self._state["nextval"] += 1
            return self._state["nextval"]
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

        if sql.startswith("SELECT source_index, event_time, account_id, apikey_id, charge_item, quantity, request_id, label"):
            rows = self._filter_rows(sql, args)
            limit = int(args[-2])
            offset = int(args[-1])
            ordered = sorted(rows, key=lambda r: r["event_time"], reverse=True)
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


def test_postgres_usage_store_retry_regenerates_source_index(monkeypatch):
    state = {
        "rows": [],
        "source_indexes": [],
        "fail_once": True,
        "nextval": 1000,
        "sequence_ensured": 0,
        "index_ensured": 0,
        "seen_keys": set(),
    }

    async def _create_pool(**kwargs):
        _ = kwargs
        return _FakePool(state)

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    async def _no_sleep(_: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    store = PostgresUsageStore(
        dsn="postgresql://fake-user:fake-pass@fake-host:5432/fake-db",
        pool_min=1,
        pool_max=2,
        write_timeout_ms=2000,
    )

    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="inference",
        quantity=77,
        request_id="req-dup",
        label="route=sampling.asample",
    )

    async def _run():
        await store.write_event(event)
        logs, count, has_more = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        summary = await store.get_account_summary("aaaaaaaaaaaaaaaaaaaaaaaa")
        await store.close()

        assert count == 1
        assert has_more is False
        assert len(logs) == 1
        assert summary["total_quantity"] == 77
        assert summary["charge_item_totals"]["inference"] == 77

        assert state["sequence_ensured"] >= 1
        assert state["index_ensured"] >= 1
        assert len(state["source_indexes"]) >= 2
        assert state["source_indexes"][0] != state["source_indexes"][1]

    asyncio.run(_run())


def test_postgres_usage_store_dedupes_replayed_request(monkeypatch, tmp_path):
    state = {
        "rows": [],
        "source_indexes": [],
        "fail_once": False,
        "fail_always": False,
        "nextval": 2000,
        "sequence_ensured": 0,
        "index_ensured": 0,
        "seen_keys": set(),
    }

    async def _create_pool(**kwargs):
        _ = kwargs
        return _FakePool(state)

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    store = PostgresUsageStore(
        dsn="postgresql://fake-user:fake-pass@fake-host:5432/fake-db",
        outbox_path=str(tmp_path / "usage_outbox.sqlite3"),
    )

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
        logs, count, _ = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        await store.close()

        assert count == 1
        assert len(logs) == 1
        assert logs[0]["request_id"] == "req-same"

    asyncio.run(_run())


def test_postgres_usage_store_outbox_flush_recovers_after_pg_failure(monkeypatch, tmp_path):
    state = {
        "rows": [],
        "source_indexes": [],
        "fail_once": False,
        "fail_always": True,
        "nextval": 3000,
        "sequence_ensured": 0,
        "index_ensured": 0,
        "seen_keys": set(),
    }

    async def _create_pool(**kwargs):
        _ = kwargs
        return _FakePool(state)

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    async def _no_sleep(_: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    store = PostgresUsageStore(
        dsn="postgresql://fake-user:fake-pass@fake-host:5432/fake-db",
        outbox_path=str(tmp_path / "usage_outbox.sqlite3"),
        outbox_flush_interval_s=3600,
    )
    monkeypatch.setattr(store, "_ensure_flush_task", lambda: None)

    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=9,
        request_id="req-outbox",
        label="model=y,route=training.forward,dimension=train",
    )

    async def _run():
        await store.write_event(event)
        logs, count, _ = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        assert count == 0
        assert logs == []

        state["fail_always"] = False
        flushed = await store.flush_outbox(limit=10)
        logs, count, _ = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
        await store.close()

        assert flushed == 1
        assert count == 1
        assert logs[0]["request_id"] == "req-outbox"

    asyncio.run(_run())


def test_postgres_usage_store_close_drains_outbox(monkeypatch, tmp_path):
    state = {
        "rows": [],
        "source_indexes": [],
        "fail_once": False,
        "fail_always": True,
        "nextval": 4000,
        "sequence_ensured": 0,
        "index_ensured": 0,
        "seen_keys": set(),
    }

    async def _create_pool(**kwargs):
        _ = kwargs
        return _FakePool(state)

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    async def _no_sleep(_: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    store = PostgresUsageStore(
        dsn="postgresql://fake-user:fake-pass@fake-host:5432/fake-db",
        outbox_path=str(tmp_path / "usage_outbox.sqlite3"),
        outbox_flush_interval_s=3600,
    )

    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=12,
        request_id="req-close",
        label="model=z,route=training.train_step,dimension=train",
    )

    async def _run():
        await store.write_event(event)
        state["fail_always"] = False
        await store.close()

        assert len(state["rows"]) == 1
        assert state["rows"][0]["request_id"] == "req-close"

    asyncio.run(_run())


def test_postgres_usage_store_close_drains_outbox_even_if_pool_never_initialized(monkeypatch, tmp_path):
    state = {
        "rows": [],
        "source_indexes": [],
        "fail_once": False,
        "fail_always": False,
        "fail_create_pool": True,
        "nextval": 5000,
        "sequence_ensured": 0,
        "index_ensured": 0,
        "seen_keys": set(),
    }

    async def _create_pool(**kwargs):
        _ = kwargs
        if state["fail_create_pool"]:
            raise RuntimeError("pool unavailable")
        return _FakePool(state)

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(create_pool=_create_pool))

    async def _no_sleep(_: float):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    store = PostgresUsageStore(
        dsn="postgresql://fake-user:fake-pass@fake-host:5432/fake-db",
        outbox_path=str(tmp_path / "usage_outbox.sqlite3"),
        outbox_flush_interval_s=3600,
    )

    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="inference",
        quantity=5,
        request_id="req-pool-none",
        label="model=z,route=sampling.compute_logprobs,dimension=prefill",
    )

    async def _run():
        await store.write_event(event)
        state["fail_create_pool"] = False
        await store.close()

        assert len(state["rows"]) == 1
        assert state["rows"][0]["request_id"] == "req-pool-none"

    asyncio.run(_run())
