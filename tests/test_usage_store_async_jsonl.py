from datetime import datetime, timedelta, timezone

import pytest

from tinker_server import usage_store as usage_store_module
from tinker_server.usage_store import UsageEvent


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_usage_event_defaults():
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=123,
        request_id="req-123",
        label="route=training.train_step",
    )
    assert event.account_id == "aaaaaaaaaaaaaaaaaaaaaaaa"
    assert event.apikey_id == "bbbbbbbbbbbbbbbbbbbbbbbb"
    assert event.charge_item == "training"
    assert event.quantity == 123
    assert event.request_id == "req-123"
    assert event.label == "route=training.train_step"
    assert event.event_time is not None


@pytest.mark.anyio
async def test_jsonl_usage_store_round_trip_preserves_billing_fields(tmp_path):
    store = usage_store_module.JsonlUsageStore(path=tmp_path / "usage_event.jsonl")
    base_time = datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc)
    first = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=123,
        request_id="req-123",
        label="route=training.train_step",
        event_time=base_time,
    )
    second = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="sampling",
        quantity=7,
        request_id="req-456",
        label="route=sampling.asample",
        event_time=base_time + timedelta(seconds=5),
    )

    await store.write_events([first, second])
    await store.write_event(first)

    logs, count, has_more = await store.query_logs(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        limit=10,
        offset=0,
    )
    summary = await store.get_account_summary("aaaaaaaaaaaaaaaaaaaaaaaa")

    assert count == 2
    assert has_more is False
    assert [log["request_id"] for log in logs] == ["req-456", "req-123"]
    assert logs[0]["source_index"] == 2
    assert logs[1]["source_index"] == 1
    assert summary == {"total_quantity": 130, "charge_item_totals": {"training": 123, "sampling": 7}}

    lines = (tmp_path / "usage_event.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = usage_store_module.json.loads(lines[0])
    assert payload == {
        "source_index": 1,
        "event_time": "2026-03-12T10:00:00+00:00",
        "account_id": "aaaaaaaaaaaaaaaaaaaaaaaa",
        "apikey_id": "bbbbbbbbbbbbbbbbbbbbbbbb",
        "charge_item": "training",
        "quantity": 123,
        "request_id": "req-123",
        "label": "route=training.train_step",
    }


@pytest.mark.anyio
async def test_jsonl_usage_store_restart_preserves_idempotency_and_source_index(tmp_path):
    path = tmp_path / "usage_event.jsonl"
    base_time = datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc)

    store = usage_store_module.JsonlUsageStore(path=path)
    await store.write_event(
        UsageEvent(
            account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            charge_item="training",
            quantity=10,
            request_id="req-1",
            label="route=training.train_step",
            event_time=base_time,
        )
    )
    await store.write_event(
        UsageEvent(
            account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            charge_item="sampling",
            quantity=5,
            request_id="req-2",
            label="route=sampling.asample",
            event_time=base_time + timedelta(seconds=1),
        )
    )
    await store.close()

    restarted = usage_store_module.JsonlUsageStore(path=path)
    await restarted.write_events(
        [
            UsageEvent(
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                charge_item="sampling",
                quantity=5,
                request_id="req-2",
                label="route=sampling.asample",
                event_time=base_time + timedelta(seconds=1),
            ),
            UsageEvent(
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                charge_item="checkpoint_storage",
                quantity=2,
                request_id="req-3",
                label="route=checkpoint.archive",
                event_time=base_time + timedelta(seconds=2),
            ),
        ]
    )

    logs, count, _ = await restarted.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)

    assert count == 3
    assert [log["source_index"] for log in logs] == [3, 2, 1]
    assert [log["request_id"] for log in logs] == ["req-3", "req-2", "req-1"]
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3


@pytest.mark.anyio
async def test_jsonl_usage_store_does_not_mutate_memory_when_append_fails(tmp_path, monkeypatch):
    path = tmp_path / "usage_event.jsonl"
    store = usage_store_module.JsonlUsageStore(path=path)
    real_fsync = usage_store_module.os.fsync
    state = {"calls": 0}

    def _flaky_fsync(fd):
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError("disk full")
        return real_fsync(fd)

    monkeypatch.setattr(usage_store_module.os, "fsync", _flaky_fsync)

    with pytest.raises(OSError, match="disk full"):
        await store.write_event(
            UsageEvent(
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                charge_item="training",
                quantity=10,
                request_id="req-1",
                label="route=training.train_step",
            )
        )

    assert path.read_text(encoding="utf-8") == ""

    logs, count, has_more = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
    summary = await store.get_account_summary("aaaaaaaaaaaaaaaaaaaaaaaa")

    assert logs == []
    assert count == 0
    assert has_more is False
    assert summary == {"total_quantity": 0, "charge_item_totals": {}}

    await store.write_event(
        UsageEvent(
            account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            charge_item="training",
            quantity=11,
            request_id="req-2",
            label="route=training.train_step",
        )
    )

    logs, count, _ = await store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)

    assert count == 1
    assert logs[0]["request_id"] == "req-2"
    assert logs[0]["source_index"] == 1


def test_build_usage_store_falls_back_to_jsonl_when_asyncpg_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "postgresql://fake")
    monkeypatch.setattr(usage_store_module.config, "usage_log_dir", str(tmp_path))
    missing = ModuleNotFoundError("No module named 'asyncpg'")
    missing.name = "asyncpg"
    monkeypatch.setattr(usage_store_module, "_import_asyncpg", lambda: (_ for _ in ()).throw(missing))

    store = usage_store_module._build_usage_store()

    assert isinstance(store, usage_store_module.JsonlUsageStore)
    assert store._path == tmp_path / "usage_event.jsonl"


def test_build_usage_store_does_not_mask_non_missing_asyncpg_errors(monkeypatch):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "postgresql://fake")
    monkeypatch.setattr(
        usage_store_module,
        "_import_asyncpg",
        lambda: (_ for _ in ()).throw(RuntimeError("broken asyncpg import")),
    )

    with pytest.raises(RuntimeError, match="broken asyncpg import"):
        usage_store_module._build_usage_store()


def test_build_usage_store_falls_back_to_jsonl_when_pg_dsn_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "")
    monkeypatch.setattr(usage_store_module.config, "usage_log_dir", str(tmp_path))
    monkeypatch.setattr(
        usage_store_module,
        "_import_asyncpg",
        lambda: (_ for _ in ()).throw(AssertionError("asyncpg import should not run without PG DSN")),
    )

    store = usage_store_module._build_usage_store()

    assert isinstance(store, usage_store_module.JsonlUsageStore)
    assert store._path == tmp_path / "usage_event.jsonl"
