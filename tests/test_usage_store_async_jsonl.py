import asyncio
import logging
import multiprocessing
from datetime import datetime, timedelta, timezone

import pytest

from tinker_server import usage_store as usage_store_module
from tinker_server.usage_store import UsageEvent


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _write_usage_event_in_process(
    path_str: str,
    request_id: str,
    charge_item: str,
    label: str,
    quantity: int,
    event_time_iso: str,
    start_event,
) -> None:
    start_event.wait(timeout=5)
    store = usage_store_module.JsonlUsageStore(path=path_str)
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item=charge_item,
        quantity=quantity,
        request_id=request_id,
        label=label,
        event_time=datetime.fromisoformat(event_time_iso),
    )
    asyncio.run(store.write_event(event))


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


def test_build_usage_store_logs_jsonl_path_without_pg_config(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "")
    monkeypatch.setattr(usage_store_module.config, "usage_log_dir", str(tmp_path))

    with caplog.at_level(logging.INFO):
        store = usage_store_module._build_usage_store()

    assert isinstance(store, usage_store_module.JsonlUsageStore)
    assert store._path == tmp_path / "usage_event.jsonl"
    assert f"usage JSONL store active at {tmp_path / 'usage_event.jsonl'}" in caplog.text


def test_build_usage_store_ignores_pg_dsn_and_keeps_jsonl(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "postgresql://fake")
    monkeypatch.setattr(usage_store_module.config, "usage_log_dir", str(tmp_path))
    monkeypatch.setattr(
        usage_store_module,
        "_import_asyncpg",
        lambda: (_ for _ in ()).throw(AssertionError("asyncpg import should not run for deprecated PG path")),
    )

    with caplog.at_level(logging.WARNING):
        store = usage_store_module._build_usage_store()

    assert isinstance(store, usage_store_module.JsonlUsageStore)
    assert store._path == tmp_path / "usage_event.jsonl"
    assert "usage PG config is deprecated and ignored" in caplog.text
    assert str(tmp_path / "usage_event.jsonl") in caplog.text


def test_build_usage_store_warns_when_pg_config_falls_back_to_default_tmp(monkeypatch, caplog):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "postgresql://fake")
    monkeypatch.setattr(usage_store_module.config, "usage_log_dir", "/tmp/tinker_usage")

    with caplog.at_level(logging.WARNING):
        store = usage_store_module._build_usage_store()

    assert isinstance(store, usage_store_module.JsonlUsageStore)
    assert store._path == usage_store_module.Path("/tmp/tinker_usage/usage_event.jsonl")
    assert "usage PG config is deprecated and ignored" in caplog.text
    assert "set TINKER_USAGE_LOG_DIR explicitly for a shared pull target" in caplog.text


def test_build_usage_store_fails_fast_for_non_postgres_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "sqlite")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "")
    monkeypatch.setattr(usage_store_module.config, "usage_log_dir", str(tmp_path))

    with pytest.raises(ValueError, match="Unsupported usage backend 'sqlite'"):
        usage_store_module._build_usage_store()


@pytest.mark.anyio
async def test_jsonl_usage_store_two_instances_refresh_source_index_before_append(tmp_path):
    path = tmp_path / "usage_event.jsonl"
    base_time = datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc)
    first_store = usage_store_module.JsonlUsageStore(path=path)
    second_store = usage_store_module.JsonlUsageStore(path=path)

    await first_store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
    await second_store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)

    await first_store.write_event(
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
    await second_store.write_event(
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

    reloaded = usage_store_module.JsonlUsageStore(path=path)
    logs, count, _ = await reloaded.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)

    assert count == 2
    assert [log["source_index"] for log in logs] == [2, 1]
    assert [log["request_id"] for log in logs] == ["req-2", "req-1"]


@pytest.mark.anyio
async def test_jsonl_usage_store_two_instances_refresh_dedupe_state_before_append(tmp_path):
    path = tmp_path / "usage_event.jsonl"
    store_a = usage_store_module.JsonlUsageStore(path=path)
    store_b = usage_store_module.JsonlUsageStore(path=path)

    await store_a.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
    await store_b.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)

    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="training",
        quantity=10,
        request_id="req-dup",
        label="route=training.train_step",
        event_time=datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc),
    )

    await store_a.write_event(event)
    await store_b.write_event(event)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


@pytest.mark.anyio
async def test_jsonl_usage_store_stale_reader_refreshes_after_external_append(tmp_path):
    path = tmp_path / "usage_event.jsonl"
    reader = usage_store_module.JsonlUsageStore(path=path)
    writer = usage_store_module.JsonlUsageStore(path=path)

    logs, count, _ = await reader.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
    assert logs == []
    assert count == 0

    await writer.write_event(
        UsageEvent(
            account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            charge_item="training",
            quantity=10,
            request_id="req-external-refresh",
            label="route=training.train_step",
            event_time=datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc),
        )
    )

    logs, count, _ = await reader.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)
    assert count == 1
    assert logs[0]["request_id"] == "req-external-refresh"


@pytest.mark.anyio
async def test_jsonl_usage_store_steady_state_writes_do_not_full_reload(tmp_path, monkeypatch):
    path = tmp_path / "usage_event.jsonl"
    store = usage_store_module.JsonlUsageStore(path=path)

    await store.write_event(
        UsageEvent(
            account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            charge_item="training",
            quantity=10,
            request_id="req-initial",
            label="route=training.train_step",
            event_time=datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc),
        )
    )

    calls = {"count": 0}
    real = store._load_from_stream_locked

    def _counting_load(stream):
        calls["count"] += 1
        return real(stream)

    monkeypatch.setattr(store, "_load_from_stream_locked", _counting_load)

    await store.write_event(
        UsageEvent(
            account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
            apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
            charge_item="sampling",
            quantity=5,
            request_id="req-second",
            label="route=sampling.asample",
            event_time=datetime(2026, 3, 12, 10, 0, 1, tzinfo=timezone.utc),
        )
    )

    assert calls["count"] == 0


@pytest.mark.anyio
async def test_jsonl_usage_store_concurrent_writers_preserve_unique_indices(tmp_path):
    path = tmp_path / "usage_event.jsonl"
    base_time = datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc)
    first_store = usage_store_module.JsonlUsageStore(path=path)
    second_store = usage_store_module.JsonlUsageStore(path=path)

    await asyncio.gather(
        first_store.write_event(
            UsageEvent(
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                charge_item="training",
                quantity=10,
                request_id="req-concurrent-1",
                label="route=training.train_step",
                event_time=base_time,
            )
        ),
        second_store.write_event(
            UsageEvent(
                account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
                apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
                charge_item="sampling",
                quantity=5,
                request_id="req-concurrent-2",
                label="route=sampling.asample",
                event_time=base_time + timedelta(seconds=1),
            )
        ),
    )

    reloaded = usage_store_module.JsonlUsageStore(path=path)
    logs, count, _ = await reloaded.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0)

    assert count == 2
    assert sorted(log["source_index"] for log in logs) == [1, 2]
    assert sorted(log["request_id"] for log in logs) == ["req-concurrent-1", "req-concurrent-2"]


def test_jsonl_usage_store_multiprocess_writers_preserve_unique_indices(tmp_path):
    path = tmp_path / "usage_event.jsonl"
    ctx = multiprocessing.get_context("fork")
    start_event = ctx.Event()
    p1 = ctx.Process(
        target=_write_usage_event_in_process,
        args=(
            str(path),
            "req-mp-1",
            "training",
            "route=training.train_step",
            10,
            datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc).isoformat(),
            start_event,
        ),
    )
    p2 = ctx.Process(
        target=_write_usage_event_in_process,
        args=(
            str(path),
            "req-mp-2",
            "sampling",
            "route=sampling.asample",
            5,
            datetime(2026, 3, 12, 10, 0, 1, tzinfo=timezone.utc).isoformat(),
            start_event,
        ),
    )

    p1.start()
    p2.start()
    start_event.set()
    p1.join(timeout=10)
    p2.join(timeout=10)

    assert p1.exitcode == 0
    assert p2.exitcode == 0

    store = usage_store_module.JsonlUsageStore(path=path)
    logs, count, _ = asyncio.run(store.query_logs(account_id="aaaaaaaaaaaaaaaaaaaaaaaa", limit=10, offset=0))

    assert count == 2
    assert sorted(log["source_index"] for log in logs) == [1, 2]
    assert sorted(log["request_id"] for log in logs) == ["req-mp-1", "req-mp-2"]
