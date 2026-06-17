import logging

import pytest

from mint_server import usage_store as usage_store_module
from mint_server.billing.usage_store import DisabledUsageStore, PostgresUsageStore, UsageEvent


def test_usage_event_defaults_include_empty_event_id():
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
    assert event.event_id == ""
    assert event.event_time is not None


def test_build_event_id_is_stable_for_same_logical_event():
    event = UsageEvent(
        account_id="aaaaaaaaaaaaaaaaaaaaaaaa",
        apikey_id="bbbbbbbbbbbbbbbbbbbbbbbb",
        charge_item="sampling",
        quantity=10,
        request_id="req-123",
        label="model=x,route=sampling.asample,dimension=prefill",
    )
    changed_quantity = UsageEvent(
        account_id=event.account_id,
        apikey_id=event.apikey_id,
        charge_item=event.charge_item,
        quantity=999,
        request_id=event.request_id,
        label=event.label,
    )
    different_label = UsageEvent(
        account_id=event.account_id,
        apikey_id=event.apikey_id,
        charge_item=event.charge_item,
        quantity=event.quantity,
        request_id=event.request_id,
        label="model=x,route=sampling.asample,dimension=sample",
    )

    assert PostgresUsageStore.build_event_id(event) == PostgresUsageStore.build_event_id(changed_quantity)
    assert PostgresUsageStore.build_event_id(event) != PostgresUsageStore.build_event_id(different_label)


def test_build_usage_store_requires_pg_dsn(monkeypatch):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_host", "")

    with pytest.raises(ValueError, match="MINT_USAGE_PG_DSN or MINT_USAGE_PG_HOST is required"):
        usage_store_module._build_usage_store()


def test_build_usage_store_rejects_non_postgres_backend(monkeypatch):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "jsonl")

    with pytest.raises(ValueError, match="Unsupported usage backend 'jsonl'"):
        usage_store_module._build_usage_store()


def test_build_usage_store_returns_disabled_store(monkeypatch):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "disabled")

    store = usage_store_module._build_usage_store()

    assert isinstance(store, DisabledUsageStore)


def test_build_usage_store_returns_postgres_store(monkeypatch):
    monkeypatch.setattr(usage_store_module.config, "usage_backend", "postgres")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_dsn", "postgresql://fake")
    monkeypatch.setattr(usage_store_module.config, "usage_pg_pool_min", 1)
    monkeypatch.setattr(usage_store_module.config, "usage_pg_pool_max", 2)
    monkeypatch.setattr(usage_store_module.config, "usage_write_timeout_ms", 1000)
    monkeypatch.setattr(usage_store_module.config, "usage_pg_table", "usage_event")

    store = usage_store_module._build_usage_store()

    assert isinstance(store, PostgresUsageStore)
    assert store._table == "public.usage_event"


def test_postgres_usage_store_warns_when_outbox_config_is_supplied(caplog):
    caplog.set_level(logging.WARNING, logger="mint_server.usage_store")
    PostgresUsageStore(
        dsn="postgresql://fake",
        outbox_path="/tmp/usage.jsonl",
        outbox_flush_interval_s=1.0,
    )

    assert "TaskStateStore owns billing outbox flushing" in caplog.text
