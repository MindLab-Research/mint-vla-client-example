from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _ready_billing_outbox_health_snapshot() -> dict:
    return {"status": "ready"}


async def _ready_future_state_store_health_snapshot() -> dict:
    return {"status": "ready"}


def _ready_cached_health_snapshot() -> dict:
    return {"status": "ready"}


@pytest.mark.anyio
async def test_public_healthz_uses_ping_cache_and_minimal_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.scheduling.model_work_scheduler as scheduler_module
    import mint_server.backend.stores.task_state_store as task_state_module
    from mint_server import health_checks

    health_checks.reset_public_healthz_cache()
    calls = {"scheduler": 0, "store": 0}

    class _Scheduler:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls["scheduler"] += 1
            return {"ok": True, "detail": "not-public"}

    class _Store:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls["store"] += 1
            return {"ok": True, "detail": "not-public"}

    class _TaskFutures:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            raise AssertionError("public healthz must not ping FutureStateStore")

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", _Scheduler())
    monkeypatch.setattr(task_state_module, "task_state_store", _Store())
    monkeypatch.setattr(task_state_module, "task_futures", _TaskFutures())
    recorded: list[str] = []
    monkeypatch.setattr(
        health_checks,
        "_record_public_healthz_refresh_metric",
        lambda result: recorded.append(result),
    )

    assert await health_checks.public_business_healthz_response() == {"status": "ready"}
    assert await health_checks.public_business_healthz_response() == {"status": "ready"}
    assert calls == {"scheduler": 1, "store": 1}
    assert recorded == ["ready"]
    assert health_checks.public_healthz_cache_age_seconds() is not None

    health_checks.reset_public_healthz_cache()


@pytest.mark.anyio
async def test_public_healthz_failed_refresh_does_not_reuse_old_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.scheduling.model_work_scheduler as scheduler_module
    import mint_server.backend.stores.task_state_store as task_state_module
    from mint_server import health_checks

    health_checks.reset_public_healthz_cache()
    fail = False

    class _Scheduler:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            if fail:
                raise RuntimeError("scheduler down")
            return {"ok": True}

    class _Store:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            return {"ok": True}

    monkeypatch.setattr(health_checks, "PUBLIC_HEALTHZ_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(scheduler_module, "model_work_scheduler", _Scheduler())
    monkeypatch.setattr(task_state_module, "task_state_store", _Store())
    recorded: list[str] = []
    monkeypatch.setattr(
        health_checks,
        "_record_public_healthz_refresh_metric",
        lambda result: recorded.append(result),
    )

    assert await health_checks.public_business_healthz_response() == {"status": "ready"}
    fail = True
    response = await health_checks.public_business_healthz_response()
    assert response.status_code == 503
    assert response.body == b'{"status":"unhealthy"}'
    assert recorded == ["ready", "unhealthy"]
    assert health_checks.public_healthz_cache_age_seconds() is None

    health_checks.reset_public_healthz_cache()


@pytest.mark.anyio
async def test_public_healthz_records_timeout_result(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import mint_server.backend.scheduling.model_work_scheduler as scheduler_module
    import mint_server.backend.stores.task_state_store as task_state_module
    from mint_server import health_checks

    health_checks.reset_public_healthz_cache()

    class _Scheduler:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            await asyncio.sleep(0.05)
            return {"ok": True}

    class _Store:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            return {"ok": True}

    recorded: list[str] = []
    monkeypatch.setattr(health_checks, "PUBLIC_HEALTHZ_REFRESH_TIMEOUT_S", 0.001)
    monkeypatch.setattr(health_checks, "PUBLIC_HEALTHZ_COMPONENT_TIMEOUT_S", 0.001)
    monkeypatch.setattr(scheduler_module, "model_work_scheduler", _Scheduler())
    monkeypatch.setattr(task_state_module, "task_state_store", _Store())
    monkeypatch.setattr(
        health_checks,
        "_record_public_healthz_refresh_metric",
        lambda result: recorded.append(result),
    )

    response = await health_checks.public_business_healthz_response()

    assert response.status_code == 503
    assert recorded == ["timeout"]
    assert health_checks.public_healthz_cache_age_seconds() is None

    health_checks.reset_public_healthz_cache()


@pytest.mark.anyio
async def test_public_healthz_single_flight_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import mint_server.backend.scheduling.model_work_scheduler as scheduler_module
    import mint_server.backend.stores.task_state_store as task_state_module
    from mint_server import health_checks

    health_checks.reset_public_healthz_cache()
    calls = {"scheduler": 0, "store": 0}

    class _Scheduler:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls["scheduler"] += 1
            await asyncio.sleep(0)
            return {"ok": True}

    class _Store:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            calls["store"] += 1
            await asyncio.sleep(0)
            return {"ok": True}

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", _Scheduler())
    monkeypatch.setattr(task_state_module, "task_state_store", _Store())

    results = await asyncio.gather(*(health_checks.public_business_healthz_response() for _ in range(8)))
    assert results == [{"status": "ready"}] * 8
    assert calls == {"scheduler": 1, "store": 1}

    health_checks.reset_public_healthz_cache()


@pytest.mark.anyio
async def test_internal_healthz_degrades_on_stale_supervisor_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    from mint_server import health_checks

    now = 1000.0

    class _Supervisor:
        def snapshot(self) -> dict:
            return {
                "desired_total": 1,
                "managed_total": 1,
                "last_reconcile_at": now - 61.0,
                "topology": {"observed_at": now - 61.0},
            }

    monkeypatch.setattr(health_checks.time, "time", lambda: now)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(health_checks, "_cached_maintenance_cron_snapshot", _ready_cached_health_snapshot)
    monkeypatch.setattr(health_checks, "_billing_outbox_health_snapshot", _ready_billing_outbox_health_snapshot)
    monkeypatch.setattr(
        health_checks,
        "_future_state_store_health_snapshot",
        _ready_future_state_store_health_snapshot,
    )

    out = await health_checks.internal_lightweight_healthz_response()
    assert out["status"] == "degraded"


@pytest.mark.anyio
async def test_internal_healthz_uses_top_level_supervisor_snapshot_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    from mint_server import health_checks

    now = 1000.0

    class _Supervisor:
        def snapshot(self) -> dict:
            return {
                "snapshot_generated_at": now,
                "desired_total": 1,
                "managed_total": 1,
                "last_reconcile_at": now - 600.0,
                "topology": {},
            }

    monkeypatch.setattr(health_checks.time, "time", lambda: now)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(health_checks, "_cached_maintenance_cron_snapshot", _ready_cached_health_snapshot)
    monkeypatch.setattr(health_checks, "_billing_outbox_health_snapshot", _ready_billing_outbox_health_snapshot)
    monkeypatch.setattr(
        health_checks,
        "_future_state_store_health_snapshot",
        _ready_future_state_store_health_snapshot,
    )

    out = await health_checks.internal_lightweight_healthz_response()
    assert out["status"] == "ready"
    assert out["model_actor_supervisor"]["snapshot_generated_at"] == now


@pytest.mark.anyio
async def test_internal_healthz_degrades_when_cron_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    from mint_server import health_checks

    now = 1000.0

    class _Supervisor:
        def snapshot(self) -> dict:
            return {
                "desired_total": 1,
                "managed_total": 1,
                "last_reconcile_at": now,
                "topology": {"observed_at": now},
            }

    monkeypatch.setattr(health_checks.time, "time", lambda: now)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(
        health_checks,
        "_cached_maintenance_cron_snapshot",
        lambda: {"status": "degraded", "reason": "maintenance_cron_actor_unavailable"},
    )
    monkeypatch.setattr(health_checks, "_billing_outbox_health_snapshot", _ready_billing_outbox_health_snapshot)
    monkeypatch.setattr(
        health_checks,
        "_future_state_store_health_snapshot",
        _ready_future_state_store_health_snapshot,
    )

    out = await health_checks.internal_lightweight_healthz_response()
    assert out["status"] == "degraded"


@pytest.mark.anyio
async def test_internal_healthz_degrades_on_startup_control_plane_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    from mint_server import health_checks
    from mint_server.health.health_state import clear_startup_degraded_state, set_startup_degraded_state

    now = 1000.0

    class _Supervisor:
        def snapshot(self) -> dict:
            return {
                "desired_total": 0,
                "managed_total": 0,
                "last_reconcile_at": now,
                "topology": {"observed_at": now},
            }

    monkeypatch.setattr(health_checks.time, "time", lambda: now)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(health_checks, "_cached_maintenance_cron_snapshot", _ready_cached_health_snapshot)
    monkeypatch.setattr(health_checks, "_billing_outbox_health_snapshot", _ready_billing_outbox_health_snapshot)
    monkeypatch.setattr(
        health_checks,
        "_future_state_store_health_snapshot",
        _ready_future_state_store_health_snapshot,
    )
    clear_startup_degraded_state()
    set_startup_degraded_state(
        reason="control_plane_unavailable",
        error="missing scheduler",
        details={"failures": {"model_work_scheduler": "missing"}},
    )
    try:
        out = await health_checks.internal_lightweight_healthz_response()
        assert out["status"] == "degraded"
        assert out["startup"]["reason"] == "control_plane_unavailable"
    finally:
        clear_startup_degraded_state()


@pytest.mark.anyio
async def test_internal_healthz_degrades_on_billing_outbox_backlog(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    import mint_server.backend.stores.task_state_store as task_state_module
    from mint_server import health_checks

    now = 1000.0

    class _Supervisor:
        def snapshot(self) -> dict:
            return {
                "desired_total": 0,
                "managed_total": 0,
                "last_reconcile_at": now,
                "topology": {"observed_at": now},
            }

    class _TaskFutures:
        async def async_billing_outbox_stats(self):
            return {
                "by_status": {
                    "pending": {"rows": 2, "oldest_age_s": 6.0},
                    "failed": {"rows": 0, "oldest_age_s": 0.0},
                },
                "metrics": {"flush_permanent_error": 0},
            }

    monkeypatch.setattr(health_checks.time, "time", lambda: now)
    monkeypatch.setenv("MINT_BILLING_OUTBOX_DEGRADED_ROWS", "10")
    monkeypatch.setenv("MINT_BILLING_OUTBOX_DEGRADED_AGE_S", "5")
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(task_state_module, "task_futures", _TaskFutures())
    monkeypatch.setattr(health_checks, "_cached_maintenance_cron_snapshot", _ready_cached_health_snapshot)
    monkeypatch.setattr(
        health_checks,
        "_future_state_store_health_snapshot",
        _ready_future_state_store_health_snapshot,
    )

    out = await health_checks.internal_lightweight_healthz_response()
    assert out["status"] == "degraded"
    assert out["billing_outbox"]["status"] == "degraded"
    assert "oldest_pending_age" in out["billing_outbox"]["reasons"]


@pytest.mark.anyio
async def test_internal_healthz_degrades_on_future_store_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    import mint_server.backend.stores.task_state_store as task_state_module
    from mint_server import health_checks

    now = 1000.0

    class _Supervisor:
        def snapshot(self) -> dict:
            return {
                "desired_total": 0,
                "managed_total": 0,
                "last_reconcile_at": now,
                "topology": {"observed_at": now},
            }

    class _TaskFutures:
        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            await asyncio.sleep(0.05)
            return {"ok": True}

    monkeypatch.setattr(health_checks.time, "time", lambda: now)
    monkeypatch.setattr(health_checks, "INTERNAL_HEALTHZ_FUTURE_STORE_TIMEOUT_S", 0.001)
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(task_state_module, "task_futures", _TaskFutures())
    monkeypatch.setattr(health_checks, "_cached_maintenance_cron_snapshot", _ready_cached_health_snapshot)
    monkeypatch.setattr(health_checks, "_billing_outbox_health_snapshot", _ready_billing_outbox_health_snapshot)

    out = await health_checks.internal_lightweight_healthz_response()

    assert out["status"] == "degraded"
    assert out["future_state_store"]["reason"] == "future_state_store_ping_timeout"


@pytest.mark.anyio
async def test_internal_healthz_unhealthy_when_supervisor_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.actors.model_actor_supervisor as supervisor_module
    from mint_server import health_checks

    def _raise() -> object:
        raise RuntimeError("supervisor unavailable")

    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", _raise)

    out = await health_checks.internal_lightweight_healthz_response()
    assert out["status"] == "unhealthy"
    assert "model_actor_supervisor" in out


def test_task_state_store_ping_checks_sqlite() -> None:
    from mint_server.backend.stores.task_state_store import TaskStateStore

    store = TaskStateStore.in_memory()
    try:
        assert store.ping() == {"ok": True}
    finally:
        store.close()


def test_model_work_scheduler_actor_ping() -> None:
    from mint_server.backend.scheduling.model_work_scheduler import _ModelWorkSchedulerActor

    out = _ModelWorkSchedulerActor().ping()
    assert out["ok"] is True
    assert out["scheduler_instance_id"]


def test_public_healthz_refresh_metric_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    from mint_server import logging_context

    calls: list[tuple[int, dict]] = []

    class _Counter:
        def add(self, count: int, *, attributes: dict) -> None:
            calls.append((count, attributes))

    monkeypatch.setattr(logging_context, "_OTEL_ENABLED", True)
    monkeypatch.setattr(logging_context, "_PUBLIC_HEALTHZ_REFRESH_COUNTER", _Counter())

    logging_context.record_public_healthz_refresh_metric(result="ready")

    assert calls == [(1, {"result": "ready"})]

    class _FailingCounter:
        def add(self, count: int, *, attributes: dict) -> None:
            raise RuntimeError("collector unavailable")

    monkeypatch.setattr(logging_context, "_PUBLIC_HEALTHZ_REFRESH_COUNTER", _FailingCounter())
    logging_context.record_public_healthz_refresh_metric(result="error")
