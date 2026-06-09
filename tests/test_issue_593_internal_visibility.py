from __future__ import annotations

import asyncio

import pytest

from mint_server.routes import internal as internal_routes


def test_issue_593_operator_routes_do_not_keep_api_v1_internal_aliases() -> None:
    from mint_server.app import _OTEL_EXCLUDED_PATHS, app

    paths = {getattr(route, "path", "") for route in app.routes}

    assert "/api/v1/internal/healthz" in paths
    assert "/internal/admission_stats" in paths
    assert "/internal/metrics" in paths
    assert "/api/v1/internal/admission_stats" not in paths
    assert "/api/v1/internal/metrics" not in paths
    assert "/api/v1/internal/admission_stats" not in _OTEL_EXCLUDED_PATHS
    assert "/api/v1/internal/metrics" not in _OTEL_EXCLUDED_PATHS


@pytest.mark.anyio
async def test_issue_593_internal_model_visibility_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.backend.model_work_scheduler as scheduler_module

    class _FakeScheduler:
        async def stats(self, *, timeout_s: float = 10.0, create_if_missing: bool = True) -> dict:
            assert create_if_missing is False
            return {"depth": 3, "backlog_depth": 1, "replica_queues": {}, "counters": {}}

    class _FakeSupervisor:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {"desired_total": 1, "managed_total": 1, "replicas": {}}

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", _FakeScheduler())
    monkeypatch.setattr(supervisor_module, "model_actor_supervisor", _FakeSupervisor())

    assert await internal_routes.model_work_scheduler_health() == {
        "depth": 3,
        "backlog_depth": 1,
        "replica_queues": {},
        "counters": {},
    }
    assert await internal_routes.model_actor_supervisor_health() == {
        "desired_total": 1,
        "managed_total": 1,
        "replicas": {},
    }


@pytest.mark.anyio
async def test_issue_593_internal_admission_stats_observes_without_creating(monkeypatch: pytest.MonkeyPatch) -> None:
    import mint_server.backend.maintenance_cron_actor as cron_module
    import mint_server.backend.model_actor_supervisor as supervisor_module
    import mint_server.backend.model_work_scheduler as scheduler_module
    import mint_server.backend.session_heartbeat_store as heartbeat_module
    import mint_server.backend.task_state_store as task_state_module

    class _FakeScheduler:
        async def stats(self, *, timeout_s: float = 10.0, create_if_missing: bool = True) -> dict:
            assert create_if_missing is False
            return {"depth": 0, "backlog_depth": 0, "replica_queues": {}, "counters": {}}

    class _FakeTaskFutures:
        async def async_stats(self) -> dict:
            return {
                "backend": "fake",
                "task_state_rpc": {"total": 1.0, "error": 0.0, "inflight": 0.0, "by_method": {}},
                "task_state_stats": {"calls": 1.0},
            }

        async def async_ensure_ready(self, *, timeout_s: float = 10.0, create_if_missing: bool = True) -> dict:
            raise AssertionError("admission_stats should collect TaskStateStore stats, not ping readiness")

        async def async_ping(self, *, timeout_s: float = 5.0) -> dict:
            return {"ok": True}

        async def async_rss_bytes(self, *, timeout_s: float = 10.0) -> int:
            return 123

    class _FakeSupervisor:
        async def async_snapshot(self, *, timeout_s: float = 10.0) -> dict:
            return {"desired_total": 0, "managed_total": 0, "replicas": {}}

        def rss_snapshot(self, *, timeout_s: float = 10.0) -> list:
            return []

        def metadata_cache_metrics_snapshot(self) -> dict:
            return {}

        def lifecycle_metrics_snapshot(self) -> dict:
            return {}

    class _FakeCron:
        async def async_health_snapshot(self, *, timeout_s: float = 10.0, create_if_missing: bool = True) -> dict:
            assert create_if_missing is False
            return {"actor_name": "mint_maintenance_cron"}

    class _FakeHeartbeatStore:
        async def async_size(self, *, create_if_missing: bool = True) -> int:
            assert create_if_missing is False
            return 0

    monkeypatch.setattr(scheduler_module, "model_work_scheduler", _FakeScheduler())
    monkeypatch.setattr(task_state_module, "task_futures", _FakeTaskFutures())
    monkeypatch.setattr(heartbeat_module, "session_heartbeat_store", _FakeHeartbeatStore())
    monkeypatch.setattr(supervisor_module, "model_actor_supervisor", _FakeSupervisor())
    monkeypatch.setattr(supervisor_module, "get_model_actor_supervisor", lambda: _FakeSupervisor())
    monkeypatch.setattr(cron_module, "maintenance_cron_actor", _FakeCron())
    monkeypatch.setattr(internal_routes, "get_ray_cluster_health_snapshot", lambda: {"ok": True})
    monkeypatch.setattr(internal_routes, "get_ray_gcs_metrics_snapshot", lambda: {"up": 1})
    async def _fake_lora_load_lock_count() -> int:
        return 0

    monkeypatch.setattr("mint_server.routes.sampling._lora_load_lock_count", _fake_lora_load_lock_count)
    monkeypatch.setattr("mint_server.routes.service.session_manager", None)

    out = await internal_routes.admission_stats(include_actor_rss=True)

    assert out["model_work_scheduler"]["depth"] == 0
    assert out["task_futures"]["backend"] == "fake"
    assert out["task_futures"]["task_state_rpc"]["total"] == 1.0
    assert out["maintenance_cron_actor"]["actor_name"] == "mint_maintenance_cron"


def test_issue_593_internal_metrics_is_sentinel_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINT_INTERNAL_PROMETHEUS_METRICS_ENABLED", "1")

    async def _fake_admission_stats(*, include_actor_rss: bool = True) -> dict:
        raise AssertionError("/internal/metrics must not collect actor snapshots after OTel migration")

    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats)

    response = asyncio.run(internal_routes.metrics())
    body = response.body.decode("utf-8")

    assert "mint_metrics_up 1" in body
    assert "mint_model_work_scheduler_" not in body
    assert "mint_model_actor_supervisor_" not in body
    assert "mint_topology_node_" not in body
    assert "mint_node_metrics_daemon_" not in body


@pytest.mark.anyio
async def test_issue_593_internal_metrics_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_INTERNAL_PROMETHEUS_METRICS_ENABLED", raising=False)

    async def _fake_admission_stats(*, include_actor_rss: bool = True) -> dict:
        raise AssertionError("disabled metrics endpoint must not collect stats")

    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats)

    with pytest.raises(Exception) as exc_info:
        await internal_routes.metrics()

    assert getattr(exc_info.value, "status_code", None) == 404
