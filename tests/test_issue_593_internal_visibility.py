from __future__ import annotations

import asyncio

import pytest

from mint_server.routes import internal as internal_routes


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
        async def async_ensure_ready(self, *, timeout_s: float = 10.0, create_if_missing: bool = True) -> dict:
            assert create_if_missing is False
            return {"backend": "fake"}

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
    assert out["maintenance_cron_actor"]["actor_name"] == "mint_maintenance_cron"


def test_issue_593_internal_metrics_exports_scheduler_and_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINT_INTERNAL_PROMETHEUS_METRICS_ENABLED", "1")

    async def _fake_admission_stats(*, include_actor_rss: bool = True) -> dict:
        assert include_actor_rss is False
        return {
            "capacity": {},
            "task_futures": {},
            "actors": {},
            "model_work_scheduler": {
                "depth": 5,
                "backlog_depth": 2,
                "backlog_depth_by_domain": {"vllm:model-a": 2},
                "replica_queues": {
                    "vllm:model-a::replica-0": {
                        "domain_key": "vllm:model-a",
                        "replica_id": "replica-0",
                        "depth": 3,
                        "status": "healthy",
                    }
                },
                "leases": [{"lease_id": "lease-a"}],
                "counters": {
                    "appended": 7,
                    "assigned": 5,
                    "claimed": 4,
                    "completed": 3,
                    "failed": 1,
                    "requeued": 2,
                },
            },
            "model_actor_supervisor": {
                "desired_total": 1,
                "managed_total": 1,
                "domain_total": 1,
                "reconcile_total": 8,
                "created_total": 1,
                "restarted_total": 1,
                "blocked_total": 0,
                "busy_recycle_skipped_total": 0,
                "scheduler_sync_failures_total": 0,
                "topology_reconcile_failures_total": 0,
                "node_metrics_created_total": 1,
                "node_metrics_reconcile_failures_total": 0,
                "topology": {
                    "nodes": {
                        "mint-worker-0": {
                            "state": "ready",
                            "provider": "volcano",
                            "gpu_count": 8,
                        }
                    }
                },
                "daemons": {
                    "node_metrics": {
                        "enabled": True,
                        "desired_total": 1,
                        "managed_total": 1,
                        "nodes": {
                            "mint-worker-0": {
                                "state": "healthy",
                                "health": {
                                    "sample_count": 3,
                                    "error_count": 0,
                                    "last_sample": {
                                        "worker_alias": "mint-worker-0",
                                        "deployment_env": "prod",
                                        "cluster_id": "volcano",
                                        "load_1m": 1.0,
                                        "load_5m": 2.0,
                                        "load_15m": 3.0,
                                        "cpu_utilization_ratio": 0.5,
                                        "memory_used_bytes": 1024,
                                        "memory_total_bytes": 2048,
                                        "disk_used_bytes": 4096,
                                        "disk_total_bytes": 8192,
                                        "sample_duration_ms": 12.5,
                                        "sampled_at": 1000.0,
                                        "gpus": [
                                            {
                                                "gpu_uuid": "GPU-test",
                                                "utilization_gpu_percent": 77,
                                                "memory_used_bytes": 512,
                                                "memory_total_bytes": 1024,
                                                "power_draw_watts": 300,
                                                "power_limit_watts": 400,
                                                "temperature_celsius": 61,
                                                "sm_clock_mhz": 1200,
                                                "memory_clock_mhz": 1500,
                                                "pcie_link_gen": 4,
                                                "pcie_link_width": 16,
                                                "processes": [
                                                    {
                                                        "process_class": "vllm",
                                                        "process_count": 2,
                                                        "memory_used_bytes": 256,
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                },
                            }
                        },
                    }
                },
                "domains": {"vllm:model-a": {"replicas": 1, "healthy": 1, "unhealthy": 0}},
                "replicas": {
                    "vllm:model-a::replica-0": {
                        "domain_key": "vllm:model-a",
                        "replica_id": "replica-0",
                        "actor_name": "actor-a",
                        "state": "healthy",
                        "generation": 42,
                    }
                },
            },
        }

    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats)

    response = asyncio.run(internal_routes.metrics())
    body = response.body.decode("utf-8")

    assert "mint_model_work_scheduler_depth 5" in body
    assert 'mint_model_work_scheduler_domain_backlog_depth{domain_key="vllm:model-a"} 2' in body
    assert (
        'mint_model_work_scheduler_replica_queue_depth{domain_key="vllm:model-a",'
        'queue_id="vllm:model-a::replica-0",replica_id="replica-0",status="healthy"} 3'
    ) in body
    assert "mint_model_work_scheduler_leases 1" in body
    assert "mint_model_actor_supervisor_desired_total 1" in body
    assert "mint_model_actor_supervisor_node_metrics_created_total 1" in body
    assert (
        'mint_topology_node_state{provider="volcano",state="ready",worker_alias="mint-worker-0"} 1'
        in body
    )
    assert (
        'mint_node_metrics_daemon_state{state="healthy",worker_alias="mint-worker-0"} 1'
        in body
    )
    assert (
        'mint_node_metrics_daemon_sample_count{state="healthy",worker_alias="mint-worker-0"} 3'
        in body
    )
    assert (
        'mint_node_gpu_memory_total_bytes{cluster_id="volcano",deployment_env="prod",'
        'gpu_uuid="GPU-test",worker_alias="mint-worker-0"} 1024'
    ) in body
    assert (
        'mint_node_gpu_processes{cluster_id="volcano",deployment_env="prod",'
        'gpu_uuid="GPU-test",process_class="vllm",worker_alias="mint-worker-0"} 2'
    ) in body
    assert (
        'mint_node_cpu_utilization_ratio{cluster_id="volcano",deployment_env="prod",'
        'worker_alias="mint-worker-0"} 0.5'
    ) in body
    assert 'mint_model_actor_supervisor_domain_healthy{domain_key="vllm:model-a"} 1' in body
    assert (
        'mint_model_actor_supervisor_replica_state{actor_name="actor-a",domain_key="vllm:model-a",'
        'replica_id="replica-0",state="healthy"} 1'
    ) in body
    assert (
        'mint_model_actor_supervisor_replica_generation{actor_name="actor-a",domain_key="vllm:model-a",'
        'replica_id="replica-0",state="healthy"} 42'
    ) in body


@pytest.mark.anyio
async def test_issue_593_internal_metrics_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINT_INTERNAL_PROMETHEUS_METRICS_ENABLED", raising=False)

    async def _fake_admission_stats(*, include_actor_rss: bool = True) -> dict:
        raise AssertionError("disabled metrics endpoint must not collect stats")

    monkeypatch.setattr(internal_routes, "admission_stats", _fake_admission_stats)

    with pytest.raises(Exception) as exc_info:
        await internal_routes.metrics()

    assert getattr(exc_info.value, "status_code", None) == 404
