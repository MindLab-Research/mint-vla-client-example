from __future__ import annotations

import asyncio

import pytest

from tinker_server.routes import internal as internal_routes


@pytest.mark.anyio
async def test_issue_593_internal_model_visibility_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    import tinker_server.backend.model_actor_supervisor as supervisor_module
    import tinker_server.backend.model_work_scheduler as scheduler_module

    class _FakeScheduler:
        async def stats(self, *, timeout_s: float = 10.0) -> dict:
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


def test_issue_593_internal_metrics_exports_scheduler_and_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_admission_stats(*, include_actor_rss: bool = True) -> dict:
        assert include_actor_rss is False
        return {
            "capacity": {},
            "work_queue": {},
            "task_state_futures": {},
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
    assert 'mint_model_actor_supervisor_domain_healthy{domain_key="vllm:model-a"} 1' in body
    assert (
        'mint_model_actor_supervisor_replica_state{actor_name="actor-a",domain_key="vllm:model-a",'
        'replica_id="replica-0",state="healthy"} 1'
    ) in body
    assert (
        'mint_model_actor_supervisor_replica_generation{actor_name="actor-a",domain_key="vllm:model-a",'
        'replica_id="replica-0",state="healthy"} 42'
    ) in body
