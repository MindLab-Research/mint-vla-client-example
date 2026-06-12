from __future__ import annotations

from typing import Any, cast

import pytest

from mint_server.backend.cluster_placement_controller import ClusterPlacementController

from .harness import SchedulerComponentWorld


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_supervisor_start_failure_removes_claimable_replica(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:

        class _FailingRuntime:
            def __init__(self, spec: Any, generation: int) -> None:
                self.actor_name = spec.normalized_actor_name()
                self.generation = int(generation)
                self.shutdown_calls = 0

            def start(self) -> dict[str, Any]:
                raise RuntimeError("synthetic runtime start failure")

            def shutdown(self) -> dict[str, Any]:
                self.shutdown_calls += 1
                return {"ok": True}

            def health_snapshot(self) -> dict[str, Any]:
                return {
                    "running": False,
                    "actor_generation": self.generation,
                    "last_error": "synthetic runtime start failure",
                }

        async def _factory(spec: Any, generation: int) -> _FailingRuntime:
            return _FailingRuntime(spec, generation)

        supervisor = world.supervisor_with_factory(_factory)
        out = await supervisor.reconcile_once()
        await world.enqueue_sampling("component-supervisor-start-failure", assign=False)

        with pytest.raises(Exception, match="not claimable|unknown replica"):
            await world.scheduler.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            )

        replica = out["snapshot"]["replicas"][f"{world.domain_key}::{world.replica_id}"]
        assert out["ok"] is True
        assert replica["state"] == "dead"
        assert "synthetic runtime start failure" in replica["last_error"]
        observed = cast(Any, await world.observe_scheduler("component-supervisor-start-failure"))
        assert observed.location == "backlog"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_supervisor_generation_restart_allows_only_new_consumer(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:

        class _Runtime:
            def __init__(self, spec: Any, generation: int) -> None:
                self.actor_name = spec.normalized_actor_name()
                self.generation = int(generation)
                self.running = True
                self.fail_health = False

            def start(self) -> dict[str, Any]:
                return {"running": True}

            def shutdown(self) -> dict[str, Any]:
                self.running = False
                return {"ok": True}

            def health_snapshot(self) -> dict[str, Any]:
                if self.fail_health:
                    raise RuntimeError("synthetic runtime health failure")
                return {
                    "running": self.running,
                    "actor_generation": self.generation,
                    "domain_key": world.domain_key,
                    "replica_id": world.replica_id,
                }

        runtimes: list[_Runtime] = []

        async def _factory(spec: Any, generation: int) -> _Runtime:
            runtime = _Runtime(spec, generation)
            runtimes.append(runtime)
            return runtime

        supervisor = world.supervisor_with_factory(_factory)
        first = await supervisor.reconcile_once()
        old_replica = first["snapshot"]["replicas"][f"{world.domain_key}::{world.replica_id}"]
        old_generation = int(old_replica["generation"])
        old_consumer_id = str(old_replica["consumer_id"])
        await world.enqueue_sampling("component-supervisor-generation", assign=False)

        runtimes[-1].fail_health = True
        second = await supervisor.reconcile_once()
        new_replica = second["snapshot"]["replicas"][f"{world.domain_key}::{world.replica_id}"]
        new_generation = int(new_replica["generation"])
        new_consumer_id = str(new_replica["consumer_id"])

        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.scheduler.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=old_consumer_id,
                consumer_generation=old_generation,
                max_items=1,
                lease_ttl_s=30.0,
            )
        claimed = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=new_consumer_id,
            consumer_generation=new_generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert first["ok"] is True
        assert second["ok"] is True
        assert new_generation > old_generation
        assert len(runtimes) == 2
        assert [lease["item"]["request_id"] for lease in claimed.leases] == [
            "component-supervisor-generation"
        ]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_placement_pg_blocked_registers_unclaimable_replica(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        async def _factory(spec: Any, generation: int) -> Any:
            raise AssertionError("runtime factory must not run while placement PG is blocked")

        placement = ClusterPlacementController(
            observed_free_gpus_by_node=lambda: {},
            initial_backoff_s=0.1,
            max_backoff_s=0.1,
        )
        supervisor = world.supervisor_with_factory(_factory)
        supervisor._placement_controller = placement

        out = await supervisor.reconcile_once()
        await world.enqueue_sampling("component-placement-pg-blocked", assign=False)

        with pytest.raises(Exception, match="not claimable|unknown replica"):
            await world.scheduler.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            )

        replica = out["snapshot"]["replicas"][f"{world.domain_key}::{world.replica_id}"]
        observed = cast(Any, await world.observe_scheduler("component-placement-pg-blocked"))
        stats = await world.scheduler.stats()

        assert out["ok"] is True
        assert replica["state"] == "blocked"
        assert replica["scheduler_status"] == "blocked"
        assert "placement group blocked" in replica["last_error"]
        replica_stats = [
            replica
            for replica in stats["replicas"]
            if replica["domain_key"] == world.domain_key and replica["replica_id"] == world.replica_id
        ]

        assert observed.location == "backlog"
        assert len(replica_stats) == 1
        assert replica_stats[0]["status"] == "blocked"
    finally:
        world.close()
