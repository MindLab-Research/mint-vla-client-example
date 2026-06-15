from __future__ import annotations

import time
from typing import Any, cast

import pytest

from mint_server.backend.cluster_placement_controller import ClusterPlacementController
from mint_server.backend.engine_adapter import EngineHealth, EngineHealthStatus
from mint_server.backend.engine_liveness import EngineLivenessPush

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
            await world.runtime_queue.claim(
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
async def test_scheduler_component_started_runtime_requires_fresh_liveness_before_claim(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:

        class _Runtime:
            def __init__(self, spec: Any, generation: int) -> None:
                self.actor_name = spec.normalized_actor_name()
                self.generation = int(generation)

            def start(self) -> dict[str, Any]:
                return {"running": True}

        async def _factory(spec: Any, generation: int) -> _Runtime:
            return _Runtime(spec, generation)

        supervisor = world.supervisor_with_factory(_factory)
        first = await supervisor.reconcile_once()
        replica_key = f"{world.domain_key}::{world.replica_id}"
        replica = first["snapshot"]["replicas"][replica_key]
        generation = int(replica["generation"])
        consumer_id = str(replica["consumer_id"])

        await world.enqueue_sampling("component-started-runtime-awaits-liveness", assign=False)

        with pytest.raises(Exception, match="not claimable"):
            await world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=consumer_id,
                consumer_generation=generation,
                max_items=1,
                lease_ttl_s=30.0,
            )

        await supervisor.push_liveness(
            EngineLivenessPush(
                actor_name=str(replica["actor_name"]),
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=consumer_id,
                actor_generation=generation,
                running=True,
                engine_ready=True,
                engine_health=EngineHealth(status=EngineHealthStatus.READY),
                pushed_at=time.time(),
            )
        )
        await supervisor.reconcile_once()
        claimed = await world.runtime_queue.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=consumer_id,
            consumer_generation=generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert replica["state"] == "starting"
        assert [lease["item"]["request_id"] for lease in claimed.leases] == [
            "component-started-runtime-awaits-liveness"
        ]
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
        await supervisor.push_liveness(
            EngineLivenessPush(
                actor_name=str(old_replica["actor_name"]),
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=old_consumer_id,
                actor_generation=old_generation,
                running=True,
                engine_ready=True,
                engine_health=EngineHealth(status=EngineHealthStatus.READY),
                pushed_at=time.time(),
            )
        )
        await supervisor.reconcile_once()
        await world.enqueue_sampling("component-supervisor-generation", assign=False)

        await supervisor.push_liveness(
            EngineLivenessPush(
                actor_name=str(old_replica["actor_name"]),
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=old_consumer_id,
                actor_generation=old_generation,
                running=True,
                engine_ready=True,
                engine_health=EngineHealth(status=EngineHealthStatus.READY),
                pushed_at=time.time() - 120.0,
            )
        )
        second = await supervisor.reconcile_once()
        new_replica = second["snapshot"]["replicas"][f"{world.domain_key}::{world.replica_id}"]
        new_generation = int(new_replica["generation"])
        new_consumer_id = str(new_replica["consumer_id"])
        await supervisor.push_liveness(
            EngineLivenessPush(
                actor_name=str(new_replica["actor_name"]),
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=new_consumer_id,
                actor_generation=new_generation,
                running=True,
                engine_ready=True,
                engine_health=EngineHealth(status=EngineHealthStatus.READY),
                pushed_at=time.time(),
            )
        )
        await supervisor.reconcile_once()

        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=old_consumer_id,
                consumer_generation=old_generation,
                max_items=1,
                lease_ttl_s=30.0,
            )
        claimed = await world.runtime_queue.claim(
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
            await world.runtime_queue.claim(
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


@pytest.mark.anyio
async def test_scheduler_component_liveness_unhealthy_push_registers_unclaimable_replica(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        supervisor = world.supervisor()
        first = await supervisor.reconcile_once()
        replica_key = f"{world.domain_key}::{world.replica_id}"
        healthy_replica = first["snapshot"]["replicas"][replica_key]
        generation = int(healthy_replica["generation"])
        consumer_id = str(healthy_replica["consumer_id"])
        await supervisor.push_liveness(
            EngineLivenessPush(
                actor_name=str(healthy_replica["actor_name"]),
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=consumer_id,
                actor_generation=generation,
                running=True,
                engine_ready=True,
                engine_health=EngineHealth(status=EngineHealthStatus.READY),
                pushed_at=time.time(),
            )
        )
        await supervisor.reconcile_once()

        await world.enqueue_sampling("component-liveness-before-unhealthy")
        healthy_claim = await world.runtime_queue.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=consumer_id,
            consumer_generation=generation,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert [lease["item"]["request_id"] for lease in healthy_claim.leases] == [
            "component-liveness-before-unhealthy"
        ]
        active_lease = healthy_claim.leases[0]

        pushed = await supervisor.push_liveness(
            EngineLivenessPush(
                actor_name=str(healthy_replica["actor_name"]),
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=consumer_id,
                actor_generation=generation,
                running=True,
                engine_ready=False,
                engine_health=EngineHealth(
                    status=EngineHealthStatus.UNHEALTHY,
                    reason="synthetic engine unhealthy",
                ),
                active_request_id=str(active_lease["item"]["request_id"]),
                active_lease_id=str(active_lease["lease_id"]),
                active_lease_count=1,
                pushed_at=time.time(),
                last_error="synthetic engine unhealthy",
            )
        )
        after_push = await supervisor.reconcile_once()
        await world.enqueue_sampling("component-liveness-after-unhealthy", assign=False)

        with pytest.raises(Exception, match="not claimable|unknown replica"):
            await world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=consumer_id,
                consumer_generation=generation,
                max_items=1,
                lease_ttl_s=30.0,
            )

        unhealthy_replica = after_push["snapshot"]["replicas"][replica_key]
        observed = cast(Any, await world.observe_scheduler("component-liveness-after-unhealthy"))
        stats = await world.scheduler.stats()
        replica_stats = [
            replica
            for replica in stats["replicas"]
            if replica["domain_key"] == world.domain_key and replica["replica_id"] == world.replica_id
        ]

        assert pushed["ok"] is True
        assert after_push["ok"] is True
        assert unhealthy_replica["state"] == "unhealthy"
        assert unhealthy_replica["scheduler_status"] == "unhealthy"
        assert unhealthy_replica["last_error"] == "synthetic engine unhealthy"
        assert observed.location == "backlog"
        assert len(replica_stats) == 1
        assert replica_stats[0]["status"] == "unhealthy"
    finally:
        world.close()
