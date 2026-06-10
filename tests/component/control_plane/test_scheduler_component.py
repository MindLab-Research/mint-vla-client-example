from __future__ import annotations

import asyncio

import pytest

from mint_server.backend.task_state_store import FutureStatus

from .harness import SchedulerComponentWorld
from .invariants import assert_terminal_not_scheduled


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_happy_path_reaches_retrieve_future(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-happy")

        await world.runtime_once()

        assert await world.future_service.async_get_status("component-happy") == FutureStatus.DONE
        status_code, payload = await world.retrieve("component-happy", monkeypatch)
        assert status_code == 200
        assert payload == {"ok": True, "request_id": "component-happy"}
        await assert_terminal_not_scheduled(world, "component-happy")
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_pending_uses_real_future_service(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-pending")

        status_code, payload = await world.retrieve("component-pending", monkeypatch)

        assert status_code == 408
        assert payload["request_id"] == "component-pending"
        assert payload["type"] == "try_again"
        assert payload["status"] == "queued"
        assert payload["queue_kind"] == "model_work_scheduler"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_supervisor_syncs_real_scheduler(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        supervisor = world.supervisor()

        out = await supervisor.reconcile_once()
        replica = out["snapshot"]["replicas"][f"{world.domain_key}::{world.replica_id}"]
        generation = int(replica["generation"])
        await world.enqueue_sampling("component-supervisor")
        claimed = await world.scheduler.claim_from_replica_queue(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=replica["consumer_id"],
            consumer_generation=generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert out["ok"] is True
        assert len(claimed["leases"]) == 1
        assert claimed["leases"][0]["item"]["request_id"] == "component-supervisor"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_claim_task_does_not_block_stats(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-claim")
        block = world.faults.block("task_state.claim_task")

        claim_task = asyncio.create_task(
            world.scheduler.claim_from_replica_queue(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        stats = await asyncio.wait_for(world.scheduler.stats(), timeout=0.5)

        block.release.set()
        claimed = await claim_task
        assert stats["scheduler_instance_id"]
        assert len(claimed["leases"]) == 1
    finally:
        world.close()
