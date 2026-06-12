from __future__ import annotations

import asyncio
import time

import pytest

from mint_server.backend.task_state_store import TaskStateStore

from .helpers import (
    assert_scheduler_surfaces_progress_while_blocked,
    assert_stats_progress_while_blocked,
    token,
    wait_for_task_state_call_count,
)
from .harness import SchedulerComponentWorld
from .invariants import (
    assert_every_terminal_has_payload_ref,
    assert_lease_consistency,
    assert_no_double_lease,
    assert_no_orphan_assigned,
    assert_terminal_not_scheduled,
)


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_invariant_helpers_cover_happy_path(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-invariants")

        lease = await world.claim_one()
        await assert_no_double_lease(world)
        await assert_lease_consistency(world)
        await assert_no_orphan_assigned(world)

        begin = await world.scheduler.begin_finalize(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            staged_payload_path=str(world.tmp_path / "component-invariants.json"),
            finalize_ttl_s=30.0,
        )
        assert begin.ok is True
        finished = await world.scheduler.finish_success(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            result_path=str(world.tmp_path / "component-invariants.json"),
            result_checksum="checksum",
            result_size_bytes=17,
        )
        assert finished.ok is True

        await assert_every_terminal_has_payload_ref(world)
        await assert_terminal_not_scheduled(world, "component-invariants")
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_owner_renew_rpc_does_not_hold_owner_lock(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        baseline_renew_calls = sum(
            1 for method, _ in world.task_state.calls if method == "renew_scheduler_owner"
        )
        block = world.faults.block("task_state.renew_scheduler_owner")

        first = asyncio.create_task(world.scheduler_actor._ensure_task_state_owner())
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        await wait_for_task_state_call_count(
            world,
            "renew_scheduler_owner",
            count=baseline_renew_calls + 1,
        )

        second = asyncio.create_task(world.scheduler_actor._ensure_task_state_owner())
        await wait_for_task_state_call_count(
            world,
            "renew_scheduler_owner",
            count=baseline_renew_calls + 2,
        )

        block.release.set()
        first_epoch, second_epoch = await asyncio.gather(first, second)
        assert first_epoch == second_epoch == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_complete_defers_while_begin_finalize_is_inflight(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-complete-during-finalize")
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize")
        finalize_task = asyncio.create_task(
            world.scheduler.begin_finalize(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        completed = await world.scheduler.complete(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        block.release.set()
        finalized = await finalize_task

        assert completed.ok is False and completed.reason == "finalize_inflight"
        assert finalized.ok is True
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_task("component-complete-during-finalize"))["status"] == "finalizing"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_stale_complete_cannot_clear_new_attempt_projection(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-stale-complete-new-attempt"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one(lease_ttl_s=1.0)
        begin = await world.scheduler.begin_finalize(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )
        committed = await world.task_state.async_commit_finalize_failure(
            request_id=request_id,
            lease_id=old_lease["lease_id"],
            attempt_id=old_lease["attempt_id"],
            scheduler_epoch=old_lease["scheduler_epoch"],
            runtime_generation=world.generation,
            error="old attempt failed",
        )

        block = world.faults.block("task_state.get_task.after")
        complete_task = asyncio.create_task(
            world.scheduler.complete(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        expired = await world.scheduler.expire(now=time.time() + 31.0)
        await world.task_state.async_forget_task(request_id=request_id)
        await world.enqueue_sampling(request_id)
        new_lease = await world.claim_one()

        block.release.set()
        stale_complete = await complete_task

        assert begin.ok is True
        assert committed.ok is True
        assert expired.ok is True and expired.expired == 0
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
        assert stale_complete.ok is False and stale_complete.reason == "stale_consumer"
        assert (
            await world.scheduler.validate(
                lease=token(new_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"
        assert (await world.observe_task(request_id))["lease_id"] == new_lease["lease_id"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_fail_defers_while_begin_finalize_is_inflight(tmp_path) -> None:
    for requeue in (True, False):
        world = SchedulerComponentWorld(tmp_path / str(requeue))
        try:
            await world.start()
            request_id = f"component-fail-during-finalize-{requeue}"
            await world.enqueue_sampling(request_id)
            lease = await world.claim_one()

            block = world.faults.block("task_state.begin_finalize")
            finalize_task = asyncio.create_task(
                world.scheduler.begin_finalize(
                    lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                    finalize_ttl_s=30.0,
                )
            )
            await asyncio.wait_for(block.entered.wait(), timeout=1.0)

            failed = await world.scheduler.fail(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=requeue,
                reason="fail-during-finalize",
            )

            block.release.set()
            finalized = await finalize_task

            assert failed.ok is False and failed.reason == "finalize_inflight"
            assert finalized.ok is True
            assert (
                await world.scheduler.validate(
                    lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                )
            ).ok is True
            assert (await world.observe_task(request_id))["status"] == "finalizing"
        finally:
            world.close()


@pytest.mark.anyio
async def test_scheduler_component_old_generation_cannot_claim_after_replica_sync(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        old_consumer_id = world.consumer_id
        old_generation = world.generation
        await world.scheduler.sync_replicas([world.replica(status="healthy", generation=old_generation + 1)])
        await world.enqueue_sampling("component-old-generation")

        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.scheduler.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=old_consumer_id,
                consumer_generation=old_generation,
                max_items=1,
                lease_ttl_s=30.0,
            )

        new_consumer_id = world.replica(generation=old_generation + 1)["consumer_id"]
        claimed = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=new_consumer_id,
            consumer_generation=old_generation + 1,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert len(claimed.leases) == 1
        assert claimed.leases[0]["item"]["request_id"] == "component-old-generation"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_pending_during_blocked_finalize_does_not_block(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-finalize-retrieve-race")
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize")
        finalize_task = asyncio.create_task(
            world.scheduler.begin_finalize(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        status_code, payload = await asyncio.wait_for(
            world.retrieve("component-finalize-retrieve-race", monkeypatch),
            timeout=0.5,
        )
        stats = await asyncio.wait_for(world.scheduler.stats(), timeout=0.5)

        block.release.set()
        finalized = await finalize_task

        assert status_code == 408
        assert payload["request_id"] == "component-finalize-retrieve-race"
        assert stats["scheduler_instance_id"]
        assert finalized.ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_begin_finalize_does_not_block_stats(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-finalize")
        lease = await world.claim_one()

        finalized = await assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.begin_finalize(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            ),
            "task_state.begin_finalize",
        )

        assert finalized.ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_begin_finalize_does_not_block_scheduler_surfaces(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-progress-probe")
        lease = await world.claim_one()

        finalized = await assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.begin_finalize(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            ),
            "task_state.begin_finalize",
        )

        assert finalized.ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sync_defers_while_begin_finalize_is_inflight(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-sync-during-finalize")
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize")
        finalize_task = asyncio.create_task(
            world.scheduler.begin_finalize(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        synced = await asyncio.wait_for(
            world.scheduler.sync_replicas([world.replica(status="unhealthy")]),
            timeout=0.5,
        )

        block.release.set()
        finalized = await finalize_task

        assert synced.extra["deferred"] == "inflight_scheduler_transition"
        assert finalized.ok is True
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        record = await world.observe_task("component-sync-during-finalize")
        assert record["status"] == "finalizing"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_durable_restart_hydrates_assigned_work(tmp_path) -> None:
    db_path = tmp_path / "task_state.sqlite3"
    first_store = TaskStateStore(db_path)
    first = SchedulerComponentWorld(tmp_path / "first", task_store=first_store)
    try:
        await first.start()
        request_id = "component-durable-restart-assigned"
        await first.enqueue_sampling(request_id)
        assert (await first.observe_task(request_id))["status"] == "assigned"
    finally:
        first.close()

    second_store = TaskStateStore(db_path)
    second = SchedulerComponentWorld(tmp_path / "second", task_store=second_store)
    try:
        await second.scheduler.sync_replicas([second.replica(status="healthy")])
        contains = await second.observe_scheduler("component-durable-restart-assigned")
        lease = await second.claim_one()

        assert contains.present is True
        assert contains.location == "assigned"
        assert lease["item"]["request_id"] == "component-durable-restart-assigned"
        assert (await second.observe_task("component-durable-restart-assigned"))["status"] == "leased"
    finally:
        second.close()


@pytest.mark.anyio
async def test_scheduler_component_renew_missing_task_cleans_orphan_lease(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-renew-missing-task"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        await world.task_state.async_forget_task(request_id=request_id)

        renewed = await world.scheduler.renew(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            lease_ttl_s=30.0,
        )

        assert renewed.ok is False and renewed.reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).present is False
        validate = await world.scheduler.validate(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
    finally:
        world.close()
