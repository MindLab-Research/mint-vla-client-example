from __future__ import annotations

import asyncio
import time

import pytest

from mint_server.backend.control_plane_contracts import ExecutorOutcome
from mint_server.backend.model_work_admission import ModelWorkAdmissionRejectedError
from mint_server.backend.task_state_store import FutureStatus
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
async def test_scheduler_component_complete_cleans_scheduler_lease_for_missing_task(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-runtime-missing-task"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        await world.task_state.async_forget_task(request_id=request_id)

        completed = await world.scheduler.complete(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        assert completed.ok is True and completed.request_id == request_id
        assert (await world.observe_scheduler(request_id)).present is False
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_runtime_drops_assigned_missing_task_without_crashing(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-runtime-assigned-missing-task"
        await world.enqueue_sampling(request_id)
        await world.task_state.async_forget_task(request_id=request_id)

        actor = await world.runtime_once()

        assert actor.health_snapshot()["processed_total"] == 0
        assert actor.health_snapshot()["last_error"] is None
        assert (await world.observe_scheduler(request_id)).present is False
        assert (await world.scheduler.stats())["counters"]["stale_dropped"] == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_claim_skips_missing_stale_head_and_claims_next(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        stale_request_id = "component-missing-head"
        valid_request_id = "component-after-missing-head"
        await world.enqueue_sampling(stale_request_id)
        await world.enqueue_sampling(valid_request_id)
        await world.task_state.async_forget_task(request_id=stale_request_id)

        claimed = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed.leases] == [valid_request_id]
        assert (await world.observe_scheduler(stale_request_id)).present is False
        assert (await world.observe_task(valid_request_id))["status"] == "leased"
        assert (await world.scheduler.stats())["counters"]["stale_dropped"] == 1
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
async def test_scheduler_component_executor_failure_commits_failed_terminal(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-exec-failed")

        async def _failing_executor(_lease: dict) -> ExecutorOutcome:
            return ExecutorOutcome(kind="user_error", error="synthetic executor failure")

        actor = await world.runtime_once(executor=_failing_executor)

        assert actor.health_snapshot()["failed_total"] == 1
        assert await world.observe_future_status("component-exec-failed") == FutureStatus.FAILED
        record = await world.observe_task("component-exec-failed")
        assert record["status"] == "failed"
        assert "synthetic executor failure" in str(record["error"])
        status_code, payload = await world.retrieve("component-exec-failed", monkeypatch)
        assert status_code == 200
        assert "synthetic executor failure" in payload["error"]
        assert payload["category"] == "system"
        await assert_terminal_not_scheduled(world, "component-exec-failed")
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_payload_write_failure_requeues_without_terminal_commit(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-payload-write-failed")
        world.inject_payload_write_failure(True)

        actor = await world.runtime_once()

        assert actor.health_snapshot()["requeued_total"] == 1
        record = await world.observe_task("component-payload-write-failed")
        assert record["status"] == "pending"
        assert record["result_path"] is None
        assert record["staged_payload_path"] is None
        assert (
            await world.scheduler.contains(request_id="component-payload-write-failed")
        ).present is True

        world.inject_payload_write_failure(False)
        assigned = await world.scheduler.assign_pending(max_items=1)
        await world.runtime_once()
        assert assigned.assigned == 1
        assert await world.observe_future_status("component-payload-write-failed") == FutureStatus.DONE
        await assert_terminal_not_scheduled(world, "component-payload-write-failed")
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
async def test_scheduler_component_blocked_claim_task_does_not_block_stats(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-claim")

        claimed = await assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            ),
            "task_state.claim_task",
        )

        assert len(claimed.leases) == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_assign_task_does_not_block_stats(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-assign", assign=False)

        assigned = await assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.assign_pending(max_items=1),
            "task_state.assign_task",
        )

        assert assigned.assigned == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_assign_task_does_not_block_scheduler_surfaces(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-progress-probe", assign=False)

        assigned = await assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.assign_pending(max_items=1),
            "task_state.assign_task",
        )

        assert assigned.assigned == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_assign_task_keeps_claim_nonblocking_and_empty(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-assigning-not-claimable", assign=False)

        block = world.faults.block("task_state.assign_task")
        assign_task = asyncio.create_task(world.scheduler.assign_pending(max_items=1))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        claimed = await asyncio.wait_for(
            world.scheduler.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            ),
            timeout=0.5,
        )

        block.release.set()
        assigned = await assign_task

        assert claimed.leases == []
        assert assigned.assigned == 1
        lease = await world.claim_one()
        assert lease["item"]["request_id"] == "component-assigning-not-claimable"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_assign_failure_restores_unprocessed_batch_to_backlog(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-assign-fails-a", assign=False)
        await world.enqueue_sampling("component-assign-fails-b", assign=False)

        world.faults.fail_on_call(
            "task_state.assign_task",
            1,
            RuntimeError("synthetic assign failure"),
        )
        with pytest.raises(RuntimeError, match="synthetic assign failure"):
            await world.scheduler.assign_pending(max_items=2)

        assert (await world.observe_scheduler("component-assign-fails-a")).location == "backlog"
        assert (await world.observe_scheduler("component-assign-fails-b")).location == "backlog"

        assigned = await world.scheduler.assign_pending(max_items=2)

        assert assigned.assigned == 2
        leases = [
            await world.claim_one(),
            await world.claim_one(),
        ]
        assert [lease["item"]["request_id"] for lease in leases] == [
            "component-assign-fails-a",
            "component-assign-fails-b",
        ]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_assign_late_failure_preserves_committed_prefix(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-assign-prefix-a", assign=False)
        await world.enqueue_sampling("component-assign-prefix-b", assign=False)
        await world.enqueue_sampling("component-assign-prefix-c", assign=False)

        world.faults.fail_on_call(
            "task_state.assign_task",
            2,
            RuntimeError("synthetic late assign failure"),
        )
        with pytest.raises(RuntimeError, match="synthetic late assign failure"):
            await world.scheduler.assign_pending(max_items=3)

        first = await world.claim_one()
        assert first["item"]["request_id"] == "component-assign-prefix-a"
        assert (await world.observe_scheduler("component-assign-prefix-b")).location == "backlog"
        assert (await world.observe_scheduler("component-assign-prefix-c")).location == "backlog"

        assigned = await world.scheduler.assign_pending(max_items=2)
        assert assigned.assigned == 2
        leases = [
            await world.claim_one(),
            await world.claim_one(),
        ]
        assert [lease["item"]["request_id"] for lease in leases] == [
            "component-assign-prefix-b",
            "component-assign-prefix-c",
        ]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sync_defers_while_assignment_is_inflight(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-sync-during-assign", assign=False)

        block = world.faults.block("task_state.assign_task")
        assign_task = asyncio.create_task(world.scheduler.assign_pending(max_items=1))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        synced = await asyncio.wait_for(
            world.scheduler.sync_replicas([world.replica(status="healthy", generation=world.generation + 1)]),
            timeout=0.5,
        )

        block.release.set()
        assigned = await assign_task
        lease = await world.claim_one()

        assert synced.extra["deferred"] == "inflight_scheduler_transition"
        assert assigned.assigned == 1
        assert lease["item"]["request_id"] == "component-sync-during-assign"
        assert lease["consumer_generation"] == world.generation
        record = await world.observe_task("component-sync-during-assign")
        assert record["status"] == "leased"
        assert record["scheduler_epoch"] == lease["scheduler_epoch"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_removed_replica_requeues_assigned_work(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-removed-replica-assigned")

        removed = await world.scheduler.sync_replicas([])

        assert removed.requeued == 1
        assert (await world.observe_scheduler("component-removed-replica-assigned")).location == "backlog"
        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])
        assert synced.assigned["assigned"] == 1
        lease = await world.claim_one()
        assert lease["item"]["request_id"] == "component-removed-replica-assigned"
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
async def test_scheduler_component_blocked_requeue_task_does_not_block_stats(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-requeue")
        lease = await world.claim_one()

        failed = await assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.fail(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=True,
                reason="component-test",
            ),
            "task_state.requeue_task",
        )

        assert failed.ok is True
        assert failed.requeued is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_requeue_task_does_not_block_scheduler_surfaces(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-progress-probe")
        lease = await world.claim_one()

        failed = await assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.fail(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=True,
                reason="component-test",
            ),
            "task_state.requeue_task",
        )

        assert failed.ok is True
        assert failed.requeued is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_direct_fail_requeue_failure_preserves_retryable_lease(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-direct-fail-requeue-error"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one()

        world.faults.fail_on_call(
            "task_state.requeue_task",
            1,
            RuntimeError("synthetic direct requeue failure"),
        )
        with pytest.raises(RuntimeError, match="synthetic direct requeue failure"):
            await world.scheduler.fail(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=True,
                reason="direct-fail-requeue-error",
            )

        assert (
            await world.scheduler.validate(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"
        assert (await world.observe_task(request_id))["status"] == "leased"

        retried = await world.scheduler.fail(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="direct-fail-requeue-retry",
        )
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert retried.ok is True and retried.request_id == request_id and retried.requeued is True
        assert assigned.assigned == 1
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
        assert (await world.observe_task(request_id))["status"] == "leased"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_direct_fail_requeue_drops_missing_task_projection(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-direct-fail-requeue-missing"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one()
        await world.task_state.async_forget_task(request_id=request_id)

        failed = await world.scheduler.fail(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="direct-fail-requeue-missing",
        )

        assert failed.ok is True and failed.request_id == request_id and failed.requeued is False
        validate = await world.scheduler.validate(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).present is False
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_direct_fail_requeue_cancellation_preserves_retryable_lease(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-direct-fail-requeue-cancel"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one()

        block = world.faults.block("task_state.requeue_task")
        fail_task = asyncio.create_task(
            world.scheduler.fail(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=True,
                reason="direct-fail-requeue-cancel",
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        fail_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await fail_task

        assert (
            await world.scheduler.validate(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"
        assert (await world.observe_task(request_id))["status"] == "leased"

        retried = await world.scheduler.fail(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="direct-fail-requeue-retry-after-cancel",
        )
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert retried.ok is True and retried.request_id == request_id and retried.requeued is True
        assert assigned.assigned == 1
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_direct_fail_requeue_cancellation_after_commit_preserves_backlog(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-direct-fail-requeue-cancel-after"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one()

        block = world.faults.block("task_state.requeue_task.after")
        fail_task = asyncio.create_task(
            world.scheduler.fail(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=True,
                reason="direct-fail-requeue-cancel-after",
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        fail_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await fail_task

        validate = await world.scheduler.validate(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).location == "backlog"
        assert (await world.observe_task(request_id))["status"] == "pending"

        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert assigned.assigned == 1
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_expire_cancellation_after_terminal_requeue_rejection_cleans_projection(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-expire-requeue-terminal-after"
        await world.enqueue_sampling(request_id)
        await world.claim_one(lease_ttl_s=1.0)

        block = world.faults.block("task_state.requeue_task.after")
        expire_task = asyncio.create_task(world.scheduler.expire(now=time.time() + 2.0))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        await asyncio.to_thread(
            world.task_store.complete_task_success,
            request_id=request_id,
            result_path=str(world.tmp_path / "result.json"),
            result_checksum=None,
            result_size_bytes=None,
        )
        expire_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await expire_task

        assert (await world.observe_task(request_id))["status"] == "done"
        assert (await world.observe_scheduler(request_id)).present is False
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_requeue_failure_restores_unprocessed_batch_to_backlog(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-requeue-fails-a")
        await world.enqueue_sampling("component-requeue-fails-b")
        first = await world.claim_one()
        second = await world.claim_one()

        world.faults.fail_on_call(
            "task_state.requeue_task",
            1,
            RuntimeError("synthetic requeue failure"),
        )
        with pytest.raises(RuntimeError, match="synthetic requeue failure"):
            await world.scheduler.sync_replicas([world.replica(status="unhealthy")])

        assert (await world.observe_scheduler(first["item"]["request_id"])).location == "leased"
        assert (await world.observe_scheduler(second["item"]["request_id"])).location == "leased"

        requeued = await world.scheduler.sync_replicas([world.replica(status="unhealthy")])
        assert requeued.requeued == 2
        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])

        assert synced.assigned["assigned"] == 2
        leases = [
            await world.claim_one(),
            await world.claim_one(),
        ]
        assert [lease["item"]["request_id"] for lease in leases] == [
            "component-requeue-fails-a",
            "component-requeue-fails-b",
        ]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_requeue_late_failure_preserves_committed_and_unprocessed_items(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-requeue-prefix-a")
        await world.enqueue_sampling("component-requeue-prefix-b")
        await world.enqueue_sampling("component-requeue-prefix-c")
        first = await world.claim_one()
        second = await world.claim_one()
        third = await world.claim_one()

        world.faults.fail_on_call(
            "task_state.requeue_task",
            2,
            RuntimeError("synthetic late requeue failure"),
        )
        with pytest.raises(RuntimeError, match="synthetic late requeue failure"):
            await world.scheduler.sync_replicas([world.replica(status="unhealthy")])

        assert (await world.observe_scheduler(first["item"]["request_id"])).location == "leased"
        assert (await world.observe_scheduler(second["item"]["request_id"])).location == "leased"
        assert (await world.observe_scheduler(third["item"]["request_id"])).location == "backlog"

        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])
        assert synced.assigned["assigned"] == 1
        first_reclaim = await world.claim_one()
        assert first_reclaim["item"]["request_id"] == "component-requeue-prefix-c"

        requeued = await world.scheduler.sync_replicas([world.replica(status="unhealthy")])
        assert requeued.requeued == 3
        reassigned = await world.scheduler.sync_replicas([world.replica(status="healthy")])
        assert reassigned.assigned["assigned"] == 3
        leases = [
            await world.claim_one(),
            await world.claim_one(),
            await world.claim_one(),
        ]
        assert [lease["item"]["request_id"] for lease in leases] == [
            "component-requeue-prefix-a",
            "component-requeue-prefix-b",
            "component-requeue-prefix-c",
        ]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sync_requeue_failure_preserves_replica_registry(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        old_consumer_id = world.consumer_id
        old_generation = world.generation
        await world.enqueue_sampling("component-sync-requeue-registry")
        lease = await world.claim_one()

        world.faults.fail_on_call(
            "task_state.requeue_task",
            1,
            RuntimeError("synthetic registry rollback failure"),
        )
        with pytest.raises(RuntimeError, match="synthetic registry rollback failure"):
            await world.scheduler.sync_replicas(
                [world.replica(status="healthy", generation=old_generation + 1)]
            )

        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            )
        ).ok is True
        await world.enqueue_sampling("component-sync-requeue-registry-after", assign=False)
        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.scheduler.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.replica(generation=old_generation + 1)["consumer_id"],
                consumer_generation=old_generation + 1,
                max_items=1,
                lease_ttl_s=30.0,
            )

        synced = await world.scheduler.sync_replicas(
            [world.replica(status="healthy", generation=old_generation + 1)]
        )
        assert synced.requeued == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_generation_bump_drains_assigned_unleased_work(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-generation-bump-assigned-drain"
        await world.enqueue_sampling(request_id, assign=False)
        assigned = await world.scheduler.assign_pending(max_items=1)

        old_record = await world.observe_task(request_id)
        next_generation = world.generation + 1
        next_consumer_id = world.replica(generation=next_generation)["consumer_id"]
        synced = await world.scheduler.sync_replicas(
            [world.replica(status="healthy", generation=next_generation)]
        )
        lease = await world.claim_one(
            consumer_id=next_consumer_id,
            consumer_generation=next_generation,
        )
        begin = await world.scheduler.begin_finalize(
            lease=token(lease, consumer_id=next_consumer_id, consumer_generation=next_generation)
        )
        finished = await world.scheduler.finish_success(
            lease=token(lease, consumer_id=next_consumer_id, consumer_generation=next_generation),
            result_path=str(world.tmp_path / "generation-bump-result.json"),
        )
        record = await world.observe_task(request_id)

        assert assigned.assigned == 1
        assert old_record["status"] == "assigned"
        assert synced.requeued == 1
        assert synced.assigned["assigned"] == 1
        assert lease["item"]["request_id"] == request_id
        assert lease["consumer_id"] == next_consumer_id
        assert begin.ok is True
        assert finished.ok is True
        assert record["status"] == "done"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_affinity_sticks_to_same_replica_surface(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.scheduler.sync_replicas(
            [
                world.replica(replica_id="replica-a", status="healthy"),
                world.replica(replica_id="replica-b", status="healthy"),
            ]
        )
        await world.enqueue_sampling("component-affinity-a1", affinity_group="lora:a")
        await world.enqueue_sampling("component-affinity-a2", affinity_group="lora:a")
        await world.enqueue_sampling("component-affinity-b1", affinity_group="lora:b")

        first_a = await world.claim_one(replica_id="replica-a")
        second_a = await world.claim_one(replica_id="replica-a")
        first_b = await world.claim_one(replica_id="replica-b")

        assert [first_a["item"]["request_id"], second_a["item"]["request_id"]] == [
            "component-affinity-a1",
            "component-affinity-a2",
        ]
        assert first_b["item"]["request_id"] == "component-affinity-b1"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_assignment_counts_active_leases_surface(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.scheduler.sync_replicas(
            [
                world.replica(replica_id="replica-a", status="healthy"),
                world.replica(replica_id="replica-b", status="healthy"),
            ]
        )
        await world.enqueue_sampling("component-active-lease", affinity_group="lora:a")
        active = await world.claim_one(replica_id="replica-a")

        await world.enqueue_sampling("component-after-active", affinity_group="lora:b")

        assert active["item"]["request_id"] == "component-active-lease"
        claimed_b = await world.claim_one(replica_id="replica-b")
        empty_a = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id="replica-a",
            consumer_id=world.replica(replica_id="replica-a")["consumer_id"],
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert claimed_b["item"]["request_id"] == "component-after-active"
        assert empty_a.leases == []
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_claims_first_item_when_it_exceeds_token_budget_surface(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-expensive-first", assign=False, token_cost=100)
        await world.enqueue_sampling("component-cheap-next", assign=False, token_cost=1)
        assigned = await world.scheduler.assign_pending(max_items=2)

        claimed = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=4,
            token_budget=50,
            lease_ttl_s=30.0,
        )
        next_lease = await world.claim_one()

        assert assigned.assigned == 2
        assert [lease["item"]["request_id"] for lease in claimed.leases] == [
            "component-expensive-first"
        ]
        assert next_lease["item"]["request_id"] == "component-cheap-next"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_same_session_sampling_is_serialized_by_ordering_key(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling(
            "component-session-serial-a",
            affinity_group="lora:session-a:generation:1",
            ordering_key="session:session-a",
        )
        await world.enqueue_sampling(
            "component-session-serial-b",
            affinity_group="lora:session-a:generation:1",
            ordering_key="session:session-a",
        )

        first = await world.claim_one()
        blocked = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert first["item"]["request_id"] == "component-session-serial-a"
        assert blocked.leases == []

        await world.future_service.async_resolve(
            "component-session-serial-a",
            {"ok": True, "request_id": "component-session-serial-a"},
        )
        assert (
            await world.scheduler.complete(
                lease=token(first, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True

        second = await world.claim_one()
        assert second["item"]["request_id"] == "component-session-serial-b"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sampling_token_budget_admission_enforce_rejects_and_releases(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "100")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_PRINCIPAL_DOMAIN", "10")
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        first = await world.enqueue_sampling("component-token-budget-a", token_cost=7)
        with pytest.raises(ModelWorkAdmissionRejectedError) as rejected_exc:
            await world.enqueue_sampling("component-token-budget-b", token_cost=4)

        assert first.scheduler_result.ok is True
        assert rejected_exc.value.scheduler_result.reason == "principal_domain_token_budget_exceeded"
        assert rejected_exc.value.scheduler_result.extra["current"] == 11
        assert rejected_exc.value.scheduler_result.extra["limit"] == 10
        assert (await world.observe_scheduler("component-token-budget-b")).present is False

        await world.runtime_once()
        accepted_after_release = await world.enqueue_sampling("component-token-budget-c", token_cost=4)
        assert accepted_after_release.scheduler_result.ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_megatron_training_session_is_serialized_by_ordering_key(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path, domain_key="megatron:glm-5")
    try:
        await world.start()
        await world.enqueue_sampling(
            "component-megatron-session-a",
            affinity_group="training_session:model-a",
            ordering_key="training_session:model-a",
        )
        await world.enqueue_sampling(
            "component-megatron-session-b",
            affinity_group="training_session:model-a",
            ordering_key="training_session:model-a",
        )

        first = await world.claim_one()
        blocked = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=4,
            lease_ttl_s=30.0,
        )

        assert first["item"]["request_id"] == "component-megatron-session-a"
        assert blocked.leases == []
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_supervisor_start_failure_removes_claimable_replica(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        class _FailingRuntime:
            def __init__(self, spec, generation: int) -> None:
                self.actor_name = spec.normalized_actor_name()
                self.generation = int(generation)
                self.shutdown_calls = 0

            def start(self) -> dict:
                raise RuntimeError("synthetic runtime start failure")

            def shutdown(self) -> dict:
                self.shutdown_calls += 1
                return {"ok": True}

            def health_snapshot(self) -> dict:
                return {
                    "running": False,
                    "actor_generation": self.generation,
                    "last_error": "synthetic runtime start failure",
                }

        async def _factory(spec, generation: int):
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
        assert (await world.observe_scheduler("component-supervisor-start-failure")).location == "backlog"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_supervisor_generation_restart_allows_only_new_consumer(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        class _Runtime:
            def __init__(self, spec, generation: int) -> None:
                self.actor_name = spec.normalized_actor_name()
                self.generation = int(generation)
                self.running = True
                self.fail_health = False

            def start(self) -> dict:
                return {"running": True}

            def shutdown(self) -> dict:
                self.running = False
                return {"ok": True}

            def health_snapshot(self) -> dict:
                if self.fail_health:
                    raise RuntimeError("synthetic runtime health failure")
                return {
                    "running": self.running,
                    "actor_generation": self.generation,
                    "domain_key": world.domain_key,
                    "replica_id": world.replica_id,
                }

        runtimes: list[_Runtime] = []

        async def _factory(spec, generation: int):
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
async def test_scheduler_component_sampling_admission_observe_allows_would_reject(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "observe")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        first = await world.enqueue_sampling("component-admission-observe-a", assign=False)
        second = await world.enqueue_sampling("component-admission-observe-b", assign=False)

        assert first.scheduler_result.ok is True
        assert second.scheduler_result.ok is True
        assert second.scheduler_result.extra["sampling_inflight_admission"]["would_reject"] is True
        assert (
            await world.observe_scheduler("component-admission-observe-b")
        ).present is True

        assigned = await world.scheduler.assign_pending(max_items=2)
        assert assigned.assigned == 2
        assert (await world.claim_one())["item"]["request_id"] == "component-admission-observe-a"
        assert (await world.claim_one())["item"]["request_id"] == "component-admission-observe-b"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sampling_admission_enforce_rejects_and_releases_after_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        first = await world.enqueue_sampling("component-admission-enforce-a")
        with pytest.raises(ModelWorkAdmissionRejectedError) as rejected_exc:
            await world.enqueue_sampling("component-admission-enforce-b")

        assert first.scheduler_result.ok is True
        assert rejected_exc.value.scheduler_result.ok is False
        assert rejected_exc.value.scheduler_result.reason == "principal_domain_inflight_limit_exceeded"
        assert (await world.observe_scheduler("component-admission-enforce-b")).present is False

        await world.runtime_once()
        await assert_terminal_not_scheduled(world, "component-admission-enforce-a")

        accepted_after_release = await world.enqueue_sampling("component-admission-enforce-c")
        assert accepted_after_release.scheduler_result.ok is True
        assert (await world.observe_scheduler("component-admission-enforce-c")).present is True
    finally:
        world.close()


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
