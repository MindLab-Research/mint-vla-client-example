from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest

from .helpers import (
    assert_scheduler_surfaces_progress_while_blocked,
    assert_stats_progress_while_blocked,
    token,
)
from .harness import SchedulerComponentWorld


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_blocked_requeue_task_does_not_block_stats(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
