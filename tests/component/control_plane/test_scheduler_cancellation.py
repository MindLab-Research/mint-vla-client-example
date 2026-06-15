from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest

from mint_server.backend.stores.task_state_store import FutureStatus

from .helpers import finish_success_for_test, token
from .harness import SchedulerComponentWorld
from .invariants import assert_no_double_lease, assert_no_orphan_assigned


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_duplicate_append_cancel_does_not_forget_existing_pending(
    tmp_path,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-duplicate-append-cancel-pending"
        block = world.faults.block("task_state.create_task.after")
        first_task = asyncio.create_task(world.enqueue_sampling(request_id, assign=False))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        duplicate_task = asyncio.create_task(world.enqueue_sampling(request_id, assign=False))

        while sum(1 for method, _ in world.task_state.calls if method == "create_task") < 2:
            await asyncio.sleep(0.001)
        duplicate_task.cancel()
        block.release.set()
        created = await first_task
        with pytest.raises(asyncio.CancelledError):
            await duplicate_task

        assigned = await world.scheduler.assign_pending(max_items=1)
        lease = await world.claim_one()

        assert created.scheduler_result.ok is True
        assert (await world.observe_task(request_id))["status"] == "leased"
        assert assigned.assigned == 1
        assert lease["item"]["request_id"] == request_id
        assert (await world.claim_none()).leases == []
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_duplicate_append_cancel_does_not_rewrite_existing_payload(
    tmp_path,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-duplicate-append-cancel-payload"
        first_payload = b'{"prompt":"first"}'
        second_payload = b'{"prompt":"second"}'
        block = world.faults.block("task_state.create_task.after")
        first_task = asyncio.create_task(
            world.enqueue_sampling(request_id, assign=False, request_json=first_payload)
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        duplicate_task = asyncio.create_task(
            world.enqueue_sampling(request_id, assign=False, request_json=second_payload)
        )

        while sum(1 for method, _ in world.task_state.calls if method == "create_task") < 2:
            await asyncio.sleep(0.001)
        duplicate_task.cancel()
        block.release.set()
        created = await first_task
        with pytest.raises(asyncio.CancelledError):
            await duplicate_task

        task = await world.observe_task(request_id)

        assert created.scheduler_result.ok is True
        assert task["status"] == "pending"
        assert task["request_json"] == first_payload
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_assign_cancellation_restores_backlog(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-assign-cancel"
        await world.enqueue_sampling(request_id, assign=False)

        block = world.faults.block("task_state.assign_task")
        assign_task = asyncio.create_task(world.scheduler.assign_pending(max_items=1))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        assign_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await assign_task

        assert (await world.observe_scheduler(request_id)).location == "backlog"
        reassigned = await world.scheduler.assign_pending(max_items=1)
        lease = await world.claim_one()

        assert reassigned.assigned == 1
        assert lease["item"]["request_id"] == request_id
        assert (await world.observe_task(request_id))["status"] == "leased"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_claim_cancellation_restores_assigned_work(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-claim-cancel"
        await world.enqueue_sampling(request_id)

        block = world.faults.block("task_state.claim_task")
        claim_task = asyncio.create_task(
            world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        claim_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await claim_task

        assert (await world.observe_scheduler(request_id)).location == "assigned"
        lease = await world.claim_one()

        assert lease["item"]["request_id"] == request_id
        assert (await world.observe_task(request_id))["status"] == "leased"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_begin_finalize_cancellation_restores_lease(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-finalize-cancel"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize")
        finalize_task = asyncio.create_task(
            world.runtime_queue.begin_finalize(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        finalize_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await finalize_task

        assert (await world.observe_scheduler(request_id)).location == "leased"
        assert (
            await world.runtime_queue.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True

        retried = await world.runtime_queue.begin_finalize(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )
        assert retried.ok is True
        assert (await world.observe_task(request_id))["status"] == "finalizing"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sync_cancellation_preserves_replica_registry_and_lease(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        old_consumer_id = world.consumer_id
        old_generation = world.generation
        request_id = "component-sync-cancel"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        block = world.faults.block("task_state.requeue_task")
        sync_task = asyncio.create_task(
            world.scheduler.sync_replicas(
                [world.replica(status="healthy", generation=old_generation + 1)]
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        sync_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await sync_task

        assert (
            await world.runtime_queue.validate(
                lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"
        await world.enqueue_sampling("component-sync-cancel-after", assign=False)
        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.replica(generation=old_generation + 1)["consumer_id"],
                consumer_generation=old_generation + 1,
                max_items=1,
                lease_ttl_s=30.0,
            )
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sync_cancellation_after_requeue_commit_preserves_backlog_and_registry(
    tmp_path,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        old_consumer_id = world.consumer_id
        old_generation = world.generation
        request_id = "component-sync-cancel-after-requeue"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one()

        block = world.faults.block("task_state.requeue_task.after")
        sync_task = asyncio.create_task(
            world.scheduler.sync_replicas(
                [world.replica(status="healthy", generation=old_generation + 1)]
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        sync_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await sync_task

        validate = await world.runtime_queue.validate(
            lease=token(old_lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).location == "backlog"
        assert (await world.observe_task(request_id))["status"] == "pending"
        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.replica(generation=old_generation + 1)["consumer_id"],
                consumer_generation=old_generation + 1,
                max_items=1,
                lease_ttl_s=30.0,
            )

        reassigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one(
            consumer_id=old_consumer_id,
            consumer_generation=old_generation,
        )

        assert reassigned.assigned == 1
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_expire_cancellation_preserves_retryable_lease(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-expire-cancel"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one(lease_ttl_s=1.0)

        block = world.faults.block("task_state.requeue_task")
        expire_task = asyncio.create_task(world.scheduler.expire(now=time.time() + 2.0))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        expire_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await expire_task

        assert (
            await world.runtime_queue.validate(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"

        expired = await world.scheduler.expire(now=time.time() + 3.0)
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert expired.ok is True and expired.expired == 1
        assert assigned.assigned == 1
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_cancel_assigned_work_removes_scheduler_projection(
    tmp_path,
    monkeypatch,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-cancel-assigned"
        await world.enqueue_sampling(request_id)

        cancelled = await world.cancel(request_id, monkeypatch, reason="component-test")
        failed_status = await world.observe_future_status(request_id)
        status_code, payload = await world.retrieve(request_id, monkeypatch)

        assert cancelled.cancelled is True
        assert (await world.observe_scheduler(request_id)).present is False
        assert failed_status == FutureStatus.FAILED
        assert status_code == 200
        assert "cancelled" in payload["error"]
        await assert_no_double_lease(world)
        await assert_no_orphan_assigned(world)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_cancel_assigned_work_survives_scheduler_restart(
    tmp_path,
    monkeypatch,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-cancel-assigned-restart"
        await world.enqueue_sampling(request_id)

        cancelled = await world.cancel(request_id, monkeypatch, reason="component-test")
        await world.acquire_owner(owner_id="component-scheduler-restart", now=time.time() + 31.0)
        world.replace_scheduler(owner_id="component-scheduler-restart")
        await world.start()
        assigned = await world.scheduler.assign_pending(max_items=1)
        claimed = await world.claim_none()
        failed_status = await world.observe_future_status(request_id)
        status_code, payload = await world.retrieve(request_id, monkeypatch)

        assert cancelled.cancelled is True
        assert assigned.assigned == 0
        assert claimed.leases == []
        assert (await world.observe_scheduler(request_id)).present is False
        assert failed_status == FutureStatus.FAILED
        assert status_code == 200
        assert "cancelled" in payload["error"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_cancel_leased_work_removes_scheduler_projection(
    tmp_path,
    monkeypatch,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-cancel-leased"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        cancelled = await world.cancel(request_id, monkeypatch, reason="component-test")
        validate = await world.runtime_queue.validate(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        failed_status = await world.observe_future_status(request_id)
        status_code, payload = await world.retrieve(request_id, monkeypatch)

        assert cancelled.cancelled is True
        assert validate.ok is False and validate.reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).present is False
        assert failed_status == FutureStatus.FAILED
        assert status_code == 200
        assert "cancelled" in payload["error"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_cancel_leased_work_survives_scheduler_restart(
    tmp_path,
    monkeypatch,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-cancel-leased-restart"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        cancelled = await world.cancel(request_id, monkeypatch, reason="component-test")
        await world.acquire_owner(owner_id="component-scheduler-restart", now=time.time() + 31.0)
        world.replace_scheduler(owner_id="component-scheduler-restart")
        await world.start()
        assigned = await world.scheduler.assign_pending(max_items=1)
        claimed = await world.claim_none()
        failed_status = await world.observe_future_status(request_id)
        status_code, payload = await world.retrieve(request_id, monkeypatch)

        assert cancelled.cancelled is True
        assert assigned.assigned == 0
        assert claimed.leases == []
        assert (
            await world.runtime_queue.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is False and (await world.runtime_queue.validate(lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation))).reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).present is False
        assert failed_status == FutureStatus.FAILED
        assert status_code == 200
        assert "cancelled" in payload["error"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_assign_cancellation_after_durable_commit_preserves_claimability(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-assign-cancel-after"
        await world.enqueue_sampling(request_id, assign=False)

        block = world.faults.block("task_state.assign_task.after")
        assign_task = asyncio.create_task(world.scheduler.assign_pending(max_items=1))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        assign_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await assign_task

        assert (await world.observe_scheduler(request_id)).location == "assigned"
        assert (await world.observe_task(request_id))["status"] == "assigned"
        lease = await world.claim_one()
        assert lease["item"]["request_id"] == request_id
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_append_assign_cancellation_after_durable_assign_preserves_claimability(
    tmp_path,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-append-assign-cancel-after"

        block = world.faults.block("task_state.assign_task.after")
        append_task = asyncio.create_task(world.enqueue_sampling(request_id, assign=True))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        append_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await append_task

        assert (await world.observe_scheduler(request_id)).location == "assigned"
        assert (await world.observe_task(request_id))["status"] == "assigned"
        lease = await world.claim_one()
        assert lease["item"]["request_id"] == request_id
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_claim_cancellation_after_durable_commit_preserves_lease(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-claim-cancel-after"
        await world.enqueue_sampling(request_id)

        block = world.faults.block("task_state.claim_task.after")
        claim_task = asyncio.create_task(
            world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        claim_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await claim_task

        task = await world.observe_task(request_id)
        assert task["status"] == "leased"
        contains = await world.observe_scheduler(request_id)
        assert contains.location == "leased"
        assert contains.lease_id == task["lease_id"]
        assert (
            await world.runtime_queue.validate(
                lease=token(task, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_begin_finalize_cancellation_after_durable_commit_preserves_finalizing(
    tmp_path,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-finalize-cancel-after"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize.after")
        finalize_task = asyncio.create_task(
            world.runtime_queue.begin_finalize(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        finalize_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await finalize_task

        assert (await world.observe_scheduler(request_id)).location == "leased"
        assert (await world.observe_task(request_id))["status"] == "finalizing"
        assert (
            await world.runtime_queue.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        terminal_fail = await world.runtime_queue.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=False,
            reason="not-yet-terminal",
        )
        assert terminal_fail.ok is False and terminal_fail.reason == "not_terminal"
        expired = await world.scheduler.expire(now=time.time() + 29.0)
        assert expired.ok is True and expired.expired == 0
        committed = await world.task_state.async_commit_finalize_success(
            request_id=request_id,
            lease_id=lease["lease_id"],
            attempt_id=lease["attempt_id"],
            scheduler_epoch=lease["scheduler_epoch"],
            runtime_generation=world.generation,
            result_path=str(world.tmp_path / "result.json"),
            result_checksum=None,
            result_size_bytes=None,
        )
        completed = await finish_success_for_test(world, lease)

        assert committed.ok is True
        assert completed.ok is True and completed.request_id == request_id
        assert (await world.observe_scheduler(request_id)).present is False
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_expire_cancellation_after_requeue_commit_preserves_backlog(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-expire-cancel-after"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one(lease_ttl_s=1.0)

        block = world.faults.block("task_state.requeue_task.after")
        expire_task = asyncio.create_task(world.scheduler.expire(now=time.time() + 2.0))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        expire_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await expire_task

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
async def test_scheduler_component_append_cancellation_after_durable_create_rolls_back_task(
    tmp_path,
    monkeypatch,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-append-cancel-after-create"

        block = world.faults.block("task_state.create_task.after")
        append_task = asyncio.create_task(world.enqueue_sampling(request_id, assign=False))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        append_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await append_task

        assert (await world.observe_scheduler(request_id)).present is False
        with pytest.raises(KeyError):
            await world.observe_task(request_id)

        retry = await world.enqueue_sampling(request_id, assign=False)
        status_code, payload = await world.retrieve(request_id, monkeypatch)

        assert retry.scheduler_result.ok is True
        assert (await world.observe_scheduler(request_id)).present is True
        assert status_code == 408
        assert payload["request_id"] == request_id
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_append_cancellation_after_duplicate_create_keeps_terminal_result(
    tmp_path,
    monkeypatch,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-append-cancel-duplicate-terminal"
        await world.enqueue_sampling(request_id)
        await world.runtime_once()
        assert await world.observe_future_status(request_id) == FutureStatus.DONE
        assert (await world.retrieve(request_id, monkeypatch))[0] == 200

        block = world.faults.block("task_state.create_task.after")
        append_task = asyncio.create_task(world.enqueue_sampling(request_id, assign=False))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        append_task.cancel()
        block.release.set()
        with pytest.raises(asyncio.CancelledError):
            await append_task

        assert (await world.observe_task(request_id))["status"] in {"done", "retrieved"}
        assert (await world.observe_scheduler(request_id)).present is False
        status_code, payload = await world.retrieve(request_id, monkeypatch)
        assert status_code == 200
        assert payload == {"ok": True, "request_id": request_id}
    finally:
        world.close()
