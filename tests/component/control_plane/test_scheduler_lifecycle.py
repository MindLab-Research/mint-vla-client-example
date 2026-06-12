from __future__ import annotations

import time
from typing import Any, cast

import pytest

from mint_server.backend.task_state_store import FutureStatus, TaskStateStore

from .harness import SchedulerComponentWorld
from .helpers import token
from .invariants import (
    assert_lease_consistency,
    assert_no_double_lease,
    assert_no_orphan_assigned,
)
from .scenarios import sampling_meta


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_stale_consumer_cannot_finalize_or_fail_active_lease(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-stale-consumer")
        lease = await world.claim_one()

        stale_finalize = await world.scheduler.begin_finalize(
            lease=token(lease, consumer_id="stale-consumer", consumer_generation=world.generation),
        )
        stale_fail = await world.scheduler.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation + 1),
            requeue=False,
            reason="stale",
        )

        assert stale_finalize.ok is False and stale_finalize.reason == "stale_consumer"
        assert stale_fail.ok is False and stale_fail.reason == "stale_consumer"
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        record = await world.observe_task("component-stale-consumer")
        assert record["status"] == "leased"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_gpu_actor_died_does_not_scheduler_fence_consumer(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-gpu-died-no-fence-a")
        lease = await world.claim_one()

        failed = await world.scheduler.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="gpu_actor_died",
        )
        await world.enqueue_sampling("component-gpu-died-no-fence-b")
        claim = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )
        stats = await world.scheduler.stats()

        assert failed.ok is True and failed.requeued is True
        assert claim.ok is True
        assert len(claim.leases) == 1
        assert claim.leases[0]["consumer_id"] == world.consumer_id
        assert "self_failed_consumers" not in stats
        assert "self_failed_consumer_count" not in stats
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_lease_expiry_requeues_for_retry(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-expired-lease")
        first_lease = await world.claim_one(lease_ttl_s=1.0)

        expired = await world.scheduler.expire(now=time.time() + 2.0)
        assigned = await world.scheduler.assign_pending(max_items=1)
        second_lease = await world.claim_one()

        assert expired.ok is True and expired.expired == 1
        assert assigned.assigned == 1
        assert second_lease["item"]["request_id"] == "component-expired-lease"
        assert second_lease["lease_id"] != first_lease["lease_id"]
        record = await world.observe_task("component-expired-lease")
        assert record["status"] == "leased"
        await assert_no_double_lease(world)
        await assert_lease_consistency(world)
        await assert_no_orphan_assigned(world)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_finalizing_expiry_requeues_for_retry(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-expired-finalize")
        first_lease = await world.claim_one(lease_ttl_s=1.0)
        begin = await world.scheduler.begin_finalize(
            lease=token(first_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=1.0,
            staged_payload_path=str(world.tmp_path / "staged.json"),
        )

        expired = await world.scheduler.expire(now=time.time() + 2.0)
        assigned = await world.scheduler.assign_pending(max_items=1)
        second_lease = await world.claim_one()

        assert begin.ok is True
        assert expired.ok is True and expired.expired == 1
        assert assigned.assigned == 1
        assert second_lease["item"]["request_id"] == "component-expired-finalize"
        assert second_lease["lease_id"] != first_lease["lease_id"]
        record = await world.observe_task("component-expired-finalize")
        assert record["status"] == "leased"
        assert record["staged_payload_path"] is None
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_expired_old_lease_cannot_finalize_new_attempt(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-stale-finalizer")
        old_lease = await world.claim_one(lease_ttl_s=1.0)
        expired = await world.scheduler.expire(now=time.time() + 2.0)
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        stale_finalize = await world.scheduler.begin_finalize(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )
        stale_complete = await world.scheduler.complete(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        valid_new = await world.scheduler.validate(
            lease=token(new_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        assert expired.ok is True and expired.expired == 1
        assert assigned.assigned == 1
        assert old_lease["lease_id"] != new_lease["lease_id"]
        assert stale_finalize.ok is False and stale_finalize.reason == "unknown_lease"
        assert stale_complete.ok is False and stale_complete.reason == "unknown_lease"
        assert valid_new.ok is True
        record = await world.observe_task("component-stale-finalizer")
        assert record["status"] == "leased"
        assert record["lease_id"] == new_lease["lease_id"]
        assert record["attempt_id"] == new_lease["attempt_id"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_new_owner_hydrates_and_fences_old_scheduler(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-owner-fencing")
        old_lease = await world.claim_one()
        old_scheduler = world.scheduler

        takeover = cast(
            Any,
            await world.acquire_owner(
                owner_id="component-scheduler-restarted",
                ttl_s=30.0,
                now=time.time() + 31.0,
            ),
        )
        world.replace_scheduler(owner_id="component-scheduler-restarted")
        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])

        with pytest.raises(Exception, match="owner_active"):
            await old_scheduler.renew(
                lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                lease_ttl_s=30.0,
            )
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert takeover.ok is True
        assert takeover.epoch == 2
        assert (synced.assigned or {}).get("assigned") == 1 or assigned.assigned == 1
        assert new_lease["item"]["request_id"] == "component-owner-fencing"
        assert new_lease["scheduler_epoch"] == 2
        assert new_lease["lease_id"] != old_lease["lease_id"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_reaper_recovers_lost_pending_after_hydration(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-reaper-lost-pending"
        created = cast(
            Any,
            await world.task_state.async_create_task(
                request_id=request_id,
                op="sampling.asample",
                domain_key=world.domain_key,
                request_json=b'{"prompt":"lost"}',
                metadata={
                    **sampling_meta(world.domain_key),
                    "user_id": "user-a",
                    "apikey_id": "key-a",
                    "throttle_principal": "apikey:key-a",
                    "affinity_group": "lora:session-a:generation:1",
                    "token_cost": 1,
                },
            ),
        )

        before = cast(Any, await world.observe_scheduler(request_id))
        reaped = await world.scheduler.reap_lost_pending_tasks(reason="component-test-reaper")
        stats = await world.scheduler.stats()
        after = cast(Any, await world.observe_scheduler(request_id))
        assigned = await world.scheduler.assign_pending(max_items=1)
        await world.runtime_once()

        assert created.created is True
        assert before.present is False
        assert reaped["ok"] is True
        assert reaped["recovered"] == 1
        assert stats["counters"]["reaper_recovered"] == 1
        assert after.present is True
        assert after.location in {"backlog", "assigned"}
        assert assigned.assigned == 1
        assert await world.observe_future_status(request_id) == FutureStatus.DONE
        assert (await world.observe_task(request_id))["status"] == "done"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_old_generation_cannot_renew_complete_or_fail_after_sync(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        old_consumer_id = world.consumer_id
        old_generation = world.generation
        await world.enqueue_sampling("component-stale-runtime-active")
        lease = await world.claim_one()

        await world.scheduler.sync_replicas([world.replica(status="healthy", generation=old_generation + 1)])

        renewed = await world.scheduler.renew(
            lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            lease_ttl_s=30.0,
        )
        completed = await world.scheduler.complete(
            lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
        )
        failed = await world.scheduler.fail(
            lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            requeue=False,
            reason="stale-runtime",
        )

        assert renewed.ok is False and renewed.reason == "unknown_lease"
        assert completed.ok is False and completed.reason == "unknown_lease"
        assert failed.ok is False and failed.reason == "unknown_lease"
        assert (await world.observe_task("component-stale-runtime-active"))["status"] == "assigned"
        assigned = await world.scheduler.assign_pending(max_items=1)
        assert assigned.assigned == 0
        new_generation = old_generation + 1
        new_consumer_id = world.replica(generation=new_generation)["consumer_id"]
        claimed = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=new_consumer_id,
            consumer_generation=new_generation,
            max_items=1,
            lease_ttl_s=30.0,
        )
        assert len(claimed.leases) == 1
        assert claimed.leases[0]["lease_id"] != lease["lease_id"]
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
        contains = cast(Any, await second.observe_scheduler("component-durable-restart-assigned"))
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
        assert cast(Any, await world.observe_scheduler(request_id)).present is False
        validate = await world.scheduler.validate(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
    finally:
        world.close()
