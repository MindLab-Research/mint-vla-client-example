from __future__ import annotations

import asyncio
import time
from typing import Any, cast

import pytest

from mint_server.backend.stores.task_state_store import FutureStatus, TaskStateStore

from .harness import SchedulerComponentWorld
from .helpers import finish_success_for_test, token
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

        stale_finalize = await world.runtime_queue.begin_finalize(
            lease=token(lease, consumer_id="stale-consumer", consumer_generation=world.generation),
        )
        stale_fail = await world.runtime_queue.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation + 1),
            requeue=False,
            reason="stale",
        )

        assert stale_finalize.ok is False and stale_finalize.reason == "stale_consumer"
        assert stale_fail.ok is False and stale_fail.reason == "stale_consumer"
        assert (
            await world.runtime_queue.validate(
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

        failed = await world.runtime_queue.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="gpu_actor_died",
        )
        await world.enqueue_sampling("component-gpu-died-no-fence-b")
        claim = await world.runtime_queue.claim(
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
        begin = await world.runtime_queue.begin_finalize(
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

        stale_finalize = await world.runtime_queue.begin_finalize(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )
        stale_finish = await finish_success_for_test(world, old_lease)
        valid_new = await world.runtime_queue.validate(
            lease=token(new_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        assert expired.ok is True and expired.expired == 1
        assert assigned.assigned == 1
        assert old_lease["lease_id"] != new_lease["lease_id"]
        assert stale_finalize.ok is False and stale_finalize.reason == "unknown_lease"
        assert stale_finish.ok is False and stale_finish.reason == "stale_consumer"
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
        old_runtime_queue = world.runtime_queue

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
            await old_runtime_queue.renew(
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
async def test_scheduler_component_concurrent_reapers_recover_lost_pending_once(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-reaper-race-lost-pending"
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

        original_list_active_tasks = world.task_state.async_list_active_tasks
        first_snapshot_ready = asyncio.Event()
        release_first_snapshot = asyncio.Event()
        list_calls = 0

        async def list_active_tasks_with_reaper_race(**kwargs: Any) -> list[dict[str, Any]]:
            nonlocal list_calls
            list_calls += 1
            records = await original_list_active_tasks(**kwargs)
            if list_calls == 1:
                first_snapshot_ready.set()
                await release_first_snapshot.wait()
            return records

        world.task_state.async_list_active_tasks = list_active_tasks_with_reaper_race
        first_reaper = asyncio.create_task(
            world.scheduler.reap_lost_pending_tasks(reason="component-test-reaper-race-first")
        )
        await asyncio.wait_for(first_snapshot_ready.wait(), timeout=1.0)
        second = await world.scheduler.reap_lost_pending_tasks(reason="component-test-reaper-race-second")
        release_first_snapshot.set()
        first = await first_reaper

        stats = await world.scheduler.stats()
        assigned = await world.scheduler.assign_pending(max_items=2)
        claimed = await world.runtime_queue.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=2,
            lease_ttl_s=30.0,
        )

        assert created.created is True
        assert first["recovered"] + second["recovered"] == 1
        assert stats["counters"]["reaper_recovered"] == 1
        assert assigned.assigned == 1
        assert len(claimed.leases) == 1
        assert claimed.leases[0]["item"]["request_id"] == request_id
        await assert_no_double_lease(world)
        await assert_lease_consistency(world)
        await assert_no_orphan_assigned(world)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_old_generation_cannot_renew_finish_or_fail_after_sync(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        old_consumer_id = world.consumer_id
        old_generation = world.generation
        await world.enqueue_sampling("component-stale-runtime-active")
        lease = await world.claim_one()

        await world.scheduler.sync_replicas([world.replica(status="healthy", generation=old_generation + 1)])

        renewed = await world.runtime_queue.renew(
            lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            lease_ttl_s=30.0,
        )
        finished = await world.runtime_queue.finish_success(
            lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            result_path=str(world.tmp_path / "result.json"),
            result_checksum=None,
            result_size_bytes=None,
            billing_observations=None,
        )
        failed = await world.runtime_queue.fail(
            lease=token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            requeue=False,
            reason="stale-runtime",
        )

        assert renewed.ok is False and renewed.reason == "unknown_lease"
        assert finished.ok is False and finished.reason == "stale_consumer"
        assert failed.ok is False and failed.reason == "unknown_lease"
        assert (await world.observe_task("component-stale-runtime-active"))["status"] == "assigned"
        assigned = await world.scheduler.assign_pending(max_items=1)
        assert assigned.assigned == 0
        new_generation = old_generation + 1
        new_consumer_id = world.replica(generation=new_generation)["consumer_id"]
        claimed = await world.runtime_queue.claim(
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
            await world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=old_consumer_id,
                consumer_generation=old_generation,
                max_items=1,
                lease_ttl_s=30.0,
            )

        new_consumer_id = world.replica(generation=old_generation + 1)["consumer_id"]
        claimed = await world.runtime_queue.claim(
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

        renewed = await world.runtime_queue.renew(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            lease_ttl_s=30.0,
        )

        assert renewed.ok is False and renewed.reason == "unknown_lease"
        assert cast(Any, await world.observe_scheduler(request_id)).present is False
        validate = await world.runtime_queue.validate(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_orphan_lease_cannot_clear_recreated_same_request_id(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-orphan-lease-recreated"
        await world.enqueue_sampling(request_id)
        old_lease = await world.claim_one()
        await world.task_state.async_forget_task(request_id=request_id)

        recreated = cast(Any, await world.enqueue_sampling(request_id))
        new_lease = await world.claim_one()

        stale_renew = await world.runtime_queue.renew(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            lease_ttl_s=30.0,
        )
        stale_validate = await world.runtime_queue.validate(
            lease=token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        valid_new = await world.runtime_queue.validate(
            lease=token(new_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        observed = cast(Any, await world.observe_scheduler(request_id))
        record = await world.observe_task(request_id)

        assert recreated.scheduler_result.ok is True
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
        assert new_lease["attempt_id"] != old_lease["attempt_id"]
        assert stale_renew.ok is False and stale_renew.reason == "unknown_lease"
        assert stale_validate.ok is False and stale_validate.reason == "unknown_lease"
        assert valid_new.ok is True
        assert observed.present is True
        assert observed.location == "leased"
        assert observed.lease_id == new_lease["lease_id"]
        assert record["status"] == "leased"
        assert record["lease_id"] == new_lease["lease_id"]
        assert record["attempt_id"] == new_lease["attempt_id"]
        await assert_no_double_lease(world)
        await assert_lease_consistency(world)
    finally:
        world.close()
