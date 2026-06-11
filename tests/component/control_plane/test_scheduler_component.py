from __future__ import annotations

import asyncio
import time

import pytest

from mint_server.backend.task_state_store import FutureStatus

from .harness import SchedulerComponentWorld
from .invariants import assert_terminal_not_scheduled


pytestmark = pytest.mark.component


async def _assert_stats_progress_while_blocked(world: SchedulerComponentWorld, call, block_name: str):
    block = world.faults.block(block_name)
    task = asyncio.create_task(call())
    await asyncio.wait_for(block.entered.wait(), timeout=1.0)

    stats = await asyncio.wait_for(world.scheduler.stats(), timeout=0.5)

    block.release.set()
    result = await task
    assert stats["scheduler_instance_id"]
    return result


async def _assert_scheduler_surfaces_progress_while_blocked(
    world: SchedulerComponentWorld,
    call,
    block_name: str,
):
    block = world.faults.block(block_name)
    task = asyncio.create_task(call())
    await asyncio.wait_for(block.entered.wait(), timeout=1.0)

    contains = await asyncio.wait_for(world.observe_scheduler("component-progress-probe"), timeout=0.5)
    stats = await asyncio.wait_for(world.scheduler.stats(), timeout=0.5)
    appended = await asyncio.wait_for(
        world.enqueue_sampling(f"component-progress-{block_name}", assign=False),
        timeout=0.5,
    )
    synced = await asyncio.wait_for(
        world.scheduler.sync_replicas([world.replica(status="healthy")]),
        timeout=0.5,
    )

    block.release.set()
    result = await task
    assert contains["ok"] is True
    assert stats["scheduler_instance_id"]
    assert appended.scheduler_result["ok"] is True
    assert synced["ok"] is True
    return result


@pytest.mark.anyio
async def test_scheduler_component_happy_path_reaches_retrieve_future(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-happy")

        await world.runtime_once()

        assert await world.observe_future_status("component-happy") == FutureStatus.DONE
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
        claimed = await world.scheduler.claim(
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
async def test_scheduler_component_duplicate_append_is_idempotent_while_pending(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()

        first = await world.enqueue_sampling("component-duplicate")
        second = await world.enqueue_sampling("component-duplicate")

        assert first.scheduler_result["ok"] is True
        assert not first.scheduler_result.get("idempotent", False)
        assert second.scheduler_result["ok"] is True
        assert second.scheduler_result["idempotent"] is True

        lease = await world.claim_one()
        assert lease["item"]["request_id"] == "component-duplicate"
        await world.claim_none()
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_stale_consumer_cannot_finalize_or_fail_active_lease(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-stale-consumer")
        lease = await world.claim_one()

        stale_finalize = await world.scheduler.begin_finalize(
            lease_id=lease["lease_id"],
            consumer_id="stale-consumer",
            consumer_generation=world.generation,
        )
        stale_fail = await world.scheduler.fail(
            lease_id=lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation + 1,
            requeue=False,
            reason="stale",
        )

        assert stale_finalize == {"ok": False, "reason": "stale_consumer"}
        assert stale_fail == {"ok": False, "reason": "stale_consumer"}
        assert (
            await world.scheduler.validate(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
        record = await world.observe_task("component-stale-consumer")
        assert record["status"] == "leased"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_complete_requires_durable_terminal_commit(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-complete-before-commit")
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease_id=lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            finalize_ttl_s=30.0,
            staged_payload_path=str(world.tmp_path / "staged.json"),
        )

        premature_complete = await world.scheduler.complete(
            lease_id=lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
        )

        assert begin["ok"] is True
        assert premature_complete == {"ok": False, "reason": "not_terminal"}
        assert (
            await world.scheduler.validate(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
        assert (await world.observe_task("component-complete-before-commit"))["status"] == "finalizing"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_fail_terminal_requires_durable_terminal_commit(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-fail-before-commit")
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease_id=lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            finalize_ttl_s=30.0,
        )

        premature_fail = await world.scheduler.fail(
            lease_id=lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            requeue=False,
            reason="premature-terminal-fail",
        )

        assert begin["ok"] is True
        assert premature_fail == {"ok": False, "reason": "not_terminal"}
        assert (
            await world.scheduler.validate(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
        assert (await world.observe_task("component-fail-before-commit"))["status"] == "finalizing"
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
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                finalize_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        completed = await world.scheduler.complete(
            lease_id=lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
        )

        block.release.set()
        finalized = await finalize_task

        assert completed == {"ok": False, "reason": "finalize_inflight"}
        assert finalized["ok"] is True
        assert (
            await world.scheduler.validate(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
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
            lease_id=old_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
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
                lease_id=old_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        expired = await world.scheduler.expire(now=time.time() + 31.0)
        await world.task_state.async_forget_task(request_id=request_id)
        await world.enqueue_sampling(request_id)
        new_lease = await world.claim_one()

        block.release.set()
        stale_complete = await complete_task

        assert begin["ok"] is True
        assert committed["ok"] is True
        assert expired == {"ok": True, "expired": 0}
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
        assert stale_complete == {"ok": False, "reason": "stale_consumer"}
        assert (
            await world.scheduler.validate(
                lease_id=new_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
        assert (await world.observe_scheduler(request_id))["location"] == "leased"
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
            lease_id=lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
        )

        assert completed == {"ok": True, "request_id": request_id}
        assert (await world.observe_scheduler(request_id))["present"] is False
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
        assert (await world.observe_scheduler(request_id))["present"] is False
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

        assert [lease["item"]["request_id"] for lease in claimed["leases"]] == [valid_request_id]
        assert (await world.observe_scheduler(stale_request_id))["present"] is False
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
                    lease_id=lease["lease_id"],
                    consumer_id=world.consumer_id,
                    consumer_generation=world.generation,
                    finalize_ttl_s=30.0,
                )
            )
            await asyncio.wait_for(block.entered.wait(), timeout=1.0)

            failed = await world.scheduler.fail(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                requeue=requeue,
                reason="fail-during-finalize",
            )

            block.release.set()
            finalized = await finalize_task

            assert failed == {"ok": False, "reason": "finalize_inflight"}
            assert finalized["ok"] is True
            assert (
                await world.scheduler.validate(
                    lease_id=lease["lease_id"],
                    consumer_id=world.consumer_id,
                    consumer_generation=world.generation,
                )
            )["ok"] is True
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
        assert len(claimed["leases"]) == 1
        assert claimed["leases"][0]["item"]["request_id"] == "component-old-generation"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_executor_failure_commits_failed_terminal(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-exec-failed")

        async def _failing_executor(_lease: dict) -> None:
            raise RuntimeError("synthetic executor failure")

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
        )["present"] is True

        world.inject_payload_write_failure(False)
        assigned = await world.scheduler.assign_pending(max_items=1)
        await world.runtime_once()
        assert assigned["assigned"] == 1
        assert await world.observe_future_status("component-payload-write-failed") == FutureStatus.DONE
        await assert_terminal_not_scheduled(world, "component-payload-write-failed")
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

        assert expired == {"ok": True, "expired": 1}
        assert assigned["assigned"] == 1
        assert second_lease["item"]["request_id"] == "component-expired-lease"
        assert second_lease["lease_id"] != first_lease["lease_id"]
        record = await world.observe_task("component-expired-lease")
        assert record["status"] == "leased"
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
            lease_id=first_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            finalize_ttl_s=1.0,
            staged_payload_path=str(world.tmp_path / "staged.json"),
        )

        expired = await world.scheduler.expire(now=time.time() + 2.0)
        assigned = await world.scheduler.assign_pending(max_items=1)
        second_lease = await world.claim_one()

        assert begin["ok"] is True
        assert expired == {"ok": True, "expired": 1}
        assert assigned["assigned"] == 1
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
            lease_id=old_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            finalize_ttl_s=30.0,
        )
        stale_complete = await world.scheduler.complete(
            lease_id=old_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
        )
        valid_new = await world.scheduler.validate(
            lease_id=new_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
        )

        assert expired == {"ok": True, "expired": 1}
        assert assigned["assigned"] == 1
        assert old_lease["lease_id"] != new_lease["lease_id"]
        assert stale_finalize == {"ok": False, "reason": "unknown_lease"}
        assert stale_complete == {"ok": False, "reason": "unknown_lease"}
        assert valid_new["ok"] is True
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

        takeover = await world.acquire_owner(
            owner_id="component-scheduler-restarted",
            ttl_s=30.0,
            now=time.time() + 31.0,
        )
        world.replace_scheduler(owner_id="component-scheduler-restarted")
        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])

        with pytest.raises(Exception, match="owner_active"):
            await old_scheduler.renew(
                lease_id=old_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                lease_ttl_s=30.0,
            )
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert takeover["ok"] is True
        assert takeover["epoch"] == 2
        assert synced["assigned"]["assigned"] == 1 or assigned["assigned"] == 1
        assert new_lease["item"]["request_id"] == "component-owner-fencing"
        assert new_lease["scheduler_epoch"] == 2
        assert new_lease["lease_id"] != old_lease["lease_id"]
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
            lease_id=lease["lease_id"],
            consumer_id=old_consumer_id,
            consumer_generation=old_generation,
            lease_ttl_s=30.0,
        )
        completed = await world.scheduler.complete(
            lease_id=lease["lease_id"],
            consumer_id=old_consumer_id,
            consumer_generation=old_generation,
        )
        failed = await world.scheduler.fail(
            lease_id=lease["lease_id"],
            consumer_id=old_consumer_id,
            consumer_generation=old_generation,
            requeue=False,
            reason="stale-runtime",
        )

        assert renewed == {"ok": False, "reason": "unknown_lease"}
        assert completed == {"ok": False, "reason": "unknown_lease"}
        assert failed == {"ok": False, "reason": "unknown_lease"}
        assert (await world.observe_task("component-stale-runtime-active"))["status"] == "assigned"
        assigned = await world.scheduler.assign_pending(max_items=1)
        assert assigned["assigned"] == 0
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
        assert len(claimed["leases"]) == 1
        assert claimed["leases"][0]["lease_id"] != lease["lease_id"]
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
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
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
        assert finalized["ok"] is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_claim_task_does_not_block_stats(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-claim")

        claimed = await _assert_stats_progress_while_blocked(
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

        assert len(claimed["leases"]) == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_assign_task_does_not_block_stats(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-assign", assign=False)

        assigned = await _assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.assign_pending(max_items=1),
            "task_state.assign_task",
        )

        assert assigned["assigned"] == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_assign_task_does_not_block_scheduler_surfaces(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-progress-probe", assign=False)

        assigned = await _assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.assign_pending(max_items=1),
            "task_state.assign_task",
        )

        assert assigned["assigned"] == 1
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

        assert claimed["leases"] == []
        assert assigned["assigned"] == 1
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

        assert (await world.observe_scheduler("component-assign-fails-a"))["location"] == "backlog"
        assert (await world.observe_scheduler("component-assign-fails-b"))["location"] == "backlog"

        assigned = await world.scheduler.assign_pending(max_items=2)

        assert assigned["assigned"] == 2
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
        assert (await world.observe_scheduler("component-assign-prefix-b"))["location"] == "backlog"
        assert (await world.observe_scheduler("component-assign-prefix-c"))["location"] == "backlog"

        assigned = await world.scheduler.assign_pending(max_items=2)
        assert assigned["assigned"] == 2
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

        assert synced["deferred"] == "inflight_scheduler_transition"
        assert assigned["assigned"] == 1
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

        assert removed["requeued"] == 1
        assert (await world.observe_scheduler("component-removed-replica-assigned"))["location"] == "backlog"
        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])
        assert synced["assigned"]["assigned"] == 1
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

        finalized = await _assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.begin_finalize(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                finalize_ttl_s=30.0,
            ),
            "task_state.begin_finalize",
        )

        assert finalized["ok"] is True
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

        finalized = await _assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.begin_finalize(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                finalize_ttl_s=30.0,
            ),
            "task_state.begin_finalize",
        )

        assert finalized["ok"] is True
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
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
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

        assert synced["deferred"] == "inflight_scheduler_transition"
        assert finalized["ok"] is True
        assert (
            await world.scheduler.validate(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
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

        failed = await _assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.fail(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                requeue=True,
                reason="component-test",
            ),
            "task_state.requeue_task",
        )

        assert failed["ok"] is True
        assert failed["requeued"] is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_requeue_task_does_not_block_scheduler_surfaces(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-progress-probe")
        lease = await world.claim_one()

        failed = await _assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.fail(
                lease_id=lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                requeue=True,
                reason="component-test",
            ),
            "task_state.requeue_task",
        )

        assert failed["ok"] is True
        assert failed["requeued"] is True
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
                lease_id=old_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
                requeue=True,
                reason="direct-fail-requeue-error",
            )

        assert (
            await world.scheduler.validate(
                lease_id=old_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
        assert (await world.observe_scheduler(request_id))["location"] == "leased"
        assert (await world.observe_task(request_id))["status"] == "leased"

        retried = await world.scheduler.fail(
            lease_id=old_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            requeue=True,
            reason="direct-fail-requeue-retry",
        )
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert retried == {"ok": True, "request_id": request_id, "requeued": True}
        assert assigned["assigned"] == 1
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
            lease_id=old_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            requeue=True,
            reason="direct-fail-requeue-missing",
        )

        assert failed == {"ok": True, "request_id": request_id, "requeued": False}
        assert (
            await world.scheduler.validate(
                lease_id=old_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        ) == {"ok": False, "reason": "unknown_lease"}
        assert (await world.observe_scheduler(request_id))["present"] is False
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
                lease_id=old_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
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
                lease_id=old_lease["lease_id"],
                consumer_id=world.consumer_id,
                consumer_generation=world.generation,
            )
        )["ok"] is True
        assert (await world.observe_scheduler(request_id))["location"] == "leased"
        assert (await world.observe_task(request_id))["status"] == "leased"

        retried = await world.scheduler.fail(
            lease_id=old_lease["lease_id"],
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            requeue=True,
            reason="direct-fail-requeue-retry-after-cancel",
        )
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert retried == {"ok": True, "request_id": request_id, "requeued": True}
        assert assigned["assigned"] == 1
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
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

        assert (await world.observe_scheduler(first["item"]["request_id"]))["location"] == "leased"
        assert (await world.observe_scheduler(second["item"]["request_id"]))["location"] == "leased"

        requeued = await world.scheduler.sync_replicas([world.replica(status="unhealthy")])
        assert requeued["requeued"] == 2
        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])

        assert synced["assigned"]["assigned"] == 2
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

        assert (await world.observe_scheduler(first["item"]["request_id"]))["location"] == "leased"
        assert (await world.observe_scheduler(second["item"]["request_id"]))["location"] == "leased"
        assert (await world.observe_scheduler(third["item"]["request_id"]))["location"] == "backlog"

        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])
        assert synced["assigned"]["assigned"] == 1
        first_reclaim = await world.claim_one()
        assert first_reclaim["item"]["request_id"] == "component-requeue-prefix-c"

        requeued = await world.scheduler.sync_replicas([world.replica(status="unhealthy")])
        assert requeued["requeued"] == 3
        reassigned = await world.scheduler.sync_replicas([world.replica(status="healthy")])
        assert reassigned["assigned"]["assigned"] == 3
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
                lease_id=lease["lease_id"],
                consumer_id=old_consumer_id,
                consumer_generation=old_generation,
            )
        )["ok"] is True
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
        assert synced["requeued"] == 1
    finally:
        world.close()
