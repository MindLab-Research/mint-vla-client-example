from __future__ import annotations

import asyncio
import time

import pytest

from mint_server.backend.control_plane_contracts import (
    AsyncSchedulerQueue,
    AsyncTaskLedger,
    ExecutorOutcome,
    LeaseToken,
    ModelWorkTaskGateway,
)
from mint_server.backend.task_state_store import FutureStatus
from mint_server.backend.model_work_admission import ModelWorkAdmissionRejectedError
from mint_server.backend.task_state_store import TaskStateStore

from .harness import SchedulerComponentWorld
from .invariants import assert_terminal_not_scheduled
from .scenarios import sampling_meta


pytestmark = pytest.mark.component


def _token(
    lease: dict,
    *,
    consumer_id: str | None = None,
    consumer_generation: int | None = None,
) -> LeaseToken:
    item = lease.get("item") if isinstance(lease.get("item"), dict) else {}
    return LeaseToken(
        request_id=str(item.get("request_id") or lease.get("request_id", "")),
        lease_id=str(lease["lease_id"]),
        attempt_id=str(lease.get("attempt_id", "")),
        scheduler_epoch=int(lease.get("scheduler_epoch", 0)),
        consumer_id=str(
            consumer_id if consumer_id is not None else lease.get("consumer_id", "")
        ),
        consumer_generation=int(
            consumer_generation
            if consumer_generation is not None
            else lease.get("consumer_generation", 0)
        ),
    )


def test_scheduler_component_world_exposes_typed_contracts(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        assert isinstance(world.task_gateway, ModelWorkTaskGateway)
        assert isinstance(world.task_ledger, AsyncTaskLedger)
        assert isinstance(world.runtime_queue, AsyncSchedulerQueue)
        assert world.scheduler is world.runtime_queue
    finally:
        world.close()


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
    assert contains.ok is True
    assert stats["scheduler_instance_id"]
    assert appended.scheduler_result.ok is True
    assert synced.ok is True
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
        assert len(claimed.leases) == 1
        assert claimed.leases[0]["item"]["request_id"] == "component-supervisor"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_duplicate_append_is_idempotent_while_pending(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()

        first = await world.enqueue_sampling("component-duplicate")
        second = await world.enqueue_sampling("component-duplicate")

        assert first.scheduler_result.ok is True
        assert not first.scheduler_result.idempotent
        assert second.scheduler_result.ok is True
        assert second.scheduler_result.idempotent is True

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
            lease=_token(lease, consumer_id="stale-consumer", consumer_generation=world.generation),
        )
        stale_fail = await world.scheduler.fail(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation + 1),
            requeue=False,
            reason="stale",
        )

        assert stale_finalize.ok is False and stale_finalize.reason == "stale_consumer"
        assert stale_fail.ok is False and stale_fail.reason == "stale_consumer"
        assert (
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        record = await world.observe_task("component-stale-consumer")
        assert record["status"] == "leased"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_gpu_actor_died_fences_consumer_until_generation_bump(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-gpu-died-fence-a")
        lease = await world.claim_one()

        failed = await world.scheduler.fail(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="gpu_actor_died",
        )
        await world.enqueue_sampling("component-gpu-died-fence-b")
        fenced_claim = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        next_generation = world.generation + 1
        next_consumer_id = world.replica(generation=next_generation)["consumer_id"]
        await world.scheduler.sync_replicas([world.replica(generation=next_generation, status="healthy")])
        recovered_claim = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=next_consumer_id,
            consumer_generation=next_generation,
            max_items=1,
            lease_ttl_s=30.0,
        )
        stats = await world.scheduler.stats()

        assert failed.ok is True and failed.requeued is True
        assert fenced_claim.ok is False
        assert fenced_claim.reason == "stale_consumer"
        assert len(recovered_claim.leases) == 1
        assert recovered_claim.leases[0]["consumer_id"] == next_consumer_id
        assert stats["self_failed_consumers"] == []
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
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
            staged_payload_path=str(world.tmp_path / "staged.json"),
        )

        premature_complete = await world.scheduler.complete(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        assert begin.ok is True
        assert premature_complete.ok is False and premature_complete.reason == "not_terminal"
        assert (
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
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
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        premature_fail = await world.scheduler.fail(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=False,
            reason="premature-terminal-fail",
        )

        assert begin.ok is True
        assert premature_fail.ok is False and premature_fail.reason == "not_terminal"
        assert (
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_task("component-fail-before-commit"))["status"] == "finalizing"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_fail_requeue_rejects_durable_finalizing_lease(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-fail-requeue-finalizing"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        failed = await world.scheduler.fail(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="ordinary-fail-after-finalize",
        )

        assert begin.ok is True
        assert failed.ok is False and failed.reason == "finalize_in_progress"
        assert (await world.observe_task(request_id))["status"] == "finalizing"
        assert (
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_fail_requeue_rejects_finalizing_after_local_ttl_drift(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-fail-requeue-finalizing-ttl-drift"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        block = world.faults.block("task_state.begin_finalize")
        finalize_task = asyncio.create_task(
            world.scheduler.begin_finalize(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=1.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        await asyncio.sleep(1.05)
        block.release.set()
        begin = await finalize_task

        failed = await world.scheduler.fail(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="ordinary-fail-after-local-ttl-drift",
        )

        assert begin.ok is True
        assert failed.ok is False and failed.reason == "finalize_in_progress"
        assert (await world.observe_task(request_id))["status"] == "finalizing"
        assert (
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_fail_requeue_recovers_after_durable_finalize_ttl_expires(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-fail-requeue-finalizing-expired"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=1.0,
        )

        await asyncio.sleep(1.05)
        failed = await world.scheduler.fail(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="recover-expired-finalize",
        )
        assigned = await world.scheduler.assign_pending(max_items=1)
        reclaimed = await world.claim_one()

        assert begin.ok is True
        assert failed.ok is True and failed.request_id == request_id and failed.requeued is True
        assert assigned.assigned == 1
        assert reclaimed["item"]["request_id"] == request_id
        assert str(reclaimed["lease_id"]) != str(lease["lease_id"])
        assert (await world.observe_task(request_id))["status"] == "leased"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_finish_success_requires_begin_finalize(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-finish-success-before-finalize"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        finished = await world.scheduler.finish_success(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            result_path=str(world.tmp_path / "result.json"),
        )

        assert finished.ok is False and finished.reason == "not_finalizing"
        assert (await world.observe_task(request_id))["status"] == "leased"
        assert (
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_finish_success_commits_terminal_and_releases_projection(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-finish-success"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
            staged_payload_path=str(world.tmp_path / "component-finish-success.json"),
        )

        finished = await world.scheduler.finish_success(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            result_path=str(world.tmp_path / "component-finish-success.json"),
            result_checksum="sha256:abc",
            result_size_bytes=123,
            billing_observations=[{"tokens": 7}],
        )

        record = await world.observe_task(request_id)
        assert begin.ok is True
        assert finished.ok is True and finished.request_id == request_id and finished.status == "done"
        assert record["status"] == "done"
        assert record["result_path"].endswith("component-finish-success.json")
        assert record["result_checksum"] == "sha256:abc"
        assert record["result_size_bytes"] == 123
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_finish_success_preserves_absent_result_metadata(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-finish-success-no-meta"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
            staged_payload_path=str(world.tmp_path / "component-finish-success-no-meta.json"),
        )

        finished = await world.scheduler.finish_success(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            result_path=str(world.tmp_path / "component-finish-success-no-meta.json"),
        )

        record = await world.observe_task(request_id)
        assert begin.ok is True
        assert finished.ok is True and finished.request_id == request_id and finished.status == "done"
        assert record["status"] == "done"
        assert record["result_checksum"] is None
        assert record["result_size_bytes"] is None
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_finish_failure_commits_terminal_and_releases_projection(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-finish-failure"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        finished = await world.scheduler.finish_failure(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            error="runtime failed",
        )

        record = await world.observe_task(request_id)
        assert begin.ok is True
        assert finished.ok is True and finished.request_id == request_id and finished.status == "failed"
        assert record["status"] == "failed"
        assert record["error"] == "runtime failed"
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_finish_cancel_after_durable_commit_releases_projection(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-finish-cancel"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        def _cancel_after_commit(**_kwargs):
            return asyncio.CancelledError()

        world.faults.fail_next("task_state.commit_finalize_success.after", _cancel_after_commit)

        with pytest.raises(asyncio.CancelledError):
            await world.scheduler.finish_success(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                result_path=str(world.tmp_path / "component-finish-cancel.json"),
                result_checksum="sha256:abc",
                result_size_bytes=123,
            )

        assert begin.ok is True
        assert (await world.observe_task(request_id))["status"] == "done"
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_duplicate_append_cancel_does_not_forget_existing_pending(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
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
    world = SchedulerComponentWorld(tmp_path)
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
async def test_scheduler_component_complete_defers_while_begin_finalize_is_inflight(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-complete-during-finalize")
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize")
        finalize_task = asyncio.create_task(
            world.scheduler.begin_finalize(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=30.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        completed = await world.scheduler.complete(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        block.release.set()
        finalized = await finalize_task

        assert completed.ok is False and completed.reason == "finalize_inflight"
        assert finalized.ok is True
        assert (
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
        assert committed["ok"] is True
        assert expired.ok is True and expired.expired == 0
        assert new_lease["item"]["request_id"] == request_id
        assert new_lease["lease_id"] != old_lease["lease_id"]
        assert stale_complete.ok is False and stale_complete.reason == "stale_consumer"
        assert (
            await world.scheduler.validate(
                lease=_token(new_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                    lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                    finalize_ttl_s=30.0,
                )
            )
            await asyncio.wait_for(block.entered.wait(), timeout=1.0)

            failed = await world.scheduler.fail(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=requeue,
                reason="fail-during-finalize",
            )

            block.release.set()
            finalized = await finalize_task

            assert failed.ok is False and failed.reason == "finalize_inflight"
            assert finalized.ok is True
            assert (
                await world.scheduler.validate(
                    lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=_token(first_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )
        stale_complete = await world.scheduler.complete(
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        valid_new = await world.scheduler.validate(
            lease=_token(new_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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

        takeover = await world.acquire_owner(
            owner_id="component-scheduler-restarted",
            ttl_s=30.0,
            now=time.time() + 31.0,
        )
        world.replace_scheduler(owner_id="component-scheduler-restarted")
        synced = await world.scheduler.sync_replicas([world.replica(status="healthy")])

        with pytest.raises(Exception, match="owner_active"):
            await old_scheduler.renew(
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                lease_ttl_s=30.0,
            )
        assigned = await world.scheduler.assign_pending(max_items=1)
        new_lease = await world.claim_one()

        assert takeover["ok"] is True
        assert takeover["epoch"] == 2
        assert synced.assigned["assigned"] == 1 or assigned.assigned == 1
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
        created = await world.task_state.async_create_task(
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
        )

        before = await world.observe_scheduler(request_id)
        reaped = await world.scheduler.reap_lost_pending_tasks(reason="component-test-reaper")
        stats = await world.scheduler.stats()
        after = await world.observe_scheduler(request_id)
        assigned = await world.scheduler.assign_pending(max_items=1)
        await world.runtime_once()

        assert created["created"] is True
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
            lease=_token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            lease_ttl_s=30.0,
        )
        completed = await world.scheduler.complete(
            lease=_token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
        )
        failed = await world.scheduler.fail(
            lease=_token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
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
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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

        assert len(claimed.leases) == 1
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

        assert assigned.assigned == 1
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

        finalized = await _assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.begin_finalize(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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

        finalized = await _assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.begin_finalize(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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

        failed = await _assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.fail(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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

        failed = await _assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.fail(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                requeue=True,
                reason="direct-fail-requeue-error",
            )

        assert (
            await world.scheduler.validate(
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"
        assert (await world.observe_task(request_id))["status"] == "leased"

        retried = await world.scheduler.fail(
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="direct-fail-requeue-missing",
        )

        assert failed.ok is True and failed.request_id == request_id and failed.requeued is False
        validate = await world.scheduler.validate(
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"
        assert (await world.observe_task(request_id))["status"] == "leased"

        retried = await world.scheduler.fail(
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
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
async def test_scheduler_component_assign_cancellation_restores_backlog(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
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
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-claim-cancel"
        await world.enqueue_sampling(request_id)

        block = world.faults.block("task_state.claim_task")
        claim_task = asyncio.create_task(
            world.scheduler.claim(
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
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-finalize-cancel"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize")
        finalize_task = asyncio.create_task(
            world.scheduler.begin_finalize(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True

        retried = await world.scheduler.begin_finalize(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )
        assert retried.ok is True
        assert (await world.observe_task(request_id))["status"] == "finalizing"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_sync_cancellation_preserves_replica_registry_and_lease(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
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
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
            )
        ).ok is True
        assert (await world.observe_scheduler(request_id)).location == "leased"
        await world.enqueue_sampling("component-sync-cancel-after", assign=False)
        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.scheduler.claim(
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
    world = SchedulerComponentWorld(tmp_path)
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

        validate = await world.scheduler.validate(
            lease=_token(old_lease, consumer_id=old_consumer_id, consumer_generation=old_generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).location == "backlog"
        assert (await world.observe_task(request_id))["status"] == "pending"
        with pytest.raises(Exception, match="consumer_id mismatch|generation mismatch"):
            await world.scheduler.claim(
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
    world = SchedulerComponentWorld(tmp_path)
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
            await world.scheduler.validate(
                lease=_token(old_lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=_token(first, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
async def test_scheduler_component_retrieve_pending_survives_scheduler_restart(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-orphan"
        await world.enqueue_sampling(request_id, assign=False)
        await world.acquire_owner(owner_id="component-scheduler-orphan-probe", now=time.time() + 31.0)
        world.replace_scheduler(owner_id="component-scheduler-orphan-probe")

        status_code, payload = await world.retrieve(request_id, monkeypatch)

        assert status_code == 408
        assert payload["request_id"] == request_id
        assert payload["type"] == "try_again"
        assert payload["status"] == "queued"
        assert payload["queue_kind"] == "model_work_scheduler"
        assert await world.observe_future_status(request_id) == FutureStatus.PENDING
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_orphan_pending_waits_for_reaper(tmp_path, monkeypatch) -> None:
    class AbsentScheduler:
        async def contains_request(
            self,
            *,
            request_id: str,
            hydrate_task_state: bool = True,
            timeout_s: float | None = None,
        ) -> dict[str, object]:
            _ = request_id, hydrate_task_state, timeout_s
            return {"ok": True, "present": False}

        async def contains(self, **kwargs) -> dict[str, object]:
            return await self.contains_request(**kwargs)

    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-orphan-pending"
        await world.enqueue_sampling(request_id, assign=False)

        status_code, payload = await world.retrieve(
            request_id,
            monkeypatch,
            scheduler_override=AbsentScheduler(),
        )

        assert status_code == 408
        assert payload["request_id"] == request_id
        assert payload["type"] == "try_again"
        assert payload["queue_kind"] == "model_work_scheduler"
        assert await world.observe_future_status(request_id) == FutureStatus.PENDING
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_wait_returns_terminal_result(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-wait-terminal"
        await world.enqueue_sampling(request_id)

        async def _complete_later() -> None:
            await asyncio.sleep(0.01)
            await world.runtime_once()

        completion = asyncio.create_task(_complete_later())
        status_code, payload = await world.retrieve(
            request_id,
            monkeypatch,
            wait_timeout_s=0.2,
        )
        await completion

        assert status_code == 200
        assert payload == {"ok": True, "request_id": request_id}
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_masks_internal_error_for_non_admin(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-mask-error"
        await world.enqueue_sampling(request_id)

        async def _failing_executor(_lease: dict) -> ExecutorOutcome:
            return ExecutorOutcome(kind="user_error", error="internal gpu traceback secret")

        await world.runtime_once(executor=_failing_executor)

        admin_status, admin_payload = await world.retrieve(request_id, monkeypatch, admin=True)
        user_status, user_payload = await world.retrieve(request_id, monkeypatch, admin=False)

        assert admin_status == 200
        assert "internal gpu traceback secret" in admin_payload["error"]
        assert user_status == 200
        assert user_payload["error"] == "Operation failed. Contact administrator if issue persists."
        assert user_payload["category"] == "system"
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
async def test_scheduler_component_cancel_assigned_work_removes_scheduler_projection(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
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
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_cancel_assigned_work_survives_scheduler_restart(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
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
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-cancel-leased"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        cancelled = await world.cancel(request_id, monkeypatch, reason="component-test")
        validate = await world.scheduler.validate(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
    world = SchedulerComponentWorld(tmp_path)
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
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is False and (await world.scheduler.validate(lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation))).reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).present is False
        assert failed_status == FutureStatus.FAILED
        assert status_code == 200
        assert "cancelled" in payload["error"]
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_assign_cancellation_after_durable_commit_preserves_claimability(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
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
    world = SchedulerComponentWorld(tmp_path)
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
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-claim-cancel-after"
        await world.enqueue_sampling(request_id)

        block = world.faults.block("task_state.claim_task.after")
        claim_task = asyncio.create_task(
            world.scheduler.claim(
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
            await world.scheduler.validate(
                lease=_token(task, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_begin_finalize_cancellation_after_durable_commit_preserves_finalizing(
    tmp_path,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-finalize-cancel-after"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()

        block = world.faults.block("task_state.begin_finalize.after")
        finalize_task = asyncio.create_task(
            world.scheduler.begin_finalize(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            await world.scheduler.validate(
                lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            )
        ).ok is True
        complete = await world.scheduler.complete(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        assert complete.ok is False and complete.reason == "not_terminal"
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
        completed = await world.scheduler.complete(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        assert committed["ok"] is True
        assert completed.ok is True and completed.request_id == request_id
        assert (await world.observe_scheduler(request_id)).present is False
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_expire_cancellation_after_requeue_commit_preserves_backlog(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
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
    world = SchedulerComponentWorld(tmp_path)
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
    world = SchedulerComponentWorld(tmp_path)
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
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            lease_ttl_s=30.0,
        )

        assert renewed.ok is False and renewed.reason == "unknown_lease"
        assert (await world.observe_scheduler(request_id)).present is False
        validate = await world.scheduler.validate(
            lease=_token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )
        assert validate.ok is False and validate.reason == "unknown_lease"
    finally:
        world.close()
