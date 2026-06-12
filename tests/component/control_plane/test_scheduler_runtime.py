from __future__ import annotations

from typing import Any, cast

import pytest

from mint_server.backend.control_plane_contracts import ExecutorOutcome
from mint_server.backend.task_state_store import FutureStatus

from .helpers import token
from .harness import SchedulerComponentWorld
from .invariants import assert_terminal_not_scheduled


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_complete_cleans_scheduler_lease_for_missing_task(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
async def test_scheduler_component_executor_failure_commits_failed_terminal(tmp_path, monkeypatch) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-exec-failed")

        async def _failing_executor(_lease: dict[str, Any]) -> ExecutorOutcome:
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
