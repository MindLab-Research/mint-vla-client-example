from __future__ import annotations

import asyncio

import pytest

from .harness import SchedulerComponentWorld
from .helpers import token
from .invariants import (
    assert_every_terminal_has_payload_ref,
    assert_no_double_lease,
    assert_terminal_not_scheduled,
)


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_complete_requires_durable_terminal_commit(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-complete-before-commit")
        lease = await world.claim_one()
        begin = await world.scheduler.begin_finalize(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
            staged_payload_path=str(world.tmp_path / "staged.json"),
        )

        premature_complete = await world.scheduler.complete(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        )

        assert begin.ok is True
        assert premature_complete.ok is False and premature_complete.reason == "not_terminal"
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        premature_fail = await world.scheduler.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=False,
            reason="premature-terminal-fail",
        )

        assert begin.ok is True
        assert premature_fail.ok is False and premature_fail.reason == "not_terminal"
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        failed = await world.scheduler.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="ordinary-fail-after-finalize",
        )

        assert begin.ok is True
        assert failed.ok is False and failed.reason == "finalize_in_progress"
        assert (await world.observe_task(request_id))["status"] == "finalizing"
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                finalize_ttl_s=1.0,
            )
        )
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)
        await asyncio.sleep(1.05)
        block.release.set()
        begin = await finalize_task

        failed = await world.scheduler.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=True,
            reason="ordinary-fail-after-local-ttl-drift",
        )

        assert begin.ok is True
        assert failed.ok is False and failed.reason == "finalize_in_progress"
        assert (await world.observe_task(request_id))["status"] == "finalizing"
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=1.0,
        )

        await asyncio.sleep(1.05)
        failed = await world.scheduler.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            result_path=str(world.tmp_path / "result.json"),
        )

        assert finished.ok is False and finished.reason == "not_finalizing"
        assert (await world.observe_task(request_id))["status"] == "leased"
        assert (
            await world.scheduler.validate(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
            staged_payload_path=str(world.tmp_path / "component-finish-success.json"),
        )

        finished = await world.scheduler.finish_success(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
        await assert_every_terminal_has_payload_ref(world)
        await assert_no_double_lease(world)
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
            staged_payload_path=str(world.tmp_path / "component-finish-success-no-meta.json"),
        )

        finished = await world.scheduler.finish_success(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        finished = await world.scheduler.finish_failure(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
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
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )

        def _cancel_after_commit(**_kwargs):
            return asyncio.CancelledError()

        world.faults.fail_next("task_state.commit_finalize_success.after", _cancel_after_commit)

        with pytest.raises(asyncio.CancelledError):
            await world.scheduler.finish_success(
                lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
                result_path=str(world.tmp_path / "component-finish-cancel.json"),
                result_checksum="sha256:abc",
                result_size_bytes=123,
            )

        assert begin.ok is True
        assert (await world.observe_task(request_id))["status"] == "done"
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()
