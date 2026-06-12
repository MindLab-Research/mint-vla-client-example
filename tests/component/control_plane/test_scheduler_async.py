from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from .helpers import (
    assert_scheduler_surfaces_progress_while_blocked,
    assert_stats_progress_while_blocked,
    token,
    wait_for_task_state_call_count,
)
from .harness import SchedulerComponentWorld


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_owner_renew_rpc_does_not_hold_owner_lock(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
async def test_scheduler_component_retrieve_pending_during_blocked_finalize_does_not_block(
    tmp_path,
    monkeypatch,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
