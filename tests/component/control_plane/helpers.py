from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any, cast

from mint_server.backend.contracts.control_plane_contracts import LeaseToken

from .harness import SchedulerComponentWorld


def token(
    lease: dict[str, Any],
    *,
    consumer_id: str | None = None,
    consumer_generation: int | None = None,
) -> LeaseToken:
    raw_item = lease.get("item")
    item = raw_item if isinstance(raw_item, dict) else {}
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


async def finish_success_for_test(
    world: SchedulerComponentWorld,
    lease: dict[str, Any],
    *,
    result_path: str | None = None,
) -> Any:
    return await world.runtime_queue.finish_success(
        lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
        result_path=result_path or str(world.tmp_path / "result.json"),
        result_checksum=None,
        result_size_bytes=None,
        billing_observations=None,
    )


async def assert_stats_progress_while_blocked(
    world: SchedulerComponentWorld,
    call: Callable[[], Coroutine[Any, Any, Any]],
    block_name: str,
) -> Any:
    block = world.faults.block(block_name)
    task = asyncio.create_task(call())
    await asyncio.wait_for(block.entered.wait(), timeout=1.0)

    stats = await asyncio.wait_for(world.scheduler.stats(), timeout=0.5)

    block.release.set()
    result = await task
    assert stats["scheduler_instance_id"]
    return result


async def assert_scheduler_surfaces_progress_while_blocked(
    world: SchedulerComponentWorld,
    call: Callable[[], Coroutine[Any, Any, Any]],
    block_name: str,
) -> Any:
    block = world.faults.block(block_name)
    task = asyncio.create_task(call())
    await asyncio.wait_for(block.entered.wait(), timeout=1.0)

    contains = cast(
        Any,
        await asyncio.wait_for(world.observe_scheduler("component-progress-probe"), timeout=0.5),
    )
    stats = await asyncio.wait_for(world.scheduler.stats(), timeout=0.5)
    appended = cast(
        Any,
        await asyncio.wait_for(
            world.enqueue_sampling(f"component-progress-{block_name}", assign=False),
            timeout=0.5,
        ),
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


async def wait_for_task_state_call_count(
    world: SchedulerComponentWorld,
    method: str,
    *,
    count: int,
    timeout_s: float = 0.5,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observed = sum(1 for call_method, _ in world.task_state.calls if call_method == method)
        if observed >= count:
            return
        await asyncio.sleep(0.005)
    observed = sum(1 for call_method, _ in world.task_state.calls if call_method == method)
    raise AssertionError(f"timed out waiting for {method} call count {count}; observed {observed}")
