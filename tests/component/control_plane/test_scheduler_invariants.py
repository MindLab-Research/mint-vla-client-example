from __future__ import annotations

import pytest

from .helpers import token
from .harness import SchedulerComponentWorld
from .invariants import (
    assert_every_terminal_has_payload_ref,
    assert_lease_consistency,
    assert_no_double_lease,
    assert_no_orphan_assigned,
    assert_terminal_not_scheduled,
)


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_invariant_helpers_cover_happy_path(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-invariants")

        lease = await world.claim_one()
        await assert_no_double_lease(world)
        await assert_lease_consistency(world)
        await assert_no_orphan_assigned(world)

        begin = await world.scheduler.begin_finalize(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            staged_payload_path=str(world.tmp_path / "component-invariants.json"),
            finalize_ttl_s=30.0,
        )
        assert begin.ok is True
        finished = await world.scheduler.finish_success(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            result_path=str(world.tmp_path / "component-invariants.json"),
            result_checksum="checksum",
            result_size_bytes=17,
        )
        assert finished.ok is True

        await assert_every_terminal_has_payload_ref(world)
        await assert_terminal_not_scheduled(world, "component-invariants")
    finally:
        world.close()
