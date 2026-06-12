from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from mint_server.backend.model_work_admission import ModelWorkAdmissionRejectedError

from .helpers import (
    assert_scheduler_surfaces_progress_while_blocked,
    assert_stats_progress_while_blocked,
    finish_success_for_test,
    token,
)
from .harness import SchedulerComponentWorld
from .invariants import (
    assert_lease_consistency,
    assert_no_double_lease,
    assert_no_orphan_assigned,
    assert_terminal_not_scheduled,
)


pytestmark = pytest.mark.component



@pytest.mark.anyio
async def test_scheduler_component_blocked_claim_task_does_not_block_stats(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-claim")

        claimed = await assert_stats_progress_while_blocked(
            world,
            lambda: world.runtime_queue.claim(
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-blocked-assign", assign=False)

        assigned = await assert_stats_progress_while_blocked(
            world,
            lambda: world.scheduler.assign_pending(max_items=1),
            "task_state.assign_task",
        )

        assert assigned.assigned == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_assign_task_does_not_block_scheduler_surfaces(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-progress-probe", assign=False)

        assigned = await assert_scheduler_surfaces_progress_while_blocked(
            world,
            lambda: world.scheduler.assign_pending(max_items=1),
            "task_state.assign_task",
        )

        assert assigned.assigned == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_blocked_assign_task_keeps_claim_nonblocking_and_empty(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-assigning-not-claimable", assign=False)

        block = world.faults.block("task_state.assign_task")
        assign_task = asyncio.create_task(world.scheduler.assign_pending(max_items=1))
        await asyncio.wait_for(block.entered.wait(), timeout=1.0)

        claimed = await asyncio.wait_for(
            world.runtime_queue.claim(
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
async def test_scheduler_component_generation_bump_drains_assigned_unleased_work(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-generation-bump-assigned-drain"
        await world.enqueue_sampling(request_id, assign=False)
        assigned = await world.scheduler.assign_pending(max_items=1)

        old_record = await world.observe_task(request_id)
        next_generation = world.generation + 1
        next_consumer_id = world.replica(generation=next_generation)["consumer_id"]
        synced = await world.scheduler.sync_replicas(
            [world.replica(status="healthy", generation=next_generation)]
        )
        lease = await world.claim_one(
            consumer_id=next_consumer_id,
            consumer_generation=next_generation,
        )
        begin = await world.runtime_queue.begin_finalize(
            lease=token(lease, consumer_id=next_consumer_id, consumer_generation=next_generation)
        )
        finished = await world.runtime_queue.finish_success(
            lease=token(lease, consumer_id=next_consumer_id, consumer_generation=next_generation),
            result_path=str(world.tmp_path / "generation-bump-result.json"),
        )
        record = await world.observe_task(request_id)

        assert assigned.assigned == 1
        assert old_record["status"] == "assigned"
        assert synced.requeued == 1
        assert synced.assigned["assigned"] == 1
        assert lease["item"]["request_id"] == request_id
        assert lease["consumer_id"] == next_consumer_id
        assert begin.ok is True
        assert finished.ok is True
        assert record["status"] == "done"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_affinity_sticks_to_same_replica_surface(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
        empty_a = await world.runtime_queue.claim(
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
async def test_scheduler_component_replica_capacity_exhaustion_leaves_extra_work_in_backlog(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.scheduler.sync_replicas(
            [
                world.replica(replica_id="replica-a", status="healthy", capacity=1),
                world.replica(replica_id="replica-b", status="healthy", capacity=1),
            ]
        )
        await world.enqueue_sampling("component-capacity-a", assign=False, affinity_group="lora:a")
        await world.enqueue_sampling("component-capacity-b", assign=False, affinity_group="lora:b")
        await world.enqueue_sampling("component-capacity-c", assign=False, affinity_group="lora:c")

        assigned = await world.scheduler.assign_pending(max_items=3)
        lease_a = await world.claim_one(replica_id="replica-a")
        lease_b = await world.claim_one(replica_id="replica-b")
        empty_a = await world.runtime_queue.claim(
            domain_key=world.domain_key,
            replica_id="replica-a",
            consumer_id=world.replica(replica_id="replica-a")["consumer_id"],
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )
        empty_b = await world.runtime_queue.claim(
            domain_key=world.domain_key,
            replica_id="replica-b",
            consumer_id=world.replica(replica_id="replica-b")["consumer_id"],
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )
        remaining = cast(Any, await world.observe_scheduler("component-capacity-c"))

        assert assigned.assigned == 2
        assert sorted(
            [
                lease_a["item"]["request_id"],
                lease_b["item"]["request_id"],
            ]
        ) == ["component-capacity-a", "component-capacity-b"]
        assert empty_a.leases == []
        assert empty_b.leases == []
        assert remaining.present is True
        assert remaining.location == "backlog"
        await assert_no_orphan_assigned(world)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_concurrent_multi_replica_claims_do_not_duplicate_work(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.scheduler.sync_replicas(
            [
                world.replica(replica_id="replica-a", status="healthy", capacity=1),
                world.replica(replica_id="replica-b", status="healthy", capacity=1),
            ]
        )
        await world.enqueue_sampling("component-concurrent-claim-a", assign=False, affinity_group="lora:a")
        await world.enqueue_sampling("component-concurrent-claim-b", assign=False, affinity_group="lora:b")
        assigned = await world.scheduler.assign_pending(max_items=2)

        claim_a, claim_b = await asyncio.gather(
            world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id="replica-a",
                consumer_id=world.replica(replica_id="replica-a")["consumer_id"],
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            ),
            world.runtime_queue.claim(
                domain_key=world.domain_key,
                replica_id="replica-b",
                consumer_id=world.replica(replica_id="replica-b")["consumer_id"],
                consumer_generation=world.generation,
                max_items=1,
                lease_ttl_s=30.0,
            ),
        )

        leases = [*claim_a.leases, *claim_b.leases]
        request_ids = sorted(lease["item"]["request_id"] for lease in leases)

        assert assigned.assigned == 2
        assert len(claim_a.leases) == 1
        assert len(claim_b.leases) == 1
        assert request_ids == [
            "component-concurrent-claim-a",
            "component-concurrent-claim-b",
        ]
        assert len({lease["lease_id"] for lease in leases}) == 2
        for lease in leases:
            record = await world.observe_task(lease["item"]["request_id"])
            assert record["status"] == "leased"
            assert record["lease_id"] == lease["lease_id"]
            assert record["attempt_id"] == lease["attempt_id"]
        await assert_no_double_lease(world)
        await assert_lease_consistency(world)
        await assert_no_orphan_assigned(world)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_claims_first_item_when_it_exceeds_token_budget_surface(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-expensive-first", assign=False, token_cost=100)
        await world.enqueue_sampling("component-cheap-next", assign=False, token_cost=1)
        assigned = await world.scheduler.assign_pending(max_items=2)

        claimed = await world.runtime_queue.claim(
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
        blocked = await world.runtime_queue.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert first["item"]["request_id"] == "component-session-serial-a"
        assert blocked.leases == []

        begin = await world.runtime_queue.begin_finalize(
            lease=token(first, consumer_id=world.consumer_id, consumer_generation=world.generation),
            finalize_ttl_s=30.0,
        )
        assert begin.ok is True
        assert (await finish_success_for_test(world, first)).ok is True

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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path, domain_key="megatron:glm-5"))
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
        blocked = await world.runtime_queue.claim(
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
async def test_scheduler_component_sampling_admission_observe_allows_would_reject(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINT_SAMPLING_INFLIGHT_ADMISSION_MODE", "observe")
    monkeypatch.setenv("MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN", "1")
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
    world = cast(Any, SchedulerComponentWorld(tmp_path))
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
