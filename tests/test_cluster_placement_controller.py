from __future__ import annotations

import asyncio

import pytest

from mint_server.backend.cluster_placement_controller import (
    ClusterPlacementController,
    PlacementBlockReason,
    PlacementGroupCreateRequest,
    PlacementGroupCreateStatus,
    PlacementReservationRequest,
    PlacementReservationStatus,
)


class _FakePlacementGroup:
    def __init__(self, *, ready_delay_s: float = 0.0) -> None:
        self.ready_delay_s = float(ready_delay_s)
        self.ready_calls = 0

    async def ready(self) -> object:
        self.ready_calls += 1
        await asyncio.sleep(self.ready_delay_s)
        return self


@pytest.mark.anyio
async def test_cluster_placement_controller_reserves_capacity_atomically() -> None:
    controller = ClusterPlacementController(
        namespace="mint",
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 4},
    )
    request_a = PlacementReservationRequest.from_mapping(
        replica_key="vllm:model-a::replica-0",
        required_gpus_by_node={"10.0.0.7": 4},
        placement_group_name="mint_model_runtime_vllm-a_replica-0_pg",
    )
    request_b = PlacementReservationRequest.from_mapping(
        replica_key="vllm:model-b::replica-0",
        required_gpus_by_node={"10.0.0.7": 4},
        placement_group_name="mint_model_runtime_vllm-b_replica-0_pg",
    )

    first, second = await asyncio.gather(controller.reserve(request_a), controller.reserve(request_b))

    statuses = {first.status, second.status}
    assert statuses == {PlacementReservationStatus.RESERVED, PlacementReservationStatus.BLOCKED}
    blocked = first if first.status == PlacementReservationStatus.BLOCKED else second
    assert blocked.reason is PlacementBlockReason.INSUFFICIENT_GPU

    snapshot = await controller.snapshot()
    assert snapshot.active_reservation_count == 1
    assert snapshot.in_flight_gpus_by_node == (("10.0.0.7", 4),)
    assert snapshot.available_gpus_by_node == (("10.0.0.7", 0),)


@pytest.mark.anyio
async def test_cluster_placement_controller_releases_capacity_for_followup_reservation() -> None:
    controller = ClusterPlacementController(
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 4},
    )
    request = PlacementReservationRequest.from_mapping(
        replica_key="dense:model-a::replica-0",
        required_gpus_by_node={"10.0.0.7": 4},
    )
    first = await controller.reserve(request)
    assert first.status is PlacementReservationStatus.RESERVED
    assert first.token is not None

    second = await controller.reserve(
        PlacementReservationRequest.from_mapping(
            replica_key="dense:model-b::replica-0",
            required_gpus_by_node={"10.0.0.7": 1},
        )
    )
    assert second.status is PlacementReservationStatus.BLOCKED

    released = await controller.release(first.token)
    assert released.status is PlacementReservationStatus.RELEASED

    third = await controller.reserve(
        PlacementReservationRequest.from_mapping(
            replica_key="dense:model-b::replica-0",
            required_gpus_by_node={"10.0.0.7": 1},
        )
    )
    assert third.status is PlacementReservationStatus.RESERVED
    snapshot = await controller.snapshot()
    assert snapshot.in_flight_gpus_by_node == (("10.0.0.7", 1),)
    assert snapshot.available_gpus_by_node == (("10.0.0.7", 3),)


@pytest.mark.anyio
async def test_cluster_placement_controller_rebuilds_existing_pg_occupancy_from_table() -> None:
    controller = ClusterPlacementController(
        namespace="mint",
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 4, "10.0.0.8": 2},
        placement_group_table=lambda: {
            "alive": {
                "name": "mint_model_runtime_vllm-a_replica-0_pg",
                "namespace": "mint",
                "state": "CREATED",
                "bundles": {
                    "0": {"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001},
                    "1": {"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001},
                },
            },
            "foreign": {
                "name": "other_pg",
                "namespace": "other",
                "state": "CREATED",
                "bundles": {"0": {"GPU": 1, "CPU": 1, "node:10.0.0.8": 0.001}},
            },
            "removed": {
                "name": "removed_pg",
                "namespace": "mint",
                "state": "REMOVED",
                "bundles": {"0": {"GPU": 1, "CPU": 1, "node:10.0.0.8": 0.001}},
            },
        },
    )

    snapshot = await controller.rebuild_from_placement_group_table()

    assert snapshot.rebuilt_gpus_by_node == (("10.0.0.7", 2),)
    assert snapshot.available_gpus_by_node == (("10.0.0.7", 2), ("10.0.0.8", 2))

    reserved = await controller.reserve(
        PlacementReservationRequest.from_mapping(
            replica_key="vllm:model-b::replica-0",
            required_gpus_by_node={"10.0.0.7": 3},
        )
    )
    assert reserved.status is PlacementReservationStatus.BLOCKED
    assert reserved.reason is PlacementBlockReason.INSUFFICIENT_GPU

    accepted = await controller.reserve(
        PlacementReservationRequest.from_mapping(
            replica_key="vllm:model-b::replica-0",
            required_gpus_by_node={"10.0.0.7": 2},
        )
    )
    assert accepted.status is PlacementReservationStatus.RESERVED


@pytest.mark.anyio
async def test_cluster_placement_controller_times_out_pending_pg_and_blocks_with_backoff() -> None:
    created: list[dict[str, object]] = []
    removed: list[_FakePlacementGroup] = []

    async def _create_pg(**kwargs) -> _FakePlacementGroup:
        created.append(dict(kwargs))
        return _FakePlacementGroup(ready_delay_s=1.0)

    controller = ClusterPlacementController(
        namespace="mint",
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 4},
        placement_group_factory=_create_pg,
        placement_group_remover=lambda pg: removed.append(pg),
        monotonic=lambda: 100.0,
        initial_backoff_s=5.0,
        max_backoff_s=60.0,
    )

    result = await controller.create_pg(
        PlacementGroupCreateRequest.from_mapping(
            replica_key="vllm:model-a::replica-0",
            placement_group_name="mint_model_runtime_vllm-a_replica-0_pg",
            required_gpus_by_node={"10.0.0.7": 4},
            bundles=({"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001},) * 4,
            ready_timeout_s=0.01,
        )
    )

    assert result.status is PlacementGroupCreateStatus.BLOCKED
    assert result.reason is PlacementBlockReason.PG_PENDING_TIMEOUT
    assert result.placement_group_name == "mint_model_runtime_vllm-a_replica-0_pg"
    assert len(created) == 1
    assert len(removed) == 1
    snapshot = await controller.snapshot()
    assert snapshot.active_reservation_count == 0
    assert snapshot.in_flight_gpus_by_node == ()
    assert snapshot.blocked_replicas == (
        ("vllm:model-a::replica-0", PlacementBlockReason.PG_PENDING_TIMEOUT, 105.0),
    )


@pytest.mark.anyio
async def test_cluster_placement_controller_recovers_blocked_replica_after_backoff() -> None:
    now = 200.0
    created: list[dict[str, object]] = []
    removed: list[_FakePlacementGroup] = []
    delays = [1.0, 0.0]

    async def _create_pg(**kwargs) -> _FakePlacementGroup:
        created.append(dict(kwargs))
        return _FakePlacementGroup(ready_delay_s=delays.pop(0))

    controller = ClusterPlacementController(
        namespace="mint",
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 2},
        placement_group_factory=_create_pg,
        placement_group_remover=lambda pg: removed.append(pg),
        monotonic=lambda: now,
        initial_backoff_s=5.0,
        max_backoff_s=60.0,
    )
    request = PlacementGroupCreateRequest.from_mapping(
        replica_key="dense:model-a::replica-0",
        placement_group_name="mint_dense_a_pg",
        required_gpus_by_node={"10.0.0.7": 1},
        bundles=({"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001},),
        ready_timeout_s=0.01,
    )

    blocked = await controller.create_pg(request)
    assert blocked.status is PlacementGroupCreateStatus.BLOCKED

    still_blocked = await controller.create_pg(request)
    assert still_blocked.status is PlacementGroupCreateStatus.BLOCKED
    assert still_blocked.reason is PlacementBlockReason.BACKOFF_ACTIVE
    assert len(created) == 1

    now = 205.0
    recovered = await controller.create_pg(request)

    assert recovered.status is PlacementGroupCreateStatus.READY
    assert recovered.placement_group is not None
    assert recovered.placement_group_name == "mint_dense_a_pg"
    assert len(created) == 2
    assert len(removed) == 1
    snapshot = await controller.snapshot()
    assert snapshot.blocked_replicas == ()
    assert snapshot.active_reservation_count == 0
