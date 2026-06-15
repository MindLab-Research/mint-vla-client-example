from __future__ import annotations

import asyncio
import sys

import pytest

from mint_server.backend.scheduling.cluster_placement_controller import (
    ClusterPlacementController,
    PlacementBlockReason,
    PlacementGroupBundleRequest,
    PlacementGroupCreateRequest,
    PlacementGroupCreateStatus,
    PlacementReconcileRequest,
    PlacementReservationRequest,
    PlacementReservationStatus,
    placement_group_bundle_request_for_spec,
)
from mint_server.backend.actors.model_actor_supervisor import ModelActorSpec
from mint_server.backend.contracts.model_placement_topology import ParallelTopology


class _FakePlacementGroup:
    def __init__(self, *, ready_delay_s: float = 0.0) -> None:
        self.ready_delay_s = float(ready_delay_s)
        self.ready_calls = 0

    async def ready(self) -> object:
        self.ready_calls += 1
        await asyncio.sleep(self.ready_delay_s)
        return self


async def assert_reservation_account_consistent(
    controller: ClusterPlacementController,
    *,
    expected_active_reservations: int | None = None,
    expected_in_flight_gpus_by_node: tuple[tuple[str, int], ...] | None = None,
) -> None:
    snapshot = await controller.snapshot()
    if expected_active_reservations is not None:
        assert snapshot.active_reservation_count == expected_active_reservations
    if expected_in_flight_gpus_by_node is not None:
        assert snapshot.in_flight_gpus_by_node == expected_in_flight_gpus_by_node
    if snapshot.active_reservation_count == 0:
        assert snapshot.in_flight_gpus_by_node == ()
    if snapshot.in_flight_gpus_by_node:
        assert snapshot.active_reservation_count > 0
    for _node_ip, gpu_count in snapshot.in_flight_gpus_by_node:
        assert gpu_count > 0
    for _node_ip, gpu_count in snapshot.available_gpus_by_node:
        assert gpu_count >= 0
    for _node_ip, gpu_count in snapshot.rebuilt_gpus_by_node:
        assert gpu_count > 0
    for _replica_key, reason, retry_at in snapshot.blocked_replicas:
        assert isinstance(reason, PlacementBlockReason)
        assert retry_at > 0


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

    await assert_reservation_account_consistent(
        controller,
        expected_active_reservations=1,
        expected_in_flight_gpus_by_node=(("10.0.0.7", 4),),
    )
    assert (await controller.snapshot()).available_gpus_by_node == (("10.0.0.7", 0),)


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
    await assert_reservation_account_consistent(
        controller,
        expected_active_reservations=1,
        expected_in_flight_gpus_by_node=(("10.0.0.7", 1),),
    )
    assert (await controller.snapshot()).available_gpus_by_node == (("10.0.0.7", 3),)


@pytest.mark.anyio
async def test_cluster_placement_controller_reserves_unpinned_cluster_capacity() -> None:
    controller = ClusterPlacementController(
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 2, "10.0.0.8": 2},
    )
    request_a = PlacementReservationRequest.from_mapping(
        replica_key="vllm:model-a::replica-0",
        required_gpus_by_node={"__cluster__": 3},
    )
    request_b = PlacementReservationRequest.from_mapping(
        replica_key="vllm:model-b::replica-0",
        required_gpus_by_node={"__cluster__": 2},
    )

    first, second = await asyncio.gather(controller.reserve(request_a), controller.reserve(request_b))

    assert {first.status, second.status} == {
        PlacementReservationStatus.RESERVED,
        PlacementReservationStatus.BLOCKED,
    }
    await assert_reservation_account_consistent(
        controller,
        expected_active_reservations=1,
        expected_in_flight_gpus_by_node=(("__cluster__", 3),),
    )
    assert (await controller.snapshot()).available_gpus_by_node == (
        ("10.0.0.7", 0),
        ("10.0.0.8", 1),
    )


def test_cluster_placement_controller_builds_backend_attach_compatible_pg_requests() -> None:
    with pytest.raises(ValueError, match="does not require GPU placement"):
        placement_group_bundle_request_for_spec(
            ModelActorSpec(
                domain_key="internal:runtime",
                replica_id="replica-0",
                launcher_key="cpu_runtime",
                gpu_count=0,
            )
        )

    dense = placement_group_bundle_request_for_spec(
        ModelActorSpec(
            domain_key="dense:Qwen/Test",
            replica_id="replica-0",
            base_model="Qwen/Test",
            launcher_key="dense",
            node_pin="10.0.0.7",
            gpu_count=1,
        )
    )
    assert dense.placement_group_name == "mint_dense_qwen__test_mint_pg"
    assert dense.required_gpus_by_node == (("10.0.0.7", 1),)

    vllm = placement_group_bundle_request_for_spec(
        ModelActorSpec(
            domain_key="vllm:Qwen/Test",
            replica_id="replica-0",
            base_model="Qwen/Test",
            actor_name="mint_model_runtime_vllm-qwen-test_replica-0",
            launcher_key="vllm",
            node_pins=("10.0.0.8", "10.0.0.9"),
            gpu_count=2,
        )
    )
    assert vllm.placement_group_name == "mint_model_runtime_vllm-qwen-test_replica-0_pg"
    assert vllm.required_gpus_by_node == (("10.0.0.8", 1), ("10.0.0.9", 1))
    assert vllm.controller_bundle_index == 2

    megatron = placement_group_bundle_request_for_spec(
        ModelActorSpec(
            domain_key="megatron:Qwen/Qwen3-30B-A3B",
            replica_id="replica-0",
            base_model="Qwen/Qwen3-30B-A3B",
            launcher_key="megatron",
            placement_slices=(("replica-0", "10.0.0.10", 2),),
            gpu_count=2,
        )
    )
    assert megatron.placement_group_name == "mint_megatron_qwen3_30b_a3b_mint_pg"
    assert megatron.required_gpus_by_node == (("10.0.0.10", 2),)


def test_cluster_placement_create_request_uses_ray_compatible_list_bundles() -> None:
    request = PlacementGroupCreateRequest.from_mapping(
        replica_key="vllm:model-a::replica-0",
        placement_group_name="mint_vllm_a_pg",
        required_gpus_by_node={"10.0.0.7": 1},
        bundles=({"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001}, {"CPU": 1}),
    )

    assert isinstance(request.bundles, tuple)
    assert request.bundle_dicts == [
        {"CPU": 1, "GPU": 1, "node:10.0.0.7": 0.001},
        {"CPU": 1},
    ]
    assert isinstance(request.bundle_dicts, list)


@pytest.mark.anyio
async def test_cluster_placement_controller_reconcile_returns_node_pins_blocked_and_pg_requests() -> None:
    controller = ClusterPlacementController(
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 1},
    )
    desired = {
        ("internal:runtime", "replica-0"): ModelActorSpec(
            domain_key="internal:runtime",
            replica_id="replica-0",
            launcher_key="cpu_runtime",
            gpu_count=0,
        ),
        ("vllm:Qwen/Test", "replica-0"): ModelActorSpec(
            domain_key="vllm:Qwen/Test",
            replica_id="replica-0",
            base_model="Qwen/Test",
            actor_name="mint_model_runtime_vllm-qwen-test_replica-0",
            node_pin="10.0.0.7",
            gpu_count=1,
        )
    }

    result = await controller.reconcile(
        PlacementReconcileRequest(
            desired=desired,
            protected_actor_names=frozenset({"mint_model_runtime_vllm-qwen-test_replica-0"}),
        )
    )

    assert result.ok is True
    assert result.node_pins_by_label == {"vllm:Qwen/Test::replica-0": ["10.0.0.7"]}
    assert result.blocked_by_label == {}
    assert len(result.placement_group_requests) == 1
    assert result.placement_group_requests[0].placement_group_name == "mint_model_runtime_vllm-qwen-test_replica-0_pg"
    assert result.protected_actor_names == frozenset({"mint_model_runtime_vllm-qwen-test_replica-0"})


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
            "alive_unpinned": {
                "name": "mint_model_runtime_vllm-b_replica-0_pg",
                "namespace": "mint",
                "state": "CREATED",
                "bundles": {"0": {"GPU": 1, "CPU": 1}},
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

    assert snapshot.rebuilt_gpus_by_node == (("10.0.0.7", 2), ("__cluster__", 1))
    assert snapshot.available_gpus_by_node == (("10.0.0.7", 1), ("10.0.0.8", 2))

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
    assert accepted.status is PlacementReservationStatus.BLOCKED


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
    await assert_reservation_account_consistent(
        controller,
        expected_active_reservations=0,
        expected_in_flight_gpus_by_node=(),
    )
    assert (await controller.snapshot()).blocked_replicas == (
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
    await assert_reservation_account_consistent(
        controller,
        expected_active_reservations=0,
        expected_in_flight_gpus_by_node=(),
    )
    assert (await controller.snapshot()).blocked_replicas == ()


@pytest.mark.anyio
async def test_cluster_placement_controller_reuses_existing_named_pg_when_create_races() -> None:
    created: list[dict[str, object]] = []
    removed: list[object] = []
    existing_pg = _FakePlacementGroup()

    async def _create_pg(**kwargs):
        created.append(dict(kwargs))
        raise RuntimeError(
            "System error: Failed to create placement group "
            "'existing_pg' because name 'mint_vllm_a_pg' already exists."
        )

    controller = ClusterPlacementController(
        namespace="mint",
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 1},
        placement_group_table=lambda: {
            "existing_pg": {
                "name": "mint_vllm_a_pg",
                "namespace": "mint",
                "state": "CREATED",
                "placement_group": existing_pg,
                "bundles": {"0": {"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001}},
            }
        },
        placement_group_factory=_create_pg,
        placement_group_remover=lambda pg: removed.append(pg),
    )

    result = await controller.create_pg(
        PlacementGroupCreateRequest.from_mapping(
            replica_key="vllm:model-a::replica-0",
            placement_group_name="mint_vllm_a_pg",
            required_gpus_by_node={"10.0.0.7": 1},
            bundles=({"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001}, {"CPU": 1}),
        )
    )

    assert result.status is PlacementGroupCreateStatus.READY
    assert result.placement_group is existing_pg
    assert result.placement_group_name == "mint_vllm_a_pg"
    assert len(created) == 1
    assert removed == []
    await assert_reservation_account_consistent(
        controller,
        expected_active_reservations=0,
        expected_in_flight_gpus_by_node=(),
    )
    assert (await controller.snapshot()).blocked_replicas == ()


def test_cluster_placement_controller_computes_backend_bundle_requests() -> None:
    dense = PlacementGroupBundleRequest.for_dense(
        replica_key="dense:model-a::replica-0",
        placement_group_name="mint_dense_a_mint_pg",
        node_ip="10.0.0.7",
        namespace="mint",
    )
    assert dense.required_gpus_by_node == (("10.0.0.7", 1),)
    assert dense.bundles == ((("CPU", 1), ("GPU", 1), ("node:10.0.0.7", 0.001)),)

    vllm = PlacementGroupBundleRequest.for_vllm(
        replica_key="vllm:model-a::replica-0",
        placement_group_name="mint_vllm_a_pg",
        worker_gpus=3,
        node_ips=("10.0.0.7", "10.0.0.7", "10.0.0.8"),
        namespace="mint",
    )
    assert vllm.required_gpus_by_node == (("10.0.0.7", 2), ("10.0.0.8", 1))
    assert vllm.controller_bundle_index == 3
    assert vllm.bundles[-1] == (("CPU", 1),)

    megatron = PlacementGroupBundleRequest.for_distributed_training(
        replica_key="megatron:model-a::replica-0",
        placement_group_name="mint_megatron_a_mint_pg",
        parallel=ParallelTopology(tensor_parallel_size=2, pipeline_parallel_size=2),
        placement_slices=(("replica-0", "10.0.0.9", 4),),
        namespace="mint",
    )
    assert megatron.required_gpus_by_node == (("10.0.0.9", 4),)
    assert len(megatron.bundles) == 4


@pytest.mark.anyio
async def test_cluster_placement_controller_default_ray_ready_wait_uses_ray_get(monkeypatch) -> None:
    ready_refs: list[object] = []

    class _RayUtil:
        @staticmethod
        def placement_group(**kwargs):
            assert kwargs["name"] == "mint_dense_a_pg"
            assert kwargs["bundles"] == [{"CPU": 1, "GPU": 1, "node:10.0.0.7": 0.001}]
            assert isinstance(kwargs["bundles"], list)
            return _SyncReadyPlacementGroup()

        @staticmethod
        def remove_placement_group(pg):
            raise AssertionError(f"unexpected remove placement group {pg!r}")

    class _RayModule:
        util = _RayUtil()

        @staticmethod
        def get(ref):
            ready_refs.append(ref)
            return "ready"

    class _SyncReadyPlacementGroup:
        def ready(self):
            return "ready-ref"

    monkeypatch.setitem(sys.modules, "ray", _RayModule())

    controller = ClusterPlacementController(
        observed_free_gpus_by_node=lambda: {"10.0.0.7": 1},
    )
    result = await controller.create_pg(
        PlacementGroupCreateRequest.from_mapping(
            replica_key="dense:model-a::replica-0",
            placement_group_name="mint_dense_a_pg",
            required_gpus_by_node={"10.0.0.7": 1},
            bundles=({"GPU": 1, "CPU": 1, "node:10.0.0.7": 0.001},),
        )
    )

    assert result.status is PlacementGroupCreateStatus.READY
    assert ready_refs == ["ready-ref"]
