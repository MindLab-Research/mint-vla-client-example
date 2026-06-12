from __future__ import annotations

import pytest

from mint_server.backend.model_placement_topology import (
    EnginePlacementTopology,
    ParallelTopology,
)


def test_parallel_topology_computes_traditional_world_size() -> None:
    topology = ParallelTopology(
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
        expert_parallel_size=4,
        context_parallel_size=1,
    )

    assert topology.world_size == 24


def test_parallel_topology_computes_expert_tensor_parallel_folding_world_size() -> None:
    topology = ParallelTopology(
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        expert_parallel_size=8,
        expert_tensor_parallel_size=1,
        context_parallel_size=3,
    )

    assert topology.world_size == 48


def test_parallel_topology_computes_context_expert_folding_world_size() -> None:
    topology = ParallelTopology(
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
        expert_parallel_size=4,
        context_parallel_size=8,
    )

    assert topology.world_size == 48


def test_parallel_topology_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        ParallelTopology(tensor_parallel_size=0)


def test_engine_placement_topology_builds_unpinned_gpu_bundles() -> None:
    topology = EnginePlacementTopology(
        parallel=ParallelTopology(tensor_parallel_size=2),
        gpu_count=2,
    )

    assert topology.worker_world_size == 2
    assert topology.gpu_bundles() == [
        {"GPU": 1, "CPU": 1},
        {"GPU": 1, "CPU": 1},
    ]


def test_engine_placement_topology_builds_node_pinned_gpu_bundles() -> None:
    topology = EnginePlacementTopology(
        parallel=ParallelTopology(tensor_parallel_size=3),
        placement_slices=(("replica-0", "10.0.0.1", 2), ("replica-0", "10.0.0.2", 1)),
    )

    assert topology.worker_world_size == 3
    assert topology.gpu_bundles() == [
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.1": 0.001},
        {"GPU": 1, "CPU": 1, "node:10.0.0.2": 0.001},
    ]


def test_engine_placement_topology_builds_vllm_controller_bundle() -> None:
    topology = EnginePlacementTopology(
        parallel=ParallelTopology(tensor_parallel_size=2),
        gpu_count=2,
    )

    layout = topology.with_controller_bundle(cpu=1)

    assert layout.controller_bundle_index == 2
    assert layout.bundles == (
        {"GPU": 1, "CPU": 1},
        {"GPU": 1, "CPU": 1},
        {"CPU": 1},
    )
