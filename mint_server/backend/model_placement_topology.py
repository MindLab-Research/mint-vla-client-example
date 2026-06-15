from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


PlacementBundle: TypeAlias = dict[str, int | float]
PlacementSlice: TypeAlias = tuple[str, str, int]


def _positive_int(value: int, *, field: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return out


@dataclass(frozen=True)
class ParallelTopology:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None
    context_parallel_size: int = 1

    def __post_init__(self) -> None:
        _positive_int(self.tensor_parallel_size, field="tensor_parallel_size")
        _positive_int(self.pipeline_parallel_size, field="pipeline_parallel_size")
        _positive_int(self.expert_parallel_size, field="expert_parallel_size")
        _positive_int(self.context_parallel_size, field="context_parallel_size")
        if self.expert_tensor_parallel_size is not None:
            _positive_int(self.expert_tensor_parallel_size, field="expert_tensor_parallel_size")

    @property
    def world_size(self) -> int:
        etp = self.expert_tensor_parallel_size
        if etp is None:
            etp = self.tensor_parallel_size

        if self.expert_parallel_size >= self.tensor_parallel_size and etp < self.tensor_parallel_size:
            return self.expert_parallel_size * self.pipeline_parallel_size * self.context_parallel_size
        if self.expert_parallel_size > 1 and self.context_parallel_size > 1:
            return (
                self.tensor_parallel_size
                * self.pipeline_parallel_size
                * max(self.expert_parallel_size, self.context_parallel_size)
            )
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.context_parallel_size
        )


@dataclass(frozen=True)
class PlacementGroupLayout:
    bundles: tuple[PlacementBundle, ...]
    controller_bundle_index: int | None = None


@dataclass(frozen=True)
class EnginePlacementTopology:
    parallel: ParallelTopology = ParallelTopology()
    gpu_count: int | None = None
    placement_slices: tuple[PlacementSlice, ...] = ()

    @property
    def worker_world_size(self) -> int:
        if self.gpu_count is not None:
            return _positive_int(self.gpu_count, field="gpu_count")
        return self.parallel.world_size

    def gpu_bundles(self, *, cpu_per_gpu: int = 1) -> list[PlacementBundle]:
        cpu = _positive_int(cpu_per_gpu, field="cpu_per_gpu")
        if self.placement_slices:
            bundles: list[PlacementBundle] = []
            total_gpus = 0
            for _replica_id, node_ip, gpu_count in self.placement_slices:
                count = _positive_int(gpu_count, field="placement_slices.gpu_count")
                total_gpus += count
                for _ in range(count):
                    bundles.append({"GPU": 1, "CPU": cpu, f"node:{str(node_ip)}": 0.001})
            if total_gpus != self.worker_world_size:
                raise ValueError(
                    "placement_slices GPU total must match worker_world_size, "
                    f"got total={total_gpus} worker_world_size={self.worker_world_size}"
                )
            return bundles
        return [{"GPU": 1, "CPU": cpu} for _ in range(self.worker_world_size)]

    def with_controller_bundle(self, *, cpu: int = 1, cpu_per_gpu: int = 1) -> PlacementGroupLayout:
        controller_cpu = _positive_int(cpu, field="cpu")
        worker_bundles = tuple(self.gpu_bundles(cpu_per_gpu=cpu_per_gpu))
        return PlacementGroupLayout(
            bundles=(*worker_bundles, {"CPU": controller_cpu}),
            controller_bundle_index=len(worker_bundles),
        )

