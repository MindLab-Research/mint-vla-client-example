from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiNodeEngineResources:
    """Scheduling requirements for MultiNodeInferenceEngine.

    Goal: worker TP uses exactly `worker_gpus` GPUs at Ray scheduling level.
    The controller is CPU-only (no extra GPU reservation).
    """

    worker_gpus: int
    controller_gpus: int
    controller_cpus: int
    total_required_gpus: int
    pg_bundles: list[dict[str, float | int]]
    controller_bundle_index: int


def compute_multinode_engine_resources(
    worker_gpus: int,
    preferred_node_ips: list[str] | None = None,
) -> MultiNodeEngineResources:
    if int(worker_gpus) <= 0:
        raise ValueError(f"worker_gpus must be > 0, got {worker_gpus!r}")

    # Controller actor does not reserve a GPU. It is pinned to a CPU-only PG bundle.
    controller_gpus = 0
    controller_cpus = 1

    total_required_gpus = int(worker_gpus)
    preferred = [ip.strip() for ip in (preferred_node_ips or []) if isinstance(ip, str) and ip.strip()]
    if preferred:
        pg_bundles: list[dict[str, float | int]] = []
        for rank in range(total_required_gpus):
            ip = preferred[rank % len(preferred)]
            pg_bundles.append({"GPU": 1, "CPU": 1, f"node:{ip}": 0.001})
        pg_bundles.append({"CPU": controller_cpus, f"node:{preferred[0]}": 0.001})
    else:
        pg_bundles = [{"GPU": 1, "CPU": 1}] * total_required_gpus + [{"CPU": controller_cpus}]
    controller_bundle_index = total_required_gpus

    return MultiNodeEngineResources(
        worker_gpus=int(worker_gpus),
        controller_gpus=controller_gpus,
        controller_cpus=controller_cpus,
        total_required_gpus=total_required_gpus,
        pg_bundles=pg_bundles,
        controller_bundle_index=controller_bundle_index,
    )
