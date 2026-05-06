"""GPU binding helpers for actor observability."""

from __future__ import annotations

import os
import subprocess


def _parse_cuda_visible_devices() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [part.strip() for part in value.split(",") if part.strip()]


def _nvidia_smi_uuid_by_index() -> dict[int, str]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    uuids: dict[int, str] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        try:
            uuids[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return uuids


def _physical_gpu_from_ray_id(raw_gpu_id: str, visible_devices: list[str]) -> tuple[int | None, str | None]:
    if raw_gpu_id.startswith("GPU-"):
        return None, raw_gpu_id
    try:
        gpu_index = int(float(raw_gpu_id))
    except (TypeError, ValueError):
        return None, None
    if visible_devices:
        if raw_gpu_id in visible_devices:
            return gpu_index, None
        if 0 <= gpu_index < len(visible_devices):
            visible = visible_devices[gpu_index]
            if visible.startswith("GPU-"):
                return None, visible
            try:
                return int(float(visible)), None
            except (TypeError, ValueError):
                return gpu_index, None
    return gpu_index, None


def gpu_bindings_from_ray_gpu_ids(*, hostname: str, node_id: str | None = None, rank: int | None = None) -> list[dict[str, object]]:
    import ray

    visible_devices = _parse_cuda_visible_devices()
    uuid_by_index = _nvidia_smi_uuid_by_index()
    bindings: list[dict[str, object]] = []
    try:
        gpu_ids = list(ray.get_gpu_ids())
    except Exception:
        gpu_ids = []
    for gpu_id in gpu_ids:
        raw_gpu_id = str(gpu_id).strip()
        gpu_index, gpu_uuid = _physical_gpu_from_ray_id(raw_gpu_id, visible_devices)
        if gpu_uuid is None and gpu_index is not None:
            gpu_uuid = uuid_by_index.get(gpu_index)
        binding: dict[str, object] = {"hostname": hostname, "ray_gpu_id": raw_gpu_id}
        if node_id is not None:
            binding["node_id"] = node_id
        if gpu_index is not None:
            binding["gpu_index"] = int(gpu_index)
        if gpu_uuid:
            binding["gpu_uuid"] = gpu_uuid
        if rank is not None:
            binding["rank"] = int(rank)
        if "gpu_index" in binding or "gpu_uuid" in binding:
            bindings.append(binding)
    return bindings
