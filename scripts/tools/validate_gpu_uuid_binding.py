#!/usr/bin/env python3
"""Validate actor-local GPU UUID binding against NVML-visible UUIDs in a Ray cluster."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import ray


def _maybe_reexec_runtime_python() -> None:
    env_root = (os.getenv("PFS_RUNTIME_ENV_ROOT") or "").strip()
    if not env_root:
        return
    target = Path(env_root) / "host-venv" / "bin" / "python"
    if not target.exists():
        return
    try:
        current = Path(sys.executable).resolve()
        wanted = target.resolve()
    except Exception:
        return
    if current == wanted:
        return
    os.execv(str(wanted), [str(wanted), *sys.argv])


def _gpu_nodes() -> list[dict[str, Any]]:
    return [
        node
        for node in ray.nodes()
        if node.get("Alive") and float(node.get("Resources", {}).get("GPU", 0.0) or 0.0) > 0
    ]


@ray.remote(num_cpus=0.01, num_gpus=0)
def inspect_node() -> dict[str, Any]:
    try:
        import ray.util  # type: ignore[attr-defined]

        node_ip = str(ray.util.get_node_ip_address())
    except Exception:
        node_ip = "unknown"
    cmd = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return {"hostname": socket.gethostname(), "node_ip": node_ip, "error": repr(exc)}
    nvml = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            nvml.append({"gpu_index": parts[0], "gpu_uuid": parts[1]})
    return {
        "hostname": socket.gethostname(),
        "node_ip": node_ip,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ray_gpu_ids": list(ray.get_gpu_ids()),
        "nvml_gpus": nvml,
    }


@ray.remote(num_cpus=0.01, num_gpus=1)
def inspect_allocated_gpu() -> dict[str, Any]:
    from mint_server.backend.ray_cluster.gpu_binding_helpers import gpu_bindings_from_ray_gpu_ids

    try:
        import ray.util  # type: ignore[attr-defined]

        node_ip = str(ray.util.get_node_ip_address())
    except Exception:
        node_ip = "unknown"
    hostname = socket.gethostname()
    return {
        "hostname": hostname,
        "node_ip": node_ip,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ray_gpu_ids": list(ray.get_gpu_ids()),
        "actor_bindings": gpu_bindings_from_ray_gpu_ids(hostname=hostname, node_id=None),
    }


def _read_model_actor_inventory_bindings() -> list[dict[str, str]]:
    try:
        from mint_server.routes import internal as internal_routes

        stats = ray.get(internal_routes.admission_stats(include_actor_rss=False))
    except Exception:
        return []
    out: list[dict[str, str]] = []
    for rec in stats.get("model_actor_inventory", []):
        if not isinstance(rec, dict):
            continue
        for binding in internal_routes._model_actor_inventory_gpu_bindings(rec):
            out.append({str(k): str(v) for k, v in binding.items()})
    return out


def main() -> int:
    _maybe_reexec_runtime_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--namespace",
        default=os.getenv("MINT_RAY_NAMESPACE") or os.getenv("MINT_RAY_NAMESPACE") or "mint",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--include-allocated-probe",
        action="store_true",
        help="Also try one num_gpus=1 probe per GPU node. This can block when the cluster has no free GPUs.",
    )
    parser.add_argument(
        "--validate-existing-bindings",
        action="store_true",
        help="Fetch current ModelActorInventory bindings and verify any gpu_uuid labels against NVML UUIDs on the same hostname.",
    )
    args = parser.parse_args()

    ray.init(address=args.ray_address, namespace=args.namespace)
    node_refs = []
    allocated_refs = []
    for node in _gpu_nodes():
        ip = str(node["NodeManagerAddress"])
        opts = {"resources": {f"node:{ip}": 0.001}}
        node_refs.append(inspect_node.options(**opts).remote())
        if args.include_allocated_probe:
            allocated_refs.append(inspect_allocated_gpu.options(**opts).remote())
    rows = ray.get(node_refs, timeout=args.timeout_s) if node_refs else []
    allocated_rows = ray.get(allocated_refs, timeout=args.timeout_s) if allocated_refs else []
    nvml_by_node = {
        row.get("node_ip"): {gpu.get("gpu_uuid") for gpu in row.get("nvml_gpus", []) if gpu.get("gpu_uuid")}
        for row in rows
    }
    nvml_by_hostname = {
        row.get("hostname"): {gpu.get("gpu_uuid") for gpu in row.get("nvml_gpus", []) if gpu.get("gpu_uuid")}
        for row in rows
    }
    model_actor_inventory_bindings = _read_model_actor_inventory_bindings() if args.validate_existing_bindings else []
    model_actor_inventory_binding_uuid_errors = []
    for binding in model_actor_inventory_bindings:
        gpu_uuid = binding.get("gpu_uuid")
        hostname = binding.get("hostname")
        if gpu_uuid and hostname and gpu_uuid not in nvml_by_hostname.get(hostname, set()):
            model_actor_inventory_binding_uuid_errors.append(binding)
    summary = []
    for row in rows:
        nvml_uuids = nvml_by_node.get(row.get("node_ip"), set())
        node_allocated = [item for item in allocated_rows if item.get("node_ip") == row.get("node_ip")]
        binding_uuids = {
            b.get("gpu_uuid")
            for item in node_allocated
            for b in item.get("actor_bindings", [])
            if b.get("gpu_uuid")
        }
        summary.append(
            {
                "hostname": row.get("hostname"),
                "node_ip": row.get("node_ip"),
                "nvml_gpu_count": len(nvml_uuids),
                "allocated_probe_count": len(node_allocated),
                "allocated_binding_uuid_count": len(binding_uuids),
                "allocated_binding_uuid_missing_from_nvml": sorted(binding_uuids - nvml_uuids),
            }
        )
    print(
        json.dumps(
            {
                "summary": summary,
                "rows": rows,
                "allocated_rows": allocated_rows,
                "model_actor_inventory_binding_uuid_count": sum(
                    1 for binding in model_actor_inventory_bindings if binding.get("gpu_uuid")
                ),
                "model_actor_inventory_binding_uuid_missing_from_nvml": model_actor_inventory_binding_uuid_errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
