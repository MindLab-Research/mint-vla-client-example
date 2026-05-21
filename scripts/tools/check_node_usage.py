#!/usr/bin/env python3
"""Check Ray cluster node usage across all namespaces.

This tool helps developers coordinate node allocation in shared Ray clusters.
It shows which nodes are occupied by actors from different namespaces.

Usage:
    python scripts/tools/check_node_usage.py
    python scripts/tools/check_node_usage.py --node-ip 192.168.38.4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import ray


def get_node_info() -> dict[str, dict]:
    """Get information about all GPU nodes in the cluster."""
    from ray._private import state as ray_state

    avail = ray_state.available_resources_per_node()
    nodes_by_ip = {}

    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        res = n.get("Resources") or {}
        total_gpus = int(float(res.get("GPU", 0) or 0))
        if total_gpus <= 0:
            continue

        node_id = str(n.get("NodeID") or "")
        node_ip = str(n.get("NodeManagerAddress") or "")
        hostname = str(n.get("NodeManagerHostname") or "")
        node_avail = avail.get(node_id) or {}
        available_gpus = int(float(node_avail.get("GPU", 0) or 0))

        nodes_by_ip[node_ip] = {
            "node_id": node_id,
            "node_ip": node_ip,
            "hostname": hostname,
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "used_gpus": total_gpus - available_gpus,
        }

    return nodes_by_ip


def get_actor_placements() -> dict[str, list[dict]]:
    """Get actor placements grouped by node IP.

    Returns:
        Dict mapping node_ip -> list of actor info dicts
    """
    all_actors = ray.util.list_named_actors(all_namespaces=True)
    placements: dict[str, list[dict]] = defaultdict(list)

    for actor_info in all_actors:
        actor_name = actor_info.get("name", "")
        actor_ns = actor_info.get("namespace", "")

        # Only track vLLM and Megatron actors (the ones that use GPUs)
        if not (
            actor_name.startswith("mint_vllm_")
            or actor_name.startswith("mint_megatron_")
            or actor_name.startswith("mint_dense_")
        ):
            continue

        # Try to find which node this actor is on
        try:
            ray.get_actor(actor_name, namespace=actor_ns)
            pg_name = f"{actor_name}_pg"

            try:
                pg = ray.util.get_placement_group(pg_name)
                pg_info = ray.util.placement_group_table(pg)
                state = pg_info.get("state", "UNKNOWN")
                gpus_by_node_ip = _gpus_by_node_ip(pg_info)

                if not gpus_by_node_ip:
                    placements["unknown"].append(
                        {
                            "actor_name": actor_name,
                            "namespace": actor_ns,
                            "gpus": _total_pg_gpus(pg_info) or 1,
                            "pg_state": state,
                        }
                    )
                    continue

                for node_ip, node_gpus in gpus_by_node_ip.items():
                    placements[node_ip].append(
                        {
                            "actor_name": actor_name,
                            "namespace": actor_ns,
                            "gpus": node_gpus,
                            "pg_state": state,
                        }
                    )
            except Exception:
                # No placement group, might be a single-GPU actor
                placements["unknown"].append(
                    {
                        "actor_name": actor_name,
                        "namespace": actor_ns,
                        "gpus": 1,
                        "pg_state": "N/A",
                    }
                )
        except Exception:
            continue

    return placements


def _gpus_by_node_ip(pg_info: dict) -> dict[str, int]:
    bundles = pg_info.get("bundles", {})
    bundles_to_node = pg_info.get("bundles_to_node_id", {})

    if isinstance(bundles, dict):
        bundle_items = bundles.items()
    elif isinstance(bundles, list):
        bundle_items = enumerate(bundles)
    else:
        bundle_items = ()

    gpu_by_bundle_key = {
        str(bundle_key): int(bundle.get("GPU", 0) or 0)
        for bundle_key, bundle in bundle_items
        if isinstance(bundle, dict) and int(bundle.get("GPU", 0) or 0) > 0
    }

    node_ip_by_id = {
        str(n.get("NodeID") or ""): str(n.get("NodeManagerAddress") or "")
        for n in ray.nodes()
        if n.get("NodeID") and n.get("NodeManagerAddress")
    }

    node_gpu_counts: dict[str, int] = defaultdict(int)
    if isinstance(bundles_to_node, dict):
        bundle_assignments = bundles_to_node.items()
    else:
        bundle_assignments = ()

    for bundle_key, node_id in bundle_assignments:
        node_ip = node_ip_by_id.get(str(node_id or ""))
        bundle_gpus = gpu_by_bundle_key.get(str(bundle_key), 0)
        if node_ip and bundle_gpus > 0:
            node_gpu_counts[node_ip] += bundle_gpus

    return dict(node_gpu_counts)


def _total_pg_gpus(pg_info: dict) -> int:
    bundles = pg_info.get("bundles", {})
    if isinstance(bundles, dict):
        bundle_values = bundles.values()
    elif isinstance(bundles, list):
        bundle_values = bundles
    else:
        bundle_values = ()
    return sum(int(bundle.get("GPU", 0) or 0) for bundle in bundle_values if isinstance(bundle, dict))


def print_node_usage(nodes_by_ip: dict, placements: dict, filter_node_ip: str | None = None):
    """Print formatted node usage information."""
    print("=" * 80)
    print("Ray Cluster Node Usage (All Namespaces)")
    print("=" * 80)
    print()

    # Filter nodes if requested
    if filter_node_ip:
        if filter_node_ip not in nodes_by_ip:
            print(f"Error: Node {filter_node_ip} not found in cluster")
            return
        nodes_to_show = {filter_node_ip: nodes_by_ip[filter_node_ip]}
    else:
        nodes_to_show = nodes_by_ip

    for node_ip in sorted(nodes_to_show.keys()):
        node = nodes_to_show[node_ip]
        print(f"Node: {node_ip}")
        print(f"  Hostname: {node['hostname']}")
        print(f"  Total GPUs: {node['total_gpus']}")
        print(f"  Available GPUs: {node['available_gpus']}")
        print(f"  Used GPUs: {node['used_gpus']}")

        actors = placements.get(node_ip, [])
        if actors:
            print(f"  Actors ({len(actors)}):")
            # Group by namespace
            by_namespace = defaultdict(list)
            for actor in actors:
                by_namespace[actor["namespace"]].append(actor)

            for ns in sorted(by_namespace.keys()):
                ns_actors = by_namespace[ns]
                total_ns_gpus = sum(a["gpus"] for a in ns_actors)
                print(f"    Namespace: {ns} ({total_ns_gpus} GPUs)")
                for actor in ns_actors:
                    print(
                        f"      - {actor['actor_name']} ({actor['gpus']} GPUs, state: {actor['pg_state']})"
                    )
        else:
            print("  Actors: None")
        print()

    # Show actors on unknown nodes
    unknown_actors = placements.get("unknown", [])
    if unknown_actors and not filter_node_ip:
        print("Actors on unknown nodes:")
        for actor in unknown_actors:
            print(f"  - {actor['actor_name']} (namespace: {actor['namespace']})")
        print()

    # Summary
    if not filter_node_ip:
        total_gpus = sum(n["total_gpus"] for n in nodes_by_ip.values())
        total_available = sum(n["available_gpus"] for n in nodes_by_ip.values())
        total_used = total_gpus - total_available
        print("=" * 80)
        print(f"Cluster Summary: {total_used}/{total_gpus} GPUs used, {total_available} available")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check Ray cluster node usage across all namespaces"
    )
    parser.add_argument(
        "--node-ip",
        type=str,
        help="Filter to show only this node IP",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    args = parser.parse_args()

    # Connect to Ray
    try:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    except Exception as e:
        print(f"Error: Failed to connect to Ray cluster: {e}", file=sys.stderr)
        print(f"Make sure RAY_ADDRESS is set or pass --ray-address", file=sys.stderr)
        sys.exit(1)

    try:
        nodes_by_ip = get_node_info()
        placements = get_actor_placements()
        print_node_usage(nodes_by_ip, placements, args.node_ip)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
