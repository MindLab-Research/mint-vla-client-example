from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass

import ray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolcGpuNode:
    node_id: str
    node_ip: str
    hostname: str
    total_gpus: int
    available_gpus: int
    volc_job_id: str | None
    volc_resource_queue_id: str | None


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _parse_volc_job_id_from_hostname(hostname: str | None) -> str | None:
    # Volcano replicas show up as: t-<job_id>-worker-<idx>
    if not hostname:
        return None
    if "-worker-" not in hostname:
        return None
    return hostname.split("-worker-", 1)[0] or None


def _parse_volc_json(payload: str) -> list[dict]:
    m = re.search(r"[\[{]", payload)
    if not m:
        raise RuntimeError("volc output did not contain JSON payload")
    return json.loads(payload[m.start() :])


def _volc_bin() -> str:
    return os.environ.get("MINT_VOLC_BIN", "/root/.volc/bin/volc")


def _volc_job_id_to_resource_queue_id(*, limit: int = 200) -> dict[str, str]:
    volc = _volc_bin()
    try:
        raw = subprocess.check_output(
            [volc, "ml_task", "list", "--output", "json", "--limit", str(int(limit))],
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"volc binary not found at {volc!r}") from e

    jobs = _parse_volc_json(raw)
    out: dict[str, str] = {}
    for j in jobs:
        jid = j.get("JobId")
        rq = j.get("ResourceQueueId")
        if jid and rq:
            out[str(jid)] = str(rq)
    return out


def _list_volc_gpu_nodes(*, resource_queue_id: str) -> list[VolcGpuNode]:
    if not ray.is_initialized():
        raise RuntimeError("ray is not initialized (expected to be connected already)")

    from ray._private import state as ray_state

    # Volcano CLI can return non-zero for very large limits; keep this bounded.
    job_to_queue = _volc_job_id_to_resource_queue_id(limit=200)
    avail = ray_state.available_resources_per_node()

    nodes: list[VolcGpuNode] = []
    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        res = n.get("Resources") or {}
        if float(res.get("GPU", 0) or 0) <= 0:
            continue

        node_id = str(n.get("NodeID") or "")
        node_ip = str(n.get("NodeManagerAddress") or "")
        hostname = str(n.get("NodeManagerHostname") or "")
        volc_job_id = _parse_volc_job_id_from_hostname(hostname)
        rq = job_to_queue.get(volc_job_id) if volc_job_id else None
        if rq != resource_queue_id:
            continue

        total_gpus = int(float(res.get("GPU", 0) or 0))
        node_avail = avail.get(node_id) or {}
        available_gpus = int(float(node_avail.get("GPU", 0) or 0))

        nodes.append(
            VolcGpuNode(
                node_id=node_id,
                node_ip=node_ip,
                hostname=hostname,
                total_gpus=total_gpus,
                available_gpus=available_gpus,
                volc_job_id=volc_job_id,
                volc_resource_queue_id=rq,
            )
        )

    return nodes


def list_node_ips_for_resource_queue(*, resource_queue_id: str) -> list[str]:
    rq = str(resource_queue_id).strip()
    if not rq:
        raise ValueError("resource_queue_id is empty")
    nodes = _list_volc_gpu_nodes(resource_queue_id=rq)
    # Preserve determinism: sort by IP
    return sorted({n.node_ip for n in nodes if n.node_ip})


def select_free_nodes_from_allowed_ips(
    *,
    allowed_node_ips: list[str],
    required_gpus: int,
) -> tuple[list[str], int]:
    """Pick nodes from an allowlist of Ray node IPs with enough free GPUs.

    For multi-node jobs we still require full nodes for all but the last node.
    For single-node jobs smaller than gpus_per_node (e.g. 4 GPUs on an 8-GPU node),
    this allows placing on partially-free nodes instead of requiring a whole free node.
    """
    if not ray.is_initialized():
        raise RuntimeError("ray is not initialized (expected to be connected already)")

    allowed = {ip.strip() for ip in allowed_node_ips if ip and ip.strip()}
    if not allowed:
        raise ValueError("allowed_node_ips is empty")
    required = int(required_gpus)
    if required <= 0:
        raise ValueError(f"required_gpus must be > 0, got {required_gpus!r}")

    from ray._private import state as ray_state

    avail = ray_state.available_resources_per_node()
    nodes: list[VolcGpuNode] = []
    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        res = n.get("Resources") or {}
        if float(res.get("GPU", 0) or 0) <= 0:
            continue

        node_ip = str(n.get("NodeManagerAddress") or "")
        if node_ip not in allowed:
            continue

        node_id = str(n.get("NodeID") or "")
        hostname = str(n.get("NodeManagerHostname") or "")
        total_gpus = int(float(res.get("GPU", 0) or 0))
        node_avail = avail.get(node_id) or {}
        available_gpus = int(float(node_avail.get("GPU", 0) or 0))

        nodes.append(
            VolcGpuNode(
                node_id=node_id,
                node_ip=node_ip,
                hostname=hostname,
                total_gpus=total_gpus,
                available_gpus=available_gpus,
                volc_job_id=_parse_volc_job_id_from_hostname(hostname),
                volc_resource_queue_id=None,
            )
        )

    if not nodes:
        raise RuntimeError(f"no alive GPU Ray nodes found within allowed_node_ips={sorted(allowed)}")

    gpus_per_node = max(1, int(max(n.total_gpus for n in nodes)))
    nodes_needed = int(math.ceil(required / gpus_per_node))

    needs_per_node: list[int] = []
    for i in range(nodes_needed):
        remaining = required - i * gpus_per_node
        needs_per_node.append(min(gpus_per_node, remaining))

    remaining_nodes = list(nodes)
    selected: list[VolcGpuNode] = []
    for need in needs_per_node:
        feasible = [n for n in remaining_nodes if n.available_gpus >= need]
        if not feasible:
            diag = sorted(
                [(n.node_ip, n.available_gpus, n.total_gpus, n.hostname) for n in nodes],
                key=lambda x: x[0],
            )
            raise RuntimeError(
                "insufficient free nodes within allowlist: "
                f"required_gpus={required} gpus_per_node={gpus_per_node} need_nodes={nodes_needed} "
                f"needs_per_node={needs_per_node} candidates={diag}"
            )

        # Best-fit (min available_gpus that satisfies need) to preserve whole free nodes for larger jobs.
        chosen = min(feasible, key=lambda n: (n.available_gpus, n.node_ip))
        selected.append(chosen)
        remaining_nodes.remove(chosen)

    if len(selected) != nodes_needed:
        diag = sorted(
            [(n.node_ip, n.available_gpus, n.total_gpus, n.hostname) for n in nodes],
            key=lambda x: x[0],
        )
        raise RuntimeError(
            "insufficient free nodes within allowlist: "
            f"required_gpus={required} gpus_per_node={gpus_per_node} need_nodes={nodes_needed} "
            f"needs_per_node={needs_per_node} candidates={diag}"
        )

    node_ips = [n.node_ip for n in selected]
    logger.info(
        f"[volc_placement] selected allowlist nodes={node_ips} required_gpus={required} gpus_per_node={gpus_per_node}"
    )
    return node_ips, gpus_per_node


def select_free_nodes_for_resource_queue(
    *,
    resource_queue_id: str,
    required_gpus: int,
) -> tuple[list[str], int]:
    """Pick nodes from a Volcano ResourceQueueID.

    Returns:
        (node_ips, gpus_per_node)
    """
    rq = str(resource_queue_id).strip()
    if not rq:
        raise ValueError("resource_queue_id is empty")
    required = int(required_gpus)
    if required <= 0:
        raise ValueError(f"required_gpus must be > 0, got {required_gpus!r}")

    nodes = _list_volc_gpu_nodes(resource_queue_id=rq)
    if not nodes:
        raise RuntimeError(f"no alive GPU Ray nodes found for volc ResourceQueueID={rq}")

    gpus_per_node = max(1, int(max(n.total_gpus for n in nodes)))
    nodes_needed = int(math.ceil(required / gpus_per_node))

    # For single-node jobs smaller than gpus_per_node (e.g. 4 GPUs on an 8-GPU node),
    # allow placing on partially-free nodes within the resource queue. This avoids
    # spurious failures when the queue has >= required_gpus free on some nodes but
    # has no fully-free nodes.
    if nodes_needed == 1 and required < gpus_per_node:
        feasible = [n for n in nodes if n.available_gpus >= required]
        if not feasible:
            diag = sorted(
                [(n.node_ip, n.available_gpus, n.total_gpus, n.hostname) for n in nodes],
                key=lambda x: x[0],
            )
            raise RuntimeError(
                "insufficient free nodes for volc queue placement: "
                f"rq={rq} required_gpus={required} gpus_per_node={gpus_per_node} "
                f"need_nodes={nodes_needed} candidates={diag}"
            )

        chosen = min(feasible, key=lambda n: (n.available_gpus, n.node_ip))
        node_ips = [chosen.node_ip]
        logger.info(
            f"[volc_placement] selected rq={rq} nodes={node_ips} required_gpus={required} gpus_per_node={gpus_per_node}"
        )
        return node_ips, gpus_per_node

    full_nodes = [n for n in nodes if n.available_gpus >= gpus_per_node]
    full_nodes_sorted = sorted(full_nodes, key=lambda n: n.node_ip)
    if len(full_nodes_sorted) < nodes_needed:
        diag = sorted(
            [(n.node_ip, n.available_gpus, n.total_gpus, n.hostname) for n in nodes],
            key=lambda x: x[0],
        )
        raise RuntimeError(
            "insufficient free nodes for volc queue placement: "
            f"rq={rq} required_gpus={required} gpus_per_node={gpus_per_node} "
            f"need_nodes={nodes_needed} free_nodes={len(full_nodes_sorted)} "
            f"candidates={diag}"
        )

    node_ips = [n.node_ip for n in full_nodes_sorted[:nodes_needed]]
    logger.info(
        f"[volc_placement] selected rq={rq} nodes={node_ips} required_gpus={required} gpus_per_node={gpus_per_node}"
    )
    return node_ips, gpus_per_node


def build_node_affinity_gpu_bundles(
    *,
    node_ips: list[str],
    gpus_per_node: int,
    required_gpus: int,
    cpu_per_gpu: int = 1,
) -> list[dict[str, int | float]]:
    """Build 1-GPU placement-group bundles pinned to specific nodes.

    Uses Ray's per-node resource key `node:<ip>` as an affinity anchor.
    """
    if gpus_per_node <= 0:
        raise ValueError(f"gpus_per_node must be > 0, got {gpus_per_node!r}")
    required = int(required_gpus)
    if required <= 0:
        raise ValueError(f"required_gpus must be > 0, got {required_gpus!r}")
    if not node_ips:
        raise ValueError("node_ips is empty")

    nodes_needed = int(math.ceil(required / int(gpus_per_node)))
    if len(node_ips) < nodes_needed:
        raise ValueError(
            f"node_ips too short: need {nodes_needed} nodes for required_gpus={required}, got {len(node_ips)}"
        )

    bundles: list[dict[str, int | float]] = []
    for i in range(required):
        ip = node_ips[i // int(gpus_per_node)]
        bundles.append(
            {
                "GPU": 1,
                "CPU": int(cpu_per_gpu),
                f"node:{ip}": 0.001,
            }
        )
    return bundles
