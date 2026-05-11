from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha1
from urllib.parse import urlsplit

import ray

logger = logging.getLogger(__name__)


def _namespace_suffix(namespace: str) -> str:
    raw = str(namespace).strip().lower()
    if not raw:
        return "default"
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    if len(sanitized) <= 24:
        return sanitized or "default"
    digest = sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[:15]}_{digest}"


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


def _iter_pg_bundle_items(bundles: object) -> list[tuple[str, dict[str, object]]]:
    items: list[tuple[str, dict[str, object]]] = []
    if isinstance(bundles, dict):
        iterator = bundles.items()
    elif isinstance(bundles, list):
        iterator = ((str(i), bundle) for i, bundle in enumerate(bundles))
    else:
        iterator = ()
    for bundle_key, bundle in iterator:
        if isinstance(bundle, dict):
            items.append((str(bundle_key), bundle))
    return items


def _is_ray_client_mode() -> bool:
    try:
        client_mod = getattr(ray.util, "client", None)
        if client_mod is None:
            return False
        ray_client = getattr(client_mod, "ray", None)
        if ray_client is None:
            return False
        is_connected = getattr(ray_client, "is_connected", None)
        if callable(is_connected):
            return bool(is_connected())
    except Exception:
        return False
    return False


def _ray_state_api_address() -> str | None:
    for name in ("MINT_RAY_CLIENT_ADDRESS", "RAY_CLIENT_ADDRESS", "RAY_ADDRESS"):
        raw = str(os.environ.get(name) or "").strip()
        if not raw:
            continue
        if raw.startswith("ray://"):
            parsed = urlsplit(raw)
            if parsed.hostname:
                return f"{parsed.hostname}:6379"
            continue
        return raw
    return None


def _actor_used_gpus_by_node_from_state_api(*, context: str) -> tuple[dict[str, float], bool]:
    try:
        from ray.util import state as ray_util_state
    except Exception:
        ray_util_state = getattr(getattr(ray, "util", None), "state", None)
    if ray_util_state is None:
        logger.debug("%s: ray.util.state unavailable for actor fallback", context)
        return {}, False

    list_actors = getattr(ray_util_state, "list_actors", None)
    if list_actors is None:
        logger.debug("%s: ray.util.state.list_actors unavailable for actor fallback", context)
        return {}, False

    address = _ray_state_api_address()
    if not address:
        logger.debug("%s: no Ray state API address configured for actor fallback", context)
        return {}, False

    try:
        actors = list_actors(
            detail=True,
            limit=10000,
        )
    except Exception as e:
        if not address:
            logger.warning("%s: actor state fallback failed: %s", context, e)
            return {}, False
        try:
            child_env = dict(os.environ)
            for name in ("RAY_ADDRESS", "MINT_RAY_CLIENT_ADDRESS", "RAY_CLIENT_ADDRESS"):
                child_env.pop(name, None)
            raw = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, sys\n"
                        "from ray.util import state as ray_state\n"
                        "rows = ray_state.list_actors(address=sys.argv[1], detail=True, limit=10000)\n"
                        "print(json.dumps([row.asdict() for row in rows]))\n"
                    ),
                    address,
                ],
                text=True,
                timeout=60,
                env=child_env,
            )
            actors = json.loads(raw)
        except Exception as e2:
            logger.warning("%s: actor state fallback failed: %s", context, e2)
            return {}, False

    used_gpus_by_node: dict[str, float] = {}
    for actor in actors:
        if not isinstance(actor, dict):
            continue
        if str(actor.get("state") or "") != "ALIVE":
            continue
        node_id = str(actor.get("node_id") or "")
        if not node_id:
            continue
        resources = actor.get("required_resources") or {}
        if not isinstance(resources, dict):
            continue
        gpu = float(resources.get("GPU", 0) or 0)
        if gpu <= 0:
            continue
        used_gpus_by_node[node_id] = used_gpus_by_node.get(node_id, 0.0) + gpu
    return used_gpus_by_node, True


def _available_resources_per_node_with_pg_fallback(
    *,
    context: str,
) -> tuple[dict[str, dict[str, float]], bool]:
    try:
        from ray._private import state as ray_state

        return ray_state.available_resources_per_node(), True
    except Exception as e:
        logger.warning(
            "%s: available_resources_per_node failed, using placement-group fallback: %s",
            context,
            e,
        )

    alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
    node_ip_to_id = {
        str(n.get("NodeManagerAddress") or ""): str(n.get("NodeID") or "")
        for n in alive_nodes
        if n.get("NodeManagerAddress") and n.get("NodeID")
    }

    try:
        table = ray.util.placement_group_table()
    except Exception as e:
        logger.warning("%s: placement_group_table fallback failed: %s", context, e)
        infos = ()
    else:
        if isinstance(table, dict):
            infos = table.values()
        elif isinstance(table, list):
            infos = table
        else:
            infos = ()

    used_gpus_by_node: dict[str, float] = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        if str(info.get("state") or "") == "REMOVED":
            continue

        bundles_to_node_id = info.get("bundles_to_node_id") or {}
        if not isinstance(bundles_to_node_id, dict):
            bundles_to_node_id = {}

        for bundle_key, bundle in _iter_pg_bundle_items(info.get("bundles") or {}):
            gpu = float(bundle.get("GPU", 0) or 0)
            if gpu <= 0:
                continue

            bundle_idx = int(bundle_key) if bundle_key.isdigit() else None
            node_id = str(
                bundles_to_node_id.get(bundle_key)
                or (bundles_to_node_id.get(bundle_idx) if bundle_idx is not None else "")
                or ""
            )
            if not node_id:
                pinned_ips = [
                    key.split("node:", 1)[1]
                    for key, value in bundle.items()
                    if isinstance(key, str)
                    and key.startswith("node:")
                    and float(value or 0) > 0
                ]
                if len(pinned_ips) == 1:
                    node_id = node_ip_to_id.get(pinned_ips[0], "")
            if not node_id:
                continue
            used_gpus_by_node[node_id] = used_gpus_by_node.get(node_id, 0.0) + gpu

    actor_used_gpus_by_node, actor_state_ok = _actor_used_gpus_by_node_from_state_api(context=context)
    if actor_state_ok:
        for node_id, used_gpus in actor_used_gpus_by_node.items():
            used_gpus_by_node[node_id] = max(
                used_gpus_by_node.get(node_id, 0.0),
                float(used_gpus),
            )

    try:
        from .resource_pool import get_resource_pool

        for node_id, used_gpus in get_resource_pool().gpus_used_by_node().items():
            used_gpus_by_node[node_id] = max(
                used_gpus_by_node.get(node_id, 0.0),
                float(used_gpus),
            )
    except Exception as e:
        logger.debug("%s: resource_pool fallback failed: %s", context, e)

    # Ray Client mode cannot always access per-node availability via ray._private.state.
    # If actor-state fallback succeeds we can still trust the derived reservations. Otherwise,
    # treat Ray Client runs as schedulable unless explicit reservations say otherwise.
    fail_closed_on_missing = not (actor_state_ok or _is_ray_client_mode())

    return (
        {node_id: {"GPU": -used_gpus} for node_id, used_gpus in used_gpus_by_node.items()},
        fail_closed_on_missing,
    )


def _available_gpus_for_node(
    *,
    total_gpus: int,
    avail: dict[str, dict[str, float]],
    node_id: str,
    fail_closed_on_missing: bool,
) -> int:
    node_avail = avail.get(node_id) or {}
    raw_available = node_avail.get("GPU")
    if raw_available is None:
        return 0 if fail_closed_on_missing else total_gpus
    raw_available_f = float(raw_available or 0)
    if raw_available_f < 0:
        reserved_gpus = int(math.ceil(-raw_available_f))
        return max(0, total_gpus - reserved_gpus)
    return max(0, int(raw_available_f))


def _list_volc_gpu_nodes(*, resource_queue_id: str) -> list[VolcGpuNode]:
    if not ray.is_initialized():
        raise RuntimeError("ray is not initialized (expected to be connected already)")

    # Volcano CLI can return non-zero for very large limits; keep this bounded.
    job_to_queue = _volc_job_id_to_resource_queue_id(limit=200)
    avail, fail_closed_on_missing = _available_resources_per_node_with_pg_fallback(
        context="_list_volc_gpu_nodes",
    )

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
        available_gpus = _available_gpus_for_node(
            total_gpus=total_gpus,
            avail=avail,
            node_id=node_id,
            fail_closed_on_missing=fail_closed_on_missing,
        )

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


def _list_alive_gpu_nodes() -> list[VolcGpuNode]:
    if not ray.is_initialized():
        raise RuntimeError("ray is not initialized (expected to be connected already)")

    avail, fail_closed_on_missing = _available_resources_per_node_with_pg_fallback(
        context="_list_alive_gpu_nodes",
    )

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
        total_gpus = int(float(res.get("GPU", 0) or 0))
        available_gpus = _available_gpus_for_node(
            total_gpus=total_gpus,
            avail=avail,
            node_id=node_id,
            fail_closed_on_missing=fail_closed_on_missing,
        )
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
    return nodes


def _gpu_placement_groups() -> list[dict[str, object]]:
    try:
        table = ray.util.placement_group_table()
    except Exception:
        return []

    groups: list[dict[str, object]] = []
    if isinstance(table, dict):
        infos = table.values()
    elif isinstance(table, list):
        infos = table
    else:
        infos = ()
    for info in infos:
        if not isinstance(info, dict):
            continue
        state = str(info.get("state") or "")
        if state == "REMOVED":
            continue

        bundles = info.get("bundles") or {}
        total_gpu = 0.0
        gpu_by_pinned_ip: dict[str, float] = {}
        pinned_ips: set[str] = set()
        if isinstance(bundles, dict):
            bundle_values = bundles.values()
        elif isinstance(bundles, list):
            bundle_values = bundles
        else:
            bundle_values = ()
        for bundle in bundle_values:
            if not isinstance(bundle, dict):
                continue
            total_gpu += float(bundle.get("GPU", 0) or 0)
            bundle_gpu = float(bundle.get("GPU", 0) or 0)
            for key, value in bundle.items():
                if not isinstance(key, str) or not key.startswith("node:"):
                    continue
                if float(value or 0) > 0:
                    ip = key.split("node:", 1)[1]
                    pinned_ips.add(ip)
                    gpu_by_pinned_ip[ip] = gpu_by_pinned_ip.get(ip, 0.0) + bundle_gpu
        if total_gpu <= 0:
            continue

        bundles_to_node_id = info.get("bundles_to_node_id") or {}
        node_ids = sorted(
            {
                str(node_id)
                for node_id in (
                    bundles_to_node_id.values()
                    if isinstance(bundles_to_node_id, dict)
                    else ()
                )
                if node_id
            }
        )
        groups.append(
            {
                "name": str(info.get("name") or "<unnamed>"),
                "namespace": str(
                    info.get("ray_namespace")
                    or info.get("namespace")
                    or info.get("rayNamespace")
                    or ""
                ),
                "state": state or "<unknown>",
                "pinned_ips": sorted(pinned_ips),
                "gpu_by_pinned_ip": dict(sorted(gpu_by_pinned_ip.items())),
                "node_ids": node_ids,
            }
        )
    return groups


def parse_model_node_ip_list(
    *,
    raw_json: str | None,
    lookup_keys: list[str],
    env_var_name: str,
    context: str,
) -> list[str]:
    raw = str(raw_json or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"{context}: {env_var_name} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{context}: {env_var_name} must be a JSON object")

    value = None
    for key in lookup_keys:
        value = data.get(key)
        if value is not None:
            break
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(
            f"{context}: {env_var_name}[{key!r}] must be a JSON list of node IPs, got {type(value).__name__}"
        )

    cleaned = [str(ip).strip() for ip in value if str(ip).strip()]
    if not cleaned:
        raise RuntimeError(f"{context}: {env_var_name}[{key!r}] resolved to an empty node list")
    return list(dict.fromkeys(cleaned))


def parse_model_single_node_ip(
    *,
    raw_json: str | None,
    lookup_keys: list[str],
    env_var_name: str,
    context: str,
) -> str | None:
    raw = str(raw_json or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"{context}: {env_var_name} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{context}: {env_var_name} must be a JSON object")

    value = None
    for key in lookup_keys:
        value = data.get(key)
        if value is not None:
            break
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"{context}: {env_var_name}[{key!r}] must be a non-empty node IP string"
        )
    return value.strip()


def assert_node_ip_capacity(
    *,
    required_gpus_by_node_ip: dict[str, int],
    context: str,
    ignore_placement_group_names: set[str] | None = None,
    ignore_placement_group_namespace: str | None = None,
) -> None:
    requested = {
        str(node_ip).strip(): int(gpus)
        for node_ip, gpus in required_gpus_by_node_ip.items()
        if str(node_ip).strip()
    }
    if not requested:
        raise ValueError("required_gpus_by_node_ip is empty")
    if any(gpus <= 0 for gpus in requested.values()):
        raise ValueError(f"required_gpus_by_node_ip must be positive, got {requested!r}")

    nodes = _list_alive_gpu_nodes()
    nodes_by_ip = {node.node_ip: node for node in nodes if node.node_ip}
    missing = sorted(ip for ip in requested if ip not in nodes_by_ip)
    if missing:
        raise RuntimeError(
            f"{context}: requested pinned node(s) are not alive Ray GPU nodes: "
            f"missing_nodes={missing} alive_gpu_nodes={sorted(nodes_by_ip)}"
        )

    ignored_pg_names = {str(name) for name in (ignore_placement_group_names or set()) if str(name)}
    placement_groups = _gpu_placement_groups()

    def _is_ignored_pg(pg: dict[str, object]) -> bool:
        name = str(pg.get("name") or "")
        if name not in ignored_pg_names:
            return False
        if ignore_placement_group_namespace is None:
            return True
        namespace = str(pg.get("namespace") or "")
        if namespace:
            return namespace == str(ignore_placement_group_namespace)
        return name.endswith(f"_{_namespace_suffix(ignore_placement_group_namespace)}_pg")

    ignored_placement_groups = [pg for pg in placement_groups if _is_ignored_pg(pg)]
    blocker_placement_groups = [pg for pg in placement_groups if not _is_ignored_pg(pg)]
    blockers: list[dict[str, object]] = []
    for node_ip, need_gpus in requested.items():
        node = nodes_by_ip[node_ip]
        ignored_reserved_gpus = 0.0
        for pg in ignored_placement_groups:
            by_ip = pg.get("gpu_by_pinned_ip")
            if isinstance(by_ip, dict):
                ignored_reserved_gpus += float(by_ip.get(node_ip, 0) or 0)
            elif node_ip in pg["pinned_ips"] or node.node_id in pg["node_ids"]:
                ignored_reserved_gpus += float(need_gpus)
        effective_available_gpus = float(node.available_gpus) + ignored_reserved_gpus
        if effective_available_gpus >= need_gpus:
            continue
        matching_pgs = [
            f"{pg['name']}:{pg['state']}"
            for pg in blocker_placement_groups
            if node_ip in pg["pinned_ips"] or node.node_id in pg["node_ids"]
        ]
        blockers.append(
            {
                "node_ip": node_ip,
                "hostname": node.hostname,
                "need_gpus": need_gpus,
                "available_gpus": node.available_gpus,
                "effective_available_gpus": effective_available_gpus,
                "total_gpus": node.total_gpus,
                "used_or_reserved_gpus": node.total_gpus - node.available_gpus,
                "placement_groups": matching_pgs[:8],
            }
        )
    if blockers:
        # Check for cross-namespace actor conflicts to provide better error messages
        conflict_info = _check_cross_namespace_conflicts(requested)
        error_msg = f"{context}: pinned node capacity check failed: required_by_node={requested} blockers={blockers}"
        if conflict_info:
            error_msg += f"\n\nPossible cross-namespace conflicts detected:\n{conflict_info}\n\nSuggestion: Coordinate with other developers or use a different node."
        raise RuntimeError(error_msg)


def _check_cross_namespace_conflicts(requested_node_ips: dict[str, int]) -> str:
    """Check if other namespaces have actors on the requested nodes.

    Returns a formatted string describing conflicts, or empty string if none found.
    """
    try:
        if not ray.is_initialized():
            return ""

        from ..config import RAY_NAMESPACE

        current_namespace = RAY_NAMESPACE
        all_actors = ray.util.list_named_actors(all_namespaces=True)

        # Group actors by namespace and node
        conflicts_by_namespace: dict[str, list[str]] = {}

        for actor_info in all_actors:
            actor_ns = actor_info.get("namespace")
            actor_name = actor_info.get("name", "")

            # Skip actors in current namespace
            if actor_ns == current_namespace:
                continue

            # Only check vLLM and Megatron actors (the ones that use pinned nodes)
            if not (actor_name.startswith("tinker_vllm_") or
                    actor_name.startswith("multinode_vllm_") or
                    actor_name.startswith("megatron_")):
                continue

            # Try to get actor's node location
            try:
                # Verify actor exists (we don't need the handle, just check it's alive)
                _ = ray.get_actor(actor_name, namespace=actor_ns)
                # Get placement group to find node
                pg_name = f"{actor_name}_pg"
                try:
                    pg = ray.util.get_placement_group(pg_name)
                    pg_info = ray.util.placement_group_table(pg)
                    bundles_to_node = pg_info.get("bundles_to_node_id", {})

                    # Check if any bundle is on requested nodes
                    for node_id in bundles_to_node.values():
                        for n in ray.nodes():
                            if n.get("NodeID") == node_id:
                                node_ip = n.get("NodeManagerAddress")
                                if node_ip in requested_node_ips:
                                    key = f"{actor_ns} (node: {node_ip})"
                                    if key not in conflicts_by_namespace:
                                        conflicts_by_namespace[key] = []
                                    conflicts_by_namespace[key].append(actor_name)
                                break
                except Exception:
                    pass
            except Exception:
                continue

        if not conflicts_by_namespace:
            return ""

        lines = []
        for ns_info, actors in sorted(conflicts_by_namespace.items()):
            lines.append(f"  - {ns_info}:")
            for actor in actors[:5]:  # Limit to 5 actors per namespace
                lines.append(f"    * {actor}")
            if len(actors) > 5:
                lines.append(f"    * ... and {len(actors) - 5} more")

        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"Failed to check cross-namespace conflicts: {e}")
        return ""



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

    nodes = [n for n in _list_alive_gpu_nodes() if n.node_ip in allowed]

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
