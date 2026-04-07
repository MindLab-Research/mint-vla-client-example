from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

_CACHE_LOCK = threading.Lock()
_CACHE_AT_MONO = 0.0
_CACHE_VALUE: dict[str, Any] | None = None
_LAST_SUCCESS_AT_UNIX = 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _timed_probe(name: str, fn, probes: dict[str, dict[str, Any]]) -> Any:
    started = time.perf_counter()
    try:
        value = fn()
        probes[name] = {"ok": True, "latency_ms": (time.perf_counter() - started) * 1000.0}
        return value
    except Exception as e:
        probes[name] = {
            "ok": False,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": f"{type(e).__name__}: {e}",
        }
        return None


def _gpu_total_from_bundles(bundles: Any) -> float:
    total_gpu = 0.0
    if isinstance(bundles, dict):
        values = bundles.values()
    elif isinstance(bundles, list):
        values = bundles
    else:
        values = []
    for bundle in values:
        if isinstance(bundle, dict):
            total_gpu += float(bundle.get("GPU", 0) or 0)
    return total_gpu


def _pending_pg_snapshot(ray: Any, *, max_names: int) -> dict[str, Any]:
    tbl = ray.util.placement_group_table()
    if not isinstance(tbl, dict):
        raise TypeError(f"placement_group_table returned non-dict: {type(tbl)}")

    total = 0
    created = 0
    removed = 0
    pending = 0
    pending_gpu = 0
    pending_names: list[str] = []

    for info in tbl.values():
        if not isinstance(info, dict):
            continue
        total += 1
        state = str(info.get("state") or "")
        name = str(info.get("name") or "")
        if state == "CREATED":
            created += 1
            continue
        if state == "REMOVED":
            removed += 1
            continue
        pending += 1

        total_gpu = _gpu_total_from_bundles(info.get("bundles"))
        if total_gpu <= 0 and name:
            try:
                pg = ray.util.get_placement_group(name)
                details = ray.util.placement_group_table(pg)
                total_gpu = _gpu_total_from_bundles(details.get("bundles") if isinstance(details, dict) else None)
            except Exception:
                total_gpu = 0.0
        if total_gpu > 0:
            pending_gpu += 1
            if name and len(pending_names) < max_names:
                pending_names.append(name)

    return {
        "total": total,
        "created": created,
        "removed": removed,
        "pending": pending,
        "pending_gpu": pending_gpu,
        "pending_gpu_names": sorted(pending_names),
    }


def _named_actor_snapshot(ray: Any, *, namespace: str) -> dict[str, Any]:
    actors = ray.util.list_named_actors(all_namespaces=True)
    if not isinstance(actors, list):
        raise TypeError(f"list_named_actors returned non-list: {type(actors)}")
    total = 0
    namespace_total = 0
    for actor in actors:
        if not isinstance(actor, dict):
            continue
        total += 1
        if str(actor.get("namespace") or "") == namespace:
            namespace_total += 1
    return {"total": total, "namespace": namespace_total}


def _node_snapshot(ray: Any, *, max_sample: int) -> dict[str, Any]:
    nodes = ray.nodes()
    if not isinstance(nodes, list):
        raise TypeError(f"ray.nodes returned non-list: {type(nodes)}")

    alive = 0
    dead = 0
    dead_missing_heartbeats = 0
    dead_missing_heartbeat_ips: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        is_alive = bool(node.get("Alive"))
        if is_alive:
            alive += 1
            continue
        dead += 1
        death_message = str(node.get("DeathReasonMessage") or "")
        if "missing too many heartbeats" in death_message.lower():
            dead_missing_heartbeats += 1
            ip = str(node.get("NodeManagerAddress") or "")
            if ip and len(dead_missing_heartbeat_ips) < max_sample:
                dead_missing_heartbeat_ips.append(ip)
    return {
        "alive": alive,
        "dead": dead,
        "dead_missing_heartbeats": dead_missing_heartbeats,
        "dead_missing_heartbeat_ips": dead_missing_heartbeat_ips,
    }


def _resource_snapshot(ray: Any) -> dict[str, Any]:
    cluster = ray.cluster_resources()
    available = ray.available_resources()
    if not isinstance(cluster, dict):
        raise TypeError(f"cluster_resources returned non-dict: {type(cluster)}")
    if not isinstance(available, dict):
        raise TypeError(f"available_resources returned non-dict: {type(available)}")
    return {
        "cpu_total": float(cluster.get("CPU", 0) or 0),
        "cpu_available": float(available.get("CPU", 0) or 0),
        "gpu_total": float(cluster.get("GPU", 0) or 0),
        "gpu_available": float(available.get("GPU", 0) or 0),
        "memory_total": float(cluster.get("memory", 0) or 0),
        "memory_available": float(available.get("memory", 0) or 0),
        "object_store_memory_total": float(cluster.get("object_store_memory", 0) or 0),
        "object_store_memory_available": float(available.get("object_store_memory", 0) or 0),
    }


def _collect_ray_cluster_health() -> dict[str, Any]:
    import ray

    from .config import RAY_NAMESPACE
    from .ray_utils import init_ray

    max_pending_pg_names = _int_env("MINT_RAY_CLUSTER_HEALTH_MAX_PENDING_PG_NAMES", 20)
    max_dead_node_sample = _int_env("MINT_RAY_CLUSTER_HEALTH_MAX_DEAD_NODE_SAMPLE", 10)
    slow_probe_ms = _float_env("MINT_RAY_CLUSTER_HEALTH_SLOW_PROBE_MS", 2000.0)

    if not ray.is_initialized():
        init_ray(namespace=RAY_NAMESPACE, ignore_reinit_error=True)

    probes: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    nodes = _timed_probe(
        "nodes",
        lambda: _node_snapshot(ray, max_sample=max_dead_node_sample),
        probes,
    )
    resources = _timed_probe("resources", lambda: _resource_snapshot(ray), probes)
    placement_groups = _timed_probe(
        "placement_groups",
        lambda: _pending_pg_snapshot(ray, max_names=max_pending_pg_names),
        probes,
    )
    named_actors = _timed_probe(
        "named_actors",
        lambda: _named_actor_snapshot(ray, namespace=RAY_NAMESPACE),
        probes,
    )

    warnings: list[str] = []
    probe_errors = [name for name, rec in probes.items() if not bool(rec.get("ok"))]
    slow_probes = [
        name for name, rec in probes.items() if float(rec.get("latency_ms", 0.0) or 0.0) >= slow_probe_ms
    ]
    if probe_errors:
        warnings.append("probe_errors")
    if slow_probes:
        warnings.append("slow_control_plane_probes")
    if isinstance(nodes, dict) and int(nodes.get("dead_missing_heartbeats", 0) or 0) > 0:
        warnings.append("dead_nodes_missing_heartbeats")
    if isinstance(placement_groups, dict) and int(placement_groups.get("pending_gpu", 0) or 0) > 0:
        warnings.append("pending_gpu_placement_groups")

    if probe_errors and all(not bool(rec.get("ok")) for rec in probes.values()):
        status = "unavailable"
    elif warnings:
        status = "degraded"
    else:
        status = "ready"

    return {
        "status": status,
        "up": status != "unavailable",
        "namespace": RAY_NAMESPACE,
        "collected_at": _utc_now_iso(),
        "probes": probes,
        "warnings": warnings,
        "warning_count": len(warnings),
        "probe_error_count": len(probe_errors),
        "slow_probe_count": len(slow_probes),
        "slow_probe_threshold_ms": slow_probe_ms,
        "nodes": nodes if isinstance(nodes, dict) else {},
        "resources": resources if isinstance(resources, dict) else {},
        "placement_groups": placement_groups if isinstance(placement_groups, dict) else {},
        "named_actors": named_actors if isinstance(named_actors, dict) else {},
        "total_probe_latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def get_ray_cluster_health_snapshot(*, force_refresh: bool = False) -> dict[str, Any]:
    global _CACHE_AT_MONO, _CACHE_VALUE, _LAST_SUCCESS_AT_UNIX

    cache_ttl_s = _float_env("MINT_RAY_CLUSTER_HEALTH_CACHE_TTL_S", 15.0)
    now = time.monotonic()
    wall_now = time.time()
    with _CACHE_LOCK:
        if not force_refresh and _CACHE_VALUE is not None and (now - _CACHE_AT_MONO) < cache_ttl_s:
            cached = dict(_CACHE_VALUE)
            cached["cached"] = True
            cached["cache_age_s"] = max(0.0, now - _CACHE_AT_MONO)
            cached["cache_ttl_s"] = cache_ttl_s
            cached["last_success_unixtime"] = float(_LAST_SUCCESS_AT_UNIX) if _LAST_SUCCESS_AT_UNIX > 0 else None
            cached["last_success_age_s"] = max(0.0, wall_now - float(_LAST_SUCCESS_AT_UNIX)) if _LAST_SUCCESS_AT_UNIX > 0 else None
            return cached

    try:
        snapshot = _collect_ray_cluster_health()
    except Exception as e:
        snapshot = {
            "status": "unavailable",
            "up": False,
            "namespace": os.environ.get("TINKER_RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE") or "tinker",
            "collected_at": _utc_now_iso(),
            "probes": {},
            "warnings": ["collector_error"],
            "warning_count": 1,
            "probe_error_count": 1,
            "slow_probe_count": 0,
            "slow_probe_threshold_ms": _float_env("MINT_RAY_CLUSTER_HEALTH_SLOW_PROBE_MS", 2000.0),
            "nodes": {},
            "resources": {},
            "placement_groups": {},
            "named_actors": {},
            "error": f"{type(e).__name__}: {e}",
            "total_probe_latency_ms": 0.0,
        }

    with _CACHE_LOCK:
        _CACHE_VALUE = dict(snapshot)
        _CACHE_AT_MONO = time.monotonic()
        if bool(snapshot.get("up")):
            _LAST_SUCCESS_AT_UNIX = float(time.time())
        current_age_s = 0.0
        last_success_unixtime = float(_LAST_SUCCESS_AT_UNIX) if _LAST_SUCCESS_AT_UNIX > 0 else None

    out = dict(snapshot)
    out["cached"] = False
    out["cache_age_s"] = current_age_s
    out["cache_ttl_s"] = cache_ttl_s
    out["last_success_unixtime"] = last_success_unixtime
    out["last_success_age_s"] = max(0.0, time.time() - last_success_unixtime) if last_success_unixtime is not None else None
    return out
