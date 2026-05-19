from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .runtime_env import env_nonempty

_CACHE_LOCK = threading.Lock()
_CACHE_AT_MONO = 0.0
_CACHE_VALUE: dict[str, Any] | None = None
_LAST_SUCCESS_AT_UNIX = 0.0

_EXACT_METRIC_NAMES = {
    "gcs_task_manager_task_events_reported",
    "gcs_task_manager_task_events_dropped",
    "gcs_task_manager_task_events_stored",
    "gcs_storage_operation_count",
    "gcs_placement_group_count",
    "gcs_actors_count",
    "grpc_server_req_new",
    "grpc_server_req_handling",
    "grpc_server_req_succeeded",
    "grpc_server_req_failed",
}

_HISTOGRAM_BASE_NAMES = {
    "gcs_storage_operation_latency_ms",
    "gcs_placement_group_creation_latency_ms",
    "gcs_placement_group_scheduling_latency_ms",
    "grpc_server_req_process_time_ms",
    "health_check_rpc_latency_ms",
}

_HISTOGRAM_SUFFIXES = {"bucket", "count", "sum"}


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


def _metrics_selected(metric_name: str) -> bool:
    if metric_name in _EXACT_METRIC_NAMES:
        return True
    for base in _HISTOGRAM_BASE_NAMES:
        if metric_name == base:
            return True
        prefix = f"{base}_"
        if metric_name.startswith(prefix) and metric_name[len(prefix) :] in _HISTOGRAM_SUFFIXES:
            return True
    return False


def _discover_candidate_addresses(ray: Any) -> tuple[list[str], list[str]]:
    override = os.environ.get("MINT_RAY_HEAD_METRICS_ADDRESS", "").strip()
    if override:
        return [override], [override]

    nodes = ray.nodes()
    if not isinstance(nodes, list):
        raise TypeError(f"ray.nodes returned non-list: {type(nodes)}")

    head_candidates: list[str] = []
    all_alive: list[str] = []
    seen: set[str] = set()

    for node in nodes:
        if not isinstance(node, dict) or not bool(node.get("Alive")):
            continue
        ip = str(node.get("NodeManagerAddress") or "").strip()
        port = int(node.get("MetricsExportPort") or 0)
        if not ip or port <= 0:
            continue
        address = f"{ip}:{port}"
        if address not in seen:
            all_alive.append(address)
            seen.add(address)

        is_head = bool(node.get("IsHeadNode"))
        if not is_head:
            resources = node.get("Resources")
            if isinstance(resources, dict):
                for key in resources:
                    if not isinstance(key, str):
                        continue
                    if key == "node:__internal_head__" or key.endswith(".__internal_head__"):
                        is_head = True
                        break
        if is_head and address not in head_candidates:
            head_candidates.append(address)

    return head_candidates, all_alive


def _scrape_metrics_text(address: str, *, timeout_s: float) -> str:
    req = urllib.request.Request(
        f"http://{address}/metrics",
        method="GET",
        headers={"Accept": "text/plain; version=0.0.4"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = resp.read()
    return payload.decode("utf-8", errors="replace")


def _extract_samples(text: str) -> list[dict[str, Any]]:
    from prometheus_client.parser import text_string_to_metric_families

    out: list[dict[str, Any]] = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if not _metrics_selected(sample.name):
                continue
            labels = dict(sample.labels)
            component = str(labels.get("Component") or "").strip().lower()
            if component and component != "gcs_server":
                continue
            out.append(
                {
                    "name": str(sample.name),
                    "labels": labels,
                    "value": float(sample.value),
                }
            )
    return out


def _aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for sample in samples:
        name = str(sample.get("name") or "")
        value = float(sample.get("value") or 0.0)
        aggregates[name] = float(aggregates.get(name, 0.0) + value)
    return aggregates


def _derived_values(aggregates: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}

    reported = float(aggregates.get("gcs_task_manager_task_events_reported", 0.0))
    dropped = float(aggregates.get("gcs_task_manager_task_events_dropped", 0.0))
    stored = float(aggregates.get("gcs_task_manager_task_events_stored", 0.0))

    if reported > 0:
        out["gcs_task_manager_task_events_drop_ratio"] = dropped / reported
        out["gcs_task_manager_task_events_store_ratio"] = stored / reported

    for base in _HISTOGRAM_BASE_NAMES:
        total = aggregates.get(f"{base}_sum")
        count = aggregates.get(f"{base}_count")
        if total is None or count is None or float(count) <= 0:
            continue
        out[f"{base}_mean"] = float(total) / float(count)

    return out


def _scrape_selected_metrics(addresses: list[str], *, timeout_s: float) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    samples: list[dict[str, Any]] = []
    sources_with_metrics: list[str] = []
    errors: list[dict[str, str]] = []
    for address in addresses:
        try:
            text = _scrape_metrics_text(address, timeout_s=timeout_s)
            extracted = _extract_samples(text)
            if extracted:
                sources_with_metrics.append(address)
                samples.extend(extracted)
        except Exception as e:
            errors.append({"address": address, "error": f"{type(e).__name__}: {e}"})
    return samples, sources_with_metrics, errors


def _collect_ray_gcs_metrics() -> dict[str, Any]:
    import ray

    from .config import RAY_NAMESPACE

    timeout_s = _float_env("MINT_RAY_GCS_METRICS_TIMEOUT_S", 2.0)
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized")

    head_candidates, all_alive = _discover_candidate_addresses(ray)
    scrape_started = time.perf_counter()

    candidate_addresses = head_candidates or all_alive
    samples, sources_with_metrics, errors = _scrape_selected_metrics(candidate_addresses, timeout_s=timeout_s)

    if not samples and head_candidates:
        remaining = [address for address in all_alive if address not in set(head_candidates)]
        extra_samples, extra_sources, extra_errors = _scrape_selected_metrics(remaining, timeout_s=timeout_s)
        samples.extend(extra_samples)
        sources_with_metrics.extend(extra_sources)
        errors.extend(extra_errors)

    aggregates = _aggregate_samples(samples)
    derived = _derived_values(aggregates)

    if samples:
        status = "ready" if not errors else "degraded"
    else:
        status = "unavailable"

    return {
        "status": status,
        "up": bool(samples),
        "namespace": RAY_NAMESPACE,
        "collected_at": _utc_now_iso(),
        "timeout_s": timeout_s,
        "candidate_addresses": candidate_addresses,
        "sources_with_metrics": sources_with_metrics,
        "scrape_error_count": len(errors),
        "scrape_errors": errors,
        "sample_count": len(samples),
        "samples": samples,
        "aggregates": aggregates,
        "derived": derived,
        "scrape_latency_ms": (time.perf_counter() - scrape_started) * 1000.0,
    }


def get_ray_gcs_metrics_snapshot(*, force_refresh: bool = False) -> dict[str, Any]:
    global _CACHE_AT_MONO, _CACHE_VALUE, _LAST_SUCCESS_AT_UNIX

    cache_ttl_s = _float_env("MINT_RAY_GCS_METRICS_CACHE_TTL_S", 15.0)
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
        snapshot = _collect_ray_gcs_metrics()
    except Exception as e:
        snapshot = {
            "status": "unavailable",
            "up": False,
            "namespace": env_nonempty(os.environ, "MINT_RAY_NAMESPACE") or "mint",
            "collected_at": _utc_now_iso(),
            "timeout_s": _float_env("MINT_RAY_GCS_METRICS_TIMEOUT_S", 2.0),
            "candidate_addresses": [],
            "sources_with_metrics": [],
            "scrape_error_count": 1,
            "scrape_errors": [{"address": "", "error": f"{type(e).__name__}: {e}"}],
            "sample_count": 0,
            "samples": [],
            "aggregates": {},
            "derived": {},
            "scrape_latency_ms": 0.0,
        }

    with _CACHE_LOCK:
        _CACHE_VALUE = dict(snapshot)
        _CACHE_AT_MONO = time.monotonic()
        if bool(snapshot.get("up")):
            _LAST_SUCCESS_AT_UNIX = float(time.time())
        last_success_unixtime = float(_LAST_SUCCESS_AT_UNIX) if _LAST_SUCCESS_AT_UNIX > 0 else None

    out = dict(snapshot)
    out["cached"] = False
    out["cache_age_s"] = 0.0
    out["cache_ttl_s"] = cache_ttl_s
    out["last_success_unixtime"] = last_success_unixtime
    out["last_success_age_s"] = max(0.0, time.time() - last_success_unixtime) if last_success_unixtime is not None else None
    return out
