#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


CORE_CONTROL_PLANE_ACTOR_NAMES = (
    "mint_config",
    "mint_model_actor_supervisor",
    "mint_model_work_scheduler",
    "mint_maintenance_cron",
    "mint_task_state_store",
)


def split_csv(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    keys = {
        "actor_id",
        "name",
        "class_name",
        "pid",
        "ray_namespace",
        "start_time_ms",
        "state",
        "job_id",
        "node_id",
    }
    return {key: getattr(row, key) for key in keys if hasattr(row, key)}


def _row_value(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def is_mint_owned_actor(data: Mapping[str, Any]) -> bool:
    name = str(data.get("name") or "")
    class_name = str(data.get("class_name") or "")
    return (
        name.startswith("mint_")
        or class_name.startswith("_Ray")
        or "Model" in class_name
        or "Engine" in class_name
    )


def stale_actor_candidates(
    rows: Iterable[Any],
    *,
    driver_namespace: str,
    now_ms: float,
    max_age_s: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        data = row_to_dict(row)
        row_namespace = str(data.get("ray_namespace") or "")
        if not row_namespace or row_namespace == driver_namespace:
            continue
        if str(data.get("state") or "ALIVE") != "ALIVE":
            continue
        if not is_mint_owned_actor(data):
            continue
        start_ms_raw = _row_value(data, "start_time_ms", "startTimeMs") or 0
        try:
            start_ms = float(start_ms_raw)
        except (TypeError, ValueError):
            start_ms = 0.0
        if start_ms <= 0:
            continue
        age_s = max(0.0, (now_ms - start_ms) / 1000.0)
        if age_s < max_age_s:
            continue
        data["age_s"] = round(age_s, 3)
        candidates.append(data)
    return candidates


def clear_task_state_dir(task_state_dir: Path, safe_root: Path) -> dict[str, Any]:
    resolved_task_state_dir = task_state_dir.resolve(strict=False)
    resolved_safe_root = safe_root.resolve(strict=False)
    if (
        resolved_task_state_dir == resolved_safe_root
        or resolved_safe_root not in resolved_task_state_dir.parents
    ):
        raise RuntimeError(
            f"refusing to remove task state outside safe root: {resolved_task_state_dir}"
        )

    record: dict[str, Any] = {
        "task_state_dir": str(task_state_dir),
        "safe_root": str(safe_root),
        "resolved_task_state_dir": str(resolved_task_state_dir),
    }
    if task_state_dir.exists():
        file_count = 0
        total_bytes = 0
        for path in task_state_dir.rglob("*"):
            if path.is_file():
                file_count += 1
                try:
                    total_bytes += path.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(task_state_dir)
        task_state_dir.joinpath("payloads").mkdir(parents=True, exist_ok=True)
        record.update(
            {
                "status": "removed",
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
        )
    else:
        task_state_dir.joinpath("payloads").mkdir(parents=True, exist_ok=True)
        record["status"] = "missing_recreated"
    return record


def _json_write(path: str | None, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _dashboard_base_url() -> str | None:
    explicit = os.environ.get("MINT_RAY_DASHBOARD_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    for key in ("MINT_RAY_GCS_ADDRESS", "MINT_RAY_CLIENT_ADDRESS", "RAY_CLIENT_ADDRESS"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        parsed = urllib.parse.urlparse(raw if "://" in raw else f"ray://{raw}")
        host = parsed.hostname
        if host:
            return f"http://{host}:8265"
    return None


def _dashboard_result(path: str, *, timeout_s: float = 10.0) -> list[dict[str, Any]]:
    base_url = _dashboard_base_url()
    if not base_url:
        raise RuntimeError("MINT_RAY_DASHBOARD_URL or Ray head address is required")
    url = f"{base_url}{path}"
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        payload = json.load(response)
    result = payload.get("data", {}).get("result", {}).get("result")
    if not isinstance(result, list):
        raise RuntimeError(f"unexpected Ray dashboard response shape for {url}")
    return [dict(row) for row in result if isinstance(row, Mapping)]


def _dashboard_no_alive_reset_snapshot(namespace: str, pg_names: Sequence[str]) -> dict[str, Any] | None:
    if not _env_flag("MINT_DEV_RESET_SKIP_RAY_WHEN_NO_ALIVE"):
        return None
    try:
        encoded_namespace = urllib.parse.quote(namespace, safe="")
        actors = _dashboard_result(
            "/api/v0/actors?"
            "limit=10000&filter_keys=ray_namespace&filter_predicates=%3D"
            f"&filter_values={encoded_namespace}"
        )
        active_actors = [
            row
            for row in actors
            if str(row.get("state") or "").upper() not in {"", "DEAD"}
        ]
        placement_groups = _dashboard_result("/api/v0/placement_groups?limit=10000")
    except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
        return {
            "fast_reset_available": False,
            "fast_reset_error": f"{type(exc).__name__}: {exc}",
        }

    pg_name_set = set(pg_names)
    active_pgs = [
        row
        for row in placement_groups
        if str(row.get("state") or "") != "REMOVED" and str(row.get("name") or "") in pg_name_set
    ]
    return {
        "fast_reset_available": not active_actors and not active_pgs,
        "dashboard_actor_count": len(actors),
        "dashboard_alive_actor_count": len(
            [row for row in actors if str(row.get("state") or "").upper() == "ALIVE"]
        ),
        "dashboard_active_actor_count": len(active_actors),
        "dashboard_active_actors": [
            {
                "name": row.get("name"),
                "state": row.get("state"),
                "actor_id": row.get("actor_id"),
                "class_name": row.get("class_name"),
            }
            for row in active_actors[:20]
        ],
        "dashboard_active_reset_pg_count": len(active_pgs),
        "dashboard_active_reset_pgs": [
            {
                "name": row.get("name"),
                "state": row.get("state"),
                "placement_group_id": row.get("placement_group_id"),
            }
            for row in active_pgs
        ],
    }


def _ensure_repo_on_path() -> None:
    repo = os.environ.get("MINT_CODE_ROOT", "").strip()
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)


def _init_ray(namespace: str) -> Any:
    _ensure_repo_on_path()
    from mint_server.ray_utils import init_ray

    init_ray(namespace=namespace, ignore_reinit_error=True)
    import ray

    return ray


def _list_alive_actors(limit: int) -> tuple[list[Any], str | None]:
    try:
        from ray.util import state as ray_state

        rows = ray_state.list_actors(
            address=os.environ.get("MINT_RAY_GCS_ADDRESS")
            or os.environ.get("MINT_RAY_CLIENT_ADDRESS"),
            detail=True,
            limit=limit,
            timeout=60,
            filters=[("state", "=", "ALIVE")],
        )
        return list(rows), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def cmd_gc_stale_actors(args: argparse.Namespace) -> int:
    namespace = args.namespace or os.environ["MINT_RAY_NAMESPACE"]
    summary_path = args.summary or os.environ.get("MINT_DEV_ACTOR_GC_SUMMARY")
    results_path = args.results or os.environ.get("MINT_DEV_ACTOR_GC_RESULTS")
    max_age_s = float(args.max_age_s)
    limit = int(args.limit)
    ray = None
    if _env_flag("MINT_DEV_GC_DASHBOARD_LIST", default=True):
        try:
            rows: list[Any] = _dashboard_result(f"/api/v0/actors?limit={limit}")
            list_error = None
        except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            rows = []
            list_error = f"{type(exc).__name__}: {exc}"
    else:
        ray = _init_ray(namespace)
        rows, list_error = _list_alive_actors(limit)
    candidates = stale_actor_candidates(
        rows,
        driver_namespace=namespace,
        now_ms=time.time() * 1000.0,
        max_age_s=max_age_s,
    )

    counts: Counter[str] = Counter()
    results_handle = open(results_path, "w", encoding="utf-8") if results_path else None
    try:
        for index, data in enumerate(candidates, 1):
            if ray is None:
                ray = _init_ray(namespace)
            name = str(data.get("name") or "")
            row_namespace = str(data.get("ray_namespace") or "")
            record = {
                "index": index,
                "actor_id": str(data.get("actor_id") or ""),
                "name": name,
                "namespace": row_namespace,
                "class_name": str(data.get("class_name") or ""),
                "pid": data.get("pid"),
                "job_id": str(data.get("job_id") or ""),
                "node_id": str(data.get("node_id") or ""),
                "age_s": data.get("age_s"),
            }
            try:
                if not name or not row_namespace:
                    raise RuntimeError("candidate is missing name or namespace")
                actor = ray.get_actor(name, namespace=row_namespace)
                ray.kill(actor, no_restart=True)
                record["status"] = "killed"
                counts["killed"] += 1
            except Exception as exc:
                record["status"] = "error"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
                counts["error"] += 1
            if results_handle is not None:
                results_handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        if results_handle is not None:
            results_handle.close()

    summary = {
        "driver_address": os.environ.get("MINT_RAY_CLIENT_ADDRESS")
        or os.environ.get("RAY_CLIENT_ADDRESS"),
        "gcs_address": os.environ.get("MINT_RAY_GCS_ADDRESS"),
        "driver_namespace": namespace,
        "max_age_s": max_age_s,
        "list_limit": limit,
        "listed_alive_count": len(rows),
        "candidate_count": len(candidates),
        "counts": dict(counts),
        "list_error": list_error,
        "results_path": results_path,
        "note": "Ray exposes no Python API to purge DEAD historical actor records; this cleanup only kills stale ALIVE actors.",
    }
    _json_write(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


async def _active_task_probe() -> dict[str, int]:
    from mint_server.backend.task_state_store import TaskStateStoreClient

    client = TaskStateStoreClient()
    tasks = await client.async_list_active_tasks(limit=1000)
    future_tasks = await client.async_future_list_active_tasks(limit=1000)
    missing_future = 0
    for record in tasks:
        request_id = str(record.get("request_id") or "")
        if not request_id:
            continue
        try:
            await client.async_future_get_task(request_id)
        except Exception:
            missing_future += 1
    return {
        "task_active_count": len(tasks),
        "future_active_count": len(future_tasks),
        "missing_future_count": missing_future,
    }


def _expected_config_snapshot(namespace: str) -> Any:
    from mint_server.runtime_config import build_config_snapshot, config_actor_name

    return build_config_snapshot(
        ray_namespace=namespace,
        actor_name=config_actor_name(),
    )


def _probe_should_reset(ray: Any, namespace: str, mode: str) -> dict[str, Any]:
    if mode not in {"auto", ""}:
        return {
            "expected_config_actor": None,
            "expected_fingerprint": None,
            "actual_fingerprint": None,
            "actual_snapshot_error": None,
            "active_task_probe": None,
            "active_task_probe_error": None,
            "should_reset": True,
        }

    expected = _expected_config_snapshot(namespace)
    should_reset = False
    actual_fingerprint = None
    actual_snapshot_error = None
    active_task_probe = None
    active_task_probe_error = None
    config_name = os.environ.get("MINT_CONFIG_ACTOR_NAME", "mint_config")

    try:
        config_actor = ray.get_actor(config_name, namespace=namespace)
        actual_snapshot = ray.get(config_actor.get_snapshot.remote(), timeout=10.0)
        actual_fingerprint = str(actual_snapshot.get("fingerprint") or "")
        if actual_fingerprint != expected.fingerprint:
            should_reset = True
    except Exception as exc:
        actual_snapshot_error = f"{type(exc).__name__}: {exc}"
        should_reset = True

    try:
        active_task_probe = asyncio.run(_active_task_probe())
        if (
            int(active_task_probe.get("task_active_count") or 0) > 0
            or int(active_task_probe.get("future_active_count") or 0) > 0
            or int(active_task_probe.get("missing_future_count") or 0) > 0
        ):
            should_reset = True
    except Exception as exc:
        active_task_probe_error = f"{type(exc).__name__}: {exc}"

    return {
        "expected_config_actor": expected.actor_name,
        "expected_fingerprint": expected.fingerprint,
        "actual_fingerprint": actual_fingerprint,
        "actual_snapshot_error": actual_snapshot_error,
        "active_task_probe": active_task_probe,
        "active_task_probe_error": active_task_probe_error,
        "should_reset": should_reset,
    }


def _kill_named_actor(ray: Any, actor_name: str, namespace: str, timeout_s: float) -> dict[str, Any]:
    record: dict[str, Any] = {"actor_name": actor_name}
    try:
        actor = ray.get_actor(actor_name, namespace=namespace)
    except Exception as exc:
        record["status"] = "missing"
        record["error_type"] = type(exc).__name__
        return record

    try:
        ray.kill(actor, no_restart=True)
    except TypeError:
        ray.kill(actor)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            ray.get_actor(actor_name, namespace=namespace)
        except Exception:
            record["status"] = "killed"
            return record
        if time.monotonic() >= deadline:
            record["status"] = "still_exists"
            return record
        time.sleep(0.25)


def _remove_named_placement_group(pg_name: str) -> dict[str, Any]:
    record: dict[str, Any] = {"placement_group_name": pg_name}
    try:
        from ray.util import get_placement_group, remove_placement_group

        pg = get_placement_group(pg_name)
    except Exception as exc:
        record["status"] = "missing"
        record["error_type"] = type(exc).__name__
        return record
    try:
        remove_placement_group(pg)
        record["status"] = "removed"
    except Exception as exc:
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    return record


def _actor_names_from_env() -> list[str]:
    extra = split_csv(os.environ.get("MINT_DEV_RESET_ACTOR_NAMES"))
    return list(dict.fromkeys([*CORE_CONTROL_PLANE_ACTOR_NAMES, *extra]))


def cmd_reset_control_plane(args: argparse.Namespace) -> int:
    namespace = args.namespace or os.environ["MINT_RAY_NAMESPACE"]
    mode = args.mode
    summary_path = args.summary or os.environ.get("MINT_DEV_RESET_SUMMARY")
    pg_names = split_csv(os.environ.get("MINT_DEV_RESET_PLACEMENT_GROUP_NAMES"))

    fast_snapshot = _dashboard_no_alive_reset_snapshot(namespace, pg_names)
    if mode not in {"auto", ""} and fast_snapshot is not None:
        summary: dict[str, Any] = {
            "namespace": namespace,
            "reset_mode": mode,
            "expected_config_actor": None,
            "expected_fingerprint": None,
            "actual_fingerprint": None,
            "actual_snapshot_error": None,
            "active_task_probe": None,
            "active_task_probe_error": None,
            "should_reset": True,
            "actors": [],
            "placement_groups": [],
            "persistent_state": {"status": "not_configured"},
            **fast_snapshot,
        }
        if fast_snapshot.get("fast_reset_available") is True:
            task_state_dir = os.environ.get("MINT_DEV_RESET_TASK_STATE_DIR")
            safe_root = os.environ.get("MINT_DEV_RESET_TASK_STATE_SAFE_ROOT")
            if task_state_dir:
                if not safe_root:
                    raise RuntimeError(
                        "MINT_DEV_RESET_TASK_STATE_SAFE_ROOT is required when task state reset is enabled"
                    )
                summary["persistent_state"] = clear_task_state_dir(
                    Path(task_state_dir),
                    Path(safe_root),
                )
            _json_write(summary_path, summary)
            print(json.dumps(summary, sort_keys=True), file=sys.stderr)
            return 0

    ray = _init_ray(namespace)
    _ensure_repo_on_path()
    probe = _probe_should_reset(ray, namespace, mode)

    summary: dict[str, Any] = {
        "namespace": namespace,
        "reset_mode": mode,
        **probe,
        "actors": [],
        "placement_groups": [],
        "persistent_state": {"status": "not_configured"},
    }

    if not probe["should_reset"]:
        _json_write(summary_path, summary)
        print(json.dumps(summary, sort_keys=True), file=sys.stderr)
        return 0

    for actor_name in _actor_names_from_env():
        record = _kill_named_actor(ray, actor_name, namespace, float(args.actor_kill_timeout_s))
        summary["actors"].append(record)
        if record.get("status") == "still_exists":
            _json_write(summary_path, summary)
            raise RuntimeError(f"actor still exists after kill: {actor_name}")

    for pg_name in pg_names:
        summary["placement_groups"].append(_remove_named_placement_group(pg_name))

    task_state_dir = os.environ.get("MINT_DEV_RESET_TASK_STATE_DIR")
    safe_root = os.environ.get("MINT_DEV_RESET_TASK_STATE_SAFE_ROOT")
    if task_state_dir:
        if not safe_root:
            raise RuntimeError(
                "MINT_DEV_RESET_TASK_STATE_SAFE_ROOT is required when task state reset is enabled"
            )
        summary["persistent_state"] = clear_task_state_dir(
            Path(task_state_dir),
            Path(safe_root),
        )

    _json_write(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mint dev Ray cleanup utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gc_parser = subparsers.add_parser("gc-stale-actors")
    gc_parser.add_argument("--namespace", default=None)
    gc_parser.add_argument(
        "--max-age-s",
        type=float,
        default=float(os.environ.get("MINT_DEV_OLD_ACTOR_MAX_AGE_S", "259200")),
    )
    gc_parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("MINT_DEV_OLD_ACTOR_LIST_LIMIT", "10000")),
    )
    gc_parser.add_argument("--summary", default=None)
    gc_parser.add_argument("--results", default=None)
    gc_parser.set_defaults(func=cmd_gc_stale_actors)

    reset_parser = subparsers.add_parser("reset-control-plane")
    reset_parser.add_argument("--namespace", default=None)
    reset_parser.add_argument("--mode", default=os.environ.get("MINT_DEV_RESET_CONTROL_PLANE", "auto"))
    reset_parser.add_argument("--summary", default=None)
    reset_parser.add_argument("--actor-kill-timeout-s", type=float, default=15.0)
    reset_parser.set_defaults(func=cmd_reset_control_plane)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
