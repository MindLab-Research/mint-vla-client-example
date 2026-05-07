from __future__ import annotations

import asyncio
import os

from fastapi.responses import JSONResponse


_REDACTED_DETAIL_KEYS = {
    "last_error_traceback",
    "traceback",
    "stack",
    "stacktrace",
    "exc_info",
}


def _public_details(value):
    if isinstance(value, dict):
        out = {}
        for key, nested in value.items():
            key_text = str(key)
            if key_text.lower() in _REDACTED_DETAIL_KEYS:
                continue
            out[key_text] = _public_details(nested)
        return out
    if isinstance(value, list):
        return [_public_details(item) for item in value]
    if isinstance(value, tuple):
        return [_public_details(item) for item in value]
    return value


def _degraded_response() -> JSONResponse | None:
    from .health_state import get_runtime_degraded_state, get_startup_degraded_state

    degraded = get_runtime_degraded_state() or get_startup_degraded_state()
    if degraded is None:
        return None
    return JSONResponse(
        status_code=503,
        content={
            "status": "degraded",
            "reason": degraded.get("reason", "startup_degraded"),
            "error": degraded.get("error", ""),
            "details": _public_details(degraded.get("details", {})),
        },
    )


def public_healthz_response() -> dict | JSONResponse:
    """Return cheap public API-worker health without probing Ray or other backends."""
    degraded = _degraded_response()
    if degraded is not None:
        return degraded
    return {"status": "ready"}


async def deep_healthz_response() -> dict | JSONResponse:
    """Return internal deep health with Ray and placement-group observations."""
    degraded = _degraded_response()
    if degraded is not None:
        return degraded

    try:
        import ray

        from .config import RAY_NAMESPACE
        from .ray_utils import init_ray

        if not ray.is_initialized():
            init_ray(namespace=RAY_NAMESPACE, ignore_reinit_error=True)

        def _pending_gpu_pg_names_in_namespace() -> list[str]:
            tbl = ray.util.placement_group_table()
            candidates: set[str] = set()
            for info in tbl.values():
                if not isinstance(info, dict):
                    continue
                name = info.get("name")
                if not isinstance(name, str) or not name:
                    continue
                state = info.get("state")
                if state in ("CREATED", "REMOVED"):
                    continue
                candidates.add(name)

            pending: list[str] = []
            for name in sorted(candidates):
                try:
                    pg = ray.util.get_placement_group(name)
                except Exception:
                    continue
                try:
                    info = ray.util.placement_group_table(pg)
                except Exception:
                    continue
                state = info.get("state")
                if state in ("CREATED", "REMOVED"):
                    continue
                bundles = info.get("bundles") or {}
                total_gpu = 0.0
                for bundle in bundles.values():
                    if isinstance(bundle, dict):
                        total_gpu += float(bundle.get("GPU", 0) or 0)
                if total_gpu <= 0:
                    continue
                pending.append(name)
            return pending

        healthz_ray_timeout_s = float(os.environ.get("MINT_HEALTHZ_RAY_TIMEOUT_S", "10.0"))
        try:
            pending_pg_names = await asyncio.wait_for(
                asyncio.to_thread(_pending_gpu_pg_names_in_namespace),
                timeout=healthz_ray_timeout_s,
            )
        except asyncio.TimeoutError:
            return {
                "status": "ready",
                "ray_observation": {
                    "reason": "ray_healthz_timeout",
                    "timeout_s": healthz_ray_timeout_s,
                },
            }

        if pending_pg_names:
            available_resources = ray.available_resources()
            cluster_resources = ray.cluster_resources()
            return {
                "status": "ready",
                "ray_observation": {
                    "reason": "pending_placement_groups",
                    "pending_pg_count": len(pending_pg_names),
                    "pending_pg_names": pending_pg_names[:20],
                    "ray_gpu_available": float(available_resources.get("GPU", 0) or 0),
                    "ray_gpu_total": float(cluster_resources.get("GPU", 0) or 0),
                },
            }

        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": "ray_unavailable",
                "error": f"{type(e).__name__}: {e}",
            },
        )
