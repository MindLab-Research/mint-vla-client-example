from __future__ import annotations

import asyncio
import os

from fastapi.responses import JSONResponse


def public_healthz_response() -> dict | JSONResponse:
    """Return public API-worker health without probing Ray or other backends."""
    return {"status": "ready"}


async def deep_healthz_response() -> dict | JSONResponse:
    """Return internal deep health with Ray and placement-group observations."""
    from .health_state import get_startup_degraded_state

    degraded = get_startup_degraded_state()
    if degraded is not None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": degraded.get("reason", "startup_degraded"),
                "error": degraded.get("error", ""),
                "details": degraded.get("details", {}),
            },
        )

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
