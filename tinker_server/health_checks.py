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
        from .backend.async_ray_control import async_pending_gpu_pg_observation

        healthz_ray_timeout_s = float(os.environ.get("MINT_HEALTHZ_RAY_TIMEOUT_S", "10.0"))
        try:
            ray_observation = await async_pending_gpu_pg_observation(timeout_s=healthz_ray_timeout_s)
        except asyncio.TimeoutError:
            return {
                "status": "ready",
                "ray_observation": {
                    "reason": "ray_healthz_timeout",
                    "timeout_s": healthz_ray_timeout_s,
                },
            }

        if ray_observation is not None:
            return {"status": "ready", "ray_observation": ray_observation}

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
