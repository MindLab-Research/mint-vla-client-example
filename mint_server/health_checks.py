from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi.responses import JSONResponse


PUBLIC_HEALTHZ_CACHE_TTL_S = 30.0
PUBLIC_HEALTHZ_REFRESH_TIMEOUT_S = 5.0
PUBLIC_HEALTHZ_COMPONENT_TIMEOUT_S = 2.0
INTERNAL_HEALTHZ_STALE_AFTER_S = 60.0


@dataclass
class _PublicHealthzCache:
    value: dict | None = None
    cached_at: float = 0.0
    lock: asyncio.Lock | None = None


_public_healthz_cache = _PublicHealthzCache()


class _PublicDependencyUnhealthy(RuntimeError):
    pass


def _public_ready() -> dict:
    return {"status": "ready"}


def _public_unhealthy() -> JSONResponse:
    return JSONResponse(status_code=503, content={"status": "unhealthy"})


def reset_public_healthz_cache() -> None:
    _public_healthz_cache.value = None
    _public_healthz_cache.cached_at = 0.0
    _public_healthz_cache.lock = None


def public_healthz_cache_age_seconds() -> float | None:
    cached_at = float(_public_healthz_cache.cached_at or 0.0)
    if _public_healthz_cache.value is None or cached_at <= 0.0:
        return None
    return max(0.0, time.monotonic() - cached_at)


async def _ping_public_dependencies(*, timeout_s: float) -> None:
    from .backend.model_work_scheduler import model_work_scheduler
    from .backend.task_state_store import task_futures, task_state_store

    scheduler_ping = getattr(model_work_scheduler, "async_ping", None)
    if scheduler_ping is None:
        scheduler_ping = getattr(model_work_scheduler, "ping")
    task_ping = getattr(task_state_store, "async_ping", None)
    if task_ping is None:
        task_ping = getattr(task_futures, "async_ping", None)
    if task_ping is None:
        task_ping = getattr(task_state_store, "ping")

    component_timeout_s = min(timeout_s, PUBLIC_HEALTHZ_COMPONENT_TIMEOUT_S)

    async def _call_ping(fn) -> object:
        try:
            out = fn(timeout_s=component_timeout_s)
            if asyncio.iscoroutine(out):
                out = await out
            if isinstance(out, dict) and not bool(out.get("ok", True)):
                raise _PublicDependencyUnhealthy(f"ping returned not ok: {out!r}")
            return out
        except _PublicDependencyUnhealthy:
            raise
        except Exception as e:
            raise _PublicDependencyUnhealthy(f"{type(e).__name__}: {e}") from e

    await asyncio.gather(
        _call_ping(scheduler_ping),
        _call_ping(task_ping),
    )


async def public_business_healthz_response() -> dict | JSONResponse:
    """Public business health for Tinker clients.

    Only the control-plane components required to accept and track business work
    are checked. Successful values are cached per API process for 30s; failed
    refreshes are never served from stale cache.
    """
    now = time.monotonic()
    cached = _public_healthz_cache.value
    if cached is not None and now - _public_healthz_cache.cached_at <= PUBLIC_HEALTHZ_CACHE_TTL_S:
        return dict(cached)

    if _public_healthz_cache.lock is None:
        _public_healthz_cache.lock = asyncio.Lock()
    async with _public_healthz_cache.lock:
        now = time.monotonic()
        cached = _public_healthz_cache.value
        if cached is not None and now - _public_healthz_cache.cached_at <= PUBLIC_HEALTHZ_CACHE_TTL_S:
            return dict(cached)
        try:
            await asyncio.wait_for(
                _ping_public_dependencies(timeout_s=PUBLIC_HEALTHZ_REFRESH_TIMEOUT_S),
                timeout=PUBLIC_HEALTHZ_REFRESH_TIMEOUT_S,
            )
        except TimeoutError:
            _public_healthz_cache.value = None
            _public_healthz_cache.cached_at = 0.0
            _record_public_healthz_refresh_metric("timeout")
            return _public_unhealthy()
        except asyncio.TimeoutError:
            _public_healthz_cache.value = None
            _public_healthz_cache.cached_at = 0.0
            _record_public_healthz_refresh_metric("timeout")
            return _public_unhealthy()
        except _PublicDependencyUnhealthy:
            _public_healthz_cache.value = None
            _public_healthz_cache.cached_at = 0.0
            _record_public_healthz_refresh_metric("unhealthy")
            return _public_unhealthy()
        except Exception:
            _public_healthz_cache.value = None
            _public_healthz_cache.cached_at = 0.0
            _record_public_healthz_refresh_metric("error")
            return _public_unhealthy()

        value = _public_ready()
        _public_healthz_cache.value = value
        _public_healthz_cache.cached_at = time.monotonic()
        _record_public_healthz_refresh_metric("ready")
        return dict(value)


def _record_public_healthz_refresh_metric(result: str) -> None:
    try:
        from .logging_context import record_public_healthz_refresh_metric

        record_public_healthz_refresh_metric(result=result)
    except Exception:
        pass


def _float_ts(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _snapshot_timestamp(snapshot: dict[str, object]) -> float | None:
    for key in ("snapshot_generated_at", "observed_at"):
        ts = _float_ts(snapshot.get(key))
        if ts is not None:
            return ts
    topology = snapshot.get("topology")
    if isinstance(topology, dict):
        ts = _float_ts(topology.get("observed_at"))
        if ts is not None:
            return ts
    return None


def _cached_maintenance_cron_snapshot() -> dict[str, object]:
    try:
        from .health_state import get_runtime_degraded_state

        degraded = get_runtime_degraded_state()
        if degraded is None:
            return {"status": "ready"}
        reason = str(degraded.get("reason") or "")
        if reason.startswith("maintenance_cron_actor"):
            return {"status": "degraded", **degraded}
        return {"status": "ready"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _cached_startup_snapshot() -> dict[str, object]:
    try:
        from .health_state import get_startup_degraded_state

        degraded = get_startup_degraded_state()
        if degraded is None:
            return {"status": "ready"}
        return {"status": "degraded", **degraded}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _cron_degraded_from_snapshot(snapshot: dict[str, object]) -> bool:
    if not snapshot:
        return False
    if "error" in snapshot:
        return True
    return str(snapshot.get("status") or "").lower() in {"degraded", "unhealthy", "unreachable"}


async def internal_lightweight_healthz_response() -> dict:
    """Internal operational health from process-local/cached control-plane state."""
    now = time.time()
    status = "ready"

    try:
        from .backend.model_actor_supervisor import get_model_actor_supervisor

        supervisor = get_model_actor_supervisor()
        supervisor_snapshot = supervisor.snapshot()
        if not isinstance(supervisor_snapshot, dict):
            raise TypeError(f"supervisor snapshot returned {type(supervisor_snapshot)}")
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_actor_supervisor": {"error": f"{type(e).__name__}: {e}"},
            "maintenance_cron_actor": {},
        }

    snapshot_ts = _snapshot_timestamp(supervisor_snapshot)
    if snapshot_ts is None or now - snapshot_ts > INTERNAL_HEALTHZ_STALE_AFTER_S:
        status = "degraded"

    cron_snapshot = _cached_maintenance_cron_snapshot()
    if _cron_degraded_from_snapshot(cron_snapshot) and status == "ready":
        status = "degraded"
    startup_snapshot = _cached_startup_snapshot()
    if _cron_degraded_from_snapshot(startup_snapshot) and status == "ready":
        status = "degraded"

    observed_at = supervisor_snapshot.get("observed_at")
    topology = supervisor_snapshot.get("topology")
    if observed_at is None and isinstance(topology, dict):
        observed_at = topology.get("observed_at")

    return {
        "status": status,
        "model_actor_supervisor": {
            "snapshot_generated_at": snapshot_ts,
            "observed_at": observed_at,
            "desired_total": supervisor_snapshot.get("desired_total"),
            "managed_total": supervisor_snapshot.get("managed_total"),
            "last_reconcile_at": supervisor_snapshot.get("last_reconcile_at"),
        },
        "maintenance_cron_actor": cron_snapshot,
        "startup": startup_snapshot,
    }
