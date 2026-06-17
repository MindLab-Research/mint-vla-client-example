from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import structlog
from fastapi.responses import JSONResponse


logger = structlog.get_logger(__name__)


PUBLIC_HEALTHZ_CACHE_TTL_S = 30.0
PUBLIC_HEALTHZ_REFRESH_TIMEOUT_S = 5.0
PUBLIC_HEALTHZ_COMPONENT_TIMEOUT_S = 2.0
INTERNAL_HEALTHZ_STALE_AFTER_S = 60.0
INTERNAL_HEALTHZ_FUTURE_STORE_TIMEOUT_S = 2.0


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
    from mint_server.backend.scheduling.model_work_scheduler import model_work_scheduler
    from mint_server.backend.stores.task_state_store import task_state_store

    scheduler_ping = getattr(model_work_scheduler, "async_ping", None)
    if scheduler_ping is None:
        scheduler_ping = getattr(model_work_scheduler, "ping")
    task_ping = getattr(task_state_store, "async_ping", None)
    if task_ping is None:
        task_ping = getattr(task_state_store, "ping")

    component_timeout_s = min(timeout_s, PUBLIC_HEALTHZ_COMPONENT_TIMEOUT_S)

    _ping_t0 = time.perf_counter()
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
        from mint_server.observability.logging_context import record_public_healthz_refresh_metric

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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


async def _billing_outbox_health_snapshot() -> dict[str, object]:
    try:
        from mint_server.backend.stores.task_state_store import task_futures

        stats = await task_futures.async_billing_outbox_stats()
    except Exception as e:
        return {"status": "degraded", "error": f"{type(e).__name__}: {e}"}

    by_status = stats.get("by_status") if isinstance(stats, dict) else {}
    metrics = stats.get("metrics") if isinstance(stats, dict) else {}
    if not isinstance(by_status, dict):
        by_status = {}
    if not isinstance(metrics, dict):
        metrics = {}

    pending = by_status.get("pending") if isinstance(by_status.get("pending"), dict) else {}
    flushing = by_status.get("flushing") if isinstance(by_status.get("flushing"), dict) else {}
    failed = by_status.get("failed") if isinstance(by_status.get("failed"), dict) else {}
    pending_rows = int((pending or {}).get("rows") or 0)
    flushing_rows = int((flushing or {}).get("rows") or 0)
    failed_rows = int((failed or {}).get("rows") or 0)
    oldest_pending_age_s = float((pending or {}).get("oldest_age_s") or 0.0)
    oldest_failed_age_s = float((failed or {}).get("oldest_age_s") or 0.0)
    degraded_rows = int(_env_float("MINT_BILLING_OUTBOX_DEGRADED_ROWS", 10000.0))
    degraded_age_s = _env_float("MINT_BILLING_OUTBOX_DEGRADED_AGE_S", 900.0)
    permanent_errors = float(metrics.get("flush_permanent_error") or 0.0)

    reasons: list[str] = []
    if failed_rows > 0:
        reasons.append("failed_rows")
    if permanent_errors > 0:
        reasons.append("permanent_flush_errors")
    if pending_rows >= degraded_rows:
        reasons.append("pending_rows")
    if oldest_pending_age_s >= degraded_age_s:
        reasons.append("oldest_pending_age")

    return {
        "status": "degraded" if reasons else "ready",
        "reasons": reasons,
        "pending_rows": pending_rows,
        "flushing_rows": flushing_rows,
        "failed_rows": failed_rows,
        "oldest_pending_age_s": oldest_pending_age_s,
        "oldest_failed_age_s": oldest_failed_age_s,
        "flush_permanent_error_total": permanent_errors,
        "degraded_rows_threshold": degraded_rows,
        "degraded_age_s_threshold": degraded_age_s,
    }


async def _future_state_store_health_snapshot() -> dict[str, object]:
    timeout_s = _env_float(
        "MINT_INTERNAL_HEALTHZ_FUTURE_STORE_TIMEOUT_S",
        INTERNAL_HEALTHZ_FUTURE_STORE_TIMEOUT_S,
    )
    started = time.monotonic()
    try:
        from mint_server.backend.stores.task_state_store import task_futures

        ping = await asyncio.wait_for(
            task_futures.async_ping(timeout_s=timeout_s),
            timeout=max(0.001, timeout_s),
        )
    except asyncio.TimeoutError:
        return {
            "status": "degraded",
            "reason": "future_state_store_ping_timeout",
            "timeout_s": timeout_s,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "reason": "future_state_store_unavailable",
            "error": f"{type(e).__name__}: {e}",
        }

    return {
        "status": "ready",
        "latency_s": max(0.0, time.monotonic() - started),
        "store": ping.get("store") if isinstance(ping, dict) else None,
        "actor_name": ping.get("actor_name") if isinstance(ping, dict) else None,
    }


async def internal_lightweight_healthz_response() -> dict:
    """Internal operational health from process-local/cached control-plane state."""
    now = time.time()
    status = "ready"

    try:
        from mint_server.backend.actors.model_actor_supervisor import get_model_actor_supervisor

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
    billing_snapshot = await _billing_outbox_health_snapshot()
    if _cron_degraded_from_snapshot(billing_snapshot) and status == "ready":
        status = "degraded"
    future_store_snapshot = await _future_state_store_health_snapshot()
    if _cron_degraded_from_snapshot(future_store_snapshot) and status == "ready":
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
        "billing_outbox": billing_snapshot,
        "future_state_store": future_store_snapshot,
    }
