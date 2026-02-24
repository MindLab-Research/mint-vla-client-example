from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StartupDegradedState:
    reason: str
    error: str
    details: dict[str, Any]


_STARTUP_DEGRADED: StartupDegradedState | None = None


def clear_startup_degraded_state() -> None:
    global _STARTUP_DEGRADED
    _STARTUP_DEGRADED = None


def set_startup_degraded_state(*, reason: str, error: str, details: dict[str, Any] | None = None) -> None:
    global _STARTUP_DEGRADED
    _STARTUP_DEGRADED = StartupDegradedState(
        reason=str(reason),
        error=str(error),
        details={} if details is None else dict(details),
    )


def get_startup_degraded_state() -> dict[str, Any] | None:
    if _STARTUP_DEGRADED is None:
        return None
    return {
        "reason": _STARTUP_DEGRADED.reason,
        "error": _STARTUP_DEGRADED.error,
        "details": dict(_STARTUP_DEGRADED.details),
    }

