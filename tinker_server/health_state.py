from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DegradedState:
    reason: str
    error: str
    details: dict[str, Any]


_STARTUP_DEGRADED: DegradedState | None = None
_RUNTIME_DEGRADED: DegradedState | None = None


def _to_payload(state: DegradedState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "reason": state.reason,
        "error": state.error,
        "details": dict(state.details),
    }


def clear_startup_degraded_state() -> None:
    global _STARTUP_DEGRADED
    _STARTUP_DEGRADED = None


def clear_runtime_degraded_state() -> None:
    global _RUNTIME_DEGRADED
    _RUNTIME_DEGRADED = None


def set_startup_degraded_state(*, reason: str, error: str, details: dict[str, Any] | None = None) -> None:
    global _STARTUP_DEGRADED
    _STARTUP_DEGRADED = DegradedState(
        reason=str(reason),
        error=str(error),
        details={} if details is None else dict(details),
    )


def set_runtime_degraded_state(*, reason: str, error: str, details: dict[str, Any] | None = None) -> None:
    global _RUNTIME_DEGRADED
    _RUNTIME_DEGRADED = DegradedState(
        reason=str(reason),
        error=str(error),
        details={} if details is None else dict(details),
    )


def get_startup_degraded_state() -> dict[str, Any] | None:
    return _to_payload(_STARTUP_DEGRADED)


def get_runtime_degraded_state() -> dict[str, Any] | None:
    return _to_payload(_RUNTIME_DEGRADED)

