"""Dependency-free helpers for the sole MANO Mode4 CLI/session lifecycle."""

from __future__ import annotations

from collections.abc import Callable


def parse_ordered_unique_csv(value: str, *, option: str) -> list[int]:
    """Parse comma-separated integers, preserving first occurrence order."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{option} must be a non-empty comma-separated list of integers")

    result: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            index = int(part)
        except ValueError as exc:
            raise ValueError(f"{option} contains a non-integer value: {part!r}") from exc
        if index not in seen:
            seen.add(index)
            result.append(index)
    return result


def acquire_action_session(
    external_session_id: str | None, create_session: Callable[[], str]
) -> tuple[str, bool]:
    """Return an action session ID and whether this caller owns its cleanup."""
    if external_session_id:
        return external_session_id, False
    return create_session(), True


def action_session_payload(
    *, session_id: str, base_model: str, model_path: str, owner_id: str
) -> dict[str, str]:
    """Build the model-identity-preserving action-session request body."""
    return {
        "session_id": session_id,
        "base_model": base_model,
        "model_path": model_path,
        "owner_id": owner_id,
    }
