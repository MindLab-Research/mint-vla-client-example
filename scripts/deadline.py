"""Cooperative training deadline helpers."""

from __future__ import annotations

import time


def parse_stop_at(value: str) -> float:
    """Parse an ISO 8601 deadline with timezone into a Unix timestamp.

    Raises ValueError for naive timestamps or past deadlines.
    """
    from datetime import datetime
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        raise ValueError(f"--stop-at requires timezone, got naive: {value!r}")
    deadline = ts.timestamp()
    if deadline <= time.time():
        raise ValueError(f"--stop-at must be in the future, got {value!r}")
    return deadline
