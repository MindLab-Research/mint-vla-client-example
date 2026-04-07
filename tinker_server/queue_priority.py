from __future__ import annotations

from typing import Any, Mapping

from fastapi import Request

QUEUE_PRIORITY_HEADER = "x-mint-priority"
QUEUE_PRIORITY_MIN = 0
QUEUE_PRIORITY_MAX = 2
QUEUE_PRIORITY_DEFAULT = 0
QUEUE_PRIORITY_AGING_S = 60.0


def normalize_queue_priority(value: Any) -> int:
    try:
        priority = int(value)
    except Exception:
        return QUEUE_PRIORITY_DEFAULT
    if priority < QUEUE_PRIORITY_MIN:
        return QUEUE_PRIORITY_MIN
    if priority > QUEUE_PRIORITY_MAX:
        return QUEUE_PRIORITY_MAX
    return priority


def extract_queue_priority_from_headers(headers: Mapping[str, Any] | None) -> int:
    if headers is None:
        return QUEUE_PRIORITY_DEFAULT
    raw = None
    for key, value in headers.items():
        if str(key).strip().lower() == QUEUE_PRIORITY_HEADER:
            raw = value
            break
    return normalize_queue_priority(raw)


def get_request_queue_priority(request: Request | None) -> int:
    if request is None:
        return QUEUE_PRIORITY_DEFAULT
    return extract_queue_priority_from_headers(request.headers)


def merge_queue_priority_extra(extra: Mapping[str, Any] | None = None, *, request: Request | None = None) -> dict[str, Any]:
    merged = {} if extra is None else dict(extra)
    merged["queue_priority"] = normalize_queue_priority(
        merged.get("queue_priority")
        if "queue_priority" in merged
        else get_request_queue_priority(request)
    )
    return merged


def effective_queue_priority(*, raw_priority: int, created_at: float, now: float, aging_s: float = QUEUE_PRIORITY_AGING_S) -> int:
    wait_s = max(0.0, float(now) - float(created_at))
    if aging_s <= 0:
        promoted = 0
    else:
        promoted = int(wait_s // float(aging_s))
    return min(QUEUE_PRIORITY_MAX, normalize_queue_priority(raw_priority) + promoted)
