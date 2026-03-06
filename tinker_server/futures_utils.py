from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PendingFutureHttpResponse:
    status_code: int
    headers: dict[str, str]
    body: dict


def pending_future_http_response(
    *, retry_after_s: int = 1, throttled: bool = False
) -> PendingFutureHttpResponse:
    headers = {"Retry-After": str(int(retry_after_s))}
    if throttled:
        headers["X-Tinker-Poll-Throttled"] = "1"
    return PendingFutureHttpResponse(
        status_code=408,
        headers=headers,
        body={"queue_state": "active"},
    )
