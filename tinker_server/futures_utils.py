from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PendingFutureHttpResponse:
    status_code: int
    headers: dict[str, str]
    body: dict


def pending_future_http_response(
    *,
    retry_after_s: int = 1,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict | None = None,
) -> PendingFutureHttpResponse:
    headers = {"Retry-After": str(int(retry_after_s))}
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    body = {"queue_state": "active", "retry_after_s": int(retry_after_s)}
    if extra_body:
        body.update(dict(extra_body))
    return PendingFutureHttpResponse(
        status_code=408,
        headers=headers,
        body=body,
    )
