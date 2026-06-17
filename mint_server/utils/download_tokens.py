from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def _b64u_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64u_decode(data: str) -> bytes:
    s = str(data)
    padding = (-len(s)) % 4
    if padding:
        s += "=" * padding
    return base64.urlsafe_b64decode(s.encode("utf-8"))


def _sign(payload_bytes: bytes, *, secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).digest()


def make_download_token(payload: dict[str, Any], *, secret: str) -> str:
    if not secret:
        raise ValueError("secret is required")
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = _sign(payload_bytes, secret=secret)
    return f"{_b64u_encode(payload_bytes)}.{_b64u_encode(sig)}"


def verify_download_token(token: str, *, secret: str) -> dict[str, Any] | None:
    if not secret:
        return None
    if not token:
        return None
    head, dot, tail = token.partition(".")
    if dot != ".":
        return None
    try:
        payload_bytes = _b64u_decode(head)
        sig = _b64u_decode(tail)
    except Exception:
        return None

    expected = _sign(payload_bytes, secret=secret)
    if not hmac.compare_digest(sig, expected):
        return None

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and time.time() > float(exp):
        return None
    return payload


def make_archive_download_token(
    *,
    secret: str,
    user_id: str | None,
    model_id: str,
    checkpoint_id: str,
    ttl_s: int = 15 * 60,
) -> tuple[str, int]:
    now = int(time.time())
    exp = now + int(ttl_s)
    token = make_download_token(
        {
            "v": 1,
            "exp": exp,
            "user_id": user_id,
            "model_id": model_id,
            "checkpoint_id": checkpoint_id,
        },
        secret=secret,
    )
    return token, exp

