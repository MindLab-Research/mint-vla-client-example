from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_tinker_sdk_user_agent(user_agent: str | None) -> bool:
    if not user_agent:
        return False
    ua = user_agent.strip()
    if not ua:
        return False
    low = ua.lower()
    # Tinker SDK sets: "User-Agent: {ClientClassName}/Python {__version__}"
    # e.g. "AsyncTinker/Python 0.2.3".
    #
    # MinT's `mint` wrapper reuses the same SDK but sets "Mint/Python {ver}".
    if "/python " not in low:
        return False
    client_name = low.split("/", 1)[0]
    return client_name.endswith("tinker") or client_name == "mint"


def _get_user_agent_from_request(request: Any) -> str | None:
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    if not isinstance(headers, Mapping):
        return None
    return headers.get("user-agent") or headers.get("User-Agent")


def prefer_tinker_uri(request: Any) -> bool:
    return is_tinker_sdk_user_agent(_get_user_agent_from_request(request))


def checkpoint_uri(
    model_id: str,
    checkpoint_name: str,
    *,
    prefer_tinker: bool,
    checkpoint_type: str,
) -> str:
    scheme = "tinker" if prefer_tinker else "mint"

    if checkpoint_type == "training":
        return f"{scheme}://{model_id}/weights/{checkpoint_name}"
    if checkpoint_type == "sampler":
        return f"{scheme}://{model_id}/sampler_weights/{checkpoint_name}"
    raise ValueError(f"Unsupported checkpoint_type: {checkpoint_type!r}")
