from __future__ import annotations

from typing import Literal

StopReason = Literal["length", "stop", "eos"]

DEFAULT_EOS_TOKENS: frozenset[int] = frozenset({151645, 151643})


def resolve_stop_reason(
    *,
    stop_reason: str | None,
    token_ids: list[int] | None,
    eos_tokens: frozenset[int] = DEFAULT_EOS_TOKENS,
) -> StopReason:
    if stop_reason in ("length", "stop", "eos"):
        return stop_reason
    if token_ids and token_ids[-1] in eos_tokens:
        return "stop"
    return "length"

