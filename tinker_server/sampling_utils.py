from __future__ import annotations

from typing import Any, Literal

StopReason = Literal["length", "stop", "eos"]

DEFAULT_EOS_TOKENS: frozenset[int] = frozenset({151645, 151643})

from .models.types import SampledSequence


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


def sampled_sequence_from_result(result: Any) -> SampledSequence:
    """Create SampledSequence from a backend generate() result.

    Handles legacy vs multi-LoRA attribute differences:
    - logprobs vs log_probs
    - optional stop_reason
    """
    logprobs = getattr(result, "logprobs", None) or getattr(result, "log_probs", None)
    stop_reason = resolve_stop_reason(
        stop_reason=getattr(result, "stop_reason", None),
        token_ids=result.token_ids,
    )
    return SampledSequence(tokens=result.token_ids, logprobs=logprobs, stop_reason=stop_reason)
