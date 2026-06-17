from __future__ import annotations

from typing import Any, Literal, Sequence

from mint_server.models.types import SampledSequence

StopReason = Literal["length", "stop", "eos"]

# Default stop/EOS tokens:
# - Qwen-style: 151645, 151643
# - Moonlight: <|im_end|>=163586, [EOS]=163585
DEFAULT_EOS_TOKENS: frozenset[int] = frozenset({151645, 151643, 163586, 163585})



def normalize_prompt_logprobs_for_tinker(
    prompt_logprobs: Sequence[float | None], *, prompt_len: int
) -> list[float | None]:
    """Normalize prompt_logprobs to official Tinker semantics.

    Required invariants:
    - len(prompt_logprobs) == prompt_len
    - prompt_logprobs[0] is None for prompt_len > 0
    """
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be >= 0, got {prompt_len}")

    xs = list(prompt_logprobs)
    if prompt_len == 0:
        if xs:
            raise ValueError(f"Expected 0 prompt_logprobs for empty prompt, got {len(xs)}")
        return []

    if len(xs) == prompt_len:
        xs[0] = None
        return xs

    if len(xs) == prompt_len - 1:
        return [None, *xs]

    raise ValueError(f"Expected {prompt_len} or {prompt_len - 1} prompt_logprobs, got {len(xs)}")


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
    routed_experts = getattr(result, "routed_experts", None)
    if routed_experts is not None and hasattr(routed_experts, "tolist"):
        routed_experts = routed_experts.tolist()
    stop_reason = resolve_stop_reason(
        stop_reason=getattr(result, "stop_reason", None),
        token_ids=result.token_ids,
    )
    return SampledSequence(
        tokens=result.token_ids,
        logprobs=logprobs,
        routed_experts=routed_experts,
        stop_reason=stop_reason,
    )
