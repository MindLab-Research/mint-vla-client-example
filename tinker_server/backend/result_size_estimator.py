from __future__ import annotations

import math

from ..models.types import ComputeLogprobsRequest, SampleRequest


def _clamp_nonneg(x: int) -> int:
    x = int(x)
    return x if x > 0 else 0


def estimate_sampling_result_bytes(req: SampleRequest) -> int:
    prompt_len = len(req.prompt.to_token_ids())
    max_new = _clamp_nonneg(req.sampling_params.max_tokens)
    num_samples = max(int(req.num_samples), 0)

    want_prompt_logprobs = bool(req.prompt_logprobs or req.include_prompt_logprobs)
    want_topk_prompt = int(req.topk_prompt_logprobs) if int(req.topk_prompt_logprobs) > 0 else 0

    # SampledSequence: tokens + optional logprobs
    # tokens: int list, logprobs: float list
    bytes_per_token = 8  # int64 in JSON is worse, but we store as Ray object; keep margin
    bytes_per_logprob = 8

    # vLLM returns logprobs for generated tokens in this codepath.
    per_generated_token = bytes_per_token + bytes_per_logprob
    per_sequence = per_generated_token * max_new

    # prompt_logprobs: list[float|None] of length prompt_len
    prompt_logprobs_bytes = 0
    if want_prompt_logprobs:
        prompt_logprobs_bytes = prompt_len * bytes_per_logprob + 64

    # topk_prompt_logprobs: per prompt token k pairs (token_id, logprob)
    topk_bytes = 0
    if want_topk_prompt > 0:
        topk_bytes = prompt_len * want_topk_prompt * (bytes_per_token + bytes_per_logprob) + 128

    # Response structure overhead and per-sequence stop_reason strings.
    overhead = 4096 + num_samples * 128

    # Safety factor for Python / Ray metadata overhead.
    raw = overhead + num_samples * per_sequence + prompt_logprobs_bytes + topk_bytes
    return int(math.ceil(raw * 2.0))


def estimate_compute_logprobs_result_bytes(req: ComputeLogprobsRequest) -> int:
    seq_len = len(req.sequence.to_token_ids())
    bytes_per_logprob = 8
    overhead = 2048
    raw = overhead + seq_len * bytes_per_logprob
    return int(math.ceil(raw * 2.0))


def estimate_small_result_bytes() -> int:
    # Training and checkpoint endpoints typically return small dicts and metadata.
    return 256 * 1024

