from __future__ import annotations

import math

from ..models.types import ComputeLogprobsRequest, ForwardBackwardRequest, ForwardRequest, SampleRequest


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


def _estimate_training_logprobs_result_bytes(data) -> int:
    num_items = len(data)
    total_targets = 0
    for datum in data:
        n = 0
        try:
            target = datum.loss_fn_inputs.get("target_tokens")
            if isinstance(target, dict):
                target_data = target.get("data")
                if isinstance(target_data, list):
                    n = len(target_data)
        except Exception:
            n = 0
        if n <= 0:
            try:
                n = len(datum.model_input.to_token_ids())
            except Exception:
                try:
                    # Multimodal VLA requests (for example pi0.5 flow matching)
                    # are not flattenable to token ids; use chunk length instead.
                    n = int(datum.model_input.length)
                except Exception:
                    n = 1
        total_targets += int(n)

    # Result payloads for training forward/forward_backward are Python dicts of Python lists
    # (see TrainingWorker.forward / forward_backward). Serialized overhead per float is
    # materially larger than 8 bytes; reserve conservatively to avoid object-store OOM.
    bytes_per_logprob = 64
    per_item_overhead = 4096
    overhead = 8192
    raw = overhead + total_targets * bytes_per_logprob + num_items * per_item_overhead
    return int(raw)


def estimate_forward_result_bytes(req: ForwardRequest) -> int:
    return _estimate_training_logprobs_result_bytes(req.forward_input.data)


def estimate_forward_backward_result_bytes(req: ForwardBackwardRequest) -> int:
    return _estimate_training_logprobs_result_bytes(req.forward_backward_input.data)


def estimate_small_result_bytes() -> int:
    # Training and checkpoint endpoints typically return small dicts and metadata.
    return 256 * 1024
