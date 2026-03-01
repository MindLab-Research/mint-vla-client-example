#!/usr/bin/env python3
"""R3 router replay validation tool (v3).

Uses DAPO-Math-17k as prompt source, but trains with Tinker `loss_fn="ppo"` only.

Key metrics tracked:
- logprobs_diff_mean/max/std: Training-inference mismatch (TIM)
- actor_rollout_pearson: Correlation between training and rollout probs
- ppo_kl_divergence: Policy divergence
- gradient_norm: Training stability indicator

Quick Start:
-----------
# 1. Run baseline (R3 disabled)
ssh mint-dev 'pkill -f run_server; sleep 2'
ssh mint-dev 'cd /vePFS-Mindverse/share/code/tinker-server && \
  MINT_ROUTER_REPLAY_MODE=disabled nohup python scripts/run_server.py &'
sleep 10
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
  python scripts/wip/r3_router_replay_smoke_v3.py --steps 100

# 2. Run with R3 enabled
ssh mint-dev 'pkill -f run_server; sleep 2'
ssh mint-dev 'cd /vePFS-Mindverse/share/code/tinker-server && \
  MINT_ROUTER_REPLAY_MODE=R3 nohup python scripts/run_server.py &'
sleep 10
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
  python scripts/wip/r3_router_replay_smoke_v3.py --steps 100
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"
DEFAULT_MAX_REQUEST_TRAINING_LOAD = 256
DEFAULT_PROBLEMS_PER_BATCH = 256
DEFAULT_NUM_SAMPLES = 8
DEFAULT_MAX_PROMPT_LENGTH = 1024 * 2
DEFAULT_MAX_RESPONSE_LENGTH = 128
DEFAULT_STEPS = 50

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _base_url(args: argparse.Namespace) -> str:
    base = _coalesce(
        getattr(args, "base_url", None),
        os.environ.get("TINKER_BASE_URL"),
        os.environ.get("MINT_BASE_URL"),
        DEFAULT_BASE_URL,
    ) or DEFAULT_BASE_URL
    return str(base).rstrip("/")


def _parse_tokens(token_str: str) -> list[int]:
    if not token_str.strip():
        return []
    out: list[int] = []
    for part in token_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _headers(args: argparse.Namespace) -> dict[str, str]:
    api_key = _coalesce(
        getattr(args, "api_key", None),
        os.environ.get("TINKER_API_KEY"),
        os.environ.get("MINT_API_KEY"),
    )
    return {"X-API-Key": api_key} if api_key else {}


def _get(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"GET {url} returned non-dict JSON: {type(out)}")
    return out


def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"POST {url} returned non-dict JSON: {type(out)}")
    return out


def _truncate_prompt_tokens(tokens: list[int], max_prompt_length: int) -> list[int]:
    if max_prompt_length <= 0:
        return tokens
    if len(tokens) <= max_prompt_length:
        return tokens
    # Match DAPO data.truncation='left': keep the rightmost tokens.
    return tokens[-max_prompt_length:]


def _wait_future(
    *,
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    timeout_s: float,
    poll_s: float = 2.0,
) -> dict[str, Any]:
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"retrieve_future timeout after {elapsed:.1f}s request_id={request_id}")
        r = requests.post(
            f"{base_url}/api/v1/retrieve_future",
            headers=headers,
            json={"request_id": request_id},
            timeout=min(120.0, timeout_s),
        )
        if r.status_code == 408:
            time.sleep(poll_s)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"retrieve_future -> {r.status_code}: {r.text[:500]!r}")
        r.raise_for_status()
        out = r.json()
        if not isinstance(out, dict):
            raise RuntimeError(f"retrieve_future returned non-dict: {type(out)}")
        return out


def _parallel_map_bounded(
    *,
    items: list[Any],
    worker_fn: Any,
    max_workers: int,
    op_name: str,
) -> list[Any]:
    if not items:
        return []

    workers = max(int(max_workers), 1)
    workers = min(workers, len(items))
    out: list[Any] = [None] * len(items)
    ex = ThreadPoolExecutor(max_workers=workers)
    failed = False
    try:
        future_to_idx = {ex.submit(worker_fn, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                out[idx] = fut.result()
            except Exception as e:
                # Fail fast on the first error and cancel pending futures.
                failed = True
                for pending in future_to_idx:
                    if pending is not fut:
                        pending.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"{op_name} failed at idx={idx}: {e}") from e
    finally:
        if not failed:
            ex.shutdown(wait=True)
    return out


def _parallel_map_bounded_return_exceptions(
    *,
    items: list[Any],
    worker_fn: Any,
    max_workers: int,
) -> list[Any | Exception]:
    if not items:
        return []

    workers = max(int(max_workers), 1)
    workers = min(workers, len(items))
    out: list[Any | Exception] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(worker_fn, item): idx for idx, item in enumerate(items)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                out[idx] = fut.result()
            except Exception as e:
                out[idx] = e
    return out


def _tokenize_many_bounded(
    *,
    base_url: str,
    headers: dict[str, str],
    texts: list[str],
    model: str,
    max_prompt_length: int,
) -> list[list[int]]:
    def _one(text: str) -> list[int]:
        tokens = _tokenize_text(base_url, headers, text, model)
        return _truncate_prompt_tokens(tokens, max_prompt_length)

    return [_one(text) for text in texts]


def _submit_asample_and_wait_many_bounded(
    *,
    base_url: str,
    headers: dict[str, str],
    sample_reqs: list[dict[str, Any]],
    submit_timeout_s: float,
    future_timeout_s: float,
    concurrency: int = DEFAULT_MAX_REQUEST_TRAINING_LOAD,
) -> list[dict[str, Any]]:
    asample_url = f"{base_url}/api/v1/asample"

    def _submit_and_wait_one(req: dict[str, Any]) -> dict[str, Any]:
        sample_fut = _post(asample_url, headers, req, submit_timeout_s)
        request_id = sample_fut.get("request_id")
        if not request_id:
            raise RuntimeError(f"asample missing request_id: {sample_fut}")
        return _wait_future(
            base_url=base_url,
            headers=headers,
            request_id=str(request_id),
            timeout_s=future_timeout_s,
        )

    out = _parallel_map_bounded(
        items=sample_reqs,
        worker_fn=_submit_and_wait_one,
        max_workers=concurrency,
        op_name="asample submit+wait",
    )
    return [dict(item) for item in out]


def _detokenize_many_bounded(
    *,
    token_batches: list[list[int]],
    model: str,
) -> list[str | Exception]:
    try:
        _get_local_tokenizer(model)
    except Exception as e:
        return [e for _ in token_batches]

    def _one(tokens: list[int]) -> str:
        return _detokenize_tokens(tokens, model)

    out: list[str | Exception] = []
    for tokens in token_batches:
        try:
            out.append(_one(tokens))
        except Exception as e:
            out.append(e)
    return [str(item) if isinstance(item, str) else item for item in out]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dapo_math_dataset(split: str = "train", num_samples: int | None = None) -> list[dict[str, Any]]:
    """Load DAPO-Math-17k dataset.

    Dataset: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
    Contains 1.79M math problems with prompts and ground truth answers.

    Returns list of dicts with 'prompt' (str) and 'answer' (str) keys.
    """
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not installed. Run: pip install datasets")

    print(f"Loading DAPO-Math-17k {split} split...", flush=True)
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split=split)

    problems = []
    for item in ds:
        # Extract prompt text from conversation format
        prompt_messages = item.get("prompt", [])
        if not prompt_messages:
            continue

        # Get user message content
        prompt_text = ""
        for msg in prompt_messages:
            if msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break

        if not prompt_text:
            continue

        # Extract ground truth answer
        reward_model = item.get("reward_model", {})
        answer = reward_model.get("ground_truth", "")

        problems.append({
            "prompt": prompt_text,
            "answer": answer,
            "data_source": item.get("data_source", ""),
            "ability": item.get("ability", ""),
        })

        if num_samples is not None and len(problems) >= num_samples:
            break

    print(f"Loaded {len(problems)} problems from DAPO-Math-17k {split}", flush=True)
    return problems


# ---------------------------------------------------------------------------
# Answer extraction and grading
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Extract answer from generated text.

    Supports multiple formats:
    - \\boxed{answer}
    - Answer: answer
    - #### answer (GSM8K format)
    """
    import re

    # Try \\boxed{} format (MATH dataset)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try "Answer:" format
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Try "####" format (GSM8K)
    gsm_match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if gsm_match:
        return gsm_match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    import re
    # Remove whitespace
    answer = answer.strip()
    # Remove commas from numbers
    answer = answer.replace(',', '')
    # Remove dollar signs
    answer = answer.replace('$', '')
    # Remove trailing periods
    answer = answer.rstrip('.')
    return answer.lower()


def grade_answer(generated: str, ground_truth: str) -> bool:
    """Grade generated answer against ground truth.

    Returns True if correct, False otherwise.
    """
    extracted = extract_answer(generated)
    if extracted is None:
        return False

    # Normalize both answers
    extracted_norm = normalize_answer(extracted)
    gt_norm = normalize_answer(ground_truth)

    # Simple string match
    return extracted_norm == gt_norm


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _tokenize_text(base_url: str, headers: dict[str, str], text: str, model: str) -> list[int]:
    """Tokenize text using local tokenizer (no server route)."""
    tokenizer = _get_local_tokenizer(model)
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    if not isinstance(tokens, list):
        raise RuntimeError(f"local tokenize returned invalid tokens: {type(tokens)}")
    return [int(t) for t in tokens]


_LOCAL_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_local_tokenizer(model: str) -> Any:
    """Get or load local HF tokenizer for detokenization."""
    tok = _LOCAL_TOKENIZER_CACHE.get(model)
    if tok is not None:
        return tok

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Local tokenize/detokenize requires transformers. Install it or use --no-grade-with-detokenize."
        ) from e

    try:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load local tokenizer for {model!r} (local_files_only=True). "
            "Ensure model tokenizer files exist locally, or use --no-grade-with-detokenize."
        ) from e

    _LOCAL_TOKENIZER_CACHE[model] = tok
    return tok


def _detokenize_tokens(tokens: list[int], model: str) -> str:
    """Detokenize tokens to text using local tokenizer (no server route)."""
    tokenizer = _get_local_tokenizer(model)
    return str(
        tokenizer.decode(
            [int(t) for t in tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _create_session(base_url: str, headers: dict[str, str], script_tag: str, timeout_s: float) -> str:
    payload = {"script_tag": script_tag}
    resp = _post(f"{base_url}/api/v1/create_session", headers, payload, timeout_s)
    session_id = resp.get("session_id")
    if not session_id:
        raise RuntimeError(f"create_session missing session_id: {resp}")
    return str(session_id)


def _create_sampling_session(
    base_url: str, headers: dict[str, str], session_id: str, base_model: str, timeout_s: float
) -> str:
    payload = {"session_id": session_id, "base_model": base_model}
    resp = _post(f"{base_url}/api/v1/create_sampling_session", headers, payload, timeout_s)
    sampling_session_id = resp.get("sampling_session_id")
    if not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {resp}")
    return str(sampling_session_id)


def _create_model(
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    base_model: str,
    lora_rank: int,
    timeout_s: float,
) -> str:
    payload = {
        "session_id": session_id,
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
    }
    resp = _post(f"{base_url}/api/v1/create_model", headers, payload, timeout_s)
    request_id = resp.get("request_id")
    if not request_id:
        raise RuntimeError(f"create_model missing request_id: {resp}")
    out = _wait_future(base_url=base_url, headers=headers, request_id=str(request_id), timeout_s=timeout_s)
    model_id = out.get("model_id")
    if not model_id:
        raise RuntimeError(f"create_model missing model_id: {out}")
    return str(model_id)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_loss_mask(prompt_len: int, response_len: int, total_len: int) -> list[float]:
    """Build shifted loss mask aligned with causal LM next-token logprobs.

    For tokenized sequence [prompt || response], the logprob at index i predicts token i+1.
    To score the first response token, mask must start at (prompt_len - 1).
    """
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be > 0, got {prompt_len}")
    if response_len < 0:
        raise ValueError(f"response_len must be >= 0, got {response_len}")
    if prompt_len + response_len != total_len:
        raise ValueError(f"prompt_len={prompt_len} + response_len={response_len} != total_len={total_len}")

    mask = [0.0] * total_len
    start = max(prompt_len - 1, 0)
    end = min(start + response_len, total_len)
    for i in range(start, end):
        mask[i] = 1.0
    if (end - start) != response_len:
        raise ValueError(
            f"shifted loss_mask length mismatch: scored={end - start} response_len={response_len} total_len={total_len}"
        )
    return mask


def _normalize_logprobs(logprobs: list[float | None]) -> list[float]:
    """Convert None to 0.0."""
    return [lp if lp is not None else 0.0 for lp in logprobs]


def _align_logprobs(logprobs: list[float], mode: str) -> list[float]:
    """Align logprobs for training.

    mode='shifted': shift left by 1 (standard causal LM alignment)
    mode='token': no shift (logprobs[i] corresponds to token[i])
    """
    if mode == "shifted":
        return logprobs[1:] + [0.0]
    elif mode == "token":
        return logprobs
    else:
        raise ValueError(f"Unknown logprobs_align mode: {mode}")


def _flatten_routed_experts(routed_experts: list, expected_seq_len: int | None) -> tuple[list[int], list[int]]:
    """Flatten nested routed_experts list to (flat_data, shape)."""
    if not routed_experts:
        return [], []
    if not isinstance(routed_experts, list):
        raise ValueError(f"routed_experts must be a list, got {type(routed_experts)}")

    seq_len = len(routed_experts)
    if expected_seq_len is not None and seq_len != expected_seq_len:
        raise ValueError(f"routed_experts seq_len={seq_len} != expected={expected_seq_len}")

    if not isinstance(routed_experts[0], list) or not routed_experts[0]:
        return [], []

    num_layers = len(routed_experts[0])
    if not isinstance(routed_experts[0][0], list):
        raise ValueError("routed_experts first layer must be a list")
    topk = len(routed_experts[0][0]) if routed_experts[0][0] else 0

    flat = []
    for pos in routed_experts:
        if not isinstance(pos, list) or len(pos) != num_layers:
            raise ValueError("routed_experts layer count mismatch across sequence")
        for layer in pos:
            if not isinstance(layer, list) or len(layer) != topk:
                raise ValueError("routed_experts topk mismatch across layers")
            flat.extend(layer)

    return flat, [seq_len, num_layers, topk]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv_header(path: Path, columns: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def _append_csv_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _get_metric(metrics: dict[str, Any], *keys: str) -> float | None:
    """Return the first numeric metric found across key aliases."""
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_logprobs_from_loss_output(output: dict[str, Any]) -> list[float]:
    """Extract flat logprobs list from one train_step loss_fn_output entry."""
    if not isinstance(output, dict):
        return []
    logprobs = output.get("logprobs")
    if isinstance(logprobs, list):
        return [float(x) for x in logprobs]
    if isinstance(logprobs, dict):
        data = logprobs.get("data")
        if isinstance(data, list):
            return [float(x) for x in data]
    return []


def _compute_ppo_kl_from_train_output(
    data_items: list[dict[str, Any]],
    loss_fn_outputs: Any,
) -> tuple[float | None, float | None]:
    """Compute cookbook-style PPO KL estimates from sampled vs training logprobs.

    Matches tinker-cookbook `compute_kl_sample_train`:
      v1 = mean(logp_sample - logp_train)
      v2 = 0.5 * mean((logp_sample - logp_train)^2)
    """
    if not isinstance(loss_fn_outputs, list) or not loss_fn_outputs:
        return None, None

    diffs: list[float] = []
    n = min(len(data_items), len(loss_fn_outputs))
    for idx in range(n):
        item = data_items[idx]
        output = loss_fn_outputs[idx]

        if not isinstance(item, dict):
            continue
        loss_fn_inputs = item.get("loss_fn_inputs")
        if not isinstance(loss_fn_inputs, dict):
            continue

        logprobs_field = loss_fn_inputs.get("logprobs")
        if not isinstance(logprobs_field, dict):
            continue
        old_full = logprobs_field.get("data")
        if not isinstance(old_full, list):
            continue

        weights_field = loss_fn_inputs.get("weights")
        if not isinstance(weights_field, dict):
            continue
        weights = weights_field.get("data")
        if not isinstance(weights, list):
            continue

        if len(old_full) != len(weights):
            continue

        old_actions = [float(lp) for lp, w in zip(old_full, weights) if float(w) != 0.0]
        if not old_actions:
            continue

        new_logprobs = _extract_logprobs_from_loss_output(output)
        if not new_logprobs:
            continue

        # Training backends usually return response-token logprobs directly.
        if len(new_logprobs) == len(old_actions):
            pairs = zip(old_actions, new_logprobs)
        # Fallback if backend returns full-sequence logprobs.
        elif len(new_logprobs) == len(weights):
            new_actions = [float(lp) for lp, w in zip(new_logprobs, weights) if float(w) != 0.0]
            m = min(len(old_actions), len(new_actions))
            if m <= 0:
                continue
            pairs = zip(old_actions[:m], new_actions[:m])
        else:
            m = min(len(old_actions), len(new_logprobs))
            if m <= 0:
                continue
            pairs = zip(old_actions[:m], new_logprobs[:m])

        for old_lp, new_lp in pairs:
            diffs.append(float(old_lp) - float(new_lp))

    if not diffs:
        return None, None

    v1 = sum(diffs) / len(diffs)
    v2 = 0.5 * sum(d * d for d in diffs) / len(diffs)
    return float(v1), float(v2)


# ---------------------------------------------------------------------------
# Main experiment: PPO training with DAPO-Math prompts
# ---------------------------------------------------------------------------

def run_ppo_experiment(args: argparse.Namespace) -> int:
    """Run PPO training experiment tracking TIM metrics."""
    base_url = _base_url(args)
    headers = _headers(args)
    if int(args.problems_per_batch) <= 0:
        raise ValueError("--problems-per-batch must be > 0")
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be > 0")
    if int(args.max_prompt_length) <= 0:
        raise ValueError("--max-prompt-length must be > 0")
    if int(args.max_response_length) <= 0:
        raise ValueError("--max-response-length must be > 0")
    if float(args.clip_ratio_low) < 0 or float(args.clip_ratio_high) < 0:
        raise ValueError("--clip-ratio-low/high must be >= 0")
    if float(args.top_p) <= 0.0 or float(args.top_p) > 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if int(args.max_request_training_load) <= 0:
        raise ValueError("--max-request-training-load must be > 0")

    # Get server info
    info = _get(f"{base_url}/api/v1/server_info", headers, timeout_s=120.0)
    config = info.get("config", {}) if isinstance(info, dict) else {}
    router_replay_mode = config.get("router_replay_mode", "unknown")
    git_sha = info.get("git_sha")

    _ = _get(f"{base_url}/api/v1/get_server_capabilities", headers, timeout_s=120.0)
    base_model = args.base_model or DEFAULT_MODEL
    fixed_prompt_tokens = _parse_tokens(args.prompt_tokens)
    if args.prompt_source == "fixed" and not fixed_prompt_tokens:
        raise ValueError("--prompt-tokens is empty but --prompt-source=fixed")
    if args.prompt_source == "dataset":
        try:
            _get_local_tokenizer(base_model)
        except Exception as e:
            raise RuntimeError(
                f"prompt_source=dataset requires local tokenizer for {base_model!r}, but load failed: {e}"
            ) from e
    grade_with_detokenize = bool(args.grade_with_detokenize)
    if grade_with_detokenize and args.prompt_source != "dataset":
        grade_with_detokenize = False
        print(
            "[Warning] prompt_source=fixed has no ground-truth answers; disable grading automatically "
            "and use unit advantages.",
            flush=True,
        )
    if grade_with_detokenize:
        try:
            _get_local_tokenizer(base_model)
        except Exception as e:
            grade_with_detokenize = False
            print(
                f"[Warning] Local detokenizer unavailable, disable grading automatically: {e}",
                flush=True,
            )

    print(f"=== R3 Router Replay Experiment (v3) ===", flush=True)
    print(f"Server: {base_url}", flush=True)
    print(f"Router replay mode: {router_replay_mode}", flush=True)
    print(f"Base model: {base_model}", flush=True)
    print(f"Git SHA: {git_sha}", flush=True)
    print(f"Steps: {args.steps}", flush=True)
    print(f"Problems per batch: {args.problems_per_batch}", flush=True)
    print(f"Responses per problem: {args.num_samples}", flush=True)
    print(f"Expected samples per step: {args.problems_per_batch * args.num_samples}", flush=True)
    print(f"Max prompt length: {args.max_prompt_length}", flush=True)
    print(f"Max response length: {args.max_response_length}", flush=True)
    print(f"Sampling: temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}", flush=True)
    print(
        f"PPO clip ratios: low={args.clip_ratio_low} high={args.clip_ratio_high} "
        f"(bounds=[{1.0-float(args.clip_ratio_low):.3f}, {1.0+float(args.clip_ratio_high):.3f}])",
        flush=True,
    )
    print(f"Max request training load: {args.max_request_training_load}", flush=True)
    print(f"LoRA rank: {args.lora_rank}", flush=True)
    print(f"Prompt source: {args.prompt_source}", flush=True)
    if args.prompt_source == "fixed":
        print(f"Prompt tokens: {len(fixed_prompt_tokens)} tokens (v2-style fixed prompt)", flush=True)
    print(f"Grade with detokenize: {grade_with_detokenize}", flush=True)
    print(
        "Note: train_prompt_mini_bsz/ppo_micro_batch_size_per_gpu are server-side in tinker-server.",
        flush=True,
    )
    print(flush=True)

    # Prompt source:
    # - fixed (default): v2-style token prompts.
    # - dataset: DAPO-Math text prompts, requires local tokenizer.
    problems: list[dict[str, Any]] = []
    if args.prompt_source == "dataset":
        problems = load_dapo_math_dataset(split="train", num_samples=args.dataset_size)
        if not problems:
            raise ValueError(
                "prompt_source=dataset but no valid prompts were loaded from DAPO-Math-17k "
                f"(dataset_size={args.dataset_size})"
            )

    # Create sessions
    session_id = _create_session(base_url, headers, script_tag="scripts/wip/r3_router_replay_smoke_v3.py", timeout_s=120.0)
    sampling_session_id = _create_sampling_session(base_url, headers, session_id=session_id, base_model=base_model, timeout_s=120.0)
    model_id = _create_model(
        base_url, headers, session_id=session_id, base_model=base_model,
        lora_rank=args.lora_rank, timeout_s=float(args.timeout_s)
    )

    print(f"Session ID: {session_id}", flush=True)
    print(f"Sampling session ID: {sampling_session_id}", flush=True)
    print(f"Model ID: {model_id}", flush=True)
    print(flush=True)

    # Setup output directory
    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "r3_router_replay" / _utc_ts()
    run_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = "r3" if router_replay_mode == "R3" else "no_r3"
    out_csv = Path(args.out) if args.out else run_dir / f"{mode_suffix}_ppo_curve.csv"
    meta_path = out_csv.with_suffix(".meta.json")

    meta = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "git_sha": git_sha,
        "router_replay_mode": router_replay_mode,
        "base_model": base_model,
        "num_steps": int(args.steps),
        "problems_per_batch": int(args.problems_per_batch),
        "responses_per_problem": int(args.num_samples),
        "expected_samples_per_step": int(args.problems_per_batch * args.num_samples),
        "max_prompt_length": int(args.max_prompt_length),
        "max_response_length": int(args.max_response_length),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "lora_rank": int(args.lora_rank),
        "learning_rate": float(args.learning_rate),
        "epsilon": float(args.epsilon),
        "clip_ratio_low": float(args.clip_ratio_low),
        "clip_ratio_high": float(args.clip_ratio_high),
        "clip_low_bound": float(1.0 - float(args.clip_ratio_low)),
        "clip_high_bound": float(1.0 + float(args.clip_ratio_high)),
        "max_request_training_load": int(args.max_request_training_load),
        "dataset": "DAPO-Math-17k (prompt source)" if args.prompt_source == "dataset" else "fixed_prompt_tokens",
        "prompt_source": str(args.prompt_source),
        "prompt_tokens_len": int(len(fixed_prompt_tokens)),
        "grade_with_detokenize": bool(grade_with_detokenize),
        "training_loss_fn": "ppo",
        "dataset_size": len(problems) if args.prompt_source == "dataset" else 0,
    }
    _write_meta(meta_path, meta)

    columns = [
        "ts", "step", "router_replay_mode",
        "rollout_probs_diff_mean", "rollout_probs_diff_max", "rollout_probs_diff_std",
        "rollout_probs_diff_valid", "actor_rollout_pearson",
        # Backward-compat aliases
        "logprobs_diff_mean", "logprobs_diff_max", "logprobs_diff_std", "logprobs_diff_valid",
        "ppo_kl_divergence", "gradient_norm", "loss",
        "num_samples", "num_problems", "avg_response_len", "avg_prompt_len", "accuracy",
    ]
    _write_csv_header(out_csv, columns)

    print(f"Output CSV: {out_csv}", flush=True)
    print(f"Metadata: {meta_path}", flush=True)
    print(flush=True)

    # Shuffle dataset prompts for variety (dataset mode only)
    if args.prompt_source == "dataset":
        random.shuffle(problems)
    problem_iter = 0

    # PPO training loop
    for step in range(int(args.steps)):
        step_start = time.time()

        # Build one training batch with N prompts, each with M responses.
        batch_problems: list[dict[str, Any]] = []
        if args.prompt_source == "dataset":
            for _ in range(int(args.problems_per_batch)):
                if problem_iter >= len(problems):
                    random.shuffle(problems)
                    problem_iter = 0
                batch_problems.append(problems[problem_iter])
                problem_iter += 1
        else:
            # v2-style fixed prompt tokens.
            batch_problems = [{} for _ in range(int(args.problems_per_batch))]

        print(
            f"[Step {step}] Batch config: problems={len(batch_problems)} "
            f"responses_per_problem={args.num_samples} "
            f"max_request_training_load={args.max_request_training_load}",
            flush=True,
        )

        if args.prompt_source == "dataset":
            prompt_tokens_list = _tokenize_many_bounded(
                base_url=base_url,
                headers=headers,
                texts=[str(problem.get("prompt", "")) for problem in batch_problems],
                model=base_model,
                max_prompt_length=int(args.max_prompt_length),
            )
        else:
            prompt_tokens_list = [list(fixed_prompt_tokens) for _ in range(int(args.problems_per_batch))]

        # Submit all sampling requests with bounded concurrency.
        sample_jobs: list[dict[str, Any]] = []
        sample_reqs: list[dict[str, Any]] = []
        for problem_idx, (problem, prompt_tokens) in enumerate(zip(batch_problems, prompt_tokens_list, strict=True)):

            sample_reqs.append({
                "sampling_session_id": sampling_session_id,
                "seq_id": int(step) * int(args.problems_per_batch) + int(problem_idx),
                "num_samples": int(args.num_samples),
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
                "sampling_params": {
                    "max_tokens": int(args.max_response_length),
                    "temperature": float(args.temperature),
                    "top_k": int(args.top_k),
                    "top_p": float(args.top_p),
                },
                "prompt_logprobs": True,
            })

            sample_jobs.append(
                {
                    "problem": problem,
                    "prompt_tokens": prompt_tokens,
                }
            )

        # Prepare training data
        data_items = []
        total_response_len = 0
        total_prompt_len = 0
        num_correct = 0
        num_sequences = 0
        pending_sequences: list[dict[str, Any]] = []

        sample_outs = _submit_asample_and_wait_many_bounded(
            base_url=base_url,
            headers=headers,
            sample_reqs=sample_reqs,
            submit_timeout_s=120.0,
            future_timeout_s=float(args.timeout_s),
            concurrency=int(args.max_request_training_load),
        )

        for problem_idx, (sample_job, sample_out) in enumerate(zip(sample_jobs, sample_outs, strict=True)):
            problem = sample_job["problem"]
            prompt_tokens = sample_job["prompt_tokens"]

            sequences = sample_out.get("sequences") or []
            if not sequences:
                if isinstance(sample_out, dict) and sample_out.get("error"):
                    raise RuntimeError(
                        f"asample failed at step={step} problem_idx={problem_idx}: {sample_out.get('error')}"
                    )
                raise RuntimeError(f"asample returned no sequences: {sample_out}")

            prompt_logprobs = sample_out.get("prompt_logprobs")
            if not isinstance(prompt_logprobs, list):
                raise RuntimeError("prompt_logprobs missing")

            if len(sequences) != int(args.num_samples):
                raise RuntimeError(
                    f"step={step} problem_idx={problem_idx} expected {args.num_samples} responses, "
                    f"got {len(sequences)}"
                )

            for seq_idx, seq in enumerate(sequences):
                gen_tokens = seq.get("tokens")
                seq_logprobs = seq.get("logprobs")
                routed_experts = seq.get("routed_experts")

                if not isinstance(gen_tokens, list) or not gen_tokens:
                    raise RuntimeError(f"invalid tokens in problem[{problem_idx}] sequence[{seq_idx}]")
                if not isinstance(seq_logprobs, list) or len(seq_logprobs) != len(gen_tokens):
                    raise RuntimeError(f"problem[{problem_idx}] sequence[{seq_idx}] logprobs invalid")

                total_response_len += len(gen_tokens)
                total_prompt_len += len(prompt_tokens)
                num_sequences += 1

                full_tokens = list(prompt_tokens) + list(gen_tokens)
                full_logprobs_raw = _normalize_logprobs(list(prompt_logprobs) + list(seq_logprobs))
                aligned_logprobs = _align_logprobs(full_logprobs_raw, mode="shifted")
                if len(aligned_logprobs) != len(full_tokens):
                    raise RuntimeError(
                        f"logprobs/tokens length mismatch at step={step} problem_idx={problem_idx} "
                        f"seq_idx={seq_idx}: aligned_logprobs={len(aligned_logprobs)} full_tokens={len(full_tokens)} "
                        f"(prompt_logprobs={len(prompt_logprobs)} prompt_tokens={len(prompt_tokens)} "
                        f"seq_logprobs={len(seq_logprobs)} gen_tokens={len(gen_tokens)})"
                    )

                prompt_len = len(prompt_tokens)
                response_len = len(gen_tokens)
                weights = _build_loss_mask(prompt_len, response_len, len(full_tokens))
                pending_sequences.append(
                    {
                        "problem_idx": problem_idx,
                        "seq_idx": seq_idx,
                        "problem": problem,
                        "gen_tokens": gen_tokens,
                        "full_tokens": full_tokens,
                        "aligned_logprobs": aligned_logprobs,
                        "weights": weights,
                        "routed_experts": routed_experts,
                    }
                )

        detok_results: list[str | Exception | None]
        if grade_with_detokenize:
            detok_results = _detokenize_many_bounded(
                token_batches=[item["gen_tokens"] for item in pending_sequences],
                model=base_model,
            )
        else:
            # PPO-only compatibility path (matches v2 behavior): use unit advantages on response tokens.
            detok_results = [None] * len(pending_sequences)

        for item, detok_result in zip(pending_sequences, detok_results, strict=True):
            problem_idx = int(item["problem_idx"])
            seq_idx = int(item["seq_idx"])
            problem = item["problem"]
            full_tokens = item["full_tokens"]
            aligned_logprobs = item["aligned_logprobs"]
            weights = item["weights"]
            routed_experts = item["routed_experts"]

            if detok_result is None:
                advantages = [1.0 if w != 0.0 else 0.0 for w in weights]
            elif isinstance(detok_result, Exception):
                print(
                    f"[Step {step}] Warning: answer grading skipped/failed for problem {problem_idx} seq {seq_idx}: {detok_result}",
                    flush=True,
                )
                advantages = [1.0 if w != 0.0 else 0.0 for w in weights]
            else:
                generated_text = detok_result
                ground_truth = problem.get("answer")
                if isinstance(ground_truth, str) and ground_truth.strip():
                    is_correct = grade_answer(generated_text, ground_truth)
                    if is_correct:
                        num_correct += 1
                    reward = 1.0 if is_correct else 0.0
                    advantages = [reward if w != 0.0 else 0.0 for w in weights]
                else:
                    # If a sample has no ground truth, fall back to unit advantages
                    # to avoid zero-gradient training steps.
                    advantages = [1.0 if w != 0.0 else 0.0 for w in weights]

            loss_fn_inputs: dict[str, Any] = {
                "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
                "logprobs": {"data": aligned_logprobs, "shape": [len(aligned_logprobs)], "dtype": "float32"},
                "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
            }

            if routed_experts is not None:
                flat, shape = _flatten_routed_experts(routed_experts, expected_seq_len=None)
                loss_fn_inputs["routed_experts"] = {"data": flat, "shape": shape, "dtype": "int64"}

            data_items.append(
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": full_tokens}]},
                    "loss_fn_inputs": loss_fn_inputs,
                }
            )

        expected_step_samples = int(args.problems_per_batch) * int(args.num_samples)
        if num_sequences != expected_step_samples:
            raise RuntimeError(f"step={step} expected {expected_step_samples} samples, got {num_sequences}")

        avg_response_len = total_response_len / max(num_sequences, 1)
        avg_prompt_len = total_prompt_len / max(num_sequences, 1)
        accuracy = (num_correct / max(num_sequences, 1)) if grade_with_detokenize else None

        # Training step
        train_step_req = {
            "type": "train_step",
            "model_id": model_id,
            "forward_backward_input": {
                "loss_fn": "ppo",
                "loss_fn_config": {
                    "epsilon": float(args.epsilon),
                    "clip_low": float(1.0 - float(args.clip_ratio_low)),
                    "clip_high": float(1.0 + float(args.clip_ratio_high)),
                },
                "data": data_items,
            },
            "adam_params": {"learning_rate": float(args.learning_rate)},
        }

        train_fut = _post(f"{base_url}/api/v1/train_step", headers, train_step_req, timeout_s=120.0)
        train_request_id = train_fut.get("request_id")
        if not train_request_id:
            raise RuntimeError(f"train_step missing request_id: {train_fut}")

        train_out = _wait_future(
            base_url=base_url, headers=headers, request_id=train_request_id, timeout_s=float(args.timeout_s)
        )

        metrics = train_out.get("metrics") or {}

        # Extract metrics
        diff_mean = _get_metric(
            metrics,
            "training/rollout_probs_diff_mean:mean",
            "training/rollout_probs_diff_mean",
        )
        diff_max = _get_metric(
            metrics,
            "training/rollout_probs_diff_max:mean",
            "training/rollout_probs_diff_max",
        )
        diff_std = _get_metric(
            metrics,
            "training/rollout_probs_diff_std:mean",
            "training/rollout_probs_diff_std",
        )
        diff_valid = _get_metric(
            metrics,
            "training/rollout_probs_diff_valid:mean",
            "training/rollout_probs_diff_valid",
        )
        pearson = _get_metric(
            metrics,
            "training/rollout_actor_probs_pearson_corr:mean",
            "training/rollout_actor_probs_pearson_corr",
        )
        ppo_kl = _get_metric(
            metrics,
            "training/ppo_kl_divergence:mean",
            "ppo_kl_divergence:mean",
            "ppo_kl_divergence",
            "approx_kl:mean",
            "approx_kl",
        )
        ppo_kl_v2 = _get_metric(
            metrics,
            "training/ppo_kl_divergence_v2:mean",
            "ppo_kl_divergence_v2:mean",
            "ppo_kl_divergence_v2",
            "approx_kl_v2:mean",
            "approx_kl_v2",
        )
        grad_norm = _get_metric(
            metrics,
            "training/grad_norm:mean",
            "grad_norm:mean",
            "grad_norm:last",
            "grad_norm",
        )
        loss = _get_metric(
            metrics,
            "training/loss:mean",
            "loss:mean",
            "loss",
        )

        # If backend doesn't expose PPO-KL metric, compute cookbook-style estimators
        # from sampled(old) vs training(new) token logprobs in this batch.
        if ppo_kl is None or ppo_kl_v2 is None:
            kl_v1_calc, kl_v2_calc = _compute_ppo_kl_from_train_output(
                data_items=data_items,
                loss_fn_outputs=train_out.get("loss_fn_outputs"),
            )
            if ppo_kl is None and kl_v1_calc is not None:
                ppo_kl = kl_v1_calc
            if ppo_kl_v2 is None and kl_v2_calc is not None:
                ppo_kl_v2 = kl_v2_calc

        if diff_mean is None:
            raise RuntimeError(f"missing training/rollout_probs_diff_mean:mean in metrics")

        step_elapsed = time.time() - step_start

        row = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "step": int(step),
            "router_replay_mode": router_replay_mode,
            "rollout_probs_diff_mean": float(diff_mean),
            "rollout_probs_diff_max": float(diff_max) if diff_max is not None else "",
            "rollout_probs_diff_std": float(diff_std) if diff_std is not None else "",
            "rollout_probs_diff_valid": int(diff_valid) if diff_valid is not None else "",
            # Backward-compat aliases
            "logprobs_diff_mean": float(diff_mean),
            "logprobs_diff_max": float(diff_max) if diff_max is not None else "",
            "logprobs_diff_std": float(diff_std) if diff_std is not None else "",
            "logprobs_diff_valid": int(diff_valid) if diff_valid is not None else "",
            "actor_rollout_pearson": float(pearson) if pearson is not None else "",
            "ppo_kl_divergence": float(ppo_kl) if ppo_kl is not None else "",
            "gradient_norm": float(grad_norm) if grad_norm is not None else "",
            "loss": float(loss) if loss is not None else "",
            "num_samples": num_sequences,
            "num_problems": len(batch_problems),
            "avg_response_len": f"{avg_response_len:.1f}",
            "avg_prompt_len": f"{avg_prompt_len:.1f}",
            "accuracy": f"{accuracy:.3f}" if accuracy is not None else "",
        }
        _append_csv_row(out_csv, row, columns)

        print(
            f"[Step {step}] "
            f"diff_mean={float(diff_mean):.6f} "
            f"diff_max={float(diff_max) if diff_max is not None else 0:.6f} "
            f"ppo_kl={(f'{float(ppo_kl):.6f}' if ppo_kl is not None else 'N/A')} "
            f"ppo_kl_v2={(f'{float(ppo_kl_v2):.6f}' if ppo_kl_v2 is not None else 'N/A')} "
            f"grad_norm={(f'{float(grad_norm):.4f}' if grad_norm is not None else 'N/A')} "
            f"samples={num_sequences} "
            f"acc={(f'{accuracy:.3f}' if accuracy is not None else 'N/A')} "
            f"({step_elapsed:.1f}s)",
            flush=True,
        )

    print(flush=True)
    print(f"=== Experiment Complete ===", flush=True)
    print(f"CSV: {out_csv}", flush=True)
    print(f"Meta: {meta_path}", flush=True)

    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="R3 router replay validation (v3) with DAPO-Math-17k",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--base-url", default=None, help="Server base URL")
    p.add_argument("--api-key", default=None, help="API key")
    p.add_argument("--base-model", default=None, help=f"Base model (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--prompt-source",
        choices=("fixed", "dataset"),
        default="fixed",
        help="Prompt source: fixed token ids (v2-style) or DAPO dataset text (requires local tokenizer)",
    )
    p.add_argument(
        "--prompt-tokens",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated token ids used when --prompt-source=fixed",
    )
    p.set_defaults(grade_with_detokenize=True)
    p.add_argument(
        "--grade-with-detokenize",
        dest="grade_with_detokenize",
        action="store_true",
        help="Enable answer grading via local tokenizer decode (default: enabled)",
    )
    p.add_argument(
        "--no-grade-with-detokenize",
        dest="grade_with_detokenize",
        action="store_false",
        help="Disable answer grading",
    )

    # DAPO-style defaults adapted from Moonlight-16B-A3B megatron config.
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of training steps")
    p.add_argument(
        "--problems-per-batch",
        "--train-prompt-bsz",
        dest="problems_per_batch",
        type=int,
        default=DEFAULT_PROBLEMS_PER_BATCH,
        help="Number of prompts per step",
    )
    p.add_argument(
        "--num-samples",
        "--n-resp-per-prompt",
        dest="num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Responses per prompt",
    )
    p.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH, help="Prompt token cap with left truncation")
    p.add_argument("--max-response-length", type=int, default=DEFAULT_MAX_RESPONSE_LENGTH, help="Max generation tokens per response")
    p.add_argument("--max-tokens", type=int, default=None, help="Deprecated alias for --max-response-length")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p")
    p.add_argument("--top-k", type=int, default=-1, help="Sampling top-k (-1 disables)")
    p.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    p.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--epsilon", type=float, default=0.2, help="PPO epsilon")
    p.add_argument("--clip-ratio-low", type=float, default=0.2, help="PPO low clip ratio delta (e.g. 0.2 => lower bound 0.8)")
    p.add_argument("--clip-ratio-high", type=float, default=0.28, help="PPO high clip ratio delta (e.g. 0.28 => upper bound 1.28)")
    p.add_argument(
        "--max-request-training-load",
        type=int,
        default=DEFAULT_MAX_REQUEST_TRAINING_LOAD,
        help="Max in-flight /asample requests (submit+wait lifecycle concurrency)",
    )
    p.add_argument("--timeout-s", type=float, default=3600.0, help="Request timeout (longer for math problems)")
    p.add_argument("--run-dir", default=None, help="Output directory")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--dataset-size", type=int, default=None, help="Limit dataset size for testing (default: None = use all 1.79M samples)")
    args = p.parse_args()
    if args.max_tokens is not None:
        args.max_response_length = int(args.max_tokens)
    return args


def main() -> int:
    args = _parse_args()
    return run_ppo_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
