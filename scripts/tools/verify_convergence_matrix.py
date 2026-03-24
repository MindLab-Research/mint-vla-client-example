#!/usr/bin/env python3
"""Correctness Verification Matrix for Issue #146.

Tests (loss_fn) × (model) combinations to verify:
1. Training behavior is sound (no NaN/Inf, metrics normal)
2. Convergence speed matches reference baseline
3. Establish baselines for future regression testing

Aligned with tinker-cookbook (using real datasets):
- Cross-Entropy (SFT): HuggingFaceH4/no_robots dataset (real conversational data)
- RL (PPO/Importance Sampling): Jiayi-Pan/Countdown-Tasks-3to4 dataset (arithmetic reasoning)
- DPO: tinker-cookbook/scripts/dpo_pairs_200.jsonl (preference pairs)

Usage:
    python scripts/tools/verify_convergence_matrix.py \
        --models "Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen3-30B-A3B-Instruct-2507" \
        --loss-fns cross_entropy dpo ppo importance_sampling \
        --seeds 3 \
        --steps 15 \
        --output-dir results/correctness_matrix
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import requests
import tinker
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizerFast

DEFAULT_MODELS = ["Qwen/Qwen3-4B-Instruct-2507"]
DEFAULT_LOSS_FNS = ["cross_entropy", "ppo", "importance_sampling"]
DEFAULT_SEEDS = 1
DEFAULT_SEED_START = 42
DEFAULT_STEPS = 1
DEFAULT_OUTPUT_DIR = "results/correctness_matrix"

LORA_RANK = 32

SFT_LEARNING_RATE = 1e-4
SFT_MAX_TOKENS = 2048
SFT_BATCH_SIZE = 16

RL_LEARNING_RATE = 4e-5
RL_MAX_TOKENS = 256
RL_BATCH_SIZE = 16

DPO_LEARNING_RATE = 1e-5
DPO_BETA = 0.1
DPO_MAX_TOKENS = 2048
DPO_BATCH_SIZE = 8

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8

MIN_LOSS_REDUCTION = 0.3
MAX_LOSS_SPIKE = 1.1


def _ts() -> str:
    """ISO timestamp."""
    return datetime.now().isoformat(timespec="milliseconds")


def _load_env() -> None:
    """Load environment variables from .env files."""
    load_dotenv()
    repo_root_env = Path(__file__).parent.parent.parent / ".env"
    if repo_root_env.exists():
        load_dotenv(repo_root_env, override=False)


def _base_url(args: argparse.Namespace) -> str:
    """Get base URL from args or environment."""
    base_url = args.base_url or os.environ.get("TINKER_BASE_URL") or os.environ.get("MINT_BASE_URL")
    if not base_url:
        raise ValueError("Base URL not provided. Set --base-url or TINKER_BASE_URL/MINT_BASE_URL")
    return base_url.rstrip("/")


def _headers(args: argparse.Namespace) -> dict[str, str]:
    """Get API headers."""
    api_key = args.api_key or os.environ.get("TINKER_API_KEY") or os.environ.get("MINT_API_KEY")
    if api_key:
        return {"X-API-Key": api_key}
    return {}


def _sdk_api_key(headers: dict[str, str]) -> str:
    api_key = headers.get("X-API-Key") or "dummy"
    return api_key if api_key.startswith("tml-") else f"tml-{api_key}"


def _get(url: str, headers: dict[str, str], timeout_s: float = 10.0) -> dict[str, Any]:
    """HTTP GET request."""
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    """HTTP POST request."""
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _poll_future(
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    timeout_s: float = 300.0,
    heartbeat_s: float = 5.0,
) -> dict[str, Any]:
    """Poll for async operation result."""
    poll_url = f"{base_url}/api/v1/retrieve_future"
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed >= timeout_s:
            raise TimeoutError(f"Operation timed out after {elapsed:.0f}s")

        resp = requests.post(
            poll_url,
            json={"request_id": request_id},
            headers=headers,
            timeout=120.0,
        )

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 408:
            if elapsed - int(elapsed / heartbeat_s) * heartbeat_s < 0.5:
                print(
                    f"[{_ts()}] Waiting for request_id={request_id[:8]}... elapsed={elapsed:.0f}s",
                    flush=True,
                )
            time.sleep(0.5)
            continue

        resp.raise_for_status()


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _tensor(data: list[Any], dtype: str) -> dict[str, Any]:
    return {"data": data, "shape": [len(data)], "dtype": dtype}


def _model_input(tokens: list[int]) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "type": "encoded_text",
                "tokens": tokens,
            }
        ]
    }


def _load_tokenizer(model: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except ValueError as e:
        if "Tokenizer class TokenizersBackend does not exist" not in str(e):
            raise

    snapshot_dir = Path(
        snapshot_download(
            model,
            allow_patterns=["tokenizer.json", "tokenizer_config.json"],
            local_files_only=False,
        )
    )
    tokenizer_json = snapshot_dir / "tokenizer.json"
    tokenizer_config = snapshot_dir / "tokenizer_config.json"
    if not tokenizer_json.exists() or not tokenizer_config.exists():
        raise ValueError(
            f"{model}: expected tokenizer.json and tokenizer_config.json for TokenizersBackend loader"
        )

    config = json.loads(tokenizer_config.read_text())
    if config.get("backend") != "tokenizers" or config.get("tokenizer_class") != "TokenizersBackend":
        raise ValueError(f"{model}: unsupported tokenizer config for TokenizersBackend loader")

    model_max_length = config.get("model_max_length")
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        eos_token=config.get("eos_token"),
        pad_token=config.get("pad_token") or config.get("eos_token"),
        model_max_length=int(model_max_length) if model_max_length is not None else None,
        padding_side=str(config.get("padding_side") or "right"),
        clean_up_tokenization_spaces=bool(config.get("clean_up_tokenization_spaces", False)),
        additional_special_tokens=list(config.get("extra_special_tokens") or []),
    )


def _hhh_parse_conversation(text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    parts = re.split(r"(Human:|Assistant:)", text)
    if parts and not parts[0].strip():
        parts = parts[1:]
    for i in range(0, len(parts), 2):
        if i + 1 >= len(parts):
            continue
        delimiter = parts[i].strip()
        content = parts[i + 1].strip()
        if delimiter == "Human:":
            messages.append({"role": "user", "content": content})
        elif delimiter == "Assistant:":
            messages.append({"role": "assistant", "content": content})
    return messages


def _hhh_example_to_dpo_row(example: dict[str, Any]) -> dict[str, Any] | None:
    chosen = _hhh_parse_conversation(example["chosen"])
    rejected = _hhh_parse_conversation(example["rejected"])
    if len(chosen) != len(rejected):
        return None
    match = [a == b for a, b in zip(chosen, rejected, strict=True)]
    if match != [True] * (len(match) - 1) + [False]:
        return None
    return {
        "prompt_conversation": chosen[:-1],
        "completion_A": [chosen[-1]],
        "completion_B": [rejected[-1]],
        "label": "A",
    }


def make_cross_entropy_data(
    tokenizer: Any,
    dataset: Any,
    seed: int,
    batch_idx: int,
    batch_size: int = SFT_BATCH_SIZE,
) -> list[dict]:
    """Generate cross_entropy training data from HuggingFaceH4/no_robots."""
    del seed

    data: list[dict[str, Any]] = []
    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, len(dataset))

    if batch_start >= len(dataset):
        batch_start = batch_start % len(dataset)
        batch_end = min(batch_start + batch_size, len(dataset))

    batch_rows = dataset.select(range(batch_start, batch_end))

    for row in batch_rows:
        messages = row["messages"]

        conversation_tokens: list[int] = []
        weights: list[float] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                tokens = tokenizer.encode(f"User: {content}\n", add_special_tokens=False)
                conversation_tokens.extend(tokens)
                weights.extend([0.0] * len(tokens))
            elif role == "assistant":
                tokens = tokenizer.encode(f"Assistant: {content}\n", add_special_tokens=False)
                conversation_tokens.extend(tokens)
                weights.extend([1.0] * len(tokens))

        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        conversation_tokens = [bos_id] + conversation_tokens
        weights = [0.0] + weights

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        if len(conversation_tokens) < SFT_MAX_TOKENS:
            pad = SFT_MAX_TOKENS - len(conversation_tokens)
            conversation_tokens = conversation_tokens + [pad_id] * pad
            weights = weights + [0.0] * pad
        else:
            conversation_tokens = conversation_tokens[:SFT_MAX_TOKENS]
            weights = weights[:SFT_MAX_TOKENS]

        input_tokens = conversation_tokens[:-1]
        target_tokens = conversation_tokens[1:]
        target_weights = weights[1:]

        data.append(
            {
                "model_input": _model_input(input_tokens),
                "loss_fn_inputs": {
                    "target_tokens": _tensor(target_tokens, "int64"),
                    "weights": _tensor(target_weights, "float32"),
                },
            }
        )

    return data


def compute_countdown_reward(generated_text: str, numbers: list[int], target: int) -> float:
    """Compute reward for Countdown task output."""
    import re

    try:
        match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if match is None:
            return 0.0

        equation = match.group(1).strip()
        if "=" in equation:
            equation = equation.split("=")[0]

        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
        if sorted(used_numbers) != sorted(numbers):
            return 0.0

        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        result = eval(equation, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def _build_countdown_prompt(numbers: list[int], target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an equation that equals {target}.\n"
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.\n"
        "Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags,\n"
        "for example <answer> (1 + 2) / 3 * 4 = 4 </answer>."
    )


def make_rl_data(
    tokenizer: Any,
    model_id: str,
    base_url: str,
    headers: dict[str, str],
    rl_dataset: Any,
    seed: int,
    batch_idx: int,
    loss_fn: str,
    batch_size: int = RL_BATCH_SIZE,
) -> list[dict]:
    """Generate RL data using real sampling from server."""
    random.seed(seed)

    data: list[dict[str, Any]] = []
    rewards: list[float] = []

    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, len(rl_dataset))

    if batch_start >= len(rl_dataset):
        batch_start = batch_start % len(rl_dataset)
        batch_end = min(batch_start + batch_size, len(rl_dataset))

    batch_rows = rl_dataset.select(range(batch_start, batch_end))

    for row in batch_rows:
        numbers = row["nums"]
        target = row.get("target") or row.get("response")

        if not numbers or target is None:
            continue

        prompt_text = _build_countdown_prompt(numbers, target)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

        sample_payload = {
            "model_id": model_id,
            "prompt": _model_input(prompt_tokens),
            "sampling_params": {
                "max_tokens": RL_MAX_TOKENS,
                "temperature": 0.7,
            },
            "num_samples": 1,
        }

        try:
            resp = _post(f"{base_url}/api/v1/asample", headers, sample_payload, timeout_s=60.0)
            result = _poll_future(base_url, headers, resp["request_id"], timeout_s=120.0)

            if "error" in result:
                continue

            sequences = result.get("sequences", [])
            if not sequences:
                continue

            seq = sequences[0]
            generated_tokens = seq.get("tokens", [])
            old_logprobs = seq.get("logprobs", [])
            if not generated_tokens:
                continue

            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            reward = compute_countdown_reward(generated_text, numbers, target)
            rewards.append(reward)

            input_tokens = prompt_tokens + generated_tokens[:-1]
            target_tokens = prompt_tokens[1:] + generated_tokens

            loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)

            baseline = 0.5
            advantage = reward - baseline
            advantages = [0.0] * (len(prompt_tokens) - 1) + [advantage] * len(generated_tokens)

            logprobs = [0.0] * (len(prompt_tokens) - 1) + old_logprobs

            datum: dict[str, Any] = {
                "model_input": _model_input(input_tokens),
                "loss_fn_inputs": {
                    "target_tokens": _tensor(target_tokens, "int64"),
                    "loss_mask": _tensor(loss_mask, "float32"),
                    "logprobs": _tensor(logprobs, "float32"),
                    "advantages": _tensor(advantages, "float32"),
                },
            }
            if loss_fn == "ppo":
                datum["loss_fn_inputs"]["response_mask"] = _tensor(loss_mask, "float32")

            data.append(datum)

        except Exception as e:
            print(f"[{_ts()}] Warning: Sampling failed for RL data generation: {e}", flush=True)
            continue

    while len(data) < batch_size:
        idx = random.randint(0, len(rl_dataset) - 1)
        row = rl_dataset[idx]

        numbers = row["nums"]
        target = row.get("target") or row.get("response")
        if not numbers or target is None:
            continue

        prompt_text = _build_countdown_prompt(numbers, target)
        completion_text = (
            f" <think>Let me try to find an equation.</think> <answer>{numbers[0]} + {numbers[1]} = {target}</answer>"
        )

        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

        input_tokens = prompt_tokens + completion_tokens[:-1]
        target_tokens = prompt_tokens[1:] + completion_tokens

        loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
        advantages = [0.0] * (len(prompt_tokens) - 1) + [0.1] * len(completion_tokens)
        logprobs = [0.0] * len(target_tokens)

        datum = {
            "model_input": _model_input(input_tokens),
            "loss_fn_inputs": {
                "target_tokens": _tensor(target_tokens, "int64"),
                "loss_mask": _tensor(loss_mask, "float32"),
                "logprobs": _tensor(logprobs, "float32"),
                "advantages": _tensor(advantages, "float32"),
            },
        }
        if loss_fn == "ppo":
            datum["loss_fn_inputs"]["response_mask"] = _tensor(loss_mask, "float32")

        data.append(datum)

    return data, rewards


def _make_dpo_datum(
    input_tokens: list[int],
    target_tokens: list[int],
    weights: list[float],
) -> tinker.types.Datum:
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": tinker.types.TensorData(
                data=list(target_tokens),
                shape=[len(target_tokens)],
                dtype="int64",
            ),
            "weights": tinker.types.TensorData(
                data=list(weights),
                shape=[len(weights)],
                dtype="float32",
            ),
        },
    )


def make_dpo_data(
    tokenizer: Any,
    dpo_dataset: Any,
    seed: int,
    batch_idx: int,
    batch_size: int = DPO_BATCH_SIZE,
) -> tuple[list[tinker.types.Datum], list[tinker.types.ModelInput]]:
    """Generate DPO training data and full sequences for the custom-loss path."""
    random.seed(seed)

    data: list[tinker.types.Datum] = []
    full_sequences: list[tinker.types.ModelInput] = []
    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, len(dpo_dataset))

    if batch_start >= len(dpo_dataset):
        batch_start = batch_start % len(dpo_dataset)
        batch_end = min(batch_start + batch_size, len(dpo_dataset))

    batch_rows = dpo_dataset.select(range(batch_start, batch_end))

    def tokenize_conversation(convo: list[dict[str, str]]) -> tuple[list[int], list[float]]:
        tokens: list[int] = []
        weights: list[float] = []

        for msg in convo:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                msg_tokens = tokenizer.encode(f"User: {content}\n", add_special_tokens=False)
                tokens.extend(msg_tokens)
                weights.extend([0.0] * len(msg_tokens))
            elif role == "assistant":
                msg_tokens = tokenizer.encode(f"Assistant: {content}\n", add_special_tokens=False)
                tokens.extend(msg_tokens)
                weights.extend([1.0] * len(msg_tokens))

        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        tokens = [bos_id] + tokens
        weights = [0.0] + weights

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        if len(tokens) < DPO_MAX_TOKENS:
            pad = DPO_MAX_TOKENS - len(tokens)
            tokens = tokens + [pad_id] * pad
            weights = weights + [0.0] * pad
            return tokens, weights

        tokens = tokens[:DPO_MAX_TOKENS]
        weights = weights[:DPO_MAX_TOKENS]
        return tokens, weights

    for row in batch_rows:
        prompt_conversation = row.get("prompt_conversation", [])
        completion_A = row.get("completion_A", [])
        completion_B = row.get("completion_B", [])
        label = row.get("label", "A")

        if not prompt_conversation or not completion_A or not completion_B:
            continue

        if label == "A":
            chosen_completion = completion_A
            rejected_completion = completion_B
        else:
            chosen_completion = completion_B
            rejected_completion = completion_A

        chosen_convo = prompt_conversation + chosen_completion
        rejected_convo = prompt_conversation + rejected_completion

        chosen_tokens, chosen_weights = tokenize_conversation(chosen_convo)
        rejected_tokens, rejected_weights = tokenize_conversation(rejected_convo)

        chosen_input = chosen_tokens[:-1]
        chosen_target = chosen_tokens[1:]
        chosen_target_weights = chosen_weights[1:]

        rejected_input = rejected_tokens[:-1]
        rejected_target = rejected_tokens[1:]
        rejected_target_weights = rejected_weights[1:]

        data.append(_make_dpo_datum(chosen_input, chosen_target, chosen_target_weights))
        data.append(_make_dpo_datum(rejected_input, rejected_target, rejected_target_weights))
        full_sequences.append(tinker.types.ModelInput.from_ints(chosen_tokens))
        full_sequences.append(tinker.types.ModelInput.from_ints(rejected_tokens))

    return data, full_sequences


def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )
    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    return loss, {
        "dpo_loss": float(loss.item()),
        "accuracy": float((chosen_log_ratio > rejected_log_ratio).float().mean().item()),
        "margin": float((chosen_rewards - rejected_rewards).mean().item()),
        "chosen_reward": float(chosen_rewards.mean().item()),
        "rejected_reward": float(rejected_rewards.mean().item()),
    }


def _save_checkpoint(
    output_file: Path,
    status: str,
    metadata: dict,
    metrics: list,
    error: str | None = None,
) -> None:
    data = {
        "status": status,
        "metadata": metadata,
        "metrics": metrics,
    }
    if error:
        data["error"] = error

    def _json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            return _json_safe(value.model_dump())
        if hasattr(value, "data") and hasattr(value, "shape") and hasattr(value, "dtype"):
            return {
                "data": _json_safe(value.data),
                "shape": _json_safe(value.shape),
                "dtype": str(value.dtype),
            }
        return repr(value)

    with open(output_file, "w") as f:
        json.dump(_json_safe(data), f, indent=2)


def run_training(
    base_url: str,
    headers: dict[str, str],
    tokenizer: Any,
    model: str,
    loss_fn: str,
    seed: int,
    steps: int,
    output_file: Path,
    sft_dataset: Any = None,
    rl_dataset: Any = None,
    dpo_dataset: Any = None,
) -> dict[str, Any]:
    """Run training for one (model, loss_fn, seed) combination.

    Args:
        sft_dataset: HuggingFace dataset for SFT (no_robots), required for cross_entropy
        rl_dataset: HuggingFace dataset for RL (Countdown), required for ppo/importance_sampling
        dpo_dataset: HuggingFace dataset for DPO (preference pairs), required for dpo

    Returns:
        dict with 'status', 'metrics', 'metadata'
    """
    print("\n" + "=" * 70)
    print(f"[{_ts()}] Starting: model={model}, loss_fn={loss_fn}, seed={seed}")
    print("=" * 70)

    start_time = time.time()
    metrics: list[dict[str, Any]] = []
    metadata = {
        "model": model,
        "loss_fn": loss_fn,
        "seed": seed,
        "steps": steps,
        "lora_rank": LORA_RANK,
        "git_sha": get_git_sha(),
        "timestamp": _ts(),
    }

    try:
        if loss_fn == "cross_entropy":
            learning_rate = SFT_LEARNING_RATE
        elif loss_fn == "dpo":
            learning_rate = DPO_LEARNING_RATE
        else:
            learning_rate = RL_LEARNING_RATE

        training_client = None
        reference_client = None

        if loss_fn == "dpo":
            print(f"[{_ts()}] Creating DPO training client (lr={learning_rate})...", flush=True)
            service_client = tinker.ServiceClient(
                base_url=base_url,
                api_key=_sdk_api_key(headers),
            )
            training_client = service_client.create_lora_training_client(
                base_model=model,
                rank=LORA_RANK,
            )
            reference_client = training_client.save_weights_and_get_sampling_client("reference")
            model_id = str(training_client.model_id)
        else:
            session_id = f"verify_matrix_{seed}_{int(time.time())}"
            create_payload = {
                "session_id": session_id,
                "model_seq_id": 1,
                "base_model": model,
                "lora_config": {"rank": LORA_RANK},
                "learning_rate": learning_rate,
            }

            print(f"[{_ts()}] Creating session (lr={learning_rate})...", flush=True)
            resp = _post(f"{base_url}/api/v1/create_model", headers, create_payload, timeout_s=300.0)
            result = _poll_future(base_url, headers, resp["request_id"], timeout_s=300.0)

            if "error" in result:
                raise RuntimeError(f"Session creation failed: {result['error']}")

            model_id = result["model_id"]

        print(f"[{_ts()}] Session created: model_id={model_id}", flush=True)

        if loss_fn in ("ppo", "importance_sampling"):
            print(f"[{_ts()}] Saving initial weights for sampling...", flush=True)
            save_payload = {"model_id": model_id, "name": "initial"}
            resp = _post(f"{base_url}/api/v1/save_weights", headers, save_payload, timeout_s=120.0)
            _poll_future(base_url, headers, resp["request_id"], timeout_s=300.0)

        for step in range(steps):
            step_start = time.time()
            batch_rewards: list[float] = []

            if loss_fn == "cross_entropy":
                if sft_dataset is None:
                    raise ValueError("sft_dataset is required for cross_entropy loss")
                data = make_cross_entropy_data(tokenizer, sft_dataset, seed, step)
            elif loss_fn == "dpo":
                if dpo_dataset is None:
                    raise ValueError("dpo_dataset is required for dpo loss")
                if training_client is None or reference_client is None:
                    raise RuntimeError("DPO path requires initialized training/reference clients")
                data, full_sequences = make_dpo_data(tokenizer, dpo_dataset, seed, step)
            else:
                if rl_dataset is None:
                    raise ValueError("rl_dataset is required for RL loss functions")
                data, batch_rewards = make_rl_data(
                    tokenizer,
                    model_id,
                    base_url,
                    headers,
                    rl_dataset,
                    seed,
                    step,
                    loss_fn,
                )

            if loss_fn == "dpo":
                all_ref_logprob_seqs = [
                    torch.tensor(reference_client.compute_logprobs(seq).result()[1:])
                    for seq in full_sequences
                ]
                chosen_data = [datum for idx, datum in enumerate(data) if idx % 2 == 0]
                rejected_data = [datum for idx, datum in enumerate(data) if idx % 2 == 1]
                chosen_ref_logprob_seqs = [all_ref_logprob_seqs[idx] for idx in range(0, len(data), 2)]
                rejected_ref_logprob_seqs = [all_ref_logprob_seqs[idx] for idx in range(1, len(data), 2)]

                def dpo_loss_fn(
                    batch: list[tinker.types.Datum], logprobs_list: list[torch.Tensor]
                ) -> tuple[torch.Tensor, dict[str, float]]:
                    chosen_logprob_seqs = [logprobs_list[idx] for idx in range(0, len(batch), 2)]
                    rejected_logprob_seqs = [logprobs_list[idx] for idx in range(1, len(batch), 2)]
                    chosen_logprobs: list[torch.Tensor] = []
                    rejected_logprobs: list[torch.Tensor] = []
                    chosen_ref_logprobs: list[torch.Tensor] = []
                    rejected_ref_logprobs: list[torch.Tensor] = []

                    for idx in range(len(chosen_data)):
                        chosen_weights = torch.tensor(chosen_data[idx].loss_fn_inputs["weights"].data)
                        chosen_logprobs.append(
                            torch.dot(chosen_logprob_seqs[idx].float(), chosen_weights.float())
                        )
                        chosen_ref_logprobs.append(
                            torch.dot(chosen_ref_logprob_seqs[idx].float(), chosen_weights.float())
                        )

                        rejected_weights = torch.tensor(rejected_data[idx].loss_fn_inputs["weights"].data)
                        rejected_logprobs.append(
                            torch.dot(rejected_logprob_seqs[idx].float(), rejected_weights.float())
                        )
                        rejected_ref_logprobs.append(
                            torch.dot(rejected_ref_logprob_seqs[idx].float(), rejected_weights.float())
                        )

                    return compute_dpo_loss(
                        chosen_logprobs=chosen_logprobs,
                        rejected_logprobs=rejected_logprobs,
                        chosen_ref_logprobs=chosen_ref_logprobs,
                        rejected_ref_logprobs=rejected_ref_logprobs,
                        dpo_beta=DPO_BETA,
                    )

                print(f"[{_ts()}] Step {step + 1}/{steps}: forward_backward_custom...", flush=True)
                backward_result = training_client.forward_backward_custom(data, dpo_loss_fn).result()
                fb_result = {
                    "metrics": dict(backward_result.metrics),
                    "loss_fn_outputs": list(backward_result.loss_fn_outputs),
                }
                print(f"[{_ts()}] Step {step + 1}/{steps}: optim_step...", flush=True)
                optim_output = training_client.optim_step(
                    tinker.AdamParams(
                        learning_rate=learning_rate,
                        beta1=ADAM_BETA1,
                        beta2=ADAM_BETA2,
                        eps=ADAM_EPS,
                    )
                ).result()
                optim_result = {
                    "metrics": dict(optim_output.metrics),
                }
            else:
                fb_payload: dict[str, Any] = {
                    "model_id": model_id,
                    "forward_backward_input": {
                        "data": data,
                        "loss_fn": loss_fn,
                    },
                }
                if loss_fn in ("ppo", "importance_sampling"):
                    fb_payload["forward_backward_input"]["loss_fn_config"] = {"clip_ratio": 0.2}

                print(f"[{_ts()}] Step {step + 1}/{steps}: forward_backward...", flush=True)
                resp = _post(f"{base_url}/api/v1/forward_backward", headers, fb_payload, timeout_s=120.0)
                fb_result = _poll_future(base_url, headers, resp["request_id"], timeout_s=300.0)

                if "error" in fb_result:
                    raise RuntimeError(f"forward_backward failed: {fb_result['error']}")

                optim_payload = {
                    "model_id": model_id,
                    "adam_params": {
                        "learning_rate": learning_rate,
                        "beta1": ADAM_BETA1,
                        "beta2": ADAM_BETA2,
                        "eps": ADAM_EPS,
                    },
                }

                print(f"[{_ts()}] Step {step + 1}/{steps}: optim_step...", flush=True)
                resp = _post(f"{base_url}/api/v1/optim_step", headers, optim_payload, timeout_s=60.0)
                optim_result = _poll_future(base_url, headers, resp["request_id"], timeout_s=60.0)

                if "error" in optim_result:
                    raise RuntimeError(f"optim_step failed: {optim_result['error']}")

            step_metrics: dict[str, Any] = {
                "step": step + 1,
                "timestamp": _ts(),
                "step_time_s": time.time() - step_start,
                "forward_backward_metrics": fb_result.get("metrics", {}),
                "optim_metrics": optim_result.get("metrics", {}),
                "loss_fn_outputs": fb_result.get("loss_fn_outputs", []),
            }

            if loss_fn == "dpo":
                loss = fb_result.get("metrics", {}).get("dpo_loss", None)
            else:
                loss = fb_result.get("metrics", {}).get("loss:mean", None)
            grad_norm = optim_result.get("metrics", {}).get("grad_norm:last", None)

            reward_mean = None
            if loss_fn in ("ppo", "importance_sampling") and batch_rewards:
                reward_mean = sum(batch_rewards) / len(batch_rewards)
                reward_max = max(batch_rewards)
                reward_min = min(batch_rewards)
                step_metrics["reward_mean"] = reward_mean
                step_metrics["reward_max"] = reward_max
                step_metrics["reward_min"] = reward_min

            loss_str = f"{loss:.6f}" if loss is not None else "N/A"
            grad_norm_str = f"{grad_norm:.6f}" if grad_norm is not None else "N/A"

            if loss_fn in ("ppo", "importance_sampling") and reward_mean is not None:
                reward_str = f"{reward_mean:.4f}"
                print(
                    f"[{_ts()}] Step {step + 1}/{steps}: "
                    f"loss={loss_str}, reward={reward_str}, grad_norm={grad_norm_str}, "
                    f"time={step_metrics['step_time_s']:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[{_ts()}] Step {step + 1}/{steps}: "
                    f"loss={loss_str}, grad_norm={grad_norm_str}, "
                    f"time={step_metrics['step_time_s']:.2f}s",
                    flush=True,
                )

            metrics.append(step_metrics)

            metadata["learning_rate"] = learning_rate
            if loss_fn == "cross_entropy":
                metadata["batch_size"] = SFT_BATCH_SIZE
                metadata["max_tokens"] = SFT_MAX_TOKENS
            elif loss_fn == "dpo":
                metadata["batch_size"] = DPO_BATCH_SIZE
                metadata["max_tokens"] = DPO_MAX_TOKENS
            else:
                metadata["batch_size"] = RL_BATCH_SIZE
                metadata["max_tokens"] = RL_MAX_TOKENS

            _save_checkpoint(output_file, "in_progress", metadata, metrics)

            if loss_fn in ("ppo", "importance_sampling") and step < steps - 1:
                save_payload = {"model_id": model_id, "name": f"step_{step + 1}"}
                resp = _post(f"{base_url}/api/v1/save_weights", headers, save_payload, timeout_s=120.0)
                _poll_future(base_url, headers, resp["request_id"], timeout_s=300.0)

        total_time = time.time() - start_time

        metadata["learning_rate"] = learning_rate
        if loss_fn == "cross_entropy":
            metadata["batch_size"] = SFT_BATCH_SIZE
            metadata["max_tokens"] = SFT_MAX_TOKENS
        elif loss_fn == "dpo":
            metadata["batch_size"] = DPO_BATCH_SIZE
            metadata["max_tokens"] = DPO_MAX_TOKENS
        else:
            metadata["batch_size"] = RL_BATCH_SIZE
            metadata["max_tokens"] = RL_MAX_TOKENS
        metadata["total_time_s"] = total_time

        _save_checkpoint(output_file, "success", metadata, metrics)

        print(f"\n[{_ts()}] SUCCESS: Saved results to {output_file}")
        print(f"[{_ts()}] Total time: {total_time:.2f}s")

        return {
            "status": "success",
            "metadata": metadata,
            "metrics": metrics,
        }

    except Exception as e:
        _save_checkpoint(output_file, "failed", metadata, metrics, error=str(e))
        _save_checkpoint(output_file, "failed", metadata, metrics, error=str(e))

        print(f"\n[{_ts()}] FAILED: {e}")
        print(f"[{_ts()}] Partial results saved to {output_file}")

        return {
            "status": "failed",
            "error": str(e),
            "metadata": metadata,
            "metrics": metrics,
        }


def analyze_results(output_dir: Path) -> None:
    """Analyze all results and generate report."""
    print("\n" + "=" * 70)
    print(f"Analyzing results in {output_dir}")
    print("=" * 70 + "\n")

    results: list[dict[str, Any]] = []
    for json_file in output_dir.glob("*.json"):
        if json_file.name == "report.json":
            continue
        with open(json_file) as f:
            results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in results:
        key = (r["metadata"]["model"], r["metadata"]["loss_fn"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    report: dict[str, Any] = {
        "timestamp": _ts(),
        "git_sha": get_git_sha(),
        "summary": {},
        "anomalies": [],
        "recommendations": [],
    }

    for (model, loss_fn), runs in groups.items():
        print(f"\n## {model} × {loss_fn}")
        print("-" * 70)

        successful = [r for r in runs if r["status"] == "success"]
        failed = [r for r in runs if r["status"] == "failed"]

        print(f"Runs: {len(successful)} successful, {len(failed)} failed")

        if failed:
            for r in failed:
                error_msg = r.get("error", "Unknown error")
                report["anomalies"].append(f"{model} × {loss_fn}: {error_msg}")
                print(f"  FAILED: {error_msg}")

        if not successful:
            continue

        all_losses: list[list[float]] = []
        for r in successful:
            losses = [
                m["forward_backward_metrics"].get("loss:mean")
                for m in r["metrics"]
                if m["forward_backward_metrics"].get("loss:mean") is not None
            ]
            if losses:
                all_losses.append(losses)

        if not all_losses:
            continue

        import numpy as np

        min_len = min(len(l) for l in all_losses)
        all_losses = [l[:min_len] for l in all_losses]

        mean_losses = np.mean(all_losses, axis=0)
        std_losses = np.std(all_losses, axis=0)

        initial_loss = mean_losses[0]
        final_loss = mean_losses[-1]
        reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

        print(f"  Initial loss: {initial_loss:.4f} ± {std_losses[0]:.4f}")
        print(f"  Final loss: {final_loss:.4f} ± {std_losses[-1]:.4f}")
        print(f"  Reduction: {reduction:.1%}")

        anomalies: list[str] = []

        if np.any(np.isnan(mean_losses)) or np.any(np.isinf(mean_losses)):
            anomalies.append("NaN or Inf detected in loss")

        if reduction < MIN_LOSS_REDUCTION:
            anomalies.append(
                f"Insufficient loss reduction: {reduction:.1%} < {MIN_LOSS_REDUCTION:.0%}"
            )

        for i in range(1, len(mean_losses)):
            if mean_losses[i] > mean_losses[i - 1] * MAX_LOSS_SPIKE:
                anomalies.append(
                    f"Loss spike at step {i + 1}: {mean_losses[i - 1]:.4f} -> {mean_losses[i]:.4f}"
                )

        if anomalies:
            print("  ANOMALIES:")
            for a in anomalies:
                print(f"    - {a}")
                report["anomalies"].append(f"{model} × {loss_fn}: {a}")
        else:
            print("  ✓ All checks passed")

        report["summary"][f"{model} × {loss_fn}"] = {
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "initial_loss": float(initial_loss),
            "final_loss": float(final_loss),
            "reduction": float(reduction),
            "anomalies": anomalies,
        }

    if report["anomalies"]:
        report["recommendations"].append("Investigate anomalies listed above")
    else:
        report["recommendations"].append("All combinations passed verification")

    report_file = output_dir / "report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Report saved to: {report_file}")
    print("=" * 70 + "\n")

    plot_file = plot_training_curves(output_dir, results)
    generate_markdown_report(output_dir, report, plot_file)


def generate_markdown_report(
    output_dir: Path,
    report: dict,
    plot_file: Path | None = None,
) -> None:
    """Generate markdown report."""
    md_file = output_dir / "report.md"

    with open(md_file, "w") as f:
        f.write("# Correctness Verification Matrix Report\n\n")
        f.write(f"**Timestamp:** {report['timestamp']}  \n")
        f.write(f"**Git SHA:** `{report['git_sha']}`  \n\n")

        if plot_file:
            f.write("## Training Curves\n\n")
            f.write(f"![Training Curves]({plot_file.name})\n\n")

        f.write("## Summary\n\n")
        f.write("| Model × Loss Function | Successful | Failed | Initial Loss | Final Loss | Reduction |\n")
        f.write("|---|---|---|---|---|---|\n")

        for key, data in report["summary"].items():
            f.write(
                f"| {key} | {data['successful_runs']} | {data['failed_runs']} | "
                f"{data['initial_loss']:.4f} | {data['final_loss']:.4f} | "
                f"{data['reduction']:.1%} |\n"
            )

        f.write("\n## Anomalies\n\n")
        if report["anomalies"]:
            for a in report["anomalies"]:
                f.write(f"- {a}\n")
        else:
            f.write("No anomalies detected. ✓\n")

        f.write("\n## Recommendations\n\n")
        for r in report["recommendations"]:
            f.write(f"- {r}\n")

        f.write("\n## Configuration\n\n")
        f.write(f"- **LoRA rank:** {LORA_RANK}\n")
        f.write(f"- **SFT Learning rate:** {SFT_LEARNING_RATE}\n")
        f.write(f"- **RL Learning rate:** {RL_LEARNING_RATE}\n")
        f.write(f"- **SFT Max tokens:** {SFT_MAX_TOKENS}\n")
        f.write(f"- **RL Max tokens:** {RL_MAX_TOKENS}\n")
        f.write(f"- **SFT Batch size:** {SFT_BATCH_SIZE}\n")
        f.write(f"- **RL Batch size:** {RL_BATCH_SIZE}\n")

        if report["summary"]:
            first_entry = list(report["summary"].values())[0]
            f.write(f"- **Successful runs:** {first_entry['successful_runs']}\n")
        else:
            f.write("- **All runs failed**\n")

    print(f"Markdown report saved to: {md_file}\n")


def plot_training_curves(output_dir: Path, results: list[dict]) -> Path | None:
    """Generate training curve plots for all experiments.

    Returns path to the generated plot, or None if matplotlib unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return None

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        if r["status"] != "success":
            continue
        key = (r["metadata"]["model"], r["metadata"]["loss_fn"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    if not groups:
        return None

    models = sorted(set(k[0] for k in groups.keys()))
    loss_fns = sorted(set(k[1] for k in groups.keys()))

    n_rows = len(models)
    n_cols = len(loss_fns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Correctness Verification Matrix: Training Curves", fontsize=16, y=0.995)

    for i, model in enumerate(models):
        for j, loss_fn in enumerate(loss_fns):
            ax = axes[i, j]
            key = (model, loss_fn)

            if key not in groups:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{model.split('/')[-1]} × {loss_fn}")
                continue

            runs = groups[key]
            is_rl = loss_fn in ("ppo", "importance_sampling")

            if is_rl:
                all_rewards: list[list[float]] = []
                for r in runs:
                    rewards = [m.get("reward_mean") for m in r["metrics"] if m.get("reward_mean") is not None]
                    if rewards:
                        all_rewards.append(rewards)

                if not all_rewards:
                    ax.text(0.5, 0.5, "No reward data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{model.split('/')[-1]} × {loss_fn}")
                    continue

                min_len = min(len(r) for r in all_rewards)
                all_rewards = [r[:min_len] for r in all_rewards]

                mean_values = np.mean(all_rewards, axis=0)
                std_values = np.std(all_rewards, axis=0)
                steps = np.arange(1, len(mean_values) + 1)

                ax.plot(steps, mean_values, "g-", linewidth=2, label="Mean")
                ax.fill_between(
                    steps,
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.3,
                    color="g",
                    label="±1 std",
                )
                for rewards in all_rewards:
                    ax.plot(steps, rewards, color="gray", alpha=0.3, linewidth=0.5)

                initial = mean_values[0]
                final = mean_values[-1]
                improvement = (final - initial) / abs(initial) if initial != 0 else 0

                ax.text(
                    0.02,
                    0.98,
                    f"Initial: {initial:.4f}\nFinal: {final:.4f}\nImprovement: {improvement:.1%}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
                )
                ax.set_xlabel("Step")
                ax.set_ylabel("Reward")
                ax.set_title(f"{model.split('/')[-1]} × {loss_fn}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="lower right", fontsize=8)
                continue

            all_losses: list[list[float]] = []
            for r in runs:
                losses = [
                    m["forward_backward_metrics"].get("loss:mean")
                    for m in r["metrics"]
                    if m["forward_backward_metrics"].get("loss:mean") is not None
                ]
                if losses:
                    all_losses.append(losses)

            if not all_losses:
                ax.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{model.split('/')[-1]} × {loss_fn}")
                continue

            min_len = min(len(l) for l in all_losses)
            all_losses = [l[:min_len] for l in all_losses]

            mean_values = np.mean(all_losses, axis=0)
            std_values = np.std(all_losses, axis=0)
            steps = np.arange(1, len(mean_values) + 1)

            ax.plot(steps, mean_values, "b-", linewidth=2, label="Mean")
            ax.fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.3,
                color="b",
                label="±1 std",
            )
            for losses in all_losses:
                ax.plot(steps, losses, color="gray", alpha=0.3, linewidth=0.5)

            initial = mean_values[0]
            final = mean_values[-1]
            reduction = (initial - final) / initial if initial > 0 else 0

            ax.text(
                0.02,
                0.98,
                f"Initial: {initial:.4f}\nFinal: {final:.4f}\nReduction: {reduction:.1%}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"{model.split('/')[-1]} × {loss_fn}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_file = output_dir / "training_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training curves plot saved to: {plot_file}")
    return plot_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Correctness verification matrix for Issue #146")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to test")
    parser.add_argument("--loss-fns", nargs="+", default=DEFAULT_LOSS_FNS, help="Loss functions to test")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of random seeds per combination")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="Starting seed value (default: 42)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of training steps per run")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--base-url", default=None, help="Base URL for Tinker server")
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results, don't run training",
    )

    args = parser.parse_args()
    _load_env()

    output_dir = Path(args.output_dir)

    if args.analyze_only:
        analyze_results(output_dir)
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_sha()[:8]
    run_dir = output_dir / f"run_{timestamp}_{git_sha}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")

    base_url = _base_url(args)
    headers = _headers(args)
    print(f"Base URL: {base_url}")

    try:
        health = _get(f"{base_url}/api/v1/healthz", headers)
        print(f"Server health: {health}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {base_url}: {e}")
        return 1

    sft_dataset = None
    if "cross_entropy" in args.loss_fns:
        try:
            import datasets

            print("\nLoading HuggingFaceH4/no_robots dataset for SFT...")
            dataset_dict = datasets.load_dataset("HuggingFaceH4/no_robots")
            sft_dataset = dataset_dict["train"].shuffle(seed=42)
            print(f"Loaded {len(sft_dataset)} SFT training examples")
        except Exception as e:
            print(f"ERROR: Failed to load no_robots dataset: {e}")
            print("Cross-entropy experiments will be skipped.")

    rl_dataset = None
    if "ppo" in args.loss_fns or "importance_sampling" in args.loss_fns:
        try:
            import datasets

            print("\nLoading Jiayi-Pan/Countdown-Tasks-3to4 dataset for RL...")
            dataset_dict = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")
            rl_dataset = dataset_dict["train"].shuffle(seed=42)
            print(f"Loaded {len(rl_dataset)} RL training examples")
        except Exception as e:
            print(f"ERROR: Failed to load Countdown dataset: {e}")
            print("RL experiments will be skipped.")

    dpo_dataset = None
    if "dpo" in args.loss_fns:
        try:
            import datasets

            print("\nLoading DPO preference pairs dataset from Anthropic/hh-rlhf...")
            dataset_dict = datasets.load_dataset("Anthropic/hh-rlhf")
            train_dataset = dataset_dict["train"].shuffle(seed=42)

            dpo_data = []
            for example in train_dataset:
                row = _hhh_example_to_dpo_row(example)
                if row is None:
                    continue
                dpo_data.append(row)
                if len(dpo_data) >= 200:
                    break
            if not dpo_data:
                raise ValueError("No valid DPO comparison rows extracted from Anthropic/hh-rlhf")

            dpo_dataset = datasets.Dataset.from_list(dpo_data)
            print(f"Loaded {len(dpo_dataset)} DPO preference pairs")
        except Exception as e:
            print(f"ERROR: Failed to load DPO dataset: {e}")
            print("DPO experiments will be skipped.")

    total_runs = len(args.models) * len(args.loss_fns) * args.seeds
    current_run = 0

    for model in args.models:
        try:
            print(f"\nLoading tokenizer for {model}...")
            tokenizer = _load_tokenizer(model)

            for loss_fn in args.loss_fns:
                if loss_fn == "cross_entropy" and sft_dataset is None:
                    print("\nSkipping cross_entropy (dataset not loaded)")
                    continue

                if loss_fn in ("ppo", "importance_sampling") and rl_dataset is None:
                    print(f"\nSkipping {loss_fn} (RL dataset not loaded)")
                    continue

                if loss_fn == "dpo" and dpo_dataset is None:
                    print("\nSkipping dpo (DPO dataset not loaded)")
                    continue

                for seed_idx in range(args.seeds):
                    seed = args.seed_start + seed_idx
                    current_run += 1

                    print("\n" + "=" * 70)
                    print(f"Run {current_run}/{total_runs} (seed={seed})")
                    print("=" * 70)

                    model_name = model.replace("/", "_")
                    output_file = run_dir / f"{model_name}_{loss_fn}_seed{seed}.json"

                    run_training(
                        base_url=base_url,
                        headers=headers,
                        tokenizer=tokenizer,
                        model=model,
                        loss_fn=loss_fn,
                        seed=seed,
                        steps=args.steps,
                        output_file=output_file,
                        sft_dataset=sft_dataset,
                        rl_dataset=rl_dataset,
                        dpo_dataset=dpo_dataset,
                    )

        except Exception as e:
            print(f"ERROR: Failed to load tokenizer for {model}: {e}")
            continue

    print("\n" + "=" * 70)
    print("All runs completed. Analyzing results...")
    print("=" * 70)

    analyze_results(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
