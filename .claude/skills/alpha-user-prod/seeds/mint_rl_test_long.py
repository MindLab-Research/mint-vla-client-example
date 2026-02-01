import argparse
import json
import os
import random
import re
import unicodedata
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mint
import math
import pandas as pd
from dotenv import load_dotenv
from mint import types

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MinT RL arithmetic demo (long prompts)")
    parser.add_argument(
        "--model",
        "--base-model",
        dest="base_model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name (as listed by server capabilities)",
    )
    parser.add_argument(
        "--system-prompt-path",
        dest="system_prompt_path",
        default="./prompts/claude_system.txt",
        help="Path to the system prompt to prepend to each input",
    )
    parser.add_argument(
        "--print-token-counts",
        dest="print_token_counts",
        action="store_true",
        default=True,
        help="Print token counts for inputs and outputs (default: enabled)",
    )
    parser.add_argument(
        "--no-print-token-counts",
        dest="print_token_counts",
        action="store_false",
        help="Disable token-count printing",
    )
    parser.add_argument(
        "--save-ckpt",
        dest="save_ckpt",
        action="store_true",
        default=True,
        help="Save checkpoints at the end (default: enabled)",
    )
    parser.add_argument(
        "--no-save-ckpt",
        dest="save_ckpt",
        action="store_false",
        help="Disable end-of-run checkpoint save",
    )
    return parser.parse_args()


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: system prompt file not found: {path!r}. Continuing without it.")
        return ""


args = _parse_args()

# Load API key from .env file
load_dotenv(override=False)

# Verify API key is set
if not os.environ.get("MINT_API_KEY"):
    print("WARNING: MINT_API_KEY not found!")
    print("Please create a .env file with: MINT_API_KEY=sk-mint-your-key-here")
else:
    print("MINT_API_KEY loaded.")

# Create timestamped experiment directory
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = Path(f"./experiments/{EXPERIMENT_TIMESTAMP}")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Experiment directory: {EXPERIMENT_DIR}")

# Create the service client
service_client = mint.ServiceClient()

# List available models
print("Connected to MinT server!")
print("\nAvailable models:")
try:
    capabilities = service_client.get_server_capabilities()
    for model in capabilities.supported_models:
        print(f"  - {model.model_name}")
except Exception as e:
    print(f"  Error: {e}")

BASE_MODEL = args.base_model

# Create a training client with LoRA configuration
training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=16,  # LoRA rank - controls adapter capacity
    train_mlp=True,  # Train MLP (feed-forward) layers
    train_attn=True,  # Train attention layers
    train_unembed=True,  # Train the output projection
)
print(f"Training client created for: {BASE_MODEL}")

# Get the tokenizer - converts text to/from token IDs
tokenizer = training_client.get_tokenizer()
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size:,} tokens")

random.seed(42)  # For reproducibility

SYSTEM_PROMPT = _read_text_file(args.system_prompt_path).strip()
if SYSTEM_PROMPT:
    print(f"Loaded system prompt from: {args.system_prompt_path} ({len(SYSTEM_PROMPT)} chars)")
else:
    print("System prompt is empty; prompts will not be extended.")

print(f"Prompt mode: {'chat_template' if hasattr(tokenizer, 'apply_chat_template') else 'plain'}")


def _save_weights_and_get_sampling_client_with_retry(
    service_client: Any,
    training_client: Any,
    base_model: str,
    name: str,
    *,
    max_retries: int = 4,
    backoff_s: float = 1.0,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            sampling_path = training_client.save_weights_for_sampler(name=name).result().path
            return service_client.create_sampling_client(
                model_path=sampling_path,
                base_model=base_model,
            )
        except Exception as e:
            last_exc = e
            print(
                f"save_weights_for_sampler({name!r}) failed: {e} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                time.sleep(min(backoff_s * (2**attempt), 10.0))
    raise RuntimeError(f"Failed to create sampling client for {name!r} after {max_retries} attempts") from last_exc


def extract_answer(response: str) -> str | None:
    """Extract the first numeric answer from response."""
    numbers = re.findall(r"\d+", response)
    return numbers[0] if numbers else None


def generate_rl_problem():
    """Generate multiplication problems for RL."""
    a = random.randint(10, 199)
    b = random.randint(10, 199)
    return f"What is {a} * {b}?", str(a * b)


def compute_reward(response: str, correct_answer: str) -> float:
    """Reward function: 1.0 if correct, 0.0 otherwise."""
    extracted = extract_answer(response)
    return 1.0 if extracted == correct_answer else 0.0


def _build_prompt(question: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages: list[dict[str, str]] = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": question})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    if SYSTEM_PROMPT:
        return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def _token_count_text(text: str) -> int:
    return len(tokenizer.encode(text))


def _as_int_list(xs: list[Any]) -> list[int]:
    return [int(x) for x in xs]


def _as_float_list(xs: list[Any]) -> list[float]:
    out: list[float] = []
    for x in xs:
        v = float(x)
        if not math.isfinite(v):
            v = 0.0
        out.append(v)
    return out


def _decode_with_visible_special_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    """
    Decode token_ids into text while making special/non-text tokens visible.

    This is diagnostic-only; do not use for reward extraction because special-token
    strings may contain digits (e.g. "reserved_special_token_0").
    """
    def _has_visible_glyph(s: str) -> bool:
        for ch in s:
            if ch.isspace():
                continue
            if unicodedata.category(ch).startswith("C"):
                continue
            if not ch.isprintable():
                continue
            return True
        return False

    parts: list[str] = []
    for tid in token_ids:
        s = tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if s and _has_visible_glyph(s):
            parts.append(s)
            continue

        if s:
            parts.append(f"<INVIS_TOKEN:{int(tid)}:{s!r}>")
            continue

        tok = None
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            try:
                tok = tokenizer.convert_ids_to_tokens(int(tid), skip_special_tokens=False)
            except Exception:
                tok = None
        parts.append(f"<NONDECODE_TOKEN:{int(tid)}:{tok}>" if tok else f"<NONDECODE_TOKEN:{int(tid)}>")
    return "".join(parts)


def _summarize_loss_inputs(loss_fn_inputs: dict[str, Any]) -> dict[str, Any]:
    def _len(x: Any) -> int | None:
        try:
            return len(x)
        except Exception:
            return None

    return {
        "keys": sorted(list(loss_fn_inputs.keys())),
        "target_tokens_len": _len(loss_fn_inputs.get("target_tokens")),
        "weights_len": _len(loss_fn_inputs.get("weights")),
        "logprobs_len": _len(loss_fn_inputs.get("logprobs")),
        "advantages_len": _len(loss_fn_inputs.get("advantages")),
        "target_tokens_type": type(loss_fn_inputs.get("target_tokens")).__name__,
        "weights_type": type(loss_fn_inputs.get("weights")).__name__,
        "logprobs_type": type(loss_fn_inputs.get("logprobs")).__name__,
        "advantages_type": type(loss_fn_inputs.get("advantages")).__name__,
    }


def _validate_loss_inputs(loss_fn_inputs: dict[str, Any]) -> tuple[bool, str]:
    required = ["target_tokens", "weights", "logprobs", "advantages"]
    missing = [k for k in required if k not in loss_fn_inputs]
    if missing:
        return False, f"missing keys: {missing}"

    lengths = {
        k: (len(loss_fn_inputs[k]) if hasattr(loss_fn_inputs[k], "__len__") else None)
        for k in required
    }
    if any(v is None for v in lengths.values()):
        return False, f"non-sequence values in loss_fn_inputs: {lengths}"

    uniq = {v for v in lengths.values() if v is not None}
    if len(uniq) != 1:
        return False, f"length mismatch: {lengths}"

    if lengths["target_tokens"] == 0:
        return False, "empty target_tokens"

    return True, "ok"


# Demo
print("Reward function demo:")
print("RL problems: 10-199 * 10-199")
print()
q, a = generate_rl_problem()
print(f"Question: {q}, Answer: {a}")
print(f"  '{a}' → reward = {compute_reward(a, a)}")
print(f"  '999' → reward = {compute_reward('999', a)}")

# RL Configuration
NUM_RL_STEPS = 1
BATCH_SIZE = 8
GROUP_SIZE = 8
MAX_TOKENS = 4096
RL_LEARNING_RATE = 2e-5
TEMPERATURE = 0.7

rl_metrics = []
print("Starting RL training (no SFT)...")
print(
    f"Config: {NUM_RL_STEPS} steps, {BATCH_SIZE} problems/batch, {GROUP_SIZE} samples/problem"
)
print()

for step in range(NUM_RL_STEPS):
    try:
        rl_sampling_client = _save_weights_and_get_sampling_client_with_retry(
            service_client,
            training_client,
            BASE_MODEL,
            name=f"rl-step-{step}",
        )
    except Exception as e:
        print(f"[step {step + 1}] Failed to create sampling client: {e}")
        break

    problems = [generate_rl_problem() for _ in range(BATCH_SIZE)]
    training_datums = []
    all_rewards = []

    for question, answer in problems:
        prompt_text = _build_prompt(question)
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_input = types.ModelInput.from_ints(prompt_tokens)

        if args.print_token_counts:
            print(
                f"[step {step + 1}] prompt_tokens={len(prompt_tokens)} "
                f"(question={question!r}, answer_tokens={_token_count_text(answer)})"
            )

        sample_result = rl_sampling_client.sample(
            prompt=prompt_input,
            num_samples=GROUP_SIZE,
            sampling_params=types.SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop_token_ids=[tokenizer.eos_token_id],
            ),
        ).result()

        group_rewards = []
        group_responses = []
        group_logprobs = []

        for seq_idx, seq in enumerate(sample_result.sequences):
            response_text_for_reward = tokenizer.decode(
                seq.tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response_text = _decode_with_visible_special_tokens(tokenizer, list(seq.tokens))
            #print(response_text)

            eos_id = tokenizer.eos_token_id
            eos_pos = list(seq.tokens).index(eos_id) if eos_id in seq.tokens else None
            hit_max = len(seq.tokens) >= MAX_TOKENS

            reward = compute_reward(response_text_for_reward, answer)
            group_rewards.append(reward)
            group_responses.append(list(seq.tokens))
            group_logprobs.append(list(seq.logprobs) if seq.logprobs else [0.0] * len(seq.tokens))

            if args.print_token_counts:
                print(
                    f"[step {step + 1}] sample[{seq_idx}] output_tokens={len(seq.tokens)} "
                    f"reward={reward} extracted={extract_answer(response_text_for_reward)!r}"
                )

        all_rewards.extend(group_rewards)

        mean_reward = sum(group_rewards) / len(group_rewards)
        advantages = [r - mean_reward for r in group_rewards]

        if all(a == 0 for a in advantages):
            continue

        for response_tokens, logprobs, adv in zip(group_responses, group_logprobs, advantages):
            if len(response_tokens) == 0:
                continue

            full_tokens = prompt_tokens + response_tokens
            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]

            weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(
                response_tokens
            )
            full_logprobs = [0.0] * (len(prompt_tokens) - 1) + list(logprobs)
            full_advantages = [0.0] * (len(prompt_tokens) - 1) + [adv] * len(
                response_tokens
            )

            loss_fn_inputs = dict(
                target_tokens=_as_int_list(target_tokens),
                weights=_as_float_list(weights),
                logprobs=_as_float_list(full_logprobs),
                advantages=_as_float_list(full_advantages),
            )

            ok, msg = _validate_loss_inputs(loss_fn_inputs)
            if not ok:
                print(
                    f"[step {step + 1}] Skipping invalid datum ({msg}). Summary: "
                    f"{_summarize_loss_inputs(loss_fn_inputs)}"
                )
                continue

            training_datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs=loss_fn_inputs,
                )
            )

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    accuracy = (
        sum(1 for r in all_rewards if r > 0) / len(all_rewards)
        if all_rewards
        else 0.0
    )

    if training_datums:
        fwdbwd_future = training_client.forward_backward(
            training_datums,
            loss_fn="importance_sampling",
        )
        fwdbwd_future.result()

        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=RL_LEARNING_RATE),
        )
        optim_future.result()

    rl_metrics.append(
        {
            "step": step,
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "num_datums": len(training_datums),
        }
    )

    print(
        f"Step {step + 1:2d}/{NUM_RL_STEPS}: Accuracy = {accuracy:5.1%}, Avg Reward = {avg_reward:.3f}"
    )

print("\nRL training complete!")

# Get final RL model
final_path = training_client.save_weights_for_sampler(
    name="arithmetic-rl-final",
).result().path
final_client = service_client.create_sampling_client(
    model_path=final_path,
    base_model=BASE_MODEL,
)

# Test on problems (10-199 range)
rl_test_problems = [
    ("What is 123 * 45?", "5535"),
    ("What is 67 * 189?", "12663"),
    ("What is 156 * 78?", "12168"),
]

print("Testing RL-only model (10-199 range):")
print("=" * 50)
rl_correct = 0

for question, correct in rl_test_problems:
    prompt_text = _build_prompt(question)
    if args.print_token_counts:
        print(
            f"[eval] prompt_tokens={_token_count_text(prompt_text)} question={question!r} "
            f"answer_tokens={_token_count_text(correct)}"
        )

    prompt_input = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

    result = final_client.sample(
        prompt=prompt_input,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=16,
            temperature=0.0,
            stop_token_ids=[tokenizer.eos_token_id],
        ),
    ).result()

    response_for_reward = tokenizer.decode(
        result.sequences[0].tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = _decode_with_visible_special_tokens(tokenizer, list(result.sequences[0].tokens))
    extracted = extract_answer(response_for_reward)
    is_correct = extracted == correct
    if is_correct:
        rl_correct += 1

    print(f"Q: {question}")
    print(
        f"A: {response.strip()} (extracted: {extracted}, correct: {correct}) "
        f"[{'PASS' if is_correct else 'FAIL'}]"
    )
    print()

print(f"RL Accuracy: {rl_correct}/{len(rl_test_problems)}")

if rl_metrics:
    df_rl = pd.DataFrame(rl_metrics)
    csv_path = (EXPERIMENT_DIR / "rl_metrics.csv").resolve()
    df_rl.to_csv(csv_path, index=False)
    print(f"Saved RL metrics CSV to: {csv_path}")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot RL Accuracy
rl_steps = [m["step"] + 1 for m in rl_metrics]
rl_accuracy = [m["accuracy"] for m in rl_metrics]

ax.plot(
    rl_steps,
    rl_accuracy,
    "g-o",
    linewidth=2,
    markersize=8,
)
ax.set_xlabel("Step")
ax.set_ylabel("Accuracy")
ax.set_title("RL Training Accuracy (No SFT)")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plot_path = (EXPERIMENT_DIR / "rl_training_results.png").resolve()
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"Plot saved to {plot_path}")

if args.save_ckpt:
    # Save final RL checkpoint
    rl_checkpoint = training_client.save_state(name="arithmetic-rl-final").result()
    print(f"Final checkpoint: {rl_checkpoint.path}")

    ckpt_info = {
        "experiment_dir": str(EXPERIMENT_DIR.resolve()),
        "base_model": BASE_MODEL,
        "system_prompt_path": args.system_prompt_path,
        "final_sampling_path": final_path,
        "final_resume_path": rl_checkpoint.path,
        "metrics_csv": str((EXPERIMENT_DIR / "rl_metrics.csv").resolve()),
        "plot_path": str((EXPERIMENT_DIR / "rl_training_results.png").resolve()),
    }
    ckpt_path = (EXPERIMENT_DIR / "checkpoints.json").resolve()
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ckpt_info, f, indent=2, ensure_ascii=False)
    print(f"Wrote checkpoint links to: {ckpt_path}")

    print("\nTo resume training later:")
    print(
        f"  client = service_client.create_training_client_from_state('{rl_checkpoint.path}')"
    )
else:
    print("Skipping checkpoint save (--no-save-ckpt).")
