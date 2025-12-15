#!/usr/bin/env python3
"""Pig Latin SFT Test for MoE Model.

Tests the same Pig Latin translation task as test_tinker_compatibility.py
but with the Qwen3-30B-A3B MoE model using the train_step endpoint.

Usage:
    TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_pig_latin_moe.py
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests


def get_base_url():
    return os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll for async result."""
    poll_url = f"{get_base_url()}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()
    raise TimeoutError(f"Operation did not complete within {timeout}s")


# Training examples from the notebook
PIG_LATIN_EXAMPLES = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]


def process_example(example: dict, tokenizer) -> dict:
    """Convert example to API format.

    From notebook:
    - Format: "English: {input}\nPig Latin:"
    - Prompt tokens get weight 0
    - Completion tokens (with leading space) get weight 1
    - Targets are shifted by 1
    """
    prompt = f"English: {example['input']}\nPig Latin:"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    # Add a space before the output string, and finish with double newline
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]  # Shifted by 1 for next-token prediction
    weights = weights[1:]

    return {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
        },
    }


def create_session(base_model: str, lora_rank: int, lr: float) -> tuple[str, str]:
    """Create training session."""
    session_id = f"pig_latin_moe_{uuid.uuid4().hex[:8]}"
    url = f"{get_base_url()}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")
    return session_id, result.get("model_id")


def train_step(model_id: str, data: list, lr: float) -> dict:
    """Execute combined forward_backward + optim_step."""
    url = f"{get_base_url()}/api/v1/train_step"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def main():
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32
    lr = 1e-4
    num_updates = 10

    print("=" * 70)
    print("TEST: Pig Latin SFT (MoE Model)")
    print("=" * 70)
    print(f"Base model: {base_model}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Learning rate: {lr}")
    print(f"Num updates: {num_updates}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Process examples
    processed_examples = [process_example(ex, tokenizer) for ex in PIG_LATIN_EXAMPLES]
    print(f"Processed {len(processed_examples)} examples")

    # Visualize first example
    print("\nFirst example visualization:")
    print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
    print("-" * 50)
    ex0 = processed_examples[0]
    input_tokens = ex0["model_input"]["chunks"][0]["tokens"]
    target_tokens = ex0["loss_fn_inputs"]["target_tokens"]["data"]
    weights = ex0["loss_fn_inputs"]["loss_mask"]["data"]
    for inp, tgt, wgt in zip(input_tokens, target_tokens, weights):
        print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")

    # Create session
    print("\nCreating session...")
    start_time = time.time()
    session_id, model_id = create_session(base_model, lora_rank, lr)
    create_time = time.time() - start_time
    print(f"Session created: {session_id} ({create_time:.1f}s)")

    # Training loop
    print("\nStarting training updates...")
    results = []
    for i in range(num_updates):
        start_time = time.time()
        result = train_step(model_id, processed_examples, lr)
        elapsed = time.time() - start_time

        metrics = result.get("metrics", {})
        loss = metrics.get("loss:mean", 0)
        grad_norm = metrics.get("grad_norm", 0)

        results.append({
            "iter": i + 1,
            "loss": loss,
            "grad_norm": grad_norm,
            "elapsed": elapsed,
        })

        print(f"Update {i+1}: Loss = {loss:.4f}, grad_norm = {grad_norm:.4f} ({elapsed:.2f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    initial_loss = results[0]["loss"]
    final_loss = results[-1]["loss"]
    min_loss = min(r["loss"] for r in results)
    avg_time = np.mean([r["elapsed"] for r in results])

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Min loss: {min_loss:.4f}")
    print(f"Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    print(f"Avg iteration time: {avg_time:.2f}s")

    # Validation
    passed = True
    if final_loss >= initial_loss:
        print("\nFAIL: Loss did not decrease")
        passed = False
    else:
        reduction = (initial_loss - final_loss) / initial_loss
        if reduction < 0.3:
            print(f"\nWARN: Loss reduction only {reduction*100:.1f}% (expected >30%)")
        else:
            print(f"\nPASS: Loss decreased by {reduction*100:.1f}%")

    # Save results
    results_dir = Path("results/pig_latin_moe")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"pig_latin_moe_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "base_model": base_model,
            "lora_rank": lora_rank,
            "learning_rate": lr,
            "num_updates": num_updates,
            "create_time": create_time,
            "results": results,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "min_loss": min_loss,
            "avg_time": avg_time,
            "passed": passed,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
