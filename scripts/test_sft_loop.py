#!/usr/bin/env python
"""SFT training loop demonstrating format learning.

Task: Teach model a complex structured output format.
Requires multiple iterations to achieve perfect accuracy.

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_sft_loop.py
"""

import os
import re
import sys
import time

tinker_path = os.path.join(os.path.dirname(__file__), "../../tinker/src")
if os.path.exists(tinker_path):
    sys.path.insert(0, tinker_path)

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

import torch
from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer


# Task: Model must respond with XML-like structured format
# <op>OP_TYPE</op><a>NUM</a><b>NUM</b><r>RESULT</r>
#
# This is challenging because:
# 1. Unusual XML-like format the model won't naturally produce
# 2. Must extract operator type and map to canonical name
# 3. Must extract both operands correctly
# 4. Must compute and include result
# 5. All fields must be present and correctly formatted

def make_response(op, a, b, result):
    return f"<op>{op}</op><a>{a}</a><b>{b}</b><r>{result}</r>"

TRAIN_DATA = [
    # Addition
    {"prompt": "Parse: 5 plus 3", "response": make_response("ADD", 5, 3, 8)},
    {"prompt": "Parse: 12 plus 7", "response": make_response("ADD", 12, 7, 19)},
    {"prompt": "Parse: 8 plus 15", "response": make_response("ADD", 8, 15, 23)},
    # Subtraction
    {"prompt": "Parse: 20 minus 7", "response": make_response("SUB", 20, 7, 13)},
    {"prompt": "Parse: 15 minus 6", "response": make_response("SUB", 15, 6, 9)},
    {"prompt": "Parse: 30 minus 12", "response": make_response("SUB", 30, 12, 18)},
    # Multiplication
    {"prompt": "Parse: 4 times 6", "response": make_response("MUL", 4, 6, 24)},
    {"prompt": "Parse: 7 times 8", "response": make_response("MUL", 7, 8, 56)},
    {"prompt": "Parse: 3 times 9", "response": make_response("MUL", 3, 9, 27)},
    # Division
    {"prompt": "Parse: 20 divided by 4", "response": make_response("DIV", 20, 4, 5)},
    {"prompt": "Parse: 36 divided by 6", "response": make_response("DIV", 36, 6, 6)},
    {"prompt": "Parse: 45 divided by 9", "response": make_response("DIV", 45, 9, 5)},
]

TEST_DATA = [
    # Different numbers, same patterns
    {"prompt": "Parse: 9 plus 4", "op": "ADD", "a": 9, "b": 4, "r": 13},
    {"prompt": "Parse: 25 minus 8", "op": "SUB", "a": 25, "b": 8, "r": 17},
    {"prompt": "Parse: 6 times 7", "op": "MUL", "a": 6, "b": 7, "r": 42},
    {"prompt": "Parse: 48 divided by 8", "op": "DIV", "a": 48, "b": 8, "r": 6},
    {"prompt": "Parse: 11 plus 9", "op": "ADD", "a": 11, "b": 9, "r": 20},
    {"prompt": "Parse: 18 minus 5", "op": "SUB", "a": 18, "b": 5, "r": 13},
    {"prompt": "Parse: 5 times 5", "op": "MUL", "a": 5, "b": 5, "r": 25},
    {"prompt": "Parse: 72 divided by 9", "op": "DIV", "a": 72, "b": 9, "r": 8},
]


def check_format(response: str, expected: dict) -> tuple[bool, dict]:
    """Check if response matches XML format with correct values.

    Returns (format_ok, field_matches) where field_matches shows which fields are correct.
    """
    response = response.strip()

    # Parse expected values
    exp_op = expected.get("op") or expected["response"].split("<op>")[1].split("</op>")[0]
    exp_a = str(expected.get("a", ""))
    exp_b = str(expected.get("b", ""))
    exp_r = str(expected.get("r", ""))

    if not exp_a:  # Extract from response string if not provided
        resp = expected["response"]
        exp_a = resp.split("<a>")[1].split("</a>")[0]
        exp_b = resp.split("<b>")[1].split("</b>")[0]
        exp_r = resp.split("<r>")[1].split("</r>")[0]

    # Check format structure
    pattern = r"^<op>(\w+)</op><a>(\d+)</a><b>(\d+)</b><r>(\d+)</r>$"
    match = re.match(pattern, response)

    if not match:
        return False, {"op": False, "a": False, "b": False, "r": False}

    got_op, got_a, got_b, got_r = match.groups()

    field_matches = {
        "op": got_op == exp_op,
        "a": got_a == exp_a,
        "b": got_b == exp_b,
        "r": got_r == exp_r,
    }

    all_correct = all(field_matches.values())
    return all_correct, field_matches


def evaluate(sampler, tokenizer, data, sample_params):
    """Evaluate model on dataset."""
    perfect = 0
    field_scores = {"op": 0, "a": 0, "b": 0, "r": 0}

    for ex in data:
        tokens = tokenizer.encode(ex["prompt"], add_special_tokens=True)
        model_input = types.ModelInput.from_ints(tokens)
        result = sampler.sample(prompt=model_input, num_samples=1, sampling_params=sample_params).result()
        resp = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

        ok, fields = check_format(resp, ex)

        status = "OK" if ok else "bad"
        print(f"  [{status}] {ex['prompt'][:25]:25} -> {resp[:45]}")

        if ok:
            perfect += 1
        for k, v in fields.items():
            if v:
                field_scores[k] += 1

    n = len(data)
    return {
        "perfect": perfect / n,
        "op": field_scores["op"] / n,
        "a": field_scores["a"] / n,
        "b": field_scores["b"] / n,
        "r": field_scores["r"] / n,
    }


def main():
    print(f"Server: {os.environ.get('TINKER_BASE_URL')}")

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    service_client = ServiceClient(base_url=os.environ.get("TINKER_BASE_URL"))

    print("\nCreating training client...")
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=32,
    )

    # Config
    NUM_ITER = 10
    TRAIN_STEPS = 3
    LR = 1e-3

    adam = types.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)
    sample_params = types.SamplingParams(max_tokens=50, temperature=0.1)

    print(f"\nTask: Learn XML-like structured output format")
    print(f"Format: <op>OP</op><a>NUM</a><b>NUM</b><r>RESULT</r>")
    print(f"Config: {NUM_ITER} iterations, {TRAIN_STEPS} steps/iter, lr={LR}")
    print(f"Train examples: {len(TRAIN_DATA)}, Test examples: {len(TEST_DATA)}")

    # Prepare training datums
    eos_token_id = tokenizer.eos_token_id
    print(f"EOS token ID: {eos_token_id}")

    train_datums = []
    for ex in TRAIN_DATA:
        prompt_tokens = tokenizer.encode(ex["prompt"], add_special_tokens=True)
        resp_tokens = tokenizer.encode(ex["response"], add_special_tokens=False)
        full = prompt_tokens + resp_tokens + [eos_token_id]
        plen = len(prompt_tokens)
        resp_len = len(resp_tokens) + 1
        mask = [0.0] * (plen - 1) + [1.0] * resp_len

        train_datums.append(types.Datum(
            model_input=types.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(full[1:], dtype=torch.long)),
                "loss_mask": TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            },
        ))

    stats = []

    # ================================================================
    # BASELINE: Evaluate BEFORE any training
    # ================================================================
    print(f"\n{'='*70}")
    print("BASELINE (before training)")
    print(f"{'='*70}")

    t0 = time.time()
    sampler = training_client.save_weights_and_get_sampling_client()
    sync_time = time.time() - t0
    print(f"Sync: {sync_time:.2f}s")

    print("\nTrain prompts (baseline):")
    train_scores = evaluate(sampler, tokenizer, TRAIN_DATA, sample_params)
    print(f"Train: perfect={train_scores['perfect']:.0%}, op={train_scores['op']:.0%}, a={train_scores['a']:.0%}, b={train_scores['b']:.0%}, r={train_scores['r']:.0%}")

    print("\nTest prompts (baseline):")
    test_scores = evaluate(sampler, tokenizer, TEST_DATA, sample_params)
    print(f"Test: perfect={test_scores['perfect']:.0%}, op={test_scores['op']:.0%}, a={test_scores['a']:.0%}, b={test_scores['b']:.0%}, r={test_scores['r']:.0%}")

    stats.append({
        "iter": 0,
        "train_perfect": train_scores["perfect"],
        "test_perfect": test_scores["perfect"],
        "test_op": test_scores["op"],
        "test_r": test_scores["r"],
        "train_time": 0,
        "sync_time": sync_time,
    })

    # ================================================================
    # TRAINING LOOP
    # ================================================================
    for iteration in range(1, NUM_ITER + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{NUM_ITER}")
        print(f"{'='*70}")

        # Train
        t_train = time.time()
        for step in range(TRAIN_STEPS):
            training_client.forward_backward(train_datums, loss_fn="cross_entropy").result()
            training_client.optim_step(adam).result()
        train_time = time.time() - t_train
        print(f"Train: {train_time:.2f}s ({len(train_datums)} examples x {TRAIN_STEPS} steps)")

        # Sync weights
        t_sync = time.time()
        sampler = training_client.save_weights_and_get_sampling_client()
        sync_time = time.time() - t_sync
        print(f"Sync: {sync_time:.2f}s")

        # Evaluate
        print("\nTrain prompts:")
        train_scores = evaluate(sampler, tokenizer, TRAIN_DATA, sample_params)
        print(f"Train: perfect={train_scores['perfect']:.0%}")

        print("\nTest prompts:")
        test_scores = evaluate(sampler, tokenizer, TEST_DATA, sample_params)
        print(f"Test: perfect={test_scores['perfect']:.0%}, op={test_scores['op']:.0%}, r={test_scores['r']:.0%}")

        stats.append({
            "iter": iteration,
            "train_perfect": train_scores["perfect"],
            "test_perfect": test_scores["perfect"],
            "test_op": test_scores["op"],
            "test_r": test_scores["r"],
            "train_time": train_time,
            "sync_time": sync_time,
        })

        # Early stopping if perfect on both train and test
        if train_scores["perfect"] == 1.0 and test_scores["perfect"] == 1.0:
            print("\n** Perfect accuracy achieved, stopping early **")
            break

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Iter':>4} | {'Train%':>6} | {'Test%':>5} | {'Op%':>4} | {'Result%':>7} | {'Train':>6} | {'Sync':>5}")
    print("-" * 70)
    for s in stats:
        train_str = f"{s['train_time']:.1f}s" if s['train_time'] > 0 else "-"
        print(f"{s['iter']:>4} | {s['train_perfect']:>5.0%} | {s['test_perfect']:>4.0%} | {s['test_op']:>3.0%} | {s['test_r']:>6.0%} | {train_str:>6} | {s['sync_time']:.1f}s")

    print(f"\nBaseline -> Final:")
    print(f"  Train perfect: {stats[0]['train_perfect']:.0%} -> {stats[-1]['train_perfect']:.0%}")
    print(f"  Test perfect:  {stats[0]['test_perfect']:.0%} -> {stats[-1]['test_perfect']:.0%}")

    total_steps = (len(stats) - 1) * TRAIN_STEPS
    print(f"\nTotal training steps: {total_steps}")


if __name__ == "__main__":
    main()
