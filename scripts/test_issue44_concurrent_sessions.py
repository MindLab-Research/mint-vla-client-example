#!/usr/bin/env python3
"""Test Issue #44: Concurrent session management with real cookbook tasks.

This test verifies that MegatronWorkerGroup correctly isolates session state
(LoRA weights + optimizer state + gradients) when training multiple sessions
concurrently.

Test design:
- Task A: Arithmetic (simple addition, short numeric output)
- Task B: Countdown (equation solving, XML-formatted output)

Comparison:
- Phase 1: Train A solo (10 steps) -> record loss curve
- Phase 2: Train B solo (10 steps) -> record loss curve
- Phase 3: Train A+B interleaved (10 steps each) -> record both curves
- Verify: Solo curves ≈ Concurrent curves

If optimizer state leaks between sessions, concurrent curves will diverge
from solo curves (different momentum causes different updates).

Usage:
    TINKER_BASE_URL=http://localhost:8000 TINKER_TELEMETRY=0 \
        python scripts/test_issue44_concurrent_sessions.py
"""

import os
import random
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SEQ_ID = 0

STEPS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# Fix random seed for reproducibility
random.seed(42)


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(2)
            continue
        else:
            resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_model(session_id: str, lora_rank: int = 8) -> str:
    """Create a training model. Returns model_id."""
    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": MODEL_SEQ_ID,
        "base_model": MODEL_NAME,
        "lora_config": {"rank": lora_rank},
    }
    resp = requests.post(url, json=payload, timeout=900)
    resp.raise_for_status()
    result = resp.json()

    if "request_id" in result:
        result = poll_future(result["request_id"], timeout=900)

    model_id = result.get("model_id", f"{session_id}_{MODEL_SEQ_ID}")
    return model_id


def forward_backward(model_id: str, data: list[dict]) -> dict:
    """Run forward-backward pass."""
    url = f"{BASE_URL}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": "cross_entropy",
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    if "request_id" in result:
        result = poll_future(result["request_id"])

    return result


def optim_step(model_id: str, learning_rate: float = LEARNING_RATE) -> dict:
    """Run optimizer step."""
    url = f"{BASE_URL}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {
            "learning_rate": learning_rate,
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    if "request_id" in result:
        result = poll_future(result["request_id"])

    return result


def create_datum(prompt: str, response: str, tokenizer) -> dict:
    """Create a Tinker Datum for training."""
    full_text = f"{prompt}{response}"
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)

    # Target tokens are shifted by 1 (predict next token)
    target_tokens = tokens[1:] + [tokenizer.eos_token_id or 0]
    loss_mask = [0.0] * len(prompt_tokens) + [1.0] * (len(tokens) - len(prompt_tokens))

    return {
        "model_input": {
            "chunks": [{"tokens": tokens, "type": "encoded_text"}]
        },
        "loss_fn_inputs": {
            "target_tokens": {
                "data": target_tokens,
                "shape": [len(target_tokens)],
                "dtype": "int64",
            },
            "loss_mask": {
                "data": loss_mask,
                "shape": [len(loss_mask)],
                "dtype": "float32",
            },
        },
    }


def create_arithmetic_data(n: int, tokenizer) -> list[dict]:
    """Task A: Simple addition problems."""
    data = []
    for _ in range(n):
        x, y = random.randint(0, 100), random.randint(0, 100)
        prompt = f"What is {x} + {y}? "
        answer = str(x + y)
        data.append(create_datum(prompt, answer, tokenizer))
    return data


def create_countdown_data(n: int, tokenizer) -> list[dict]:
    """Task B: Countdown equation problems."""
    # Fixed set of solvable problems for reproducibility
    problems = [
        ([44, 19, 35], 98, "44 + 35 + 19"),
        ([10, 5, 2], 17, "10 + 5 + 2"),
        ([20, 8, 3], 15, "20 - 8 + 3"),
        ([50, 25, 10], 65, "50 + 25 - 10"),
        ([100, 50, 25], 75, "100 - 50 + 25"),
        ([30, 20, 10], 40, "30 + 20 - 10"),
        ([15, 7, 3], 25, "15 + 7 + 3"),
        ([80, 40, 20], 60, "80 - 40 + 20"),
        ([12, 6, 4], 14, "12 + 6 - 4"),
        ([90, 45, 15], 60, "90 - 45 + 15"),
    ]
    data = []
    for i in range(n):
        nums, target, solution = problems[i % len(problems)]
        prompt = f"Using the numbers {nums}, create an equation that equals {target}. "
        answer = f"<answer> {solution} = {target} </answer>"
        data.append(create_datum(prompt, answer, tokenizer))
    return data


def train_solo(session_id: str, data: list[dict], steps: int) -> list[float]:
    """Train a single session for N steps, return loss curve."""
    print(f"\n  Creating model for {session_id}...")
    model_id = create_model(session_id)
    print(f"  Model created: {model_id}")

    losses = []
    for step in range(steps):
        batch = data[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]

        t0 = time.time()
        result = forward_backward(model_id, batch)
        optim_step(model_id)
        t1 = time.time()

        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"  Step {step + 1}/{steps}: loss={loss:.4f}, time={t1 - t0:.2f}s")

    return losses


def train_interleaved(
    session_a: str, data_a: list[dict],
    session_b: str, data_b: list[dict],
    steps: int
) -> tuple[list[float], list[float]]:
    """Train two sessions interleaved: A.fb -> B.fb -> A.optim -> B.optim."""
    print(f"\n  Creating model for {session_a}...")
    model_a_id = create_model(session_a)
    print(f"  Model A created: {model_a_id}")

    print(f"  Creating model for {session_b}...")
    model_b_id = create_model(session_b)
    print(f"  Model B created: {model_b_id}")

    losses_a, losses_b = [], []
    for step in range(steps):
        batch_a = data_a[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        batch_b = data_b[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]

        t0 = time.time()

        # Interleaved pattern (tests gradient + optimizer isolation)
        # A's forward-backward
        result_a = forward_backward(model_a_id, batch_a)
        # B's forward-backward (A's gradients must be preserved)
        result_b = forward_backward(model_b_id, batch_b)
        # A's optim_step (must use A's gradients, not B's)
        optim_step(model_a_id)
        # B's optim_step (must use B's gradients)
        optim_step(model_b_id)

        t1 = time.time()

        loss_a = result_a.get("metrics", {}).get("loss:mean", 0)
        loss_b = result_b.get("metrics", {}).get("loss:mean", 0)
        losses_a.append(loss_a)
        losses_b.append(loss_b)

        print(f"  Step {step + 1}/{steps}: A_loss={loss_a:.4f}, B_loss={loss_b:.4f}, time={t1 - t0:.2f}s")

    return losses_a, losses_b


def main():
    from transformers import AutoTokenizer

    print("=" * 70)
    print("Test Issue #44: Concurrent Session Management")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Steps: {STEPS}, Batch size: {BATCH_SIZE}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Generate unique session IDs
    test_id = uuid.uuid4().hex[:8]

    # Generate data (need enough for all phases)
    # Arithmetic: 3 phases (solo1, solo2, concurrent)
    # Countdown: 2 phases (solo, concurrent)
    print("Generating training data...")
    arith_data = create_arithmetic_data(STEPS * BATCH_SIZE * 3, tokenizer)
    countdown_data = create_countdown_data(STEPS * BATCH_SIZE * 2, tokenizer)
    print(f"  Arithmetic samples: {len(arith_data)}")
    print(f"  Countdown samples: {len(countdown_data)}")

    # Phase 1: Train Arithmetic solo (first time)
    print("\n" + "=" * 70)
    print("PHASE 1: Train Arithmetic solo (FIRST)")
    print("=" * 70)
    arith_solo_1 = train_solo(
        f"arith-solo-{test_id}",
        arith_data[:STEPS * BATCH_SIZE],
        STEPS
    )

    # Phase 2: Train Countdown solo
    print("\n" + "=" * 70)
    print("PHASE 2: Train Countdown solo")
    print("=" * 70)
    countdown_solo = train_solo(
        f"countdown-solo-{test_id}",
        countdown_data[:STEPS * BATCH_SIZE],
        STEPS
    )

    # Phase 3: Train Arithmetic solo AGAIN (NEW session ID!)
    print("\n" + "=" * 70)
    print("PHASE 3: Train Arithmetic solo (SECOND - NEW session ID)")
    print("=" * 70)
    arith_solo_2 = train_solo(
        f"arith-solo-NEW-{test_id}",  # DIFFERENT session ID!
        arith_data[STEPS * BATCH_SIZE:STEPS * BATCH_SIZE * 2],  # Different data
        STEPS
    )

    # Phase 4: Train Arithmetic + Countdown concurrently (interleaved)
    print("\n" + "=" * 70)
    print("PHASE 4: Train Arithmetic + Countdown concurrent (INTERLEAVED)")
    print("=" * 70)
    arith_concurrent, countdown_concurrent = train_interleaved(
        f"arith-concurrent-{test_id}",
        arith_data[STEPS * BATCH_SIZE * 2:STEPS * BATCH_SIZE * 3],
        f"countdown-concurrent-{test_id}",
        countdown_data[STEPS * BATCH_SIZE:STEPS * BATCH_SIZE * 2],
        STEPS,
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: Loss Curves")
    print("=" * 70)

    print("\nArithmetic - Solo (FIRST):")
    print(f"{'Step':<6} {'Loss':<12}")
    print("-" * 18)
    for i, loss in enumerate(arith_solo_1):
        print(f"{i:<6} {loss:<12.4f}")

    print("\nCountdown - Solo:")
    print(f"{'Step':<6} {'Loss':<12}")
    print("-" * 18)
    for i, loss in enumerate(countdown_solo):
        print(f"{i:<6} {loss:<12.4f}")

    print("\nArithmetic - Solo (SECOND - new session):")
    print(f"{'Step':<6} {'Loss':<12}")
    print("-" * 18)
    for i, loss in enumerate(arith_solo_2):
        print(f"{i:<6} {loss:<12.4f}")

    print("\nArithmetic - Concurrent:")
    print(f"{'Step':<6} {'Loss':<12}")
    print("-" * 18)
    for i, loss in enumerate(arith_concurrent):
        print(f"{i:<6} {loss:<12.4f}")

    print("\nCountdown - Concurrent:")
    print(f"{'Step':<6} {'Loss':<12}")
    print("-" * 18)
    for i, loss in enumerate(countdown_concurrent):
        print(f"{i:<6} {loss:<12.4f}")

    # Check for the bug
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"\nArith FIRST:  {arith_solo_1[0]:.4f} -> {arith_solo_1[-1]:.4f}")
    print(f"Arith SECOND: {arith_solo_2[0]:.4f} -> {arith_solo_2[-1]:.4f}")

    if arith_solo_2[0] < 0.1:
        print("\n*** BUG CONFIRMED: Second Arith starts with near-zero loss! ***")
        print("*** Session is loading trained weights instead of fresh weights ***")


if __name__ == "__main__":
    main()
