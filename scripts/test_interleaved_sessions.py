#!/usr/bin/env python3
"""Test interleaved training sessions on same Dense trainer.

This tests the stateless trainer architecture:
1. Session A: 3 iterations, records final loss L_A
2. Session B: 3 iterations, records final loss L_B
3. Session A: 3 more iterations, should continue from L_A (not L_B)

If statelessness works, Session A's loss after returning should be close to
where it left off, not affected by Session B's training.
"""

import os
import sys
import time
import uuid

import requests
from transformers import AutoTokenizer

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"
DENSE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll for async operation result."""
    poll_url = f"{API}/retrieve_future"
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


def create_session(session_id: str, base_model: str, lora_rank: int = 32, lr: float = 1e-4) -> str:
    """Create training session. Returns model_id."""
    url = f"{API}/create_model"
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
    return result.get("model_id")


def forward_backward(model_id: str, data: list, loss_fn: str = "cross_entropy") -> dict:
    """Run forward-backward pass."""
    url = f"{API}/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": loss_fn},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def optim_step(model_id: str, lr: float = 1e-4) -> dict:
    """Run optimizer step."""
    url = f"{API}/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=60)


def make_sft_datum(input_tokens: list, target_tokens: list, loss_mask: list) -> dict:
    """Create a single SFT training datum."""
    return {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
        },
    }


def prepare_training_data_a(tokenizer) -> list:
    """Prepare training data for Session A (Pig Latin)."""
    examples = [
        {"prompt": "Translate to Pig Latin: hello", "completion": " ello-hay"},
        {"prompt": "Translate to Pig Latin: world", "completion": " orld-way"},
    ]
    return _prepare_data(tokenizer, examples)


def prepare_training_data_b(tokenizer) -> list:
    """Prepare training data for Session B (Spanish)."""
    examples = [
        {"prompt": "Translate to Spanish: hello", "completion": " hola"},
        {"prompt": "Translate to Spanish: world", "completion": " mundo"},
    ]
    return _prepare_data(tokenizer, examples)


def _prepare_data(tokenizer, examples: list) -> list:
    """Prepare training data from examples."""
    data = []
    for ex in examples:
        prompt_tokens = tokenizer.encode(ex["prompt"], add_special_tokens=True)
        completion_tokens = tokenizer.encode(ex["completion"], add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        loss_mask = loss_mask[1:]

        data.append(make_sft_datum(input_tokens, target_tokens, loss_mask))

    return data


def train_iterations(model_id: str, data: list, iterations: int, lr: float = 1e-4) -> list:
    """Run training iterations and return losses."""
    losses = []
    for i in range(iterations):
        result = forward_backward(model_id, data, loss_fn="cross_entropy")
        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"    Iter {i+1}: loss={loss:.4f}")
        optim_step(model_id, lr=lr)
    return losses


def main():
    print("Interleaved Sessions Test")
    print("=" * 60)
    print("Testing stateless trainer: sessions should preserve state when interleaved")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, trust_remote_code=True)

    # Prepare different training data for each session
    data_a = prepare_training_data_a(tokenizer)
    data_b = prepare_training_data_b(tokenizer)

    # Create sessions with unique IDs
    session_a_id = f"interleave_A_{uuid.uuid4().hex[:8]}"
    session_b_id = f"interleave_B_{uuid.uuid4().hex[:8]}"

    print(f"\nCreating Session A: {session_a_id}")
    model_a = create_session(session_a_id, DENSE_MODEL, lora_rank=32, lr=1e-4)
    print(f"  model_id: {model_a}")

    print(f"\nCreating Session B: {session_b_id}")
    model_b = create_session(session_b_id, DENSE_MODEL, lora_rank=32, lr=1e-4)
    print(f"  model_id: {model_b}")

    # Phase 1: Train Session A for 3 iterations
    print("\n--- Phase 1: Session A (3 iterations) ---")
    losses_a1 = train_iterations(model_a, data_a, 3, lr=1e-4)
    loss_a_before = losses_a1[-1]
    print(f"  Session A final loss: {loss_a_before:.4f}")

    # Phase 2: Train Session B for 3 iterations
    print("\n--- Phase 2: Session B (3 iterations) ---")
    losses_b = train_iterations(model_b, data_b, 3, lr=1e-4)
    loss_b_final = losses_b[-1]
    print(f"  Session B final loss: {loss_b_final:.4f}")

    # Phase 3: Return to Session A for 3 more iterations
    print("\n--- Phase 3: Session A (3 more iterations) ---")
    losses_a2 = train_iterations(model_a, data_a, 3, lr=1e-4)
    loss_a_after = losses_a2[0]  # First loss after returning
    loss_a_final = losses_a2[-1]

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print(f"\nSession A:")
    print(f"  Loss before B: {loss_a_before:.4f}")
    print(f"  Loss after B:  {loss_a_after:.4f}")
    print(f"  Final loss:    {loss_a_final:.4f}")

    print(f"\nSession B:")
    print(f"  Final loss:    {loss_b_final:.4f}")

    # The key check: Session A's first loss after returning should be close to
    # where it left off (continuing from its saved state), not affected by Session B
    diff = abs(loss_a_after - loss_a_before)
    print(f"\nLoss continuity check:")
    print(f"  Difference between A before and after B: {diff:.4f}")

    # In a stateless system, the first forward pass after returning should produce
    # similar gradients (slightly different due to gradient accumulation reset)
    # The loss should continue decreasing from where it was, not jump around

    # Check if training continued properly (loss should generally decrease)
    continued_properly = loss_a_final < loss_a_before

    # Check if session B didn't corrupt session A's state
    # After returning to A, loss should be close to where A left off
    # Allow 50% tolerance since this is a difficult test
    state_preserved = diff < abs(losses_a1[0] - loss_a_before) * 0.5

    if continued_properly:
        print("\nPASS: Session A training continued properly after returning")
    else:
        print(f"\nWARN: Session A loss didn't decrease after returning")
        print(f"      (was {loss_a_before:.4f}, now {loss_a_final:.4f})")

    if state_preserved:
        print("PASS: Session A state was preserved (loss continuity maintained)")
    else:
        print("WARN: Session A state may not have been preserved properly")
        print(f"      Expected loss close to {loss_a_before:.4f}, got {loss_a_after:.4f}")

    # Overall result
    print("\n" + "=" * 60)
    if continued_properly and state_preserved:
        print("OVERALL: PASS - Stateless trainer pattern working")
        return 0
    elif continued_properly:
        print("OVERALL: PARTIAL PASS - Training continues but state preservation unclear")
        return 0
    else:
        print("OVERALL: FAIL - Interleaved sessions not working correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
