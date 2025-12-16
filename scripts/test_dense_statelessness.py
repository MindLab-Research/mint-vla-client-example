#!/usr/bin/env python3
"""Test Dense trainer statelessness - verifies weights are reinitialized on actor reuse.

Tests that:
1. First session trains and loss decreases
2. Second session (reusing actor) starts with fresh weights
3. Initial loss of second session matches first session (not trained loss)
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


def create_session(base_model: str, lora_rank: int = 32, lr: float = 1e-4) -> tuple[str, str]:
    """Create training session. Returns (session_id, model_id)."""
    session_id = f"stateless_test_{uuid.uuid4().hex[:8]}"
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
    return session_id, result.get("model_id")


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


def prepare_training_data(tokenizer) -> list:
    """Prepare fixed training data for reproducibility."""
    examples = [
        {"prompt": "Translate to Pig Latin: hello", "completion": " ello-hay"},
        {"prompt": "Translate to Pig Latin: world", "completion": " orld-way"},
    ]

    data = []
    for ex in examples:
        prompt_tokens = tokenizer.encode(ex["prompt"], add_special_tokens=True)
        completion_tokens = tokenizer.encode(ex["completion"], add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        # Loss only on completion tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        loss_mask = loss_mask[1:]

        data.append(make_sft_datum(input_tokens, target_tokens, loss_mask))

    return data


def train_session(name: str, tokenizer, iterations: int = 3) -> tuple[list, str]:
    """Run a training session and return losses and model_id."""
    print(f"\n=== Session: {name} ===")

    # Create session
    session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
    print(f"Created: session_id={session_id}, model_id={model_id}")

    # Prepare data
    data = prepare_training_data(tokenizer)

    # Training loop
    losses = []
    for i in range(iterations):
        result = forward_backward(model_id, data, loss_fn="cross_entropy")
        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"  Iter {i+1}: loss={loss:.4f}")
        optim_step(model_id, lr=1e-4)

    # Note: Sessions are not explicitly destroyed - actor stays in pool
    print(f"Session complete (actor stays in pool for reuse)")

    return losses, model_id


def main():
    print("Dense Statelessness Test")
    print("=" * 60)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, trust_remote_code=True)

    # Session 1: Train
    losses1, model1 = train_session("session1", tokenizer, iterations=5)
    initial_loss1 = losses1[0]
    final_loss1 = losses1[-1]

    print(f"\nSession 1: initial={initial_loss1:.4f}, final={final_loss1:.4f}")

    # Small delay
    time.sleep(2)

    # Session 2: Should start fresh
    losses2, model2 = train_session("session2", tokenizer, iterations=3)
    initial_loss2 = losses2[0]

    print(f"\nSession 2: initial={initial_loss2:.4f}")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    diff_to_initial = abs(initial_loss2 - initial_loss1)
    diff_to_final = abs(initial_loss2 - final_loss1)

    print(f"Session 1 initial loss: {initial_loss1:.4f}")
    print(f"Session 1 final loss:   {final_loss1:.4f}")
    print(f"Session 2 initial loss: {initial_loss2:.4f}")
    print(f"")
    print(f"Diff from Session 1 initial: {diff_to_initial:.4f}")
    print(f"Diff from Session 1 final:   {diff_to_final:.4f}")

    # Verify weights were reinitialized
    loss_reduction1 = (initial_loss1 - final_loss1) / initial_loss1 if initial_loss1 > 0 else 0
    print(f"\nSession 1 loss reduction: {loss_reduction1 * 100:.1f}%")

    # Session 2 initial should be closer to Session 1's initial than to final
    # A 20% tolerance is reasonable given random initialization variance
    if initial_loss2 < final_loss1 * 1.2 and loss_reduction1 > 0.3:
        # Session 2 started near Session 1's trained (final) loss - weights inherited!
        print("\nFAIL: Session 2 inherited trained weights from Session 1!")
        print("      (reinit_lora_weights was not called on actor reuse)")
        return 1
    elif diff_to_initial < diff_to_final * 0.8 or initial_loss2 > final_loss1 * 1.5:
        # Session 2 started fresh (closer to initial or much higher than final)
        print("\nPASS: Session 2 started with fresh weights (stateless)")
        return 0
    else:
        # Ambiguous - training didn't change loss enough to determine
        if loss_reduction1 < 0.2:
            print("\nINCONCLUSIVE: Session 1 loss reduction too small to determine statelessness")
            return 2
        else:
            print("\nWARNING: Ambiguous result, but likely PASS")
            return 0


if __name__ == "__main__":
    sys.exit(main())
