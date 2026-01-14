#!/usr/bin/env python3
"""Test Issue #44: Automatic session state management in MegatronWorkerGroup.

This test verifies that MegatronWorkerGroup automatically saves/loads LoRA weights
when sessions switch during forward_backward() calls.

Before the fix:
- Session B would use Session A's LoRA weights (wrong!)
- LoRA weights persisted in GPU memory forever

After the fix:
- forward_backward() calls _ensure_session_loaded() which triggers swap_session()
- LoRA weights are saved to disk when switching away from a session
- LoRA weights are loaded from disk when switching back to a session

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_issue44_auto_session_swap.py
"""

import os
import sys
import time
import uuid
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SEQ_ID = 0  # Fixed model sequence ID for this test


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


def create_model(session_id: str, lora_rank: int = 8) -> tuple[str, dict]:
    """Create a training model. Returns (model_id, result)."""
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

    # Poll if async
    if "request_id" in result:
        result = poll_future(result["request_id"], timeout=900)

    model_id = result.get("model_id", f"{session_id}_{MODEL_SEQ_ID}")
    return model_id, result


def forward_backward(model_id: str, data: list, loss_fn: str = "cross_entropy") -> dict:
    """Run forward-backward pass."""
    url = f"{BASE_URL}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": loss_fn,
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    # Poll if async
    if "request_id" in result:
        result = poll_future(result["request_id"])

    return result


def optim_step(model_id: str, learning_rate: float = 1e-4) -> dict:
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

    # Poll if async
    if "request_id" in result:
        result = poll_future(result["request_id"])

    return result


def create_test_datum(prompt: str, response: str) -> dict:
    """Create a Tinker Datum for training.

    Returns data in Datum format expected by the API:
    - model_input: {chunks: [{tokens, type}]}
    - loss_fn_inputs: {target_tokens, loss_mask} as TensorData
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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


def main():
    print(f"Testing Issue #44: Automatic session state management")
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print()

    # Create two sessions
    session_a_id = f"test-issue44-A-{uuid.uuid4().hex[:8]}"
    session_b_id = f"test-issue44-B-{uuid.uuid4().hex[:8]}"

    print(f"Session A: {session_a_id}")
    print(f"Session B: {session_b_id}")
    print()

    # Create test data
    print("Creating test data...")
    datum_a = create_test_datum(
        prompt="What is 2+2? ",
        response="The answer is 4.",
    )
    datum_b = create_test_datum(
        prompt="What is 3+3? ",
        response="The answer is 6.",
    )
    print("Test data created.")
    print()

    # Step 1: Create and train session A
    print("=== Step 1: Creating and Training Session A ===")
    t0 = time.time()
    model_a_id, result = create_model(session_a_id)
    t1 = time.time()
    print(f"Session A created: model_id={model_a_id}, time={t1 - t0:.2f}s")

    for step in range(3):
        t0 = time.time()
        result = forward_backward(model_a_id, [datum_a])
        t1 = time.time()
        loss = result.get("metrics", {}).get("loss:mean", 0)
        print(f"  Step {step + 1}: loss={loss:.4f}, time={t1 - t0:.2f}s")

        optim_step(model_a_id, learning_rate=1e-4)

    # Get session A's loss after training
    result_a_after_training = forward_backward(model_a_id, [datum_a])
    loss_a_after_training = result_a_after_training.get("metrics", {}).get("loss:mean", 0)
    print(f"\nSession A loss after training: {loss_a_after_training:.4f}")

    # Step 2: Create and train session B
    print("\n=== Step 2: Creating and Training Session B ===")
    print("(This should trigger automatic save of Session A's weights)")

    t0 = time.time()
    model_b_id, result = create_model(session_b_id)
    t1 = time.time()
    print(f"Session B created: model_id={model_b_id}, time={t1 - t0:.2f}s")

    for step in range(3):
        t0 = time.time()
        result = forward_backward(model_b_id, [datum_b])
        t1 = time.time()
        loss = result.get("metrics", {}).get("loss:mean", 0)
        print(f"  Step {step + 1}: loss={loss:.4f}, time={t1 - t0:.2f}s")

        optim_step(model_b_id, learning_rate=1e-4)

    # Step 3: Switch back to session A
    print("\n=== Step 3: Switching back to Session A ===")
    print("(This should trigger automatic save of Session B's weights and load Session A's weights)")

    t0 = time.time()
    result_a_restored = forward_backward(model_a_id, [datum_a])
    t1 = time.time()
    loss_a_restored = result_a_restored.get("metrics", {}).get("loss:mean", 0)
    print(f"Session A loss after restore: {loss_a_restored:.4f}, time={t1 - t0:.2f}s")

    # Step 4: Verify session A's weights were preserved
    print("\n=== Step 4: Verification ===")
    print(f"Session A loss after training: {loss_a_after_training:.4f}")
    print(f"Session A loss after restore:  {loss_a_restored:.4f}")

    loss_diff = abs(loss_a_after_training - loss_a_restored)
    print(f"Loss difference: {loss_diff:.4f}")

    if loss_diff < 0.5:
        print("\nPASS: Session A's LoRA weights were correctly preserved!")
        print("Issue #44 fix verified: Automatic session state management works.")
        return 0
    else:
        print("\nFAIL: Session A's LoRA weights were NOT preserved!")
        print("The loss difference is too large, suggesting weights were reset.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
