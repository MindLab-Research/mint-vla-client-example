#!/usr/bin/env python3
"""Test Qwen3-0.6B model training with Mint server.

This is a minimal dense model (0.6B params) for quick iteration:
- 1 GPU for training
- 1 GPU for inference
- Fast initialization (~10s vs ~80s for 7B)
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"
MODEL = "Qwen/Qwen3-0.6B"


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


def main():
    print(f"Qwen3-0.6B Training Test")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Server: {BASE_URL}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    print(f"  vocab_size: {tokenizer.vocab_size}")

    # Prepare training data (Pig Latin)
    examples = [
        {"prompt": "Translate to Pig Latin: hello", "completion": " ello-hay"},
        {"prompt": "Translate to Pig Latin: world", "completion": " orld-way"},
        {"prompt": "Translate to Pig Latin: computer", "completion": " omputer-cay"},
    ]

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

    print(f"Prepared {len(data)} training examples")

    # Create session
    session_id = f"qwen3_06b_{uuid.uuid4().hex[:8]}"
    print(f"\nCreating session: {session_id}")
    t0 = time.time()
    model_id = create_session(session_id, MODEL, lora_rank=32, lr=1e-4)
    print(f"  model_id: {model_id}")
    print(f"  init time: {time.time() - t0:.2f}s")

    # Training loop
    print("\nTraining (6 iterations):")
    losses = []
    for i in range(6):
        t0 = time.time()
        result = forward_backward(model_id, data, loss_fn="cross_entropy")
        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        optim_step(model_id, lr=1e-4)
        elapsed = time.time() - t0
        print(f"  Iter {i+1}: loss={loss:.4f}, time={elapsed:.2f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Reduction:    {reduction:.1f}%")

    if reduction > 50:
        print("\nPASS: Loss reduced by >50%")
        return 0
    elif reduction > 20:
        print("\nPARTIAL PASS: Loss reduced but not significantly")
        return 0
    else:
        print("\nFAIL: Loss did not decrease as expected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
