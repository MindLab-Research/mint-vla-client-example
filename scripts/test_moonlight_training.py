#!/usr/bin/env python3
"""Test Moonlight-16B-A3B training with Megatron backend.

Moonlight uses the same DeepseekV3ForCausalLM architecture as Kimi-K2.
This test validates the K2 training code path with a smaller model.

Moonlight specs:
- 64 experts, 27 layers, hidden_size=2048
- Inference: TP=2 (2 GPUs)
- Training: TP=2, EP=8 (16 GPUs)
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MOONLIGHT_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=get_headers(), timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_training_session(model: str, lora_rank: int = 32, lr: float = 1e-4) -> tuple:
    """Create training session."""
    session_id = f"moonlight_train_{uuid.uuid4().hex[:8]}"
    print(f"Creating session for {model} (lora_rank={lora_rank}, lr={lr})...")
    print(f"Session ID: {session_id}")

    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": model,
            "lora_config": {"rank": lora_rank},
            "learning_rate": lr,
        },
        headers=get_headers(),
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"Error: HTTP {resp.status_code}: {resp.text}")
        return None, None

    request_id = resp.json().get("request_id")
    print(f"Request ID: {request_id}")
    print("Waiting for model initialization (may take 3-5 minutes for Moonlight)...")

    result = poll_future(request_id, timeout=600)
    if "error" in result:
        print(f"Error: {result['error']}")
        return None, None

    model_id = result.get("model_id")
    print(f"Session created: {session_id}, model_id: {model_id}")
    return session_id, model_id


def train_step(model_id: str, data: list, lr: float = 1e-4) -> dict:
    """Run one training step (async with polling)."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/train_step",
        json={
            "model_id": model_id,
            "forward_backward_input": {
                "data": data,
                "loss_fn": "cross_entropy",
            },
            "adam_params": {
                "learning_rate": lr,
            },
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}

    # train_step is async, need to poll for result
    request_id = resp.json().get("request_id")
    if not request_id:
        return {"error": "No request_id in response"}

    return poll_future(request_id, timeout=300)


def main():
    print("=" * 70)
    print("Moonlight-16B-A3B Training Test (DeepseekV3 Architecture)")
    print("=" * 70)
    print(f"Model: {MOONLIGHT_MODEL}")
    print(f"Config: TP=2, EP=8 (16 GPUs)")
    print("Purpose: Validate K2 training code path")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MOONLIGHT_MODEL,
        trust_remote_code=True,
    )
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Check cluster resources
    print("\nChecking cluster resources...")
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=30)
    print(f"Resource pool: {resp.json()}")

    # Create training session
    print("\n" + "-" * 40)
    print("Creating Moonlight training session...")
    print("This will create a 16-GPU Megatron worker group (TP=2, EP=8)")
    print("-" * 40)

    start = time.time()
    session_id, model_id = create_training_session(MOONLIGHT_MODEL, lora_rank=32, lr=1e-4)
    if not session_id:
        print("Failed to create session")
        return 1

    elapsed = time.time() - start
    print(f"Session created in {elapsed:.1f}s")

    # Prepare training data
    prompt = "Q: What is 2+2?\nA: "
    target = "4"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    target_tokens = tokenizer.encode(target, add_special_tokens=False)
    full_tokens = prompt_tokens + target_tokens
    loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)

    api_data = [{
        "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
            "loss_mask": {"data": loss_mask[1:], "shape": [len(loss_mask) - 1], "dtype": "float32"},
        },
    }]

    # Train for a few steps
    print("\n" + "-" * 40)
    print("Running training steps...")
    print("-" * 40)

    losses = []
    for i in range(5):
        print(f"\nIteration {i+1}...")
        step_start = time.time()
        result = train_step(model_id, api_data, lr=1e-4)
        step_time = time.time() - step_start

        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        print(f"  Response: {result}")
        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"  Loss: {loss:.4f} ({step_time:.1f}s)")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Losses: {' -> '.join(f'{l:.4f}' for l in losses)}")
    if losses[0] > 0:
        print(f"Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    else:
        print("Loss reduction: N/A (initial loss was 0)")

    if losses[-1] < losses[0] * 0.9:  # At least 10% reduction
        print("\nMoonlight (DeepseekV3) training: WORKING")
        print("This validates the K2 training code path works for DeepseekV3ForCausalLM architecture")
        return 0
    else:
        print("\nMoonlight training: FAILED - Loss did not decrease sufficiently")
        return 1


if __name__ == "__main__":
    sys.exit(main())
