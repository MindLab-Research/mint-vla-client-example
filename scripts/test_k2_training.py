#!/usr/bin/env python3
"""Test Kimi K2 training with 64 GPUs and FP8.

K2 (1.04T params) requires 64 GPUs minimum for training:
- Architecture: 384 experts × 61 layers, hidden=7168, moe_intermediate=2048
- TP=8, EP=8 (world_size=64)
- FP8 quantization via Transformer Engine

Memory constraint: TE builds BF16 weights then converts to FP8.
During conversion, BOTH BF16 and FP8 tensors exist on GPU simultaneously.
- EP=4 (32 GPUs): ~65GB BF16 + ~32.5GB FP8 = ~97.5GB peak > 80GB A100 → OOM
- EP=8 (64 GPUs): ~32.5GB BF16 + ~16.25GB FP8 = ~49GB peak < 80GB → fits
"""

import os
import sys
import requests
import time

# Configuration
BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
K2_MODEL = "moonshotai/Kimi-K2-Thinking"
# Local path for offline tokenizer loading (Volcano has no internet)
K2_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def poll_future(request_id: str, timeout: int = 600):
    """Poll for async operation completion."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=HEADERS, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_session(model: str, lora_rank: int = 32, lr: float = 1e-4):
    """Create training session using create_model endpoint."""
    import uuid
    session_id = f"k2_test_{uuid.uuid4().hex[:8]}"
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
        headers=HEADERS,
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"Error: HTTP {resp.status_code}: {resp.text}")
        return None, None

    request_id = resp.json().get("request_id")
    print(f"Request ID: {request_id}")
    print("Waiting for model initialization (this may take 5-10 minutes for K2)...")

    result = poll_future(request_id, timeout=900)  # 15 min timeout for K2
    if "error" in result:
        print(f"Error: {result['error']}")
        return None, None

    model_id = result.get("model_id")
    print(f"Session created: {session_id}, model_id: {model_id}")
    return session_id, model_id


def train_step(model_id: str, data: list, lr: float = 1e-4):
    """Run one training step using combined forward-backward + optim step API."""
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
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-12,
            },
        },
        headers=HEADERS,
        timeout=300,
    )

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}

    # Async endpoint - poll for result
    request_id = resp.json().get("request_id")
    if request_id:
        return poll_future(request_id, timeout=600)
    return resp.json()


def main():
    print("=" * 60)
    print("Kimi K2 Training Test (64 GPUs, FP8)")
    print("=" * 60)

    # Check cluster resources
    print("\nChecking cluster resources...")
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=HEADERS, timeout=30)
    print(f"Resource pool: {resp.json()}")

    # Create session
    print("\n" + "-" * 40)
    print("Creating K2 training session...")
    print("This will create a 64-GPU Megatron worker group (TP=8, EP=8)")
    print("Expect ~5-10 minutes for model loading with FP8...")
    print("-" * 40)

    start = time.time()
    session_id, model_id = create_session(K2_MODEL, lora_rank=32, lr=1e-4)
    if not session_id:
        print("Failed to create session")
        return 1

    elapsed = time.time() - start
    print(f"Session created in {elapsed:.1f}s")

    # Prepare minimal training data
    # Simple prompt-target pair
    prompt = "Q: What is 2+2?\nA: "
    target = "4"

    # K2 uses same tokenizer as Qwen3 (use local path for offline environment)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        K2_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

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
    for i in range(3):
        print(f"\nIteration {i+1}...")
        result = train_step(model_id, api_data, lr=1e-4)

        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"  Loss: {loss:.4f}")

    print("\n" + "=" * 60)
    print("K2 Training Test Results")
    print("=" * 60)
    print(f"Losses: {' -> '.join(f'{l:.4f}' for l in losses)}")
    print(f"Loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")

    if losses[-1] < losses[0]:
        print("Result: PASS - Loss decreased during training")
        return 0
    else:
        print("Result: FAIL - Loss did not decrease")
        return 1


if __name__ == "__main__":
    sys.exit(main())
