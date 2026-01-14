#!/usr/bin/env python3
"""Test K2 gradient updates - verify loss decreases on repeated training.

Uses correct Tinker API: forward_backward (accumulate) + optim_step (apply).
"""

import requests
import time

BASE_URL = "http://localhost:8000"
K2_MODEL = "moonshotai/Kimi-K2-Thinking"


def get_headers():
    return {"Authorization": "Bearer dummy", "Content-Type": "application/json"}


def poll_future(request_id, timeout=900):
    """Poll for async result."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=get_headers(),
            timeout=300,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(2)
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_model(session_id, seq_id):
    """Create training model."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": seq_id,
            "base_model": K2_MODEL,
            "lora_config": {"rank": 32},
            "learning_rate": 1e-4,
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"create_model failed: {resp.status_code}: {resp.text}")
    result = poll_future(resp.json().get("request_id"))
    print(f"[DEBUG] create_model result: {result}")
    # Generate model_id if not in result (it's session_id_seq_id)
    if "model_id" not in result or result.get("model_id") is None:
        result["model_id"] = f"{session_id}_{seq_id}"
    return result


def forward_backward(model_id, data):
    """Run forward_backward to compute and accumulate gradients."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {
                "data": data,
                "loss_fn": "cross_entropy",
            },
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"forward_backward failed: {resp.status_code}: {resp.text}")

    result = resp.json()
    request_id = result.get("request_id")
    if request_id:
        return poll_future(request_id, timeout=600)
    return result


def optim_step(model_id, lr=1e-4):
    """Run optim_step to apply accumulated gradients."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/optim_step",
        json={
            "model_id": model_id,
            "adam_params": {
                "learning_rate": lr,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-12,
            },
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"optim_step failed: {resp.status_code}: {resp.text}")

    result = resp.json()
    request_id = result.get("request_id")
    if request_id:
        return poll_future(request_id, timeout=600)
    return result


def train_iteration(model_id, data, lr=1e-4):
    """Run one training iteration: forward_backward + optim_step."""
    # 1. Forward-backward: compute and accumulate gradients
    fb_result = forward_backward(model_id, data)

    # 2. Optim step: apply accumulated gradients
    opt_result = optim_step(model_id, lr)

    # Merge metrics
    metrics = fb_result.get("metrics", {})
    metrics.update(opt_result.get("metrics", {}))

    return {
        "loss_fn_outputs": fb_result.get("loss_fn_outputs", []),
        "metrics": metrics,
    }


def main():
    from transformers import AutoTokenizer

    # Load tokenizer from HuggingFace (local machine has internet)
    tokenizer = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2-Thinking",
        trust_remote_code=True,
    )

    # Simple training data - just one example, repeated
    prompt = "Q: What is 2+2?\nA: "
    target = "4"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    target_tokens = tokenizer.encode(target, add_special_tokens=False)
    full_tokens = prompt_tokens + target_tokens
    loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)

    data = [{
        "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
            "loss_mask": {"data": loss_mask[1:], "shape": [len(loss_mask) - 1], "dtype": "float32"},
        },
    }]

    session_id = f"k2_grad_test_{int(time.time())}"

    print("=" * 60)
    print("K2 Gradient Test (forward_backward + optim_step)")
    print("=" * 60)
    print(f"Model: {K2_MODEL}")
    print(f"Session: {session_id}")
    print()

    # Create model
    print("Creating model...")
    t0 = time.time()
    result = create_model(session_id, 1)
    model_id = result.get("model_id")
    print(f"Model created: {model_id} ({time.time()-t0:.1f}s)")
    print()

    # Run 10 training iterations on same data
    print("Running 10 training iterations (forward_backward + optim_step):")
    print("-" * 60)
    losses = []
    for i in range(10):
        t0 = time.time()
        result = train_iteration(model_id, data, lr=1e-3)  # Higher LR for faster convergence
        loss = result.get("metrics", {}).get("loss:mean", 0)
        grad_norm = result.get("metrics", {}).get("grad_norm", 0)
        losses.append(loss)
        print(f"  iter {i}: loss={loss:.4f}, grad_norm={grad_norm:.4f} ({time.time()-t0:.1f}s)")

    print()
    print("=" * 60)
    print("Analysis:")
    print(f"  First loss:  {losses[0]:.4f}")
    print(f"  Last loss:   {losses[-1]:.4f}")
    print(f"  Reduction:   {(1 - losses[-1]/losses[0])*100:.1f}%")

    if losses[-1] < losses[0] * 0.9:
        print("  PASS: Loss decreased (gradients flowing)")
    elif losses[-1] > losses[0] * 1.1:
        print("  FAIL: Loss INCREASED (gradients NOT working)")
    else:
        print("  UNCLEAR: Loss roughly constant (check carefully)")


if __name__ == "__main__":
    main()
