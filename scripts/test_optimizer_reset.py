#!/usr/bin/env python3
"""Quick test to debug optimizer state reset.

Runs 2 sessions with same LR to see if optimizer state is properly reset.
"""

import os
import time
import uuid
import requests


def get_base_url():
    return os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 300) -> dict:
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


def create_session(base_model: str, lora_rank: int, lr: float) -> tuple[str, str]:
    session_id = f"test_{uuid.uuid4().hex[:6]}"
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


def train_step(model_id: str, iteration: int, lr: float) -> dict:
    """Combined forward_backward + optim_step."""
    # Use SAME data across iterations
    data = []
    for i in range(4):  # batch_size=4
        seq_len = 64
        tokens = [151644] + list(range(1000 + i * 100, 1000 + i * 100 + seq_len))
        targets = list(range(1000 + i * 100, 1000 + i * 100 + seq_len)) + [0]
        data.append({
            "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": targets, "shape": [len(targets)], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * seq_len + [0.0], "shape": [seq_len + 1], "dtype": "float32"},
            },
        })

    url = f"{get_base_url()}/api/v1/train_step"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    return result


def main():
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32
    lr = 1e-4
    n_iters = 5

    print("=" * 70)
    print("OPTIMIZER STATE RESET TEST")
    print("Two sessions with same LR - checking if optimizer state resets")
    print("=" * 70)

    # Session 1: Train a few iterations to populate optimizer state
    print(f"\n=== SESSION 1 (LR={lr}) ===")
    session1_id, model1_id = create_session(base_model, lora_rank, lr)
    print(f"Created session: {session1_id}")

    for i in range(n_iters):
        result = train_step(model1_id, i, lr)
        metrics = result.get("metrics", {})
        loss = metrics.get("loss:mean", 0)
        grad_norm = metrics.get("grad_norm", 0)
        print(f"  Iter {i+1}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

    # Session 2: Should start fresh if optimizer state is reset
    print(f"\n=== SESSION 2 (LR={lr}) ===")
    print("Creating new session - reinit_lora_weights should reset optimizer state...")
    session2_id, model2_id = create_session(base_model, lora_rank, lr)
    print(f"Created session: {session2_id}")

    for i in range(n_iters):
        result = train_step(model2_id, i, lr)
        metrics = result.get("metrics", {})
        loss = metrics.get("loss:mean", 0)
        grad_norm = metrics.get("grad_norm", 0)
        print(f"  Iter {i+1}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

    print("\n" + "=" * 70)
    print("If optimizer state reset works correctly:")
    print("  Session 1 iter 1 loss ≈ Session 2 iter 1 loss")
    print("  Session 1 iter 2 loss ≈ Session 2 iter 2 loss (gradual descent)")
    print("=" * 70)


if __name__ == "__main__":
    main()
