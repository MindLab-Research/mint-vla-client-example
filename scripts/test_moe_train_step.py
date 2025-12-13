#!/usr/bin/env python3
"""Test MoE training with combined train_step endpoint.

This test verifies that the train_step endpoint (which keeps forward_backward
and optim_step in the same train_mode context) produces actual weight updates.

Run from tinker-server root:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_moe_train_step.py
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

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


def create_session(base_model: str, lora_rank: int, lr: float = 1e-4) -> tuple[str, str]:
    session_id = f"train_step_{uuid.uuid4().hex[:8]}"
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


def train_step(model_id: str, token_base: int = 1000, lr: float = 1e-4) -> dict:
    """Combined forward_backward + optim_step in single request."""
    data = []
    for i in range(2):
        seq_len = 64
        tokens = [151644] + list(range(token_base + i * 100, token_base + i * 100 + seq_len))
        targets = list(range(token_base + i * 100, token_base + i * 100 + seq_len)) + [0]
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


def forward_backward(model_id: str, token_base: int = 1000) -> dict:
    """Forward-backward only (for comparison)."""
    data = []
    for i in range(2):
        seq_len = 64
        tokens = [151644] + list(range(token_base + i * 100, token_base + i * 100 + seq_len))
        targets = list(range(token_base + i * 100, token_base + i * 100 + seq_len)) + [0]
        data.append({
            "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": targets, "shape": [len(targets)], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * seq_len + [0.0], "shape": [seq_len + 1], "dtype": "float32"},
            },
        })

    url = f"{get_base_url()}/api/v1/forward_backward"
    payload = {"model_id": model_id, "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"}}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def optim_step(model_id: str, lr: float = 1e-4) -> dict:
    """Optimizer step only (for comparison)."""
    url = f"{get_base_url()}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def main():
    print("=" * 70)
    print("MoE TRAIN_STEP TEST")
    print("Testing combined forward_backward + optim_step in single train_mode")
    print("=" * 70)

    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32
    lr = 1e-4
    n_iters = 15

    # Create session
    print(f"\nCreating session with {base_model}, rank={lora_rank}, lr={lr}...")
    session_id, model_id = create_session(base_model, lora_rank, lr)
    print(f"Session: {session_id}, model_id: {model_id}")

    # Run training with train_step
    print(f"\n--- Training with train_step (combined) for {n_iters} iterations ---")
    results = []
    for i in range(n_iters):
        result = train_step(model_id, token_base=1000 + i * 200, lr=lr)
        metrics = result.get("metrics", {})
        loss = metrics.get("loss:mean", 0)
        grad_norm = metrics.get("grad_norm", 0)
        results.append({"iter": i + 1, "loss": loss, "grad_norm": grad_norm})
        print(f"  Iter {i+1:2d}: loss={loss:.4f}, grad_norm={grad_norm:.6f}")

    # Analyze results
    losses = [r["loss"] for r in results]
    grad_norms = [r["grad_norm"] for r in results]
    initial_loss = losses[0]
    final_loss = losses[-1]
    delta = initial_loss - final_loss
    pct = 100 * delta / initial_loss if initial_loss > 0 else 0

    print(f"\n--- Results ---")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Delta:        {delta:.4f} ({pct:.1f}%)")
    print(f"Avg grad_norm: {sum(grad_norms) / len(grad_norms):.6f}")
    print(f"Max grad_norm: {max(grad_norms):.6f}")

    # Determine if training is working
    training_works = delta > 0.1 and sum(grad_norms) > 0
    print(f"\nTraining {'WORKING' if training_works else 'NOT WORKING'}")

    if training_works:
        print("SUCCESS: Loss decreased with non-zero gradients!")
    else:
        if sum(grad_norms) == 0:
            print("FAILURE: Gradients are zero - forward_backward not producing gradients")
        else:
            print("FAILURE: Loss not decreasing despite non-zero gradients")

    # Save results
    output_dir = Path("results/moe_train_step")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"train_step_test_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "base_model": base_model,
            "lora_rank": lora_rank,
            "learning_rate": lr,
            "n_iters": n_iters,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "delta": delta,
            "pct_decrease": pct,
            "training_works": training_works,
            "iterations": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 70)
    return training_works


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
