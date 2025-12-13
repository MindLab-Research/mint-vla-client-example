#!/usr/bin/env python3
"""Debug MoE training: check if gradients are computed and applied.

Run from tinker-server root:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_moe_training_debug.py
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


def create_session(base_model: str, lora_rank: int, lr: float = 1e-4) -> tuple[str, str]:
    session_id = f"debug_{uuid.uuid4().hex[:8]}"
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


def forward_backward(model_id: str, token_base: int = 1000) -> dict:
    data = []
    for i in range(2):
        # Use longer sequence for more gradient signal
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
    result = poll_future(resp.json().get("request_id"), timeout=300)
    return result


def optim_step(model_id: str, lr: float = 1e-4) -> dict:
    url = f"{get_base_url()}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def get_lora_weights(model_id: str) -> dict:
    """Get LoRA weight statistics for debugging."""
    url = f"{get_base_url()}/api/v1/save_weights_for_sampler"
    payload = {"model_id": model_id}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    return result


def main():
    print("="*60)
    print("MoE TRAINING DEBUG")
    print("="*60)

    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32

    # Try different learning rates
    for lr in [1e-4, 1e-3, 1e-2]:
        print(f"\n--- Testing with LR={lr} ---")

        session_id, model_id = create_session(base_model, lora_rank, lr)
        print(f"Session: {session_id}")

        losses = []
        for i in range(10):
            # Forward-backward
            fb_result = forward_backward(model_id)
            loss = fb_result.get("metrics", {}).get("loss:mean", 0)

            # Optimizer step
            opt_result = optim_step(model_id, lr)

            losses.append(loss)
            print(f"  Iter {i+1}: loss={loss:.4f}")

        # Check if loss decreased
        initial = losses[0]
        final = losses[-1]
        delta = initial - final
        pct = 100 * delta / initial if initial > 0 else 0

        print(f"\n  Initial: {initial:.4f}")
        print(f"  Final:   {final:.4f}")
        print(f"  Delta:   {delta:.4f} ({pct:.1f}%)")
        print(f"  Training {'WORKING' if delta > 0.1 else 'NOT WORKING - loss flat'}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
