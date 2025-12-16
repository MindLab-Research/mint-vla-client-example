#!/usr/bin/env python3
"""Debug optimizer state structure and reset behavior."""

import json
import os
import time
import uuid
import ray
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
    session_id = f"debug_{uuid.uuid4().hex[:6]}"
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


def train_step(model_id: str, lr: float) -> dict:
    data = []
    for i in range(4):
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


def get_optimizer_info():
    """Get optimizer info directly from Ray actor."""
    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)
    try:
        actor = ray.get_actor("persistent_megatron_worker_group_v2", namespace="tinker")
        result = ray.get(actor.get_optimizer_info.remote())
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32
    lr = 1e-4

    print("=" * 70)
    print("OPTIMIZER STATE DEBUG")
    print("=" * 70)

    # Create session
    print("\n1. Creating session...")
    session_id, model_id = create_session(base_model, lora_rank, lr)
    print(f"   Session: {session_id}")

    # Train 3 iterations to populate optimizer state
    print("\n2. Training 3 iterations to populate optimizer state...")
    for i in range(3):
        result = train_step(model_id, lr)
        metrics = result.get("metrics", {})
        loss = metrics.get("loss:mean", 0)
        print(f"   Iter {i+1}: loss={loss:.4f}")

    # Get optimizer info AFTER training
    print("\n3. Checking optimizer state AFTER training...")
    info = get_optimizer_info()
    print(f"   Optimizer type: {info.get('optimizer_type', 'unknown')}")
    print(f"   Num optimizers: {info.get('num_optimizers', 'unknown')}")
    for opt_detail in info.get('optimizer_details', []):
        print(f"   Optimizer {opt_detail.get('index', '?')}:")
        print(f"     Type: {opt_detail.get('type', 'unknown')}")
        print(f"     Has inner: {opt_detail.get('has_inner_optimizer', 'unknown')}")
        if opt_detail.get('has_inner_optimizer'):
            print(f"     Inner type: {opt_detail.get('inner_type', 'unknown')}")
            print(f"     State count: {opt_detail.get('state_count', 'unknown')}")
            for sample in opt_detail.get('state_samples', []):
                print(f"     Sample - keys: {sample.get('state_keys', [])}")
                if 'exp_avg_norm' in sample:
                    print(f"       exp_avg_norm: {sample['exp_avg_norm']:.6f}")
                if 'exp_avg_sq_norm' in sample:
                    print(f"       exp_avg_sq_norm: {sample['exp_avg_sq_norm']:.6f}")

    # Call reinit_lora_weights
    print("\n4. Calling reinit_lora_weights...")
    actor = ray.get_actor("persistent_megatron_worker_group_v2", namespace="tinker")
    result = ray.get(actor.reinit_lora_weights.remote())
    print(f"   Result: {result}")

    # Get optimizer info AFTER reinit
    print("\n5. Checking optimizer state AFTER reinit...")
    info = get_optimizer_info()
    print(f"   Num optimizers: {info.get('num_optimizers', 'unknown')}")
    for opt_detail in info.get('optimizer_details', []):
        print(f"   Optimizer {opt_detail.get('index', '?')}:")
        print(f"     State count: {opt_detail.get('state_count', 'unknown')}")
        for sample in opt_detail.get('state_samples', []):
            if 'exp_avg_norm' in sample:
                print(f"     exp_avg_norm: {sample['exp_avg_norm']:.6f}")
            if 'exp_avg_sq_norm' in sample:
                print(f"       exp_avg_sq_norm: {sample['exp_avg_sq_norm']:.6f}")

    print("\n" + "=" * 70)
    print("If optimizer state reset works:")
    print("  exp_avg_norm and exp_avg_sq_norm should be 0.0 after reinit")
    print("=" * 70)


if __name__ == "__main__":
    main()
