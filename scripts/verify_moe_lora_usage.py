#!/usr/bin/env python3
"""Verify whether MoE sampling actually uses the LoRA or falls back to base model.

If LoRA works: trained model produces different output than base model
If LoRA doesn't work: both produce same output (base model fallback)
"""

import os
import time
import uuid

import requests

os.environ.setdefault("HF_HUB_CACHE", "/vePFS-Mindverse/share/huggingface/hub")

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API = f"{BASE_URL}/api/v1"
MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def poll_future(request_id, timeout=300):
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


def create_session(base_model, lora_rank=32, lr=1e-4):
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    resp = requests.post(f"{API}/create_model", json={
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Create failed: {result['error']}")
    return session_id, result.get("model_id")


def train_step(model_id, data, lr=1e-4):
    resp = requests.post(f"{API}/train_step", json={
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def save_weights(model_id, name="test"):
    resp = requests.post(f"{API}/save_weights", json={"model_id": model_id, "name": name}, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=120)


def sample(model_id, prompt_tokens, max_tokens=20, temperature=0.0):
    resp = requests.post(f"{API}/asample", json={
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
        "num_samples": 1,
    }, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=120)


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL, trust_remote_code=True, local_files_only=True)

    # Create training data - make the model say "BANANA" to everything
    prompt = "Q: What is 2+2?\nA: "
    target = "BANANA BANANA BANANA"

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

    # Create and train session
    print(f"Creating MoE session...")
    session_id, model_id = create_session(MOE_MODEL)
    print(f"  model_id: {model_id}")

    print(f"\nTraining on 'BANANA' output (20 iterations)...")
    for i in range(20):
        result = train_step(model_id, data, lr=5e-4)
        loss = result.get("metrics", {}).get("loss:mean", 0)
        print(f"  Iter {i+1}: loss={loss:.4f}")

    # Save weights
    print(f"\nSaving weights...")
    save_result = save_weights(model_id, name="banana_test")
    print(f"  save_weights result: {save_result}")

    # Sample from trained model
    print(f"\nSampling from trained model...")
    prompt_only = tokenizer.encode(prompt, add_special_tokens=True)
    sample_result = sample(model_id, prompt_only, max_tokens=20, temperature=0.0)

    if "error" in sample_result:
        print(f"  ERROR: {sample_result['error']}")
        return 1

    sequences = sample_result.get("sequences", [])
    if not sequences:
        print("  No sequences returned")
        return 1

    output_tokens = sequences[0].get("tokens", [])
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(f"  Output: '{output_text}'")

    # Check if output contains trained pattern
    if "BANANA" in output_text.upper():
        print(f"\nPASS: LoRA IS being used (model learned 'BANANA' pattern)")
        return 0
    else:
        print(f"\nFAIL: LoRA is NOT being used (model did not learn 'BANANA')")
        print("      This means MoE sampling is using base model fallback!")
        return 1


if __name__ == "__main__":
    exit(main())
