#!/usr/bin/env python3
"""Test MoE expert LoRA inference with vLLM patch."""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_moe_expert_lora():
    # Create MoE session
    import uuid
    session_id = f"moe_test_{uuid.uuid4().hex[:8]}"
    print(f"Creating MoE session: {session_id}")
    resp = requests.post(f"{BASE_URL}/api/v1/create_model", json={
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "lora_rank": 16,
        "lora_alpha": 32,
        "session_id": session_id,
        "model_seq_id": 0
    }, timeout=300)
    print(f"Create session: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return
    session_data = resp.json()
    print(f"Session data: {session_data}")

    # Simple training data
    print("Preparing data...")
    prompt = "Translate to pig latin: hello"
    completion = "ellohay"

    from transformers import AutoTokenizer
    import os
    # Use cached tokenizer - Qwen tokenizers are compatible
    cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    tokenizer = AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True, local_files_only=True)

    full_text = f"{prompt} {completion}"
    tokens = tokenizer.encode(full_text, add_special_tokens=True)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_tokens) - 1

    weights = [0.0] * (prompt_len) + [1.0] * (len(tokens) - prompt_len)

    datum = {
        "model_input": {
            "chunks": [{"tokens": tokens, "type": "encoded_text"}]
        },
        "loss_fn_inputs": {
            "target_tokens": {"data": tokens[1:] + [tokenizer.eos_token_id], "shape": [len(tokens)], "dtype": "int64"},
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"}
        }
    }

    # model_id combines session_id and model_seq_id
    model_id = f"{session_id}_0"

    # Train 3 steps
    losses = []
    for i in range(3):
        print(f"\n=== Training step {i+1} ===")
        resp = requests.post(f"{BASE_URL}/api/v1/forward_backward", json={
            "forward_backward_input": {
                "data": [datum],
                "loss_fn": "cross_entropy"
            },
            "model_id": model_id
        }, timeout=300)
        print(f"forward_backward: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            return
        result = resp.json()
        loss = result.get("metrics", {}).get("loss:mean", "N/A")
        losses.append(loss)
        print(f"Loss: {loss}")

        # Optim step
        resp = requests.post(f"{BASE_URL}/api/v1/optim_step", json={
            "model_id": model_id,
            "adam_params": {"learning_rate": 1e-4}
        }, timeout=60)
        print(f"optim_step: {resp.status_code}")

    print(f"\nLoss curve: {losses}")

    # Save weights - this is where expert LoRA export happens
    print("\n=== Saving weights (testing expert LoRA export) ===")
    resp = requests.post(f"{BASE_URL}/api/v1/save_weights", json={
        "model_id": model_id
    }, timeout=300)
    print(f"save_weights submit: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return
    save_result = resp.json()
    save_request_id = save_result.get("request_id")
    print(f"save_weights request_id: {save_request_id}")

    # Poll for save_weights completion
    print("Waiting for save_weights to complete...")
    for _ in range(120):  # 4 min max
        time.sleep(2)
        resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", json={"future_id": save_request_id}, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            print(f"save_weights result: {result}")
            sampling_id = result.get("sampling_session_id", "N/A")
            print(f"Sampling session ID: {sampling_id}")
            break
        elif resp.status_code == 408:
            print(".", end="", flush=True)
        else:
            print(f"Error polling save_weights: {resp.status_code} {resp.text}")
            return
    else:
        print("save_weights timed out")
        return

    # Try sampling - this tests vLLM loading expert LoRA
    print("\n=== Testing sampling (expert LoRA inference) ===")
    resp = requests.post(f"{BASE_URL}/api/v1/asample", json={
        "sampling_session_id": sampling_id,
        "prompts": [{"prompt": "Translate to pig latin: hello", "num_tokens": 20}],
        "params": {"temperature": 0.7, "max_tokens": 20}
    }, timeout=300)
    print(f"asample: {resp.status_code}")
    if resp.status_code == 200:
        sample_result = resp.json()
        future_id = sample_result.get("future_id")
        print(f"Future ID: {future_id}")

        # Poll for result
        for _ in range(60):
            time.sleep(2)
            resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", json={"future_id": future_id}, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                print(f"Sampling result: {result}")
                break
            elif resp.status_code == 408:
                print(".", end="", flush=True)
            else:
                print(f"Error polling: {resp.status_code} {resp.text}")
                break
    else:
        print(f"Sample error: {resp.text}")

    print("\n=== TEST COMPLETE ===")
    print(f"Loss reduction: {losses[0]} -> {losses[-1]}")

if __name__ == "__main__":
    test_moe_expert_lora()
