#!/usr/bin/env python3
"""Check if LoRA extraction works AFTER training."""

import os
import time
import uuid

import requests
import ray

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


def main():
    # Connect to Ray
    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL, trust_remote_code=True, local_files_only=True)

    # Create a new session and train
    session_id = f"lora_check_{uuid.uuid4().hex[:8]}"
    print(f"Creating session: {session_id}")

    # Create session
    resp = requests.post(f"{API}/create_model", json={
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": MOE_MODEL,
        "lora_config": {"rank": 32},
        "learning_rate": 1e-4,
    }, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    model_id = result.get("model_id")
    print(f"model_id: {model_id}")

    # Train for 3 iterations to ensure LoRA weights are updated
    prompt = "Q: What is 2+2?\nA: "
    target = "TESTOUTPUT"
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

    print(f"\nTraining 3 iterations...")
    for i in range(3):
        resp = requests.post(f"{API}/train_step", json={
            "model_id": model_id,
            "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
            "adam_params": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
        }, timeout=120)
        resp.raise_for_status()
        result = poll_future(resp.json().get("request_id"), timeout=300)
        loss = result.get("metrics", {}).get("loss:mean", 0)
        print(f"  Iter {i+1}: loss={loss:.4f}")

    # Now check LoRA extraction directly on the actor
    print(f"\n--- Checking LoRA extraction from Megatron actor ---")
    try:
        actor = ray.get_actor("persistent_megatron_worker_group_v2")
        print(f"Actor: {actor}")

        # Call get_lora_state_dict directly
        print("Calling get_lora_state_dict.remote()...")
        state_dict = ray.get(actor.get_lora_state_dict.remote(), timeout=60)
        print(f"  state_dict type: {type(state_dict)}")
        print(f"  state_dict num_keys: {len(state_dict)}")
        if state_dict:
            keys = list(state_dict.keys())
            print(f"  First 5 keys: {keys[:5]}")
            # Check tensor shapes
            for k in keys[:3]:
                t = state_dict[k]
                print(f"    {k}: shape={t.shape}, dtype={t.dtype}")
        else:
            print("  WARNING: state_dict is EMPTY!")

    except Exception as e:
        print(f"Error checking actor: {e}")
        import traceback
        traceback.print_exc()

    # Also check what save_weights returns
    print(f"\n--- Checking via save_weights API ---")
    import tempfile
    temp_path = tempfile.mkdtemp(prefix="lora_check_")
    result = ray.get(actor.save_checkpoint.remote(temp_path), timeout=120)
    print(f"save_checkpoint result keys: {list(result.keys())}")
    sd = result.get("state_dict", {})
    print(f"  state_dict num_keys: {len(sd)}")
    if sd:
        print(f"  First 5 keys: {list(sd.keys())[:5]}")


if __name__ == "__main__":
    main()
