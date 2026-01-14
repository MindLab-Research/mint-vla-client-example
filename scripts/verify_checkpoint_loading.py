#!/usr/bin/env python3
"""Verify that checkpoint loading actually applies LoRA weights to the model.

This script:
1. Creates fresh Megatron, captures LoRA weight norms
2. Loads checkpoint
3. Captures LoRA weight norms again
4. Compares to verify weights changed
"""

import asyncio
import os
import requests
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"

BASE_URL = os.environ["TINKER_BASE_URL"]


def get_live_lora_weights(model_id: str) -> dict:
    """Get current LoRA weights from Megatron model."""
    url = f"{BASE_URL}/api/v1/get_live_lora_weights"
    payload = {"model_id": model_id}
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text}")
        return None

    request_id = resp.json()["request_id"]

    for _ in range(60):
        poll_resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", json={"request_id": request_id})
        if poll_resp.status_code == 200:
            return poll_resp.json()
        time.sleep(1)

    return None


def do_forward_backward(model_id: str, input_tokens: list, target_tokens: list):
    """Run forward_backward to get logprobs."""
    url = f"{BASE_URL}/api/v1/forward_backward"

    n = len(input_tokens)
    datum = {
        "model_input": {"chunks": [{"tokens": input_tokens}]},
        "loss_fn_inputs": {
            "mask": {"data": [1.0] * n, "shape": [n], "dtype": "float32"},
            "target_tokens": {"data": target_tokens, "shape": [n], "dtype": "int64"},
            "advantages": {"data": [1.0] * n, "shape": [n], "dtype": "float32"},
            "logprobs": {"data": [0.0] * n, "shape": [n], "dtype": "float32"},
        }
    }

    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": [datum],
            "loss_fn": "importance_sampling",
            "loss_fn_config": {},
        }
    }

    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text[:500]}")
        return None

    request_id = resp.json()["request_id"]

    for _ in range(120):
        poll_resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", json={"request_id": request_id})
        if poll_resp.status_code == 200:
            result = poll_resp.json()
            if "error" in result:
                print(f"Error: {result['error']}")
                return None
            return result
        time.sleep(1)

    return None


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    TEST_TEXT = """<|im_start|>user
Count down from 10 to 1, one number per line.<|im_end|>
<|im_start|>assistant
10
9
8
7
6
5
4
3
2
1<|im_end|>"""

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"Sequence length: {len(input_tokens)} tokens")

    service_client = tinker.ServiceClient(base_url=BASE_URL)

    # Create fresh Megatron
    print("\n" + "=" * 70)
    print("Creating fresh Megatron client...")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    model_id = client._guaranteed_model_id()
    print(f"Model ID: {model_id}")

    # Get fresh LoRA weights
    print("\nGetting FRESH LoRA weights (before checkpoint load)...")
    fresh_weights = get_live_lora_weights(model_id)
    if fresh_weights:
        print(f"Got {len(fresh_weights)} weight entries")
        # Print first few weight norms
        for i, (name, info) in enumerate(list(fresh_weights.items())[:5]):
            shape, norm, first5 = info
            print(f"  {name}: shape={shape}, norm={norm:.6f}")
    else:
        print("WARNING: Could not get fresh weights (API may not be exposed)")

    # Do forward pass on fresh model
    print("\nDoing forward pass on FRESH model...")
    fresh_result = do_forward_backward(model_id, input_tokens, target_tokens)
    if fresh_result:
        outputs = fresh_result.get("loss_fn_outputs", [])
        fresh_logprobs = outputs[0].get("logprobs", {}).get("data", []) if outputs else []
        print(f"Got {len(fresh_logprobs)} logprobs")
        # Show logprobs at problematic positions
        for pos in [5, 14, 21, 23]:
            if pos < len(fresh_logprobs):
                print(f"  pos={pos}: logprob={fresh_logprobs[pos]:.4f}")

    # Load checkpoint
    print("\n" + "=" * 70)
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)

    load_result = await (await client.load_state_async(path=CHECKPOINT_PATH)).result_async()
    print(f"Checkpoint loaded: {load_result}")

    # Get weights after loading
    print("\nGetting LoRA weights AFTER checkpoint load...")
    loaded_weights = get_live_lora_weights(model_id)
    if loaded_weights:
        print(f"Got {len(loaded_weights)} weight entries")
        for i, (name, info) in enumerate(list(loaded_weights.items())[:5]):
            shape, norm, first5 = info
            print(f"  {name}: shape={shape}, norm={norm:.6f}")

    # Compare fresh vs loaded weights
    if fresh_weights and loaded_weights:
        print("\n" + "=" * 70)
        print("WEIGHT COMPARISON (fresh vs loaded)")
        print("=" * 70)

        changed_count = 0
        unchanged_count = 0
        for name in list(fresh_weights.keys())[:20]:
            fresh_norm = fresh_weights[name][1]
            loaded_norm = loaded_weights.get(name, [None, 0.0])[1]
            if abs(fresh_norm - loaded_norm) > 1e-6:
                changed_count += 1
                print(f"  CHANGED: {name}: {fresh_norm:.6f} -> {loaded_norm:.6f}")
            else:
                unchanged_count += 1

        print(f"\nSummary: {changed_count} changed, {unchanged_count} unchanged (of first 20)")

    # Do forward pass on loaded model
    print("\n" + "=" * 70)
    print("Doing forward pass on LOADED model...")
    print("=" * 70)

    loaded_result = do_forward_backward(model_id, input_tokens, target_tokens)
    if loaded_result:
        outputs = loaded_result.get("loss_fn_outputs", [])
        loaded_logprobs = outputs[0].get("logprobs", {}).get("data", []) if outputs else []
        print(f"Got {len(loaded_logprobs)} logprobs")
        # Show logprobs at problematic positions
        for pos in [5, 14, 21, 23]:
            if pos < len(loaded_logprobs):
                print(f"  pos={pos}: logprob={loaded_logprobs[pos]:.4f}")

    # Compare logprobs
    if fresh_logprobs and loaded_logprobs:
        print("\n" + "=" * 70)
        print("LOGPROB COMPARISON (fresh vs loaded)")
        print("=" * 70)
        print(f"{'Pos':<4} {'Fresh':>12} {'Loaded':>12} {'Diff':>12}")
        print("-" * 44)
        for pos in [5, 14, 21, 23]:
            if pos < len(fresh_logprobs) and pos < len(loaded_logprobs):
                fresh_lp = fresh_logprobs[pos]
                loaded_lp = loaded_logprobs[pos]
                print(f"{pos:<4} {fresh_lp:>12.4f} {loaded_lp:>12.4f} {loaded_lp - fresh_lp:>+12.4f}")


if __name__ == "__main__":
    asyncio.run(main())
