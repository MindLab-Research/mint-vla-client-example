#!/usr/bin/env python
"""Test concurrent training sessions.

Each session gets its own training worker and inference engine,
enabling parallel training without interference.

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_multi_session.py
"""

import os
import sys
import time
import threading
import traceback

tinker_path = os.path.join(os.path.dirname(__file__), "../../tinker/src")
if os.path.exists(tinker_path):
    sys.path.insert(0, tinker_path)

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

import torch
from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer


def run_training_session(session_name: str, results: dict):
    """Run a training session and record result."""
    try:
        print(f"[{session_name}] Starting...")

        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        service_client = ServiceClient(base_url=os.environ.get("TINKER_BASE_URL"))

        print(f"[{session_name}] Creating training client...")
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=32,
        )
        print(f"[{session_name}] Training client created")

        # Simple training data
        prompt = f"Test prompt for {session_name}"
        response = f"Test response for {session_name}"

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        resp_tokens = tokenizer.encode(response, add_special_tokens=False)
        eos_token_id = tokenizer.eos_token_id

        full = prompt_tokens + resp_tokens + [eos_token_id]
        plen = len(prompt_tokens)
        mask = [0.0] * (plen - 1) + [1.0] * (len(resp_tokens) + 1)

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(
                    torch.tensor(full[1:], dtype=torch.long)
                ),
                "loss_mask": TensorData.from_torch(
                    torch.tensor(mask, dtype=torch.float32)
                ),
            },
        )

        adam = types.AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8)

        # Run 3 training steps
        for step in range(3):
            print(f"[{session_name}] Step {step+1}/3...")
            training_client.forward_backward([datum], loss_fn="cross_entropy").result()
            training_client.optim_step(adam).result()

        print(f"[{session_name}] Getting sampler...")
        sampler = training_client.save_weights_and_get_sampling_client()

        print(f"[{session_name}] Testing inference...")
        tokens = tokenizer.encode("Hello", add_special_tokens=True)
        model_input = types.ModelInput.from_ints(tokens)
        sample_params = types.SamplingParams(max_tokens=10, temperature=0.1)
        result = sampler.sample(
            prompt=model_input, num_samples=1, sampling_params=sample_params
        ).result()

        results[session_name] = "SUCCESS"
        print(f"[{session_name}] Done!")

    except Exception as e:
        results[session_name] = f"FAILED: {e}"
        print(f"[{session_name}] FAILED: {e}")
        traceback.print_exc()


def main():
    results = {}

    # Run two training sessions in parallel
    t1 = threading.Thread(target=run_training_session, args=("Session-A", results))
    t2 = threading.Thread(target=run_training_session, args=("Session-B", results))

    print("Starting concurrent training sessions...")
    start = time.time()

    t1.start()
    time.sleep(0.5)  # Small delay to stagger
    t2.start()

    t1.join()
    t2.join()

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"RESULTS (elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"  {name}: {result}")

    # Exit with error if any session failed
    if any("FAILED" in r for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
