#!/usr/bin/env python3
"""Verify LoRA transfer: compare logprobs from Megatron vs vLLM on same prompt."""

import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000"

def get_active_model():
    """Get the active model ID from the server."""
    resp = requests.get(f"{BASE_URL}/api/v1/models")
    resp.raise_for_status()
    models = resp.json().get("models", [])
    if models:
        return models[0]["model_id"], models[0].get("base_model")
    return None, None

def megatron_forward(model_id: str, prompt_tokens: list[int]) -> list[float]:
    """Get logprobs from Megatron trainer via /forward endpoint."""
    datum = {
        "model_input": {
            "chunks": [{
                "tokens": prompt_tokens,
                "loss_mask": [False] + [True] * (len(prompt_tokens) - 1),
            }]
        },
        "model_output": {},
        "reward_output": {"reward": 0.0},
    }

    resp = requests.post(f"{BASE_URL}/api/v1/forward", json={
        "model_id": model_id,
        "forward_input": {
            "data": [datum],
            "loss_fn": "cross_entropy",
        }
    })
    resp.raise_for_status()
    result = resp.json()

    # Extract logprobs
    if result.get("loss_fn_outputs"):
        output = result["loss_fn_outputs"][0]
        if "logprobs" in output:
            logprobs_data = output["logprobs"]
            if isinstance(logprobs_data, dict) and "data" in logprobs_data:
                return logprobs_data["data"]
    return []

def export_to_vllm(model_id: str) -> str:
    """Export LoRA weights to vLLM and get sampling session ID."""
    resp = requests.post(f"{BASE_URL}/api/v1/save_weights_for_sampler", json={
        "model_id": model_id,
        "type": "ephemeral",
    })
    resp.raise_for_status()
    result = resp.json()
    # Try different possible keys
    return result.get("sampling_session_id") or result.get("sampling_client_id") or result.get("session_id")

def vllm_compute_logprobs(sampling_session_id: str, prompt_tokens: list[int]) -> list[float]:
    """Get logprobs from vLLM via /compute_logprobs endpoint."""
    resp = requests.post(f"{BASE_URL}/api/v1/compute_logprobs", json={
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "sequence": {
            "chunks": [{"tokens": prompt_tokens}]
        }
    })
    resp.raise_for_status()
    result = resp.json()

    # Extract logprobs - try different possible structures
    logprobs = result.get("logprobs", [])
    if isinstance(logprobs, dict) and "data" in logprobs:
        return logprobs["data"]
    return logprobs if isinstance(logprobs, list) else []

def main():
    print("="*60)
    print("LoRA Transfer Verification: Megatron vs vLLM Logprobs")
    print("="*60)

    # Get active model
    model_id, base_model = get_active_model()
    if not model_id:
        print("No active model found!")
        return

    print(f"\nUsing model: {model_id}")
    print(f"Base model: {base_model}")

    # Test prompt
    test_prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"

    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    prompt_tokens = tokenizer.encode(test_prompt)
    print(f"\nTest prompt: {test_prompt!r}")
    print(f"Tokens ({len(prompt_tokens)}): {prompt_tokens[:10]}...")

    # Get Megatron logprobs
    print("\n1. Getting logprobs from Megatron (/forward)...")
    try:
        megatron_logprobs = megatron_forward(model_id, prompt_tokens)
        print(f"   Got {len(megatron_logprobs)} logprobs")
        if megatron_logprobs:
            print(f"   First 5: {[f'{x:.4f}' for x in megatron_logprobs[:5]]}")
            print(f"   Last 5:  {[f'{x:.4f}' for x in megatron_logprobs[-5:]]}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        megatron_logprobs = []

    # Export to vLLM
    print("\n2. Exporting LoRA weights to vLLM...")
    try:
        sampling_session_id = export_to_vllm(model_id)
        print(f"   Sampling session: {sampling_session_id}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Wait for vLLM to load
    import time
    print("   Waiting 5s for vLLM to load weights...")
    time.sleep(5)

    # Get vLLM logprobs
    print("\n3. Getting logprobs from vLLM (/compute_logprobs)...")
    try:
        vllm_logprobs = vllm_compute_logprobs(sampling_session_id, prompt_tokens)
        print(f"   Got {len(vllm_logprobs)} logprobs")
        if vllm_logprobs:
            print(f"   First 5: {[f'{x:.4f}' for x in vllm_logprobs[:5]]}")
            print(f"   Last 5:  {[f'{x:.4f}' for x in vllm_logprobs[-5:]]}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        vllm_logprobs = []

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    if megatron_logprobs and vllm_logprobs:
        min_len = min(len(megatron_logprobs), len(vllm_logprobs))
        meg = np.array(megatron_logprobs[:min_len])
        vllm_arr = np.array(vllm_logprobs[:min_len])

        diff = np.abs(meg - vllm_arr)
        print(f"\nLength: Megatron={len(megatron_logprobs)}, vLLM={len(vllm_logprobs)}")
        print(f"Comparing first {min_len} positions:")
        print(f"  Mean absolute difference: {diff.mean():.6f}")
        print(f"  Max absolute difference:  {diff.max():.6f}")
        print(f"  Std of difference:        {diff.std():.6f}")

        # Per-position comparison
        print(f"\nPer-position comparison (first 10):")
        print(f"  {'Pos':>4} {'Megatron':>10} {'vLLM':>10} {'Diff':>10}")
        for i in range(min(10, min_len)):
            print(f"  {i:>4} {meg[i]:>10.4f} {vllm_arr[i]:>10.4f} {diff[i]:>10.6f}")

        # Verdict
        if diff.mean() < 0.01 and diff.max() < 0.1:
            print("\n✓ MATCH - LoRA transfer appears correct")
        elif diff.mean() < 0.1:
            print("\n⚠ SMALL MISMATCH - May be numerical precision differences")
        else:
            print("\n✗ SIGNIFICANT MISMATCH - Train-inference inconsistency detected!")
            print(f"   This explains the exponential KL divergence growth!")
    else:
        print("Could not compare - missing logprobs")
        print(f"  Megatron: {len(megatron_logprobs)}")
        print(f"  vLLM: {len(vllm_logprobs)}")

if __name__ == "__main__":
    main()
