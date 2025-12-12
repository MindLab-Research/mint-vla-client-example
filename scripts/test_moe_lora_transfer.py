#!/usr/bin/env python
"""MoE LoRA Transfer Integration Test - Phase 3 Validation.

Tests end-to-end flow: Megatron MoE training -> LoRA extraction -> vLLM inference

Steps:
1. Create MoE training session (Qwen3-30B-A3B via Megatron)
2. Run forward_backward + optim_step (modify LoRA weights)
3. Save weights for sampler -> extracts LoRA in PEFT format
4. Create sampling session with trained LoRA
5. Generate with LoRA and verify output

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_moe_lora_transfer.py
"""

import json
import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(2)
            continue
        else:
            resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_training_session(base_model: str, rank: int = 16, timeout: int = 900) -> tuple[str, str]:
    """Create training session, return (session_id, model_id)."""
    session_id = str(uuid.uuid4())
    model_seq_id = 1

    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": model_seq_id,
        "base_model": base_model,
        "lora_config": {"rank": rank},
    }
    resp = requests.post(url, json=payload, timeout=900)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    result = poll_future(request_id, timeout=timeout)

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    model_id = result.get("model_id")
    backend = result.get("backend")

    print(f"Session created: model_id={model_id}, backend={backend}")
    return session_id, model_id


def forward_backward(model_id: str, data: list) -> dict:
    """Run forward_backward and poll for result."""
    url = f"{BASE_URL}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": "cross_entropy",
        },
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=300)


def optim_step(model_id: str, learning_rate: float = 1e-4) -> dict:
    """Run optim_step and poll for result."""
    url = f"{BASE_URL}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": learning_rate},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=120)


def save_weights_for_sampler(model_id: str) -> str:
    """Save weights for sampler and return sampling_session_id."""
    url = f"{BASE_URL}/api/v1/save_weights_for_sampler"
    payload = {"model_id": model_id}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    result = poll_future(request_id, timeout=300)

    if "error" in result:
        raise RuntimeError(f"save_weights_for_sampler failed: {result['error']}")

    sampling_session_id = result.get("sampling_session_id")
    print(f"LoRA transferred to vLLM: sampling_session_id={sampling_session_id}")
    return sampling_session_id


def sample_with_lora(sampling_session_id: str, prompt_tokens: list, max_tokens: int = 50) -> dict:
    """Generate tokens using LoRA adapter."""
    url = f"{BASE_URL}/api/v1/asample"
    payload = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "num_samples": 1,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=120)


def create_base_sampling_session(session_id: str) -> str:
    """Create sampling session with base model (no LoRA)."""
    url = f"{BASE_URL}/api/v1/create_sampling_session"
    payload = {"session_id": session_id}
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["sampling_session_id"]


def create_sample_data() -> list:
    """Create sample data for SFT training."""
    # "Hello world" style sequences
    input_tokens_1 = [9707, 1917, 0]
    target_tokens_1 = [1917, 0, 0]
    loss_mask_1 = [1.0, 1.0, 0.0]

    input_tokens_2 = [9707, 0]
    target_tokens_2 = [0, 0]
    loss_mask_2 = [1.0, 0.0]

    return [
        {
            "model_input": {
                "chunks": [{"tokens": input_tokens_1, "type": "encoded_text"}]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens_1, "shape": [len(target_tokens_1)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask_1, "shape": [len(loss_mask_1)], "dtype": "float32"},
            },
        },
        {
            "model_input": {
                "chunks": [{"tokens": input_tokens_2, "type": "encoded_text"}]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens_2, "shape": [len(target_tokens_2)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask_2, "shape": [len(loss_mask_2)], "dtype": "float32"},
            },
        },
    ]


def test_moe_lora_transfer():
    """Test MoE LoRA transfer: training -> extraction -> inference."""
    print("=" * 70)
    print("TEST: MoE LoRA Transfer (Phase 3)")
    print("=" * 70)

    moe_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # Step 1: Create training session
    print(f"\n[1/6] Creating MoE training session for: {moe_model}")
    print("This may take several minutes for Megatron initialization...")

    t0 = time.time()
    try:
        session_id, model_id = create_training_session(moe_model, rank=16)
        init_time = time.time() - t0
        print(f"Session created in {init_time:.2f}s")
    except Exception as e:
        print(f"[FAIL] create_model failed: {e}")
        return False

    # Step 2: forward_backward
    print(f"\n[2/6] Running forward_backward...")
    data = create_sample_data()

    t0 = time.time()
    try:
        fb_result = forward_backward(model_id, data)
        fb_time = time.time() - t0
        print(f"forward_backward completed in {fb_time:.2f}s")
        metrics = fb_result.get("metrics", {})
        loss = metrics.get("loss:mean", "N/A")
        print(f"Loss: {loss}")
    except Exception as e:
        print(f"[FAIL] forward_backward failed: {e}")
        return False

    # Step 3: optim_step (modify LoRA weights)
    print(f"\n[3/6] Running optim_step...")

    t0 = time.time()
    try:
        optim_result = optim_step(model_id, learning_rate=1e-4)
        optim_time = time.time() - t0
        print(f"optim_step completed in {optim_time:.2f}s")
    except Exception as e:
        print(f"[FAIL] optim_step failed: {e}")
        return False

    # Step 4: Save weights for sampler (transfers LoRA to vLLM)
    print(f"\n[4/6] Extracting LoRA and transferring to vLLM...")
    print("(PEFT format conversion: mbridge -> base_model.model.model.*)")

    t0 = time.time()
    try:
        sampling_session_id = save_weights_for_sampler(model_id)
        transfer_time = time.time() - t0
        print(f"LoRA transfer completed in {transfer_time:.2f}s")
    except Exception as e:
        print(f"[FAIL] save_weights_for_sampler failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Generate with LoRA
    print(f"\n[5/6] Generating with trained LoRA adapter...")
    # "Hello" token
    prompt_tokens = [9707]

    t0 = time.time()
    try:
        lora_result = sample_with_lora(sampling_session_id, prompt_tokens, max_tokens=30)
        gen_time = time.time() - t0
        lora_tokens = lora_result["sequences"][0]["tokens"]
        print(f"LoRA generation completed in {gen_time:.2f}s")
        print(f"Generated {len(lora_tokens)} tokens: {lora_tokens[:10]}...")
    except Exception as e:
        print(f"[FAIL] LoRA sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Compare with base model (optional but informative)
    print(f"\n[6/6] Generating with base model for comparison...")

    t0 = time.time()
    try:
        base_session_id = create_base_sampling_session(session_id)
        base_result = sample_with_lora(base_session_id, prompt_tokens, max_tokens=30)
        base_time = time.time() - t0
        base_tokens = base_result["sequences"][0]["tokens"]
        print(f"Base generation completed in {base_time:.2f}s")
        print(f"Generated {len(base_tokens)} tokens: {base_tokens[:10]}...")

        # Compare
        if lora_tokens != base_tokens:
            print("\nLoRA output differs from base model - training affected model behavior")
        else:
            print("\nLoRA output same as base model (expected with minimal training)")
    except Exception as e:
        print(f"[WARN] Base model comparison failed: {e}")
        # Not a test failure - the main test passed

    print("\n" + "=" * 70)
    print("[PASS] MoE LoRA Transfer completed successfully")
    print("=" * 70)
    return True


def main():
    print(f"Server: {BASE_URL}")
    print("=" * 70)
    print("MOE LORA TRANSFER INTEGRATION TEST (Phase 3)")
    print("=" * 70)

    # Check server health
    print("\nChecking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=10)
        health = resp.json()
        print(f"Server status: {health.get('status')}")
    except Exception as e:
        print(f"Server not available: {e}")
        return 1

    # Run test
    success = test_moe_lora_transfer()

    print("\n" + "=" * 70)
    print("RESULT: " + ("PASS" if success else "FAIL"))
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
