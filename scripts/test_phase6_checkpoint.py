#!/usr/bin/env python3
"""Phase 6 Test: Checkpoint Save/Restore.

Tests:
1. Create session A and train for N iterations
2. Save checkpoint
3. Train more (modifies weights)
4. Restore checkpoint
5. Verify loss returns to pre-restore value

Run from tinker-server root:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_phase6_checkpoint.py
"""

import argparse
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


def create_session(base_model: str, lora_rank: int) -> tuple[str, str, float]:
    session_id = f"ckpt_test_{uuid.uuid4().hex[:8]}"
    url = f"{get_base_url()}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")
    return session_id, result.get("model_id"), time.time() - t0


def forward_backward(model_id: str, token_base: int = 1000) -> tuple[float, float]:
    data = []
    for i in range(2):
        tokens = [151644] + list(range(token_base + i * 100, token_base + i * 100 + 32))
        targets = list(range(token_base + i * 100, token_base + i * 100 + 32)) + [0]
        data.append({
            "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": targets, "shape": [len(targets)], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * 32 + [0.0], "shape": [33], "dtype": "float32"},
            },
        })

    url = f"{get_base_url()}/api/v1/forward_backward"
    payload = {"model_id": model_id, "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"}}
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"forward_backward error: {result['error']}")
    loss = result.get("metrics", {}).get("loss:mean", 0)
    if loss == 0:
        loss_outputs = result.get("loss_fn_outputs", [])
        if loss_outputs:
            losses = [o.get("loss", {}).get("data", [0])[0] for o in loss_outputs]
            loss = sum(losses) / len(losses) if losses else 0
    return loss, time.time() - t0


def optim_step(model_id: str) -> float:
    url = f"{get_base_url()}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    poll_future(resp.json().get("request_id"), timeout=300)
    return time.time() - t0


def save_weights(model_id: str, checkpoint_name: str) -> dict:
    """Save adapter weights to checkpoint."""
    url = f"{get_base_url()}/api/v1/save_weights"
    # path is checkpoint name (relative), not full path
    payload = {"model_id": model_id, "path": checkpoint_name}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    return result


def load_weights(model_id: str, tinker_path: str) -> dict:
    """Load adapter weights from checkpoint."""
    url = f"{get_base_url()}/api/v1/load_weights"
    # path must be tinker:// URI or file:// path
    payload = {"model_id": model_id, "path": tinker_path}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Checkpoint Test")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--checkpoint-path", default="/vePFS-Mindverse/share/code/tinker-server/checkpoints/phase6_test")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PHASE 6: CHECKPOINT SAVE/RESTORE TEST")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"{'='*60}\n")

    # Step 1: Create session
    print("Step 1: Creating session...")
    session_id, model_id, create_time = create_session(args.model, args.lora_rank)
    print(f"  Session: {session_id}, model_id: {model_id}, time: {create_time:.2f}s\n")

    # Step 2: Train 5 iterations
    print("Step 2: Training 5 iterations...")
    for i in range(5):
        loss, _ = forward_backward(model_id)
        optim_step(model_id)
        print(f"  Iter {i+1}: loss = {loss:.4f}")
    print()

    # Step 3: Get baseline loss (before save)
    print("Step 3: Getting baseline loss...")
    baseline_loss, _ = forward_backward(model_id)
    print(f"  Baseline loss: {baseline_loss:.4f}\n")

    # Step 4: Save checkpoint
    print("Step 4: Saving checkpoint...")
    checkpoint_name = "phase6_ckpt"
    save_result = save_weights(model_id, checkpoint_name)
    print(f"  Save result: {save_result}")
    saved_path = save_result.get("path", "")
    print(f"  Saved to: {saved_path}\n")

    # Step 5: Train 5 more iterations (modifies weights)
    print("Step 5: Training 5 more iterations (modifies weights)...")
    for i in range(5):
        loss, _ = forward_backward(model_id)
        optim_step(model_id)
        print(f"  Iter {i+1}: loss = {loss:.4f}")
    print()

    # Step 6: Get post-training loss
    print("Step 6: Getting post-training loss...")
    post_train_loss, _ = forward_backward(model_id)
    print(f"  Post-training loss: {post_train_loss:.4f}\n")

    # Step 7: Restore checkpoint
    print("Step 7: Restoring checkpoint...")
    if saved_path:
        load_result = load_weights(model_id, saved_path)
        print(f"  Load result: {load_result}\n")
    else:
        print("  ERROR: No saved path available\n")
        load_result = {}

    # Step 8: Get restored loss
    print("Step 8: Getting restored loss...")
    restored_loss, _ = forward_backward(model_id)
    print(f"  Restored loss: {restored_loss:.4f}\n")

    # Step 9: Verify
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline loss (after 5 iters):  {baseline_loss:.4f}")
    print(f"Post-training loss (after 10 iters): {post_train_loss:.4f}")
    print(f"Restored loss (after restore): {restored_loss:.4f}")
    print()

    delta_baseline_restored = abs(baseline_loss - restored_loss)
    delta_baseline_posttrain = abs(baseline_loss - post_train_loss)

    print(f"|Baseline - Restored|: {delta_baseline_restored:.4f}")
    print(f"|Baseline - PostTrain|: {delta_baseline_posttrain:.4f}")

    # Checkpoint is successful if restored loss is closer to baseline than post-train loss
    checkpoint_works = delta_baseline_restored < delta_baseline_posttrain
    print(f"\nCheckpoint restore works: {checkpoint_works}")
    print("="*60)


if __name__ == "__main__":
    main()
