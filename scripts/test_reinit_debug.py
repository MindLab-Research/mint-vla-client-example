#!/usr/bin/env python
"""Minimal test to debug reinit_lora_weights optimizer state clearing."""

import os
import sys
import requests
import json
import time
import random

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

def create_session(session_id: str, learning_rate: float):
    """Create a training session."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_config": {
                "base_model": "/vePFS-Mindverse/share/checkpoint/qwen2.5_moe/Qwen2.5-3B-Instruct-AWQ-A100",
                "lora_rank": 32,
                "use_lora": True,
                "learning_rate": learning_rate,
                "max_new_tokens": 256,
                "temperature": 1.0,
            }
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()

def train_step(session_id: str, batch):
    """Run one training step."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/train_step",
        json={"session_id": session_id, "batch": batch},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()

def delete_session(session_id: str):
    """Delete a session."""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/delete_session",
            json={"session_id": session_id},
            timeout=30,
        )
    except:
        pass

def make_batch():
    """Create a simple training batch."""
    prompt = "What is 2 + 2?"
    response = "The answer is 4."
    return [{"prompt": prompt, "response": response}]

def run_session(session_id: str, learning_rate: float, n_iters: int = 5):
    """Run a session with n_iters training steps."""
    print(f"\n{'='*60}")
    print(f"SESSION: {session_id}, LR={learning_rate}")
    print('='*60)

    print(f"Creating session...")
    create_session(session_id, learning_rate)
    print(f"Session created")

    batch = make_batch()
    for i in range(1, n_iters + 1):
        result = train_step(session_id, batch)
        loss = result.get("loss", 0)
        grad_norm = result.get("grad_norm", 0)
        print(f"  Iter {i}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

    delete_session(session_id)
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("REINIT DEBUG TEST")
    print("Testing optimizer state clearing between sessions")
    print("=" * 60)

    # Session 1: Train with LR=1e-4
    run_session(f"debug_lr1e4_{random.randint(0,999999):06d}", learning_rate=1e-4, n_iters=10)

    # Session 2: Should start fresh with LR=1e-5
    # If optimizer state carries over, will show instant convergence
    run_session(f"debug_lr1e5_{random.randint(0,999999):06d}", learning_rate=1e-5, n_iters=10)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("Check Ray worker logs for [REINIT DEBUG] output")
    print("=" * 60)
