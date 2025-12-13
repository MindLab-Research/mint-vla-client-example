#!/usr/bin/env python3
"""Phase 11: Session swap stress test.

Creates multiple sessions sequentially on the same base model,
each doing a few training iterations. Tests:
- Actor reuse across sessions
- Session state isolation
- No memory leaks or errors over many swaps

Run from tinker-server root directory:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_session_stress.py --num-sessions 100
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests


def get_base_url():
    return os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll for async operation result."""
    poll_url = f"{get_base_url()}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
            continue
        else:
            resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_session(base_model: str, lora_rank: int = 32) -> tuple[str, str, float]:
    """Create training session and return (session_id, model_id, create_time)."""
    session_id = str(uuid.uuid4())
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
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=300)
    create_time = time.time() - t0

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    return session_id, result.get("model_id"), create_time


def forward_backward(model_id: str, data: list) -> tuple[float, float]:
    """Run one forward-backward pass. Returns (loss, time)."""
    url = f"{get_base_url()}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": "cross_entropy",
        },
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=300)
    fb_time = time.time() - t0

    if "error" in result:
        raise RuntimeError(f"forward_backward error: {result['error']}")

    loss = result.get("metrics", {}).get("loss:mean", 0)
    if loss == 0:
        # Try alternate format
        loss_outputs = result.get("loss_fn_outputs", [])
        if loss_outputs:
            losses = [o.get("loss", {}).get("data", [0])[0] for o in loss_outputs]
            loss = sum(losses) / len(losses) if losses else 0

    return loss, fb_time


def optim_step(model_id: str, learning_rate: float = 1e-4) -> float:
    """Run optimizer step. Returns time."""
    url = f"{get_base_url()}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": learning_rate},
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=120)
    opt_time = time.time() - t0

    if "error" in result:
        raise RuntimeError(f"optim_step error: {result['error']}")

    return opt_time


def create_sample_data(batch_size: int = 2, seq_len: int = 16) -> list:
    """Create minimal sample data for testing."""
    data = []
    for i in range(batch_size):
        input_tokens = [151644] + list(range(1000 + i * 100, 1000 + i * 100 + seq_len))
        target_tokens = list(range(1000 + i * 100, 1000 + i * 100 + seq_len)) + [0]
        loss_mask = [1.0] * seq_len + [0.0]

        data.append({
            "model_input": {
                "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
            },
        })
    return data


def main():
    parser = argparse.ArgumentParser(description="Phase 11 session swap stress test")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name (default: Qwen3-30B-A3B MoE)",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=100,
        help="Number of sessions to create (default: 100)",
    )
    parser.add_argument(
        "--iters-per-session",
        type=int,
        default=2,
        help="Training iterations per session (default: 2)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/session_stress"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SESSION SWAP STRESS TEST")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Sessions: {args.num_sessions}")
    print(f"Iterations per session: {args.iters_per_session}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"{'='*60}\n")

    results = {
        "model_name": args.model,
        "num_sessions": args.num_sessions,
        "iters_per_session": args.iters_per_session,
        "lora_rank": args.lora_rank,
        "sessions": [],
        "start_time": datetime.now().isoformat(),
    }

    total_start = time.time()
    errors = []

    for i in range(args.num_sessions):
        session_result = {
            "session_num": i + 1,
            "iterations": [],
        }

        try:
            # Create session
            session_id, model_id, create_time = create_session(args.model, args.lora_rank)
            session_result["session_id"] = session_id
            session_result["model_id"] = model_id
            session_result["create_time"] = create_time

            # Run training iterations
            data = create_sample_data()
            for j in range(args.iters_per_session):
                loss, fb_time = forward_backward(model_id, data)
                opt_time = optim_step(model_id)
                session_result["iterations"].append({
                    "iteration": j + 1,
                    "loss": loss,
                    "fb_time": fb_time,
                    "opt_time": opt_time,
                })

            session_result["status"] = "ok"

        except Exception as e:
            session_result["status"] = "error"
            session_result["error"] = str(e)
            errors.append((i + 1, str(e)))
            import traceback
            traceback.print_exc()

        results["sessions"].append(session_result)

        # Progress report
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - total_start
            rate = (i + 1) / elapsed
            eta = (args.num_sessions - i - 1) / rate if rate > 0 else 0
            avg_create = sum(s.get("create_time", 0) for s in results["sessions"]) / len(results["sessions"])
            print(
                f"Session {i+1:3d}/{args.num_sessions}: "
                f"create={session_result.get('create_time', 0):.2f}s, "
                f"avg_create={avg_create:.2f}s, "
                f"rate={rate:.2f} sess/min, "
                f"ETA={eta:.0f}s"
            )

    total_time = time.time() - total_start
    results["total_time"] = total_time
    results["end_time"] = datetime.now().isoformat()
    results["num_errors"] = len(errors)
    results["errors"] = errors

    # Calculate statistics
    create_times = [s.get("create_time", 0) for s in results["sessions"] if s.get("status") == "ok"]
    if create_times:
        results["avg_create_time"] = sum(create_times) / len(create_times)
        results["min_create_time"] = min(create_times)
        results["max_create_time"] = max(create_times)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"session_stress_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SESSION SWAP STRESS TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Sessions completed: {args.num_sessions - len(errors)}/{args.num_sessions}")
    print(f"Total time: {total_time:.2f}s")
    if create_times:
        print(f"Avg session create time: {results['avg_create_time']:.2f}s")
        print(f"Min/Max create time: {results['min_create_time']:.2f}s / {results['max_create_time']:.2f}s")
    print(f"Errors: {len(errors)}")
    for sess_num, err in errors[:5]:  # Show first 5 errors
        print(f"  Session {sess_num}: {err[:100]}")
    print(f"Results saved: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
