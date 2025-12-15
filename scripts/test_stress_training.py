#!/usr/bin/env python3
"""Phase 11: Stress testing for MoE training.

Records loss curves over multiple iterations and saves results for plotting.

Run from tinker-server root directory:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_stress_training.py --iterations 100
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


def poll_future(request_id: str, timeout: int = 600) -> dict:
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


def create_training_session(base_model: str, lora_rank: int = 32, learning_rate: float = 1e-4) -> tuple[str, str, str]:
    """Create training session via create_model API.

    Returns (session_id, model_id, backend).
    """
    session_id = str(uuid.uuid4())
    model_seq_id = 1

    url = f"{get_base_url()}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": model_seq_id,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
    }
    resp = requests.post(url, json=payload, timeout=900)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    result = poll_future(request_id, timeout=900)

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    model_id = result.get("model_id")
    backend = result.get("backend", "unknown")

    return session_id, model_id, backend


def forward_backward(model_id: str, data: list) -> dict:
    """Run forward_backward and poll for result."""
    url = f"{get_base_url()}/api/v1/forward_backward"
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
    url = f"{get_base_url()}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {
            "learning_rate": learning_rate,
        },
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=120)


def create_sample_data(batch_size: int = 4, seq_len: int = 32) -> list:
    """Create sample data for SFT training in Datum format.

    Returns data in Datum format expected by the API:
    - model_input: {chunks: [{tokens, type}]}
    - loss_fn_inputs: {target_tokens, loss_mask}
    """
    data = []
    for i in range(batch_size):
        # Generate varying length sequences for more realistic training
        actual_len = seq_len + (i % 8)

        # Create input tokens (some padding to vary length)
        input_tokens = [151644] + list(range(1000 + i * 100, 1000 + i * 100 + actual_len))
        target_tokens = list(range(1000 + i * 100, 1000 + i * 100 + actual_len)) + [0]
        loss_mask = [1.0] * actual_len + [0.0]

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


def run_stress_test(
    model_name: str,
    num_iterations: int,
    batch_size: int,
    lora_rank: int,
    learning_rate: float,
    output_dir: Path,
):
    """Run stress test with loss curve tracking."""
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {model_name}")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}")
    print(f"Batch size: {batch_size}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Create session
    print("[1] Creating training session...")
    session_start = time.time()
    session_id, model_id, backend = create_training_session(model_name, lora_rank, learning_rate)
    session_time = time.time() - session_start
    print(f"    Session created in {session_time:.2f}s")
    print(f"    model_id: {model_id}")
    print(f"    backend: {backend}")

    # Results tracking
    results = {
        "model_name": model_name,
        "session_id": session_id,
        "model_id": model_id,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "session_create_time": session_time,
        "backend": backend,
        "iterations": [],
        "start_time": datetime.now().isoformat(),
    }

    # Training loop
    print(f"\n[2] Running {num_iterations} training iterations...")
    total_start = time.time()

    try:
        for i in range(num_iterations):
            iter_start = time.time()

            # Generate fresh batch each iteration
            data = create_sample_data(batch_size)

            # Forward-backward
            fb_start = time.time()
            fb_result = forward_backward(model_id, data)
            fb_time = time.time() - fb_start

            # Check for errors
            if "error" in fb_result:
                raise RuntimeError(f"forward_backward error: {fb_result['error']}")

            loss = fb_result.get("metrics", {}).get("loss:mean")
            if loss is None:
                # Try alternate format
                loss_outputs = fb_result.get("loss_fn_outputs", [])
                if loss_outputs:
                    losses = [o.get("loss", {}).get("data", [0])[0] for o in loss_outputs]
                    loss = sum(losses) / len(losses) if losses else 0

            # Optimizer step
            opt_start = time.time()
            opt_result = optim_step(model_id, learning_rate)
            opt_time = time.time() - opt_start

            # Check for errors
            if "error" in opt_result:
                raise RuntimeError(f"optim_step error: {opt_result['error']}")

            iter_time = time.time() - iter_start
            step = opt_result.get("metrics", {}).get("step", i + 1)

            # Record iteration data
            iter_data = {
                "iteration": i + 1,
                "step": step,
                "loss": loss,
                "fb_time": fb_time,
                "opt_time": opt_time,
                "total_time": iter_time,
                "wall_time": time.time() - total_start,
            }
            results["iterations"].append(iter_data)

            # Progress report every 10 iterations
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - total_start
                rate = (i + 1) / elapsed
                eta = (num_iterations - i - 1) / rate if rate > 0 else 0
                print(
                    f"    Iter {i+1:4d}/{num_iterations}: "
                    f"loss={loss:.4f}, "
                    f"fb={fb_time:.2f}s, opt={opt_time:.2f}s, "
                    f"rate={rate:.2f} it/s, ETA={eta:.0f}s"
                )

    except Exception as e:
        print(f"\n    ERROR at iteration {i+1}: {e}")
        results["error"] = str(e)
        results["error_iteration"] = i + 1
        import traceback
        traceback.print_exc()

    # Final stats
    total_time = time.time() - total_start
    results["total_training_time"] = total_time
    results["end_time"] = datetime.now().isoformat()

    if results["iterations"]:
        losses = [it["loss"] for it in results["iterations"]]
        results["initial_loss"] = losses[0]
        results["final_loss"] = losses[-1]
        results["min_loss"] = min(losses)
        results["max_loss"] = max(losses)
        results["avg_iter_time"] = sum(it["total_time"] for it in results["iterations"]) / len(results["iterations"])

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"stress_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("STRESS TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Iterations completed: {len(results['iterations'])}/{num_iterations}")
    print(f"Total time: {total_time:.2f}s")
    if results["iterations"]:
        print(f"Initial loss: {results['initial_loss']:.4f}")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Avg iteration time: {results['avg_iter_time']:.2f}s")
    print(f"Results saved: {output_file}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 11 stress testing")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name (default: Qwen3-30B-A3B MoE)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per iteration (default: 4)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stress_tests"),
        help="Output directory for results (default: results/stress_tests)",
    )
    args = parser.parse_args()

    run_stress_test(
        model_name=args.model,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
