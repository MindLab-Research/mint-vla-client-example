#!/usr/bin/env python3
"""Phase 7 Test: Unified Rank Support via Max-Rank Padding.

Tests:
1. Create sessions with different LoRA ranks (16, 32, 64)
2. Verify all sessions reuse the same Megatron actor
3. Train each session and track loss
4. Verify iteration times are similar across ranks (max-rank padding overhead)

Run from tinker-server root:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_phase7_unified_rank.py
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import matplotlib.pyplot as plt
import numpy as np


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
    session_id = f"rank{lora_rank}_{uuid.uuid4().hex[:8]}"
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


def get_megatron_status() -> dict:
    """Get Megatron actor status."""
    url = f"{get_base_url()}/api/v1/megatron_status"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def plot_results(results: dict, output_path: Path):
    """Generate comparison plots for different ranks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ranks = sorted(results["sessions"].keys())
    colors = {"16": "b", "32": "g", "64": "r"}

    # Plot 1: Loss vs Iteration
    for rank in ranks:
        data = results["sessions"][rank]
        iters = [r["iteration"] for r in data["iterations"]]
        losses = [r["loss"] for r in data["iterations"]]
        axes[0, 0].plot(iters, losses, f'{colors.get(rank, "k")}-o',
                        label=f'Rank {rank}', markersize=4)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss (NLL)")
    axes[0, 0].set_title("Loss vs Iteration by LoRA Rank")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Iteration Time by Rank
    rank_times = {}
    for rank in ranks:
        data = results["sessions"][rank]
        times = [r["fb_time"] + r["opt_time"] for r in data["iterations"]]
        rank_times[rank] = times

    x = np.arange(len(ranks))
    avg_times = [np.mean(rank_times[r]) for r in ranks]
    std_times = [np.std(rank_times[r]) for r in ranks]
    bar_colors = ["blue", "green", "red"][:len(ranks)]
    axes[0, 1].bar(x, avg_times, yerr=std_times, color=bar_colors, alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'Rank {r}' for r in ranks])
    axes[0, 1].set_ylabel("Avg Iteration Time (s)")
    axes[0, 1].set_title("Iteration Time by LoRA Rank (should be similar)")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Session Create Time
    create_times = [results["sessions"][r]["create_time"] for r in ranks]
    axes[1, 0].bar(x, create_times, color=bar_colors, alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'Rank {r}' for r in ranks])
    axes[1, 0].set_ylabel("Session Create Time (s)")
    axes[1, 0].set_title("Session Create Time (actor reuse)")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary text
    axes[1, 1].axis('off')
    summary_text = f"""
    Phase 7: Unified Rank Support Test Summary

    Model: {results['model']}
    Ranks tested: {', '.join(ranks)}

    Session Create Times:
    {chr(10).join([f'  Rank {r}: {results["sessions"][r]["create_time"]:.2f}s' for r in ranks])}

    Avg Iteration Times:
    {chr(10).join([f'  Rank {r}: {np.mean(rank_times[r]):.2f}s' for r in ranks])}

    Actor Reuse Verified: {results.get('actor_reuse_verified', 'N/A')}

    Key Finding:
    Iteration times should be similar across ranks due to
    max-rank padding. Small create times indicate actor reuse.
    """
    axes[1, 1].text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
                    verticalalignment='top', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Unified Rank Test")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--ranks", type=str, default="16,32,64", help="Comma-separated ranks to test")
    parser.add_argument("--num-iters", type=int, default=10, help="Iterations per rank")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase7_unified_rank"))
    args = parser.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PHASE 7: UNIFIED RANK SUPPORT TEST")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Ranks to test: {ranks}")
    print(f"Iterations per rank: {args.num_iters}")
    print(f"{'='*60}\n")

    results = {
        "model": args.model,
        "ranks": ranks,
        "num_iters": args.num_iters,
        "start_time": datetime.now().isoformat(),
        "sessions": {},
    }

    # Check initial Megatron status
    initial_status = get_megatron_status()
    print(f"Initial Megatron status: {initial_status}\n")

    # Test each rank
    for rank in ranks:
        print(f"=== Testing Rank {rank} ===")

        # Create session
        session_id, model_id, create_time = create_session(args.model, rank)
        print(f"  Session: {session_id}, create_time: {create_time:.2f}s")

        # Check Megatron status after session creation
        status = get_megatron_status()
        actor_alive = status.get("alive", False)
        print(f"  Megatron actor alive: {actor_alive}")

        # Train iterations
        iterations = []
        for i in range(args.num_iters):
            loss, fb_time = forward_backward(model_id, token_base=1000 + rank * 100)
            opt_time = optim_step(model_id)
            iterations.append({
                "iteration": i + 1,
                "loss": loss,
                "fb_time": fb_time,
                "opt_time": opt_time,
            })
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Iter {i+1}: loss={loss:.4f}, time={fb_time + opt_time:.2f}s")

        results["sessions"][str(rank)] = {
            "session_id": session_id,
            "model_id": model_id,
            "create_time": create_time,
            "iterations": iterations,
        }
        print()

    # Verify actor reuse
    first_create = results["sessions"][str(ranks[0])]["create_time"]
    actor_reuse = all(
        results["sessions"][str(r)]["create_time"] < 5.0  # Fast create = reuse
        for r in ranks[1:]
    )
    results["actor_reuse_verified"] = actor_reuse
    print(f"Actor reuse verified: {actor_reuse}")
    print(f"  First session create: {first_create:.2f}s")
    for r in ranks[1:]:
        print(f"  Rank {r} create: {results['sessions'][str(r)]['create_time']:.2f}s")

    # Calculate timing consistency
    print("\n=== Iteration Time Consistency ===")
    for rank in ranks:
        times = [r["fb_time"] + r["opt_time"] for r in results["sessions"][str(rank)]["iterations"]]
        avg = np.mean(times)
        std = np.std(times)
        print(f"Rank {rank}: avg={avg:.2f}s, std={std:.3f}s")

    results["end_time"] = datetime.now().isoformat()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"unified_rank_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_file}")

    # Generate plots
    plot_path = args.output_dir / f"unified_rank_test_{timestamp}.png"
    plot_results(results, plot_path)

    print(f"\n{'='*60}")
    print("PHASE 7 UNIFIED RANK TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
