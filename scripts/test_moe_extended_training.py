#!/usr/bin/env python3
"""Extended MoE training test with plotting.

Runs 100 iterations using train_step endpoint and generates training curve.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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


def create_session(base_model: str, lora_rank: int, lr: float = 1e-4) -> tuple[str, str]:
    session_id = f"extended_{uuid.uuid4().hex[:8]}"
    url = f"{get_base_url()}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")
    return session_id, result.get("model_id")


def train_step(model_id: str, iteration: int, lr: float = 1e-4) -> dict:
    """Combined forward_backward + optim_step in single request."""
    data = []
    # Use different token ranges per iteration to avoid memorization
    token_base = 1000 + iteration * 200
    for i in range(4):  # batch_size=4
        seq_len = 64
        tokens = [151644] + list(range(token_base + i * 100, token_base + i * 100 + seq_len))
        targets = list(range(token_base + i * 100, token_base + i * 100 + seq_len)) + [0]
        data.append({
            "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": targets, "shape": [len(targets)], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * seq_len + [0.0], "shape": [seq_len + 1], "dtype": "float32"},
            },
        })

    url = f"{get_base_url()}/api/v1/train_step"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    return result


def plot_training_curve(results: list[dict], output_path: str, title: str = "MoE Training Curve"):
    """Generate training curve plot."""
    iterations = [r["iter"] for r in results]
    losses = [r["loss"] for r in results]
    grad_norms = [r["grad_norm"] for r in results]
    times = [r.get("elapsed", 0) for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss vs Iteration
    ax1 = axes[0, 0]
    ax1.plot(iterations, losses, 'b-', linewidth=1.5, marker='o', markersize=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Iteration")
    ax1.grid(True, alpha=0.3)

    # Loss vs Wall Time
    ax2 = axes[0, 1]
    ax2.plot(times, losses, 'g-', linewidth=1.5, marker='o', markersize=2)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Wall Time")
    ax2.grid(True, alpha=0.3)

    # Gradient Norm vs Iteration
    ax3 = axes[1, 0]
    ax3.plot(iterations, grad_norms, 'r-', linewidth=1.5, marker='o', markersize=2)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Gradient Norm")
    ax3.set_title("Gradient Norm vs Iteration")
    ax3.grid(True, alpha=0.3)

    # Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')

    initial_loss = losses[0]
    final_loss = losses[-1]
    delta = initial_loss - final_loss
    pct = 100 * delta / initial_loss if initial_loss > 0 else 0
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    total_time = times[-1] if times else 0
    avg_iter_time = total_time / len(results) if results else 0

    stats_text = f"""Training Summary

Initial Loss: {initial_loss:.4f}
Final Loss: {final_loss:.4f}
Loss Change: {delta:.4f} ({pct:.2f}%)

Avg Gradient Norm: {avg_grad_norm:.4f}
Max Gradient Norm: {max(grad_norms):.4f}
Min Gradient Norm: {min(grad_norms):.4f}

Total Iterations: {len(results)}
Total Time: {total_time:.1f}s
Avg Time/Iter: {avg_iter_time:.2f}s

Training Status: {"WORKING" if delta > 0.05 else "FLAT"}
"""
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    print("=" * 70)
    print("EXTENDED MOE TRAINING TEST (100 iterations)")
    print("Testing combined train_step endpoint")
    print("=" * 70)

    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32
    lr = 1e-4
    n_iters = 100

    # Create output directory
    output_dir = Path("results/moe_extended_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create session
    print(f"\nCreating session with {base_model}, rank={lora_rank}, lr={lr}...")
    start_time = time.time()
    session_id, model_id = create_session(base_model, lora_rank, lr)
    create_time = time.time() - start_time
    print(f"Session created in {create_time:.2f}s: {session_id}, model_id: {model_id}")

    # Run training
    print(f"\n--- Training for {n_iters} iterations ---")
    results = []
    training_start = time.time()

    for i in range(n_iters):
        iter_start = time.time()
        result = train_step(model_id, i, lr=lr)
        iter_time = time.time() - iter_start
        elapsed = time.time() - training_start

        metrics = result.get("metrics", {})
        loss = metrics.get("loss:mean", 0)
        grad_norm = metrics.get("grad_norm", 0)

        results.append({
            "iter": i + 1,
            "loss": loss,
            "grad_norm": grad_norm,
            "iter_time": iter_time,
            "elapsed": elapsed,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iter {i+1:3d}: loss={loss:.4f}, grad_norm={grad_norm:.4f}, time={iter_time:.2f}s")

    total_time = time.time() - training_start

    # Analyze results
    losses = [r["loss"] for r in results]
    grad_norms = [r["grad_norm"] for r in results]
    initial_loss = losses[0]
    final_loss = losses[-1]
    delta = initial_loss - final_loss
    pct = 100 * delta / initial_loss if initial_loss > 0 else 0

    print(f"\n--- Results ---")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Delta:        {delta:.4f} ({pct:.2f}%)")
    print(f"Min loss:     {min(losses):.4f}")
    print(f"Max loss:     {max(losses):.4f}")
    print(f"Avg grad_norm: {sum(grad_norms) / len(grad_norms):.4f}")
    print(f"Total time:   {total_time:.1f}s")
    print(f"Avg iter time: {total_time / n_iters:.2f}s")

    # Determine if training is working
    training_works = delta > 0.05 and sum(grad_norms) > 0
    print(f"\nTraining {'WORKING' if training_works else 'NOT WORKING'}")

    # Save JSON results
    output_json = output_dir / f"extended_training_{timestamp}.json"
    with open(output_json, "w") as f:
        json.dump({
            "config": {
                "base_model": base_model,
                "lora_rank": lora_rank,
                "learning_rate": lr,
                "n_iters": n_iters,
            },
            "summary": {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "delta": delta,
                "pct_decrease": pct,
                "min_loss": min(losses),
                "max_loss": max(losses),
                "avg_grad_norm": sum(grad_norms) / len(grad_norms),
                "total_time": total_time,
                "avg_iter_time": total_time / n_iters,
                "training_works": training_works,
            },
            "iterations": results,
        }, f, indent=2)
    print(f"Results saved to {output_json}")

    # Generate plot
    output_plot = output_dir / f"training_curve_{timestamp}.png"
    plot_training_curve(
        results,
        str(output_plot),
        title=f"MoE Training (rank={lora_rank}, lr={lr}, iters={n_iters})"
    )

    print("\n" + "=" * 70)
    return training_works


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
