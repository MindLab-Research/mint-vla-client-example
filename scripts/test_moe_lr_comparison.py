#!/usr/bin/env python3
"""MoE training with different learning rates.

Compares LR=1e-4, 1e-5, 1e-6 to find optimal range.
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


def create_session(base_model: str, lora_rank: int, lr: float) -> tuple[str, str]:
    session_id = f"lr_{lr:.0e}_{uuid.uuid4().hex[:6]}"
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


def train_step(model_id: str, iteration: int, lr: float) -> dict:
    """Combined forward_backward + optim_step."""
    # Use SAME data across iterations to test pure overfitting
    data = []
    for i in range(4):  # batch_size=4
        seq_len = 64
        tokens = [151644] + list(range(1000 + i * 100, 1000 + i * 100 + seq_len))
        targets = list(range(1000 + i * 100, 1000 + i * 100 + seq_len)) + [0]
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


def run_experiment(lr: float, n_iters: int = 50) -> dict:
    """Run training with given LR."""
    base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    lora_rank = 32

    print(f"\n{'='*50}")
    print(f"LR = {lr}")
    print(f"{'='*50}")

    session_id, model_id = create_session(base_model, lora_rank, lr)
    print(f"Session: {session_id}")

    results = []
    start_time = time.time()

    for i in range(n_iters):
        result = train_step(model_id, i, lr=lr)
        elapsed = time.time() - start_time
        metrics = result.get("metrics", {})
        loss = metrics.get("loss:mean", 0)
        grad_norm = metrics.get("grad_norm", 0)
        results.append({"iter": i + 1, "loss": loss, "grad_norm": grad_norm, "elapsed": elapsed})

        if (i + 1) % 10 == 0:
            print(f"  Iter {i+1:3d}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

    losses = [r["loss"] for r in results]
    grad_norms = [r["grad_norm"] for r in results]

    return {
        "lr": lr,
        "results": results,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "delta": losses[0] - losses[-1],
        "pct_change": 100 * (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0,
        "avg_grad_norm": sum(grad_norms) / len(grad_norms),
    }


def main():
    print("=" * 70)
    print("MOE LEARNING RATE COMPARISON")
    print("Testing LR: 1e-4, 1e-5, 1e-6 with SAME training data each iteration")
    print("=" * 70)

    output_dir = Path("results/moe_lr_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    learning_rates = [1e-4, 1e-5, 1e-6]
    n_iters = 50
    all_results = {}

    for lr in learning_rates:
        try:
            result = run_experiment(lr, n_iters)
            all_results[str(lr)] = result
            print(f"\nLR={lr}: loss {result['initial_loss']:.4f} -> {result['final_loss']:.4f} ({result['pct_change']:.2f}%)")
        except Exception as e:
            print(f"\nLR={lr} FAILED: {e}")
            all_results[str(lr)] = {"error": str(e)}

    # Save JSON
    output_json = output_dir / f"lr_comparison_{timestamp}.json"
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax2 = axes[1]

    for lr_str, data in all_results.items():
        if "error" in data:
            continue
        iters = [r["iter"] for r in data["results"]]
        losses = [r["loss"] for r in data["results"]]
        grad_norms = [r["grad_norm"] for r in data["results"]]

        ax1.plot(iters, losses, label=f"LR={lr_str}", linewidth=1.5)
        ax2.plot(iters, grad_norms, label=f"LR={lr_str}", linewidth=1.5)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Iteration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Norm vs Iteration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"MoE Training - Learning Rate Comparison (rank=32, {n_iters} iters)", fontsize=12)
    plt.tight_layout()

    output_plot = output_dir / f"lr_comparison_{timestamp}.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_plot}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for lr_str, data in all_results.items():
        if "error" in data:
            print(f"LR={lr_str}: ERROR - {data['error']}")
        else:
            print(f"LR={lr_str}: {data['initial_loss']:.4f} -> {data['final_loss']:.4f} ({data['pct_change']:+.2f}%), min={data['min_loss']:.4f}")


if __name__ == "__main__":
    main()
