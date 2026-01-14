#!/usr/bin/env python3
"""Plot comparison of 4 RL training experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Experiment directories
EXPERIMENTS = {
    "Exp1: Existing+IS": "/tmp/tinker-examples/math_rl/countdown-moonshotai-Moonlight-16B-A3B-Instruct-16rank-5e-05lr-8group-96batch-importance_sampling-seed42-2026-01-07-02-04",
    "Exp2: Existing+PPO": "/tmp/tinker-examples/math_rl/countdown-moonshotai-Moonlight-16B-A3B-Instruct-16rank-5e-05lr-8group-96batch-ppo-seed42-2026-01-06-19-43",
    "Exp3: M-Bridge+IS": "/tmp/tinker-examples/math_rl/countdown-moonshotai-Moonlight-16B-A3B-Instruct-16rank-5e-05lr-8group-96batch-importance_sampling-seed42-2026-01-07-03-57",
    "Exp4: M-Bridge+PPO": "/tmp/tinker-examples/math_rl/countdown-moonshotai-Moonlight-16B-A3B-Instruct-16rank-5e-05lr-8group-96batch-ppo-seed42-2026-01-07-07-13",
}

def load_metrics(exp_dir):
    """Load metrics from jsonl file."""
    metrics_file = Path(exp_dir) / "metrics.jsonl"
    if not metrics_file.exists():
        return None

    data = []
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_metric(data, key):
    """Extract a specific metric from data."""
    return [d.get(key, 0) for d in data]

def main():
    # Load all experiments
    exp_data = {}
    for name, path in EXPERIMENTS.items():
        data = load_metrics(path)
        if data:
            exp_data[name] = data
            print(f"{name}: {len(data)} steps loaded")
        else:
            print(f"{name}: No data found")

    if not exp_data:
        print("No experiment data found!")
        return

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "Exp1: Existing+IS": "blue",
        "Exp2: Existing+PPO": "green",
        "Exp3: M-Bridge+IS": "red",
        "Exp4: M-Bridge+PPO": "purple",
    }

    # Plot 1: Correctness
    ax = axes[0, 0]
    for name, data in exp_data.items():
        steps = list(range(len(data)))
        correct = [d.get("env/all/correct", 0) * 100 for d in data]
        ax.plot(steps, correct, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Correctness (%)")
    ax.set_title("Correctness over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 35)

    # Plot 2: KL Divergence
    ax = axes[0, 1]
    for name, data in exp_data.items():
        steps = list(range(len(data)))
        kl = [d.get("optim/kl_sample_train_v1", 0) for d in data]
        ax.plot(steps, kl, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Divergence (v1)")
    ax.set_title("KL Divergence over Training")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Plot 3: Format Adherence
    ax = axes[1, 0]
    for name, data in exp_data.items():
        steps = list(range(len(data)))
        fmt = [d.get("env/all/format", 0) * 100 for d in data]
        ax.plot(steps, fmt, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Format Adherence (%)")
    ax.set_title("Format Adherence over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 105)

    # Plot 4: Reward
    ax = axes[1, 1]
    for name, data in exp_data.items():
        steps = list(range(len(data)))
        reward = [d.get("env/all/reward/total", 0) for d in data]
        ax.plot(steps, reward, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward over Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("RL Training Experiments Comparison\nMoonlight-16B-A3B, Countdown Task", fontsize=14)
    plt.tight_layout()

    # Save plot
    output_path = Path(__file__).parent.parent / "experiment_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for name, data in exp_data.items():
        correct = [d.get("env/all/correct", 0) * 100 for d in data]
        kl = [d.get("optim/kl_sample_train_v1", 0) for d in data]
        print(f"\n{name}:")
        print(f"  Steps: {len(data)}")
        print(f"  Correct: {correct[0]:.1f}% -> peak {max(correct):.1f}% (step {correct.index(max(correct))})")
        print(f"  KL: {kl[0]:.2f} -> {kl[-1]:.2f}")

if __name__ == "__main__":
    main()
