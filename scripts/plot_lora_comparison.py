#!/usr/bin/env python3
"""Generate comparison plots for different LoRA ranks.

Usage:
    python scripts/plot_lora_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(file_path: Path) -> dict:
    """Load stress test results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def main():
    # Load results from different runs
    results_dir = Path("results")
    comparison_dir = results_dir / "lora_comparison"

    # Find all result files
    all_results = []

    # Load from lora_comparison
    for f in comparison_dir.glob("stress_test_*.json"):
        data = load_results(f)
        all_results.append(data)

    # Also include rank 32 from stress_tests (100-iter run)
    stress_dir = results_dir / "stress_tests"
    for f in stress_dir.glob("stress_test_*.json"):
        data = load_results(f)
        # Only include 100-iteration runs with rank 32
        if data.get("num_iterations") == 100 and data.get("lora_rank") == 32:
            all_results.append(data)
            break  # Just take one

    if not all_results:
        print("No results found!")
        return

    # Sort by lora_rank
    all_results.sort(key=lambda x: x.get("lora_rank", 0))

    print(f"Found {len(all_results)} result files:")
    for r in all_results:
        print(f"  - Rank {r.get('lora_rank')}: {r.get('num_iterations')} iterations")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves comparison
    ax1 = axes[0, 0]
    for data in all_results:
        rank = data.get("lora_rank", "?")
        iterations = [it["iteration"] for it in data["iterations"]]
        losses = [it["loss"] for it in data["iterations"]]
        ax1.plot(iterations, losses, label=f"Rank {rank}", alpha=0.8)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs Iteration by LoRA Rank")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss vs wall time
    ax2 = axes[0, 1]
    for data in all_results:
        rank = data.get("lora_rank", "?")
        wall_times = [it["wall_time"] for it in data["iterations"]]
        losses = [it["loss"] for it in data["iterations"]]
        ax2.plot(wall_times, losses, label=f"Rank {rank}", alpha=0.8)

    ax2.set_xlabel("Wall Time (s)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Wall Time by LoRA Rank")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Iteration time comparison (bar chart)
    ax3 = axes[1, 0]
    ranks = [str(data.get("lora_rank", "?")) for data in all_results]
    fb_times = [np.mean([it["fb_time"] for it in data["iterations"]]) for data in all_results]
    opt_times = [np.mean([it["opt_time"] for it in data["iterations"]]) for data in all_results]

    x = np.arange(len(ranks))
    width = 0.35

    bars1 = ax3.bar(x - width/2, fb_times, width, label='Forward-Backward')
    bars2 = ax3.bar(x + width/2, opt_times, width, label='Optimizer Step')

    ax3.set_xlabel("LoRA Rank")
    ax3.set_ylabel("Time (s)")
    ax3.set_title("Avg Iteration Time by LoRA Rank")
    ax3.set_xticks(x)
    ax3.set_xticklabels(ranks)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Plot 4: Summary metrics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary table
    table_data = []
    headers = ["Rank", "Initial Loss", "Final Loss", "Avg Iter (s)", "Total Time (s)"]

    for data in all_results:
        rank = data.get("lora_rank", "?")
        initial = data.get("initial_loss", 0)
        final = data.get("final_loss", 0)
        avg_iter = data.get("avg_iter_time", 0)
        total = data.get("total_training_time", 0)
        table_data.append([str(rank), f"{initial:.4f}", f"{final:.4f}", f"{avg_iter:.2f}", f"{total:.1f}"])

    table = ax4.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Summary Metrics by LoRA Rank", pad=20)

    # Save figure
    output_path = comparison_dir / "lora_rank_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("LORA RANK COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Rank':<10} {'Initial':<12} {'Final':<12} {'Avg Iter':<12} {'Total':<12}")
    print("-"*60)
    for data in all_results:
        rank = data.get("lora_rank", "?")
        initial = data.get("initial_loss", 0)
        final = data.get("final_loss", 0)
        avg_iter = data.get("avg_iter_time", 0)
        total = data.get("total_training_time", 0)
        print(f"{rank:<10} {initial:<12.4f} {final:<12.4f} {avg_iter:<12.2f} {total:<12.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
