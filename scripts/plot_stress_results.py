#!/usr/bin/env python3
"""Plot stress test results: loss curves vs iteration and wall time."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(file_path: Path) -> dict:
    """Load stress test results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def plot_results(results: dict, output_dir: Path):
    """Generate plots from stress test results."""
    iterations = results.get("iterations", [])
    if not iterations:
        print("No iteration data to plot")
        return

    # Extract data
    iter_nums = [it["iteration"] for it in iterations]
    losses = [it["loss"] for it in iterations]
    wall_times = [it["wall_time"] for it in iterations]
    fb_times = [it["fb_time"] for it in iterations]
    opt_times = [it["opt_time"] for it in iterations]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Loss vs Iteration
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iter_nums, losses, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss vs Iteration\n{results.get('model_name', 'Unknown')} - {results.get('backend', 'Unknown')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_iteration.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'loss_vs_iteration.png'}")

    # Plot 2: Loss vs Wall Time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wall_times, losses, "r-", linewidth=1.5)
    ax.set_xlabel("Wall Time (seconds)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss vs Wall Time\n{results.get('model_name', 'Unknown')} - {results.get('backend', 'Unknown')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_walltime.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'loss_vs_walltime.png'}")

    # Plot 3: Iteration Timing Breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        iter_nums,
        fb_times,
        opt_times,
        labels=["Forward-Backward", "Optimizer Step"],
        alpha=0.7,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Iteration Time Breakdown\n{results.get('model_name', 'Unknown')} - {results.get('backend', 'Unknown')}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "timing_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'timing_breakdown.png'}")

    # Print summary
    print("\nSummary:")
    print(f"  Model: {results.get('model_name')}")
    print(f"  Backend: {results.get('backend')}")
    print(f"  Iterations: {len(iterations)}")
    print(f"  Total time: {results.get('total_training_time', 0):.2f}s")
    print(f"  Initial loss: {results.get('initial_loss', 0):.4f}")
    print(f"  Final loss: {results.get('final_loss', 0):.4f}")
    print(f"  Loss change: {results.get('final_loss', 0) - results.get('initial_loss', 0):.4f}")
    print(f"  Avg iter time: {results.get('avg_iter_time', 0):.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Plot stress test results")
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to stress test results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as results file)",
    )
    args = parser.parse_args()

    results = load_results(args.results_file)
    output_dir = args.output_dir or args.results_file.parent

    plot_results(results, output_dir)


if __name__ == "__main__":
    main()
