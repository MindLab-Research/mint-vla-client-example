"""Shared utilities for merge gate tests.

Provides:
- Training curve saving and plotting
- Anomaly detection
- Result aggregation for visual inspection
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_training_curve(
    metrics: dict[str, list],
    test_name: str,
    metadata: dict = None,
    plot_title: str = None,
) -> tuple[Path, Path | None]:
    """Save training metrics and generate visualization.

    Args:
        metrics: Dict of metric_name -> list of values per iteration
                 Must include "losses" key
        test_name: Name for output files
        metadata: Additional metadata to save
        plot_title: Custom plot title

    Returns:
        (data_path, plot_path) - plot_path is None if matplotlib unavailable
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iterations = list(range(1, len(metrics["losses"]) + 1))

    # Save raw data
    data = {
        "test_name": test_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "metrics": metrics,
        "metadata": metadata or {},
    }

    data_file = RESULTS_DIR / f"{test_name}_{timestamp}.json"
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)

    # Generate plot
    plot_file = None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        losses = metrics["losses"]
        n_metrics = sum(1 for k in metrics if k != "losses" and len(metrics[k]) > 0)

        # Create subplots
        fig, axes = plt.subplots(1 + n_metrics, 1, figsize=(10, 4 * (1 + n_metrics)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Loss plot
        ax = axes[0]
        ax.plot(iterations, losses, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(plot_title or f"{test_name}: Training Curve")
        ax.grid(True, alpha=0.3)

        # Annotate
        if losses:
            ax.annotate(f"Initial: {losses[0]:.4f}", xy=(1, losses[0]),
                       xytext=(2, losses[0]), fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='gray'))
            ax.annotate(f"Final: {losses[-1]:.4f}", xy=(len(losses), losses[-1]),
                       xytext=(len(losses)-1, losses[-1]), fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='gray'))
            reduction = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0
            ax.text(0.02, 0.98, f"Reduction: {reduction:.1f}%",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Additional metrics
        metric_idx = 1
        colors = ['g', 'orange', 'purple', 'r']  # Use short color codes or names
        for key, values in metrics.items():
            if key == "losses" or not values:
                continue
            ax = axes[metric_idx]
            color = colors[(metric_idx - 1) % len(colors)]
            ax.plot(iterations[:len(values)], values, '-o', color=color, linewidth=2, markersize=6)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(key.replace("_", " ").title())
            ax.set_title(f"{key}")
            ax.grid(True, alpha=0.3)
            metric_idx += 1

        plt.tight_layout()
        plot_file = RESULTS_DIR / f"{test_name}_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

    except ImportError:
        pass

    print(f"\nResults saved:")
    print(f"  Data: {data_file}")
    if plot_file:
        print(f"  Plot: {plot_file}")

    return data_file, plot_file


def detect_anomalies(losses: list, metric_name: str = "loss") -> list[str]:
    """Detect anomalies in training curve.

    Checks for:
    - Loss spikes (>10% increase)
    - Plateaus (no change in last 3 iterations)
    - NaN/Inf values
    - Insufficient learning (<30% decrease)

    Returns list of anomaly descriptions.
    """
    anomalies = []

    if not losses:
        return ["No data collected"]

    # Check for loss spikes
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] * 1.1:
            anomalies.append(f"{metric_name} spike at iter {i+1}: {losses[i-1]:.4f} -> {losses[i]:.4f} (+{(losses[i]/losses[i-1]-1)*100:.1f}%)")

    # Check for plateaus
    if len(losses) >= 3:
        recent = losses[-3:]
        spread = max(recent) - min(recent)
        if spread < 0.01:
            anomalies.append(f"{metric_name} plateau: last 3 values within {spread:.4f}")

    # Check for NaN/Inf
    for i, val in enumerate(losses):
        if np.isnan(val) or np.isinf(val):
            anomalies.append(f"Invalid {metric_name} at iter {i+1}: {val}")

    # Check for insufficient decrease
    if len(losses) >= 2 and losses[0] > 0:
        decrease = (losses[0] - losses[-1]) / losses[0]
        if decrease < 0.3:
            anomalies.append(f"Insufficient learning: only {decrease:.1%} decrease")

    return anomalies


def print_test_summary(
    test_name: str,
    metrics: dict[str, list],
    anomalies: list[str],
    plot_path: Path | None,
    extra_info: dict[str, Any] = None,
):
    """Print formatted test summary."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {test_name}")
    print(f"{'='*60}")

    losses = metrics.get("losses", [])
    if losses:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        if losses[0] > 0:
            reduction = (losses[0] - losses[-1]) / losses[0]
            print(f"  Reduction: {reduction:.1%}")

    if extra_info:
        for key, val in extra_info.items():
            print(f"  {key}: {val}")

    print(f"  Anomalies: {len(anomalies)}")

    if anomalies:
        print(f"\n*** ANOMALIES DETECTED - REQUIRE VISUAL INSPECTION ***")
        for a in anomalies:
            print(f"  - {a}")

    if plot_path:
        print(f"\nInspect plot: {plot_path}")


def aggregate_results(results_dir: Path = RESULTS_DIR) -> dict:
    """Aggregate all test results for summary report."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            test_name = data.get("test_name", json_file.stem)
            results[test_name] = {
                "timestamp": data.get("timestamp"),
                "initial_loss": data["metrics"]["losses"][0] if data.get("metrics", {}).get("losses") else None,
                "final_loss": data["metrics"]["losses"][-1] if data.get("metrics", {}).get("losses") else None,
                "metadata": data.get("metadata", {}),
            }
    return results
