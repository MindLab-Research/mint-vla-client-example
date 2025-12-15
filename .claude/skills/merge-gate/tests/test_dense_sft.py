"""Dense SFT Test: Pig Latin Translation.

Based on official tinker_test.ipynb. Teaches model to translate English to Pig Latin.

Pass criteria: Loss decreases >70% over 10 iterations.

This test collects training curves and saves plots for visual inspection.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from .conftest import (
    DENSE_MODEL,
    create_session,
    forward_backward,
    optim_step,
)


# Results directory for plots and data
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Pig Latin training examples from tinker_test.ipynb
PIG_LATIN_EXAMPLES = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]


def prepare_pig_latin_data(tokenizer) -> tuple[list, list]:
    """Prepare Pig Latin training data."""
    api_data = []
    all_weights = []

    for ex in PIG_LATIN_EXAMPLES:
        prompt = f"English: {ex['input']}\nPig Latin:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(f" {ex['output']}\n\n", add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = weights[1:]
        all_weights.extend(weights)

        api_data.append({
            "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
            },
        })

    return api_data, all_weights


def compute_weighted_loss(result: dict, all_weights: list) -> float:
    """Compute weighted loss from logprobs."""
    logprobs = []
    for item in result.get("loss_fn_outputs", []):
        lp = item.get("logprobs", [])
        logprobs.extend(lp)

    # Fall back to mean loss if logprobs not available or mismatch
    if not logprobs:
        return result.get("metrics", {}).get("loss:mean", 0)

    # Use mean loss - weighted loss calculation requires matching lengths
    return result.get("metrics", {}).get("loss:mean", 0)


def save_training_curve(losses: list, test_name: str, metadata: dict = None):
    """Save training curve data and generate plot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw data
    data = {
        "test_name": test_name,
        "timestamp": timestamp,
        "losses": losses,
        "iterations": list(range(1, len(losses) + 1)),
        "metadata": metadata or {},
    }

    data_file = RESULTS_DIR / f"{test_name}_{timestamp}.json"
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(data["iterations"], losses, 'b-o', linewidth=2, markersize=8)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"{test_name}: Training Curve", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add annotations
        ax.annotate(f"Initial: {losses[0]:.4f}", xy=(1, losses[0]),
                   xytext=(2, losses[0] + 0.2), fontsize=10)
        ax.annotate(f"Final: {losses[-1]:.4f}", xy=(len(losses), losses[-1]),
                   xytext=(len(losses) - 1, losses[-1] + 0.2), fontsize=10)

        # Compute and show reduction
        reduction = (losses[0] - losses[-1]) / losses[0] * 100
        ax.text(0.5, 0.95, f"Reduction: {reduction:.1f}%",
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        plot_file = RESULTS_DIR / f"{test_name}_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nTraining curve saved:")
        print(f"  Data: {data_file}")
        print(f"  Plot: {plot_file}")

        return str(plot_file)

    except ImportError:
        print(f"\nmatplotlib not available, data saved to: {data_file}")
        return None


def detect_anomalies(losses: list) -> list[str]:
    """Detect anomalies in training curve.

    Returns list of detected issues for manual inspection.
    """
    anomalies = []

    # Check for loss increase
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] * 1.1:  # >10% increase
            anomalies.append(f"Loss spike at iteration {i+1}: {losses[i-1]:.4f} -> {losses[i]:.4f}")

    # Check for flat loss (no learning)
    if len(losses) >= 3:
        recent = losses[-3:]
        if max(recent) - min(recent) < 0.01:
            anomalies.append(f"Loss plateau detected: last 3 values within 0.01")

    # Check for NaN/Inf
    for i, loss in enumerate(losses):
        if np.isnan(loss) or np.isinf(loss):
            anomalies.append(f"Invalid loss at iteration {i+1}: {loss}")

    # Check for insufficient decrease
    if len(losses) >= 2:
        total_decrease = (losses[0] - losses[-1]) / losses[0]
        if total_decrease < 0.3:
            anomalies.append(f"Insufficient learning: only {total_decrease:.1%} decrease")

    return anomalies


class TestDenseSFT:
    """Dense model SFT training tests."""

    def test_pig_latin_training(self, tokenizer):
        """Test Pig Latin SFT training with curve analysis.

        Expected: Loss decreases >70% over 10 iterations.
        Saves training curve for visual inspection.
        """
        num_iterations = 10
        min_reduction = 0.70
        lr = 1e-4

        # Create session
        session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=lr)

        # Prepare data
        api_data, all_weights = prepare_pig_latin_data(tokenizer)

        # Training loop - collect all metrics
        metrics = {
            "losses": [],
            "grad_norms": [],
            "iteration_times": [],
        }

        import time
        for i in range(num_iterations):
            t0 = time.time()

            # Forward-backward
            result = forward_backward(model_id, api_data, loss_fn="cross_entropy")
            loss = compute_weighted_loss(result, all_weights)
            metrics["losses"].append(loss)

            # Optim step
            optim_result = optim_step(model_id, lr=lr)
            grad_norm = optim_result.get("metrics", {}).get("grad_norm", 0)
            metrics["grad_norms"].append(grad_norm)

            iteration_time = time.time() - t0
            metrics["iteration_times"].append(iteration_time)

            print(f"Iteration {i+1}: loss={loss:.4f}, grad_norm={grad_norm:.6f}, time={iteration_time:.2f}s")

        # Save training curve
        plot_path = save_training_curve(
            metrics["losses"],
            "dense_sft_pig_latin",
            metadata={
                "model": DENSE_MODEL,
                "lr": lr,
                "num_iterations": num_iterations,
                "grad_norms": metrics["grad_norms"],
            }
        )

        # Detect anomalies
        anomalies = detect_anomalies(metrics["losses"])
        if anomalies:
            print("\n*** ANOMALIES DETECTED - REQUIRE VISUAL INSPECTION ***")
            for a in anomalies:
                print(f"  - {a}")
            if plot_path:
                print(f"\nInspect plot: {plot_path}")

        # Evaluate
        initial_loss = metrics["losses"][0]
        final_loss = metrics["losses"][-1]
        reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

        print(f"\n{'='*60}")
        print(f"RESULTS: Dense SFT (Pig Latin)")
        print(f"{'='*60}")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Reduction: {reduction:.1%}")
        print(f"  Avg iteration time: {np.mean(metrics['iteration_times']):.2f}s")
        print(f"  Anomalies: {len(anomalies)}")

        # Programmatic check (soft - anomalies may override)
        assert reduction >= min_reduction, (
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f} "
            f"({reduction:.1%} < {min_reduction:.0%} required)\n"
            f"Inspect training curve: {plot_path}"
        )

        # Warn if anomalies but passed
        if anomalies:
            pytest.warns(UserWarning, match="Anomalies detected")
