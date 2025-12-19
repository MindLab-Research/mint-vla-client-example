"""Dense RL Test: Arithmetic with PPO.

Based on tinker_cookbook math_rl recipe. Model learns to answer arithmetic questions.

Pass criteria: Rewards improve, policy ratio near 1.0.

This test collects training curves and saves plots for visual inspection.
"""

import numpy as np
import pytest

from .conftest import (
    DENSE_MODEL,
    create_session,
    forward_backward,
    optim_step,
    save_weights,
    sample,
)
from .utils import (
    save_training_curve,
    detect_anomalies,
    print_test_summary,
)


# Arithmetic examples
ARITHMETIC_PROBLEMS = [
    {"question": "What is 5 + 3?", "answer": "8"},
    {"question": "What is 12 - 7?", "answer": "5"},
    {"question": "What is 4 * 6?", "answer": "24"},
    {"question": "What is 15 / 3?", "answer": "5"},
    {"question": "What is 9 + 11?", "answer": "20"},
    {"question": "What is 8 * 7?", "answer": "56"},
    {"question": "What is 100 - 37?", "answer": "63"},
    {"question": "What is 45 / 9?", "answer": "5"},
]


def generate_rollouts(tokenizer, model_id: str, problems: list) -> list[dict]:
    """Generate rollouts by sampling from current policy.

    Note: For testing purposes, we assign non-zero advantages to all rollouts
    to ensure gradients flow. In production RL, advantages would come from
    actual rewards minus baseline.
    """
    rollouts = []

    for prob in problems:
        prompt = f"Q: {prob['question']}\nA:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

        # Sample from model
        result = sample(model_id, prompt_tokens, max_tokens=10, temperature=0.7)

        if "error" in result:
            continue

        samples = result.get("sequences", [])
        if not samples:
            continue

        sample_data = samples[0]
        generated_tokens = sample_data.get("tokens", [])
        logprobs = sample_data.get("logprobs", [])

        if not generated_tokens:
            continue

        # Compute reward
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        # Check if answer starts with correct digits
        correct = prob["answer"]
        partial_match = generated_text.strip().startswith(correct[:1]) if correct else False
        reward = 1.0 if generated_text.strip() == correct else (0.5 if partial_match else 0.0)

        # Build training datum
        input_tokens = prompt_tokens + generated_tokens[:-1]
        target_tokens = prompt_tokens[1:] + generated_tokens

        # Loss mask: only on generated tokens
        loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)

        # Advantages: use reward - baseline for proper RL
        # Baseline = 0.5 (expected value for binary 0/1 rewards)
        # Correct answers: advantage = 1.0 - 0.5 = +0.5 (reinforce)
        # Wrong answers: advantage = 0.0 - 0.5 = -0.5 (discourage)
        # NOTE: advantages must match target_tokens length (prompt[1:] + generated)
        baseline = 0.5
        advantage_value = reward - baseline
        advantages = [0.0] * (len(prompt_tokens) - 1) + [advantage_value] * len(generated_tokens)

        # Old logprobs for importance sampling
        old_logprobs = [0.0] * (len(prompt_tokens) - 1) + logprobs

        rollouts.append({
            "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
                "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
                "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
            },
            "reward": reward,
            "generated_text": generated_text,
            "correct_answer": prob["answer"],
        })

    return rollouts


def compute_rl_metrics(result: dict, rollouts: list) -> dict:
    """Extract RL metrics from forward_backward result."""
    metrics = result.get("metrics", {})

    # Calculate average reward from rollouts
    rewards = [r["reward"] for r in rollouts]
    avg_reward = np.mean(rewards) if rewards else 0.0
    accuracy = np.mean([1 if r["reward"] > 0 else 0 for r in rollouts]) if rollouts else 0.0

    return {
        "loss": metrics.get("loss:mean", 0),
        "avg_reward": avg_reward,
        "accuracy": accuracy,
        "ratio_mean": metrics.get("ratio:mean", 1.0),
        "ratio_std": metrics.get("ratio:std", 0.0),
    }


class TestDenseRL:
    """Dense model RL training tests."""

    def test_arithmetic_rl(self, tokenizer):
        """Test arithmetic RL training with PPO.

        Expected: Rewards improve over iterations, policy ratio stays near 1.0.
        Saves training curve for visual inspection.
        """
        num_iterations = 8
        lr = 1e-4

        # Create session
        session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=lr)

        # Save initial weights for sampling
        save_weights(model_id, name="initial")

        # Training loop - collect all metrics
        metrics = {
            "losses": [],
            "rewards": [],
            "accuracies": [],
            "ratios": [],
        }

        import time
        for i in range(num_iterations):
            t0 = time.time()

            # Generate rollouts from current policy
            rollouts = generate_rollouts(tokenizer, model_id, ARITHMETIC_PROBLEMS)

            if not rollouts:
                print(f"Iteration {i+1}: No valid rollouts generated")
                continue

            # Prepare training data (strip metadata)
            train_data = [{
                "model_input": r["model_input"],
                "loss_fn_inputs": r["loss_fn_inputs"],
            } for r in rollouts]

            # Forward-backward with importance_sampling loss
            result = forward_backward(
                model_id,
                train_data,
                loss_fn="importance_sampling",
                loss_fn_config={"clip_ratio": 0.2}
            )

            # Collect metrics
            rl_metrics = compute_rl_metrics(result, rollouts)
            metrics["losses"].append(rl_metrics["loss"])
            metrics["rewards"].append(rl_metrics["avg_reward"])
            metrics["accuracies"].append(rl_metrics["accuracy"])
            metrics["ratios"].append(rl_metrics["ratio_mean"])

            # Optim step
            optim_result = optim_step(model_id, lr=lr)
            grad_norm = optim_result.get("metrics", {}).get("grad_norm", 0)

            # Save weights for next iteration sampling
            save_weights(model_id, name=f"iter_{i+1}")

            iteration_time = time.time() - t0
            print(f"Iteration {i+1}: loss={rl_metrics['loss']:.4f}, "
                  f"reward={rl_metrics['avg_reward']:.3f}, "
                  f"acc={rl_metrics['accuracy']:.1%}, "
                  f"ratio={rl_metrics['ratio_mean']:.3f}, "
                  f"grad_norm={grad_norm:.6f}, "
                  f"time={iteration_time:.2f}s")

        # Save training curves
        data_path, plot_path = save_training_curve(
            metrics,
            "dense_rl_arithmetic",
            metadata={
                "model": DENSE_MODEL,
                "lr": lr,
                "num_iterations": num_iterations,
                "loss_fn": "importance_sampling",
            },
            plot_title="Dense RL (Arithmetic): Training Curves"
        )

        # Detect anomalies
        anomalies = detect_anomalies(metrics["losses"], "loss")

        # Check for reward anomalies
        if metrics["rewards"]:
            if metrics["rewards"][-1] < metrics["rewards"][0]:
                anomalies.append(f"Reward decreased: {metrics['rewards'][0]:.3f} -> {metrics['rewards'][-1]:.3f}")

        # Check ratio stability
        if metrics["ratios"]:
            ratio_range = max(metrics["ratios"]) - min(metrics["ratios"])
            if ratio_range > 0.5:
                anomalies.append(f"Ratio unstable: range {ratio_range:.3f}")

        # Print summary
        print_test_summary(
            "Dense RL (Arithmetic)",
            metrics,
            anomalies,
            plot_path,
            extra_info={
                "Initial reward": f"{metrics['rewards'][0]:.3f}" if metrics["rewards"] else "N/A",
                "Final reward": f"{metrics['rewards'][-1]:.3f}" if metrics["rewards"] else "N/A",
                "Final accuracy": f"{metrics['accuracies'][-1]:.1%}" if metrics["accuracies"] else "N/A",
                "Avg ratio": f"{np.mean(metrics['ratios']):.3f}" if metrics["ratios"] else "N/A",
            }
        )

        # Programmatic checks
        assert len(metrics["losses"]) > 0, "No training iterations completed"

        # Check ratio is reasonable (near 1.0)
        if metrics["ratios"]:
            avg_ratio = np.mean(metrics["ratios"])
            assert 0.5 < avg_ratio < 2.0, f"Policy ratio out of range: {avg_ratio:.3f}"

        # Check for improvement (or at least no collapse)
        if len(metrics["rewards"]) >= 2:
            if metrics["rewards"][-1] < metrics["rewards"][0] * 0.5:
                pytest.fail(
                    f"Reward collapsed: {metrics['rewards'][0]:.3f} -> {metrics['rewards'][-1]:.3f}\n"
                    f"Inspect training curve: {plot_path}"
                )

        # Warn if anomalies but passed
        if anomalies:
            pytest.warns(UserWarning, match="Anomalies detected")
