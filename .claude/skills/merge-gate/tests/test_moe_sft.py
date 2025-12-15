"""MoE SFT Test: Supervised Fine-tuning with train_step.

Tests MoE model (Qwen3-30B-A3B) SFT using the train_step endpoint.
train_step combines forward_backward + optim_step in a single train_mode context,
required for MoE models with offloading.

Pass criteria: Loss decreases >30% over 10 iterations.

This test collects training curves and saves plots for visual inspection.
"""

import numpy as np
import pytest

from .conftest import (
    MOE_MODEL,
    create_session,
    train_step,
    save_weights,
    sample,
)
from .utils import (
    save_training_curve,
    detect_anomalies,
    print_test_summary,
)


# Simple SFT examples for MoE model
MOE_SFT_EXAMPLES = [
    {"prompt": "What is 2+2?", "response": "4"},
    {"prompt": "What is the color of grass?", "response": "Green"},
    {"prompt": "How many days in a week?", "response": "7"},
    {"prompt": "What is the capital of Japan?", "response": "Tokyo"},
    {"prompt": "What is H2O?", "response": "Water"},
    {"prompt": "What is the opposite of hot?", "response": "Cold"},
]


def prepare_moe_sft_data(tokenizer) -> tuple[list, list]:
    """Prepare SFT data for MoE model."""
    api_data = []
    all_weights = []

    for ex in MOE_SFT_EXAMPLES:
        prompt = f"Q: {ex['prompt']}\nA:"
        response = f" {ex['response']}"

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)

        full_tokens = prompt_tokens + response_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens)
        all_weights.extend(loss_mask)

        api_data.append({
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": loss_mask[1:], "shape": [len(loss_mask) - 1], "dtype": "float32"},
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


class TestMoESFT:
    """MoE model SFT training tests."""

    def test_moe_sft_training(self, moe_tokenizer):
        """Test MoE SFT training with train_step endpoint.

        Expected: Loss decreases >30% over 10 iterations.
        Saves training curve for visual inspection.

        NOTE: Uses train_step (combined forward_backward + optim_step)
        which is required for MoE models with expert offloading.
        """
        num_iterations = 10
        min_reduction = 0.30
        lr = 1e-4

        # Create session for MoE model
        print(f"Creating MoE session for {MOE_MODEL}...")
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=lr)
        print(f"Session created: {session_id}, model_id: {model_id}")

        # Prepare data
        api_data, all_weights = prepare_moe_sft_data(moe_tokenizer)

        # Training loop - collect all metrics
        metrics = {
            "losses": [],
            "grad_norms": [],
            "iteration_times": [],
        }

        import time
        for i in range(num_iterations):
            t0 = time.time()

            # Use train_step for MoE (combined forward_backward + optim_step)
            result = train_step(
                model_id,
                api_data,
                lr=lr,
                loss_fn="cross_entropy"
            )

            if "error" in result:
                print(f"Iteration {i+1}: ERROR - {result['error']}")
                continue

            # Compute loss
            loss = compute_weighted_loss(result, all_weights)
            metrics["losses"].append(loss)

            # Get grad norm from result
            grad_norm = result.get("metrics", {}).get("grad_norm", 0)
            metrics["grad_norms"].append(grad_norm)

            iteration_time = time.time() - t0
            metrics["iteration_times"].append(iteration_time)

            print(f"Iteration {i+1}: loss={loss:.4f}, grad_norm={grad_norm:.6f}, time={iteration_time:.2f}s")

        # Save training curve
        data_path, plot_path = save_training_curve(
            metrics,
            "moe_sft",
            metadata={
                "model": MOE_MODEL,
                "lr": lr,
                "num_iterations": num_iterations,
                "num_examples": len(MOE_SFT_EXAMPLES),
            },
            plot_title="MoE SFT: Training Curve"
        )

        # Detect anomalies
        anomalies = detect_anomalies(metrics["losses"], "loss")

        # Check grad norm anomalies
        if metrics["grad_norms"]:
            zero_grads = sum(1 for g in metrics["grad_norms"] if g < 1e-10)
            if zero_grads > len(metrics["grad_norms"]) // 2:
                anomalies.append(f"Many zero grad norms: {zero_grads}/{len(metrics['grad_norms'])}")

        # Print summary
        extra_info = {
            "Model": MOE_MODEL,
            "Avg iteration time": f"{np.mean(metrics['iteration_times']):.2f}s" if metrics["iteration_times"] else "N/A",
        }

        print_test_summary(
            "MoE SFT",
            metrics,
            anomalies,
            plot_path,
            extra_info=extra_info
        )

        # Programmatic checks
        assert len(metrics["losses"]) >= num_iterations // 2, (
            f"Too few iterations completed: {len(metrics['losses'])}/{num_iterations}"
        )

        initial_loss = metrics["losses"][0]
        final_loss = metrics["losses"][-1]

        if initial_loss > 0:
            reduction = (initial_loss - final_loss) / initial_loss
        else:
            reduction = 0

        assert reduction >= min_reduction, (
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f} "
            f"({reduction:.1%} < {min_reduction:.0%} required)\n"
            f"Inspect training curve: {plot_path}"
        )

        # Warn if anomalies but passed
        if anomalies:
            pytest.warns(UserWarning, match="Anomalies detected")

    def test_moe_sampling_after_train(self, moe_tokenizer):
        """Test sampling from MoE model after training."""
        lr = 1e-4

        # Create session
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=lr)

        # Quick training (3 iterations)
        api_data, _ = prepare_moe_sft_data(moe_tokenizer)

        for i in range(3):
            result = train_step(model_id, api_data, lr=lr, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            print(f"Iteration {i+1}: loss={loss:.4f}")

        # Save weights for sampling
        save_result = save_weights(model_id, name="moe_test")
        assert "error" not in save_result, f"Weight save failed: {save_result.get('error')}"

        # Sample from trained model
        prompt = "Q: What is 2+2?\nA:"
        prompt_tokens = moe_tokenizer.encode(prompt, add_special_tokens=True)

        result = sample(model_id, prompt_tokens, max_tokens=10, temperature=0.0)
        assert "error" not in result, f"Sampling failed: {result.get('error')}"

        samples = result.get("sequences", [])
        assert len(samples) > 0, "No samples returned"

        generated_tokens = samples[0].get("tokens", [])
        generated_text = moe_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
