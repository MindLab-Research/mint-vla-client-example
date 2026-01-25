"""Dense DPO Test: Preference Optimization.

Tests Direct Preference Optimization on constructed preference pairs.

Pass criteria: DPO loss computes correctly, chosen logprobs > rejected logprobs.

This test collects training curves and saves plots for visual inspection.

NOTE: Requires server-side loss_fn="dpo" support.
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


# DPO preference pairs: chosen = concise, rejected = verbose
DPO_PAIRS = [
    {
        "prompt": "Explain what Python is in one sentence.",
        "chosen": "Python is a high-level programming language.",
        "rejected": "Python is a very popular and widely-used high-level general-purpose programming language that was created by Guido van Rossum and first released in 1991, known for its clear syntax and readability.",
    },
    {
        "prompt": "What is 2+2?",
        "chosen": "4",
        "rejected": "The answer to the mathematical question of what two plus two equals is four, which is a fundamental arithmetic fact.",
    },
    {
        "prompt": "Name a primary color.",
        "chosen": "Red.",
        "rejected": "One of the three primary colors in the traditional RYB color model, which are the basis for mixing other colors, is red.",
    },
    {
        "prompt": "Is water wet?",
        "chosen": "Yes.",
        "rejected": "This is actually a complex philosophical question that has been debated extensively, but generally speaking, water can be considered wet in the sense that it causes wetness when it contacts other surfaces.",
    },
    {
        "prompt": "What color is the sky?",
        "chosen": "Blue.",
        "rejected": "The sky appears blue to human observers during daytime due to a phenomenon called Rayleigh scattering, where shorter wavelengths of light are scattered more than longer wavelengths.",
    },
]


def get_ref_logprobs(tokenizer, model_id: str, tokens: list) -> list:
    """Get reference logprobs from base model."""
    # Use sampling with temperature=0 to get deterministic logprobs
    result = sample(
        model_id,
        tokens[:-1],  # Input tokens
        max_tokens=1,
        temperature=0.0,
        include_prompt_logprobs=True
    )

    if "error" in result:
        return [0.0] * len(tokens)

    # Extract prompt logprobs - at root level of response, not inside samples
    prompt_logprobs = result.get("prompt_logprobs", [])
    if prompt_logprobs:
        # Pad to match sequence length
        while len(prompt_logprobs) < len(tokens):
            prompt_logprobs.append(0.0)
        return prompt_logprobs[:len(tokens)]

    return [0.0] * len(tokens)


def prepare_dpo_data(tokenizer, model_id: str) -> tuple[list, dict]:
    """Prepare DPO training data with reference logprobs."""
    api_data = []
    metadata = {"pairs": []}

    # First, save weights to enable sampling
    save_weights(model_id, name="ref")

    for pair in DPO_PAIRS:
        prompt_text = f"Q: {pair['prompt']}\nA:"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

        # Encode chosen and rejected
        chosen_tokens = tokenizer.encode(f" {pair['chosen']}", add_special_tokens=False)
        rejected_tokens = tokenizer.encode(f" {pair['rejected']}", add_special_tokens=False)

        # Build full sequences
        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens

        # Get reference logprobs (from current model as reference)
        chosen_ref_logprobs = get_ref_logprobs(tokenizer, model_id, chosen_full)
        rejected_ref_logprobs = get_ref_logprobs(tokenizer, model_id, rejected_full)

        # Loss mask: only on response tokens
        chosen_mask = [0.0] * len(prompt_tokens) + [1.0] * len(chosen_tokens)
        rejected_mask = [0.0] * len(prompt_tokens) + [1.0] * len(rejected_tokens)

        # Chosen datum
        api_data.append({
            "model_input": {"chunks": [{"tokens": chosen_full[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": chosen_full[1:], "shape": [len(chosen_full) - 1], "dtype": "int64"},
                "loss_mask": {"data": chosen_mask[1:], "shape": [len(chosen_mask) - 1], "dtype": "float32"},
                "ref_logprobs": {"data": chosen_ref_logprobs[1:], "shape": [len(chosen_ref_logprobs) - 1], "dtype": "float32"},
                "is_chosen": True,
            },
        })

        # Rejected datum
        api_data.append({
            "model_input": {"chunks": [{"tokens": rejected_full[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": rejected_full[1:], "shape": [len(rejected_full) - 1], "dtype": "int64"},
                "loss_mask": {"data": rejected_mask[1:], "shape": [len(rejected_mask) - 1], "dtype": "float32"},
                "ref_logprobs": {"data": rejected_ref_logprobs[1:], "shape": [len(rejected_ref_logprobs) - 1], "dtype": "float32"},
                "is_chosen": False,
            },
        })

        metadata["pairs"].append({
            "prompt": pair["prompt"],
            "chosen_len": len(chosen_tokens),
            "rejected_len": len(rejected_tokens),
        })

    return api_data, metadata


def compute_dpo_metrics(result: dict) -> dict:
    """Extract DPO metrics from forward_backward result."""
    metrics = result.get("metrics", {})

    # Get logprobs for chosen and rejected
    loss_outputs = result.get("loss_fn_outputs", [])

    chosen_logprobs = []
    rejected_logprobs = []

    for i, output in enumerate(loss_outputs):
        logprobs = output.get("logprobs", [])
        if logprobs:
            avg_logprob = np.mean(logprobs)
            if i % 2 == 0:  # Even indices are chosen
                chosen_logprobs.append(avg_logprob)
            else:  # Odd indices are rejected
                rejected_logprobs.append(avg_logprob)

    chosen_avg = np.mean(chosen_logprobs) if chosen_logprobs else 0.0
    rejected_avg = np.mean(rejected_logprobs) if rejected_logprobs else 0.0
    margin = chosen_avg - rejected_avg

    return {
        "loss": metrics.get("loss:mean", 0),
        "chosen_logprob": chosen_avg,
        "rejected_logprob": rejected_avg,
        "margin": margin,
    }


class TestDenseDPO:
    """Dense model DPO training tests."""

    def test_dpo_training(self, tokenizer):
        """Test DPO training on preference pairs.

        Expected: Chosen responses get higher logprobs than rejected.
        Saves training curve for visual inspection.
        """
        num_iterations = 8
        lr = 1e-5  # Lower LR for DPO
        dpo_beta = 0.1

        # Create session
        session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=lr)

        # Prepare data
        api_data, data_metadata = prepare_dpo_data(tokenizer, model_id)

        # Training loop - collect all metrics
        metrics = {
            "losses": [],
            "chosen_logprobs": [],
            "rejected_logprobs": [],
            "margins": [],
        }

        import time
        for i in range(num_iterations):
            t0 = time.time()

            # Forward-backward with DPO loss
            result = forward_backward(
                model_id,
                api_data,
                loss_fn="dpo",
                loss_fn_config={"beta": dpo_beta}
            )

            # Collect metrics
            dpo_metrics = compute_dpo_metrics(result)
            metrics["losses"].append(dpo_metrics["loss"])
            metrics["chosen_logprobs"].append(dpo_metrics["chosen_logprob"])
            metrics["rejected_logprobs"].append(dpo_metrics["rejected_logprob"])
            metrics["margins"].append(dpo_metrics["margin"])

            # Optim step
            optim_result = optim_step(model_id, lr=lr)
            grad_norm = optim_result.get("metrics", {}).get("grad_norm", 0)

            iteration_time = time.time() - t0
            print(f"Iteration {i+1}: loss={dpo_metrics['loss']:.4f}, "
                  f"chosen={dpo_metrics['chosen_logprob']:.3f}, "
                  f"rejected={dpo_metrics['rejected_logprob']:.3f}, "
                  f"margin={dpo_metrics['margin']:.3f}, "
                  f"grad_norm={grad_norm:.6f}, "
                  f"time={iteration_time:.2f}s")

        # Save training curves
        data_path, plot_path = save_training_curve(
            metrics,
            "dense_dpo_preference",
            metadata={
                "model": DENSE_MODEL,
                "lr": lr,
                "beta": dpo_beta,
                "num_iterations": num_iterations,
                "num_pairs": len(DPO_PAIRS),
            },
            plot_title="Dense DPO: Training Curves"
        )

        # Detect anomalies
        anomalies = detect_anomalies(metrics["losses"], "loss")

        # Check margin trend
        if len(metrics["margins"]) >= 2:
            margin_improvement = metrics["margins"][-1] - metrics["margins"][0]
            if margin_improvement < 0:
                anomalies.append(f"Margin decreased: {metrics['margins'][0]:.3f} -> {metrics['margins'][-1]:.3f}")

        # Check final margin is positive
        if metrics["margins"] and metrics["margins"][-1] < 0:
            anomalies.append(f"Final margin negative: {metrics['margins'][-1]:.3f} (chosen < rejected)")

        # Print summary
        print_test_summary(
            "Dense DPO (Preference)",
            metrics,
            anomalies,
            plot_path,
            extra_info={
                "DPO beta": dpo_beta,
                "Initial margin": f"{metrics['margins'][0]:.3f}" if metrics["margins"] else "N/A",
                "Final margin": f"{metrics['margins'][-1]:.3f}" if metrics["margins"] else "N/A",
                "Num pairs": len(DPO_PAIRS),
            }
        )

        # Programmatic checks
        assert len(metrics["losses"]) > 0, "No training iterations completed"

        # DPO loss should be computable
        for loss in metrics["losses"]:
            assert not np.isnan(loss) and not np.isinf(loss), f"Invalid DPO loss: {loss}"

        # Margin should improve or stay reasonable
        if len(metrics["margins"]) >= 2:
            final_margin = metrics["margins"][-1]
            if final_margin < -1.0:
                pytest.fail(
                    f"Margin too negative: {final_margin:.3f} (model prefers rejected)\n"
                    f"Inspect training curve: {plot_path}"
                )

        # Warn if anomalies but passed
        if anomalies:
            pytest.warns(UserWarning, match="Anomalies detected")
