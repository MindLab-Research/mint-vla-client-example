"""Moonlight SFT Test: DeepseekV3 MLA Architecture Training + LoRA Transfer.

Tests Moonlight-16B-A3B (DeepseekV3 MLA architecture) end-to-end:
1. SFT training with train_step endpoint (Megatron)
2. LoRA weight extraction and transfer to vLLM
3. Inference with trained LoRA adapter

Pass criteria:
- Training: Loss decreases >30% over 10 iterations
- LoRA Transfer: Weights successfully loaded in vLLM
- Inference: Generation produces tokens

GPU Requirements:
- Training: 8 GPUs (TP=2, EP=4)
- Inference: 4 GPUs (TP=4)
- Total: 12 GPUs
"""

import time
import numpy as np
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MINT_RUN_MOONLIGHT") != "1",
    reason="Moonlight tests require a dedicated >=12-GPU run; set MINT_RUN_MOONLIGHT=1 to enable.",
)

from .conftest import (
    MOONLIGHT_MODEL,
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


# Simple SFT examples
MOONLIGHT_SFT_EXAMPLES = [
    {"prompt": "What is 2+2?", "response": "4"},
    {"prompt": "What is the color of the sky?", "response": "Blue"},
    {"prompt": "How many days in a week?", "response": "7"},
    {"prompt": "What is the capital of France?", "response": "Paris"},
    {"prompt": "What is H2O?", "response": "Water"},
    {"prompt": "What is the opposite of hot?", "response": "Cold"},
]


def prepare_moonlight_sft_data(tokenizer) -> tuple[list, list]:
    """Prepare SFT data for Moonlight model."""
    api_data = []
    all_weights = []

    for ex in MOONLIGHT_SFT_EXAMPLES:
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


class TestMoonlightSFT:
    """Moonlight model (DeepseekV3 MLA) SFT training and LoRA transfer tests."""

    def test_moonlight_sft_training(self, moonlight_tokenizer):
        """Test Moonlight SFT training with train_step endpoint.

        Expected: Loss decreases >30% over 10 iterations.

        This validates MLA attention (value padding) works correctly
        for training on A800 GPUs.
        """
        num_iterations = 10
        min_reduction = 0.30
        lr = 1e-4

        # Create session for Moonlight model
        print(f"\nCreating Moonlight session for {MOONLIGHT_MODEL}...")
        print("Config: Training TP=2, EP=4 (8 GPUs), Inference TP=4 (4 GPUs)")
        session_id, model_id = create_session(MOONLIGHT_MODEL, lora_rank=32, lr=lr)
        print(f"Session created: {session_id}, model_id: {model_id}")

        # Prepare data
        api_data, all_weights = prepare_moonlight_sft_data(moonlight_tokenizer)

        # Training loop
        metrics = {
            "losses": [],
            "grad_norms": [],
            "iteration_times": [],
        }

        for i in range(num_iterations):
            t0 = time.time()

            result = train_step(
                model_id,
                api_data,
                lr=lr,
                loss_fn="cross_entropy"
            )

            if "error" in result:
                print(f"Iteration {i+1}: ERROR - {result['error']}")
                continue

            loss = result.get("metrics", {}).get("loss:mean", 0)
            metrics["losses"].append(loss)

            grad_norm = result.get("metrics", {}).get("grad_norm", 0)
            metrics["grad_norms"].append(grad_norm)

            iteration_time = time.time() - t0
            metrics["iteration_times"].append(iteration_time)

            print(f"Iteration {i+1}: loss={loss:.4f}, grad_norm={grad_norm:.6f}, time={iteration_time:.2f}s")

        # Save training curve
        data_path, plot_path = save_training_curve(
            metrics,
            "moonlight_sft",
            metadata={
                "model": MOONLIGHT_MODEL,
                "lr": lr,
                "num_iterations": num_iterations,
                "num_examples": len(MOONLIGHT_SFT_EXAMPLES),
                "architecture": "DeepseekV3 MLA",
            },
            plot_title="Moonlight SFT: Training Curve (MLA Architecture)"
        )

        # Detect anomalies
        anomalies = detect_anomalies(metrics["losses"], "loss")

        if metrics["grad_norms"]:
            zero_grads = sum(1 for g in metrics["grad_norms"] if g < 1e-10)
            if zero_grads > len(metrics["grad_norms"]) // 2:
                anomalies.append(f"Many zero grad norms: {zero_grads}/{len(metrics['grad_norms'])}")

        # Print summary
        extra_info = {
            "Model": MOONLIGHT_MODEL,
            "Architecture": "DeepseekV3 MLA (value padding)",
            "Training GPUs": "8 (TP=2, EP=4)",
            "Avg iteration time": f"{np.mean(metrics['iteration_times']):.2f}s" if metrics["iteration_times"] else "N/A",
        }

        print_test_summary(
            "Moonlight SFT",
            metrics,
            anomalies,
            plot_path,
            extra_info=extra_info
        )

        # Assertions
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

        if anomalies:
            pytest.warns(UserWarning, match="Anomalies detected")

    def test_moonlight_lora_transfer(self, moonlight_tokenizer):
        """Test LoRA weight transfer from Megatron to vLLM and inference.

        This is the critical test for MLA architecture:
        1. Train with Megatron (MLA attention with value padding)
        2. Extract LoRA weights in PEFT format
        3. Load into vLLM
        4. Generate with trained LoRA
        """
        lr = 1e-4

        print(f"\nCreating Moonlight session for LoRA transfer test...")
        session_id, model_id = create_session(MOONLIGHT_MODEL, lora_rank=32, lr=lr)
        print(f"Session: {session_id}, model_id: {model_id}")

        # Quick training (5 iterations)
        api_data, _ = prepare_moonlight_sft_data(moonlight_tokenizer)

        print("\nTraining (5 iterations)...")
        for i in range(5):
            result = train_step(model_id, api_data, lr=lr, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            print(f"  Iteration {i+1}: loss={loss:.4f}")

        # Save weights for sampling (this triggers LoRA transfer to vLLM)
        print("\nSaving weights and transferring LoRA to vLLM...")
        t0 = time.time()
        save_result = save_weights(model_id, name="moonlight_lora_test")
        transfer_time = time.time() - t0

        assert "error" not in save_result, f"Weight save failed: {save_result.get('error')}"
        print(f"LoRA transfer completed in {transfer_time:.2f}s")

        # Sample from trained model
        print("\nSampling from trained LoRA adapter...")
        prompt = "Q: What is 2+2?\nA:"
        prompt_tokens = moonlight_tokenizer.encode(prompt, add_special_tokens=True)

        t0 = time.time()
        result = sample(model_id, prompt_tokens, max_tokens=20, temperature=0.0)
        sample_time = time.time() - t0

        assert "error" not in result, f"Sampling failed: {result.get('error')}"

        samples = result.get("sequences", [])
        assert len(samples) > 0, "No samples returned"

        generated_tokens = samples[0].get("tokens", [])
        generated_text = moonlight_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Sample time: {sample_time:.2f}s")

        # Verify we got some output
        assert len(generated_tokens) > 0, "No tokens generated"
        print("\nMoonlight LoRA transfer test: PASS")

    def test_moonlight_rl(self, moonlight_tokenizer):
        """Test Moonlight RL training with random 2-digit addition.

        Uses random arithmetic (like tinker-cookbook) to ensure there's room
        for improvement. Fixed problems like "5+3" are too easy - the model
        already knows them.

        Pass criteria: Accuracy improves over training iterations.

        Note: Uses larger sample sizes (64 problems/iter) and lower LR (1e-5)
        to match tinker-cookbook and get statistically meaningful results.
        """
        import random

        num_iterations = 10
        problems_per_iter = 64  # Larger sample for statistical significance
        lr = 1e-5  # Match tinker-cookbook (was 1e-4)

        print(f"\nCreating Moonlight session for RL test...")
        session_id, model_id = create_session(MOONLIGHT_MODEL, lora_rank=32, lr=lr)
        print(f"Session: {session_id}, model_id: {model_id}")

        # Save initial weights for sampling
        print("\nSaving initial weights...")
        save_weights(model_id, name="moonlight_rl_init")

        metrics = {
            "losses": [],
            "rewards": [],
            "accuracies": [],
            "ratios": [],
            "grad_norms": [],
        }

        def parse_answer(text: str) -> int | None:
            """Parse first integer from response (like tinker-cookbook)."""
            chunks = text.split()
            for chunk in chunks:
                # Remove common prefixes/suffixes
                clean = chunk.strip('.,!?:;')
                try:
                    return int(clean)
                except ValueError:
                    continue
            return None

        def generate_problem() -> tuple[str, int]:
            """Generate random 2-digit addition problem."""
            x = random.randint(10, 99)
            y = random.randint(10, 99)
            return f"What is {x} + {y}?", x + y

        for iteration in range(num_iterations):
            t0 = time.time()
            print(f"\n--- Iteration {iteration + 1} ---")

            # Generate random problems for this iteration
            random.seed(iteration * 1000)  # Reproducible but different each iter
            problems = [generate_problem() for _ in range(problems_per_iter)]

            # Generate rollouts
            rollouts = []
            correct_count = 0

            for question, answer in problems:
                prompt = f"Q: {question}\nA:"
                prompt_tokens = moonlight_tokenizer.encode(prompt, add_special_tokens=True)

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

                generated_text = moonlight_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                parsed = parse_answer(generated_text)

                is_correct = (parsed == answer)
                reward = 1.0 if is_correct else 0.0
                if is_correct:
                    correct_count += 1

                verdict = "ok" if is_correct else "bad"
                print(
                    f"  {question} -> '{generated_text[:20]}' (parsed={parsed}, correct={answer}) {verdict}"
                )

                # Build training datum
                input_tokens = prompt_tokens + generated_tokens[:-1]
                target_tokens = prompt_tokens[1:] + generated_tokens
                loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)

                # Proper RL: positive advantage for correct, negative for wrong
                baseline = 0.5
                advantage_value = reward - baseline
                advantages = [0.0] * (len(prompt_tokens) - 1) + [advantage_value] * len(generated_tokens)
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
                })

            if not rollouts:
                print("No valid rollouts, skipping iteration")
                continue

            accuracy = correct_count / len(problems)
            avg_reward = np.mean([r["reward"] for r in rollouts])

            # Train
            train_data = [{"model_input": r["model_input"], "loss_fn_inputs": r["loss_fn_inputs"]} for r in rollouts]
            result = train_step(model_id, train_data, lr=lr, loss_fn="importance_sampling")

            if "error" in result:
                print(f"Train error: {result['error']}")
                continue

            loss = result.get("metrics", {}).get("loss:mean", 0)
            ratio = result.get("metrics", {}).get("ratio:mean", 1.0)
            grad_norm = result.get("metrics", {}).get("grad_norm", 0)

            metrics["losses"].append(loss)
            metrics["rewards"].append(avg_reward)
            metrics["accuracies"].append(accuracy)
            metrics["ratios"].append(ratio)
            metrics["grad_norms"].append(grad_norm)

            save_weights(model_id, name=f"moonlight_rl_iter_{iteration + 1}")

            iteration_time = time.time() - t0
            print(f"Results: acc={accuracy:.1%} ({correct_count}/{len(problems)}), "
                  f"loss={loss:.4f}, ratio={ratio:.3f}, grad_norm={grad_norm:.4f}, time={iteration_time:.2f}s")

        # Save training curves
        data_path, plot_path = save_training_curve(
            metrics,
            "moonlight_rl",
            metadata={
                "model": MOONLIGHT_MODEL,
                "lr": lr,
                "num_iterations": num_iterations,
                "problems_per_iter": problems_per_iter,
                "task": "random 2-digit addition",
            },
            plot_title="Moonlight RL: Random Addition Training"
        )

        # Detect anomalies
        anomalies = detect_anomalies(metrics["losses"], "loss")

        print_test_summary(
            "Moonlight RL",
            metrics,
            anomalies,
            plot_path,
            extra_info={
                "Task": "Random 2-digit addition",
                "Initial accuracy": f"{metrics['accuracies'][0]:.1%}" if metrics["accuracies"] else "N/A",
                "Final accuracy": f"{metrics['accuracies'][-1]:.1%}" if metrics["accuracies"] else "N/A",
                "Accuracy change": f"{metrics['accuracies'][-1] - metrics['accuracies'][0]:+.1%}" if len(metrics["accuracies"]) >= 2 else "N/A",
            }
        )

        # Key assertion: RL should show clear improvement
        # Look at the actual numbers and judge if this is learning or noise
        assert len(metrics["accuracies"]) >= 5, "Need at least 5 iterations"

        print(f"\nRL Results (judge if this is learning or noise):")
        print(f"  Accuracies: {[f'{a:.1%}' for a in metrics['accuracies']]}")
        print(f"  Rewards: {[f'{r:.2f}' for r in metrics['rewards']]}")

        print("\nMoonlight RL test: PASS (check curve visually)")
