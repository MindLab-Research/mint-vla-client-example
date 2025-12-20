#!/usr/bin/env python
"""Full RL Training Test for Moonlight-16B-A3B.

This script runs a complete RL training loop similar to tinker-cookbook math_rl:
1. Sample rollouts from current policy (vLLM)
2. Compute rewards based on arithmetic correctness
3. Train with importance_sampling loss (Megatron)
4. Repeat for multiple iterations
5. Track reward curve improvement

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_moonlight_full_rl.py
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import requests
from transformers import AutoTokenizer

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"

# Arithmetic problems (similar to tinker-cookbook arithmetic_env)
ARITHMETIC_PROBLEMS = [
    # Addition
    {"question": "What is 5 + 3?", "answer": "8"},
    {"question": "What is 12 + 7?", "answer": "19"},
    {"question": "What is 9 + 11?", "answer": "20"},
    {"question": "What is 25 + 17?", "answer": "42"},
    {"question": "What is 8 + 6?", "answer": "14"},
    {"question": "What is 15 + 9?", "answer": "24"},
    {"question": "What is 23 + 19?", "answer": "42"},
    {"question": "What is 7 + 8?", "answer": "15"},
    # Subtraction
    {"question": "What is 12 - 7?", "answer": "5"},
    {"question": "What is 20 - 8?", "answer": "12"},
    {"question": "What is 15 - 6?", "answer": "9"},
    {"question": "What is 30 - 12?", "answer": "18"},
    # Multiplication
    {"question": "What is 4 * 6?", "answer": "24"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "What is 5 * 9?", "answer": "45"},
    {"question": "What is 6 * 7?", "answer": "42"},
    # Division
    {"question": "What is 15 / 3?", "answer": "5"},
    {"question": "What is 24 / 6?", "answer": "4"},
    {"question": "What is 45 / 9?", "answer": "5"},
    {"question": "What is 56 / 8?", "answer": "7"},
]


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
            continue
        else:
            resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_session(model: str, lora_rank: int = 32, lr: float = 1e-4) -> tuple[str, str]:
    """Create training session."""
    import uuid
    session_id = str(uuid.uuid4())

    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    resp = requests.post(url, json=payload, timeout=900)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=600)

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    return session_id, result.get("model_id")


def save_weights(model_id: str, name: str = "checkpoint") -> dict:
    """Save weights for sampling."""
    url = f"{BASE_URL}/api/v1/save_weights"
    payload = {"model_id": model_id, "name": name}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def sample(model_id: str, prompt_tokens: list, max_tokens: int = 10, temperature: float = 0.7) -> dict:
    """Sample from model."""
    url = f"{BASE_URL}/api/v1/asample"
    payload = {
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
        "num_samples": 1,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=120)


def train_step(model_id: str, data: list, lr: float = 1e-4, loss_fn: str = "importance_sampling") -> dict:
    """Combined forward_backward + optim_step."""
    url = f"{BASE_URL}/api/v1/train_step"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": loss_fn},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def extract_answer(text: str) -> str:
    """Extract the numeric answer from generated text."""
    # Try to find the first number in the response
    text = text.strip()

    # If text starts with the answer directly
    for word in text.split():
        # Remove punctuation
        word = word.strip(".,!?;:")
        if word.isdigit():
            return word
        # Try negative numbers
        if word.startswith("-") and word[1:].isdigit():
            return word

    return text.split()[0] if text else ""


def generate_rollouts(tokenizer, model_id: str, problems: list, group_size: int = 4) -> list:
    """Generate rollouts by sampling from current policy."""
    rollouts = []

    # Sample group_size times for each problem
    for prob in problems:
        prompt = f"Q: {prob['question']}\nA:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

        for _ in range(group_size):
            result = sample(model_id, prompt_tokens, max_tokens=10, temperature=0.7)

            if "error" in result:
                continue

            samples = result.get("sequences", [])
            if not samples:
                continue

            sample_data = samples[0]
            generated_tokens = sample_data.get("tokens", [])
            logprobs = sample_data.get("logprobs", [])

            if not generated_tokens or not logprobs:
                continue

            # Decode and compute reward
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            extracted = extract_answer(generated_text)
            correct = prob["answer"]
            reward = 1.0 if extracted == correct else 0.0

            # Build training datum
            input_tokens = prompt_tokens + generated_tokens[:-1]
            target_tokens = prompt_tokens[1:] + generated_tokens

            # Loss mask: only on generated tokens
            loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)

            # Advantages with baseline subtraction (use mean reward as baseline)
            advantage = reward - 0.5  # Simple baseline
            advantages = [0.0] * (len(prompt_tokens) - 1) + [advantage] * len(generated_tokens)

            # Old logprobs
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
                "question": prob["question"],
                "answer": prob["answer"],
                "generated": generated_text,
                "extracted": extracted,
            })

    return rollouts


def main():
    print("=" * 70)
    print("MOONLIGHT FULL RL TRAINING TEST")
    print("=" * 70)
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL}")

    # Config
    num_iterations = 10
    group_size = 2  # Samples per problem per iteration
    batch_size = 10  # Problems per batch
    lr = 1e-4
    lora_rank = 32

    print(f"\nConfig:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Group size: {group_size}")
    print(f"  Batch size: {batch_size} problems")
    print(f"  Learning rate: {lr}")
    print(f"  LoRA rank: {lora_rank}")

    # Check server health
    print("\nChecking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=10)
        print(f"Server status: {resp.json().get('status')}")
    except Exception as e:
        print(f"Server not available: {e}")
        return 1

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Create session
    print("\nCreating training session...")
    t0 = time.time()
    session_id, model_id = create_session(MODEL, lora_rank=lora_rank, lr=lr)
    init_time = time.time() - t0
    print(f"Session created in {init_time:.2f}s: {model_id}")

    # Save initial weights
    print("\nSaving initial weights for sampling...")
    save_weights(model_id, name="init")

    # Track metrics
    all_rewards = []
    all_accuracies = []
    all_losses = []
    all_ratios = []

    print("\n" + "=" * 70)
    print("TRAINING LOOP")
    print("=" * 70)

    for iteration in range(num_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Sample batch of problems
        batch_problems = random.sample(ARITHMETIC_PROBLEMS, min(batch_size, len(ARITHMETIC_PROBLEMS)))

        # Generate rollouts
        print(f"Generating rollouts ({len(batch_problems)} problems × {group_size} samples)...")
        t0 = time.time()
        rollouts = generate_rollouts(tokenizer, model_id, batch_problems, group_size=group_size)
        sample_time = time.time() - t0

        if not rollouts:
            print("No valid rollouts generated, skipping")
            continue

        # Compute metrics
        rewards = [r["reward"] for r in rollouts]
        avg_reward = np.mean(rewards)
        accuracy = np.mean([1 if r["reward"] > 0 else 0 for r in rollouts])

        # Show some examples
        print(f"Sample rollouts:")
        for r in rollouts[:3]:
            status = "✓" if r["reward"] > 0 else "✗"
            print(f"  {status} Q: {r['question']} -> '{r['extracted']}' (correct: {r['answer']})")

        # Train
        print(f"Training on {len(rollouts)} rollouts...")
        train_data = [{
            "model_input": r["model_input"],
            "loss_fn_inputs": r["loss_fn_inputs"],
        } for r in rollouts]

        t0 = time.time()
        result = train_step(model_id, train_data, lr=lr, loss_fn="importance_sampling")
        train_time = time.time() - t0

        if "error" in result:
            print(f"Train error: {result['error']}")
            continue

        loss = result.get("metrics", {}).get("loss:mean", 0)
        ratio = result.get("metrics", {}).get("ratio:mean", 1.0)
        grad_norm = result.get("metrics", {}).get("grad_norm", 0)

        # Save weights for next iteration
        save_weights(model_id, name=f"iter_{iteration + 1}")

        # Record metrics
        all_rewards.append(avg_reward)
        all_accuracies.append(accuracy)
        all_losses.append(loss)
        all_ratios.append(ratio)

        iter_time = time.time() - iter_start
        print(f"Results: reward={avg_reward:.3f}, acc={accuracy:.1%}, loss={loss:.4f}, "
              f"ratio={ratio:.3f}, grad_norm={grad_norm:.4f}")
        print(f"Timing: sample={sample_time:.1f}s, train={train_time:.1f}s, total={iter_time:.1f}s")

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    if all_rewards:
        print(f"\nReward Curve:")
        for i, r in enumerate(all_rewards):
            bar = "█" * int(r * 20)
            print(f"  Iter {i+1:2d}: {r:.3f} {bar}")

        print(f"\nMetrics:")
        print(f"  Initial reward: {all_rewards[0]:.3f}")
        print(f"  Final reward:   {all_rewards[-1]:.3f}")
        print(f"  Reward change:  {all_rewards[-1] - all_rewards[0]:+.3f}")
        print(f"  Initial acc:    {all_accuracies[0]:.1%}")
        print(f"  Final acc:      {all_accuracies[-1]:.1%}")
        print(f"  Avg ratio:      {np.mean(all_ratios):.3f}")

        # Check for improvement
        if all_rewards[-1] > all_rewards[0]:
            print("\n✓ REWARD IMPROVED - Training is working!")
        elif all_rewards[-1] >= all_rewards[0] * 0.9:
            print("\n~ REWARD STABLE - Training is functional")
        else:
            print("\n✗ REWARD DECREASED - Potential issue")

        # Save results
        results = {
            "model": MODEL,
            "config": {
                "num_iterations": num_iterations,
                "group_size": group_size,
                "batch_size": batch_size,
                "lr": lr,
                "lora_rank": lora_rank,
            },
            "rewards": all_rewards,
            "accuracies": all_accuracies,
            "losses": all_losses,
            "ratios": all_ratios,
        }

        results_path = "/tmp/moonlight_rl_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
