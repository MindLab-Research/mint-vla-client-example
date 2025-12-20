#!/usr/bin/env python3
"""Standalone Moonlight RL test - verify RL actually works."""

import os
import sys
import time
import random
import requests

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

MOONLIGHT_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"
BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def create_session(model, lora_rank=32, lr=1e-5):
    """Create training session."""
    import uuid
    session_id = f"rl_test_{uuid.uuid4().hex[:8]}"
    resp = requests.post(f"{BASE_URL}/api/v1/create_model", json={
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")
    return session_id, result.get("model_id")


def save_weights(model_id, name=None):
    """Save weights for sampling."""
    resp = requests.post(f"{BASE_URL}/api/v1/save_weights", json={
        "model_id": model_id,
        "name": name,
    }, timeout=300)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def poll_future(request_id, timeout=300):
    """Poll for async result."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", json={
            "request_id": request_id
        }, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(1)
    raise TimeoutError(f"Request {request_id} timed out after {timeout}s")


def sample(model_id, tokens, max_tokens=10, temperature=0.7):
    """Sample from model."""
    resp = requests.post(f"{BASE_URL}/api/v1/asample", json={
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
        "num_samples": 1,
    }, timeout=60)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=120)


def train_step(model_id, data, lr=1e-5, loss_fn="importance_sampling"):
    """Execute train step."""
    resp = requests.post(f"{BASE_URL}/api/v1/train_step", json={
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": loss_fn},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }, timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def parse_answer(text: str) -> int | None:
    """Parse first integer from response."""
    for chunk in text.split():
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


def main():
    print("=" * 60)
    print("Moonlight RL Test - Verify RL Actually Works")
    print("=" * 60)

    # Parameters matching tinker-cookbook
    num_iterations = 10
    problems_per_iter = 64
    lr = 1e-5

    print(f"\nParameters:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Problems/iter: {problems_per_iter}")
    print(f"  Learning rate: {lr}")
    print(f"  Total samples: {num_iterations * problems_per_iter}")

    # Load tokenizer
    print(f"\nLoading tokenizer for {MOONLIGHT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MOONLIGHT_MODEL, trust_remote_code=True)

    # Create session
    print(f"\nCreating session...")
    session_id, model_id = create_session(MOONLIGHT_MODEL, lora_rank=32, lr=lr)
    print(f"Session: {session_id}, model_id: {model_id}")

    # Save initial weights
    print("\nSaving initial weights for sampling...")
    save_weights(model_id, name="moonlight_rl_init")

    accuracies = []
    rewards = []

    for iteration in range(num_iterations):
        t0 = time.time()
        print(f"\n{'='*40}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*40}")

        # Generate problems
        random.seed(iteration * 1000)
        problems = [generate_problem() for _ in range(problems_per_iter)]

        # Generate rollouts
        rollouts = []
        correct_count = 0

        for i, (question, answer) in enumerate(problems):
            prompt = f"Q: {question}\nA:"
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

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

            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            parsed = parse_answer(generated_text)

            is_correct = (parsed == answer)
            reward = 1.0 if is_correct else 0.0
            if is_correct:
                correct_count += 1

            # Show some examples
            if i < 5:
                mark = "✓" if is_correct else "✗"
                print(f"  {question} -> '{generated_text[:15]}' (parsed={parsed}, correct={answer}) {mark}")

            # Build training datum
            input_tokens = prompt_tokens + generated_tokens[:-1]
            target_tokens = prompt_tokens[1:] + generated_tokens
            loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)

            # Proper advantage: +0.5 for correct, -0.5 for wrong
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
            })

        if not rollouts:
            print("  No valid rollouts, skipping")
            continue

        accuracy = correct_count / len(problems)
        avg_reward = sum(1 if r["loss_fn_inputs"]["advantages"]["data"][-1] > 0 else 0 for r in rollouts) / len(rollouts)

        # Train
        train_data = [{"model_input": r["model_input"], "loss_fn_inputs": r["loss_fn_inputs"]} for r in rollouts]
        result = train_step(model_id, train_data, lr=lr, loss_fn="importance_sampling")

        if "error" in result:
            print(f"  Train error: {result['error']}")
            continue

        loss = result.get("metrics", {}).get("loss:mean", 0)
        ratio = result.get("metrics", {}).get("ratio:mean", 1.0)

        accuracies.append(accuracy)
        rewards.append(avg_reward)

        # Save weights for next iteration sampling
        save_weights(model_id, name=f"moonlight_rl_iter_{iteration + 1}")

        elapsed = time.time() - t0
        print(f"\n  Results: acc={accuracy:.1%} ({correct_count}/{len(problems)}), loss={loss:.4f}, ratio={ratio:.3f}, time={elapsed:.1f}s")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nAccuracies by iteration:")
    for i, acc in enumerate(accuracies):
        bar = "█" * int(acc * 40)
        print(f"  Iter {i+1:2d}: {acc:5.1%} {bar}")

    if len(accuracies) >= 2:
        print(f"\nFirst accuracy:  {accuracies[0]:.1%}")
        print(f"Last accuracy:   {accuracies[-1]:.1%}")
        print(f"Change:          {accuracies[-1] - accuracies[0]:+.1%}")

    print("\n" + "=" * 60)
    print("JUDGMENT: Does this look like learning or noise?")
    print("=" * 60)


if __name__ == "__main__":
    main()
