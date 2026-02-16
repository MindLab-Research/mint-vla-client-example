"""Dense RL Test: Arithmetic with PPO-style importance_sampling.

Agent-analyzed test:
- Emit a single JSON report + optional plot under `.claude/skills/merge-gate/results/`
- Fail only on structural invariants (API errors, NaN/Inf, missing metrics)
"""

import time

import numpy as np

from .conftest import (
    DENSE_MODEL,
    create_session,
    forward_backward,
    optim_step,
    save_weights,
    sample,
)
from .framework import (
    PlotGenerator,
    SessionData,
    create_test_report,
    print_report_summary,
)


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


def generate_rollouts(tokenizer, model_id: str, problems: list[dict]) -> list[dict]:
    rollouts = []
    for prob in problems:
        prompt = f"Q: {prob['question']}\nA:"
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
        correct = prob["answer"]
        partial_match = generated_text.strip().startswith(correct[:1]) if correct else False
        reward = 1.0 if generated_text.strip() == correct else (0.5 if partial_match else 0.0)

        input_tokens = prompt_tokens + generated_tokens[:-1]
        target_tokens = prompt_tokens[1:] + generated_tokens
        loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(generated_tokens)

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

    return rollouts


class TestDenseRL:
    def test_arithmetic_rl(self, tokenizer):
        num_iterations = 8
        lr = 1e-4

        start_time = time.time()
        session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=lr)

        save_weights(model_id, name="dense_rl_init")

        session = SessionData(
            session_id=session_id,
            model_id=model_id,
            base_model=DENSE_MODEL,
            lora_rank=32,
            learning_rate=lr,
        )

        for i in range(num_iterations):
            t0 = time.time()

            rollouts = generate_rollouts(tokenizer, model_id, ARITHMETIC_PROBLEMS)
            if not rollouts:
                continue

            train_data = [{"model_input": r["model_input"], "loss_fn_inputs": r["loss_fn_inputs"]} for r in rollouts]
            rewards = [float(r["reward"]) for r in rollouts]
            avg_reward = float(np.mean(rewards)) if rewards else 0.0
            accuracy = float(np.mean([1.0 if r > 0.9 else 0.0 for r in rewards])) if rewards else 0.0

            result = forward_backward(
                model_id,
                train_data,
                loss_fn="importance_sampling",
                loss_fn_config={"clip_ratio": 0.2},
            )
            assert "error" not in result, f"forward_backward error: {result.get('error')}"

            loss = float(result.get("metrics", {}).get("loss:mean", 0.0))
            ratio = float(result.get("metrics", {}).get("ratio:mean", 1.0))
            assert np.isfinite(loss), f"non-finite loss: {loss!r}"
            assert np.isfinite(ratio), f"non-finite ratio: {ratio!r}"

            optim_result = optim_step(model_id, lr=lr)
            grad_norm = optim_result.get("metrics", {}).get("grad_norm")
            if grad_norm is None:
                grad_norm = optim_result.get("metrics", {}).get("grad_norm:last", None)
            grad_norm = float(grad_norm) if grad_norm is not None else None

            save_weights(model_id, name=f"dense_rl_iter_{i+1}")

            session.add_iteration(
                iteration=i + 1,
                loss=loss,
                reward=avg_reward,
                accuracy=accuracy,
                ratio_mean=ratio,
                grad_norm=grad_norm,
                wall_time_seconds=time.time() - t0,
            )

            print(
                f"iter={i+1} loss={loss:.4f} reward={avg_reward:.3f} acc={accuracy:.3f} ratio={ratio:.3f} "
                f"grad_norm={grad_norm if grad_norm is not None else 'n/a'} dt_s={session.iterations[-1].wall_time_seconds:.2f}"
            )

        plot = PlotGenerator().rl_training_curves(session, title="Dense RL (Arithmetic)")
        report = create_test_report(
            test_name="dense_rl_arithmetic",
            test_type="training",
            data=session,
            start_time=start_time,
            plots=[plot] if plot else [],
            metadata={
                "base_model": DENSE_MODEL,
                "learning_rate": lr,
                "num_iterations": num_iterations,
                "num_problems": len(ARITHMETIC_PROBLEMS),
            },
        )
        report_path = report.save()
        print_report_summary(report)
        print(f"report_json={report_path}")

        assert len(session.iterations) >= 1, "no RL iterations completed"
