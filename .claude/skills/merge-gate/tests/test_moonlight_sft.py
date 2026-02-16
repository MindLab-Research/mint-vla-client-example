"""Moonlight: MLA SFT + LoRA transfer + RL smoke.

Agent-analyzed tests:
- Emit a single JSON report + optional plot under `.claude/skills/merge-gate/results/`
- Fail only on structural invariants (API errors, NaN/Inf, missing metrics)

GPU target (merge gate):
- Training: 4 GPUs (TP=1, EP=4)
- Inference: 4 GPUs (TP=4)
- Total: 8 GPUs
"""

import os
import random
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MINT_RUN_MOONLIGHT") != "1",
    reason="Moonlight tests require a dedicated >=8-GPU run; set MINT_RUN_MOONLIGHT=1 to enable.",
)

from .conftest import (
    MOONLIGHT_MODEL,
    create_session,
    sample,
    save_weights,
    train_step,
)
from .framework import (
    PlotGenerator,
    SessionData,
    TestReport,
    create_test_report,
    print_report_summary,
)


MOONLIGHT_SFT_EXAMPLES = [
    {"prompt": "What is 2+2?", "response": "4"},
    {"prompt": "What is the color of the sky?", "response": "Blue"},
    {"prompt": "How many days in a week?", "response": "7"},
    {"prompt": "What is the capital of France?", "response": "Paris"},
    {"prompt": "What is H2O?", "response": "Water"},
    {"prompt": "What is the opposite of hot?", "response": "Cold"},
]


def prepare_moonlight_sft_data(tokenizer) -> list[dict]:
    api_data = []
    for ex in MOONLIGHT_SFT_EXAMPLES:
        prompt = f"Q: {ex['prompt']}\nA:"
        response = f" {ex['response']}"

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)

        full_tokens = prompt_tokens + response_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens)

        api_data.append({
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": loss_mask[1:], "shape": [len(loss_mask) - 1], "dtype": "float32"},
            },
        })
    return api_data


def _parse_answer(text: str) -> int | None:
    chunks = text.split()
    for chunk in chunks:
        clean = chunk.strip(".,!?:;")
        try:
            return int(clean)
        except ValueError:
            continue
    return None


def _generate_problem(rng: random.Random) -> tuple[str, int]:
    x = rng.randint(10, 99)
    y = rng.randint(10, 99)
    return f"What is {x} + {y}?", x + y


class TestMoonlight:
    def test_moonlight_sft_training(self, moonlight_tokenizer):
        num_iterations = 10
        lr = 1e-4

        print(f"\nCreating Moonlight session for {MOONLIGHT_MODEL}...")
        print("Config: Training TP=1, EP=4 (4 GPUs), Inference TP=4 (4 GPUs)")

        start_time = time.time()
        session_id, model_id = create_session(MOONLIGHT_MODEL, lora_rank=32, lr=lr)
        api_data = prepare_moonlight_sft_data(moonlight_tokenizer)

        session = SessionData(
            session_id=session_id,
            model_id=model_id,
            base_model=MOONLIGHT_MODEL,
            lora_rank=32,
            learning_rate=lr,
        )

        for i in range(num_iterations):
            t0 = time.time()
            result = train_step(model_id, api_data, lr=lr, loss_fn="cross_entropy")
            assert "error" not in result, f"train_step error: {result.get('error')}"

            loss = result.get("metrics", {}).get("loss:mean")
            assert loss is not None, "missing metrics.loss:mean"
            loss = float(loss)
            assert np.isfinite(loss), f"non-finite loss: {loss!r}"

            grad_norm = result.get("metrics", {}).get("grad_norm", None)
            grad_norm = float(grad_norm) if grad_norm is not None else None

            session.add_iteration(
                iteration=i + 1,
                loss=loss,
                grad_norm=grad_norm,
                wall_time_seconds=time.time() - t0,
            )
            print(
                f"iter={i+1} loss={loss:.4f} grad_norm={grad_norm if grad_norm is not None else 'n/a'} "
                f"dt_s={session.iterations[-1].wall_time_seconds:.2f}"
            )

        plot = PlotGenerator().training_curve(session, title="Moonlight SFT (MLA)")
        report = create_test_report(
            test_name="moonlight_sft_training",
            test_type="training",
            data=session,
            start_time=start_time,
            plots=[plot] if plot else [],
            metadata={
                "base_model": MOONLIGHT_MODEL,
                "learning_rate": lr,
                "num_iterations": num_iterations,
            },
        )
        report_path = report.save()
        print_report_summary(report)
        print(f"report_json={report_path}")

        assert len(session.iterations) == num_iterations, "incomplete training loop"

    def test_moonlight_lora_transfer(self, moonlight_tokenizer):
        lr = 1e-4
        start_time = time.time()

        session_id, model_id = create_session(MOONLIGHT_MODEL, lora_rank=32, lr=lr)
        api_data = prepare_moonlight_sft_data(moonlight_tokenizer)

        losses: list[float] = []
        for i in range(5):
            result = train_step(model_id, api_data, lr=lr, loss_fn="cross_entropy")
            assert "error" not in result, f"train_step error: {result.get('error')}"
            loss = float(result.get("metrics", {}).get("loss:mean", 0.0))
            assert np.isfinite(loss), f"non-finite loss: {loss!r}"
            losses.append(loss)
            print(f"iter={i+1} loss={loss:.4f}")

        t0 = time.time()
        save_result = save_weights(model_id, name="moonlight_lora_transfer")
        transfer_time = time.time() - t0
        assert "error" not in save_result, f"save_weights failed: {save_result.get('error')}"

        prompt = "Q: What is 2+2?\nA:"
        prompt_tokens = moonlight_tokenizer.encode(prompt, add_special_tokens=True)
        t0 = time.time()
        sample_result = sample(model_id, prompt_tokens, max_tokens=20, temperature=0.0)
        sample_time = time.time() - t0
        assert "error" not in sample_result, f"sample failed: {sample_result.get('error')}"

        samples = sample_result.get("sequences", [])
        assert samples, "no sequences returned"
        generated_tokens = samples[0].get("tokens", [])
        assert generated_tokens, "no generated tokens returned"
        generated_text = moonlight_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        report = TestReport(
            test_id=session_id,
            test_name="moonlight_lora_transfer",
            test_type="checkpoint",
            timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            duration_seconds=time.time() - start_time,
            data={
                "base_model": MOONLIGHT_MODEL,
                "train_losses": losses,
                "transfer_time_s": transfer_time,
                "sample_time_s": sample_time,
                "prompt": prompt,
                "generated_text": generated_text,
                "generated_token_count": len(generated_tokens),
            },
            plots=[],
            anomalies=[],
            metadata={},
        )
        report_path = report.save()
        print(f"report_json={report_path}")

    def test_moonlight_rl_smoke(self, moonlight_tokenizer):
        num_iterations = 10
        problems_per_iter = 64
        lr = 1e-5

        start_time = time.time()
        session_id, model_id = create_session(MOONLIGHT_MODEL, lora_rank=32, lr=lr)

        save_weights(model_id, name="moonlight_rl_init")

        session = SessionData(
            session_id=session_id,
            model_id=model_id,
            base_model=MOONLIGHT_MODEL,
            lora_rank=32,
            learning_rate=lr,
        )

        for iteration in range(num_iterations):
            t0 = time.time()
            rng = random.Random(iteration * 1000)
            problems = [_generate_problem(rng) for _ in range(problems_per_iter)]

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
                parsed = _parse_answer(generated_text)
                is_correct = (parsed == answer)
                reward = 1.0 if is_correct else 0.0
                if is_correct:
                    correct_count += 1

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

            if not rollouts:
                continue

            accuracy = correct_count / len(problems)
            avg_reward = float(np.mean([r["reward"] for r in rollouts]))

            train_data = [{"model_input": r["model_input"], "loss_fn_inputs": r["loss_fn_inputs"]} for r in rollouts]
            result = train_step(model_id, train_data, lr=lr, loss_fn="importance_sampling")
            assert "error" not in result, f"train_step error: {result.get('error')}"

            loss = float(result.get("metrics", {}).get("loss:mean", 0.0))
            ratio = float(result.get("metrics", {}).get("ratio:mean", 1.0))
            grad_norm = result.get("metrics", {}).get("grad_norm", None)
            grad_norm = float(grad_norm) if grad_norm is not None else None

            assert np.isfinite(loss), f"non-finite loss: {loss!r}"
            assert np.isfinite(ratio), f"non-finite ratio: {ratio!r}"

            session.add_iteration(
                iteration=iteration + 1,
                loss=loss,
                reward=avg_reward,
                accuracy=accuracy,
                ratio_mean=ratio,
                grad_norm=grad_norm,
                wall_time_seconds=time.time() - t0,
            )

            save_weights(model_id, name=f"moonlight_rl_iter_{iteration + 1}")

        plot = PlotGenerator().rl_training_curves(session, title="Moonlight RL Smoke (Random Addition)")
        report = create_test_report(
            test_name="moonlight_rl_smoke",
            test_type="training",
            data=session,
            start_time=start_time,
            plots=[plot] if plot else [],
            metadata={
                "base_model": MOONLIGHT_MODEL,
                "learning_rate": lr,
                "num_iterations": num_iterations,
                "problems_per_iter": problems_per_iter,
                "task": "random 2-digit addition",
            },
        )
        report_path = report.save()
        print_report_summary(report)
        print(f"report_json={report_path}")

        assert len(session.iterations) >= 1, "no iterations produced RL metrics"
