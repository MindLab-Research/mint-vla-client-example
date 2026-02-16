"""MoE SFT Test: Supervised Fine-tuning (Qwen3-30B-A3B).

Agent-analyzed test:
- Emit a single JSON report + optional plot under `.claude/skills/merge-gate/results/`
- Fail only on structural invariants (API errors, NaN/Inf, missing metrics)
"""

import time

import numpy as np

from .conftest import (
    MOE_MODEL,
    create_session,
    forward_backward,
    optim_step,
    save_weights,
    sample,
)
from .framework import (
    PlotGenerator,
    SessionData,
    TestReport,
    create_test_report,
    print_report_summary,
)


MOE_SFT_EXAMPLES = [
    {"prompt": "What is 2+2?", "response": "4"},
    {"prompt": "What is the color of grass?", "response": "Green"},
    {"prompt": "How many days in a week?", "response": "7"},
    {"prompt": "What is the capital of Japan?", "response": "Tokyo"},
    {"prompt": "What is H2O?", "response": "Water"},
    {"prompt": "What is the opposite of hot?", "response": "Cold"},
]


def prepare_moe_sft_data(tokenizer) -> list[dict]:
    api_data = []
    for ex in MOE_SFT_EXAMPLES:
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


class TestMoESFT:
    def test_moe_sft_training(self, moe_tokenizer):
        num_iterations = 10
        lr = 1e-4

        start_time = time.time()
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=lr)
        api_data = prepare_moe_sft_data(moe_tokenizer)

        session = SessionData(
            session_id=session_id,
            model_id=model_id,
            base_model=MOE_MODEL,
            lora_rank=32,
            learning_rate=lr,
        )

        for i in range(num_iterations):
            t0 = time.time()
            result = forward_backward(model_id, api_data, loss_fn="cross_entropy")
            assert "error" not in result, f"forward_backward error: {result.get('error')}"

            loss = result.get("metrics", {}).get("loss:mean")
            assert loss is not None, "missing metrics.loss:mean"
            loss = float(loss)
            assert np.isfinite(loss), f"non-finite loss: {loss!r}"

            optim_result = optim_step(model_id, lr=lr)
            grad_norm = optim_result.get("metrics", {}).get("grad_norm")
            if grad_norm is None:
                grad_norm = optim_result.get("metrics", {}).get("grad_norm:last", None)
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

        plot = PlotGenerator().training_curve(session, title="MoE SFT (Qwen3-30B-A3B)")
        report = create_test_report(
            test_name="moe_sft_training",
            test_type="training",
            data=session,
            start_time=start_time,
            plots=[plot] if plot else [],
            metadata={
                "base_model": MOE_MODEL,
                "num_iterations": num_iterations,
                "learning_rate": lr,
                "num_examples": len(MOE_SFT_EXAMPLES),
            },
        )
        report_path = report.save()
        print_report_summary(report)
        print(f"report_json={report_path}")

        assert len(session.iterations) == num_iterations, "incomplete training loop"

    def test_moe_sampling_after_train(self, moe_tokenizer):
        lr = 1e-4
        start_time = time.time()

        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=lr)
        api_data = prepare_moe_sft_data(moe_tokenizer)

        losses: list[float] = []
        for i in range(3):
            result = forward_backward(model_id, api_data, loss_fn="cross_entropy")
            assert "error" not in result, f"forward_backward error: {result.get('error')}"
            loss = float(result.get("metrics", {}).get("loss:mean", 0.0))
            assert np.isfinite(loss), f"non-finite loss: {loss!r}"
            losses.append(loss)
            optim_step(model_id, lr=lr)
            print(f"iter={i+1} loss={loss:.4f}")

        save_result = save_weights(model_id, name="moe_merge_gate_sampling")
        assert "error" not in save_result, f"save_weights failed: {save_result.get('error')}"

        prompt = "Q: What is 2+2?\nA:"
        prompt_tokens = moe_tokenizer.encode(prompt, add_special_tokens=True)
        sample_result = sample(model_id, prompt_tokens, max_tokens=10, temperature=0.0)
        assert "error" not in sample_result, f"sample failed: {sample_result.get('error')}"

        samples = sample_result.get("sequences", [])
        assert samples, "no sequences returned"
        generated_tokens = samples[0].get("tokens", [])
        assert generated_tokens, "no generated tokens returned"

        generated_text = moe_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"prompt={prompt!r}")
        print(f"generated_text={generated_text!r}")

        report = TestReport(
            test_id=session_id,
            test_name="moe_sampling_after_train",
            test_type="concurrent",
            timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            duration_seconds=time.time() - start_time,
            data={
                "base_model": MOE_MODEL,
                "train_losses": losses,
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
