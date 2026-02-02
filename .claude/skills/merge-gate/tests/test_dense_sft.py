"""Dense SFT Test: Pig Latin Translation.

Agent-analyzed test:
- Collect full per-iteration metrics
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
)
from .framework import (
    PlotGenerator,
    SessionData,
    create_test_report,
    print_report_summary,
)


PIG_LATIN_EXAMPLES = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]


def prepare_pig_latin_data(tokenizer) -> list[dict]:
    api_data = []
    for ex in PIG_LATIN_EXAMPLES:
        prompt = f"English: {ex['input']}\nPig Latin:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(f" {ex['output']}\n\n", add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        loss_mask = weights[1:]

        api_data.append({
            "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
            },
        })
    return api_data


class TestDenseSFT:
    def test_pig_latin_training(self, tokenizer):
        num_iterations = 10
        lr = 1e-4

        start_time = time.time()
        session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=lr)
        api_data = prepare_pig_latin_data(tokenizer)

        session = SessionData(
            session_id=session_id,
            model_id=model_id,
            base_model=DENSE_MODEL,
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

        plot = PlotGenerator().training_curve(
            session,
            title="Dense SFT (Pig Latin)",
        )

        report = create_test_report(
            test_name="dense_sft_pig_latin",
            test_type="training",
            data=session,
            start_time=start_time,
            plots=[plot] if plot else [],
            metadata={
                "base_model": DENSE_MODEL,
                "num_iterations": num_iterations,
                "learning_rate": lr,
            },
        )
        report_path = report.save()
        print_report_summary(report)
        print(f"report_json={report_path}")

        assert len(session.iterations) == num_iterations, "incomplete training loop"
