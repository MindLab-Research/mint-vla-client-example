"""Dense DPO test via the SDK custom-loss path."""

from __future__ import annotations

import numpy as np
import pytest
import tinker
import torch
import torch.nn.functional as F

from .conftest import API_KEY, BASE_URL, DENSE_MODEL
from .utils import detect_anomalies, print_test_summary, save_training_curve


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


def _tensor(values: list[int] | list[float], dtype: str) -> tinker.types.TensorData:
    return tinker.types.TensorData(data=list(values), shape=[len(values)], dtype=dtype)


def _sdk_api_key(api_key: str) -> str:
    return api_key if api_key.startswith("tml-") else f"tml-{api_key}"


def _prepare_dpo_data(tokenizer) -> tuple[list[tinker.types.Datum], list[tinker.types.ModelInput], dict]:
    data: list[tinker.types.Datum] = []
    full_sequences: list[tinker.types.ModelInput] = []
    metadata = {"pairs": []}

    for pair in DPO_PAIRS:
        prompt_text = f"Q: {pair['prompt']}\nA:"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        chosen_tokens = tokenizer.encode(f" {pair['chosen']}", add_special_tokens=False)
        rejected_tokens = tokenizer.encode(f" {pair['rejected']}", add_special_tokens=False)

        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens
        chosen_weights = [0.0] * len(prompt_tokens) + [1.0] * len(chosen_tokens)
        rejected_weights = [0.0] * len(prompt_tokens) + [1.0] * len(rejected_tokens)

        data.append(
            tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(chosen_full[:-1]),
                loss_fn_inputs={
                    "target_tokens": _tensor(chosen_full[1:], "int64"),
                    "weights": _tensor(chosen_weights[1:], "float32"),
                },
            )
        )
        data.append(
            tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(rejected_full[:-1]),
                loss_fn_inputs={
                    "target_tokens": _tensor(rejected_full[1:], "int64"),
                    "weights": _tensor(rejected_weights[1:], "float32"),
                },
            )
        )
        full_sequences.append(tinker.types.ModelInput.from_ints(chosen_full))
        full_sequences.append(tinker.types.ModelInput.from_ints(rejected_full))
        metadata["pairs"].append(
            {
                "prompt": pair["prompt"],
                "chosen_len": len(chosen_tokens),
                "rejected_len": len(rejected_tokens),
            }
        )

    return data, full_sequences, metadata


def _compute_dpo_loss(
    *,
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )
    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    metrics = {
        "dpo_loss": float(loss.item()),
        "accuracy": float((chosen_log_ratio > rejected_log_ratio).float().mean().item()),
        "margin": float((chosen_rewards - rejected_rewards).mean().item()),
        "chosen_reward": float(chosen_rewards.mean().item()),
        "rejected_reward": float(rejected_rewards.mean().item()),
    }
    return loss, metrics


class TestDenseDPO:
    def test_dpo_training(self, tokenizer):
        num_iterations = 8
        lr = 1e-5
        dpo_beta = 0.1

        service_client = tinker.ServiceClient(base_url=BASE_URL, api_key=_sdk_api_key(API_KEY))
        training_client = service_client.create_lora_training_client(
            base_model=DENSE_MODEL,
            rank=32,
        )
        reference_client = training_client.save_weights_and_get_sampling_client("reference")

        data, full_sequences, _ = _prepare_dpo_data(tokenizer)
        chosen_data = [datum for idx, datum in enumerate(data) if idx % 2 == 0]
        rejected_data = [datum for idx, datum in enumerate(data) if idx % 2 == 1]
        all_ref_logprob_seqs = [
            torch.tensor(reference_client.compute_logprobs(seq).result()[1:])
            for seq in full_sequences
        ]
        chosen_ref_logprob_seqs = [all_ref_logprob_seqs[idx] for idx in range(0, len(data), 2)]
        rejected_ref_logprob_seqs = [all_ref_logprob_seqs[idx] for idx in range(1, len(data), 2)]

        def dpo_loss_fn(
            batch: list[tinker.types.Datum], logprobs_list: list[torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            chosen_logprob_seqs = [logprobs_list[idx] for idx in range(0, len(batch), 2)]
            rejected_logprob_seqs = [logprobs_list[idx] for idx in range(1, len(batch), 2)]
            chosen_logprobs: list[torch.Tensor] = []
            rejected_logprobs: list[torch.Tensor] = []
            chosen_ref_logprobs: list[torch.Tensor] = []
            rejected_ref_logprobs: list[torch.Tensor] = []

            for idx in range(len(chosen_data)):
                chosen_weights = torch.tensor(chosen_data[idx].loss_fn_inputs["weights"].data)
                chosen_logprobs.append(torch.dot(chosen_logprob_seqs[idx].float(), chosen_weights.float()))
                chosen_ref_logprobs.append(
                    torch.dot(chosen_ref_logprob_seqs[idx].float(), chosen_weights.float())
                )

                rejected_weights = torch.tensor(rejected_data[idx].loss_fn_inputs["weights"].data)
                rejected_logprobs.append(torch.dot(rejected_logprob_seqs[idx].float(), rejected_weights.float()))
                rejected_ref_logprobs.append(
                    torch.dot(rejected_ref_logprob_seqs[idx].float(), rejected_weights.float())
                )

            return _compute_dpo_loss(
                chosen_logprobs=chosen_logprobs,
                rejected_logprobs=rejected_logprobs,
                chosen_ref_logprobs=chosen_ref_logprobs,
                rejected_ref_logprobs=rejected_ref_logprobs,
                dpo_beta=dpo_beta,
            )

        metrics = {
            "losses": [],
            "chosen_rewards": [],
            "rejected_rewards": [],
            "margins": [],
        }

        import time

        for i in range(num_iterations):
            t0 = time.time()
            backward_result = training_client.forward_backward_custom(data, dpo_loss_fn).result()
            dpo_metrics = backward_result.metrics
            metrics["losses"].append(dpo_metrics["dpo_loss"])
            metrics["chosen_rewards"].append(dpo_metrics["chosen_reward"])
            metrics["rejected_rewards"].append(dpo_metrics["rejected_reward"])
            metrics["margins"].append(dpo_metrics["margin"])

            optim_result = training_client.optim_step(
                tinker.AdamParams(
                    learning_rate=lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-12,
                )
            ).result()
            grad_norm = optim_result.metrics.get("grad_norm:last") or optim_result.metrics.get("grad_norm") or 0.0
            iteration_time = time.time() - t0
            print(
                f"Iteration {i+1}: loss={dpo_metrics['dpo_loss']:.4f}, "
                f"chosen={dpo_metrics['chosen_reward']:.3f}, "
                f"rejected={dpo_metrics['rejected_reward']:.3f}, "
                f"margin={dpo_metrics['margin']:.3f}, "
                f"grad_norm={grad_norm:.6f}, "
                f"time={iteration_time:.2f}s"
            )

        _, plot_path = save_training_curve(
            metrics,
            "dense_dpo_preference",
            metadata={
                "model": DENSE_MODEL,
                "lr": lr,
                "beta": dpo_beta,
                "num_iterations": num_iterations,
                "num_pairs": len(DPO_PAIRS),
            },
            plot_title="Dense DPO: Training Curves",
        )

        anomalies = detect_anomalies(metrics["losses"], "loss")
        if len(metrics["margins"]) >= 2:
            margin_improvement = metrics["margins"][-1] - metrics["margins"][0]
            if margin_improvement < 0:
                anomalies.append(
                    f"Margin decreased: {metrics['margins'][0]:.3f} -> {metrics['margins'][-1]:.3f}"
                )
        if metrics["margins"] and metrics["margins"][-1] < 0:
            anomalies.append(f"Final margin negative: {metrics['margins'][-1]:.3f} (chosen < rejected)")

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
            },
        )

        assert len(metrics["losses"]) > 0, "No training iterations completed"
        for loss in metrics["losses"]:
            assert not np.isnan(loss) and not np.isinf(loss), f"Invalid DPO loss: {loss}"
        if len(metrics["margins"]) >= 2 and metrics["margins"][-1] < -1.0:
            pytest.fail(
                f"Margin too negative: {metrics['margins'][-1]:.3f} (model prefers rejected)\n"
                f"Inspect training curve: {plot_path}"
            )
