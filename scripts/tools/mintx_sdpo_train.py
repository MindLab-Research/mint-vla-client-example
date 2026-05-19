#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import random
import string
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mint
from mint import mint as mintx
import mint.mint._mintx as mintx_impl
import tinker
from transformers import AutoTokenizer

OWNER_ID = (
    os.environ.get("MINT_OWNER_ID")
    or os.environ.get("MINT_USER_ID")
    or os.environ.get("MINT_GATEWAY_USER_ID")
    or "mintx_sdpo_user"
)
_MINTX_MODEL_DUMP = mintx_impl.model_dump


def _model_dump_with_owner(obj, *args, **kwargs):
    body = _MINTX_MODEL_DUMP(obj, *args, **kwargs)
    if isinstance(body, dict) and body.get("type") in {
        "mint_forward_backward_reverse_kl",
        "mint_interpolate_checkpoints",
    }:
        body["owner_id"] = OWNER_ID
    return body


mintx_impl.model_dump = _model_dump_with_owner

@dataclass(frozen=True)
class Example:
    key: str
    value: str


def target_answer(example: Example) -> str:
    return f"value={example.value}"


def teacher_prompt(example: Example) -> str:
    return (
        "You are an exact lookup engine for an arbitrary key-value table.\n"
        "Each key maps to one opaque lowercase code.\n"
        "Do not explain or transform the code.\n"
        "Output exactly one line in the format value=<code>.\n"
        "Example 1:\n"
        "Input: key=atlas42\n"
        "Answer:\n"
        "value=pkmvrn\n"
        "Example 2:\n"
        "Input: key=delta77\n"
        "Answer:\n"
        "value=qsbhxe\n"
        f"Input: key={example.key}\n"
        "Answer:"
    )


def student_prompt(example: Example) -> str:
    return (
        "Lookup the stored code.\n"
        f"key={example.key}\n"
        "value:"
    )


def make_examples(*, seed: int, count: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    seen_keys: set[str] = set()
    alphabet = string.ascii_lowercase + string.digits
    while len(examples) < count:
        key = "".join(rng.choice(alphabet) for _ in range(8))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        value = "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
        examples.append(Example(key=key, value=value))
    return examples


def build_reverse_kl_datum(tokenizer, example: Example, completion_tokens: list[int]) -> mintx.ReverseKLDatum:
    student_ids = tokenizer.encode(student_prompt(example))
    teacher_ids = tokenizer.encode(teacher_prompt(example))
    return mintx.ReverseKLDatum(
        student_input=tinker.types.ModelInput.from_ints(student_ids),
        reference_input=tinker.types.ModelInput.from_ints(teacher_ids),
        target_tokens=tinker.types.TensorData(
            data=[int(x) for x in completion_tokens],
            shape=[len(completion_tokens)],
            dtype="int64",
        ),
        weights=tinker.types.TensorData(
            data=[1.0] * len(completion_tokens),
            shape=[len(completion_tokens)],
            dtype="float32",
        ),
    )


def load_tokenizer_for_model(model_name: str):
    hf_home = Path(os.environ.get("HF_HOME", "/vePFS-Mindverse/share/huggingface"))
    if "/" in model_name:
        org, name = model_name.split("/", 1)
        snapshot_root = hf_home / "hub" / f"models--{org}--{name}" / "snapshots"
        if snapshot_root.is_dir():
            main_path = snapshot_root / "main"
            if main_path.is_dir():
                return AutoTokenizer.from_pretrained(main_path, trust_remote_code=True, local_files_only=True)
            snapshots = sorted(p for p in snapshot_root.iterdir() if p.is_dir())
            if snapshots:
                return AutoTokenizer.from_pretrained(snapshots[-1], trust_remote_code=True, local_files_only=True)
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)


def build_scoring_datum(tokenizer, example: Example, completion_tokens: list[int]) -> tinker.types.Datum:
    prompt_ids = tokenizer.encode(student_prompt(example))
    if not completion_tokens:
        raise ValueError("completion_tokens must be non-empty")
    model_input_ids = prompt_ids + completion_tokens[:-1]
    target_tokens = model_input_ids[1:] + [completion_tokens[-1]]
    weights = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(completion_tokens)
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(model_input_ids),
        loss_fn_inputs={
            "target_tokens": tinker.types.TensorData(
                data=[int(x) for x in target_tokens],
                shape=[len(target_tokens)],
                dtype="int64",
            ),
            "weights": tinker.types.TensorData(
                data=weights,
                shape=[len(weights)],
                dtype="float32",
            ),
        },
    )


def sample_completion(sampling_client: tinker.SamplingClient, tokenizer, prompt_text: str, *, temperature: float, top_k: int, max_tokens: int, seed: int | None) -> list[int]:
    prompt = tinker.types.ModelInput.from_ints(tokenizer.encode(prompt_text))
    response = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=["\n"],
            seed=seed,
        ),
    ).result()
    if not response.sequences:
        return []
    return list(response.sequences[0].tokens)


def decode_completion(tokenizer, tokens: list[int]) -> str:
    if not tokens:
        return ""
    return tokenizer.decode(tokens).strip()


def evaluate_sampling(
    sampling_client: tinker.SamplingClient, tokenizer, examples: list[Example], *, max_tokens: int
) -> tuple[float, float]:
    correct = 0
    similarity = 0.0
    for idx, example in enumerate(examples):
        tokens = sample_completion(
            sampling_client,
            tokenizer,
            student_prompt(example),
            temperature=0.1,
            top_k=1,
            max_tokens=max_tokens,
            seed=idx,
        )
        pred = decode_completion(tokenizer, tokens)
        target = target_answer(example)
        if pred == target:
            correct += 1
        similarity += difflib.SequenceMatcher(a=pred, b=target).ratio()
    denom = max(len(examples), 1)
    return correct / denom, similarity / denom


def probe_target_metrics(
    training_client: tinker.TrainingClient, tokenizer, examples: list[Example]
) -> tuple[float, float]:
    total_logprob = 0.0
    total_tokens = 0
    for example in examples:
        completion_tokens = tokenizer.encode(target_answer(example))
        datum = build_scoring_datum(tokenizer, example, completion_tokens)
        result = training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        logprobs = result.loss_fn_outputs[0]["logprobs"]
        values = logprobs.data if hasattr(logprobs, "data") else logprobs["data"]
        used = values[-len(completion_tokens):]
        total_logprob += sum(float(x) for x in used)
        total_tokens += len(used)
    mean_logprob = total_logprob / max(total_tokens, 1)
    mean_nll = -mean_logprob
    return mean_logprob, mean_nll


def plot_curves(rows: list[dict[str, float]], out_dir: Path) -> None:
    steps = [row["step"] for row in rows]
    losses = [row["reverse_kl_mean"] for row in rows]
    probe_logprobs = [row["probe_mean_logprob"] for row in rows]
    probe_nlls = [row["probe_mean_nll"] for row in rows]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, marker="o")
    plt.yscale("log")
    plt.title("Reverse KL")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(steps, probe_nlls, marker="o", label="probe nll")
    plt.plot(steps, probe_logprobs, marker="x", label="probe logprob")
    plt.title("Probe Target Metrics")
    plt.xlabel("Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "convergence.png", dpi=180)
    plt.close()


def main() -> None:
    global OWNER_ID
    parser = argparse.ArgumentParser(description="Run MintX SDPO training against /api/v1/mint endpoints")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--owner-id", default=OWNER_ID)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=24)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bootstrap-teacher-learning-rate", type=float, default=None)
    parser.add_argument("--reverse-kl-temperature", type=float, default=1.0)
    parser.add_argument("--ema-alpha", type=float, default=0.95)
    parser.add_argument("--ema-sync-steps", type=int, default=1)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-checkpoint-kind", choices=["sampler", "state"], default="sampler")
    parser.add_argument("--sampling-mode", choices=["student", "ground_truth"], default="student")
    parser.add_argument("--train-mlp", action="store_true")
    parser.add_argument("--train-unembed", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--bootstrap-teacher-steps", type=int, default=0)
    parser.add_argument("--probe-size", type=int, default=8)
    args = parser.parse_args()

    OWNER_ID = args.owner_id

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"mintx-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    train_examples = make_examples(seed=args.seed, count=args.train_size)
    eval_examples = make_examples(seed=args.seed + 1000, count=args.eval_size)
    probe_examples = train_examples[: min(args.probe_size, len(train_examples))]

    service = mint.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    def create_client(tag: str) -> tinker.TrainingClient:
        return service.create_lora_training_client(
            base_model=args.model,
            rank=args.lora_rank,
            train_mlp=args.train_mlp,
            train_unembed=args.train_unembed,
            user_metadata={"mintx_task": tag},
        )

    training_client = create_client("sdpo_string_transform_student")
    tokenizer = load_tokenizer_for_model(args.model)

    if args.bootstrap_teacher_steps > 0:
        teacher_client = create_client("sdpo_string_transform_teacher")
        rng_teacher = random.Random(args.seed + 17)
        teacher_lr = (
            float(args.bootstrap_teacher_learning_rate)
            if args.bootstrap_teacher_learning_rate is not None
            else float(args.learning_rate)
        )
        for teacher_step in range(1, args.bootstrap_teacher_steps + 1):
            batch_examples = [rng_teacher.choice(train_examples) for _ in range(args.train_batch_size)]
            batch = [
                build_scoring_datum(tokenizer, example, tokenizer.encode(target_answer(example)))
                for example in batch_examples
            ]
            teacher_client.forward_backward(batch, loss_fn="cross_entropy").result()
            teacher_client.optim_step(
                tinker.types.AdamParams(
                    learning_rate=teacher_lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            ).result()
            print(
                json.dumps(
                    {
                        "bootstrap_teacher_step": float(teacher_step),
                        "bootstrap_total": float(args.bootstrap_teacher_steps),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.teacher_checkpoint_kind == "state":
            teacher_path = teacher_client.save_state(f"{run_tag}-teacher-bootstrap").result().path
        else:
            teacher_path = teacher_client.save_weights_for_sampler(f"{run_tag}-teacher-bootstrap").result().path
    elif args.teacher_checkpoint_kind == "state":
        teacher_path = training_client.save_state(f"{run_tag}-teacher-init").result().path
    else:
        teacher_path = training_client.save_weights_for_sampler(f"{run_tag}-teacher-init").result().path
    sampler = None
    if args.sampling_mode == "student":
        sampler = training_client.save_weights_and_get_sampling_client()

    rows: list[dict[str, float]] = []
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "reverse_kl_mean",
                "probe_mean_logprob",
                "probe_mean_nll",
                "elapsed_s",
            ],
        )
        writer.writeheader()

        rng = random.Random(args.seed + 7)
        start = time.time()
        for step in range(1, args.steps + 1):
            batch_examples = [rng.choice(train_examples) for _ in range(args.train_batch_size)]
            batch_data: list[mintx.ReverseKLDatum] = []
            for batch_idx, example in enumerate(batch_examples):
                if args.sampling_mode == "ground_truth":
                    completion_tokens = tokenizer.encode(target_answer(example))
                else:
                    assert sampler is not None
                    completion_tokens = sample_completion(
                        sampler,
                        tokenizer,
                        student_prompt(example),
                        temperature=args.sample_temperature,
                        top_k=-1,
                        max_tokens=args.max_tokens,
                        seed=args.seed * 1000 + step * 100 + batch_idx,
                    )
                    if not completion_tokens:
                        continue
                batch_data.append(build_reverse_kl_datum(tokenizer, example, completion_tokens))

            if not batch_data:
                continue

            reverse_kl_result = mintx.forward_backward_reverse_kl(
                training_client,
                reference_model_path=teacher_path,
                data=batch_data,
                temperature=args.reverse_kl_temperature,
            )
            training_client.optim_step(
                tinker.types.AdamParams(
                    learning_rate=args.learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            ).result()

            if args.ema_sync_steps > 0 and step % args.ema_sync_steps == 0:
                student_path = training_client.save_weights_for_sampler(
                    f"{run_tag}-student-{step:04d}"
                ).result().path
                teacher_path = mintx.interpolate_checkpoints(
                    service,
                    source_paths=[teacher_path, student_path],
                    coefficients=[args.ema_alpha, 1.0 - args.ema_alpha],
                    output_path=f"{run_tag}-teacher-{step:04d}",
                ).path

            if args.skip_eval:
                probe_mean_logprob = float("nan")
                probe_mean_nll = float("nan")
            elif args.sampling_mode == "student":
                sampler = training_client.save_weights_and_get_sampling_client()
                eval_accuracy, eval_similarity = evaluate_sampling(
                    sampler,
                    tokenizer,
                    eval_examples,
                    max_tokens=args.max_tokens,
                )
                probe_mean_logprob = float(eval_similarity)
                probe_mean_nll = float(1.0 - eval_accuracy)
            else:
                probe_mean_logprob, probe_mean_nll = probe_target_metrics(
                    training_client,
                    tokenizer,
                    probe_examples,
                )
            row = {
                "step": float(step),
                "reverse_kl_mean": float(reverse_kl_result.metrics.get("reverse_kl:mean", 0.0)),
                "probe_mean_logprob": float(probe_mean_logprob),
                "probe_mean_nll": float(probe_mean_nll),
                "elapsed_s": float(time.time() - start),
            }
            rows.append(row)
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, sort_keys=True), flush=True)

    plot_curves(rows, out_dir)


if __name__ == "__main__":
    main()
