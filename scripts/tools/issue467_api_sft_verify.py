import argparse
import json
import math
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DISH_DIR = REPO_ROOT / "context" / "dishlist_mint_e2e"
sys.path.insert(0, str(DISH_DIR))

from common import (  # noqa: E402
    MEM_UNIT_QA_PROMPT,
    SYSTEM_PROMPT,
    build_lora_records,
    load_capacity_context_and_samples,
    post_process_prediction,
    resolve_model_path,
    sample_to_target_text,
    score_prediction,
    summarize_predictions,
)
from mint_client import (  # noqa: E402
    create_lora_model,
    create_session,
    forward_backward,
    make_service_config,
    optim_step,
    save_state,
)
from mint_eval_utils import create_sampling_session, sample_tokens_via_mint  # noqa: E402
from mint_train_utils import append_jsonl, dataset_to_http_examples, shuffled_batches, write_json  # noqa: E402


DEFAULT_DATA_FILE = DISH_DIR / "data" / "turn10_tcov_1_0.json"
DEFAULT_CHECKPOINT_STEPS = (0, 40, 80, 160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--future-timeout-s", type=float, default=21600.0)
    parser.add_argument("--poll-heartbeat-s", type=float, default=2.0)
    parser.add_argument("--poll-http-timeout-s", type=float, default=60.0)
    parser.add_argument("--poll-log-every-s", type=float, default=30.0)
    parser.add_argument("--eval-remote-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _make_prompt_tokens(tokenizer, question: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": MEM_UNIT_QA_PROMPT.format(question=question)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(prompt_text, add_special_tokens=True)["input_ids"]


def _evaluate_split(
    *,
    args: argparse.Namespace,
    tokenizer,
    sampling_session_id: str,
    split_name: str,
    samples: list[dict],
    output_dir: Path,
) -> dict:
    rows = []
    pred_path = output_dir / f"{split_name}_predictions.jsonl"
    seq_offset = 0 if split_name == "train" else 1_000_000
    for row_idx, sample in enumerate(samples):
        prompt_tokens = _make_prompt_tokens(tokenizer, sample["input"])
        output_tokens = sample_tokens_via_mint(
            base_url=args.base_url,
            api_key=args.api_key,
            sampling_session_id=sampling_session_id,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.eval_remote_timeout_s,
            seq_id=seq_offset + row_idx,
        )
        prediction_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        prediction = post_process_prediction(prediction_text)
        row = {
            **sample,
            "target_text": sample_to_target_text(sample),
            "prediction": prediction,
            "score": score_prediction(sample, prediction),
            "split": split_name,
        }
        rows.append(row)
        append_jsonl(pred_path, row)
        print(
            f"[eval {split_name}] idx={row_idx + 1}/{len(samples)} "
            f"score={row['score']:.1f} pred={prediction!r} target={row['target_text']!r}",
            flush=True,
        )
    summary = summarize_predictions(rows)
    write_json(output_dir / f"{split_name}_summary.json", summary)
    return {"summary": summary, "predictions_path": str(pred_path)}


def _export_and_eval_checkpoint(
    *,
    args: argparse.Namespace,
    cfg,
    tokenizer,
    model_id: str,
    checkpoint_name: str,
    output_root: Path,
    train_samples: list[dict],
    test_samples: list[dict],
) -> dict:
    state_path = save_state(
        cfg=cfg,
        model_id=model_id,
        name=checkpoint_name,
        timeout_s=args.future_timeout_s,
        heartbeat_s=args.poll_heartbeat_s,
        http_timeout_s=args.poll_http_timeout_s,
        log_every_s=args.poll_log_every_s,
    )
    step_dir = output_root / checkpoint_name
    step_dir.mkdir(parents=True, exist_ok=True)
    sampling_session_id = create_sampling_session(
        base_url=args.base_url,
        api_key=args.api_key,
        base_model=args.model_name,
        model_path=state_path,
        lora_rank=args.lora_rank,
        session_id=f"issue467-{checkpoint_name}-{int(time.time())}",
    )
    train_eval = _evaluate_split(
        args=args,
        tokenizer=tokenizer,
        sampling_session_id=sampling_session_id,
        split_name="train",
        samples=train_samples,
        output_dir=step_dir,
    )
    test_eval = _evaluate_split(
        args=args,
        tokenizer=tokenizer,
        sampling_session_id=sampling_session_id,
        split_name="test",
        samples=test_samples,
        output_dir=step_dir,
    )
    return {
        "checkpoint_name": checkpoint_name,
        "state_path": state_path,
        "sampling_session_id": sampling_session_id,
        "train": train_eval,
        "test": test_eval,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model_name, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = make_service_config(base_url=args.base_url, api_key=args.api_key)

    _, train_context, train_samples = load_capacity_context_and_samples(args.data_file, "train")
    _, _, test_samples = load_capacity_context_and_samples(args.data_file, "test")
    if train_context["context_id"] != load_capacity_context_and_samples(args.data_file, "test")[1]["context_id"]:
        raise RuntimeError("train/test context_id mismatch")

    train_records = build_lora_records(train_context, train_samples, use_context=False)
    train_examples = dataset_to_http_examples(train_records, tokenizer, max_seq_length=4096)
    steps_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    total_steps = steps_per_epoch * args.n_epochs
    checkpoint_steps = set(DEFAULT_CHECKPOINT_STEPS)
    if max(checkpoint_steps) != total_steps:
        raise RuntimeError(
            f"Requested final checkpoint step {max(checkpoint_steps)} does not match total_steps={total_steps}"
        )

    session_id = create_session(
        cfg=cfg,
        tags=["dishlist-issue467-full-sft-verify"],
        sdk_version="issue467_api_sft_verify",
        timeout_s=30.0,
    )
    create_out = create_lora_model(
        cfg=cfg,
        session_id=session_id,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_seed=args.seed,
        train_attn=True,
        train_mlp=False,
        train_unembed=False,
        model_seq_id=0,
        timeout_s=args.future_timeout_s,
        heartbeat_s=args.poll_heartbeat_s,
        http_timeout_s=args.poll_http_timeout_s,
        log_every_s=args.poll_log_every_s,
    )
    model_id = create_out.get("model_id")
    if not model_id:
        raise RuntimeError(f"create_model missing model_id: {create_out}")

    metrics_path = output_root / "train_metrics.jsonl"
    run_meta = {
        "model_name": args.model_name,
        "model_path": model_path,
        "session_id": session_id,
        "model_id": str(model_id),
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "learning_rate": args.lr,
        "lora_rank": args.lora_rank,
        "data_file": str(Path(args.data_file).resolve()),
        "base_url": args.base_url,
    }
    write_json(output_root / "run_meta.json", run_meta)

    checkpoint_results: dict[str, dict] = {}

    checkpoint_name = f"step_{0:04d}"
    print(f"[checkpoint] exporting {checkpoint_name}", flush=True)
    checkpoint_results[checkpoint_name] = _export_and_eval_checkpoint(
        args=args,
        cfg=cfg,
        tokenizer=tokenizer,
        model_id=str(model_id),
        checkpoint_name=checkpoint_name,
        output_root=output_root,
        train_samples=train_samples,
        test_samples=test_samples,
    )

    step = 0
    for epoch in range(args.n_epochs):
        for batch_idx, batch in enumerate(shuffled_batches(train_examples, args.batch_size, args.seed, epoch)):
            step += 1
            step_start = time.time()
            backward_out = forward_backward(
                cfg=cfg,
                model_id=str(model_id),
                data=batch,
                loss_fn="cross_entropy",
                label=f"forward_backward step={step}/{total_steps}",
                timeout_s=args.future_timeout_s,
                heartbeat_s=args.poll_heartbeat_s,
                http_timeout_s=args.poll_http_timeout_s,
                log_every_s=args.poll_log_every_s,
            )
            optim_out = optim_step(
                cfg=cfg,
                model_id=str(model_id),
                learning_rate=float(args.lr),
                label=f"optim_step step={step}/{total_steps}",
                timeout_s=args.future_timeout_s,
                heartbeat_s=args.poll_heartbeat_s,
                http_timeout_s=args.poll_http_timeout_s,
                log_every_s=args.poll_log_every_s,
            )
            step_elapsed = time.time() - step_start
            row = {
                "step": step,
                "epoch": epoch,
                "batch": batch_idx,
                "step_time_sec": round(step_elapsed, 6),
                "loss_metrics": backward_out.get("metrics", {}),
                "optim_metrics": optim_out.get("metrics", {}),
            }
            append_jsonl(metrics_path, row)
            loss_value = backward_out.get("metrics", {}).get("loss:mean")
            grad_norm = optim_out.get("metrics", {}).get("grad_norm")
            print(
                f"[train] step={step}/{total_steps} epoch={epoch} batch={batch_idx} "
                f"loss={loss_value} grad_norm={grad_norm} step_time={step_elapsed:.2f}s",
                flush=True,
            )
            if step in checkpoint_steps and step != 0:
                checkpoint_name = f"step_{step:04d}"
                print(f"[checkpoint] exporting {checkpoint_name}", flush=True)
                checkpoint_results[checkpoint_name] = _export_and_eval_checkpoint(
                    args=args,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    model_id=str(model_id),
                    checkpoint_name=checkpoint_name,
                    output_root=output_root,
                    train_samples=train_samples,
                    test_samples=test_samples,
                )

    final_summary = {
        "run_meta": run_meta,
        "checkpoint_results": checkpoint_results,
    }
    write_json(output_root / "summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
