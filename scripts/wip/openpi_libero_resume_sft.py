#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import requests

from openpi_libero_sft import (
    _build_transform,
    _collect_transformed_items,
    _create_model,
    _fast_datum_from_transformed,
    _load_tasks,
    _pi05_datum_from_transformed,
    _plot_curve,
    CONFIG_NAME_BY_BASE_MODEL,
)


def _poll_future(base_url: str, request_id: str, *, timeout_s: float = 3600.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(f"{base_url}/api/v1/retrieve_future", json={"request_id": request_id}, timeout=120)
        if resp.status_code == 408:
            time.sleep(1.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise TimeoutError(f"timed out waiting for {request_id}")


def _delete_model(base_url: str, model_id: str) -> None:
    resp = requests.delete(f"{base_url}/api/v1/models/{model_id}", timeout=300)
    if resp.status_code not in {200, 404}:
        resp.raise_for_status()


def _save_state(base_url: str, model_id: str, checkpoint_name: str) -> str:
    resp = requests.post(
        f"{base_url}/api/v1/save_state",
        json={"model_id": model_id, "path": checkpoint_name},
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    path = result.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"save_state missing path: {result!r}")
    return path


def _create_model_from_state(base_url: str, *, session_id: str, model_seq_id: int, base_model: str, state_path: str):
    payload = {
        "session_id": session_id,
        "model_seq_id": model_seq_id,
        "base_model": base_model,
        "state_path": state_path,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "load_optimizer": True,
        "user_metadata": {"script": "scripts/wip/openpi_libero_resume_sft.py"},
    }
    resp = requests.post(f"{base_url}/api/v1/create_model_from_state", json=payload, timeout=120)
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model_from_state missing model_id: {result!r}")
    return model_id


def _train_step(base_url: str, *, model_id: str, datum_builder, base_model: str, items, batch_indices, loss_fn: str):
    batch = [datum_builder(base_model, items[idx]) for idx in batch_indices]
    resp = requests.post(
        f"{base_url}/api/v1/mint/vla/train_step",
        json={"model_id": model_id, "loss_fn": loss_fn, "data": batch},
        timeout=120,
    )
    resp.raise_for_status()
    result = _poll_future(base_url, resp.json()["request_id"], timeout_s=3600)
    return float(result["metrics"]["loss:mean"]), result["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--base-model", required=True, choices=sorted(CONFIG_NAME_BY_BASE_MODEL))
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--pre-steps", type=int, default=4)
    parser.add_argument("--post-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    tasks = _load_tasks()
    task_text = tasks[args.task_index]
    cfg, tx = _build_transform(args.base_model)
    items, pool_meta = _collect_transformed_items(
        args.base_model,
        tx,
        task_text,
        int(cfg.model.action_horizon),
        max_episodes=args.max_episodes,
        stride=args.stride,
    )
    rng = random.Random(args.seed)
    total_steps = args.pre_steps + args.post_steps
    batch_plan = [[rng.randrange(len(items)) for _ in range(args.batch_size)] for _ in range(total_steps)]
    loss_fn = "cross_entropy" if "pi0-fast" in args.base_model else "flow_matching"
    datum_builder = _fast_datum_from_transformed if "pi0-fast" in args.base_model else _pi05_datum_from_transformed

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    steps: list[int] = []
    losses: list[float] = []
    resume_after_step = args.pre_steps
    model_a = _create_model(base_url, args.base_model)[0]
    model_b = None
    try:
        for step_idx in range(args.pre_steps):
            loss, metrics = _train_step(
                base_url,
                model_id=model_a,
                datum_builder=datum_builder,
                base_model=args.base_model,
                items=items,
                batch_indices=batch_plan[step_idx],
                loss_fn=loss_fn,
            )
            record = {"step": step_idx + 1, "phase": "pre", "loss": loss, "metrics": metrics}
            metrics_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            steps.append(step_idx + 1)
            losses.append(loss)

        ckpt_path = _save_state(base_url, model_a, f"resume-{uuid.uuid4().hex[:8]}")
        _delete_model(base_url, model_a)
        model_b = _create_model_from_state(
            base_url,
            session_id=f"resume-{uuid.uuid4().hex[:10]}",
            model_seq_id=0,
            base_model=args.base_model,
            state_path=ckpt_path,
        )

        for offset in range(args.post_steps):
            step_no = args.pre_steps + offset + 1
            loss, metrics = _train_step(
                base_url,
                model_id=model_b,
                datum_builder=datum_builder,
                base_model=args.base_model,
                items=items,
                batch_indices=batch_plan[args.pre_steps + offset],
                loss_fn=loss_fn,
            )
            record = {"step": step_no, "phase": "post", "loss": loss, "metrics": metrics}
            metrics_path.open("a", encoding="utf-8").write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            steps.append(step_no)
            losses.append(loss)

        _plot_curve(steps, losses, out_dir / "loss_curve.png", f"{args.base_model} | resume task={args.task_index}")
        summary = {
            "base_model": args.base_model,
            "task_index": args.task_index,
            "task": task_text,
            "pre_steps": args.pre_steps,
            "post_steps": args.post_steps,
            "resume_after_step": resume_after_step,
            "initial_loss": losses[0],
            "presave_loss": losses[args.pre_steps - 1],
            "first_resumed_loss": losses[args.pre_steps],
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "curve_path": str(out_dir / "loss_curve.png"),
            **pool_meta,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"event": "done", **summary}), flush=True)
    finally:
        if model_b:
            _delete_model(base_url, model_b)
        else:
            _delete_model(base_url, model_a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
