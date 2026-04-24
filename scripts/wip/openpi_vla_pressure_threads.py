#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from openpi_libero_fast_rl import (
    _create_action_session,
    _create_model as _rl_create_model,
    _delete_action_session,
    _delete_model as _delete_model,
    _forward_logprobs,
    _make_rl_datum,
    _ppo_train_step,
    _sample_actions,
    _save_weights_for_sampler,
    _tokenize_sampled_actions,
)
from openpi_libero_sft import (
    _build_transform,
    _collect_transformed_items,
    _create_model as _sft_create_model,
    _fast_datum_from_transformed,
    _load_tasks,
    _pi05_datum_from_transformed,
    _poll_future,
)

FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"
PI05_MODEL = "openpi/pi05-libero-low-mem-finetune"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj) + "\n")


def _run_sft_client(
    *,
    name: str,
    base_url: str,
    base_model: str,
    task_index: int,
    task_text: str,
    items: list[dict[str, Any]],
    pool_meta: dict[str, Any],
    batch_size: int,
    seed: int,
    output_dir: Path,
    start_event: threading.Event,
) -> dict[str, Any]:
    rng = random.Random(seed)
    run_log = output_dir.with_suffix(".run.log")
    metrics_path = output_dir / "metrics.jsonl"
    exit_path = output_dir.with_suffix(".exit")
    start_event.wait()
    model_id = None
    session_id = None
    try:
        model_id, session_id = _sft_create_model(base_url, base_model)
        _append_jsonl(run_log, {
            "event": "model_created",
            "name": name,
            "model_id": model_id,
            "session_id": session_id,
            "task": task_text,
            **pool_meta,
        })
        loss_fn = "cross_entropy" if "pi0-fast" in base_model else "flow_matching"
        datum_builder = _fast_datum_from_transformed if "pi0-fast" in base_model else _pi05_datum_from_transformed
        batch = [datum_builder(base_model, rng.choice(items)) for _ in range(batch_size)]
        resp = requests.post(
            f"{base_url}/api/v1/mint/vla/train_step",
            json={"model_id": model_id, "loss_fn": loss_fn, "data": batch},
            timeout=120,
        )
        resp.raise_for_status()
        result = _poll_future(base_url, resp.json()["request_id"], timeout_s=1800)
        record = {
            "step": 1,
            "loss": float(result["metrics"]["loss:mean"]),
            "metrics": result["metrics"],
            "name": name,
            "task": task_text,
        }
        _append_jsonl(metrics_path, record)
        _append_jsonl(run_log, record)
        summary = {
            "name": name,
            "kind": "sft",
            "base_model": base_model,
            "task_index": task_index,
            "task": task_text,
            "model_id": model_id,
            "steps": 1,
            "batch_size": batch_size,
            "loss": record["loss"],
            **pool_meta,
        }
        _write_text(output_dir / "summary.json", json.dumps(summary, indent=2))
        _write_text(exit_path, "0")
        return {"name": name, "ok": True, **summary}
    except Exception:
        _write_text(exit_path, "1")
        _append_jsonl(run_log, {"event": "error", "name": name, "traceback": traceback.format_exc()})
        return {"name": name, "ok": False}
    finally:
        if model_id:
            try:
                _delete_model(base_url, model_id)
            except Exception:
                _append_jsonl(run_log, {"event": "delete_model_error", "name": name, "traceback": traceback.format_exc()})


def _run_fast_rl_client(
    *,
    name: str,
    base_url: str,
    task_index: int,
    task_text: str,
    items: list[dict[str, Any]],
    pool_meta: dict[str, Any],
    batch_size: int,
    seed: int,
    output_dir: Path,
    start_event: threading.Event,
) -> dict[str, Any]:
    rng = random.Random(seed)
    run_log = output_dir.with_suffix(".run.log")
    metrics_path = output_dir / "metrics.jsonl"
    exit_path = output_dir.with_suffix(".exit")
    start_event.wait()
    model_id = None
    action_session_id = None
    try:
        model_id = _rl_create_model(base_url, FAST_MODEL)
        checkpoint_path = _save_weights_for_sampler(base_url, model_id, f"{name}-{seed:08x}")
        action_session_id = _create_action_session(base_url, FAST_MODEL, checkpoint_path)
        item = items[rng.randrange(len(items))]
        sampled_actions = _sample_actions(base_url, action_session_id, item)
        expert_actions = np.asarray(item["actions"], dtype=np.float32)
        mse = float(np.mean((sampled_actions - expert_actions) ** 2))
        reward = math.exp(-5.0 * mse)
        from openpi.models.tokenizer import FASTTokenizer
        from openpi_libero_fast_rl import _resolve_fast_tokenizer_path
        from openpi_libero_sft import _build_transform as _build_transform_inner
        cfg, _ = _build_transform_inner(FAST_MODEL)
        tokenizer = FASTTokenizer(cfg.model.max_token_len, fast_tokenizer_path=_resolve_fast_tokenizer_path())
        prefix_tokens, target_tokens, suffix_mask = _tokenize_sampled_actions(tokenizer, task_text, item, sampled_actions)
        probe_datum = _make_rl_datum(item, prefix_tokens, target_tokens, suffix_mask, logprobs=[0.0] * len(target_tokens), advantages=[0.0] * len(target_tokens))
        old_logprobs = _forward_logprobs(base_url, model_id, probe_datum)
        datum = _make_rl_datum(item, prefix_tokens, target_tokens, suffix_mask, logprobs=old_logprobs, advantages=[reward] * len(target_tokens))
        result = _ppo_train_step(base_url, model_id, datum)
        record = {
            "step": 1,
            "reward": reward,
            "loss": float(result["metrics"]["loss:mean"]),
            "num_samples": batch_size,
            "name": name,
            "task": task_text,
        }
        _append_jsonl(metrics_path, record)
        _append_jsonl(run_log, record)
        summary = {
            "name": name,
            "kind": "fast_rl",
            "base_model": FAST_MODEL,
            "task_index": task_index,
            "task": task_text,
            "model_id": model_id,
            "reward": record["reward"],
            "loss": record["loss"],
            **pool_meta,
        }
        _write_text(output_dir / "summary.json", json.dumps(summary, indent=2))
        _write_text(exit_path, "0")
        return {"name": name, "ok": True, **summary}
    except Exception:
        _write_text(exit_path, "1")
        _append_jsonl(run_log, {"event": "error", "name": name, "traceback": traceback.format_exc()})
        return {"name": name, "ok": False}
    finally:
        if action_session_id:
            try:
                _delete_action_session(base_url, action_session_id)
            except Exception:
                _append_jsonl(run_log, {"event": "delete_action_error", "name": name, "traceback": traceback.format_exc()})
        if model_id:
            try:
                _delete_model(base_url, model_id)
            except Exception:
                _append_jsonl(run_log, {"event": "delete_model_error", "name": name, "traceback": traceback.format_exc()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("OPENPI_DATA_HOME", "/vePFS-Mindverse/share/data/openpi")
    os.environ.setdefault("HF_HOME", "/vePFS-Mindverse/share/hf")

    base_url = args.base_url.rstrip("/")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks()

    fast_cfg, fast_tx = _build_transform(FAST_MODEL)
    pi05_cfg, pi05_tx = _build_transform(PI05_MODEL)
    fast_pools: dict[int, tuple[str, list[dict[str, Any]], dict[str, Any]]] = {}
    pi05_pools: dict[int, tuple[str, list[dict[str, Any]], dict[str, Any]]] = {}

    for task_idx in range(0, 20):
        task_text = tasks[task_idx]
        fast_pools[task_idx] = (
            task_text,
            *_collect_transformed_items(FAST_MODEL, fast_tx, task_text, int(fast_cfg.model.action_horizon), max_episodes=8, stride=10),
        )
    for task_idx in range(20, 30):
        task_text = tasks[task_idx]
        pi05_pools[task_idx] = (
            task_text,
            *_collect_transformed_items(PI05_MODEL, pi05_tx, task_text, int(pi05_cfg.model.action_horizon), max_episodes=8, stride=10),
        )

    jobs: list[tuple[str, Any, dict[str, Any]]] = []
    for task_idx in range(0, 10):
        task_text, items, meta = fast_pools[task_idx]
        name = f"pressure2_fast_sft_{task_idx:02d}_w"
        jobs.append((name, _run_sft_client, {
            "name": name,
            "base_url": base_url,
            "base_model": FAST_MODEL,
            "task_index": task_idx,
            "task_text": task_text,
            "items": items,
            "pool_meta": meta,
            "batch_size": 2,
            "seed": args.seed + task_idx,
            "output_dir": output_root / name,
        }))
    for task_idx in range(10, 20):
        task_text, items, meta = fast_pools[task_idx]
        name = f"pressure2_fast_rl_{task_idx:02d}_w"
        jobs.append((name, _run_fast_rl_client, {
            "name": name,
            "base_url": base_url,
            "task_index": task_idx,
            "task_text": task_text,
            "items": items,
            "pool_meta": meta,
            "batch_size": 1,
            "seed": args.seed + task_idx,
            "output_dir": output_root / name,
        }))
    for task_idx in range(20, 30):
        task_text, items, meta = pi05_pools[task_idx]
        name = f"pressure2_pi05_sft_{task_idx:02d}_w"
        jobs.append((name, _run_sft_client, {
            "name": name,
            "base_url": base_url,
            "base_model": PI05_MODEL,
            "task_index": task_idx,
            "task_text": task_text,
            "items": items,
            "pool_meta": meta,
            "batch_size": 2,
            "seed": args.seed + task_idx,
            "output_dir": output_root / name,
        }))

    for name, _, kwargs in jobs:
        exit_path = kwargs["output_dir"].with_suffix(".exit")
        if exit_path.exists():
            exit_path.unlink()

    start_event = threading.Event()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = []
        for _, fn, kwargs in jobs:
            futures.append(pool.submit(fn, start_event=start_event, **kwargs))
        start_event.set()
        for fut in as_completed(futures):
            results.append(fut.result())

    summary = {
        "count": len(results),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "results": sorted(results, key=lambda r: r["name"]),
    }
    _write_text(output_root / "batch_summary.json", json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
