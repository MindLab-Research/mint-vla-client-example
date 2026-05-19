#!/usr/bin/env python3
"""Benchmark LoRA training forward_backward latency vs microbatch size.

This isolates the trainer path (no vLLM sampling) for a fixed max_seq_len.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


def _ts_dir() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("empty list")
    return out


def _get(base_url: str, path: str, *, timeout_s: float) -> dict[str, Any]:
    r = requests.get(f"{base_url}{path}", timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


def _wait_future(fut: Any, *, label: str, heartbeat_s: float) -> Any:
    start = time.time()
    while True:
        try:
            return fut.result(timeout=heartbeat_s)
        except TimeoutError:
            elapsed = time.time() - start
            print(f"[{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}] waiting {label} elapsed_s={elapsed:.0f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        default=os.environ.get("MINT_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:8000",
    )
    p.add_argument("--api-key", default=os.environ.get("MINT_API_KEY") or os.environ.get("MINT_API_KEY") or "dummy")
    p.add_argument("--model", required=True, help="HF model name")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=32000)
    p.add_argument("--batch-sizes", default="1,2,4,8", help="Comma-separated microbatch sizes to test")
    p.add_argument("--heartbeat-s", type=float, default=60.0)
    p.add_argument("--call-timeout-s", type=float, default=3600.0)
    p.add_argument("--run-dir", default=None, help="Directory to write JSONL results")
    args = p.parse_args()

    base_url = str(args.base_url).rstrip("/")
    bs = _parse_int_list(args.batch_sizes)
    if any(x < 1 for x in bs):
        raise SystemExit(f"invalid --batch-sizes: {bs}")
    if args.max_seq_len < 2:
        raise SystemExit("--max-seq-len must be >= 2")

    info = _get(base_url, "/api/v1/server_info", timeout_s=30)
    git_sha = info.get("git_sha")

    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "issue87" / _ts_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"bench_fb_{args.model.replace('/', '_')}_rank{int(args.rank)}_seqlen{int(args.max_seq_len)}.jsonl"

    import mint
    from mint import types

    service_client = mint.ServiceClient(base_url=base_url, api_key=args.api_key)
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=int(args.rank),
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    tokenizer = training_client.get_tokenizer()
    filler_ids = tokenizer.encode(" a", add_special_tokens=False) or tokenizer.encode("0", add_special_tokens=False)
    if not filler_ids:
        raise SystemExit("failed to get filler token id from tokenizer")
    filler_id = int(filler_ids[0])

    full_tokens = [filler_id] * int(args.max_seq_len)
    target_tokens = full_tokens[1:]
    n = int(args.max_seq_len) - 1
    logprobs = [0.0] * n
    advantages = [1.0] * n

    base_datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": logprobs,
            "advantages": advantages,
        },
    )

    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "kind": "meta",
                    "base_url": base_url,
                    "git_sha": git_sha,
                    "model": args.model,
                    "rank": int(args.rank),
                    "max_seq_len": int(args.max_seq_len),
                    "batch_sizes": bs,
                    "heartbeat_s": float(args.heartbeat_s),
                },
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()

        for b in bs:
            datums = [base_datum] * int(b)
            t0 = time.time()
            fut = training_client.forward_backward(datums, loss_fn="ppo")
            try:
                _wait_future(fut, label=f"forward_backward batch={b}", heartbeat_s=float(args.heartbeat_s))
                ok = True
                err: str | None = None
            except Exception as e:
                ok = False
                err = f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0
            rec = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "kind": "result",
                "batch_size": int(b),
                "ok": ok,
                "elapsed_s": float(elapsed),
                "error": err,
            }
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            print(f"[{rec['ts']}] batch={b} ok={ok} elapsed_s={elapsed:.3f}", flush=True)

    print(str(out_path), flush=True)


if __name__ == "__main__":
    main()

