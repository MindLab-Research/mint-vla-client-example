#!/usr/bin/env python3
"""Benchmark LoRA sampling latency vs concurrency for one base model.

This measures the /save_weights_and_get_sampling_client (LoRA export + engine hookup)
and then /asample + /retrieve_future sampling latency for multiple concurrent LoRAs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


def _p50(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _get(base_url: str, path: str, *, timeout_s: float) -> dict[str, Any]:
    r = requests.get(f"{base_url}{path}", timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


@dataclass(frozen=True)
class OneResult:
    ok: bool
    elapsed_s: float
    sequences: int | None
    error: str | None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        default=os.environ.get("TINKER_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:8000",
    )
    p.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY") or os.environ.get("MINT_API_KEY") or None)
    p.add_argument("--model", required=True, help="HF model name")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=32000)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--unique-prompts", action="store_true", help="Make prompt tokens unique per request (avoid prefix caching)")
    p.add_argument("--concurrency", default="1,2,4", help="Comma-separated concurrency values (number of LoRAs in-flight)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--poll-s", type=float, default=0.2)
    p.add_argument("--call-timeout-s", type=float, default=3600.0)
    p.add_argument("--run-dir", default=None, help="Directory to write JSONL results")
    args = p.parse_args()

    base_url = str(args.base_url).rstrip("/")
    conc = _parse_int_list(args.concurrency)
    if any(c < 1 for c in conc):
        raise SystemExit(f"invalid --concurrency: {conc}")
    if args.prompt_len < 1:
        raise SystemExit("--prompt-len must be >= 1")
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if args.rank < 1:
        raise SystemExit("--rank must be >= 1")

    info = _get(base_url, "/api/v1/server_info", timeout_s=30)
    git_sha = info.get("git_sha")

    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "issue87" / _ts_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"bench_lora_{args.model.replace('/', '_')}_pl{args.prompt_len}_ns{args.num_samples}_mt{args.max_tokens}.jsonl"

    import mint
    from mint import types

    service_client = mint.ServiceClient(base_url=base_url, api_key=args.api_key)
    training_clients = []
    sampling_clients = []
    save_elapsed_s = []

    max_c = max(conc)
    t_create0 = time.time()
    for i in range(max_c):
        tc = service_client.create_lora_training_client(
            base_model=args.model,
            rank=int(args.rank),
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        training_clients.append(tc)
    t_create1 = time.time()

    tokenizer = training_clients[0].get_tokenizer()
    filler_ids = tokenizer.encode(" a", add_special_tokens=False) or tokenizer.encode("0", add_special_tokens=False)
    if not filler_ids:
        raise SystemExit("failed to get filler token id from tokenizer")
    filler_id = int(filler_ids[0])

    base_prompt_tokens = [filler_id] * int(args.prompt_len)

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
                    "prompt_len": int(args.prompt_len),
                    "num_samples": int(args.num_samples),
                    "max_tokens": int(args.max_tokens),
                    "unique_prompts": bool(args.unique_prompts),
                    "concurrency": conc,
                    "repeats": int(args.repeats),
                    "create_training_clients_s": float(t_create1 - t_create0),
                },
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()

        for i, tc in enumerate(training_clients):
            t0 = time.time()
            sc = tc.save_weights_and_get_sampling_client(name=f"bench_lora_session_{i:02d}")
            save_elapsed_s.append(time.time() - t0)
            sampling_clients.append(sc)
            f.write(
                json.dumps(
                    {
                        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                        "kind": "save",
                        "session_idx": i,
                        "elapsed_s": save_elapsed_s[-1],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            f.flush()

        def _run_one(*, sc: Any, prompt_tokens: list[int]) -> OneResult:
            t0 = time.time()
            try:
                fut = sc.sample(
                    prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                    num_samples=int(args.num_samples),
                    sampling_params=types.SamplingParams(
                        max_tokens=int(args.max_tokens),
                        temperature=0.7,
                        top_k=-1,
                        top_p=1.0,
                    ),
                )
                res = fut.result(timeout=float(args.call_timeout_s))
                seqs = res.sequences if res is not None else None
                n = len(seqs) if isinstance(seqs, list) else None
                return OneResult(ok=True, elapsed_s=time.time() - t0, sequences=n, error=None)
            except Exception as e:
                return OneResult(ok=False, elapsed_s=time.time() - t0, sequences=None, error=f"{type(e).__name__}: {e}")

        for c in conc:
            for rep in range(int(args.repeats)):
                barrier = threading.Barrier(c + 1)

                def _worker(sc: Any, prompt_tokens: list[int]) -> OneResult:
                    barrier.wait()
                    return _run_one(sc=sc, prompt_tokens=prompt_tokens)

                prompt_tokens_by_req: list[list[int]] = []
                for i in range(c):
                    toks = list(base_prompt_tokens)
                    if args.unique_prompts:
                        toks[0] = 1000 + (c * 1000 + rep * 100 + i)
                    prompt_tokens_by_req.append(toks)

                t0 = time.time()
                with ThreadPoolExecutor(max_workers=c) as ex:
                    futs = [
                        ex.submit(_worker, sampling_clients[i], prompt_tokens_by_req[i])
                        for i in range(c)
                    ]
                    barrier.wait()
                    results = [fu.result() for fu in futs]
                elapsed_wall = time.time() - t0

                latencies = [r.elapsed_s for r in results if r.ok]
                errors = [r.error for r in results if not r.ok]
                rec = {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "kind": "result",
                    "concurrency": c,
                    "repeat": rep,
                    "ok": len(errors) == 0,
                    "n_ok": len(latencies),
                    "n_err": len(errors),
                    "wall_s": elapsed_wall,
                    "p50_s": _p50(latencies),
                    "mean_s": statistics.mean(latencies) if latencies else None,
                    "errors": errors[:5],
                }
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                print(f"[{rec['ts']}] model={args.model} c={c} rep={rep} ok={rec['ok']} p50_s={rec['p50_s']}", flush=True)

    print(str(out_path), flush=True)


if __name__ == "__main__":
    main()

