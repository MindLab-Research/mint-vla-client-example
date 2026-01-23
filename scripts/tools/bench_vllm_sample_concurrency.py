#!/usr/bin/env python3
"""Benchmark /api/v1/asample latency vs concurrency for one model.

Focus: client-perceived latency of sampling with long prompts and multi-sample batches.
This hits the server's /asample + /retrieve_future API (no Ray CLI usage).
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
    xs2 = sorted(xs)
    mid = (len(xs2) - 1) // 2
    return xs2[mid]


def _post(base_url: str, path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = requests.post(f"{base_url}{path}", json=payload, timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


def _get(base_url: str, path: str, *, timeout_s: float) -> dict[str, Any]:
    r = requests.get(f"{base_url}{path}", timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


def _create_sampling_session(base_url: str, model: str, *, timeout_s: float) -> str:
    session = _post(
        base_url,
        "/api/v1/create_session",
        {"tags": [], "user_metadata": {}, "sdk_version": "", "type": "create_session"},
        timeout_s=timeout_s,
    )["session_id"]
    sampling_session_id = _post(
        base_url,
        "/api/v1/create_sampling_session",
        {"session_id": session, "base_model": model},
        timeout_s=timeout_s,
    )["sampling_session_id"]
    return str(sampling_session_id)


@dataclass(frozen=True)
class OneResult:
    ok: bool
    elapsed_s: float
    sequences: int | None
    error: str | None


def _run_one(
    *,
    base_url: str,
    sampling_session_id: str,
    prompt_tokens: list[int],
    num_samples: int,
    max_tokens: int,
    prompt_logprobs: bool,
    topk_prompt_logprobs: int,
    poll_s: float,
    call_timeout_s: float,
    expect_sequences: int | None,
) -> OneResult:
    t0 = time.time()
    try:
        fut = _post(
            base_url,
            "/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": int(num_samples),
                "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
                "sampling_params": {
                    "max_tokens": int(max_tokens),
                    "temperature": 0.7,
                    "top_k": -1,
                    "top_p": 1.0,
                },
                "prompt_logprobs": bool(prompt_logprobs),
                "topk_prompt_logprobs": int(topk_prompt_logprobs),
            },
            timeout_s=call_timeout_s,
        )["request_id"]

        while True:
            r = requests.post(
                f"{base_url}/api/v1/retrieve_future",
                json={"request_id": fut, "model_id": sampling_session_id},
                timeout=call_timeout_s,
            )
            if r.status_code == 408:
                time.sleep(poll_s)
                continue
            r.raise_for_status()
            out = r.json()
            seqs = out.get("sequences") if isinstance(out, dict) else None
            n = len(seqs) if isinstance(seqs, list) else None
            elapsed = time.time() - t0
            if expect_sequences is not None and n is not None and n != expect_sequences:
                return OneResult(
                    ok=False,
                    elapsed_s=elapsed,
                    sequences=n,
                    error=f"sequence_count_mismatch expected={expect_sequences} got={n}",
                )
            return OneResult(ok=True, elapsed_s=elapsed, sequences=n, error=None)
    except Exception as e:
        return OneResult(ok=False, elapsed_s=time.time() - t0, sequences=None, error=f"{type(e).__name__}: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:8000")
    p.add_argument("--model", required=True, help="HF model name")
    p.add_argument("--prompt-len", type=int, default=32000)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--prompt-logprobs", action="store_true")
    p.add_argument("--topk-prompt-logprobs", type=int, default=0)
    p.add_argument("--concurrency", default="1,2,4,8", help="Comma-separated concurrency values")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--poll-s", type=float, default=1.0)
    p.add_argument("--call-timeout-s", type=float, default=3600.0)
    p.add_argument("--expect-sequences", action="store_true", help="Fail if response sequences != num_samples")
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

    info = _get(base_url, "/api/v1/server_info", timeout_s=30)
    git_sha = info.get("git_sha")

    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "issue87" / _ts_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.prompt_logprobs:
        suffix += "_plp1"
    if args.topk_prompt_logprobs:
        suffix += f"_topk{int(args.topk_prompt_logprobs)}"
    out_path = run_dir / f"bench_sample_{args.model.replace('/', '_')}_pl{args.prompt_len}_ns{args.num_samples}_mt{args.max_tokens}{suffix}.jsonl"

    prompt_tokens = [10] * int(args.prompt_len)
    expect_sequences = int(args.num_samples) if args.expect_sequences else None

    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "kind": "meta",
                    "base_url": base_url,
                    "git_sha": git_sha,
                    "model": args.model,
                    "prompt_len": args.prompt_len,
                    "num_samples": args.num_samples,
                    "max_tokens": args.max_tokens,
                    "prompt_logprobs": bool(args.prompt_logprobs),
                    "topk_prompt_logprobs": int(args.topk_prompt_logprobs),
                    "concurrency": conc,
                    "repeats": args.repeats,
                },
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()

        for c in conc:
            for rep in range(args.repeats):
                sessions = [
                    _create_sampling_session(base_url, args.model, timeout_s=args.call_timeout_s)
                    for _ in range(c)
                ]

                barrier = threading.Barrier(c + 1)

                def _worker(sid: str) -> OneResult:
                    barrier.wait()
                    return _run_one(
                        base_url=base_url,
                        sampling_session_id=sid,
                        prompt_tokens=prompt_tokens,
                        num_samples=args.num_samples,
                        max_tokens=args.max_tokens,
                        prompt_logprobs=bool(args.prompt_logprobs),
                        topk_prompt_logprobs=int(args.topk_prompt_logprobs),
                        poll_s=args.poll_s,
                        call_timeout_s=args.call_timeout_s,
                        expect_sequences=expect_sequences,
                    )

                t0 = time.time()
                with ThreadPoolExecutor(max_workers=c) as ex:
                    futs = [ex.submit(_worker, sid) for sid in sessions]
                    barrier.wait()
                    results = [fu.result() for fu in futs]
                elapsed_wall = time.time() - t0

                latencies = [r.elapsed_s for r in results if r.ok]
                errors = [r.error for r in results if not r.ok]
                rec = {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "kind": "result",
                    "model": args.model,
                    "prompt_len": args.prompt_len,
                    "num_samples": args.num_samples,
                    "max_tokens": args.max_tokens,
                    "prompt_logprobs": bool(args.prompt_logprobs),
                    "topk_prompt_logprobs": int(args.topk_prompt_logprobs),
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
