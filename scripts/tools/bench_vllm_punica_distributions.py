#!/usr/bin/env python3
"""Run Punica-style request-family distributions against Mint /asample.

Prerequisite: the target 235B server must already be healthy and serving. This
script benchmarks the clean-cluster workload only; it does not repair node-ip,
Ray attach, worker runtime package, or placement problems.

This benchmark adapts Punica's four popularity distributions to a single-base-model
Mint workload by treating prompt-prefix families as the analog of LoRA IDs.
Requests inside the same family share a large prompt prefix so prefix caching can
coalesce some prefill work, while the suffix remains request-specific.

The script does not try to reproduce Punica's exact ShareGPT corpus. Instead it
uses configurable prompt/output length buckets chosen for the issue512 235B path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import random
import re
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


METRIC_NAMES = [
    "mint_vllm_scheduler_waiting_requests",
    "mint_vllm_scheduler_running_requests",
    "mint_vllm_scheduler_kv_cache_usage_ratio",
    "mint_vllm_prefix_cache_queries_total",
    "mint_vllm_prefix_cache_hits_total",
    "mint_vllm_queue_time_s_sum",
    "mint_vllm_queue_time_s_count",
    "mint_vllm_prefill_time_s_sum",
    "mint_vllm_prefill_time_s_count",
    "mint_vllm_decode_time_s_sum",
    "mint_vllm_decode_time_s_count",
    "mint_vllm_time_per_output_token_s_sum",
    "mint_vllm_time_per_output_token_s_count",
    "mint_vllm_workload_requests_total",
    "mint_vllm_workload_prompt_tokens_total",
    "mint_vllm_workload_generated_tokens_total",
]


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _ts_dir() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_int_list(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_float_list(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"non-positive weight sum: {weights}")
    return [w / total for w in weights]


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q}")
    ys = sorted(float(x) for x in xs)
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def _weighted_choice(rng: random.Random, values: list[int], weights: list[float]) -> int:
    return int(rng.choices(values, weights=weights, k=1)[0])


def _headers(api_key: str | None) -> dict[str, str]:
    key = (api_key or "").strip()
    return {"X-API-Key": key} if key else {}


def _post(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    headers: dict[str, str],
) -> dict[str, Any]:
    r = requests.post(f"{base_url}{path}", json=payload, timeout=timeout_s, headers=headers)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


def _get(base_url: str, path: str, *, timeout_s: float, headers: dict[str, str]) -> dict[str, Any]:
    r = requests.get(f"{base_url}{path}", timeout=timeout_s, headers=headers)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


_METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})? (?P<value>[-+0-9.eE]+)$")


def _fetch_metrics(
    base_url: str,
    model: str,
    *,
    timeout_s: float,
    headers: dict[str, str],
) -> dict[str, float]:
    r = requests.get(f"{base_url}/internal/metrics", timeout=timeout_s, headers=headers)
    r.raise_for_status()
    model_key = f'base_model="{model}"'
    out: dict[str, float] = {}
    for line in r.text.splitlines():
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in METRIC_NAMES:
            continue
        labels = m.group("labels") or ""
        if model_key not in labels:
            continue
        out[name] = float(m.group("value"))
    return out


@dataclass(frozen=True)
class RequestSpec:
    request_index: int
    family: int
    prompt_len: int
    max_tokens: int
    arrival_s: float
    session_id: str
    prompt_tokens: list[int]


@dataclass(frozen=True)
class RequestResult:
    request_index: int
    family: int
    prompt_len: int
    max_tokens: int
    arrival_s: float
    ok: bool
    elapsed_s: float
    generated_tokens: int
    sequence_count: int | None
    error: str | None


def _create_sampling_session(
    base_url: str,
    model: str,
    *,
    timeout_s: float,
    headers: dict[str, str],
) -> str:
    session_id = _post(
        base_url,
        "/api/v1/create_session",
        {"tags": [], "user_metadata": {}, "sdk_version": "", "type": "create_session"},
        timeout_s=timeout_s,
        headers=headers,
    )["session_id"]
    sampling_session_id = _post(
        base_url,
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "base_model": model},
        timeout_s=timeout_s,
        headers=headers,
    )["sampling_session_id"]
    return str(sampling_session_id)


def _make_family_prefix(family: int, length: int) -> list[int]:
    base = 10000 + family * 97
    return [base + (i % 31) for i in range(length)]


def _make_prompt_tokens(spec_index: int, family: int, prompt_len: int, shared_prefix_ratio: float) -> list[int]:
    if prompt_len < 1:
        raise ValueError(f"prompt_len must be >= 1, got {prompt_len}")
    shared = int(prompt_len * shared_prefix_ratio)
    shared = max(1, min(shared, prompt_len - 1)) if prompt_len > 1 else 1
    suffix = prompt_len - shared
    toks = _make_family_prefix(family, shared)
    if suffix > 0:
        tail_base = 50000 + spec_index * 131
        toks.extend(tail_base + (i % 47) for i in range(suffix))
    return toks


def _distribution_families(dist: str, num_requests: int, rng: random.Random, skew_alpha: float) -> list[int]:
    k = max(1, math.ceil(math.sqrt(num_requests)))
    if dist == "identical":
        return [0 for _ in range(num_requests)]
    if dist == "distinct":
        return list(range(num_requests))
    if dist == "uniform":
        return [int(rng.randrange(k)) for _ in range(num_requests)]
    if dist == "skewed":
        weights = [1.0 / ((i + 1) ** skew_alpha) for i in range(k)]
        return [int(rng.choices(range(k), weights=weights, k=1)[0]) for _ in range(num_requests)]
    raise ValueError(f"unknown distribution: {dist}")


def _build_specs(
    *,
    base_url: str,
    model: str,
    num_requests: int,
    dist: str,
    prompt_lens: list[int],
    prompt_len_weights: list[float],
    max_tokens_values: list[int],
    max_tokens_weights: list[float],
    shared_prefix_ratio: float,
    arrival_rate_qps: float,
    timeout_s: float,
    seed: int,
    headers: dict[str, str],
) -> list[RequestSpec]:
    rng = random.Random(seed)
    families = _distribution_families(dist, num_requests, rng, skew_alpha=1.5)
    arrivals: list[float] = []
    t = 0.0
    for _ in range(num_requests):
        gap = rng.expovariate(arrival_rate_qps) if arrival_rate_qps > 0 else 0.0
        t += gap
        arrivals.append(t)
    specs: list[RequestSpec] = []
    session_ids = [
        _create_sampling_session(base_url, model, timeout_s=timeout_s, headers=headers)
        for _ in range(num_requests)
    ]
    for i in range(num_requests):
        prompt_len = _weighted_choice(rng, prompt_lens, prompt_len_weights)
        max_tokens = _weighted_choice(rng, max_tokens_values, max_tokens_weights)
        prompt_tokens = _make_prompt_tokens(i, families[i], prompt_len, shared_prefix_ratio)
        specs.append(
            RequestSpec(
                request_index=i,
                family=families[i],
                prompt_len=prompt_len,
                max_tokens=max_tokens,
                arrival_s=arrivals[i],
                session_id=session_ids[i],
                prompt_tokens=prompt_tokens,
            )
        )
    return specs


def _run_one(
    *,
    base_url: str,
    model: str,
    spec: RequestSpec,
    num_samples: int,
    call_timeout_s: float,
    poll_s: float,
    expect_sequences: int | None,
    headers: dict[str, str],
) -> RequestResult:
    t0 = time.time()
    try:
        req_id = _post(
            base_url,
            "/api/v1/asample",
            {
                "sampling_session_id": spec.session_id,
                "seq_id": 0,
                "num_samples": int(num_samples),
                "prompt": {"chunks": [{"tokens": spec.prompt_tokens, "type": "encoded_text"}]},
                "sampling_params": {
                    "max_tokens": int(spec.max_tokens),
                    "temperature": 0.7,
                    "top_k": -1,
                    "top_p": 1.0,
                },
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
            timeout_s=call_timeout_s,
            headers=headers,
        )["request_id"]
        while True:
            if time.time() - t0 > call_timeout_s:
                return RequestResult(
                    request_index=spec.request_index,
                    family=spec.family,
                    prompt_len=spec.prompt_len,
                    max_tokens=spec.max_tokens,
                    arrival_s=spec.arrival_s,
                    ok=False,
                    elapsed_s=time.time() - t0,
                    generated_tokens=0,
                    sequence_count=None,
                    error=f"TimeoutError: retrieve_future exceeded {call_timeout_s}s",
                )
            r = requests.post(
                f"{base_url}/api/v1/retrieve_future",
                json={"request_id": req_id, "model_id": spec.session_id},
                timeout=min(call_timeout_s, 30.0),
                headers=headers,
            )
            if r.status_code == 408:
                time.sleep(poll_s)
                continue
            r.raise_for_status()
            out = r.json()
            if isinstance(out, dict) and "error" in out:
                err = out.get("error")
                return RequestResult(
                    request_index=spec.request_index,
                    family=spec.family,
                    prompt_len=spec.prompt_len,
                    max_tokens=spec.max_tokens,
                    arrival_s=spec.arrival_s,
                    ok=False,
                    elapsed_s=time.time() - t0,
                    generated_tokens=0,
                    sequence_count=None,
                    error=str(err)[:500] if err is not None else "unknown_error",
                )
            seqs = out.get("sequences") if isinstance(out, dict) else None
            n = len(seqs) if isinstance(seqs, list) else None
            if expect_sequences is not None and n is not None and n != expect_sequences:
                return RequestResult(
                    request_index=spec.request_index,
                    family=spec.family,
                    prompt_len=spec.prompt_len,
                    max_tokens=spec.max_tokens,
                    arrival_s=spec.arrival_s,
                    ok=False,
                    elapsed_s=time.time() - t0,
                    generated_tokens=0,
                    sequence_count=n,
                    error=f"sequence_count_mismatch expected={expect_sequences} got={n}",
                )
            generated_tokens = 0
            if isinstance(seqs, list):
                for seq in seqs:
                    if isinstance(seq, dict):
                        toks = seq.get("tokens")
                        if isinstance(toks, list):
                            generated_tokens += len(toks)
            return RequestResult(
                request_index=spec.request_index,
                family=spec.family,
                prompt_len=spec.prompt_len,
                max_tokens=spec.max_tokens,
                arrival_s=spec.arrival_s,
                ok=True,
                elapsed_s=time.time() - t0,
                generated_tokens=generated_tokens,
                sequence_count=n,
                error=None,
            )
    except Exception as e:
        return RequestResult(
            request_index=spec.request_index,
            family=spec.family,
            prompt_len=spec.prompt_len,
            max_tokens=spec.max_tokens,
            arrival_s=spec.arrival_s,
            ok=False,
            elapsed_s=time.time() - t0,
            generated_tokens=0,
            sequence_count=None,
            error=f"{type(e).__name__}: {e}",
        )


def _summarize_metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    def delta(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    q_n = delta("mint_vllm_queue_time_s_count")
    p_n = delta("mint_vllm_prefill_time_s_count")
    d_n = delta("mint_vllm_decode_time_s_count")
    t_n = delta("mint_vllm_time_per_output_token_s_count")
    prefix_q = delta("mint_vllm_prefix_cache_queries_total")
    prefix_h = delta("mint_vllm_prefix_cache_hits_total")
    return {
        "queue_time_avg_s": (delta("mint_vllm_queue_time_s_sum") / q_n) if q_n > 0 else None,
        "prefill_time_avg_s": (delta("mint_vllm_prefill_time_s_sum") / p_n) if p_n > 0 else None,
        "decode_time_avg_s": (delta("mint_vllm_decode_time_s_sum") / d_n) if d_n > 0 else None,
        "time_per_output_token_avg_s": (delta("mint_vllm_time_per_output_token_s_sum") / t_n) if t_n > 0 else None,
        "prefix_cache_query_delta": prefix_q,
        "prefix_cache_hit_delta": prefix_h,
        "prefix_cache_hit_ratio_delta": (prefix_h / prefix_q) if prefix_q > 0 else None,
        "workload_request_delta": delta("mint_vllm_workload_requests_total"),
        "workload_prompt_token_delta": delta("mint_vllm_workload_prompt_tokens_total"),
        "workload_generated_token_delta": delta("mint_vllm_workload_generated_tokens_total"),
    }


def _summarize_poll(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    waits = [s.get("mint_vllm_scheduler_waiting_requests", 0.0) for s in samples]
    runs = [s.get("mint_vllm_scheduler_running_requests", 0.0) for s in samples]
    kvs = [s.get("mint_vllm_scheduler_kv_cache_usage_ratio", 0.0) for s in samples]
    if not samples:
        return {
            "poll_samples": 0,
            "waiting_avg": None,
            "waiting_max": None,
            "running_avg": None,
            "running_max": None,
            "kv_avg": None,
            "kv_max": None,
        }
    return {
        "poll_samples": len(samples),
        "waiting_avg": statistics.mean(waits),
        "waiting_max": max(waits),
        "running_avg": statistics.mean(runs),
        "running_max": max(runs),
        "kv_avg": statistics.mean(kvs),
        "kv_max": max(kvs),
    }


def _run_distribution(
    *,
    base_url: str,
    model: str,
    distribution: str,
    repeat: int,
    num_requests: int,
    num_samples: int,
    prompt_lens: list[int],
    prompt_len_weights: list[float],
    max_tokens_values: list[int],
    max_tokens_weights: list[float],
    shared_prefix_ratio: float,
    arrival_rate_qps: float,
    call_timeout_s: float,
    poll_s: float,
    metrics_poll_s: float,
    metrics_settle_s: float,
    out_dir: Path,
    seed: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    specs = _build_specs(
        base_url=base_url,
        model=model,
        num_requests=num_requests,
        dist=distribution,
        prompt_lens=prompt_lens,
        prompt_len_weights=prompt_len_weights,
        max_tokens_values=max_tokens_values,
        max_tokens_weights=max_tokens_weights,
        shared_prefix_ratio=shared_prefix_ratio,
        arrival_rate_qps=arrival_rate_qps,
        timeout_s=call_timeout_s,
        seed=seed,
        headers=headers,
    )

    before = _fetch_metrics(base_url, model, timeout_s=30.0, headers=headers)
    poll_stop = threading.Event()
    poll_rows: list[dict[str, Any]] = []

    def _poller() -> None:
        while not poll_stop.is_set():
            try:
                row = _fetch_metrics(base_url, model, timeout_s=30.0, headers=headers)
                row["ts"] = time.time()
                poll_rows.append(row)
            except Exception as e:
                poll_rows.append({"ts": time.time(), "error": f"{type(e).__name__}: {e}"})
            poll_stop.wait(metrics_poll_s)

    poll_thread = threading.Thread(target=_poller, daemon=True)
    poll_thread.start()

    req_path = out_dir / f"requests_{distribution}_rep{repeat}.jsonl"
    wall_t0 = time.monotonic()
    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as ex:
        futs: list[concurrent.futures.Future[RequestResult]] = []
        for spec in specs:
            target = wall_t0 + spec.arrival_s
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            futs.append(
                ex.submit(
                    _run_one,
                    base_url=base_url,
                    model=model,
                    spec=spec,
                    num_samples=num_samples,
                    call_timeout_s=call_timeout_s,
                    poll_s=poll_s,
                    expect_sequences=num_samples,
                    headers=headers,
                )
            )
        for fut in futs:
            results.append(fut.result())
    wall_s = time.monotonic() - wall_t0
    if metrics_settle_s > 0:
        time.sleep(metrics_settle_s)
    poll_stop.set()
    poll_thread.join(timeout=metrics_poll_s + metrics_settle_s + 5.0)
    after = _fetch_metrics(base_url, model, timeout_s=30.0, headers=headers)

    req_path.parent.mkdir(parents=True, exist_ok=True)
    with req_path.open("w", encoding="utf-8") as f:
        for spec in specs:
            f.write(json.dumps({"kind": "spec", **asdict(spec)}, sort_keys=True) + "\n")
        for res in results:
            f.write(json.dumps({"kind": "result", **asdict(res)}, sort_keys=True) + "\n")

    ok = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]
    latencies = [r.elapsed_s for r in ok]
    gen_toks = sum(r.generated_tokens for r in ok)
    prompt_toks = sum(r.prompt_len for r in ok)
    family_counts: dict[int, int] = {}
    for spec in specs:
        family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
    summary: dict[str, Any] = {
        "ts": _ts(),
        "distribution": distribution,
        "repeat": repeat,
        "ok": len(err) == 0,
        "n_ok": len(ok),
        "n_err": len(err),
        "num_requests": num_requests,
        "num_samples": num_samples,
        "arrival_rate_qps": arrival_rate_qps,
        "shared_prefix_ratio": shared_prefix_ratio,
        "wall_s": wall_s,
        "request_throughput_rps": (len(ok) / wall_s) if wall_s > 0 else None,
        "prompt_token_throughput_tps": (prompt_toks / wall_s) if wall_s > 0 else None,
        "generated_token_throughput_tps": (gen_toks / wall_s) if wall_s > 0 else None,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_max_s": max(latencies) if latencies else None,
        "family_count": len(family_counts),
        "family_histogram": {str(k): v for k, v in sorted(family_counts.items())},
        "errors": [r.error for r in err[:5]],
        "metrics_delta": _summarize_metric_deltas(before, after),
        "metrics_poll": _summarize_poll([x for x in poll_rows if "error" not in x]),
        "request_jsonl": str(req_path),
    }
    metrics_path = out_dir / f"metrics_{distribution}_rep{repeat}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "before": before,
                "after": after,
                "poll": poll_rows,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary["metrics_json"] = str(metrics_path)
    print(
        json.dumps(
            {
                "distribution": distribution,
                "repeat": repeat,
                "ok": summary["ok"],
                "latency_p50_s": summary["latency_p50_s"],
                "request_throughput_rps": summary["request_throughput_rps"],
                "waiting_max": summary["metrics_poll"]["waiting_max"],
                "running_max": summary["metrics_poll"]["running_max"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--distributions", default="distinct,uniform,skewed,identical")
    p.add_argument("--api-key", default=os.environ.get("MINT_API_KEY") or os.environ.get("MINT_API_KEY") or "")
    p.add_argument("--num-requests", type=int, default=24)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--prompt-lens", default="2048,8192,16000")
    p.add_argument("--prompt-len-weights", default="0.2,0.35,0.45")
    p.add_argument("--max-tokens-values", default="16,32,64")
    p.add_argument("--max-tokens-weights", default="0.2,0.6,0.2")
    p.add_argument("--shared-prefix-ratio", type=float, default=0.75)
    p.add_argument("--arrival-rate-qps", type=float, default=0.35)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--call-timeout-s", type=float, default=3600.0)
    p.add_argument("--poll-s", type=float, default=1.0)
    p.add_argument("--metrics-poll-s", type=float, default=1.0)
    p.add_argument("--metrics-settle-s", type=float, default=5.0)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    base_url = str(args.base_url).rstrip("/")
    distributions = [x.strip() for x in str(args.distributions).split(",") if x.strip()]
    prompt_lens = _parse_int_list(args.prompt_lens)
    prompt_len_weights = _normalize(_parse_float_list(args.prompt_len_weights))
    max_tokens_values = _parse_int_list(args.max_tokens_values)
    max_tokens_weights = _normalize(_parse_float_list(args.max_tokens_weights))
    if len(prompt_lens) != len(prompt_len_weights):
        raise SystemExit("prompt lens and weights length mismatch")
    if len(max_tokens_values) != len(max_tokens_weights):
        raise SystemExit("max tokens values and weights length mismatch")
    if args.num_requests < 1:
        raise SystemExit("--num-requests must be >= 1")
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if not (0.0 <= args.shared_prefix_ratio <= 1.0):
        raise SystemExit("--shared-prefix-ratio must be in [0, 1]")

    headers = _headers(args.api_key)
    _get(base_url, "/api/v1/server_info", timeout_s=30, headers=headers)
    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "issue512" / _ts_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "ts": _ts(),
        "base_url": base_url,
        "model": args.model,
        "distributions": distributions,
        "num_requests": args.num_requests,
        "num_samples": args.num_samples,
        "prompt_lens": prompt_lens,
        "prompt_len_weights": prompt_len_weights,
        "max_tokens_values": max_tokens_values,
        "max_tokens_weights": max_tokens_weights,
        "shared_prefix_ratio": args.shared_prefix_ratio,
        "arrival_rate_qps": args.arrival_rate_qps,
        "repeats": args.repeats,
        "seed": args.seed,
        "api_key_present": bool(str(args.api_key).strip()),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summaries: list[dict[str, Any]] = []
    for rep in range(args.repeats):
        for i, dist in enumerate(distributions):
            summaries.append(
                _run_distribution(
                    base_url=base_url,
                    model=args.model,
                    distribution=dist,
                    repeat=rep,
                    num_requests=args.num_requests,
                    num_samples=args.num_samples,
                    prompt_lens=prompt_lens,
                    prompt_len_weights=prompt_len_weights,
                    max_tokens_values=max_tokens_values,
                    max_tokens_weights=max_tokens_weights,
                    shared_prefix_ratio=args.shared_prefix_ratio,
                    arrival_rate_qps=args.arrival_rate_qps,
                    call_timeout_s=args.call_timeout_s,
                    poll_s=args.poll_s,
                    metrics_poll_s=args.metrics_poll_s,
                    metrics_settle_s=args.metrics_settle_s,
                    out_dir=run_dir,
                    seed=args.seed + rep * 1000 + i * 100,
                    headers=headers,
                )
            )

    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(run_dir), flush=True)


if __name__ == "__main__":
    main()
