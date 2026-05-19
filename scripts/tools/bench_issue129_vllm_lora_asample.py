#!/usr/bin/env python3
"""Issue #129 benchmark: concurrent /asample for vLLM LoRA with long prompts.

Hard constraint: measure only concurrent /asample usage (no sequential sampling loops).
This script submits concurrent /api/v1/asample requests and polls /retrieve_future.

Artifacts:
- JSONL with per-request client timing + output token counts, plus run metadata.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


def _ts_dir() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


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


@dataclass(frozen=True)
class OneRequestResult:
    ok: bool
    request_id: str | None
    client_total_s: float
    output_tokens: int | None
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
) -> OneRequestResult:
    t0 = time.perf_counter()
    request_id: str | None = None
    try:
        out = _post(
            base_url,
            "/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": int(num_samples),
                "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
                "sampling_params": {
                    "max_tokens": int(max_tokens),
                    "temperature": 1.0,
                    "top_k": -1,
                    "top_p": 1.0,
                },
                "prompt_logprobs": bool(prompt_logprobs),
                "topk_prompt_logprobs": int(topk_prompt_logprobs),
            },
            timeout_s=min(call_timeout_s, 300.0),
        )
        request_id = str(out["request_id"])
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed > call_timeout_s:
                return OneRequestResult(
                    ok=False,
                    request_id=request_id,
                    client_total_s=elapsed,
                    output_tokens=None,
                    sequences=None,
                    error=f"TimeoutError: retrieve_future exceeded {call_timeout_s}s",
                )
            r = requests.post(
                f"{base_url}/api/v1/retrieve_future",
                json={"request_id": request_id, "model_id": sampling_session_id},
                timeout=min(call_timeout_s, 30.0),
            )
            if r.status_code == 408:
                time.sleep(poll_s)
                continue
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and "error" in payload:
                err = payload.get("error")
                return OneRequestResult(
                    ok=False,
                    request_id=request_id,
                    client_total_s=time.perf_counter() - t0,
                    output_tokens=None,
                    sequences=None,
                    error=str(err)[:800] if err is not None else "unknown_error",
                )
            seqs = payload.get("sequences") if isinstance(payload, dict) else None
            n_seqs = len(seqs) if isinstance(seqs, list) else None
            out_tokens = None
            if isinstance(seqs, list):
                total = 0
                for s in seqs:
                    toks = s.get("tokens") if isinstance(s, dict) else None
                    if isinstance(toks, list):
                        total += len(toks)
                out_tokens = total
            return OneRequestResult(
                ok=True,
                request_id=request_id,
                client_total_s=time.perf_counter() - t0,
                output_tokens=out_tokens,
                sequences=n_seqs,
                error=None,
            )
    except Exception as e:
        return OneRequestResult(
            ok=False,
            request_id=request_id,
            client_total_s=time.perf_counter() - t0,
            output_tokens=None,
            sequences=None,
            error=f"{type(e).__name__}: {e}",
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:8000")
    p.add_argument("--api-key", default=os.environ.get("MINT_API_KEY") or os.environ.get("MINT_API_KEY") or None)
    p.add_argument("--label", default="", help="Freeform label to tag the run (e.g. coalesce0/coalesce1)")
    p.add_argument("--model", required=True, help="HF base model name")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank for the benchmark session")
    p.add_argument("--prompt-len", type=int, default=32000)
    p.add_argument("--gen-max-tokens", type=int, default=256)
    p.add_argument("--n-prompts", default="1,2,4,8", help="Comma-separated n_prompts values (concurrent /asample requests)")
    p.add_argument("--n-samples", default="1,2,4,8", help="Comma-separated n_samples values (num_samples per /asample)")
    p.add_argument("--prompt-logprobs", default="0,1", help="Comma-separated {0,1} include_prompt_logprobs values")
    p.add_argument(
        "--prompt-reuse",
        default="same_prompt,unique_prompts",
        help="Comma-separated {same_prompt,unique_prompts}",
    )
    p.add_argument(
        "--unique-batches",
        action="store_true",
        help="Make prompt identical within a batch, but unique across (case,repeat) to avoid cross-repeat prefix caching.",
    )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--poll-s", type=float, default=0.2)
    p.add_argument("--call-timeout-s", type=float, default=3600.0)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    base_url = str(args.base_url).rstrip("/")
    n_prompts_list = _parse_int_list(args.n_prompts)
    n_samples_list = _parse_int_list(args.n_samples)
    prompt_logprobs_list = _parse_int_list(args.prompt_logprobs)
    prompt_reuse_list = [s.strip() for s in str(args.prompt_reuse).split(",") if s.strip()]
    if any(n < 1 for n in n_prompts_list):
        raise SystemExit(f"invalid --n-prompts: {n_prompts_list}")
    if any(n < 1 for n in n_samples_list):
        raise SystemExit(f"invalid --n-samples: {n_samples_list}")
    if any(v not in (0, 1) for v in prompt_logprobs_list):
        raise SystemExit(f"invalid --prompt-logprobs: {prompt_logprobs_list} (expected 0/1)")
    if any(v not in ("same_prompt", "unique_prompts") for v in prompt_reuse_list):
        raise SystemExit(f"invalid --prompt-reuse: {prompt_reuse_list}")
    if int(args.prompt_len) < 1:
        raise SystemExit("--prompt-len must be >= 1")
    if int(args.gen_max_tokens) < 1:
        raise SystemExit("--gen-max-tokens must be >= 1")
    if int(args.rank) < 1:
        raise SystemExit("--rank must be >= 1")

    info = _get(base_url, "/api/v1/server_info", timeout_s=30)
    git_sha = info.get("git_sha")

    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "issue87" / _ts_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"issue129_{args.model.replace('/', '_')}_pl{int(args.prompt_len)}_mt{int(args.gen_max_tokens)}.jsonl"

    import mint

    service_client = mint.ServiceClient(base_url=base_url, api_key=args.api_key)

    t_create0 = time.perf_counter()
    tc = service_client.create_lora_training_client(
        base_model=str(args.model),
        rank=int(args.rank),
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    t_create1 = time.perf_counter()

    tokenizer = tc.get_tokenizer()
    filler_ids = tokenizer.encode(" a", add_special_tokens=False) or tokenizer.encode("0", add_special_tokens=False)
    if not filler_ids:
        raise SystemExit("failed to get filler token id from tokenizer")
    filler_id = int(filler_ids[0])
    vocab_size = getattr(tokenizer, "vocab_size", None)
    vocab_size_i = int(vocab_size) if isinstance(vocab_size, int) else None

    t_save0 = time.perf_counter()
    sc = tc.save_weights_and_get_sampling_client(name="issue129_bench")
    t_save1 = time.perf_counter()
    sampling_session_id = str(getattr(sc, "_sampling_session_id"))

    base_prompt_tokens = [filler_id] * int(args.prompt_len)

    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": _now_iso(),
                    "kind": "meta",
                    "base_url": base_url,
                    "git_sha": git_sha,
                    "label": str(args.label),
                    "model": str(args.model),
                    "rank": int(args.rank),
                    "sampling_session_id": sampling_session_id,
                    "prompt_len": int(args.prompt_len),
                    "gen_max_tokens": int(args.gen_max_tokens),
                    "n_prompts": n_prompts_list,
                    "n_samples": n_samples_list,
                    "prompt_reuse": prompt_reuse_list,
                    "prompt_logprobs": prompt_logprobs_list,
                    "repeats": int(args.repeats),
                    "unique_batches": bool(args.unique_batches),
                    "filler_id": filler_id,
                    "tokenizer_vocab_size": vocab_size_i,
                    "create_training_client_s": float(t_create1 - t_create0),
                    "save_weights_and_get_sampling_client_s": float(t_save1 - t_save0),
                },
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()

        case_idx = 0
        for n_prompts in n_prompts_list:
            for n_samples in n_samples_list:
                for prompt_reuse in prompt_reuse_list:
                    for plp in prompt_logprobs_list:
                        for rep in range(int(args.repeats)):
                            case_idx += 1
                            barrier = threading.Barrier(n_prompts + 1)

                            if prompt_reuse == "same_prompt":
                                if args.unique_batches:
                                    first_tok = 1000 + case_idx + rep
                                    if vocab_size_i is not None and first_tok >= vocab_size_i:
                                        raise SystemExit(f"unique batch token {first_tok} >= vocab_size {vocab_size_i}")
                                    toks = list(base_prompt_tokens)
                                    toks[0] = int(first_tok)
                                    prompt_tokens_by_req = [toks] * n_prompts
                                else:
                                    prompt_tokens_by_req = [base_prompt_tokens] * n_prompts
                            else:
                                prompt_tokens_by_req = []
                                for i in range(n_prompts):
                                    first_tok = 1000 + (case_idx * 1000 + rep * 100 + i)
                                    if vocab_size_i is not None and first_tok >= vocab_size_i:
                                        raise SystemExit(f"unique prompt token {first_tok} >= vocab_size {vocab_size_i}")
                                    toks = list(base_prompt_tokens)
                                    toks[0] = int(first_tok)
                                    prompt_tokens_by_req.append(toks)

                            f.write(
                                json.dumps(
                                    {
                                        "ts": _now_iso(),
                                        "kind": "case",
                                        "case_idx": case_idx,
                                        "repeat": rep,
                                        "n_prompts": n_prompts,
                                        "n_samples": int(n_samples),
                                        "prompt_reuse": prompt_reuse,
                                        "prompt_logprobs": int(plp),
                                        "topk_prompt_logprobs": 0,
                                        "poll_s": float(args.poll_s),
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            f.flush()

                            def _worker(prompt_tokens: list[int]) -> OneRequestResult:
                                barrier.wait()
                                return _run_one(
                                    base_url=base_url,
                                    sampling_session_id=sampling_session_id,
                                    prompt_tokens=prompt_tokens,
                                    num_samples=int(n_samples),
                                    max_tokens=int(args.gen_max_tokens),
                                    prompt_logprobs=bool(plp),
                                    topk_prompt_logprobs=0,
                                    poll_s=float(args.poll_s),
                                    call_timeout_s=float(args.call_timeout_s),
                                )

                            t_wall0 = time.perf_counter()
                            with ThreadPoolExecutor(max_workers=n_prompts) as ex:
                                futs = [ex.submit(_worker, pt) for pt in prompt_tokens_by_req]
                                barrier.wait()
                                results = [fu.result() for fu in futs]
                            t_wall1 = time.perf_counter()

                            for i, r in enumerate(results):
                                f.write(
                                    json.dumps(
                                        {
                                            "ts": _now_iso(),
                                            "kind": "request",
                                            "case_idx": case_idx,
                                            "repeat": rep,
                                            "req_idx": i,
                                            "n_prompts": n_prompts,
                                            "n_samples": int(n_samples),
                                            "prompt_reuse": prompt_reuse,
                                            "prompt_logprobs": int(plp),
                                            "prompt_len": int(args.prompt_len),
                                            "gen_max_tokens": int(args.gen_max_tokens),
                                            "prompt_token0": int(prompt_tokens_by_req[i][0]),
                                            "ok": bool(r.ok),
                                            "request_id": r.request_id,
                                            "client_total_s": float(r.client_total_s),
                                            "output_tokens": r.output_tokens,
                                            "sequences": r.sequences,
                                            "error": r.error,
                                            "batch_wall_s": float(t_wall1 - t_wall0),
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                            f.flush()

                            ok = sum(1 for r in results if r.ok)
                            err = sum(1 for r in results if not r.ok)
                            print(
                                f"[{_now_iso()}] case={case_idx} n_prompts={n_prompts} n_samples={n_samples} "
                                f"reuse={prompt_reuse} plp={plp} rep={rep} ok={ok} err={err} wall_s={t_wall1 - t_wall0:.2f}",
                                flush=True,
                            )

    print(str(out_path), flush=True)


if __name__ == "__main__":
    main()

