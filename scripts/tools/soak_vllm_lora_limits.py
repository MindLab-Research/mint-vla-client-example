#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

import requests


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _post_json(base_url: str, api_key: str, path: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    r = requests.post(f"{base_url}{path}", headers=_headers(api_key), json=payload, timeout=timeout_s)
    try:
        out = r.json()
    except Exception:
        out = {"_non_json_body": r.text[:2000]}
    if not isinstance(out, dict):
        out = {"_non_dict_json": repr(out)}
    return r.status_code, out


def _get_json(base_url: str, api_key: str, path: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
    r = requests.get(f"{base_url}{path}", headers=_headers(api_key), timeout=timeout_s)
    try:
        out = r.json()
    except Exception:
        out = {"_non_json_body": r.text[:2000]}
    if not isinstance(out, dict):
        out = {"_non_dict_json": repr(out)}
    return r.status_code, out


def _tokens(length: int, seed: int) -> list[int]:
    return [100 + ((seed + i) % 1000) for i in range(length)]


def _create_sampling_session(base_url: str, api_key: str, base_model: str, model_path: str, lora_rank: int) -> str:
    st, out = _post_json(
        base_url,
        api_key,
        "/api/v1/create_session",
        {"tags": ["soak_vllm_lora_limits"], "user_metadata": {}, "sdk_version": "soak_vllm_lora_limits"},
        60.0,
    )
    if st != 200:
        raise RuntimeError(f"create_session returned {st}: {out!r}")
    session_id = out.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"missing session_id: {out!r}")
    st, out = _post_json(
        base_url,
        api_key,
        "/api/v1/create_sampling_session",
        {
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": base_model,
            "model_path": model_path,
            "lora_rank": lora_rank,
        },
        1800.0,
    )
    if st != 200:
        raise RuntimeError(f"create_sampling_session returned {st}: {out!r}")
    sid = out.get("sampling_session_id")
    if not isinstance(sid, str) or not sid:
        raise RuntimeError(f"missing sampling_session_id: {out!r}")
    return sid


@dataclass
class RequestOutcome:
    idx: int
    kind: str
    request_id: str | None
    terminal_status: int | None
    ok: bool
    elapsed_s: float
    error: str | None


def _poll_future(base_url: str, api_key: str, request_id: str, timeout_s: float, poll_s: float) -> tuple[int, dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st, out = _post_json(base_url, api_key, "/api/v1/retrieve_future", {"request_id": request_id}, 60.0)
        if st != 408:
            return st, out
        time.sleep(poll_s)
    raise TimeoutError(f"timeout polling {request_id}")


def _run_request(
    *,
    base_url: str,
    api_key: str,
    sampling_session_id: str,
    idx: int,
    kind: str,
    prompt_len: int,
    max_tokens: int,
    prompt_logprobs: bool,
    timeout_s: float,
    poll_s: float,
) -> RequestOutcome:
    start = time.time()
    request_id: str | None = None
    try:
        st, out = _post_json(
            base_url,
            api_key,
            "/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": 1,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": _tokens(prompt_len, idx * 7919 + prompt_len)}]},
                "sampling_params": {
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                },
                "prompt_logprobs": prompt_logprobs,
            },
            120.0,
        )
        if st != 200:
            return RequestOutcome(idx, kind, None, st, False, time.time() - start, f"submit_status={st} out={out!r}")
        request_id = out.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return RequestOutcome(idx, kind, None, st, False, time.time() - start, f"missing request_id: {out!r}")
        st, out = _poll_future(base_url, api_key, request_id, timeout_s, poll_s)
        if st != 200:
            return RequestOutcome(idx, kind, request_id, st, False, time.time() - start, f"terminal_status={st} out={out!r}")
        if "error" in out:
            return RequestOutcome(idx, kind, request_id, st, False, time.time() - start, str(out.get("error"))[:4000])
        return RequestOutcome(idx, kind, request_id, st, True, time.time() - start, None)
    except Exception as e:
        return RequestOutcome(idx, kind, request_id, None, False, time.time() - start, f"{type(e).__name__}: {e}")


def _run_compute_logprobs(
    *,
    base_url: str,
    api_key: str,
    sampling_session_id: str,
    idx: int,
    prompt_len: int,
    timeout_s: float,
    poll_s: float,
) -> RequestOutcome:
    start = time.time()
    request_id: str | None = None
    try:
        st, out = _post_json(
            base_url,
            api_key,
            "/api/v1/compute_logprobs",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "sequence": {"chunks": [{"type": "encoded_text", "tokens": _tokens(prompt_len, idx * 3571 + prompt_len)}]},
            },
            120.0,
        )
        if st != 200:
            return RequestOutcome(idx, "compute_logprobs", None, st, False, time.time() - start, f"submit_status={st} out={out!r}")
        request_id = out.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return RequestOutcome(idx, "compute_logprobs", None, st, False, time.time() - start, f"missing request_id: {out!r}")
        st, out = _poll_future(base_url, api_key, request_id, timeout_s, poll_s)
        if st != 200:
            return RequestOutcome(idx, "compute_logprobs", request_id, st, False, time.time() - start, f"terminal_status={st} out={out!r}")
        if "error" in out:
            return RequestOutcome(idx, "compute_logprobs", request_id, st, False, time.time() - start, str(out.get("error"))[:4000])
        return RequestOutcome(idx, "compute_logprobs", request_id, st, True, time.time() - start, None)
    except Exception as e:
        return RequestOutcome(idx, "compute_logprobs", request_id, None, False, time.time() - start, f"{type(e).__name__}: {e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=(os.environ.get("MINT_BASE_URL") or "http://localhost:8002").rstrip("/"))
    p.add_argument("--api-key", default=os.environ.get("MINT_API_KEY", "dummy"))
    p.add_argument("--base-model", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--lora-rank", type=int, required=True)
    p.add_argument("--max-loras", type=int, required=True)
    p.add_argument("--overflow", type=int, default=4)
    p.add_argument("--long-prompt-len", type=int, default=32000)
    p.add_argument("--short-prompt-len", type=int, default=512)
    p.add_argument("--timeout-s", type=float, default=1800.0)
    p.add_argument("--poll-s", type=float, default=0.5)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    base_url = str(args.base_url).rstrip("/")
    api_key = str(args.api_key)
    session_count = int(args.max_loras) + int(args.overflow)

    result: dict[str, Any] = {
        "base_url": base_url,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "lora_rank": args.lora_rank,
        "max_loras": args.max_loras,
        "overflow": args.overflow,
        "session_count": session_count,
        "started_at": time.time(),
    }

    st, health = _get_json(base_url, api_key, "/api/v1/healthz", 10.0)
    result["healthz"] = {"status": st, "body": health}
    if st != 200:
        raise SystemExit(json.dumps({"error": f"healthz failed: {st} {health!r}"}))

    sessions = [
        _create_sampling_session(base_url, api_key, args.base_model, args.model_path, args.lora_rank)
        for _ in range(session_count)
    ]
    result["sampling_session_ids"] = sessions

    outcomes: list[RequestOutcome] = []
    with ThreadPoolExecutor(max_workers=session_count) as pool:
        futs = []
        for i, sid in enumerate(sessions):
            mod = i % 4
            if mod == 0:
                futs.append(
                    pool.submit(
                        _run_request,
                        base_url=base_url,
                        api_key=api_key,
                        sampling_session_id=sid,
                        idx=i,
                        kind="long_prompt_logprobs",
                        prompt_len=args.long_prompt_len,
                        max_tokens=1,
                        prompt_logprobs=True,
                        timeout_s=args.timeout_s,
                        poll_s=args.poll_s,
                    )
                )
            elif mod == 1:
                futs.append(
                    pool.submit(
                        _run_compute_logprobs,
                        base_url=base_url,
                        api_key=api_key,
                        sampling_session_id=sid,
                        idx=i,
                        prompt_len=args.long_prompt_len,
                        timeout_s=args.timeout_s,
                        poll_s=args.poll_s,
                    )
                )
            elif mod == 2:
                futs.append(
                    pool.submit(
                        _run_request,
                        base_url=base_url,
                        api_key=api_key,
                        sampling_session_id=sid,
                        idx=i,
                        kind="long_sample",
                        prompt_len=args.long_prompt_len,
                        max_tokens=8,
                        prompt_logprobs=False,
                        timeout_s=args.timeout_s,
                        poll_s=args.poll_s,
                    )
                )
            else:
                futs.append(
                    pool.submit(
                        _run_request,
                        base_url=base_url,
                        api_key=api_key,
                        sampling_session_id=sid,
                        idx=i,
                        kind="short_sample",
                        prompt_len=args.short_prompt_len,
                        max_tokens=16,
                        prompt_logprobs=False,
                        timeout_s=args.timeout_s,
                        poll_s=args.poll_s,
                    )
                )
        for fut in as_completed(futs):
            outcomes.append(fut.result())

    st, actors = _get_json(base_url, api_key, "/internal/actors", 10.0)
    result["actors_after"] = {"status": st, "body": actors}
    result["finished_at"] = time.time()
    result["outcomes"] = [asdict(x) for x in sorted(outcomes, key=lambda r: r.idx)]
    result["summary"] = {
        "ok_count": sum(1 for x in outcomes if x.ok),
        "error_count": sum(1 for x in outcomes if not x.ok),
        "engine_dead_like": [
            asdict(x)
            for x in outcomes
            if x.error and any(s in x.error for s in ("EngineDeadError", "ActorDiedError", "mint_vllm_multinode_ray_get_failed", "out of memory", "CUDA out of memory"))
        ],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"output_json": args.output_json, "summary": result["summary"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
