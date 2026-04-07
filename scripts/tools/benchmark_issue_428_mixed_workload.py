#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import requests


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


BASE_URL = _env("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = _env("TINKER_API_KEY", "")
MODEL = _env("TINKER_BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
REQUEST_TIMEOUT_S = float(_env("TINKER_REQUEST_TIMEOUT_S", "30"))
POLL_TIMEOUT_S = float(_env("TINKER_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(_env("TINKER_POLL_SLEEP_S", "0.2"))
TRIALS = int(_env("TINKER_TRIALS", "1"))
PROMPT_LEN = int(_env("TINKER_PROMPT_LEN", "4096"))
MULTI_NUM_SAMPLES = int(_env("TINKER_MULTI_NUM_SAMPLES", "8"))
MULTI_MAX_TOKENS = int(_env("TINKER_MULTI_MAX_TOKENS", "256"))
ORDINARY_CONCURRENCY = int(_env("TINKER_ORDINARY_CONCURRENCY", "8"))
ORDINARY_MAX_TOKENS = int(_env("TINKER_ORDINARY_MAX_TOKENS", "256"))
MULTI_SUBMIT_GAP_S = float(_env("TINKER_MULTI_SUBMIT_GAP_S", "0.3"))
ORDINARY_SUBMIT_GAP_S = float(_env("TINKER_ORDINARY_SUBMIT_GAP_S", "0.0"))
SECOND_MULTISAMPLE = _env("TINKER_SECOND_MULTISAMPLE", "0") in {"1", "true", "yes", "on"}
SECOND_MULTI_GAP_S = float(_env("TINKER_SECOND_MULTI_GAP_S", "0.3"))
TEMPERATURE = float(_env("TINKER_TEMPERATURE", "0.7"))
TOP_P = float(_env("TINKER_TOP_P", "1.0"))
SEED = int(_env("TINKER_SEED", "123"))
RUN_LABEL = _env("TINKER_RUN_LABEL", "issue428-mixed")
WARMUP = _env("TINKER_WARMUP", "1") in {"1", "true", "yes", "on"}
WARMUP_MAX_TOKENS = int(_env("TINKER_WARMUP_MAX_TOKENS", "16"))


@dataclass
class RequestResult:
    kind: str
    request_id: str
    submitted_at_s: float
    completed_order: int
    completed_at_s: float
    latency_s: float
    sequence_count: int
    total_output_tokens: int
    error: str | None


@dataclass
class TrialSummary:
    trial: int
    ordinary_before_multi1_count: int
    ordinary_before_multi2_count: int | None
    ordinary_total: int
    ordinary_latency_min_s: float | None
    ordinary_latency_p50_s: float | None
    ordinary_latency_p95_s: float | None
    ordinary_latency_max_s: float | None
    multi1_latency_s: float
    multi2_latency_s: float | None
    total_output_tokens: int
    wall_clock_s: float
    throughput_tokens_per_s: float
    cache_metrics: dict[str, Any] | None


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _fail(message: str, *, payload: dict[str, Any] | None = None) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    if payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 1


def _request(method: str, path: str, *, payload: dict[str, Any] | None = None, timeout_s: float) -> requests.Response:
    return requests.request(
        method=method,
        url=f"{BASE_URL}{path}",
        headers=_headers(),
        json=payload,
        timeout=timeout_s,
    )


def _request_json(method: str, path: str, *, payload: dict[str, Any] | None = None, timeout_s: float, allowed_statuses: set[int] | None = None) -> tuple[int, dict[str, Any]]:
    if allowed_statuses is None:
        allowed_statuses = {200}
    resp = _request(method, path, payload=payload, timeout_s=timeout_s)
    if resp.status_code not in allowed_statuses:
        raise RuntimeError(f"{path} -> {resp.status_code}: {resp.text[:800]!r}")
    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"{path} returned non-JSON body: {type(exc).__name__}: {resp.text[:800]!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} returned non-dict JSON: {type(data)}")
    return resp.status_code, data


def _make_prompt_tokens(prompt_len: int, *, salt: int) -> list[int]:
    # Build a deterministic but non-trivial encoded_text token list.
    base = [151644, 77091, 198]
    filler_len = max(0, prompt_len - len(base) - 1)
    filler = [1000 + ((salt + i) % 200) for i in range(filler_len)]
    return [*base, *filler, 151645]


def _create_session() -> str:
    _status, body = _request_json(
        "POST",
        "/api/v1/create_session",
        payload={
            "tags": ["issue428", RUN_LABEL],
            "user_metadata": {"script": "benchmark_issue_428_mixed_workload.py", "label": RUN_LABEL},
            "sdk_version": "benchmark_issue_428_mixed_workload",
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {body!r}")
    return session_id


def _create_sampling_session(session_id: str) -> str:
    _status, body = _request_json(
        "POST",
        "/api/v1/create_sampling_session",
        payload={
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": MODEL,
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    sampling_session_id = body.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {body!r}")
    return sampling_session_id


def _submit_sample(*, sampling_session_id: str, kind: str, num_samples: int, prompt_tokens: list[int], max_tokens: int) -> tuple[str, float]:
    submitted_at = time.time()
    _status, body = _request_json(
        "POST",
        "/api/v1/asample",
        payload={
            "sampling_session_id": sampling_session_id,
            "num_samples": num_samples,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
            },
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id for {kind}: {body!r}")
    print(f"submitted {kind} request_id={request_id} at={submitted_at:.3f}", flush=True)
    return request_id, submitted_at


def _extract_result(*, kind: str, request_id: str, submitted_at_s: float, body: dict[str, Any], completed_order: int) -> RequestResult:
    err = body.get("error")
    seqs = body.get("sequences")
    sequence_count = len(seqs) if isinstance(seqs, list) else 0
    total_output_tokens = 0
    if isinstance(seqs, list):
        for seq in seqs:
            if isinstance(seq, dict):
                toks = seq.get("tokens")
                if isinstance(toks, list):
                    total_output_tokens += len(toks)
    return RequestResult(
        kind=kind,
        request_id=request_id,
        submitted_at_s=submitted_at_s,
        completed_order=completed_order,
        completed_at_s=time.time(),
        latency_s=time.time() - submitted_at_s,
        sequence_count=sequence_count,
        total_output_tokens=total_output_tokens,
        error=str(err) if err else None,
    )


def _poll_all(requests_by_id: dict[str, tuple[str, float]]) -> list[RequestResult]:
    pending = dict(requests_by_id)
    results: list[RequestResult] = []
    started = time.time()
    completed_order = 0
    while pending:
        if time.time() - started > POLL_TIMEOUT_S:
            raise TimeoutError(f"pending futures remain after {POLL_TIMEOUT_S}s: {list(pending)[:8]}")
        for request_id, (kind, submitted_at_s) in list(pending.items()):
            code, body = _request_json(
                "POST",
                "/api/v1/retrieve_future",
                payload={"request_id": request_id},
                timeout_s=REQUEST_TIMEOUT_S,
                allowed_statuses={200, 408},
            )
            if code == 408:
                continue
            completed_order += 1
            results.append(
                _extract_result(
                    kind=kind,
                    request_id=request_id,
                    submitted_at_s=submitted_at_s,
                    body=body,
                    completed_order=completed_order,
                )
            )
            pending.pop(request_id)
        if pending:
            time.sleep(POLL_SLEEP_S)
    return results


def _maybe_warmup() -> None:
    if not WARMUP:
        return
    sid = _create_session()
    ssid = _create_sampling_session(sid)
    rid, submitted = _submit_sample(
        sampling_session_id=ssid,
        kind="warmup",
        num_samples=1,
        prompt_tokens=_make_prompt_tokens(PROMPT_LEN // 4, salt=999),
        max_tokens=WARMUP_MAX_TOKENS,
    )
    results = _poll_all({rid: ("warmup", submitted)})
    warmup = results[0]
    if warmup.error or warmup.sequence_count != 1:
        raise RuntimeError(f"warmup failed: {asdict(warmup)}")


def _p50(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[len(ys) // 2]


def _p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = int((len(ys) * 95 + 99) // 100) - 1
    idx = max(0, min(idx, len(ys) - 1))
    return ys[idx]


def _run_trial(trial: int) -> TrialSummary:
    session_id = _create_session()
    sampling_session_id = _create_sampling_session(session_id)

    submitted: dict[str, tuple[str, float]] = {}

    multi1_prompt = _make_prompt_tokens(PROMPT_LEN, salt=trial * 1000 + 1)
    multi1_request_id, multi1_submitted = _submit_sample(
        sampling_session_id=sampling_session_id,
        kind="multi1",
        num_samples=MULTI_NUM_SAMPLES,
        prompt_tokens=multi1_prompt,
        max_tokens=MULTI_MAX_TOKENS,
    )
    submitted[multi1_request_id] = ("multi1", multi1_submitted)

    time.sleep(MULTI_SUBMIT_GAP_S)

    for idx in range(ORDINARY_CONCURRENCY):
        ordinary_prompt = _make_prompt_tokens(PROMPT_LEN, salt=trial * 1000 + 100 + idx)
        ordinary_request_id, ordinary_submitted = _submit_sample(
            sampling_session_id=sampling_session_id,
            kind=f"ordinary_{idx}",
            num_samples=1,
            prompt_tokens=ordinary_prompt,
            max_tokens=ORDINARY_MAX_TOKENS,
        )
        submitted[ordinary_request_id] = (f"ordinary_{idx}", ordinary_submitted)
        if ORDINARY_SUBMIT_GAP_S > 0:
            time.sleep(ORDINARY_SUBMIT_GAP_S)

    if SECOND_MULTISAMPLE:
        time.sleep(SECOND_MULTI_GAP_S)
        multi2_prompt = _make_prompt_tokens(PROMPT_LEN, salt=trial * 1000 + 2)
        multi2_request_id, multi2_submitted = _submit_sample(
            sampling_session_id=sampling_session_id,
            kind="multi2",
            num_samples=MULTI_NUM_SAMPLES,
            prompt_tokens=multi2_prompt,
            max_tokens=MULTI_MAX_TOKENS,
        )
        submitted[multi2_request_id] = ("multi2", multi2_submitted)

    results = _poll_all(submitted)
    by_kind = {result.kind: result for result in results}
    multi1 = by_kind["multi1"]
    multi2 = by_kind.get("multi2")
    ordinary = [result for result in results if result.kind.startswith("ordinary_")]

    if multi1.error or multi1.sequence_count != MULTI_NUM_SAMPLES:
        raise RuntimeError(f"multi1 failed: {asdict(multi1)}")
    if multi2 is not None and (multi2.error or multi2.sequence_count != MULTI_NUM_SAMPLES):
        raise RuntimeError(f"multi2 failed: {asdict(multi2)}")
    for ordinary_req in ordinary:
        if ordinary_req.error or ordinary_req.sequence_count != 1:
            raise RuntimeError(f"ordinary failed: {asdict(ordinary_req)}")

    ordinary_before_multi1 = [result for result in ordinary if result.completed_order < multi1.completed_order]
    ordinary_before_multi2 = None
    if multi2 is not None:
        ordinary_before_multi2 = len([result for result in ordinary if result.completed_order < multi2.completed_order])

    all_results = results
    first_submit = min(result.submitted_at_s for result in all_results)
    last_complete = max(result.completed_at_s for result in all_results)
    wall_clock_s = last_complete - first_submit
    total_output_tokens = sum(result.total_output_tokens for result in all_results)
    throughput_tokens_per_s = total_output_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

    ordinary_latencies = [result.latency_s for result in ordinary]
    return TrialSummary(
        trial=trial,
        ordinary_before_multi1_count=len(ordinary_before_multi1),
        ordinary_before_multi2_count=ordinary_before_multi2,
        ordinary_total=len(ordinary),
        ordinary_latency_min_s=min(ordinary_latencies) if ordinary_latencies else None,
        ordinary_latency_p50_s=_p50(ordinary_latencies),
        ordinary_latency_p95_s=_p95(ordinary_latencies),
        ordinary_latency_max_s=max(ordinary_latencies) if ordinary_latencies else None,
        multi1_latency_s=multi1.latency_s,
        multi2_latency_s=multi2.latency_s if multi2 is not None else None,
        total_output_tokens=total_output_tokens,
        wall_clock_s=wall_clock_s,
        throughput_tokens_per_s=throughput_tokens_per_s,
        cache_metrics=None,
    )


def main() -> int:
    print(
        json.dumps(
            {
                "label": RUN_LABEL,
                "base_url": BASE_URL,
                "model": MODEL,
                "prompt_len": PROMPT_LEN,
                "multi_num_samples": MULTI_NUM_SAMPLES,
                "multi_max_tokens": MULTI_MAX_TOKENS,
                "ordinary_concurrency": ORDINARY_CONCURRENCY,
                "ordinary_max_tokens": ORDINARY_MAX_TOKENS,
                "second_multisample": SECOND_MULTISAMPLE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        _maybe_warmup()
        summaries: list[TrialSummary] = []
        for trial in range(1, TRIALS + 1):
            summary = _run_trial(trial)
            summaries.append(summary)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True), flush=True)

        final_summary = {
            "label": RUN_LABEL,
            "trials": [asdict(summary) for summary in summaries],
            "cache_metrics": None,
        }
        print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
