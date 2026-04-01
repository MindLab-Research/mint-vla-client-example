#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import threading
import time
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
POLL_TIMEOUT_S = float(_env("TINKER_POLL_TIMEOUT_S", "600"))
POLL_SLEEP_S = float(_env("TINKER_POLL_SLEEP_S", "0.5"))
SUBMIT_GAP_S = float(_env("TINKER_SUBMIT_GAP_S", "1.0"))
TRIALS = int(_env("TINKER_TRIALS", "2"))
MULTI_NUM_SAMPLES = int(_env("TINKER_MULTI_NUM_SAMPLES", "4"))
MULTI_MAX_TOKENS = int(_env("TINKER_MULTI_MAX_TOKENS", "128"))
ORDINARY_CONCURRENCY = int(_env("TINKER_ORDINARY_CONCURRENCY", "1"))
ORDINARY_MAX_TOKENS = int(_env("TINKER_ORDINARY_MAX_TOKENS", "16"))
EXPECT_ORDINARY_BEFORE_MULTI_MIN = int(_env("TINKER_EXPECT_ORDINARY_BEFORE_MULTI_MIN", str(ORDINARY_CONCURRENCY)))
WARMUP_MAX_TOKENS = int(_env("TINKER_WARMUP_MAX_TOKENS", "8"))
RUN_WARMUP = _env("TINKER_RUN_WARMUP", "1") in {"1", "true", "yes", "on"}

# Minimal qwen-ish encoded prompt prefix that works with the existing mint API.
MULTI_PROMPT_TOKENS = [151644, 77091, 198, 4321, 4322, 4323, 151645]
ORDINARY_PROMPT_TOKENS = [151644, 77091, 198, 2507, 151645]
WARMUP_PROMPT_TOKENS = [151644, 77091, 198, 1111, 151645]


@dataclass
class FutureResult:
    request_id: str
    status_code: int
    completed_at_s: float
    latency_s: float
    sequence_count: int
    first_sequence_len: int
    error: str | None


@dataclass
class TrialSummary:
    trial: int
    multi_request_id: str
    ordinary_request_ids: list[str]
    ordinary_latencies_s: list[float]
    multi_latency_s: float
    ordinary_completed_before_multi_count: int
    ordinary_completed_before_multi_all: bool


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _fail(message: str, *, summary: dict[str, Any] | None = None) -> int:
    print(f"FAIL: {message}", file=sys.stderr, flush=True)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
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


def _create_session() -> str:
    _status, body = _request_json(
        "POST",
        "/api/v1/create_session",
        payload={
            "tags": ["issue428"],
            "user_metadata": {"script": "scripts/tools/reproduce_issue_428.py"},
            "sdk_version": "reproduce_issue_428",
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


def _submit_sample(*, sampling_session_id: str, num_samples: int, prompt_tokens: list[int], max_tokens: int) -> tuple[str, float]:
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
                "temperature": 0.7,
                "top_p": 1.0,
            },
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {body!r}")
    return request_id, submitted_at


def _poll_future(request_id: str, *, submitted_at_s: float, out: dict[str, FutureResult], key: str) -> None:
    started = time.time()
    while True:
        code, body = _request_json(
            "POST",
            "/api/v1/retrieve_future",
            payload={"request_id": request_id},
            timeout_s=REQUEST_TIMEOUT_S,
            allowed_statuses={200, 408},
        )
        if code == 408:
            if time.time() - started > POLL_TIMEOUT_S:
                out[key] = FutureResult(
                    request_id=request_id,
                    status_code=408,
                    completed_at_s=time.time(),
                    latency_s=time.time() - submitted_at_s,
                    sequence_count=0,
                    first_sequence_len=0,
                    error=f"timeout after {POLL_TIMEOUT_S}s",
                )
                return
            time.sleep(POLL_SLEEP_S)
            continue

        error = body.get("error")
        sequences = body.get("sequences")
        sequence_count = len(sequences) if isinstance(sequences, list) else 0
        first_sequence_len = 0
        if isinstance(sequences, list) and sequences and isinstance(sequences[0], dict):
            toks = sequences[0].get("tokens")
            if isinstance(toks, list):
                first_sequence_len = len(toks)
        out[key] = FutureResult(
            request_id=request_id,
            status_code=code,
            completed_at_s=time.time(),
            latency_s=time.time() - submitted_at_s,
            sequence_count=sequence_count,
            first_sequence_len=first_sequence_len,
            error=str(error) if error else None,
        )
        return


def _run_trial(trial: int) -> TrialSummary:
    session_id = _create_session()
    sampling_session_id = _create_sampling_session(session_id)

    multi_request_id, multi_submitted_at = _submit_sample(
        sampling_session_id=sampling_session_id,
        num_samples=MULTI_NUM_SAMPLES,
        prompt_tokens=MULTI_PROMPT_TOKENS,
        max_tokens=MULTI_MAX_TOKENS,
    )
    time.sleep(SUBMIT_GAP_S)

    ordinary_submissions: list[tuple[str, float]] = []
    for _idx in range(ORDINARY_CONCURRENCY):
        ordinary_submissions.append(
            _submit_sample(
                sampling_session_id=sampling_session_id,
                num_samples=1,
                prompt_tokens=ORDINARY_PROMPT_TOKENS,
                max_tokens=ORDINARY_MAX_TOKENS,
            )
        )

    results: dict[str, FutureResult] = {}
    threads: list[threading.Thread] = []
    threads.append(
        threading.Thread(
            target=_poll_future,
            kwargs={
                "request_id": multi_request_id,
                "submitted_at_s": multi_submitted_at,
                "out": results,
                "key": "multi",
            },
            daemon=True,
        )
    )
    for idx, (ordinary_request_id, ordinary_submitted_at) in enumerate(ordinary_submissions):
        threads.append(
            threading.Thread(
                target=_poll_future,
                kwargs={
                    "request_id": ordinary_request_id,
                    "submitted_at_s": ordinary_submitted_at,
                    "out": results,
                    "key": f"ordinary_{idx}",
                },
                daemon=True,
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    multi = results["multi"]
    ordinary_results = [results[f"ordinary_{idx}"] for idx in range(len(ordinary_submissions))]

    if multi.status_code != 200 or multi.error is not None:
        raise RuntimeError(f"multi request failed: {asdict(multi)}")
    if multi.sequence_count != MULTI_NUM_SAMPLES:
        raise RuntimeError(f"multi request returned wrong sequence_count: {asdict(multi)}")

    for ordinary in ordinary_results:
        if ordinary.status_code != 200 or ordinary.error is not None:
            raise RuntimeError(f"ordinary request failed: {asdict(ordinary)}")
        if ordinary.sequence_count != 1:
            raise RuntimeError(f"ordinary request returned wrong sequence_count: {asdict(ordinary)}")

    ordinary_completed_before_multi_count = sum(
        1 for ordinary in ordinary_results if ordinary.completed_at_s < multi.completed_at_s
    )
    return TrialSummary(
        trial=trial,
        multi_request_id=multi_request_id,
        ordinary_request_ids=[request_id for request_id, _ts in ordinary_submissions],
        ordinary_latencies_s=[ordinary.latency_s for ordinary in ordinary_results],
        multi_latency_s=multi.latency_s,
        ordinary_completed_before_multi_count=ordinary_completed_before_multi_count,
        ordinary_completed_before_multi_all=(ordinary_completed_before_multi_count == len(ordinary_results)),
    )


def _maybe_warmup() -> None:
    if not RUN_WARMUP:
        return
    session_id = _create_session()
    sampling_session_id = _create_sampling_session(session_id)
    request_id, submitted_at = _submit_sample(
        sampling_session_id=sampling_session_id,
        num_samples=1,
        prompt_tokens=WARMUP_PROMPT_TOKENS,
        max_tokens=WARMUP_MAX_TOKENS,
    )
    out: dict[str, FutureResult] = {}
    _poll_future(request_id, submitted_at_s=submitted_at, out=out, key="warmup")
    warmup = out["warmup"]
    if warmup.status_code != 200 or warmup.error is not None:
        raise RuntimeError(f"warmup failed: {asdict(warmup)}")


def main() -> int:
    print(
        json.dumps(
            {
                "base_url": BASE_URL,
                "model": MODEL,
                "trials": TRIALS,
                "multi_num_samples": MULTI_NUM_SAMPLES,
                "multi_max_tokens": MULTI_MAX_TOKENS,
                "ordinary_concurrency": ORDINARY_CONCURRENCY,
                "ordinary_max_tokens": ORDINARY_MAX_TOKENS,
                "expect_ordinary_before_multi_min": EXPECT_ORDINARY_BEFORE_MULTI_MIN,
                "submit_gap_s": SUBMIT_GAP_S,
                "warmup": RUN_WARMUP,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        _maybe_warmup()
        trials: list[TrialSummary] = []
        for trial in range(1, TRIALS + 1):
            summary = _run_trial(trial)
            trials.append(summary)
            print(json.dumps(asdict(summary), indent=2, sort_keys=True), flush=True)

        trials_meeting_threshold = sum(
            1 for t in trials if t.ordinary_completed_before_multi_count >= EXPECT_ORDINARY_BEFORE_MULTI_MIN
        )
        final_summary = {
            "trials": [asdict(t) for t in trials],
            "trials_meeting_threshold": trials_meeting_threshold,
            "expect_ordinary_before_multi_min": EXPECT_ORDINARY_BEFORE_MULTI_MIN,
            "total_trials": len(trials),
        }
        if trials_meeting_threshold != len(trials):
            return _fail(
                "ordinary request concurrency did not meet completion-before-multi threshold in every trial",
                summary=final_summary,
            )

        print("PASS: ordinary request concurrency met completion-before-multi threshold in every trial", flush=True)
        print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
