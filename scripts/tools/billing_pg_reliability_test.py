#!/usr/bin/env python3
"""End-to-end billing (usage_events) reliability test against a running mint-server.

This script:
1) Creates a sampling session for a base model.
2) Runs N /asample requests (optionally concurrent) and waits for completion.
3) Queries /internal/usage_logs since the run start time.
4) Verifies that usage events were persisted and match expected counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "http://localhost:8000"


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _load_env() -> None:
    load_dotenv()
    repo_root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(repo_root_env):
        load_dotenv(repo_root_env, override=False)


def _iso(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"X-API-Key": api_key} if api_key else {}
    gateway_user_id = _coalesce(os.environ.get("MINT_BILLING_USER_ID"), os.environ.get("X_MINT_USER_ID"))
    gateway_apikey_id = _coalesce(os.environ.get("MINT_BILLING_APIKEY_ID"), os.environ.get("X_MINT_APIKEY_ID"))
    if gateway_user_id and gateway_apikey_id:
        headers["X-MinT-User-Id"] = gateway_user_id
        headers["X-MinT-Apikey-Id"] = gateway_apikey_id
        internal_token = _coalesce(os.environ.get("INTERNAL_API_TOKEN"), os.environ.get("MINT_INTERNAL_API_TOKEN"))
        if internal_token:
            headers["X-Internal-Token"] = internal_token
    return headers


def _request_headers(headers: dict[str, str]) -> dict[str, str]:
    out = dict(headers)
    if out.get("X-MinT-User-Id") and out.get("X-MinT-Apikey-Id"):
        out["X-MinT-Request-Id"] = uuid.uuid4().hex
    return out


def _get(base_url: str, path: str, headers: dict[str, str], *, timeout_s: float) -> dict[str, Any]:
    r = requests.get(f"{base_url}{path}", headers=_request_headers(headers), timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


def _post(base_url: str, path: str, headers: dict[str, str], payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = requests.post(f"{base_url}{path}", headers=_request_headers(headers), json=payload, timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    assert isinstance(out, dict)
    return out


def _create_sampling_session(
    *,
    base_url: str,
    headers: dict[str, str],
    base_model: str,
    timeout_s: float,
) -> tuple[str, str]:
    sess = _post(
        base_url,
        "/api/v1/create_session",
        headers,
        payload={"tags": ["billing_pg_reliability_test"], "user_metadata": {}, "sdk_version": "scripts/tools/billing_pg_reliability_test.py"},
        timeout_s=timeout_s,
    )
    session_id = str(sess.get("session_id") or "")
    if not session_id:
        raise RuntimeError(f"create_session missing session_id: {sess}")

    out = _post(
        base_url,
        "/api/v1/create_sampling_session",
        headers,
        payload={"session_id": session_id, "sampling_session_seq_id": 0, "base_model": base_model},
        timeout_s=max(timeout_s, 90.0),
    )
    sampling_session_id = str(out.get("sampling_session_id") or "")
    if not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {out}")
    return session_id, sampling_session_id


def _wait_future(
    *,
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    model_id: str,
    poll_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"retrieve_future timeout request_id={request_id}")
        r = requests.post(
            f"{base_url}/api/v1/retrieve_future",
            headers=headers,
            json={"request_id": request_id, "model_id": model_id},
            timeout=min(30.0, timeout_s),
        )
        if r.status_code == 408:
            time.sleep(poll_s)
            continue
        r.raise_for_status()
        out = r.json()
        assert isinstance(out, dict)
        if "error" in out:
            raise RuntimeError(f"future error request_id={request_id}: {out.get('error')}")
        return out


def _run_one(
    *,
    base_url: str,
    headers: dict[str, str],
    sampling_session_id: str,
    seq_id: int,
    prompt_tokens: list[int],
    max_tokens: int,
    poll_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.time()
    fut = _post(
        base_url,
        "/api/v1/asample",
        headers,
        payload={
            "sampling_session_id": sampling_session_id,
            "seq_id": int(seq_id),
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
            "sampling_params": {"max_tokens": int(max_tokens), "temperature": 0.7, "top_k": -1, "top_p": 1.0},
        },
        timeout_s=timeout_s,
    )
    request_id = str(fut.get("request_id") or "")
    if not request_id:
        raise RuntimeError(f"asample missing request_id: {fut}")

    out = _wait_future(
        base_url=base_url,
        headers=headers,
        request_id=request_id,
        model_id=sampling_session_id,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )

    sequences = out.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError(f"retrieve_future missing sequences: {out}")

    gen_tokens = 0
    for s in sequences:
        toks = s.get("tokens") if isinstance(s, dict) else None
        if not isinstance(toks, list):
            raise RuntimeError(f"sequence missing tokens: {s}")
        gen_tokens += len(toks)

    return {
        "ok": True,
        "request_id": request_id,
        "elapsed_s": time.time() - t0,
        "prefill_tokens": len(prompt_tokens),
        "generation_tokens": int(gen_tokens),
    }


def _fetch_usage_logs_since(
    *,
    base_url: str,
    headers: dict[str, str],
    since_iso: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = _get(
            base_url,
            f"/internal/usage_logs?since={since_iso}&limit=1000&offset={offset}",
            headers,
            timeout_s=timeout_s,
        )
        logs = page.get("logs")
        if not isinstance(logs, list):
            raise RuntimeError(f"usage_logs invalid response: {page}")
        out.extend([x for x in logs if isinstance(x, dict)])
        has_more = bool(page.get("has_more"))
        if not has_more:
            return out
        next_offset = page.get("next_offset")
        if not isinstance(next_offset, int):
            raise RuntimeError(f"usage_logs missing next_offset: {page}")
        offset = next_offset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--requests", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--prompt-len", type=int, default=16)
    p.add_argument("--prompt-token", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--poll-s", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--out-json", default=None, help="Write a JSON report to this path")
    args = p.parse_args()

    if args.requests < 1:
        raise SystemExit("--requests must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.prompt_len < 1:
        raise SystemExit("--prompt-len must be >= 1")

    _load_env()
    base_url = (
        _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("MINT_BASE_URL"), DEFAULT_BASE_URL)
        .rstrip("/")
    )
    api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("MINT_API_KEY"))
    headers = _headers(api_key)
    summary_account_id = _coalesce(
        os.environ.get("MINT_BILLING_ACCOUNT_ID"),
        os.environ.get("MINT_BILLING_USER_ID"),
        "admin",
    )

    pre_summary = _get(base_url, f"/internal/usage_summary/{summary_account_id}", headers, timeout_s=float(args.timeout_s))
    since_iso = _iso(dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1))

    session_id, sampling_session_id = _create_sampling_session(
        base_url=base_url,
        headers=headers,
        base_model=str(args.base_model),
        timeout_s=float(args.timeout_s),
    )

    prompt_tokens = [int(args.prompt_token)] * int(args.prompt_len)
    t0 = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _task(i: int) -> dict[str, Any]:
        return _run_one(
            base_url=base_url,
            headers=headers,
            sampling_session_id=sampling_session_id,
            seq_id=i,
            prompt_tokens=prompt_tokens,
            max_tokens=int(args.max_tokens),
            poll_s=float(args.poll_s),
            timeout_s=float(args.timeout_s),
        )

    with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
        futs = {ex.submit(_task, i): i for i in range(int(args.requests))}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                failures.append({"ok": False, "i": i, "error": f"{type(e).__name__}: {e}"})

    elapsed_s = time.time() - t0
    ok = len(failures) == 0

    post_summary = _get(base_url, f"/internal/usage_summary/{summary_account_id}", headers, timeout_s=float(args.timeout_s))
    logs = _fetch_usage_logs_since(base_url=base_url, headers=headers, since_iso=since_iso, timeout_s=float(args.timeout_s))

    expected_events = len(results) * 2
    observed_events = len(logs)

    expected_prefill = int(sum(int(r["prefill_tokens"]) for r in results))
    expected_gen = int(sum(int(r["generation_tokens"]) for r in results))
    expected_total = expected_prefill + expected_gen

    charge_item_totals: dict[str, int] = {}
    charge_item_counts: dict[str, int] = {}
    dimension_totals: dict[str, int] = {}
    dimension_counts: dict[str, int] = {}
    seen_source_indexes: set[int] = set()
    dup_source_indexes = 0
    for row in logs:
        charge_item = str(row.get("charge_item") or "")
        quantity = int(row.get("quantity") or 0)
        label = str(row.get("label") or "")
        source_index = int(row.get("source_index") or 0)
        if source_index in seen_source_indexes:
            dup_source_indexes += 1
        seen_source_indexes.add(source_index)
        charge_item_totals[charge_item] = charge_item_totals.get(charge_item, 0) + quantity
        charge_item_counts[charge_item] = charge_item_counts.get(charge_item, 0) + 1
        for part in label.split(","):
            if not part.startswith("dimension="):
                continue
            dimension = part.split("=", 1)[1]
            if not dimension:
                continue
            dimension_totals[dimension] = dimension_totals.get(dimension, 0) + quantity
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
            break

    def _summary_counts(summary: dict[str, Any]) -> dict[str, int]:
        oc = summary.get("charge_item_totals")
        if not isinstance(oc, dict):
            return {}
        out: dict[str, int] = {}
        for k, v in oc.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out

    pre_op = _summary_counts(pre_summary)
    post_op = _summary_counts(post_summary)
    delta_op = {k: post_op.get(k, 0) - pre_op.get(k, 0) for k in set(pre_op) | set(post_op)}

    per_req_lat = [float(r["elapsed_s"]) for r in results if r.get("ok")]
    report = {
        "ok": ok,
        "base_url": base_url,
        "base_model": str(args.base_model),
        "session_id": session_id,
        "sampling_session_id": sampling_session_id,
        "start_server_timestamp": _iso(start_server_ts),
        "since_iso": since_iso,
        "requests": int(args.requests),
        "concurrency": int(args.concurrency),
        "prompt_len": int(args.prompt_len),
        "max_tokens": int(args.max_tokens),
        "elapsed_s": float(elapsed_s),
        "latency_s": {
            "p50": float(statistics.median(per_req_lat)) if per_req_lat else None,
            "p95": float(statistics.quantiles(per_req_lat, n=20)[18]) if len(per_req_lat) >= 20 else None,
            "max": float(max(per_req_lat)) if per_req_lat else None,
        },
        "results_ok": len(results),
        "results_failed": len(failures),
        "failures": failures[:20],
        "usage_events": {
            "expected_events": int(expected_events),
            "observed_events": int(observed_events),
            "duplicate_source_indexes": int(dup_source_indexes),
            "observed_counts_by_charge_item": charge_item_counts,
            "observed_quantity_by_charge_item": charge_item_totals,
            "observed_counts_by_dimension": dimension_counts,
            "observed_quantity_by_dimension": dimension_totals,
            "expected_quantity": {"sampling": expected_total},
            "expected_quantity_by_dimension": {
                "prefill": expected_prefill,
                "sample": expected_gen,
            },
            "delta_summary_quantity_by_charge_item": delta_op,
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.out_json:
        out_path = str(args.out_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
