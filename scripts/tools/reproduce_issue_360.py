#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


BASE_URL = _env("MINT_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
API_KEY = _env("MINT_API_KEY", "dummy")
BASE_MODEL = _env("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")


@dataclass
class ProbeResult:
    path: str
    status_code: int | None
    latency_s: float
    error: str | None


@dataclass
class SampleRun:
    seq_id: int
    submit_s: float
    total_poll_s: float
    first_pending_s: float | None
    pending_count: int
    pending_statuses: list[str]
    final_keys: list[str]
    sequence_count: int
    stop_reasons: list[str]
    error: str | None


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


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    return statistics.quantiles(xs, n=100)[max(0, min(99, int(q * 100) - 1))]


def _request(method: str, path: str, *, payload: dict[str, Any] | None = None, timeout_s: float) -> requests.Response:
    return requests.request(
        method=method,
        url=f"{BASE_URL}{path}",
        headers=_headers(),
        json=payload,
        timeout=timeout_s,
    )


def _request_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_s: float,
    allowed_statuses: set[int] | None = None,
) -> tuple[int, dict[str, Any]]:
    resp = _request(method, path, payload=payload, timeout_s=timeout_s)
    if allowed_statuses is None:
        allowed_statuses = {200}
    if resp.status_code not in allowed_statuses:
        raise RuntimeError(f"{path} -> {resp.status_code}: {resp.text[:800]!r}")
    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"{path} returned non-JSON body: {type(exc).__name__}: {resp.text[:800]!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} returned non-dict JSON: {type(data)}")
    return resp.status_code, data


def _probe_loop(
    path: str,
    *,
    stop_event: threading.Event,
    interval_s: float,
    timeout_s: float,
    sink: list[ProbeResult],
) -> None:
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            resp = _request("GET", path, timeout_s=timeout_s)
            sink.append(
                ProbeResult(
                    path=path,
                    status_code=int(resp.status_code),
                    latency_s=time.perf_counter() - t0,
                    error=None,
                )
            )
        except Exception as exc:
            sink.append(
                ProbeResult(
                    path=path,
                    status_code=None,
                    latency_s=time.perf_counter() - t0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        stop_event.wait(interval_s)


def _start_probe_threads(
    *,
    stop_event: threading.Event,
    interval_s: float,
    timeout_s: float,
    sink: list[ProbeResult],
) -> list[threading.Thread]:
    threads = [
        threading.Thread(
            target=_probe_loop,
            kwargs={
                "path": "/api/v1/healthz",
                "stop_event": stop_event,
                "interval_s": interval_s,
                "timeout_s": timeout_s,
                "sink": sink,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_probe_loop,
            kwargs={
                "path": "/internal/actors",
                "stop_event": stop_event,
                "interval_s": interval_s,
                "timeout_s": timeout_s,
                "sink": sink,
            },
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    return threads


def _create_session(*, timeout_s: float) -> str:
    _status, body = _request_json(
        "POST",
        "/api/v1/create_session",
        payload={
            "tags": ["issue360-e2e"],
            "user_metadata": {},
            "sdk_version": "scripts/tools/reproduce_issue_360.py",
        },
        timeout_s=timeout_s,
    )
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {body!r}")
    return session_id


def _create_sampling_session(session_id: str, *, timeout_s: float) -> tuple[str, float]:
    t0 = time.perf_counter()
    _status, body = _request_json(
        "POST",
        "/api/v1/create_sampling_session",
        payload={
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": BASE_MODEL,
        },
        timeout_s=timeout_s,
    )
    sampling_session_id = body.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {body!r}")
    return sampling_session_id, time.perf_counter() - t0


def _advertised_model_names(payload: dict[str, Any]) -> list[str]:
    items = payload.get("supported_models")
    if not isinstance(items, list):
        raise RuntimeError(f"get_server_capabilities missing supported_models list: {payload!r}")
    names: list[str] = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
            continue
        if isinstance(item, dict) and isinstance(item.get("model_name"), str):
            names.append(item["model_name"])
            continue
        raise RuntimeError(f"unsupported supported_models entry: {item!r}")
    return names


def _validate_terminal_payload(body: dict[str, Any]) -> tuple[list[str], int, list[str]]:
    if "error" in body:
        raise RuntimeError(f"terminal payload carried error: {body.get('error')!r}")
    sequences = body.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise RuntimeError(f"terminal payload missing sequences: {body!r}")
    stop_reasons: list[str] = []
    for seq in sequences:
        if not isinstance(seq, dict):
            raise RuntimeError(f"terminal sequence is not dict: {seq!r}")
        tokens = seq.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError(f"terminal sequence missing tokens: {seq!r}")
        stop_reason = seq.get("stop_reason")
        if not isinstance(stop_reason, str) or not stop_reason:
            raise RuntimeError(f"terminal sequence missing stop_reason: {seq!r}")
        stop_reasons.append(stop_reason)
    return sorted(body.keys()), len(sequences), stop_reasons


def _run_sample_iteration(
    *,
    sampling_session_id: str,
    seq_id: int,
    poll_timeout_s: float,
    poll_sleep_s: float,
) -> SampleRun:
    submit_t0 = time.perf_counter()
    _status, future = _request_json(
        "POST",
        "/api/v1/asample",
        payload={
            "sampling_session_id": sampling_session_id,
            "seq_id": seq_id,
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": [1, 2, 3, 4], "type": "encoded_text"}]},
            "sampling_params": {
                "max_tokens": 8,
                "temperature": 0.7,
                "top_k": -1,
                "top_p": 1.0,
                "stop": None,
                "seed": None,
            },
        },
        timeout_s=60.0,
    )
    submit_s = time.perf_counter() - submit_t0
    request_id = future.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {future!r}")

    poll_started_at = time.perf_counter()
    first_pending_s: float | None = None
    pending_count = 0
    pending_statuses: list[str] = []
    while True:
        elapsed = time.perf_counter() - poll_started_at
        if elapsed > poll_timeout_s:
            raise TimeoutError(
                f"retrieve_future timed out after {elapsed:.2f}s request_id={request_id} seq_id={seq_id}"
            )
        status_code, body = _request_json(
            "POST",
            "/api/v1/retrieve_future",
            payload={"request_id": request_id},
            timeout_s=30.0,
            allowed_statuses={200, 408},
        )
        if status_code == 408:
            pending_count += 1
            status = body.get("status")
            if isinstance(status, str) and status not in pending_statuses:
                pending_statuses.append(status)
            if first_pending_s is None:
                first_pending_s = elapsed
            time.sleep(poll_sleep_s)
            continue
        final_keys, sequence_count, stop_reasons = _validate_terminal_payload(body)
        return SampleRun(
            seq_id=seq_id,
            submit_s=submit_s,
            total_poll_s=time.perf_counter() - poll_started_at,
            first_pending_s=first_pending_s,
            pending_count=pending_count,
            pending_statuses=pending_statuses,
            final_keys=final_keys,
            sequence_count=sequence_count,
            stop_reasons=stop_reasons,
            error=None,
        )


def _summarize_probes(rows: list[ProbeResult]) -> dict[str, Any]:
    by_path: dict[str, dict[str, Any]] = {}
    for path in sorted({row.path for row in rows}):
        items = [row for row in rows if row.path == path]
        ok_rows = [row for row in items if row.error is None and row.status_code is not None]
        latencies = [row.latency_s for row in ok_rows]
        by_path[path] = {
            "n_total": len(items),
            "n_ok": len(ok_rows),
            "n_error": len(items) - len(ok_rows),
            "status_codes": sorted({row.status_code for row in ok_rows}),
            "latency_p50_s": statistics.median(latencies) if latencies else None,
            "latency_p95_s": _quantile(latencies, 0.95),
            "sample_errors": [row.error for row in items if row.error][:5],
        }
    return by_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--poll-timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-sleep-s", type=float, default=0.2)
    parser.add_argument("--probe-interval-s", type=float, default=0.1)
    parser.add_argument("--probe-timeout-s", type=float, default=5.0)
    parser.add_argument("--require-pending", action="store_true")
    parser.add_argument("--healthz-p95-budget-s", type=float, default=0.5)
    parser.add_argument("--actors-p95-budget-s", type=float, default=1.0)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> int:
    global BASE_URL, API_KEY, BASE_MODEL

    args = _parse_args()
    BASE_URL = str(args.base_url).rstrip("/")
    API_KEY = str(args.api_key)
    BASE_MODEL = str(args.base_model)

    try:
        health_status, health = _request_json(
            "GET",
            "/api/v1/healthz",
            timeout_s=10.0,
            allowed_statuses={200, 503},
        )
        if health_status == 200 and health.get("status") != "ready":
            return _fail(f"healthz unexpected payload: {health!r}")

        _status, caps = _request_json("GET", "/api/v1/get_server_capabilities", timeout_s=30.0)
        supported_model_names = _advertised_model_names(caps)
        if BASE_MODEL not in supported_model_names:
            return _fail(f"base_model not advertised by server: base_model={BASE_MODEL!r} caps={caps!r}")

        session_id = _create_session(timeout_s=30.0)
        sample_runs: list[SampleRun] = []
        probe_rows: list[ProbeResult] = []
        probe_stop_event = threading.Event()
        probe_threads = _start_probe_threads(
            stop_event=probe_stop_event,
            interval_s=float(args.probe_interval_s),
            timeout_s=float(args.probe_timeout_s),
            sink=probe_rows,
        )
        try:
            sampling_session_id, create_sampling_session_s = _create_sampling_session(session_id, timeout_s=180.0)
            for seq_id in range(int(args.iterations)):
                sample_run = _run_sample_iteration(
                    sampling_session_id=sampling_session_id,
                    seq_id=seq_id,
                    poll_timeout_s=float(args.poll_timeout_s),
                    poll_sleep_s=float(args.poll_sleep_s),
                )
                sample_runs.append(sample_run)
        finally:
            probe_stop_event.set()
            for thread in probe_threads:
                thread.join(timeout=float(args.probe_timeout_s))

        summary = {
            "base_url": BASE_URL,
            "base_model": BASE_MODEL,
            "preflight_healthz": {
                "status_code": health_status,
                "payload": health,
            },
            "session_id": session_id,
            "sampling_session_id": sampling_session_id,
            "iterations": int(args.iterations),
            "create_sampling_session_s": round(create_sampling_session_s, 3),
            "samples": {
                "runs": [asdict(run) for run in sample_runs],
                "pending_total": sum(run.pending_count for run in sample_runs),
                "submit_p50_s": statistics.median([run.submit_s for run in sample_runs]) if sample_runs else None,
                "submit_p95_s": _quantile([run.submit_s for run in sample_runs], 0.95),
                "terminal_p50_s": statistics.median([run.total_poll_s for run in sample_runs]) if sample_runs else None,
                "terminal_p95_s": _quantile([run.total_poll_s for run in sample_runs], 0.95),
                "first_pending_p50_s": statistics.median(
                    [run.first_pending_s for run in sample_runs if run.first_pending_s is not None]
                )
                if any(run.first_pending_s is not None for run in sample_runs)
                else None,
            },
            "probes": _summarize_probes(probe_rows),
        }

        if args.require_pending and summary["samples"]["pending_total"] <= 0:
            return _fail("no 408 pending state observed during retrieve_future polling", summary=summary)

        for run in sample_runs:
            if run.error:
                return _fail(f"sample seq_id={run.seq_id} failed: {run.error}", summary=summary)

        for path, budget_s, allowed_statuses in [
            ("/api/v1/healthz", float(args.healthz_p95_budget_s), {200, 503}),
            ("/internal/actors", float(args.actors_p95_budget_s), {200}),
        ]:
            stats = summary["probes"].get(path)
            if not isinstance(stats, dict):
                return _fail(f"missing probe stats for {path}", summary=summary)
            if stats["n_total"] <= 0:
                return _fail(f"no probe samples collected for {path}", summary=summary)
            if stats["n_error"] > 0:
                return _fail(f"probe errors observed for {path}", summary=summary)
            statuses = set(stats["status_codes"])
            if not statuses.issubset(allowed_statuses):
                return _fail(f"unexpected status codes for {path}: {statuses}", summary=summary)
            p95 = stats["latency_p95_s"]
            if p95 is not None and p95 > budget_s:
                return _fail(f"{path} latency_p95_s={p95:.3f} exceeded budget {budget_s:.3f}", summary=summary)

        rendered = json.dumps(summary, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.json_output:
            with open(args.json_output, "w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.write("\n")
        return 0
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
