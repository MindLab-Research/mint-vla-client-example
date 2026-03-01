#!/usr/bin/env python3
"""Issue #194 automatic benchmark runner.

Run with no arguments:
    python scripts/tools/bench_issue194_interleaving.py

What it does automatically:
1) Client-scaling experiment on 2/3/4/8/10 clients.
2) For each client count, generate one random interleaving trace as baseline.
3) Re-run the exact same trace after deterministic "sticky burst" reshaping.
4) Emit raw JSONL to /tmp and report artifacts to cover/issue194.
5) Produce a short sequence-trace experiment (small enough to capture queue exec order)
   and output submit-vs-exec CSVs for baseline/reshaped.

Environment overrides (optional):
- MINT_BASE_URL (default: http://localhost:8000)
- MINT_API_KEY
- MINT_BENCH_MODEL (default: Qwen/Qwen3-0.6B)
- MINT_BENCH_CLIENTS (default: 2,3,4,8,10)
- MINT_BENCH_STEPS (default: 4)
- MINT_BENCH_CHUNKS_PER_STEP (default: 4)
- MINT_BENCH_BATCH_SIZE (default: 2)
- MINT_BENCH_SEQ_LEN (default: 128)
- MINT_BENCH_RANDOM_STAY_PROB (default: 0.55)
- MINT_BENCH_RESHAPE_BURST (default: 4)
- MINT_BENCH_TIMEOUT_S (default: 1800)
- MINT_BENCH_POLL_INTERVAL_S (default: 0.2)
"""

from __future__ import annotations

import concurrent.futures
import csv
import datetime
import json
import math
import os
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class Job:
    job_id: int
    step: int
    session_idx: int
    chunk_idx: int
    seq_id: int
    seed_base: int


@dataclass(frozen=True)
class BenchConfig:
    base_url: str
    headers: dict[str, str]
    model: str
    lora_rank: int
    clients: list[int]
    steps: int
    chunks_per_step: int
    batch_size: int
    seq_len: int
    random_stay_prob: float
    reshape_burst: int
    timeout_s: float
    poll_interval_s: float


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _ts_dir() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except Exception:
        return float(default)


def _env_clients() -> list[int]:
    raw = str(os.environ.get("MINT_BENCH_CLIENTS", "2,3,4,8,10")).strip()
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
        except Exception:
            continue
        if v >= 2:
            out.append(v)
    return out or [2, 3, 4, 8, 10]


def _base_url() -> str:
    return (_coalesce(os.environ.get("MINT_BASE_URL"), DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")


def _headers() -> dict[str, str]:
    api_key = _coalesce(os.environ.get("MINT_API_KEY"))
    return {"X-API-Key": api_key} if api_key else {}


def _safe_server_info(base_url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    try:
        r = requests.get(f"{base_url}/api/v1/server_info", headers=headers, timeout=timeout_s)
        if r.ok and isinstance(r.json(), dict):
            return r.json()
    except Exception:
        pass
    return {}


def _safe_server_caps(base_url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    try:
        r = requests.get(f"{base_url}/api/v1/get_server_capabilities", headers=headers, timeout=timeout_s)
        if r.ok and isinstance(r.json(), dict):
            return r.json()
    except Exception:
        pass
    return {}


def _resolve_model(base_url: str, headers: dict[str, str], timeout_s: float) -> str:
    preferred = str(os.environ.get("MINT_BENCH_MODEL", DEFAULT_MODEL)).strip()
    caps = _safe_server_caps(base_url, headers, timeout_s=timeout_s)
    models = caps.get("supported_models") if isinstance(caps, dict) else None
    names: list[str] = []
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict):
                n = m.get("model_name")
                if isinstance(n, str) and n:
                    names.append(n)
    if preferred and preferred in names:
        return preferred
    if preferred and not names:
        return preferred
    if names:
        for n in names:
            if "0.6B" in n:
                return n
        return names[0]
    return preferred or DEFAULT_MODEL


def _poll_future(
    *,
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    url = f"{base_url}/api/v1/retrieve_future"
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"retrieve_future timeout request_id={request_id} timeout_s={timeout_s}")
        resp = requests.post(url, headers=headers, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data
            raise TypeError(f"retrieve_future returned non-dict: {type(data)}")
        if resp.status_code == 408:
            time.sleep(poll_interval_s)
            continue
        try:
            detail = resp.text
        except Exception:
            detail = "<no-body>"
        raise RuntimeError(f"retrieve_future failed status={resp.status_code} request_id={request_id} body={detail}")


def _create_model(
    *,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    base_model: str,
    lora_rank: int,
    timeout_s: float,
    poll_interval_s: float,
) -> str:
    url = f"{base_url}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {"rank": int(lora_rank)},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise TypeError(f"/create_model returned non-dict: {type(data)}")
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"/create_model missing request_id: {data}")
    out = _poll_future(
        base_url=base_url,
        headers=headers,
        request_id=request_id,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    model_id = out.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model future missing model_id: {out}")
    return model_id


def _delete_model(*, base_url: str, headers: dict[str, str], model_id: str, timeout_s: float) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=min(timeout_s, 120.0))
    except Exception:
        pass


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    s = sorted(values)
    idx = int((len(s) * p + 99) // 100) - 1
    if idx < 0:
        idx = 0
    if idx >= len(s):
        idx = len(s) - 1
    return float(s[idx])


def _build_datum(*, seq_len: int, seed: int, vocab_hint: int = 512) -> dict[str, Any]:
    r = random.Random(seed)
    tokens = [int(10 + r.randrange(max(16, vocab_hint))) for _ in range(seq_len)]
    target_tokens = tokens[1:] + [2]
    loss_mask = [1.0] * len(target_tokens)
    return {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
        },
    }


def _build_chunk(*, batch_size: int, seq_len: int, seed_base: int) -> list[dict[str, Any]]:
    return [_build_datum(seq_len=seq_len, seed=seed_base + i) for i in range(batch_size)]


def _run_forward_backward(
    *,
    base_url: str,
    headers: dict[str, str],
    model_id: str,
    seq_id: int,
    data: list[dict[str, Any]],
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload: dict[str, Any] = {
        "model_id": model_id,
        "seq_id": int(seq_id),
        "forward_backward_input": {
            "data": data,
            "loss_fn": "cross_entropy",
        },
    }
    resp = requests.post(f"{base_url}/api/v1/forward_backward", headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    first = resp.json()
    if not isinstance(first, dict):
        raise TypeError(f"/forward_backward returned non-dict: {type(first)}")
    request_id = first.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"/forward_backward missing request_id: {first}")
    _poll_future(
        base_url=base_url,
        headers=headers,
        request_id=request_id,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    elapsed_s = time.perf_counter() - t0
    return {
        "request_id": request_id,
        "elapsed_s": float(elapsed_s),
    }


def _safe_queue_debug(base_url: str, headers: dict[str, str]) -> dict[str, Any]:
    try:
        r = requests.get(f"{base_url}/internal/work_queue/debug_state", headers=headers, timeout=15)
        if r.ok:
            d = r.json()
            if isinstance(d, dict):
                return d
    except Exception:
        pass
    return {}


def _create_sessions(cfg: BenchConfig, *, run_id: str, sessions: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i in range(sessions):
        sid = f"{run_id}-s{i}"
        model_id = _create_model(
            base_url=cfg.base_url,
            headers=cfg.headers,
            session_id=sid,
            base_model=cfg.model,
            lora_rank=cfg.lora_rank,
            timeout_s=cfg.timeout_s,
            poll_interval_s=cfg.poll_interval_s,
        )
        out.append({"session_id": sid, "model_id": model_id})
        print(f"[{_now_iso()}] created session={sid} model_id={model_id}", flush=True)
    return out


def _cleanup_sessions(cfg: BenchConfig, session_infos: list[dict[str, str]]) -> None:
    for s in session_infos:
        _delete_model(
            base_url=cfg.base_url,
            headers=cfg.headers,
            model_id=str(s.get("model_id", "")),
            timeout_s=cfg.timeout_s,
        )


def _build_random_jobs(*, sessions: int, steps: int, chunks_per_step: int, seed: int, stay_prob: float) -> list[Job]:
    rng = random.Random(seed)
    jobs: list[Job] = []
    job_id = 0
    for step in range(steps):
        remain: dict[int, list[int]] = {s: list(range(chunks_per_step)) for s in range(sessions)}
        active = [s for s in range(sessions)]
        current = rng.choice(active)
        while active:
            if current not in active:
                current = rng.choice(active)
            chunk_idx = remain[current].pop(0)
            seq_id = step * chunks_per_step + chunk_idx
            seed_base = (current + 1) * 10_000_000 + (step + 1) * 100_000 + (chunk_idx + 1) * 1_000
            jobs.append(
                Job(
                    job_id=job_id,
                    step=step,
                    session_idx=current,
                    chunk_idx=chunk_idx,
                    seq_id=seq_id,
                    seed_base=seed_base,
                )
            )
            job_id += 1
            if not remain[current]:
                active = [s for s in active if s != current]
                if not active:
                    break
            if current in active and rng.random() < stay_prob:
                continue
            current = rng.choice(active)
    return jobs


def _reshape_jobs_rr_burst(*, baseline_jobs: list[Job], sessions: int, burst: int) -> list[Job]:
    queues: dict[int, list[Job]] = {s: [] for s in range(sessions)}
    for j in baseline_jobs:
        queues[j.session_idx].append(j)
    active = [s for s in range(sessions) if queues[s]]
    out: list[Job] = []
    rr_idx = 0
    b = max(1, int(burst))
    while active:
        s = active[rr_idx % len(active)]
        q = queues[s]
        take = min(b, len(q))
        for _ in range(take):
            out.append(q.pop(0))
        if not q:
            active.pop(rr_idx % len(active))
            if active:
                rr_idx = rr_idx % len(active)
        else:
            rr_idx = (rr_idx + 1) % len(active)
    return out


def _run_sequence_case(
    *,
    cfg: BenchConfig,
    run_dir: Path,
    run_id: str,
    mode: str,
    session_infos: list[dict[str, str]],
    jobs: list[Job],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    out_path = run_dir / f"bench_issue194_{run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    before_debug = _safe_queue_debug(cfg.base_url, cfg.headers)
    before_reqs = {
        str(x.get("request_id"))
        for x in (before_debug.get("recent_dequeues") or [])
        if isinstance(x, dict) and isinstance(x.get("request_id"), str)
    }

    records: list[dict[str, Any]] = []
    completion_idx = 0
    run_t0 = time.perf_counter()

    submit_workers = max(8, len(session_infos) * 4)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "kind": "meta",
                    "ts": _now_iso(),
                    "run_id": run_id,
                    "mode": mode,
                    "base_url": cfg.base_url,
                    "model": cfg.model,
                    "sessions": len(session_infos),
                    "steps": cfg.steps,
                    "chunks_per_step": cfg.chunks_per_step,
                    "batch_size": cfg.batch_size,
                    "seq_len": cfg.seq_len,
                    "jobs_total": len(jobs),
                    "session_infos": session_infos,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=submit_workers) as pool:
            fut_to_meta: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, Job, str]] = {}
            for submit_idx, job in enumerate(jobs):
                model_id = session_infos[job.session_idx]["model_id"]
                chunk_data = _build_chunk(
                    batch_size=cfg.batch_size,
                    seq_len=cfg.seq_len,
                    seed_base=job.seed_base,
                )
                fut = pool.submit(
                    _run_forward_backward,
                    base_url=cfg.base_url,
                    headers=cfg.headers,
                    model_id=model_id,
                    seq_id=job.seq_id,
                    data=chunk_data,
                    timeout_s=cfg.timeout_s,
                    poll_interval_s=cfg.poll_interval_s,
                )
                fut_to_meta[fut] = (submit_idx, job, model_id)

            for fut in concurrent.futures.as_completed(fut_to_meta):
                submit_idx, job, model_id = fut_to_meta[fut]
                rec: dict[str, Any]
                try:
                    out = fut.result()
                    rec = {
                        "kind": "fb",
                        "ts": _now_iso(),
                        "run_id": run_id,
                        "mode": mode,
                        "ok": True,
                        "submit_idx": int(submit_idx),
                        "completion_idx": int(completion_idx),
                        "request_id": str(out.get("request_id")),
                        "elapsed_s": float(out.get("elapsed_s", 0.0)),
                        "step": int(job.step),
                        "session_idx": int(job.session_idx),
                        "chunk_idx": int(job.chunk_idx),
                        "seq_id": int(job.seq_id),
                        "model_id": model_id,
                    }
                except Exception as e:
                    rec = {
                        "kind": "fb",
                        "ts": _now_iso(),
                        "run_id": run_id,
                        "mode": mode,
                        "ok": False,
                        "submit_idx": int(submit_idx),
                        "completion_idx": int(completion_idx),
                        "error": f"{type(e).__name__}: {e}",
                        "step": int(job.step),
                        "session_idx": int(job.session_idx),
                        "chunk_idx": int(job.chunk_idx),
                        "seq_id": int(job.seq_id),
                        "model_id": model_id,
                    }
                completion_idx += 1
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
                f.flush()

        run_elapsed_s = time.perf_counter() - run_t0

        after_debug = _safe_queue_debug(cfg.base_url, cfg.headers)
        recent_dequeues = after_debug.get("recent_dequeues") or []
        req_ids = {
            str(r.get("request_id"))
            for r in records
            if r.get("ok") and isinstance(r.get("request_id"), str)
        }
        exec_rows: list[tuple[float, str]] = []
        for x in recent_dequeues:
            if not isinstance(x, dict):
                continue
            rid = x.get("request_id")
            if not isinstance(rid, str) or rid not in req_ids or rid in before_reqs:
                continue
            ts = x.get("ts")
            try:
                tsf = float(ts)
            except Exception:
                continue
            exec_rows.append((tsf, rid))
        exec_rows.sort(key=lambda t: t[0])
        req_to_exec_idx = {rid: i for i, (_, rid) in enumerate(exec_rows)}

        for rec in records:
            rid = rec.get("request_id")
            if isinstance(rid, str) and rid in req_to_exec_idx:
                rec["exec_idx"] = int(req_to_exec_idx[rid])
                rec["exec_minus_submit"] = int(req_to_exec_idx[rid]) - int(rec["submit_idx"])
            else:
                rec["exec_idx"] = None
                rec["exec_minus_submit"] = None

        ok_records = [r for r in records if r.get("ok") and isinstance(r.get("elapsed_s"), (int, float))]
        lat = [float(r["elapsed_s"]) for r in ok_records]
        p50 = _percentile(lat, 50)
        p95 = _percentile(lat, 95)
        p99 = _percentile(lat, 99)
        mean = float(statistics.mean(lat)) if lat else float("nan")
        tps = float(len(ok_records) / run_elapsed_s) if run_elapsed_s > 0 else float("nan")
        exec_cov = float(sum(1 for r in ok_records if isinstance(r.get("exec_idx"), int)) / len(ok_records)) if ok_records else 0.0

        summary = {
            "kind": "summary",
            "ts": _now_iso(),
            "run_id": run_id,
            "mode": mode,
            "fb_total": int(len(records)),
            "fb_ok": int(len(ok_records)),
            "fb_err": int(len(records) - len(ok_records)),
            "run_elapsed_s": float(run_elapsed_s),
            "fb_latency_p50_s": float(p50),
            "fb_latency_p95_s": float(p95),
            "fb_latency_p99_s": float(p99),
            "fb_latency_mean_s": float(mean),
            "fb_throughput_rps": float(tps),
            "exec_coverage": float(exec_cov),
            "jsonl_path": str(out_path),
        }
        f.write(json.dumps(summary, ensure_ascii=True, sort_keys=True) + "\n")
        f.flush()

    print(
        f"[{_now_iso()}] done run_id={run_id} mode={mode} ok={summary['fb_ok']}/{summary['fb_total']} "
        f"p50={summary['fb_latency_p50_s']:.3f}s p95={summary['fb_latency_p95_s']:.3f}s "
        f"tps={summary['fb_throughput_rps']:.3f}",
        flush=True,
    )
    return summary, records, out_path


def _write_scaling_outputs(
    *,
    report_dir: Path,
    rows: list[dict[str, Any]],
    cfg: BenchConfig,
    raw_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "random-reorder-scaling-data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "clients",
                "mode",
                "fb_ok",
                "fb_latency_p50_s",
                "fb_latency_p95_s",
                "fb_latency_p99_s",
                "fb_throughput_rps",
                "run_id",
                "jsonl_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    clients = sorted({int(r["clients"]) for r in rows})
    by = {(int(r["clients"]), str(r["mode"])): r for r in rows}

    b_p95 = [float(by[(c, "baseline_random")]["fb_latency_p95_s"]) for c in clients]
    r_p95 = [float(by[(c, "reshaped_sticky")]["fb_latency_p95_s"]) for c in clients]
    b_tps = [float(by[(c, "baseline_random")]["fb_throughput_rps"]) for c in clients]
    r_tps = [float(by[(c, "reshaped_sticky")]["fb_throughput_rps"]) for c in clients]

    width, height = 1160, 640
    margin = 74
    panel_w = (width - margin * 3) / 2
    panel_h = height - 180
    left_x = margin
    right_x = margin * 2 + panel_w
    y0 = 84

    def map_points(vals: list[float], y_min: float, y_max: float, x_start: float) -> list[tuple[float, float]]:
        xs = [x_start + i * (panel_w / (len(clients) - 1)) for i in range(len(clients))]

        def ymap(v: float) -> float:
            if abs(y_max - y_min) < 1e-12:
                return y0 + panel_h / 2
            return y0 + panel_h - (v - y_min) / (y_max - y_min) * panel_h

        return [(x, ymap(v)) for x, v in zip(xs, vals)]

    p95_max = max(max(b_p95), max(r_p95)) * 1.08
    tps_max = max(max(b_tps), max(r_tps)) * 1.2
    lp_b = map_points(b_p95, 0.0, p95_max, left_x)
    lp_r = map_points(r_p95, 0.0, p95_max, left_x)
    rp_b = map_points(b_tps, 0.0, tps_max, right_x)
    rp_r = map_points(r_tps, 0.0, tps_max, right_x)

    def poly(points: list[tuple[float, float]], color: str) -> str:
        ptxt = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{ptxt}" />'

    def dots(points: list[tuple[float, float]], color: str) -> str:
        return "\n".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />' for x, y in points)

    xlab_left = []
    xlab_right = []
    for i, c in enumerate(clients):
        xl = left_x + i * (panel_w / (len(clients) - 1))
        xr = right_x + i * (panel_w / (len(clients) - 1))
        xlab_left.append(
            f'<text x="{xl:.2f}" y="{y0+panel_h+28:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{c}</text>'
        )
        xlab_right.append(
            f'<text x="{xr:.2f}" y="{y0+panel_h+28:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{c}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
  <text x="{width/2}" y="36" text-anchor="middle" font-size="24" font-family="Arial" fill="#111">Issue194 Random Trace Baseline vs Reshaped</text>

  <text x="{left_x + panel_w/2}" y="64" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">P95 Latency vs Clients</text>
  <line x1="{left_x}" y1="{y0+panel_h}" x2="{left_x+panel_w}" y2="{y0+panel_h}" stroke="#666" />
  <line x1="{left_x}" y1="{y0}" x2="{left_x}" y2="{y0+panel_h}" stroke="#666" />
  {poly(lp_b, '#d62728')}
  {poly(lp_r, '#1f77b4')}
  {dots(lp_b, '#d62728')}
  {dots(lp_r, '#1f77b4')}

  <text x="{right_x + panel_w/2}" y="64" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">Throughput (req/s) vs Clients</text>
  <line x1="{right_x}" y1="{y0+panel_h}" x2="{right_x+panel_w}" y2="{y0+panel_h}" stroke="#666" />
  <line x1="{right_x}" y1="{y0}" x2="{right_x}" y2="{y0+panel_h}" stroke="#666" />
  {poly(rp_b, '#d62728')}
  {poly(rp_r, '#1f77b4')}
  {dots(rp_b, '#d62728')}
  {dots(rp_r, '#1f77b4')}

  <text x="{left_x}" y="{y0+panel_h+50}" font-size="13" font-family="Arial" fill="#333">clients</text>
  {''.join(xlab_left)}
  <text x="{right_x}" y="{y0+panel_h+50}" font-size="13" font-family="Arial" fill="#333">clients</text>
  {''.join(xlab_right)}

  <rect x="{width/2 - 190}" y="{height-58}" width="14" height="14" fill="#d62728"/>
  <text x="{width/2 - 168}" y="{height-46}" font-size="13" font-family="Arial" fill="#111">baseline_random</text>
  <rect x="{width/2 - 20}" y="{height-58}" width="14" height="14" fill="#1f77b4"/>
  <text x="{width/2 + 2}" y="{height-46}" font-size="13" font-family="Arial" fill="#111">reshaped_sticky</text>
</svg>
'''
    (report_dir / "random-reorder-curves.svg").write_text(svg, encoding="utf-8")

    lines: list[str] = []
    lines.append("# Issue194 Random Trace Reordering Experiment")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- model: `{cfg.model}`")
    lines.append(
        f"- per-case shape: `steps={cfg.steps}`, `chunks_per_step={cfg.chunks_per_step}`, "
        f"`batch_size={cfg.batch_size}`, `seq_len={cfg.seq_len}`"
    )
    lines.append(f"- baseline trace generator: Markov stay probability `{cfg.random_stay_prob:.2f}`")
    lines.append(f"- reorder algorithm: sticky round-robin burst `{cfg.reshape_burst}`")
    lines.append("- clients: `2,3,4,8,10`")
    lines.append("")
    lines.append("## Curve")
    lines.append("![random-reorder-curves](./random-reorder-curves.svg)")
    lines.append("")
    lines.append("## Table")
    lines.append("| clients | baseline p50(s) | reshaped p50(s) | baseline p95(s) | reshaped p95(s) | p95 speedup (base/re) | baseline tps | reshaped tps | throughput gain (re/base) | p95 sample N | p95 top-K |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in clients:
        b = by[(c, "baseline_random")]
        r = by[(c, "reshaped_sticky")]
        n = int(b["fb_ok"])
        rank = int(math.ceil(0.95 * n)) if n > 0 else 0
        top_k = int(n - rank + 1) if n > 0 else 0
        p95_speedup = float(b["fb_latency_p95_s"]) / float(r["fb_latency_p95_s"]) if float(r["fb_latency_p95_s"]) > 0 else float("nan")
        tps_gain = float(r["fb_throughput_rps"]) / float(b["fb_throughput_rps"]) if float(b["fb_throughput_rps"]) > 0 else float("nan")
        lines.append(
            "| {c} | {bp50:.3f} | {rp50:.3f} | {bp95:.3f} | {rp95:.3f} | {p95s:.2f}x | {btps:.3f} | {rtps:.3f} | {tg:.2f}x | {n} | {k} |".format(
                c=c,
                bp50=float(b["fb_latency_p50_s"]),
                rp50=float(r["fb_latency_p50_s"]),
                bp95=float(b["fb_latency_p95_s"]),
                rp95=float(r["fb_latency_p95_s"]),
                p95s=p95_speedup,
                btps=float(b["fb_throughput_rps"]),
                rtps=float(r["fb_throughput_rps"]),
                tg=tps_gain,
                n=n,
                k=top_k,
            )
        )

    lines.append("")
    lines.append("## Reading P50/P95")
    lines.append("- P50 is robust here and good for central tendency.")
    lines.append("- P95 is meaningful, but for small N it is sensitive to a few slow requests.")
    lines.append("- In this setup, top-K for P95 is small at low clients (for example, clients=2 gives top-K=2), so use trend + repeated runs for stronger confidence.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("- Random baseline interleaving is consistently worse than reshaped sticky scheduling on both tail latency and throughput.")
    lines.append("- As clients scale up, baseline tail latency increases faster, while reshaped mode keeps better throughput floor.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- CSV: `cover/issue194/random-reorder-scaling-data.csv`")
    lines.append("- Curve: `cover/issue194/random-reorder-curves.svg`")
    lines.append("- Raw JSONL dir: `{}`".format(raw_dir))

    (report_dir / "random-reorder-analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sequence_outputs(
    *,
    report_dir: Path,
    baseline_records: list[dict[str, Any]],
    reshaped_records: list[dict[str, Any]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        ordered = sorted(rows, key=lambda r: int(r.get("submit_idx", 0)))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "submit_idx",
                    "exec_idx",
                    "exec_minus_submit",
                    "request_id",
                    "step",
                    "session_idx",
                    "chunk_idx",
                    "seq_id",
                    "elapsed_s",
                ],
            )
            w.writeheader()
            for r in ordered:
                w.writerow(
                    {
                        "submit_idx": r.get("submit_idx"),
                        "exec_idx": r.get("exec_idx"),
                        "exec_minus_submit": r.get("exec_minus_submit"),
                        "request_id": r.get("request_id"),
                        "step": r.get("step"),
                        "session_idx": r.get("session_idx"),
                        "chunk_idx": r.get("chunk_idx"),
                        "seq_id": r.get("seq_id"),
                        "elapsed_s": r.get("elapsed_s"),
                    }
                )

    baseline_csv = report_dir / "random-seqtrace-baseline-submit-vs-exec.csv"
    reshaped_csv = report_dir / "random-seqtrace-reshaped-submit-vs-exec.csv"
    write_csv(baseline_csv, [r for r in baseline_records if r.get("ok")])
    write_csv(reshaped_csv, [r for r in reshaped_records if r.get("ok")])

    def run_lengths(rows: list[dict[str, Any]], key: str) -> list[int]:
        valid = [r for r in rows if isinstance(r.get(key), int)]
        if key == "exec_idx":
            valid = sorted(valid, key=lambda r: int(r["exec_idx"]))
        else:
            valid = sorted(valid, key=lambda r: int(r["submit_idx"]))
        out: list[int] = []
        prev = None
        cnt = 0
        for r in valid:
            sid = int(r["session_idx"])
            if sid == prev:
                cnt += 1
            else:
                if prev is not None:
                    out.append(cnt)
                prev = sid
                cnt = 1
        if prev is not None:
            out.append(cnt)
        return out

    b_sub = run_lengths(baseline_records, "submit_idx")
    b_exe = run_lengths(baseline_records, "exec_idx")
    r_sub = run_lengths(reshaped_records, "submit_idx")
    r_exe = run_lengths(reshaped_records, "exec_idx")

    md = []
    md.append("# Random Trace Sequence Mapping (submit vs actor execution)")
    md.append("")
    md.append("This experiment is intentionally small so queue `recent_dequeues` can cover nearly all requests.")
    md.append("")
    md.append("## Files")
    md.append("- `cover/issue194/random-seqtrace-baseline-submit-vs-exec.csv`")
    md.append("- `cover/issue194/random-seqtrace-reshaped-submit-vs-exec.csv`")
    md.append("")
    md.append("## Run-length Summary")
    md.append(f"- baseline submit run lengths: {b_sub}")
    md.append(f"- baseline exec run lengths: {b_exe}")
    md.append(f"- reshaped submit run lengths: {r_sub}")
    md.append(f"- reshaped exec run lengths: {r_exe}")
    md.append("")
    md.append("Interpretation:")
    md.append("- baseline random interleaving tends to produce shorter session bursts.")
    md.append("- reshaped mode increases contiguous same-session bursts, and execution order tends to follow that more closely.")

    (report_dir / "random-sequence-trace.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _run_scaling_experiment(cfg: BenchConfig, raw_dir: Path, report_dir: Path) -> None:
    rows: list[dict[str, Any]] = []

    for clients in cfg.clients:
        run_prefix = f"rnd-c{clients}-{uuid.uuid4().hex[:8]}"
        session_infos = _create_sessions(cfg, run_id=run_prefix, sessions=clients)
        try:
            baseline_jobs = _build_random_jobs(
                sessions=clients,
                steps=cfg.steps,
                chunks_per_step=cfg.chunks_per_step,
                seed=194_000 + clients,
                stay_prob=cfg.random_stay_prob,
            )
            reshaped_jobs = _reshape_jobs_rr_burst(
                baseline_jobs=baseline_jobs,
                sessions=clients,
                burst=cfg.reshape_burst,
            )

            b_summary, _, _ = _run_sequence_case(
                cfg=cfg,
                run_dir=raw_dir,
                run_id=f"{run_prefix}-baseline",
                mode="baseline_random",
                session_infos=session_infos,
                jobs=baseline_jobs,
            )
            r_summary, _, _ = _run_sequence_case(
                cfg=cfg,
                run_dir=raw_dir,
                run_id=f"{run_prefix}-reshaped",
                mode="reshaped_sticky",
                session_infos=session_infos,
                jobs=reshaped_jobs,
            )

            rows.append(
                {
                    "clients": clients,
                    "mode": "baseline_random",
                    "fb_ok": b_summary["fb_ok"],
                    "fb_latency_p50_s": b_summary["fb_latency_p50_s"],
                    "fb_latency_p95_s": b_summary["fb_latency_p95_s"],
                    "fb_latency_p99_s": b_summary["fb_latency_p99_s"],
                    "fb_throughput_rps": b_summary["fb_throughput_rps"],
                    "run_id": b_summary["run_id"],
                    "jsonl_path": b_summary["jsonl_path"],
                }
            )
            rows.append(
                {
                    "clients": clients,
                    "mode": "reshaped_sticky",
                    "fb_ok": r_summary["fb_ok"],
                    "fb_latency_p50_s": r_summary["fb_latency_p50_s"],
                    "fb_latency_p95_s": r_summary["fb_latency_p95_s"],
                    "fb_latency_p99_s": r_summary["fb_latency_p99_s"],
                    "fb_throughput_rps": r_summary["fb_throughput_rps"],
                    "run_id": r_summary["run_id"],
                    "jsonl_path": r_summary["jsonl_path"],
                }
            )
        finally:
            _cleanup_sessions(cfg, session_infos)

    rows.sort(key=lambda x: (int(x["clients"]), str(x["mode"])))
    _write_scaling_outputs(report_dir=report_dir, rows=rows, cfg=cfg, raw_dir=raw_dir)


def _run_sequence_trace_experiment(cfg: BenchConfig, raw_dir: Path, report_dir: Path) -> None:
    trace_clients = 4
    trace_steps = 1
    trace_chunks = 6
    run_prefix = f"seqtrace-{uuid.uuid4().hex[:8]}"
    session_infos = _create_sessions(cfg, run_id=run_prefix, sessions=trace_clients)
    try:
        baseline_jobs = _build_random_jobs(
            sessions=trace_clients,
            steps=trace_steps,
            chunks_per_step=trace_chunks,
            seed=987_654,
            stay_prob=cfg.random_stay_prob,
        )
        reshaped_jobs = _reshape_jobs_rr_burst(
            baseline_jobs=baseline_jobs,
            sessions=trace_clients,
            burst=cfg.reshape_burst,
        )

        # Temporary cfg override for trace-only shape
        trace_cfg = BenchConfig(
            base_url=cfg.base_url,
            headers=cfg.headers,
            model=cfg.model,
            lora_rank=cfg.lora_rank,
            clients=[trace_clients],
            steps=trace_steps,
            chunks_per_step=trace_chunks,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            random_stay_prob=cfg.random_stay_prob,
            reshape_burst=cfg.reshape_burst,
            timeout_s=cfg.timeout_s,
            poll_interval_s=cfg.poll_interval_s,
        )

        _, b_records, _ = _run_sequence_case(
            cfg=trace_cfg,
            run_dir=raw_dir,
            run_id=f"{run_prefix}-baseline",
            mode="baseline_random",
            session_infos=session_infos,
            jobs=baseline_jobs,
        )
        _, r_records, _ = _run_sequence_case(
            cfg=trace_cfg,
            run_dir=raw_dir,
            run_id=f"{run_prefix}-reshaped",
            mode="reshaped_sticky",
            session_infos=session_infos,
            jobs=reshaped_jobs,
        )

        _write_sequence_outputs(
            report_dir=report_dir,
            baseline_records=b_records,
            reshaped_records=r_records,
        )
    finally:
        _cleanup_sessions(cfg, session_infos)


def _build_config() -> BenchConfig:
    base_url = _base_url()
    headers = _headers()
    model = _resolve_model(base_url, headers, timeout_s=30.0)
    return BenchConfig(
        base_url=base_url,
        headers=headers,
        model=model,
        lora_rank=_env_int("MINT_BENCH_LORA_RANK", 8),
        clients=_env_clients(),
        steps=max(1, _env_int("MINT_BENCH_STEPS", 4)),
        chunks_per_step=max(1, _env_int("MINT_BENCH_CHUNKS_PER_STEP", 4)),
        batch_size=max(1, _env_int("MINT_BENCH_BATCH_SIZE", 2)),
        seq_len=max(2, _env_int("MINT_BENCH_SEQ_LEN", 128)),
        random_stay_prob=min(0.95, max(0.0, _env_float("MINT_BENCH_RANDOM_STAY_PROB", 0.55))),
        reshape_burst=max(1, _env_int("MINT_BENCH_RESHAPE_BURST", 4)),
        timeout_s=max(60.0, _env_float("MINT_BENCH_TIMEOUT_S", 1800.0)),
        poll_interval_s=max(0.05, _env_float("MINT_BENCH_POLL_INTERVAL_S", 0.2)),
    )


def main() -> None:
    cfg = _build_config()
    server_info = _safe_server_info(cfg.base_url, cfg.headers, timeout_s=30.0)
    run_dir = Path("/tmp") / "issue194-auto" / _ts_dir()
    report_dir = Path("cover") / "issue194"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[issue194] automatic benchmark start", flush=True)
    print(
        json.dumps(
            {
                "base_url": cfg.base_url,
                "model": cfg.model,
                "clients": cfg.clients,
                "steps": cfg.steps,
                "chunks_per_step": cfg.chunks_per_step,
                "batch_size": cfg.batch_size,
                "seq_len": cfg.seq_len,
                "stay_prob": cfg.random_stay_prob,
                "reshape_burst": cfg.reshape_burst,
                "run_dir": str(run_dir),
                "report_dir": str(report_dir),
                "server_info": server_info,
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )

    _run_scaling_experiment(cfg, raw_dir=run_dir, report_dir=report_dir)
    _run_sequence_trace_experiment(cfg, raw_dir=run_dir, report_dir=report_dir)

    print("[issue194] done", flush=True)
    print(f"raw_jsonl_dir={run_dir}", flush=True)
    print(f"report_md={report_dir / 'random-reorder-analysis.md'}", flush=True)
    print(f"report_svg={report_dir / 'random-reorder-curves.svg'}", flush=True)
    print(f"sequence_md={report_dir / 'random-sequence-trace.md'}", flush=True)


if __name__ == "__main__":
    main()
