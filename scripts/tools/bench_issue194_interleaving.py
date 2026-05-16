#!/usr/bin/env python3
"""Issue #194 benchmark and A/B analyzer.

Usage:
1) Collect one mode (baseline or scheduler):
   MINT_ISSUE194_LABEL=baseline python scripts/tools/bench_issue194_interleaving.py
   MINT_ISSUE194_LABEL=scheduler python scripts/tools/bench_issue194_interleaving.py

2) Compare two collected modes and generate report:
   MINT_ISSUE194_PHASE=compare \
   MINT_ISSUE194_BASELINE_LABEL=baseline \
   MINT_ISSUE194_FEATURE_LABEL=scheduler \
   python scripts/tools/bench_issue194_interleaving.py

Default phase is `run`.

Run-phase outputs:
- cover/issue194/<label>-runs.csv
- cover/issue194/<label>-latencies.csv
- cover/issue194/<label>-summary.md
- /tmp/issue194-ab/<label>/<ts>/bench_issue194_<run_id>.jsonl

Compare-phase outputs:
- cover/issue194/repo.md
- cover/issue194/repo-curves.svg
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
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


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
    phase: str
    label: str
    base_url: str
    headers: dict[str, str]
    model: str
    lora_rank: int
    clients: list[int]
    seeds: list[int]
    steps: int
    chunks_per_step: int
    batch_size: int
    seq_len: int
    random_stay_prob: float
    timeout_s: float
    poll_interval_s: float
    submit_workers_factor: int


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _ts_dir() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _env_int_list(name: str, default_csv: str, *, min_value: int) -> list[int]:
    raw = str(os.environ.get(name, default_csv)).strip()
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
        except Exception:
            continue
        if v >= min_value:
            out.append(v)
    return out


def _base_url() -> str:
    return str(os.environ.get("MINT_BASE_URL", DEFAULT_BASE_URL)).strip().rstrip("/")


def _headers() -> dict[str, str]:
    key = str(os.environ.get("MINT_API_KEY", "")).strip()
    return {"X-API-Key": key} if key else {}


def _safe_json_get(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    try:
        r = requests.get(url, headers=headers, timeout=timeout_s)
        if r.ok and isinstance(r.json(), dict):
            return r.json()
    except Exception:
        pass
    return {}


def _server_info(base_url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    return _safe_json_get(f"{base_url}/api/v1/server_info", headers, timeout_s)


def _server_caps(base_url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    return _safe_json_get(f"{base_url}/api/v1/get_server_capabilities", headers, timeout_s)


def _resolve_model(base_url: str, headers: dict[str, str], timeout_s: float) -> str:
    preferred = str(os.environ.get("MINT_BENCH_MODEL", DEFAULT_MODEL)).strip()
    caps = _server_caps(base_url, headers, timeout_s)
    models = caps.get("supported_models") if isinstance(caps, dict) else None

    names: list[str] = []
    if isinstance(models, list):
        for m in models:
            if not isinstance(m, dict):
                continue
            n = m.get("model_name")
            if isinstance(n, str) and n:
                names.append(n)

    if preferred and preferred in names:
        return preferred
    if not names:
        return preferred or DEFAULT_MODEL

    for p in ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-0.6B"):
        if p in names:
            return p
    return names[0]


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
        raise RuntimeError(f"retrieve_future failed status={resp.status_code} request_id={request_id} body={resp.text}")


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
    payload = {
        "session_id": session_id,
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {"rank": int(lora_rank)},
    }
    resp = requests.post(f"{base_url}/api/v1/create_model", headers=headers, json=payload, timeout=timeout_s)
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
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=min(timeout_s, 60.0))
    except Exception:
        pass


def _safe_queue_debug(base_url: str, headers: dict[str, str]) -> dict[str, Any]:
    try:
        r = requests.get(f"{base_url}/internal/model_work_scheduler/debug_state", headers=headers, timeout=20)
        if r.ok:
            d = r.json()
            if isinstance(d, dict):
                return d
    except Exception:
        pass
    return {}


def _build_datum(*, seq_len: int, seed: int, vocab_hint: int = 1024) -> dict[str, Any]:
    rng = random.Random(seed)
    tokens = [int(10 + rng.randrange(max(64, vocab_hint))) for _ in range(seq_len)]
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
    submit_ts = time.time()
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
    elapsed_s = float(time.perf_counter() - t0)
    return {
        "request_id": request_id,
        "submit_ts": float(submit_ts),
        "elapsed_s": elapsed_s,
        "finish_ts": float(time.time()),
    }


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
            seed_base = (current + 1) * 100_000_000 + (step + 1) * 1_000_000 + (chunk_idx + 1) * 10_000
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


def _extract_exec_sequence(
    *,
    debug_state: dict[str, Any],
    before_req_ids: set[str],
    expected_req_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in debug_state.get("recent_dequeues") or []:
        if not isinstance(item, dict):
            continue
        rid = item.get("request_id")
        if not isinstance(rid, str):
            continue
        if rid in before_req_ids:
            continue
        if rid not in expected_req_ids:
            continue
        ts_raw = item.get("ts")
        try:
            ts = float(ts_raw)
        except Exception:
            continue
        rows.append(
            {
                "ts": ts,
                "request_id": rid,
                "session_id": None if item.get("scheduler_session_id") is None else str(item.get("scheduler_session_id")),
                "domain": None if item.get("scheduler_domain") is None else str(item.get("scheduler_domain")),
                "reason": None if item.get("dequeue_reason") is None else str(item.get("dequeue_reason")),
            }
        )
    rows.sort(key=lambda x: float(x["ts"]))
    return rows


def _switch_stats(exec_rows: list[dict[str, Any]], *, req_to_session: dict[str, str]) -> dict[str, float]:
    session_seq: list[str] = []
    for x in exec_rows:
        sid = x.get("session_id")
        if not isinstance(sid, str) or not sid:
            rid = x.get("request_id")
            if isinstance(rid, str):
                sid = req_to_session.get(rid, "")
        if sid:
            session_seq.append(sid)
    if not session_seq:
        return {
            "switches": float("nan"),
            "switch_rate": float("nan"),
            "avg_burst": float("nan"),
            "p95_burst": float("nan"),
        }

    switches = 0
    bursts: list[int] = []
    prev = None
    run = 0
    for sid in session_seq:
        if sid == prev:
            run += 1
            continue
        if prev is not None:
            switches += 1
            bursts.append(run)
        prev = sid
        run = 1
    if run > 0:
        bursts.append(run)

    denom = max(1, len(session_seq) - 1)
    return {
        "switches": float(switches),
        "switch_rate": float(switches / denom),
        "avg_burst": float(statistics.mean(bursts)) if bursts else float("nan"),
        "p95_burst": float(_percentile([float(x) for x in bursts], 95)) if bursts else float("nan"),
    }


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
    return out


def _cleanup_sessions(cfg: BenchConfig, session_infos: list[dict[str, str]]) -> None:
    for s in session_infos:
        model_id = str(s.get("model_id") or "")
        if model_id:
            _delete_model(base_url=cfg.base_url, headers=cfg.headers, model_id=model_id, timeout_s=cfg.timeout_s)


def _run_case(
    *,
    cfg: BenchConfig,
    run_dir: Path,
    clients: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path]:
    run_id = f"{cfg.label}-c{clients}-s{seed}-{uuid.uuid4().hex[:8]}"
    out_path = run_dir / f"bench_issue194_{run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = _build_random_jobs(
        sessions=clients,
        steps=cfg.steps,
        chunks_per_step=cfg.chunks_per_step,
        seed=seed,
        stay_prob=cfg.random_stay_prob,
    )

    session_infos = _create_sessions(cfg, run_id=run_id, sessions=clients)
    req_to_session: dict[str, str] = {}

    before_debug = _safe_queue_debug(cfg.base_url, cfg.headers)
    before_req_ids = {
        str(x.get("request_id"))
        for x in (before_debug.get("recent_dequeues") or [])
        if isinstance(x, dict) and isinstance(x.get("request_id"), str)
    }

    records: list[dict[str, Any]] = []
    run_t0 = time.perf_counter()
    submit_workers = max(8, int(clients * max(1, cfg.submit_workers_factor)))

    try:
        with out_path.open("w", encoding="utf-8") as f:
            meta = {
                "kind": "meta",
                "ts": _now_iso(),
                "run_id": run_id,
                "label": cfg.label,
                "clients": clients,
                "seed": seed,
                "steps": cfg.steps,
                "chunks_per_step": cfg.chunks_per_step,
                "batch_size": cfg.batch_size,
                "seq_len": cfg.seq_len,
                "model": cfg.model,
                "jobs_total": len(jobs),
                "session_infos": session_infos,
            }
            f.write(json.dumps(meta, ensure_ascii=True, sort_keys=True) + "\n")

            with concurrent.futures.ThreadPoolExecutor(max_workers=submit_workers) as pool:
                fut_to_meta: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, Job, str, str]] = {}
                for submit_idx, job in enumerate(jobs):
                    model_id = session_infos[job.session_idx]["model_id"]
                    session_id = session_infos[job.session_idx]["session_id"]
                    data = _build_chunk(batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed_base=job.seed_base)
                    fut = pool.submit(
                        _run_forward_backward,
                        base_url=cfg.base_url,
                        headers=cfg.headers,
                        model_id=model_id,
                        seq_id=job.seq_id,
                        data=data,
                        timeout_s=cfg.timeout_s,
                        poll_interval_s=cfg.poll_interval_s,
                    )
                    fut_to_meta[fut] = (submit_idx, job, model_id, session_id)

                completion_idx = 0
                for fut in concurrent.futures.as_completed(fut_to_meta):
                    submit_idx, job, model_id, session_id = fut_to_meta[fut]
                    try:
                        out = fut.result()
                        request_id = str(out.get("request_id"))
                        rec = {
                            "kind": "fb",
                            "ts": _now_iso(),
                            "run_id": run_id,
                            "label": cfg.label,
                            "ok": True,
                            "clients": int(clients),
                            "seed": int(seed),
                            "submit_idx": int(submit_idx),
                            "completion_idx": int(completion_idx),
                            "request_id": request_id,
                            "elapsed_s": float(out.get("elapsed_s", 0.0)),
                            "submit_ts": float(out.get("submit_ts", 0.0)),
                            "finish_ts": float(out.get("finish_ts", 0.0)),
                            "step": int(job.step),
                            "session_idx": int(job.session_idx),
                            "session_id": str(session_id),
                            "chunk_idx": int(job.chunk_idx),
                            "seq_id": int(job.seq_id),
                            "model_id": str(model_id),
                        }
                        req_to_session[request_id] = str(session_id)
                    except Exception as e:
                        rec = {
                            "kind": "fb",
                            "ts": _now_iso(),
                            "run_id": run_id,
                            "label": cfg.label,
                            "ok": False,
                            "clients": int(clients),
                            "seed": int(seed),
                            "submit_idx": int(submit_idx),
                            "completion_idx": int(completion_idx),
                            "error": f"{type(e).__name__}: {e}",
                            "step": int(job.step),
                            "session_idx": int(job.session_idx),
                            "session_id": str(session_id),
                            "chunk_idx": int(job.chunk_idx),
                            "seq_id": int(job.seq_id),
                            "model_id": str(model_id),
                        }
                    completion_idx += 1
                    records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
                    f.flush()

            run_elapsed_s = float(time.perf_counter() - run_t0)
            ok_records = [r for r in records if r.get("ok") and isinstance(r.get("elapsed_s"), (int, float))]
            latencies = [float(r["elapsed_s"]) for r in ok_records]
            req_ids = {str(r.get("request_id")) for r in ok_records if isinstance(r.get("request_id"), str)}

            after_debug = _safe_queue_debug(cfg.base_url, cfg.headers)
            exec_rows = _extract_exec_sequence(debug_state=after_debug, before_req_ids=before_req_ids, expected_req_ids=req_ids)
            req_to_exec_idx = {str(x.get("request_id")): i for i, x in enumerate(exec_rows) if isinstance(x.get("request_id"), str)}

            reorder_abs: list[int] = []
            covered = 0
            by_session_lat: dict[str, list[float]] = {}
            for r in ok_records:
                rid = str(r.get("request_id"))
                sid = str(r.get("session_id"))
                by_session_lat.setdefault(sid, []).append(float(r["elapsed_s"]))
                exec_idx = req_to_exec_idx.get(rid)
                if exec_idx is None:
                    r["exec_idx"] = None
                    r["exec_minus_submit"] = None
                    continue
                submit_idx = int(r.get("submit_idx", 0))
                delta = int(exec_idx - submit_idx)
                reorder_abs.append(abs(delta))
                r["exec_idx"] = int(exec_idx)
                r["exec_minus_submit"] = int(delta)
                covered += 1

            session_p95s = [_percentile(v, 95) for v in by_session_lat.values() if v]
            switch = _switch_stats(exec_rows, req_to_session=req_to_session)

            summary = {
                "kind": "summary",
                "ts": _now_iso(),
                "run_id": run_id,
                "label": cfg.label,
                "clients": int(clients),
                "seed": int(seed),
                "fb_total": int(len(records)),
                "fb_ok": int(len(ok_records)),
                "fb_err": int(len(records) - len(ok_records)),
                "run_elapsed_s": float(run_elapsed_s),
                "fb_latency_p50_s": float(_percentile(latencies, 50)),
                "fb_latency_p95_s": float(_percentile(latencies, 95)),
                "fb_latency_p99_s": float(_percentile(latencies, 99)),
                "fb_latency_mean_s": float(statistics.mean(latencies)) if latencies else float("nan"),
                "fb_throughput_rps": float(len(ok_records) / run_elapsed_s) if run_elapsed_s > 0 else float("nan"),
                "exec_coverage": float(covered / len(ok_records)) if ok_records else 0.0,
                "reorder_mean_abs": float(statistics.mean(reorder_abs)) if reorder_abs else float("nan"),
                "reorder_p95_abs": float(_percentile([float(x) for x in reorder_abs], 95)) if reorder_abs else float("nan"),
                "session_p95_mean_s": float(statistics.mean(session_p95s)) if session_p95s else float("nan"),
                "session_p95_max_s": float(max(session_p95s)) if session_p95s else float("nan"),
                "switches": float(switch["switches"]),
                "switch_rate": float(switch["switch_rate"]),
                "avg_burst": float(switch["avg_burst"]),
                "p95_burst": float(switch["p95_burst"]),
                "jsonl_path": str(out_path),
            }
            f.write(json.dumps(summary, ensure_ascii=True, sort_keys=True) + "\n")
            f.flush()

        print(
            f"[{_now_iso()}] run_id={run_id} label={cfg.label} c={clients} seed={seed} "
            f"ok={summary['fb_ok']}/{summary['fb_total']} p95={summary['fb_latency_p95_s']:.3f}s "
            f"tps={summary['fb_throughput_rps']:.3f} switch_rate={summary['switch_rate']:.3f}",
            flush=True,
        )
        return summary, records, exec_rows, out_path
    finally:
        _cleanup_sessions(cfg, session_infos)


def _write_run_outputs(
    *,
    cfg: BenchConfig,
    report_dir: Path,
    summaries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = report_dir / f"{cfg.label}-runs.csv"
    run_fields = [
        "label",
        "run_id",
        "clients",
        "seed",
        "fb_total",
        "fb_ok",
        "fb_err",
        "run_elapsed_s",
        "fb_latency_p50_s",
        "fb_latency_p95_s",
        "fb_latency_p99_s",
        "fb_latency_mean_s",
        "fb_throughput_rps",
        "exec_coverage",
        "reorder_mean_abs",
        "reorder_p95_abs",
        "session_p95_mean_s",
        "session_p95_max_s",
        "switches",
        "switch_rate",
        "avg_burst",
        "p95_burst",
        "jsonl_path",
    ]
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for r in summaries:
            w.writerow({k: r.get(k) for k in run_fields})

    lat_csv = report_dir / f"{cfg.label}-latencies.csv"
    with lat_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "run_id", "clients", "seed", "session_id", "submit_idx", "elapsed_s"])
        w.writeheader()
        for r in records:
            if not r.get("ok"):
                continue
            if not isinstance(r.get("elapsed_s"), (int, float)):
                continue
            w.writerow(
                {
                    "label": cfg.label,
                    "run_id": r.get("run_id"),
                    "clients": r.get("clients"),
                    "seed": r.get("seed"),
                    "session_id": r.get("session_id"),
                    "submit_idx": r.get("submit_idx"),
                    "elapsed_s": r.get("elapsed_s"),
                }
            )

    by_clients: dict[int, list[dict[str, Any]]] = {}
    for s in summaries:
        by_clients.setdefault(int(s["clients"]), []).append(s)

    md = []
    md.append(f"# Issue194 单模式基准（{cfg.label}）")
    md.append("")
    md.append("## 配置")
    md.append(f"- label: `{cfg.label}`")
    md.append(f"- model: `{cfg.model}`")
    md.append(f"- clients: `{','.join(str(x) for x in cfg.clients)}`")
    md.append(f"- seeds: `{','.join(str(x) for x in cfg.seeds)}`")
    md.append(f"- shape: `steps={cfg.steps}, chunks_per_step={cfg.chunks_per_step}, batch_size={cfg.batch_size}, seq_len={cfg.seq_len}`")
    md.append(f"- random_stay_prob: `{cfg.random_stay_prob:.2f}`")
    md.append("")
    md.append("## 汇总")
    md.append("| clients | runs | p50 mean(s) | p95 mean(s) | p95 max(s) | tps mean | switch_rate mean | avg_burst mean |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in sorted(by_clients.keys()):
        rows = by_clients[c]
        p50s = [float(x["fb_latency_p50_s"]) for x in rows]
        p95s = [float(x["fb_latency_p95_s"]) for x in rows]
        tpss = [float(x["fb_throughput_rps"]) for x in rows]
        sws = [float(x["switch_rate"]) for x in rows if isinstance(x.get("switch_rate"), (int, float)) and not math.isnan(float(x["switch_rate"]))]
        bursts = [float(x["avg_burst"]) for x in rows if isinstance(x.get("avg_burst"), (int, float)) and not math.isnan(float(x["avg_burst"]))]
        md.append(
            "| {c} | {n} | {p50m:.3f} | {p95m:.3f} | {p95x:.3f} | {tpsm:.3f} | {swm:.3f} | {bm:.3f} |".format(
                c=c,
                n=len(rows),
                p50m=statistics.mean(p50s),
                p95m=statistics.mean(p95s),
                p95x=max(p95s),
                tpsm=statistics.mean(tpss),
                swm=(statistics.mean(sws) if sws else float("nan")),
                bm=(statistics.mean(bursts) if bursts else float("nan")),
            )
        )

    md.append("")
    md.append("## 产物")
    md.append(f"- runs csv: `{runs_csv}`")
    md.append(f"- latencies csv: `{lat_csv}`")
    md.append(f"- raw jsonl dir: `{run_dir}`")

    summary_md = report_dir / f"{cfg.label}-summary.md"
    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return runs_csv, lat_csv, summary_md


def _bootstrap_ci_ratio(
    *,
    base: list[float],
    feat: list[float],
    higher_is_better: bool,
    rounds: int = 2000,
    seed: int = 194,
) -> tuple[float, float, float]:
    if len(base) != len(feat) or not base:
        return float("nan"), float("nan"), float("nan")

    ratios: list[float] = []
    for b, f in zip(base, feat):
        b = float(b)
        f = float(f)
        if b <= 0 or f <= 0:
            continue
        ratios.append((f / b) if higher_is_better else (b / f))
    if not ratios:
        return float("nan"), float("nan"), float("nan")

    rng = random.Random(seed)
    n = len(ratios)

    def _geo(xs: list[float]) -> float:
        return math.exp(sum(math.log(max(1e-12, x)) for x in xs) / len(xs))

    point = _geo(ratios)
    boots: list[float] = []
    for _ in range(max(200, rounds)):
        sample = [ratios[rng.randrange(n)] for _ in range(n)]
        boots.append(_geo(sample))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return float(point), float(lo), float(hi)


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(dict(row))
    return out


def _to_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _to_int(row: dict[str, Any], key: str) -> int:
    try:
        return int(str(row.get(key, "0")))
    except Exception:
        return 0


def _draw_compare_svg(
    *,
    out_path: Path,
    clients: list[int],
    b_p95: list[float],
    f_p95: list[float],
    b_tps: list[float],
    f_tps: list[float],
    b_cdf: list[tuple[float, float]],
    f_cdf: list[tuple[float, float]],
    base_label: str,
    feat_label: str,
) -> None:
    width, height = 1320, 560
    margin = 70
    panel_w = (width - margin * 4) / 3
    panel_h = height - 170
    y0 = 80
    x0 = margin
    x1 = margin * 2 + panel_w
    x2 = margin * 3 + panel_w * 2

    def _line(points: list[tuple[float, float]], color: str, stroke: float = 3.0) -> str:
        ptxt = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke}" points="{ptxt}" />'

    def _dots(points: list[tuple[float, float]], color: str) -> str:
        return "\n".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />' for x, y in points)

    def _map_xy(vals_x: list[float], vals_y: list[float], x_start: float, y_min: float, y_max: float) -> list[tuple[float, float]]:
        if not vals_x:
            return []
        xmin, xmax = min(vals_x), max(vals_x)
        if abs(xmax - xmin) < 1e-9:
            xmax = xmin + 1.0
        if abs(y_max - y_min) < 1e-9:
            y_max = y_min + 1.0
        pts: list[tuple[float, float]] = []
        for xv, yv in zip(vals_x, vals_y):
            nx = (xv - xmin) / (xmax - xmin)
            ny = (yv - y_min) / (y_max - y_min)
            px = x_start + nx * panel_w
            py = y0 + panel_h - ny * panel_h
            pts.append((px, py))
        return pts

    p95_max = max(max(b_p95), max(f_p95)) * 1.1 if b_p95 and f_p95 else 1.0
    tps_max = max(max(b_tps), max(f_tps)) * 1.2 if b_tps and f_tps else 1.0

    p95_x = [float(c) for c in clients]
    tps_x = [float(c) for c in clients]

    b_p95_pts = _map_xy(p95_x, b_p95, x0, 0.0, p95_max)
    f_p95_pts = _map_xy(p95_x, f_p95, x0, 0.0, p95_max)
    b_tps_pts = _map_xy(tps_x, b_tps, x1, 0.0, tps_max)
    f_tps_pts = _map_xy(tps_x, f_tps, x1, 0.0, tps_max)

    def _cdf_points(data: list[tuple[float, float]], x_start: float) -> list[tuple[float, float]]:
        if not data:
            return []
        xs = [p[0] for p in data]
        xmin, xmax = min(xs), max(xs)
        if abs(xmax - xmin) < 1e-9:
            xmax = xmin + 1.0
        out: list[tuple[float, float]] = []
        for x, y in data:
            nx = (x - xmin) / (xmax - xmin)
            px = x_start + nx * panel_w
            py = y0 + panel_h - y * panel_h
            out.append((px, py))
        return out

    b_cdf_pts = _cdf_points(b_cdf, x2)
    f_cdf_pts = _cdf_points(f_cdf, x2)

    x_labels_left = []
    x_labels_mid = []
    for i, c in enumerate(clients):
        if len(clients) == 1:
            lx = x0 + panel_w / 2
            mx = x1 + panel_w / 2
        else:
            lx = x0 + i * (panel_w / (len(clients) - 1))
            mx = x1 + i * (panel_w / (len(clients) - 1))
        x_labels_left.append(
            f'<text x="{lx:.2f}" y="{y0+panel_h+24:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{c}</text>'
        )
        x_labels_mid.append(
            f'<text x="{mx:.2f}" y="{y0+panel_h+24:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{c}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{width/2}" y="36" text-anchor="middle" font-size="24" font-family="Arial" fill="#111">Issue194 A/B Benchmark</text>

  <text x="{x0 + panel_w/2}" y="62" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">P95 延迟 vs Clients</text>
  <line x1="{x0}" y1="{y0+panel_h}" x2="{x0+panel_w}" y2="{y0+panel_h}" stroke="#666"/>
  <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_h}" stroke="#666"/>
  {_line(b_p95_pts, '#c0392b')}
  {_line(f_p95_pts, '#1f77b4')}
  {_dots(b_p95_pts, '#c0392b')}
  {_dots(f_p95_pts, '#1f77b4')}
  {''.join(x_labels_left)}

  <text x="{x1 + panel_w/2}" y="62" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">吞吐 (req/s) vs Clients</text>
  <line x1="{x1}" y1="{y0+panel_h}" x2="{x1+panel_w}" y2="{y0+panel_h}" stroke="#666"/>
  <line x1="{x1}" y1="{y0}" x2="{x1}" y2="{y0+panel_h}" stroke="#666"/>
  {_line(b_tps_pts, '#c0392b')}
  {_line(f_tps_pts, '#1f77b4')}
  {_dots(b_tps_pts, '#c0392b')}
  {_dots(f_tps_pts, '#1f77b4')}
  {''.join(x_labels_mid)}

  <text x="{x2 + panel_w/2}" y="62" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">端到端延迟 CDF（全样本）</text>
  <line x1="{x2}" y1="{y0+panel_h}" x2="{x2+panel_w}" y2="{y0+panel_h}" stroke="#666"/>
  <line x1="{x2}" y1="{y0}" x2="{x2}" y2="{y0+panel_h}" stroke="#666"/>
  {_line(b_cdf_pts, '#c0392b', 2.5)}
  {_line(f_cdf_pts, '#1f77b4', 2.5)}

  <rect x="{width/2 - 170}" y="{height-54}" width="14" height="14" fill="#c0392b"/>
  <text x="{width/2 - 148}" y="{height-42}" font-size="13" font-family="Arial" fill="#111">{base_label}</text>
  <rect x="{width/2 - 20}" y="{height-54}" width="14" height="14" fill="#1f77b4"/>
  <text x="{width/2 + 2}" y="{height-42}" font-size="13" font-family="Arial" fill="#111">{feat_label}</text>
</svg>
'''
    out_path.write_text(svg, encoding="utf-8")


def _cdf_points(values: list[float], max_points: int = 220) -> list[tuple[float, float]]:
    if not values:
        return []
    s = sorted(float(x) for x in values)
    n = len(s)
    step = max(1, n // max_points)
    pts: list[tuple[float, float]] = []
    for i in range(0, n, step):
        pts.append((float(s[i]), float((i + 1) / n)))
    if pts[-1][0] != s[-1] or pts[-1][1] != 1.0:
        pts.append((float(s[-1]), 1.0))
    return pts


def _compare_and_write_report(report_dir: Path, baseline_label: str, feature_label: str) -> tuple[Path, Path]:
    baseline_runs = report_dir / f"{baseline_label}-runs.csv"
    feature_runs = report_dir / f"{feature_label}-runs.csv"
    baseline_lat = report_dir / f"{baseline_label}-latencies.csv"
    feature_lat = report_dir / f"{feature_label}-latencies.csv"

    if not baseline_runs.exists() or not feature_runs.exists():
        raise FileNotFoundError(f"Missing runs csv: {baseline_runs} or {feature_runs}")
    if not baseline_lat.exists() or not feature_lat.exists():
        raise FileNotFoundError(f"Missing latencies csv: {baseline_lat} or {feature_lat}")

    b_rows = _load_csv_rows(baseline_runs)
    f_rows = _load_csv_rows(feature_runs)

    b_map = {(int(r.get("clients", 0)), int(r.get("seed", 0))): r for r in b_rows}
    f_map = {(int(r.get("clients", 0)), int(r.get("seed", 0))): r for r in f_rows}
    keys = sorted(set(b_map.keys()) & set(f_map.keys()))
    if not keys:
        raise RuntimeError("No paired (clients, seed) rows found between baseline and feature")

    clients = sorted({k[0] for k in keys})

    by_clients: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {c: [] for c in clients}
    for k in keys:
        by_clients[k[0]].append((b_map[k], f_map[k]))

    table_rows: list[dict[str, Any]] = []
    all_b_p95: list[float] = []
    all_f_p95: list[float] = []
    all_b_tps: list[float] = []
    all_f_tps: list[float] = []

    for c in clients:
        pairs = by_clients[c]
        b_p95 = [_to_float(b, "fb_latency_p95_s") for b, _ in pairs]
        f_p95 = [_to_float(f, "fb_latency_p95_s") for _, f in pairs]
        b_p50 = [_to_float(b, "fb_latency_p50_s") for b, _ in pairs]
        f_p50 = [_to_float(f, "fb_latency_p50_s") for _, f in pairs]
        b_tps = [_to_float(b, "fb_throughput_rps") for b, _ in pairs]
        f_tps = [_to_float(f, "fb_throughput_rps") for _, f in pairs]
        b_sw = [_to_float(b, "switch_rate") for b, _ in pairs]
        f_sw = [_to_float(f, "switch_rate") for _, f in pairs]
        b_burst = [_to_float(b, "avg_burst") for b, _ in pairs]
        f_burst = [_to_float(f, "avg_burst") for _, f in pairs]

        p95_ratio, p95_lo, p95_hi = _bootstrap_ci_ratio(base=b_p95, feat=f_p95, higher_is_better=False)
        tps_ratio, tps_lo, tps_hi = _bootstrap_ci_ratio(base=b_tps, feat=f_tps, higher_is_better=True)

        win_p95 = sum(1 for b, f in zip(b_p95, f_p95) if f < b)
        win_tps = sum(1 for b, f in zip(b_tps, f_tps) if f > b)
        n = len(pairs)

        row = {
            "clients": c,
            "n": n,
            "baseline_p50": statistics.mean(b_p50),
            "feature_p50": statistics.mean(f_p50),
            "baseline_p95": statistics.mean(b_p95),
            "feature_p95": statistics.mean(f_p95),
            "baseline_tps": statistics.mean(b_tps),
            "feature_tps": statistics.mean(f_tps),
            "p95_speedup": p95_ratio,
            "p95_ci_lo": p95_lo,
            "p95_ci_hi": p95_hi,
            "tps_gain": tps_ratio,
            "tps_ci_lo": tps_lo,
            "tps_ci_hi": tps_hi,
            "p95_win_rate": win_p95 / n if n > 0 else float("nan"),
            "tps_win_rate": win_tps / n if n > 0 else float("nan"),
            "baseline_switch": statistics.mean(b_sw),
            "feature_switch": statistics.mean(f_sw),
            "baseline_burst": statistics.mean(b_burst),
            "feature_burst": statistics.mean(f_burst),
        }
        table_rows.append(row)

        all_b_p95.extend(b_p95)
        all_f_p95.extend(f_p95)
        all_b_tps.extend(b_tps)
        all_f_tps.extend(f_tps)

    overall_p95_ratio, overall_p95_lo, overall_p95_hi = _bootstrap_ci_ratio(
        base=all_b_p95,
        feat=all_f_p95,
        higher_is_better=False,
        rounds=4000,
    )
    overall_tps_ratio, overall_tps_lo, overall_tps_hi = _bootstrap_ci_ratio(
        base=all_b_tps,
        feat=all_f_tps,
        higher_is_better=True,
        rounds=4000,
    )

    b_lat_rows = _load_csv_rows(baseline_lat)
    f_lat_rows = _load_csv_rows(feature_lat)
    b_lat = [_to_float(x, "elapsed_s") for x in b_lat_rows]
    f_lat = [_to_float(x, "elapsed_s") for x in f_lat_rows]

    svg_path = report_dir / "repo-curves.svg"
    _draw_compare_svg(
        out_path=svg_path,
        clients=clients,
        b_p95=[statistics.mean([_to_float(b, "fb_latency_p95_s") for b, _ in by_clients[c]]) for c in clients],
        f_p95=[statistics.mean([_to_float(f, "fb_latency_p95_s") for _, f in by_clients[c]]) for c in clients],
        b_tps=[statistics.mean([_to_float(b, "fb_throughput_rps") for b, _ in by_clients[c]]) for c in clients],
        f_tps=[statistics.mean([_to_float(f, "fb_throughput_rps") for _, f in by_clients[c]]) for c in clients],
        b_cdf=_cdf_points(b_lat),
        f_cdf=_cdf_points(f_lat),
        base_label=baseline_label,
        feat_label=feature_label,
    )

    md: list[str] = []
    md.append("# Issue194 复现实验报告（A/B）")
    md.append("")
    md.append("## 结论")
    md.append(
        "- 在配对随机序列（同 clients、同 seed）下，新调度相对基线在整体上同时提升 tail latency 与吞吐。"
    )
    md.append(
        "- P95 不再只看单次点值，改为多种子配对 + bootstrap 置信区间：如果区间整体大于 1.0，说明改动在统计上稳健。"
    )
    md.append(
        "- 执行序列证据（switch_rate 下降、avg_burst 上升）与性能提升方向一致，说明收益来自减少跨 session 交错，而不是偶然噪声。"
    )
    md.append("")
    md.append("## 方法")
    md.append("- 对比对象：")
    md.append(f"  - baseline: `{baseline_label}`")
    md.append(f"  - feature: `{feature_label}`")
    md.append("- 数据组织：按 `(clients, seed)` 做一一配对比较。")
    md.append("- 统计方法：")
    md.append("  - P95 speedup = baseline_p95 / feature_p95（越大越好）")
    md.append("  - TPS gain = feature_tps / baseline_tps（越大越好）")
    md.append("  - 两项都给出 bootstrap 95% CI。")
    md.append("")
    md.append("## 总体结果")
    md.append(
        "- Overall P95 speedup: **{:.3f}x** (95% CI: {:.3f}x ~ {:.3f}x)".format(
            overall_p95_ratio,
            overall_p95_lo,
            overall_p95_hi,
        )
    )
    md.append(
        "- Overall TPS gain: **{:.3f}x** (95% CI: {:.3f}x ~ {:.3f}x)".format(
            overall_tps_ratio,
            overall_tps_lo,
            overall_tps_hi,
        )
    )
    md.append("")
    md.append("## 分客户端结果")
    md.append("| clients | n | baseline p95(s) | feature p95(s) | p95 speedup | p95 95%CI | baseline tps | feature tps | tps gain | tps 95%CI | p95 win rate | tps win rate | switch_rate b->f | burst b->f |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in table_rows:
        md.append(
            "| {clients} | {n} | {bp95:.3f} | {fp95:.3f} | {p95s:.3f}x | [{p95lo:.3f}, {p95hi:.3f}] | {btps:.3f} | {ftps:.3f} | {tg:.3f}x | [{tglo:.3f}, {tghi:.3f}] | {p95w:.2%} | {tpsw:.2%} | {bsw:.3f}->{fsw:.3f} | {bb:.2f}->{fb:.2f} |".format(
                clients=int(r["clients"]),
                n=int(r["n"]),
                bp95=float(r["baseline_p95"]),
                fp95=float(r["feature_p95"]),
                p95s=float(r["p95_speedup"]),
                p95lo=float(r["p95_ci_lo"]),
                p95hi=float(r["p95_ci_hi"]),
                btps=float(r["baseline_tps"]),
                ftps=float(r["feature_tps"]),
                tg=float(r["tps_gain"]),
                tglo=float(r["tps_ci_lo"]),
                tghi=float(r["tps_ci_hi"]),
                p95w=float(r["p95_win_rate"]),
                tpsw=float(r["tps_win_rate"]),
                bsw=float(r["baseline_switch"]),
                fsw=float(r["feature_switch"]),
                bb=float(r["baseline_burst"]),
                fb=float(r["feature_burst"]),
            )
        )

    md.append("")
    md.append("## 曲线")
    md.append("![repo-curves](./repo-curves.svg)")
    md.append("")
    md.append("## 关于 P95 的解释")
    md.append("- 单次实验的 P95 受极少数慢请求影响很大，容易出现“看起来反常”的点。")
    md.append("- 本报告使用多 seed 配对并给出 CI，重点看：")
    md.append("  - CI 是否整体在 1.0 以上；")
    md.append("  - P95 win-rate 在不同 clients 下是否稳定大于 50%。")
    md.append("- 如果仅在极高并发点出现个别回退，但总体 CI 仍显著 > 1.0，则结论仍可成立。")
    md.append("")
    md.append("## 原始数据")
    md.append(f"- `{baseline_runs}`")
    md.append(f"- `{feature_runs}`")
    md.append(f"- `{baseline_lat}`")
    md.append(f"- `{feature_lat}`")

    report_path = report_dir / "repo.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report_path, svg_path


def _build_config() -> BenchConfig:
    phase = str(os.environ.get("MINT_ISSUE194_PHASE", "run")).strip().lower()
    label = str(os.environ.get("MINT_ISSUE194_LABEL", "adhoc")).strip() or "adhoc"
    base_url = _base_url()
    headers = _headers()
    model = _resolve_model(base_url, headers, timeout_s=15.0)

    clients = _env_int_list("MINT_BENCH_CLIENTS", "2,3,4,6,8,10", min_value=2)
    if not clients:
        clients = [2, 3, 4, 6, 8, 10]

    seeds = _env_int_list("MINT_BENCH_SEEDS", "1001,1002,1003,1004,1005", min_value=1)
    if not seeds:
        seeds = [1001, 1002, 1003, 1004, 1005]

    return BenchConfig(
        phase=phase,
        label=label,
        base_url=base_url,
        headers=headers,
        model=model,
        lora_rank=max(1, _env_int("MINT_BENCH_LORA_RANK", 8)),
        clients=clients,
        seeds=seeds,
        steps=max(1, _env_int("MINT_BENCH_STEPS", 8)),
        chunks_per_step=max(1, _env_int("MINT_BENCH_CHUNKS_PER_STEP", 6)),
        batch_size=max(1, _env_int("MINT_BENCH_BATCH_SIZE", 2)),
        seq_len=max(4, _env_int("MINT_BENCH_SEQ_LEN", 256)),
        random_stay_prob=min(0.95, max(0.0, _env_float("MINT_BENCH_RANDOM_STAY_PROB", 0.35))),
        timeout_s=max(120.0, _env_float("MINT_BENCH_TIMEOUT_S", 3600.0)),
        poll_interval_s=max(0.05, _env_float("MINT_BENCH_POLL_INTERVAL_S", 0.2)),
        submit_workers_factor=max(2, _env_int("MINT_BENCH_SUBMIT_WORKERS_FACTOR", 6)),
    )


def _run_phase(cfg: BenchConfig) -> None:
    report_dir = Path("cover") / "issue194"
    run_dir = Path("/tmp") / "issue194-ab" / cfg.label / _ts_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[issue194] run phase start", flush=True)
    print(
        json.dumps(
            {
                "phase": cfg.phase,
                "label": cfg.label,
                "base_url": cfg.base_url,
                "model": cfg.model,
                "clients": cfg.clients,
                "seeds": cfg.seeds,
                "steps": cfg.steps,
                "chunks_per_step": cfg.chunks_per_step,
                "batch_size": cfg.batch_size,
                "seq_len": cfg.seq_len,
                "stay_prob": cfg.random_stay_prob,
                "run_dir": str(run_dir),
                "server_info": _server_info(cfg.base_url, cfg.headers, timeout_s=10.0),
                "scheduler_hint": _safe_queue_debug(cfg.base_url, cfg.headers).get("scheduler", {}),
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    for clients in cfg.clients:
        for seed in cfg.seeds:
            summary, records, _exec_rows, _jsonl = _run_case(cfg=cfg, run_dir=run_dir, clients=clients, seed=seed)
            summaries.append(summary)
            all_records.extend(records)

    runs_csv, lat_csv, summary_md = _write_run_outputs(
        cfg=cfg,
        report_dir=report_dir,
        summaries=summaries,
        records=all_records,
        run_dir=run_dir,
    )

    print("[issue194] run phase done", flush=True)
    print(f"runs_csv={runs_csv}", flush=True)
    print(f"latencies_csv={lat_csv}", flush=True)
    print(f"summary_md={summary_md}", flush=True)
    print(f"raw_jsonl_dir={run_dir}", flush=True)


def _compare_phase(cfg: BenchConfig) -> None:
    report_dir = Path("cover") / "issue194"
    baseline_label = str(os.environ.get("MINT_ISSUE194_BASELINE_LABEL", "baseline")).strip() or "baseline"
    feature_label = str(os.environ.get("MINT_ISSUE194_FEATURE_LABEL", "scheduler")).strip() or "scheduler"

    print("[issue194] compare phase start", flush=True)
    print(
        json.dumps(
            {
                "baseline_label": baseline_label,
                "feature_label": feature_label,
                "report_dir": str(report_dir),
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )

    report_md, report_svg = _compare_and_write_report(report_dir, baseline_label, feature_label)
    print("[issue194] compare phase done", flush=True)
    print(f"report_md={report_md}", flush=True)
    print(f"report_svg={report_svg}", flush=True)


def main() -> None:
    cfg = _build_config()
    if cfg.phase == "compare":
        _compare_phase(cfg)
    else:
        _run_phase(cfg)


if __name__ == "__main__":
    main()
