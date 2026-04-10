#!/usr/bin/env python3
"""Issue 432 local mixed-load harness.

Purpose:
- generate sustained local legacy FIFO pressure with `/internal/work_queue/noop`
- generate scheduled local training pressure with `training.forward_backward`
- capture the Gate 5 artifacts needed to explain global arbitration decisions

This script keeps scope tight to issue432. It does not try to reproduce the
production incident byte-for-byte. It isolates the local legacy-vs-scheduled
arbiter with deterministic inputs and captures the required debug evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import random
import statistics
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class Config:
    base_url: str
    headers: dict[str, str]
    base_model: str
    lora_rank: int
    training_sessions: int
    training_steps: int
    batch_size: int
    seq_len: int
    legacy_target_outstanding: int
    legacy_poll_batch: int
    legacy_warmup_s: float
    snapshot_interval_s: float
    timeout_s: float
    poll_interval_s: float
    output_dir: Path
    raw_dir: Path
    label: str


@dataclass(frozen=True)
class TrainingJob:
    job_id: int
    session_idx: int
    session_id: str
    model_id: str
    seq_id: int
    seed_base: int


class LegacyController:
    def __init__(self, *, cfg: Config, stop_event: threading.Event, out_path: Path) -> None:
        self.cfg = cfg
        self.stop_event = stop_event
        self.out_path = out_path
        self.submitted = 0
        self.completed = 0
        self.errors = 0
        self.max_outstanding = 0
        self.outstanding: dict[str, float] = {}
        self.records: list[dict[str, Any]] = []

    def _log(self, rec: dict[str, Any]) -> None:
        _append_jsonl(self.out_path, rec)

    def _submit_noop(self, session: requests.Session) -> str:
        r = session.post(
            f"{self.cfg.base_url}/internal/work_queue/noop",
            headers=self.cfg.headers,
            timeout=min(self.cfg.timeout_s, 30.0),
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise TypeError(f"/internal/work_queue/noop returned non-dict: {type(data)}")
        request_id = data.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(f"/internal/work_queue/noop missing request_id: {data}")
        return request_id

    def _poll_once(self, session: requests.Session, request_id: str) -> tuple[str, dict[str, Any] | None]:
        r = session.post(
            f"{self.cfg.base_url}/api/v1/retrieve_future",
            headers=self.cfg.headers,
            json={"request_id": request_id},
            timeout=min(self.cfg.timeout_s, 30.0),
        )
        if r.status_code == 408:
            data = r.json()
            return "pending", data if isinstance(data, dict) else None
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise TypeError(f"retrieve_future returned non-dict: {type(data)}")
        if "error" in data:
            return "error", data
        return "done", data

    def run(self) -> None:
        session = requests.Session()
        try:
            while not self.stop_event.is_set():
                while len(self.outstanding) < self.cfg.legacy_target_outstanding and not self.stop_event.is_set():
                    submit_ts = time.time()
                    try:
                        request_id = self._submit_noop(session)
                    except Exception as e:
                        self.errors += 1
                        self._log(
                            {
                                "kind": "legacy_submit_error",
                                "ts": _now_iso(),
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                        time.sleep(0.05)
                        break
                    self.outstanding[request_id] = submit_ts
                    self.submitted += 1
                    self.max_outstanding = max(self.max_outstanding, len(self.outstanding))
                    self._log(
                        {
                            "kind": "legacy_submit",
                            "ts": _now_iso(),
                            "request_id": request_id,
                            "submit_ts": submit_ts,
                            "outstanding": len(self.outstanding),
                        }
                    )
                self._poll_batch(session)
                time.sleep(0.01)

            deadline = time.time() + self.cfg.timeout_s
            while self.outstanding and time.time() < deadline:
                self._poll_batch(session)
                time.sleep(self.cfg.poll_interval_s)
        finally:
            session.close()

    def _poll_batch(self, session: requests.Session) -> None:
        request_ids = list(self.outstanding.keys())[: self.cfg.legacy_poll_batch]
        for request_id in request_ids:
            try:
                status, payload = self._poll_once(session, request_id)
            except Exception as e:
                self.errors += 1
                submit_ts = self.outstanding.pop(request_id, None)
                self._log(
                    {
                        "kind": "legacy_poll_error",
                        "ts": _now_iso(),
                        "request_id": request_id,
                        "submit_ts": submit_ts,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            if status == "pending":
                continue
            submit_ts = self.outstanding.pop(request_id, None)
            finish_ts = time.time()
            rec = {
                "kind": "legacy_done" if status == "done" else "legacy_result_error",
                "ts": _now_iso(),
                "request_id": request_id,
                "submit_ts": submit_ts,
                "finish_ts": finish_ts,
                "elapsed_s": None if submit_ts is None else max(0.0, finish_ts - float(submit_ts)),
                "payload": payload,
            }
            if status == "done":
                self.completed += 1
            else:
                self.errors += 1
            self.records.append(rec)
            self._log(rec)


class SnapshotSampler:
    def __init__(self, *, cfg: Config, stop_event: threading.Event, out_path: Path) -> None:
        self.cfg = cfg
        self.stop_event = stop_event
        self.out_path = out_path
        self.seen_dequeues: set[tuple[str, float]] = set()
        self.seen_enqueues: set[tuple[str, float]] = set()

    def run(self) -> None:
        session = requests.Session()
        try:
            while not self.stop_event.is_set():
                self._sample_once(session)
                time.sleep(self.cfg.snapshot_interval_s)
        finally:
            session.close()

    def _sample_once(self, session: requests.Session) -> None:
        sample_ts = time.time()
        try:
            admission = _get_json(
                session,
                f"{self.cfg.base_url}/internal/admission_stats",
                headers=self.cfg.headers,
                timeout_s=self.cfg.timeout_s,
            )
        except Exception as e:
            _append_jsonl(
                self.out_path,
                {
                    "kind": "snapshot_error",
                    "ts": _now_iso(),
                    "target": "admission_stats",
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            admission = {}
        try:
            debug = _get_json(
                session,
                f"{self.cfg.base_url}/internal/work_queue/debug_state",
                headers=self.cfg.headers,
                timeout_s=self.cfg.timeout_s,
            )
        except Exception as e:
            _append_jsonl(
                self.out_path,
                {
                    "kind": "snapshot_error",
                    "ts": _now_iso(),
                    "target": "debug_state",
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            debug = {}

        scheduler = debug.get("scheduler") if isinstance(debug, dict) else {}
        stats = debug.get("stats") if isinstance(debug, dict) else {}
        domains = scheduler.get("domains") if isinstance(scheduler, dict) else {}
        recent_dequeues = debug.get("recent_dequeues") if isinstance(debug, dict) else []
        recent_enqueues = debug.get("recent_enqueues") if isinstance(debug, dict) else []

        if isinstance(recent_dequeues, list):
            for rec in recent_dequeues:
                if not isinstance(rec, dict):
                    continue
                request_id = rec.get("request_id")
                raw_ts = rec.get("ts")
                if not isinstance(request_id, str):
                    continue
                try:
                    ts_f = float(raw_ts)
                except Exception:
                    continue
                key = (request_id, ts_f)
                if key in self.seen_dequeues:
                    continue
                self.seen_dequeues.add(key)
                _append_jsonl(self.out_path, {"kind": "recent_dequeue", "captured_at": _now_iso(), "record": rec})

        if isinstance(recent_enqueues, list):
            for rec in recent_enqueues:
                if not isinstance(rec, dict):
                    continue
                request_id = rec.get("request_id")
                raw_ts = rec.get("ts")
                if not isinstance(request_id, str):
                    continue
                try:
                    ts_f = float(raw_ts)
                except Exception:
                    continue
                key = (request_id, ts_f)
                if key in self.seen_enqueues:
                    continue
                self.seen_enqueues.add(key)
                _append_jsonl(self.out_path, {"kind": "recent_enqueue", "captured_at": _now_iso(), "record": rec})

        work_queue = admission.get("work_queue") if isinstance(admission, dict) else {}
        _append_jsonl(
            self.out_path,
            {
                "kind": "snapshot",
                "ts": _now_iso(),
                "sample_ts": sample_ts,
                "depth": work_queue.get("depth") if isinstance(work_queue, dict) else None,
                "scheduler_arbitration_total": work_queue.get("scheduler_arbitration_total") if isinstance(work_queue, dict) else None,
                "scheduler_arbitration_by_winner": work_queue.get("scheduler_arbitration_by_winner") if isinstance(work_queue, dict) else None,
                "scheduler_arbitration_by_reason": work_queue.get("scheduler_arbitration_by_reason") if isinstance(work_queue, dict) else None,
                "scheduler_legacy_streak": scheduler.get("legacy_streak") if isinstance(scheduler, dict) else None,
                "scheduler_domains": domains,
                "debug_stats": stats,
            },
        )


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _ts_dir() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _post_json(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    r = session.post(url, headers=headers, json=payload, timeout=min(timeout_s, 60.0))
    if r.status_code >= 400:
        raise RuntimeError(f"POST {url} failed status={r.status_code} body={r.text[:500]}")
    out = r.json()
    if not isinstance(out, dict):
        raise TypeError(f"POST {url} returned non-dict: {type(out)}")
    return out


def _get_json(session: requests.Session, url: str, *, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    r = session.get(url, headers=headers, timeout=min(timeout_s, 60.0))
    if r.status_code >= 400:
        raise RuntimeError(f"GET {url} failed status={r.status_code} body={r.text[:500]}")
    out = r.json()
    if not isinstance(out, dict):
        raise TypeError(f"GET {url} returned non-dict: {type(out)}")
    return out


def _get_text(session: requests.Session, url: str, *, headers: dict[str, str], timeout_s: float) -> str:
    r = session.get(url, headers=headers, timeout=min(timeout_s, 60.0))
    if r.status_code >= 400:
        raise RuntimeError(f"GET {url} failed status={r.status_code} body={r.text[:500]}")
    return r.text


def _poll_future(
    session: requests.Session,
    *,
    cfg: Config,
    request_id: str,
    event_path: Path,
    op_name: str,
) -> dict[str, Any]:
    t0 = time.time()
    while True:
        if time.time() - t0 > cfg.timeout_s:
            raise TimeoutError(f"retrieve_future timeout request_id={request_id} op={op_name}")
        r = session.post(
            f"{cfg.base_url}/api/v1/retrieve_future",
            headers=cfg.headers,
            json={"request_id": request_id},
            timeout=min(cfg.timeout_s, 30.0),
        )
        if r.status_code == 408:
            try:
                payload = r.json()
            except Exception:
                payload = None
            _append_jsonl(
                event_path,
                {
                    "kind": "pending",
                    "ts": _now_iso(),
                    "request_id": request_id,
                    "op": op_name,
                    "status_code": 408,
                    "payload": payload,
                },
            )
            time.sleep(cfg.poll_interval_s)
            continue
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, dict):
            raise TypeError(f"retrieve_future returned non-dict: {type(payload)}")
        _append_jsonl(
            event_path,
            {
                "kind": "done",
                "ts": _now_iso(),
                "request_id": request_id,
                "op": op_name,
                "status_code": r.status_code,
                "payload": payload,
            },
        )
        return payload


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


def _build_batch(*, batch_size: int, seq_len: int, seed_base: int) -> list[dict[str, Any]]:
    return [_build_datum(seq_len=seq_len, seed=seed_base + i) for i in range(batch_size)]


def _create_model(session: requests.Session, *, cfg: Config, session_id: str, event_path: Path) -> str:
    payload = {
        "session_id": session_id,
        "model_seq_id": 0,
        "base_model": cfg.base_model,
        "lora_config": {"rank": int(cfg.lora_rank)},
    }
    first = _post_json(
        session,
        f"{cfg.base_url}/api/v1/create_model",
        headers=cfg.headers,
        payload=payload,
        timeout_s=cfg.timeout_s,
    )
    request_id = first.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"create_model missing request_id: {first}")
    _append_jsonl(
        event_path,
        {
            "kind": "submit",
            "ts": _now_iso(),
            "request_id": request_id,
            "op": "training.create_model",
            "session_id": session_id,
        },
    )
    out = _poll_future(session, cfg=cfg, request_id=request_id, event_path=event_path, op_name="training.create_model")
    model_id = out.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model future missing model_id: {out}")
    return model_id


def _delete_model(session: requests.Session, *, cfg: Config, model_id: str) -> None:
    try:
        session.delete(
            f"{cfg.base_url}/api/v1/models/{model_id}",
            headers=cfg.headers,
            timeout=min(cfg.timeout_s, 60.0),
        )
    except Exception:
        pass


def _run_training_job(cfg: Config, job: TrainingJob, event_path: Path) -> dict[str, Any]:
    session = requests.Session()
    try:
        submit_ts = time.time()
        payload = {
            "model_id": job.model_id,
            "seq_id": int(job.seq_id),
            "forward_backward_input": {
                "data": _build_batch(batch_size=cfg.batch_size, seq_len=cfg.seq_len, seed_base=job.seed_base),
                "loss_fn": "cross_entropy",
            },
        }
        first = _post_json(
            session,
            f"{cfg.base_url}/api/v1/forward_backward",
            headers=cfg.headers,
            payload=payload,
            timeout_s=cfg.timeout_s,
        )
        request_id = first.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(f"forward_backward missing request_id: {first}")
        _append_jsonl(
            event_path,
            {
                "kind": "submit",
                "ts": _now_iso(),
                "request_id": request_id,
                "op": "training.forward_backward",
                "job_id": job.job_id,
                "session_id": job.session_id,
                "model_id": job.model_id,
                "seq_id": job.seq_id,
                "submit_ts": submit_ts,
            },
        )
        out = _poll_future(
            session,
            cfg=cfg,
            request_id=request_id,
            event_path=event_path,
            op_name="training.forward_backward",
        )
        finish_ts = time.time()
        return {
            "ok": True,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "model_id": job.model_id,
            "seq_id": job.seq_id,
            "request_id": request_id,
            "submit_ts": submit_ts,
            "finish_ts": finish_ts,
            "elapsed_s": max(0.0, finish_ts - submit_ts),
            "result": out,
        }
    except Exception as e:
        finish_ts = time.time()
        return {
            "ok": False,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "model_id": job.model_id,
            "seq_id": job.seq_id,
            "request_id": None,
            "submit_ts": None,
            "finish_ts": finish_ts,
            "elapsed_s": None,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        session.close()


def _build_jobs(session_infos: list[dict[str, str]], *, training_steps: int) -> list[TrainingJob]:
    jobs: list[TrainingJob] = []
    job_id = 0
    for step in range(training_steps):
        for session_idx, info in enumerate(session_infos):
            jobs.append(
                TrainingJob(
                    job_id=job_id,
                    session_idx=session_idx,
                    session_id=str(info["session_id"]),
                    model_id=str(info["model_id"]),
                    seq_id=step,
                    seed_base=(session_idx + 1) * 100_000 + (step + 1) * 1_000,
                )
            )
            job_id += 1
    return jobs


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    s = sorted(values)
    idx = int((len(s) * p + 99) // 100) - 1
    idx = max(0, min(len(s) - 1, idx))
    return float(s[idx])


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    vals = [float(x) for x in values]
    return {
        "count": len(vals),
        "mean": float(statistics.mean(vals)),
        "p50": float(_percentile(vals, 50)),
        "p95": float(_percentile(vals, 95)),
        "max": float(max(vals)),
    }


def _dict_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in sorted(set(before) | set(after)):
        try:
            a = int(after.get(key, 0))
        except Exception:
            a = 0
        try:
            b = int(before.get(key, 0))
        except Exception:
            b = 0
        out[str(key)] = a - b
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _collect_dequeues(sample_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in sample_rows:
        if row.get("kind") != "recent_dequeue":
            continue
        rec = row.get("record")
        if not isinstance(rec, dict):
            continue
        request_id = rec.get("request_id")
        if isinstance(request_id, str) and request_id:
            out[request_id] = rec
    return out


def _domain_peak(sample_rows: list[dict[str, Any]], domain: str | None) -> dict[str, Any]:
    peak = {
        "max_service_gap_s": None,
        "max_pending_requests": None,
        "max_inflight_workers": None,
        "max_capacity_workers": None,
        "max_oldest_queued_s": None,
        "admissible_false_samples": 0,
    }
    if not domain:
        return peak
    for row in sample_rows:
        if row.get("kind") != "snapshot":
            continue
        domains = row.get("scheduler_domains")
        if not isinstance(domains, dict):
            continue
        rec = domains.get(domain)
        if not isinstance(rec, dict):
            continue
        for src_key, dst_key in (
            ("service_gap_s", "max_service_gap_s"),
            ("pending_requests", "max_pending_requests"),
            ("inflight_workers", "max_inflight_workers"),
            ("capacity_workers", "max_capacity_workers"),
            ("oldest_queued_s", "max_oldest_queued_s"),
        ):
            val = rec.get(src_key)
            if not isinstance(val, (int, float)):
                continue
            cur = peak.get(dst_key)
            if cur is None or float(val) > float(cur):
                peak[dst_key] = float(val)
        if rec.get("admissible") is False:
            peak["admissible_false_samples"] = int(peak["admissible_false_samples"]) + 1
    return peak


def _build_summary(
    *,
    cfg: Config,
    before_admission: dict[str, Any],
    after_admission: dict[str, Any],
    training_results: list[dict[str, Any]],
    legacy: LegacyController,
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    deq_by_id = _collect_dequeues(sample_rows)
    ok_rows = [row for row in training_results if row.get("ok")]
    err_rows = [row for row in training_results if not row.get("ok")]

    latencies: list[float] = []
    queue_waits: list[float] = []
    exec_to_done: list[float] = []
    by_session_lat: dict[str, list[float]] = {}
    by_session_qw: dict[str, list[float]] = {}
    scheduled_domains: list[str] = []

    for row in ok_rows:
        elapsed = row.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            latencies.append(float(elapsed))
        sid = str(row.get("session_id") or "")
        if sid and isinstance(elapsed, (int, float)):
            by_session_lat.setdefault(sid, []).append(float(elapsed))
        request_id = row.get("request_id")
        deq = deq_by_id.get(str(request_id)) if isinstance(request_id, str) else None
        if isinstance(deq, dict):
            domain = deq.get("scheduler_domain")
            if isinstance(domain, str) and domain:
                scheduled_domains.append(domain)
            submit_ts = row.get("submit_ts")
            deq_ts = deq.get("ts")
            finish_ts = row.get("finish_ts")
            if isinstance(submit_ts, (int, float)) and isinstance(deq_ts, (int, float)):
                qw = max(0.0, float(deq_ts) - float(submit_ts))
                queue_waits.append(qw)
                if sid:
                    by_session_qw.setdefault(sid, []).append(qw)
            if isinstance(finish_ts, (int, float)) and isinstance(deq_ts, (int, float)):
                exec_to_done.append(max(0.0, float(finish_ts) - float(deq_ts)))

    target_domain = statistics.mode(scheduled_domains) if scheduled_domains else None
    if scheduled_domains:
        try:
            target_domain = statistics.mode(scheduled_domains)
        except statistics.StatisticsError:
            target_domain = scheduled_domains[0]

    before_wq = before_admission.get("work_queue") if isinstance(before_admission, dict) else {}
    after_wq = after_admission.get("work_queue") if isinstance(after_admission, dict) else {}
    max_depth = None
    max_legacy_streak = None
    for row in sample_rows:
        if row.get("kind") != "snapshot":
            continue
        depth = row.get("depth")
        if isinstance(depth, (int, float)):
            max_depth = float(depth) if max_depth is None else max(float(depth), float(max_depth))
        streak = row.get("scheduler_legacy_streak")
        if isinstance(streak, (int, float)):
            max_legacy_streak = float(streak) if max_legacy_streak is None else max(float(streak), float(max_legacy_streak))

    by_session = {}
    for sid in sorted(set(by_session_lat) | set(by_session_qw)):
        by_session[sid] = {
            "latency_s": _stats(by_session_lat.get(sid, [])),
            "queue_wait_s": _stats(by_session_qw.get(sid, [])),
        }

    return {
        "label": cfg.label,
        "base_url": cfg.base_url,
        "base_model": cfg.base_model,
        "target_scheduler_domain": target_domain,
        "training_sessions": cfg.training_sessions,
        "training_steps": cfg.training_steps,
        "training_total": len(training_results),
        "training_ok": len(ok_rows),
        "training_errors": err_rows,
        "scheduled_latency_s": _stats(latencies),
        "scheduled_queue_wait_s": _stats(queue_waits),
        "scheduled_exec_to_done_s": _stats(exec_to_done),
        "by_session": by_session,
        "legacy": {
            "submitted": legacy.submitted,
            "completed": legacy.completed,
            "errors": legacy.errors,
            "max_outstanding": legacy.max_outstanding,
            "final_outstanding": len(legacy.outstanding),
            "latency_s": _stats(
                [float(rec["elapsed_s"]) for rec in legacy.records if isinstance(rec.get("elapsed_s"), (int, float))]
            ),
        },
        "arbitration_delta_total": int((after_wq or {}).get("scheduler_arbitration_total", 0)) - int((before_wq or {}).get("scheduler_arbitration_total", 0)),
        "arbitration_delta_by_winner": _dict_delta(
            (after_wq or {}).get("scheduler_arbitration_by_winner", {}) if isinstance(after_wq, dict) else {},
            (before_wq or {}).get("scheduler_arbitration_by_winner", {}) if isinstance(before_wq, dict) else {},
        ),
        "arbitration_delta_by_reason": _dict_delta(
            (after_wq or {}).get("scheduler_arbitration_by_reason", {}) if isinstance(after_wq, dict) else {},
            (before_wq or {}).get("scheduler_arbitration_by_reason", {}) if isinstance(before_wq, dict) else {},
        ),
        "max_sampled_depth": max_depth,
        "max_sampled_legacy_streak": max_legacy_streak,
        "domain_peak": _domain_peak(sample_rows, target_domain),
    }


def _summary_md(summary: dict[str, Any], *, cfg: Config, paths: dict[str, Path]) -> str:
    lines = [
        "# Issue432 Gate 5 local mixed-load run",
        "",
        f"- label: `{cfg.label}`",
        f"- base_url: `{cfg.base_url}`",
        f"- base_model: `{cfg.base_model}`",
        f"- target_scheduler_domain: `{summary.get('target_scheduler_domain')}`",
        f"- training_ok: `{summary.get('training_ok')}` / `{summary.get('training_total')}`",
        f"- legacy_submitted: `{summary.get('legacy', {}).get('submitted')}`",
        f"- legacy_completed: `{summary.get('legacy', {}).get('completed')}`",
        f"- arbitration_delta_total: `{summary.get('arbitration_delta_total')}`",
        f"- arbitration_delta_by_winner: `{summary.get('arbitration_delta_by_winner')}`",
        f"- arbitration_delta_by_reason: `{summary.get('arbitration_delta_by_reason')}`",
        f"- scheduled_latency_s: `{summary.get('scheduled_latency_s')}`",
        f"- scheduled_queue_wait_s: `{summary.get('scheduled_queue_wait_s')}`",
        f"- domain_peak: `{summary.get('domain_peak')}`",
        "",
        "## Artifacts",
    ]
    for key, path in paths.items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--base-model", default=os.environ.get("MINT_ISSUE432_MODEL", DEFAULT_MODEL))
    p.add_argument("--lora-rank", type=int, default=int(os.environ.get("MINT_ISSUE432_LORA_RANK", "8")))
    p.add_argument("--training-sessions", type=int, default=int(os.environ.get("MINT_ISSUE432_TRAINING_SESSIONS", "2")))
    p.add_argument("--training-steps", type=int, default=int(os.environ.get("MINT_ISSUE432_TRAINING_STEPS", "6")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("MINT_ISSUE432_BATCH_SIZE", "2")))
    p.add_argument("--seq-len", type=int, default=int(os.environ.get("MINT_ISSUE432_SEQ_LEN", "256")))
    p.add_argument("--legacy-target-outstanding", type=int, default=int(os.environ.get("MINT_ISSUE432_LEGACY_TARGET_OUTSTANDING", "128")))
    p.add_argument("--legacy-poll-batch", type=int, default=int(os.environ.get("MINT_ISSUE432_LEGACY_POLL_BATCH", "32")))
    p.add_argument("--legacy-warmup-s", type=float, default=float(os.environ.get("MINT_ISSUE432_LEGACY_WARMUP_S", "1.0")))
    p.add_argument("--snapshot-interval-s", type=float, default=float(os.environ.get("MINT_ISSUE432_SNAPSHOT_INTERVAL_S", "0.25")))
    p.add_argument("--timeout-s", type=float, default=float(os.environ.get("MINT_ISSUE432_TIMEOUT_S", "1800")))
    p.add_argument("--poll-interval-s", type=float, default=float(os.environ.get("MINT_ISSUE432_POLL_INTERVAL_S", "0.1")))
    p.add_argument("--label", default=os.environ.get("MINT_ISSUE432_LABEL", f"issue432-{_ts_dir()}"))
    p.add_argument("--output-dir", default=os.environ.get("MINT_ISSUE432_OUTPUT_DIR", "cover/issue432"))
    p.add_argument("--raw-root", default=os.environ.get("MINT_ISSUE432_RAW_ROOT", "/tmp/issue432-mixed"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Config(
        base_url=str(args.base_url).rstrip("/"),
        headers=({"X-API-Key": str(os.environ.get("MINT_API_KEY", "")).strip()} if str(os.environ.get("MINT_API_KEY", "")).strip() else {}),
        base_model=str(args.base_model),
        lora_rank=max(1, int(args.lora_rank)),
        training_sessions=max(1, int(args.training_sessions)),
        training_steps=max(1, int(args.training_steps)),
        batch_size=max(1, int(args.batch_size)),
        seq_len=max(4, int(args.seq_len)),
        legacy_target_outstanding=max(1, int(args.legacy_target_outstanding)),
        legacy_poll_batch=max(1, int(args.legacy_poll_batch)),
        legacy_warmup_s=max(0.0, float(args.legacy_warmup_s)),
        snapshot_interval_s=max(0.05, float(args.snapshot_interval_s)),
        timeout_s=max(30.0, float(args.timeout_s)),
        poll_interval_s=max(0.05, float(args.poll_interval_s)),
        output_dir=Path(args.output_dir),
        raw_dir=Path(args.raw_root) / str(args.label),
        label=str(args.label),
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    events_path = cfg.raw_dir / "events.jsonl"
    legacy_path = cfg.raw_dir / "legacy.jsonl"
    samples_path = cfg.raw_dir / "samples.jsonl"

    session_infos: list[dict[str, str]] = []
    stop_legacy = threading.Event()
    stop_sampler = threading.Event()
    legacy = LegacyController(cfg=cfg, stop_event=stop_legacy, out_path=legacy_path)
    sampler = SnapshotSampler(cfg=cfg, stop_event=stop_sampler, out_path=samples_path)
    before_admission: dict[str, Any] = {}
    after_admission: dict[str, Any] = {}

    with requests.Session() as session:
        server_info = _get_json(session, f"{cfg.base_url}/api/v1/server_info", headers=cfg.headers, timeout_s=cfg.timeout_s)
        _write_json(
            cfg.raw_dir / "meta.json",
            {
                "ts": _now_iso(),
                "label": cfg.label,
                "config": {
                    **cfg.__dict__,
                    "output_dir": str(cfg.output_dir),
                    "raw_dir": str(cfg.raw_dir),
                },
                "server_info": server_info,
            },
        )

        try:
            for idx in range(cfg.training_sessions):
                session_id = f"{cfg.label}-sess-{idx}-{uuid.uuid4().hex[:8]}"
                model_id = _create_model(session, cfg=cfg, session_id=session_id, event_path=events_path)
                session_infos.append({"session_id": session_id, "model_id": model_id})

            before_admission = _get_json(session, f"{cfg.base_url}/internal/admission_stats", headers=cfg.headers, timeout_s=cfg.timeout_s)
            before_debug = _get_json(session, f"{cfg.base_url}/internal/work_queue/debug_state", headers=cfg.headers, timeout_s=cfg.timeout_s)
            before_metrics = _get_text(session, f"{cfg.base_url}/internal/metrics", headers=cfg.headers, timeout_s=cfg.timeout_s)
            _write_json(cfg.raw_dir / "before.admission.json", before_admission)
            _write_json(cfg.raw_dir / "before.debug.json", before_debug)
            _write_text(cfg.raw_dir / "before.metrics.prom", before_metrics)

            sampler_thread = threading.Thread(target=sampler.run, name="issue432-sampler", daemon=True)
            legacy_thread = threading.Thread(target=legacy.run, name="issue432-legacy", daemon=True)
            sampler_thread.start()
            legacy_thread.start()
            if cfg.legacy_warmup_s > 0.0:
                time.sleep(cfg.legacy_warmup_s)

            jobs = _build_jobs(session_infos, training_steps=cfg.training_steps)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                results = list(pool.map(lambda job: _run_training_job(cfg, job, events_path), jobs))

            stop_legacy.set()
            legacy_thread.join(timeout=cfg.timeout_s)
            time.sleep(cfg.snapshot_interval_s)
            stop_sampler.set()
            sampler_thread.join(timeout=cfg.timeout_s)

            after_admission = _get_json(session, f"{cfg.base_url}/internal/admission_stats", headers=cfg.headers, timeout_s=cfg.timeout_s)
            after_debug = _get_json(session, f"{cfg.base_url}/internal/work_queue/debug_state", headers=cfg.headers, timeout_s=cfg.timeout_s)
            after_metrics = _get_text(session, f"{cfg.base_url}/internal/metrics", headers=cfg.headers, timeout_s=cfg.timeout_s)
            _write_json(cfg.raw_dir / "after.admission.json", after_admission)
            _write_json(cfg.raw_dir / "after.debug.json", after_debug)
            _write_text(cfg.raw_dir / "after.metrics.prom", after_metrics)

            _write_json(cfg.raw_dir / "training.results.json", results)
            sample_rows = _load_jsonl(samples_path)
            summary = _build_summary(
                cfg=cfg,
                before_admission=before_admission,
                after_admission=after_admission,
                training_results=results,
                legacy=legacy,
                sample_rows=sample_rows,
            )
            summary_path = cfg.output_dir / f"{cfg.label}-summary.json"
            md_path = cfg.output_dir / f"{cfg.label}-summary.md"
            paths = {
                "summary_json": summary_path,
                "summary_md": md_path,
                "events_jsonl": events_path,
                "legacy_jsonl": legacy_path,
                "samples_jsonl": samples_path,
                "before_admission": cfg.raw_dir / "before.admission.json",
                "after_admission": cfg.raw_dir / "after.admission.json",
                "before_debug": cfg.raw_dir / "before.debug.json",
                "after_debug": cfg.raw_dir / "after.debug.json",
                "before_metrics": cfg.raw_dir / "before.metrics.prom",
                "after_metrics": cfg.raw_dir / "after.metrics.prom",
                "training_results": cfg.raw_dir / "training.results.json",
            }
            _write_json(summary_path, summary)
            _write_text(md_path, _summary_md(summary, cfg=cfg, paths=paths))

            print(json.dumps({"label": cfg.label, "summary_json": str(summary_path), "summary_md": str(md_path), "raw_dir": str(cfg.raw_dir)}, ensure_ascii=True, sort_keys=True))
            print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        finally:
            stop_legacy.set()
            stop_sampler.set()
            for info in session_infos:
                model_id = str(info.get("model_id") or "")
                if model_id:
                    _delete_model(session, cfg=cfg, model_id=model_id)


if __name__ == "__main__":
    main()
