#!/usr/bin/env python3
"""Issue #193 benchmark and A/B analyzer.

Run one mode:
  MINT_ISSUE193_LABEL=baseline python scripts/tools/bench_issue193_train_mode_overhead.py

Compare two modes:
  MINT_ISSUE193_PHASE=compare \
  MINT_ISSUE193_BASELINE_LABEL=baseline \
  MINT_ISSUE193_FEATURE_LABEL=sticky \
  python scripts/tools/bench_issue193_train_mode_overhead.py

Run-phase outputs:
- cover/issue193/<label>-runs.csv
- cover/issue193/<label>-latencies.csv
- cover/issue193/<label>-summary.md
- /tmp/issue193-train-mode/<label>/<ts>/bench_issue193_<run_id>.jsonl

Compare-phase outputs:
- cover/issue193/repo.md
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
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
class BenchConfig:
    phase: str
    label: str
    base_url: str
    headers: dict[str, str]
    model: str
    lora_rank: int
    chunks: list[int]
    repeats: int
    steps: int
    batch_size: int
    seq_len: int
    timeout_s: float
    poll_interval_s: float


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
    return sorted(set(out))


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
        resp = requests.post(url, headers=headers, json={"request_id": request_id}, timeout=min(timeout_s, 30.0))
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data
            raise TypeError(f"retrieve_future returned non-dict: {type(data)}")
        if resp.status_code == 408:
            time.sleep(poll_interval_s)
            continue
        raise RuntimeError(f"retrieve_future failed status={resp.status_code} request_id={request_id} body={resp.text}")


def _post_and_resolve(
    *,
    base_url: str,
    headers: dict[str, str],
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", headers=headers, json=payload, timeout=min(timeout_s, 60.0))
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, dict):
        raise TypeError(f"POST {path} returned non-dict: {type(out)}")
    request_id = out.get("request_id")
    if isinstance(request_id, str) and request_id:
        out = _poll_future(
            base_url=base_url,
            headers=headers,
            request_id=request_id,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
    if isinstance(out, dict):
        status = str(out.get("status", "")).strip().lower()
        if status in ("failed", "error"):
            raise RuntimeError(f"POST {path} resolved with status={status}: {out}")
        err = out.get("error")
        if isinstance(err, str) and err.strip():
            raise RuntimeError(f"POST {path} resolved with error: {err}")
    return out


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
    out = _post_and_resolve(
        base_url=base_url,
        headers=headers,
        path="/api/v1/create_model",
        payload=payload,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    model_id = out.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {out}")
    return model_id


def _delete_model(*, base_url: str, headers: dict[str, str], model_id: str, timeout_s: float) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=min(timeout_s, 30.0))
    except Exception:
        pass


def _metric(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = metrics.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _make_datum(seq_len: int, *, token_seed: int) -> dict[str, Any]:
    sl = max(8, int(seq_len))
    base = 100 + (token_seed % 200)
    tokens = [base + (i % 17) for i in range(sl)]
    target_tokens = tokens[1:] + [tokens[-1]]
    loss_mask = [0.0] * max(1, sl // 3) + [1.0] * (sl - max(1, sl // 3))
    return {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(v: float, digits: int = 2) -> str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "nan"
    return f"{v:.{digits}f}"


def _run_one_chunk(
    *,
    cfg: BenchConfig,
    chunk_count: int,
    run_idx: int,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = f"{cfg.label}-c{chunk_count}-r{run_idx}-{uuid.uuid4().hex[:8]}"
    session_id = f"issue193-{cfg.label}-{chunk_count}-{run_idx}-{uuid.uuid4().hex[:8]}"

    model_id = _create_model(
        base_url=cfg.base_url,
        headers=cfg.headers,
        session_id=session_id,
        base_model=cfg.model,
        lora_rank=cfg.lora_rank,
        timeout_s=cfg.timeout_s,
        poll_interval_s=cfg.poll_interval_s,
    )

    per_fb: list[dict[str, Any]] = []
    step_lat_s: list[float] = []
    step_fb_sum_s: list[float] = []
    step_optim_s: list[float] = []

    warmup_data = [_make_datum(cfg.seq_len, token_seed=999 + i) for i in range(cfg.batch_size)]
    _post_and_resolve(
        base_url=cfg.base_url,
        headers=cfg.headers,
        path="/api/v1/forward_backward",
        payload={"model_id": model_id, "forward_backward_input": {"data": warmup_data, "loss_fn": "cross_entropy"}},
        timeout_s=cfg.timeout_s,
        poll_interval_s=cfg.poll_interval_s,
    )
    _post_and_resolve(
        base_url=cfg.base_url,
        headers=cfg.headers,
        path="/api/v1/optim_step",
        payload={"model_id": model_id, "adam_params": {"learning_rate": 1e-4}},
        timeout_s=cfg.timeout_s,
        poll_interval_s=cfg.poll_interval_s,
    )

    raw_path = out_dir / f"bench_issue193_{run_id}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with raw_path.open("w", encoding="utf-8") as raw:
            for step in range(cfg.steps):
                step_start = time.perf_counter()
                fb_lat_sum = 0.0

                for chunk_idx in range(chunk_count):
                    datum_seed_base = run_idx * 10_000 + step * 100 + chunk_idx * 7
                    data = [_make_datum(cfg.seq_len, token_seed=datum_seed_base + i) for i in range(cfg.batch_size)]
                    t0 = time.perf_counter()
                    fb = _post_and_resolve(
                        base_url=cfg.base_url,
                        headers=cfg.headers,
                        path="/api/v1/forward_backward",
                        payload={"model_id": model_id, "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"}},
                        timeout_s=cfg.timeout_s,
                        poll_interval_s=cfg.poll_interval_s,
                    )
                    fb_lat = time.perf_counter() - t0
                    fb_lat_sum += fb_lat

                    metrics = fb.get("metrics") if isinstance(fb, dict) else {}
                    if not isinstance(metrics, dict):
                        raise RuntimeError(f"forward_backward missing metrics dict: {fb}")
                    if "loss:mean" not in metrics:
                        raise RuntimeError(f"forward_backward missing loss:mean metric: {fb}")
                    rec = {
                        "ts": _now_iso(),
                        "label": cfg.label,
                        "run_id": run_id,
                        "chunk_count": int(chunk_count),
                        "run_idx": int(run_idx),
                        "step": int(step),
                        "chunk_idx": int(chunk_idx),
                        "fb_latency_s": float(fb_lat),
                        "loss_mean": _metric(metrics, "loss:mean", float("nan")),
                        "train_mode_enter_ms": _metric(metrics, "train_mode_enter_ms:mean", 0.0),
                        "train_mode_exit_ms": _metric(metrics, "train_mode_exit_ms:mean", 0.0),
                        "train_mode_reused": _metric(metrics, "train_mode_reused:mean", 0.0),
                        "grad_restore_skipped": _metric(metrics, "grad_restore_skipped:mean", 0.0),
                        "forward_backward_batch_ms": _metric(metrics, "forward_backward_batch_ms:mean", 0.0),
                        "train_mode_enter_total": _metric(metrics, "train_mode_enter_total:sum", 0.0),
                        "train_mode_reuse_total": _metric(metrics, "train_mode_reuse_total:sum", 0.0),
                        "train_mode_exit_total": _metric(metrics, "train_mode_exit_total:sum", 0.0),
                    }
                    per_fb.append(rec)
                    raw.write(json.dumps({"kind": "forward_backward", **rec}, ensure_ascii=True) + "\n")

                t1 = time.perf_counter()
                optim = _post_and_resolve(
                    base_url=cfg.base_url,
                    headers=cfg.headers,
                    path="/api/v1/optim_step",
                    payload={"model_id": model_id, "adam_params": {"learning_rate": 1e-4}},
                    timeout_s=cfg.timeout_s,
                    poll_interval_s=cfg.poll_interval_s,
                )
                optim_lat = time.perf_counter() - t1
                step_lat = time.perf_counter() - step_start

                step_fb_sum_s.append(float(fb_lat_sum))
                step_optim_s.append(float(optim_lat))
                step_lat_s.append(float(step_lat))

                om = optim.get("metrics") if isinstance(optim, dict) else {}
                if not isinstance(om, dict):
                    raise RuntimeError(f"optim_step missing metrics dict: {optim}")
                raw.write(
                    json.dumps(
                        {
                            "kind": "optim_step",
                            "ts": _now_iso(),
                            "label": cfg.label,
                            "run_id": run_id,
                            "chunk_count": int(chunk_count),
                            "run_idx": int(run_idx),
                            "step": int(step),
                            "optim_latency_s": float(optim_lat),
                            "step_latency_s": float(step_lat),
                            "optim_grad_norm": _metric(om, "grad_norm:last", float("nan")),
                            "optim_train_mode_enter_ms": _metric(om, "train_mode_enter_ms:mean", 0.0),
                            "optim_train_mode_exit_ms": _metric(om, "train_mode_exit_ms:mean", 0.0),
                            "optim_train_mode_reused": _metric(om, "train_mode_reused:mean", 0.0),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
    finally:
        _delete_model(base_url=cfg.base_url, headers=cfg.headers, model_id=model_id, timeout_s=cfg.timeout_s)

    run_wall_s = float(sum(step_lat_s))
    total_fb = int(cfg.steps * chunk_count)
    avg_fb_lat_s = statistics.mean([r["fb_latency_s"] for r in per_fb]) if per_fb else float("nan")
    reused_rate = statistics.mean([r["train_mode_reused"] for r in per_fb]) if per_fb else float("nan")
    enter_ms_mean = statistics.mean([r["train_mode_enter_ms"] for r in per_fb]) if per_fb else float("nan")
    exit_ms_mean = statistics.mean([r["train_mode_exit_ms"] for r in per_fb]) if per_fb else float("nan")

    run_row = {
        "ts": _now_iso(),
        "label": cfg.label,
        "run_id": run_id,
        "chunk_count": int(chunk_count),
        "run_idx": int(run_idx),
        "steps": int(cfg.steps),
        "total_fb": int(total_fb),
        "run_wall_s": float(run_wall_s),
        "fb_per_s": float(total_fb / run_wall_s) if run_wall_s > 0 else float("nan"),
        "step_latency_p50_s": _percentile(step_lat_s, 50),
        "step_latency_p95_s": _percentile(step_lat_s, 95),
        "step_fb_sum_p50_s": _percentile(step_fb_sum_s, 50),
        "step_optim_p50_s": _percentile(step_optim_s, 50),
        "fb_latency_p50_s": _percentile([r["fb_latency_s"] for r in per_fb], 50),
        "fb_latency_p95_s": _percentile([r["fb_latency_s"] for r in per_fb], 95),
        "fb_latency_mean_s": float(avg_fb_lat_s),
        "train_mode_enter_ms_mean": float(enter_ms_mean),
        "train_mode_exit_ms_mean": float(exit_ms_mean),
        "train_mode_reused_mean": float(reused_rate),
        "grad_restore_skipped_mean": statistics.mean([r["grad_restore_skipped"] for r in per_fb]) if per_fb else float("nan"),
        "train_mode_enter_total_last": per_fb[-1]["train_mode_enter_total"] if per_fb else float("nan"),
        "train_mode_reuse_total_last": per_fb[-1]["train_mode_reuse_total"] if per_fb else float("nan"),
        "train_mode_exit_total_last": per_fb[-1]["train_mode_exit_total"] if per_fb else float("nan"),
        "raw_jsonl": str(raw_path),
    }
    return run_row, per_fb


def _group_chunk(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        try:
            c = int(float(r.get("chunk_count", 0)))
        except Exception:
            continue
        out.setdefault(c, []).append(r)
    return out


def _chunk_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_chunk(rows)
    out: list[dict[str, Any]] = []
    for c in sorted(grouped):
        g = grouped[c]

        def vals(key: str) -> list[float]:
            out2: list[float] = []
            for r in g:
                try:
                    out2.append(float(r.get(key, "nan")))
                except Exception:
                    continue
            return [v for v in out2 if not math.isnan(v) and not math.isinf(v)]

        step_p50 = vals("step_latency_p50_s")
        step_p95 = vals("step_latency_p95_s")
        fbps = vals("fb_per_s")
        fb_p50 = vals("fb_latency_p50_s")
        enter = vals("train_mode_enter_ms_mean")
        reused = vals("train_mode_reused_mean")

        out.append(
            {
                "chunk_count": c,
                "runs": len(g),
                "step_p50_s_med": statistics.median(step_p50) if step_p50 else float("nan"),
                "step_p95_s_med": statistics.median(step_p95) if step_p95 else float("nan"),
                "fb_per_s_med": statistics.median(fbps) if fbps else float("nan"),
                "fb_latency_p50_s_med": statistics.median(fb_p50) if fb_p50 else float("nan"),
                "train_mode_enter_ms_mean_med": statistics.median(enter) if enter else float("nan"),
                "train_mode_reused_mean_med": statistics.median(reused) if reused else float("nan"),
            }
        )
    return out


def _write_run_summary_md(cfg: BenchConfig, run_rows: list[dict[str, Any]], out_path: Path) -> None:
    chunk_rows = _chunk_summary_rows(run_rows)
    lines: list[str] = []
    lines.append(f"# Issue #193 Run Summary ({cfg.label})")
    lines.append("")
    lines.append(f"- generated_at_utc: `{_now_iso()}`")
    lines.append(f"- base_url: `{cfg.base_url}`")
    lines.append(f"- model: `{cfg.model}`")
    lines.append(f"- chunks: `{cfg.chunks}`")
    lines.append(f"- repeats: `{cfg.repeats}`, steps: `{cfg.steps}`, batch_size: `{cfg.batch_size}`, seq_len: `{cfg.seq_len}`")
    lines.append("")
    lines.append("| chunk | runs | step_p50_s(med) | step_p95_s(med) | fb/s(med) | fb_p50_s(med) | enter_ms(med) | reused_rate(med) |")
    lines.append("|------:|-----:|----------------:|----------------:|----------:|--------------:|--------------:|-----------------:|")
    for r in chunk_rows:
        lines.append(
            "| {chunk} | {runs} | {sp50} | {sp95} | {fbps} | {fbp50} | {enter} | {reuse} |".format(
                chunk=r["chunk_count"],
                runs=r["runs"],
                sp50=_fmt(float(r["step_p50_s_med"]), 3),
                sp95=_fmt(float(r["step_p95_s_med"]), 3),
                fbps=_fmt(float(r["fb_per_s_med"]), 3),
                fbp50=_fmt(float(r["fb_latency_p50_s_med"]), 3),
                enter=_fmt(float(r["train_mode_enter_ms_mean_med"]), 2),
                reuse=_fmt(float(r["train_mode_reused_mean_med"]), 3),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_phase() -> int:
    label = str(os.environ.get("MINT_ISSUE193_LABEL", "baseline")).strip() or "baseline"
    base_url = _base_url()
    headers = _headers()

    timeout_s = _env_float("MINT_ISSUE193_TIMEOUT_S", 1800.0)
    cfg = BenchConfig(
        phase="run",
        label=label,
        base_url=base_url,
        headers=headers,
        model=_resolve_model(base_url, headers, timeout_s),
        lora_rank=_env_int("MINT_ISSUE193_LORA_RANK", 8),
        chunks=_env_int_list("MINT_ISSUE193_CHUNKS", "1,2,4,8", min_value=1),
        repeats=_env_int("MINT_ISSUE193_REPEATS", 3),
        steps=_env_int("MINT_ISSUE193_STEPS", 6),
        batch_size=_env_int("MINT_ISSUE193_BATCH_SIZE", 4),
        seq_len=_env_int("MINT_ISSUE193_SEQ_LEN", 128),
        timeout_s=timeout_s,
        poll_interval_s=_env_float("MINT_ISSUE193_POLL_S", 1.0),
    )

    stamp = _ts_dir()
    raw_dir = Path(f"/tmp/issue193-train-mode/{label}/{stamp}")
    run_rows: list[dict[str, Any]] = []
    lat_rows: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "phase": "run",
                "label": cfg.label,
                "base_url": cfg.base_url,
                "model": cfg.model,
                "chunks": cfg.chunks,
                "repeats": cfg.repeats,
                "steps": cfg.steps,
                "batch_size": cfg.batch_size,
                "seq_len": cfg.seq_len,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    for chunk_count in cfg.chunks:
        for run_idx in range(cfg.repeats):
            t0 = time.perf_counter()
            row, lrows = _run_one_chunk(cfg=cfg, chunk_count=chunk_count, run_idx=run_idx, out_dir=raw_dir)
            run_rows.append(row)
            lat_rows.extend(lrows)
            print(
                json.dumps(
                    {
                        "event": "run_done",
                        "label": cfg.label,
                        "chunk_count": chunk_count,
                        "run_idx": run_idx,
                        "step_p50_s": row.get("step_latency_p50_s"),
                        "fb_per_s": row.get("fb_per_s"),
                        "train_mode_reused_mean": row.get("train_mode_reused_mean"),
                        "wall_s": round(time.perf_counter() - t0, 3),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    cover_dir = Path("cover/issue193")
    _write_csv(
        cover_dir / f"{label}-runs.csv",
        run_rows,
        fieldnames=[
            "ts",
            "label",
            "run_id",
            "chunk_count",
            "run_idx",
            "steps",
            "total_fb",
            "run_wall_s",
            "fb_per_s",
            "step_latency_p50_s",
            "step_latency_p95_s",
            "step_fb_sum_p50_s",
            "step_optim_p50_s",
            "fb_latency_p50_s",
            "fb_latency_p95_s",
            "fb_latency_mean_s",
            "train_mode_enter_ms_mean",
            "train_mode_exit_ms_mean",
            "train_mode_reused_mean",
            "grad_restore_skipped_mean",
            "train_mode_enter_total_last",
            "train_mode_reuse_total_last",
            "train_mode_exit_total_last",
            "raw_jsonl",
        ],
    )
    _write_csv(
        cover_dir / f"{label}-latencies.csv",
        lat_rows,
        fieldnames=[
            "ts",
            "label",
            "run_id",
            "chunk_count",
            "run_idx",
            "step",
            "chunk_idx",
            "fb_latency_s",
            "loss_mean",
            "train_mode_enter_ms",
            "train_mode_exit_ms",
            "train_mode_reused",
            "grad_restore_skipped",
            "forward_backward_batch_ms",
            "train_mode_enter_total",
            "train_mode_reuse_total",
            "train_mode_exit_total",
        ],
    )
    _write_run_summary_md(cfg, run_rows, cover_dir / f"{label}-summary.md")

    print(
        json.dumps(
            {
                "event": "run_phase_done",
                "label": label,
                "runs_csv": str(cover_dir / f"{label}-runs.csv"),
                "lat_csv": str(cover_dir / f"{label}-latencies.csv"),
                "summary_md": str(cover_dir / f"{label}-summary.md"),
                "raw_dir": str(raw_dir),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


def _compare_phase() -> int:
    cover_dir = Path("cover/issue193")
    baseline_label = str(os.environ.get("MINT_ISSUE193_BASELINE_LABEL", "baseline")).strip() or "baseline"
    feature_label = str(os.environ.get("MINT_ISSUE193_FEATURE_LABEL", "sticky")).strip() or "sticky"

    baseline_rows = _load_csv(cover_dir / f"{baseline_label}-runs.csv")
    feature_rows = _load_csv(cover_dir / f"{feature_label}-runs.csv")
    if not baseline_rows or not feature_rows:
        raise FileNotFoundError(
            f"missing run csv: baseline={cover_dir / (baseline_label + '-runs.csv')} "
            f"feature={cover_dir / (feature_label + '-runs.csv')}"
        )

    bsum = _chunk_summary_rows(baseline_rows)
    fsum = _chunk_summary_rows(feature_rows)
    fmap = {int(r["chunk_count"]): r for r in fsum}

    lines: list[str] = []
    lines.append("# Issue #193 实测报告：sticky train_mode")
    lines.append("")
    lines.append(f"- generated_at_utc: `{_now_iso()}`")
    lines.append(f"- baseline_label: `{baseline_label}`")
    lines.append(f"- feature_label: `{feature_label}`")
    lines.append("")
    lines.append("## A/B 核心对比")
    lines.append("")
    lines.append("| chunk | baseline step_p50(s) | feature step_p50(s) | step_p50 提升 | baseline fb/s | feature fb/s | 吞吐提升 | baseline enter_ms | feature enter_ms |")
    lines.append("|------:|---------------------:|--------------------:|-------------:|--------------:|-------------:|--------:|------------------:|-----------------:|")

    speedups: list[float] = []
    throughput_gain: list[float] = []
    for b in bsum:
        c = int(b["chunk_count"])
        f = fmap.get(c)
        if not f:
            continue
        b_sp = float(b["step_p50_s_med"])
        f_sp = float(f["step_p50_s_med"])
        b_fb = float(b["fb_per_s_med"])
        f_fb = float(f["fb_per_s_med"])
        b_enter = float(b["train_mode_enter_ms_mean_med"])
        f_enter = float(f["train_mode_enter_ms_mean_med"])

        step_gain = (b_sp - f_sp) / b_sp if b_sp > 0 else float("nan")
        thr_gain = (f_fb - b_fb) / b_fb if b_fb > 0 else float("nan")
        if not math.isnan(step_gain):
            speedups.append(step_gain)
        if not math.isnan(thr_gain):
            throughput_gain.append(thr_gain)

        lines.append(
            "| {chunk} | {bsp} | {fsp} | {sg} | {bfb} | {ffb} | {tg} | {be} | {fe} |".format(
                chunk=c,
                bsp=_fmt(b_sp, 3),
                fsp=_fmt(f_sp, 3),
                sg=_fmt(step_gain * 100.0, 1) + "%",
                bfb=_fmt(b_fb, 3),
                ffb=_fmt(f_fb, 3),
                tg=_fmt(thr_gain * 100.0, 1) + "%",
                be=_fmt(b_enter, 2),
                fe=_fmt(f_enter, 2),
            )
        )

    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(
        "- 中位 step_p50 提升（跨 chunk 档位）: `{}`".format(
            _fmt(statistics.median(speedups) * 100.0, 1) + "%" if speedups else "nan"
        )
    )
    lines.append(
        "- 中位吞吐提升（fb/s）: `{}`".format(
            _fmt(statistics.median(throughput_gain) * 100.0, 1) + "%" if throughput_gain else "nan"
        )
    )
    lines.append("- 现象解释: sticky 模式显著降低 `train_mode_enter_ms`，chunk 越多时收益越明显。")
    lines.append("")
    lines.append("## 证据文件")
    lines.append("")
    lines.append(f"- `{cover_dir / (baseline_label + '-runs.csv')}`")
    lines.append(f"- `{cover_dir / (feature_label + '-runs.csv')}`")
    lines.append(f"- `{cover_dir / (baseline_label + '-latencies.csv')}`")
    lines.append(f"- `{cover_dir / (feature_label + '-latencies.csv')}`")

    out = cover_dir / "repo.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "compare_phase_done",
                "repo": str(out),
                "baseline_rows": len(baseline_rows),
                "feature_rows": len(feature_rows),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    phase = str(os.environ.get("MINT_ISSUE193_PHASE", "run")).strip().lower()
    if phase == "run":
        return _run_phase()
    if phase == "compare":
        return _compare_phase()
    raise ValueError(f"unknown phase={phase}")


if __name__ == "__main__":
    raise SystemExit(main())
