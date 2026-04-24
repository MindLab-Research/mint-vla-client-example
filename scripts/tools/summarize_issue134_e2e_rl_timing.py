#!/usr/bin/env python3
"""Summarize Issue #134 E2E RL timing runs into a standardized report.

Inputs:
- Manifest JSONL from scripts/tools/bench_issue134_e2e_rl_timing.py
- Optional server log file(s) for trace_id/request_id-based queue timing
- Optional OTEL span export file(s)

Outputs:
- normalized_stage_rows.csv
- stage_summary.csv
- issue-134-repo.md
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CLIENT_MAJOR_STAGES = [
    "save_weights_and_get_sampling_client",
    "compute_logprobs",
    "rollout_sample",
    "forward_backward",
    "optim_step",
]


_EXEC_START_RE = re.compile(r"\[api_work_queue\] executing request_id=(?P<rid>\S+) op=(?P<op>\S+)")
_EXEC_DONE_RE = re.compile(r"\[api_work_queue\] executor completed request_id=(?P<rid>\S+) op=(?P<op>\S+)")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_ts(ts: str | None) -> float | None:
    if not ts or not isinstance(ts, str):
        return None
    s = ts.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    if p <= 0:
        return float(min(xs))
    if p >= 100:
        return float(max(xs))
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def _fmt(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return "nan"
    return f"{x:.3f}"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = _iter_jsonl(path)
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r.get("run_id"), str):
            continue
        out.append(r)
    return out


def _load_client_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in manifest_rows:
        run_id = str(run["run_id"])
        trace_id = str(run.get("trace_id") or "")
        model = str(run.get("model") or "")
        num_sessions = int(run.get("num_sessions") or 1)
        plp = int(run.get("prompt_logprobs") or 0)
        clp = int(run.get("compute_logprobs") or 0)
        repeat = int(run.get("repeat") or 0)
        rc = int(run.get("rc") or 0)
        jsonl_paths = run.get("jsonl_paths")
        if not isinstance(jsonl_paths, list):
            continue

        step_acc: dict[tuple[int, int], float] = defaultdict(float)
        for p in jsonl_paths:
            jp = Path(str(p))
            for rec in _iter_jsonl(jp):
                stage = rec.get("stage")
                elapsed = rec.get("elapsed_s")
                if not isinstance(stage, str) or not isinstance(elapsed, (int, float)):
                    continue
                session_idx = int(rec.get("session_idx") or 0)
                step_idx = int(rec.get("step_idx") or 0)
                row = {
                    "source": "client",
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "model": model,
                    "num_sessions": num_sessions,
                    "prompt_logprobs": plp,
                    "compute_logprobs": clp,
                    "repeat": repeat,
                    "rc": rc,
                    "session_idx": session_idx,
                    "step_idx": step_idx,
                    "stage": stage,
                    "elapsed_s": float(elapsed),
                    "request_id": "",
                    "op": "",
                    "ts": str(rec.get("ts") or ""),
                }
                rows.append(row)
                if stage in CLIENT_MAJOR_STAGES and step_idx >= 0:
                    step_acc[(session_idx, step_idx)] += float(elapsed)

        for (session_idx, step_idx), total_s in sorted(step_acc.items()):
            rows.append(
                {
                    "source": "client",
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "model": model,
                    "num_sessions": num_sessions,
                    "prompt_logprobs": plp,
                    "compute_logprobs": clp,
                    "repeat": repeat,
                    "rc": rc,
                    "session_idx": int(session_idx),
                    "step_idx": int(step_idx),
                    "stage": "step_total",
                    "elapsed_s": float(total_s),
                    "request_id": "",
                    "op": "",
                    "ts": "",
                }
            )
    return rows


def _load_server_log_rows(
    *,
    server_logs: list[Path],
    trace_ids: set[str],
    run_by_trace: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    starts: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for path in server_logs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            payload: dict[str, Any] | None = None
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    payload = obj
            except Exception:
                payload = None

            if payload is None:
                continue
            trace_id = payload.get("trace_id")
            if not isinstance(trace_id, str) or trace_id not in trace_ids:
                continue
            event = payload.get("event")
            if not isinstance(event, str):
                continue
            ts = _parse_ts(payload.get("timestamp"))
            if ts is None:
                continue

            m = _EXEC_START_RE.search(event)
            if m:
                key = (trace_id, m.group("rid"), m.group("op"))
                starts[key].append(ts)
                continue

            m = _EXEC_DONE_RE.search(event)
            if m:
                key = (trace_id, m.group("rid"), m.group("op"))
                if not starts[key]:
                    continue
                t0 = starts[key].pop(0)
                elapsed_s = max(0.0, ts - t0)
                run = run_by_trace.get(trace_id)
                if run is None:
                    continue
                rows.append(
                    {
                        "source": "server_log",
                        "run_id": str(run["run_id"]),
                        "trace_id": trace_id,
                        "model": str(run.get("model") or ""),
                        "num_sessions": int(run.get("num_sessions") or 1),
                        "prompt_logprobs": int(run.get("prompt_logprobs") or 0),
                        "compute_logprobs": int(run.get("compute_logprobs") or 0),
                        "repeat": int(run.get("repeat") or 0),
                        "rc": int(run.get("rc") or 0),
                        "session_idx": -1,
                        "step_idx": -1,
                        "stage": f"queue_execute:{m.group('op')}",
                        "elapsed_s": float(elapsed_s),
                        "request_id": m.group("rid"),
                        "op": m.group("op"),
                        "ts": "",
                    }
                )
    return rows


def _otlp_attrs_to_map(attrs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(attrs, dict):
        return attrs
    if not isinstance(attrs, list):
        return out
    for item in attrs:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value_obj = item.get("value")
        if not isinstance(key, str):
            continue
        if isinstance(value_obj, dict):
            for vk in ("stringValue", "intValue", "doubleValue", "boolValue"):
                if vk in value_obj:
                    out[key] = value_obj[vk]
                    break
    return out


def _extract_spans_from_obj(obj: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(obj, dict):
        return out
    if "resourceSpans" in obj and isinstance(obj["resourceSpans"], list):
        for rs in obj["resourceSpans"]:
            if not isinstance(rs, dict):
                continue
            for ss in rs.get("scopeSpans", []) or []:
                if not isinstance(ss, dict):
                    continue
                for sp in ss.get("spans", []) or []:
                    if isinstance(sp, dict):
                        out.append(sp)
        return out
    # JSONL span-like line fallback
    if any(k in obj for k in ("traceId", "startTimeUnixNano", "endTimeUnixNano", "name")):
        out.append(obj)
    return out


def _load_otel_rows(
    *,
    otel_files: list[Path],
    trace_ids: set[str],
    run_by_trace: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in otel_files:
        if not path.exists():
            continue
        txt = path.read_text(encoding="utf-8", errors="replace")
        span_objs: list[dict[str, Any]] = []
        try:
            parsed = json.loads(txt)
            span_objs.extend(_extract_spans_from_obj(parsed))
        except Exception:
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                span_objs.extend(_extract_spans_from_obj(parsed))

        for sp in span_objs:
            trace_id = sp.get("traceId") or sp.get("trace_id")
            if not isinstance(trace_id, str):
                continue
            trace_id = trace_id.strip().lower()
            if trace_id not in trace_ids:
                continue
            name = str(sp.get("name") or "unknown")
            start_ns = sp.get("startTimeUnixNano") or sp.get("start_time_unix_nano")
            end_ns = sp.get("endTimeUnixNano") or sp.get("end_time_unix_nano")
            try:
                elapsed_s = max(0.0, (int(end_ns) - int(start_ns)) / 1_000_000_000.0)
            except Exception:
                continue
            attrs = _otlp_attrs_to_map(sp.get("attributes"))
            request_id = str(attrs.get("request_id") or "")
            op = str(attrs.get("op") or "")
            run = run_by_trace.get(trace_id)
            if run is None:
                continue
            rows.append(
                {
                    "source": "otel_span",
                    "run_id": str(run["run_id"]),
                    "trace_id": trace_id,
                    "model": str(run.get("model") or ""),
                    "num_sessions": int(run.get("num_sessions") or 1),
                    "prompt_logprobs": int(run.get("prompt_logprobs") or 0),
                    "compute_logprobs": int(run.get("compute_logprobs") or 0),
                    "repeat": int(run.get("repeat") or 0),
                    "rc": int(run.get("rc") or 0),
                    "session_idx": -1,
                    "step_idx": -1,
                    "stage": f"span:{name}",
                    "elapsed_s": float(elapsed_s),
                    "request_id": request_id,
                    "op": op,
                    "ts": "",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True, help="manifest.jsonl from bench_issue134_e2e_rl_timing.py")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--server-log", type=Path, nargs="*", default=[])
    p.add_argument("--otel-trace", type=Path, nargs="*", default=[])
    p.add_argument("--report-path", type=Path, default=None, help="Override markdown report output path")
    args = p.parse_args()

    manifest_rows = _load_manifest(args.manifest)
    if not manifest_rows:
        raise SystemExit(f"empty or invalid manifest: {args.manifest}")

    out_dir = args.out_dir or (args.manifest.parent / "summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or (out_dir / "issue-134-repo.md")

    client_rows = _load_client_rows(manifest_rows)
    trace_ids = {str(r.get("trace_id") or "").strip().lower() for r in manifest_rows if str(r.get("trace_id") or "").strip()}
    run_by_trace = {
        str(r.get("trace_id") or "").strip().lower(): r
        for r in manifest_rows
        if str(r.get("trace_id") or "").strip()
    }

    server_rows = _load_server_log_rows(
        server_logs=[Path(p) for p in args.server_log],
        trace_ids=trace_ids,
        run_by_trace=run_by_trace,
    )
    otel_rows = _load_otel_rows(
        otel_files=[Path(p) for p in args.otel_trace],
        trace_ids=trace_ids,
        run_by_trace=run_by_trace,
    )

    all_rows = client_rows + server_rows + otel_rows
    fieldnames = [
        "source",
        "run_id",
        "trace_id",
        "model",
        "num_sessions",
        "prompt_logprobs",
        "compute_logprobs",
        "repeat",
        "rc",
        "session_idx",
        "step_idx",
        "stage",
        "elapsed_s",
        "request_id",
        "op",
        "ts",
    ]
    _write_csv(out_dir / "normalized_stage_rows.csv", all_rows, fieldnames)

    by_group: dict[tuple[str, str, int, int, int, str], list[float]] = defaultdict(list)
    for r in all_rows:
        key = (
            str(r.get("source") or ""),
            str(r.get("model") or ""),
            int(r.get("prompt_logprobs") or 0),
            int(r.get("compute_logprobs") or 0),
            int(r.get("num_sessions") or 1),
            str(r.get("stage") or ""),
        )
        by_group[key].append(float(r["elapsed_s"]))

    summary_rows: list[dict[str, Any]] = []
    for (source, model, plp, clp, c, stage), xs in sorted(by_group.items()):
        summary_rows.append(
            {
                "source": source,
                "model": model,
                "prompt_logprobs": plp,
                "compute_logprobs": clp,
                "num_sessions": c,
                "stage": stage,
                "count": len(xs),
                "p50_s": _pctl(xs, 50),
                "p90_s": _pctl(xs, 90),
                "mean_s": (sum(xs) / len(xs)) if xs else float("nan"),
                "max_s": max(xs) if xs else float("nan"),
            }
        )
    _write_csv(
        out_dir / "stage_summary.csv",
        summary_rows,
        [
            "source",
            "model",
            "prompt_logprobs",
            "compute_logprobs",
            "num_sessions",
            "stage",
            "count",
            "p50_s",
            "p90_s",
            "mean_s",
            "max_s",
        ],
    )

    # Build bottleneck share using client rows only.
    client_summary_map: dict[tuple[str, int, int, int, str], float] = {}
    for r in summary_rows:
        if r["source"] != "client":
            continue
        k = (
            str(r["model"]),
            int(r["prompt_logprobs"]),
            int(r["compute_logprobs"]),
            int(r["num_sessions"]),
            str(r["stage"]),
        )
        client_summary_map[k] = float(r["mean_s"])

    share_rows: list[dict[str, Any]] = []
    models = sorted({str(r.get("model") or "") for r in manifest_rows})
    for model in models:
        for plp in (0, 1):
            for clp in (0, 1):
                for c in sorted({int(r.get("num_sessions") or 1) for r in manifest_rows if str(r.get("model") or "") == model}):
                    step_total = client_summary_map.get((model, plp, clp, c, "step_total"))
                    if step_total is None or step_total <= 0:
                        continue
                    for st in CLIENT_MAJOR_STAGES:
                        v = client_summary_map.get((model, plp, clp, c, st))
                        if v is None:
                            continue
                        share_rows.append(
                            {
                                "model": model,
                                "prompt_logprobs": plp,
                                "compute_logprobs": clp,
                                "num_sessions": c,
                                "stage": st,
                                "mean_s": v,
                                "step_total_mean_s": step_total,
                                "share": (v / step_total),
                            }
                        )

    # Markdown report
    ok_jobs = sum(1 for r in manifest_rows if int(r.get("rc") or 0) == 0)
    fail_jobs = len(manifest_rows) - ok_jobs
    md: list[str] = []
    md.append("# issue-134-repo")
    md.append("")
    md.append(f"- generated_at: {_now_iso()}")
    md.append(f"- manifest: {args.manifest}")
    md.append(f"- jobs_total: {len(manifest_rows)}")
    md.append(f"- jobs_ok: {ok_jobs}")
    md.append(f"- jobs_failed: {fail_jobs}")
    md.append(f"- normalized_csv: {out_dir / 'normalized_stage_rows.csv'}")
    md.append(f"- summary_csv: {out_dir / 'stage_summary.csv'}")
    md.append(f"- server_log_rows: {len(server_rows)}")
    md.append(f"- otel_span_rows: {len(otel_rows)}")
    md.append("")

    md.append("## Client Stage P50/P90")
    md.append("")
    md.append("| model | c | plp | clp | stage | count | p50_s | p90_s | mean_s |")
    md.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for r in summary_rows:
        if r["source"] != "client":
            continue
        md.append(
            "| {model} | {c} | {plp} | {clp} | {stage} | {count} | {p50} | {p90} | {mean} |".format(
                model=r["model"],
                c=int(r["num_sessions"]),
                plp=int(r["prompt_logprobs"]),
                clp=int(r["compute_logprobs"]),
                stage=r["stage"],
                count=int(r["count"]),
                p50=_fmt(float(r["p50_s"])),
                p90=_fmt(float(r["p90_s"])),
                mean=_fmt(float(r["mean_s"])),
            )
        )
    md.append("")

    md.append("## Bottleneck Share (Client Mean)")
    md.append("")
    md.append("| model | c | plp | clp | stage | mean_s | step_total_mean_s | share |")
    md.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for r in sorted(
        share_rows,
        key=lambda x: (
            str(x["model"]),
            int(x["num_sessions"]),
            int(x["prompt_logprobs"]),
            int(x["compute_logprobs"]),
            -float(x["share"]),
        ),
    ):
        md.append(
            "| {model} | {c} | {plp} | {clp} | {stage} | {mean} | {step_total} | {share} |".format(
                model=r["model"],
                c=int(r["num_sessions"]),
                plp=int(r["prompt_logprobs"]),
                clp=int(r["compute_logprobs"]),
                stage=r["stage"],
                mean=_fmt(float(r["mean_s"])),
                step_total=_fmt(float(r["step_total_mean_s"])),
                share=_fmt(float(r["share"])),
            )
        )
    md.append("")

    non_client = [r for r in summary_rows if r["source"] != "client"]
    if non_client:
        md.append("## Trace/Queue Timing (Server/Otel)")
        md.append("")
        md.append("| source | model | plp | clp | stage | count | p50_s | p90_s |")
        md.append("|---|---|---:|---:|---|---:|---:|---:|")
        for r in non_client:
            md.append(
                "| {source} | {model} | {plp} | {clp} | {stage} | {count} | {p50} | {p90} |".format(
                    source=r["source"],
                    model=r["model"],
                    plp=int(r["prompt_logprobs"]),
                    clp=int(r["compute_logprobs"]),
                    stage=r["stage"],
                    count=int(r["count"]),
                    p50=_fmt(float(r["p50_s"])),
                    p90=_fmt(float(r["p90_s"])),
                )
            )
        md.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(str(report_path), flush=True)


if __name__ == "__main__":
    main()
