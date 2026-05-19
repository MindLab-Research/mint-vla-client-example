#!/usr/bin/env python3
"""Summarize Issue #129 benchmark JSONL + vLLM timing logs into tables and plots.

Inputs:
- One or more JSONL files produced by scripts/tools/bench_issue129_vllm_lora_asample.py
- A text file containing server logs with `[vLLM timing] ... req=... total_s=... first_tok_s=...`
  plus optional `[coalesce flush] ... rid_ns=...` lines (added in mint_server/routes/sampling.py).

Outputs (written to --out-dir):
- summary.csv
- summary.md
- plot_{ttft,client_total,decode_tps}_<model>_<label>.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_VLLM_TIMING_RE = re.compile(
    r"\[vLLM timing\].*?\breq=(?P<req>\S+)\b.*?\btotal_s=(?P<total>[-+0-9.eE]+)\b.*?\bfirst_tok_s=(?P<first>[-+0-9.eE]+)",
)

_COALESCE_FLUSH_RE = re.compile(
    r"\[coalesce flush\].*?\bleader=(?P<leader>\S+)\b.*?\bvllm_req=(?P<vllm_req>\S+)\b.*?\brid_ns=(?P<rid_ns>\S+)",
)


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


@dataclass(frozen=True)
class Timing:
    req: str
    total_s: float
    first_tok_s: float


@dataclass(frozen=True)
class ReqRec:
    model: str
    label: str
    n_prompts: int
    n_samples: int
    prompt_reuse: str
    prompt_logprobs: int
    request_id: str
    client_total_s: float
    output_tokens: int


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_reqs(jsonl_paths: list[Path]) -> tuple[list[ReqRec], set[tuple[str, str]]]:
    reqs: list[ReqRec] = []
    meta_seen: set[tuple[str, str]] = set()
    meta_by_file: dict[Path, dict[str, Any]] = {}
    for p in jsonl_paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            kind = rec.get("kind")
            if kind == "meta":
                meta_by_file[p] = rec
                model = str(rec.get("model") or "")
                label = str(rec.get("label") or "")
                if model:
                    meta_seen.add((model, label))
                continue
            if kind != "request":
                continue
            if not rec.get("ok"):
                continue
            rid = rec.get("request_id")
            if not isinstance(rid, str) or not rid:
                continue
            out_tokens = rec.get("output_tokens")
            if not isinstance(out_tokens, int) or out_tokens < 0:
                continue
            model = str(rec.get("model") or meta_by_file.get(p, {}).get("model") or "")
            label = str(rec.get("label") or meta_by_file.get(p, {}).get("label") or "")
            reqs.append(
                ReqRec(
                    model=model,
                    label=label,
                    n_prompts=int(rec["n_prompts"]),
                    n_samples=int(rec["n_samples"]),
                    prompt_reuse=str(rec["prompt_reuse"]),
                    prompt_logprobs=int(rec["prompt_logprobs"]),
                    request_id=rid,
                    client_total_s=float(rec["client_total_s"]),
                    output_tokens=int(out_tokens),
                )
            )
    return reqs, meta_seen


def _parse_logs(log_paths: list[Path]) -> tuple[dict[str, Timing], dict[str, str]]:
    timing_by_req: dict[str, Timing] = {}
    rid_to_vllm_req: dict[str, str] = {}
    for p in log_paths:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _VLLM_TIMING_RE.search(line)
            if m:
                req = m.group("req")
                timing_by_req[req] = Timing(
                    req=req,
                    total_s=float(m.group("total")),
                    first_tok_s=float(m.group("first")),
                )
                continue
            m = _COALESCE_FLUSH_RE.search(line)
            if m:
                vllm_req = m.group("vllm_req")
                rid_ns = m.group("rid_ns")
                for part in rid_ns.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if ":" not in part:
                        continue
                    rid, _ns = part.split(":", 1)
                    rid = rid.strip()
                    if rid:
                        rid_to_vllm_req[rid] = vllm_req
    return timing_by_req, rid_to_vllm_req


def _group_output_tokens(reqs: list[ReqRec], rid_to_vllm_req: dict[str, str]) -> dict[str, int]:
    total_by_vllm_req: dict[str, int] = defaultdict(int)
    for r in reqs:
        vllm_req = rid_to_vllm_req.get(r.request_id, r.request_id)
        total_by_vllm_req[vllm_req] += int(r.output_tokens)
    return total_by_vllm_req


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", nargs="+", required=True, type=Path, help="Benchmark JSONL path(s)")
    p.add_argument("--logs", nargs="+", required=True, type=Path, help="Server log text path(s)")
    p.add_argument("--out-dir", type=Path, default=Path("results") / "issue87" / "issue129_summary")
    args = p.parse_args()

    reqs, meta_seen = _load_reqs(args.jsonl)
    timing_by_req, rid_to_vllm_req = _parse_logs(args.logs)
    out_tokens_by_vllm_req = _group_output_tokens(reqs, rid_to_vllm_req)

    joined: list[dict[str, Any]] = []
    missing_timing = 0
    for r in reqs:
        vllm_req = rid_to_vllm_req.get(r.request_id, r.request_id)
        t = timing_by_req.get(vllm_req)
        if t is None:
            missing_timing += 1
            continue
        decode_s = max(1e-9, float(t.total_s - t.first_tok_s))
        group_out = int(out_tokens_by_vllm_req.get(vllm_req, 0))
        joined.append(
            {
                "model": r.model,
                "label": r.label,
                "n_prompts": r.n_prompts,
                "n_samples": r.n_samples,
                "prompt_reuse": r.prompt_reuse,
                "prompt_logprobs": r.prompt_logprobs,
                "request_id": r.request_id,
                "vllm_req": vllm_req,
                "client_total_s": r.client_total_s,
                "server_ttft_s": float(t.first_tok_s),
                "server_total_s": float(t.total_s),
                "server_overhead_s": float(r.client_total_s - t.total_s),
                "decode_tps": float(group_out / decode_s),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "summary.csv"
    summary_md = args.out_dir / "summary.md"

    by_group: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in joined:
        key = (
            r["model"],
            r["label"],
            int(r["n_prompts"]),
            int(r["n_samples"]),
            r["prompt_reuse"],
            int(r["prompt_logprobs"]),
        )
        by_group[key].append(r)

    rows: list[dict[str, Any]] = []
    for (model, label, n_prompts, n_samples, prompt_reuse, plp), xs in sorted(by_group.items()):
        ttft = [float(r["server_ttft_s"]) for r in xs]
        tps = [float(r["decode_tps"]) for r in xs]
        total = [float(r["client_total_s"]) for r in xs]
        stotal = [float(r["server_total_s"]) for r in xs]
        overhead = [float(r["server_overhead_s"]) for r in xs]
        rows.append(
            {
                "model": model,
                "label": label,
                "n_prompts": n_prompts,
                "n_samples": n_samples,
                "prompt_reuse": prompt_reuse,
                "prompt_logprobs": plp,
                "count": len(xs),
                "ttft_p50_s": _pctl(ttft, 50),
                "ttft_p90_s": _pctl(ttft, 90),
                "decode_tps_p50": _pctl(tps, 50),
                "decode_tps_p90": _pctl(tps, 90),
                "client_total_p50_s": _pctl(total, 50),
                "client_total_p90_s": _pctl(total, 90),
                "server_total_p50_s": _pctl(stotal, 50),
                "server_total_p90_s": _pctl(stotal, 90),
                "overhead_p50_s": _pctl(overhead, 50),
                "overhead_p90_s": _pctl(overhead, 90),
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def _fmt(x: float) -> str:
        if math.isnan(x) or math.isinf(x):
            return "nan"
        return f"{x:.3f}"

    by_model_label: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model_label[(str(r["model"]), str(r["label"]))].append(r)

    md_lines: list[str] = []
    md_lines.append(f"- parsed req rows: {len(reqs)} ok requests")
    md_lines.append(f"- timing join rows: {len(joined)} (missing_timing={missing_timing})")
    if meta_seen:
        md_lines.append(f"- meta runs: {sorted(meta_seen)}")
    md_lines.append("")

    for (model, label), rs in sorted(by_model_label.items()):
        md_lines.append(f"## {model} ({label or 'no-label'})")
        md_lines.append("")
        md_lines.append("| n_prompts | n_samples | prompt_reuse | prompt_logprobs | count | ttft_p50_s | ttft_p90_s | decode_tps_p50 | decode_tps_p90 | client_total_p50_s | client_total_p90_s |")
        md_lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(
            rs,
            key=lambda x: (int(x["n_prompts"]), int(x["n_samples"]), str(x["prompt_reuse"]), int(x["prompt_logprobs"])),
        ):
            md_lines.append(
                "| {n_prompts} | {n_samples} | {prompt_reuse} | {prompt_logprobs} | {count} | {ttft_p50_s} | {ttft_p90_s} | {decode_tps_p50} | {decode_tps_p90} | {client_total_p50_s} | {client_total_p90_s} |".format(
                    n_prompts=int(r["n_prompts"]),
                    n_samples=int(r["n_samples"]),
                    prompt_reuse=str(r["prompt_reuse"]),
                    prompt_logprobs=int(r["prompt_logprobs"]),
                    count=int(r["count"]),
                    ttft_p50_s=_fmt(float(r["ttft_p50_s"])),
                    ttft_p90_s=_fmt(float(r["ttft_p90_s"])),
                    decode_tps_p50=_fmt(float(r["decode_tps_p50"])),
                    decode_tps_p90=_fmt(float(r["decode_tps_p90"])),
                    client_total_p50_s=_fmt(float(r["client_total_p50_s"])),
                    client_total_p90_s=_fmt(float(r["client_total_p90_s"])),
                )
            )
        md_lines.append("")

    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"wrote {summary_csv} and {summary_md}; matplotlib import failed, skipping plots")
        return

    # Plots: per (model,label), 2x2 grid (reuse x plp), lines by n_samples, x=n_prompts, y=p50(metric)
    def _plot_metric(metric_key: str, y_label: str, fname_prefix: str) -> None:
        for (model, label), rs in sorted(by_model_label.items()):
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharex=True)
            for i_reuse, reuse in enumerate(["same_prompt", "unique_prompts"]):
                for j_plp, plp in enumerate([0, 1]):
                    ax = axes[i_reuse][j_plp]
                    ax.set_title(f"{reuse}, prompt_logprobs={plp}")
                    subset = [r for r in rs if r["prompt_reuse"] == reuse and int(r["prompt_logprobs"]) == plp]
                    by_ns: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    for r in subset:
                        by_ns[int(r["n_samples"])].append(r)
                    for ns, vals in sorted(by_ns.items()):
                        vals_sorted = sorted(vals, key=lambda x: int(x["n_prompts"]))
                        xs = [int(v["n_prompts"]) for v in vals_sorted]
                        ys = [float(v[metric_key]) for v in vals_sorted]
                        ax.plot(xs, ys, marker="o", label=f"n_samples={ns}")
                    ax.set_xlabel("n_prompts")
                    ax.set_ylabel(y_label)
                    ax.grid(True, alpha=0.2)
                    if subset:
                        ax.legend(fontsize=8)
            fig.suptitle(f"{model} {label or ''}".strip())
            fig.tight_layout()
            safe_model = model.replace("/", "_").replace(" ", "_")
            safe_label = (label or "no-label").replace("/", "_").replace(" ", "_")
            out_png = args.out_dir / f"{fname_prefix}_{safe_model}_{safe_label}.png"
            out_svg = args.out_dir / f"{fname_prefix}_{safe_model}_{safe_label}.svg"
            fig.savefig(out_png, dpi=180)
            fig.savefig(out_svg)
            plt.close(fig)

    _plot_metric("ttft_p50_s", "TTFT p50 (s)", "plot_ttft")
    _plot_metric("client_total_p50_s", "Client total p50 (s)", "plot_client_total")
    _plot_metric("decode_tps_p50", "Decode TPS p50 (tokens/s)", "plot_decode_tps")

    print(f"wrote {summary_csv} and {summary_md}")


if __name__ == "__main__":
    main()
