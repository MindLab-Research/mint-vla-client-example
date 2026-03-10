#!/usr/bin/env python3
"""Issue #134 matrix runner for E2E RL step timing.

This script orchestrates runs of `scripts/tools/test_32k_rl_concurrent.py`
across (prompt_logprobs, compute_logprobs, model, num_sessions, repeat) and
writes a normalized run manifest for downstream reporting.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODELS = ",".join(
    [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ]
)


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ts_dir() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slug(s: str) -> str:
    return s.replace("/", "_").replace("-", "_").replace(" ", "_").lower()


def _parse_int_list(s: str, *, minimum: int = 0) -> list[int]:
    out: list[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        v = int(p)
        if v < minimum:
            raise ValueError(f"value must be >= {minimum}: {v}")
        out.append(v)
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_bool01_list(s: str) -> list[int]:
    vals = _parse_int_list(s, minimum=0)
    if any(v not in (0, 1) for v in vals):
        raise ValueError(f"expected 0/1 list, got: {vals}")
    return vals


def _trace_id_for_run(run_id: str) -> str:
    trace_id = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]
    if trace_id == "0" * 32:
        trace_id = "1" + trace_id[1:]
    return trace_id


def _redact_cmd(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for i, token in enumerate(redacted[:-1]):
        if token == "--api-key":
            redacted[i + 1] = "<redacted>"
    return redacted


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL") or os.environ.get("MINT_BASE_URL") or DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY") or os.environ.get("MINT_API_KEY") or "")
    p.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated base models")
    p.add_argument("--num-sessions", default="1", help="Comma-separated session counts, e.g. 1,2,4")
    p.add_argument("--prompt-logprobs", default="0,1", help="Comma-separated {0,1}")
    p.add_argument("--compute-logprobs", default="0,1", help="Comma-separated {0,1}")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--continue-on-error", action="store_true")

    # Forwarded RL config
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--prompts-per-step", type=int, default=2)
    p.add_argument("--samples-per-prompt", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--gen-max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--rollout-max-inflight", type=int, default=1)
    p.add_argument("--train-microbatch", type=int, default=0)
    p.add_argument("--future-heartbeat-s", type=float, default=60.0)
    p.add_argument("--heartbeat-s", type=float, default=30.0, help="multi-session heartbeat")
    p.add_argument("--stall-timeout-s", type=float, default=1800.0, help="multi-session stall timeout")
    p.add_argument("--stagger-s", type=float, default=0.0)
    args = p.parse_args()

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise SystemExit("no models configured")
    session_counts = _parse_int_list(args.num_sessions, minimum=1)
    plp_values = _parse_bool01_list(args.prompt_logprobs)
    clp_values = _parse_bool01_list(args.compute_logprobs)
    if int(args.repeats) < 1:
        raise SystemExit("--repeats must be >= 1")

    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "issue134" / _ts_dir()
    raw_dir = run_dir / "raw"
    run_logs = run_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_logs.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"

    script = Path(__file__).resolve().parent / "test_32k_rl_concurrent.py"
    if not script.exists():
        raise SystemExit(f"missing dependency script: {script}")

    rows: list[dict[str, Any]] = []
    total_jobs = len(models) * len(session_counts) * len(plp_values) * len(clp_values) * int(args.repeats)
    job_idx = 0
    for model in models:
        for num_sessions in session_counts:
            for plp in plp_values:
                for clp in clp_values:
                    for rep in range(int(args.repeats)):
                        job_idx += 1
                        run_id = f"issue134_{_slug(model)}_c{num_sessions}_plp{plp}_clp{clp}_r{rep:02d}"
                        trace_id = _trace_id_for_run(run_id)
                        log_path = run_logs / f"{run_id}.log"

                        if num_sessions == 1:
                            jsonl_path = raw_dir / f"{run_id}.jsonl"
                            cmd = [
                                str(args.python),
                                "-u",
                                str(script),
                                "single",
                                "--base-url",
                                str(args.base_url),
                                "--api-key",
                                str(args.api_key),
                                "--model",
                                model,
                                "--jsonl-path",
                                str(jsonl_path),
                                "--steps",
                                str(args.steps),
                                "--prompts-per-step",
                                str(args.prompts_per_step),
                                "--samples-per-prompt",
                                str(args.samples_per_prompt),
                                "--max-seq-len",
                                str(args.max_seq_len),
                                "--gen-max-tokens",
                                str(args.gen_max_tokens),
                                "--temperature",
                                str(args.temperature),
                                "--lora-rank",
                                str(args.lora_rank),
                                "--learning-rate",
                                str(args.learning_rate),
                                "--rollout-max-inflight",
                                str(args.rollout_max_inflight),
                                "--train-microbatch",
                                str(args.train_microbatch),
                                "--future-heartbeat-s",
                                str(args.future_heartbeat_s),
                                "--prompt-logprobs",
                                str(plp),
                                "--compute-logprobs",
                                str(clp),
                                "--trace-id",
                                trace_id,
                                "--run-id",
                                run_id,
                            ]
                            output_paths = [str(jsonl_path)]
                        else:
                            run_subdir = raw_dir / run_id
                            run_subdir.mkdir(parents=True, exist_ok=True)
                            cmd = [
                                str(args.python),
                                "-u",
                                str(script),
                                "multi-session",
                                "--base-url",
                                str(args.base_url),
                                "--api-key",
                                str(args.api_key),
                                "--model",
                                model,
                                "--num-sessions",
                                str(num_sessions),
                                "--run-dir",
                                str(run_subdir),
                                "--steps",
                                str(args.steps),
                                "--prompts-per-step",
                                str(args.prompts_per_step),
                                "--samples-per-prompt",
                                str(args.samples_per_prompt),
                                "--max-seq-len",
                                str(args.max_seq_len),
                                "--gen-max-tokens",
                                str(args.gen_max_tokens),
                                "--temperature",
                                str(args.temperature),
                                "--lora-rank",
                                str(args.lora_rank),
                                "--learning-rate",
                                str(args.learning_rate),
                                "--rollout-max-inflight",
                                str(args.rollout_max_inflight),
                                "--train-microbatch",
                                str(args.train_microbatch),
                                "--future-heartbeat-s",
                                str(args.future_heartbeat_s),
                                "--heartbeat-s",
                                str(args.heartbeat_s),
                                "--stall-timeout-s",
                                str(args.stall_timeout_s),
                                "--stagger-s",
                                str(args.stagger_s),
                                "--prompt-logprobs",
                                str(plp),
                                "--compute-logprobs",
                                str(clp),
                                "--trace-id",
                                trace_id,
                                "--run-id",
                                run_id,
                            ]
                            output_paths = [str(pth) for pth in sorted(run_subdir.glob("session_*.jsonl"))]

                        print(
                            f"[{_ts()}] [{job_idx}/{total_jobs}] run_id={run_id} model={model} c={num_sessions} plp={plp} clp={clp}",
                            flush=True,
                        )
                        t0 = time.time()
                        with log_path.open("w", encoding="utf-8") as lf:
                            lf.write(f"# started={_ts()}\n")
                            lf.write("CMD: " + " ".join(_redact_cmd(cmd)) + "\n\n")
                            lf.flush()
                            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                        elapsed_s = time.time() - t0

                        if num_sessions > 1:
                            run_subdir = raw_dir / run_id
                            output_paths = [str(pth) for pth in sorted(run_subdir.glob("session_*.jsonl"))]

                        row = {
                            "ts": _ts(),
                            "run_id": run_id,
                            "trace_id": trace_id,
                            "model": model,
                            "num_sessions": int(num_sessions),
                            "prompt_logprobs": int(plp),
                            "compute_logprobs": int(clp),
                            "repeat": int(rep),
                            "rc": int(proc.returncode),
                            "elapsed_s": float(elapsed_s),
                            "log_path": str(log_path),
                            "jsonl_paths": output_paths,
                            "base_url": str(args.base_url).rstrip("/"),
                            "cfg": {
                                "steps": int(args.steps),
                                "prompts_per_step": int(args.prompts_per_step),
                                "samples_per_prompt": int(args.samples_per_prompt),
                                "max_seq_len": int(args.max_seq_len),
                                "gen_max_tokens": int(args.gen_max_tokens),
                                "temperature": float(args.temperature),
                                "lora_rank": int(args.lora_rank),
                                "learning_rate": float(args.learning_rate),
                                "rollout_max_inflight": int(args.rollout_max_inflight),
                                "train_microbatch": int(args.train_microbatch),
                                "future_heartbeat_s": float(args.future_heartbeat_s),
                            },
                        }
                        rows.append(row)
                        with manifest_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, sort_keys=True) + "\n")

                        ok = proc.returncode == 0
                        print(
                            f"[{_ts()}] completed run_id={run_id} rc={proc.returncode} elapsed_s={elapsed_s:.1f} jsonl={len(output_paths)} ok={ok}",
                            flush=True,
                        )
                        if (not ok) and (not args.continue_on_error):
                            print(f"[{_ts()}] stopping on first failure (use --continue-on-error to keep going)", flush=True)
                            print(str(manifest_path), flush=True)
                            return 1

    success = sum(1 for r in rows if int(r["rc"]) == 0)
    failed = len(rows) - success
    summary = {
        "ts": _ts(),
        "jobs_total": len(rows),
        "jobs_success": success,
        "jobs_failed": failed,
        "manifest": str(manifest_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(manifest_path), flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
