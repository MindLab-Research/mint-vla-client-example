#!/usr/bin/env python3
"""Pressure test: concurrent 32k-context RL loops across supported models.

Spawns one `scripts/prod_32k_rl_loop.py` process per base model, so RL sampling
(vLLM) and training (dense pool / Megatron) are exercised concurrently.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODELS = ",".join(
    [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ]
)


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sanitize_model(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").lower()


def _tail1(path: Path) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end == 0:
                return ""
            # Read up to 4KB from the end and split lines.
            n = min(4096, end)
            f.seek(-n, os.SEEK_END)
            chunk = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=None, help="MINT_BASE_URL/TINKER_BASE_URL override")
    p.add_argument("--api-key", default=None, help="MINT_API_KEY/TINKER_API_KEY override")
    p.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated HF model names")
    p.add_argument("--run-dir", default=None, help="Directory to write per-model logs")
    p.add_argument("--stagger-s", type=float, default=0.0, help="Sleep between process launches")
    p.add_argument("--heartbeat-s", type=float, default=60.0, help="Print status every N seconds")
    p.add_argument("--max-runtime-s", type=float, default=0.0, help="0 = no limit")

    # Forwarded to prod_32k_rl_loop.py via env vars.
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--samples-per-prompt", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=32000)
    p.add_argument("--gen-max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = (
        args.base_url
        or os.environ.get("MINT_BASE_URL")
        or os.environ.get("TINKER_BASE_URL")
        or "http://localhost:18000"
    )
    api_key = args.api_key or os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY")

    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not models:
        print("No models specified", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir or f"/tmp/32k_rl_concurrent.{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve().parents[1] / "prod_32k_rl_loop.py"
    if not script_path.exists():
        print(f"Missing script: {script_path}", file=sys.stderr)
        return 2

    print(f"[{_ts()}] base_url={base_url} models={models} run_dir={run_dir}", flush=True)
    print(
        f"[{_ts()}] cfg steps={args.steps} prompts_per_step={args.prompts_per_step} "
        f"samples_per_prompt={args.samples_per_prompt} max_seq_len={args.max_seq_len} "
        f"gen_max_tokens={args.gen_max_tokens} temperature={args.temperature} lora_rank={args.lora_rank}",
        flush=True,
    )

    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, Path] = {}

    for model in models:
        log_path = run_dir / f"{_sanitize_model(model)}.log"
        logs[model] = log_path
        log_f = log_path.open("wb")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["BASE_MODEL"] = model
        env["MINT_BASE_URL"] = base_url
        if api_key:
            env["MINT_API_KEY"] = api_key

        env["RL_STEPS"] = str(args.steps)
        env["RL_PROMPTS_PER_STEP"] = str(args.prompts_per_step)
        env["RL_SAMPLES_PER_PROMPT"] = str(args.samples_per_prompt)
        env["RL_MAX_SEQ_LEN"] = str(args.max_seq_len)
        env["RL_GEN_MAX_TOKENS"] = str(args.gen_max_tokens)
        env["RL_TEMPERATURE"] = str(args.temperature)
        env["LORA_RANK"] = str(args.lora_rank)
        env["RL_LEARNING_RATE"] = str(args.learning_rate)

        proc = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs[model] = proc
        print(f"[{_ts()}] started model={model} pid={proc.pid} log={log_path}", flush=True)
        if args.stagger_s > 0:
            time.sleep(args.stagger_s)

    start = time.time()
    last_report = 0.0
    timed_out = False

    try:
        while True:
            now = time.time()
            if args.max_runtime_s > 0 and (now - start) > args.max_runtime_s:
                timed_out = True
                break

            alive = [m for m, p in procs.items() if p.poll() is None]
            if not alive:
                break

            if args.heartbeat_s > 0 and (now - last_report) >= args.heartbeat_s:
                last_report = now
                print(f"[{_ts()}] heartbeat elapsed_s={now - start:.0f} alive={len(alive)}/{len(models)}", flush=True)
                for model in models:
                    p = procs[model]
                    log_path = logs[model]
                    rc = p.poll()
                    mtime = log_path.stat().st_mtime if log_path.exists() else 0.0
                    last_line = _tail1(log_path)
                    print(
                        f"[{_ts()}] model={model} pid={p.pid} rc={rc} log_age_s={now - mtime:.0f} last={last_line}",
                        flush=True,
                    )

            time.sleep(1.0)
    finally:
        if timed_out:
            print(f"[{_ts()}] max-runtime exceeded; terminating children", flush=True)
            for p in procs.values():
                if p.poll() is None:
                    p.terminate()
            for p in procs.values():
                try:
                    p.wait(timeout=30)
                except Exception:
                    pass

    rc = 0
    for model, p in procs.items():
        if p.returncode not in (0, None):
            rc = 1
    print(f"[{_ts()}] finished rc={rc} timed_out={timed_out} run_dir={run_dir}", flush=True)
    for model in models:
        p = procs[model]
        print(f"[{_ts()}] model={model} rc={p.returncode} log={logs[model]}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
