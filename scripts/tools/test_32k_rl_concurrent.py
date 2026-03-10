#!/usr/bin/env python3
"""Pressure test: 32k-context RL loops (single or concurrent).

Modes:
- `concurrent` (default): spawns one child process per base model.
- `multi-session`: spawns multiple sessions for the same base model.
- `single`: runs the RL loop for one base model in-process.

Why one file:
- Avoids splitting orchestration and loop logic across multiple scripts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODELS = ",".join(
    [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ]
)

STAGE_SAVE = "save_weights_and_get_sampling_client"
STAGE_ROLLOUT = "rollout_sample"
STAGE_ROLLOUT_ITEM = "rollout_sample_item"
STAGE_COMPUTE_LOGPROBS = "compute_logprobs"
STAGE_FORWARD_BACKWARD = "forward_backward"
STAGE_OPTIM_STEP = "optim_step"
STAGE_HEARTBEAT = "heartbeat"
STAGE_BARRIER = "barrier"


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _parse_bool_flag(v: str | int | bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid bool flag: {v!r}")


def _load_env() -> None:
    # Load env from the CWD (common case) and from the repo root (when invoked via wrapper).
    load_dotenv()
    repo_root_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_root_env.exists():
        load_dotenv(repo_root_env, override=False)

    # Prevent mint SDK from defaulting to https://mint.macaron.im when running tools
    # without explicit target configuration.
    if "MINT_BASE_URL" not in os.environ and "TINKER_BASE_URL" not in os.environ:
        os.environ["MINT_BASE_URL"] = DEFAULT_BASE_URL


def _sanitize_model(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").lower()


def _tail1(path: Path) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end == 0:
                return ""
            n = min(4096, end)
            f.seek(-n, os.SEEK_END)
            chunk = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def _tail_jsonl(path: Path) -> dict[str, Any] | None:
    line = _tail1(path)
    if not line:
        return None
    try:
        out = json.loads(line)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = path.open("a", encoding="utf-8")

    def write(self, rec: dict[str, Any]) -> None:
        self._f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


@dataclass(frozen=True)
class RLConfig:
    steps: int
    prompts_per_step: int
    samples_per_prompt: int
    max_seq_len: int
    gen_max_tokens: int
    temperature: float
    lora_rank: int
    learning_rate: float
    rollout_max_inflight: int
    train_microbatch: int
    heartbeat_s: float


def _first_int(s: str) -> int | None:
    m = re.search(r"(-?\\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pad_to(tokens: list[int], length: int, pad_id: int) -> list[int]:
    if len(tokens) >= length:
        return tokens[:length]
    return tokens + [pad_id] * (length - len(tokens))


def _wait_future(
    fut: Any,
    *,
    label: str,
    heartbeat_s: float,
    on_heartbeat: Callable[[float], None] | None = None,
) -> Any:
    start = time.time()
    while True:
        try:
            return fut.result(timeout=heartbeat_s)
        except TimeoutError:
            elapsed = time.time() - start
            print(f"[{_ts()}] waiting {label} elapsed_s={elapsed:.0f}", flush=True)
            if on_heartbeat:
                on_heartbeat(elapsed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    p_single = sub.add_parser("single", help="Run RL loop for one model")
    p_single.add_argument("--base-url", default=None, help="MINT_BASE_URL/TINKER_BASE_URL override")
    p_single.add_argument("--api-key", default=None, help="MINT_API_KEY/TINKER_API_KEY override")
    p_single.add_argument("--model", required=True, help="HF model name")
    p_single.add_argument("--session-idx", type=int, default=0, help="Optional session index for logging")
    p_single.add_argument("--jsonl-path", default=None, help="Write per-stage timing logs to this JSONL file")
    p_single.add_argument("--barrier-dir", default=None, help="Optional directory for step barrier files")
    p_single.add_argument("--barrier-sessions", type=int, default=0, help="Expected session count for step barrier")
    p_conc = sub.add_parser("concurrent", help="Run RL loops concurrently (default)")
    p_conc.add_argument("--base-url", default=None, help="MINT_BASE_URL/TINKER_BASE_URL override")
    p_conc.add_argument("--api-key", default=None, help="MINT_API_KEY/TINKER_API_KEY override")
    p_conc.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated HF model names")
    p_conc.add_argument("--run-dir", default=None, help="Directory to write per-model logs")
    p_conc.add_argument("--stagger-s", type=float, default=0.0, help="Sleep between process launches")
    p_conc.add_argument("--heartbeat-s", type=float, default=60.0, help="Print status every N seconds")
    p_conc.add_argument("--max-runtime-s", type=float, default=0.0, help="0 = no limit")

    p_ms = sub.add_parser("multi-session", help="Run multiple RL sessions concurrently for one base model")
    p_ms.add_argument("--base-url", default=None, help="MINT_BASE_URL/TINKER_BASE_URL override")
    p_ms.add_argument("--api-key", default=None, help="MINT_API_KEY/TINKER_API_KEY override")
    p_ms.add_argument("--model", required=True, help="HF model name")
    p_ms.add_argument("--num-sessions", type=int, required=True, help="Number of concurrent sessions (>=2)")
    p_ms.add_argument("--run-dir", default=None, help="Directory to write per-session logs")
    p_ms.add_argument("--stagger-s", type=float, default=0.0, help="Sleep between process launches")
    p_ms.add_argument("--heartbeat-s", type=float, default=30.0, help="Print status every N seconds")
    p_ms.add_argument("--stall-timeout-s", type=float, default=1800.0, help="Fail if no progress for this long")
    p_ms.add_argument("--sync-steps", action="store_true", help="Barrier at step boundaries across sessions")

    for pp in (p_single, p_conc, p_ms):
        pp.add_argument("--steps", type=int, default=2)
        pp.add_argument("--prompts-per-step", type=int, default=4)
        pp.add_argument("--samples-per-prompt", type=int, default=2)
        pp.add_argument("--max-seq-len", type=int, default=32000)
        pp.add_argument("--gen-max-tokens", type=int, default=256)
        pp.add_argument("--temperature", type=float, default=0.7)
        pp.add_argument("--lora-rank", type=int, default=16)
        pp.add_argument("--learning-rate", type=float, default=5e-5)
        pp.add_argument(
            "--rollout-max-inflight",
            type=int,
            default=1,
            help="Max in-flight sampling requests per step (higher enables vLLM continuous batching)",
        )
        pp.add_argument(
            "--train-microbatch",
            type=int,
            default=0,
            help="If >0, split forward_backward into chunks of this size (gradient accumulates until optim_step)",
        )
        pp.add_argument("--future-heartbeat-s", type=float, default=60.0)
        pp.add_argument("--prompt-logprobs", default="0", help="0/1: include prompt logprobs in sample()")
        pp.add_argument("--compute-logprobs", default="0", help="0/1: call compute_logprobs() before sample()")
        pp.add_argument("--trace-id", default=None, help="Optional fixed X-Trace-Id for all HTTP calls in this run")
        pp.add_argument("--run-id", default=None, help="Optional run identifier for JSONL/report joins")

    return p.parse_args()


def _rl_cfg_from_args(args: argparse.Namespace) -> RLConfig:
    return RLConfig(
        steps=int(args.steps),
        prompts_per_step=int(args.prompts_per_step),
        samples_per_prompt=int(args.samples_per_prompt),
        max_seq_len=int(args.max_seq_len),
        gen_max_tokens=int(args.gen_max_tokens),
        temperature=float(args.temperature),
        lora_rank=int(args.lora_rank),
        learning_rate=float(args.learning_rate),
        rollout_max_inflight=int(args.rollout_max_inflight),
        train_microbatch=int(args.train_microbatch),
        heartbeat_s=float(args.future_heartbeat_s),
    )


def _run_single(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    cfg: RLConfig,
    session_idx: int = 0,
    jsonl_path: Path | None = None,
    barrier_dir: Path | None = None,
    barrier_sessions: int = 0,
    prompt_logprobs: bool = False,
    compute_logprobs: bool = False,
    trace_id: str | None = None,
    run_id: str | None = None,
) -> int:
    import mint
    from mint import types

    writer = _JsonlWriter(jsonl_path) if jsonl_path else None
    pid = os.getpid()

    def _emit(stage: str, *, step_idx: int, elapsed_s: float, **extra: Any) -> None:
        if not writer:
            return
        rec: dict[str, Any] = {
            "ts": _ts(),
            "session_idx": session_idx,
            "step_idx": step_idx,
            "stage": stage,
            "elapsed_s": float(elapsed_s),
            "model": model,
            "pid": pid,
            "prompt_logprobs": bool(prompt_logprobs),
            "compute_logprobs": bool(compute_logprobs),
        }
        if trace_id:
            rec["trace_id"] = str(trace_id)
        if run_id:
            rec["run_id"] = str(run_id)
        rec.update(extra)
        writer.write(rec)

    def _heartbeat(*, stage: str, step_idx: int, elapsed_s: float, label: str) -> None:
        if not writer:
            return
        writer.write(
            {
                "ts": _ts(),
                "session_idx": session_idx,
                "step_idx": step_idx,
                "stage": STAGE_HEARTBEAT,
                "current_stage": stage,
                "elapsed_s": float(elapsed_s),
                "label": label,
                "model": model,
                "pid": pid,
                "prompt_logprobs": bool(prompt_logprobs),
                "compute_logprobs": bool(compute_logprobs),
            }
        )

    def _barrier(step_idx: int) -> None:
        if not barrier_dir or barrier_sessions <= 0:
            return
        step_dir = barrier_dir / f"step_{step_idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        ready_path = step_dir / f"session_{session_idx:02d}.ready"
        ready_path.write_text(_ts() + "\n", encoding="utf-8")

        t0 = time.time()
        if writer:
            writer.write(
                {
                    "ts": _ts(),
                    "session_idx": session_idx,
                    "step_idx": step_idx,
                    "stage": STAGE_BARRIER,
                    "event": "arrive",
                    "model": model,
                    "pid": pid,
                    "prompt_logprobs": bool(prompt_logprobs),
                    "compute_logprobs": bool(compute_logprobs),
                }
            )

        while True:
            if len(list(step_dir.glob("session_*.ready"))) >= barrier_sessions:
                break
            time.sleep(0.25)

        if writer:
            writer.write(
                {
                    "ts": _ts(),
                    "session_idx": session_idx,
                    "step_idx": step_idx,
                    "stage": STAGE_BARRIER,
                    "event": "release",
                    "wait_s": float(time.time() - t0),
                    "model": model,
                    "pid": pid,
                    "prompt_logprobs": bool(prompt_logprobs),
                    "compute_logprobs": bool(compute_logprobs),
                }
            )

    if writer:
        writer.write(
            {
                "ts": _ts(),
                "session_idx": session_idx,
                "step_idx": -1,
                "stage": "start",
                "model": model,
                "pid": pid,
                "prompt_logprobs": bool(prompt_logprobs),
                "compute_logprobs": bool(compute_logprobs),
                "trace_id": trace_id,
                "run_id": run_id,
            }
        )

    try:
        print(f"[{_ts()}] RL start base_model={model} session_idx={session_idx} lora_rank={cfg.lora_rank} lr={cfg.learning_rate}", flush=True)
        print(
            f"[{_ts()}] cfg steps={cfg.steps} prompts_per_step={cfg.prompts_per_step} "
            f"samples_per_prompt={cfg.samples_per_prompt} max_seq_len={cfg.max_seq_len} "
            f"gen_max_tokens={cfg.gen_max_tokens} temperature={cfg.temperature} "
            f"rollout_max_inflight={cfg.rollout_max_inflight} "
            f"train_microbatch={cfg.train_microbatch}",
            flush=True,
        )

        random.seed(42 + int(session_idx))
        default_headers: dict[str, str] = {}
        if trace_id:
            default_headers["X-Trace-Id"] = str(trace_id)
        service_client = mint.ServiceClient(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )

        print(f"[{_ts()}] create_lora_training_client start base_model={model}", flush=True)
        training_client = service_client.create_lora_training_client(
            base_model=model,
            rank=cfg.lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        print(f"[{_ts()}] get_tokenizer start base_model={model}", flush=True)
        tokenizer = training_client.get_tokenizer()
        print(f"[{_ts()}] get_tokenizer done base_model={model} vocab_size={tokenizer.vocab_size}", flush=True)
        eos_id = tokenizer.eos_token_id

        filler_ids = tokenizer.encode(" a", add_special_tokens=False) or tokenizer.encode("0", add_special_tokens=False)
        if not filler_ids:
            raise RuntimeError("Failed to get filler token id from tokenizer")
        filler_id = int(filler_ids[0])

        for step in range(cfg.steps):
            if barrier_dir and barrier_sessions > 0:
                _barrier(step)

            ckpt_name = f"{model.replace('/', '_')}_session_{session_idx:02d}_rl_step_{step:04d}"
            print(f"[{_ts()}] step {step+1}/{cfg.steps}: save_weights_and_get_sampling_client start name={ckpt_name}", flush=True)

            save_t0 = time.time()
            out: dict[str, object] = {}

            def _do_save() -> None:
                try:
                    out["client"] = training_client.save_weights_and_get_sampling_client(name=ckpt_name)
                except BaseException as e:
                    out["err"] = e

            th = threading.Thread(target=_do_save, daemon=True)
            th.start()
            while th.is_alive():
                th.join(timeout=cfg.heartbeat_s)
                if th.is_alive():
                    elapsed = time.time() - save_t0
                    print(f"[{_ts()}] waiting save_weights_and_get_sampling_client elapsed_s={elapsed:.0f}", flush=True)
                    _heartbeat(stage=STAGE_SAVE, step_idx=step, elapsed_s=elapsed, label="save_weights_and_get_sampling_client")

            if "err" in out:
                raise out["err"]  # type: ignore[misc]
            sampling_client = out["client"]  # type: ignore[assignment]
            save_elapsed = time.time() - save_t0
            print(f"[{_ts()}] step {step+1}/{cfg.steps}: save_weights_and_get_sampling_client done elapsed_s={save_elapsed:.1f}", flush=True)
            _emit(STAGE_SAVE, step_idx=step, elapsed_s=save_elapsed)

            step_rewards: list[float] = []
            datums: list[types.Datum] = []

            rollout_t0 = time.time()
            desired_prompt_len = max(2, cfg.max_seq_len - cfg.gen_max_tokens - 1)
            prompt_specs: list[tuple[int, int, int, list[int]]] = []
            for p in range(cfg.prompts_per_step):
                a = random.randint(10, 99)
                b = random.randint(10, 99)
                expected = a * b
                prompt_text = f"Question: What is {a} * {b}?\\nAnswer:"
                base_prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
                prompt_tokens = _pad_to(base_prompt_tokens, desired_prompt_len, filler_id)
                prompt_specs.append((p, expected, len(prompt_tokens), prompt_tokens))

            rollout_inflight = max(1, int(cfg.rollout_max_inflight))
            prompt_cursor = 0
            while prompt_cursor < len(prompt_specs):
                chunk = prompt_specs[prompt_cursor : prompt_cursor + rollout_inflight]
                prompt_cursor += rollout_inflight

                futures: list[tuple[int, int, list[int], float, concurrent.futures.Future]] = []
                for p_idx, expected, prompt_len, prompt_tokens in chunk:
                    if compute_logprobs:
                        clp_t0 = time.time()
                        clp_future = sampling_client.compute_logprobs(types.ModelInput.from_ints(tokens=prompt_tokens))
                        _wait_future(
                            clp_future,
                            label=f"compute_logprobs model={model} session={session_idx} step {step+1}/{cfg.steps} prompt {p_idx+1}/{cfg.prompts_per_step}",
                            heartbeat_s=cfg.heartbeat_s,
                            on_heartbeat=lambda elapsed: _heartbeat(
                                stage=STAGE_COMPUTE_LOGPROBS,
                                step_idx=step,
                                elapsed_s=elapsed,
                                label=f"compute_logprobs prompt={p_idx}",
                            ),
                        )
                        _emit(
                            STAGE_COMPUTE_LOGPROBS,
                            step_idx=step,
                            elapsed_s=time.time() - clp_t0,
                            prompt_idx=p_idx,
                            prompt_len=len(prompt_tokens),
                        )
                    print(
                        f"[{_ts()}] step {step+1}/{cfg.steps}: rollout prompt {p_idx+1}/{cfg.prompts_per_step} "
                        f"num_samples={cfg.samples_per_prompt} prompt_len={prompt_len}",
                        flush=True,
                    )
                    submit_t0 = time.time()
                    futures.append(
                        (
                            p_idx,
                            expected,
                            prompt_tokens,
                            submit_t0,
                            sampling_client.sample(
                                prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                                num_samples=cfg.samples_per_prompt,
                                sampling_params=types.SamplingParams(
                                    max_tokens=cfg.gen_max_tokens,
                                    temperature=cfg.temperature,
                                    top_k=-1,
                                    top_p=1.0,
                                ),
                                include_prompt_logprobs=bool(prompt_logprobs),
                            ),
                        )
                    )

                pending: dict[concurrent.futures.Future, tuple[int, int, list[int], float]] = {
                    fut: (p_idx, expected, prompt_tokens, submit_t0)
                    for p_idx, expected, prompt_tokens, submit_t0, fut in futures
                }
                # Mint futures can be lazy (polling starts on first .result() call).
                # Prime them so concurrent wait() reflects actual completion order.
                for fut in list(pending.keys()):
                    try:
                        fut.result(timeout=0)
                    except concurrent.futures.TimeoutError:
                        pass
                while pending:
                    done, _not_done = concurrent.futures.wait(
                        pending.keys(),
                        timeout=cfg.heartbeat_s,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        _heartbeat(
                            stage=STAGE_ROLLOUT,
                            step_idx=step,
                            elapsed_s=time.time() - rollout_t0,
                            label=f"sample pending={len(pending)}",
                        )
                        continue

                    for fut in done:
                        p_idx, expected, prompt_tokens, submit_t0 = pending.pop(fut)
                        sample_res = fut.result()
                        sample_latency_s = time.time() - submit_t0
                        _emit(
                            STAGE_ROLLOUT_ITEM,
                            step_idx=step,
                            elapsed_s=time.time() - rollout_t0,
                            prompt_idx=p_idx,
                            prompt_len=len(prompt_tokens),
                            num_samples=cfg.samples_per_prompt,
                            sample_latency_s=sample_latency_s,
                        )
                        _heartbeat(
                            stage=STAGE_ROLLOUT,
                            step_idx=step,
                            elapsed_s=time.time() - rollout_t0,
                            label=f"prompt {p_idx+1}/{cfg.prompts_per_step} complete",
                        )

                        rewards: list[float] = []
                        payloads: list[tuple[list[int], int, list[int], list[float], float]] = []
                        for seq in sample_res.sequences:
                            completion_tokens = list(seq.tokens)
                            completion_logprobs = seq.logprobs
                            if completion_logprobs is None:
                                raise RuntimeError("sample() returned no logprobs; expected per-token logprobs for PPO")
                            if len(completion_logprobs) != len(completion_tokens):
                                raise RuntimeError(
                                    f"sample() returned logprobs length mismatch: "
                                    f"tokens={len(completion_tokens)} logprobs={len(completion_logprobs)}"
                                )
                            if completion_tokens[: len(prompt_tokens)] == prompt_tokens:
                                full_tokens = completion_tokens
                                completion_logprobs = completion_logprobs[len(prompt_tokens) :]
                                completion_tokens = completion_tokens[len(prompt_tokens) :]
                            else:
                                full_tokens = prompt_tokens + completion_tokens

                            full_tokens = _pad_to(full_tokens, cfg.max_seq_len, filler_id)
                            completion_text = tokenizer.decode(completion_tokens)
                            pred = _first_int(completion_text)
                            reward = 1.0 if pred == expected else 0.0
                            rewards.append(reward)
                            payloads.append(
                                (full_tokens, len(prompt_tokens), completion_tokens, list(completion_logprobs), reward)
                            )

                        mean_reward = sum(rewards) / max(1, len(rewards))
                        # With samples_per_prompt=1 (or with identical rewards across samples),
                        # reward-mean_reward produces all-zero advantages, which can yield num_tokens=0
                        # in PPO and trigger NaN loss logging in Megatron.
                        baseline = 0.5 if len(set(rewards)) <= 1 else mean_reward
                        n = cfg.max_seq_len - 1
                        for full_tokens, prompt_len, completion_tokens, completion_logprobs, reward in payloads:
                            adv_scalar = float(reward - baseline)

                            sampling_logprobs = [0.0] * n
                            start_i = max(0, prompt_len - 1)
                            end_i = min(n, start_i + len(completion_tokens))
                            for j in range(max(0, end_i - start_i)):
                                sampling_logprobs[start_i + j] = float(completion_logprobs[j])

                            adv = [0.0] * n
                            for i in range(start_i, end_i):
                                adv[i] = float(adv_scalar)

                            datum = types.Datum(
                                model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
                                loss_fn_inputs={
                                    "target_tokens": full_tokens[1:],
                                    "logprobs": sampling_logprobs,
                                    "advantages": adv,
                                },
                            )
                            datums.append(datum)
                            step_rewards.append(reward)

            rollout_elapsed = time.time() - rollout_t0
            _emit(STAGE_ROLLOUT, step_idx=step, elapsed_s=rollout_elapsed, num_samples=len(datums))

            train_microbatch = int(cfg.train_microbatch)
            if train_microbatch <= 0:
                if len(datums) > 64:
                    raise RuntimeError(
                        f"forward_backward batch too large ({len(datums)} datums); set --train-microbatch to avoid OOM"
                    )
                microbatches = [datums]
            else:
                microbatches = [datums[i : i + train_microbatch] for i in range(0, len(datums), train_microbatch)]

            fb_total_t0 = time.time()
            for mb_idx, mb in enumerate(microbatches):
                print(
                    f"[{_ts()}] step {step+1}/{cfg.steps}: forward_backward loss_fn=ppo microbatch {mb_idx+1}/{len(microbatches)} "
                    f"batch={len(mb)}",
                    flush=True,
                )
                fb_t0 = time.time()
                fb_future = training_client.forward_backward(mb, loss_fn="ppo")
                _wait_future(
                    fb_future,
                    label=f"forward_backward model={model} session={session_idx} step {step+1}/{cfg.steps} microbatch {mb_idx+1}/{len(microbatches)}",
                    heartbeat_s=cfg.heartbeat_s,
                    on_heartbeat=lambda elapsed: _heartbeat(
                        stage=STAGE_FORWARD_BACKWARD, step_idx=step, elapsed_s=elapsed, label="forward_backward"
                    ),
                )
                fb_elapsed = time.time() - fb_t0
                _emit(
                    STAGE_FORWARD_BACKWARD,
                    step_idx=step,
                    elapsed_s=fb_elapsed,
                    microbatch_idx=mb_idx,
                    microbatch_count=len(microbatches),
                    microbatch_size=len(mb),
                )

            fb_total_elapsed = time.time() - fb_total_t0
            print(f"[{_ts()}] step {step+1}/{cfg.steps}: forward_backward total elapsed_s={fb_total_elapsed:.1f}", flush=True)

            print(f"[{_ts()}] step {step+1}/{cfg.steps}: optim_step lr={cfg.learning_rate}", flush=True)
            opt_t0 = time.time()
            opt_future = training_client.optim_step(types.AdamParams(learning_rate=cfg.learning_rate))
            _wait_future(
                opt_future,
                label=f"optim_step model={model} session={session_idx} step {step+1}/{cfg.steps}",
                heartbeat_s=cfg.heartbeat_s,
                on_heartbeat=lambda elapsed: _heartbeat(
                    stage=STAGE_OPTIM_STEP, step_idx=step, elapsed_s=elapsed, label="optim_step"
                ),
            )
            opt_elapsed = time.time() - opt_t0
            print(f"[{_ts()}] step {step+1}/{cfg.steps}: optim_step done elapsed_s={opt_elapsed:.1f}", flush=True)
            _emit(STAGE_OPTIM_STEP, step_idx=step, elapsed_s=opt_elapsed)

            avg_reward = sum(step_rewards) / max(1, len(step_rewards))
            print(f"[{_ts()}] step {step+1}/{cfg.steps}: avg_reward={avg_reward:.4f}", flush=True)

        if writer:
            writer.write(
                {
                    "ts": _ts(),
                    "session_idx": session_idx,
                    "step_idx": cfg.steps,
                    "stage": "end",
                    "model": model,
                    "pid": pid,
                    "prompt_logprobs": bool(prompt_logprobs),
                    "compute_logprobs": bool(compute_logprobs),
                    "trace_id": trace_id,
                    "run_id": run_id,
                }
            )
        return 0
    finally:
        if writer:
            writer.close()


_SUMMARY_STAGES = [
    STAGE_SAVE,
    STAGE_COMPUTE_LOGPROBS,
    STAGE_ROLLOUT,
    STAGE_FORWARD_BACKWARD,
    STAGE_OPTIM_STEP,
]


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    if p <= 0:
        return min(xs)
    if p >= 1:
        return max(xs)
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    idx = int(math.ceil(p * n) - 1)
    idx = max(0, min(n - 1, idx))
    return float(xs_sorted[idx])


def _stage_stats(xs: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(xs),
        "p50_s": _percentile(xs, 0.50),
        "p95_s": _percentile(xs, 0.95),
        "max_s": (max(xs) if xs else None),
    }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
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
    except FileNotFoundError:
        pass
    return out


def _write_summary(*, run_dir: Path, model: str, num_sessions: int, cfg: RLConfig, sessions: dict[int, Path]) -> None:
    by_session: dict[str, dict[str, dict[str, float | int | None]]] = {}
    all_values: dict[str, list[float]] = {stage: [] for stage in _SUMMARY_STAGES}
    run_ids: set[str] = set()
    trace_ids: set[str] = set()
    prompt_logprobs_flag: bool | None = None
    compute_logprobs_flag: bool | None = None

    for session_idx, jsonl_path in sorted(sessions.items()):
        vals: dict[str, list[float]] = {stage: [] for stage in _SUMMARY_STAGES}
        for rec in _iter_jsonl(jsonl_path):
            rid = rec.get("run_id")
            if isinstance(rid, str) and rid:
                run_ids.add(rid)
            tid = rec.get("trace_id")
            if isinstance(tid, str) and tid:
                trace_ids.add(tid)
            plp = rec.get("prompt_logprobs")
            if isinstance(plp, bool):
                prompt_logprobs_flag = plp if prompt_logprobs_flag is None else prompt_logprobs_flag
            clp = rec.get("compute_logprobs")
            if isinstance(clp, bool):
                compute_logprobs_flag = clp if compute_logprobs_flag is None else compute_logprobs_flag
            stage = rec.get("stage")
            if stage not in _SUMMARY_STAGES:
                continue
            elapsed = rec.get("elapsed_s")
            if isinstance(elapsed, (int, float)):
                vals[stage].append(float(elapsed))

        by_session[f"session_{session_idx:02d}"] = {stage: _stage_stats(v) for stage, v in vals.items()}
        for stage, v in vals.items():
            all_values[stage].extend(v)

    summary = {
        "model": model,
        "num_sessions": num_sessions,
        "cfg": {
            "steps": cfg.steps,
            "prompts_per_step": cfg.prompts_per_step,
            "samples_per_prompt": cfg.samples_per_prompt,
            "max_seq_len": cfg.max_seq_len,
            "gen_max_tokens": cfg.gen_max_tokens,
            "temperature": cfg.temperature,
            "lora_rank": cfg.lora_rank,
            "learning_rate": cfg.learning_rate,
            "rollout_max_inflight": cfg.rollout_max_inflight,
            "train_microbatch": cfg.train_microbatch,
            "future_heartbeat_s": cfg.heartbeat_s,
            "prompt_logprobs": prompt_logprobs_flag,
            "compute_logprobs": compute_logprobs_flag,
        },
        "run_ids": sorted(run_ids),
        "trace_ids": sorted(trace_ids),
        "by_session": by_session,
        "all_sessions": {stage: _stage_stats(v) for stage, v in all_values.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_multi_session(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    num_sessions: int,
    cfg: RLConfig,
    run_dir: Path,
    stagger_s: float,
    heartbeat_s: float,
    stall_timeout_s: float,
    sync_steps: bool,
    prompt_logprobs: bool,
    compute_logprobs: bool,
    trace_id: str | None = None,
    run_id: str | None = None,
) -> int:
    if num_sessions < 2:
        print("--num-sessions must be >= 2", file=sys.stderr)
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir = run_dir / "barrier" if sync_steps else None
    if barrier_dir:
        barrier_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()

    procs: dict[int, subprocess.Popen] = {}
    jsonls: dict[int, Path] = {}
    logs: dict[int, Path] = {}
    starts: dict[int, float] = {}

    print(f"[{_ts()}] base_url={base_url} model={model} num_sessions={num_sessions} run_dir={run_dir}", flush=True)
    print(
        f"[{_ts()}] cfg steps={cfg.steps} prompts_per_step={cfg.prompts_per_step} "
        f"samples_per_prompt={cfg.samples_per_prompt} max_seq_len={cfg.max_seq_len} "
        f"gen_max_tokens={cfg.gen_max_tokens} temperature={cfg.temperature} lora_rank={cfg.lora_rank} "
        f"rollout_max_inflight={cfg.rollout_max_inflight} "
        f"train_microbatch={cfg.train_microbatch}",
        flush=True,
    )

    for session_idx in range(num_sessions):
        jsonl_path = run_dir / f"session_{session_idx:02d}.jsonl"
        log_path = run_dir / f"session_{session_idx:02d}.log"
        jsonls[session_idx] = jsonl_path
        logs[session_idx] = log_path
        starts[session_idx] = time.time()

        log_f = log_path.open("wb")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MINT_BASE_URL"] = base_url
        if api_key:
            env["MINT_API_KEY"] = api_key

        argv = [
            sys.executable,
            "-u",
            str(script_path),
            "single",
            "--model",
            model,
            "--session-idx",
            str(session_idx),
            "--jsonl-path",
            str(jsonl_path),
            "--steps",
            str(cfg.steps),
            "--prompts-per-step",
            str(cfg.prompts_per_step),
            "--samples-per-prompt",
            str(cfg.samples_per_prompt),
            "--max-seq-len",
            str(cfg.max_seq_len),
            "--gen-max-tokens",
            str(cfg.gen_max_tokens),
            "--temperature",
            str(cfg.temperature),
            "--lora-rank",
            str(cfg.lora_rank),
            "--learning-rate",
            str(cfg.learning_rate),
            "--rollout-max-inflight",
            str(cfg.rollout_max_inflight),
            "--train-microbatch",
            str(cfg.train_microbatch),
            "--future-heartbeat-s",
            str(cfg.heartbeat_s),
            "--prompt-logprobs",
            "1" if prompt_logprobs else "0",
            "--compute-logprobs",
            "1" if compute_logprobs else "0",
        ]
        if trace_id:
            argv += ["--trace-id", str(trace_id)]
        if run_id:
            argv += ["--run-id", str(run_id)]
        if barrier_dir:
            argv += ["--barrier-dir", str(barrier_dir), "--barrier-sessions", str(num_sessions)]

        proc = subprocess.Popen(argv, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        procs[session_idx] = proc
        print(f"[{_ts()}] started session_idx={session_idx} pid={proc.pid} log={log_path} jsonl={jsonl_path}", flush=True)
        if stagger_s > 0:
            time.sleep(stagger_s)

    start = time.time()
    last_report = 0.0
    stalled = False
    stalled_session: int | None = None
    last_key: dict[int, tuple[int, str] | None] = {i: None for i in range(num_sessions)}
    last_key_ts: dict[int, float] = {i: starts[i] for i in range(num_sessions)}

    def _progress_key(last: dict[str, Any] | None) -> tuple[int, str] | None:
        if not last:
            return None
        if last.get("stage") == STAGE_HEARTBEAT:
            stage = last.get("current_stage") or STAGE_HEARTBEAT
            step = last.get("step_idx")
        else:
            stage = last.get("stage")
            step = last.get("step_idx")
        if not stage or step is None:
            return None
        try:
            step_i = int(step)
        except Exception:
            return None
        return (step_i, str(stage))

    try:
        while True:
            now = time.time()
            alive = [i for i, p in procs.items() if p.poll() is None]
            if not alive:
                break

            for session_idx in alive:
                jsonl_path = jsonls[session_idx]
                last = _tail_jsonl(jsonl_path)
                key = _progress_key(last)
                if key != last_key[session_idx]:
                    last_key[session_idx] = key
                    last_key_ts[session_idx] = now

                if (now - last_key_ts[session_idx]) > stall_timeout_s:
                    stalled = True
                    stalled_session = session_idx
                    break

            if stalled:
                break

            if heartbeat_s > 0 and (now - last_report) >= heartbeat_s:
                last_report = now
                print(f"[{_ts()}] heartbeat elapsed_s={now - start:.0f} alive={len(alive)}/{num_sessions}", flush=True)
                for session_idx in range(num_sessions):
                    p = procs[session_idx]
                    jsonl_path = jsonls[session_idx]
                    rc = p.poll()
                    mtime = jsonl_path.stat().st_mtime if jsonl_path.exists() else starts[session_idx]
                    last = _tail_jsonl(jsonl_path)
                    if not last:
                        last_desc = "<no jsonl>"
                    elif last.get("stage") == STAGE_HEARTBEAT:
                        last_desc = f"heartbeat({last.get('current_stage')}) step={last.get('step_idx')}"
                    else:
                        last_desc = f"{last.get('stage')} step={last.get('step_idx')}"
                    print(
                        f"[{_ts()}] session_idx={session_idx} pid={p.pid} rc={rc} "
                        f"jsonl_age_s={now - mtime:.0f} stage_age_s={now - last_key_ts[session_idx]:.0f} last={last_desc}",
                        flush=True,
                    )

            time.sleep(1.0)
    finally:
        if stalled:
            print(f"[{_ts()}] stall detected session_idx={stalled_session} timeout_s={stall_timeout_s:.0f}; terminating children", flush=True)
            for session_idx, p in procs.items():
                if p.poll() is None:
                    p.terminate()
            for session_idx, p in procs.items():
                try:
                    p.wait(timeout=30)
                except Exception:
                    pass

            for session_idx in range(num_sessions):
                last = _tail_jsonl(jsonls[session_idx])
                if not last:
                    last_desc = "<no jsonl>"
                elif last.get("stage") == STAGE_HEARTBEAT:
                    last_desc = f"heartbeat({last.get('current_stage')}) step={last.get('step_idx')}"
                else:
                    last_desc = f"{last.get('stage')} step={last.get('step_idx')}"
                print(f"[{_ts()}] session_idx={session_idx} last={last_desc}", flush=True)

        _write_summary(run_dir=run_dir, model=model, num_sessions=num_sessions, cfg=cfg, sessions=jsonls)

    rc = 0
    for p in procs.values():
        if p.returncode not in (0, None):
            rc = 1
    if stalled:
        rc = 1
    print(f"[{_ts()}] finished rc={rc} stalled={stalled} run_dir={run_dir}", flush=True)
    for session_idx in range(num_sessions):
        p = procs[session_idx]
        print(f"[{_ts()}] session_idx={session_idx} rc={p.returncode} log={logs[session_idx]} jsonl={jsonls[session_idx]}", flush=True)
    return rc


def main() -> int:
    _load_env()
    args = _parse_args()
    cmd = args.cmd or "concurrent"

    if cmd == "single":
        base_url = _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("TINKER_BASE_URL"))
        if not base_url:
            print("Set --base-url or MINT_BASE_URL/TINKER_BASE_URL", file=sys.stderr)
            return 2
        api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("TINKER_API_KEY"))
        cfg = _rl_cfg_from_args(args)
        jsonl_path = Path(args.jsonl_path) if getattr(args, "jsonl_path", None) else None
        barrier_dir = Path(args.barrier_dir) if getattr(args, "barrier_dir", None) else None
        barrier_sessions = int(getattr(args, "barrier_sessions", 0) or 0)
        session_idx = int(getattr(args, "session_idx", 0) or 0)
        prompt_logprobs = _parse_bool_flag(getattr(args, "prompt_logprobs", "0"))
        compute_logprobs = _parse_bool_flag(getattr(args, "compute_logprobs", "0"))
        trace_id = _coalesce(getattr(args, "trace_id", None), os.environ.get("MINT_TRACE_ID"))
        run_id = _coalesce(getattr(args, "run_id", None), os.environ.get("MINT_RUN_ID"))
        try:
            return _run_single(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                cfg=cfg,
                session_idx=session_idx,
                jsonl_path=jsonl_path,
                barrier_dir=barrier_dir,
                barrier_sessions=barrier_sessions,
                prompt_logprobs=bool(prompt_logprobs),
                compute_logprobs=bool(compute_logprobs),
                trace_id=trace_id,
                run_id=run_id,
            )
        except Exception as e:
            print(
                f"[{_ts()}] error model={args.model} session_idx={session_idx} {type(e).__name__}: {e}",
                flush=True,
            )
            return 1

    if cmd == "multi-session":
        base_url = _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("TINKER_BASE_URL"), DEFAULT_BASE_URL)
        api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("TINKER_API_KEY"))
        cfg = _rl_cfg_from_args(args)
        prompt_logprobs = _parse_bool_flag(getattr(args, "prompt_logprobs", "0"))
        compute_logprobs = _parse_bool_flag(getattr(args, "compute_logprobs", "0"))
        trace_id = _coalesce(getattr(args, "trace_id", None), os.environ.get("MINT_TRACE_ID"))
        run_id = _coalesce(getattr(args, "run_id", None), os.environ.get("MINT_RUN_ID"))
        run_dir = Path(args.run_dir or f"/tmp/32k_rl_multi_session.{int(time.time())}")
        try:
            return _run_multi_session(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                num_sessions=int(args.num_sessions),
                cfg=cfg,
                run_dir=run_dir,
                stagger_s=float(args.stagger_s),
                heartbeat_s=float(args.heartbeat_s),
                stall_timeout_s=float(args.stall_timeout_s),
                sync_steps=bool(args.sync_steps),
                prompt_logprobs=bool(prompt_logprobs),
                compute_logprobs=bool(compute_logprobs),
                trace_id=trace_id,
                run_id=run_id,
            )
        except Exception as e:
            print(f"[{_ts()}] error multi-session model={args.model} {type(e).__name__}: {e}", flush=True)
            return 1

    if cmd != "concurrent":
        print(f"Unknown cmd: {cmd}", file=sys.stderr)
        return 2

    base_url = _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("TINKER_BASE_URL"), DEFAULT_BASE_URL)
    api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("TINKER_API_KEY"))
    cfg = _rl_cfg_from_args(args)
    prompt_logprobs = _parse_bool_flag(getattr(args, "prompt_logprobs", "0"))
    compute_logprobs = _parse_bool_flag(getattr(args, "compute_logprobs", "0"))
    trace_id = _coalesce(getattr(args, "trace_id", None), os.environ.get("MINT_TRACE_ID"))
    run_id = _coalesce(getattr(args, "run_id", None), os.environ.get("MINT_RUN_ID"))

    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not models:
        print("No models specified", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir or f"/tmp/32k_rl_concurrent.{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()

    print(f"[{_ts()}] base_url={base_url} models={models} run_dir={run_dir}", flush=True)
    print(
        f"[{_ts()}] cfg steps={cfg.steps} prompts_per_step={cfg.prompts_per_step} "
        f"samples_per_prompt={cfg.samples_per_prompt} max_seq_len={cfg.max_seq_len} "
        f"gen_max_tokens={cfg.gen_max_tokens} temperature={cfg.temperature} lora_rank={cfg.lora_rank} "
        f"rollout_max_inflight={cfg.rollout_max_inflight} "
        f"train_microbatch={cfg.train_microbatch}",
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
        env["MINT_BASE_URL"] = base_url
        if api_key:
            env["MINT_API_KEY"] = api_key

        argv = [
            sys.executable,
            "-u",
            str(script_path),
            "single",
            "--model",
            model,
            "--steps",
            str(cfg.steps),
            "--prompts-per-step",
            str(cfg.prompts_per_step),
            "--samples-per-prompt",
            str(cfg.samples_per_prompt),
            "--max-seq-len",
            str(cfg.max_seq_len),
            "--gen-max-tokens",
            str(cfg.gen_max_tokens),
            "--temperature",
            str(cfg.temperature),
            "--lora-rank",
            str(cfg.lora_rank),
            "--learning-rate",
            str(cfg.learning_rate),
            "--rollout-max-inflight",
            str(cfg.rollout_max_inflight),
            "--train-microbatch",
            str(cfg.train_microbatch),
            "--future-heartbeat-s",
            str(cfg.heartbeat_s),
            "--prompt-logprobs",
            "1" if prompt_logprobs else "0",
            "--compute-logprobs",
            "1" if compute_logprobs else "0",
        ]
        if trace_id:
            argv += ["--trace-id", str(trace_id)]
        if run_id:
            argv += ["--run-id", str(run_id)]

        proc = subprocess.Popen(
            argv,
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
    for p in procs.values():
        if p.returncode not in (0, None):
            rc = 1
    print(f"[{_ts()}] finished rc={rc} timed_out={timed_out} run_dir={run_dir}", flush=True)
    for model in models:
        p = procs[model]
        print(f"[{_ts()}] model={model} rc={p.returncode} log={logs[model]}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
