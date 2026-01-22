#!/usr/bin/env python3
"""Pressure test: 32k-context RL loops (single or concurrent).

Modes:
- `concurrent` (default): spawns one child process per base model.
- `single`: runs the RL loop for one base model in-process.

Why one file:
- Avoids splitting orchestration and loop logic across multiple scripts.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


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


def _wait_future(fut, *, label: str, heartbeat_s: float) -> object:
    start = time.time()
    while True:
        try:
            return fut.result(timeout=heartbeat_s)
        except TimeoutError:
            elapsed = time.time() - start
            print(f"[{_ts()}] waiting {label} elapsed_s={elapsed:.0f}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    p_single = sub.add_parser("single", help="Run RL loop for one model")
    p_single.add_argument("--base-url", default=None, help="MINT_BASE_URL/TINKER_BASE_URL override")
    p_single.add_argument("--api-key", default=None, help="MINT_API_KEY/TINKER_API_KEY override")
    p_single.add_argument("--model", required=True, help="HF model name")

    p_conc = sub.add_parser("concurrent", help="Run RL loops concurrently (default)")
    p_conc.add_argument("--base-url", default=None, help="MINT_BASE_URL/TINKER_BASE_URL override")
    p_conc.add_argument("--api-key", default=None, help="MINT_API_KEY/TINKER_API_KEY override")
    p_conc.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated HF model names")
    p_conc.add_argument("--run-dir", default=None, help="Directory to write per-model logs")
    p_conc.add_argument("--stagger-s", type=float, default=0.0, help="Sleep between process launches")
    p_conc.add_argument("--heartbeat-s", type=float, default=60.0, help="Print status every N seconds")
    p_conc.add_argument("--max-runtime-s", type=float, default=0.0, help="0 = no limit")

    for pp in (p_single, p_conc):
        pp.add_argument("--steps", type=int, default=2)
        pp.add_argument("--prompts-per-step", type=int, default=4)
        pp.add_argument("--samples-per-prompt", type=int, default=2)
        pp.add_argument("--max-seq-len", type=int, default=32000)
        pp.add_argument("--gen-max-tokens", type=int, default=256)
        pp.add_argument("--temperature", type=float, default=0.7)
        pp.add_argument("--lora-rank", type=int, default=16)
        pp.add_argument("--learning-rate", type=float, default=5e-5)
        pp.add_argument("--future-heartbeat-s", type=float, default=60.0)

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
        heartbeat_s=float(args.future_heartbeat_s),
    )


def _run_single(*, base_url: str, api_key: str | None, model: str, cfg: RLConfig) -> int:
    import mint
    from mint import types

    print(f"[{_ts()}] RL start base_model={model} lora_rank={cfg.lora_rank} lr={cfg.learning_rate}", flush=True)
    print(
        f"[{_ts()}] cfg steps={cfg.steps} prompts_per_step={cfg.prompts_per_step} "
        f"samples_per_prompt={cfg.samples_per_prompt} max_seq_len={cfg.max_seq_len} "
        f"gen_max_tokens={cfg.gen_max_tokens} temperature={cfg.temperature}",
        flush=True,
    )

    random.seed(42)
    service_client = mint.ServiceClient(base_url=base_url, api_key=api_key)

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
        ckpt_name = f"{model.replace('/', '_')}_rl_step_{step:04d}"
        print(f"[{_ts()}] step {step+1}/{cfg.steps}: save_weights_and_get_sampling_client start name={ckpt_name}", flush=True)
        t0 = time.time()
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
                print(f"[{_ts()}] waiting save_weights_and_get_sampling_client elapsed_s={time.time()-t0:.0f}", flush=True)
        if "err" in out:
            raise out["err"]  # type: ignore[misc]
        sampling_client = out["client"]  # type: ignore[assignment]
        print(f"[{_ts()}] step {step+1}/{cfg.steps}: save_weights_and_get_sampling_client done elapsed_s={time.time()-t0:.1f}", flush=True)

        datums: list[types.Datum] = []
        step_rewards: list[float] = []

        for p in range(cfg.prompts_per_step):
            a = random.randint(10, 99)
            b = random.randint(10, 99)
            expected = a * b

            prompt_text = f"Question: What is {a} * {b}?\\nAnswer:"
            base_prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

            desired_prompt_len = max(2, cfg.max_seq_len - cfg.gen_max_tokens - 1)
            prompt_tokens = _pad_to(base_prompt_tokens, desired_prompt_len, filler_id)
            prompt = types.ModelInput.from_ints(tokens=prompt_tokens)

            print(
                f"[{_ts()}] step {step+1}/{cfg.steps}: rollout prompt {p+1}/{cfg.prompts_per_step} "
                f"num_samples={cfg.samples_per_prompt} prompt_len={len(prompt_tokens)}",
                flush=True,
            )

            sample_future = sampling_client.sample(
                prompt=prompt,
                num_samples=cfg.samples_per_prompt,
                sampling_params=types.SamplingParams(
                    max_tokens=cfg.gen_max_tokens,
                    temperature=cfg.temperature,
                    top_k=-1,
                    top_p=1.0,
                ),
            )
            sample_res = _wait_future(
                sample_future,
                label=f"sample model={model} prompt {p+1}/{cfg.prompts_per_step} step {step+1}/{cfg.steps}",
                heartbeat_s=cfg.heartbeat_s,
            )

            rewards: list[float] = []
            token_payloads: list[tuple[list[int], int, list[int]]] = []

            for seq in sample_res.sequences:
                completion_tokens = list(seq.tokens)
                if completion_tokens[: len(prompt_tokens)] == prompt_tokens:
                    full_tokens = completion_tokens
                    completion_tokens = completion_tokens[len(prompt_tokens) :]
                else:
                    full_tokens = prompt_tokens + completion_tokens

                if not full_tokens or full_tokens[-1] != eos_id:
                    full_tokens = full_tokens + [eos_id]

                full_tokens = _pad_to(full_tokens, cfg.max_seq_len, filler_id)
                completion_text = tokenizer.decode(completion_tokens)
                pred = _first_int(completion_text)
                reward = 1.0 if pred == expected else 0.0
                rewards.append(reward)
                token_payloads.append((full_tokens, len(prompt_tokens), completion_tokens))

            mean_reward = sum(rewards) / max(1, len(rewards))

            for reward, (full_tokens, prompt_len, completion_tokens) in zip(rewards, token_payloads):
                adv_scalar = float(reward - mean_reward)

                lp_future = sampling_client.compute_logprobs(types.ModelInput.from_ints(tokens=full_tokens))
                lp_res = _wait_future(
                    lp_future,
                    label=f"compute_logprobs model={model} prompt {p+1}/{cfg.prompts_per_step} step {step+1}/{cfg.steps}",
                    heartbeat_s=cfg.heartbeat_s,
                )
                lp = [(-100.0 if x is None else float(x)) for x in list(lp_res)]
                sampling_logprobs = lp[1:]

                n = cfg.max_seq_len - 1
                adv = [0.0] * n
                start_i = max(0, prompt_len - 1)
                end_i = min(n, start_i + len(completion_tokens) + 1)
                for i in range(start_i, end_i):
                    adv[i] = adv_scalar

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

        print(f"[{_ts()}] step {step+1}/{cfg.steps}: forward_backward loss_fn=ppo batch={len(datums)}", flush=True)
        fb_t0 = time.time()
        fb_future = training_client.forward_backward(datums, loss_fn="ppo")
        _wait_future(
            fb_future,
            label=f"forward_backward model={model} step {step+1}/{cfg.steps}",
            heartbeat_s=cfg.heartbeat_s,
        )
        print(f"[{_ts()}] step {step+1}/{cfg.steps}: forward_backward done elapsed_s={time.time()-fb_t0:.1f}", flush=True)

        print(f"[{_ts()}] step {step+1}/{cfg.steps}: optim_step lr={cfg.learning_rate}", flush=True)
        opt_t0 = time.time()
        opt_future = training_client.optim_step(types.AdamParams(learning_rate=cfg.learning_rate))
        _wait_future(
            opt_future,
            label=f"optim_step model={model} step {step+1}/{cfg.steps}",
            heartbeat_s=cfg.heartbeat_s,
        )
        print(f"[{_ts()}] step {step+1}/{cfg.steps}: optim_step done elapsed_s={time.time()-opt_t0:.1f}", flush=True)

        avg_reward = sum(step_rewards) / max(1, len(step_rewards))
        print(f"[{_ts()}] step {step+1}/{cfg.steps}: avg_reward={avg_reward:.4f}", flush=True)

    return 0


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
        try:
            return _run_single(base_url=base_url, api_key=api_key, model=args.model, cfg=cfg)
        except Exception as e:
            print(f"[{_ts()}] error model={args.model} {type(e).__name__}: {e}", flush=True)
            return 1

    if cmd != "concurrent":
        print(f"Unknown cmd: {cmd}", file=sys.stderr)
        return 2

    base_url = _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("TINKER_BASE_URL"), DEFAULT_BASE_URL)
    api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("TINKER_API_KEY"))
    cfg = _rl_cfg_from_args(args)

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
        f"gen_max_tokens={cfg.gen_max_tokens} temperature={cfg.temperature} lora_rank={cfg.lora_rank}",
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

        env["RL_STEPS"] = str(cfg.steps)
        env["RL_PROMPTS_PER_STEP"] = str(cfg.prompts_per_step)
        env["RL_SAMPLES_PER_PROMPT"] = str(cfg.samples_per_prompt)
        env["RL_MAX_SEQ_LEN"] = str(cfg.max_seq_len)
        env["RL_GEN_MAX_TOKENS"] = str(cfg.gen_max_tokens)
        env["RL_TEMPERATURE"] = str(cfg.temperature)
        env["LORA_RANK"] = str(cfg.lora_rank)
        env["RL_LEARNING_RATE"] = str(cfg.learning_rate)
        env["MINT_FUTURE_HEARTBEAT_S"] = str(cfg.heartbeat_s)

        proc = subprocess.Popen(
            [sys.executable, "-u", str(script_path), "single", "--model", model],
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
