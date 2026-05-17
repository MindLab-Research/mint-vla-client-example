#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = (os.environ.get("TINKER_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:8000").rstrip("/")
DEFAULT_API_KEY = os.environ.get("TINKER_API_KEY") or os.environ.get("MINT_API_KEY") or "dummy"

MODEL = os.environ.get("TINKER_INFER_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _ts_dir() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    return 1


def _post_json(base_url: str, api_key: str, path: str, payload: dict[str, Any], *, timeout_s: float) -> tuple[int, dict[str, Any]]:
    url = f"{base_url}{path}"
    r = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout_s)
    try:
        out = r.json()
    except Exception:
        out = {"_non_json_body": r.text[:800]}
    if not isinstance(out, dict):
        out = {"_non_dict_json": str(type(out))}
    return r.status_code, out


def _get_json(base_url: str, api_key: str, path: str, *, timeout_s: float) -> tuple[int, dict[str, Any]]:
    url = f"{base_url}{path}"
    r = requests.get(url, headers=_headers(api_key), timeout=timeout_s)
    try:
        out = r.json()
    except Exception:
        out = {"_non_json_body": r.text[:800]}
    if not isinstance(out, dict):
        out = {"_non_dict_json": str(type(out))}
    return r.status_code, out


def _create_sampling_session(*, base_url: str, api_key: str, model: str, timeout_s: float) -> str:
    st, sess = _post_json(
        base_url,
        api_key,
        "/api/v1/create_session",
        {"tags": ["scripts/tools/reproduce_issue_174_aliyun_235b.py"], "user_metadata": {}, "sdk_version": "repro-174"},
        timeout_s=timeout_s,
    )
    if st != 200:
        raise RuntimeError(f"create_session returned {st}: {sess!r}")
    session_id = sess.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {sess!r}")

    st, out = _post_json(
        base_url,
        api_key,
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": model},
        timeout_s=timeout_s,
    )
    if st != 200:
        raise RuntimeError(f"create_sampling_session returned {st}: {out!r}")
    sampling_session_id = out.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {out!r}")
    return sampling_session_id


def _prompt_tokens(*, prompt_len: int, uniq: int) -> list[int]:
    # Keep token ids small to avoid tokenizer/vocab mismatches.
    if prompt_len <= 0:
        return [10]
    toks = [10] * prompt_len
    u = uniq % 1000
    toks[0] = 100 + u
    if prompt_len >= 2:
        toks[-1] = 200 + u
    if prompt_len >= 8:
        toks[prompt_len // 2] = 300 + u
    return toks


def _prompt_tokens_random(*, prompt_len: int, seed: int, max_token_id: int) -> list[int]:
    if prompt_len <= 0:
        return [10]
    if max_token_id < 32:
        max_token_id = 32
    rng = random.Random(seed)
    toks = [rng.randint(10, max_token_id) for _ in range(prompt_len)]
    toks[0] = 100 + (seed % 1000)
    if prompt_len >= 2:
        toks[-1] = 200 + (seed % 1000)
    return toks


@dataclass(frozen=True)
class OneRequest:
    worker: int
    iter: int
    request_id: str
    ok: bool
    elapsed_s: float
    status: int
    error: str | None


def _looks_like_issue_174(err: str) -> bool:
    s = err.lower()
    needles = [
        "raychanneltimeouterror",
        "enginedeaderror",
        "timed out acquiring the read lock",
        "compiled dag",
        "ray compiled",
        "cgraph",
        "unserializableexception",
        "failed to deserialize exception",
    ]
    return any(n in s for n in needles)


def _parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("TINKER_CONCURRENCY", "16")))
    p.add_argument("--prompt-len", type=int, default=int(os.environ.get("TINKER_PROMPT_LEN", "512")))
    p.add_argument(
        "--prompt-lens",
        default=os.environ.get("TINKER_PROMPT_LENS") or "",
        help="Optional comma-separated prompt lengths. If set, overrides --prompt-len per request.",
    )
    p.add_argument("--num-samples", type=int, default=int(os.environ.get("TINKER_NUM_SAMPLES", "1")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("TINKER_MAX_TOKENS", "64")))
    p.add_argument("--warmup-prompt-len", type=int, default=int(os.environ.get("TINKER_WARMUP_PROMPT_LEN", "512")))
    p.add_argument("--warmup-max-tokens", type=int, default=int(os.environ.get("TINKER_WARMUP_MAX_TOKENS", "8")))
    p.add_argument(
        "--warmup-num-samples",
        type=int,
        default=int(os.environ.get("TINKER_WARMUP_NUM_SAMPLES", "0")),
        help="If >0, overrides --num-samples for the warmup request only.",
    )
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TINKER_TEMPERATURE", "0.7")))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TINKER_TOP_P", "1.0")))
    p.add_argument("--poll-s", type=float, default=float(os.environ.get("TINKER_POLL_S", "0.2")))
    p.add_argument("--http-timeout-s", type=float, default=float(os.environ.get("TINKER_HTTP_TIMEOUT_S", "30")))
    p.add_argument("--req-timeout-s", type=float, default=float(os.environ.get("TINKER_REQ_TIMEOUT_S", "900")))
    p.add_argument("--stall-timeout-s", type=float, default=float(os.environ.get("TINKER_STALL_TIMEOUT_S", "180")))
    p.add_argument("--total-seconds", type=float, default=float(os.environ.get("TINKER_TOTAL_SECONDS", "1200")))
    p.add_argument("--warmup-timeout-s", type=float, default=float(os.environ.get("TINKER_WARMUP_TIMEOUT_S", "900")))
    p.add_argument("--reuse-sampling-session", action="store_true")
    p.add_argument(
        "--session-mode",
        choices=["shared", "per_worker", "per_request"],
        default=os.environ.get("TINKER_SESSION_MODE") or "",
        help="Sampling session strategy. Default matches legacy: shared if --reuse-sampling-session, else per_request.",
    )
    p.add_argument(
        "--continue-after-repro",
        action="store_true",
        default=(os.environ.get("TINKER_CONTINUE_AFTER_REPRO", "").strip() == "1"),
        help="Do not exit immediately on the first issue signature; run until --total-seconds.",
    )
    p.add_argument(
        "--auto-kill-actors-on-repro",
        action="store_true",
        default=(os.environ.get("TINKER_AUTO_KILL_ACTORS_ON_REPRO", "").strip() == "1"),
        help="POST /internal/actors/kill for the vLLM actor when an issue signature or stall is detected.",
    )
    p.add_argument(
        "--auto-kill-min-interval-s",
        type=float,
        default=float(os.environ.get("TINKER_AUTO_KILL_MIN_INTERVAL_S", "120")),
        help="Minimum seconds between actor kill calls (rate limit).",
    )
    p.add_argument(
        "--prompt-mode",
        choices=["pattern", "random"],
        default=os.environ.get("TINKER_PROMPT_MODE", "pattern"),
        help="Prompt token generation strategy.",
    )
    p.add_argument(
        "--prompt-random-max-token-id",
        type=int,
        default=int(os.environ.get("TINKER_PROMPT_RANDOM_MAX_TOKEN_ID", "2000")),
        help="Upper bound for random token ids when --prompt-mode=random.",
    )
    p.add_argument("--progress-s", type=float, default=float(os.environ.get("TINKER_PROGRESS_S", "10")))
    p.add_argument("--run-dir", default=os.environ.get("TINKER_RUN_DIR") or None)
    args = p.parse_args()

    base_url = str(args.base_url).rstrip("/")
    api_key = str(args.api_key)
    if "--api-key" in sys.argv:
        print("WARNING: --api-key puts the key in argv/ps; prefer TINKER_API_KEY/MINT_API_KEY env var.", file=sys.stderr, flush=True)

    if args.concurrency < 1:
        return _fail("--concurrency must be >= 1")
    if args.num_samples < 1:
        return _fail("--num-samples must be >= 1")
    if args.max_tokens < 1:
        return _fail("--max-tokens must be >= 1")
    if args.warmup_prompt_len < 1:
        return _fail("--warmup-prompt-len must be >= 1")
    if args.warmup_max_tokens < 1:
        return _fail("--warmup-max-tokens must be >= 1")
    if args.total_seconds < 1:
        return _fail("--total-seconds must be >= 1")

    prompt_lens: list[int] = []
    if str(args.prompt_lens).strip():
        try:
            prompt_lens = _parse_int_list(str(args.prompt_lens))
        except Exception as e:
            return _fail(f"invalid --prompt-lens: {type(e).__name__}: {e}")
        if not prompt_lens:
            return _fail("--prompt-lens is set but parsed empty")
        if any(pl < 1 for pl in prompt_lens):
            return _fail(f"--prompt-lens must be >= 1, got {prompt_lens!r}")

    session_mode = str(args.session_mode).strip()
    if not session_mode:
        session_mode = "shared" if args.reuse_sampling_session else "per_request"
    if args.reuse_sampling_session and session_mode != "shared":
        return _fail("--reuse-sampling-session implies --session-mode shared (remove one of them)")

    st, health = _get_json(base_url, api_key, "/api/v1/healthz", timeout_s=10.0)
    if st != 200 or health.get("status") != "ready":
        return _fail(f"healthz not ready status={st} body={health!r}")

    # Auth check: required on prod for stress endpoints; fail early if missing.
    st, _ = _get_json(base_url, api_key, "/api/v1/server_info", timeout_s=10.0)
    if st == 401:
        return _fail("unauthorized: set TINKER_API_KEY/MINT_API_KEY (or --api-key) for this base-url")
    if st != 200:
        return _fail(f"unexpected /api/v1/server_info status={st}")

    run_dir = Path(args.run_dir) if args.run_dir else Path("results") / "issue174" / _ts_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"issue174_{args.model.replace('/', '_')}_c{args.concurrency}_pl{args.prompt_len}_ns{args.num_samples}_mt{args.max_tokens}.jsonl"

    try:
        st, info = _get_json(base_url, api_key, "/api/v1/server_info", timeout_s=10.0)
        git_sha = info.get("git_sha") if st == 200 else None
    except Exception:
        git_sha = None

    sampling_session_id: str | None = None
    if session_mode == "shared":
        sampling_session_id = _create_sampling_session(
            base_url=base_url,
            api_key=api_key,
            model=str(args.model),
            timeout_s=max(args.http_timeout_s, 30.0),
        )

    lock = threading.Lock()
    stop = threading.Event()
    reproduced = threading.Event()
    repro_events = 0
    kill_lock = threading.Lock()
    last_kill_at = 0.0
    last_complete = time.monotonic()
    total_submitted = 0
    total_completed = 0
    total_failed = 0
    stress_completed = 0

    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": _now_iso(),
                    "kind": "meta",
                    "base_url": base_url,
                    "git_sha": git_sha,
                    "model": str(args.model),
                    "concurrency": int(args.concurrency),
                    "prompt_len": int(args.prompt_len),
                    "prompt_lens": prompt_lens,
                    "num_samples": int(args.num_samples),
                    "max_tokens": int(args.max_tokens),
                    "warmup_prompt_len": int(args.warmup_prompt_len),
                    "warmup_max_tokens": int(args.warmup_max_tokens),
                    "warmup_num_samples": int(args.warmup_num_samples),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "poll_s": float(args.poll_s),
                    "http_timeout_s": float(args.http_timeout_s),
                    "req_timeout_s": float(args.req_timeout_s),
                    "stall_timeout_s": float(args.stall_timeout_s),
                    "total_seconds": float(args.total_seconds),
                    "warmup_timeout_s": float(args.warmup_timeout_s),
                    "session_mode": session_mode,
                    "continue_after_repro": bool(args.continue_after_repro),
                    "auto_kill_actors_on_repro": bool(args.auto_kill_actors_on_repro),
                    "auto_kill_min_interval_s": float(args.auto_kill_min_interval_s),
                    "prompt_mode": str(args.prompt_mode),
                    "prompt_random_max_token_id": int(args.prompt_random_max_token_id),
                    "progress_s": float(args.progress_s),
                },
                sort_keys=True,
            )
            + "\n"
        )
        f.flush()

        def record_obj(obj: dict[str, Any]) -> None:
            f.write(json.dumps(obj, sort_keys=True) + "\n")
            f.flush()

        def record_event(event_type: str, extra: dict[str, Any]) -> None:
            record_obj({"ts": _now_iso(), "kind": "event", "type": event_type, **extra})

        def maybe_kill_actors(reason: str) -> None:
            nonlocal last_kill_at
            if not args.auto_kill_actors_on_repro:
                return
            now = time.monotonic()
            with kill_lock:
                if (now - last_kill_at) < float(args.auto_kill_min_interval_s):
                    return
                last_kill_at = now
            try:
                st, out = _post_json(
                    base_url,
                    api_key,
                    "/internal/actors/kill",
                    {"actor_type": "vllm", "model_name": str(args.model)},
                    timeout_s=min(max(args.http_timeout_s, 1.0), 30.0),
                )
                record_event("kill_actors", {"reason": reason, "status": int(st), "resp": str(out)[:800]})
            except Exception as e:
                record_event("kill_actors", {"reason": reason, "status": 0, "resp": f"{type(e).__name__}: {e}"})

        def record(r: OneRequest) -> None:
            nonlocal last_complete, total_completed, total_failed, stress_completed
            with lock:
                last_complete = time.monotonic()
                if r.ok:
                    total_completed += 1
                    if r.worker >= 0:
                        stress_completed += 1
                else:
                    total_failed += 1
                record_obj(
                    {
                        "ts": _now_iso(),
                        "kind": "result",
                        "worker": r.worker,
                        "iter": r.iter,
                        "request_id": r.request_id,
                        "ok": bool(r.ok),
                        "elapsed_s": float(r.elapsed_s),
                        "status": int(r.status),
                        "error": r.error,
                    }
                )

        def run_one(*, worker: int, it: int, sampling_session_id: str, req_timeout_s: float) -> OneRequest:
            nonlocal total_submitted
            t0 = time.monotonic()
            req_id = f"repro-174-{worker}-{it}-{uuid.uuid4()}"
            uniq = worker * 100000 + it
            if worker < 0:
                pl = int(args.prompt_len)
            elif prompt_lens:
                pl = int(prompt_lens[(worker + it) % len(prompt_lens)])
            else:
                pl = int(args.prompt_len)
            try:
                if args.prompt_mode == "random":
                    toks = _prompt_tokens_random(
                        prompt_len=pl,
                        seed=uniq,
                        max_token_id=int(args.prompt_random_max_token_id),
                    )
                else:
                    toks = _prompt_tokens(prompt_len=pl, uniq=uniq)
                st, out = _post_json(
                    base_url,
                    api_key,
                    "/api/v1/asample",
                    {
                        "sampling_session_id": sampling_session_id,
                        "seq_id": 0,
                        "num_samples": int(args.num_samples),
                        "prompt": {"chunks": [{"type": "encoded_text", "tokens": toks}]},
                        "sampling_params": {
                            "max_tokens": int(args.max_tokens),
                            "temperature": float(args.temperature),
                            "top_k": -1,
                            "top_p": float(args.top_p),
                        },
                        "request_id": req_id,
                    },
                    timeout_s=min(max(args.http_timeout_s, 1.0), 300.0),
                )
                if st != 200:
                    return OneRequest(worker=worker, iter=it, request_id=req_id, ok=False, elapsed_s=time.monotonic() - t0, status=st, error=str(out)[:900])
                server_rid = out.get("request_id")
                if isinstance(server_rid, str) and server_rid:
                    req_id = server_rid
                with lock:
                    total_submitted += 1
                    record_obj(
                        {
                            "ts": _now_iso(),
                            "kind": "submit",
                            "worker": worker,
                            "iter": it,
                            "request_id": req_id,
                            "prompt_len": pl,
                            "submit_elapsed_s": float(time.monotonic() - t0),
                        }
                    )

                start = time.monotonic()
                while True:
                    elapsed = time.monotonic() - start
                    if elapsed > req_timeout_s:
                        return OneRequest(
                            worker=worker,
                            iter=it,
                            request_id=req_id,
                            ok=False,
                            elapsed_s=time.monotonic() - t0,
                            status=408,
                            error=f"TimeoutError: retrieve_future exceeded req_timeout_s={req_timeout_s}",
                        )
                    st, fut = _post_json(
                        base_url,
                        api_key,
                        "/api/v1/retrieve_future",
                        {"request_id": req_id, "model_id": sampling_session_id},
                        timeout_s=min(max(args.http_timeout_s, 1.0), 30.0),
                    )
                    if st == 408:
                        time.sleep(args.poll_s)
                        continue
                    if st != 200:
                        return OneRequest(
                            worker=worker,
                            iter=it,
                            request_id=req_id,
                            ok=False,
                            elapsed_s=time.monotonic() - t0,
                            status=st,
                            error=str(fut)[:900],
                        )
                    if "error" in fut:
                        return OneRequest(
                            worker=worker,
                            iter=it,
                            request_id=req_id,
                            ok=False,
                            elapsed_s=time.monotonic() - t0,
                            status=st,
                            error=str(fut.get("error"))[:900],
                        )
                    return OneRequest(worker=worker, iter=it, request_id=req_id, ok=True, elapsed_s=time.monotonic() - t0, status=st, error=None)
            except Exception as e:
                return OneRequest(worker=worker, iter=it, request_id=req_id, ok=False, elapsed_s=time.monotonic() - t0, status=0, error=f"{type(e).__name__}: {e}")

        def worker_loop(worker: int) -> None:
            nonlocal sampling_session_id, repro_events
            it = 0
            worker_sid: str | None = None
            if session_mode == "per_worker":
                worker_sid = _create_sampling_session(
                    base_url=base_url,
                    api_key=api_key,
                    model=str(args.model),
                    timeout_s=max(args.http_timeout_s, 30.0),
                )
                record_event("create_sampling_session", {"mode": "per_worker", "worker": worker, "sampling_session_id": worker_sid})
            while not stop.is_set():
                try:
                    if session_mode == "shared":
                        sid = sampling_session_id
                        assert sid is not None
                    elif session_mode == "per_worker":
                        sid = worker_sid
                        assert sid is not None
                    else:
                        sid = _create_sampling_session(
                            base_url=base_url,
                            api_key=api_key,
                            model=str(args.model),
                            timeout_s=max(args.http_timeout_s, 30.0),
                        )
                    r = run_one(worker=worker, it=it, sampling_session_id=sid, req_timeout_s=float(args.req_timeout_s))
                    record(r)
                    if (not r.ok) and r.error and _looks_like_issue_174(r.error):
                        with lock:
                            repro_events += 1
                        reproduced.set()
                        record_event(
                            "issue_signature",
                            {
                                "worker": worker,
                                "iter": it,
                                "request_id": r.request_id,
                                "error": r.error,
                            },
                        )
                        maybe_kill_actors("issue_signature")
                        if not args.continue_after_repro:
                            stop.set()
                            return
                except Exception as e:
                    record(
                        OneRequest(
                            worker=worker,
                            iter=it,
                            request_id=f"repro-174-worker-{worker}-{uuid.uuid4()}",
                            ok=False,
                            elapsed_s=0.0,
                            status=0,
                            error=f"{type(e).__name__}: {e}",
                        )
                    )
                it += 1

        def watchdog(deadline: float) -> None:
            nonlocal last_complete
            while not stop.is_set():
                now = time.monotonic()
                if now >= deadline:
                    record_event("time_limit_reached", {"elapsed_s": float(args.total_seconds)})
                    stop.set()
                    return
                with lock:
                    dt = now - last_complete
                if dt > args.stall_timeout_s:
                    reproduced.set()
                    record_event("stall_timeout", {"no_complete_for_s": float(dt), "stall_timeout_s": float(args.stall_timeout_s)})
                    maybe_kill_actors("stall_timeout")
                    stop.set()
                    return
                time.sleep(1.0)

        def progress_printer() -> None:
            while not stop.is_set():
                time.sleep(max(args.progress_s, 1.0))
                with lock:
                    submitted = total_submitted
                    completed = total_completed
                    failed = total_failed
                    dt = time.monotonic() - last_complete
                print(
                    f"[{_now_iso()}] progress submitted={submitted} completed={completed} failed={failed} "
                    f"no_complete_for_s={dt:.1f}",
                    flush=True,
                )

        pt = threading.Thread(target=progress_printer, daemon=True)
        pt.start()

        print(f"[{_now_iso()}] start base_url={base_url} model={args.model} concurrency={args.concurrency} out={out_path}", flush=True)

        if sampling_session_id is not None:
            orig_prompt_len = int(args.prompt_len)
            orig_max_tokens = int(args.max_tokens)
            orig_num_samples = int(args.num_samples)
            warm_prompt_len = int(args.warmup_prompt_len)
            warm_max_tokens = int(args.warmup_max_tokens)
            warm_num_samples = int(args.warmup_num_samples)
            args.prompt_len, args.max_tokens = warm_prompt_len, warm_max_tokens
            if warm_num_samples > 0:
                args.num_samples = warm_num_samples
            warm = run_one(worker=-1, it=-1, sampling_session_id=sampling_session_id, req_timeout_s=float(args.warmup_timeout_s))
            args.prompt_len, args.max_tokens, args.num_samples = orig_prompt_len, orig_max_tokens, orig_num_samples
            record(warm)
            if not warm.ok:
                return _fail(f"warmup_failed status={warm.status} error={warm.error!r} (details in {out_path})")

        deadline = time.monotonic() + float(args.total_seconds)
        wd = threading.Thread(target=watchdog, args=(deadline,), daemon=True)
        wd.start()

        with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
            for w in range(int(args.concurrency)):
                ex.submit(worker_loop, w)
            while not stop.is_set():
                time.sleep(0.5)

    if reproduced.is_set():
        return _fail(f"issue_174_reproduced (details in {out_path})")

    if stress_completed == 0:
        return _fail(f"no_stress_completions_within_total_seconds={args.total_seconds} (details in {out_path})")

    print(f"PASS (no repro within total_seconds={args.total_seconds}, details in {out_path})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
