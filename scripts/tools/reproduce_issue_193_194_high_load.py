#!/usr/bin/env python3
"""High-load repro harness for issue 193/194 style same-session crossing.

This script is intended to run from a workstation against a live mint-server.

What it does:
- creates one target training model
- rapidly enqueues same-session `forward_backward` -> `optim_step` pairs
- optionally injects `save_weights_for_sampler` on the same model
- optionally starts background training sessions to increase step latency
- polls all futures concurrently and records completion order

Why this catches the old bug:
- before the execution-serialization fix, a later `optim_step` could overtake an
  earlier `forward_backward` for the same `model_id` under load
- that shows up as a completion-order inversion (`optim_step` finishing before
  its matching `forward_backward`) and often as abrupt loss spikes

Outputs:
- jsonl event log under `/tmp/issue193-194-high-load/<timestamp>/events.jsonl`
- summary json under `/tmp/issue193-194-high-load/<timestamp>/summary.json`
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@dataclass(frozen=True)
class RequestRecord:
    op: str
    seq_id: int
    request_id: str
    enqueue_idx: int
    enqueued_at_s: float


@dataclass(frozen=True)
class CompletionRecord:
    op: str
    seq_id: int
    request_id: str
    enqueue_idx: int
    completed_at_s: float
    latency_s: float
    ok: bool
    error: str | None
    loss: float | None
    payload_type: str | None


def _coalesce(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _base_url(args: argparse.Namespace) -> str:
    return (
        _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("MINT_BASE_URL"), DEFAULT_BASE_URL)
        .rstrip("/")
    )


def _headers(args: argparse.Namespace) -> dict[str, str]:
    key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("MINT_API_KEY"))
    return {"X-API-Key": key} if key else {}


def _now_s() -> float:
    return time.time()


def _event_dir(args: argparse.Namespace) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    root = Path(args.output_dir).expanduser().resolve()
    path = root / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def _poll_future(
    *,
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    url = f"{base_url}/api/v1/retrieve_future"
    deadline = _now_s() + timeout_s
    while _now_s() < deadline:
        resp = requests.post(url, headers=headers, json={"request_id": request_id}, timeout=min(timeout_s, 60.0))
        if resp.status_code == 200:
            out = resp.json()
            if not isinstance(out, dict):
                raise TypeError(f"retrieve_future({request_id}) returned non-dict: {type(out)}")
            return out
        if resp.status_code == 408:
            time.sleep(poll_interval_s)
            continue
        raise RuntimeError(f"retrieve_future({request_id}) -> {resp.status_code}: {resp.text[:800]!r}")
    raise TimeoutError(f"retrieve_future timeout request_id={request_id} timeout_s={timeout_s}")


def _post_async(
    *,
    base_url: str,
    headers: dict[str, str],
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> str:
    resp = requests.post(f"{base_url}{path}", headers=headers, json=payload, timeout=min(timeout_s, 60.0))
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, dict):
        raise TypeError(f"POST {path} returned non-dict: {type(out)}")
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"POST {path} missing request_id: {out!r}")
    return request_id


def _post_and_wait(
    *,
    base_url: str,
    headers: dict[str, str],
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    request_id = _post_async(
        base_url=base_url,
        headers=headers,
        path=path,
        payload=payload,
        timeout_s=timeout_s,
    )
    return _poll_future(
        base_url=base_url,
        headers=headers,
        request_id=request_id,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )


def _create_model(
    *,
    base_url: str,
    headers: dict[str, str],
    base_model: str,
    lora_rank: int,
    learning_rate: float,
    timeout_s: float,
    poll_interval_s: float,
    tag: str,
) -> str:
    session_id = f"{tag}-{uuid.uuid4().hex[:10]}"
    out = _post_and_wait(
        base_url=base_url,
        headers=headers,
        path="/api/v1/create_model",
        payload={
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": base_model,
            "lora_config": {"rank": int(lora_rank)},
            "learning_rate": float(learning_rate),
            "user_metadata": {"script": "reproduce_issue_193_194_high_load.py", "tag": tag},
        },
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    model_id = out.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {out!r}")
    return model_id


def _delete_model(*, base_url: str, headers: dict[str, str], model_id: str, timeout_s: float) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=min(timeout_s, 30.0))
    except Exception:
        pass


def _build_batch(*, model: str, batch_size: int, prompt_repeats: int, target_repeats: int) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    data: list[dict[str, Any]] = []
    for idx in range(batch_size):
        prompt = (
            f"Session-isolation stress sample {idx}. "
            + ("alpha bravo charlie delta echo " * int(prompt_repeats))
            + " Predict the stable answer. "
        )
        target = f"stable-answer-{idx % 7} " + ("omega sigma tau " * int(target_repeats))
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        target_tokens = tokenizer.encode(target, add_special_tokens=False)
        full_tokens = prompt_tokens + target_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)
        if len(full_tokens) < 2:
            raise RuntimeError("constructed training example is too short")
        data.append(
            {
                "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
                "loss_fn_inputs": {
                    "target_tokens": {
                        "data": full_tokens[1:],
                        "shape": [len(full_tokens) - 1],
                        "dtype": "int64",
                    },
                    "loss_mask": {
                        "data": loss_mask[1:],
                        "shape": [len(loss_mask) - 1],
                        "dtype": "float32",
                    },
                },
            }
        )
    return data


def _extract_loss(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("loss:mean")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _await_record(
    record: RequestRecord,
    *,
    base_url: str,
    headers: dict[str, str],
    timeout_s: float,
    poll_interval_s: float,
) -> CompletionRecord:
    err: str | None = None
    payload: dict[str, Any] | None = None
    ok = False
    try:
        payload = _poll_future(
            base_url=base_url,
            headers=headers,
            request_id=record.request_id,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        status = str(payload.get("status", "")).strip().lower()
        error = payload.get("error")
        ok = not (status in ("failed", "error") or (isinstance(error, str) and error.strip()))
        if not ok:
            err = str(error or payload)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        ok = False
        payload = None

    completed_at_s = _now_s()
    return CompletionRecord(
        op=record.op,
        seq_id=record.seq_id,
        request_id=record.request_id,
        enqueue_idx=record.enqueue_idx,
        completed_at_s=completed_at_s,
        latency_s=completed_at_s - record.enqueued_at_s,
        ok=ok,
        error=err,
        loss=_extract_loss(payload or {}),
        payload_type=str((payload or {}).get("type")) if isinstance(payload, dict) and payload.get("type") is not None else None,
    )


def _background_worker(
    *,
    stop_event: threading.Event,
    errors: list[str],
    base_url: str,
    headers: dict[str, str],
    base_model: str,
    lora_rank: int,
    learning_rate: float,
    timeout_s: float,
    poll_interval_s: float,
    batch: list[dict[str, Any]],
    name: str,
) -> None:
    model_id = ""
    try:
        model_id = _create_model(
            base_url=base_url,
            headers=headers,
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            tag=name,
        )
        step = 0
        while not stop_event.is_set():
            step += 1
            _post_and_wait(
                base_url=base_url,
                headers=headers,
                path="/api/v1/forward_backward",
                payload={
                    "model_id": model_id,
                    "seq_id": step,
                    "forward_backward_input": {"data": batch, "loss_fn": "cross_entropy"},
                },
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
            _post_and_wait(
                base_url=base_url,
                headers=headers,
                path="/api/v1/optim_step",
                payload={
                    "model_id": model_id,
                    "seq_id": step,
                    "adam_params": {
                        "learning_rate": float(learning_rate),
                        "beta1": 0.9,
                        "beta2": 0.95,
                        "eps": 1e-12,
                    },
                },
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
    except Exception as exc:
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        stop_event.set()
    finally:
        if model_id:
            _delete_model(base_url=base_url, headers=headers, model_id=model_id, timeout_s=timeout_s)


def _detect_ordering_inversions(records: list[CompletionRecord], *, min_inversion_gap_s: float) -> list[str]:
    by_step: dict[int, dict[str, CompletionRecord]] = {}
    for rec in records:
        by_step.setdefault(rec.seq_id, {})[rec.op] = rec

    findings: list[str] = []
    for seq_id in sorted(by_step):
        ops = by_step[seq_id]
        fb = ops.get("forward_backward")
        opt = ops.get("optim_step")
        save = ops.get("save_weights_for_sampler")
        if (
            fb is not None
            and opt is not None
            and (fb.completed_at_s - opt.completed_at_s) > float(min_inversion_gap_s)
        ):
            findings.append(
                f"seq={seq_id}: optim_step completed before forward_backward "
                f"(opt={opt.completed_at_s:.3f}, fb={fb.completed_at_s:.3f})"
            )
        if (
            opt is not None
            and save is not None
            and (opt.completed_at_s - save.completed_at_s) > float(min_inversion_gap_s)
        ):
            findings.append(
                f"seq={seq_id}: save_weights_for_sampler completed before optim_step "
                f"(save={save.completed_at_s:.3f}, opt={opt.completed_at_s:.3f})"
            )
        if (
            fb is not None
            and save is not None
            and (fb.completed_at_s - save.completed_at_s) > float(min_inversion_gap_s)
        ):
            findings.append(
                f"seq={seq_id}: save_weights_for_sampler completed before forward_backward "
                f"(save={save.completed_at_s:.3f}, fb={fb.completed_at_s:.3f})"
            )
    return findings


def _detect_loss_spikes(
    records: list[CompletionRecord],
    *,
    baseline_window: int,
    spike_factor: float,
    min_abs_increase: float,
) -> list[str]:
    losses = [r for r in sorted(records, key=lambda x: x.seq_id) if r.op == "forward_backward" and r.loss is not None]
    findings: list[str] = []
    if len(losses) <= baseline_window:
        return findings
    for idx in range(baseline_window, len(losses)):
        current = float(losses[idx].loss or 0.0)
        history = [float(item.loss or 0.0) for item in losses[idx - baseline_window : idx]]
        baseline = statistics.median(history)
        if baseline <= 0.0:
            continue
        if current >= baseline * spike_factor and (current - baseline) >= min_abs_increase:
            findings.append(
                f"seq={losses[idx].seq_id}: loss spike current={current:.6f} baseline={baseline:.6f} "
                f"factor={current / baseline:.3f}"
            )
    return findings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=os.environ.get("MINT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--lora-rank", type=int, default=int(os.environ.get("MINT_LORA_RANK", "8")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("MINT_LR", "1e-4")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("MINT_STEPS", "8")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("MINT_BATCH_SIZE", "8")))
    parser.add_argument("--prompt-repeats", type=int, default=int(os.environ.get("MINT_PROMPT_REPEATS", "256")))
    parser.add_argument("--target-repeats", type=int, default=int(os.environ.get("MINT_TARGET_REPEATS", "48")))
    parser.add_argument("--background-models", type=int, default=int(os.environ.get("MINT_BG_MODELS", "2")))
    parser.add_argument("--save-every", type=int, default=int(os.environ.get("MINT_SAVE_EVERY", "0")))
    parser.add_argument("--dispatch-gap-s", type=float, default=float(os.environ.get("MINT_DISPATCH_GAP_S", "0.0")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("MINT_TIMEOUT_S", "3600")))
    parser.add_argument("--poll-interval-s", type=float, default=float(os.environ.get("MINT_POLL_INTERVAL_S", "2.0")))
    parser.add_argument(
        "--ordering-tolerance-s",
        type=float,
        default=float(os.environ.get("MINT_ORDERING_TOLERANCE_S", "2.0")),
    )
    parser.add_argument("--baseline-window", type=int, default=int(os.environ.get("MINT_BASELINE_WINDOW", "3")))
    parser.add_argument("--loss-spike-factor", type=float, default=float(os.environ.get("MINT_LOSS_SPIKE_FACTOR", "3.0")))
    parser.add_argument("--loss-spike-abs", type=float, default=float(os.environ.get("MINT_LOSS_SPIKE_ABS", "0.5")))
    parser.add_argument("--output-dir", default=os.environ.get("MINT_REPRO_OUTPUT_DIR", "/tmp/issue193-194-high-load"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = _base_url(args)
    headers = _headers(args)
    out_dir = _event_dir(args)
    events_path = out_dir / "events.jsonl"
    summary_path = out_dir / "summary.json"

    target_batch = _build_batch(
        model=args.model,
        batch_size=int(args.batch_size),
        prompt_repeats=int(args.prompt_repeats),
        target_repeats=int(args.target_repeats),
    )
    bg_batch = _build_batch(
        model=args.model,
        batch_size=max(2, int(args.batch_size) // 2),
        prompt_repeats=max(8, int(args.prompt_repeats) // 2),
        target_repeats=max(8, int(args.target_repeats) // 2),
    )

    target_model_id = ""
    stop_event = threading.Event()
    bg_errors: list[str] = []
    bg_threads: list[threading.Thread] = []
    request_records: list[RequestRecord] = []
    completions: list[CompletionRecord] = []
    enqueue_idx = 0

    try:
        target_model_id = _create_model(
            base_url=base_url,
            headers=headers,
            base_model=args.model,
            lora_rank=int(args.lora_rank),
            learning_rate=float(args.learning_rate),
            timeout_s=float(args.timeout_s),
            poll_interval_s=float(args.poll_interval_s),
            tag="issue193-194-target",
        )
        print(f"target_model_id={target_model_id}", flush=True)

        for bg_idx in range(int(args.background_models)):
            thread = threading.Thread(
                target=_background_worker,
                kwargs={
                    "stop_event": stop_event,
                    "errors": bg_errors,
                    "base_url": base_url,
                    "headers": headers,
                    "base_model": args.model,
                    "lora_rank": int(args.lora_rank),
                    "learning_rate": float(args.learning_rate),
                    "timeout_s": float(args.timeout_s),
                    "poll_interval_s": float(args.poll_interval_s),
                    "batch": bg_batch,
                    "name": f"issue193-194-bg-{bg_idx}",
                },
                daemon=True,
            )
            thread.start()
            bg_threads.append(thread)

        for seq_id in range(1, int(args.steps) + 1):
            fb_request_id = _post_async(
                base_url=base_url,
                headers=headers,
                path="/api/v1/forward_backward",
                payload={
                    "model_id": target_model_id,
                    "seq_id": seq_id,
                    "forward_backward_input": {"data": target_batch, "loss_fn": "cross_entropy"},
                },
                timeout_s=float(args.timeout_s),
            )
            request_records.append(
                RequestRecord(
                    op="forward_backward",
                    seq_id=seq_id,
                    request_id=fb_request_id,
                    enqueue_idx=enqueue_idx,
                    enqueued_at_s=_now_s(),
                )
            )
            enqueue_idx += 1

            opt_request_id = _post_async(
                base_url=base_url,
                headers=headers,
                path="/api/v1/optim_step",
                payload={
                    "model_id": target_model_id,
                    "seq_id": seq_id,
                    "adam_params": {
                        "learning_rate": float(args.learning_rate),
                        "beta1": 0.9,
                        "beta2": 0.95,
                        "eps": 1e-12,
                    },
                },
                timeout_s=float(args.timeout_s),
            )
            request_records.append(
                RequestRecord(
                    op="optim_step",
                    seq_id=seq_id,
                    request_id=opt_request_id,
                    enqueue_idx=enqueue_idx,
                    enqueued_at_s=_now_s(),
                )
            )
            enqueue_idx += 1

            if int(args.save_every) > 0 and seq_id % int(args.save_every) == 0:
                save_request_id = _post_async(
                    base_url=base_url,
                    headers=headers,
                    path="/api/v1/save_weights_for_sampler",
                    payload={
                        "model_id": target_model_id,
                        "path": f"issue193-194-step-{seq_id}",
                        "seq_id": seq_id,
                    },
                    timeout_s=float(args.timeout_s),
                )
                request_records.append(
                    RequestRecord(
                        op="save_weights_for_sampler",
                        seq_id=seq_id,
                        request_id=save_request_id,
                        enqueue_idx=enqueue_idx,
                        enqueued_at_s=_now_s(),
                    )
                )
                enqueue_idx += 1

            if float(args.dispatch_gap_s) > 0:
                time.sleep(float(args.dispatch_gap_s))

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(4, len(request_records)))) as pool:
            completions = list(
                pool.map(
                    lambda rec: _await_record(
                        rec,
                        base_url=base_url,
                        headers=headers,
                        timeout_s=float(args.timeout_s),
                        poll_interval_s=float(args.poll_interval_s),
                    ),
                    request_records,
                )
            )

    finally:
        stop_event.set()
        for thread in bg_threads:
            thread.join(timeout=5.0)
        if target_model_id:
            _delete_model(base_url=base_url, headers=headers, model_id=target_model_id, timeout_s=float(args.timeout_s))

    completions_sorted = sorted(completions, key=lambda rec: rec.completed_at_s)
    ordering_findings = _detect_ordering_inversions(
        completions_sorted,
        min_inversion_gap_s=float(args.ordering_tolerance_s),
    )
    spike_findings = _detect_loss_spikes(
        completions_sorted,
        baseline_window=int(args.baseline_window),
        spike_factor=float(args.loss_spike_factor),
        min_abs_increase=float(args.loss_spike_abs),
    )
    failures = [rec for rec in completions_sorted if not rec.ok]

    with events_path.open("w", encoding="utf-8") as f:
        for rec in completions_sorted:
            f.write(json.dumps(asdict(rec), ensure_ascii=True) + "\n")

    summary = {
        "base_url": base_url,
        "model": args.model,
        "target_model_id": target_model_id,
        "request_count": len(request_records),
        "steps": int(args.steps),
        "background_models": int(args.background_models),
        "save_every": int(args.save_every),
        "ordering_findings": ordering_findings,
        "loss_spike_findings": spike_findings,
        "background_errors": list(bg_errors),
        "failures": [asdict(rec) for rec in failures],
        "events_path": str(events_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"events={events_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    if ordering_findings:
        print("ORDERING_INVERSIONS:", flush=True)
        for finding in ordering_findings:
            print(f"  {finding}", flush=True)
    if spike_findings:
        print("LOSS_SPIKES:", flush=True)
        for finding in spike_findings:
            print(f"  {finding}", flush=True)
    if bg_errors:
        print("BACKGROUND_ERRORS:", flush=True)
        for err in bg_errors:
            print(f"  {err}", flush=True)
    if failures:
        print("REQUEST_FAILURES:", flush=True)
        for rec in failures:
            print(f"  seq={rec.seq_id} op={rec.op} error={rec.error}", flush=True)

    if ordering_findings or spike_findings or bg_errors or failures:
        return 1
    print("PASS: no ordering inversion, loss spike, or request failure observed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
