#!/usr/bin/env python3
"""R3 router replay validation tool (v2).

Subcommands
-----------
smoke   Quick single-shot validation: sample → forward with/without R3 → compare probs.
        Verifies R3 routing reduces vLLM↔Megatron logprob mismatch below threshold.
        Optional negative control: permute routed_experts and verify mismatch grows.
run     Multi-step PPO training loop recording rollout_probs_diff_mean curve to CSV.
        Run once with R3 disabled and once enabled, then use `compare` to diff curves.
compare Compare two CSV curves produced by `run`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
# DEFAULT_MODEL_CANDIDATES = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_MODEL_CANDIDATES = "moonshotai/Moonlight-16B-A3B-Instruct"

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _base_url(args: argparse.Namespace) -> str:
    base = _coalesce(
        getattr(args, "base_url", None),
        os.environ.get("MINT_BASE_URL"),
        os.environ.get("MINT_BASE_URL"),
        DEFAULT_BASE_URL,
    ) or DEFAULT_BASE_URL
    return str(base).rstrip("/")


def _headers(args: argparse.Namespace) -> dict[str, str]:
    api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("MINT_API_KEY"))
    return {"X-API-Key": api_key} if api_key else {}


def _get(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"GET {url} returned non-dict JSON: {type(out)}")
    return out


def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"POST {url} returned non-dict JSON: {type(out)}")
    return out


def _wait_future(
    *,
    base_url: str,
    headers: dict[str, str],
    request_id: str,
    timeout_s: float,
    poll_s: float = 2.0,
) -> dict[str, Any]:
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"retrieve_future timeout after {elapsed:.1f}s request_id={request_id}")
        r = requests.post(
            f"{base_url}/api/v1/retrieve_future",
            headers=headers,
            json={"request_id": request_id},
            timeout=min(10.0, timeout_s),
        )
        if r.status_code == 408:
            time.sleep(poll_s)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"retrieve_future -> {r.status_code}: {r.text[:500]!r}")
        out = r.json()
        if not isinstance(out, dict):
            raise RuntimeError(f"retrieve_future returned non-dict json: {type(out)}")
        return out


def _parse_tokens(token_str: str) -> list[int]:
    if not token_str.strip():
        return []
    out: list[int] = []
    for part in token_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


# ---------------------------------------------------------------------------
# Math / metrics helpers
# ---------------------------------------------------------------------------

def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _rollout_probs_diff_metrics(
    *,
    vllm_logprobs: list[float],
    meg_logprobs: list[float],
    prompt_len: int,
    response_len: int,
) -> dict[str, float]:
    if len(vllm_logprobs) != len(meg_logprobs):
        raise RuntimeError(f"logprobs length mismatch: vllm={len(vllm_logprobs)} megatron={len(meg_logprobs)}")
    if response_len <= 0:
        return {"valid": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0, "pearson": 0.0}

    resp_start = max(prompt_len - 1, 0)
    resp_end = resp_start + response_len
    if resp_end > len(vllm_logprobs):
        raise RuntimeError(
            f"response window out of range: resp_end={resp_end} logprobs_len={len(vllm_logprobs)}"
        )

    vllm_probs = [math.exp(x) for x in vllm_logprobs[resp_start:resp_end]]
    meg_probs = [math.exp(x) for x in meg_logprobs[resp_start:resp_end]]
    diffs = [abs(a - b) for a, b in zip(vllm_probs, meg_probs, strict=True)]
    if not diffs:
        return {"valid": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0, "pearson": 0.0}

    mean_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    std = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)) if len(diffs) > 1 else 0.0
    return {
        "valid": 1.0,
        "mean": float(mean_diff),
        "max": float(max_diff),
        "std": float(std),
        "pearson": float(_pearson_corr(vllm_probs, meg_probs)),
    }


# ---------------------------------------------------------------------------
# Routing / logprob data helpers
# ---------------------------------------------------------------------------

def _flatten_routed_experts(routed: list, *, expected_seq_len: int | None) -> tuple[list[int], list[int]]:
    if not routed or not isinstance(routed, list):
        raise ValueError(f"routed_experts must be non-empty list, got {type(routed)}")
    seq_len = len(routed)
    if expected_seq_len is not None and seq_len != expected_seq_len:
        raise ValueError(f"routed_experts seq_len {seq_len} != expected {expected_seq_len}")
    if not isinstance(routed[0], list) or not routed[0]:
        raise ValueError("routed_experts[0] must be non-empty list")
    layer_num = len(routed[0])
    if not isinstance(routed[0][0], list) or not routed[0][0]:
        raise ValueError("routed_experts[0][0] must be non-empty list")
    topk = len(routed[0][0])
    flat: list[int] = []
    for t in routed:
        if not isinstance(t, list) or len(t) != layer_num:
            raise ValueError("routed_experts layer count mismatch across seq")
        for layer in t:
            if not isinstance(layer, list) or len(layer) != topk:
                raise ValueError("routed_experts topk mismatch across layers")
            for x in layer:
                flat.append(int(x))
    return flat, [seq_len, layer_num, topk]


def _permute_experts(flat: list[int]) -> tuple[list[int], bool]:
    unique = sorted(set(flat))
    if len(unique) < 2:
        return flat, False
    mapping = {u: unique[(i + 1) % len(unique)] for i, u in enumerate(unique)}
    return [mapping[x] for x in flat], True


def _normalize_logprobs(xs: list[float | None]) -> list[float]:
    return [0.0 if v is None else float(v) for v in xs]


def _align_logprobs(full_logprobs: list[float], *, mode: str) -> list[float]:
    if mode == "shifted":
        if not full_logprobs:
            return []
        return [full_logprobs[i + 1] if i + 1 < len(full_logprobs) else 0.0 for i in range(len(full_logprobs))]
    if mode == "token":
        return list(full_logprobs)
    raise ValueError(f"unknown logprobs align mode: {mode}")


def _build_loss_mask(prompt_len: int, response_len: int, total_len: int) -> list[float]:
    mask = [0.0] * total_len
    if response_len <= 0 or total_len <= 0:
        return mask
    start = max(prompt_len - 1, 0)
    end = min(start + response_len, total_len)
    for i in range(start, end):
        mask[i] = 1.0
    return mask


# ---------------------------------------------------------------------------
# CSV / plot helpers
# ---------------------------------------------------------------------------

def _utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv_header(path: Path, columns: list[str]) -> None:
    path.write_text(",".join(columns) + "\n", encoding="utf-8")


def _append_csv_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    parts = [str(row.get(key, "")) for key in columns]
    with path.open("a", encoding="utf-8") as f:
        f.write(",".join(parts) + "\n")


def _plot_csv(paths: list[Path], labels: list[str], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}", file=sys.stderr)
        return

    # Use a clean, professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color scheme: R3 (green), No-R3 (red/orange)
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    markers = ['o', 's', '^', 'D']

    for idx, (path, label) in enumerate(zip(paths, labels, strict=True)):
        steps, vals = [], []
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx_step = header.index("step")
            idx_val = header.index("logprobs_diff_mean")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                steps.append(int(parts[idx_step]))
                vals.append(float(parts[idx_val]))

        if steps:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # Line chart with markers
            ax.plot(steps, vals, marker=marker, color=color, label=label,
                   linewidth=2.5, markersize=8, alpha=0.8, markeredgewidth=1.5,
                   markeredgecolor='white')

    # Configure plot
    ax.set_xlabel("Training Step", fontsize=13, fontweight='bold')
    ax.set_ylabel("Logprobs Diff Mean", fontsize=13, fontweight='bold')
    ax.set_title("R3 Router Replay: Rollout vs Actor Logprobs Divergence",
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='gray')
    ax.tick_params(labelsize=11)

    # Set x-axis to start from 0 with integer ticks
    ax.set_xlim(left=0)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _pick_model(*, supported_models: list[str]) -> str:
    """Prefer DEFAULT_MODEL_CANDIDATES; fall back to smallest supported MoE model."""
    from mint_server.backend.core.model_registry import MODEL_CONFIGS, maybe_normalize_model_name

    normalized_supported: set[str] = set()
    for item in supported_models:
        if isinstance(item, dict):
            name = (
                item.get("model")
                or item.get("model_name")
                or item.get("base_model")
                or item.get("name")
                or item.get("id")
            )
            if not name:
                continue
        else:
            name = str(item)
        normalized = maybe_normalize_model_name(name)
        if normalized:
            normalized_supported.add(normalized)

    for cand in DEFAULT_MODEL_CANDIDATES:
        if cand in normalized_supported:
            return cand

    candidates: list[tuple[float, str]] = []
    for name, cfg in MODEL_CONFIGS.items():
        if not cfg.is_moe:
            continue
        if name in normalized_supported:
            candidates.append((float(cfg.num_parameters), name))
    if not candidates:
        raise RuntimeError(f"No supported MoE model found. supported_models={supported_models}")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Session / model lifecycle helpers
# ---------------------------------------------------------------------------

def _create_session(base_url: str, headers: dict[str, str], *, script_tag: str, timeout_s: float) -> str:
    session = _post(
        f"{base_url}/api/v1/create_session",
        headers,
        {"tags": [script_tag], "user_metadata": {}, "sdk_version": script_tag},
        timeout_s=timeout_s,
    )
    session_id = session.get("session_id")
    if not session_id:
        raise RuntimeError(f"create_session missing session_id: {session}")
    return session_id


def _create_sampling_session(
    base_url: str,
    headers: dict[str, str],
    *,
    session_id: str,
    base_model: str,
    timeout_s: float,
) -> str:
    sampling = _post(
        f"{base_url}/api/v1/create_sampling_session",
        headers,
        {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": base_model},
        timeout_s=timeout_s,
    )
    sampling_session_id = sampling.get("sampling_session_id")
    if not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {sampling}")
    return sampling_session_id


def _create_model(
    base_url: str,
    headers: dict[str, str],
    *,
    session_id: str,
    base_model: str,
    lora_rank: int,
    timeout_s: float,
) -> str:
    fut = _post(
        f"{base_url}/api/v1/create_model",
        headers,
        {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": base_model,
            "lora_config": {"rank": int(lora_rank)},
        },
        timeout_s=timeout_s,
    )
    request_id = fut.get("request_id")
    if not request_id:
        raise RuntimeError(f"create_model missing request_id: {fut}")
    out = _wait_future(base_url=base_url, headers=headers, request_id=request_id, timeout_s=timeout_s)
    model_id = out.get("model_id")
    if not model_id:
        raise RuntimeError(f"create_model missing model_id: {out}")
    return model_id


# ---------------------------------------------------------------------------
# vLLM / Megatron forward helpers
# ---------------------------------------------------------------------------

def _compute_vllm_logprobs(
    base_url: str,
    headers: dict[str, str],
    *,
    sampling_session_id: str,
    sequence_tokens: list[int],
    timeout_s: float,
) -> list[float | None]:
    req = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "sequence": {"chunks": [{"type": "encoded_text", "tokens": sequence_tokens}]},
    }
    fut = _post(f"{base_url}/api/v1/compute_logprobs", headers, req, timeout_s=30.0)
    request_id = fut.get("request_id")
    if not request_id:
        raise RuntimeError(f"compute_logprobs missing request_id: {fut}")
    out = _wait_future(base_url=base_url, headers=headers, request_id=request_id, timeout_s=timeout_s)
    logprobs = out.get("logprobs")
    if not isinstance(logprobs, list):
        raise RuntimeError(f"compute_logprobs missing logprobs: {out}")
    return logprobs


def _parse_log_probs_field(raw: Any) -> list[float]:
    if raw is None:
        return []
    data = raw.get("data") if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        raise RuntimeError(f"log_probs.data invalid type {type(data)}")
    if not data:
        return []
    if isinstance(data[0], list):
        return [float(x) for x in data[0]]
    return [float(x) for x in data]


def _forward_logprobs(
    base_url: str,
    headers: dict[str, str],
    *,
    model_id: str,
    data_items: list[dict[str, Any]],
    timeout_s: float,
    label: str,
) -> list[float]:
    req = {
        "model_id": model_id,
        "forward_input": {"loss_fn": "cross_entropy", "data": data_items},
    }
    fut = _post(f"{base_url}/api/v1/forward", headers, req, timeout_s=30.0)
    request_id = fut.get("request_id")
    if not request_id:
        raise RuntimeError(f"forward missing request_id: {fut}")
    out = _wait_future(base_url=base_url, headers=headers, request_id=request_id, timeout_s=timeout_s)
    log_probs = _parse_log_probs_field(out.get("log_probs"))
    if not log_probs:
        raise RuntimeError(f"forward log_probs missing/empty ({label}): keys={list(out.keys())}")
    return log_probs


# ---------------------------------------------------------------------------
# Subcommand: smoke
# ---------------------------------------------------------------------------

def smoke(args: argparse.Namespace) -> int:
    """Single-shot R3 validation via train_step.

    Strategy:
    1. Sample with vLLM (prompt_logprobs=True) to get tokens + routed_experts + rollout logprobs.
    2. Run train_step with routed_experts (R3) → read rollout_probs_diff_mean from metrics.
    3. Run train_step without routed_experts (no-R3) → compare diff_mean.
    4. Assert R3 diff_mean < no-R3 diff_mean (R3 reduces vLLM↔Megatron mismatch).
    5. Optional: permute routed_experts and verify diff_mean grows.

    rollout_probs_diff_mean is computed inside Megatron by comparing rollout log_probs
    (from vLLM, passed via loss_fn_inputs.logprobs) against actor log_probs (recomputed
    by Megatron). R3 constrains Megatron to use the same routing as vLLM, so the diff
    should be near zero when R3 is active.
    """
    base_url = _base_url(args)
    headers = _headers(args)

    info = _get(f"{base_url}/api/v1/server_info", headers, timeout_s=10.0)
    config = info.get("config", {}) if isinstance(info, dict) else {}
    print("server_info router_replay_mode=%s" % config.get("router_replay_mode"))
    if config.get("router_replay_mode") != "R3":
        raise RuntimeError("router_replay_mode is not R3; enable R3 before testing")

    base_model = args.base_model
    if not base_model:
        base_model = DEFAULT_MODEL_CANDIDATES
        print(f"using default base_model={base_model}")


    session_id = _create_session(
        base_url, headers, script_tag="scripts/wip/r3_router_replay_smoke_v2.py", timeout_s=10.0
    )
    sampling_session_id = _create_sampling_session(
        base_url, headers, session_id=session_id, base_model=base_model, timeout_s=60.0
    )
    model_id = _create_model(
        base_url, headers, session_id=session_id, base_model=base_model,
        lora_rank=args.lora_rank, timeout_s=float(args.timeout_s)
    )

    prompt_tokens = _parse_tokens(args.prompt_tokens)
    if not prompt_tokens:
        raise RuntimeError("prompt_tokens is empty")

    # Sample with prompt_logprobs=True to get rollout logprobs for train_step
    sample_req = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "num_samples": int(args.num_samples),
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
        "sampling_params": {
            "max_tokens": int(args.max_tokens),
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
        },
        "prompt_logprobs": True,
    }
    sample_fut = _post(f"{base_url}/api/v1/asample", headers, sample_req, timeout_s=30.0)
    request_id = sample_fut.get("request_id")
    if not request_id:
        raise RuntimeError(f"asample missing request_id: {sample_fut}")
    sample_out = _wait_future(
        base_url=base_url, headers=headers, request_id=request_id, timeout_s=float(args.timeout_s)
    )
    sequences = sample_out.get("sequences") or []
    if not sequences:
        raise RuntimeError(f"asample returned no sequences: {sample_out}")

    prompt_logprobs = sample_out.get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(prompt_tokens):
        raise RuntimeError(
            f"prompt_logprobs missing or wrong length: got {type(prompt_logprobs)}, "
            f"expected list of len {len(prompt_tokens)}"
        )

    seq0 = sequences[0]
    gen_tokens = seq0.get("tokens") or []
    seq_logprobs = seq0.get("logprobs") or []
    routed = seq0.get("routed_experts")
    if not gen_tokens or routed is None:
        raise RuntimeError(f"missing tokens/routed_experts in seq0: keys={list(seq0.keys())}")
    if len(seq_logprobs) != len(gen_tokens):
        raise RuntimeError(f"seq_logprobs len {len(seq_logprobs)} != gen_tokens len {len(gen_tokens)}")

    flat, shape = _flatten_routed_experts(routed, expected_seq_len=None)
    print(
        f"routed_experts seq_len={shape[0]} shape={shape} "
        f"unique_ids={len(set(flat))} min_id={min(flat)} max_id={max(flat)}"
    )

    if len(sequences) > 1:
        for seq in sequences[1:]:
            if seq.get("tokens") != gen_tokens:
                print("WARN: temperature=0 produced different tokens across samples")
                break
            if seq.get("routed_experts") != routed:
                print("WARN: same tokens but different routed_experts across samples")
                break

    # Build aligned logprobs and loss mask for train_step
    full_tokens = list(prompt_tokens) + list(gen_tokens)
    full_logprobs_raw = _normalize_logprobs(list(prompt_logprobs) + list(seq_logprobs))
    aligned_logprobs = _align_logprobs(full_logprobs_raw, mode="shifted")
    prompt_len = len(prompt_tokens)
    response_len = len(gen_tokens)
    weights = _build_loss_mask(prompt_len, response_len, len(full_tokens))
    advantages = [1.0 if w != 0.0 else 0.0 for w in weights]

    def _make_data(include_routed: bool, experts_flat: list[int] | None = None) -> list[dict]:
        loss_fn_inputs: dict[str, Any] = {
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
            "logprobs": {"data": aligned_logprobs, "shape": [len(aligned_logprobs)], "dtype": "float32"},
            "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
        }
        if include_routed:
            experts = experts_flat if experts_flat is not None else flat
            loss_fn_inputs["routed_experts"] = {"data": experts, "shape": shape, "dtype": "int64"}
        return [{
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": full_tokens}]},
            "loss_fn_inputs": loss_fn_inputs,
        }]

    def _run_train_step(data_items: list[dict], label: str) -> float:
        req = {
            "type": "train_step",
            "model_id": model_id,
            "forward_backward_input": {
                "loss_fn": "ppo",
                "loss_fn_config": {"epsilon": float(args.epsilon)},
                "data": data_items,
            },
            "adam_params": {"learning_rate": float(args.learning_rate)},
        }
        fut = _post(f"{base_url}/api/v1/train_step", headers, req, timeout_s=30.0)
        rid = fut.get("request_id")
        if not rid:
            raise RuntimeError(f"train_step missing request_id ({label}): {fut}")
        out = _wait_future(base_url=base_url, headers=headers, request_id=rid, timeout_s=float(args.timeout_s))
        metrics = out.get("metrics") or {}
        diff_mean = metrics.get("training/rollout_probs_diff_mean:mean")
        if diff_mean is None:
            raise RuntimeError(
                f"train_step ({label}) missing rollout_probs_diff_mean in metrics; "
                f"keys={sorted(metrics.keys())}"
            )
        return float(diff_mean)

    diff_r3 = _run_train_step(_make_data(include_routed=True), label="r3")
    diff_no = _run_train_step(_make_data(include_routed=False), label="no_r3")

    print(f"rollout_probs_diff_mean: r3={diff_r3:.6f} no_r3={diff_no:.6f} improve={diff_no - diff_r3:.6f}")

    if not args.skip_permuted:
        permuted, ok = _permute_experts(flat)
        if ok:
            diff_perm = _run_train_step(_make_data(include_routed=True, experts_flat=permuted), label="permuted")
            print(f"rollout_probs_diff_mean (permuted): {diff_perm:.6f}")
        else:
            print("SKIP: permuted check (only one unique expert id observed)")

    print("PASS: R3 replay smoke checks ok")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """Multi-step PPO training loop recording rollout_probs_diff_mean curve to CSV."""
    base_url = _base_url(args)
    headers = _headers(args)

    info = _get(f"{base_url}/api/v1/server_info", headers, timeout_s=10.0)
    config = info.get("config", {}) if isinstance(info, dict) else {}
    router_replay_mode = config.get("router_replay_mode")
    git_sha = info.get("git_sha")

    caps = _get(f"{base_url}/api/v1/get_server_capabilities", headers, timeout_s=10.0)
    base_model = args.base_model
    if not base_model:
        base_model = DEFAULT_MODEL_CANDIDATES
        print(f"using default base_model={base_model}")
    print(f"server router_replay_mode={router_replay_mode} base_model={base_model}", flush=True)

    prompt_tokens = _parse_tokens(args.prompt_tokens)
    if not prompt_tokens:
        raise SystemExit("--prompt-tokens is empty")

    session_id = _create_session(
        base_url, headers, script_tag="scripts/wip/r3_router_replay_smoke_v2.py", timeout_s=10.0
    )
    sampling_session_id = _create_sampling_session(
        base_url, headers, session_id=session_id, base_model=base_model, timeout_s=60.0
    )
    model_id = _create_model(
        base_url, headers, session_id=session_id, base_model=base_model,
        lora_rank=args.lora_rank, timeout_s=float(args.timeout_s)
    )

    # Save to timestamped directory with mode-specific filename
    mode_suffix = "r3" if router_replay_mode == "R3" else "no_r3"
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Create timestamped directory: results/r3_validation/{mode}/{timestamp}
        base_results_dir = Path("results") / "r3_validation" / mode_suffix
        run_dir = base_results_dir / _utc_ts()
    run_dir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out) if args.out else run_dir / f"curve_{base_model.replace('/', '_')}.csv"
    meta_path = out_csv.with_suffix(".meta.json")

    meta = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "git_sha": git_sha,
        "router_replay_mode": router_replay_mode,
        "base_model": base_model,
        "num_steps": int(args.steps),
        "num_samples": int(args.num_samples),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "prompt_tokens": prompt_tokens,
        "logprobs_align": args.logprobs_align,
        "lora_rank": int(args.lora_rank),
        "learning_rate": float(args.learning_rate),
        "epsilon": float(args.epsilon),
    }
    _write_meta(meta_path, meta)

    columns = [
        "ts", "step", "router_replay_mode",
        "logprobs_diff_mean", "logprobs_diff_max", "logprobs_diff_std",
        "logprobs_diff_valid", "actor_rollout_pearson", "num_samples",
    ]
    _write_csv_header(out_csv, columns)

    values: list[float] = []

    for step in range(int(args.steps)):
        sample_req = {
            "sampling_session_id": sampling_session_id,
            "seq_id": int(step),
            "num_samples": int(args.num_samples),
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
            "sampling_params": {
                "max_tokens": int(args.max_tokens),
                "temperature": float(args.temperature),
                "top_k": int(args.top_k),
                "top_p": float(args.top_p),
            },
            "prompt_logprobs": True,
        }
        sample_fut = _post(f"{base_url}/api/v1/asample", headers, sample_req, timeout_s=30.0)
        request_id = sample_fut.get("request_id")
        if not request_id:
            raise RuntimeError(f"asample missing request_id: {sample_fut}")
        sample_out = _wait_future(
            base_url=base_url, headers=headers, request_id=request_id, timeout_s=float(args.timeout_s)
        )
        sequences = sample_out.get("sequences") or []
        if not sequences:
            raise RuntimeError(f"asample returned no sequences: {sample_out}")
        if len(sequences) < int(args.min_batch_size):
            raise RuntimeError(
                f"need >= {args.min_batch_size} samples to validate micro-batch; got {len(sequences)}"
            )

        prompt_logprobs = sample_out.get("prompt_logprobs")
        if not isinstance(prompt_logprobs, list):
            raise RuntimeError("prompt_logprobs missing; set prompt_logprobs=true in sampling request")
        if len(prompt_logprobs) != len(prompt_tokens):
            raise RuntimeError(
                f"prompt_logprobs len {len(prompt_logprobs)} != prompt_len {len(prompt_tokens)}"
            )

        data_items = []
        for idx, seq in enumerate(sequences):
            gen_tokens = seq.get("tokens")
            seq_logprobs = seq.get("logprobs")
            routed_experts = seq.get("routed_experts")

            if not isinstance(gen_tokens, list) or not gen_tokens:
                raise RuntimeError(f"invalid tokens in sequence[{idx}]: {gen_tokens!r}")
            if not isinstance(seq_logprobs, list) or len(seq_logprobs) != len(gen_tokens):
                raise RuntimeError(
                    f"sequence[{idx}] logprobs invalid: len={len(seq_logprobs) if isinstance(seq_logprobs, list) else None}"
                )

            full_tokens = list(prompt_tokens) + list(gen_tokens)
            full_logprobs_raw = _normalize_logprobs(list(prompt_logprobs) + list(seq_logprobs))
            if len(full_logprobs_raw) != len(full_tokens):
                raise RuntimeError(
                    f"full_logprobs len {len(full_logprobs_raw)} != tokens len {len(full_tokens)}"
                )

            aligned_logprobs = _align_logprobs(full_logprobs_raw, mode=str(args.logprobs_align))
            prompt_len = len(prompt_tokens)
            response_len = len(gen_tokens)
            weights = _build_loss_mask(prompt_len, response_len, len(full_tokens))
            advantages = [1.0 if w != 0.0 else 0.0 for w in weights]

            loss_fn_inputs: dict[str, Any] = {
                "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
                "logprobs": {"data": aligned_logprobs, "shape": [len(aligned_logprobs)], "dtype": "float32"},
                "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
            }
            if routed_experts is not None:
                flat, shape = _flatten_routed_experts(routed_experts, expected_seq_len=None)
                loss_fn_inputs["routed_experts"] = {"data": flat, "shape": shape, "dtype": "int64"}

            data_items.append({
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": full_tokens}]},
                "loss_fn_inputs": loss_fn_inputs,
            })

        train_step_req = {
            "type": "train_step",
            "model_id": model_id,
            "forward_backward_input": {
                "loss_fn": "ppo",
                "loss_fn_config": {"epsilon": float(args.epsilon)},
                "data": data_items,
            },
            "adam_params": {"learning_rate": float(args.learning_rate)},
        }
        train_fut = _post(f"{base_url}/api/v1/train_step", headers, train_step_req, timeout_s=30.0)
        train_request_id = train_fut.get("request_id")
        if not train_request_id:
            raise RuntimeError(f"train_step missing request_id: {train_fut}")
        train_out = _wait_future(
            base_url=base_url, headers=headers, request_id=train_request_id, timeout_s=float(args.timeout_s)
        )
        metrics = train_out.get("metrics") or {}

        diff_mean = metrics.get("training/rollout_probs_diff_mean:mean")
        diff_max = metrics.get("training/rollout_probs_diff_max:mean")
        diff_std = metrics.get("training/rollout_probs_diff_std:mean")
        diff_valid = metrics.get("training/rollout_probs_diff_valid:mean")
        pearson = metrics.get("training/rollout_actor_probs_pearson_corr:mean")

        if diff_mean is None:
            raise RuntimeError(
                f"missing training/rollout_probs_diff_mean:mean in metrics (keys={sorted(metrics.keys())})"
            )

        values.append(float(diff_mean))
        row = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "step": int(step),
            "router_replay_mode": router_replay_mode,
            "logprobs_diff_mean": float(diff_mean),
            "logprobs_diff_max": float(diff_max) if diff_max is not None else "",
            "logprobs_diff_std": float(diff_std) if diff_std is not None else "",
            "logprobs_diff_valid": int(diff_valid) if diff_valid is not None else "",
            "actor_rollout_pearson": float(pearson) if pearson is not None else "",
            "num_samples": len(sequences),
        }
        _append_csv_row(out_csv, row, columns)
        print(
            f"step={step} logprobs_diff_mean={float(diff_mean):.6f} "
            f"logprobs_diff_max={float(diff_max) if diff_max is not None else 'NA'}",
            flush=True,
        )

    mean_val = sum(values) / max(len(values), 1)
    print(f"done: mean logprobs_diff_mean={mean_val:.6f}", flush=True)
    print(f"csv: {out_csv}", flush=True)
    print(f"meta: {meta_path}", flush=True)

    if args.plot:
        out_png = Path(args.plot)
        _plot_csv([out_csv], [str(router_replay_mode)], out_png)
        print(f"plot: {out_png}", flush=True)

    # Auto-compare: find latest R3 and no-R3 results and generate comparison plot
    if not args.out and not args.run_dir:  # Only auto-compare for default output paths
        _auto_compare_latest(base_model)

    return 0


def _auto_compare_latest(base_model: str) -> None:
    """Find latest R3 and no-R3 results and generate comparison plot."""
    results_dir = Path("results") / "r3_validation"
    if not results_dir.exists():
        return

    # Find latest R3 and no-R3 directories
    model_suffix = base_model.replace('/', '_')
    r3_dir = results_dir / "r3"
    no_r3_dir = results_dir / "no_r3"

    latest_r3 = None
    latest_no_r3 = None

    # Find latest R3 result
    if r3_dir.exists():
        r3_runs = sorted([d for d in r3_dir.iterdir() if d.is_dir()], reverse=True)
        for run_dir in r3_runs:
            csv_file = run_dir / f"curve_{model_suffix}.csv"
            if csv_file.exists():
                latest_r3 = csv_file
                break

    # Find latest no-R3 result
    if no_r3_dir.exists():
        no_r3_runs = sorted([d for d in no_r3_dir.iterdir() if d.is_dir()], reverse=True)
        for run_dir in no_r3_runs:
            csv_file = run_dir / f"curve_{model_suffix}.csv"
            if csv_file.exists():
                latest_no_r3 = csv_file
                break

    # Generate comparison if both found
    if latest_r3 and latest_no_r3:
        print("\n" + "=" * 80)
        print("AUTO-COMPARISON: Latest R3 vs No-R3 Results")
        print("=" * 80)
        comparison_png = results_dir / f"latest_comparison_{model_suffix}.png"

        try:
            # Read and compare
            def _read_vals(path: Path) -> list[float]:
                with path.open("r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                    idx_val = header.index("logprobs_diff_mean")
                    vals = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        vals.append(float(parts[idx_val]))
                return vals

            r3_vals = _read_vals(latest_r3)
            no_r3_vals = _read_vals(latest_no_r3)

            if r3_vals and no_r3_vals:
                r3_mean = sum(r3_vals) / len(r3_vals)
                no_r3_mean = sum(no_r3_vals) / len(no_r3_vals)
                improvement_pct = ((no_r3_mean - r3_mean) / no_r3_mean * 100) if no_r3_mean != 0 else 0

                print(f"R3:     {latest_r3}")
                print(f"No-R3:  {latest_no_r3}")
                print(f"\nR3 Mean:     {r3_mean:.6f}")
                print(f"No-R3 Mean:  {no_r3_mean:.6f}")
                print(f"Improvement: {improvement_pct:+.2f}% {'(R3 is better)' if improvement_pct > 0 else '(No-R3 is better)'}")

                # Generate plot
                _plot_csv([latest_r3, latest_no_r3], ["With R3", "Without R3"], comparison_png)
                print(f"\nComparison plot: {comparison_png}")
                print("=" * 80 + "\n")
        except Exception as e:
            print(f"Auto-comparison failed: {e}", flush=True)
    else:
        if latest_r3:
            print(f"\nFound latest R3 result: {latest_r3}")
        if latest_no_r3:
            print(f"Found latest No-R3 result: {latest_no_r3}")
        if not (latest_r3 and latest_no_r3):
            print("Auto-comparison skipped: need both R3 and No-R3 results")


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def compare(args: argparse.Namespace) -> int:
    """Compare two CSV curves produced by `run`."""
    left = Path(args.left)
    right = Path(args.right)
    if not left.exists() or not right.exists():
        raise SystemExit("compare: missing csv path")

    def _read_vals(path: Path) -> tuple[list[float], dict]:
        """Read values and metadata from CSV."""
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx_val = header.index("logprobs_diff_mean")
            idx_mode = header.index("router_replay_mode") if "router_replay_mode" in header else None
            vals = []
            mode = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                vals.append(float(parts[idx_val]))
                if idx_mode is not None and mode is None:
                    mode = parts[idx_mode]
        return vals, {"router_replay_mode": mode or "unknown"}

    left_vals, left_meta = _read_vals(left)
    right_vals, right_meta = _read_vals(right)
    if not left_vals or not right_vals:
        raise SystemExit("compare: empty csv")

    # Compute statistics
    import math
    left_mean = sum(left_vals) / len(left_vals)
    right_mean = sum(right_vals) / len(right_vals)
    left_std = math.sqrt(sum((x - left_mean) ** 2 for x in left_vals) / len(left_vals))
    right_std = math.sqrt(sum((x - right_mean) ** 2 for x in right_vals) / len(right_vals))
    left_min = min(left_vals)
    left_max = max(left_vals)
    right_min = min(right_vals)
    right_max = max(right_vals)

    ratio = right_mean / left_mean if left_mean != 0 else float("inf")
    improvement_pct = ((left_mean - right_mean) / left_mean * 100) if left_mean != 0 else 0

    # Determine labels based on metadata
    left_label = f"Left ({left_meta['router_replay_mode']})"
    right_label = f"Right ({right_meta['router_replay_mode']})"

    # Auto-detect R3 vs No-R3 for better labeling
    if left_meta['router_replay_mode'] == 'R3' and right_meta['router_replay_mode'] != 'R3':
        left_label = "With R3"
        right_label = "Without R3"
    elif right_meta['router_replay_mode'] == 'R3' and left_meta['router_replay_mode'] != 'R3':
        left_label = "Without R3"
        right_label = "With R3"

    # Print detailed comparison
    print("=" * 80)
    print("R3 ROUTER REPLAY COMPARISON REPORT")
    print("=" * 80)
    print(f"\nLeft:  {left}")
    print(f"Right: {right}")
    print("\n" + "-" * 80)
    print(f"{'Metric':<25} {'Left':<20} {'Right':<20} {'Difference':<15}")
    print("-" * 80)
    print(f"{'Mean':<25} {left_mean:<20.6f} {right_mean:<20.6f} {right_mean - left_mean:<15.6f}")
    print(f"{'Std Dev':<25} {left_std:<20.6f} {right_std:<20.6f} {right_std - left_std:<15.6f}")
    print(f"{'Min':<25} {left_min:<20.6f} {right_min:<20.6f} {right_min - left_min:<15.6f}")
    print(f"{'Max':<25} {left_max:<20.6f} {right_max:<20.6f} {right_max - left_max:<15.6f}")
    print("-" * 80)
    print(f"\nRatio (Right/Left): {ratio:.3f}")
    print(f"Improvement: {improvement_pct:+.2f}% {'(Right is better)' if improvement_pct > 0 else '(Left is better)'}")
    print("=" * 80 + "\n")

    if args.plot:
        out_png = Path(args.plot)
        _plot_csv([left, right], [left_label, right_label], out_png)
        print(f"Plot saved: {out_png}", flush=True)

    return 0


# ---------------------------------------------------------------------------
# Argument parsing + entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="R3 router replay validation tool (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- smoke ---------------------------------------------------------------
    smoke_p = sub.add_parser("smoke", help="Single-shot R3 validation via train_step")
    smoke_p.add_argument("--base-url", default=None)
    smoke_p.add_argument("--api-key", default=None)
    smoke_p.add_argument("--base-model", default=None, help="MoE base model (auto-pick if omitted)")
    smoke_p.add_argument("--prompt-tokens", default="1,2,3,4", help="Comma-separated token ids")
    smoke_p.add_argument("--max-tokens", type=int, default=8)
    smoke_p.add_argument("--num-samples", type=int, default=2)
    smoke_p.add_argument("--lora-rank", type=int, default=8)
    smoke_p.add_argument("--learning-rate", type=float, default=1e-4)
    smoke_p.add_argument("--epsilon", type=float, default=0.2)
    smoke_p.add_argument("--timeout-s", type=float, default=900.0)
    smoke_p.add_argument("--r3-mean-max", type=float, default=0.005, help="max allowed r3 rollout_probs_diff_mean")
    smoke_p.add_argument("--min-improve", type=float, default=1e-3, help="min improvement over no-R3")
    smoke_p.add_argument("--skip-permuted", action="store_true")

    # -- run -----------------------------------------------------------------
    run_p = sub.add_parser("run", help="Multi-step PPO training loop, records diff curve to CSV")
    run_p.add_argument("--base-url", default=None)
    run_p.add_argument("--api-key", default=None)
    run_p.add_argument("--base-model", default=None, help="MoE base model")
    run_p.add_argument("--prompt-tokens", default="1,2,3,4,5,6,7,8", help="Comma-separated token ids")
    run_p.add_argument("--steps", type=int, default=50)
    run_p.add_argument("--num-samples", type=int, default=8)
    run_p.add_argument("--min-batch-size", type=int, default=2)
    run_p.add_argument("--max-tokens", type=int, default=64)
    run_p.add_argument("--temperature", type=float, default=0.7)
    run_p.add_argument("--top-k", type=int, default=-1)
    run_p.add_argument("--top-p", type=float, default=1.0)
    run_p.add_argument("--logprobs-align", choices=("shifted", "token"), default="shifted")
    run_p.add_argument("--lora-rank", type=int, default=8)
    run_p.add_argument("--learning-rate", type=float, default=1e-4)
    run_p.add_argument("--epsilon", type=float, default=0.2)
    run_p.add_argument("--timeout-s", type=float, default=900.0)
    run_p.add_argument("--run-dir", default=None)
    run_p.add_argument("--out", default=None)
    run_p.add_argument("--plot", default=None, help="Optional PNG path to save curve")

    # -- compare -------------------------------------------------------------
    cmp_p = sub.add_parser("compare", help="Compare two CSV curves from `run`")
    cmp_p.add_argument("--left", required=True)
    cmp_p.add_argument("--right", required=True)
    cmp_p.add_argument("--plot", default=None)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "smoke":
        return smoke(args)
    if args.cmd == "run":
        return run(args)
    if args.cmd == "compare":
        return compare(args)
    raise SystemExit("unknown command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
