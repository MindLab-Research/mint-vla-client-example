#!/usr/bin/env python3
"""Smoke test for R3 router replay (sampling -> routed_experts -> train_step).

Runs locally against a dev server. Requires a MoE base model to be available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _base_url(args: argparse.Namespace) -> str:
    # Be defensive: always fall back to DEFAULT_BASE_URL and handle missing attributes.
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
    return r.json()


def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


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


def _flatten_routed_experts(routed: list, *, expected_seq_len: int) -> tuple[list[int], list[int]]:
    if not routed:
        raise ValueError("routed_experts is empty")
    if not isinstance(routed, list):
        raise ValueError(f"routed_experts must be list, got {type(routed)}")
    seq_len = len(routed)
    if seq_len != expected_seq_len:
        raise ValueError(f"routed_experts seq_len {seq_len} != tokens_len {expected_seq_len}")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-model", default=None, help="MoE base model name to test (auto-pick if omitted)")
    p.add_argument("--prompt-tokens", default="1,2,3,4", help="Comma-separated token ids")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--min-batch-size", type=int, default=2, help="Require >= this many samples to validate R3")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--timeout-s", type=float, default=900.0)
    return p.parse_args()


def _pick_min_moe_model(*, supported_models: list[str]) -> str:
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

    candidates: list[tuple[float, str]] = []
    for name, cfg in MODEL_CONFIGS.items():
        if not cfg.is_moe:
            continue
        if name in normalized_supported:
            candidates.append((float(cfg.num_parameters), name))

    if not candidates:
        raise RuntimeError(
            "No MoE model found in supported_models. "
            f"supported_models={supported_models}"
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def main() -> int:
    args = _parse_args()
    base_url = _base_url(args)
    headers = _headers(args)

    info = _get(f"{base_url}/api/v1/server_info", headers, timeout_s=10.0)
    config = info.get("config", {}) if isinstance(info, dict) else {}
    print(
        f"server_info router_replay_mode={config.get('router_replay_mode')} "
        f"enable_rollout_routing_replay={config.get('enable_rollout_routing_replay')}"
    )
    if config.get("router_replay_mode") != "R3":
        raise RuntimeError("router_replay_mode is not R3; enable R3 before testing")

    base_model = args.base_model
    if not base_model:
        caps = _get(f"{base_url}/api/v1/get_server_capabilities", headers, timeout_s=10.0)
        supported_models = caps.get("supported_models") or []
        base_model = _pick_min_moe_model(supported_models=supported_models)
        print(f"auto-selected MoE base_model={base_model}")

    session = _post(
        f"{base_url}/api/v1/create_session",
        headers,
        {
            "tags": ["scripts/wip/r3_router_replay_smoke.py"],
            "user_metadata": {},
            "sdk_version": "scripts/wip/r3_router_replay_smoke.py",
        },
        timeout_s=10.0,
    )
    session_id = session.get("session_id")
    if not session_id:
        raise RuntimeError(f"create_session missing session_id: {session}")

    sampling = _post(
        f"{base_url}/api/v1/create_sampling_session",
        headers,
        {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": base_model},
        timeout_s=60.0,
    )
    sampling_session_id = sampling.get("sampling_session_id")
    if not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {sampling}")

    prompt_tokens = _parse_tokens(args.prompt_tokens)
    if not prompt_tokens:
        raise RuntimeError("prompt_tokens is empty")

    sample_req = {
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "num_samples": int(args.num_samples),
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
        "sampling_params": {
            "max_tokens": int(args.max_tokens),
            "temperature": 0.7,
            "top_k": -1,
            "top_p": 1.0,
        },
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
            f"need >= {args.min_batch_size} samples to validate micro-batch replay; got {len(sequences)}"
        )

    data_items = []
    first_shape = None
    for idx, seq in enumerate(sequences):
        gen_tokens = seq.get("tokens")
        routed = seq.get("routed_experts")
        if routed is None:
            raise RuntimeError("routed_experts missing; vLLM patch likely not applied")
        if not isinstance(gen_tokens, list) or not gen_tokens:
            raise RuntimeError(f"invalid tokens in sequence[{idx}]: {gen_tokens!r}")
        # routed_experts covers input positions of the full sequence (prompt + generated),
        # which is (prompt_len + gen_len - 1) entries because the last generated token
        # is never processed as model input.
        full_tokens = list(prompt_tokens) + list(gen_tokens)
        # Validate routed_experts structure (don't enforce exact seq_len match here).
        flat, shape = _flatten_routed_experts(routed, expected_seq_len=len(routed))
        if first_shape is None:
            first_shape = shape
        elif shape[1:] != first_shape[1:]:
            raise RuntimeError(f"routed_experts shape mismatch across samples: {shape} vs {first_shape}")
        data_items.append(
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": full_tokens}]},
                "loss_fn_inputs": {
                    "weights": {"data": [1.0] * len(full_tokens), "shape": [len(full_tokens)], "dtype": "float32"},
                    "routed_experts": {"data": flat, "shape": shape, "dtype": "int64"},
                },
            }
        )
    print(f"routed_experts shape={first_shape} batch_size={len(data_items)}")

    create_model_fut = _post(
        f"{base_url}/api/v1/create_model",
        headers,
        {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": base_model,
            "lora_config": {"rank": int(args.lora_rank)},
        },
        timeout_s=10.0,
    )
    create_request_id = create_model_fut.get("request_id")
    if not create_request_id:
        raise RuntimeError(f"create_model missing request_id: {create_model_fut}")
    create_out = _wait_future(
        base_url=base_url, headers=headers, request_id=create_request_id, timeout_s=float(args.timeout_s)
    )
    model_id = create_out.get("model_id")
    if not model_id:
        raise RuntimeError(f"create_model missing model_id: {create_out}")

    train_step_req = {
        "type": "train_step",
        "model_id": model_id,
        "forward_backward_input": {
            "loss_fn": "cross_entropy",
            "data": data_items,
        },
        "adam_params": {"learning_rate": 1e-4},
    }
    train_fut = _post(f"{base_url}/api/v1/train_step", headers, train_step_req, timeout_s=30.0)
    train_request_id = train_fut.get("request_id")
    if not train_request_id:
        raise RuntimeError(f"train_step missing request_id: {train_fut}")
    train_out = _wait_future(
        base_url=base_url, headers=headers, request_id=train_request_id, timeout_s=float(args.timeout_s)
    )
    metrics = train_out.get("metrics") if isinstance(train_out, dict) else None
    print(f"train_step ok metrics_keys={sorted(metrics.keys()) if isinstance(metrics, dict) else None}")
    print("PASS: R3 replay ok (routed_experts present, batch_size>=2, train_step succeeded)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
