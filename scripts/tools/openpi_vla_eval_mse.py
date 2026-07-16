#!/usr/bin/env python3
"""pi0.5 量化评估：对已落盘 sampler 权重推理，与 Lance 真值动作算 MSE/L1。

This is the *productized*, importable counterpart to
``scripts/wip/openpi_vla_eval_lance.py`` (same algorithm, same
normalized-space comparison semantics), split so the core evaluation logic
can be imported by ``scripts/tools/openpi_vla_lora_finetune.py`` as an
optional post-training step, while still being runnable standalone against
any already-saved checkpoint.

Correctness invariant (identical to the wip script): the worker's
``act()`` does NOT unnormalize, and the driver's ground truth goes through
the same ``Normalize`` transform as training -- so pred and gt are compared
in **normalized space**. Do not unnormalize here; doing so would make the
MSE numbers incomparable to the loss values logged during training.

Standalone usage:
    python scripts/tools/openpi_vla_eval_mse.py \\
        --model-path mint://<model_id>/sampler_weights/<name> \\
        --owner-id <owner_id> \\
        --lance-dataset /path/to/data.lance \\
        --indices 0,1,2,5,10 \\
        --output-json /tmp/eval.json

As a library (used by openpi_vla_lora_finetune.py):
    from openpi_vla_eval_mse import run_mse_evaluation
    result = run_mse_evaluation(
        base_url, headers, base_model=..., model_path=..., owner_id=...,
        dataset=dataset, data_config=data_config, indices=[0, 1, 2],
    )
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_SCRIPT_PATH = _REPO_ROOT / "scripts" / "wip" / "openpi_vla_smoke_lance.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("openpi_vla_smoke_lance", _SMOKE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load smoke script module from {_SMOKE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke = _load_smoke_module()
_headers = _smoke._headers
_post_json = _smoke._post_json
_await_result = _smoke._await_result
_build_batch = _smoke._build_batch


def _parse_indices(spec: str) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return [0]
    return [int(t) for t in spec.split(",") if t.strip()] or [0]


def _actions_from_result(act: dict[str, Any]) -> np.ndarray | None:
    a = act.get("actions") if isinstance(act, dict) else None
    if not isinstance(a, dict):
        return None
    data = a.get("data")
    shape = a.get("shape")
    if not isinstance(data, list) or not isinstance(shape, list):
        return None
    return np.asarray(data, dtype=np.float32).reshape(shape)


def run_mse_evaluation(
    base_url: str,
    headers: dict[str, str],
    *,
    base_model: str,
    model_path: str,
    owner_id: str | None,
    dataset: Any,
    data_config: Any,
    indices: list[int],
) -> dict[str, Any]:
    """Create a throwaway action_session against `model_path`, run inference
    on each of `indices`, and compare against the dataset's ground-truth
    actions (both in normalized space -- see module docstring).

    Returns {"action_session_id", "per_sample": [...], "aggregate": {...}}.
    Cleans up the action_session it creates before returning, even on error.
    """
    action_session_id = ""
    per_sample: list[dict[str, Any]] = []
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    base_sq_sum = 0.0  # zero-prediction baseline: squared error of an all-zero prediction (== gt^2)
    elem_count = 0
    try:
        created = _post_json(
            base_url,
            "/api/v1/mint/action_sessions",
            headers,
            {
                "session_id": f"vla-lora-eval-{uuid.uuid4().hex[:12]}",
                "base_model": base_model,
                "model_path": model_path,
                "owner_id": owner_id,
            },
        )
        action_session_id = created["action_session_id"]

        for idx in indices:
            datum = _build_batch(dataset, data_config, base_model=base_model, indices=[idx])[0]
            obs = datum["observation"]
            gt = np.asarray(datum["supervision"]["actions"]["data"], dtype=np.float32).reshape(
                datum["supervision"]["actions"]["shape"]
            )

            act = _await_result(
                base_url,
                headers,
                _post_json(
                    base_url,
                    f"/api/v1/mint/action_sessions/{action_session_id}/act",
                    headers,
                    {"observation": obs},
                ),
            )
            pred = _actions_from_result(act)
            if pred is None:
                per_sample.append({"index": idx, "error": act})
                continue

            # Align on horizon/dim (take the shorter side, guards against a mismatch).
            h = min(pred.shape[0], gt.shape[0])
            d = min(pred.shape[1], gt.shape[1])
            pv, gv = pred[:h, :d], gt[:h, :d]
            diff = pv - gv
            mse = float(np.mean(diff**2))
            l1 = float(np.mean(np.abs(diff)))
            base_mse = float(np.mean(gv**2))
            has_nan = bool(np.isnan(pv).any() or np.isinf(pv).any())

            sq_err_sum += float(np.sum(diff**2))
            abs_err_sum += float(np.sum(np.abs(diff)))
            base_sq_sum += float(np.sum(gv**2))
            elem_count += pv.size

            per_sample.append(
                {
                    "index": idx,
                    "pred_shape": list(pred.shape),
                    "mse": mse,
                    "l1": l1,
                    "baseline_mse_zero": base_mse,
                    "improvement_vs_zero": (base_mse - mse),
                    "pred_has_nan_inf": has_nan,
                    "pred_range": [float(pv.min()), float(pv.max())],
                    "gt_range": [float(gv.min()), float(gv.max())],
                }
            )

        aggregate = {
            "num_samples": len([r for r in per_sample if "mse" in r]),
            "overall_mse": (sq_err_sum / elem_count) if elem_count else None,
            "overall_l1": (abs_err_sum / elem_count) if elem_count else None,
            "baseline_mse_zero": (base_sq_sum / elem_count) if elem_count else None,
        }
        if aggregate["overall_mse"] is not None and aggregate["baseline_mse_zero"]:
            aggregate["mse_vs_baseline_ratio"] = aggregate["overall_mse"] / aggregate["baseline_mse_zero"]

        return {
            "action_session_id": action_session_id,
            "per_sample": per_sample,
            "aggregate": aggregate,
        }
    finally:
        if action_session_id:
            try:
                requests.delete(
                    f"{base_url}/api/v1/mint/action_sessions/{action_session_id}",
                    headers=headers,
                    timeout=120.0,
                )
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    default_ds = (
        "/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_lance_smoke.lance"
    )
    p = argparse.ArgumentParser(description="pi0.5 sampler MSE/L1 quantitative evaluation")
    p.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("MINT_API_KEY", "dummy"))
    p.add_argument("--base-model", default=_smoke.PI05_MODEL)
    p.add_argument("--model-path", default=os.environ.get("MINT_SAMPLER_PATH", ""), required=False)
    p.add_argument("--owner-id", default=os.environ.get("MINT_SAMPLER_OWNER", "anonymous"))
    p.add_argument("--lance-dataset", default=os.environ.get("MINT_LANCE_DATASET", default_ds))
    p.add_argument("--action-horizon", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--indices", default="0")
    p.add_argument("--output-json", default="")
    args = p.parse_args()

    if not args.model_path:
        print("error: --model-path is required (the mint:// path from save_weights_for_sampler)", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    headers = _headers(args.api_key)
    indices = _parse_indices(args.indices)

    dataset = _smoke.LanceViewpi05Dataset(
        Path(args.lance_dataset), action_horizon=args.action_horizon, max_samples=args.max_samples
    )
    model_cfg = _smoke._build_model_config(args.action_horizon)
    norm_stats = _smoke._compute_norm_stats(dataset)
    data_config = _smoke._make_data_config(model_cfg, norm_stats)
    print(f"lance_dataset: {args.lance_dataset}  samples={len(dataset)}  eval_indices={indices}")

    result = run_mse_evaluation(
        base_url,
        headers,
        base_model=args.base_model,
        model_path=args.model_path,
        owner_id=args.owner_id,
        dataset=dataset,
        data_config=data_config,
        indices=indices,
    )

    for row in result["per_sample"]:
        print(json.dumps(row), flush=True)
    print("=== aggregate ===")
    print(json.dumps(result["aggregate"], indent=2))

    if args.output_json:
        payload = {
            "base_model": args.base_model,
            "model_path": args.model_path,
            "lance_dataset": args.lance_dataset,
            **result,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
