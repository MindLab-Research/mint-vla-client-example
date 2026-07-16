#!/usr/bin/env python3
"""pi0.5 逐帧推理，把预测 action 合并回原 Lance 结构，写成新 Lance 数据集。

This is the *productized*, importable counterpart to
``scripts/wip/openpi_vla_merge_infer_lance.py`` (identical output schema and
correctness semantics -- normalized-space MSE, quantile/z-score
unnormalization), split so it can be invoked either standalone against an
already-saved checkpoint, or as an optional post-training step from
``scripts/tools/openpi_vla_lora_finetune.py``.

Output schema (identical to the wip script): original Lance columns are kept
unchanged; four new per-frame columns are appended, parallel to the original
`actions` column (outer length == total_frames):

  pred_actions            : per-frame [action_horizon, action_dim] prediction window (normalized space)
  pred_actions_physical   : unnormalized back to physical units (same space as the original `actions`)
  pred_action_mse         : per-frame pred vs ground-truth MSE (normalized space)
  pred_meta (episode-level): model_path / action_horizon / action_dim

Frames that weren't inferred (shouldn't happen under normal use) are padded
with NaN, not silently dropped -- this keeps every row's column lengths
consistent, which Lance/Arrow requires.

Standalone usage:
    python scripts/tools/openpi_vla_infer_to_lance.py \\
        --model-path mint://<model_id>/sampler_weights/<name> \\
        --owner-id <owner_id> \\
        --lance-dataset /path/to/data.lance \\
        --output-lance /path/to/output_merged.lance

As a library (used by openpi_vla_lora_finetune.py):
    from openpi_vla_infer_to_lance import run_inference_and_merge_to_lance
    result = run_inference_and_merge_to_lance(
        base_url, headers, base_model=..., model_path=..., owner_id=...,
        dataset=dataset, data_config=data_config,
        lance_dataset_path=..., output_lance_path=...,
    )
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_SCRIPT_PATH = _REPO_ROOT / "scripts" / "wip" / "openpi_vla_smoke_lance.py"
_INFER_OBS_SCRIPT_PATH = _REPO_ROOT / "scripts" / "wip" / "openpi_vla_infer_obs.py"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_module_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke = _load_module_by_path("openpi_vla_smoke_lance", _SMOKE_SCRIPT_PATH)
_obs = _load_module_by_path("openpi_vla_infer_obs", _INFER_OBS_SCRIPT_PATH)

_headers = _smoke._headers
_post_json = _smoke._post_json
_await_result = _smoke._await_result
_build_batch = _smoke._build_batch
_unnormalize_actions = _obs._unnormalize_actions


def _f32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def run_inference_and_merge_to_lance(
    base_url: str,
    headers: dict[str, str],
    *,
    base_model: str,
    model_path: str,
    owner_id: str | None,
    dataset: Any,
    data_config: Any,
    action_horizon: int,
    lance_dataset_path: str,
    output_lance_path: str,
) -> dict[str, Any]:
    """Run inference on every frame in `dataset`, merge predictions back into
    the original Lance structure, write `output_lance_path`.

    Creates its own throwaway action_session and cleans it up before
    returning (even on error). Returns a small summary dict; the actual
    merged dataset is the side effect written to `output_lance_path`.
    """
    per_row_pred_norm: dict[int, dict[int, np.ndarray]] = {}
    per_row_pred_phys: dict[int, dict[int, np.ndarray]] = {}
    per_row_mse: dict[int, dict[int, float]] = {}
    action_dim_seen = 0
    action_session_id = ""
    try:
        created = _post_json(
            base_url,
            "/api/v1/mint/action_sessions",
            headers,
            {
                "session_id": f"vla-lora-mergeinfer-{uuid.uuid4().hex[:12]}",
                "base_model": base_model,
                "model_path": model_path,
                "owner_id": owner_id,
            },
        )
        action_session_id = created["action_session_id"]

        for idx in range(len(dataset)):
            row_index, frame = dataset._index[idx]
            datum = _build_batch(dataset, data_config, base_model=base_model, indices=[idx])[0]
            obs = datum["observation"]
            gt_norm = np.asarray(datum["supervision"]["actions"]["data"], dtype=np.float32).reshape(
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
            a = act.get("actions") or {}
            pred_norm = np.asarray(a["data"], dtype=np.float32).reshape(a["shape"])
            action_dim_seen = int(pred_norm.shape[1])

            h = min(pred_norm.shape[0], gt_norm.shape[0])
            d = min(pred_norm.shape[1], gt_norm.shape[1])
            mse = float(np.mean((pred_norm[:h, :d] - gt_norm[:h, :d]) ** 2))
            pred_phys = _unnormalize_actions(pred_norm, data_config)

            per_row_pred_norm.setdefault(row_index, {})[frame] = _f32(pred_norm)
            per_row_pred_phys.setdefault(row_index, {})[frame] = _f32(pred_phys)
            per_row_mse.setdefault(row_index, {})[frame] = mse
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

    num_frames_written = _write_merged(
        dataset=dataset,
        lance_dataset_path=lance_dataset_path,
        output_lance_path=output_lance_path,
        model_path=model_path,
        action_horizon=action_horizon,
        per_row_pred_norm=per_row_pred_norm,
        per_row_pred_phys=per_row_pred_phys,
        per_row_mse=per_row_mse,
        action_dim=action_dim_seen,
    )

    return {
        "action_session_id": action_session_id,
        "output_lance_path": output_lance_path,
        "num_frames_inferred": sum(len(v) for v in per_row_pred_norm.values()),
        "num_frames_written": num_frames_written,
    }


def _write_merged(
    *,
    dataset: Any,
    lance_dataset_path: str,
    output_lance_path: str,
    model_path: str,
    action_horizon: int,
    per_row_pred_norm: dict[int, dict[int, np.ndarray]],
    per_row_pred_phys: dict[int, dict[int, np.ndarray]],
    per_row_mse: dict[int, dict[int, float]],
    action_dim: int,
) -> int:
    """Read the original Lance table in full, append per-frame prediction
    columns parallel to `actions`, write a new Lance dataset.

    All original columns (image/wrist_image/state/actions/mujoco/camera/...)
    are kept unchanged. Prediction columns have outer length == total_frames;
    a frame that wasn't inferred (shouldn't happen in normal use) is padded
    with NaN so every row's column lengths stay consistent.
    """
    src = lance.dataset(str(lance_dataset_path))
    table = src.to_table()
    n_rows = table.num_rows
    horizon = int(action_horizon)

    pred_norm_col: list[list[list[float]]] = []
    pred_phys_col: list[list[list[float]]] = []
    mse_col: list[list[float]] = []
    meta_col: list[dict[str, Any]] = []
    total_frames_written = 0

    for row_index in range(n_rows):
        total_frames = int(dataset._rows[row_index]["episode_metadata"]["total_frames"])
        pn = per_row_pred_norm.get(row_index, {})
        pp = per_row_pred_phys.get(row_index, {})
        ms = per_row_mse.get(row_index, {})
        nan_win = [[float("nan")] * action_dim for _ in range(horizon)]
        row_norm, row_phys, row_mse = [], [], []
        for f in range(total_frames):
            if f in pn:
                row_norm.append([[float(x) for x in step] for step in pn[f].tolist()])
                row_phys.append([[float(x) for x in step] for step in pp[f].tolist()])
                row_mse.append(float(ms[f]))
                total_frames_written += 1
            else:
                row_norm.append(copy.deepcopy(nan_win))
                row_phys.append(copy.deepcopy(nan_win))
                row_mse.append(float("nan"))
        pred_norm_col.append(row_norm)
        pred_phys_col.append(row_phys)
        mse_col.append(row_mse)
        meta_col.append(
            {
                "model_path": str(model_path),
                "action_horizon": horizon,
                "action_dim": int(action_dim),
            }
        )

    # One element per episode; inner shape is [frame][horizon_step][action_dim].
    win_ty = pa.list_(pa.list_(pa.list_(pa.float32())))
    meta_ty = pa.struct(
        [
            ("model_path", pa.string()),
            ("action_horizon", pa.int32()),
            ("action_dim", pa.int32()),
        ]
    )
    out = table
    out = out.append_column("pred_actions", pa.array(pred_norm_col, type=win_ty))
    out = out.append_column("pred_actions_physical", pa.array(pred_phys_col, type=win_ty))
    out = out.append_column("pred_action_mse", pa.array(mse_col, type=pa.list_(pa.float32())))
    out = out.append_column("pred_meta", pa.array(meta_col, type=meta_ty))

    lance.write_dataset(out, output_lance_path, mode="overwrite")
    return total_frames_written


def main() -> int:
    default_ds = (
        "/vePFS-Mindverse/user/intern/wenxi/pi-finetune/data_source/lance/pi_video_streams_lance_smoke.lance"
    )
    p = argparse.ArgumentParser(description="pi0.5 inference results merged back into the original Lance dataset")
    p.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("MINT_API_KEY", "dummy"))
    p.add_argument("--base-model", default=_smoke.PI05_MODEL)
    p.add_argument("--model-path", default=os.environ.get("MINT_SAMPLER_PATH", ""), required=False)
    p.add_argument("--owner-id", default=os.environ.get("MINT_SAMPLER_OWNER", "anonymous"))
    p.add_argument("--lance-dataset", default=os.environ.get("MINT_LANCE_DATASET", default_ds))
    p.add_argument("--action-horizon", type=int, default=10)
    p.add_argument("--output-lance", required=True)
    args = p.parse_args()

    if not args.model_path:
        print("error: --model-path is required (the mint:// path from save_weights_for_sampler)", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    headers = _headers(args.api_key)

    dataset = _smoke.LanceViewpi05Dataset(Path(args.lance_dataset), action_horizon=args.action_horizon)
    model_cfg = _smoke._build_model_config(args.action_horizon)
    norm_stats = _smoke._compute_norm_stats(dataset)
    data_config = _smoke._make_data_config(model_cfg, norm_stats)
    print(f"lance_in={args.lance_dataset}  episodes={len(dataset._rows)}  frames={len(dataset)}")

    result = run_inference_and_merge_to_lance(
        base_url,
        headers,
        base_model=args.base_model,
        model_path=args.model_path,
        owner_id=args.owner_id,
        dataset=dataset,
        data_config=data_config,
        action_horizon=args.action_horizon,
        lance_dataset_path=args.lance_dataset,
        output_lance_path=args.output_lance,
    )
    print(f"OK: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
