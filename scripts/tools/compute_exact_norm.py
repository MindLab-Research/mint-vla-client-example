#!/usr/bin/env python3
"""Compute exact OpenPI norm stats for a selected MANO row population."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import openpi_vla_smoke_lance_base as L
import train_cube1_01_compare as C


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--gesture-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--action-horizon", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    norm_path = args.output_dir / "norm_stats.json"
    if norm_path.exists():
        raise FileExistsError(f"refusing to overwrite {norm_path}")

    rows = [int(value) for value in args.rows_csv.read_text().strip().split(",") if value]
    dataset = C.SelectedLanceDataset(
        args.dataset,
        row_indices=rows,
        action_horizon=args.action_horizon,
        frame_window="contact",
        contact_context_frames=100,
        contact_window_manifest=args.contact_manifest,
        missing_contact_policy="error",
        action_source="urdf_target_absolute",
        language_conditioning="gesture",
        gesture_index=args.gesture_index,
        target_lance_dataset=args.target_dataset,
        extended_state=True,
    )
    print(
        json.dumps(
            {
                "phase": "dataset_ready",
                "rows": len(rows),
                "selected_query_frames": len(dataset),
                "window_summary": dataset.window_summary(),
            }
        ),
        flush=True,
    )

    stats = C.selected_norm_stats(dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    L.normalize.save(args.output_dir, stats)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    contract = {
        "contract_version": 1,
        "selection_id": args.selection_id,
        "client_git_commit": commit,
        "image_dataset": str(args.dataset),
        "target_dataset": str(args.target_dataset),
        "target_field": "hands[].urdf_dof_target",
        "row_indices_file": str(args.rows_csv),
        "row_indices_sha256": hashlib.sha256(
            ",".join(map(str, rows)).encode()
        ).hexdigest(),
        "row_count": len(rows),
        "row_min": min(rows),
        "row_max": max(rows),
        "selected_query_frames": len(dataset),
        "action_horizon": args.action_horizon,
        "action_norm_vectors": len(dataset) * args.action_horizon,
        "state_contract": "mano_five_finger_contact_lift_v1",
        "extended_state": True,
        "action_source": "urdf_target_absolute",
        "action_semantics": (
            "xyz/finger target[t+k]-query_state[t]; "
            "Euler absolute target[t+k]; pad zero"
        ),
        "action_norm_population": (
            "exact contact-selected query frame x horizon "
            "with selected-window repeat padding"
        ),
        "delta_mask": "make_bool_mask(3,-3,20,-6)",
        "contact_manifest": str(args.contact_manifest),
        "contact_manifest_sha256": _sha256(args.contact_manifest),
        "gesture_index": str(args.gesture_index),
        "gesture_index_sha256": _sha256(args.gesture_index),
        "language_conditioning": "gesture",
        "prompt_format": "pick up the <object> using gesture <NN>",
        "norm_stats_sha256": _sha256(norm_path),
        "window_summary": dataset.window_summary(),
    }
    (args.output_dir / "data_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    print(json.dumps(contract, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
