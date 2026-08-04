#!/usr/bin/env python3
"""Validate the four-row train-only replay-State54 snapshot Mode4 gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.train.train_cube1_01_compare import SelectedLanceDataset

ROWS = [7, 8, 619, 620]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--data-contract-sha256", required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--norm-sha256", required=True)
    parser.add_argument("--train-rows-csv", type=Path, required=True)
    parser.add_argument("--train-rows-csv-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contact-window-manifest", type=Path, required=True)
    parser.add_argument("--gesture-index", type=Path, required=True)
    parser.add_argument("--feature-release", type=Path, required=True)
    parser.add_argument("--feature-release-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.data_contract) != args.data_contract_sha256:
        raise ValueError("data contract SHA mismatch")
    if sha256(args.norm_stats) != args.norm_sha256:
        raise ValueError("norm SHA mismatch")
    if sha256(args.train_rows_csv) != args.train_rows_csv_sha256:
        raise ValueError("train rows SHA mismatch")
    contract = json.loads(args.data_contract.read_text())
    if contract.get("status") != "accepted" or contract.get("contract_id") != "state54_replay_train_only_noaug_v1" or contract.get("norm_stats_sha256") != args.norm_sha256 or contract.get("feature_release_sha256") != args.feature_release_sha256:
        raise ValueError("wrong formal State54 data contract")
    train_rows = [int(value) for value in args.train_rows_csv.read_text().strip().split(",") if value]
    if len(train_rows) != 813:
        raise ValueError("formal normalization population must contain813 rows")
    summary = json.loads(args.summary.read_text())
    actual_rows = [int(item["row_index"]) for item in summary["rows"]]
    if actual_rows != ROWS:
        raise ValueError(f"unexpected Mode4 rows: {actual_rows}")
    if summary.get("video_mode") != "none" or summary.get("state_contract") != "mano_object_dynamics_state54_v1":
        raise ValueError("summary state/video contract mismatch")
    normalization_rows = [int(value) for value in summary["normalization_row_indices"]]
    if normalization_rows != train_rows:
        raise ValueError("Mode4 normalization rows differ from formal train split")
    if summary.get("norm_sha_expected") != args.norm_sha256 or summary.get("norm_sha_actual") != args.norm_sha256:
        raise ValueError("Mode4 norm SHA mismatch")
    if summary.get("action_session", {}).get("retained") is not False:
        raise ValueError("Mode4 action session was retained")
    dataset = SelectedLanceDataset(
        args.dataset, row_indices=ROWS, action_horizon=10,
        frame_window="contact", contact_context_frames=60,
        contact_window_manifest=args.contact_window_manifest,
        missing_contact_policy="error", action_source="urdf_target_absolute",
        language_conditioning="gesture", gesture_index=args.gesture_index,
        target_lance_dataset=args.dataset,
        state_contract="mano_object_dynamics_state54_v1",
        state54_replay_feature_release=args.feature_release,
        state54_replay_feature_release_sha256=args.feature_release_sha256,
    )
    rows_report = []
    query_count = 0
    for local_row, row_summary in enumerate(summary["rows"]):
        row = int(row_summary["row_index"])
        result = row_summary["results"][0]
        if result.get("row_index") != row or result.get("trajectory_frame_count") != 7 or result.get("pred_has_nan_inf") is not False:
            raise ValueError(f"row{row} rollout contract mismatch")
        if result.get("state54_data_contract", {}).get("sha256") != args.data_contract_sha256:
            raise ValueError(f"row{row} data contract mismatch")
        initialization = result["initialization"]
        if initialization.get("initialization_mode") != "accepted_pose_backward_qvel_snapshot_v1" or initialization.get("qvel_source") != "mj_differentiatePos_previous_to_current_5ms" or initialization.get("source_frame") != result["frame_window"]["start_frame"]:
            raise ValueError(f"row{row} initialization mismatch")
        if result.get("state_contract") != "mano_object_dynamics_state54_v1" or result.get("closed_loop") is not True:
            raise ValueError(f"row{row} is not closed-loop State54")
        queries = result.get("query_timings", [])
        if len(queries) != 2:
            raise ValueError(f"row{row} expected two queries")
        for query in queries:
            query_count += 1
            if query.get("used_data_sharding") is not True or query.get("request_batch_size") != 4 or query.get("response_batch_size") != 4 or query.get("actual_observation_count") != 4 or query.get("padding_count") != 0:
                raise ValueError(f"row{row} batch4/no-padding gate failed")
        for name, path in result["arrays"].items():
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if not np.isfinite(values).all():
                raise ValueError(f"row{row} array {name} is nonfinite")
        observation = np.load(result["arrays"]["rollout_observation_state"], allow_pickle=False)
        expected = np.asarray(dataset.load_state54_numeric_row(local_row)["_state54_window"][0], dtype=np.float32)
        delta = np.abs(observation[0] - expected)
        maximum = float(delta.max())
        if observation.shape[1:] != (54,) or expected.shape != (54,) or maximum > 2e-5 or abs(float(observation[0, 52])) > 1e-7:
            raise ValueError(f"row{row} first State54 mismatch: {maximum}")
        rows_report.append({
            "row_index": row,
            "object_name": result["object_name"],
            "window_start": result["frame_window"]["start_frame"],
            "first_state_max_abs_error": maximum,
            "first_state_max_abs_error_dimension": int(delta.argmax()),
            "array_count": len(result["arrays"]),
            "all_arrays_finite": True,
            "query_count": len(queries),
            "padding_count": 0,
        })
    report = {
        "schema_version": 1,
        "protocol": "state54_replay_train_only_mode4_gate_v1",
        "status": "accepted",
        "summary": str(args.summary.resolve()),
        "summary_sha256": sha256(args.summary),
        "rows": rows_report,
        "row_count": len(rows_report),
        "query_count_row_duplicated": query_count,
        "all_queries_fixed_batch4_sharded": True,
        "all_arrays_finite": True,
        "first_observation_matches_training_state54": True,
        "first_observation_tolerance": 2e-5,
        "state54_data_contract_sha256": args.data_contract_sha256,
        "norm_sha256": args.norm_sha256,
        "feature_release_sha256": args.feature_release_sha256,
        "train_rows_csv_sha256": args.train_rows_csv_sha256,
        "action_session_released": True,
        "claim_scope": "State54 train-only integration/parity only; not grasp quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    print("validation_sha256", sha256(args.output))


if __name__ == "__main__":
    main()
