#!/usr/bin/env python3
"""Build the train-only replay-State54 contract for the State54 mainline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected.lower():
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def rows_csv(path: Path) -> list[int]:
    lines = path.read_text().strip().splitlines()
    if len(lines) != 1:
        raise ValueError("train rows CSV must contain exactly one line")
    rows = [int(value) for value in lines[0].split(",") if value]
    if len(rows) != 813 or len(set(rows)) != 813:
        raise ValueError("formal State54 train split must contain813 unique rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--source-release-sha256", required=True)
    parser.add_argument("--personal-data-release", type=Path, required=True)
    parser.add_argument("--personal-data-release-sha256", required=True)
    parser.add_argument("--feature-release", type=Path, required=True)
    parser.add_argument("--feature-release-sha256", required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--rows-csv-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--window-manifest-sha256", required=True)
    parser.add_argument("--gesture-index", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--norm-summary", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--formal-protocol", type=Path, required=True)
    parser.add_argument("--formal-protocol-sha256", required=True)
    parser.add_argument("--coverage-schedule", type=Path, required=True)
    parser.add_argument("--coverage-schedule-sha256", required=True)
    parser.add_argument("--client-commit", required=True)
    parser.add_argument("--mint-commit", required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    require_sha(args.source_release, args.source_release_sha256, "source release")
    require_sha(args.personal_data_release, args.personal_data_release_sha256, "personal data release")
    require_sha(args.feature_release, args.feature_release_sha256, "feature release")
    require_sha(args.rows_csv, args.rows_csv_sha256, "train rows")
    require_sha(args.split_manifest, args.split_manifest_sha256, "split manifest")
    require_sha(args.window_manifest, args.window_manifest_sha256, "window manifest")
    require_sha(args.formal_protocol, args.formal_protocol_sha256, "formal protocol")
    require_sha(args.coverage_schedule, args.coverage_schedule_sha256, "coverage schedule")
    rows = rows_csv(args.rows_csv)
    row_digest = hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()
    if row_digest != "c2113eb790250206c3a5b860cb37197715ba44c4f3b5a86f0d42182bc0f79ad4":
        raise ValueError("train row digest mismatch")
    split = load_json(args.split_manifest)
    train = split.get("splits", {}).get("train", {})
    if split.get("status") != "accepted" or train.get("row_indices_sha256") != row_digest or train.get("rows_csv_sha256") != args.rows_csv_sha256:
        raise ValueError("split manifest does not authenticate train rows")
    windows_payload = load_json(args.window_manifest)
    windows = windows_payload.get("windows", {})
    active_frames = sum(int(windows[str(row)]["frame_count"]) for row in rows)
    if active_frames != 423450 or train.get("active_frame_count") != active_frames:
        raise ValueError(f"active frame mismatch: {active_frames}")
    source_release = load_json(args.source_release)
    data_release = load_json(args.personal_data_release)
    feature_release = load_json(args.feature_release)
    if source_release.get("status") != "accepted" or data_release.get("status") != "accepted" or feature_release.get("status") != "accepted":
        raise ValueError("source/data/feature release status is not accepted")
    if data_release.get("feature_release_sha256") != args.feature_release_sha256:
        raise ValueError("data/feature release binding mismatch")
    norm_sha = sha256(args.norm_stats)
    norm_summary = load_json(args.norm_summary)
    if norm_summary.get("state_contract") != "mano_object_dynamics_state54_v1" or norm_summary.get("state_dim") != 54 or norm_summary.get("action_dim") != 32 or norm_summary.get("active_frames") != active_frames or norm_summary.get("row_indices_sha256") != row_digest or norm_summary.get("norm_stats_sha256") != norm_sha:
        raise ValueError("State54 norm summary mismatch")
    token_sha = sha256(args.token_audit)
    token = load_json(args.token_audit)
    if token.get("state_contract") != "mano_object_dynamics_state54_v1" or token.get("population_row_indices_sha256") != row_digest or token.get("audited_active_frames") != active_frames or token.get("overflow_count") != 0 or token.get("zero_truncation") is not True or int(token.get("maximum_token_length", 10**9)) > 256:
        raise ValueError("State54 token audit failed")
    protocol = load_json(args.formal_protocol)
    if protocol.get("protocol_id") != "state54_replay_train_only_v1" or protocol.get("status") != "frozen_not_launched":
        raise ValueError("wrong State54-only formal protocol")
    schedule = load_json(args.coverage_schedule)
    if schedule.get("status") != "accepted" or schedule.get("row_count") != 813 or schedule.get("sampling", {}).get("samples_per_run") != 1_200_000:
        raise ValueError("coverage schedule mismatch")
    contract = {
        "schema_version": 2,
        "contract_id": "state54_replay_train_only_noaug_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted",
        "state_contract": "mano_object_dynamics_state54_v1",
        "state_dim": 54,
        "action_dim": 32,
        "action_horizon": 10,
        "max_token_len": 256,
        "fail_on_token_truncation": True,
        "dataset": str(args.dataset.resolve()),
        "source_release_sha256": args.source_release_sha256,
        "data_release_sha256": args.personal_data_release_sha256,
        "feature_release_sha256": args.feature_release_sha256,
        "gesture_index": str(args.gesture_index.resolve()),
        "gesture_index_sha256": sha256(args.gesture_index),
        "row_indices_sha256": row_digest,
        "trajectory_count": 813,
        "active_frame_count": active_frames,
        "action_vector_count": active_frames * 10,
        "split": "train",
        "split_manifest_sha256": args.split_manifest_sha256,
        "rows_csv_sha256": args.rows_csv_sha256,
        "contact_window_manifest_sha256": args.window_manifest_sha256,
        "norm_stats": str(args.norm_stats.resolve()),
        "norm_stats_sha256": norm_sha,
        "norm_summary_sha256": sha256(args.norm_summary),
        "token_audit": str(args.token_audit.resolve()),
        "token_audit_sha256": token_sha,
        "token_overflow_count": 0,
        "augmentation": {"state_noise_std": 0.0, "target_noise_std": 0.0},
        "profile_id": "pi05_action_lora_r16_state54_v1",
        "model": "openpi/pi05-action-lora-r16-state54-finetune",
        "action_lora_rank": 16,
        "trainable_parameters": 13224992,
        "formal_protocol_sha256": args.formal_protocol_sha256,
        "coverage_schedule_sha256": args.coverage_schedule_sha256,
        "mode4_initialization": "accepted_pose_backward_qvel_current_target_snapshot_window_v1",
        "runtime_commits": {"client": args.client_commit, "mint": args.mint_commit, "openpi": args.openpi_commit},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
