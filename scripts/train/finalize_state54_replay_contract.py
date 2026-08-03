#!/usr/bin/env python3
"""Finalize an authenticated replay-State54 data_contract.json."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.gesture_language import GestureIndex
from scripts.mano_state54_contract import (
    CONTACT_AGE_CLIP_SECONDS,
    FORCE_REFERENCE_NEWTONS,
    SOURCE_INTERVAL_SECONDS,
    STATE_CONTRACT_ID,
)
from scripts.replay_state54_data import sha256_file


def git_head_clean(path: Path) -> str:
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError(f"source repo must be clean before contract finalization: {path}\n{status}")
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def load_authenticated(path: Path, expected_sha: str, label: str) -> tuple[dict, str]:
    actual = sha256_file(path)
    if actual != expected_sha.lower():
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected_sha}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in (None, "accepted"):
        raise ValueError(f"{label} is not accepted")
    return payload, actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm-dir", type=Path, required=True)
    parser.add_argument("--data-release", type=Path, required=True)
    parser.add_argument("--data-release-sha256", required=True)
    parser.add_argument("--snapshot-force-audit", type=Path, required=True)
    parser.add_argument("--snapshot-force-audit-sha256", required=True)
    parser.add_argument("--client-repo", type=Path, required=True)
    parser.add_argument("--mint-repo", type=Path, required=True)
    parser.add_argument("--openpi-repo", type=Path, required=True)
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--state-noise-std", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data, data_sha = load_authenticated(
        args.data_release.resolve(), args.data_release_sha256, "data release"
    )
    force_audit, force_audit_sha = load_authenticated(
        args.snapshot_force_audit.resolve(),
        args.snapshot_force_audit_sha256,
        "snapshot force audit",
    )
    feature_release_path = Path(data["feature_release"]) / "release.json"
    features, feature_sha = load_authenticated(
        feature_release_path,
        data["feature_release_sha256"],
        "feature release",
    )

    norm_path = args.norm_dir / "norm_stats.json"
    norm_summary_path = args.norm_dir / "norm_summary.json"
    token_path = args.norm_dir / "token_audit.json"
    norm_sha = sha256_file(norm_path)
    norm_summary = json.loads(norm_summary_path.read_text(encoding="utf-8"))
    token = json.loads(token_path.read_text(encoding="utf-8"))
    token_sha = sha256_file(token_path)
    if norm_summary["norm_stats_sha256"] != norm_sha:
        raise ValueError("norm summary SHA mismatch")
    if norm_summary["row_indices_sha256"] != data["row_indices_sha256"]:
        raise ValueError("norm population digest mismatch")
    if int(norm_summary["active_frames"]) != int(data["active_frames"]):
        raise ValueError("norm active-frame mismatch")
    if token["population_row_indices_sha256"] != data["row_indices_sha256"]:
        raise ValueError("token population digest mismatch")
    if int(token["audited_active_frames"]) != int(data["active_frames"]):
        raise ValueError("token active-frame mismatch")
    if token["norm_stats_sha256"] != norm_sha:
        raise ValueError("token/norm SHA mismatch")
    augmentation = token.get("augmentation") or {}
    if token.get("zero_truncation") is not True or int(token.get("overflow_count", -1)) != 0:
        raise ValueError("clean token audit did not prove zero truncation")
    if augmentation.get("zero_truncation") is not True or int(augmentation.get("overflow_count", -1)) != 0:
        raise ValueError("augmented token audit did not prove zero truncation")
    if float(augmentation.get("requested_sigma", -1)) != args.state_noise_std:
        raise ValueError("augmented token audit sigma mismatch")
    if max(int(token["maximum_token_length"]), int(augmentation["maximum_token_length"])) > args.max_token_len:
        raise ValueError("observed token maximum exceeds profile")

    gesture = GestureIndex.load(Path(data["gesture_index"]))
    if gesture.sha256 != data["gesture_index_sha256"]:
        raise ValueError("gesture binding mismatch")
    if feature_sha != data["feature_release_sha256"]:
        raise ValueError("data/feature release SHA mismatch")
    if features["feature_manifest_sha256"] != data["feature_manifest_sha256"]:
        raise ValueError("data/feature manifest mismatch")

    contract = {
        "contract_version": 1,
        "status": "accepted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_id": "state44_replay_cube1_cube2_state54_action32_v1",
        "state_contract": STATE_CONTRACT_ID,
        "feature_schema_id": features["feature_schema_id"],
        "state_dim": 54,
        "state_layout": {
            "hand_qpos": [0, 26],
            "finger_contact": [26, 31],
            "source_frame0_lift": [31, 32],
            "fingertip_collision_box_xyz": [32, 47],
            "finger_log1p_normal_load": [47, 52],
            "object_minus_palm_vertical_velocity": [52, 53],
            "window_local_multifinger_contact_age": [53, 54],
        },
        "action_dim": 32,
        "action_horizon": 10,
        "action_source": "urdf_target_absolute",
        "model": "openpi/pi05-action-lora-r16-state54-finetune",
        "profile": "pi05_action_lora_r16_state54_v1",
        "alora_rank": 16,
        "expected_trainable_parameters": 13224992,
        "frame_window": "full_contact_span_pm60",
        "contact_context_frames": 60,
        "row_indices_sha256": data["row_indices_sha256"],
        "trajectory_count": data["row_count"],
        "object_counts": data["object_counts"],
        "source_frame_count": data["source_frames"],
        "active_frame_count": data["active_frames"],
        "action_vector_count": int(data["active_frames"]) * 10,
        "data_release": str(args.data_release.resolve()),
        "data_release_sha256": data_sha,
        "dataset": data["dataset"],
        "source_release": data["source_release"],
        "source_release_sha256": data["source_release_sha256"],
        "source_lance_file_manifest_sha256": data["source_lance_file_manifest_sha256"],
        "feature_release": data["feature_release"],
        "feature_release_sha256": feature_sha,
        "feature_manifest_sha256": data["feature_manifest_sha256"],
        "snapshot_contact_reconciliation": data["snapshot_contact_reconciliation"],
        "snapshot_force_audit": str(args.snapshot_force_audit.resolve()),
        "snapshot_force_audit_sha256": force_audit_sha,
        "snapshot_force_reference_rows": force_audit["reference_row_count"],
        "snapshot_force_global_correlation": force_audit["global_force_correlation"],
        "snapshot_force_global_mae_newtons": force_audit["global_mae_newtons"],
        "norm_stats": str(norm_path.resolve()),
        "norm_stats_sha256": norm_sha,
        "token_audit": str(token_path.resolve()),
        "token_audit_sha256": token_sha,
        "max_token_len": args.max_token_len,
        "clean_token_range": [token["minimum_token_length"], token["maximum_token_length"]],
        "augmented_token_range": [
            augmentation["minimum_token_length"],
            augmentation["maximum_token_length"],
        ],
        "clean_token_overflow_count": 0,
        "augmented_token_overflow_count": 0,
        "state_augmentation": {
            "sigma": args.state_noise_std,
            "seed": augmentation["seed"],
            "realized_sigma": augmentation["realized_sigma"],
            "rule": augmentation["causal_recomputation"],
        },
        "force_derivation": (
            "accepted_pose_mj_differentiatePos_backward_5ms_current_target_mj_forward_"
            "sum_pair_normal_load_log1p_masked_by_accepted_contact_v1"
        ),
        "force_reference_newtons": FORCE_REFERENCE_NEWTONS,
        "source_interval_seconds": SOURCE_INTERVAL_SECONDS,
        "contact_age_rule": "consecutive_at_least_two_fingers_window_local_clipped_v1",
        "contact_age_clip_seconds": CONTACT_AGE_CLIP_SECONDS,
        "lift_baseline": "source_frame_0_object_body_z_v1",
        "palm_vertical_position": "hand_qpos_root_z_v1",
        "fingertip_frame": "object_mesh_aabb_center_half_extents_v1",
        "finger_order": ["index", "thumb", "ring", "middle", "pinky"],
        "mode4_initialization": "accepted_pose_backward_qvel_current_target_snapshot_window_v1",
        "mode4_temporal_reset": "at_policy_takeover_window_start",
        "gesture_index": data["gesture_index"],
        "gesture_index_sha256": gesture.sha256,
        "contact_window_manifest": data["contact_window_manifest"],
        "contact_window_manifest_sha256": data["contact_window_manifest_sha256"],
        "client_commit": git_head_clean(args.client_repo),
        "mint_commit": git_head_clean(args.mint_repo),
        "openpi_commit": git_head_clean(args.openpi_repo),
    }
    destination = args.norm_dir / "data_contract.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
