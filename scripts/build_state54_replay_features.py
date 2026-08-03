#!/usr/bin/env python3
"""Build an authenticated per-row State54 feature release from replay traces."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from multiprocessing import get_context
from pathlib import Path
import tempfile
import time
from typing import Any

import lance

from scripts.replay_state54_data import (
    DETERMINISTIC_REPLAY_FEATURE_SCHEMA_ID,
    SNAPSHOT_FEATURE_SCHEMA_ID,
    atomic_json,
    atomic_npz,
    replay_trace_state54_features,
    sha256_file,
    snapshot_trace_state54_features,
)

SCHEMA_VERSION = 1
SNAPSHOT_RAW_CONTACT_MISMATCH_RATE_MAX = 1e-6
SNAPSHOT_FALSE_NEGATIVE_MAX = 0
SNAPSHOT_MISMATCHED_LOAD_NEWTONS_MAX = 0.1
FEATURE_SCHEMA_BY_MODE = {
    "deterministic-replay": DETERMINISTIC_REPLAY_FEATURE_SCHEMA_ID,
    "snapshot-backward": SNAPSHOT_FEATURE_SCHEMA_ID,
}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accepted_existing(report_path: Path, feature_path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "passed":
            return None
        for key in ("row_index", "row_uuid", "object_name", "trace_sha256", "frame_count"):
            if report.get(key) != expected[key]:
                return None
        if report.get("output_npz_sha256") != sha256_file(feature_path):
            return None
        return report
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None


def derive_one(job: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(job["output_dir"])
    feature_path = output_dir / "features" / f"row{job['row_index']:05d}.npz"
    report_path = output_dir / "reports" / f"row{job['row_index']:05d}.json"
    expected = {
        "row_index": job["row_index"],
        "row_uuid": job["row_uuid"],
        "object_name": job["object_name"],
        "trace_sha256": job["trace_sha256"],
        "frame_count": job["frame_count"],
    }
    if job["resume"] and feature_path.is_file() and report_path.is_file():
        existing = _accepted_existing(report_path, feature_path, expected)
        if existing is not None:
            return {**existing, "worker_status": "skipped"}

    actual_trace_sha = sha256_file(job["trace_path"])
    if actual_trace_sha != job["trace_sha256"]:
        raise ValueError(
            f"row{job['row_index']} trace SHA mismatch: "
            f"{actual_trace_sha} != {job['trace_sha256']}"
        )
    derive = (
        replay_trace_state54_features
        if job["derivation_mode"] == "deterministic-replay"
        else snapshot_trace_state54_features
    )
    report, arrays = derive(job["trace_path"], object_name=job["object_name"])
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "feature_schema_id": job["feature_schema_id"],
            "row_index": job["row_index"],
            "row_uuid": job["row_uuid"],
            "seed_uuid": job["seed_uuid"],
            "source_row_index": job["source_row_index"],
            "object_name": job["object_name"],
            "frame_count": job["frame_count"],
        }
    )
    if report["status"] != "passed":
        atomic_json(report_path, report)
        return {**report, "worker_status": "failed"}
    atomic_npz(feature_path, **arrays)
    report["output_npz"] = str(feature_path.resolve())
    report["output_npz_sha256"] = sha256_file(feature_path)
    atomic_json(report_path, report)
    return {**report, "worker_status": "built"}


def _source_jobs(args: argparse.Namespace, release: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_path = Path(release["dataset"]).resolve()
    dataset = lance.dataset(str(dataset_path))
    row_indices = list(range(args.row_start, args.row_end_inclusive + 1))
    rows = dataset.take(
        row_indices, columns=["index", "trajectory_metadata", "episode_metadata", "source"]
    ).to_pylist()
    jobs = []
    for row_index, row in zip(row_indices, rows, strict=True):
        index = row["index"]
        metadata = row["trajectory_metadata"]
        objects = metadata.get("object_names") or []
        if len(objects) != 1 or objects[0] not in {"cube1", "cube2"}:
            raise ValueError(f"row{row_index} is outside cube1/cube2 scope: {objects!r}")
        source = row["source"]
        trace_path = Path(source["target_replay_trace"]).resolve()
        trace_sha = str(source["target_replay_trace_sha256"]).lower()
        frame_count = int(row["episode_metadata"]["total_frames"])
        jobs.append(
            {
                "output_dir": str(args.output_dir.resolve()),
                "resume": bool(args.resume),
                "row_index": row_index,
                "row_uuid": str(index["uuid"]),
                "seed_uuid": str(index["seed_uuid"]),
                "source_row_index": int(source["source_row_index"]),
                "object_name": str(objects[0]),
                "frame_count": frame_count,
                "trace_path": str(trace_path),
                "trace_sha256": trace_sha,
                "derivation_mode": args.derivation_mode,
                "feature_schema_id": FEATURE_SCHEMA_BY_MODE[args.derivation_mode],
            }
        )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--source-release-sha256", required=True)
    parser.add_argument("--row-start", type=int, default=7)
    parser.add_argument("--row-end-inclusive", type=int, default=1020)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--derivation-mode",
        choices=tuple(FEATURE_SCHEMA_BY_MODE),
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.row_start < 0 or args.row_end_inclusive < args.row_start:
        raise ValueError("invalid inclusive row range")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    source_release_sha = sha256_file(args.source_release)
    if source_release_sha != args.source_release_sha256.lower():
        raise ValueError(
            f"source release SHA mismatch: {source_release_sha} != "
            f"{args.source_release_sha256.lower()}"
        )
    release = json.loads(args.source_release.read_text(encoding="utf-8"))
    if release.get("status") != "accepted":
        raise ValueError("source replay release is not accepted")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = _source_jobs(args, release)
    population = [
        {
            key: job[key]
            for key in (
                "row_index",
                "row_uuid",
                "seed_uuid",
                "source_row_index",
                "object_name",
                "frame_count",
                "trace_path",
                "trace_sha256",
            )
        }
        for job in jobs
    ]
    population_sha = canonical_sha256(population)
    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_BY_MODE[args.derivation_mode],
        "status": "building",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_release": str(args.source_release.resolve()),
        "source_release_sha256": source_release_sha,
        "source_dataset": str(Path(release["dataset"]).resolve()),
        "source_lance_file_manifest_sha256": release.get("lance_file_manifest_sha256"),
        "row_start": args.row_start,
        "row_end_inclusive": args.row_end_inclusive,
        "row_count": len(jobs),
        "population_sha256": population_sha,
        "population": population,
        "workers": args.workers,
        "derivation_mode": args.derivation_mode,
    }
    manifest_path = args.output_dir / "build_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "feature_schema_id",
            "source_release_sha256",
            "source_dataset",
            "row_start",
            "row_end_inclusive",
            "row_count",
            "population_sha256",
            "derivation_mode",
        ):
            if existing.get(key) != build_manifest[key]:
                raise ValueError(f"existing build manifest disagrees on {key!r}")
    else:
        atomic_json(manifest_path, build_manifest)

    started = time.monotonic()
    reports: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as executor:
        for completed, report in enumerate(executor.map(derive_one, jobs), 1):
            reports.append(report)
            status_counts[report["worker_status"]] += 1
            object_counts[report["object_name"]] += 1
            if completed == 1 or completed % 25 == 0 or completed == len(jobs):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(jobs),
                            "status_counts": dict(status_counts),
                            "object_counts": dict(object_counts),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    reports.sort(key=lambda item: int(item["row_index"]))
    failed = [report for report in reports if report["status"] != "passed"]
    common_entry_keys = (
        "derivation_mode",
        "row_index",
        "row_uuid",
        "seed_uuid",
        "source_row_index",
        "object_name",
        "frame_count",
        "trace_sha256",
        "output_npz",
        "output_npz_sha256",
        "contact_mismatch_values",
        "force_nonzero_frames",
        "force_nonzero_values",
        "max_log1p_force",
    )
    optional_entry_keys = (
        "max_abs_hand_qpos_error",
        "max_abs_object_position_error",
        "max_sign_invariant_object_quaternion_error",
        "snapshot_state_source",
        "snapshot_qvel_source",
        "snapshot_ctrl_source",
        "contact_false_positive_values",
        "contact_false_negative_values",
        "max_mismatched_snapshot_load_newtons",
        "contact_output_source",
        "force_mask",
    )
    entries = []
    for report in reports:
        if report["status"] != "passed":
            continue
        entry = {key: report[key] for key in common_entry_keys}
        entry.update(
            {key: report[key] for key in optional_entry_keys if key in report}
        )
        entries.append(entry)
    feature_manifest_sha = canonical_sha256(entries)
    snapshot_reconciliation = None
    snapshot_reconciliation_accepted = True
    if args.derivation_mode == "snapshot-backward":
        total_contact_values = sum(int(report["frame_count"]) * 5 for report in reports)
        mismatch_values = sum(int(report["contact_mismatch_values"]) for report in reports)
        false_positive_values = sum(
            int(report["contact_false_positive_values"]) for report in reports
        )
        false_negative_values = sum(
            int(report["contact_false_negative_values"]) for report in reports
        )
        max_mismatched_load = max(
            float(report["max_mismatched_snapshot_load_newtons"]) for report in reports
        )
        mismatch_rate = mismatch_values / total_contact_values
        reconciliation_checks = {
            "raw_contact_mismatch_rate_at_most_max": (
                mismatch_rate <= SNAPSHOT_RAW_CONTACT_MISMATCH_RATE_MAX
            ),
            "false_negative_values_at_most_max": (
                false_negative_values <= SNAPSHOT_FALSE_NEGATIVE_MAX
            ),
            "max_mismatched_load_newtons_at_most_max": (
                max_mismatched_load <= SNAPSHOT_MISMATCHED_LOAD_NEWTONS_MAX
            ),
        }
        snapshot_reconciliation_accepted = all(reconciliation_checks.values())
        snapshot_reconciliation = {
            "status": "accepted" if snapshot_reconciliation_accepted else "rejected",
            "contact_output_source": "accepted_target_replay_trace",
            "force_mask": "accepted_target_replay_contact",
            "total_contact_values": total_contact_values,
            "raw_contact_mismatch_values": mismatch_values,
            "raw_contact_mismatch_rate": mismatch_rate,
            "raw_contact_false_positive_values": false_positive_values,
            "raw_contact_false_negative_values": false_negative_values,
            "max_mismatched_snapshot_load_newtons": max_mismatched_load,
            "thresholds": {
                "raw_contact_mismatch_rate_max": SNAPSHOT_RAW_CONTACT_MISMATCH_RATE_MAX,
                "false_negative_values_max": SNAPSHOT_FALSE_NEGATIVE_MAX,
                "max_mismatched_load_newtons_max": SNAPSHOT_MISMATCHED_LOAD_NEWTONS_MAX,
            },
            "checks": reconciliation_checks,
        }
    release_accepted = (
        not failed
        and len(entries) == len(jobs)
        and snapshot_reconciliation_accepted
    )
    final_release = {
        **{key: value for key, value in build_manifest.items() if key != "status"},
        "status": "accepted" if release_accepted else "failed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "status_counts": dict(status_counts),
        "object_counts": dict(object_counts),
        "entry_count": len(entries),
        "failed_rows": [int(report["row_index"]) for report in failed],
        "feature_manifest_sha256": feature_manifest_sha,
        "snapshot_contact_reconciliation": snapshot_reconciliation,
        "entries": entries,
    }
    atomic_json(args.output_dir / "release.json", final_release)
    print(
        json.dumps(
            {
                key: final_release[key]
                for key in (
                    "status",
                    "row_count",
                    "entry_count",
                    "failed_rows",
                    "feature_manifest_sha256",
                    "elapsed_seconds",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_release["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
