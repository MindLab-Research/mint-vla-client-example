#!/usr/bin/env python3
"""Build a personal authenticated virtual State54 replay-data release."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import lance
import numpy as np

from scripts.replay_state54_data import atomic_json, sha256_file

CONTACT_CONTEXT_FRAMES = 60
EXPECTED_OBJECT_COUNTS = {"cube1": 612, "cube2": 402}
SOURCE_RELEASE_SHA256 = "e05598979dbc08827f169b44f4fa655a01b8af85efe23682fc620b7ba5c544bd"


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--source-release-sha256", default=SOURCE_RELEASE_SHA256)
    parser.add_argument("--feature-release", type=Path, required=True)
    parser.add_argument("--feature-release-sha256", required=True)
    parser.add_argument("--row-start", type=int, default=7)
    parser.add_argument("--row-end-inclusive", type=int, default=1020)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_release_path = args.source_release.resolve()
    source_sha = sha256_file(source_release_path)
    if source_sha != args.source_release_sha256.lower():
        raise ValueError(f"source release SHA mismatch: {source_sha}")
    source = json.loads(source_release_path.read_text(encoding="utf-8"))
    if source.get("status") != "accepted":
        raise ValueError("source release is not accepted")
    dataset_path = Path(source["dataset"]).resolve()

    feature_release_path = (args.feature_release / "release.json").resolve()
    feature_sha = sha256_file(feature_release_path)
    if feature_sha != args.feature_release_sha256.lower():
        raise ValueError(f"feature release SHA mismatch: {feature_sha}")
    features = json.loads(feature_release_path.read_text(encoding="utf-8"))
    if features.get("status") != "accepted":
        raise ValueError("feature release is not accepted")
    if Path(features["source_dataset"]).resolve() != dataset_path:
        raise ValueError("feature/source dataset mismatch")

    rows = list(range(args.row_start, args.row_end_inclusive + 1))
    if rows != list(range(7, 1021)):
        raise ValueError("first replay State54 release is fixed to source rows7-1020")
    row_digest = hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()
    feature_by_row = {int(entry["row_index"]): entry for entry in features["entries"]}
    if sorted(feature_by_row) != rows:
        raise ValueError("feature release population does not equal rows7-1020")

    dataset = lance.dataset(str(dataset_path))
    selected = dataset.take(
        rows,
        columns=["index", "trajectory_metadata", "episode_metadata", "state44", "source"],
    ).to_pylist()
    population = []
    windows: dict[str, dict[str, Any]] = {}
    object_counts: Counter[str] = Counter()
    active_frames = 0
    source_frames = 0
    for row_index, row in zip(rows, selected, strict=True):
        index = row["index"]
        metadata = row["trajectory_metadata"]
        object_names = metadata.get("object_names") or []
        if len(object_names) != 1:
            raise ValueError(f"row{row_index} object identity invalid: {object_names!r}")
        object_name = str(object_names[0])
        object_counts[object_name] += 1
        feature = feature_by_row[row_index]
        frame_count = int(row["episode_metadata"]["total_frames"])
        if feature["row_uuid"] != index["uuid"] or feature["object_name"] != object_name:
            raise ValueError(f"row{row_index} feature identity mismatch")
        if int(feature["frame_count"]) != frame_count:
            raise ValueError(f"row{row_index} feature frame mismatch")
        state44 = np.asarray(row["state44"], dtype=np.float32)
        if state44.shape != (frame_count, 44) or not np.all(np.isfinite(state44)):
            raise ValueError(f"row{row_index} invalid State44 source shape")
        contact = state44[:, 26:31] > 0.5
        contact_frames = np.flatnonzero(np.any(contact, axis=1))
        if not len(contact_frames):
            raise ValueError(f"row{row_index} has no target-object contact")
        first = int(contact_frames[0])
        last = int(contact_frames[-1])
        start = max(0, first - CONTACT_CONTEXT_FRAMES)
        end = min(frame_count - 1, last + CONTACT_CONTEXT_FRAMES)
        active = end - start + 1
        active_frames += active
        source_frames += frame_count
        windows[str(row_index)] = {
            "status": "full_contact_span_pm60",
            "row_index": row_index,
            "object_name": object_name,
            "total_frames": frame_count,
            "first_contact_frame": first,
            "last_contact_frame": last,
            "contact_frame_count": int(len(contact_frames)),
            "start_frame": start,
            "end_frame": end,
            "frame_count": active,
            "context_frames": CONTACT_CONTEXT_FRAMES,
        }
        population.append(
            {
                "row_index": row_index,
                "row_uuid": str(index["uuid"]),
                "seed_uuid": str(index["seed_uuid"]),
                "object_name": object_name,
                "frame_count": frame_count,
                "active_start_frame": start,
                "active_end_frame": end,
                "active_frame_count": active,
                "source_row_index": int(row["source"]["source_row_index"]),
                "trace_sha256": str(row["source"]["target_replay_trace_sha256"]),
                "feature_npz_sha256": str(feature["output_npz_sha256"]),
            }
        )
    if dict(object_counts) != EXPECTED_OBJECT_COUNTS:
        raise ValueError(f"object count mismatch: {dict(object_counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.csv"
    rows_path.write_text(",".join(map(str, rows)) + "\n", encoding="utf-8")
    contact_manifest = {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "source_release": str(source_release_path),
        "source_release_sha256": source_sha,
        "row_count": int(source["rows"]),
        "selected_row_count": len(rows),
        "context_frames": CONTACT_CONTEXT_FRAMES,
        "post_context_frames": CONTACT_CONTEXT_FRAMES,
        "missing_policy": "error",
        "selection": {"rows": rows, "row_indices_sha256": row_digest},
        "phase_contract": {
            "signal": "accepted replay contact5 from source State44[26:31]",
            "start": "max(0, first contact frame - 60)",
            "end": "min(T-1, last contact frame + 60)",
            "interior": "retain all frames including contact gaps/lift/hold/descent/release",
            "action_horizon_boundary": "clip and repeat-pad final target",
        },
        "windows": windows,
    }
    contact_path = args.output_dir / "contact_pm60_window_manifest.json"
    atomic_json(contact_path, contact_manifest)
    population_payload = {
        "schema_version": 1,
        "row_count": len(rows),
        "row_indices_sha256": row_digest,
        "object_counts": dict(object_counts),
        "source_frames": source_frames,
        "active_frames": active_frames,
        "entries_sha256": canonical_sha256(population),
        "entries": population,
    }
    population_path = args.output_dir / "population.json"
    atomic_json(population_path, population_payload)

    source_gesture = Path(source["artifacts"]["gesture_index"]["path"]).resolve()
    source_gesture_sha = sha256_file(source_gesture)
    if source_gesture_sha != source["artifacts"]["gesture_index"]["sha256"]:
        raise ValueError("source gesture-index SHA mismatch")
    gesture_path = args.output_dir / "gesture_index.json"
    shutil.copyfile(source_gesture, gesture_path)
    if sha256_file(gesture_path) != source_gesture_sha:
        raise ValueError("copied gesture-index SHA mismatch")

    release = {
        "schema_version": 1,
        "release_id": args.output_dir.name,
        "status": "accepted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_kind": "authenticated_virtual_lance_view_with_state54_feature_sidecars",
        "state_contract": "mano_object_dynamics_state54_v1",
        "feature_schema_id": features["feature_schema_id"],
        "action_contract": "urdf_target_absolute_action32_horizon10",
        "dataset": str(dataset_path),
        "dataset_ownership": "external_read_only_authenticated_source",
        "source_release": str(source_release_path),
        "source_release_sha256": source_sha,
        "source_lance_file_manifest_sha256": source["lance_file_manifest_sha256"],
        "feature_release": str(args.feature_release.resolve()),
        "feature_release_sha256": feature_sha,
        "feature_manifest_sha256": features["feature_manifest_sha256"],
        "snapshot_contact_reconciliation": features.get("snapshot_contact_reconciliation"),
        "rows_csv": str(rows_path.resolve()),
        "rows_csv_sha256": sha256_file(rows_path),
        "population": str(population_path.resolve()),
        "population_sha256": sha256_file(population_path),
        "row_indices_sha256": row_digest,
        "row_count": len(rows),
        "object_counts": dict(object_counts),
        "source_frames": source_frames,
        "active_frames": active_frames,
        "contact_window_manifest": str(contact_path.resolve()),
        "contact_window_manifest_sha256": sha256_file(contact_path),
        "gesture_index": str(gesture_path.resolve()),
        "gesture_index_sha256": source_gesture_sha,
        "immutable_scope": "release manifests and personal feature files; source Lance authenticated by source release",
    }
    atomic_json(args.output_dir / "release.json", release)
    print(json.dumps(release, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
