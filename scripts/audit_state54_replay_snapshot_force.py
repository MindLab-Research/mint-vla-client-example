#!/usr/bin/env python3
"""Quantify snapshot-load approximation against exact deterministic replays."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.replay_state54_data import atomic_json, sha256_file

GLOBAL_CORRELATION_MIN = 0.98
GLOBAL_MAE_NEWTONS_MAX = 0.5
ROW_MAE_P95_NEWTONS_MAX = 1.0


def _entries(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    release = json.loads(path.read_text(encoding="utf-8"))
    entries = release.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"feature release has no accepted entries: {path}")
    mapping = {int(entry["row_index"]): entry for entry in entries}
    if len(mapping) != len(entries):
        raise ValueError(f"duplicate feature rows in {path}")
    return release, mapping


def _load(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(entry["output_npz"])
    if sha256_file(path) != entry["output_npz_sha256"]:
        raise ValueError(f"feature SHA mismatch: {path}")
    with np.load(path) as archive:
        contact = np.asarray(archive["finger_contacts"], dtype=np.float32)
        force = np.asarray(archive["finger_log1p_force"], dtype=np.float32)
    if contact.shape != force.shape or contact.ndim != 2 or contact.shape[1] != 5:
        raise ValueError(f"invalid feature shapes in {path}: {contact.shape}/{force.shape}")
    return contact, force


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-release", type=Path, required=True)
    parser.add_argument("--snapshot-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exact_release, exact = _entries(args.exact_release)
    snapshot_release, snapshot = _entries(args.snapshot_release)
    missing = sorted(set(exact) - set(snapshot))
    if missing:
        raise ValueError(f"snapshot release is missing exact-reference rows: {missing[:20]}")

    exact_values = []
    snapshot_values = []
    row_metrics = []
    total_contact_mismatch = 0
    for row_index in sorted(exact):
        exact_contact, exact_log = _load(exact[row_index])
        snapshot_contact, snapshot_log = _load(snapshot[row_index])
        if exact_contact.shape != snapshot_contact.shape:
            raise ValueError(f"row{row_index} feature shape mismatch")
        contact_mismatch = int(np.count_nonzero(exact_contact != snapshot_contact))
        total_contact_mismatch += contact_mismatch
        exact_load = np.expm1(exact_log.astype(np.float64))
        snapshot_load = np.expm1(snapshot_log.astype(np.float64))
        mask = (exact_load > 0) | (snapshot_load > 0)
        difference = np.abs(exact_load[mask] - snapshot_load[mask])
        exact_values.append(exact_load[mask])
        snapshot_values.append(snapshot_load[mask])
        row_metrics.append(
            {
                "row_index": row_index,
                "object_name": exact[row_index]["object_name"],
                "frame_count": int(exact[row_index]["frame_count"]),
                "contact_mismatch_values": contact_mismatch,
                "compared_load_values": int(np.count_nonzero(mask)),
                "mae_newtons": float(np.mean(difference)) if len(difference) else 0.0,
                "median_abs_error_newtons": float(np.median(difference)) if len(difference) else 0.0,
                "max_abs_error_newtons": float(np.max(difference)) if len(difference) else 0.0,
            }
        )

    x = np.concatenate(exact_values)
    y = np.concatenate(snapshot_values)
    absolute_error = np.abs(x - y)
    correlation = float(np.corrcoef(x, y)[0, 1])
    row_mae = np.asarray([row["mae_newtons"] for row in row_metrics], dtype=np.float64)
    checks = {
        "contact_values_exact": total_contact_mismatch == 0,
        "global_correlation_at_least_min": correlation >= GLOBAL_CORRELATION_MIN,
        "global_mae_at_most_max": float(np.mean(absolute_error)) <= GLOBAL_MAE_NEWTONS_MAX,
        "row_mae_p95_at_most_max": float(np.quantile(row_mae, 0.95)) <= ROW_MAE_P95_NEWTONS_MAX,
    }
    report = {
        "status": "accepted" if all(checks.values()) else "rejected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "accepted_pose_backward_qvel_snapshot_force_audit_v1",
        "exact_release": str(args.exact_release.resolve()),
        "exact_release_sha256": sha256_file(args.exact_release),
        "snapshot_release": str(args.snapshot_release.resolve()),
        "snapshot_release_sha256": sha256_file(args.snapshot_release),
        "reference_row_count": len(row_metrics),
        "compared_load_values": len(x),
        "contact_mismatch_values": total_contact_mismatch,
        "global_force_correlation": correlation,
        "global_mae_newtons": float(np.mean(absolute_error)),
        "global_median_abs_error_newtons": float(np.median(absolute_error)),
        "global_p95_abs_error_newtons": float(np.quantile(absolute_error, 0.95)),
        "global_p99_abs_error_newtons": float(np.quantile(absolute_error, 0.99)),
        "global_max_abs_error_newtons": float(np.max(absolute_error)),
        "exact_mean_positive_union_load_newtons": float(np.mean(x)),
        "snapshot_mean_positive_union_load_newtons": float(np.mean(y)),
        "row_mae_p50_newtons": float(np.quantile(row_mae, 0.5)),
        "row_mae_p95_newtons": float(np.quantile(row_mae, 0.95)),
        "row_mae_max_newtons": float(np.max(row_mae)),
        "thresholds": {
            "global_correlation_min": GLOBAL_CORRELATION_MIN,
            "global_mae_newtons_max": GLOBAL_MAE_NEWTONS_MAX,
            "row_mae_p95_newtons_max": ROW_MAE_P95_NEWTONS_MAX,
            "contact_values_exact": True,
        },
        "checks": checks,
        "rows": row_metrics,
    }
    atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in report if key != "rows"}, indent=2, sort_keys=True))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
