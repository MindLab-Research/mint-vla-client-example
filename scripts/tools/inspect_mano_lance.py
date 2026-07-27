#!/usr/bin/env python3
"""Inspect Dataset_B/MANO Lance rows without loading OpenPI or image data."""

from __future__ import annotations

from collections import Counter
import json
import sys

import lance


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} DATASET.lance")
    dataset = lance.dataset(sys.argv[1])
    row_count = dataset.count_rows()
    trajectories = []
    counts: Counter[str] = Counter()
    for row_index in range(row_count):
        row = dataset.take(
            [row_index],
            columns=["trajectory_metadata", "episode_metadata", "prompt"],
        ).to_pylist()[0]
        names = (row.get("trajectory_metadata") or {}).get("object_names") or []
        object_name = str(names[0]) if names else "unknown"
        frame_count = int((row.get("episode_metadata") or {}).get("total_frames") or 0)
        counts[object_name] += 1
        trajectories.append(
            {
                "row_index": row_index,
                "object_name": object_name,
                "frame_count": frame_count,
                "prompt": str(row.get("prompt") or ""),
            }
        )
    print(
        json.dumps(
            {
                "dataset": sys.argv[1],
                "row_count": row_count,
                "object_counts": dict(sorted(counts.items())),
                "trajectories": trajectories,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
