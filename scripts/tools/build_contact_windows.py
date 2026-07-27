#!/usr/bin/env python3
"""Build or extend a per-trajectory contact-window manifest for a Lance dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import lance

from scripts import contact_windows as cw


def _parse_indices(value: str, row_count: int) -> list[int]:
    if not value.strip() or value.strip().lower() == "all":
        return list(range(row_count))
    indices = [int(item) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--row-indices did not contain any indices")
    invalid = [index for index in indices if not 0 <= index < row_count]
    if invalid:
        raise IndexError(f"row indices outside [0, {row_count}): {invalid}")
    return list(dict.fromkeys(indices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--row-indices", default="all")
    parser.add_argument("--context-frames", type=int, default=cw.DEFAULT_CONTACT_CONTEXT_FRAMES)
    parser.add_argument("--missing-contact-policy", choices=("full", "skip", "error"), default="full")
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.context_frames < 0:
        raise ValueError("--context-frames must be non-negative")
    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    dataset = lance.dataset(str(args.lance_dataset))
    row_count = dataset.count_rows()
    selected = _parse_indices(args.row_indices, row_count)
    output = args.output or cw.default_manifest_path(args.lance_dataset)
    if args.overwrite and output.exists():
        output.unlink()

    started = time.monotonic()
    resolved: dict[int, dict] = {}
    for offset in range(0, len(selected), args.batch_rows):
        batch = selected[offset : offset + args.batch_rows]
        current = cw.load_or_build_windows(
            dataset,
            args.lance_dataset,
            batch,
            manifest_path=output,
            context_frames=args.context_frames,
            missing_policy=args.missing_contact_policy,
            batch_rows=args.batch_rows,
            cache=True,
        )
        resolved.update(current)
        print(
            json.dumps(
                {
                    "processed": min(offset + len(batch), len(selected)),
                    "selected": len(selected),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "manifest": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    statuses = Counter(str(entry.get("status")) for entry in resolved.values())
    source_frames = sum(int(entry.get("total_frames") or 0) for entry in resolved.values())
    selected_frames = sum(int(entry.get("frame_count") or 0) for entry in resolved.values())
    summary = {
        "dataset": str(args.lance_dataset),
        "manifest": str(output),
        "dataset_row_count": row_count,
        "selected_row_count": len(selected),
        "resolved_row_count": len(resolved),
        "context_frames": args.context_frames,
        "missing_contact_policy": args.missing_contact_policy,
        "status_counts": dict(sorted(statuses.items())),
        "source_frame_count": source_frames,
        "selected_frame_count": selected_frames,
        "retained_fraction": (selected_frames / source_frames) if source_frames else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
