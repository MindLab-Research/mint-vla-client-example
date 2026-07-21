#!/usr/bin/env python3
"""Extract a small, standalone Lance subset (by prompt match or explicit row indices).

Motivation: several downstream tools in this repo (``openpi_vla_infer_to_lance.py``'s
``_write_merged``, in particular) always read the *entire* source table
(``lance.dataset(path).to_table()``) before doing their work. That is fine for a
dataset with a handful of episodes, but on a real-scale dataset such as
``new_all_generated_mano_with_images.lance`` (7539 episodes, ~803GB, mostly JPEG
image bytes) it means materializing hundreds of GB just to run inference on a
handful of episodes. This script selects a small number of rows (episodes) up
front and writes them to their own standalone Lance dataset, so any downstream
tool that operates on "the whole table" only ever sees the small subset.

Row selection reads only the ``prompt`` column first (cheap -- confirmed
elsewhere in this repo that reading just ``prompt`` for all rows of this
dataset takes ~0.01s, since Lance only reads that column's data pages, not
every row's image/wrist_image bytes), then uses ``lance.Dataset.take(...)`` to
fetch only the selected rows' full data. Neither step reads the full table.

Usage:
    python scripts/tools/openpi_vla_extract_lance_subset.py \\
        --lance-dataset /path/to/big.lance \\
        --prompt-filter "pick up the cube1" --num-episodes 1 --seed 0 \\
        --output-lance /path/to/subset_cube1.lance

    # Or select specific rows directly instead of filtering by prompt:
    python scripts/tools/openpi_vla_extract_lance_subset.py \\
        --lance-dataset /path/to/big.lance --row-indices 12,340,5001 \\
        --output-lance /path/to/subset.lance
"""

from __future__ import annotations

import argparse
from typing import Any

import lance
import numpy as np


def _matching_row_indices(ds: "lance.LanceDataset", prompt_filter: str) -> list[int]:
    """Row indices whose `prompt` column exactly equals `prompt_filter`.

    Reads only the `prompt` column, not the full table.
    """
    prompts = ds.to_table(columns=["prompt"]).column("prompt").to_pylist()
    return [i for i, p in enumerate(prompts) if p == prompt_filter]


def extract_subset(
    lance_dataset: str,
    *,
    output_lance: str,
    prompt_filter: str | None = None,
    row_indices: list[int] | None = None,
    num_episodes: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Select `num_episodes` rows (by prompt match or explicit row_indices)
    from `lance_dataset` and write them as a standalone Lance dataset at
    `output_lance`.

    Uses `lance.Dataset.take(selected_indices)` to fetch only the selected
    rows -- this does NOT read the full source table (unlike
    `lance.Dataset.to_table()` with no row filter).
    """
    ds = lance.dataset(lance_dataset)

    if row_indices is not None:
        selected = list(row_indices)
    else:
        if not prompt_filter:
            raise ValueError("must supply either --prompt-filter or --row-indices")
        candidates = _matching_row_indices(ds, prompt_filter)
        if not candidates:
            raise SystemExit(f"no rows with prompt == {prompt_filter!r} found in {lance_dataset!r}")
        rng = np.random.default_rng(seed)
        n = min(num_episodes, len(candidates))
        selected = sorted(int(x) for x in rng.choice(candidates, size=n, replace=False))

    table = ds.take(selected)
    lance.write_dataset(table, output_lance, mode="overwrite")

    return {
        "source": lance_dataset,
        "output": output_lance,
        "selected_row_indices": selected,
        "num_rows_written": table.num_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract a small standalone Lance subset (by prompt match or explicit row "
        "indices) from a large source Lance dataset, without reading the full source table."
    )
    p.add_argument("--lance-dataset", required=True, help="source Lance dataset path")
    p.add_argument("--output-lance", required=True, help="output path for the extracted subset")
    p.add_argument(
        "--prompt-filter",
        default=None,
        help='exact-match prompt string to select from, e.g. "pick up the cube1"',
    )
    p.add_argument(
        "--row-indices",
        default=None,
        help="comma-separated explicit row indices to select instead of --prompt-filter",
    )
    p.add_argument("--num-episodes", type=int, default=1, help="how many matching rows to randomly select")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    row_indices = None
    if args.row_indices:
        row_indices = [int(t) for t in args.row_indices.split(",") if t.strip()]

    result = extract_subset(
        args.lance_dataset,
        output_lance=args.output_lance,
        prompt_filter=args.prompt_filter,
        row_indices=row_indices,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
    print(f"OK: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
