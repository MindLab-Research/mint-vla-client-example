#!/usr/bin/env python3
"""Compute exact state54 population norm and audit every active prompt/state token length."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import openpi_vla_smoke_lance_base as L
from openpi.models.tokenizer import PaligemmaTokenizer
from scripts.mano_state54_contract import STATE_CONTRACT_ID, STATE_DIM
from scripts.train.train_cube1_01_compare import (
    GESTURE_LANGUAGE,
    SelectedLanceDataset,
    format_language_prompt,
    selected_norm_stats,
)
from scripts.target_actions import URDF_TARGET_ABSOLUTE



def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_rows(path: Path) -> tuple[list[int], str]:
    text = path.read_text(encoding="utf-8").strip()
    rows = [int(value) for value in text.split(",") if value]
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("population rows must be nonempty and unique")
    digest = hashlib.sha256(",".join(map(str, rows)).encode()).hexdigest()
    return rows, digest


def make_dataset(args, rows):
    return SelectedLanceDataset(
        args.lance_dataset,
        row_indices=rows,
        action_horizon=10,
        frame_window="contact",
        contact_context_frames=100,
        contact_window_manifest=args.contact_window_manifest,
        missing_contact_policy="error",
        action_source=URDF_TARGET_ABSOLUTE,
        language_conditioning=GESTURE_LANGUAGE,
        gesture_index=args.gesture_index,
        target_lance_dataset=args.lance_dataset,
        state_contract=STATE_CONTRACT_ID,
        state54_replay_feature_release=args.state54_replay_feature_release,
        state54_replay_feature_release_sha256=args.state54_replay_feature_release_sha256,
    )


def compute_norm(dataset, output_dir: Path, row_digest: str) -> dict:
    started = time.time()
    stats = selected_norm_stats(dataset)
    if np.asarray(stats["state"].mean).shape != (54,):
        raise ValueError("computed state norm is not 54D")
    if np.asarray(stats["actions"].mean).shape != (32,):
        raise ValueError("computed action norm is not 32D")
    # The five sparse binary features use the fixed 0/1 quantile map, matching
    # the authenticated 32D contract and guaranteeing 0->-1, 1->+1.
    np.testing.assert_array_equal(np.asarray(stats["state"].q01)[26:31], 0.0)
    np.testing.assert_array_equal(np.asarray(stats["state"].q99)[26:31], 1.0)
    L.normalize.save(output_dir, stats)
    norm_path = output_dir / "norm_stats.json"
    sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    summary = {
        "state_contract": STATE_CONTRACT_ID,
        "state_dim": 54,
        "action_dim": 32,
        "action_horizon": 10,
        "active_frames": len(dataset),
        "trajectory_count": len(dataset._row_windows),
        "row_indices_sha256": row_digest,
        "norm_stats_sha256": sha,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "norm_summary.json", summary)
    return summary


def effective_prompt(dataset, local_row: int) -> str:
    source_row = dataset._source_row_indices[local_row]
    base = dataset._dataset.take([source_row], columns=["prompt"]).to_pylist()[0]["prompt"]
    return format_language_prompt(
        base,
        dataset._rows[local_row]["trajectory_metadata"],
        dataset._language_conditioning,
        gesture=dataset._gesture_records[local_row].gesture,
    )


def audit_tokens(dataset, norm_dir: Path, output_dir: Path, max_token_len: int, row_digest: str) -> dict:
    stats = L.normalize.load(norm_dir)
    q01 = np.asarray(stats["state"].q01, dtype=np.float32)
    q99 = np.asarray(stats["state"].q99, dtype=np.float32)
    if q01.shape != (STATE_DIM,) or q99.shape != (STATE_DIM,):
        raise ValueError("token audit requires exact 54D quantile stats")
    # 4096 is only an audit instrument.  It prevents the production tokenizer's
    # max-length branch from hiding the raw token length under test.
    tokenizer = PaligemmaTokenizer(max_len=4096)
    histogram: dict[int, int] = {}
    maximum = -1
    maximum_examples: list[dict] = []
    total = 0
    overflow = 0
    started = time.time()
    progress_path = output_dir / "token_audit.progress.json"
    for ordinal, local_row in enumerate(sorted(dataset._row_windows)):
        numeric = dataset.load_state54_numeric_row(local_row)
        raw = np.asarray(numeric["_state54_window"], dtype=np.float32)
        normalized = (raw - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        if normalized.shape[1:] != (54,) or not np.all(np.isfinite(normalized)):
            raise ValueError(f"invalid normalized state at local row {local_row}")
        prompt = effective_prompt(dataset, local_row)
        source_row = dataset._source_row_indices[local_row]
        window = dataset._row_windows[local_row]
        for offset, state in enumerate(normalized):
            _tokens, mask = tokenizer.tokenize(prompt, state)
            length = int(np.count_nonzero(mask))
            if length >= 4096:
                raise ValueError("audit tokenizer ceiling was reached; raw length is unknown")
            histogram[length] = histogram.get(length, 0) + 1
            total += 1
            if length > max_token_len:
                overflow += 1
            if length > maximum:
                maximum = length
                maximum_examples = [{
                    "source_row": source_row,
                    "source_frame": window.start_frame + offset,
                    "prompt": prompt,
                    "token_length": length,
                }]
            elif length == maximum and len(maximum_examples) < 20:
                maximum_examples.append({
                    "source_row": source_row,
                    "source_frame": window.start_frame + offset,
                    "prompt": prompt,
                    "token_length": length,
                })
        if ordinal % 10 == 0 or ordinal + 1 == len(dataset._row_windows):
            write_json(progress_path, {
                "completed_rows": ordinal + 1,
                "total_rows": len(dataset._row_windows),
                "audited_frames": total,
                "current_max": maximum,
                "overflow_count": overflow,
                "elapsed_seconds": time.time() - started,
            })
    if total != len(dataset):
        raise ValueError(f"token audit population mismatch: {total} != {len(dataset)}")
    result = {
        "state_contract": STATE_CONTRACT_ID,
        "population_row_indices_sha256": row_digest,
        "trajectory_count": len(dataset._row_windows),
        "audited_active_frames": total,
        "effective_prompt_count": len(dataset._row_windows),
        "profile_max_token_len": max_token_len,
        "minimum_token_length": min(histogram),
        "maximum_token_length": maximum,
        "headroom_at_maximum": max_token_len - maximum,
        "overflow_count": overflow,
        "zero_truncation": overflow == 0,
        "token_length_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "maximum_examples": maximum_examples,
        "elapsed_seconds": time.time() - started,
        "norm_stats_sha256": hashlib.sha256((norm_dir / "norm_stats.json").read_bytes()).hexdigest(),
    }
    write_json(output_dir / "token_audit.json", result)
    if overflow:
        raise ValueError(
            f"max_token_len={max_token_len} truncates {overflow} frames; observed max={maximum}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--contact-window-manifest", type=Path, required=True)
    parser.add_argument("--gesture-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state54-replay-feature-release", type=Path, required=True)
    parser.add_argument("--state54-replay-feature-release-sha256", required=True)
    parser.add_argument("--mode", choices=("norm", "tokens", "both"), default="both")
    parser.add_argument("--max-token-len", type=int, default=256)
    args = parser.parse_args()
    rows, row_digest = load_rows(args.rows_csv)
    dataset = make_dataset(args, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("norm", "both"):
        print(json.dumps(compute_norm(dataset, args.output_dir, row_digest), sort_keys=True), flush=True)
    if args.mode in ("tokens", "both"):
        print(json.dumps(audit_tokens(
            dataset, args.output_dir, args.output_dir, args.max_token_len, row_digest
        ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
