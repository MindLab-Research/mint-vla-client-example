#!/usr/bin/env python3
"""Compute authenticated state44/action32 norms and audit the raw 200-token budget."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import lance
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize

from scripts import contact_windows
from scripts.gesture_language import DEFAULT_GESTURE_INDEX_PATH
from scripts.mano_state44_contract import (
    STATE44_ACTION_DIM,
    STATE44_CONTRACT_ID,
    STATE44_DIM,
    STATE44_PERSISTENCE_CLIP_SECONDS,
    STATE44_RATE_WINDOW_SECONDS,
    STATE44_SAMPLE_DT_SECONDS,
)
from scripts.openpi_profiles import ACTION_LORA_R16_STATE44_MODEL, resolve_profile
from scripts.target_actions import URDF_TARGET_ABSOLUTE
from scripts.train.train_cube1_01_compare import (
    GESTURE_LANGUAGE,
    SelectedLanceDataset,
    format_language_prompt,
    parse_row_indices,
    selected_norm_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--target-lance-dataset", type=Path, default=None)
    parser.add_argument("--row-indices", required=True, help="comma-separated source rows or all")
    parser.add_argument("--frame-window", choices=("contact", "full"), default="contact")
    parser.add_argument("--contact-context-frames", type=int, default=100)
    parser.add_argument("--contact-window-manifest", type=Path, default=None)
    parser.add_argument("--gesture-index", type=Path, default=DEFAULT_GESTURE_INDEX_PATH)
    parser.add_argument("--norm-output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def _percentile(lengths: list[int], percentile: float) -> float:
    return float(np.percentile(np.asarray(lengths, dtype=np.float64), percentile))


def raw_pi05_token_length(
    tokenizer: PaligemmaTokenizer, prompt: str, normalized_state: np.ndarray
) -> int:
    """Count the exact pre-padding/pre-truncation tokens used by pi0.5."""
    cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ")
    discretized_state = (
        np.digitize(normalized_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    )
    state_text = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {cleaned_prompt}, State: {state_text};\nAction: "
    return len(tokenizer._tokenizer.encode(full_prompt, add_bos=True))


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.norm_output_dir.exists():
        raise FileExistsError(f"norm output already exists: {args.norm_output_dir}")
    if args.report_json.exists():
        raise FileExistsError(f"report output already exists: {args.report_json}")
    target = args.target_lance_dataset or args.lance_dataset
    source = lance.dataset(str(args.lance_dataset))
    rows, row_selection = parse_row_indices(args.row_indices, int(source.count_rows()))
    manifest = args.contact_window_manifest
    if args.frame_window == "contact" and manifest is None:
        manifest = contact_windows.default_manifest_path(args.lance_dataset)
    dataset = SelectedLanceDataset(
        args.lance_dataset,
        row_indices=rows,
        action_horizon=10,
        frame_window=args.frame_window,
        contact_context_frames=args.contact_context_frames,
        contact_window_manifest=manifest,
        missing_contact_policy="error",
        action_source=URDF_TARGET_ABSOLUTE,
        language_conditioning=GESTURE_LANGUAGE,
        gesture_index=args.gesture_index,
        target_lance_dataset=target,
        state_contract="state44",
    )

    started = time.perf_counter()
    local_rows = sorted(dataset._row_start_offset)
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="state44-feature") as pool:
        for completed, _ in enumerate(
            pool.map(dataset._get_state44_sequence, local_rows), start=1
        ):
            if completed % 100 == 0 or completed == len(local_rows):
                print(
                    f"state44 feature rows {completed}/{len(local_rows)}",
                    flush=True,
                )
    feature_seconds = time.perf_counter() - started

    norm_stats = selected_norm_stats(dataset)
    if np.asarray(norm_stats["state"].mean).shape != (STATE44_DIM,):
        raise RuntimeError("computed state normalization width is not 44")
    if np.asarray(norm_stats["actions"].mean).shape != (STATE44_ACTION_DIM,):
        raise RuntimeError("computed action normalization width is not 32")
    q01 = np.asarray(norm_stats["state"].q01, dtype=np.float32)
    q99 = np.asarray(norm_stats["state"].q99, dtype=np.float32)
    state_range = q99 - q01 + 1e-6

    profile = resolve_profile(ACTION_LORA_R16_STATE44_MODEL)
    raw_tokenizer = PaligemmaTokenizer(
        max_len=profile.max_tokens, fail_on_truncation=True
    )
    lengths: list[int] = []
    overflow_examples: list[dict] = []
    maximum_examples: list[dict] = []
    for audit_row_number, local_row in enumerate(local_rows, start=1):
        sequence = dataset._get_state44_sequence(local_row)
        window = dataset._row_windows[local_row]
        prompt_row = dataset._dataset.take(
            [dataset._source_row_indices[local_row]], columns=["prompt"]
        ).to_pylist()[0]
        prompt = format_language_prompt(
            prompt_row["prompt"],
            dataset._rows[local_row]["trajectory_metadata"],
            GESTURE_LANGUAGE,
            gesture=dataset._gesture_records[local_row].gesture,
        )
        for frame in range(window.start_frame, window.end_frame + 1):
            normalized = (sequence[frame] - q01) / state_range * 2.0 - 1.0
            raw_length = raw_pi05_token_length(raw_tokenizer, prompt, normalized)
            lengths.append(raw_length)
            record = {
                "source_row": int(dataset._source_row_indices[local_row]),
                "source_frame": int(frame),
                "raw_tokens": raw_length,
                "prompt": prompt,
            }
            if raw_length > profile.max_tokens and len(overflow_examples) < 100:
                overflow_examples.append(record)
            if not maximum_examples or raw_length >= maximum_examples[-1]["raw_tokens"]:
                maximum_examples.append(record)
                maximum_examples = sorted(
                    maximum_examples, key=lambda value: value["raw_tokens"], reverse=True
                )[:20]
        if audit_row_number % 100 == 0 or audit_row_number == len(local_rows):
            print(
                f"state44 token rows {audit_row_number}/{len(local_rows)} "
                f"samples={len(lengths)} max={max(lengths)}",
                flush=True,
            )

    overflow_count = int(sum(value > profile.max_tokens for value in lengths))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if overflow_count == 0 else "failed_token_budget",
        "model": ACTION_LORA_R16_STATE44_MODEL,
        "profile_id": profile.profile_id,
        "state_contract": STATE44_CONTRACT_ID,
        "state_schema": {
            "0:26": "mano_qpos26",
            "26:31": "target_object_contact5_index_thumb_ring_middle_pinky",
            "31": "object_lift_from_trajectory_initialization_m",
            "32:37": "signed_fingertip_sphere_to_target_collision_surface_distance5_m",
            "37:42": "causal25ms_fingertip_radial_rate5_m_per_s",
            "42": "object_floor_contact_pair_presence",
            "43": "elapsed_current_at_least_two_finger_contact_run_s",
        },
        "state_dim": STATE44_DIM,
        "action_dim": STATE44_ACTION_DIM,
        "action_horizon": profile.action_horizon,
        "max_token_len": profile.max_tokens,
        "fail_on_token_truncation": profile.fail_on_token_truncation,
        "clock": {
            "sample_dt_seconds": STATE44_SAMPLE_DT_SECONDS,
            "radial_rate_window_seconds": STATE44_RATE_WINDOW_SECONDS,
            "causal": True,
            "multicontact_reset": "reset_to_zero_when_fewer_than_two_fingers_contact",
            "multicontact_clip_seconds": STATE44_PERSISTENCE_CLIP_SECONDS,
            "normalization": "authenticated_population_quantile_q01_q99",
        },
        "client_git_commit": os.environ.get("VLA_CLIENT_GIT_COMMIT", "unknown"),
        "implementation_sha256": {
            str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).parents[1] / "mano_state44_contract.py",
                Path(__file__).parents[1] / "eval" / "mano_physics_core.py",
            )
        },
        "dataset": str(args.lance_dataset.resolve()),
        "target_dataset": str(target.resolve()),
        "row_selection": row_selection,
        "selected_frames": len(dataset),
        "frame_window": args.frame_window,
        "contact_context_frames": args.contact_context_frames,
        "contact_window_manifest": str(manifest) if manifest is not None else None,
        "contact_window_manifest_sha256": (
            hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest is not None else None
        ),
        "gesture_index": str(dataset._gesture_index_path),
        "gesture_index_sha256": dataset._gesture_index_sha256,
        "feature_precompute_seconds": feature_seconds,
        "token_audit": {
            "samples": len(lengths),
            "overflow_count": overflow_count,
            "min": min(lengths),
            "p50": _percentile(lengths, 50),
            "p95": _percentile(lengths, 95),
            "p99": _percentile(lengths, 99),
            "max": max(lengths),
            "maximum_examples": maximum_examples,
            "overflow_examples": overflow_examples,
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    if overflow_count:
        raise RuntimeError(
            f"state44 raw token audit found {overflow_count}/{len(lengths)} samples above "
            f"max_token_len={profile.max_tokens}; norm output was not written"
        )

    normalize.save(args.norm_output_dir, norm_stats)
    norm_path = args.norm_output_dir / "norm_stats.json"
    norm_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    report["norm"] = {
        "directory": str(args.norm_output_dir.resolve()),
        "path": str(norm_path.resolve()),
        "sha256": norm_sha,
        "state_width": STATE44_DIM,
        "action_width": STATE44_ACTION_DIM,
    }
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
