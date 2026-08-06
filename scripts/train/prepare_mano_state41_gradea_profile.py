#!/usr/bin/env python3
"""Publish the full Grade-A state41 train/validation profile atomically."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import lance
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize

from scripts import contact_windows
from scripts.mano_state41_contract import ACTION_DIM, STATE41_CONTRACT_ID, STATE_DIM
from scripts.openpi_profiles import ACTION_LORA_R16_STATE41_MODEL, resolve_profile
from scripts.target_actions import URDF_TARGET_ABSOLUTE
from scripts.train.state41_gradea_contract import (
    canonical_release_gesture_prompt,
    canonical_sha256,
    selection_manifest,
    split_grade_a_rows,
)
from scripts.train.train_cube1_01_compare import (
    GESTURE_LANGUAGE,
    SelectedLanceDataset,
    selected_norm_stats,
)
from scripts.train.prepare_mano_state41_profile import raw_token_length


EXPECTED_RELEASE_CONTRACT = "mano_28d_native_replay_state41_rgb_v1"
PROFILE_CONTRACT = "mano_state41_grade_a_train_profile_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incoming-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--release-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--contact-context-frames", type=int, default=100)
    return parser.parse_args()


def grade_a_records(light_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for release_row_index, source in enumerate(light_rows):
        index = source.get("index") or {}
        if index.get("grade") != "A":
            continue
        provenance = source.get("provenance") or {}
        if provenance.get("state_contract") != STATE41_CONTRACT_ID:
            raise ValueError(
                f"Grade-A release row {release_row_index} does not use state41"
            )
        record = {
            "release_row_index": int(release_row_index),
            "filtered_row_index": int(index["filtered_row_index"]),
            "original_merged_row_index": int(index["original_merged_row_index"]),
            "uuid": str(index["uuid"]),
            "seed_uuid": str(index["seed_uuid"]),
            "object": str(index["object"]),
            "gesture": str(index["gesture"]),
            "grade": str(index["grade"]),
            "frames": int(source["frame_count"]),
        }
        record["prompt"] = canonical_release_gesture_prompt(record)
        records.append(record)
    if not records:
        raise ValueError("formal release contains no Grade-A rows")
    return records


def _token_audit(
    *,
    dataset: SelectedLanceDataset,
    prompts: Mapping[int, str],
    norm_stats: Mapping[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    tokenizer = PaligemmaTokenizer(max_len=max_tokens, fail_on_truncation=True)
    q01 = np.asarray(norm_stats["state"].q01, dtype=np.float32)
    q99 = np.asarray(norm_stats["state"].q99, dtype=np.float32)
    scale = q99 - q01 + 1e-6
    lengths: list[int] = []
    maximum_examples: list[dict[str, Any]] = []
    for local_row in sorted(dataset._row_start_offset):
        source_row = int(dataset._source_row_indices[local_row])
        row = dataset._dataset.take([source_row], columns=["state", "index"]).to_pylist()[0]
        prompt = prompts[source_row]
        expected = canonical_release_gesture_prompt(row["index"])
        if prompt != expected:
            raise ValueError(
                f"canonical prompt mismatch at release row {source_row}: {prompt!r} != {expected!r}"
            )
        sequence = np.asarray(row["state"], dtype=np.float32)
        window = dataset._row_windows[local_row]
        for frame in range(window.start_frame, window.end_frame + 1):
            normalized = (sequence[frame] - q01) / scale * 2.0 - 1.0
            length = raw_token_length(tokenizer, prompt, normalized)
            lengths.append(length)
            maximum_examples.append(
                {
                    "release_row_index": source_row,
                    "source_frame": int(frame),
                    "raw_tokens": int(length),
                    "prompt": prompt,
                }
            )
            maximum_examples = sorted(
                maximum_examples,
                key=lambda value: value["raw_tokens"],
                reverse=True,
            )[:20]
    if not lengths:
        raise RuntimeError("Grade-A token audit has no training samples")
    overflow_count = sum(length > max_tokens for length in lengths)
    if overflow_count:
        raise RuntimeError(
            f"state41 Grade-A token audit found {overflow_count}/{len(lengths)} "
            f"samples above max_token_len={max_tokens}"
        )
    return {
        "samples": len(lengths),
        "overflow_count": int(overflow_count),
        "min": int(min(lengths)),
        "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(max(lengths)),
        "maximum_examples": maximum_examples,
    }


def main() -> int:
    args = parse_args()
    if args.contact_context_frames < 0:
        raise ValueError("--contact-context-frames must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing profile output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.incoming-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging output exists: {staging}")
    staging.mkdir()

    started = time.perf_counter()
    try:
        lance_dataset = args.lance_dataset.expanduser().resolve()
        release_verification = args.release_verification.expanduser().resolve()
        verification = json.loads(release_verification.read_text())
        if verification.get("contract") != EXPECTED_RELEASE_CONTRACT:
            raise ValueError("release verification contract mismatch")
        if Path(verification.get("release", "")).resolve() != lance_dataset:
            raise ValueError("release verification points to a different Lance dataset")

        source = lance.dataset(str(lance_dataset))
        light = source.to_table(
            columns=["index", "provenance", "frame_count"]
        ).to_pylist()
        if len(light) != int(verification["rows"]):
            raise ValueError("release row population disagrees with verification")
        population = grade_a_records(light)
        split = split_grade_a_rows(
            population,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
        )
        release_sha = sha256(release_verification)
        train_manifest = selection_manifest(
            split,
            split_name="train",
            dataset=str(lance_dataset),
            release_verification_sha256=release_sha,
        )
        validation_manifest = selection_manifest(
            split,
            split_name="validation",
            dataset=str(lance_dataset),
            release_verification_sha256=release_sha,
        )
        train_selection_path = staging / "train_selection.json"
        validation_selection_path = staging / "validation_selection.json"
        atomic_json(train_selection_path, train_manifest)
        atomic_json(validation_selection_path, validation_manifest)

        train_indices = [int(row["release_row_index"]) for row in split["train"]]
        validation_indices = [
            int(row["release_row_index"]) for row in split["validation"]
        ]
        train_contact_path = staging / "train_contact_windows.json"
        validation_contact_path = staging / "validation_contact_windows.json"
        train_dataset = SelectedLanceDataset(
            lance_dataset,
            row_indices=train_indices,
            action_horizon=10,
            frame_window="contact",
            contact_context_frames=args.contact_context_frames,
            contact_window_manifest=train_contact_path,
            missing_contact_policy="error",
            action_source=URDF_TARGET_ABSOLUTE,
            language_conditioning=GESTURE_LANGUAGE,
            target_lance_dataset=None,
            state_contract="state41",
        )
        contact_windows.load_or_build_windows(
            source,
            lance_dataset,
            validation_indices,
            manifest_path=validation_contact_path,
            context_frames=args.contact_context_frames,
            missing_policy="error",
        )

        norm_stats = selected_norm_stats(train_dataset)
        if np.asarray(norm_stats["state"].mean).shape != (STATE_DIM,):
            raise RuntimeError("computed state normalization width is not 41")
        if np.asarray(norm_stats["actions"].mean).shape != (ACTION_DIM,):
            raise RuntimeError("computed action normalization width is not 32")
        normalize.save(staging / "norm", norm_stats)
        norm_path = staging / "norm/norm_stats.json"

        profile = resolve_profile(ACTION_LORA_R16_STATE41_MODEL)
        prompts = {
            int(row["release_row_index"]): str(row["prompt"])
            for row in split["train"]
        }
        token_audit = _token_audit(
            dataset=train_dataset,
            prompts=prompts,
            norm_stats=norm_stats,
            max_tokens=profile.max_tokens,
        )
        final = output_dir
        report = {
            "contract": PROFILE_CONTRACT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "dataset": str(lance_dataset),
            "release_verification": str(release_verification),
            "release_verification_sha256": release_sha,
            "release_plan_sha256": verification["release_plan_sha256"],
            "population": "grade_a",
            "population_rows": split["population_rows"],
            "train_rows": split["train_rows"],
            "validation_rows": split["validation_rows"],
            "validation_fraction": split["validation_fraction"],
            "split_seed": split["split_seed"],
            "strata": split["strata"],
            "train_selection_manifest": str(final / "train_selection.json"),
            "train_selection_manifest_sha256": sha256(train_selection_path),
            "validation_selection_manifest": str(final / "validation_selection.json"),
            "validation_selection_manifest_sha256": sha256(
                validation_selection_path
            ),
            "train_uuid_sha256": split["train_uuid_sha256"],
            "validation_uuid_sha256": split["validation_uuid_sha256"],
            "frame_window": "contact",
            "contact_context_frames": args.contact_context_frames,
            "missing_contact_policy": "error",
            "train_contact_window_manifest": str(
                final / "train_contact_windows.json"
            ),
            "train_contact_window_manifest_sha256": sha256(train_contact_path),
            "validation_contact_window_manifest": str(
                final / "validation_contact_windows.json"
            ),
            "validation_contact_window_manifest_sha256": sha256(
                validation_contact_path
            ),
            "training_frames": len(train_dataset),
            "language_conditioning": GESTURE_LANGUAGE,
            "prompt_template": "pick up the {object} using gesture {gesture}",
            "model": ACTION_LORA_R16_STATE41_MODEL,
            "profile_id": profile.profile_id,
            "state_contract": STATE41_CONTRACT_ID,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": profile.action_horizon,
            "delta_mask_segments": list(profile.delta_mask_segments),
            "max_token_len": profile.max_tokens,
            "fail_on_token_truncation": profile.fail_on_token_truncation,
            "norm_population": "train_only_contact_window",
            "norm": {
                "directory": str(final / "norm"),
                "path": str(final / "norm/norm_stats.json"),
                "sha256": sha256(norm_path),
                "state_width": STATE_DIM,
                "action_width": ACTION_DIM,
            },
            "token_audit": token_audit,
            "sampling_default": {
                "strategy": "sqrt_tempered",
                "coverage_anchors_per_row": 8,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "client_git_commit": os.environ.get("VLA_CLIENT_GIT_COMMIT", "unknown"),
            "implementation_sha256": {
                "prepare": sha256(Path(__file__).resolve()),
                "split_contract": sha256(
                    Path(__file__).with_name("state41_gradea_contract.py")
                ),
            },
        }
        atomic_json(staging / "profile_report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
