#!/usr/bin/env python3
"""Freeze cube/gesture selection, contact windows, state41 norms, and token audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import lance
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize

from scripts.mano_state41_contract import (
    ACTION_DIM,
    STATE41_CONTRACT_ID,
    STATE_DIM,
)
from scripts.openpi_profiles import ACTION_LORA_R16_STATE41_MODEL, resolve_profile
from scripts.target_actions import URDF_TARGET_ABSOLUTE
from scripts.train.train_cube1_01_compare import (
    OBJECT_ONLY_LANGUAGE,
    SelectedLanceDataset,
    selected_norm_stats,
)

EXPECTED_RELEASE_CONTRACT = "mano_28d_native_replay_state41_rgb_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".json.tmp", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def raw_token_length(
    tokenizer: PaligemmaTokenizer, prompt: str, normalized_state: np.ndarray
) -> int:
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    discrete = np.digitize(
        normalized_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]
    ) - 1
    state_text = " ".join(map(str, discrete))
    text = f"Task: {cleaned}, State: {state_text};\nAction: "
    return len(tokenizer._tokenizer.encode(text, add_bos=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--release-verification", type=Path, required=True)
    parser.add_argument("--object", default="cube1")
    parser.add_argument("--gesture", default="03")
    parser.add_argument("--contact-context-frames", type=int, default=100)
    parser.add_argument("--contact-window-manifest", type=Path, required=True)
    parser.add_argument("--norm-output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.contact_context_frames < 0:
        raise ValueError("--contact-context-frames must be non-negative")
    for path in (
        args.contact_window_manifest,
        args.norm_output_dir,
        args.report_json,
    ):
        if path.exists():
            raise FileExistsError(f"refusing existing output: {path}")
    verification = json.loads(args.release_verification.read_text())
    if verification.get("contract") != EXPECTED_RELEASE_CONTRACT:
        raise ValueError("release verification contract mismatch")
    if Path(verification.get("release", "")).resolve() != args.lance_dataset.resolve():
        raise ValueError("release verification points to a different Lance dataset")

    source = lance.dataset(str(args.lance_dataset))
    light = source.to_table(columns=["index", "provenance", "frame_count"]).to_pylist()
    if len(light) != int(verification["rows"]):
        raise ValueError("release row population disagrees with verification")
    selected = [
        (local_row, row)
        for local_row, row in enumerate(light)
        if row["index"]["object"] == args.object
        and str(row["index"]["gesture"]) == str(args.gesture)
        and row["index"]["grade"] in ("A", "B")
    ]
    if not selected:
        raise ValueError(
            f"no qualified rows for object={args.object!r} gesture={args.gesture!r}"
        )
    row_indices = [local_row for local_row, _ in selected]
    selection = [
        {
            "release_row_index": local_row,
            "filtered_row_index": int(row["index"]["filtered_row_index"]),
            "original_merged_row_index": int(
                row["index"]["original_merged_row_index"]
            ),
            "uuid": row["index"]["uuid"],
            "grade": row["index"]["grade"],
            "frames": int(row["frame_count"]),
        }
        for local_row, row in selected
    ]
    if len({row["uuid"] for row in selection}) != len(selection):
        raise ValueError("selected UUIDs are not unique")
    if any(
        row["provenance"]["state_contract"] != STATE41_CONTRACT_ID
        for _, row in selected
    ):
        raise ValueError("selected release rows do not use state41")

    dataset = SelectedLanceDataset(
        args.lance_dataset,
        row_indices=row_indices,
        action_horizon=10,
        frame_window="contact",
        contact_context_frames=args.contact_context_frames,
        contact_window_manifest=args.contact_window_manifest,
        missing_contact_policy="error",
        action_source=URDF_TARGET_ABSOLUTE,
        language_conditioning=OBJECT_ONLY_LANGUAGE,
        target_lance_dataset=None,
        state_contract="state41",
    )
    started = time.perf_counter()
    norm_stats = selected_norm_stats(dataset)
    if np.asarray(norm_stats["state"].mean).shape != (STATE_DIM,):
        raise RuntimeError("computed state normalization width is not 41")
    if np.asarray(norm_stats["actions"].mean).shape != (ACTION_DIM,):
        raise RuntimeError("computed action normalization width is not 32")

    profile = resolve_profile(ACTION_LORA_R16_STATE41_MODEL)
    tokenizer = PaligemmaTokenizer(
        max_len=profile.max_tokens, fail_on_truncation=True
    )
    q01 = np.asarray(norm_stats["state"].q01, dtype=np.float32)
    q99 = np.asarray(norm_stats["state"].q99, dtype=np.float32)
    scale = q99 - q01 + 1e-6
    lengths: list[int] = []
    maximum_examples: list[dict[str, Any]] = []
    for local_row in sorted(dataset._row_start_offset):
        source_row = dataset._source_row_indices[local_row]
        row = dataset._dataset.take(
            [source_row], columns=["state", "prompt"]
        ).to_pylist()[0]
        sequence = np.asarray(row["state"], dtype=np.float32)
        window = dataset._row_windows[local_row]
        for frame in range(window.start_frame, window.end_frame + 1):
            normalized = (sequence[frame] - q01) / scale * 2.0 - 1.0
            length = raw_token_length(tokenizer, row["prompt"], normalized)
            lengths.append(length)
            record = {
                "release_row_index": int(source_row),
                "source_frame": int(frame),
                "raw_tokens": int(length),
                "prompt": row["prompt"],
            }
            maximum_examples.append(record)
            maximum_examples = sorted(
                maximum_examples,
                key=lambda value: value["raw_tokens"],
                reverse=True,
            )[:20]
    overflow_count = sum(length > profile.max_tokens for length in lengths)
    if overflow_count:
        raise RuntimeError(
            f"state41 token audit found {overflow_count}/{len(lengths)} samples "
            f"above max_token_len={profile.max_tokens}"
        )

    temporary_norm = args.norm_output_dir.with_name(
        f".{args.norm_output_dir.name}.incoming-{os.getpid()}"
    )
    if temporary_norm.exists():
        raise FileExistsError(f"temporary norm output exists: {temporary_norm}")
    normalize.save(temporary_norm, norm_stats)
    os.replace(temporary_norm, args.norm_output_dir)
    norm_path = args.norm_output_dir / "norm_stats.json"
    manifest_sha = sha256(args.contact_window_manifest)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "model": ACTION_LORA_R16_STATE41_MODEL,
        "profile_id": profile.profile_id,
        "state_contract": STATE41_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": profile.action_horizon,
        "max_token_len": profile.max_tokens,
        "fail_on_token_truncation": profile.fail_on_token_truncation,
        "dataset": str(args.lance_dataset.resolve()),
        "release_verification": str(args.release_verification.resolve()),
        "release_verification_sha256": sha256(args.release_verification),
        "release_plan_sha256": verification["release_plan_sha256"],
        "object": args.object,
        "gesture": str(args.gesture),
        "selection": selection,
        "selection_uuid_sha256": canonical_sha256(
            [row["uuid"] for row in selection]
        ),
        "selected_rows": len(selection),
        "source_full_frames": sum(row["frames"] for row in selection),
        "training_frames": len(dataset),
        "frame_window": "contact",
        "contact_context_frames": args.contact_context_frames,
        "contact_window_manifest": str(args.contact_window_manifest.resolve()),
        "contact_window_manifest_sha256": manifest_sha,
        "norm": {
            "directory": str(args.norm_output_dir.resolve()),
            "path": str(norm_path.resolve()),
            "sha256": sha256(norm_path),
            "state_width": STATE_DIM,
            "action_width": ACTION_DIM,
        },
        "token_audit": {
            "samples": len(lengths),
            "overflow_count": int(overflow_count),
            "min": int(min(lengths)),
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(max(lengths)),
            "maximum_examples": maximum_examples,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "client_git_commit": os.environ.get("VLA_CLIENT_GIT_COMMIT", "unknown"),
        "implementation_sha256": {
            "prepare": sha256(Path(__file__).resolve()),
            "state_contract": sha256(
                Path(__file__).parents[1] / "mano_state41_contract.py"
            ),
        },
    }
    atomic_json(args.report_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
