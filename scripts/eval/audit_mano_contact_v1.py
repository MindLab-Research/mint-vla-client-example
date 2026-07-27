#!/usr/bin/env python3
"""Audit v1 training contact records against MuJoCo keypoint-pair presence.

The audit replays exact recorded hand/object poses with ``mj_forward`` and
separates three mechanisms per finger:
1. pair present but every solved force is <= 0.01 N (old threshold-only skew),
2. training record present but MuJoCo pair absent,
3. MuJoCo pair present but training record absent.

Pinky is reported but excluded from the blocking gate because this population
contains only six positive pinky frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lance
import mujoco
import numpy as np

from scripts.eval import mano_physics_core as physics
from scripts.eval import mode4_data_support as mode4_data
from scripts.mano_state_contract import (
    CONTACT_SEMANTICS,
    FINGER_NAMES,
    STATE_CONTRACT_ID,
    aggregate_finger_contacts,
)

OLD_FORCE_THRESHOLD_N = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--row-start", type=int, default=810)
    parser.add_argument("--row-end", type=int, default=994)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-examples-per-bin", type=int, default=20)
    return parser.parse_args()


def pair_and_force_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    keypoint_geom_ids: set[int],
    object_geom_ids: set[int],
    geom_id_to_finger: dict[int, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pair presence, old-force-qualified presence, and max force."""
    pair = np.zeros(len(FINGER_NAMES), dtype=np.bool_)
    force_qualified = np.zeros(len(FINGER_NAMES), dtype=np.bool_)
    max_force = np.zeros(len(FINGER_NAMES), dtype=np.float64)
    force_6d = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in object_geom_ids and geom2 in keypoint_geom_ids:
            hand_geom = geom2
        elif geom2 in object_geom_ids and geom1 in keypoint_geom_ids:
            hand_geom = geom1
        else:
            continue
        finger = geom_id_to_finger.get(hand_geom)
        if finger is None:  # palm integrity geom
            continue
        finger_index = FINGER_NAMES.index(finger)
        pair[finger_index] = True
        mujoco.mj_contactForce(model, data, contact_index, force_6d)
        force_norm = float(np.linalg.norm(force_6d[:3]))
        max_force[finger_index] = max(max_force[finger_index], force_norm)
        if force_norm > OLD_FORCE_THRESHOLD_N:
            force_qualified[finger_index] = True
    return pair, force_qualified, max_force


def main() -> int:
    args = parse_args()
    if args.row_start > args.row_end:
        raise ValueError("row-start must be <= row-end")
    rows = list(range(args.row_start, args.row_end + 1))
    dataset = lance.dataset(str(args.lance_dataset))
    if rows[-1] >= dataset.count_rows():
        raise IndexError(f"row {rows[-1]} outside dataset")

    tmp, model, data, _, object_addr, _, hand_addrs, _, _ = physics.make_scene(
        "cube1", 64, 64, physics=True, physics_timestep=physics.DT, create_renderer=False
    )
    keypoint_ids, object_ids, geom_to_finger = physics.resolve_keypoint_geom_ids(
        model, "cube1"
    )

    bins = {
        name: {finger: 0 for finger in FINGER_NAMES}
        for name in (
            "pair_present_force_le_0p01",
            "record_present_pair_absent",
            "pair_present_record_absent",
        )
    }
    positives = {
        name: {finger: 0 for finger in FINGER_NAMES}
        for name in ("record", "pair", "force_gt_0p01")
    }
    examples: dict[str, list[dict]] = {name: [] for name in bins}
    total_frames = 0

    try:
        for row_index in rows:
            row = dataset.take(
                [row_index], columns=["state", "objects", "contact", "trajectory_metadata"]
            ).to_pylist()[0]
            metadata = row.get("trajectory_metadata") or {}
            object_names = metadata.get("object_names") or []
            if object_names != ["cube1"] or len(row["objects"]) != 1:
                raise ValueError(
                    f"row {row_index}: expected one cube1 object, got {object_names!r}"
                )
            state = np.asarray(row["state"], dtype=np.float64)
            object_pos = np.asarray(row["objects"][0]["pos"], dtype=np.float64)
            object_rot = np.asarray(row["objects"][0]["rot_aa"], dtype=np.float64)
            contacts = row["contact"]
            frame_count = state.shape[0]
            if object_pos.shape[0] != frame_count or len(contacts) != frame_count:
                raise ValueError(f"row {row_index}: frame-alignment mismatch")

            for frame_index in range(frame_count):
                record = aggregate_finger_contacts(
                    contacts[frame_index] or [], "cube1"
                ).astype(np.bool_)
                mode4_data.set_scene_state(
                    model,
                    data,
                    state=state[frame_index, :26],
                    object_pos=object_pos[frame_index],
                    object_rot_aa=object_rot[frame_index],
                    object_addr=object_addr,
                    hand_addrs=hand_addrs,
                )
                pair, force_qualified, max_force = pair_and_force_contacts(
                    model,
                    data,
                    keypoint_geom_ids=keypoint_ids,
                    object_geom_ids=object_ids,
                    geom_id_to_finger=geom_to_finger,
                )
                masks = {
                    "pair_present_force_le_0p01": pair & ~force_qualified,
                    "record_present_pair_absent": record & ~pair,
                    "pair_present_record_absent": pair & ~record,
                }
                for finger_index, finger in enumerate(FINGER_NAMES):
                    positives["record"][finger] += int(record[finger_index])
                    positives["pair"][finger] += int(pair[finger_index])
                    positives["force_gt_0p01"][finger] += int(
                        force_qualified[finger_index]
                    )
                    for bin_name, mask in masks.items():
                        if not mask[finger_index]:
                            continue
                        bins[bin_name][finger] += 1
                        if len(examples[bin_name]) < args.max_examples_per_bin:
                            examples[bin_name].append(
                                {
                                    "row": row_index,
                                    "frame": frame_index,
                                    "finger": finger,
                                    "record": bool(record[finger_index]),
                                    "pair": bool(pair[finger_index]),
                                    "force_gt_0p01": bool(force_qualified[finger_index]),
                                    "max_force_n": float(max_force[finger_index]),
                                }
                            )
                total_frames += 1
            if (row_index - args.row_start + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "rows_complete": row_index - args.row_start + 1,
                            "total_rows": len(rows),
                            "frames_complete": total_frames,
                        }
                    ),
                    flush=True,
                )
    finally:
        tmp.cleanup()

    blocking_fingers = [finger for finger in FINGER_NAMES if finger != "pinky"]
    blocking_disagreements = sum(
        bins[bin_name][finger]
        for bin_name in ("record_present_pair_absent", "pair_present_record_absent")
        for finger in blocking_fingers
    )
    result = {
        "state_contract": STATE_CONTRACT_ID,
        "contact_semantics": CONTACT_SEMANTICS,
        "dataset": str(args.lance_dataset),
        "rows": {"start": args.row_start, "end": args.row_end, "count": len(rows)},
        "total_frames": total_frames,
        "finger_decisions": total_frames * len(FINGER_NAMES),
        "old_force_threshold_n": OLD_FORCE_THRESHOLD_N,
        "positives": positives,
        "bins": bins,
        "examples": examples,
        "blocking_fingers": blocking_fingers,
        "blocking_disagreements": blocking_disagreements,
        "passed": blocking_disagreements == 0,
        "pinky_reported_not_gated": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
