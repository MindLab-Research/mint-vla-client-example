#!/usr/bin/env python3
"""Derive State54 geometry/load features from accepted target-replay traces.

The source replay Lance intentionally omits MANO joint positions and per-contact
normal loads.  This module reconstructs both from the exact saved physical
trace without weakening the State54 contract:

* fingertip XYZ comes from the same audited MuJoCo FK used by Mode4;
* finger loads come from the same live MuJoCo contact-force aggregation;
* replayed hand/object/contact arrays must match the accepted trace.

The module never writes to the source release.  CLI outputs are explicit paths
owned by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import mujoco
import numpy as np

from scripts.eval import mano_physics_core as physics
from scripts.mano_state54_contract import (
    FINGER_NAMES,
    fingertips_in_collision_box_frame,
    fingertip_world_from_mujoco,
)

HAND_DIM = 26
SOURCE_DT_SECONDS = 0.005
PHYSICS_SUBSTEPS = 2
HAND_ATOL = 2e-6
OBJECT_POSITION_ATOL = 2e-6
OBJECT_QUATERNION_ATOL = 2e-6


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".npz") as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _require_trace_array(
    archive: Any, name: str, shape: tuple[int, ...], *, dtype: np.dtype[Any]
) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"accepted replay trace is missing {name!r}")
    value = np.asarray(archive[name], dtype=dtype)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"accepted replay trace {name!r} must be finite {shape}, got {value.shape}")
    return value


def load_accepted_trace(path: str | Path) -> dict[str, np.ndarray]:
    trace_path = Path(path)
    with np.load(trace_path) as archive:
        if "timestamp" not in archive:
            raise ValueError("accepted replay trace is missing 'timestamp'")
        timestamp = np.asarray(archive["timestamp"], dtype=np.float64)
        if timestamp.ndim != 1 or len(timestamp) < 2 or not np.all(np.isfinite(timestamp)):
            raise ValueError("accepted replay timestamp must be a finite vector with at least two frames")
        frame_count = len(timestamp)
        if not np.allclose(
            np.diff(timestamp), SOURCE_DT_SECONDS, rtol=0, atol=1e-10
        ):
            raise ValueError("accepted replay trace must use exact 5 ms source intervals")
        result = {
            "timestamp": timestamp,
            "hand_qpos": _require_trace_array(
                archive, "hand_qpos", (frame_count, HAND_DIM), dtype=np.float32
            ),
            "object_position": _require_trace_array(
                archive, "object_position", (frame_count, 3), dtype=np.float32
            ),
            "object_quaternion_wxyz": _require_trace_array(
                archive, "object_quaternion_wxyz", (frame_count, 4), dtype=np.float32
            ),
            "contacts": _require_trace_array(
                archive, "contacts", (frame_count, len(FINGER_NAMES)), dtype=np.float32
            ),
            "absolute_target_qpos": _require_trace_array(
                archive, "absolute_target_qpos", (frame_count, HAND_DIM), dtype=np.float32
            ),
        }
    return result


def _quaternion_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    direct = np.max(np.abs(actual - expected), axis=1)
    negated = np.max(np.abs(actual + expected), axis=1)
    return np.minimum(direct, negated)


def replay_trace_state54_features(
    trace_path: str | Path,
    *,
    object_name: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    trace = load_accepted_trace(trace_path)
    hand_expected = trace["hand_qpos"]
    position_expected = trace["object_position"]
    quaternion_expected = trace["object_quaternion_wxyz"]
    contacts_expected = trace["contacts"]
    targets = trace["absolute_target_qpos"]
    frame_count = len(trace["timestamp"])

    temporary, model, _unused_data, renderer, object_addr, _, hand_addrs, _, limits = (
        physics.make_scene(
            object_name,
            1,
            1,
            physics=True,
            physics_timestep=physics.DT,
            create_renderer=False,
        )
    )
    if renderer is not None:
        temporary.cleanup()
        raise RuntimeError("force replay unexpectedly created a renderer")
    try:
        keypoint_geom_ids, object_geom_ids, geom_id_to_finger = (
            physics.resolve_keypoint_geom_ids(model, object_name)
        )
        object_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{object_name}_body"
        )
        if object_body_id < 0:
            raise ValueError(f"cannot resolve MuJoCo body for {object_name!r}")

        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        data.qpos[object_addr : object_addr + 3] = position_expected[0]
        data.qpos[object_addr + 3 : object_addr + 7] = quaternion_expected[0]
        data.qpos[hand_addrs] = hand_expected[0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        replay_hand = np.empty_like(hand_expected)
        replay_position = np.empty_like(position_expected)
        replay_quaternion = np.empty_like(quaternion_expected)
        replay_contacts = np.empty_like(contacts_expected)
        replay_log1p_force = np.empty_like(contacts_expected)
        replay_tip_box_xyz = np.empty((frame_count, 5, 3), dtype=np.float32)
        max_target_limit_correction = 0.0

        for frame in range(frame_count):
            hand = np.asarray(data.qpos[hand_addrs], dtype=np.float32)
            position = np.asarray(data.qpos[object_addr : object_addr + 3], dtype=np.float32)
            quaternion = np.asarray(data.qpos[object_addr + 3 : object_addr + 7], dtype=np.float32)
            contacts, log1p_force = physics.finger_contact_and_force_from_mujoco(
                model,
                data,
                object_name,
                keypoint_geom_ids=keypoint_geom_ids,
                object_geom_ids=object_geom_ids,
                geom_id_to_finger=geom_id_to_finger,
            )
            tip_world = fingertip_world_from_mujoco(model, data)
            object_rotation = np.asarray(
                data.xmat[object_body_id], dtype=np.float64
            ).reshape(3, 3)
            tip_box_xyz = fingertips_in_collision_box_frame(
                tip_world,
                np.asarray(data.xpos[object_body_id], dtype=np.float64),
                object_rotation,
                object_name,
            )
            replay_hand[frame] = hand
            replay_position[frame] = position
            replay_quaternion[frame] = quaternion
            replay_contacts[frame] = contacts
            replay_log1p_force[frame] = log1p_force
            replay_tip_box_xyz[frame] = tip_box_xyz

            if frame == frame_count - 1:
                break
            target, event = physics.nearest_wrapped_position_target(
                np.asarray(hand, dtype=np.float64),
                np.asarray(targets[frame], dtype=np.float64)
                - np.asarray(hand, dtype=np.float64),
                limits,
            )
            if event is not None:
                max_target_limit_correction = max(
                    max_target_limit_correction, float(event["max_correction"])
                )
            physics.step_servo(
                model=model,
                data=data,
                target=target,
                substeps=PHYSICS_SUBSTEPS,
                object_name=object_name,
            )
            # Current dev_v3 predates the replay fix that refreshes derived
            # kinematics/contact state after the final mj_step integration.
            # Refresh explicitly so observations correspond to data.qpos and
            # accepted State44 replay semantics.
            mujoco.mj_forward(model, data)

        expected_time = PHYSICS_SUBSTEPS * (frame_count - 1) * physics.DT
        if not np.isclose(data.time, expected_time, rtol=0, atol=1e-9):
            raise RuntimeError(f"force replay time mismatch: {data.time} != {expected_time}")

        hand_error = np.max(np.abs(replay_hand - hand_expected), axis=1)
        position_error = np.max(np.abs(replay_position - position_expected), axis=1)
        quaternion_error = _quaternion_error(replay_quaternion, quaternion_expected)
        contact_mismatch = replay_contacts != contacts_expected
        finite_features = bool(
            np.all(np.isfinite(replay_log1p_force))
            and np.all(np.isfinite(replay_tip_box_xyz))
        )
        diagnostics: dict[str, Any] = {
            "status": "passed"
            if (
                float(np.max(hand_error)) <= HAND_ATOL
                and float(np.max(position_error)) <= OBJECT_POSITION_ATOL
                and float(np.max(quaternion_error)) <= OBJECT_QUATERNION_ATOL
                and not bool(np.any(contact_mismatch))
                and finite_features
            )
            else "failed",
            "object_name": object_name,
            "trace_path": str(Path(trace_path).resolve()),
            "trace_sha256": sha256_file(trace_path),
            "frame_count": frame_count,
            "source_dt_seconds": SOURCE_DT_SECONDS,
            "mujoco_dt_seconds": float(physics.DT),
            "physics_substeps": PHYSICS_SUBSTEPS,
            "max_abs_hand_qpos_error": float(np.max(hand_error)),
            "max_abs_object_position_error": float(np.max(position_error)),
            "max_sign_invariant_object_quaternion_error": float(np.max(quaternion_error)),
            "contact_mismatch_frames": int(np.count_nonzero(np.any(contact_mismatch, axis=1))),
            "contact_mismatch_values": int(np.count_nonzero(contact_mismatch)),
            "force_nonzero_frames": int(
                np.count_nonzero(np.any(replay_log1p_force > 0.0, axis=1))
            ),
            "force_nonzero_values": int(np.count_nonzero(replay_log1p_force > 0.0)),
            "max_log1p_force": float(np.max(replay_log1p_force)),
            "features_finite": finite_features,
            "max_target_limit_correction": max_target_limit_correction,
            "thresholds": {
                "hand_qpos_atol": HAND_ATOL,
                "object_position_atol": OBJECT_POSITION_ATOL,
                "object_quaternion_atol": OBJECT_QUATERNION_ATOL,
                "contacts_exact": True,
            },
        }
        arrays = {
            "timestamp": trace["timestamp"],
            "finger_contacts": replay_contacts,
            "finger_log1p_force": replay_log1p_force,
            "fingertip_collision_box_xyz": replay_tip_box_xyz,
            "replayed_hand_qpos": replay_hand,
            "replayed_object_position": replay_position,
            "replayed_object_quaternion_wxyz": replay_quaternion,
        }
        return diagnostics, arrays
    finally:
        temporary.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-trace-sha256")
    parser.add_argument("--require-parity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actual_trace_sha = sha256_file(args.trace)
    if args.expected_trace_sha256 and actual_trace_sha != args.expected_trace_sha256.lower():
        raise ValueError(
            f"trace SHA mismatch: expected {args.expected_trace_sha256.lower()}, got {actual_trace_sha}"
        )
    report, arrays = replay_trace_state54_features(args.trace, object_name=args.object)
    if args.output_npz is not None:
        atomic_npz(args.output_npz, **arrays)
        report["output_npz"] = str(args.output_npz.resolve())
        report["output_npz_sha256"] = sha256_file(args.output_npz)
    if args.report is not None:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_parity and report["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
