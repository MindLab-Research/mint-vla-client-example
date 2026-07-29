#!/usr/bin/env python3
"""Replay dataset MANO absolute target DOFs through the canonical Mode4 physics.

No policy or MINT action session is used. The replay initializes simulation once
at source frame 0, holds target[f] over interval f->f+1, and records the same
MuJoCo/controller diagnostics and side-by-side videos as policy Mode4.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import lance
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM") or "egl"
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import mode4_data_support as full
import mano_physics_core as physics
from mano_joint_limits import clip_hand_state, new_clipping_diagnostics, record_clipping
from mode4_support import parse_ordered_unique_csv

HAND_DIM = 26


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lance-dataset", type=Path, required=True)
    rows = p.add_mutually_exclusive_group(required=True)
    rows.add_argument("--row-index", type=int)
    rows.add_argument("--row-indices")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    return p.parse_args()


def decode_jpeg(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return np.asarray(frame, dtype=np.uint8)
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR))


def label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    ImageDraw.Draw(image).rectangle((0, 0, min(image.width, 600), 24), fill=(0, 0, 0))
    ImageDraw.Draw(image).text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def load_row(source, row_index: int) -> dict:
    columns = [
        "state", "hands", "objects", "timestamp", "trajectory_metadata",
        "episode_metadata", "image", "wrist_image", "index", "prompt",
    ]
    return source.take([row_index], columns=columns).to_pylist()[0]


def run_row(args: argparse.Namespace, row: dict, row_index: int, output_dir: Path) -> dict:
    object_name = full.safe_object_name(row)
    source_frames = full.row_frame_count(row)
    timestamps = np.asarray(row["timestamp"], dtype=np.float64)
    if timestamps.shape != (source_frames,) or not np.allclose(
        np.diff(timestamps), 0.005, rtol=0, atol=1e-10
    ):
        raise ValueError("GT Mode4 replay requires exact 200 Hz timestamps")
    targets = np.asarray(row["hands"][0]["urdf_dof_target"], dtype=np.float64)
    if targets.shape != (source_frames, HAND_DIM) or not np.isfinite(targets).all():
        raise ValueError(f"invalid target DOFs: {targets.shape}")
    frame_count = source_frames if args.max_frames <= 0 else min(source_frames, args.max_frames)
    if frame_count < 2:
        raise ValueError("GT replay needs at least two frames")
    out = output_dir / "gt_replay"
    out.mkdir(parents=True, exist_ok=True)
    tmp = renderer = None
    try:
        tmp, model, data, renderer, object_addr, _, hand_addrs, _, limits = physics.make_scene(
            object_name, args.width, args.height, physics=True,
            physics_timestep=physics.DT, create_renderer=True,
        )
        initial = np.asarray(row["state"][0], dtype=np.float64)[:HAND_DIM]
        data.qvel[:] = 0.0
        full.set_scene_state(
            model, data, state=initial,
            object_pos=row["objects"][0]["pos"][0],
            object_rot_aa=row["objects"][0]["rot_aa"][0],
            object_addr=object_addr, hand_addrs=hand_addrs,
        )
        mujoco.mj_forward(model, data)
        clipping = new_clipping_diagnostics(limits)
        contacts = {"hand_object": 0, "object_floor": 0, "hand_floor": 0}
        max_ncon = max_force = max_actuator = max_qvel = 0.0
        hand_states = [np.asarray(data.qpos[hand_addrs], dtype=np.float32).copy()]
        object_positions = [np.asarray(data.qpos[object_addr:object_addr + 3], dtype=np.float32).copy()]
        object_quaternions = [np.asarray(data.qpos[object_addr + 3:object_addr + 7], dtype=np.float32).copy()]
        raw_targets, applied_targets = [], []
        head_path = out / "gt_physics_vs_dataset_head.mp4"
        wrist_path = out / "gt_physics_vs_dataset_wrist.mp4"
        dataset_path = out / "dataset_reference.mp4"
        with imageio.get_writer(str(head_path), fps=args.fps, macro_block_size=1, codec="libx264") as hw, \
             imageio.get_writer(str(wrist_path), fps=args.fps, macro_block_size=1, codec="libx264") as ww, \
             imageio.get_writer(str(dataset_path), fps=args.fps, macro_block_size=1, codec="libx264") as dw:
            sim_h, sim_w = physics.render_current_state(model, data, renderer)
            data_h = resize(decode_jpeg(row["image"][0]), args.width, args.height)
            data_w = resize(decode_jpeg(row["wrist_image"][0]), args.width, args.height)
            hw.append_data(np.concatenate((label(sim_h, "GT PHYSICS frame 0"), label(data_h, "DATASET frame 0")), axis=1))
            ww.append_data(np.concatenate((label(sim_w, "GT PHYSICS wrist 0"), label(data_w, "DATASET wrist 0")), axis=1))
            dw.append_data(label(data_h, "DATASET reference frame 0"))
            for frame in range(frame_count - 1):
                raw = targets[frame]
                bounded, event = clip_hand_state(raw, limits)
                record_clipping(clipping, event)
                stats = physics.step_servo(
                    model=model, data=data, target=bounded,
                    substeps=physics.NATIVE_SUBSTEPS, object_name=object_name,
                )
                for key, field in (("hand_object", "hand_object_contact"),
                                   ("object_floor", "object_floor_contact"),
                                   ("hand_floor", "hand_floor_contact")):
                    contacts[key] += int(stats[field])
                max_ncon = max(max_ncon, int(stats["max_ncon"]))
                max_force = max(max_force, float(stats["max_contact_force"]))
                max_actuator = max(max_actuator, float(stats["max_abs_actuator_force"]))
                max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel))))
                raw_targets.append(raw.astype(np.float32))
                applied_targets.append(np.asarray(bounded, dtype=np.float32))
                hand_states.append(np.asarray(data.qpos[hand_addrs], dtype=np.float32).copy())
                object_positions.append(np.asarray(data.qpos[object_addr:object_addr + 3], dtype=np.float32).copy())
                object_quaternions.append(np.asarray(data.qpos[object_addr + 3:object_addr + 7], dtype=np.float32).copy())
                idx = frame + 1
                sim_h, sim_w = physics.render_current_state(model, data, renderer)
                data_h = resize(decode_jpeg(row["image"][idx]), args.width, args.height)
                data_w = resize(decode_jpeg(row["wrist_image"][idx]), args.width, args.height)
                hw.append_data(np.concatenate((label(sim_h, f"GT PHYSICS frame {idx}"), label(data_h, f"DATASET frame {idx}")), axis=1))
                ww.append_data(np.concatenate((label(sim_w, f"GT PHYSICS wrist {idx}"), label(data_w, f"DATASET wrist {idx}")), axis=1))
                dw.append_data(label(data_h, f"DATASET reference frame {idx}"))
        sim_pos = np.asarray(object_positions, dtype=np.float32)
        ref_pos = np.asarray(row["objects"][0]["pos"][:frame_count], dtype=np.float32)
        error = np.linalg.norm(sim_pos - ref_pos, axis=1)
        sim_lift = sim_pos[:, 2] - sim_pos[0, 2]
        ref_lift = ref_pos[:, 2] - ref_pos[0, 2]
        arrays = {
            "raw_gt_targets": np.asarray(raw_targets, dtype=np.float32),
            "applied_gt_targets": np.asarray(applied_targets, dtype=np.float32),
            "hand_state_sim": np.asarray(hand_states, dtype=np.float32),
            "object_position_sim": sim_pos,
            "object_quaternion_sim": np.asarray(object_quaternions, dtype=np.float32),
            "object_position_reference": ref_pos,
        }
        array_paths = {}
        for name, value in arrays.items():
            path = out / f"{name}.npy"; np.save(path, value); array_paths[name] = str(path)
        result = {
            "mode": "mode4_gt_absolute_target_physics_replay",
            "policy_inference": False,
            "row_index": row_index,
            "object_name": object_name,
            "target_alignment": "target[f]_held_over_interval_f_to_f_plus_1",
            "frame_count": frame_count,
            "physics": {
                "engine": "MuJoCo", "controller": "manorl_native_position_servo",
                "timestep_seconds": physics.DT, "substeps_per_source_interval": physics.NATIVE_SUBSTEPS,
                "mj_step_calls": physics.NATIVE_SUBSTEPS * (frame_count - 1),
                "contacts": contacts, "max_ncon": max_ncon,
                "max_contact_force": max_force, "max_abs_actuator_force": max_actuator,
                "max_abs_qvel": max_qvel,
            },
            "joint_limit_clipping": clipping,
            "object_metrics": {
                "sim_max_lift_m": float(sim_lift.max()), "sim_final_lift_m": float(sim_lift[-1]),
                "reference_max_lift_m": float(ref_lift.max()),
                "position_error_mean_m": float(error.mean()),
                "position_error_p95_m": float(np.quantile(error, 0.95)),
                "position_error_max_m": float(error.max()),
            },
            "head_video": str(head_path), "wrist_video": str(wrist_path),
            "dataset_replay_video": str(dataset_path), "arrays": array_paths,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if renderer is not None:
            renderer.close()
        if tmp is not None:
            tmp.cleanup()


def main() -> int:
    args = parse_args()
    eval_rows = parse_ordered_unique_csv(args.row_indices, option="--row-indices") if args.row_indices else [args.row_index]
    source = lance.dataset(str(args.lance_dataset))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row_index in eval_rows:
        row_dir = args.output_dir / f"row{row_index}"
        result = run_row(args, load_row(source, row_index), row_index, row_dir)
        (row_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
    summary = {"mode": "mode4_gt_absolute_target_physics_replay_batch", "lance_dataset": str(args.lance_dataset), "row_indices": eval_rows, "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
