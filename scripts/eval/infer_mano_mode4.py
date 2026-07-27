#!/usr/bin/env python3
"""The sole MANO Mode4 evaluator: policy target-DOF control in real MuJoCo physics.

The object is initialized at frame 0 and thereafter owned by MuJoCo. Model B
outputs are reconstructed into absolute target DOFs, temporally ensembled, and
executed by the same 200 Hz native position-servo contract as quality replay.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import time
import uuid

import imageio.v2 as imageio
import lance

# The rollout renders headless observations on the remote server.
# Empty inherited variables are common on the shared host; set the backend
# explicitly before importing mujoco/PyOpenGL.
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM") or "egl"

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import requests

import mano_action_support as chunk_helper
import mode4_data_support as full
import mano_physics_core as physics
from mano_joint_limits import new_clipping_diagnostics, record_clipping
from mode4_support import (
    acquire_action_session,
    action_session_payload,
    parse_ordered_unique_csv,
)
from scripts.gesture_language import DEFAULT_GESTURE_INDEX_PATH, GestureIndex
from scripts.mano_state_contract import (
    CONTACT_RULE,
    CONTACT_SEMANTICS,
    EXPECTED_NORM_SHA256,
    STATE_CONTRACT_ID,
    verify_locked_norm_stats,
)
from scripts.target_actions import URDF_TARGET_ABSOLUTE, project_row_actions
from scripts.train.train_cube1_01_compare import (
    GESTURE_LANGUAGE,
    LANGUAGE_CONDITIONING_CHOICES,
    format_language_prompt,
)


L = chunk_helper.L
OBS = chunk_helper.OBS
HORIZON = 10
HAND_DIM = 26

# Fixed timing exported for audit/tests.
MANORL_PHYSICS_TIMESTEP = physics.DT
MANORL_PHYSICS_SUBSTEPS = physics.NATIVE_SUBSTEPS


def reconstruct_absolute_target_chunk(query_q: np.ndarray, pred_phys: np.ndarray) -> np.ndarray:
    """Invert B semantics once: xyz/fingers are query deltas, Euler is absolute."""
    query = np.asarray(query_q, dtype=np.float32)
    output = np.asarray(pred_phys, dtype=np.float32)
    if query.shape != (HAND_DIM,) or output.ndim != 2 or output.shape[1] < HAND_DIM:
        raise ValueError(
            f"invalid B reconstruction shapes query={query.shape} output={output.shape}"
        )
    target = np.empty((output.shape[0], HAND_DIM), dtype=np.float32)
    target[:, :3] = query[:3] + output[:, :3]
    target[:, 3:6] = output[:, 3:6]
    target[:, 6:26] = query[6:26] + output[:, 6:26]
    if not np.isfinite(target).all():
        raise ValueError("non-finite absolute target chunk")
    return target


def expected_mode4_mj_steps(frame_count: int) -> int:
    if frame_count < 2:
        raise ValueError("Mode4 needs at least two frames")
    return MANORL_PHYSICS_SUBSTEPS * (frame_count - 1)


def condition_row_language(
    row: dict,
    language_conditioning: str,
    *,
    row_index: int,
    gesture_index: GestureIndex | None,
) -> dict:
    """Return an eval row whose prompt exactly matches the training contract."""
    gesture = None
    if language_conditioning == GESTURE_LANGUAGE:
        if gesture_index is None:
            raise ValueError("gesture language requires --gesture-index")
        object_names = row["trajectory_metadata"].get("object_names") or []
        if len(object_names) != 1 or not isinstance(object_names[0], str):
            raise ValueError(f"gesture language requires one object at row {row_index}")
        index = row["index"]
        gesture = gesture_index.record_for(
            row_index,
            uuid=index["uuid"],
            seed_uuid=index["seed_uuid"],
            object_type=object_names[0],
            total_frames=int(row["episode_metadata"]["total_frames"]),
        ).gesture
    return {
        **row,
        "prompt": format_language_prompt(
            row["prompt"],
            row["trajectory_metadata"],
            language_conditioning,
            gesture=gesture,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30530")
    parser.add_argument("--api-key", default="tml-dummy")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", default=L.PI05_ACTION_LORA_R16_MODEL, choices=L.MODEL_CHOICES)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    row_selection = parser.add_mutually_exclusive_group(required=True)
    row_selection.add_argument("--row-index", type=int)
    row_selection.add_argument(
        "--row-indices",
        help="ordered comma-separated row IDs; duplicate IDs are evaluated once",
    )
    parser.set_defaults(action_source=URDF_TARGET_ABSOLUTE)
    parser.add_argument(
        "--extended-state",
        action="store_true",
        help="use MANO extended 32-dim state: finger contacts from live MuJoCo at [26:31], "
        "lift height from sim object at [31]. Must match training contract.",
    )
    parser.add_argument(
        "--language-conditioning",
        choices=LANGUAGE_CONDITIONING_CHOICES,
        default=GESTURE_LANGUAGE,
        help="must match the checkpoint's training prompt contract",
    )
    parser.add_argument(
        "--gesture-index",
        type=Path,
        default=DEFAULT_GESTURE_INDEX_PATH,
        help="canonical generated-MANO index.json used by gesture checkpoints",
    )
    parser.add_argument(
        "--normalization-row-indices",
        required=True,
        help="comma-separated rows or 'all'; provenance population for normalization",
    )
    parser.add_argument(
        "--norm-stats-dir",
        type=Path,
        required=True,
        help="locked training norm_stats directory; required for B target recovery",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--act-mode",
        choices=("batch", "single"),
        default="batch",
        help="batch uses fixed-shape warm-JIT act_batch; single is diagnostic",
    )
    parser.add_argument("--act-batch-size", type=int, default=4)
    parser.add_argument(
        "--max-warm-request-seconds",
        type=float,
        default=2.0,
        help="abort after the second batch request if warm latency exceeds this; 0 disables",
    )
    parser.add_argument("--temporal-decay", type=float, default=0.4)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def decode_jpeg(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def resize_for_video(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return np.asarray(frame, dtype=np.uint8)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    return np.asarray(
        image.resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    width = min(image.width, max(320, 8 + len(text) * 7))
    draw.rectangle((0, 0, width, 31), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return np.asarray(image)


def pad_actions(actions: np.ndarray, frame: int, window_end: int) -> np.ndarray:
    end = min(frame + HORIZON, window_end + 1)
    # DeltaActions mutates its input in place. Own this chunk so repeated or
    # overlapping eval queries cannot corrupt the row's absolute action labels.
    result = np.array(actions[frame:end], dtype=np.float32, copy=True)
    if result.shape[0] < HORIZON:
        result = np.concatenate(
            [result, np.repeat(result[-1:], HORIZON - result.shape[0], axis=0)],
            axis=0,
        )
    return result


def build_datum(
    row: dict,
    *,
    frame: int,
    state_input: np.ndarray,
    head_image: np.ndarray,
    wrist_image: np.ndarray,
    data_config,
    base_model: str,
    window_end: int,
) -> dict:
    raw = {
        "observation/image": head_image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.asarray(state_input, dtype=np.float32),
        "actions": pad_actions(np.asarray(row["actions"]), frame, window_end),
        "prompt": str(row["prompt"]),
    }
    transformed = L._transform_sample(raw, data_config)
    return L._pi05_datum_from_transformed(base_model, transformed)


def create_session(args: argparse.Namespace, headers: dict[str, str]) -> str:
    created = L._post_json(
        args.base_url,
        "/api/v1/mint/action_sessions",
        headers,
        action_session_payload(
            session_id=f"mano-rollout-{uuid.uuid4().hex[:12]}",
            base_model=args.model,
            model_path=args.model_path,
            owner_id=args.owner_id,
        ),
    )
    return str(created["action_session_id"])


def delete_session(args: argparse.Namespace, headers: dict[str, str], session_id: str) -> None:
    try:
        requests.delete(
            f"{args.base_url}/api/v1/mint/action_sessions/{session_id}",
            headers=headers,
            timeout=120.0,
        )
    except Exception:
        pass


def query_action(
    *,
    args: argparse.Namespace,
    headers: dict[str, str],
    session_id: str,
    datum: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if args.act_mode == "batch":
        payloads, timing = L._request_action_batch(
            args.base_url,
            headers,
            session_id,
            [datum["observation"]],
            fixed_batch_size=args.act_batch_size,
        )
        payload = payloads[0]
    else:
        started = time.monotonic()
        result = L._await_result(
            args.base_url,
            headers,
            L._post_json(
                args.base_url,
                f"/api/v1/mint/action_sessions/{session_id}/act",
                headers,
                {"observation": datum["observation"]},
            ),
        )
        timing = {
            "wall_seconds": time.monotonic() - started,
            "actual_observation_count": 1,
            "request_batch_size": 1,
            "padding_count": 0,
            "server_elapsed_ms": None,
            "used_data_sharding": False,
            "response_batch_size": 1,
        }
        payload = result["actions"]
    pred_norm = np.asarray(payload["data"], dtype=np.float32).reshape(payload["shape"])
    pred_phys = np.asarray(
        OBS._unnormalize_actions(pred_norm, datum["data_config"]), dtype=np.float32
    )
    gt_payload = datum["supervision"]["actions"]
    gt_norm = np.asarray(gt_payload["data"], dtype=np.float32).reshape(gt_payload["shape"])
    return pred_norm, pred_phys, gt_norm, timing


def run_variant(
    *,
    args,
    row,
    data_config,
    mode,
    headers,
    object_name,
    manifest_entry=None,
    session_id=None,
    row_index=None,
    output_dir=None,
):
    """The sole Mode4: policy-in-the-loop, target-DOF, real MuJoCo dynamics."""
    if mode != "mode4":
        raise ValueError(f"only mode4 exists, got {mode!r}")
    if args.action_source != URDF_TARGET_ABSOLUTE:
        raise ValueError("Mode4 requires urdf_target_absolute checkpoint semantics")
    selected_row_index = args.row_index if row_index is None else row_index
    source_frames = full.row_frame_count(row)
    timestamps = np.asarray(row["timestamp"], dtype=np.float64)
    if timestamps.shape != (source_frames,) or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"invalid source timestamps: {timestamps.shape}")
    if not np.allclose(np.diff(timestamps), 0.005, rtol=0, atol=1e-10):
        raise ValueError("Mode4 requires exact monotonic 200 Hz source timestamps")
    window = full.resolve_row_window(
        row,
        row_index=selected_row_index,
        frame_window="full",
        contact_context_frames=0,
        missing_contact_policy="error",
        manifest_entry=None,
    )
    if window is None or window.start_frame != 0:
        raise ValueError("Mode4 must initialize once at source frame 0")
    window_end = window.end_frame
    if args.max_frames > 0:
        window_end = min(window_end, args.max_frames - 1)
    frame_count = window_end + 1
    if frame_count < 2:
        raise ValueError("Mode4 needs at least two frames")
    # Canonical MANO metadata says 100 Hz, but the validated quality-replay
    # timeline is authoritative: adjacent source rows are 0.005 s apart.
    # Mode4 therefore executes exactly two 0.0025 s MuJoCo steps per interval.
    substeps = MANORL_PHYSICS_SUBSTEPS

    out = (args.output_dir if output_dir is None else output_dir) / "mode4"
    out.mkdir(parents=True, exist_ok=True)
    tmp, model, data, renderer, object_addr, _, hand_addrs, _, limits = physics.make_scene(
        object_name,
        args.width,
        args.height,
        physics=True,
        physics_timestep=physics.DT,
        create_renderer=True,
    )
    # Pre-resolve keypoint/object geom IDs once at scene init (not per-frame string scan).
    kp_geom_ids, obj_geom_ids, geom_to_finger = (
        physics.resolve_keypoint_geom_ids(model, object_name)
        if args.extended_state else (None, None, None)
    )
    state = np.zeros(32, dtype=np.float32)
    state[:HAND_DIM] = np.asarray(row["state"][0], dtype=np.float32)[:HAND_DIM]
    clipping = new_clipping_diagnostics(limits)
    data.qvel[:] = 0.0
    full.set_scene_state(
        model,
        data,
        state=state[:HAND_DIM],
        object_pos=row["objects"][0]["pos"][0],
        object_rot_aa=row["objects"][0]["rot_aa"][0],
        object_addr=object_addr,
        hand_addrs=hand_addrs,
    )
    import mujoco as _mujoco
    _mujoco.mj_forward(model, data)
    # Record lift baseline from Mode 4's actual initialized sim state.
    object_z_initial = (
        float(data.qpos[object_addr + 2]) if args.extended_state else 0.0
    )

    candidates = []
    query_timings = []
    raw_norm_steps = []
    raw_phys_steps = []
    commanded_steps = []
    servo_targets = []
    applied_steps = []
    observation_states = []
    observation_contacts = []
    observation_lift = []
    preclip_targets = []
    clipping_corrections = []
    physics_contact_flags = []
    step_max_contact_forces = []
    gt_norm_steps = []
    hand_states = [np.asarray(data.qpos[hand_addrs], dtype=np.float32).copy()]
    object_positions = [
        np.asarray(data.qpos[object_addr : object_addr + 3], dtype=np.float32).copy()
    ]
    object_quaternions = [
        np.asarray(data.qpos[object_addr + 3 : object_addr + 7], dtype=np.float32).copy()
    ]
    contacts = {"hand_object": 0, "object_floor": 0, "hand_floor": 0}
    max_ncon = 0
    max_contact_force = 0.0
    max_actuator_force = 0.0
    max_qvel = 0.0
    head_path = out / "mode4_physics_vs_dataset_head.mp4"
    wrist_path = out / "mode4_physics_vs_dataset_wrist.mp4"
    dataset_path = out / "dataset_reference.mp4"
    active_session_id = session_id or ""
    owns_session = False
    headers = dict(headers)
    try:
        active_session_id, owns_session = acquire_action_session(
            session_id, lambda: create_session(args, headers)
        )
        with imageio.get_writer(
            str(head_path), fps=args.fps, macro_block_size=1, codec="libx264"
        ) as head_writer, imageio.get_writer(
            str(wrist_path), fps=args.fps, macro_block_size=1, codec="libx264"
        ) as wrist_writer, imageio.get_writer(
            str(dataset_path), fps=args.fps, macro_block_size=1, codec="libx264"
        ) as dataset_writer:
            sim_head, sim_wrist = physics.render_current_state(model, data, renderer)
            dataset_head = resize_for_video(decode_jpeg(row["image"][0]), args.width, args.height)
            dataset_wrist = resize_for_video(
                decode_jpeg(row["wrist_image"][0]), args.width, args.height
            )
            head_writer.append_data(
                np.concatenate(
                    [
                        label(sim_head, "MODE4 PHYSICS frame 0"),
                        label(dataset_head, "DATASET frame 0"),
                    ],
                    axis=1,
                )
            )
            wrist_writer.append_data(
                np.concatenate(
                    [
                        label(sim_wrist, "MODE4 PHYSICS wrist 0"),
                        label(dataset_wrist, "DATASET wrist 0"),
                    ],
                    axis=1,
                )
            )
            dataset_writer.append_data(label(dataset_head, "DATASET reference frame 0"))

            for frame in range(window_end):
                current_q = np.asarray(data.qpos[hand_addrs], dtype=np.float64).copy()
                state[:HAND_DIM] = current_q.astype(np.float32)
                if args.extended_state:
                    state[HAND_DIM:] = 0
                    fc = physics.finger_contacts_from_mujoco(
                        model, data, object_name,
                        keypoint_geom_ids=kp_geom_ids,
                        object_geom_ids=obj_geom_ids,
                        geom_id_to_finger=geom_to_finger,
                    )
                    state[26:31] = fc
                    state[31] = np.float32(
                        float(data.qpos[object_addr + 2]) - object_z_initial
                    )
                else:
                    state[HAND_DIM:] = 0
                # Record the policy observation before selecting this transition.
                observation_states.append(state.copy())
                observation_contacts.append(state[26:31].copy())
                observation_lift.append(state[31])
                if frame % args.chunk_stride == 0:
                    head_image, wrist_image = physics.render_current_state(model, data, renderer)
                    datum = build_datum(
                        row,
                        frame=frame,
                        state_input=state.copy(),
                        head_image=head_image,
                        wrist_image=wrist_image,
                        data_config=data_config,
                        base_model=args.model,
                        window_end=window_end,
                    )
                    datum["data_config"] = data_config
                    pred_norm, pred_phys, gt_norm, timing = query_action(
                        args=args,
                        headers=headers,
                        session_id=active_session_id,
                        datum=datum,
                    )
                    query_index = len(query_timings)
                    timing = {"query_index": query_index, "source_frame": frame, **timing}
                    query_timings.append(timing)
                    if args.act_mode == "batch" and timing.get("used_data_sharding") is not True:
                        raise RuntimeError("act_batch did not use data sharding")
                    if (
                        args.act_mode == "batch"
                        and query_index == 1
                        and args.max_warm_request_seconds > 0
                        and float(timing["wall_seconds"]) > args.max_warm_request_seconds
                    ):
                        raise RuntimeError(
                            f"warm latency {timing['wall_seconds']:.3f}s exceeds limit"
                        )
                    rollout_phys = np.asarray(pred_phys, dtype=np.float32).copy()
                    query_q = state[:HAND_DIM].copy()
                    target_hand = reconstruct_absolute_target_chunk(query_q, rollout_phys[:HORIZON])
                    candidates.append(
                        {
                            "start": frame,
                            "pred_norm": pred_norm,
                            "pred_phys": pred_phys,
                            "gt_norm": gt_norm,
                            "target_hand": target_hand,
                        }
                    )
                    candidates = [c for c in candidates if frame < c["start"] + HORIZON]
                active = [c for c in candidates if c["start"] <= frame < c["start"] + HORIZON]
                if not active:
                    raise RuntimeError(f"no action candidate at frame {frame}")
                newest_start = max(c["start"] for c in active)
                newest = max(active, key=lambda c: c["start"])
                local_newest = frame - newest["start"]
                weights = np.asarray(
                    [
                        args.temporal_decay ** ((newest_start - c["start"]) // args.chunk_stride)
                        for c in active
                    ],
                    dtype=np.float64,
                )
                weights /= weights.sum()
                absolute_target = sum(
                    w * c["target_hand"][frame - c["start"]]
                    for w, c in zip(weights, active, strict=True)
                )
                target, clip_event = physics.nearest_wrapped_position_target(
                    current_q, absolute_target - current_q, limits
                )
                record_clipping(clipping, clip_event)
                preclip_target = np.asarray(absolute_target, dtype=np.float32)
                servo_target = np.asarray(target, dtype=np.float32)
                before = current_q.copy()
                diagnostics = physics.step_servo(
                    model=model,
                    data=data,
                    target=target,
                    substeps=substeps,
                    object_name=object_name,
                )
                after = np.asarray(data.qpos[hand_addrs], dtype=np.float64).copy()
                for key in contacts:
                    contacts[key] += int(diagnostics[f"{key}_contact"])
                max_ncon = max(max_ncon, int(diagnostics["max_ncon"]))
                max_contact_force = max(max_contact_force, float(diagnostics["max_contact_force"]))
                max_actuator_force = max(
                    max_actuator_force, float(diagnostics["max_abs_actuator_force"])
                )
                max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel))))
                physics_contact_flags.append(
                    [
                        diagnostics["hand_object_contact"],
                        diagnostics["object_floor_contact"],
                        diagnostics["hand_floor_contact"],
                    ]
                )
                step_max_contact_forces.append(diagnostics["max_contact_force"])
                raw_norm_steps.append(newest["pred_norm"][local_newest])
                raw_phys_steps.append(newest["pred_phys"][local_newest])
                commanded = np.zeros(32, dtype=np.float32)
                commanded[:HAND_DIM] = absolute_target - current_q
                commanded_steps.append(commanded)
                preclip_targets.append(preclip_target)
                servo_targets.append(servo_target)
                clipping_corrections.append(servo_target - preclip_target)
                applied = np.zeros(32, dtype=np.float32)
                applied[:HAND_DIM] = (after - before).astype(np.float32)
                applied_steps.append(applied)
                gt_norm_steps.append(newest["gt_norm"][local_newest])
                hand_states.append(after.astype(np.float32))
                object_positions.append(
                    np.asarray(data.qpos[object_addr : object_addr + 3], dtype=np.float32).copy()
                )
                object_quaternions.append(
                    np.asarray(
                        data.qpos[object_addr + 3 : object_addr + 7], dtype=np.float32
                    ).copy()
                )
                next_frame = frame + 1
                sim_head, sim_wrist = physics.render_current_state(model, data, renderer)
                dataset_head = resize_for_video(
                    decode_jpeg(row["image"][next_frame]), args.width, args.height
                )
                dataset_wrist = resize_for_video(
                    decode_jpeg(row["wrist_image"][next_frame]), args.width, args.height
                )
                head_writer.append_data(
                    np.concatenate(
                        [
                            label(sim_head, f"MODE4 PHYSICS frame {next_frame}"),
                            label(dataset_head, f"DATASET frame {next_frame}"),
                        ],
                        axis=1,
                    )
                )
                wrist_writer.append_data(
                    np.concatenate(
                        [
                            label(sim_wrist, f"MODE4 PHYSICS wrist {next_frame}"),
                            label(dataset_wrist, f"DATASET wrist {next_frame}"),
                        ],
                        axis=1,
                    )
                )
                dataset_writer.append_data(
                    label(dataset_head, f"DATASET reference frame {next_frame}")
                )
    finally:
        if owns_session and active_session_id:
            delete_session(args, headers, active_session_id)
        renderer.close()
        tmp.cleanup()

    expected_steps = expected_mode4_mj_steps(frame_count)
    if not np.isclose(data.time, expected_steps * physics.DT, rtol=0, atol=1e-9):
        raise RuntimeError(f"Mode4 time mismatch {data.time}")
    raw_norm_all = np.asarray(raw_norm_steps, dtype=np.float32)
    raw_phys_all = np.asarray(raw_phys_steps, dtype=np.float32)
    applied_all = np.asarray(applied_steps, dtype=np.float32)
    gt_norm_all = np.asarray(gt_norm_steps, dtype=np.float32)
    arrays = {
        "actions_raw_pred_normalized": out / "actions_raw_pred_normalized.npy",
        "actions_raw_pred_physical": out / "actions_raw_pred_physical.npy",
        "actions_commanded_physical": out / "actions_commanded_physical.npy",
        "preclip_absolute_targets": out / "preclip_absolute_targets.npy",
        "servo_position_targets": out / "servo_position_targets.npy",
        "servo_target_clipping_correction": out / "servo_target_clipping_correction.npy",
        "actions_applied_physical": out / "actions_applied_physical.npy",
        "rollout_observation_state": out / "rollout_observation_state.npy",
        "rollout_observation_contacts": out / "rollout_observation_contacts.npy",
        "rollout_observation_lift": out / "rollout_observation_lift.npy",
        "physics_contact_flags": out / "physics_contact_flags.npy",
        "step_max_contact_force": out / "step_max_contact_force.npy",
        "hand_state_sim": out / "hand_state_sim.npy",
        "object_position_sim": out / "object_position_sim.npy",
        "object_quaternion_sim": out / "object_quaternion_sim.npy",
    }
    np.save(arrays["actions_raw_pred_normalized"], raw_norm_all)
    np.save(arrays["actions_raw_pred_physical"], raw_phys_all)
    np.save(arrays["actions_commanded_physical"], np.asarray(commanded_steps, dtype=np.float32))
    np.save(arrays["preclip_absolute_targets"], np.asarray(preclip_targets, dtype=np.float32))
    np.save(arrays["servo_position_targets"], np.asarray(servo_targets, dtype=np.float32))
    np.save(
        arrays["servo_target_clipping_correction"],
        np.asarray(clipping_corrections, dtype=np.float32),
    )
    np.save(arrays["actions_applied_physical"], applied_all)
    np.save(arrays["rollout_observation_state"], np.asarray(observation_states, dtype=np.float32))
    np.save(
        arrays["rollout_observation_contacts"],
        np.asarray(observation_contacts, dtype=np.float32),
    )
    np.save(arrays["rollout_observation_lift"], np.asarray(observation_lift, dtype=np.float32))
    np.save(arrays["physics_contact_flags"], np.asarray(physics_contact_flags, dtype=np.bool_))
    np.save(
        arrays["step_max_contact_force"],
        np.asarray(step_max_contact_forces, dtype=np.float32),
    )
    np.save(arrays["hand_state_sim"], np.asarray(hand_states, dtype=np.float32))
    np.save(arrays["object_position_sim"], np.asarray(object_positions, dtype=np.float32))
    np.save(arrays["object_quaternion_sim"], np.asarray(object_quaternions, dtype=np.float32))
    total_request_seconds = float(sum(float(t["wall_seconds"]) for t in query_timings))
    result = {
        "mode": "mode4",
        "physics_dynamics": True,
        "closed_loop": True,
        "observation_feedback": True,
        "state_observation_source": "integrated_mujoco_qpos",
        "image_observation_source": "integrated_mujoco_renderer",
        "object_pose_source": "sim_owned_after_frame0",
        "action_source": args.action_source,
        "extended_state": bool(args.extended_state),
        "state_contract": (
            STATE_CONTRACT_ID if args.extended_state else None
        ),
        "contact_semantics": (
            CONTACT_SEMANTICS if args.extended_state else None
        ),
        "contact_rule": CONTACT_RULE if args.extended_state else None,
        "norm_sha_expected": getattr(args, "norm_sha_expected", None),
        "norm_sha_actual": getattr(args, "norm_sha_actual", None),
        "language_conditioning": args.language_conditioning,
        "prompt": str(row["prompt"]),
        "rollout_dynamics": "B_query_anchored_absolute_target_to_native_position_servo",
        "temporal_ensemble": True,
        "temporal_decay": args.temporal_decay,
        "chunk_horizon": HORIZON,
        "query_stride": args.chunk_stride,
        "row_index": selected_row_index,
        "object_name": object_name,
        "source_frame_count": source_frames,
        "trajectory_frame_count": frame_count,
        "frame_window": {"start_frame": 0, "end_frame": window_end, "frame_count": frame_count},
        "physics": {
            "engine": "MuJoCo",
            "controller": "manorl_native_position_servo",
            "timestep_seconds": physics.DT,
            "steps_per_source_interval": substeps,
            "control_hz": 200,
            "source_interval_seconds": 0.005,
            "declared_dataset_fps": row.get("trajectory_metadata", {}).get("data_fps"),
            "intervals": frame_count - 1,
            "mj_step_calls": expected_steps,
            "simulated_seconds": float(data.time),
            "contacts": contacts,
            "max_ncon": max_ncon,
            "max_contact_force": max_contact_force,
            "max_abs_actuator_force": max_actuator_force,
            "max_abs_qvel": max_qvel,
        },
        "joint_limit_clipping": clipping,
        "query_count": len(query_timings),
        "total_request_seconds": total_request_seconds,
        "mean_request_seconds": total_request_seconds / max(len(query_timings), 1),
        "query_timings": query_timings,
        "pred_has_nan_inf": bool(not np.isfinite(raw_norm_all).all()),
        "head_video": str(head_path),
        "wrist_video": str(wrist_path),
        "dataset_replay_video": str(dataset_path),
        "arrays": {name: str(path) for name, path in arrays.items()},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_row_summary(
    *,
    args: argparse.Namespace,
    row_index: int,
    normalization_rows: list[int],
    results: list[dict],
) -> dict:
    return {
        "mode": "mode4_policy_target_dof_mujoco_physics",
        "model_path": args.model_path,
        "model": args.model,
        "action_source": args.action_source,
        "language_conditioning": args.language_conditioning,
        "gesture_index": (
            str(args._gesture_index.path) if args._gesture_index is not None else None
        ),
        "gesture_index_sha256": (
            args._gesture_index.sha256 if args._gesture_index is not None else None
        ),
        "lance_dataset": str(args.lance_dataset),
        "row_index": row_index,
        "normalization_row_indices": normalization_rows,
        "norm_stats_dir": str(args.norm_stats_dir) if args.norm_stats_dir else None,
        "state_contract": (
            STATE_CONTRACT_ID if args.extended_state else None
        ),
        "contact_semantics": (
            CONTACT_SEMANTICS if args.extended_state else None
        ),
        "contact_rule": (
            CONTACT_RULE if args.extended_state else None
        ),
        "norm_sha_expected": getattr(args, "norm_sha_expected", None),
        "norm_sha_actual": getattr(args, "norm_sha_actual", None),
        "frame_window": "full_from_frame0",
        "object_pose_source": "sim_owned_after_frame0",
        "act_mode": args.act_mode,
        "act_batch_size": args.act_batch_size if args.act_mode == "batch" else 1,
        "max_warm_request_seconds": args.max_warm_request_seconds,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    args._gesture_index = (
        GestureIndex.load(args.gesture_index)
        if args.language_conditioning == GESTURE_LANGUAGE
        else None
    )
    if not 1 <= args.chunk_stride < HORIZON:
        raise ValueError(f"--chunk-stride must be between 1 and {HORIZON - 1}")
    if not 0 < args.temporal_decay <= 1:
        raise ValueError("--temporal-decay must be in (0, 1]")
    if args.act_batch_size <= 0:
        raise ValueError("--act-batch-size must be positive")

    multi_row = args.row_indices is not None
    eval_rows = (
        parse_ordered_unique_csv(args.row_indices, option="--row-indices")
        if multi_row
        else [args.row_index]
    )
    if any(index is None for index in eval_rows):
        raise ValueError("row selection must not be empty")
    eval_rows = [int(index) for index in eval_rows]

    source = lance.dataset(str(args.lance_dataset))
    row_count = source.count_rows()
    target_source = source
    args.rollout_action_gains = np.ones(HAND_DIM, dtype=np.float32)
    if any(not 0 <= index < row_count for index in eval_rows):
        raise IndexError(f"row index out of range: {eval_rows}")
    normalization_rows = (
        list(range(row_count))
        if args.normalization_row_indices.strip().lower() == "all"
        else [int(value) for value in args.normalization_row_indices.split(",") if value.strip()]
    )
    if not normalization_rows:
        raise ValueError("--normalization-row-indices must not be empty")
    if any(not 0 <= index < row_count for index in normalization_rows):
        raise IndexError(f"normalization row index out of range: {normalization_rows}")

    metadata_columns = [
        "state",
        "actions",
        "prompt",
        "objects",
        "timestamp",
        "trajectory_metadata",
        "episode_metadata",
    ]
    if args.extended_state:
        # Authenticate the exact v1 normalization bytes before loading them.
        _, actual_sha = verify_locked_norm_stats(args.norm_stats_dir)
        args.norm_sha_expected = EXPECTED_NORM_SHA256
        args.norm_sha_actual = actual_sha
    norm_stats = L.normalize.load(args.norm_stats_dir)
    if args.extended_state:
        # Structural validation follows the exact-byte check.
        _sq01 = np.asarray(norm_stats["state"].q01, dtype=np.float32)
        _sq99 = np.asarray(norm_stats["state"].q99, dtype=np.float32)
        if not np.allclose(_sq01[26:31], 0.0):
            raise ValueError(
                f"extended-state norm cache q01[26:31] must be 0, got {_sq01[26:31]}"
            )
        if not np.allclose(_sq99[26:31], 1.0):
            raise ValueError(
                f"extended-state norm cache q99[26:31] must be 1, got {_sq99[26:31]}"
            )
        _lift_range = _sq99[31] - _sq01[31]
        if _lift_range < 1e-4:
            raise ValueError(
                f"extended-state norm cache lift range must be > 1e-4, got {_lift_range}"
            )
    data_config = L._make_data_config(
        full.build_model_config(args.model), norm_stats, action_source=args.action_source
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    headers = L._headers(args.api_key)

    def load_eval_row(row_index: int) -> dict:
        row = source.take(
            [row_index], columns=[*metadata_columns, "image", "wrist_image", "index"]
        ).to_pylist()[0]
        if target_source is not None:
            target_row = target_source.take([row_index], columns=["hands", "index"]).to_pylist()[0]
            if row["index"]["uuid"] != target_row["index"]["uuid"]:
                raise ValueError(f"target/image row UUID mismatch at {row_index}")
            target_q = np.asarray(target_row["hands"][0]["urdf_dof"], dtype=np.float32)
            image_q = np.asarray(row["state"], dtype=np.float32)[:, :26]
            if target_q.shape != image_q.shape or not np.array_equal(target_q, image_q):
                raise ValueError(f"target/image q mismatch at row {row_index}")
            row = {**row, "hands": target_row["hands"]}
        projected = project_row_actions(row, args.action_source)
        return condition_row_language(
            projected,
            args.language_conditioning,
            row_index=row_index,
            gesture_index=args._gesture_index,
        )

    if not multi_row:
        row_index = eval_rows[0]
        row = load_eval_row(row_index)
        object_name = full.safe_object_name(row)
        results = [
            run_variant(
                args=args,
                row=row,
                data_config=data_config,
                mode="mode4",
                headers=headers,
                object_name=object_name,
            )
        ]
        summary = build_row_summary(
            args=args,
            row_index=row_index,
            normalization_rows=normalization_rows,
            results=results,
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    shared_session_id = create_session(args, headers)
    row_summaries: list[dict] = []
    try:
        for row_index in eval_rows:
            row_output_dir = args.output_dir / f"row{row_index}"
            row_output_dir.mkdir(parents=True, exist_ok=True)
            row = load_eval_row(row_index)
            object_name = full.safe_object_name(row)
            results = [
                run_variant(
                    args=args,
                    row=row,
                    data_config=data_config,
                    mode="mode4",
                    headers=headers,
                    object_name=object_name,
                    session_id=shared_session_id,
                    row_index=row_index,
                    output_dir=row_output_dir,
                )
            ]
            row_summary = build_row_summary(
                args=args,
                row_index=row_index,
                normalization_rows=normalization_rows,
                results=results,
            )
            (row_output_dir / "summary.json").write_text(
                json.dumps(row_summary, indent=2), encoding="utf-8"
            )
            row_summaries.append(row_summary)
    finally:
        delete_session(args, headers, shared_session_id)

    summary = {
        "mode": "mode4_policy_target_dof_mujoco_physics_multi_row",
        "model_path": args.model_path,
        "row_indices": eval_rows,
        "action_source": args.action_source,
        "language_conditioning": args.language_conditioning,
        "gesture_index": (
            str(args._gesture_index.path) if args._gesture_index is not None else None
        ),
        "gesture_index_sha256": (
            args._gesture_index.sha256 if args._gesture_index is not None else None
        ),
        "lance_dataset": str(args.lance_dataset),
        "shared_session": True,
        "normalization_row_indices": normalization_rows,
        "norm_stats_dir": str(args.norm_stats_dir) if args.norm_stats_dir else None,
        "state_contract": (
            STATE_CONTRACT_ID if args.extended_state else None
        ),
        "contact_semantics": (
            CONTACT_SEMANTICS if args.extended_state else None
        ),
        "contact_rule": (
            CONTACT_RULE if args.extended_state else None
        ),
        "norm_sha_expected": getattr(args, "norm_sha_expected", None),
        "norm_sha_actual": getattr(args, "norm_sha_actual", None),
        "frame_window": "full_from_frame0",
        "object_pose_source": "sim_owned_after_frame0",
        "act_mode": args.act_mode,
        "act_batch_size": args.act_batch_size if args.act_mode == "batch" else 1,
        "max_warm_request_seconds": args.max_warm_request_seconds,
        "rows": row_summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
