#!/usr/bin/env python3
"""Historical kinematic MANO Mode3 diagnostic for B-exact 32D v1 checkpoints.

Mode3 is deliberately separate from Mode4.  It renders the current predicted
hand against the *reference* object pose, forwards MuJoCo's kinematic caches,
and never advances physics.  Each policy query supplies exactly one disjoint
10-frame action chunk; no temporal ensemble is involved.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import uuid

import imageio.v2 as imageio
import lance

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM") or "egl"

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import requests

import mano_action_support as action_support
import mano_physics_core as physics
from mano_joint_limits import clip_hand_state, new_clipping_diagnostics, record_clipping
from mano_servo_lag import (
    CONTRACT_ID as SERVO_LAG_CONTRACT_ID,
    load_gain_file,
    servo_lag_step,
    wrap_euler_target_near_current,
)
from mode4_support import acquire_action_session, action_session_payload, parse_ordered_unique_csv
from scripts import contact_windows
from scripts.eval.result_paths import default_inference_output_dir
from scripts.gesture_language import DEFAULT_GESTURE_INDEX_PATH, GestureIndex
from scripts.mano_state_contract import (
    CONTACT_RULE, CONTACT_SEMANTICS, EXPECTED_NORM_SHA256, STATE_CONTRACT_ID,
    verify_locked_norm_stats,
)
from scripts.target_actions import URDF_TARGET_ABSOLUTE, project_row_actions
from scripts.train.train_cube1_01_compare import (
    GESTURE_LANGUAGE, LANGUAGE_CONDITIONING_CHOICES, format_language_prompt,
)

L = action_support.L
OBS = action_support.OBS
HORIZON = 10
HAND_DIM = 26
ACTION_DIM = 32


def reconstruct_absolute_target_chunk(query_q: np.ndarray, pred_phys: np.ndarray) -> np.ndarray:
    """Invert B output semantics using the qpos presented at this query."""
    query = np.asarray(query_q, dtype=np.float32)
    output = np.asarray(pred_phys, dtype=np.float32)
    if query.shape != (HAND_DIM,) or output.ndim != 2 or output.shape[1] < HAND_DIM:
        raise ValueError(f"invalid B reconstruction shapes query={query.shape} output={output.shape}")
    target = np.empty((output.shape[0], HAND_DIM), dtype=np.float32)
    target[:, :3] = query[:3] + output[:, :3]
    target[:, 3:6] = output[:, 3:6]
    target[:, 6:26] = query[6:26] + output[:, 6:26]
    if not np.isfinite(target).all():
        raise ValueError("non-finite absolute target chunk")
    return target


def mode3_query_frames(start: int, end: int, query_stride: int = HORIZON) -> list[int]:
    """Return query frames for historical chunk-10 or receding-horizon replan-1."""
    if query_stride not in (1, HORIZON):
        raise ValueError(f"Mode3 query stride must be 1 or {HORIZON}, got {query_stride}")
    if end < start:
        return []
    return list(range(start, end + 1, query_stride))


def mode3_row_frame_count(row: dict) -> int:
    """Reject misaligned source trajectories before a kinematic rollout."""
    objects = row.get("objects") or []
    if len(objects) != 1:
        raise ValueError(f"Mode3 requires exactly one object trajectory, got {len(objects)}")
    lengths = [len(row[key]) for key in ("state", "actions", "image", "wrist_image")]
    lengths.extend((len(objects[0]["pos"]), len(objects[0]["rot_aa"])))
    declared = int((row.get("episode_metadata") or {}).get("total_frames") or 0)
    if declared:
        lengths.append(declared)
    if not lengths or min(lengths) <= 0 or len(set(lengths)) != 1:
        raise ValueError(f"inconsistent Mode3 frame counts: {lengths}")
    return lengths[0]


def canonicalize_physical_padding(pred_phys: np.ndarray) -> np.ndarray:
    """Restore B's exact physical-zero padding after epsilon-producing unnormalization."""
    values = np.asarray(pred_phys, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"expected physical action shape [N,{ACTION_DIM}], got {values.shape}")
    values[:, HAND_DIM:] = 0.0
    return values


def assert_physical_padding_zero(pred_phys: np.ndarray) -> None:
    """B's final six model dimensions are padding, never a kinematic command."""
    values = np.asarray(pred_phys, dtype=np.float32)
    if values.shape != (HORIZON, ACTION_DIM):
        raise ValueError(f"expected physical action shape ({HORIZON},{ACTION_DIM}), got {values.shape}")
    if not np.array_equal(values[:, HAND_DIM:], np.zeros((HORIZON, ACTION_DIM - HAND_DIM), dtype=np.float32)):
        raise ValueError("B checkpoint violated physical-zero action[26:32] contract")


def build_extended_sim_state(*, hand_qpos: np.ndarray, model, data, object_name: str,
                             keypoint_geom_ids: set[int], object_geom_ids: set[int],
                             geom_id_to_finger: dict[int, str], reference_object_z: float,
                             source_object_z0: float) -> np.ndarray:
    """Build v1 Mode3 state after forward at the current reference object pose."""
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    state[:HAND_DIM] = np.asarray(hand_qpos, dtype=np.float32)[:HAND_DIM]
    state[26:31] = physics.finger_contacts_from_mujoco(
        model, data, object_name, keypoint_geom_ids=keypoint_geom_ids,
        object_geom_ids=object_geom_ids, geom_id_to_finger=geom_id_to_finger,
    )
    state[31] = np.float32(reference_object_z - source_object_z0)
    return state


def condition_row_language(row: dict, language_conditioning: str, *, row_index: int,
                           gesture_index: GestureIndex | None) -> dict:
    gesture = None
    if language_conditioning == GESTURE_LANGUAGE:
        if gesture_index is None:
            raise ValueError("gesture language requires --gesture-index")
        names = row["trajectory_metadata"].get("object_names") or []
        if len(names) != 1 or not isinstance(names[0], str):
            raise ValueError(f"gesture language requires one object at row {row_index}")
        index = row["index"]
        gesture = gesture_index.record_for(
            row_index, uuid=index["uuid"], seed_uuid=index["seed_uuid"],
            object_type=names[0], total_frames=int(row["episode_metadata"]["total_frames"]),
        ).gesture
    return {**row, "prompt": format_language_prompt(
        row["prompt"], row["trajectory_metadata"], language_conditioning, gesture=gesture)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:30530")
    p.add_argument("--api-key", default="tml-dummy")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model", default=L.PI05_ACTION_LORA_R16_MODEL, choices=L.MODEL_CHOICES)
    p.add_argument("--owner-id", required=True)
    p.add_argument("--lance-dataset", type=Path, required=True)
    rows = p.add_mutually_exclusive_group(required=True)
    rows.add_argument("--row-index", type=int)
    rows.add_argument("--row-indices", help="ordered comma-separated row IDs; duplicates run once")
    p.add_argument("--extended-state", action="store_true", required=True,
                   help="required B-exact v1 32D contact/lift observation")
    p.add_argument("--norm-stats-dir", type=Path, required=True,
                   help="locked v1 norm directory; SHA is checked before loading")
    p.add_argument("--normalization-row-indices", required=True)
    p.add_argument("--language-conditioning", choices=LANGUAGE_CONDITIONING_CHOICES,
                   default=GESTURE_LANGUAGE)
    p.add_argument("--gesture-index", type=Path, default=DEFAULT_GESTURE_INDEX_PATH)
    p.add_argument("--contact-window-manifest", default="")
    p.add_argument("--contact-context-frames", type=int,
                   default=contact_windows.DEFAULT_CONTACT_CONTEXT_FRAMES)
    p.add_argument("--missing-contact-policy", choices=("full", "skip", "error"), default="full")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--act-batch-size", type=int, default=4,
                   help="must remain 4 for the deployed B checkpoint serving contract")
    p.add_argument(
        "--query-stride", type=int, choices=(1, HORIZON), default=HORIZON,
        help="10 preserves historical Mode3; 1 replans every frame and executes only action[0]",
    )
    p.add_argument(
        "--hand-transition", choices=("instant_setpoint", "calibrated_servo_lag"),
        default="instant_setpoint",
        help="historical direct target write or calibrated one-source-step setpoint response",
    )
    p.add_argument("--servo-gain-file", type=Path,
                   help="required by calibrated_servo_lag; fitted 0.005-second per-DoF gains")
    p.add_argument("--max-warm-request-seconds", type=float, default=2.0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_inference_output_dir("mode3"),
        help="result root (default: client-local results/inference/mode3_<UTC>_<pid>)",
    )
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--client-commit", default=os.environ.get("VLA_CLIENT_GIT_COMMIT"),
                   help="optional client source SHA (launcher supplies this when available)")
    p.add_argument("--backend-commit", default=None, help="optional paired MINT backend source SHA")
    p.add_argument("--model-commit", default=None, help="optional paired OpenPI model source SHA")
    return p.parse_args()


def decode_jpeg(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def resize_for_video(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return np.asarray(frame, dtype=np.uint8)
    return np.asarray(Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, max(320, 8 + 7 * len(text))), 31), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return np.asarray(image)


def side_by_side_video_frame(sim: np.ndarray, dataset: np.ndarray, *, sim_label: str,
                              dataset_label: str) -> np.ndarray:
    """Preserve historical Mode3 SIM-versus-dataset video presentation."""
    return np.concatenate((label(sim, sim_label), label(dataset, dataset_label)), axis=1)


def pad_actions(actions: np.ndarray, frame: int, window_end: int) -> np.ndarray:
    result = np.asarray(actions[frame:min(frame + HORIZON, window_end + 1)], dtype=np.float32)
    if result.shape[0] == 0:
        raise ValueError("cannot pad empty action chunk")
    if result.shape[0] < HORIZON:
        result = np.concatenate((result, np.repeat(result[-1:], HORIZON - result.shape[0], axis=0)))
    return result


def build_datum(row: dict, *, frame: int, state_input: np.ndarray, head_image: np.ndarray,
                wrist_image: np.ndarray, data_config, base_model: str, window_end: int) -> dict:
    raw = {"observation/image": head_image, "observation/wrist_image": wrist_image,
           "observation/state": np.asarray(state_input, dtype=np.float32),
           "actions": pad_actions(np.asarray(row["actions"]), frame, window_end),
           "prompt": str(row["prompt"])}
    return L._pi05_datum_from_transformed(base_model, L._transform_sample(raw, data_config))


def create_session(args: argparse.Namespace, headers: dict[str, str]) -> str:
    created = L._post_json(args.base_url, "/api/v1/mint/action_sessions", headers,
        action_session_payload(session_id=f"mano-mode3-{uuid.uuid4().hex[:12]}",
            base_model=args.model, model_path=args.model_path, owner_id=args.owner_id))
    return str(created["action_session_id"])


def delete_session(args: argparse.Namespace, headers: dict[str, str], session_id: str) -> None:
    try:
        requests.delete(f"{args.base_url}/api/v1/mint/action_sessions/{session_id}", headers=headers, timeout=120)
    except Exception:
        pass


def query_action(*, args, headers, session_id, datum) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    payloads, timing = L._request_action_batch(args.base_url, headers, session_id,
        [datum["observation"]], fixed_batch_size=args.act_batch_size)
    payload = payloads[0]
    pred_norm = np.asarray(payload["data"], dtype=np.float32).reshape(payload["shape"])
    pred_phys = canonicalize_physical_padding(
        np.asarray(OBS._unnormalize_actions(pred_norm, datum["data_config"]), dtype=np.float32)
    )
    gt = datum["supervision"]["actions"]
    gt_norm = np.asarray(gt["data"], dtype=np.float32).reshape(gt["shape"])
    return pred_norm, pred_phys, gt_norm, timing


def render_reference_state(*, model, data, renderer, object_addr: int, hand_addrs: list[int],
                           hand_state: np.ndarray, object_pos: np.ndarray, object_rot: np.ndarray):
    """Set kinematic state and refresh contacts/cameras.  No integration occurs."""
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[object_addr:object_addr + 3] = np.asarray(object_pos, dtype=np.float64)
    data.qpos[object_addr + 3:object_addr + 7] = action_support.axis_angle_to_wxyz(object_rot)
    data.qpos[hand_addrs] = np.asarray(hand_state, dtype=np.float64)
    mujoco.mj_forward(model, data)
    head, wrist = physics.render_current_state(model, data, renderer)
    return head, wrist


def _result_metadata(args, *, row_index: int, object_name: str, window, source_frames: int,
                     query_timings: list[dict], out: Path, head_path: Path, wrist_path: Path,
                     row: dict | None = None) -> dict:
    row = row or {}
    index = row.get("index") or {}
    total_request_seconds = float(
        sum(float(timing.get("wall_seconds", 0.0)) for timing in query_timings)
    )
    return {
        "mode": (
            "kinematic_mode3_servo_lag_replan1"
            if args.hand_transition == "calibrated_servo_lag"
            else (
                "historical_kinematic_mode3_sim_no_smooth"
                if args.query_stride == HORIZON
                else "kinematic_mode3_replan1_first_action"
            )
        ),
        "physics_dynamics": False,
        "mujoco_update": "mj_forward_only; mj_step_never_called",
        "closed_loop": True,
        "observation_feedback": True,
        "state_observation_source": "sim",
        "image_observation_source": "sim",
        "video_presentation": "side_by_side_sim_vs_dataset_same_source_frame",
        "object_pose_source": "reference_trajectory",
        "action_source": URDF_TARGET_ABSOLUTE,
        "rollout_dynamics": "B_query_anchored_absolute_target_kinematic",
        "temporal_ensemble": False,
        "chunk_horizon": HORIZON,
        "query_stride": args.query_stride,
        "action_execution": (
            "consume_full_nonoverlap_chunk"
            if args.query_stride == HORIZON
            else "replan_each_frame_execute_action_0"
        ),
        "hand_transition": args.hand_transition,
        "servo_lag": (
            {
                "contract_id": SERVO_LAG_CONTRACT_ID,
                "gain_file": str(args.servo_gain_file),
                "gain_file_sha256": args.servo_gain_sha256,
                "source_interval_seconds": args.servo_gain_payload["source_interval_seconds"],
                "fit_row_count": args.servo_gain_payload["row_count"],
                "fit_transition_count": args.servo_gain_payload["transition_count"],
                "gains": args.servo_gains.tolist(),
            }
            if args.hand_transition == "calibrated_servo_lag"
            else None
        ),
        "act_mode": "batch",
        "act_batch_size": args.act_batch_size,
        "action_padding_contract": "physical_action[26:32]=0; never_executed",
        "extended_state": True,
        "state_contract": STATE_CONTRACT_ID,
        "contact_semantics": CONTACT_SEMANTICS,
        "contact_rule": CONTACT_RULE,
        "norm_stats_dir": str(args.norm_stats_dir),
        "norm_sha_expected": args.norm_sha_expected,
        "norm_sha_actual": args.norm_sha_actual,
        "model_path": args.model_path,
        "model": args.model,
        "row_index": row_index,
        "row_uuid": index.get("uuid"),
        "seed_uuid": index.get("seed_uuid"),
        "prompt": row.get("prompt"),
        "object_name": object_name,
        "source_frame_count": source_frames,
        "trajectory_frame_count": window.frame_count,
        "frame_window": window.as_dict(),
        "query_count": len(query_timings),
        "total_request_seconds": total_request_seconds,
        "mean_request_seconds": (
            total_request_seconds / len(query_timings) if query_timings else 0.0
        ),
        "query_timings": query_timings,
        "head_video": str(head_path),
        "wrist_video": str(wrist_path),
        "arrays": {
            "state_observation_32d": str(out / "state_observation_32d.npy"),
            "actions_raw_pred_normalized": str(out / "actions_raw_pred_normalized.npy"),
            "actions_raw_pred_physical": str(out / "actions_raw_pred_physical.npy"),
            "actions_commanded_physical": str(out / "actions_commanded_physical.npy"),
            "actions_applied_physical": str(out / "actions_applied_physical.npy"),
            "actions_gt_normalized": str(out / "actions_gt_normalized.npy"),
            "actions_gt_physical": str(out / "actions_gt_physical.npy"),
            "hand_state_sim": str(out / "hand_state_sim.npy"),
            "setpoint_targets_physical": str(out / "setpoint_targets_physical.npy"),
        },
        "client_commit": args.client_commit,
        "backend_commit": args.backend_commit,
        "model_commit": args.model_commit,
        "language_conditioning": args.language_conditioning,
    }


def run_mode3(*, args, row, data_config, headers, object_name, row_index, manifest_entry=None,
              session_id=None, output_dir=None) -> dict:
    source_frames = mode3_row_frame_count(row)
    window = contact_windows.select_window(row, row_index=row_index, total_frames=source_frames,
        manifest_entry=manifest_entry, context_frames=args.contact_context_frames,
        missing_policy=args.missing_contact_policy)
    if window is None or window.frame_count <= 0:
        raise ValueError(f"row {row_index} has no selected contact-window frames")
    if args.max_frames > 0:
        window = contact_windows.clamp_window(window, min(source_frames, window.start_frame + args.max_frames))
    out = (args.output_dir if output_dir is None else output_dir) / "mode3"
    out.mkdir(parents=True, exist_ok=True)
    # physics=True installs collision geoms required by v1 pair-presence contact;
    # Mode3 still calls only mj_forward and therefore never advances dynamics.
    tmp, model, data, renderer, object_addr, _, hand_addrs, _, limits = physics.make_scene(
        object_name, args.width, args.height, physics=True, create_renderer=True)
    kp_ids, object_ids, geom_to_finger = physics.resolve_keypoint_geom_ids(model, object_name)
    hand = np.asarray(row["state"][window.start_frame], dtype=np.float32)[:HAND_DIM].copy()
    hand, initial_clip = clip_hand_state(hand, limits)
    clipping = new_clipping_diagnostics(limits)
    record_clipping(clipping, initial_clip, initial=True)
    object_pos = np.asarray(row["objects"][0]["pos"], dtype=np.float64)
    object_rot = np.asarray(row["objects"][0]["rot_aa"], dtype=np.float64)
    object_z0 = float(object_pos[0, 2])
    raw_norm, raw_phys, gt_norm_steps, commanded, applied, setpoint_targets = [], [], [], [], [], []
    sim_hands, state_observations = [hand.copy()], []
    query_timings: list[dict] = []
    candidates: dict[int, dict] = {}
    query_frames = set(
        mode3_query_frames(window.start_frame, window.end_frame, args.query_stride)
    )
    active_session, owns_session = session_id or "", False
    head_path, wrist_path = out / "mode3_kinematic_head.mp4", out / "mode3_kinematic_wrist.mp4"
    try:
        active_session, owns_session = acquire_action_session(session_id, lambda: create_session(args, headers))
        with imageio.get_writer(str(head_path), fps=args.fps, macro_block_size=1, codec="libx264") as head_writer, \
             imageio.get_writer(str(wrist_path), fps=args.fps, macro_block_size=1, codec="libx264") as wrist_writer:
            for frame in range(window.start_frame, window.end_frame + 1):
                head, wrist = render_reference_state(model=model, data=data, renderer=renderer,
                    object_addr=object_addr, hand_addrs=hand_addrs, hand_state=hand,
                    object_pos=object_pos[frame], object_rot=object_rot[frame])
                state = build_extended_sim_state(hand_qpos=hand, model=model, data=data, object_name=object_name,
                    keypoint_geom_ids=kp_ids, object_geom_ids=object_ids, geom_id_to_finger=geom_to_finger,
                    reference_object_z=float(object_pos[frame, 2]), source_object_z0=object_z0)
                state_observations.append(state.copy())
                if frame in query_frames:
                    datum = build_datum(row, frame=frame, state_input=state, head_image=head, wrist_image=wrist,
                                        data_config=data_config, base_model=args.model, window_end=window.end_frame)
                    datum["data_config"] = data_config
                    pred_n, pred_p, gt_n, timing = query_action(args=args, headers=headers,
                        session_id=active_session, datum=datum)
                    qi = len(query_timings)
                    timing = {"query_index": qi, "source_frame": frame, **timing}
                    if timing.get("used_data_sharding") is not True:
                        raise RuntimeError("Mode3 requires sharded fixed batch act_batch")
                    if qi == 1 and args.max_warm_request_seconds > 0 and float(timing["wall_seconds"]) > args.max_warm_request_seconds:
                        raise RuntimeError(f"warm act_batch latency {timing['wall_seconds']:.3f}s exceeds limit")
                    assert_physical_padding_zero(pred_p[:HORIZON])
                    candidates[frame] = {"pred_norm": pred_n, "pred_phys": pred_p, "gt_norm": gt_n,
                                         "targets": reconstruct_absolute_target_chunk(hand.copy(), pred_p[:HORIZON])}
                    query_timings.append(timing)
                start = max(k for k in candidates if k <= frame)
                candidate = candidates[start]
                offset = frame - start
                if offset >= HORIZON:
                    raise RuntimeError(f"no non-overlapping Mode3 chunk at frame {frame}")
                target = candidate["targets"][offset]
                if args.hand_transition == "calibrated_servo_lag":
                    # Euler predictions are absolute and may choose an equivalent 2pi branch.
                    # Put the setpoint on the branch nearest current qpos before clipping/lag.
                    target = wrap_euler_target_near_current(target, hand)
                bounded_target, clip_event = clip_hand_state(target, limits)
                if args.hand_transition == "calibrated_servo_lag":
                    next_hand = servo_lag_step(hand, bounded_target, args.servo_gains)
                    next_hand, applied_clip = clip_hand_state(next_hand, limits)
                    if applied_clip["clipped_values"]:
                        raise RuntimeError("calibrated convex servo step unexpectedly left joint limits")
                else:
                    next_hand = bounded_target
                command = np.zeros(ACTION_DIM, dtype=np.float32)
                command[:HAND_DIM] = next_hand - hand
                # Physical B padding is fixed-zero and is never executed.
                applied_step = command.copy()
                hand = next_hand
                record_clipping(clipping, clip_event)
                raw_norm.append(candidate["pred_norm"][offset]); raw_phys.append(candidate["pred_phys"][offset])
                setpoint_targets.append(bounded_target.copy())
                gt_norm_steps.append(candidate["gt_norm"][offset])
                commanded.append(command); applied.append(applied_step); sim_hands.append(hand.copy())
                dataset_head = resize_for_video(decode_jpeg(row["image"][frame]), args.width, args.height)
                dataset_wrist = resize_for_video(decode_jpeg(row["wrist_image"][frame]), args.width, args.height)
                head_writer.append_data(side_by_side_video_frame(
                    head, dataset_head, sim_label=f"SIM Mode3 kinematic frame {frame}",
                    dataset_label=f"DATASET source frame {frame}"))
                wrist_writer.append_data(side_by_side_video_frame(
                    wrist, dataset_wrist, sim_label=f"SIM Mode3 kinematic wrist frame {frame}",
                    dataset_label=f"DATASET wrist source frame {frame}"))
    finally:
        if owns_session and active_session:
            delete_session(args, headers, active_session)
        renderer.close(); tmp.cleanup()
    for name, values in (("actions_raw_pred_normalized", raw_norm), ("actions_raw_pred_physical", raw_phys),
                         ("actions_commanded_physical", commanded), ("actions_applied_physical", applied),
                         ("actions_gt_normalized", gt_norm_steps),
                         ("actions_gt_physical", np.asarray(row["actions"])[window.start_frame:window.end_frame + 1]),
                         ("hand_state_sim", sim_hands),
                         ("setpoint_targets_physical", setpoint_targets),
                         ("state_observation_32d", state_observations)):
        np.save(out / f"{name}.npy", np.asarray(values, dtype=np.float32))
    result = _result_metadata(args, row_index=row_index, object_name=object_name, window=window,
        source_frames=source_frames, query_timings=query_timings, out=out, head_path=head_path,
        wrist_path=wrist_path, row=row)
    result["joint_limit_clipping"] = clipping
    result["pred_has_nan_inf"] = bool(not np.isfinite(np.asarray(raw_norm)).all())
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _client_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args(); args.base_url = args.base_url.rstrip("/")
    args.client_commit = args.client_commit or _client_commit()
    if args.act_batch_size != 4:
        raise ValueError("Mode3 requires --act-batch-size 4")
    if args.hand_transition == "calibrated_servo_lag":
        if args.query_stride != 1:
            raise ValueError("calibrated_servo_lag currently requires --query-stride 1")
        if args.servo_gain_file is None:
            raise ValueError("calibrated_servo_lag requires --servo-gain-file")
        args.servo_gains, args.servo_gain_payload, args.servo_gain_sha256 = load_gain_file(
            args.servo_gain_file
        )
    else:
        if args.servo_gain_file is not None:
            raise ValueError("--servo-gain-file requires calibrated_servo_lag")
        args.servo_gains = None; args.servo_gain_payload = None; args.servo_gain_sha256 = None
    if args.contact_context_frames < 0:
        raise ValueError("--contact-context-frames must be non-negative")
    _, args.norm_sha_actual = verify_locked_norm_stats(args.norm_stats_dir)
    args.norm_sha_expected = EXPECTED_NORM_SHA256
    args._gesture_index = GestureIndex.load(args.gesture_index) if args.language_conditioning == GESTURE_LANGUAGE else None
    eval_rows = parse_ordered_unique_csv(args.row_indices, option="--row-indices") if args.row_indices else [args.row_index]
    source = lance.dataset(str(args.lance_dataset)); row_count = source.count_rows()
    if any(index is None or not 0 <= int(index) < row_count for index in eval_rows):
        raise IndexError(f"row index out of range: {eval_rows}")
    norm_rows = list(range(row_count)) if args.normalization_row_indices.strip().lower() == "all" else [int(x) for x in args.normalization_row_indices.split(",") if x.strip()]
    if not norm_rows or any(not 0 <= x < row_count for x in norm_rows):
        raise IndexError("invalid --normalization-row-indices")
    manifest_path = Path(args.contact_window_manifest) if args.contact_window_manifest else contact_windows.default_manifest_path(args.lance_dataset)
    entries = contact_windows.load_or_build_windows(source, args.lance_dataset, sorted(set(norm_rows) | set(eval_rows)),
        manifest_path=manifest_path, context_frames=args.contact_context_frames, missing_policy=args.missing_contact_policy)
    norm_stats = L.normalize.load(args.norm_stats_dir)
    q01, q99 = np.asarray(norm_stats["state"].q01), np.asarray(norm_stats["state"].q99)
    if not np.allclose(q01[26:31], 0) or not np.allclose(q99[26:31], 1) or q99[31] - q01[31] < 1e-4:
        raise ValueError("locked v1 norm stats fail required contact/lift structure")
    data_config = L._make_data_config(L._build_model_config(HORIZON, action_dim=ACTION_DIM, base_model=args.model), norm_stats, action_source=URDF_TARGET_ABSOLUTE)
    args.output_dir.mkdir(parents=True, exist_ok=True); headers = L._headers(args.api_key)
    columns = ["state", "actions", "prompt", "objects", "trajectory_metadata", "episode_metadata", "image", "wrist_image", "index", "hands"]
    def load_row(index: int) -> dict:
        row = source.take([index], columns=columns).to_pylist()[0]
        return condition_row_language(project_row_actions(row, URDF_TARGET_ABSOLUTE), args.language_conditioning, row_index=index, gesture_index=args._gesture_index)
    shared = create_session(args, headers) if len(eval_rows) > 1 else None
    summaries = []
    try:
        for index in eval_rows:
            row = load_row(index); names = row["trajectory_metadata"].get("object_names") or []
            if len(names) != 1 or not isinstance(names[0], str): raise ValueError(f"Mode3 requires exactly one object at row {index}")
            root = args.output_dir if len(eval_rows) == 1 else args.output_dir / f"row{index}"
            result = run_mode3(args=args, row=row, data_config=data_config, headers=headers, object_name=names[0],
                row_index=index, manifest_entry=entries.get(index), session_id=shared, output_dir=root)
            summary = {"mode": "historical_kinematic_mode3", "row_index": index, "result": result,
                       "normalization_row_indices": norm_rows, "contact_window_manifest": str(manifest_path),
                       "client_commit": args.client_commit, "backend_commit": args.backend_commit, "model_commit": args.model_commit}
            (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); summaries.append(summary)
    finally:
        if shared: delete_session(args, headers, shared)
    final = summaries[0] if len(summaries) == 1 else {"mode": "historical_kinematic_mode3_multi_row", "shared_session": True, "rows": summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
