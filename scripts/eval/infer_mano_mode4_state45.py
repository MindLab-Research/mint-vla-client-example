#!/usr/bin/env python3
"""State45 phase-aware persistent Mode4 rollout with native ManoRL MuJoCo."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import imageio.v2 as imageio
import lance
import numpy as np

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM") or "egl"

import infer_mano_mode4 as base
import manorl_native_physics as physics
import mode4_data_support as full
from scripts import contact_windows
from scripts.mano_state41_contract import (
    ACTION_DIM,
    CONTACT_SLICE,
    FLOOR_SUPPORT_INDEX,
    HAND_QPOS_DIM,
    LIFT_HEIGHT_INDEX,
    MIN_MULTICONTACT_FINGERS,
    MULTICONTACT_PERSISTENCE_INDEX,
    SAMPLE_DT_SECONDS,
    STATE41_CONTRACT_ID,
    STATE_DIM as STATE41_DIM,
    SURFACE_DISTANCE_SLICE,
)
from scripts.mano_state45_contract import (
    STABLE_LIFT_INDEX,
    STATE45_CONTRACT_ID,
    STATE_DIM,
    TASK_PHASE_INDEX,
    ManoTaskPhaseTracker,
    PhaseTrackerConfig,
    assemble_live_state45,
)
from scripts.mano_forced_grasp_retry import (
    FORCED_GRASP_RETRY_CONTRACT_ID,
    ForcedGraspRetryController,
    ForcedRetryConfig,
)
from scripts.mano_task_phase import PHASE_TRACKER_CONTRACT_ID, TaskPhase
from scripts.target_actions import URDF_TARGET_ABSOLUTE, project_row_actions
from scripts.train.state45_gradea_contract import canonical_release_full_task_prompt

MODEL = "openpi/pi05-action-lora-r16-state45-phase-28dof-finetune"
HORIZON = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="tml-dummy")
    parser.add_argument("--model", choices=(MODEL,), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--row-indices", required=True)
    parser.add_argument("--normalization-row-indices", required=True)
    parser.add_argument("--state-contract", choices=("state45",), required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--norm-sha-expected", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--language-conditioning", choices=("gesture",), required=True
    )
    parser.add_argument("--contact-window-manifest", type=Path, required=True)
    parser.add_argument("--contact-context-frames", type=int, default=100)
    parser.add_argument("--missing-contact-policy", choices=("error",), default="error")
    parser.add_argument(
        "--frame-window", choices=("persistent_task",), default="persistent_task"
    )
    parser.add_argument("--max-control-seconds", type=float, default=15.0)
    parser.add_argument("--chunk-stride", type=int, default=1)
    parser.add_argument("--temporal-decay", type=float, default=0.4)
    parser.add_argument("--act-mode", choices=("batch", "single"), default="batch")
    parser.add_argument("--act-batch-size", type=int, default=4)
    parser.add_argument("--row-execution", choices=("sequential", "lockstep"), default="sequential")
    parser.add_argument("--row-batch-size", type=int, default=1)
    parser.add_argument("--max-warm-request-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--video-mode", choices=("full", "none"), default="full")
    parser.add_argument("--client-commit")
    parser.add_argument("--backend-commit")
    parser.add_argument("--model-commit")
    parser.add_argument("--action-session-id")
    parser.add_argument("--keep-action-session", action="store_true")
    parser.add_argument("--forced-first-failure", action="store_true")
    parser.add_argument("--forced-release-trigger-lift-m", type=float, default=0.03)
    parser.add_argument("--forced-release-floor-frames", type=int, default=10)
    parser.add_argument("--forced-release-no-contact-frames", type=int, default=20)
    parser.add_argument("--forced-release-max-frames", type=int, default=150)
    args = parser.parse_args()
    rows = base.parse_ordered_unique_csv(args.row_indices, option="--row-indices")
    if len(rows) != 1:
        raise ValueError("State45 Mode4 video evaluation currently requires exactly one row")
    args.row_index = rows[0]
    if not 1 <= args.chunk_stride < HORIZON:
        raise ValueError("--chunk-stride must be in [1,9]")
    if args.max_frames:
        raise ValueError("State45 persistent-task uses --max-control-seconds, not --max-frames")
    if not np.isfinite(args.max_control_seconds) or args.max_control_seconds <= 0:
        raise ValueError("--max-control-seconds must be finite and positive")
    if args.action_session_id and args.keep_action_session:
        raise ValueError("external and retained action sessions are mutually exclusive")
    args.forced_retry_config = ForcedRetryConfig(
        trigger_lift_m=args.forced_release_trigger_lift_m,
        floor_contact_frames=args.forced_release_floor_frames,
        no_hand_contact_frames=args.forced_release_no_contact_frames,
        max_forced_release_frames=args.forced_release_max_frames,
    )
    return args


def persistent_termination_reason(
    *, phase: TaskPhase | int, control_steps: int, max_control_frames: int
) -> str | None:
    """Return the causal task termination, with DONE taking priority at timeout."""
    if control_steps < 0 or max_control_frames <= 0:
        raise ValueError("persistent control counters must be non-negative/positive")
    if int(phase) == int(TaskPhase.DONE):
        return "done"
    if control_steps >= max_control_frames:
        return "timeout"
    return None


def reconstruct_absolute_target_chunk(query_q: np.ndarray, pred_phys: np.ndarray) -> np.ndarray:
    query = np.asarray(query_q, dtype=np.float32)
    output = np.asarray(pred_phys, dtype=np.float32)
    if query.shape != (HAND_QPOS_DIM,) or output.ndim != 2 or output.shape[1] < HAND_QPOS_DIM:
        raise ValueError(f"invalid State45 target shapes {query.shape}/{output.shape}")
    target = np.empty((len(output), HAND_QPOS_DIM), dtype=np.float32)
    target[:, :3] = query[:3] + output[:, :3]
    target[:, 3:6] = output[:, 3:6]
    target[:, 6:HAND_QPOS_DIM] = query[6:HAND_QPOS_DIM] + output[:, 6:HAND_QPOS_DIM]
    if not np.isfinite(target).all():
        raise FloatingPointError("State45 target reconstruction is non-finite")
    return target


def assemble_live_state41(
    *, hand_qpos: np.ndarray, contacts: np.ndarray, object_lift: float,
    signed_surface_distances: np.ndarray, floor_support: float, persistence: float,
) -> np.ndarray:
    state = np.empty(STATE41_DIM, dtype=np.float32)
    state[:HAND_QPOS_DIM] = np.asarray(hand_qpos, dtype=np.float32)
    state[CONTACT_SLICE] = np.asarray(contacts, dtype=np.float32)
    state[LIFT_HEIGHT_INDEX] = np.float32(object_lift)
    state[SURFACE_DISTANCE_SLICE] = np.asarray(signed_surface_distances, dtype=np.float32)
    state[FLOOR_SUPPORT_INDEX] = np.float32(floor_support)
    state[MULTICONTACT_PERSISTENCE_INDEX] = np.float32(persistence)
    if not np.isfinite(state).all():
        raise FloatingPointError("live state41 is non-finite")
    return state


def update_multicontact_run(run_frames: int, contacts: np.ndarray) -> tuple[int, np.float32]:
    if int(np.sum(np.asarray(contacts) > 0.5)) >= MIN_MULTICONTACT_FINGERS:
        run_frames += 1
        return run_frames, np.float32((run_frames - 1) * SAMPLE_DT_SECONDS)
    return 0, np.float32(0.0)


def condition_state45_language(row: dict, language_conditioning: str) -> dict:
    """Apply the same formal full-task prompt used by State45 training."""
    if language_conditioning == "gesture":
        prompt = canonical_release_full_task_prompt(
            row["index"], row["trajectory_metadata"]
        )
        return {**row, "prompt": prompt}
    return base.condition_row_language(
        row, language_conditioning, row_index=-1, gesture_index=None
    )


def load_row(args: argparse.Namespace) -> tuple[dict, dict]:
    source = lance.dataset(str(args.lance_dataset))
    if not 0 <= args.row_index < source.count_rows():
        raise IndexError(f"row out of range: {args.row_index}")
    columns = [
        "state", "actions", "prompt", "objects", "timestamp", "trajectory_metadata",
        "episode_metadata", "image", "wrist_image", "index", "hands",
    ]
    row = source.take([args.row_index], columns=columns).to_pylist()[0]
    right_hand = full.resolve_right_hand(row)
    qpos = np.asarray(right_hand.get("urdf_dof"), dtype=np.float32)
    state_qpos = np.asarray(row["state"], dtype=np.float32)[:, :HAND_QPOS_DIM]
    if qpos.shape != state_qpos.shape or not np.array_equal(qpos, state_qpos):
        raise ValueError(f"right-hand/state qpos mismatch: {qpos.shape}/{state_qpos.shape}")
    target = np.asarray(right_hand.get("urdf_dof_target"), dtype=np.float32)
    if target.shape != qpos.shape or not np.isfinite(target).all():
        raise ValueError(f"right-hand target mismatch: {target.shape}/{qpos.shape}")
    row = {**row, "hands": [right_hand]}
    row = project_row_actions(row, URDF_TARGET_ABSOLUTE, action_dim=ACTION_DIM)
    row = condition_state45_language(row, args.language_conditioning)
    return row, source


def run(args: argparse.Namespace) -> dict:
    profile = base.L.resolve_profile(args.model)
    if profile.state_dim != STATE_DIM or profile.action_dim != ACTION_DIM:
        raise ValueError("State45 profile width mismatch")
    _, norm_sha = base.verify_locked_norm_stats(
        args.norm_stats_dir, expected_sha256=args.norm_sha_expected
    )
    norm_stats = base.L.normalize.load(args.norm_stats_dir)
    for key, width in (("state", STATE_DIM), ("actions", ACTION_DIM)):
        if np.asarray(norm_stats[key].mean).shape != (width,):
            raise ValueError(f"norm width mismatch for {key}")
    data_config = base.L._make_data_config(
        full.build_model_config(args.model), norm_stats,
        action_source=URDF_TARGET_ABSOLUTE,
        delta_mask_segments=profile.delta_mask_segments,
    )
    row, _source = load_row(args)
    object_name = full.safe_object_name(row)
    manifest_raw, manifest_entries = contact_windows.load_manifest(args.contact_window_manifest)
    if manifest_raw.get("dataset") not in (None, str(args.lance_dataset)):
        raise ValueError("contact manifest dataset mismatch")
    # Contact metadata selects only the reproducible initialization frame.  The
    # persistent rollout end is determined by causal DONE or a wall-clock task
    # timeout, never by the source demonstration's last contact.
    window = full.resolve_row_window(
        row, row_index=args.row_index, frame_window="contact",
        contact_context_frames=args.contact_context_frames,
        missing_contact_policy=args.missing_contact_policy,
        manifest_entry=manifest_entries.get(args.row_index),
    )
    if window is None:
        raise ValueError("State45 Mode4 initialization window was skipped")
    window_start = int(window.start_frame)
    source_last_frame = len(row["state"]) - 1
    max_control_frames = int(np.ceil(args.max_control_seconds / SAMPLE_DT_SECONDS))
    if max_control_frames < 1:
        raise ValueError("persistent task timeout resolves to zero control frames")

    out = args.output_dir / "mode4"
    out.mkdir(parents=True, exist_ok=True)
    scene = physics.make_scene(
        object_name, args.width, args.height, physics=True,
        physics_timestep=physics.DT, create_renderer=True,
    )
    tmp, model, data, renderer, object_addr, _, hand_addrs, _, limits = scene
    del tmp
    feature_ids = physics.resolve_state41_feature_ids(model, object_name)
    object_positions_source = np.asarray(row["objects"][0]["pos"], dtype=np.float64)
    object_rotations_source = np.asarray(row["objects"][0]["rot_aa"], dtype=np.float64)
    source_state = np.asarray(row["state"], dtype=np.float32)
    object_z_reference = float(object_positions_source[0, 2])

    # Preserve both causal clocks across the source-derived initialization
    # prefix.  PhaseTracker consumes the persisted State41 evidence directly,
    # exactly as the offline training transform does.
    run_frames = 0
    phase_tracker = ManoTaskPhaseTracker(PhaseTrackerConfig())
    for prior_frame in range(window_start):
        full.set_scene_state(
            model, data, state=source_state[prior_frame, :HAND_QPOS_DIM],
            object_pos=object_positions_source[prior_frame],
            object_rot_aa=object_rotations_source[prior_frame], object_addr=object_addr,
            hand_addrs=list(hand_addrs),
        )
        prior_contacts, _surface, _floor, _pairs = physics.state41_features_from_mujoco(
            model, data, object_name, feature_ids=feature_ids
        )
        run_frames, _ = update_multicontact_run(run_frames, prior_contacts)
        phase_tracker.update(
            object_lift_m=float(source_state[prior_frame, LIFT_HEIGHT_INDEX]),
            hand_object_contact=bool(np.any(source_state[prior_frame, CONTACT_SLICE] > 0.5)),
            object_floor_contact=bool(source_state[prior_frame, FLOOR_SUPPORT_INDEX] > 0.5),
        )

    full.set_scene_state(
        model, data, state=source_state[window_start, :HAND_QPOS_DIM],
        object_pos=object_positions_source[window_start],
        object_rot_aa=object_rotations_source[window_start], object_addr=object_addr,
        hand_addrs=list(hand_addrs),
    )
    forced_retry: ForcedGraspRetryController | None = None
    forced_open_hand_qpos: np.ndarray | None = None
    if args.forced_first_failure:
        if window_start != 0:
            raise ValueError(
                "forced-first-failure requires frame0 initialization so the open pose and "
                "first attempt share one causal episode"
            )
        forced_retry = ForcedGraspRetryController(args.forced_retry_config)
        forced_open_hand_qpos = np.asarray(
            source_state[0, :HAND_QPOS_DIM], dtype=np.float32
        ).copy()
        if forced_open_hand_qpos.shape != (HAND_QPOS_DIM,) or not np.isfinite(
            forced_open_hand_qpos
        ).all():
            raise ValueError("forced-release source-frame0 open pose is invalid")

    headers = base.L._headers(args.api_key)
    try:
        session_id = args.action_session_id or base.create_session(args, headers)
    except BaseException:
        renderer.close()
        raise
    owns_session = args.action_session_id is None
    candidates: list[dict] = []
    query_timings: list[dict] = []
    arrays: dict[str, list] = {
        "actions_raw_pred_normalized": [], "actions_raw_pred_physical": [],
        "actions_commanded_physical": [], "preclip_absolute_targets": [],
        "servo_position_targets": [], "servo_target_clipping_correction": [],
        "actions_applied_physical": [], "rollout_observation_state": [],
        "rollout_observation_contacts": [], "rollout_observation_lift": [],
        "rollout_observation_surface_distance": [], "rollout_observation_floor_support": [],
        "rollout_observation_multicontact_persistence": [],
        "rollout_observation_phase_features": [], "physics_contact_flags": [],
        "step_max_contact_force": [],
    }
    if forced_retry is not None:
        arrays.update(
            {
                "policy_preintervention_absolute_targets": [],
                "forced_release_target_correction": [],
                "forced_release_active": [],
            }
        )
    hand_states = [np.asarray(data.qpos[hand_addrs], dtype=np.float32).copy()]
    object_states = [np.asarray(data.qpos[object_addr:object_addr + 3], dtype=np.float32).copy()]
    object_quaternions = [
        np.asarray(data.qpos[object_addr + 3:object_addr + 7], dtype=np.float32).copy()
    ]
    contact_counts = {"hand_object": 0, "object_floor": 0, "hand_floor": 0}
    max_ncon = 0
    max_contact_force = 0.0
    max_abs_actuator_force = 0.0
    max_abs_qvel = 0.0
    head_path = out / "mode4_physics_vs_dataset_head.mp4"
    wrist_path = out / "mode4_physics_vs_dataset_wrist.mp4"
    dataset_path = out / "dataset_reference.mp4"
    termination_reason = "timeout"
    terminal_state: np.ndarray | None = None
    control_steps = 0
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            head_writer = wrist_writer = dataset_writer = None
            if args.video_mode == "full":
                head_writer = stack.enter_context(
                    imageio.get_writer(str(head_path), fps=args.fps, macro_block_size=1, codec="libx264")
                )
                wrist_writer = stack.enter_context(
                    imageio.get_writer(str(wrist_path), fps=args.fps, macro_block_size=1, codec="libx264")
                )
                dataset_writer = stack.enter_context(
                    imageio.get_writer(str(dataset_path), fps=args.fps, macro_block_size=1, codec="libx264")
                )
                for source_frame, blob in enumerate(row["image"]):
                    dataset_writer.append_data(
                        base.label(base.resize_for_video(base.decode_jpeg(blob), args.width, args.height),
                                   f"DATASET reference source frame {source_frame}")
                    )

            for control_frame in range(max_control_frames):
                source_frame = min(window_start + control_frame, source_last_frame)
                current_q = np.asarray(data.qpos[hand_addrs], dtype=np.float64).copy()
                contacts, surface, floor_support, _pairs = physics.state41_features_from_mujoco(
                    model, data, object_name, feature_ids=feature_ids
                )
                run_frames, persistence = update_multicontact_run(run_frames, contacts)
                object_lift = float(data.qpos[object_addr + 2]) - object_z_reference
                state41 = assemble_live_state41(
                    hand_qpos=current_q, contacts=contacts,
                    object_lift=object_lift,
                    signed_surface_distances=surface, floor_support=float(floor_support),
                    persistence=float(persistence),
                )
                phase_observation = phase_tracker.update(
                    object_lift_m=object_lift,
                    hand_object_contact=bool(np.any(np.asarray(contacts) > 0.5)),
                    object_floor_contact=float(floor_support),
                )
                state = assemble_live_state45(state41, phase_observation)
                if forced_retry is not None:
                    forced_retry.observe(
                        control_frame=control_steps,
                        object_lift_m=object_lift,
                        hand_object_contact=bool(np.any(np.asarray(contacts) > 0.5)),
                        object_floor_contact=float(floor_support),
                        stable_lift_achieved=float(state[STABLE_LIFT_INDEX]),
                        task_phase=float(state[TASK_PHASE_INDEX]),
                    )
                    if forced_retry.intervention_invalid:
                        termination_reason = "intervention_invalid"
                        terminal_state = state.copy()
                        break
                reason = persistent_termination_reason(
                    phase=phase_tracker.phase,
                    control_steps=control_steps,
                    max_control_frames=max_control_frames,
                )
                if reason == "done":
                    termination_reason = reason
                    terminal_state = state.copy()
                    break
                arrays["rollout_observation_state"].append(state.copy())
                arrays["rollout_observation_contacts"].append(contacts.copy())
                arrays["rollout_observation_lift"].append(state[LIFT_HEIGHT_INDEX])
                arrays["rollout_observation_surface_distance"].append(surface.copy())
                arrays["rollout_observation_floor_support"].append(state[FLOOR_SUPPORT_INDEX])
                arrays["rollout_observation_multicontact_persistence"].append(
                    state[MULTICONTACT_PERSISTENCE_INDEX]
                )
                arrays["rollout_observation_phase_features"].append(
                    state[STATE41_DIM:STATE_DIM].copy()
                )

                if args.video_mode == "full" and control_frame == 0:
                    sim_head, sim_wrist = physics.render_current_state(model, data, renderer)
                    dataset_head = base.resize_for_video(
                        base.decode_jpeg(row["image"][source_frame]), args.width, args.height
                    )
                    dataset_wrist = base.resize_for_video(
                        base.decode_jpeg(row["wrist_image"][source_frame]), args.width, args.height
                    )
                    head_writer.append_data(np.concatenate([
                        base.label(sim_head, f"MODE4 PHYSICS control frame {control_frame}"),
                        base.label(dataset_head, f"DATASET source frame {source_frame}"),
                    ], axis=1))
                    wrist_writer.append_data(np.concatenate([
                        base.label(sim_wrist, f"MODE4 PHYSICS wrist control frame {control_frame}"),
                        base.label(dataset_wrist, f"DATASET wrist source frame {source_frame}"),
                    ], axis=1))

                if control_frame % args.chunk_stride == 0:
                    head_image, wrist_image = physics.render_current_state(model, data, renderer)
                    datum = base.build_datum(
                        row, frame=source_frame, state_input=state, head_image=head_image,
                        wrist_image=wrist_image, data_config=data_config,
                        base_model=args.model, window_end=source_last_frame,
                    )
                    datum["data_config"] = data_config
                    pred_norm, pred_phys, _gt_norm, timing = base.query_action(
                        args=args, headers=headers, session_id=session_id, datum=datum
                    )
                    query_timings.append({
                        "control_frame": control_frame,
                        "source_reference_frame": source_frame,
                        **timing,
                    })
                    if len(query_timings) == 2 and args.max_warm_request_seconds > 0:
                        if float(timing["wall_seconds"]) > args.max_warm_request_seconds:
                            raise RuntimeError(
                                f"warm latency {timing['wall_seconds']:.3f}s exceeds limit"
                            )
                    candidates.append({
                        "start": control_frame, "pred_norm": pred_norm, "pred_phys": pred_phys,
                        "target_hand": reconstruct_absolute_target_chunk(current_q, pred_phys[:HORIZON]),
                    })
                    candidates = [
                        item for item in candidates
                        if control_frame < item["start"] + HORIZON
                    ]

                active = [
                    item for item in candidates
                    if item["start"] <= control_frame < item["start"] + HORIZON
                ]
                if not active:
                    raise RuntimeError(f"no action candidate at control frame {control_frame}")
                newest = max(active, key=lambda item: item["start"])
                newest_start = newest["start"]
                local_newest = control_frame - newest_start
                weights = np.asarray([
                    args.temporal_decay ** ((newest_start - item["start"]) // args.chunk_stride)
                    for item in active
                ], dtype=np.float64)
                weights /= weights.sum()
                policy_absolute_target = np.asarray(
                    sum(
                        weight * item["target_hand"][control_frame - item["start"]]
                        for weight, item in zip(weights, active, strict=True)
                    ),
                    dtype=np.float32,
                )
                absolute_target = policy_absolute_target
                forced_release_correction = np.zeros(HAND_QPOS_DIM, dtype=np.float32)
                forced_release_active = False
                if forced_retry is not None:
                    if forced_open_hand_qpos is None:
                        raise RuntimeError("forced-retry open pose is missing")
                    forced_release_active = forced_retry.override_active
                    absolute_target, forced_release_correction = (
                        forced_retry.apply_action_override(
                            policy_absolute_target, forced_open_hand_qpos
                        )
                    )
                target, clipping = physics.nearest_wrapped_position_target(
                    current_q, absolute_target - current_q, limits
                )
                before = current_q.copy()
                diagnostics = physics.step_servo(
                    model=model, data=data, target=target,
                    substeps=physics.NATIVE_SUBSTEPS, object_name=object_name,
                )
                after = np.asarray(data.qpos[hand_addrs], dtype=np.float64).copy()
                if forced_retry is not None and forced_release_active:
                    forced_retry.record_override_action()
                for key in contact_counts:
                    contact_counts[key] += int(diagnostics[f"{key}_contact"])
                max_ncon = max(max_ncon, int(diagnostics["max_ncon"]))
                max_contact_force = max(max_contact_force, float(diagnostics["max_contact_force"]))
                max_abs_actuator_force = max(
                    max_abs_actuator_force, float(diagnostics["max_abs_actuator_force"])
                )
                max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(data.qvel))))
                arrays["actions_raw_pred_normalized"].append(newest["pred_norm"][local_newest])
                arrays["actions_raw_pred_physical"].append(newest["pred_phys"][local_newest])
                commanded = np.zeros(ACTION_DIM, dtype=np.float32)
                commanded[:HAND_QPOS_DIM] = absolute_target - current_q
                arrays["actions_commanded_physical"].append(commanded)
                arrays["preclip_absolute_targets"].append(np.asarray(absolute_target, dtype=np.float32))
                arrays["servo_position_targets"].append(np.asarray(target, dtype=np.float32))
                arrays["servo_target_clipping_correction"].append(
                    np.asarray(target - absolute_target, dtype=np.float32)
                )
                applied = np.zeros(ACTION_DIM, dtype=np.float32)
                applied[:HAND_QPOS_DIM] = after - before
                arrays["actions_applied_physical"].append(applied)
                if forced_retry is not None:
                    arrays["policy_preintervention_absolute_targets"].append(
                        policy_absolute_target.copy()
                    )
                    arrays["forced_release_target_correction"].append(
                        forced_release_correction.copy()
                    )
                    arrays["forced_release_active"].append(forced_release_active)
                arrays["physics_contact_flags"].append([
                    diagnostics["hand_object_contact"], diagnostics["object_floor_contact"],
                    diagnostics["hand_floor_contact"],
                ])
                arrays["step_max_contact_force"].append(diagnostics["max_contact_force"])
                hand_states.append(after.astype(np.float32))
                object_states.append(
                    np.asarray(data.qpos[object_addr:object_addr + 3], dtype=np.float32).copy()
                )
                object_quaternions.append(
                    np.asarray(data.qpos[object_addr + 3:object_addr + 7], dtype=np.float32).copy()
                )

                control_steps += 1
                if args.video_mode == "full":
                    next_control_frame = control_frame + 1
                    next_source_frame = min(source_frame + 1, source_last_frame)
                    sim_head, sim_wrist = physics.render_current_state(model, data, renderer)
                    dataset_head = base.resize_for_video(
                        base.decode_jpeg(row["image"][next_source_frame]), args.width, args.height
                    )
                    dataset_wrist = base.resize_for_video(
                        base.decode_jpeg(row["wrist_image"][next_source_frame]), args.width, args.height
                    )
                    head_writer.append_data(np.concatenate([
                        base.label(sim_head, f"MODE4 PHYSICS control frame {next_control_frame}"),
                        base.label(dataset_head, f"DATASET source frame {next_source_frame}"),
                    ], axis=1))
                    wrist_writer.append_data(np.concatenate([
                        base.label(sim_wrist, f"MODE4 PHYSICS wrist control frame {next_control_frame}"),
                        base.label(dataset_wrist, f"DATASET wrist source frame {next_source_frame}"),
                    ], axis=1))

        # The state after the last permitted action is still an observation
        # frame.  Evaluate it before declaring timeout so a DONE transition
        # caused by that action wins at the exact 15-second boundary.
        if termination_reason not in {"done", "intervention_invalid"}:
            current_q = np.asarray(data.qpos[hand_addrs], dtype=np.float64).copy()
            contacts, surface, floor_support, _pairs = physics.state41_features_from_mujoco(
                model, data, object_name, feature_ids=feature_ids
            )
            run_frames, persistence = update_multicontact_run(run_frames, contacts)
            object_lift = float(data.qpos[object_addr + 2]) - object_z_reference
            state41 = assemble_live_state41(
                hand_qpos=current_q,
                contacts=contacts,
                object_lift=object_lift,
                signed_surface_distances=surface,
                floor_support=float(floor_support),
                persistence=float(persistence),
            )
            phase_observation = phase_tracker.update(
                object_lift_m=object_lift,
                hand_object_contact=bool(np.any(np.asarray(contacts) > 0.5)),
                object_floor_contact=float(floor_support),
            )
            terminal_state = assemble_live_state45(state41, phase_observation)
            if forced_retry is not None:
                forced_retry.observe(
                    control_frame=control_steps,
                    object_lift_m=object_lift,
                    hand_object_contact=bool(np.any(np.asarray(contacts) > 0.5)),
                    object_floor_contact=float(floor_support),
                    stable_lift_achieved=float(terminal_state[STABLE_LIFT_INDEX]),
                    task_phase=float(terminal_state[TASK_PHASE_INDEX]),
                )
            if forced_retry is not None and forced_retry.intervention_invalid:
                termination_reason = "intervention_invalid"
            else:
                termination_reason = persistent_termination_reason(
                    phase=phase_tracker.phase,
                    control_steps=control_steps,
                    max_control_frames=max_control_frames,
                ) or "timeout"
    finally:
        if owns_session and session_id and not args.keep_action_session:
            base.delete_session(args, headers, session_id)
        renderer.close()

    final_reason = persistent_termination_reason(
        phase=phase_tracker.phase,
        control_steps=control_steps,
        max_control_frames=max_control_frames,
    )
    if termination_reason == "intervention_invalid":
        if forced_retry is None or not forced_retry.intervention_invalid:
            raise RuntimeError("intervention-invalid termination lacks controller evidence")
    elif termination_reason != "done":
        if final_reason != "timeout":
            raise RuntimeError(
                "persistent rollout ended without causal DONE or exhausted timeout"
            )
        termination_reason = final_reason

    forced_retry_ledger = None
    if forced_retry is not None:
        forced_retry_ledger = forced_retry.finalize(
            termination_reason=termination_reason,
            control_frame=control_steps,
        )

    np_arrays = {name: np.asarray(values) for name, values in arrays.items()}
    np_arrays["hand_state_sim"] = np.asarray(hand_states, dtype=np.float32)
    np_arrays["object_position_sim"] = np.asarray(object_states, dtype=np.float32)
    np_arrays["object_quaternion_sim"] = np.asarray(object_quaternions, dtype=np.float32)
    if terminal_state is not None:
        np_arrays["terminal_observation_state"] = np.asarray(terminal_state, dtype=np.float32)
    for name, value in np_arrays.items():
        np.save(out / f"{name}.npy", value)
    if np_arrays["rollout_observation_state"].shape != (control_steps, STATE_DIM):
        raise RuntimeError("State45 rollout state shape mismatch")
    if np_arrays["hand_state_sim"].shape != (control_steps + 1, HAND_QPOS_DIM):
        raise RuntimeError("State45 hand-state/control length mismatch")
    if np_arrays.get("terminal_observation_state", np.empty(0)).shape != (STATE_DIM,):
        raise RuntimeError("State45 terminal observation is missing or malformed")
    control_array_names = [
        "actions_raw_pred_normalized",
        "actions_raw_pred_physical",
        "actions_commanded_physical",
        "actions_applied_physical",
        "rollout_observation_phase_features",
    ]
    if forced_retry is not None:
        control_array_names.extend(
            [
                "policy_preintervention_absolute_targets",
                "forced_release_target_correction",
                "forced_release_active",
            ]
        )
    for name in control_array_names:
        if len(np_arrays[name]) != control_steps:
            raise RuntimeError(f"State45 control-array length mismatch for {name}")
    if not all(np.isfinite(value).all() for value in np_arrays.values() if value.dtype.kind == "f"):
        raise FloatingPointError("State45 Mode4 artifact contains non-finite values")
    expected_steps = physics.NATIVE_SUBSTEPS * control_steps
    if not np.isclose(data.time, expected_steps * physics.DT, rtol=0, atol=1e-9):
        raise RuntimeError("State45 Mode4 physics time mismatch")

    result = {
        "mode": "mode4_state45_phase_28dof_native",
        "model": args.model,
        "model_path": args.model_path,
        "state_contract": STATE45_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": HORIZON,
        "row_index": args.row_index,
        "object_name": object_name,
        "prompt": row["prompt"],
        "frame_window": {
            "type": "persistent_task",
            "initialization_start_frame": window_start,
            "source_contact_end_frame": int(window.end_frame),
            "source_reference_last_frame": source_last_frame,
            "control_steps": control_steps,
            "state_frame_count": control_steps + 1,
            "max_control_seconds": args.max_control_seconds,
            "max_control_frames": max_control_frames,
            "termination_reason": termination_reason,
            "context_frames": args.contact_context_frames,
            "manifest": str(args.contact_window_manifest),
        },
        "task_phase": {
            "tracker_contract": PHASE_TRACKER_CONTRACT_ID,
            "tracker_config": PhaseTrackerConfig().__dict__,
            "terminal_phase": int(phase_tracker.phase),
            "done": termination_reason == "done",
        },
        "forced_retry": forced_retry_ledger,
        "forced_retry_contract": (
            FORCED_GRASP_RETRY_CONTRACT_ID if forced_retry is not None else None
        ),
        "forced_release_open_pose_source_frame": 0 if forced_retry is not None else None,
        "physics": {
            "engine": "native MuJoCo", "controller": "manorl_native_position_servo",
            "timestep_seconds": physics.DT,
            "steps_per_source_interval": physics.NATIVE_SUBSTEPS,
            "mj_step_calls": expected_steps, "simulated_seconds": float(data.time),
            "contacts": contact_counts, "max_ncon": max_ncon,
            "max_contact_force": max_contact_force,
            "max_abs_actuator_force": max_abs_actuator_force,
            "max_abs_qvel": max_abs_qvel,
        },
        "query_count": len(query_timings),
        "query_timings": query_timings,
        "norm_sha_expected": args.norm_sha_expected,
        "norm_sha_actual": norm_sha,
        "client_commit": args.client_commit,
        "backend_commit": args.backend_commit,
        "model_commit": args.model_commit,
        "action_session_id": session_id,
        "video_mode": args.video_mode,
        "head_video": str(head_path) if args.video_mode == "full" else None,
        "wrist_video": str(wrist_path) if args.video_mode == "full" else None,
        "dataset_replay_video": str(dataset_path) if args.video_mode == "full" else None,
        "arrays": {name: str(out / f"{name}.npy") for name in np_arrays},
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.keep_action_session and owns_session:
        marker = base.write_retained_session_marker(args, session_id)
        result["action_session_marker"] = str(marker)
    return result


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.action_source = URDF_TARGET_ABSOLUTE
    args.extended_state = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    summary = {
        "mode": "mode4_state45_phase_28dof_native",
        "model": args.model,
        "model_path": args.model_path,
        "row_indices": [args.row_index],
        "state_contract": STATE45_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "norm_stats_dir": str(args.norm_stats_dir),
        "norm_sha_expected": args.norm_sha_expected,
        "result": result,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
