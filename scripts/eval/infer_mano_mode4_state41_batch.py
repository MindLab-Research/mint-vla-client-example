#!/usr/bin/env python3
"""Batched state41/28DoF Mode4 rollout with independent native MuJoCo rows."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import lance
import numpy as np

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM") or "egl"

import infer_mano_mode4 as base
import infer_mano_mode4_state41 as single
import manorl_native_physics as physics
import mode4_data_support as full
from scripts import contact_windows
from scripts.mano_state41_contract import (
    ACTION_DIM,
    CONTACT_SLICE,
    FLOOR_SUPPORT_INDEX,
    HAND_QPOS_DIM,
    LIFT_HEIGHT_INDEX,
    MULTICONTACT_PERSISTENCE_INDEX,
    STATE41_CONTRACT_ID,
    STATE_DIM,
    SURFACE_DISTANCE_SLICE,
)
from scripts.target_actions import URDF_TARGET_ABSOLUTE

MODEL = "openpi/pi05-action-lora-r16-state41-28dof-finetune"
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
    parser.add_argument("--state-contract", choices=("state41",), required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--norm-sha-expected", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--language-conditioning", choices=("gesture", "object_only"), required=True
    )
    parser.add_argument("--contact-window-manifest", type=Path, required=True)
    parser.add_argument("--contact-context-frames", type=int, default=100)
    parser.add_argument("--missing-contact-policy", choices=("error",), default="error")
    parser.add_argument("--frame-window", choices=("contact",), default="contact")
    parser.add_argument("--chunk-stride", type=int, default=5)
    parser.add_argument("--temporal-decay", type=float, default=0.4)
    parser.add_argument("--act-mode", choices=("batch",), default="batch")
    parser.add_argument("--act-batch-size", type=int, default=4)
    parser.add_argument("--row-execution", choices=("lockstep",), default="lockstep")
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--max-warm-request-seconds", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--video-mode", choices=("none",), default="none")
    parser.add_argument("--client-commit")
    parser.add_argument("--backend-commit")
    parser.add_argument("--model-commit")
    args = parser.parse_args()
    args.row_indices_list = base.parse_ordered_unique_csv(
        args.row_indices, option="--row-indices"
    )
    if not args.row_indices_list:
        raise ValueError("at least one row is required")
    if not 1 <= args.chunk_stride < HORIZON:
        raise ValueError("--chunk-stride must be in [1,9]")
    if args.act_batch_size < 1 or args.row_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.row_batch_size > args.act_batch_size:
        raise ValueError("--row-batch-size must be <= --act-batch-size")
    if args.max_frames not in (0,) and args.max_frames < 2:
        raise ValueError("--max-frames must be zero or at least two")
    return args


def _empty_arrays() -> dict[str, list]:
    return {
        "actions_raw_pred_normalized": [],
        "actions_raw_pred_physical": [],
        "actions_commanded_physical": [],
        "preclip_absolute_targets": [],
        "servo_position_targets": [],
        "servo_target_clipping_correction": [],
        "actions_applied_physical": [],
        "rollout_observation_state": [],
        "rollout_observation_contacts": [],
        "rollout_observation_lift": [],
        "rollout_observation_surface_distance": [],
        "rollout_observation_floor_support": [],
        "rollout_observation_multicontact_persistence": [],
        "physics_contact_flags": [],
        "step_max_contact_force": [],
    }


def _initialize_context(args, row_index: int, data_config, manifest_entries) -> dict:
    args.row_index = row_index
    row, _source = single.load_row(args)
    object_name = full.safe_object_name(row)
    window = full.resolve_row_window(
        row,
        row_index=row_index,
        frame_window=args.frame_window,
        contact_context_frames=args.contact_context_frames,
        missing_contact_policy=args.missing_contact_policy,
        manifest_entry=manifest_entries.get(row_index),
    )
    if window is None:
        raise ValueError(f"row {row_index} was skipped by contact window policy")
    window_start, window_end = int(window.start_frame), int(window.end_frame)
    if args.max_frames:
        window_end = min(window_end, window_start + args.max_frames - 1)
    frame_count = window_end - window_start + 1
    if frame_count < 2:
        raise ValueError(f"row {row_index} needs at least two frames")

    scene = physics.make_scene(
        object_name,
        args.width,
        args.height,
        physics=True,
        physics_timestep=physics.DT,
        create_renderer=True,
    )
    tmp, model, data, renderer, object_addr, _, hand_addrs, _, limits = scene
    del tmp
    feature_ids = physics.resolve_state41_feature_ids(model, object_name)
    source_state = np.asarray(row["state"], dtype=np.float32)
    object_positions = np.asarray(row["objects"][0]["pos"], dtype=np.float64)
    object_rotations = np.asarray(row["objects"][0]["rot_aa"], dtype=np.float64)
    object_z_reference = float(object_positions[0, 2])

    run_frames = 0
    for prior_frame in range(window_start):
        full.set_scene_state(
            model,
            data,
            state=source_state[prior_frame, :HAND_QPOS_DIM],
            object_pos=object_positions[prior_frame],
            object_rot_aa=object_rotations[prior_frame],
            object_addr=object_addr,
            hand_addrs=list(hand_addrs),
        )
        prior_contacts, _surface, _floor, _pairs = physics.state41_features_from_mujoco(
            model, data, object_name, feature_ids=feature_ids
        )
        run_frames, _ = single.update_multicontact_run(run_frames, prior_contacts)
    full.set_scene_state(
        model,
        data,
        state=source_state[window_start, :HAND_QPOS_DIM],
        object_pos=object_positions[window_start],
        object_rot_aa=object_rotations[window_start],
        object_addr=object_addr,
        hand_addrs=list(hand_addrs),
    )

    hand_states = [np.asarray(data.qpos[hand_addrs], dtype=np.float32).copy()]
    object_states = [
        np.asarray(data.qpos[object_addr : object_addr + 3], dtype=np.float32).copy()
    ]
    object_quaternions = [
        np.asarray(data.qpos[object_addr + 3 : object_addr + 7], dtype=np.float32).copy()
    ]
    return {
        "row_index": row_index,
        "row": row,
        "object_name": object_name,
        "window": window,
        "window_start": window_start,
        "window_end": window_end,
        "frame_count": frame_count,
        "model": model,
        "data": data,
        "renderer": renderer,
        "object_addr": object_addr,
        "hand_addrs": hand_addrs,
        "limits": limits,
        "feature_ids": feature_ids,
        "object_positions": object_positions,
        "object_rotations": object_rotations,
        "object_z_reference": object_z_reference,
        "run_frames": run_frames,
        "candidates": [],
        "pending": None,
        "arrays": _empty_arrays(),
        "hand_states": hand_states,
        "object_states": object_states,
        "object_quaternions": object_quaternions,
        "query_timings": [],
        "contact_counts": {"hand_object": 0, "object_floor": 0, "hand_floor": 0},
        "max_ncon": 0,
        "max_contact_force": 0.0,
        "max_abs_actuator_force": 0.0,
        "max_abs_qvel": 0.0,
    }


def _observe(context: dict) -> tuple[np.ndarray, np.ndarray]:
    model, data = context["model"], context["data"]
    hand_addrs = context["hand_addrs"]
    current_q = np.asarray(data.qpos[hand_addrs], dtype=np.float64).copy()
    contacts, surface, floor_support, _pairs = physics.state41_features_from_mujoco(
        model,
        data,
        context["object_name"],
        feature_ids=context["feature_ids"],
    )
    context["run_frames"], persistence = single.update_multicontact_run(
        context["run_frames"], contacts
    )
    state = single.assemble_live_state41(
        hand_qpos=current_q,
        contacts=contacts,
        object_lift=float(data.qpos[context["object_addr"] + 2])
        - context["object_z_reference"],
        signed_surface_distances=surface,
        floor_support=float(floor_support),
        persistence=float(persistence),
    )
    arrays = context["arrays"]
    arrays["rollout_observation_state"].append(state.copy())
    arrays["rollout_observation_contacts"].append(contacts.copy())
    arrays["rollout_observation_lift"].append(state[LIFT_HEIGHT_INDEX])
    arrays["rollout_observation_surface_distance"].append(surface.copy())
    arrays["rollout_observation_floor_support"].append(state[FLOOR_SUPPORT_INDEX])
    arrays["rollout_observation_multicontact_persistence"].append(
        state[MULTICONTACT_PERSISTENCE_INDEX]
    )
    return state, current_q


def _record_action_and_step(context: dict, relative_frame: int, temporal_decay: float) -> None:
    candidates = [
        item
        for item in context["candidates"]
        if relative_frame < item["start"] + HORIZON
    ]
    context["candidates"] = candidates
    if not candidates:
        raise RuntimeError(f"row {context['row_index']} has no action at frame {relative_frame}")
    newest_start = max(item["start"] for item in candidates)
    weights = np.asarray(
        [
            temporal_decay ** ((newest_start - item["start"]) // context["chunk_stride"])
            for item in candidates
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    absolute_target = sum(
        weight * item["target_hand"][relative_frame - item["start"]]
        for weight, item in zip(weights, candidates, strict=True)
    )
    current_q = context["pending"][1]
    target, _clipping = physics.nearest_wrapped_position_target(
        current_q,
        absolute_target - current_q,
        context["limits"],
    )
    diagnostics = physics.step_servo(
        model=context["model"],
        data=context["data"],
        target=target,
        substeps=physics.NATIVE_SUBSTEPS,
        object_name=context["object_name"],
    )
    after = np.asarray(context["data"].qpos[context["hand_addrs"]], dtype=np.float64).copy()
    before = current_q
    for key in context["contact_counts"]:
        context["contact_counts"][key] += int(diagnostics[f"{key}_contact"])
    context["max_ncon"] = max(context["max_ncon"], int(diagnostics["max_ncon"]))
    context["max_contact_force"] = max(
        context["max_contact_force"], float(diagnostics["max_contact_force"])
    )
    context["max_abs_actuator_force"] = max(
        context["max_abs_actuator_force"], float(diagnostics["max_abs_actuator_force"])
    )
    context["max_abs_qvel"] = max(
        context["max_abs_qvel"], float(np.max(np.abs(context["data"].qvel)))
    )
    newest = max(candidates, key=lambda item: item["start"])
    local_newest = relative_frame - newest["start"]
    arrays = context["arrays"]
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
    arrays["physics_contact_flags"].append(
        [
            diagnostics["hand_object_contact"],
            diagnostics["object_floor_contact"],
            diagnostics["hand_floor_contact"],
        ]
    )
    arrays["step_max_contact_force"].append(diagnostics["max_contact_force"])
    context["hand_states"].append(after.astype(np.float32))
    context["object_states"].append(
        np.asarray(
            context["data"].qpos[context["object_addr"] : context["object_addr"] + 3],
            dtype=np.float32,
        ).copy()
    )
    context["object_quaternions"].append(
        np.asarray(
            context["data"].qpos[context["object_addr"] + 3 : context["object_addr"] + 7],
            dtype=np.float32,
        ).copy()
    )
    context["pending"] = None


def _finalize_context(context: dict, output_root: Path, args: argparse.Namespace) -> dict:
    out = output_root / "rows" / f"row_{context['row_index']}" / "artifacts" / "mode4"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(values) for name, values in context["arrays"].items()}
    arrays["hand_state_sim"] = np.asarray(context["hand_states"], dtype=np.float32)
    arrays["object_position_sim"] = np.asarray(context["object_states"], dtype=np.float32)
    arrays["object_quaternion_sim"] = np.asarray(
        context["object_quaternions"], dtype=np.float32
    )
    expected_steps = physics.NATIVE_SUBSTEPS * (context["frame_count"] - 1)
    if arrays["rollout_observation_state"].shape != (
        context["frame_count"] - 1,
        STATE_DIM,
    ):
        raise RuntimeError(f"row {context['row_index']} state shape mismatch")
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise FloatingPointError(f"row {context['row_index']} non-finite {name}")
        np.save(out / f"{name}.npy", value)
    if not np.isclose(
        context["data"].time,
        expected_steps * physics.DT,
        rtol=0,
        atol=1e-9,
    ):
        raise RuntimeError(f"row {context['row_index']} MuJoCo time mismatch")
    result = {
        "mode": "mode4_state41_28dof_native",
        "row_index": context["row_index"],
        "object_name": context["object_name"],
        "prompt": context["row"]["prompt"],
        "state_contract": STATE41_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": HORIZON,
        "frame_window": {
            "start_frame": context["window_start"],
            "end_frame": context["window_end"],
            "frame_count": context["frame_count"],
            "context_frames": args.contact_context_frames,
            "manifest": str(args.contact_window_manifest),
        },
        "physics": {
            "engine": "native MuJoCo",
            "controller": "manorl_native_position_servo",
            "timestep_seconds": physics.DT,
            "steps_per_source_interval": physics.NATIVE_SUBSTEPS,
            "mj_step_calls": expected_steps,
            "simulated_seconds": float(context["data"].time),
            "contacts": context["contact_counts"],
            "max_ncon": context["max_ncon"],
            "max_contact_force": context["max_contact_force"],
            "max_abs_actuator_force": context["max_abs_actuator_force"],
            "max_abs_qvel": context["max_abs_qvel"],
        },
        "query_count": len(context["query_timings"]),
        "query_timings": context["query_timings"],
        "arrays": {name: str(out / f"{name}.npy") for name in arrays},
    }
    summary = {
        "mode": "mode4_state41_28dof_native",
        "model": args.model,
        "model_path": args.model_path,
        "row_indices": [context["row_index"]],
        "state_contract": STATE41_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "norm_stats_dir": str(args.norm_stats_dir),
        "norm_sha_expected": args.norm_sha_expected,
        "result": result,
    }
    (out.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return result


def run(args: argparse.Namespace) -> dict:
    if args.video_mode != "none":
        raise ValueError("batch state41 inference requires video-mode none; encode rows sequentially afterward")
    profile = base.L.resolve_profile(args.model)
    if profile.state_dim != STATE_DIM or profile.action_dim != ACTION_DIM:
        raise ValueError("state41 profile width mismatch")
    _, norm_sha = base.verify_locked_norm_stats(
        args.norm_stats_dir, expected_sha256=args.norm_sha_expected
    )
    norm_stats = base.L.normalize.load(args.norm_stats_dir)
    data_config = base.L._make_data_config(
        full.build_model_config(args.model),
        norm_stats,
        action_source=URDF_TARGET_ABSOLUTE,
        delta_mask_segments=profile.delta_mask_segments,
    )
    manifest_raw, manifest_entries = contact_windows.load_manifest(args.contact_window_manifest)
    if manifest_raw.get("dataset") not in (None, str(args.lance_dataset)):
        raise ValueError("contact manifest dataset mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts = [
        _initialize_context(args, row_index, data_config, manifest_entries)
        for row_index in args.row_indices_list
    ]
    headers = base.L._headers(args.api_key)
    session_id = base.create_session(args, headers)
    global_batch_count = 0
    global_observation_count = 0
    started = time.perf_counter()
    try:
        max_steps = max(context["frame_count"] - 1 for context in contexts)
        for relative_frame in range(max_steps):
            active = [
                context
                for context in contexts
                if relative_frame < context["frame_count"] - 1
            ]
            due: list[tuple[dict, dict]] = []
            for context in active:
                state, current_q = _observe(context)
                context["pending"] = (state, current_q)
                if relative_frame % args.chunk_stride == 0:
                    head, wrist = physics.render_current_state(
                        context["model"], context["data"], context["renderer"]
                    )
                    datum = base.build_datum(
                        context["row"],
                        frame=context["window_start"] + relative_frame,
                        state_input=state,
                        head_image=head,
                        wrist_image=wrist,
                        data_config=data_config,
                        base_model=args.model,
                        window_end=context["window_end"],
                    )
                    datum["data_config"] = data_config
                    due.append((context, datum))
            for group_start in range(0, len(due), args.act_batch_size):
                group = due[group_start : group_start + args.act_batch_size]
                results = base.query_action_group(
                    args=args,
                    headers=headers,
                    session_id=session_id,
                    datums=[datum for _context, datum in group],
                )
                global_batch_count += 1
                global_observation_count += len(group)
                for (context, _datum), (pred_norm, pred_phys, _gt_norm, timing) in zip(
                    group, results, strict=True
                ):
                    context["query_timings"].append(
                        {
                            "relative_frame": relative_frame,
                            "source_frame": context["window_start"] + relative_frame,
                            "batch_index": global_batch_count - 1,
                            **timing,
                        }
                    )
                    query_q = context["pending"][1]
                    context["candidates"].append(
                        {
                            "start": relative_frame,
                            "pred_norm": pred_norm,
                            "pred_phys": pred_phys,
                            "target_hand": single.reconstruct_absolute_target_chunk(
                                query_q, pred_phys[:HORIZON]
                            ),
                        }
                    )
            for context in active:
                context["chunk_stride"] = args.chunk_stride
                _record_action_and_step(context, relative_frame, args.temporal_decay)
    finally:
        try:
            base.delete_session(args, headers, session_id)
        finally:
            for context in contexts:
                context["renderer"].close()
    results = [_finalize_context(context, args.output_dir, args) for context in contexts]
    aggregate = {
        "mode": "mode4_state41_28dof_native_batch",
        "model": args.model,
        "model_path": args.model_path,
        "state_contract": STATE41_CONTRACT_ID,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": HORIZON,
        "row_indices": args.row_indices_list,
        "row_count": len(results),
        "act_batch_size": args.act_batch_size,
        "policy_batch_requests": global_batch_count,
        "policy_real_observations": global_observation_count,
        "norm_sha_expected": args.norm_sha_expected,
        "norm_sha_actual": norm_sha,
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return aggregate


def main() -> int:
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.action_source = URDF_TARGET_ABSOLUTE
    args.extended_state = True
    result = run(args)
    (args.output_dir / "batch_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
