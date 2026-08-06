#!/usr/bin/env python3
"""Authenticated batched State56/native28 closed-loop Mode4 evaluation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "egl"
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM") or "egl"

import lance
import mujoco
import numpy as np
import requests

import infer_mano_mode4 as base
import mode4_data_support as full
from scripts import contact_windows
from scripts import mano_state56_contract as contract
from scripts.eval import manorl_native28_mode4_physics as physics
from scripts.gesture_language import format_gesture_prompt
from scripts.openpi_profiles import ACTION_LORA_R16_STATE56_28DOF_MODEL, resolve_profile
from scripts.target_actions import URDF_TARGET_ABSOLUTE, project_row_actions

MODEL = ACTION_LORA_R16_STATE56_28DOF_MODEL
HORIZON = contract.ACTION_HORIZON


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="tml-dummy")
    parser.add_argument("--model", choices=(MODEL,), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--lance-dataset", type=Path, required=True)
    parser.add_argument("--source-release-verification", type=Path, required=True)
    parser.add_argument("--source-release-verification-sha256", required=True)
    parser.add_argument("--row-indices", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--norm-stats-dir", type=Path, required=True)
    parser.add_argument("--norm-sha-expected", required=True)
    parser.add_argument("--state56-data-contract", type=Path, required=True)
    parser.add_argument("--state56-data-contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-window-manifest", type=Path, required=True)
    parser.add_argument("--contact-window-manifest-sha256", required=True)
    parser.add_argument("--contact-context-frames", type=int, default=100)
    parser.add_argument("--chunk-stride", type=int, default=5)
    parser.add_argument("--temporal-decay", type=float, default=0.4)
    parser.add_argument("--act-batch-size", type=int, default=4)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--max-warm-request-seconds", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--training-client-commit", required=True)
    parser.add_argument("--evaluation-client-commit", required=True)
    parser.add_argument("--backend-commit", required=True)
    parser.add_argument("--model-commit", required=True)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.row_indices_list = base.parse_ordered_unique_csv(args.row_indices, option="--row-indices")
    if not args.row_indices_list:
        raise ValueError("at least one row is required")
    if not 1 <= args.chunk_stride < HORIZON:
        raise ValueError("chunk stride must be in [1,9]")
    if not 1 <= args.row_batch_size <= args.act_batch_size:
        raise ValueError("row batch size must be within action batch size")
    if args.act_batch_size != 4:
        raise ValueError("formal State56 evaluation requires fixed action batch4")
    if args.max_frames not in (0,) and args.max_frames < 2:
        raise ValueError("max frames must be zero or at least2")
    return args


def _verify_preflight(args: argparse.Namespace) -> tuple[dict, dict, Any, str]:
    for path, expected, name in (
        (args.source_release_verification, args.source_release_verification_sha256, "source verification"),
        (args.selection, args.selection_sha256, "selection"),
        (args.state56_data_contract, args.state56_data_contract_sha256, "data contract"),
        (args.contact_window_manifest, args.contact_window_manifest_sha256, "contact windows"),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name} SHA mismatch: {path}")
    source_verification = json.loads(args.source_release_verification.read_text())
    selection = json.loads(args.selection.read_text())
    data_contract = json.loads(args.state56_data_contract.read_text())
    if source_verification.get("contract") != "mano_28d_native_replay_state41_rgb_v1":
        raise ValueError("unsupported source release identity")
    if Path(source_verification.get("release", "")).resolve() != args.lance_dataset.resolve():
        raise ValueError("source verification release path mismatch")
    source_dataset = lance.dataset(str(args.lance_dataset))
    if source_dataset.count_rows() != int(source_verification.get("rows", -1)):
        raise ValueError("source release row count mismatch")
    if selection.get("contract") != "mano_state56_native28_experiment_selection_v1":
        raise ValueError("unsupported selection identity")
    expected_rows = [int(row["release_row_index"]) for row in selection.get("rows", [])]
    if expected_rows != args.row_indices_list or int(selection.get("selected_rows", -1)) != len(expected_rows):
        raise ValueError("requested rows do not exactly equal authenticated selection")
    if Path(selection.get("dataset", "")).resolve() != args.lance_dataset.resolve():
        raise ValueError("selection dataset mismatch")
    if selection.get("source_release_verification_sha256") != args.source_release_verification_sha256:
        raise ValueError("selection source verification mismatch")
    if data_contract.get("contract_id") != "mano_state56_native28_experiment_data_v1":
        raise ValueError("unsupported data contract")
    required = {
        "state_contract": contract.STATE_CONTRACT_ID,
        "action_contract": contract.ACTION_CONTRACT_ID,
        "model": MODEL,
        "state_dim": 56,
        "action_dim": 32,
        "action_physical_dim": 28,
        "action_padding_dim": 4,
        "action_horizon": 10,
        "action_delta_mask": [3, -3, 22, -4],
        "action_source": URDF_TARGET_ABSOLUTE,
        "norm_stats_sha256": args.norm_sha_expected,
        "source_release_verification_sha256": args.source_release_verification_sha256,
        "client_commit": args.training_client_commit,
        "mint_commit": args.backend_commit,
        "openpi_commit": args.model_commit,
    }
    for key, expected in required.items():
        if data_contract.get(key) != expected:
            raise ValueError(f"data contract {key} mismatch: {data_contract.get(key)!r} != {expected!r}")
    role = str(selection.get("split"))
    selection_key = "train_selection" if role == "train" else "validation_selection"
    windows_key = "train_contact_windows" if role == "train" else "validation_contact_windows"
    if data_contract.get(selection_key+"_sha256") != args.selection_sha256:
        raise ValueError("data contract selection mismatch")
    if data_contract.get(windows_key+"_sha256") != args.contact_window_manifest_sha256:
        raise ValueError("data contract contact-window mismatch")
    norm_path, norm_sha = contract.verify_locked_state56_norm_stats(
        args.norm_stats_dir,
        expected_sha256=args.norm_sha_expected,
        data_contract_path=args.state56_data_contract,
    )
    if norm_sha != args.norm_sha_expected:
        raise ValueError("verified norm SHA mismatch")
    metadata_path = args.model_path / "metadata.json"
    profile_path = args.model_path / "mint_pi05_profile.json"
    if not metadata_path.is_file() or not profile_path.is_file():
        raise ValueError("model path is not a complete sampler export")
    metadata = json.loads(metadata_path.read_text())
    profile_manifest = json.loads(profile_path.read_text())
    if metadata.get("step") != 20000 or metadata.get("model_name") != MODEL or metadata.get("checkpoint_type") != "sampler":
        raise ValueError("sampler metadata is not the requested State56 step20K model")
    profile_required = {"state_dim":56,"action_dim":32,"action_physical_dim":28,"action_horizon":10,"delta_mask_segments":[3,-3,22,-4],"profile_id":contract.PROFILE_ID}
    for key, expected in profile_required.items():
        if profile_manifest.get(key) != expected:
            raise ValueError(f"sampler profile {key} mismatch")
    profile = resolve_profile(MODEL)
    if profile.state_dim != 56 or profile.action_dim != 32 or profile.physical_action_dim != 28:
        raise ValueError("runtime State56 profile mismatch")
    norm_stats = base.L.normalize.load(args.norm_stats_dir)
    model_config = base.L._build_model_config(10, action_dim=32, base_model=profile.base_model, profile=profile)
    data_config = base.L._make_data_config(
        model_config,
        norm_stats,
        action_source=URDF_TARGET_ABSOLUTE,
        delta_mask_segments=profile.delta_mask_segments,
        physical_action_dim=profile.physical_action_dim,
    )
    return selection, data_contract, data_config, norm_sha


def _resolve_right_hand(row: dict) -> dict:
    hands = row.get("hands") or []
    metadata = row.get("trajectory_metadata") or {}
    slots = metadata.get("hand_slots")
    names = metadata.get("hand_names")
    if not isinstance(slots, list) or len(slots) != len(hands) or slots.count("right") != 1:
        raise ValueError("trajectory_metadata.hand_slots cannot resolve one right hand")
    if not isinstance(names, list) or "right" not in names:
        raise ValueError("trajectory_metadata.hand_names does not contain right")
    hand = hands[slots.index("right")]
    if not isinstance(hand, dict) or hand.get("hand_name") != "right":
        raise ValueError("resolved hand slot does not report hand_name=right")
    return hand


def _load_row(args: argparse.Namespace, row_index: int, selection_record: dict) -> dict:
    source = lance.dataset(str(args.lance_dataset))
    columns = ["state","actions","prompt","objects","timestamp","trajectory_metadata","episode_metadata","image","wrist_image","index","hands","row_payload_sha256"]
    row = source.take([row_index], columns=columns).to_pylist()[0]
    if row["index"]["uuid"] != selection_record["uuid"] or row["index"]["seed_uuid"] != selection_record["seed_uuid"]:
        raise ValueError(f"row identity mismatch for release row {row_index}")
    if row["index"]["object"] != selection_record["object"] or row["index"]["gesture"] != selection_record["gesture"]:
        raise ValueError(f"row population metadata mismatch for release row {row_index}")
    right = _resolve_right_hand(row)
    qpos = np.asarray(right["urdf_dof"], dtype=np.float32)
    source_state = np.asarray(row["state"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 28 or source_state.shape[0] != qpos.shape[0] or not np.array_equal(qpos, source_state[:, :28]):
        raise ValueError(f"source qpos mismatch row {row_index}")
    target = np.asarray(right["urdf_dof_target"], dtype=np.float32)
    if target.shape != qpos.shape or not np.isfinite(target).all():
        raise ValueError(f"source target mismatch row {row_index}")
    row = {**row, "hands":[right], "state":source_state}
    row = project_row_actions(row, URDF_TARGET_ABSOLUTE, action_dim=32, physical_dim=28)
    base_prompt = str(row["prompt"]).strip()
    suffix = f" using gesture {row['index']['gesture']}"
    prompt = base_prompt if base_prompt.endswith(suffix) else format_gesture_prompt(base_prompt, str(row["index"]["gesture"]))
    if prompt != selection_record["prompt"]:
        raise ValueError(f"prompt mismatch row {row_index}: {prompt!r}")
    return {**row, "prompt":prompt}


def _initialize_snapshot(context: dict) -> None:
    model, data = context["model"], context["data"]
    hand_qpos = context["hand_qpos_source"]
    object_positions = context["object_positions_source"]
    object_quaternions = context["object_quaternions_source"]
    target = context["target_source"]
    frame = context["window_start"]
    current_full = np.zeros(model.nq, dtype=np.float64)
    previous_full = np.zeros(model.nq, dtype=np.float64)
    current_full[context["hand_addrs"]] = hand_qpos[frame]
    oa = context["object_addr"]
    current_full[oa:oa+3] = object_positions[frame]
    current_full[oa+3:oa+7] = object_quaternions[frame] / np.linalg.norm(object_quaternions[frame])
    qvel = np.zeros(model.nv, dtype=np.float64)
    if frame > 0:
        previous_full[context["hand_addrs"]] = hand_qpos[frame-1]
        previous_full[oa:oa+3] = object_positions[frame-1]
        previous_full[oa+3:oa+7] = object_quaternions[frame-1] / np.linalg.norm(object_quaternions[frame-1])
        mujoco.mj_differentiatePos(model, qvel, contract.SOURCE_INTERVAL_SECONDS, previous_full, current_full)
    data.qpos[:] = current_full
    data.qvel[:] = qvel
    data.ctrl[:] = target[frame]
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise FloatingPointError("non-finite initial snapshot")


def _empty_arrays() -> dict[str,list]:
    return {name:[] for name in (
        "actions_raw_pred_normalized","actions_raw_pred_physical","actions_commanded_physical",
        "preclip_absolute_targets","servo_position_targets","servo_target_clipping_correction",
        "actions_applied_physical","rollout_observation_state","rollout_observation_contacts",
        "rollout_observation_lift","rollout_observation_tipxyz15","rollout_observation_log1p_force",
        "rollout_observation_relative_vertical_velocity","rollout_observation_contact_age",
        "physics_contact_flags","step_max_contact_force")}


def _initialize_context(args: argparse.Namespace, row_index: int, record: dict, data_config: Any, manifest_entries: dict) -> dict:
    row = _load_row(args,row_index,record)
    object_name = str(row["index"]["object"])
    window = full.resolve_row_window(row,row_index=row_index,frame_window="contact",contact_context_frames=args.contact_context_frames,missing_contact_policy="error",manifest_entry=manifest_entries.get(row_index))
    if window is None:
        raise ValueError(f"row {row_index} skipped by contact window")
    window_start,window_end=int(window.start_frame),int(window.end_frame)
    if args.max_frames:
        window_end=min(window_end,window_start+args.max_frames-1)
    frame_count=window_end-window_start+1
    if frame_count<2:
        raise ValueError(f"row {row_index} has fewer than2 evaluation frames")
    scene=physics.make_scene(object_name,args.width,args.height,physics=True,physics_timestep=physics.DT,create_renderer=True)
    tmp,model,data,renderer,object_addr,object_dof_addr,hand_addrs,hand_dof_addrs,limits=scene
    obj=row["objects"][0]
    context={"row_index":row_index,"record":record,"row":row,"object_name":object_name,"window_start":window_start,"window_end":window_end,"frame_count":frame_count,"tmp":tmp,"model":model,"data":data,"renderer":renderer,"object_addr":object_addr,"object_dof_addr":object_dof_addr,"hand_addrs":np.asarray(hand_addrs,dtype=np.int64),"hand_dof_addrs":np.asarray(hand_dof_addrs,dtype=np.int64),"limits":limits,"hand_qpos_source":np.asarray(row["hands"][0]["urdf_dof"],dtype=np.float64),"target_source":np.asarray(row["hands"][0]["urdf_dof_target"],dtype=np.float64),"object_positions_source":np.asarray(obj["pos"],dtype=np.float64),"object_quaternions_source":np.asarray(obj["quat_wxyz"],dtype=np.float64),"object_z_reference":float(np.asarray(obj["pos"])[0,2]),"tracker":contract.State56TemporalTracker(),"candidates":[],"pending":None,"arrays":_empty_arrays(),"query_timings":[],"contact_counts":{"hand_object":0,"object_floor":0,"hand_floor":0},"max_ncon":0,"max_contact_force":0.0,"max_abs_actuator_force":0.0,"max_abs_qvel":0.0}
    _initialize_snapshot(context)
    context["hand_states"]=[np.asarray(data.qpos[context["hand_addrs"]],dtype=np.float32).copy()]
    context["object_states"]=[np.asarray(data.qpos[object_addr:object_addr+3],dtype=np.float32).copy()]
    context["object_quaternions"]=[np.asarray(data.qpos[object_addr+3:object_addr+7],dtype=np.float32).copy()]
    return context


def _observe(context: dict) -> tuple[np.ndarray,np.ndarray]:
    data,model=context["data"],context["model"]
    current_q=np.asarray(data.qpos[context["hand_addrs"]],dtype=np.float64).copy()
    contacts,forces,_pairs=physics.state56_features_from_mujoco(model,data,context["object_name"])
    oa=context["object_addr"]
    object_position=np.asarray(data.qpos[oa:oa+3],dtype=np.float64)
    object_quaternion=np.asarray(data.qpos[oa+3:oa+7],dtype=np.float64)
    tips_world=contract.fingertip_world_from_mujoco(model,data)
    tips=contract.fingertips_in_collision_box_frame(tips_world,object_position,contract.quaternion_wxyz_to_matrix(object_quaternion),context["object_name"])
    velocity,age=context["tracker"].update(object_z=float(object_position[2]),palm_z=float(current_q[2]),finger_contacts=contacts)
    lift=float(object_position[2])-context["object_z_reference"]
    state=contract.build_state56(hand_qpos=current_q,finger_contacts=contacts,lift_height=lift,fingertip_collision_box_xyz=tips,finger_log1p_force=forces,relative_vertical_velocity=velocity,multifinger_contact_age=age)
    a=context["arrays"]
    a["rollout_observation_state"].append(state.copy());a["rollout_observation_contacts"].append(contacts.copy());a["rollout_observation_lift"].append(lift);a["rollout_observation_tipxyz15"].append(tips.reshape(-1).copy());a["rollout_observation_log1p_force"].append(forces.copy());a["rollout_observation_relative_vertical_velocity"].append(velocity);a["rollout_observation_contact_age"].append(age)
    return state,current_q


def _reconstruct_absolute_target_chunk(query_q: np.ndarray,pred_phys: np.ndarray)->np.ndarray:
    q=np.asarray(query_q,dtype=np.float32);pred=np.asarray(pred_phys,dtype=np.float32)
    if q.shape!=(28,) or pred.shape!=(10,32) or not np.isfinite(pred).all():
        raise ValueError(f"invalid State56 action reconstruction shapes {q.shape}/{pred.shape}")
    target=np.empty((10,28),dtype=np.float32)
    target[:,:3]=q[:3]+pred[:,:3];target[:,3:6]=pred[:,3:6];target[:,6:28]=q[6:28]+pred[:,6:28]
    if not np.isfinite(target).all():
        raise FloatingPointError("non-finite reconstructed State56 targets")
    return target


def _record_action_and_step(context: dict,relative_frame: int,temporal_decay: float)->None:
    candidates=[item for item in context["candidates"] if relative_frame<item["start"]+HORIZON];context["candidates"]=candidates
    if not candidates:raise RuntimeError(f"row {context['row_index']} has no action at frame {relative_frame}")
    newest_start=max(item["start"] for item in candidates)
    weights=np.asarray([temporal_decay**((newest_start-item["start"])//context["chunk_stride"]) for item in candidates],dtype=np.float64);weights/=weights.sum()
    absolute_target=sum(weight*item["target_hand"][relative_frame-item["start"]] for weight,item in zip(weights,candidates,strict=True))
    current_q=context["pending"][1]
    target,_clip=physics.nearest_wrapped_position_target(current_q,absolute_target-current_q,context["limits"])
    diagnostics=physics.step_servo(model=context["model"],data=context["data"],target=target,substeps=physics.NATIVE_SUBSTEPS,object_name=context["object_name"])
    after=np.asarray(context["data"].qpos[context["hand_addrs"]],dtype=np.float64).copy()
    for key in context["contact_counts"]:context["contact_counts"][key]+=int(diagnostics[f"{key}_contact"])
    context["max_ncon"]=max(context["max_ncon"],int(diagnostics["max_ncon"]));context["max_contact_force"]=max(context["max_contact_force"],float(diagnostics["max_contact_force"]));context["max_abs_actuator_force"]=max(context["max_abs_actuator_force"],float(diagnostics["max_abs_actuator_force"]));context["max_abs_qvel"]=max(context["max_abs_qvel"],float(np.max(np.abs(context["data"].qvel))))
    newest=max(candidates,key=lambda item:item["start"]);local=relative_frame-newest["start"]
    a=context["arrays"];a["actions_raw_pred_normalized"].append(newest["pred_norm"][local]);a["actions_raw_pred_physical"].append(newest["pred_phys"][local])
    commanded=np.zeros(32,dtype=np.float32);commanded[:28]=absolute_target-current_q;a["actions_commanded_physical"].append(commanded);a["preclip_absolute_targets"].append(np.asarray(absolute_target,dtype=np.float32));a["servo_position_targets"].append(np.asarray(target,dtype=np.float32));a["servo_target_clipping_correction"].append(np.asarray(target-absolute_target,dtype=np.float32))
    applied=np.zeros(32,dtype=np.float32);applied[:28]=after-current_q;a["actions_applied_physical"].append(applied);a["physics_contact_flags"].append([diagnostics["hand_object_contact"],diagnostics["object_floor_contact"],diagnostics["hand_floor_contact"]]);a["step_max_contact_force"].append(diagnostics["max_contact_force"])
    context["hand_states"].append(after.astype(np.float32));oa=context["object_addr"];context["object_states"].append(np.asarray(context["data"].qpos[oa:oa+3],dtype=np.float32).copy());context["object_quaternions"].append(np.asarray(context["data"].qpos[oa+3:oa+7],dtype=np.float32).copy());context["pending"]=None


def _finalize_context(context: dict,output_root: Path,args: argparse.Namespace)->dict:
    out=output_root/"rows"/f"row_{context['row_index']}"/"artifacts"/"mode4";out.mkdir(parents=True,exist_ok=True)
    arrays={name:np.asarray(values) for name,values in context["arrays"].items()};arrays["hand_state_sim"]=np.asarray(context["hand_states"],dtype=np.float32);arrays["object_position_sim"]=np.asarray(context["object_states"],dtype=np.float32);arrays["object_quaternion_sim"]=np.asarray(context["object_quaternions"],dtype=np.float32)
    if arrays["rollout_observation_state"].shape!=(context["frame_count"]-1,56):raise RuntimeError("State56 rollout observation shape mismatch")
    for name,value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():raise FloatingPointError(f"row {context['row_index']} non-finite {name}")
        np.save(out/f"{name}.npy",value)
    expected_steps=physics.NATIVE_SUBSTEPS*(context["frame_count"]-1)
    expected_seconds = expected_steps * physics.DT
    if not np.isclose(context["data"].time, expected_seconds, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"row {context['row_index']} MuJoCo time mismatch")
    result={"mode":"mode4_state56_native28_snapshot_qvel","row_index":context["row_index"],"uuid":context["record"]["uuid"],"seed_uuid":context["record"]["seed_uuid"],"object_name":context["object_name"],"gesture":context["record"]["gesture"],"prompt":context["row"]["prompt"],"object_z_reference":context["object_z_reference"],"state_contract":contract.STATE_CONTRACT_ID,"state_dim":56,"action_contract":contract.ACTION_CONTRACT_ID,"action_dim":32,"action_physical_dim":28,"action_horizon":10,"frame_window":{"start_frame":context["window_start"],"end_frame":context["window_end"],"frame_count":context["frame_count"],"context_frames":args.contact_context_frames,"manifest":str(args.contact_window_manifest)},"initialization":"source-window snapshot pose + backward-difference5ms qvel + current source target","physics":{"engine":"native MuJoCo","controller":"manorl_native_position_servo","timestep_seconds":physics.DT,"steps_per_source_interval":physics.NATIVE_SUBSTEPS,"mj_step_calls":expected_steps,"simulated_seconds":float(context["data"].time),"contacts":context["contact_counts"],"max_ncon":context["max_ncon"],"max_contact_force":context["max_contact_force"],"max_abs_actuator_force":context["max_abs_actuator_force"],"max_abs_qvel":context["max_abs_qvel"]},"query_count":len(context["query_timings"]),"query_timings":context["query_timings"],"arrays":{name:str(out/f"{name}.npy") for name in arrays}}
    (out.parent/"summary.json").write_text(json.dumps({"status":"completed","result":result},indent=2)+"\n")
    return result


def _write_progress(path: Path,payload: dict)->None:
    tmp=path.with_name("."+path.name+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");tmp.replace(path)


def _delete_session_verified(args: argparse.Namespace, headers: dict[str, str], session_id: str) -> None:
    response = requests.delete(
        f"{args.base_url}/api/v1/mint/action_sessions/{session_id}",
        headers=headers,
        timeout=120.0,
    )
    if response.status_code != 200:
        raise RuntimeError(f"action-session deletion failed: HTTP {response.status_code}: {response.text[:500]}")


def run(args: argparse.Namespace)->dict:
    selection,data_contract,data_config,norm_sha=_verify_preflight(args)
    manifest_raw,manifest_entries=contact_windows.load_manifest(args.contact_window_manifest)
    if manifest_raw.get("dataset") not in (None,str(args.lance_dataset)):raise ValueError("contact manifest dataset mismatch")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    progress_path=args.output_dir/"progress.json";headers=base.L._headers(args.api_key);session_id=None;global_batch_count=0;global_observation_count=0;global_padding_count=0;results=[];completed=[];started=time.perf_counter();records={int(x["release_row_index"]):x for x in selection["rows"]}
    _write_progress(progress_path,{"status":"running","row_count":len(args.row_indices_list),"completed_row_count":0,"policy_batch_requests":0,"elapsed_seconds":0.0})
    try:
        session_id=base.create_session(args,headers)
        for row_batch_index,row_start in enumerate(range(0,len(args.row_indices_list),args.row_batch_size)):
            row_indices=args.row_indices_list[row_start:row_start+args.row_batch_size];contexts=[]
            with ExitStack() as stack:
                for row_index in row_indices:
                    c=_initialize_context(args,row_index,records[row_index],data_config,manifest_entries);contexts.append(c);stack.callback(c["renderer"].close)
                    if c["tmp"] is not None:
                        stack.callback(c["tmp"].cleanup)
                max_steps=max(c["frame_count"]-1 for c in contexts)
                for relative_frame in range(max_steps):
                    active=[c for c in contexts if relative_frame<c["frame_count"]-1];due=[]
                    for c in active:
                        state,current_q=_observe(c);c["pending"]=(state,current_q)
                        if relative_frame%args.chunk_stride==0:
                            head,wrist=physics.render_current_state(c["model"],c["data"],c["renderer"]);datum=base.build_datum(c["row"],frame=c["window_start"]+relative_frame,state_input=state,head_image=head,wrist_image=wrist,data_config=data_config,base_model=args.model,window_end=c["window_end"]);datum["data_config"]=data_config;due.append((c,datum))
                    for group_start in range(0,len(due),args.act_batch_size):
                        group=due[group_start:group_start+args.act_batch_size];group_results=base.query_action_group(args=args,headers=headers,session_id=session_id,datums=[d for _c,d in group]);global_batch_count+=1;global_observation_count+=len(group);global_padding_count+=args.act_batch_size-len(group)
                        for (c,_datum),(pred_norm,pred_phys,_gt_norm,timing) in zip(group,group_results,strict=True):
                            if global_batch_count>=2 and args.max_warm_request_seconds>0 and float(timing.get("wall_seconds",0))>args.max_warm_request_seconds:raise TimeoutError(f"warm action request exceeded gate: {timing.get('wall_seconds')}s")
                            c["query_timings"].append({"relative_frame":relative_frame,"source_frame":c["window_start"]+relative_frame,"batch_index":global_batch_count-1,**timing});query_q=c["pending"][1];c["candidates"].append({"start":relative_frame,"pred_norm":pred_norm,"pred_phys":pred_phys,"target_hand":_reconstruct_absolute_target_chunk(query_q,pred_phys[:HORIZON])})
                    for c in active:c["chunk_stride"]=args.chunk_stride;_record_action_and_step(c,relative_frame,args.temporal_decay)
                results.extend(_finalize_context(c,args.output_dir,args) for c in contexts)
            completed.extend(row_indices);_write_progress(progress_path,{"status":"running","row_count":len(args.row_indices_list),"last_completed_row_batch_index":row_batch_index,"completed_row_count":len(completed),"completed_row_indices":completed,"policy_batch_requests":global_batch_count,"policy_real_observations":global_observation_count,"transport_padding_observations":global_padding_count,"elapsed_seconds":time.perf_counter()-started})
    except BaseException as exc:
        _write_progress(progress_path,{"status":"failed","completed_row_count":len(completed),"completed_row_indices":completed,"policy_batch_requests":global_batch_count,"elapsed_seconds":time.perf_counter()-started,"error_type":type(exc).__name__,"error":str(exc)});raise
    finally:
        if session_id is not None:
            _delete_session_verified(args, headers, session_id)
    aggregate={"contract":"mano_state56_native28_20k_mode4_batch_v1","status":"completed","model":args.model,"model_path":str(args.model_path.resolve()),"model_metadata_sha256":sha256(args.model_path/"metadata.json"),"model_profile_sha256":sha256(args.model_path/"mint_pi05_profile.json"),"checkpoint_step":20000,"state_contract":contract.STATE_CONTRACT_ID,"state_dim":56,"action_contract":contract.ACTION_CONTRACT_ID,"action_dim":32,"action_physical_dim":28,"row_indices":args.row_indices_list,"row_count":len(results),"population_role":selection["split"],"selection":str(args.selection.resolve()),"selection_sha256":args.selection_sha256,"data_contract":str(args.state56_data_contract.resolve()),"data_contract_sha256":args.state56_data_contract_sha256,"norm_sha256":norm_sha,"source_release_verification_sha256":args.source_release_verification_sha256,"contact_window_manifest_sha256":args.contact_window_manifest_sha256,"training_client_commit":args.training_client_commit,"evaluation_client_commit":args.evaluation_client_commit,"backend_commit":args.backend_commit,"model_commit":args.model_commit,"initialization":"native28 source-window snapshot-qvel","row_batch_size":args.row_batch_size,"act_batch_size":args.act_batch_size,"policy_batch_requests":global_batch_count,"policy_real_observations":global_observation_count,"transport_padding_observations":global_padding_count,"all_rows_complete":True,"all_arrays_finite":True,"action_session_released":True,"results":results,"elapsed_seconds":time.perf_counter()-started}
    (args.output_dir/"summary.json").write_text(json.dumps(aggregate,indent=2)+"\n");_write_progress(progress_path,{"status":"completed","row_count":len(results),"completed_row_count":len(completed),"completed_row_indices":completed,"policy_batch_requests":global_batch_count,"policy_real_observations":global_observation_count,"transport_padding_observations":global_padding_count,"elapsed_seconds":aggregate["elapsed_seconds"],"summary":str(args.output_dir/"summary.json")});return aggregate


def main()->int:
    args=parse_args();result=run(args);print(json.dumps({"status":result["status"],"row_count":result["row_count"],"elapsed_seconds":result["elapsed_seconds"]},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
