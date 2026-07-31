#!/usr/bin/env python3
"""Finalize data_contract.json after the exact state54 norm and token audit exist."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path

from scripts.gesture_language import GestureIndex
from scripts.mano_state54_contract import (
    CONTACT_AGE_CLIP_SECONDS, FORCE_REFERENCE_NEWTONS, POPULATION_ACTIVE_FRAMES,
    POPULATION_ROW_INDICES_SHA256, POPULATION_TRAJECTORIES, PROFILE_MAX_TOKEN_LEN,
    SOURCE_INTERVAL_SECONDS, STATE54_NORM_SHA256, STATE_CONTRACT_ID,
)

def git_head_clean(path: Path) -> str:
    if subprocess.check_output(["git","-C",str(path),"status","--porcelain"],text=True).strip():
        raise ValueError(f"source repo must be clean before contract finalization: {path}")
    return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--norm-dir",type=Path,required=True);ap.add_argument("--client-repo",type=Path,required=True);ap.add_argument("--mint-repo",type=Path,required=True);ap.add_argument("--openpi-repo",type=Path,required=True);ap.add_argument("--gesture-index",type=Path,required=True);ap.add_argument("--dataset",type=Path,required=True);ap.add_argument("--contact-window-manifest",type=Path,required=True);args=ap.parse_args()
    norm=args.norm_dir/"norm_stats.json";audit=args.norm_dir/"token_audit.json"
    norm_sha=hashlib.sha256(norm.read_bytes()).hexdigest();audit_sha=hashlib.sha256(audit.read_bytes()).hexdigest()
    if norm_sha!=STATE54_NORM_SHA256:raise ValueError(f"norm SHA {norm_sha} is not state54 allowlist")
    token=json.loads(audit.read_text())
    if token.get("zero_truncation") is not True or token.get("overflow_count")!=0:raise ValueError("token audit did not prove zero truncation")
    gesture=GestureIndex.load(args.gesture_index)
    contract={
      "contract_version":1,"state_contract":STATE_CONTRACT_ID,"state_dim":54,"action_dim":32,"action_horizon":10,
      "action_source":"urdf_target_absolute","frame_window":"contact","contact_context_frames":100,
      "row_indices_sha256":POPULATION_ROW_INDICES_SHA256,"trajectory_count":POPULATION_TRAJECTORIES,
      "active_frame_count":POPULATION_ACTIVE_FRAMES,"action_vector_count":POPULATION_ACTIVE_FRAMES*10,
      "norm_stats_sha256":norm_sha,"token_audit_sha256":audit_sha,"max_token_len":PROFILE_MAX_TOKEN_LEN,
      "observed_max_token_length":token["maximum_token_length"],"zero_truncation":True,
      "force_aggregation":"sum_norm_each_contact_pairs.force_normal_then_log1p_v1",
      "force_reference_newtons":FORCE_REFERENCE_NEWTONS,"source_interval_seconds":SOURCE_INTERVAL_SECONDS,
      "contact_age_rule":"consecutive_at_least_two_fingers_window_local_clipped_v1",
      "contact_age_clip_seconds":CONTACT_AGE_CLIP_SECONDS,
      "lift_baseline":"source_frame_0_object_body_z_v1","palm_vertical_position":"hand_qpos_root_z_v1",
      "fingertip_frame":"object_mesh_aabb_center_half_extents_v1","finger_order":["index","thumb","ring","middle","pinky"],
      "gesture_index_sha256":gesture.sha256,"dataset":str(args.dataset.resolve()),
      "contact_window_manifest":str(args.contact_window_manifest.resolve()),
      "client_commit":git_head_clean(args.client_repo),"mint_commit":git_head_clean(args.mint_repo),"openpi_commit":git_head_clean(args.openpi_repo),
    }
    dst=args.norm_dir/"data_contract.json";tmp=dst.with_suffix(".json.tmp");tmp.write_text(json.dumps(contract,indent=2,sort_keys=True)+"\n");tmp.replace(dst);print(json.dumps(contract,sort_keys=True))
if __name__=="__main__":main()
