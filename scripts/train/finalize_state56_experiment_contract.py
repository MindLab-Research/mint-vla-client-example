#!/usr/bin/env python3
"""Finalize a fail-closed State56 experiment data contract."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,uuid
from datetime import datetime,timezone
from pathlib import Path
from scripts import mano_state56_contract as C

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(r,*a):return subprocess.check_output(["git","-C",str(r),*a],text=True,stderr=subprocess.DEVNULL).strip()
def clean(r):
 if git(r,"status","--porcelain","--ignore-submodules=dirty"):raise RuntimeError(f"dirty repo {r}")
 return git(r,"rev-parse","HEAD")
def atomic(p,x):
 t=p.with_name("."+p.name+".tmp-"+uuid.uuid4().hex);t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
def main():
 ap=argparse.ArgumentParser()
 for n in ["source","source_release_verification","sidecar","sidecar_verification","train_selection","validation_selection","train_windows","validation_windows","norm_dir","norm_provenance","token_audit","client_root","mint_root","openpi_root","output"]:ap.add_argument("--"+n.replace("_","-"),type=Path,required=True)
 ap.add_argument("--task-id",required=True);ap.add_argument("--state-noise-std",type=float,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(f"refusing existing contract {a.output}")
 src=json.loads(a.source_release_verification.read_text());sv=json.loads(a.sidecar_verification.read_text());tr=json.loads(a.train_selection.read_text());va=json.loads(a.validation_selection.read_text());npv=json.loads(a.norm_provenance.read_text());tok=json.loads(a.token_audit.read_text())
 if src.get("release")!=str(a.source.resolve()) or (src.get("rows"),src.get("frames"))!=(5060,2811006):raise ValueError("source identity mismatch")
 if sv.get("status")!="passed" or Path(sv["path"]).resolve()!=a.sidecar.resolve() or sv.get("rows")!=4856:raise ValueError("sidecar identity mismatch")
 for x,name in [(tr,"train"),(va,"validation")]:
  if x.get("contract")!="mano_state56_native28_experiment_selection_v1" or x.get("split")!=name or x.get("dataset")!=str(a.source.resolve()):raise ValueError(f"{name} selection mismatch")
 if npv.get("contract")!="mano_state56_native28_trainonly_norm_v1" or npv.get("status")!="passed" or npv.get("task_id")!=a.task_id or npv.get("selection_sha256")!=sha(a.train_selection) or npv.get("norm_stats_sha256")!=sha(a.norm_dir/"norm_stats.json"):raise ValueError("norm provenance mismatch")
 if tok.get("contract")!="mano_state56_experiment_clean_stateaug_token_audit_v1" or tok.get("status")!="passed" or tok.get("task_id")!=a.task_id or tok.get("selection_sha256")!=sha(a.train_selection) or tok.get("trajectory_count")!=tr["selected_rows"] or tok.get("audited_active_frames")!=tr["active_frame_count"] or tok.get("norm_stats_sha256")!=npv["norm_stats_sha256"] or tok.get("overflow_count")!=0 or tok.get("zero_truncation") is not True:raise ValueError("token audit mismatch")
 aug=tok.get("augmentation") or {}
 if aug.get("seed")!=43 or float(aug.get("state_noise_std",-1))!=a.state_noise_std or float(aug.get("target_noise_std",-1))!=0.0 or aug.get("overflow_count")!=0 or aug.get("zero_truncation") is not True:raise ValueError("augmentation token audit mismatch")
 contract={"contract_id":"mano_state56_native28_experiment_data_v1","status":"accepted","created_at":datetime.now(timezone.utc).isoformat(),"task_id":a.task_id,"state_contract":C.STATE_CONTRACT_ID,"action_contract":C.ACTION_CONTRACT_ID,"profile_id":C.PROFILE_ID,"model":C.MODEL_ID,"state_dim":56,"action_dim":32,"action_physical_dim":28,"action_padding_dim":4,"action_horizon":10,"action_delta_mask":[3,-3,22,-4],"action_source":"urdf_target_absolute","max_token_len":256,"source_interval_seconds":.005,"contact_age_clip_seconds":1.0,"geometry_contract_sha256":C.GEOMETRY_CONTRACT_SHA256,"dataset":str(a.source.resolve()),"source_release_verification":str(a.source_release_verification.resolve()),"source_release_verification_sha256":sha(a.source_release_verification),"sidecar":str(a.sidecar.resolve()),"sidecar_verification":str(a.sidecar_verification.resolve()),"sidecar_verification_sha256":sha(a.sidecar_verification),"train_trajectory_count":int(tr["selected_rows"]),"validation_trajectory_count":int(va["selected_rows"]),"held_out_trajectory_count":0,"train_active_frame_count":int(tr["active_frame_count"]),"validation_active_frame_count":int(va["active_frame_count"]),"action_vector_count":int(tr["active_frame_count"])*10,"train_selection":str(a.train_selection.resolve()),"train_selection_sha256":sha(a.train_selection),"train_row_indices_sha256":tr["row_index_sha256"],"train_uuid_sha256":tr["uuid_sha256"],"validation_selection":str(a.validation_selection.resolve()),"validation_selection_sha256":sha(a.validation_selection),"validation_row_indices_sha256":va["row_index_sha256"],"validation_uuid_sha256":va["uuid_sha256"],"train_contact_windows":str(a.train_windows.resolve()),"train_contact_windows_sha256":sha(a.train_windows),"validation_contact_windows":str(a.validation_windows.resolve()),"validation_contact_windows_sha256":sha(a.validation_windows),"contact_context_frames":100,"missing_contact_policy":"error","norm_stats":str((a.norm_dir/"norm_stats.json").resolve()),"norm_stats_sha256":sha(a.norm_dir/"norm_stats.json"),"norm_provenance":str(a.norm_provenance.resolve()),"norm_provenance_sha256":sha(a.norm_provenance),"token_audit":str(a.token_audit.resolve()),"token_audit_sha256":sha(a.token_audit),"augmentation":{"state_noise_std":a.state_noise_std,"augmentation_seed":43,"target_noise_std":0.0,"qpos_dimensions":28,"tipXYZ_recomputed_under_noisy_qpos":True,"action_residual_compensation_dimensions":[0,1,2,*range(6,28)]},"prompt_template":"pick up the {object} using gesture {gesture}","population_grade":"A","initialization":"original pi0.5 Action-LoRA base","expected_trainable_count":13224992,"optimizer_steps":20000,"checkpoint_every_steps":4000,"checkpoint_steps":[4000,8000,12000,16000,20000],"training_status":"approved_not_started","client_commit":clean(a.client_root.resolve()),"mint_commit":clean(a.mint_root.resolve()),"openpi_commit":clean(a.openpi_root.resolve()),"builder_sha256":sha(Path(__file__).resolve())}
 atomic(a.output,contract);print(json.dumps({"task_id":a.task_id,"contract":str(a.output),"sha256":sha(a.output)},sort_keys=True))
if __name__=="__main__":main()
