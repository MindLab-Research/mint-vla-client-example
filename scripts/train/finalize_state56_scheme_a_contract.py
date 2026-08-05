#!/usr/bin/env python3
"""Finalize the fail-closed State56 Scheme-A data/token/norm contract."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess,uuid
from scripts import mano_state56_contract as C

def sha256(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def git(r:Path,*a:str)->str:return subprocess.check_output(['git','-C',str(r),*a],text=True,stderr=subprocess.DEVNULL).strip()
def clean_commit(r:Path)->str:
 if git(r,'status','--porcelain','--ignore-submodules=dirty'):raise RuntimeError(f'repository is dirty: {r}')
 return git(r,'rev-parse','HEAD')
def load(p:Path):return json.loads(p.read_text())
def atomic_json(p:Path,x:dict):
 t=p.with_name(f'.{p.name}.tmp-{uuid.uuid4().hex}');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def main():
 q=argparse.ArgumentParser(description=__doc__)
 for name in ['source_dataset','source_release_verification','sidecar','sidecar_verification','train_selection','validation_selection','train_windows','validation_windows','norm_dir','norm_provenance','token_audit','client_root','mint_root','openpi_root','output']:q.add_argument('--'+name.replace('_','-'),type=Path,required=True)
 a=q.parse_args();
 if a.output.exists():raise FileExistsError(f'refusing existing State56 data contract: {a.output}')
 src=load(a.source_release_verification);side=load(a.sidecar_verification);tr=load(a.train_selection);va=load(a.validation_selection);npv=load(a.norm_provenance);tok=load(a.token_audit);norm=a.norm_dir/'norm_stats.json'
 if src.get('contract')!='mano_28d_native_replay_state41_rgb_v1':raise ValueError('source release verification mismatch')
 if Path(src.get('release','')).resolve()!=a.source_dataset.resolve() or (src.get('rows'),src.get('frames'))!=(5060,2811006):raise ValueError('source release path/population mismatch')
 if side.get('contract')!='mano_state56_native28_virtual_sidecar_v1' or side.get('status')!='passed':raise ValueError('sidecar verification mismatch')
 if Path(side.get('path','')).resolve()!=a.sidecar.resolve() or (side.get('rows'),side.get('train_rows'),side.get('validation_rows'),side.get('held_out_rows'),side.get('frames'))!=(4856,4613,243,0,2695132):raise ValueError('sidecar Scheme-A population mismatch')
 if len(tr.get('rows',[]))!=4613 or len(va.get('rows',[]))!=243 or tr.get('population_rows')!=4856 or va.get('population_rows')!=4856:raise ValueError('Scheme-A selection mismatch')
 if npv.get('contract')!='mano_state56_scheme_a_trainonly_norm_v1' or npv.get('status')!='passed' or npv.get('norm_stats_sha256')!=sha256(norm):raise ValueError('State56 norm provenance mismatch')
 if tok.get('contract')!='mano_state56_scheme_a_clean_stateaug005_token_audit_v1' or tok.get('status')!='passed' or tok.get('zero_truncation') is not True or tok.get('overflow_count')!=0 or tok.get('audited_active_frames')!=2560614 or tok.get('maximum_token_length',999)>256:raise ValueError('State56 clean token audit mismatch')
 aug=tok.get('augmentation') or {}
 if aug.get('seed')!=43 or aug.get('state_noise_std')!=.05 or aug.get('target_noise_std')!=0.0 or aug.get('zero_truncation') is not True or aug.get('overflow_count')!=0 or aug.get('maximum_token_length',999)>256 or aug.get('action_compensation_dimensions')!=[0,1,2,*range(6,28)]:raise ValueError('State56 augmented token audit mismatch')
 client=clean_commit(a.client_root.resolve());mint=clean_commit(a.mint_root.resolve());openpi=clean_commit(a.openpi_root.resolve())
 contract={
  'contract_id':'mano_state56_native28_scheme_a_data_v1','status':'accepted','created_at':datetime.now(timezone.utc).isoformat(),
  'state_contract':C.STATE_CONTRACT_ID,'action_contract':C.ACTION_CONTRACT_ID,'profile_id':C.PROFILE_ID,'model':C.MODEL_ID,
  'state_dim':56,'state_layout':{'qpos28':[0,28],'contact5':[28,33],'source_frame0_lift':[33,34],'object_frame_tipXYZ15':[34,49],'log1p_force5':[49,54],'object_minus_palm_vertical_velocity':[54,55],'window_local_multicontact_age':[55,56]},
  'action_dim':32,'action_physical_dim':28,'action_padding_dim':4,'action_horizon':10,'action_delta_mask':[3,-3,22,-4],'action_source':'urdf_target_absolute','action_layout':'target28+pad4','max_token_len':256,
  'source_interval_seconds':.005,'contact_age_clip_seconds':1.0,'force_reference_newtons':50.0,'geometry_contract_sha256':C.GEOMETRY_CONTRACT_SHA256,
  'population_grade':'A','population_trajectory_count':4856,'train_trajectory_count':4613,'validation_trajectory_count':243,'held_out_trajectory_count':0,'train_active_frame_count':2560614,'validation_active_frame_count':134518,'active_frame_count':2560614,'action_vector_count':25606140,
  'split_scheme':'A: exact existing object×gesture stratified seed42 train/validation manifests','validation_boundary':'already used for checkpoint selection; not an untouched final test','validation_seed_leakage':{'cross_split_seed_uuid_groups':216,'validation_rows_sharing_train_seed_uuid':239,'validation_rows':243,'unseen_seed_generalization_claim_allowed':False},
  'dataset':str(a.source_dataset.resolve()),'source_dataset_version':16,'source_release_verification':str(a.source_release_verification.resolve()),'source_release_verification_sha256':sha256(a.source_release_verification),
  'sidecar':str(a.sidecar.resolve()),'sidecar_verification':str(a.sidecar_verification.resolve()),'sidecar_verification_sha256':sha256(a.sidecar_verification),'sidecar_plan_sha256':side['plan_sha256'],
  'train_selection':str(a.train_selection.resolve()),'train_selection_sha256':sha256(a.train_selection),'train_row_indices_sha256':tr['row_index_sha256'],'train_uuid_sha256':tr['uuid_sha256'],
  'validation_selection':str(a.validation_selection.resolve()),'validation_selection_sha256':sha256(a.validation_selection),'validation_row_indices_sha256':va['row_index_sha256'],'validation_uuid_sha256':va['uuid_sha256'],
  'train_contact_windows':str(a.train_windows.resolve()),'train_contact_windows_sha256':sha256(a.train_windows),'validation_contact_windows':str(a.validation_windows.resolve()),'validation_contact_windows_sha256':sha256(a.validation_windows),'contact_context_frames':100,'missing_contact_policy':'error','observed_full_window_rows':4856,
  'norm_population':'Scheme-A train-only contact windows','norm_stats':str(norm.resolve()),'norm_stats_sha256':sha256(norm),'norm_provenance':str(a.norm_provenance.resolve()),'norm_provenance_sha256':sha256(a.norm_provenance),
  'token_audit':str(a.token_audit.resolve()),'token_audit_sha256':sha256(a.token_audit),'tokenizer_model_sha256':'8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6','token_capacity_constructive_bound':245,'token_capacity_audit_sha256':'431402c671bc9d8b27f6f51e40cdcdc85b105e94df4d465175a7ce43818b8972',
  'prompt_template':'pick up the {object} using gesture {gesture}','prompt_source':'State41 formal release metadata','objects':['banana','bowl','cube1','cube2','cylinder3','cylinder4','cylinder7','iphone','largeclamp','mayonnaisebottle','powerdrill'],'gesture_count':13,'object_gesture_strata':53,
  'augmentation':{'state_noise_std':.05,'augmentation_seed':43,'target_noise_std':0.0,'qpos_dimensions':28,'tipXYZ_recomputed_under_noisy_qpos':True,'action_residual_compensation_dimensions':[0,1,2,*range(6,28)],'preserved_clean_features':['contact5','lift1','force5','relative_velocity1','contact_age1']},
  'initialization':'original pi0.5 Action-LoRA base weights; no State41 sampler reuse','expected_trainable_count':13224992,'training_status':'not_started_requires_separate_user_approval',
  'client_commit':client,'mint_commit':mint,'openpi_commit':openpi,'contract_builder_sha256':sha256(Path(__file__).resolve()),
 }
 atomic_json(a.output,contract);print(json.dumps({'data_contract':str(a.output),'sha256':sha256(a.output),'norm_sha256':contract['norm_stats_sha256'],'token_audit_sha256':contract['token_audit_sha256']},sort_keys=True))
if __name__=='__main__':main()
