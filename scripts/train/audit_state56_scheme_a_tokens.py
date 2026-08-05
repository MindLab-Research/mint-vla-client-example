#!/usr/bin/env python3
"""Exhaustively audit clean and deterministic StateAug0.05 State56 tokens."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import lance
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from scripts import mano_state56_contract as C
from scripts.eval.manorl_native28_physics import Native28FingertipFK
from scripts.state56_virtual_data import State56SidecarStore

AUDIT_CONTRACT="mano_state56_scheme_a_clean_stateaug005_token_audit_v1"


def sha256(path: Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path,payload: dict)->None:
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(temp,path)


def git(root: Path,*args: str)->str:return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()


def record_length(tokenizer: PaligemmaTokenizer,prompt: str,state: np.ndarray)->int:
    _tokens,mask=tokenizer.tokenize(prompt,state)
    length=int(np.count_nonzero(mask))
    if length>=4096:raise ValueError('token audit ceiling reached')
    return length


def update_max(current: int,examples: list[dict],value: int,example: dict)->tuple[int,list[dict]]:
    if value>current:return value,[example]
    if value==current and len(examples)<20:return current,[*examples,example]
    return current,examples


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dataset',type=Path,required=True)
    parser.add_argument('--sidecar',type=Path,required=True)
    parser.add_argument('--sidecar-verification',type=Path,required=True)
    parser.add_argument('--sidecar-verification-sha256',required=True)
    parser.add_argument('--norm-dir',type=Path,required=True)
    parser.add_argument('--norm-provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--state-noise-std',type=float,default=.05)
    parser.add_argument('--augmentation-seed',type=int,default=43)
    parser.add_argument('--max-token-len',type=int,default=256)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'refusing existing token audit: {args.output}')
    if args.state_noise_std!=.05 or args.augmentation_seed!=43 or args.max_token_len!=256:
        raise ValueError('State56 formal token audit requires sigma0.05 seed43 max256')
    client_root=Path(__file__).resolve().parents[2]
    if git(client_root,'status','--porcelain','--ignore-submodules=dirty'):
        raise RuntimeError('State56 token audit requires a clean client checkout')
    source_path=args.source_dataset.expanduser().resolve();sidecar_path=args.sidecar.expanduser().resolve()
    store=State56SidecarStore(sidecar_path,verification_path=args.sidecar_verification,expected_verification_sha256=args.sidecar_verification_sha256,source_dataset=source_path)
    norm_provenance=json.loads(args.norm_provenance.read_text())
    norm_path=args.norm_dir/'norm_stats.json';norm_sha=sha256(norm_path)
    if norm_provenance.get('norm_stats_sha256')!=norm_sha or norm_provenance.get('train_active_frame_count')!=2560614:
        raise ValueError('State56 norm provenance mismatch')
    stats=normalize.load(args.norm_dir);q01=np.asarray(stats['state'].q01,dtype=np.float32);q99=np.asarray(stats['state'].q99,dtype=np.float32)
    aq01=np.asarray(stats['actions'].q01,dtype=np.float32);aq99=np.asarray(stats['actions'].q99,dtype=np.float32)
    if q01.shape!=(56,) or q99.shape!=(56,) or aq01.shape!=(32,) or aq99.shape!=(32,):raise ValueError('State56 norm widths mismatch')
    qpos_valid=(q99[:28]-q01[:28])>1e-6
    state_range=q99[:32]-q01[:32];action_range=aq99-aq01
    compensation_scale=np.where((state_range>1e-6)&(action_range>1e-6),state_range/np.maximum(action_range,1e-8),0.0)
    delta_dims=np.asarray([0,1,2,*range(6,28)],dtype=np.int64)
    tokenizer=PaligemmaTokenizer(max_len=4096)
    fk=Native28FingertipFK();rng=np.random.default_rng(43)
    source=lance.dataset(str(source_path),version=store.source_dataset_version)
    light=store.dataset.to_table(columns=['index','window']).to_pylist()
    train_positions=sorted((int(row['index']['release_row_index']),position) for position,row in enumerate(light) if row['index']['split']=='train')
    if len(train_positions)!=4613:raise ValueError('State56 token train-row count mismatch')
    clean_hist={};aug_hist={};clean_max=aug_max=-1;clean_examples=[];aug_examples=[]
    total=clean_overflow=aug_overflow=0;noise_sq=0.0;noise_count=0;correction_sq=np.zeros(32,dtype=np.float64)
    started=time.monotonic();progress=args.output.with_name(args.output.stem+'.progress.json')
    for ordinal,(source_row,position) in enumerate(train_positions):
        side=store.dataset.take([position],columns=['index','state','prompt','provenance']).to_pylist()[0]
        raw=source.take([source_row],columns=['index','objects','row_payload_sha256']).to_pylist()[0]
        if raw['index']['uuid']!=side['index']['uuid'] or raw['row_payload_sha256']!=side['provenance']['source_row_payload_sha256']:
            raise ValueError(f'State56 token source join mismatch row{source_row}')
        states=np.asarray(side['state'],dtype=np.float32);normalized=(states-q01)/(q99-q01+1e-6)*2.0-1.0
        if not np.all(np.isfinite(normalized)):raise ValueError(f'non-finite normalized State56 row{source_row}')
        prompt=side['prompt'];obj=raw['objects'][0];positions=np.asarray(obj['pos'],dtype=np.float32);quaternions=np.asarray(obj['quat_wxyz'],dtype=np.float32);object_name=side['index']['object']
        for frame,state in enumerate(normalized):
            clean_length=record_length(tokenizer,prompt,state);clean_hist[clean_length]=clean_hist.get(clean_length,0)+1;clean_overflow+=int(clean_length>256)
            example={'release_row_index':source_row,'source_frame':frame,'prompt':prompt,'token_length':clean_length}
            clean_max,clean_examples=update_max(clean_max,clean_examples,clean_length,example)
            noise=rng.normal(0.0,.05,size=32).astype(np.float32);augmented=state.copy();augmented[:28][qpos_valid]+=noise[:28][qpos_valid]
            valid_noise=noise[:28][qpos_valid];noise_sq+=float(np.sum(np.square(valid_noise)));noise_count+=valid_noise.size
            raw_qpos=q01[:28]+(augmented[:28]+1.0)*.5*(q99[:28]-q01[:28]+1e-6)
            tip_world=fk(object_name=object_name,hand_qpos=raw_qpos,object_position=positions[frame],object_quaternion_wxyz=quaternions[frame])
            tip_box=C.fingertips_in_collision_box_frame(tip_world,positions[frame],C.quaternion_wxyz_to_matrix(quaternions[frame]),object_name).reshape(-1)
            augmented[34:49]=((tip_box-q01[34:49])/(q99[34:49]-q01[34:49]+1e-6)*2.0-1.0).astype(np.float32)
            aug_length=record_length(tokenizer,prompt,augmented);aug_hist[aug_length]=aug_hist.get(aug_length,0)+1;aug_overflow+=int(aug_length>256)
            aug_example={**example,'token_length':aug_length};aug_max,aug_examples=update_max(aug_max,aug_examples,aug_length,aug_example)
            correction=np.zeros(32,dtype=np.float32);correction[delta_dims]=-noise[delta_dims]*compensation_scale[delta_dims];correction_sq+=np.square(correction,dtype=np.float64)
            total+=1
        if ordinal%10==0 or ordinal+1==len(train_positions):
            atomic_json(progress,{'completed_rows':ordinal+1,'total_rows':4613,'audited_frames':total,'clean_current_max':clean_max,'augmented_current_max':aug_max,'clean_overflow_count':clean_overflow,'augmented_overflow_count':aug_overflow,'elapsed_seconds':time.monotonic()-started})
    if total!=2560614:raise ValueError(f'State56 token frame count mismatch: {total}')
    result={
        'contract':AUDIT_CONTRACT,'status':'passed','created_at':datetime.now(timezone.utc).isoformat(),
        'state_contract':C.STATE_CONTRACT_ID,'action_contract':C.ACTION_CONTRACT_ID,
        'population':'Scheme-A train-only contact window','trajectory_count':4613,'audited_active_frames':total,
        'profile_max_token_len':256,'minimum_token_length':min(clean_hist),'maximum_token_length':clean_max,
        'headroom_at_maximum':256-clean_max,'overflow_count':clean_overflow,'zero_truncation':clean_overflow==0,
        'token_length_histogram':{str(k):clean_hist[k] for k in sorted(clean_hist)},'maximum_examples':clean_examples,
        'augmentation':{
            'contract':'normalized_qpos28_sigma005_tipFK_action_residual28_v1','seed':43,'state_noise_std':.05,'target_noise_std':0.0,
            'samples':total,'realized_sigma':float(np.sqrt(noise_sq/noise_count)),'qpos_valid_dimensions':np.flatnonzero(qpos_valid).tolist(),
            'causal_recomputation':'qpos28 noise then pinned native28 MuJoCo FK tipXYZ15; contact/lift/force/velocity/age clean',
            'action_compensation_dimensions':delta_dims.tolist(),'action_compensation_rule':'normalized residual -= qpos_noise * state_quantile_range/action_quantile_range',
            'action_compensation_rms_by_dim':np.sqrt(correction_sq/total).tolist(),
            'minimum_token_length':min(aug_hist),'maximum_token_length':aug_max,'headroom_at_maximum':256-aug_max,
            'overflow_count':aug_overflow,'zero_truncation':aug_overflow==0,
            'token_length_histogram':{str(k):aug_hist[k] for k in sorted(aug_hist)},'maximum_examples':aug_examples,
        },
        'source_dataset':str(source_path),'source_dataset_version':store.source_dataset_version,
        'sidecar':str(sidecar_path),'sidecar_verification_sha256':store.verification_sha256,
        'norm_stats_sha256':norm_sha,'norm_provenance_sha256':sha256(args.norm_provenance),
        'geometry_contract_sha256':C.GEOMETRY_CONTRACT_SHA256,'client_commit':git(client_root,'rev-parse','HEAD'),
        'builder_sha256':sha256(Path(__file__).resolve()),'elapsed_seconds':time.monotonic()-started,
    }
    atomic_json(args.output,result)
    if clean_overflow or aug_overflow:raise ValueError(f'State56 token overflow clean={clean_overflow} augmented={aug_overflow}')
    print(json.dumps({'token_audit':str(args.output),'sha256':sha256(args.output),'clean_max':clean_max,'augmented_max':aug_max,'frames':total},sort_keys=True))


if __name__=='__main__':main()
