#!/usr/bin/env python3
"""Compute State56 Scheme-A normalization from exactly4,613 train windows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import lance
import numpy as np

from openpi.shared import normalize
from scripts import mano_state56_contract as C

NORM_CONTRACT = "mano_state56_scheme_a_trainonly_norm_v1"
SIDECAR_CONTRACT = "mano_state56_native28_virtual_sidecar_v1"


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def git(root: Path,*args: str)->str:
    return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()


def atomic_json(path: Path,payload: dict)->None:
    temp=path.with_name(f'.{path.name}.tmp-{uuid.uuid4().hex}')
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    os.replace(temp,path)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sidecar',type=Path,required=True)
    parser.add_argument('--sidecar-verification',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--batch-rows',type=int,default=24)
    args=parser.parse_args()
    sidecar_path=args.sidecar.expanduser().resolve();verification_path=args.sidecar_verification.expanduser().resolve();output=args.output_dir.expanduser().resolve()
    if output.exists():raise FileExistsError(f'refusing existing State56 norm output: {output}')
    client_root=Path(__file__).resolve().parents[2]
    if git(client_root,'status','--porcelain','--ignore-submodules=dirty'):
        raise RuntimeError('State56 norm requires a clean client checkout')
    verification=json.loads(verification_path.read_text())
    if verification.get('contract')!=SIDECAR_CONTRACT or verification.get('status')!='passed':
        raise ValueError('State56 sidecar verification contract/status mismatch')
    if Path(verification.get('path','')).resolve()!=sidecar_path:
        raise ValueError('State56 sidecar verification points to another release')
    if (verification.get('rows'),verification.get('train_rows'),verification.get('validation_rows'),verification.get('held_out_rows'))!=(4856,4613,243,0):
        raise ValueError('State56 sidecar Scheme-A counts mismatch')
    dataset=lance.dataset(str(sidecar_path),version=int(verification['lance_version']))
    state_stats=normalize.RunningStats();action_stats=normalize.RunningStats()
    train_rows=0;validation_rows=0;train_frames=0;validation_frames=0
    source_dataset=None;source_version=None;plan_sha=None
    started=time.monotonic();step=max(1,args.batch_rows)
    for offset in range(0,4856,step):
        side_rows=dataset.take(list(range(offset,min(offset+step,4856))),columns=['index','window','state','provenance']).to_pylist()
        train_side=[row for row in side_rows if row['index']['split']=='train']
        validation_rows += sum(row['index']['split']=='validation' for row in side_rows)
        validation_frames += sum(int(row['window']['frame_count']) for row in side_rows if row['index']['split']=='validation')
        if any(row['index']['split'] not in {'train','validation'} for row in side_rows):
            raise ValueError('State56 sidecar contains an unknown split')
        if not train_side:continue
        for row in train_side:
            provenance=row['provenance']
            identity=(provenance['source_dataset'],int(provenance['source_dataset_version']),provenance['plan_sha256'])
            if source_dataset is None:source_dataset,source_version,plan_sha=identity
            elif identity!=(source_dataset,source_version,plan_sha):raise ValueError('State56 sidecar source identity varies by row')
        source=lance.dataset(source_dataset,version=source_version)
        source_rows=source.take([int(row['index']['release_row_index']) for row in train_side],columns=['index','actions','row_payload_sha256']).to_pylist()
        for side,raw in zip(train_side,source_rows,strict=True):
            if raw['index']['uuid']!=side['index']['uuid'] or raw['row_payload_sha256']!=side['provenance']['source_row_payload_sha256']:
                raise ValueError(f"State56 source join mismatch row{side['index']['release_row_index']}")
            state=np.asarray(side['state'],dtype=np.float32)
            window=side['window'];start,end=int(window['start_frame']),int(window['end_frame'])
            absolute_actions=np.asarray(raw['actions'],dtype=np.float32)
            expected=(int(window['frame_count']),C.STATE_DIM)
            if state.shape!=expected or absolute_actions.ndim!=2 or absolute_actions.shape[1]!=C.ACTION_DIM:
                raise ValueError(f"State56 norm shape mismatch row{side['index']['release_row_index']}: {state.shape}/{absolute_actions.shape}")
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(absolute_actions)):
                raise ValueError('State56 norm population contains non-finite values')
            if not np.array_equal(absolute_actions[:,28:],np.zeros((absolute_actions.shape[0],4),dtype=np.float32)):
                raise ValueError('State56 norm population action pad4 is nonzero')
            query_frames=np.arange(start,end+1,dtype=np.int64)
            target_frames=np.minimum(query_frames[:,None]+np.arange(C.ACTION_HORIZON,dtype=np.int64)[None,:],end)
            residual_actions=absolute_actions[target_frames].copy()
            residual_actions[:,:,:3]-=state[:,None,:3]
            residual_actions[:,:,6:28]-=state[:,None,6:28]
            state_stats.update(state);action_stats.update(residual_actions.reshape(-1,C.ACTION_DIM));train_rows+=1;train_frames+=state.shape[0]
    if (train_rows,validation_rows,train_frames,validation_frames)!=(4613,243,2560614,134518):
        raise ValueError(f'State56 Scheme-A norm population mismatch: {(train_rows,validation_rows,train_frames,validation_frames)}')
    empirical_state=state_stats.get_statistics()
    state_mean=np.asarray(empirical_state.mean,dtype=np.float32).copy();state_std=np.asarray(empirical_state.std,dtype=np.float32).copy()
    state_q01=np.asarray(empirical_state.q01,dtype=np.float32).copy();state_q99=np.asarray(empirical_state.q99,dtype=np.float32).copy()
    state_mean[28:33]=0.5;state_std[28:33]=0.5;state_q01[28:33]=0.0;state_q99[28:33]=1.0
    state_q01[49:54]=0.0;state_q99[49:54]=float(np.log1p(C.FORCE_REFERENCE_NEWTONS))
    state_q01[55]=0.0;state_q99[55]=C.CONTACT_AGE_CLIP_SECONDS
    state_result=normalize.NormStats(mean=state_mean,std=state_std,q01=state_q01,q99=state_q99)
    action_result=action_stats.get_statistics()
    stats={'state':state_result,'actions':action_result}
    if np.asarray(state_result.mean).shape!=(56,) or np.asarray(action_result.mean).shape!=(32,):
        raise RuntimeError('State56 norm widths mismatch')
    if not np.array_equal(np.asarray(action_result.q01)[28:],np.zeros(4)) or not np.array_equal(np.asarray(action_result.q99)[28:],np.zeros(4)):
        raise RuntimeError('State56 action pad4 normalization is not exactly zero')
    staging=output.with_name(f'.{output.name}.incoming-{uuid.uuid4().hex}');staging.mkdir(parents=True)
    try:
        normalize.save(staging,stats)
        norm_path=staging/'norm_stats.json';norm_sha=sha256(norm_path)
        report={
            'contract':NORM_CONTRACT,'status':'passed','created_at':datetime.now(timezone.utc).isoformat(),
            'state_contract':C.STATE_CONTRACT_ID,'action_contract':C.ACTION_CONTRACT_ID,
            'state_dim':56,'action_dim':32,'action_physical_dim':28,'action_padding_dim':4,
            'action_delta_mask':[3,-3,22,-4],'action_source':'urdf_target_absolute',
            'scheme':'A','population_grade':'A','population_rows':4856,'train_trajectory_count':train_rows,
            'validation_trajectory_count':validation_rows,'held_out_trajectory_count':0,
            'train_active_frame_count':train_frames,'validation_active_frame_count':validation_frames,
            'action_vector_count':train_frames*C.ACTION_HORIZON,
            'action_norm_semantics':'target28 residual to query qpos on mask(3,-3,22,-4), horizon10, repeat-pad at window end',
            'state_fixed_ranges':{'contact28_33':[0.0,1.0],'force49_54':[0.0,float(np.log1p(C.FORCE_REFERENCE_NEWTONS))],'age55':[0.0,1.0]},
            'norm_population':'train_only_contact_window','validation_excluded':True,
            'sidecar':str(sidecar_path),'sidecar_verification':str(verification_path),
            'sidecar_verification_sha256':sha256(verification_path),'sidecar_plan_sha256':plan_sha,
            'source_dataset':source_dataset,'source_dataset_version':source_version,
            'norm_stats_sha256':norm_sha,'geometry_contract_sha256':C.GEOMETRY_CONTRACT_SHA256,
            'client_commit':git(client_root,'rev-parse','HEAD'),'builder_sha256':sha256(Path(__file__).resolve()),
            'elapsed_seconds':time.monotonic()-started,
        }
        atomic_json(staging/'norm_provenance.json',report)
        output.parent.mkdir(parents=True,exist_ok=True);os.replace(staging,output)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps({'norm':str(output/'norm_stats.json'),'norm_sha256':norm_sha,'provenance':str(output/'norm_provenance.json'),'provenance_sha256':sha256(output/'norm_provenance.json'),'train_frames':train_frames},sort_keys=True))


if __name__=='__main__':main()
