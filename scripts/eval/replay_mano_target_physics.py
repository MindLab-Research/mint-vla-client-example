#!/usr/bin/env python3
"""Headless, frame-aligned 200 Hz MuJoCo quality replay for canonical MANO rows."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile, time, traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from pathlib import Path
from multiprocessing import get_context
import uuid
import numpy as np

from scripts.eval import mano_action_support
from scripts.eval import manorl_native_physics as physics

DT=physics.DT; FRAME_DT=0.005; SUBSTEPS=physics.NATIVE_SUBSTEPS; HAND_DIM=physics.HAND_DIM
CONTRACT='mano_native_target_replay_28d_200hz_v1'
SOURCE_CONTRACT='synthetic_mano_28d_checkpoint_rollout_v2_2'
EXPECTED_DATASET_VERSION=1; EXPECTED_ROWS=5_425; EXPECTED_FRAGMENTS=43
RELEASE_ROOT=Path('/vePFS-Mindverse/user/intern/wenxi/results/datas/28dof_manohand')
DEFAULT_DATASET=str(RELEASE_ROOT/'source/guangguan_all_actions_checkpoint2200_ratio5_seed42_v22.lance')
DEFAULT_ACCEPTED_MANIFEST=str(RELEASE_ROOT/'manifests/guangguan_merged_v1_quality_rules_8cm100f_rot30deg.accepted_rows.json')
DEFAULT_FILTER_VERIFICATION=str(RELEASE_ROOT/'manifests/source_filter_verification.json')
EXPECTED_ACCEPTED_MANIFEST_SHA256='2ae0dbd86a4fadc25d25fbc4ea692651761119ac0a7142d205ec5921186d582c'
EXPECTED_FILTER_VERIFICATION_SHA256='f85351f5ecb145fc720f5909e0b439e76f92d3fa43fc3321d43c772e4fe7e9c3'
SOURCE_COLUMNS=['index','trajectory_metadata','timestamp','hands','objects','provenance']
_SCENES={}

@contextmanager
def single_object_lock(root:Path):
    """Fail rather than run two objects concurrently under one output root."""
    root.mkdir(parents=True,exist_ok=True); path=root/'.single_object_run.lock'
    with path.open('a+') as f:
        try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError(f'another object job holds {path}') from e
        f.seek(0); f.truncate(); f.write(f'pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n'); f.flush()
        try: yield
        finally: fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def sha256(p: Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def lance_fingerprint(path: Path) -> str:
    """Stable, cheap source identity: hashes Lance metadata names and bytes, never RGB data."""
    h=hashlib.sha256()
    for p in sorted(path.rglob('*')):
        if p.is_file() and ('manifest' in p.name or p.name.startswith('_versions')):
            h.update(str(p.relative_to(path)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def atomic_json(path:Path, obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf8') as f:
        json.dump(obj,f,sort_keys=True,indent=2); f.write('\n'); tmp=Path(f.name)
    os.replace(tmp,path)

def run_lengths(mask):
    best=cur=frames=0
    for x in mask:
        if x: cur+=1; frames+=1; best=max(best,cur)
        else: cur=0
    return {'frames':int(frames),'longest_run':int(best)}

def path_metrics(pos):
    d=np.linalg.norm(np.diff(pos,axis=0),axis=1) if len(pos)>1 else np.array([])
    return {'path_length_m':float(d.sum()),'net_displacement_m':float(np.linalg.norm(pos[-1]-pos[0])),
            'max_height_m':float(pos[:,2].max())}

def grade_from_max_error(value):
    if not np.isfinite(value): raise ValueError('max error must be finite')
    return 'A' if value<.03 else 'B' if value<.08 else 'C'

def object_rows(entries,object_name):
    rows=[int(e['row_index']) for e in entries if e.get('object_type')==object_name]
    if not rows: raise ValueError(f'unknown object {object_name!r}')
    return rows

def validate_dataset(ds,path,version):
    rows=int(ds.count_rows()); fragments=len(ds.get_fragments()); names=list(ds.schema.names)
    expected=['index','trajectory_metadata','timestamp','hands','objects','contact','reference','rollout','provenance']
    if version!=EXPECTED_DATASET_VERSION or ds.version!=version: raise ValueError(f'source Lance version mismatch: requested={version} actual={ds.version}')
    if rows!=EXPECTED_ROWS or fragments!=EXPECTED_FRAGMENTS: raise ValueError(f'source population mismatch rows={rows} fragments={fragments}')
    if names!=expected: raise ValueError(f'source schema mismatch: {names}')
    return {'path':str(path),'version':version,'row_count':rows,'fragment_count':fragments,'schema_names':names,'metadata_sha256':lance_fingerprint(path)}


def validate_filter_contract(accepted_path,verification_path,dataset_path,source_summary):
    accepted_path=Path(accepted_path).expanduser().resolve(); verification_path=Path(verification_path).expanduser().resolve()
    if sha256(accepted_path)!=EXPECTED_ACCEPTED_MANIFEST_SHA256: raise ValueError('accepted-row manifest SHA256 mismatch')
    if sha256(verification_path)!=EXPECTED_FILTER_VERIFICATION_SHA256: raise ValueError('source filter verification SHA256 mismatch')
    accepted=json.loads(accepted_path.read_text()); verification=json.loads(verification_path.read_text()); rows=accepted.get('rows')
    selection=accepted.get('selection') or {}
    if accepted.get('schema')!='manorl.synthetic_quality_accepted_row_manifest.v1' or not isinstance(rows,list): raise ValueError('invalid accepted-row manifest schema')
    if len(rows)!=EXPECTED_ROWS or selection.get('rows_to_retain')!=EXPECTED_ROWS or selection.get('rows_to_reject')!=11_480: raise ValueError('accepted-row population mismatch')
    original_indices=[int(row.get('row_index',-1)) for row in rows]
    if original_indices!=sorted(original_indices) or len(set(original_indices))!=EXPECTED_ROWS or not all(row.get('accepted') is True for row in rows): raise ValueError('accepted-row order/identity contract mismatch')
    if verification.get('status')!='verified_and_rejected_rows_deleted' or verification.get('rows_before')!=16_905 or verification.get('rows_retained')!=EXPECTED_ROWS or verification.get('rows_deleted')!=11_480: raise ValueError('source filter verification population mismatch')
    if Path(verification.get('server_source_lance','')).resolve()!=Path(dataset_path).resolve() or verification.get('schema')!=source_summary['schema_names']: raise ValueError('source filter verification dataset/schema mismatch')
    if verification.get('accepted_manifest_sha256')!=EXPECTED_ACCEPTED_MANIFEST_SHA256 or verification.get('nas_source_modified') is not False or verification.get('original_server_backup_deleted') is not True: raise ValueError('source filter verification integrity mismatch')
    filtered_manifest=Path(verification.get('filtered_tree_manifest','')).resolve()
    if not filtered_manifest.is_file() or sha256(filtered_manifest)!=verification.get('filtered_tree_manifest_sha256'): raise ValueError('filtered source tree manifest mismatch')
    summary={'accepted_manifest':str(accepted_path),'accepted_manifest_sha256':EXPECTED_ACCEPTED_MANIFEST_SHA256,
             'filter_verification':str(verification_path),'filter_verification_sha256':EXPECTED_FILTER_VERIFICATION_SHA256,
             'filtered_tree_manifest':str(filtered_manifest),'filtered_tree_manifest_sha256':verification['filtered_tree_manifest_sha256'],
             'rows_before':verification['rows_before'],'rows_retained':verification['rows_retained'],'rows_deleted':verification['rows_deleted'],
             'predicate':selection.get('predicate'),'row_index_order':selection.get('row_index_order'),
             'row_index_sha256':selection.get('row_index_sha256'),'uuid_sha256':selection.get('uuid_sha256')}
    return rows,summary


def build_source_index(ds,accepted_rows,batch_size=128):
    """Stream compact IDs and bind every filtered row to its accepted lineage."""
    if len(accepted_rows)!=EXPECTED_ROWS: raise ValueError(f'accepted identity count {len(accepted_rows)} != {EXPECTED_ROWS}')
    entries=[]; scanner=ds.scanner(columns=['index','provenance'],batch_size=batch_size)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            i=len(entries)
            if i>=len(accepted_rows): raise ValueError('filtered source contains rows beyond accepted manifest')
            accepted=accepted_rows[i]; idx=row.get('index') or {}; prov=row.get('provenance') or {}
            obj=idx.get('scene'); identity=prov.get('source_identity'); gesture=str(accepted.get('pair','')).rsplit('_',1)[-1]
            observed=(str(idx.get('uuid')),str(obj),str(identity)); expected=(str(accepted.get('uuid')),str(accepted.get('object')),str(accepted.get('source_identity')))
            if idx.get('is_generated') is not True or prov.get('contract')!=SOURCE_CONTRACT or observed!=expected: raise ValueError(f'filtered/accepted identity mismatch at row {i}: {observed} != {expected}')
            if not identity.startswith(f'{obj}_{gesture}_'): raise ValueError(f'source identity/gesture mismatch at filtered row {i}')
            entries.append({'row_index':i,'original_merged_row_index':int(accepted['row_index']),
                            'uuid':str(idx.get('uuid')),'seed_uuid':str(idx.get('seed_uuid')),
                            'object_type':obj,'gesture':gesture,'total_frames':int(accepted.get('frames',-1)),
                            'source_identity':identity,'source_dataset':prov.get('dataset_path'),
                            'source_dataset_version':prov.get('dataset_version'),'source_row_index':prov.get('row_index'),
                            'checkpoint_update':prov.get('checkpoint_update'),'checkpoint_sha256':prov.get('checkpoint_sha256')})
    if len(entries)!=EXPECTED_ROWS: raise ValueError(f'streamed identity count {len(entries)} != {EXPECTED_ROWS}')
    uuids=[entry['uuid'] for entry in entries]; original=[entry['original_merged_row_index'] for entry in entries]
    if len(set(uuids))!=len(uuids) or original!=sorted(original) or len(set(original))!=len(original): raise ValueError('filtered UUID/original-row identity is not unique and ordered')
    return entries


def make_run_identity(args,object_name,source_summary,filter_summary):
    return {'contract':CONTRACT,'source_contract':SOURCE_CONTRACT,'object':object_name,
            'source_dataset':str(Path(args.dataset).resolve()),'source_dataset_version':int(args.dataset_version),
            'source_metadata_sha256':source_summary['metadata_sha256'],'filtered_population':filter_summary,
            'script_sha256':sha256(Path(__file__)),
            'physics_adapter_sha256':sha256(Path(physics.__file__)),'client_commit':os.environ.get('VLA_CLIENT_GIT_COMMIT','unknown'),
            'manorl':physics.runtime_provenance(object_name),'source_dt':FRAME_DT,'mujoco_dt':DT,'steps_per_interval':SUBSTEPS,
            'control_input':'hands[right].urdf_dof_target[t] absolute position target','external_target_modification':False}

def ensure_manifest(out,run_id,rows):
    path=out/'manifest.json'; expected=[int(x) for x in rows]
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('run_identity')!=run_id or old.get('row_indices')!=expected:
            raise RuntimeError(f'existing {path} has different provenance/population; use a new output directory')
        return
    atomic_json(path,{'contract':CONTRACT,'created_at':datetime.now(timezone.utc).isoformat(),
                      'object':run_id['object'],'row_count':len(expected),'row_indices':expected,'run_identity':run_id})

def resume_valid(js,npz,row_index,run_id):
    if not js.exists() or not npz.exists(): return False
    try:
        r=json.loads(js.read_text())
        if r.get('status')!='ok' or int(r.get('row_index',-1))!=row_index or r.get('provenance')!=run_id or r.get('trace_sha256')!=sha256(npz): return False
        T=int(r['frames']); required={'object_position_error':(T,),'simulated_full_qpos':(T,HAND_DIM+7),'simulated_hand_qpos':(T,HAND_DIM),'source_target_qpos':(T,HAND_DIM)}
        with np.load(npz) as z: return all(name in z and z[name].shape==shape and np.isfinite(z[name]).all() for name,shape in required.items())
    except Exception: return False


def _right_hand(row,meta):
    hands=row.get('hands') or []; slots=meta.get('hand_slots'); names=meta.get('hand_names')
    if not isinstance(slots,list) or len(slots)!=len(hands) or slots.count('right')!=1: raise ValueError('trajectory_metadata.hand_slots cannot resolve one right hand')
    if not isinstance(names,list) or 'right' not in names: raise ValueError('trajectory_metadata.hand_names does not contain right')
    hand=hands[slots.index('right')]
    if not isinstance(hand,dict) or hand.get('hand_name')!='right': raise ValueError("resolved right slot does not report hand_name='right'")
    return hand


def _selected_object(row,meta,obj):
    objects=row.get('objects') or []; names=meta.get('object_names')
    if not isinstance(names,list) or len(names)!=len(objects) or names.count(obj)!=1: raise ValueError('trajectory_metadata.object_names cannot resolve selected object')
    selected=objects[names.index(obj)]
    if not isinstance(selected,dict): raise ValueError('selected object slot is not a mapping')
    return selected


def validate_row(row,row_index,ident):
    idx=row.get('index') or {}; meta=row.get('trajectory_metadata') or {}; prov=row.get('provenance') or {}; obj=idx.get('scene')
    if int(ident.get('row_index',-1))!=row_index or str(ident.get('uuid'))!=str(idx.get('uuid')) or str(ident.get('seed_uuid'))!=str(idx.get('seed_uuid')): raise ValueError('row/compact index identity mismatch')
    if ident.get('object_type')!=obj or ident.get('gesture')!=meta.get('gesture') or ident.get('source_identity')!=prov.get('source_identity') or prov.get('contract')!=SOURCE_CONTRACT: raise ValueError('row/compact lineage mismatch')
    hand=_right_hand(row,meta); selected=_selected_object(row,meta,obj); ts=np.asarray(row.get('timestamp'),dtype=np.float64); T=len(ts)
    q=np.asarray(hand.get('urdf_dof'),dtype=np.float64); target=np.asarray(hand.get('urdf_dof_target'),dtype=np.float64)
    pos=np.asarray(selected.get('pos'),dtype=np.float64); aa=np.asarray(selected.get('rot_aa'),dtype=np.float64)
    arrays={'timestamp':(ts,(T,)),'urdf_dof':(q,(T,HAND_DIM)),'urdf_dof_target':(target,(T,HAND_DIM)),'object_position':(pos,(T,3)),'object_axis_angle':(aa,(T,3))}
    if T<2 or int(meta.get('total_frames',-1))!=T: raise ValueError(f'invalid frame count {T}')
    for name,(value,shape) in arrays.items():
        if value.shape!=shape or not np.isfinite(value).all(): raise ValueError(f'invalid {name}: shape={value.shape}')
    if not np.isclose(ts[0],0,rtol=0,atol=1e-12) or not np.allclose(np.diff(ts),FRAME_DT,rtol=0,atol=1e-10): raise ValueError('timestamps are not exact 200Hz intervals from zero')
    return T,q,target,pos,aa,ts,str(obj)


def scene(object_name):
    if object_name not in _SCENES: _SCENES[object_name]=physics.make_scene(object_name,1,1,physics=True,physics_timestep=DT,create_renderer=False)
    return _SCENES[object_name]


def _rotation_error(first,second): return 2*np.arccos(np.clip(np.abs(np.sum(first*second,axis=-1)),-1,1))
def _control_limits(targets,limits):
    exceed=np.maximum(np.maximum(limits[:,0][None]-targets,targets-limits[:,1][None]),0); outside=exceed>0; frame,joint=np.unravel_index(int(np.argmax(exceed)),exceed.shape)
    return {'external_preclipping_applied':False,'compiled_model_explains_limit_handling':True,'outside_element_count':int(outside.sum()),'outside_frame_count':int(np.any(outside,axis=1).sum()),'max_exceedance':float(exceed[frame,joint]),'max_exceedance_frame':int(frame),'max_exceedance_joint_index':int(joint)}


def replay(row_index,row,ident,run_id):
    import mujoco
    T,q,targets,ref_pos,ref_aa,ts,obj=validate_row(row,row_index,ident); _,model,_,_,objaddr,_,handaddr,_,limits=scene(obj)
    data=mujoco.MjData(model); data.qpos[:]=0; data.qpos[objaddr:objaddr+3]=ref_pos[0]; data.qpos[objaddr+3:objaddr+7]=mano_action_support.axis_angle_to_wxyz(ref_aa[0]); data.qpos[handaddr]=q[0]; data.qvel[:]=0; mujoco.mj_forward(model,data)
    body=physics.object_body_id(model,obj)
    full_q=np.empty((T,model.nq)); full_v=np.empty((T,model.nv)); sim_q=np.empty((T,HAND_DIM)); sim_pos=np.empty((T,3)); sim_quat=np.empty((T,4)); sim_time=np.empty(T)
    ref_quat=np.stack([mano_action_support.axis_angle_to_wxyz(value) for value in ref_aa]); qerr=np.empty(T); poserr=np.empty(T); roterr=np.empty(T)
    contacts={'hand_object':0,'object_floor':0,'hand_floor':0}; first={k:None for k in contacts}; max_ncon=max_force=max_act=max_qvel=0.; warnings=[]
    for i in range(T):
        full_q[i]=data.qpos; full_v[i]=data.qvel; sim_q[i]=data.qpos[handaddr]; sim_pos[i]=data.xpos[body]; sim_quat[i]=data.xquat[body]; sim_time[i]=data.time
        qerr[i]=np.max(np.abs(sim_q[i]-q[i])); poserr[i]=np.linalg.norm(sim_pos[i]-ref_pos[i]); roterr[i]=_rotation_error(sim_quat[i:i+1],ref_quat[i:i+1])[0]
        if i==T-1: break
        d=physics.step_servo(model=model,data=data,target=targets[i],substeps=SUBSTEPS,object_name=obj); warnings.extend(d['warnings'])
        for k in contacts:
            if d[k+'_contact']:
                contacts[k]+=1
                if first[k] is None: first[k]=i+1
        max_ncon=max(max_ncon,d['max_ncon']); max_force=max(max_force,d['max_contact_force']); max_act=max(max_act,d['max_abs_actuator_force']); max_qvel=max(max_qvel,float(np.max(np.abs(data.qvel))))
    expected_steps=SUBSTEPS*(T-1); expected_time=expected_steps*DT
    if not np.isclose(data.time,expected_time,rtol=0,atol=1e-12): raise RuntimeError(f'time {data.time} != {expected_time}')
    if poserr[0]>1e-6: raise RuntimeError(f'frame0 object error {poserr[0]}')
    if warnings: raise RuntimeError(f'MuJoCo warnings: {warnings[:8]}')
    if not all(np.isfinite(x).all() for x in (full_q,full_v,sim_pos,sim_quat,qerr,poserr,roterr)): raise FloatingPointError('nonfinite simulated trajectory')
    qe,pe,re=qerr[1:],poserr[1:],roterr[1:]; qarg=int(qe.argmax())+1; parg=int(pe.argmax())+1; rarg=int(re.argmax())+1; grade=grade_from_max_error(float(pe.max()))
    metrics={'max_position_error_m':float(pe.max()),'argmax_frame':parg,'argmax_timestamp_s':float(ts[parg]),'mean_position_error_m':float(pe.mean()),'rms_position_error_m':float(np.sqrt(np.mean(pe**2))),'p95_position_error_m':float(np.percentile(pe,95)),'final_position_error_m':float(poserr[-1]),
             'max_rotation_error_rad':float(re.max()),'mean_rotation_error_rad':float(re.mean()),'rms_rotation_error_rad':float(np.sqrt(np.mean(re**2))),'rotation_argmax_frame':rarg,
             'max_qpos_abs_error':float(qe.max()),'mean_qpos_abs_error':float(qe.mean()),'rms_qpos_abs_error':float(np.sqrt(np.mean(qe**2))),'qpos_argmax_frame':qarg,
             'over_3cm':run_lengths(poserr>=.03),'over_8cm':run_lengths(poserr>=.08),'reference_path':path_metrics(ref_pos),'sim_path':path_metrics(sim_pos),'contacts':{'counts':contacts,'first_frame':first},
             'dynamics':{'max_contacts':int(max_ncon),'max_contact_force':float(max_force),'max_abs_actuator_force':float(max_act),'max_abs_qvel':float(max_qvel)},'diagnostics':_control_limits(targets[:-1],limits),
             'timing':{'source_dt':FRAME_DT,'mujoco_dt':DT,'steps_per_interval':SUBSTEPS,'intervals':T-1,'mj_steps':expected_steps,'sim_time':float(data.time),'object_sim_owned_after_frame0':True,'final_source_target_executed':False},
             'initial_errors':{'qpos':float(qerr[0]),'position_m':float(poserr[0]),'rotation_rad':float(roterr[0])}}
    result={'status':'ok','grade':grade,'qualified':grade in ('A','B'),'row_index':row_index,
            'original_merged_row_index':int(ident['original_merged_row_index']),
            'row_uuid':str(row['index']['uuid']),'seed_uuid':str(row['index']['seed_uuid']),'source_identity':ident['source_identity'],'object':obj,'gesture':ident['gesture'],'frames':T,'valid_transition_count':T-1,'metrics':metrics,'provenance':run_id}
    arrays={'timestamp':ts.astype(np.float64),'simulated_time':sim_time.astype(np.float64),'state_producing_target_index':np.arange(-1,T-1,dtype=np.int64),'target_index':np.arange(T-1,dtype=np.int64),'target_qpos':targets[:-1].astype(np.float32),'source_target_qpos':targets.astype(np.float32),
            'simulated_full_qpos':full_q.astype(np.float32),'simulated_full_qvel':full_v.astype(np.float32),'simulated_hand_qpos':sim_q.astype(np.float32),'recorded_hand_qpos':q.astype(np.float32),'simulated_object_position':sim_pos.astype(np.float32),'recorded_object_position':ref_pos.astype(np.float32),
            'simulated_object_quaternion':sim_quat.astype(np.float32),'recorded_object_quaternion':ref_quat.astype(np.float32),'qpos_error':qerr.astype(np.float32),'object_position_error':poserr.astype(np.float32),'object_rotation_error':roterr.astype(np.float32)}
    return result,arrays

def record_one(job):
    row_index,row,ident,out,resume,run_id=job; root=Path(out); stem=root/'records'/f'{row_index:05d}'
    js=stem.with_suffix('.json'); npz=stem.with_suffix('.npz')
    if resume and resume_valid(js,npz,row_index,run_id): return {'row_index':row_index,'status':'skipped'}
    js.unlink(missing_ok=True); npz.unlink(missing_ok=True)
    try:
        result,arrays=replay(row_index,row,ident,run_id)
        npz.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=npz.parent,delete=False,suffix='.npz') as f: np.savez_compressed(f,**arrays); temp=f.name
        os.replace(temp,npz); result['trace_npz']=str(npz); result['trace_sha256']=sha256(npz); atomic_json(js,result)
        return {'row_index':row_index,'status':'ok','grade':result['grade']}
    except Exception as e:
        idx=row.get('index') or {}
        invalid={'status':'invalid','grade':None,'qualified':False,'row_index':row_index,
                 'original_merged_row_index':ident.get('original_merged_row_index'),
                 'row_uuid':str(idx.get('uuid')),'seed_uuid':str(idx.get('seed_uuid')),'object':ident.get('object_type'),'gesture':ident.get('gesture'),'provenance':run_id,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(limit=10)}
        atomic_json(js,invalid); return {'row_index':row_index,'status':'invalid','error':f'{type(e).__name__}: {e}'}

def validate_record_population(records,expected_rows,entries,object_name,run_id=None):
    actual=[int(r.get('row_index',-1)) for r in records]
    if actual!=expected_rows:
        missing=sorted(set(expected_rows)-set(actual)); extra=sorted(set(actual)-set(expected_rows))
        raise ValueError(f'record population mismatch missing={missing[:20]} extra={extra[:20]}')
    bad=[r for r in records if r.get('status')!='ok']
    if bad: raise ValueError(f'cannot aggregate {len(bad)} non-ok rows: {[r["row_index"] for r in bad[:20]]}')
    uuids=[r.get('row_uuid') for r in records]; expected=[str(entries[i].get('uuid')) for i in expected_rows]
    if uuids!=expected or len(set(uuids))!=len(uuids): raise ValueError('record UUID order/uniqueness mismatch')
    original=[r.get('original_merged_row_index') for r in records]; expected_original=[entries[i].get('original_merged_row_index') for i in expected_rows]
    if original!=expected_original: raise ValueError('record original merged-row identity mismatch')
    if any(r.get('object')!=object_name for r in records): raise ValueError('record object mismatch')
    if run_id is not None and any(r.get('provenance')!=run_id for r in records): raise ValueError('record provenance mismatch')

def aggregate(out,expected_rows,entries,object_name,target,overwrite,run_id):
    import lance,pyarrow as pa
    records=[]
    for p in sorted((out/'records').glob('*.json')):
        r=json.loads(p.read_text()); r['trace_json']=str(p); records.append(r)
    records.sort(key=lambda x:x['row_index']); validate_record_population(records,expected_rows,entries,object_name,run_id)
    for r in records:
        trace=Path(r.get('trace_npz',''))
        if not trace.is_file() or r.get('trace_sha256')!=sha256(trace): raise ValueError(f'trace missing/checksum mismatch for row {r["row_index"]}')
    target=Path(target); tmp=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex)
    lance.write_dataset(pa.Table.from_pylist(records),str(tmp),mode='create')
    written=lance.dataset(str(tmp))
    if written.count_rows()!=len(expected_rows): shutil.rmtree(tmp); raise RuntimeError('written shard row count mismatch')
    uuids=written.to_table(columns=['row_uuid'])['row_uuid'].to_pylist()
    if uuids!=[str(entries[i]['uuid']) for i in expected_rows]: shutil.rmtree(tmp); raise RuntimeError('written shard UUID order mismatch')
    if target.exists():
        if not overwrite: shutil.rmtree(tmp); raise FileExistsError(f'{target} exists; use --overwrite-aggregate')
        backup=target.with_name(target.name+'.previous-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')); os.replace(target,backup)
    os.replace(tmp,target)
    verification={'created_at':datetime.now(timezone.utc).isoformat(),'object':object_name,'row_count':len(expected_rows),'unique_row_uuid_count':len(set(uuids)),'grade_counts':dict(Counter(r['grade'] for r in records)),'qualified_count':sum(bool(r['qualified']) for r in records),'output':str(target)}
    atomic_json(out/'aggregate_verification.json',verification); print(json.dumps(verification,indent=2)); return target

def parse_rows(s,count):
    if not s:return list(range(count))
    ans=[]
    for part in s.split(','):
        a=part.strip().split(':')
        if len(a)==1: ans.append(int(a[0]))
        elif len(a)==2 and int(a[1])>int(a[0]): ans.extend(range(int(a[0]),int(a[1])))
        else: raise ValueError(f'invalid end-exclusive row range {part!r}')
    ans=sorted(set(ans)); bad=[x for x in ans if x<0 or x>=count]
    if bad: raise ValueError(f'rows outside [0,{count}): {bad[:20]}')
    return ans

def iter_jobs(ds,rows,entries,out,resume,run_id,batch_size):
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]; payload=ds.take(batch,columns=SOURCE_COLUMNS).to_pylist()
        for i,r in zip(batch,payload,strict=True): yield (i,r,entries[i],str(out),resume,run_id)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',default=DEFAULT_DATASET); p.add_argument('--dataset-version',type=int,default=EXPECTED_DATASET_VERSION)
 p.add_argument('--accepted-manifest',default=DEFAULT_ACCEPTED_MANIFEST); p.add_argument('--filter-verification',default=DEFAULT_FILTER_VERIFICATION)
 p.add_argument('--output-dir',required=True); p.add_argument('--object',required=True)
 p.add_argument('--rows',default='',help='global indices/ranges; ranges are end-exclusive and must belong to --object')
 p.add_argument('--workers',type=int,default=8); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--index-batch-size',type=int,default=128)
 p.add_argument('--resume',action='store_true'); p.add_argument('--aggregate',action='store_true'); p.add_argument('--aggregate-output',default=''); p.add_argument('--overwrite-aggregate',action='store_true')
 a=p.parse_args()
 if min(a.workers,a.batch_size,a.index_batch_size)<1: raise ValueError('workers and batch sizes must be positive')
 import lance
 dataset_path=Path(a.dataset).expanduser().resolve(); ds=lance.dataset(str(dataset_path),version=a.dataset_version)
 source=validate_dataset(ds,dataset_path,a.dataset_version)
 accepted_rows,filter_summary=validate_filter_contract(a.accepted_manifest,a.filter_verification,dataset_path,source)
 entries=build_source_index(ds,accepted_rows,a.index_batch_size); all_rows=object_rows(entries,a.object); selected=all_rows if not a.rows else parse_rows(a.rows,source['row_count'])
 outside=[i for i in selected if entries[i].get('object_type')!=a.object]
 if outside: raise ValueError(f'rows not belonging to {a.object}: {outside[:20]}')
 root=Path(a.output_dir).expanduser().resolve(); out=root/'objects'/a.object; run_id=make_run_identity(a,a.object,source,filter_summary)
 with single_object_lock(root):
  ensure_manifest(out,run_id,all_rows)
  if a.aggregate:
   target=Path(a.aggregate_output) if a.aggregate_output else out/f'{a.object}.{CONTRACT}.lance'
   print(aggregate(out,all_rows,entries,a.object,target,a.overwrite_aggregate,run_id)); return
  invocation={'started_at':datetime.now(timezone.utc).isoformat(),'object':a.object,'selected_rows':selected,'workers':a.workers,'batch_size':a.batch_size,'resume':a.resume}
  inv=out/'invocations'/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")}.json'; atomic_json(inv,invocation)
  jobs=iter_jobs(ds,selected,entries,out,a.resume,run_id,a.batch_size); started=time.monotonic(); counts=Counter(); grades=Counter(); invalid=[]
  def consume(results):
   for n,r in enumerate(results,1):
    counts[r['status']]+=1
    if r.get('grade'): grades[r['grade']]+=1
    if r['status']=='invalid': invalid.append(r)
    if n==1 or n%25==0 or n==len(selected): print(json.dumps({'object':a.object,'completed':n,'total':len(selected),'rows_per_second':n/max(time.monotonic()-started,1e-9),'status':dict(counts),'grades':dict(grades)},sort_keys=True),flush=True)
  if a.workers==1: consume(map(record_one,jobs))
  else:
   with get_context('spawn').Pool(a.workers) as pool: consume(pool.imap_unordered(record_one,jobs,chunksize=1))
  summary={'finished_at':datetime.now(timezone.utc).isoformat(),'object':a.object,'selected_row_count':len(selected),'status_counts':dict(counts),'grade_counts':dict(grades),'invalid_rows':invalid,'elapsed_seconds':time.monotonic()-started,'invocation':str(inv)}
  summary_path=out/'summaries'/f'{inv.stem}.json'; atomic_json(summary_path,summary); print(json.dumps({**summary,'summary':str(summary_path)},indent=2))
if __name__=='__main__': main()
