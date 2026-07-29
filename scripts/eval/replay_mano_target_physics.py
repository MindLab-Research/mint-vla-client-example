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

from scripts import mano_dataset_release
from scripts.eval import mano_action_support
from scripts.eval import mano_physics_core as physics
from scripts.tools import validate_mano_dataset_release as release_validator

DT=physics.DT; FRAME_DT=0.005; SUBSTEPS=physics.NATIVE_SUBSTEPS; HAND_DIM=physics.HAND_DIM
CONTRACT='mano_target_physics_200hz_v1'
DEFAULT_DATASET=str(mano_dataset_release.resolve_role('training_dataset'))
DEFAULT_INDEX=str(mano_dataset_release.resolve_role('language_index'))
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

def new_clip_stats(): return {'clipped_steps':0,'clipped_values':0,'max_correction':0.0}
def update_clip_stats(stats,event):
    n=int(event['clipped_values']); stats['clipped_steps']+=int(n>0); stats['clipped_values']+=n
    stats['max_correction']=max(float(stats['max_correction']),float(event['max_correction']))

def object_rows(entries,object_name):
    rows=[int(e['row_index']) for e in entries if e.get('object_type')==object_name]
    if not rows: raise ValueError(f'unknown object {object_name!r}')
    return rows

def validate_release_inputs(args):
    release_path=Path(args.release_manifest).expanduser().resolve(); release=mano_dataset_release.load_release(release_path)
    dataset_path=Path(args.dataset).expanduser().resolve(); index_path=Path(args.index).expanduser().resolve()
    expected_dataset=mano_dataset_release.resolve_role('training_dataset',release=release,manifest_path=release_path)
    expected_index=mano_dataset_release.resolve_role('language_index',release=release,manifest_path=release_path)
    if dataset_path!=expected_dataset:
        raise ValueError(f'--dataset is not release role training_dataset: {dataset_path} != {expected_dataset}')
    if index_path!=expected_index:
        raise ValueError(f'--index is not release role language_index: {index_path} != {expected_index}')
    artifacts=release['artifacts']
    release_validator.validate_lance('canonical_lance',artifacts['canonical_lance'],dataset_path,deep=False)
    release_validator.validate_gesture_index('gesture_index',artifacts['gesture_index'],index_path)
    asset_id=release['roles']['assets']; asset_spec=artifacts[asset_id]
    asset_path=mano_dataset_release.resolve_artifact(asset_id,release=release,manifest_path=release_path)
    release_validator.validate_asset_bundle(asset_id,asset_spec,asset_path,release_path)
    generator_id=release['roles']['physics_quality_generator']; generator_spec=artifacts[generator_id]
    generator_path=mano_dataset_release.resolve_artifact(generator_id,release=release,manifest_path=release_path)
    if generator_path!=Path(__file__).resolve():
        raise ValueError(f'release physics generator is {generator_path}, not {Path(__file__).resolve()}')
    release_validator.validate_file(generator_id,generator_spec,generator_path)
    scene_spec=artifacts['scene_camera_code']; release_validator.validate_file('scene_camera_code',scene_spec,Path(mano_action_support.__file__).resolve())
    core_spec=artifacts['physics_contract_code']; release_validator.validate_file('physics_contract_code',core_spec,Path(physics.__file__).resolve())
    return release


def make_run_identity(args,object_name):
    release_path=Path(args.release_manifest).expanduser().resolve(); release=mano_dataset_release.load_release(release_path)
    asset_spec=release['artifacts'][release['roles']['assets']]
    return {'contract':CONTRACT,'object':object_name,'source_dataset':str(Path(args.dataset).resolve()),
            'dataset_release_id':release['release_id'],'dataset_release_manifest':str(release_path),
            'dataset_release_manifest_sha256':sha256(release_path),
            'dataset_metadata_sha256':lance_fingerprint(Path(args.dataset)),'gesture_index':str(Path(args.index).resolve()),
            'index_sha256':sha256(Path(args.index)),'script_sha256':sha256(Path(__file__)),
            'physics_core_sha256':sha256(Path(physics.__file__)),'asset_bundle_sha256':asset_spec['sha256'],
            'client_commit':os.environ.get('VLA_CLIENT_GIT_COMMIT','unknown'),
            'target_offset':int(args.target_offset),'source_dt':FRAME_DT,'mujoco_dt':DT,'steps_per_interval':SUBSTEPS}

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
        if r.get('status')!='ok' or int(r.get('row_index',-1))!=row_index or r.get('provenance')!=run_id: return False
        if r.get('trace_sha256')!=sha256(npz): return False
        with np.load(npz) as z:
            e=z['position_error_m']; return e.shape==(int(r['frames']),) and np.isfinite(e).all()
    except Exception: return False

def validate_row(row, row_index, ident):
    if int(ident.get('row_index',-1))!=row_index: raise ValueError('config row_index mismatch')
    hands=row.get('hands') or []; objects=row.get('objects') or []
    if len(hands)!=1 or len(objects)!=1: raise ValueError('requires exactly one hand and one object')
    h,o=hands[0],objects[0]
    arrays={'urdf_dof':h.get('urdf_dof'),'urdf_dof_target':h.get('urdf_dof_target'),
            'object_pos':o.get('pos'),'object_rot_aa':o.get('rot_aa'),'timestamps':row.get('timestamp'),
            'state':row.get('state'),'actions':row.get('actions')}
    if any(v is None for v in arrays.values()): raise ValueError(f'missing arrays: {[k for k,v in arrays.items() if v is None]}')
    lengths={k:len(v) for k,v in arrays.items()}
    if len(set(lengths.values()))!=1: raise ValueError(f'aligned array lengths disagree: {lengths}')
    T=next(iter(lengths.values()))
    if T<2: raise ValueError(f'need T>=2, got {T}')
    q=np.asarray(h['urdf_dof'],dtype=np.float64); target=np.asarray(h['urdf_dof_target'],dtype=np.float64)
    pos=np.asarray(o['pos'],dtype=np.float64); aa=np.asarray(o['rot_aa'],dtype=np.float64); ts=np.asarray(row['timestamp'],dtype=np.float64)
    if q.shape!=(T,HAND_DIM) or target.shape!=(T,HAND_DIM) or pos.shape!=(T,3) or aa.shape!=(T,3) or ts.shape!=(T,): raise ValueError(f'shape mismatch q={q.shape} target={target.shape} pos={pos.shape} aa={aa.shape} ts={ts.shape}')
    if not all(np.isfinite(x).all() for x in (q,target,pos,aa,ts)): raise ValueError('non-finite source state')
    intervals=np.diff(ts)
    if not np.all(intervals>0) or not np.allclose(intervals,FRAME_DT,rtol=0,atol=1e-10): raise ValueError('timestamps are not exact monotonic 200Hz intervals')
    idx=row.get('index') or {}
    if str(ident.get('uuid'))!=str(idx.get('uuid')) or str(ident.get('seed_uuid'))!=str(idx.get('seed_uuid')): raise ValueError('config index identity mismatch')
    obj=((row.get('trajectory_metadata') or {}).get('object_names') or [None])[0]
    if obj != ident.get('object_type'): raise ValueError(f'config object mismatch {obj} != {ident.get("object_type")}')
    return T,q,target,pos,aa,ts,str(obj)

def scene(object_name):
    if object_name not in _SCENES:
        _SCENES[object_name]=physics.make_scene(object_name,1,1,physics=True,physics_timestep=DT,create_renderer=False)
    return _SCENES[object_name]

def replay(row_index,row,ident,target_offset,run_id):
    import mujoco
    T,q,targets,ref_pos,ref_aa,ts,obj=validate_row(row,row_index,ident)
    tmp,model,data,renderer,objaddr,_,handaddr,_,limits=scene(obj)
    # Fresh MjData per row; object qpos is never written again after this initialization.
    data=mujoco.MjData(model)
    data.qpos[:]=0
    data.qpos[objaddr:objaddr+3]=ref_pos[0]
    data.qpos[objaddr+3:objaddr+7]=mano_action_support.axis_angle_to_wxyz(ref_aa[0])
    data.qpos[handaddr]=q[0]; data.qvel[:]=0; mujoco.mj_forward(model,data)
    sim_pos=np.empty((T,3),np.float64); sim_quat=np.empty((T,4),np.float64); err=np.empty(T,np.float64); hand_err=np.empty(T,np.float64)
    contacts={'hand_object':0,'object_floor':0,'hand_floor':0}; first={k:None for k in contacts}; max_ncon=max_force=max_act=max_qvel=0.; wrap_events=0; clip_stats=new_clip_stats()
    for i in range(T):
        sim_pos[i]=data.qpos[objaddr:objaddr+3]; sim_quat[i]=data.qpos[objaddr+3:objaddr+7]
        err[i]=np.linalg.norm(sim_pos[i]-ref_pos[i]); hand_err[i]=np.linalg.norm(data.qpos[handaddr]-q[i])
        if i==T-1: break
        j=min(T-1,max(0,i+target_offset)); current=np.asarray(data.qpos[handaddr],dtype=np.float64); raw=targets[j]
        wrapped=current[3:6]+(raw[3:6]-current[3:6]+np.pi)%(2*np.pi)-np.pi
        wrap_events+=int(not np.allclose(raw[3:6],wrapped,atol=1e-12))
        target,event=physics.nearest_wrapped_position_target(current,raw-current,limits); update_clip_stats(clip_stats,event)
        d=physics.step_servo(model=model,data=data,target=target,substeps=SUBSTEPS,object_name=obj)
        for k in contacts:
            if d[k+'_contact']:
                contacts[k]+=1
                if first[k] is None:first[k]=i+1
        max_ncon=max(max_ncon,d['max_ncon']); max_force=max(max_force,d['max_contact_force']); max_act=max(max_act,d['max_abs_actuator_force']); max_qvel=max(max_qvel,float(np.max(np.abs(data.qvel))))
    expected_steps=2*(T-1); expected_time=expected_steps*DT
    if not np.isclose(data.time,expected_time,rtol=0,atol=1e-9): raise RuntimeError(f'time {data.time} != {expected_time}')
    if err[0]>=1e-6: raise RuntimeError(f'frame0 object error {err[0]}')
    if not all(np.isfinite(x).all() for x in (sim_pos,sim_quat,err)): raise FloatingPointError('nonfinite simulated trajectory')
    grade=grade_from_max_error(float(err.max()))
    metrics={'max_position_error_m':float(err.max()),'argmax_frame':int(err.argmax()),'argmax_timestamp_s':float(ts[err.argmax()]),'mean_position_error_m':float(err.mean()),'rms_position_error_m':float(np.sqrt(np.mean(err**2))),'p95_position_error_m':float(np.percentile(err,95)),'final_position_error_m':float(err[-1]),'over_3cm':run_lengths(err>=.03),'over_8cm':run_lengths(err>=.08),'reference_path':path_metrics(ref_pos),'sim_path':path_metrics(sim_pos),'hand_tracking':{'mean_l2':float(hand_err.mean()),'max_l2':float(hand_err.max()),'final_l2':float(hand_err[-1])},'contacts':{'counts':contacts,'first_frame':first},'dynamics':{'max_contacts':int(max_ncon),'max_contact_force':float(max_force),'max_abs_actuator_force':float(max_act),'max_abs_qvel':float(max_qvel)},'diagnostics':{'target_offset':target_offset,'wrapped_wrist_commands':wrap_events,**clip_stats},'timing':{'source_dt':FRAME_DT,'mujoco_dt':DT,'steps_per_interval':SUBSTEPS,'intervals':T-1,'mj_steps':expected_steps,'sim_time':float(data.time),'object_sim_owned_after_frame0':True}}
    result={'status':'ok','grade':grade,'qualified':grade in ('A','B'),'row_index':row_index,'row_uuid':str(row['index']['uuid']),'seed_uuid':str(row['index']['seed_uuid']),'object':obj,'gesture':ident.get('gesture'),'frames':T,'metrics':metrics,'provenance':run_id}
    arrays={'frame_timestamp_s':ts.astype(np.float64),'reference_object_position':ref_pos.astype(np.float32),'sim_object_position':sim_pos.astype(np.float32),'sim_object_quaternion_wxyz':sim_quat.astype(np.float32),'position_error_m':err.astype(np.float32)}
    return result,arrays

def record_one(job):
    row_index,row,ident,offset,out,resume,run_id=job; root=Path(out); stem=root/'records'/f'{row_index:05d}'
    js=stem.with_suffix('.json'); npz=stem.with_suffix('.npz')
    if resume and resume_valid(js,npz,row_index,run_id): return {'row_index':row_index,'status':'skipped'}
    js.unlink(missing_ok=True); npz.unlink(missing_ok=True)
    try:
        result,arrays=replay(row_index,row,ident,offset,run_id)
        npz.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=npz.parent,delete=False,suffix='.npz') as f: np.savez_compressed(f,**arrays); temp=f.name
        os.replace(temp,npz); result['trace_npz']=str(npz); result['trace_sha256']=sha256(npz); atomic_json(js,result)
        return {'row_index':row_index,'status':'ok','grade':result['grade']}
    except Exception as e:
        idx=row.get('index') or {}
        invalid={'status':'invalid','grade':None,'qualified':False,'row_index':row_index,'row_uuid':str(idx.get('uuid')),'seed_uuid':str(idx.get('seed_uuid')),'object':ident.get('object_type'),'gesture':ident.get('gesture'),'provenance':run_id,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(limit=10)}
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
    if any(r.get('object')!=object_name for r in records): raise ValueError('record object mismatch')
    if run_id is not None and any(r.get('provenance')!=run_id for r in records): raise ValueError('record provenance mismatch')

def aggregate(out,dataset_path,target_offset,expected_rows,entries,object_name,target,overwrite,run_id):
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

def iter_jobs(ds,rows,entries,offset,out,resume,run_id,batch_size):
    cols=['index','trajectory_metadata','episode_metadata','hands','objects','timestamp','state','actions']
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]; payload=ds.take(batch,columns=cols).to_pylist()
        for i,r in zip(batch,payload,strict=True): yield (i,r,entries[i],offset,str(out),resume,run_id)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--release-manifest',default=str(mano_dataset_release.DEFAULT_RELEASE_MANIFEST))
 p.add_argument('--dataset',default=DEFAULT_DATASET)
 p.add_argument('--output-dir',required=True,help='new result root; the historical evidence root is immutable')
 p.add_argument('--index',default=DEFAULT_INDEX)
 p.add_argument('--object',required=True)
 p.add_argument('--rows',default='',help='global indices/ranges; ranges are end-exclusive and must belong to --object')
 p.add_argument('--target-offset',type=int,choices=(-1,0,1),default=0)
 p.add_argument('--workers',type=int,default=8)
 p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--resume',action='store_true')
 p.add_argument('--aggregate',action='store_true')
 p.add_argument('--aggregate-output',default='')
 p.add_argument('--overwrite-aggregate',action='store_true')
 a=p.parse_args()
 if a.workers<1 or a.batch_size<1: raise ValueError('workers and batch-size must be positive')
 validate_release_inputs(a)
 import lance
 ds=lance.dataset(a.dataset); count=ds.count_rows(); meta=json.loads(Path(a.index).read_text()); entries=meta.get('entries')
 if not isinstance(entries,list) or len(entries)!=count: raise ValueError(f'index/source count mismatch: {0 if not isinstance(entries,list) else len(entries)} != {count}')
 for i,e in enumerate(entries):
  if int(e.get('row_index',-1))!=i: raise ValueError(f'index entry {i} has row_index={e.get("row_index")}')
 all_rows=object_rows(entries,a.object); selected=all_rows if not a.rows else parse_rows(a.rows,count)
 outside=[i for i in selected if entries[i].get('object_type')!=a.object]
 if outside: raise ValueError(f'rows not belonging to {a.object}: {outside[:20]}')
 root=Path(a.output_dir).expanduser().resolve()
 release=mano_dataset_release.load_release(a.release_manifest)
 historical=mano_dataset_release.resolve_role('physics_quality',release=release,manifest_path=a.release_manifest)
 if root==historical or historical in root.parents:
  raise ValueError(f'historical physics evidence is immutable; choose a new --output-dir, not {root}')
 out=root/'objects'/a.object; run_id=make_run_identity(a,a.object)
 with single_object_lock(root):
  ensure_manifest(out,run_id,all_rows)
  if a.aggregate:
   target=Path(a.aggregate_output) if a.aggregate_output else out/f'{a.object}.{CONTRACT}.lance'
   print(aggregate(out,a.dataset,a.target_offset,all_rows,entries,a.object,target,a.overwrite_aggregate,run_id)); return
  invocation={'started_at':datetime.now(timezone.utc).isoformat(),'object':a.object,'selected_rows':selected,'workers':a.workers,'batch_size':a.batch_size,'resume':a.resume,'target_offset':a.target_offset}
  inv=out/'invocations'/f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")}.json'; atomic_json(inv,invocation)
  jobs=iter_jobs(ds,selected,entries,a.target_offset,out,a.resume,run_id,a.batch_size); started=time.monotonic(); counts=Counter(); grades=Counter(); invalid=[]
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
