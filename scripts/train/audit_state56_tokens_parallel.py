#!/usr/bin/env python3
"""Deterministic parallel executor for the exhaustive State56 token audit."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess,time,uuid
import lance,numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from scripts import mano_state56_contract as C
from scripts.eval.manorl_native28_physics import Native28FingertipFK
from scripts.state56_virtual_data import State56SidecarStore
from scripts.train.audit_state56_scheme_a_tokens import record_length,update_max,AUDIT_CONTRACT

PLAN_CONTRACT='mano_state56_scheme_a_token_parallel_plan_v1'
PART_CONTRACT='mano_state56_scheme_a_token_parallel_part_v1'

def sha256(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_json(p:Path,x:dict)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.tmp-{uuid.uuid4().hex}');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def git(r:Path,*a:str)->str:return subprocess.check_output(['git','-C',str(r),*a],text=True,stderr=subprocess.DEVNULL).strip()
def code_identity()->dict:
 root=Path(__file__).resolve().parents[2]
 if git(root,'status','--porcelain','--ignore-submodules=dirty'):raise RuntimeError('parallel token audit requires clean client checkout')
 serial=Path(__file__).with_name('audit_state56_scheme_a_tokens.py')
 return {'client_commit':git(root,'rev-parse','HEAD'),'parallel_builder_sha256':sha256(Path(__file__).resolve()),'serial_mechanism_sha256':sha256(serial)}
def store_from(plan):return State56SidecarStore(Path(plan['sidecar']),verification_path=Path(plan['sidecar_verification']),expected_verification_sha256=plan['sidecar_verification_sha256'],source_dataset=Path(plan['source_dataset']))
def load_plan(path:Path):
 p=json.loads(path.read_text());
 if p.get('contract')!=PLAN_CONTRACT or p.get('code')!=code_identity():raise ValueError('parallel token plan identity mismatch')
 return p,sha256(path)
def make_plan(args):
 out=args.output_root.resolve();
 if out.exists() and any(out.iterdir()):raise FileExistsError(f'nonempty token parallel root: {out}')
 out.mkdir(parents=True,exist_ok=True)
 store=State56SidecarStore(args.sidecar,verification_path=args.sidecar_verification,expected_verification_sha256=args.sidecar_verification_sha256,source_dataset=args.source_dataset)
 light=store.dataset.to_table(columns=['index','window']).to_pylist();entries=sorted(({'source_row':int(r['index']['release_row_index']),'sidecar_position':i,'frames':int(r['window']['frame_count'])} for i,r in enumerate(light) if r['index']['split']=='train'),key=lambda x:x['source_row'])
 if len(entries)!=4613 or sum(e['frames'] for e in entries)!=2560614:raise ValueError('parallel token plan population mismatch')
 target=sum(e['frames'] for e in entries)/args.shards;chunks=[];current=[];frames=0
 for e in entries:
  if current and frames>=target and len(chunks)<args.shards-1:chunks.append(current);current=[];frames=0
  current.append(e);frames+=e['frames']
 chunks.append(current)
 if len(chunks)!=args.shards:raise ValueError(f'expected{args.shards} token shards got{len(chunks)}')
 rng=np.random.default_rng(43);shards=[]
 for i,chunk in enumerate(chunks):
  state=rng.bit_generator.state
  for e in chunk:rng.normal(0.0,.05,size=(e['frames'],32))
  shards.append({'shard_index':i,'rng_state':state,'entries':chunk,'rows':len(chunk),'frames':sum(e['frames'] for e in chunk)})
 plan={'contract':PLAN_CONTRACT,'created_at':datetime.now(timezone.utc).isoformat(),'source_dataset':str(args.source_dataset.resolve()),'sidecar':str(args.sidecar.resolve()),'sidecar_verification':str(args.sidecar_verification.resolve()),'sidecar_verification_sha256':args.sidecar_verification_sha256,'norm_dir':str(args.norm_dir.resolve()),'norm_provenance':str(args.norm_provenance.resolve()),'norm_stats_sha256':sha256(args.norm_dir/'norm_stats.json'),'state_noise_std':.05,'augmentation_seed':43,'max_token_len':256,'rows':4613,'frames':2560614,'shard_count':args.shards,'code':code_identity(),'shards':shards}
 path=out/'token_audit_plan.json';atomic_json(path,plan);print(json.dumps({'plan':str(path),'sha256':sha256(path),'rows':4613,'frames':2560614,'shards':args.shards},sort_keys=True))
def run_shard(args):
 plan,plan_sha=load_plan(args.plan);shard=plan['shards'][args.shard_index];store=store_from(plan);stats=normalize.load(Path(plan['norm_dir']));q01=np.asarray(stats['state'].q01,dtype=np.float32);q99=np.asarray(stats['state'].q99,dtype=np.float32);aq01=np.asarray(stats['actions'].q01,dtype=np.float32);aq99=np.asarray(stats['actions'].q99,dtype=np.float32);qvalid=(q99[:28]-q01[:28])>1e-6;sr=q99[:32]-q01[:32];ar=aq99-aq01;scale=np.where((sr>1e-6)&(ar>1e-6),sr/np.maximum(ar,1e-8),0.0);dims=np.asarray([0,1,2,*range(6,28)],dtype=np.int64)
 tokenizer=PaligemmaTokenizer(max_len=4096);fk=Native28FingertipFK();source=lance.dataset(plan['source_dataset'],version=store.source_dataset_version);rng=np.random.default_rng();rng.bit_generator.state=shard['rng_state'];clean_hist={};aug_hist={};clean_max=aug_max=-1;clean_ex=[];aug_ex=[];total=co=ao=0;noise_sq=0.;noise_count=0;corr_sq=np.zeros(32,dtype=np.float64);started=time.monotonic()
 for ordinal,e in enumerate(shard['entries']):
  side=store.dataset.take([e['sidecar_position']],columns=['index','state','prompt','provenance']).to_pylist()[0];raw=source.take([e['source_row']],columns=['index','objects','row_payload_sha256']).to_pylist()[0]
  if raw['index']['uuid']!=side['index']['uuid'] or raw['row_payload_sha256']!=side['provenance']['source_row_payload_sha256']:raise ValueError(f'parallel token join mismatch row{e["source_row"]}')
  states=np.asarray(side['state'],dtype=np.float32);normed=(states-q01)/(q99-q01+1e-6)*2-1;obj=raw['objects'][0];pos=np.asarray(obj['pos'],dtype=np.float32);quat=np.asarray(obj['quat_wxyz'],dtype=np.float32);name=side['index']['object'];prompt=side['prompt']
  for frame,state in enumerate(normed):
   cl=record_length(tokenizer,prompt,state);clean_hist[cl]=clean_hist.get(cl,0)+1;co+=int(cl>256);ex={'release_row_index':e['source_row'],'source_frame':frame,'prompt':prompt,'token_length':cl};clean_max,clean_ex=update_max(clean_max,clean_ex,cl,ex)
   noise=rng.normal(0,.05,size=32).astype(np.float32);aug=state.copy();aug[:28][qvalid]+=noise[:28][qvalid];vn=noise[:28][qvalid];noise_sq+=float(np.sum(vn*vn));noise_count+=vn.size;rq=q01[:28]+(aug[:28]+1)*.5*(q99[:28]-q01[:28]+1e-6);tw=fk(object_name=name,hand_qpos=rq,object_position=pos[frame],object_quaternion_wxyz=quat[frame]);tb=C.fingertips_in_collision_box_frame(tw,pos[frame],C.quaternion_wxyz_to_matrix(quat[frame]),name).reshape(-1);aug[34:49]=((tb-q01[34:49])/(q99[34:49]-q01[34:49]+1e-6)*2-1).astype(np.float32);al=record_length(tokenizer,prompt,aug);aug_hist[al]=aug_hist.get(al,0)+1;ao+=int(al>256);aex={**ex,'token_length':al};aug_max,aug_ex=update_max(aug_max,aug_ex,al,aex);corr=np.zeros(32,dtype=np.float32);corr[dims]=-noise[dims]*scale[dims];corr_sq+=np.square(corr,dtype=np.float64);total+=1
  if ordinal%10==0 or ordinal+1==shard['rows']:atomic_json(args.plan.parent/'progress'/f'shard-{args.shard_index:03d}.json',{'completed_rows':ordinal+1,'total_rows':shard['rows'],'frames':total,'total_frames':shard['frames'],'elapsed_seconds':time.monotonic()-started})
 part={'contract':PART_CONTRACT,'plan_sha256':plan_sha,'shard_index':args.shard_index,'rows':shard['rows'],'frames':total,'clean_hist':clean_hist,'aug_hist':aug_hist,'clean_max':clean_max,'aug_max':aug_max,'clean_examples':clean_ex,'aug_examples':aug_ex,'clean_overflow':co,'aug_overflow':ao,'noise_square_sum':noise_sq,'noise_count':noise_count,'correction_square_sum':corr_sq.tolist(),'elapsed_seconds':time.monotonic()-started};atomic_json(args.plan.parent/'parts'/f'shard-{args.shard_index:03d}.json',part);print(json.dumps({'shard':args.shard_index,'rows':shard['rows'],'frames':total,'clean_max':clean_max,'aug_max':aug_max},sort_keys=True))
def aggregate(args):
 plan,plan_sha=load_plan(args.plan);parts=[]
 for i in range(plan['shard_count']):
  p=json.loads((args.plan.parent/'parts'/f'shard-{i:03d}.json').read_text());
  if p.get('contract')!=PART_CONTRACT or p.get('plan_sha256')!=plan_sha or p.get('shard_index')!=i:raise ValueError(f'token part{i} identity mismatch')
  parts.append(p)
 def hist(key):
  h={}
  for p in parts:
   for k,v in p[key].items():h[int(k)]=h.get(int(k),0)+int(v)
  return h
 ch,ah=hist('clean_hist'),hist('aug_hist');frames=sum(p['frames'] for p in parts);rows=sum(p['rows'] for p in parts)
 if (rows,frames)!=(4613,2560614) or sum(ch.values())!=frames or sum(ah.values())!=frames:raise ValueError('token aggregate population mismatch')
 cm=max(p['clean_max'] for p in parts);am=max(p['aug_max'] for p in parts);ce=[e for p in parts if p['clean_max']==cm for e in p['clean_examples']][:20];ae=[e for p in parts if p['aug_max']==am for e in p['aug_examples']][:20];ns=sum(p['noise_square_sum'] for p in parts);nc=sum(p['noise_count'] for p in parts);cs=np.sum([np.asarray(p['correction_square_sum']) for p in parts],axis=0);co=sum(p['clean_overflow'] for p in parts);ao=sum(p['aug_overflow'] for p in parts);stats=normalize.load(Path(plan['norm_dir']));q01=np.asarray(stats['state'].q01);q99=np.asarray(stats['state'].q99);dims=[0,1,2,*range(6,28)]
 result={'contract':AUDIT_CONTRACT,'status':'passed','created_at':datetime.now(timezone.utc).isoformat(),'state_contract':C.STATE_CONTRACT_ID,'action_contract':C.ACTION_CONTRACT_ID,'population':'Scheme-A train-only contact window','trajectory_count':rows,'audited_active_frames':frames,'profile_max_token_len':256,'minimum_token_length':min(ch),'maximum_token_length':cm,'headroom_at_maximum':256-cm,'overflow_count':co,'zero_truncation':co==0,'token_length_histogram':{str(k):ch[k] for k in sorted(ch)},'maximum_examples':ce,'augmentation':{'contract':'normalized_qpos28_sigma005_tipFK_action_residual28_v1','seed':43,'state_noise_std':.05,'target_noise_std':0.0,'samples':frames,'realized_sigma':float(np.sqrt(ns/nc)),'qpos_valid_dimensions':np.flatnonzero((q99[:28]-q01[:28])>1e-6).tolist(),'causal_recomputation':'qpos28 noise then pinned native28 MuJoCo FK tipXYZ15; contact/lift/force/velocity/age clean','action_compensation_dimensions':dims,'action_compensation_rule':'normalized residual -= qpos_noise * state_quantile_range/action_quantile_range','action_compensation_rms_by_dim':np.sqrt(cs/frames).tolist(),'minimum_token_length':min(ah),'maximum_token_length':am,'headroom_at_maximum':256-am,'overflow_count':ao,'zero_truncation':ao==0,'token_length_histogram':{str(k):ah[k] for k in sorted(ah)},'maximum_examples':ae},'source_dataset':plan['source_dataset'],'sidecar':plan['sidecar'],'sidecar_verification_sha256':plan['sidecar_verification_sha256'],'norm_stats_sha256':plan['norm_stats_sha256'],'norm_provenance_sha256':sha256(Path(plan['norm_provenance'])),'geometry_contract_sha256':C.GEOMETRY_CONTRACT_SHA256,'parallel_plan_sha256':plan_sha,'client_commit':plan['code']['client_commit'],'builder_sha256':plan['code']['parallel_builder_sha256'],'elapsed_seconds':sum(p['elapsed_seconds'] for p in parts)}
 atomic_json(args.output,result)
 if co or ao:raise ValueError(f'token overflow clean={co} aug={ao}')
 print(json.dumps({'token_audit':str(args.output),'sha256':sha256(args.output),'clean_max':cm,'augmented_max':am,'frames':frames},sort_keys=True))
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);a=s.add_parser('plan');
 for name in ['source_dataset','sidecar','sidecar_verification','norm_dir','norm_provenance','output_root']:a.add_argument('--'+name.replace('_','-'),type=Path,required=True)
 a.add_argument('--sidecar-verification-sha256',required=True);a.add_argument('--shards',type=int,default=16);b=s.add_parser('run-shard');b.add_argument('--plan',type=Path,required=True);b.add_argument('--shard-index',type=int,required=True);c=s.add_parser('aggregate');c.add_argument('--plan',type=Path,required=True);c.add_argument('--output',type=Path,required=True);x=p.parse_args();make_plan(x) if x.cmd=='plan' else run_shard(x) if x.cmd=='run-shard' else aggregate(x)
if __name__=='__main__':main()
