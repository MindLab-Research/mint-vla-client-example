#!/usr/bin/env python3
"""Offline Task-B failure gallery from accepted State56 rollout arrays; no policy query."""
from __future__ import annotations
import argparse,hashlib,json
from io import BytesIO
from pathlib import Path
import cv2,imageio.v2 as imageio,lance,mujoco,numpy as np
from PIL import Image
from scripts.eval import manorl_native28_mode4_physics as physics

def sha(p:Path)->str:
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def decode(blob)->np.ndarray:return np.asarray(Image.open(BytesIO(blob)).convert('RGB'),dtype=np.uint8)
def resize(frame,w,h)->np.ndarray:return np.asarray(Image.fromarray(frame).resize((w,h),Image.Resampling.BILINEAR),dtype=np.uint8)
def label(frame,text)->np.ndarray:
 out=np.asarray(frame,dtype=np.uint8).copy();cv2.rectangle(out,(0,0),(min(out.shape[1],12+len(text)*9),28),(0,0,0),-1);cv2.putText(out,text,(7,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);return out
def header(comp,lines):
 h=78;canvas=np.zeros((comp.shape[0]+h,comp.shape[1],3),dtype=np.uint8);canvas[h:]=comp
 colors=((255,255,255),(120,255,120),(120,210,255))
 for i,(text,color) in enumerate(zip(lines,colors)):cv2.putText(canvas,text,(8,21+i*24),cv2.FONT_HERSHEY_SIMPLEX,.49,color,1,cv2.LINE_AA)
 return canvas
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--lance-dataset',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--fps',type=float,default=20);ap.add_argument('--stride',type=int,default=5);ap.add_argument('--panel-width',type=int,default=400);ap.add_argument('--panel-height',type=int,default=225);a=ap.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 sel=json.loads(a.selection.read_text());rows=sel['rows']
 if sel.get('status')!='accepted' or len(rows)!=12:raise ValueError('selection must be accepted12')
 a.output_dir.mkdir(parents=True);ds=lance.dataset(str(a.lance_dataset));reports=[]
 for metric in rows:
  idx=int(metric['row_index']);gesture=str(metric['gesture']);arrays=metric['rollout_arrays'];hp=Path(arrays['hand_state_sim']);pp=Path(arrays['object_position_sim']);qp=Path(arrays['object_quaternion_sim']);cp=Path(arrays['rollout_observation_contacts']);fp=Path(arrays['rollout_observation_log1p_force'])
  hand=np.asarray(np.load(hp),dtype=np.float64);pos=np.asarray(np.load(pp),dtype=np.float64);quat=np.asarray(np.load(qp),dtype=np.float64);contacts=np.asarray(np.load(cp),dtype=np.float64);forces=np.asarray(np.load(fp),dtype=np.float64)
  n=int(metric['frame_window']['frame_count']);start=int(metric['frame_window']['start_frame']);end=int(metric['frame_window']['end_frame'])
  if hand.shape!=(n,28) or pos.shape!=(n,3) or quat.shape!=(n,4) or contacts.shape!=(n-1,5) or forces.shape!=(n-1,5) or end-start+1!=n:raise ValueError(f'array shape mismatch row{idx}')
  if not all(np.isfinite(x).all() for x in (hand,pos,quat,contacts,forces)):raise ValueError(f'nonfinite row{idx}')
  row=ds.take([idx],columns=['image','wrist_image','objects']).to_pylist()[0];source_pos=np.asarray(row['objects'][0]['pos'],dtype=np.float64);source_lift=source_pos[:,2]-source_pos[0,2]
  if end>=len(row['image']) or len(row['wrist_image'])!=len(row['image']):raise ValueError(f'source frame mismatch row{idx}')
  offsets=list(range(0,n,a.stride));
  if offsets[-1]!=n-1:offsets.append(n-1)
  failure=str(metric['failure_type']);name=f"gesture{gesture}_slot{metric['slot']}_row{idx}_{failure}.mp4";out=a.output_dir/name
  tmp,model,data,renderer,object_addr,_,hand_addrs,_,_=physics.make_scene('cube1',a.panel_width,a.panel_height,physics=True,physics_timestep=physics.DT,create_renderer=True)
  try:
   with imageio.get_writer(str(out),fps=a.fps,macro_block_size=1,codec='libx264',quality=8) as writer:
    for off in offsets:
     data.qpos[hand_addrs]=hand[off];data.qpos[object_addr:object_addr+3]=pos[off];data.qpos[object_addr+3:object_addr+7]=quat[off]/np.linalg.norm(quat[off]);data.qvel[:]=0;data.ctrl[:]=hand[off];mujoco.mj_forward(model,data)
     mh,mw=physics.render_current_state(model,data,renderer);source_frame=start+off;sh=resize(decode(row['image'][source_frame]),a.panel_width,a.panel_height);sw=resize(decode(row['wrist_image'][source_frame]),a.panel_width,a.panel_height)
     comp=np.concatenate([np.concatenate([label(mh,'MODEL 20K saved rollout - head'),label(sh,'GRADE-A source reference - head')],axis=1),np.concatenate([label(mw,'MODEL 20K saved rollout - wrist'),label(sw,'GRADE-A source reference - wrist')],axis=1)],axis=0)
     obs=min(off,n-2);model_lift=float(pos[off,2]-metric['object_z_reference']);src_lift=float(source_lift[source_frame]);finger_count=int(np.sum(contacts[obs]>.5));force_sum=float(np.expm1(forces[obs]).sum())
     lines=[f"Task B 20K | gesture {gesture} | row {idx} | {failure}",f"t={off*.005:.3f}s | model lift={model_lift*100:.1f}cm | source lift={src_lift*100:.1f}cm | fingers={finger_count} | normal load={force_sum:.1f}N",f"model max={metric['max_lift_m']*100:.1f}cm | longest>=5cm={metric['longest_5cm_frames']*.005:.3f}s | playback 0.5x | OFFLINE arrays only"]
     writer.append_data(header(comp,lines))
  finally:
   renderer.close()
   if tmp is not None:tmp.cleanup()
  if not out.is_file() or out.stat().st_size==0:raise RuntimeError(f'missing video {out}')
  with imageio.get_reader(str(out)) as reader:count=sum(1 for _ in reader)
  if count!=len(offsets):raise RuntimeError(f'video frame mismatch row{idx}: {count}!={len(offsets)}')
  reports.append({'gesture':gesture,'slot':metric['slot'],'row_index':idx,'failure_type':failure,'path':str(out),'bytes':out.stat().st_size,'frames':count,'fps':a.fps,'source_frame_stride':a.stride,'playback_speed':a.fps/(200/a.stride),'sha256':sha(out),'policy_requeried':False,'source_arrays':{k:sha(Path(arrays[k])) for k in ('hand_state_sim','object_position_sim','object_quaternion_sim','rollout_observation_contacts','rollout_observation_log1p_force')}})
 manifest={'contract':'state56_taskb20k_representative_failure_video_manifest_v1','status':'accepted','render_mechanism':'offline_mujoco_forward_from_accepted_rollout_arrays plus authenticated source JPEG reference','policy_requeried':False,'selection':str(a.selection.resolve()),'selection_sha256':sha(a.selection),'video_count':len(reports),'videos':reports}
 mp=a.output_dir/'video_manifest.json';mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'manifest':str(mp),'sha256':sha(mp),'videos':len(reports)},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
