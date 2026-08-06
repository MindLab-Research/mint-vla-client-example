#!/usr/bin/env python3
"""Select two representative non-sustained Task-B20K rows per gesture."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p:Path)->str:return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def medoid(pool:list[dict])->dict:
    features=np.asarray([[r['max'],r['multi_contact_fraction'],r['max_qvel']] for r in pool],dtype=float)
    center=np.median(features,axis=0);scale=np.std(features,axis=0);scale[scale<1e-9]=1.0
    scores=np.sum(np.abs((features-center)/scale),axis=1)
    return pool[int(np.argmin(scores))]
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--task-b-summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    analysis=json.loads(a.analysis.read_text());summary=json.loads(a.task_b_summary.read_text())
    if analysis.get('status')!='accepted' or analysis.get('row_count')!=115:raise ValueError('analysis is not accepted validation115')
    if summary.get('status')!='completed' or summary.get('row_count')!=115 or not summary.get('all_arrays_finite'):raise ValueError('rollout summary is not complete')
    results={int(r['row_index']):r for r in summary['results']};selected=[]
    for gesture in sorted({r['gesture'] for r in analysis['rows']}):
        failures=[r for r in analysis['rows'] if r['gesture']==gesture and not r['sustained']]
        gross=[r for r in failures if r['max']<.02];partial=[r for r in failures if r['max']>=.02]
        if not gross:raise ValueError(f'gesture{gesture} has no gross failure')
        first=medoid(gross)
        if partial:
            second=medoid(partial);second_type='partial_or_unstable_pickup'
        else:
            remaining=[r for r in gross if r['row']!=first['row']]
            second=max(remaining,key=lambda r:r['multi_contact_fraction']);second_type='contact_without_lift'
        for slot,(row,failure_type) in enumerate(((first,'gross_contact_acquisition_failure'),(second,second_type)),1):
            result=results[int(row['row'])]
            selected.append({'gesture':gesture,'slot':slot,'row_index':int(row['row']),'uuid':result['uuid'],'seed_uuid':result['seed_uuid'],'failure_type':failure_type,'selection_rule':'robust feature medoid within gesture/category' if slot==1 or partial else 'highest multi-contact fraction among remaining gross failures','max_lift_m':row['max'],'final_lift_m':row['final'],'longest_5cm_frames':row['longest'],'multi_contact_fraction':row['multi_contact_fraction'],'max_qvel':row['max_qvel'],'clip_abs_max':row['clip_abs_max'],'rollout_arrays':result['arrays'],'frame_window':result['frame_window'],'object_z_reference':result['object_z_reference']})
    if len(selected)!=12 or len({r['row_index'] for r in selected})!=12:raise ValueError('expected12 distinct selections')
    out={'contract':'state56_taskb20k_representative_failure_video_selection_v1','status':'accepted','checkpoint_step':20000,'population':'Task-B seed-disjoint validation115','selection_semantics':'two non-sustained rows per gesture: one typical gross no-lift failure plus one partial/unstable failure when available; gesture10 has no partial failure, so second row maximizes multi-contact among gross failures','analysis':str(a.analysis.resolve()),'analysis_sha256':sha(a.analysis),'task_b_summary':str(a.task_b_summary.resolve()),'task_b_summary_sha256':sha(a.task_b_summary),'video_count':12,'rows':selected,'claim_scope':'representative by deterministic feature-space rule, not worst-case or random sample'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'rows':[(r['gesture'],r['row_index'],r['failure_type']) for r in selected]},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
