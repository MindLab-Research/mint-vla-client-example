#!/usr/bin/env python3
"""Summarize Task-A/Task-B State56 step20K closed-loop lift success."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

THRESHOLD=0.05
SUSTAINED_N=100
TAIL_N=100
TAIL_REQUIRED=90


def sha(path:Path)->str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def longest_true(values:np.ndarray)->int:
    best=run=0
    for value in values:
        run=run+1 if bool(value) else 0;best=max(best,run)
    return best

def lift_metrics(lift:np.ndarray)->dict:
    lift=np.asarray(lift,dtype=np.float64)
    if lift.ndim!=1 or len(lift)<TAIL_N or not np.isfinite(lift).all():raise ValueError("lift must be finite 1D with at least100 frames")
    above=lift>=THRESHOLD;longest=longest_true(above);tail=above[-TAIL_N:]
    transient=bool(above.any());sustained=bool(longest>=SUSTAINED_N);tail_stable=bool(above[-1] and int(tail.sum())>=TAIL_REQUIRED)
    rank=3 if tail_stable else 2 if sustained else 1 if transient else 0
    return {"transient_5cm":transient,"sustained_5cm":sustained,"tail_stable_5cm":tail_stable,"tier":"tail_stable_5cm" if rank==3 else "sustained_5cm" if rank==2 else "transient_5cm" if rank==1 else "failure","tier_rank":rank,"frame_count":len(lift),"max_lift_m":float(lift.max()),"final_lift_m":float(lift[-1]),"longest_5cm_frames":longest,"longest_5cm_seconds":longest*.005,"tail_5cm_count":int(tail.sum()),"tail_5cm_fraction":float(tail.mean()),"tail_lift_mean_m":float(lift[-TAIL_N:].mean()),"tail_lift_std_m":float(lift[-TAIL_N:].std())}

def rate(rows:list[dict],metric:str)->dict:
    n=len(rows);count=sum(bool(r[metric]) for r in rows);return {"count":count,"denominator":n,"rate":count/n if n else None}

def group_bootstrap(rows:list[dict],metric:str,seed:int=20260806,reps:int=10000)->list[float]|None:
    groups={}
    for row in rows:groups.setdefault(row["seed_uuid"],[]).append(row)
    if len(groups)<2:return None
    keys=sorted(groups);rng=np.random.default_rng(seed);values=[]
    for _ in range(reps):
        sampled=rng.choice(keys,size=len(keys),replace=True);pool=[r for key in sampled for r in groups[str(key)]];values.append(np.mean([bool(r[metric]) for r in pool]))
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]

def load_population(path:Path,label:str)->tuple[dict,list[dict]]:
    summary=json.loads(path.read_text())
    if summary.get("status")!="completed" or summary.get("checkpoint_step")!=20000 or not summary.get("all_rows_complete") or not summary.get("all_arrays_finite") or not summary.get("action_session_released"):
        raise ValueError(f"incomplete evaluation summary: {path}")
    rows=[]
    for result in summary["results"]:
        positions=np.load(result["arrays"]["object_position_sim"])
        if positions.shape!=(result["frame_window"]["frame_count"],3) or not np.isfinite(positions).all():raise ValueError(f"invalid object array row {result['row_index']}")
        lift=positions[:,2].astype(np.float64)-float(result["object_z_reference"])
        metrics=lift_metrics(lift)
        rows.append({"population":label,"row_index":int(result["row_index"]),"uuid":result["uuid"],"seed_uuid":result["seed_uuid"],"object_name":result["object_name"],"gesture":result["gesture"],**metrics})
    if len(rows)!=summary["row_count"] or len({r["row_index"] for r in rows})!=len(rows):raise ValueError("evaluation row identity mismatch")
    return summary,rows

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--task-a-summary",type=Path,required=True);ap.add_argument("--task-b-summary",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
    sa,ra=load_population(a.task_a_summary,"task_a_train15_seen");sb,rb=load_population(a.task_b_summary,"task_b_validation115_seed_disjoint")
    if len(ra)!=15 or len(rb)!=115:raise ValueError("formal population count mismatch")
    a_by={r["row_index"]:r for r in ra};b_by={r["row_index"]:r for r in rb};common_ids=sorted(set(a_by)&set(b_by));
    if len(common_ids)!=5:raise ValueError(f"expected five matched cube1/03 rows, got {common_ids}")
    common_a=[a_by[i] for i in common_ids];common_b=[b_by[i] for i in common_ids]
    metrics=("tail_stable_5cm","sustained_5cm","transient_5cm")
    def summarize(rows):return {m:{**rate(rows,m),"seed_group_bootstrap_ci95":group_bootstrap(rows,m)} for m in metrics}
    paired={m:{"task_a_only":sum(a_by[i][m] and not b_by[i][m] for i in common_ids),"task_b_only":sum(b_by[i][m] and not a_by[i][m] for i in common_ids),"both":sum(a_by[i][m] and b_by[i][m] for i in common_ids),"neither":sum(not a_by[i][m] and not b_by[i][m] for i in common_ids)} for m in metrics}
    output={"contract":"mano_state56_native28_aug010_20k_success_comparison_v1","status":"accepted","checkpoint_step":20000,"threshold_m":THRESHOLD,"definitions":{"transient_5cm":"maximum lift>=0.05m","sustained_5cm":"at least100 consecutive5ms frames lift>=0.05m","tail_stable_5cm":"final lift>=0.05m and at least90 of final100 frames>=0.05m"},"task_a":{"scope":"all15 optimizer-train cube1/03 rows; seen-data fit evidence only","summary":str(a.task_a_summary.resolve()),"summary_sha256":sha(a.task_a_summary),"rates":summarize(ra)},"task_b":{"scope":"all115 whole-seed-group-disjoint cube1 validation rows; validation not untouched final test","summary":str(a.task_b_summary.resolve()),"summary_sha256":sha(a.task_b_summary),"rates":summarize(rb)},"common_five":{"row_indices":common_ids,"scope":"same cube1/03 rows; seen by Task A and seed-disjoint validation for Task B, so outcomes are paired but generalization semantics differ","task_a_rates":summarize(common_a),"task_b_rates":summarize(common_b),"paired_outcomes":paired},"all_arrays_finite":True,"all_rows_complete":True,"videos_generated":False,"claim_scope":"step20K only; no claim about earlier checkpoint optimum or Task-A held-out generalization"}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    rows=ra+rb
    fields=list(rows[0]);
    with (a.output_dir/"rows.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (a.output_dir/"rows.json").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n")
    p=a.output_dir/"quality_summary.json";p.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    completion={"status":"accepted","checkpoint_step":20000,"task_a_row_count":15,"task_b_row_count":115,"common_row_count":5,"quality_summary_sha256":sha(p),"rows_csv_sha256":sha(a.output_dir/"rows.csv"),"rows_json_sha256":sha(a.output_dir/"rows.json")};q=a.output_dir/"completion_summary.json";q.write_text(json.dumps(completion,indent=2,sort_keys=True)+"\n");print(json.dumps({"output":str(p),"sha256":sha(p),"task_a_tail":output["task_a"]["rates"]["tail_stable_5cm"],"task_b_tail":output["task_b"]["rates"]["tail_stable_5cm"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
