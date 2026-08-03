#!/usr/bin/env python3
"""Build a deterministic seed-grouped train/validation/held-out replay split."""
from __future__ import annotations
import argparse, collections, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPLIT_SALT="state54-replay-split-v1"
SPLITS=("train","validation","held_out")

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def digest_values(values:list[Any])->str:return hashlib.sha256(",".join(map(str,values)).encode()).hexdigest()
def atomic_json(path:Path,value:Any)->None:
 tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");tmp.replace(path)

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--data-release",type=Path,required=True);p.add_argument("--data-release-sha256",required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 release_path=a.data_release.resolve();actual=sha(release_path)
 if actual!=a.data_release_sha256.lower():raise ValueError(f"data release SHA mismatch: {actual}")
 release=json.loads(release_path.read_text());
 if release.get("status")!="accepted" or release.get("row_count")!=1014:raise ValueError("unexpected data release")
 population_path=Path(release["population"]);gesture_path=Path(release["gesture_index"])
 if sha(population_path)!=release["population_sha256"]:raise ValueError("population SHA mismatch")
 if sha(gesture_path)!=release["gesture_index_sha256"]:raise ValueError("gesture SHA mismatch")
 population=json.loads(population_path.read_text())["entries"]
 gesture={int(e["row_index"]):e for e in json.loads(gesture_path.read_text())["entries"]}
 rows=[int(e["row_index"]) for e in population]
 if digest_values(rows)!=release["row_indices_sha256"]:raise ValueError("population row digest mismatch")
 groups:dict[str,list[dict[str,Any]]]=collections.defaultdict(list)
 for e in population:groups[str(e["seed_uuid"])].append(e)
 strata:dict[tuple[str,str],list[tuple[str,list[dict[str,Any]]]]]=collections.defaultdict(list)
 for seed,entries in groups.items():
  objects={e["object_name"] for e in entries};gestures={gesture[int(e["row_index"])]["gesture"] for e in entries}
  if len(objects)!=1 or len(gestures)!=1:raise ValueError(f"seed group crosses stratum: {seed}")
  strata[(next(iter(objects)),next(iter(gestures)))].append((seed,entries))
 assigned={name:[] for name in SPLITS};group_assignment={}
 strata_plan={}
 for stratum in sorted(strata):
  gs=sorted(strata[stratum],key=lambda item:hashlib.sha256(f"{SPLIT_SALT}|{item[0]}".encode()).hexdigest())
  n=len(gs);validation_groups=max(1,round(.1*n)) if n>=3 else 0;held_groups=max(1,round(.1*n)) if n>=2 else 0
  while validation_groups+held_groups>=n and validation_groups>0:validation_groups-=1
  while validation_groups+held_groups>=n and held_groups>0:held_groups-=1
  cuts={"held_out":gs[:held_groups],"validation":gs[held_groups:held_groups+validation_groups],"train":gs[held_groups+validation_groups:]}
  for split,items in cuts.items():
   for seed,entries in items:
    if seed in group_assignment:raise ValueError(f"duplicate group assignment: {seed}")
    group_assignment[seed]=split;assigned[split].extend(entries)
  strata_plan[f"{stratum[0]}|{stratum[1]}"]={"total_groups":n,"train_groups":len(cuts["train"]),"validation_groups":len(cuts["validation"]),"held_out_groups":len(cuts["held_out"])}
 for split in SPLITS:assigned[split].sort(key=lambda e:int(e["row_index"]))
 split_rows={s:[int(e["row_index"]) for e in assigned[s]] for s in SPLITS}
 split_groups={s:sorted({str(e["seed_uuid"]) for e in assigned[s]}) for s in SPLITS}
 row_sets={s:set(v) for s,v in split_rows.items()};group_sets={s:set(v) for s,v in split_groups.items()}
 checks={
  "row_sets_pairwise_disjoint":all(row_sets[a].isdisjoint(row_sets[b]) for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]),
  "seed_groups_pairwise_disjoint":all(group_sets[a].isdisjoint(group_sets[b]) for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]),
  "complete_row_coverage":set().union(*row_sets.values())==set(rows),
  "complete_seed_group_coverage":set().union(*group_sets.values())==set(groups),
  "all_object_gesture_strata_in_all_splits":True,
 }
 for stratum,items in strata.items():
  seeds={seed for seed,_ in items}
  if len(seeds)>=3:
   checks["all_object_gesture_strata_in_all_splits"] &= all(bool(seeds&group_sets[s]) for s in SPLITS)
 if not all(checks.values()):raise ValueError(f"split validation failed: {checks}")
 a.output_dir.mkdir(parents=True,exist_ok=False)
 split_summary={}
 for split in SPLITS:
  rows_path=a.output_dir/f"{split}_rows.csv";rows_path.write_text(",".join(map(str,split_rows[split]))+"\n")
  object_counts=collections.Counter(e["object_name"] for e in assigned[split])
  strata_counts=collections.Counter(f"{e['object_name']}|{gesture[int(e['row_index'])]['gesture']}" for e in assigned[split])
  split_summary[split]={"row_count":len(split_rows[split]),"row_fraction":len(split_rows[split])/len(rows),"seed_group_count":len(split_groups[split]),"source_frame_count":sum(int(e["frame_count"]) for e in assigned[split]),"active_frame_count":sum(int(e["active_frame_count"]) for e in assigned[split]),"row_indices_sha256":digest_values(split_rows[split]),"seed_groups_sha256":digest_values(split_groups[split]),"rows_csv":str(rows_path.resolve()),"rows_csv_sha256":sha(rows_path),"object_counts":dict(sorted(object_counts.items())),"object_gesture_counts":dict(sorted(strata_counts.items()))}
 manifest={"schema_version":1,"status":"accepted","created_at":datetime.now(timezone.utc).isoformat(),"split_id":"state54_replay_seed_grouped_stratified_v1","algorithm":{"group_unit":"seed_uuid","stratum":"object_name|gesture","ordering":f"sha256({SPLIT_SALT}|seed_uuid)","group_quotas":"per-stratum round(10%) validation and held-out with >=1 each when >=3 groups; remainder train","salt":SPLIT_SALT},"data_release":str(release_path),"data_release_sha256":actual,"population_row_indices_sha256":release["row_indices_sha256"],"population_row_count":len(rows),"population_seed_group_count":len(groups),"strata_plan":strata_plan,"splits":split_summary,"checks":checks,"group_assignment":dict(sorted(group_assignment.items()))}
 atomic_json(a.output_dir/"split_manifest.json",manifest)
 validation={"status":"accepted","split_manifest":str((a.output_dir/"split_manifest.json").resolve()),"split_manifest_sha256":sha(a.output_dir/"split_manifest.json"),"checks":checks,"row_counts":{s:len(split_rows[s]) for s in SPLITS},"seed_group_counts":{s:len(split_groups[s]) for s in SPLITS},"total_rows":sum(map(len,split_rows.values())),"total_seed_groups":sum(map(len,split_groups.values()))}
 atomic_json(a.output_dir/"validation.json",validation)
 print(json.dumps({"status":"accepted","output":str(a.output_dir.resolve()),"manifest_sha256":sha(a.output_dir/"split_manifest.json"),"validation_sha256":sha(a.output_dir/"validation.json"),"row_counts":validation["row_counts"],"seed_group_counts":validation["seed_group_counts"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
