#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path


def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 r=a.run_root;manifest=r/'launch_manifest.json'
 if sha(manifest)!=a.manifest_sha256:raise ValueError('manifest SHA mismatch')
 m=json.loads(manifest.read_text());result=json.loads((r/'result.json').read_text())
 expected_contract=m['data_contract_sha256'];expected_norm=m['norm_sha256']
 if result.get('state_contract')!='mano_object_dynamics_state54_v1':raise ValueError('state contract mismatch')
 if result.get('state54_data_contract',{}).get('sha256')!=expected_contract:raise ValueError('data contract mismatch')
 if result.get('norm_sha_actual')!=expected_norm or result.get('norm_sha_expected')!=expected_norm:raise ValueError('norm mismatch')
 if result.get('completed_step')!=20 or result.get('batch_size')!=8 or result.get('seed')!=42:raise ValueError('step/batch/seed mismatch')
 if result.get('state_noise_std_normalized')!=0.05 or result.get('target_noise_std_normalized')!=0:raise ValueError('StateAug0.05 contract mismatch')
 aug=result.get('augmentation',{})
 if aug.get('samples')!=160 or aug.get('requested_sigma')!=0.05 or not (0.04<=float(aug.get('realized_sigma',-1))<=0.06) or aug.get('causal_derived_valid_coordinates')!=2400 or aug.get('causal_derived_bin_changed_fraction',0)<=0:
  raise ValueError('StateAug0.05 realization mismatch')
 for key in ('clean_token_length_range','augmented_token_length_range'):
  values=aug.get(key) or []
  if len(values)!=2 or max(values)>256:raise ValueError(f'{key} exceeds max256: {values}')
 if result.get('skip_final_save') is not False:raise ValueError('final save was skipped')
 if result.get('row_selection',{}).get('count')!=813 or result.get('window_summary',{}).get('selected_frame_count')!=423450:raise ValueError('train population mismatch')
 sampling=result.get('sampling',{})
 if sampling.get('strategy')!='coverage' or sampling.get('valid_rows')!=813:raise ValueError('sampling mismatch')
 metrics=[]
 for line in (r/'metrics.jsonl').read_text().splitlines():
  if line.strip():metrics.append(json.loads(line))
 if len(metrics)!=20:raise ValueError(f'metrics lines={len(metrics)}')
 for item in metrics:
  if not math.isfinite(float(item.get('loss',float('nan')))):raise ValueError('nonfinite loss')
  if not math.isfinite(float(item.get('metrics',{}).get('grad_norm:mean',float('nan')))):raise ValueError('nonfinite grad')
 save=result.get('save_result',{});sampler=Path(save.get('filesystem_path',''))
 files={'metadata':sampler/'metadata.json','profile':sampler/'mint_pi05_profile.json','params_metadata':sampler/'params/_METADATA','norm_provenance':sampler/'mint_pi05_norm_provenance.json','embedded_norm':sampler/'assets/physical-intelligence/libero/norm_stats.json'}
 for name,path in files.items():
  if not path.is_file() or path.stat().st_size==0:raise ValueError(f'missing sampler {name}: {path}')
 meta=json.loads(files['metadata'].read_text());profile=json.loads(files['profile'].read_text());normprov=json.loads(files['norm_provenance'].read_text())
 if meta.get('step')!=20 or meta.get('optimizer_present') is not False:raise ValueError('sampler metadata mismatch')
 expected_profile={'profile_id':'pi05_action_lora_r16_state54_v1','state_dim':54,'action_dim':32,'action_horizon':10,'max_token_len':256,'fail_on_token_truncation':True,'expected_trainable_count':13224992}
 for k,v in expected_profile.items():
  if profile.get(k)!=v:raise ValueError(f'profile {k} mismatch')
 if normprov.get('sha256')!=expected_norm or normprov.get('state_dim')!=54 or normprov.get('action_dim')!=32:raise ValueError('norm provenance mismatch')
 if sha(files['embedded_norm'])!=expected_norm:raise ValueError('embedded norm bytes mismatch')
 report={'schema_version':1,'status':'accepted','protocol':'state54_replay_train_stateaug005_batch8_smoke20_v1','run_root':str(r.resolve()),'manifest_sha256':a.manifest_sha256,'data_contract_sha256':expected_contract,'norm_sha256':expected_norm,'steps':20,'optimizer_samples':160,'batch_size':8,'seed':42,'all_losses_finite':True,'all_grad_norms_finite':True,'first_loss':metrics[0]['loss'],'final_loss':metrics[-1]['loss'],'sampling_schedule_hash_prefix160':sampling.get('schedule_hash'),'sampler':str(sampler.resolve()),'sampler_metadata_sha256':sha(files['metadata']),'sampler_profile_sha256':sha(files['profile']),'sampler_params_metadata_sha256':sha(files['params_metadata']),'sampler_norm_provenance_sha256':sha(files['norm_provenance']),'embedded_norm_sha256':sha(files['embedded_norm']),'optimizer_present':False,'claim_scope':'optimizer/runtime/export integration only; not grasp quality'}
 a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,sort_keys=True));print('sha256',sha(a.output))
if __name__=='__main__':main()
