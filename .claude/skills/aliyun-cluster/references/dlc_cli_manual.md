# DLC CLI manual (PAI-DLC)

This is a condensed manual for the Aliyun PAI DLC CLI binary `dlc`.

Docs often use `./dlc`. If you install it into PATH (for example `/usr/local/bin/dlc`), use `dlc`.

## Installation

Download the latest Linux client and make it executable:
```bash
wget -O dlc "https://dlc-release.oss-cn-zhangjiakou.aliyuncs.com/console/public/latest/dlc"
chmod +x dlc
```

If you see certificate/CA errors:
```bash
sudo apt-get update && sudo apt-get install -y ca-certificates
```

## Authentication and config

`dlc` stores credentials and defaults in `~/.dlc/config`.

Command format:
```bash
dlc config --protocol https --access_id <ACCESS_KEY_ID> --access_key <ACCESS_KEY_SECRET> --endpoint <ENDPOINT> --region <REGION>
```

Notes:
- `endpoint` is region-specific, example: `pai-dlc.cn-beijing.aliyuncs.com`
- `region` is the region id, example: `cn-beijing`

Project note:
- This project stores `ALI_ACCESSKEY_ID`/`ALI_ACCESSKEY_SECRET` in `.env` at the repo root.

## Shell completion

Generate completion script for your shell:
```bash
dlc completion <bash|fish|powershell|zsh>
```

Example (bash):
```bash
source <(dlc completion bash)
```

## Getting help

```bash
dlc help
dlc help <command>
dlc <command> --help
```

## Job lifecycle

### List jobs / get job details

Command format:
```bash
dlc get job [JOB_ID] \
  [--workspace_id <WORKSPACE_ID>] \
  [--display_name <NAME_SUBSTRING>] \
  [--job_type <JOB_TYPE>] \
  [--status <JOB_STATUS>] \
  [--start_time <RFC3339_UTC>] \
  [--end_time <RFC3339_UTC>] \
  [--page_num <N>] \
  [--page_size <N>] \
  [--max_events_num <N>] \
  [--events] \
  [--events_only]
```

Operational notes:
- Use `dlc get job <JOB_ID>` to extract pod ids and pod IPs for multi-node jobs.
- `start_time`/`end_time` are UTC timestamps, example: `2022-08-04T02:09:32Z`.

### Stop a job

Command format:
```bash
dlc stop job <JOB_ID> [--force]
```

### Fetch pod logs

Command format:
```bash
dlc logs <JOB_ID> <POD_ID> \
  [--max_events_num <N>] \
  [--start_time <RFC3339_UTC>] \
  [--end_time <RFC3339_UTC>]
```

Operational notes:
- `POD_ID` comes from `dlc get job <JOB_ID>`.
- Default `--max_events_num` is 2000 in the official docs; use a smaller number when polling.

## Submitting jobs

All job types share a set of "common parameters". The official docs cover `tfjob`, `pytorchjob`, and `xgboostjob` in detail, plus separate docs for `rayjob`.

## CPFS mount (Mindverse PFS)

Aliyun PAI documents CPFS mounting in PAI via NAS-style URIs (example shown for DSW dynamic mount):
- `nas://<nas-endpoint>/`

For Mindverse BMCPFS on `cn-beijing`, mount the existing BMCPFS filesystem into the DLC job.

In this project environment:
- `--data_sources` can mount CPFS read-only (pods show `/vePFS-Mindverse ... (ro,...)`), which breaks checkpoint saving.
- Prefer `--data_source_uris` with a BMCPFS URI, which mounts CPFS read-write (pods show `/vePFS-Mindverse ... (rw,...)`).

```bash
--data_source_uris "bmcpfs://bmcpfs-03001407yug37qgafv7j5.cn-beijing/::/vePFS-Mindverse"
```

Operational check:
- `dlc logs <JOB_ID> <POD_ID>` should contain a mount line with `(rw,...)` for `/vePFS-Mindverse`.

References:
- "PAI挂载CPFS智算版文件系统" (CPFS docs)
- "为实例挂载存储的配置方法与管理" (PAI docs; shows `nas://...` URI format)

### Submission mode A: flags

```bash
dlc submit <job_type> [flags]
```

### Submission mode B: job parameter file

`--job_file` points to a text file containing `key=value` lines (keys match flag names):
```bash
dlc submit <job_type> --job_file <path>
```

### Common submit parameters (shared by tfjob/pytorchjob/xgboostjob)

Required:
- `--name`: job name
- `--command`: entrypoint command executed on nodes

Optional (selected; non-exhaustive but matches official "common parameters" list):
- `--data_sources`: comma-separated dataset ids
- `--code_source`: code source id
- `--code_branch`, `--code_commit`: code version selectors (with `--code_source`)
- `--thirdparty_libs`: comma-separated python deps
- `--thirdparty_lib_dir`: directory containing `requirements.txt`
- `--vpc_id`, `--switch_id`, `--security_group_id`: VPC networking
- `--job_file`: `key=value` file (takes priority)
- `--interactive`: interactive mode
- `--job_max_running_time_minutes`: max runtime (0 = unlimited)
- `--success_policy`: TFJob only; success condition policy
- `--envs`: `k=v,k2=v2` worker env vars
- `--tags`: `k=v,k2=v2` tags
- `--oversold_type`: idle-resource policy
- `--driver`: GPU driver version selector
- `--default_route`: VPC egress routing (`eth0`/`eth1`)
- `--priority`: job priority (1..9)
- `--exit_code_on_stopped`: exit code in interactive stop case
- `--job_reserved_minutes`: keep job resources after finish
- `--job_reserved_policy`: retention policy (`Always`/`OnFailure`/`OnSucceed`)

### tfjob-specific parameters

`tfjob` format:
```bash
dlc submit tfjob [flags]
```

Key tfjob parameters (official table):
- `--workspace_id` (required)
- `--chief` (bool), `--chief_image`, `--chief_spec`
- `--masters`, `--master_image`, `--master_spec`
- `--ps`, `--ps_image`, `--ps_spec`
- `--workers`, `--worker_image`, `--worker_spec`
- `--evaluators`, `--evaluator_image`, `--evaluator_spec`
- `--graphlearns`, `--graphlearn_image`, `--graphlearn_spec`

Private quota / dedicated resource group parameters include:
- `--resource_id`
- Role resource configs for CPU/GPU/memory/shared-memory:
  - `chief_cpu`, `chief_gpu`, `chief_gpu_type`, `chief_memory`, `chief_shared_memory`
  - `master_cpu`, `master_gpu`, `master_gpu_type`, `master_memory`, `master_shared_memory`
  - Patterned role params: `ps_*`, `worker_*`, `evaluator_*`, `graphlearn_*`

### pytorchjob-specific parameters

`pytorchjob` format:
```bash
dlc submit pytorchjob [flags]
```

Key pytorchjob parameters (official table):
- `--workspace_id` (required)
- `--masters`, `--master_image`, `--master_spec`
- `--workers`, `--worker_image`, `--worker_spec`

Private quota / dedicated resource group parameters include:
- `--resource_id`
- `--priority`
- Per-role resource configs:
  - `master_cpu`, `master_gpu`, `master_gpu_type`, `master_memory`, `master_shared_memory`
  - `worker_cpu`, `worker_gpu`, `worker_gpu_type`, `worker_memory`, `worker_shared_memory`

### xgboostjob-specific parameters

`xgboostjob` format:
```bash
dlc submit xgboostjob [flags]
```

Key xgboostjob parameters (official table):
- `--workspace_id` (required)
- `--masters`, `--master_image`, `--master_spec`
- `--workers`, `--worker_image`, `--worker_spec`

Private quota / dedicated resource group parameters include:
- `--resource_id`
- `--priority`
- Per-role resource configs:
  - `master_cpu`, `master_gpu`, `master_gpu_type`, `master_memory`, `master_shared_memory`
  - `worker_cpu`, `worker_gpu`, `worker_gpu_type`, `worker_memory`, `worker_shared_memory`

### rayjob (Ray head/workers)

`rayjob` is documented as a separate use case. Minimal CLI shape:
```bash
dlc submit rayjob --name=<NAME> \
  --workers=<N> --worker_spec=<ECS_SPEC> --worker_image=<IMAGE> \
  --heads=<N> --head_spec=<ECS_SPEC> --head_image=<IMAGE> \
  --command="<entrypoint>" \
  --workspace_id=<WORKSPACE_ID>
```

Operational notes:
- The service creates a Ray head and workers, then submits a `ray job` entrypoint internally.
- Use `dlc get job <JOB_ID>` to obtain pod ids and IPs for debugging.

## Project recipe: 24-GPU Ray cluster as 4 pytorchjobs

This repo uses 4 separate `pytorchjob`s to build a Ray cluster (1 head + 3 workers).

Role selection:
- Set `MINT_RAY_ROLE=head|worker` (preferred over relying on DLC-injected `RANK`/`PAI_TASK_ROLE`).
- Worker jobs also need `HEAD_IP=<head_ip>` (or `MASTER_ADDR=<head_ip>`).

Entrypoint script (stored in this repo):
- `${ALIYUN_CODE_ROOT}/.claude/skills/aliyun-cluster/scripts/ray_entrypoint.sh`
