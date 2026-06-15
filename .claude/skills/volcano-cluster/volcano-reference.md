# Volcano ML Platform Reference

This reference is subordinate to `SKILL.md`. Node lifecycle must use the
Volcano Engine Python SDK through `scripts/tools/volcano_sdk_jobs.py` or the
topology reconciler in `mint_model_actor_supervisor`.

## Network Access

| Component | Internet | Proxy |
|-----------|----------|-------|
| SSH server | Via proxy | `localhost:1081` (HTTP), `localhost:1080` (SOCKS5) |
| Ray workers | None | N/A |

Workers must use packages pre-installed in the image or via PFS runtime paths.

## Instance Flavors

| Flavor | GPUs | Use Case |
|--------|------|----------|
| `ml.hpcpni2l.7xlarge` | 2x A800 80GB | Small training |
| `ml.hpcpni2l.14xlarge` | 4x A800 80GB | Medium training |
| `ml.hpcpni2l.28xlarge` | 8x A800 80GB | Standard Mint GPU worker |
| `ml.r3i.4xlarge` | 0 | Ray head node |

## Storage Mounts

```yaml
Storages:
    - Type: "Vepfs"
      MountPath: "/vePFS-Mindverse/share"
      SubPath: "share"
      ReadOnly: false
```

## PFS Directory Structure

| Path | Purpose |
|------|---------|
| `/vePFS-Mindverse/share/<owner>/mint-server/` | Explicit dev server checkout passed as `MINT_CODE_ROOT` |
| `/vePFS-Mindverse/share/mint/prod/mint-server/` | Prod server git checkout |
| `/vePFS-Mindverse/share/mint/dev/runtime/` | Dev runtime symlink |
| `/vePFS-Mindverse/share/mint/prod/runtime/` | Prod runtime symlink |
| `/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt` | Dev Ray head pointer |
| `/vePFS-Mindverse/share/mint/prod/ray/head-address/ray_head_ip.txt` | Prod Ray head pointer |
| `/vePFS-Mindverse/share/mint/<env>/runtime/topology_state.yaml` | Supervisor-written topology debug state |
| `/vePFS-Mindverse/share/huggingface/` | HuggingFace cache |
| `/vePFS-Mindverse/share/models/` | Model checkpoints |
| `/vePFS-Mindverse/share/dataset/` | Training datasets |

## SDK Commands

```bash
PY=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing list --name-contains mint-<env>-worker- --limit 200
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing instances --job-id <job_id>
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing submit-topology-node --config <topology.yaml> --alias mint-worker-0
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing stop --job-id <job_id>
```

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Worker remains queued | Check Volcano console capacity; keep one worker node per alias. |
| Worker fails to join | Verify the head IP file, runtime path, PFS mounts, and network reachability. |
| Cannot obtain worker IP | Use `instances --job-id` and `topology_state.yaml`; do not scrape provider logs. |
