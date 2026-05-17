# Volcano ML Platform Reference

## Network Access

| Component | Internet | Proxy |
|-----------|----------|-------|
| SSH server | Via proxy | `localhost:1081` (HTTP), `localhost:1080` (SOCKS5) |
| Ray workers | None | N/A |

Workers must use packages pre-installed in image or via PFS PYTHONPATH.

## Instance Flavors

| Flavor | GPUs | Memory | Use Case |
|--------|------|--------|----------|
| `ml.hpcpni2l.7xlarge` | 2x A800 80GB | 490 GiB | Small training (RDMA) |
| `ml.hpcpni2l.14xlarge` | 4x A800 80GB | 980 GiB | Medium training (RDMA) |
| `ml.hpcpni2l.28xlarge` | 8x A800 80GB | 1960 GiB | Large training/MoE (RDMA) |

**GPU allocation is flexible.** Adjust `Flavor` and `--num-gpus` in configs as needed.

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
| `/vePFS-Mindverse/share/mint/dev/mint-server/` | Dev server git checkout |
| `/vePFS-Mindverse/share/mint/prod/mint-server/` | Prod server git checkout |
| `/vePFS-Mindverse/share/mint/dev/runtime/` | Dev runtime symlink |
| `/vePFS-Mindverse/share/mint/prod/runtime/` | Prod runtime symlink |
| `/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt` | Dev Ray head pointer |
| `/vePFS-Mindverse/share/mint/prod/ray/head-address/ray_head_ip.txt` | Prod Ray head pointer |
| `/vePFS-Mindverse/share/huggingface/` | HuggingFace cache (models, tokenizers) |
| `/vePFS-Mindverse/share/models/` | Model checkpoints |
| `/vePFS-Mindverse/share/dataset/` | Training datasets |

## Common YAML Parameters

| Parameter | Description |
|-----------|-------------|
| `ActiveDeadlineSeconds` | Max runtime in seconds (432000 = 5 days) |
| `DelayExitTimeSeconds` | Keep instance alive after completion |
| `ResourceQueueID` | Queue for resource allocation |

## CLI Commands

**Important:** Use `--output json` to avoid interactive TUI mode.

```bash
volc ml_task list --output json                 # List tasks
volc ml_task submit -c config.yaml --output json  # Submit new task
volc ml_task cancel --id TASK_ID                # Cancel task (no --output json support on current CLI)
volc ml_task logs -t TASK_ID -i worker_0        # View logs (find Ray IP here)
```

**Finding Ray head IP:** Check logs for "Local node IP: 192.x.x.x"

## Troubleshooting

### Task stuck in Queue
- Check queue capacity: `volc ml_task list`
- Try lower priority or different queue

### Worker fails to join
- Verify HEAD_IP is correct
- Check network connectivity between instances
- Ensure both use same image version

### Out of memory
- Reduce batch size
- Enable gradient checkpointing
- Use parameter offloading
