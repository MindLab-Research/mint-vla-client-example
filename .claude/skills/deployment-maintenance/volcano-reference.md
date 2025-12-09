# Volcano ML Platform Reference

## Instance Flavors

| Flavor | GPUs | Memory | Use Case |
|--------|------|--------|----------|
| `ml.g2a.xlarge` | 0 | 8GB | Head node (CPU only) |
| `ml.pni2l.7xlarge` | 1x A100 80GB | 80GB | Small inference |
| `ml.pni2l.14xlarge` | 2x A100 80GB | 160GB | Medium inference |
| `ml.pni2l.28xlarge` | 8x A100 80GB | 640GB | Training/Large inference |

**GPU allocation is flexible.** Adjust `Flavor` and `--num-gpus` in configs as needed.

## Storage Mounts

```yaml
Storages:
    - Type: "Vepfs"
      MountPath: "/vePFS-Mindverse/share"
      SubPath: "share"
      ReadOnly: false
```

## Common YAML Parameters

| Parameter | Description |
|-----------|-------------|
| `ActiveDeadlineSeconds` | Max runtime in seconds (432000 = 5 days) |
| `DelayExitTimeSeconds` | Keep instance alive after completion |
| `ResourceQueueID` | Queue for resource allocation |

## CLI Commands

```bash
volc ml_task list                          # List tasks
volc ml_task logs -t TASK_ID -i worker_0   # View logs
volc ml_task cancel --id TASK_ID           # Cancel task
volc ml_task export --id TASK_ID           # Export config
volc ml_task get --id TASK_ID              # Task details
```

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
