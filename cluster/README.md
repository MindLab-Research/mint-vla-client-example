# Ray Cluster Management for Volcano ML Platform

Configuration files and scripts for deploying Ray clusters on Volcano ML Platform.

## Prerequisites

Install the Volcano CLI:
```bash
sh -c "$(curl -fsSL https://ml-platform-public-examples-cn-beijing.tos-cn-beijing.volces.com/cli-binary/install.sh)"
export PATH=$HOME/.volc/bin:$PATH
```

Configure credentials:
```bash
volc configure
# Enter AK, SK, and region (cn-beijing)
```

## Deployment Options

### Option 1: Combined Head + Worker (Simple)

Single node with 8 GPUs running both Ray head and workers.

```bash
volc ml_task submit -c ray_cluster_8gpu.yaml
```

### Option 2: Separate Head + Workers (Scalable)

CPU-only head node (persistent) + ephemeral GPU workers.

```bash
# 1. Start head node
volc ml_task submit -c ray_master.yaml

# 2. Get head node IP
volc ml_task list  # Select ray-master, note the IP

# 3. Update ray_worker_8gpu.yaml with HEAD_IP
sed -i "s/HEAD_IP/192.168.x.x/" ray_worker_8gpu.yaml

# 4. Start GPU worker
volc ml_task submit -c ray_worker_8gpu.yaml
```

## Cluster Operations

### List Running Tasks
```bash
volc ml_task list
```

### View Task Logs
```bash
volc ml_task logs -t TASK_ID -i worker_0
```

### Cancel Task
```bash
volc ml_task cancel --id TASK_ID
```

### Export Task Config
```bash
volc ml_task export --id TASK_ID
```

## Configuration Reference

### Instance Flavors

| Flavor | GPUs | Memory | Use Case |
|--------|------|--------|----------|
| `ml.g2a.xlarge` | 0 | 8GB | Head node |
| `ml.pni2l.28xlarge` | 8x A100 80GB | 640GB | Training/Inference |

### Storage Mounts

```yaml
Storages:
    - Type: "Vepfs"
      MountPath: "/vePFS-Mindverse/share"  # Shared filesystem
      SubPath: "share"
      ReadOnly: false
```

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| `ActiveDeadlineSeconds` | Max runtime in seconds (432000 = 5 days) |
| `DelayExitTimeSeconds` | Keep instance alive after completion |
| `ResourceQueueID` | Queue for resource allocation |

## Connecting to Ray Cluster

From the head node or any worker:
```bash
# Connect via Ray address
ray.init(address='auto')

# Or explicit address
ray.init(address='HEAD_IP:6379')
```

Dashboard available at: `http://HEAD_IP:8265`

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
