---
name: deployment-maintenance
description: Guides server deployment to Volcano GPU cluster, code synchronization via Unison, and cluster maintenance including vLLM actor management, Ray cluster coordination, and health monitoring for the Tinker inference server.
---

# Server Deployment and Cluster Maintenance

## When to Use This Skill

- Deploy/restart Tinker server on Volcano cluster
- Manage code sync via Unison
- Manage Ray cluster (submit tasks, check status, cancel)
- Handle vLLM actor lifecycle
- Troubleshoot distributed inference issues
- Set up SSH tunnels for remote testing

## Skill Resources

- [configs/](configs/) - Ray cluster YAML configs + Unison profile
- [scripts/](scripts/) - Deployment and status scripts
- [volcano-reference.md](volcano-reference.md) - Instance flavors, storage, CLI commands

## Code Synchronization (Unison)

**NEVER** manually sync code (no rsync, scp, git for syncing). Use Unison.

### Start Sync (Background)

```bash
unison volcano-tinker -repeat watch
```

Runs continuously, syncing on file changes.

### One-Time Sync

```bash
unison volcano-tinker
```

### Check Sync Status

```bash
pgrep -af "unison.*volcano-tinker"
```

### Stop Sync

```bash
pkill -f "unison.*volcano-tinker"
```

### Setup (First Time)

Copy profile to `~/.unison/`:

```bash
cp configs/volcano-tinker.prf ~/.unison/
```

Profile syncs `/home/yiwen/tinker_project` <-> `volcano:/root/tinker_project`.

Ignores: `*.pyc`, `__pycache__`, `.git`, `.venv`, `node_modules`.

## Server Management

### Environment Variables

```bash
HF_HUB_OFFLINE=1                    # Force offline mode
HF_HOME=/vePFS-Mindverse/share/huggingface
TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
TINKER_CHECKPOINT_DIR=<path>        # LoRA checkpoints (shared filesystem required)
```

### SSH Tunnel Setup

```bash
ssh -f -N -L 8000:localhost:8000 volcano
```

### Start Server

```bash
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'
```

### Stop Server

```bash
ssh volcano 'pkill -f "python scripts/run_server.py"'
```

### Health Check

```bash
curl -s http://localhost:8000/api/v1/healthz  # {"status":"ready"}
```

### View Logs

```bash
ssh volcano "tail -50 /tmp/tinker_server.log"
```

## vLLM Actor Management

| Operation | Time | Command |
|-----------|------|---------|
| First start | ~80s | Automatic on server start |
| Restart (actor reused) | ~2s | Kill server, restart |
| Kill actor | - | `curl -X POST http://localhost:8000/api/v1/kill_vllm` |
| Check status | - | `curl http://localhost:8000/api/v1/vllm_status` |

**When to kill actor:** Base model changed, OOM, need GPU memory.

## Ray Cluster

### Connect to Existing Cluster

```bash
ray start --address='192.168.47.158:6379'
```

Dashboard: http://192.168.47.158:8265

### Deploy New Cluster via Volcano

**Prerequisites:** Run `scripts/setup_volc_cli.sh` to install CLI, then `volc configure`.

#### Option 1: Simple (Single Node 8 GPU)

```bash
scripts/deploy_cluster.sh simple
# Or directly: volc ml_task submit -c configs/ray_cluster_8gpu.yaml
```

#### Option 2: Scalable (Separate Head + Workers)

```bash
scripts/deploy_cluster.sh scalable
```

Then follow printed instructions to get HEAD_IP and submit worker.

### Check Cluster Status

```bash
scripts/cluster_status.sh
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/healthz` | GET | Health check |
| `/api/v1/vllm_status` | GET | vLLM actor status |
| `/api/v1/kill_vllm` | POST | Kill vLLM actor |
| `/api/v1/create_session` | POST | Create session |
| `/api/v1/create_sampling_session` | POST | Create sampling session |
| `/api/v1/asample` | POST | Submit async sample |
| `/api/v1/retrieve_future` | POST | Poll result (408=pending, 200=ready) |

## Troubleshooting

### Unison not syncing
- Check process running: `pgrep -af "unison.*volcano-tinker"`
- Check SSH connectivity: `ssh volcano echo ok`
- View logs: `tail -100 ~/.unison/unison.log`
- Restart: `pkill -f "unison.*volcano-tinker" && unison volcano-tinker -repeat watch`

### Server won't start
- Check logs: `ssh volcano "tail -100 /tmp/tinker_server.log"`
- Verify model path exists
- Check Ray cluster connectivity

### vLLM OOM
- Kill actor: `curl -X POST http://localhost:8000/api/v1/kill_vllm`
- Restart server
- Monitor GPU on dashboard

See [volcano-reference.md](volcano-reference.md) for Volcano-specific troubleshooting.
