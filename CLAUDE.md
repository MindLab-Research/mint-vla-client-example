# tinker-server

## Code Synchronization

**IMPORTANT:** Code sync between local and server is handled by a background `unison` process. **NEVER** manually synchronize code (no rsync, scp, or git operations for syncing).

## Running the Server

```bash
HF_HUB_OFFLINE=1 \
HF_HOME=/vePFS-Mindverse/share/huggingface \
TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
python scripts/run_server.py
```

Environment variables:
- `HF_HUB_OFFLINE=1` - Force offline mode
- `HF_HOME` - Local HuggingFace cache directory
- `TINKER_MODEL_PATH` - Model snapshot directory
- `TINKER_CHECKPOINT_DIR` - LoRA checkpoint directory (must be shared filesystem in distributed deployments)

## Ray Cluster

Join cluster: `ray start --address='192.168.47.158:6379'`

Server auto-connects via `ray.init(address='auto')`. Dashboard: `http://192.168.47.158:8265`

## Fixed Issues

| Issue | Fix | Location |
|-------|-----|----------|
| max_tokens ignored | Override `generate()` to respect user param | `verl_inference.py:138-213` |
| EOS not detected | Add `stop_token_ids=[151645, 151643]` | `verl_inference.py:283`, `sampling.py:72-76` |
| Slow weight sync (60s) | Hot LoRA reload via shared engine (0.68s) | `session_manager.py:103-184`, `verl_inference.py:77-132` |

## Testing

```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_client.py
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/create_session` | POST | Create session (optional) |
| `/api/v1/create_sampling_session` | POST | Create sampling session |
| `/api/v1/asample` | POST | Submit async sample request |
| `/api/v1/retrieve_future` | POST | Poll result (408=pending, 200=ready) |
| `/api/v1/healthz` | GET | Health check |
| `/api/v1/vllm_status` | GET | Check vLLM actor status |
| `/api/v1/kill_vllm` | POST | Kill vLLM actor (forces reinit) |

## Distributed Architecture

```
Client Machine              API Server (volcano)           GPU Workers
──────────────              ────────────────────           ───────────
tinker-cookbook  ──HTTP──>  tinker-server:8000  ──Ray──>  TrainingWorker
                                   |                       vLLM Engine
                            SSH tunnel (8000)
```

**Data transfer:** Weights transferred via Ray object store, not file paths (different filesystems).

## Architecture Notes

### Inference Modes

**Named Sessions:** `create_sampling_session(model_path=...)` spawns dedicated engine (~60s init). Use for stable, long-running deployments.

**Ephemeral Sessions:** `save_weights_and_get_sampling_client()` uses per-training-session engine. First call ~60s, subsequent hot-reloads ~0.7s. Use for RL training loops.

### Persistent vLLM Actor

vLLM runs as detached Ray actor surviving server restarts:
- First start: ~80s (model load + CUDA graph capture)
- Subsequent restarts: ~2s (reuses actor)
- Kill when: base model changed, OOM, need GPU memory

## Remote Testing (volcano)

### Setup

```bash
ssh -f -N -L 8000:localhost:8000 volcano  # SSH tunnel (run once)
```

### Server Management

```bash
# Kill server
ssh volcano 'pkill -f "python scripts/run_server.py"'

# Start server (uses env vars from "Running the Server" section)
ssh volcano 'cd /root/tinker_project/tinker-server && nohup bash -c "HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 python scripts/run_server.py" > /tmp/tinker_server.log 2>&1 &'

# Check status
curl -s http://localhost:8000/api/v1/healthz  # Expected: {"status":"ready"}

# View logs
ssh volcano "tail -50 /tmp/tinker_server.log"
```

**Fast restart** (code changes): Kill server, restart. vLLM actor reused (~2s).

**Full restart** (vLLM changes): `curl -X POST http://localhost:8000/api/v1/kill_vllm`, then kill/restart server (~80s).

### Cookbook Tests

```bash
cd /home/yiwen/tinker_project/tinker-cookbook

# Arithmetic RL (~5 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    renderer_name="qwen3_instruct" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4

# Chat SL (~30-60 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    renderer_name="qwen3_instruct" \
    dataset=no_robots learning_rate=5e-4 batch_size=64 lora_rank=64 eval_every=20
```

Note: `renderer_name="qwen3_instruct"` bypasses model lookup (cookbook only has Qwen3 in model_info).
