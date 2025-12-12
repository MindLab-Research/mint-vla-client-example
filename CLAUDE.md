# Mint Server

## Quick Start (Local)

```bash
HF_HUB_OFFLINE=1 \
HF_HOME=/vePFS-Mindverse/share/huggingface \
TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
python scripts/run_server.py
```

Environment variables:
- `HF_HUB_OFFLINE=1` - Force offline mode
- `HF_HOME` - HuggingFace cache directory
- `TINKER_MODEL_PATH` - Model snapshot path
- `TINKER_CHECKPOINT_DIR` - LoRA checkpoint directory (shared filesystem for distributed)

## Remote Deployment

**For remote deployment and cluster management, use skills:**
- `mint-dev` - Development environment (volcano, port 8000)
- `mint-prod` - Production environment (mint-prod, port 18000)
- `volcano-cluster` - Ray cluster lifecycle

## Testing

### Using tinker SamplingClient

```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_client.py
```

### Using curl (raw API)

```bash
# 1. Create session
curl -X POST http://localhost:8000/api/v1/create_session \
  -H "Content-Type: application/json" \
  -d '{"tags": [], "user_metadata": {}}'

# 2. Create sampling session
curl -X POST http://localhost:8000/api/v1/create_sampling_session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>"}'

# 3. Submit async sample
curl -X POST http://localhost:8000/api/v1/asample \
  -H "Content-Type: application/json" \
  -d '{
    "sampling_session_id": "<sampling_session_id>",
    "seq_id": 0,
    "num_samples": 1,
    "prompt": {"chunks": [{"tokens": [9707], "type": "encoded_text"}]},
    "sampling_params": {"max_tokens": 32, "temperature": 0.7}
  }'

# 4. Poll result
curl -i -X POST http://localhost:8000/api/v1/retrieve_future \
  -H "Content-Type: application/json" \
  -d '{"request_id": "<request_id>"}'
# Returns: HTTP 408 (pending) or HTTP 200 with result
```

### Tinker Cookbook Integration

```bash
cd /home/yiwen/tinker_project/tinker-cookbook

# Arithmetic RL (~5 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    renderer_name="qwen3_instruct" \
    group_size=4 \
    groups_per_batch=100 \
    learning_rate=1e-4

# Chat SL (~30-60 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    renderer_name="qwen3_instruct" \
    dataset=no_robots \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    eval_every=20
```

Note: Use `renderer_name="qwen3_instruct"` to bypass model lookup.

## Architecture

### Distributed Deployment

```
Client Machine                     API Server                    GPU Workers
--------------                     ----------                    -----------
tinker-cookbook  ──HTTP──>  FastAPI (no GPU)  ──Ray──>  TrainingWorker
                            sessions, routing           vLLM Engine
```

- **Client**: Python client, connects via HTTP
- **API Server**: FastAPI, manages sessions, no GPU, connects to Ray cluster
- **GPU Workers**: Ray actors for training (TrainingWorker) and inference (vLLM)

Data transfer uses Ray object store, not filesystem paths.

### Inference Modes

**Named Sessions** - dedicated engine per session
- `create_sampling_session(model_path=...)` spawns VerlInferenceEngine
- ~60s init, complete isolation
- Use for: long-running sessions, stable weights

**Ephemeral Sessions** - per-training-session engine
- `save_weights_and_get_sampling_client()` uses shared engine
- First call ~60s, subsequent hot-reload ~0.7s
- Use for: RL training loops with frequent weight updates

### Persistent vLLM Actor

vLLM runs as detached Ray actor surviving server restarts:
- First start: ~80s (model loading + CUDA graphs)
- Subsequent restarts: ~2s (reuses actor)
- Kill with `/api/v1/kill_vllm` when: model changed, OOM, vLLM code changed

## Known Limitations (Fixed)

### max_tokens
User's `max_tokens` respected via monkey-patch in `ExtendedVLLMHttpServer.generate()`.
Location: `tinker_server/backend/verl_inference.py:138-213`

### EOS token detection
`stop_token_ids=[151645, 151643]` added to sampling parameters.
Location: `tinker_server/backend/verl_inference.py:283`, `tinker_server/routes/sampling.py:72-76`

### Training-to-inference weight sync
Hot LoRA reload via shared engine pattern. 88x speedup (60s → 0.68s).
Location: `tinker_server/backend/session_manager.py:103-184`, `tinker_server/backend/verl_inference.py:77-132`
