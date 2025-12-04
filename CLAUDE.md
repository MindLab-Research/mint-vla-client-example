# tinker-server

## Running the Server

Working command (offline, local HF cache):

```bash
HF_HUB_OFFLINE=1 \
HF_HOME=/vePFS-Mindverse/share/huggingface \
TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
python scripts/run_server.py
```

Environment variables:
- `HF_HUB_OFFLINE=1` - Force offline mode (no network access to HuggingFace)
- `HF_HOME` - Path to local HuggingFace cache directory
- `TINKER_MODEL_PATH` - Full path to model snapshot directory
- `TINKER_CHECKPOINT_DIR` - Directory for saving LoRA checkpoints (must be shared filesystem in distributed deployments)

## Running with Ray Cluster

To use an existing Ray cluster instead of starting a local instance:

### 1. Join the cluster from client node

```bash
ray start --address='<HEAD_IP>:6379'
```

Example:
```bash
ray start --address='192.168.47.143:6379'
```

### 2. Run the server

The server auto-connects to the cluster via `ray.init(address='auto')`:

```bash
HF_HUB_OFFLINE=1 \
HF_HOME=/vePFS-Mindverse/share/huggingface \
TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
python scripts/run_server.py
```

### Cluster info

The vLLMHttpServer actor is scheduled on cluster nodes with available GPUs. Check cluster resources:

```python
import ray
ray.init(address='auto')
print(ray.cluster_resources())  # Shows CPUs, GPUs, nodes
```

Dashboard: `http://<HEAD_IP>:8265`

## Known Limitations

### 1. max_tokens support (FIXED)
User's `max_tokens` parameter is now respected via monkey-patch in `ExtendedVLLMHttpServer.generate()`.

**Fix:** Override `generate()` method to use `min(user_max_tokens, max_model_len - prompt_len)`.

**Location:** `tinker_server/backend/verl_inference.py:138-213`

### 2. EOS token detection (FIXED)
EOS token detection is now working correctly. The fix adds `stop_token_ids=[151645, 151643]` to the sampling parameters in verl_inference.py, and the sampling route correctly detects the stop reason based on presence of EOS tokens in the generated sequence.

**Location:** `tinker_server/backend/verl_inference.py:283` and `tinker_server/routes/sampling.py:72-76`

### 3. Slow training-to-inference weight sync (FIXED)
Hot LoRA reload implemented via shared engine pattern. Achieved 88x speedup.

**Implementation:**
- `SessionManager._shared_engine` - Lazily initialized shared vLLM engine
- `create_ephemeral_session()` - Uses shared engine with hot LoRA reload (0.68s)
- `add_lora_from_tensors()` - Transfers tensors via Ray to GPU worker, saves locally, loads via file-based LoRARequest

**Performance:**
- Named flow (new engine per session): ~60s
- Ephemeral flow first call (init shared engine): ~60s
- Ephemeral flow subsequent calls (hot reload only): ~0.68s (88x faster)

**Location:** `tinker_server/backend/session_manager.py:103-184` and `tinker_server/backend/verl_inference.py:77-132`

## Testing

### Using tinker SamplingClient (recommended)

```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_client.py
```

Expected output:
```
Connecting to: http://localhost:8000
ServiceClient created, session_id: ...
Creating SamplingClient...
SamplingClient created, sampling_session_id: ...

============================================================
PROMPT (29 tokens):
============================================================
User message: What is the capital of France?

Formatted with chat template:
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
<|im_end|>

Token IDs: [151644, 8198, 271, 1017, 271, 951, 271, 151645, 151644, 9062, ...]

============================================================
GENERATING...
============================================================

============================================================
RESPONSE (1 sequence(s)):
============================================================

EOS token ID: 151645

Sequence 0:
  Length: 8 tokens
  Stop reason: stop
  EOS found at position: 7

  First 20 token IDs: [95806, 41519, 234, 6, 248, 24163, 151643, 151645]
  Last 20 token IDs: [95806, 41519, 234, 6, 248, 24163, 151643, 151645]

  Text:
Paris

============================================================
Test passed!
============================================================
```

### Using curl (raw API)

Server listens on `0.0.0.0:8000` by default (override with `TINKER_HOST`, `TINKER_PORT`).

#### 1. Create session (optional)
```bash
curl -X POST http://localhost:8000/api/v1/create_session \
  -H "Content-Type: application/json" \
  -d '{"tags": [], "user_metadata": {}}'
# Returns: {"session_id": "..."}
```

#### 2. Create sampling session
```bash
curl -X POST http://localhost:8000/api/v1/create_sampling_session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>"}'
# Returns: {"sampling_session_id": "..."}
```

#### 3. Submit async sample
```bash
curl -X POST http://localhost:8000/api/v1/asample \
  -H "Content-Type: application/json" \
  -d '{
    "sampling_session_id": "<sampling_session_id>",
    "seq_id": 0,
    "num_samples": 1,
    "prompt": {"chunks": [{"tokens": [9707], "type": "encoded_text"}]},
    "sampling_params": {"max_tokens": 32, "temperature": 0.7}
  }'
# Returns: {"request_id": "..."}
```

#### 4. Retrieve result (poll until complete)
```bash
curl -i -X POST http://localhost:8000/api/v1/retrieve_future \
  -H "Content-Type: application/json" \
  -d '{"request_id": "<request_id>"}'
# Returns: HTTP 408 (pending) or HTTP 200 with {"sequences": [...], "type": "sample"}
```

Note:
- Steps 1-2 don't validate - any string works for `sampling_session_id` in step 3
- Polling returns HTTP 408 while pending (tinker client protocol)

## Distributed Architecture (Ray Cluster)

**Three distinct machines in the deployment:**

1. **Client Machine** - Runs the tinker Python client (test scripts, user applications)
   - Connects to API server via HTTP
   - No GPU required

2. **API Server Machine** - Runs FastAPI server (`scripts/run_server.py`)
   - Hosts `/api/v1/*` endpoints
   - Manages sessions, routes requests
   - Checkpoints saved locally in `./checkpoints/`
   - Connects to Ray cluster

3. **GPU Worker Node(s)** - Ray actors for training and inference
   - `TrainingWorker` Ray actors (LoRA training on GPU)
   - `ExtendedVLLMHttpServer` Ray actors (vLLM inference on GPU)
   - Different filesystem than API server (`/root/...` vs `/home/yiwen/...`)

**Data transfer between machines uses Ray object store:**
- Training weights: `worker.get_lora_state_dict.remote()` → API server saves to disk
- Inference weights: API server loads from disk → `server.add_lora_from_tensors.remote()`

**Path considerations:**
- Checkpoints saved on API server are NOT accessible from GPU workers
- Must transfer tensors via Ray, not file paths
- Or use shared filesystem (NFS) mounted at same path on all nodes

## Architecture Notes

### Inference Modes

Two session creation modes with different performance tradeoffs:

**1. Named Sessions (per-session engine)**
- `create_sampling_session(model_path=...)` spawns dedicated VerlInferenceEngine
- Full engine initialization (~60s) but complete isolation
- Use for: Long-running sessions, production deployments with stable weights

**2. Ephemeral Sessions (per-training-session engine)**
- `save_weights_and_get_sampling_client()` uses per-training-session VerlInferenceEngine
- First call initializes dedicated engine (~60s), subsequent calls hot-reload LoRA (~0.7s)
- Each training session gets its own isolated engine
- Concurrent training sessions don't interfere with each other
- Use for: RL training loops with frequent weight updates

**Implementation:**
- `TrainingSession.inference_engine` - Per-training-session inference engine
- `SessionManager.create_session_with_engine()` - Registers external engines
- `TrainingSessionManager.shutdown_all()` - Cleans up inference engines on exit

### Concurrent Training Sessions

Multiple training sessions can run in parallel. Each session has:
- Its own `TrainingWorker` Ray actor for training
- Its own `VerlInferenceEngine` for inference (lazily initialized)

This ensures complete isolation - one session's LoRA reload doesn't affect another session's inference.
