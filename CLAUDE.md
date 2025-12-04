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

### 1. max_tokens ignored
User's `max_tokens` parameter in sampling requests is currently ignored. verl computes max_tokens internally as `max_model_len - prompt_len`, generating up to that limit.

**Impact:** Model generates thousands of tokens instead of requested amount (e.g., request 64 tokens, get 4089).

**To fix:** Requires upstream change to verl's `vLLMHttpServer.generate()` method to accept user-specified max_tokens.

**Location:** `/root/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:400`

### 2. EOS token detection (FIXED)
EOS token detection is now working correctly. The fix adds `stop_token_ids=[151645, 151643]` to the sampling parameters in verl_inference.py, and the sampling route correctly detects the stop reason based on presence of EOS tokens in the generated sequence.

**Location:** `tinker_server/backend/verl_inference.py:151` and `tinker_server/routes/sampling.py:72-76`

### 3. Slow training-to-inference weight sync (BLOCKING)
When syncing LoRA weights from training to inference via `save_weights_for_sampler`, each new sampling session spawns a fresh vLLM engine. This causes significant latency (tens of seconds) per weight sync.

**Root cause:** Current architecture creates one vLLM engine per sampling session. Loading LoRA at init requires full engine startup.

**Impact:** RL training loops that frequently sync weights between training and inference are bottlenecked by engine initialization time.

**To fix:** Implement hot LoRA reload into running vLLM engine. Requires:
1. Apply verl's `VLLMHijack` to patch vLLM for `TensorLoRARequest` support
2. Use `engine.add_lora()` to hot-swap adapters without restart
3. Alternatively, use vLLM's native multi-LoRA with dynamic adapter loading

**Workaround:** For now, minimize weight sync frequency or batch multiple training steps before sync.

**Related:** See "Future: vLLM Multi-LoRA" in Architecture Notes.

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

## Architecture Notes

### Current: One Server Per Session

Each sampling session spawns a dedicated verl server with its own LoRA adapter.

- SessionManager maps session_id → VerlInferenceEngine
- LoRA weights randomly initialized at session creation
- Engine shutdown on session end via `/end_session`

### Future: vLLM Multi-LoRA (Planned)

For efficient multi-tenant inference under high load, bypass verl and use vLLM directly:

- Single vLLM server with `enable_lora=True`, `max_loras=N`
- Dynamic adapter loading via `/v1/load_lora_adapter`
- Per-request `lora_request` routing
- No per-session server spawning overhead

Requires direct vLLM integration (verl's generate() doesn't expose lora_request parameter).
