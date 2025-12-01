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

## Known Limitations

### 1. max_tokens ignored
User's `max_tokens` parameter in sampling requests is currently ignored. verl computes max_tokens internally as `max_model_len - prompt_len`, generating up to that limit.

**Impact:** Model generates thousands of tokens instead of requested amount (e.g., request 64 tokens, get 4089).

**To fix:** Requires upstream change to verl's `vLLMHttpServer.generate()` method to accept user-specified max_tokens.

**Location:** `/root/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:400`

### 2. EOS token detection broken
Model doesn't stop at EOS (end-of-sequence) tokens. Generation continues until hitting max length, producing repetitive/degraded output.

**Impact:** Responses contain correct answer followed by repetitive hashtags/filler until length limit.

**Root cause:** verl's `TokenOutput` at `vllm_async_server.py:433` only returns `token_ids` and `log_probs`, discarding vLLM's `finish_reason`. Server hardcodes `stop_reason="length"`.

**To fix:**
1. Extend verl's `TokenOutput` to include `finish_reason` from vLLM's `RequestOutput`
2. Update `tinker_server/routes/sampling.py:82` to use actual finish reason instead of hardcoded "length"
3. Alternative: Detect EOS token ID in response and truncate

**Location:** `tinker_server/routes/sampling.py:82`

## Testing

### Using tinker SamplingClient (recommended)

```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_client.py
```

Note: Telemetry errors (404 for `/api/v1/telemetry/send`) are expected - telemetry endpoint not implemented in MVP.

Expected output:
```
Connecting to: http://localhost:8000
ServiceClient created, session_id: ...
Creating SamplingClient...
SamplingClient created, sampling_session_id: ...
Generating with 2 prompt tokens...
Waiting for result...
Generated 1 sequence(s):
  Sequence 0:
    Tokens: 809 tokens
    First 10: [678, 0, 358, 2776, 1588, 311, 3061, 911, 279, 7897]
    Stop reason: length
    Logprobs (first 5): [-0.338, -2.100, -0.937, -1.950, -3.605]
Test passed!
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
