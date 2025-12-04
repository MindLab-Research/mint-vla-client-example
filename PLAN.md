# tinker-server Development Plan

## Completed: Phase 1 - Training + Inference Integration

All features from original plan implemented and working.

### Implemented Features

| Component | Status | Commit |
|-----------|--------|--------|
| Training types in types.py | Done | - |
| TrainingSessionManager | Done | - |
| VerlTrainingEngine | Done | - |
| Training routes (/create_model, /forward_backward, /optim_step) | Done | - |
| Save weights for sampler | Done | - |
| Telemetry endpoint | Done | - |
| Per-session inference engines | Done | - |
| Hot LoRA reload (88x speedup) | Done | da69d06 |

### Test Scripts

- `scripts/test_client.py` - Inference flow
- `scripts/test_training_client.py` - Training + inference integration

---

## Next: Phase 2 - Stability Fixes (HIGH PRIORITY)

Two critical issues identified that cause poor performance and crashes:

### 1. max_tokens Ignored (FIXED)

**Problem:** User's `max_tokens` parameter was ignored by verl. Model generated up to `max_model_len - prompt_len` tokens (e.g., 4000+) instead of requested amount.

**Root cause:** `verl/workers/rollout/vllm_rollout/vllm_async_server.py:400`
```python
max_tokens = self.config.max_model_len - len(prompt_ids)  # Ignores user's max_tokens
```

**Fix:** Monkey-patched via `ExtendedVLLMHttpServer.generate()` override in `verl_inference.py`.
Uses `min(user_max_tokens, max_model_len - prompt_len)` to respect user's limit while staying within engine bounds.

### 2. Concurrent Session Crash (FIXED)

**Problem:** Running two training sessions in parallel caused `EngineDeadError` crash.

**Root cause:** Shared inference engine had no locking around hot-reload + inference operations.
- `_shared_engine_lock` only protected engine creation
- When session B hot-reloaded LoRA while session A was inferring, vLLM engine crashed

**Fix:** Implemented per-session inference engines.

Each training session now gets its own dedicated `VerlInferenceEngine`:
- On first `save_weights_and_get_sampling_client()` call: Creates new engine (~60s)
- On subsequent calls: Hot-reloads LoRA on session's own engine (~0.7s)
- Engines are isolated - no interference between concurrent sessions

**Implementation:**
- `TrainingSession.inference_engine` field holds per-session engine
- `SessionManager.create_session_with_engine()` registers external engines
- `TrainingSessionManager.shutdown_all()` cleans up inference engines

**Location:**
- `tinker_server/backend/training_session_manager.py:39-41` (new field)
- `tinker_server/backend/session_manager.py:263-294` (new method)
- `tinker_server/routes/training.py:266-328` (modified flow)

---

## Phase 3 - Production Readiness

### 3. Observability (Medium Priority)

- Prometheus metrics (request latency, throughput, GPU utilization)
- Structured logging with request tracing
- Health check endpoints

### 4. Error Resilience (Medium Priority)

- Graceful handling of Ray actor failures
- Automatic session recovery after GPU OOM
- Circuit breaker for inference requests

### 5. Testing (Low Priority)

- Unit tests for session managers
- Integration tests with mock Ray actors
- Load testing scripts

---

## Architecture Reference

```
Client (HTTP)
    │
    ▼
API Server (FastAPI)
    ├── SessionManager (inference)
    │   └── Per-session VerlInferenceEngine (named flow)
    │
    └── TrainingSessionManager
        └── TrainingSession
            ├── VerlTrainingEngine → TrainingWorker Ray actors
            └── VerlInferenceEngine (per-session, for ephemeral flow)

GPU Workers (Ray cluster)
    ├── ExtendedVLLMHttpServer (inference - one per training session)
    └── TrainingWorker (LoRA training)
```

Data flow for ephemeral weight sync:
1. TrainingWorker.get_lora_state_dict() → tensors via Ray
2. API server saves to checkpoint directory
3. API server loads tensors from checkpoint
4. server.add_lora_from_tensors.remote() → tensors via Ray
5. GPU worker saves to temp, loads via LoRARequest
