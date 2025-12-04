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

## Next: Phase 2 - Production Readiness

Candidates for next development phase:

### 1. Multi-Session Isolation (Medium Priority)

**Problem:** Current shared engine only supports one active LoRA. Multiple concurrent training sessions overwrite each other's weights.

**Solution options:**
- vLLM multi-LoRA with `max_loras=N` and per-request adapter routing
- Session queuing (serialize ephemeral requests)
- Multiple shared engines (one per LoRA rank)

### 2. max_tokens Support (Low Priority - Upstream)

**Problem:** User's `max_tokens` parameter ignored. Requires verl upstream change.

**Workaround:** Use `stop_token_ids` for EOS detection (already implemented).

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
    │   ├── Per-session VerlInferenceEngine (named flow)
    │   └── Shared VerlInferenceEngine (ephemeral flow, 88x faster)
    │
    └── TrainingSessionManager
        └── VerlTrainingEngine → TrainingWorker Ray actors

GPU Workers (Ray cluster)
    ├── ExtendedVLLMHttpServer (inference)
    └── TrainingWorker (LoRA training)
```

Data flow for ephemeral weight sync:
1. TrainingWorker.get_lora_state_dict() → tensors via Ray
2. API server saves to checkpoint directory
3. API server loads tensors from checkpoint
4. server.add_lora_from_tensors.remote() → tensors via Ray
5. GPU worker saves to temp, loads via LoRARequest
