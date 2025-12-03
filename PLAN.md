# tinker-server: Multi-Session LoRA + Training Support

## Current Task: One Server Per Session

Implement per-session verl server with dedicated LoRA adapter.

### Architecture

```
Current:
  app.py → single VerlInferenceEngine → all sessions

New:
  app.py → SessionManager
             ├── session_1 → VerlInferenceEngine (lora_1)
             ├── session_2 → VerlInferenceEngine (lora_2)
             └── session_N → VerlInferenceEngine (lora_N)
```

### Implementation

**1. SessionManager (`tinker_server/backend/session_manager.py`)**

```python
class SessionManager:
    _sessions: dict[str, VerlInferenceEngine]

    async def create_session(self, session_id: str, lora_rank: int) -> VerlInferenceEngine
    def get_engine(self, session_id: str) -> VerlInferenceEngine | None
    async def end_session(self, session_id: str) -> bool
    async def shutdown_all(self) -> None
```

**2. Modify VerlInferenceEngine**

Add `lora_rank` parameter:
```python
def __init__(self, ..., lora_rank: int = 0):
    self.lora_rank = lora_rank

# In initialize():
model_config = HFModelConfig(
    path=self.model_path,
    lora_rank=self.lora_rank,  # Was hardcoded 0
)
```

**3. Modify Routes**

`service.py`:
- `create_sampling_session`: spawn engine via SessionManager

`sampling.py`:
- Replace global `verl_engine` with `session_manager.get_engine(session_id)`

**4. Modify app.py**

- Initialize SessionManager instead of single engine
- Shutdown all sessions on app exit

### New Types

```python
class CreateSamplingSessionRequest:
    session_id: str
    base_model: str | None = None
    lora_rank: int = 32  # NEW
```

### Files

New:
- `tinker_server/backend/session_manager.py`
- `scripts/test_multi_lora_sessions.py`

Modified:
- `tinker_server/backend/verl_inference.py`
- `tinker_server/routes/service.py`
- `tinker_server/routes/sampling.py`
- `tinker_server/app.py`
- `tinker_server/models/types.py`

### Testing

```python
# Create two sessions with different LoRA
session_1 = create_sampling_session(lora_rank=32)
session_2 = create_sampling_session(lora_rank=32)

# Same prompt, different outputs (different random LoRA weights)
result_1 = sample(session_1, prompt)
result_2 = sample(session_2, prompt)
assert result_1 != result_2
```

---

## Future: vLLM Multi-LoRA (Not Implemented)

Current approach spawns one server per session. For high-load multi-tenant:

- Bypass verl, use vLLM directly
- Single vLLM server with `enable_lora=True`, `max_loras=N`
- Dynamic adapter loading via `/v1/load_lora_adapter`
- Per-request `lora_request` parameter

This requires direct vLLM integration, not through verl wrapper.

---

## Planned: Training Support

Implement TrainingClient API with LoRA fine-tuning.

**Endpoints:**
- `POST /api/v1/create_model` - spawn training session with LoRA config
- `POST /api/v1/forward_backward` - compute loss and gradients
- `POST /api/v1/optim_step` - update weights, sync to rollout

**Architecture change:**
```
Current: VerlInferenceEngine (STANDALONE mode, inference only)

New: VerlTrainingEngine (HYBRID mode)
  ├── ActorWorker (FSDP + LoRA rank 32)
  └── RolloutReplica (vLLM with LoRA weights)
      └── Weight sync after optim_step
```

**LoRA config:**
- Rank: 32
- Target: all-linear layers (attn, mlp, unembed)
- Base model: Qwen2.5-7B-Instruct

**Implementation:**
- `tinker_server/backend/verl_training.py` - VerlTrainingEngine wrapper
- `tinker_server/backend/session_manager.py` - model_id → TrainingSession mapping
- `tinker_server/routes/training.py` - forward_backward, optim_step endpoints
- `tinker_server/models/types.py` - add ForwardBackwardRequest, OptimStepRequest, CreateModelRequest

**Weight sync flow:**
1. optim_step completes on ActorWorker
2. collect_lora_params() extracts LoRA weights
3. RolloutReplica.update_weights() loads into vLLM

**verl configuration:**
- RolloutMode.HYBRID (actor + rollout share GPUs)
- Ray placement group with max_colocate_count=2
- Context switching: actor training → rollout sleep, rollout sampling → actor sleep

**Testing:**
```python
# Create training session
client = ServiceClient()
training = client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    rank=32
)

# Prepare batch
data = [Datum(
    model_input=ModelInput.from_ints([...]),
    loss_fn_inputs={"target_tokens": TensorData.from_ints([...])}
)]

# Train 10 steps
for i in range(10):
    fwd_result = training.forward_backward(data, "cross_entropy").result()
    opt_result = training.optim_step(AdamParams(learning_rate=1e-4)).result()
    print(f"Step {i}: loss={fwd_result.metrics['loss']}")
```

**Verify:**
- Loss decreases (initial ~2.5 → after 10 steps ~1.8)
- RolloutReplica weights updated (sample from trained model shows changed output)
- Multiple sessions don't interfere (create 2 training sessions, verify independent losses)

**New files:**
- `tinker_server/backend/verl_training.py`
- `tinker_server/routes/training.py`
- `scripts/test_training.py`

---

## Dependencies

Add to pyproject.toml:
```toml
"peft>=0.7.0"  # LoRA implementation
```
