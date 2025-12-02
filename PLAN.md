# tinker-server: Training + Multi-Session Support

## Tasks

### 1. Training Support

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

### 2. Concurrent Sessions

Support multiple training/sampling sessions simultaneously.

**SessionManager:**
```python
class SessionManager:
    training_sessions: dict[str, TrainingSession]  # model_id -> session
    sampling_sessions: dict[str, SamplingSession]  # session_id -> session
```

**Isolation:**
- Each TrainingSession spawns independent Ray placement group
- LoRA adapters isolated per model_id
- Existing VerlInferenceEngine (STANDALONE) handles sampling-only sessions
- No shared state between sessions

**Request ordering:**
- Use seq_id field to order requests per model_id
- Background task queue per TrainingSession
- forward_backward must complete before optim_step

## Testing

**Basic training loop:**
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

**Test script:** `scripts/test_training.py`

## Dependencies

Add to pyproject.toml:
```toml
"peft>=0.7.0"  # LoRA implementation
```

## Files

New:
- `tinker_server/backend/verl_training.py`
- `tinker_server/backend/session_manager.py`
- `tinker_server/routes/training.py`
- `scripts/test_training.py`

Modified:
- `tinker_server/models/types.py`
- `tinker_server/routes/service.py` (add /create_model)
- `tinker_server/app.py` (initialize SessionManager)
