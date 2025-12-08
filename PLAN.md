# Tinker API Compatibility Checklist

Reference: `/home/yiwen/tinker_project/tinker-cookbook/CLIENT_INTERFACE.md`

## Priority 1: API Naming Fixes (Breaking Changes) - DONE

These rename existing endpoints to match Tinker spec exactly.

| Endpoint | Current Name | Tinker Name | Status |
|----------|-------------|-------------|--------|
| Save full state (LoRA + optimizer) | `POST /save_weights` | `POST /save_weights` | Done |
| Load full state | `POST /load_weights` | `POST /load_weights` | Done |
| Response types | `SaveStateResponse`, `LoadStateResponse` | `SaveStateResponse`, `LoadStateResponse` | Done |

## Priority 2: Stage 1 - Inference - DONE

Core inference functionality required for all workflows.

| Interface Method | Endpoint | Status | Notes |
|-----------------|----------|--------|-------|
| `SamplingClient.sample_async()` | `POST /asample` | Done | Working |
| `SamplingClient.compute_logprobs_async()` | `POST /compute_logprobs` | Done | Uses vLLM `prompt_logprobs` |
| `ServiceClient.create_sampling_client()` | `POST /create_sampling_session` | Done | Working |

### `compute_logprobs_async` Implementation

```
Input: sequence (ModelInput)
Output: list[float] where logprobs[i] = log P(token[i+1] | token[0:i+1])
Length: len(sequence) - 1
```

Implementation:
- `POST /compute_logprobs` endpoint added to sampling.py
- Uses `prompt_logprobs=1` in vLLM SamplingParams
- Supports both multi-LoRA and legacy per-session modes

## Priority 3: Stage 3 - Checkpointing - DONE

Required for training resumption and state management.

| Interface Method | Endpoint | Status | Notes |
|-----------------|----------|--------|-------|
| `TrainingClient.save_state_async()` | `POST /save_weights` | Done | Working |
| `TrainingClient.load_state_async()` | `POST /load_weights` | Done | Working |
| `TrainingClient.save_weights_for_sampler_async()` | `POST /save_weights_for_sampler` | Done | Working |
| `TrainingClient.save_weights_and_get_sampling_client()` | `POST /save_weights_for_sampler` (path=None) | Done | Ephemeral flow |
| `ServiceClient.create_training_client_from_state()` | `POST /create_model_from_state` | Done | Composes create + load |

### `create_training_client_from_state` Implementation

```
Input: session_id, model_seq_id, base_model, state_path, lora_config, load_optimizer
Output: model_id (str)
```

Implementation:
- Compose: `create_model` + `load_weights`
- Create new Ray actor with LoRA config
- Load checkpoint into actor
- Return model_id

Location: `tinker_server/routes/training.py:147-229`

## Priority 4: Stage 2 - SFT Training - DONE

| Interface Method | Endpoint | Status | Notes |
|-----------------|----------|--------|-------|
| `ServiceClient.create_lora_training_client()` | `POST /create_model` | Done | Working |
| `TrainingClient.forward_backward_async()` | `POST /forward_backward` | Done | Only `cross_entropy` implemented |
| `TrainingClient.forward_async()` | `POST /forward` | Done | Forward only, no backward |
| `TrainingClient.optim_step_async()` | `POST /optim_step` | Done | Working |
| `TrainingClient.get_tokenizer()` | `GET /models/{model_id}/tokenizer` | Done | Return tokenizer config |

### `forward_async` Implementation

```
Input: data (list[Datum]), loss_fn (str)
Output: ForwardBackwardOutput (logprobs in loss_fn_outputs, no gradient computation)
```

Implementation:
- `POST /forward` endpoint added to training.py
- Uses `torch.no_grad()` context, model in eval mode
- Returns logprobs for target tokens in `loss_fn_outputs`
- Same input format as `forward_backward`

Location: `tinker_server/routes/training.py:279-316`

## Priority 5: Stage 4 - RL Training - DONE

Required for RLHF workflows.

| Interface Method | Endpoint | Status | Notes |
|-----------------|----------|--------|-------|
| `forward_backward_async(loss_fn="importance_sampling")` | `POST /forward_backward` | Done | Policy gradient with IS |
| `forward_backward_async(loss_fn="ppo")` | `POST /forward_backward` | Done | PPO with clipping |

### Loss Function Inputs (from spec)

| Loss | Required `loss_fn_inputs` |
|------|--------------------------|
| `cross_entropy` | `target_tokens`, `weights` |
| `importance_sampling` | `target_tokens`, `weights`, `logprobs`, `advantages` |
| `ppo` | `target_tokens`, `weights`, `logprobs`, `advantages` |

### `importance_sampling` Loss

```python
# Policy gradient with importance sampling
ratio = exp(new_logprobs - old_logprobs)
loss = -ratio * advantages * mask
```

### `ppo` Loss

```python
# PPO with clipping (epsilon from loss_fn_config, default 0.2)
ratio = exp(new_logprobs - old_logprobs)
clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
loss = -max(ratio * advantages, clipped_ratio * advantages) * mask
# Note: max() because we negate, equivalent to min() on positive objective
```

### Implementation

Location: `tinker_server/backend/verl_training.py:84-210`

- `loss_fn` and `loss_fn_config` passed from request to worker
- PPO epsilon configurable via `loss_fn_config.epsilon` (default: 0.2)
- Asymmetric clipping via `loss_fn_config.clip_low` / `clip_high`
- Returns RL metrics: `ratio:mean`, `clipfrac:mean` (PPO only)

## Priority 6: Stage 5 - Custom Losses - DONE

Required for DPO and custom training objectives.

| Interface Method | Endpoint | Status | Notes |
|-----------------|----------|--------|-------|
| `TrainingClient.forward_backward_custom()` | Client-side | Done | Server supports via `weights` |

### `forward_backward_custom` Implementation

The tinker client implements `forward_backward_custom` **client-side** using existing server endpoints:

1. Client calls `POST /forward` with `cross_entropy` → gets logprobs with `requires_grad=True`
2. Client runs user's callback: `loss, metrics = loss_fn(data, logprobs_list)`
3. Client calls `loss.backward()` → extracts gradients from logprob tensors
4. Client calls `POST /forward_backward` with `weights: -grad` (negative gradients)
5. Server computes `loss = sum(cross_entropy * weights)` → backpropagates custom loss gradients

### Server-side Support

No new endpoint needed. Server supports custom losses via `weights` in `loss_fn_inputs`:

- When `weights` has negative values → custom loss backward → sum without averaging
- When `weights` is non-negative → standard SFT/RL → average by sum of weights

### Data Flow

```
Client                              Server
------                              ------
forward(data, "cross_entropy")  --> /forward --> logprobs

loss_fn(data, logprobs)         [client-side]
loss.backward()                 [client-side]
grads = logprobs.grad           [client-side]

forward_backward(               --> /forward_backward
  data with weights=-grads,         (detects negative weights)
  "cross_entropy"                   loss = sum(ce * weights)
)                                   loss.backward()
```

Location: `tinker_server/backend/verl_training.py:85-283`

## Data Types Status

| Type | Status | Notes |
|------|--------|-------|
| `ModelInput` | Done | `from_ints()`, `to_ints()` implemented |
| `EncodedTextChunk` | Done | |
| `TensorData` | Done | `from_torch()`, `to_torch()` in client |
| `Datum` | Done | |
| `AdamParams` | Done | |
| `SamplingParams` | Done | |
| `LoRAConfig` | Done | |
| `ForwardBackwardOutput` | Done | |
| `SampleResponse` | Done | |
| `Sequence` | Done | As `SampledSequence` |
| `SaveStateResponse` | Done | |
| `LoadStateResponse` | Done | |
| `APIFuture` | Done | As `UntypedAPIFuture` |

## Implementation Order

1. **Endpoint alignment** (Priority 1) - Match tinker client exactly - DONE
   - [x] `POST /save_weights` - Save full checkpoint
   - [x] `POST /load_weights` - Load checkpoint
   - [x] Update types and routes

2. **Add `compute_logprobs`** (Priority 2) - Enables RL data collection - DONE
   - [x] Add endpoint
   - [x] Use vLLM `prompt_logprobs` feature

3. **Add `create_model_from_state`** (Priority 3) - Enables training resumption - DONE
   - [x] Add endpoint
   - [x] Compose create + load

4. **Add `forward` (forward-only)** (Priority 4) - Enables inference-time logprob computation - DONE
   - [x] Add endpoint
   - [x] Skip backward pass
   - [x] Add `GET /models/{model_id}/tokenizer` endpoint

5. **Add RL losses** (Priority 5) - Enables RLHF - DONE
   - [x] `importance_sampling` loss
   - [x] `ppo` loss

6. **Add custom loss support** (Priority 6) - Enables DPO - DONE
   - [x] Use `weights` in `loss_fn_inputs` (tinker standard)
   - [x] Detect negative weights for custom loss backward (sum without averaging)
   - [x] Client-side `forward_backward_custom` composes `/forward` + `/forward_backward`

## Integration Testing with Tinker Cookbook

Run unmodified Tinker Cookbook recipes against our local server by setting environment variables:

```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy
```

### Test Environment

```bash
# Terminal 1: Start tinker-server
HF_HUB_OFFLINE=1 \
HF_HOME=/vePFS-Mindverse/share/huggingface \
TINKER_MODEL_PATH=/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
python scripts/run_server.py

# Terminal 2: Run cookbook recipes
cd ../tinker-cookbook
export TINKER_BASE_URL=http://localhost:8000
export TINKER_API_KEY=dummy
```

### Phase 1: Quick Validation (5 min)

| Recipe | Type | Loss Function | Status |
|--------|------|---------------|--------|
| Arithmetic RL | RL | importance_sampling/ppo | [x] Done |

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    group_size=4 \
    groups_per_batch=100 \
    learning_rate=1e-4
```

Expected: Reward 0.66 → 1.0 in first few steps.

### Phase 2: SFT Baseline (30-60 min)

| Recipe | Type | Loss Function | Status |
|--------|------|---------------|--------|
| Chat SL (NoRobots) | SFT | cross_entropy | [x] Done |

```bash
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    dataset=no_robots \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    eval_every=20
```

Expected: test/nll drops to ~1.78 after 140 steps.

### Phase 3: RL on Math (2-4 hours)

| Recipe | Type | Loss Function | Status |
|--------|------|---------------|--------|
| MATH RL | RL | importance_sampling/ppo | [ ] |

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    env=math \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    group_size=16 \
    groups_per_batch=64 \
    learning_rate=2e-5 \
    max_tokens=512
```

Expected: test/env/all/correct = 0.767 after 180 steps.

### Phase 4: Custom Loss - DPO (1-2 hours)

| Recipe | Type | Loss Function | Status |
|--------|------|---------------|--------|
| DPO (HHH) | Preference | custom (via weights) | [x] Done |

```bash
python -m tinker_cookbook.recipes.preference.dpo.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    dataset=hhh \
    learning_rate=1e-5 \
    dpo_beta=0.1
```

Expected: accuracy ~0.57 after 50 steps.

### Phase 5: Full Pipeline (8+ hours, optional)

| Recipe | Type | Coverage | Status |
|--------|------|----------|--------|
| RLHF 3-stage | SFT→RM→RL | All loss types | [ ] |

```bash
python -m tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline
```

### Test Checklist

- [x] Phase 1: Arithmetic RL completes without errors (reward 0.66 → 1.0)
- [x] Phase 2: Chat SL trains and loss decreases (test/nll dropped to ~1.78)
- [ ] Phase 3: MATH RL trains and accuracy improves
- [x] Phase 4: DPO trains with custom loss via weights (accuracy 0.41 → 0.61, dpo_loss 0.71 → 0.69)
- [~] Phase 5: Full RLHF pipeline - SKIPPED (cookbook model_info.py lacks Qwen2.5 support, not a server issue)

### Known Requirements

All recipes auto-download data from HuggingFace. No external services required for Phases 1-4.

## Unit Test Coverage

- [ ] `compute_logprobs` returns correct format (length = seq_len - 1)
- [x] `create_model_from_state` restores LoRA + optimizer correctly
- [x] `forward` returns logprobs without updating gradients
- [ ] `importance_sampling` loss computes correct gradients
- [ ] `ppo` loss clips ratios correctly
