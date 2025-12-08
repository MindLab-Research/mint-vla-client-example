# Tinker API Compatibility Checklist

Reference: `/home/yiwen/tinker_project/tinker-cookbook/CLIENT_INTERFACE.md`

## Priority 1: API Naming Fixes (Breaking Changes) - DONE

These rename existing endpoints to match Tinker spec exactly.

| Endpoint | Current Name | Tinker Name | Status |
|----------|-------------|-------------|--------|
| Save full state (LoRA + optimizer) | `POST /save_state` | `POST /save_state` | Done |
| Load full state | `POST /load_state` | `POST /load_state` | Done |
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
| `TrainingClient.save_state_async()` | `POST /save_state` | Done | Working |
| `TrainingClient.load_state_async()` | `POST /load_state` | Done | Working |
| `TrainingClient.save_weights_for_sampler_async()` | `POST /save_weights_for_sampler` | Done | Working |
| `TrainingClient.save_weights_and_get_sampling_client()` | `POST /save_weights_for_sampler` (path=None) | Done | Ephemeral flow |
| `ServiceClient.create_training_client_from_state()` | `POST /create_model_from_state` | Done | Composes create + load |

### `create_training_client_from_state` Implementation

```
Input: session_id, model_seq_id, base_model, state_path, lora_config, load_optimizer
Output: model_id (str)
```

Implementation:
- Compose: `create_model` + `load_state`
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
| `cross_entropy` | `target_tokens`, `loss_mask` |
| `importance_sampling` | `target_tokens`, `loss_mask`, `logprobs`, `advantages` |
| `ppo` | `target_tokens`, `loss_mask`, `logprobs`, `advantages` |

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

## Priority 6: Stage 5 - Custom Losses (Not Implemented)

Required for DPO and custom training objectives.

| Interface Method | Endpoint | Status | Notes |
|-----------------|----------|--------|-------|
| `TrainingClient.forward_backward_custom()` | `POST /forward_backward_custom` | **Missing** | Callback execution |

### `forward_backward_custom` Implementation

Challenge: Callback functions cannot be serialized over HTTP.

Options:
1. **Named callbacks** - Server defines callback registry, client passes callback name
2. **Code execution** - Client sends Python code (security risk)
3. **DSL** - Define a loss function DSL that server interprets

Recommended: Option 1 (named callbacks) for MVP, with built-in DPO loss.

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

1. **Rename endpoints** (Priority 1) - Breaking change, do first - DONE
   - [x] `save_weights` → `save_state`
   - [x] `load_weights` → `load_state`
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

6. **Add custom loss support** (Priority 6) - Enables DPO
   - [ ] Design callback mechanism
   - [ ] Implement `forward_backward_custom`

## Test Coverage Needed

- [ ] `compute_logprobs` returns correct format (length = seq_len - 1)
- [x] `create_model_from_state` restores LoRA + optimizer correctly
- [x] `forward` returns logprobs without updating gradients
- [ ] `importance_sampling` loss computes correct gradients
- [ ] `ppo` loss clips ratios correctly
