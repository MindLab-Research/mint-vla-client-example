# Core logic

## 1. Scope

This skill is for Megatron memory-related back-of-envelope estimation.

That includes:

- training microbatch token admission
- safe `max_token_len_per_gpu`
- safe `max_num_batched_tokens`
- recompute tradeoffs
- offload tradeoffs
- full finetune versus adapter-style training

The starting reference is the original estimator repository:

- `https://huggingface.co/spaces/ISEEKYAN/megatron_memory_estimator`

Treat that project as the structural baseline, not as an infallible runtime
oracle.

## 2. The layered model

For a given per-rank batch shape, split memory into:

- base weights
- trainable-state memory
- activation memory from the estimator
- runtime-only scratch and duplicated state

So the practical peak model is:

- `peak ~= baseline_persistent + batch_activation + logits_scratch + calibrated_residual`

where:

- `baseline_persistent` is measured after model creation and before forward
- `batch_activation` comes from the estimator for the actual per-rank shape
- `logits_scratch` is the large `[B, T, V_shard]` promotion term if present
- `calibrated_residual` covers runtime overhead the estimator does not model
  tightly enough

## 3. Structural lower bound

Start from the estimator's useful terms:

- per-rank base weights
- activation term
- optimizer or gradient term

But interpret the optimizer term correctly.

For full finetune:
- the estimator's full-model optimizer term is relevant

For adapter-style training:
- the base model may be frozen
- the trainable-state term should be based on the adapter parameter count, not
  the full model

So a better lower bound is:

- `lower_bound = base_weights_rank + trainable_state + estimator_activations`

## 4. Runtime-only terms

These are the terms that often explain the large gap between the estimator and
the real OOM.

### Dense padding

If remove-padding is disabled, runtime scales with:

- `batch_size * max_len`

not packed token count.

### fp32 logits promotion

If logits are upcast on the last stage:

- `logits_scratch = batch * padded_seq_len * vocab_shard * 4`

This can add several GiB immediately.

### DDP grad buffers for frozen parameters

Even with a frozen base model, runtime may still allocate gradient buffers or
related structures before freezing is honored.

### TransformerEngine or duplicated parameter storage

TE or wrapper-specific storage can add another weight-scale persistent term.

### Allocator reserve and fragmentation

This is real memory pressure and must be counted in calibrated peak estimates.

## 5. Recompute and offload

### Recompute

Recompute primarily trades compute for activation memory.

Use it when the dominant batch-dependent term is activation memory.

The estimator already models some recompute settings, so recompute changes
should first be reflected in the estimator-side activation term before any
runtime residual calibration.

### Offload

Offload primarily trades latency or bandwidth for persistent memory.

Use it when the dominant term is:

- optimizer state
- gradients
- other persistent training state

Offload is less relevant when the true peak is dominated by a dense logits
scratch term or another large per-step temporary.

## 6. How to estimate `max_token_len_per_gpu`

1. Identify the largest per-rank microbatch shape the current limit admits.
2. Compute the structural lower-bound activation term for that shape.
3. Add runtime-only terms that apply on this path.
4. Compare with measured baseline and the device budget.
5. If the peak is too close to the limit, lower `max_token_len_per_gpu` until
   the batch must split earlier.

Important:

- if a failing batch still has total tokens below `max_token_len_per_gpu`, it
  will stay a single microbatch
- so if the goal is to force splitting, the limit must be below that batch's
  actual admitted token count

## 7. How to estimate `max_num_batched_tokens`

For inference-like Megatron admission, use the same structure:

- persistent baseline
- activation term for the scheduled batch
- logits or output scratch if relevant
- calibrated residual

Then choose the maximum scheduled token count that leaves enough headroom.

So `max_num_batched_tokens` is the same style of knob as
`max_token_len_per_gpu`, but for the inference scheduler rather than the
training microbatch splitter.

## 8. Output discipline

When explaining a Megatron estimate:

- separate structural lower-bound terms from calibrated runtime terms
- state explicitly whether the estimator is being used as a lower bound or as a
  calibrated predictor
- identify which knob each term informs:
  - `max_token_len_per_gpu`
  - `max_num_batched_tokens`
  - recompute
  - offload
  - optimizer strategy
