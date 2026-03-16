# Sizing logic

## 1. Budget layers

The sizing problem has four memory consumers:

- Base model weights
- Steady-state LoRA runtime capacity
- KV cache for live requests
- Runtime scratch, especially long-prompt logprob scratch

The first three determine the steady-state serving envelope.
The fourth determines how much extra headroom is required for some request
classes.

## 2. Planning-time budget

Define:

- `requested_memory = total_gpu_memory_per_gpu * gpu_memory_utilization`

vLLM profiles non-KV memory first and then gives the remainder of
`requested_memory` to KV cache.

So the planning-time inequality is:

- `requested_memory >= profiled_non_kv + kv_budget`

where:

- `profiled_non_kv` is vLLM's original startup-profile term:
  - `weights_memory`
  - `peak_activation_memory`
  - `non_torch_increase`
- `weights_memory` already includes persistent LoRA slot tensors allocated
  during model load when LoRA is enabled
- `kv_budget` is the intended KV reservation

Do not replace `profiled_non_kv` with `weights_memory`.
If local patches bypass `Worker.determine_available_memory()` or mutate
`DeviceMemoryProfiler`, the sizing math in this reference no longer applies.

## 3. KV geometry

Per token, per GPU:

- `kv_bytes_per_token_per_gpu = 2 * num_layers * local_kv_heads * head_dim * bytes_per_element`

where:

- first `2` is K and V
- `local_kv_heads` is after TP partitioning and any head-replication rule
- `bytes_per_element` is usually `2` for bf16/fp16

Then:

- `kv_bytes_per_full_context_seq_per_gpu = kv_bytes_per_token_per_gpu * context_len`

If the conservative long-request concurrency target is `N_long`, the KV term is:

- `kv_budget >= N_long * kv_bytes_per_full_context_seq_per_gpu`

## 4. LoRA term

Do not use checkpoint size as the LoRA runtime estimate.

What matters is steady-state slot capacity on each GPU:

- `lora_runtime_per_gpu = max_loras * slot_cost_per_gpu(max_lora_rank, architecture)`

Hard rule:

- derive `slot_cost_per_gpu` from the active vLLM replacement class and tensor
  shapes, not from a generic MoE shorthand

For MoE this especially means checking all of:

- whether the active class is `FusedMoEWithLoRA` or `FusedMoE3DWithLoRA`
- whether `fully_sharded_loras` is actually enabled
- whether expert parallel is enabled, because that determines
  `local_num_experts`

Do not assume `local_num_experts = global_num_experts / tp`.
If expert parallel is disabled, `local_num_experts` can still equal the full
expert count even when tensor parallel is greater than 1.

For `FusedMoEWithLoRA`, do not collapse the expert slot cost to a single
"experts term". The persistent slot tensors are:

- `w13_lora_a_stacked`
- `w13_lora_b_stacked`
- `w2_lora_a_stacked`
- `w2_lora_b_stacked`

Even when `fully_sharded_loras=True`, only part of that state is TP-divided.
On the current path:

- `w13_lora_a` is rank-sharded
- `w2_lora_b` is hidden-sharded
- `w13_lora_b` is still intermediate-sharded only
- `w2_lora_a` is still intermediate-sharded only

So for `FusedMoEWithLoRA` with expert parallel disabled, the per-layer slot
bytes are:

- `2 * E * (R / TP) * H * bytes`
- `2 * E * (I / TP) * R * bytes`
- `E * R * (I / TP) * bytes`
- `E * (H / TP) * R * bytes`

where:

- `E = local_num_experts`
- `R = max_lora_rank`
- `H = hidden_size`
- `I = moe_intermediate_size`
- `bytes = lora_dtype_size`

Only after the active tensor shapes are identified should you apply the usual
linear scaling observations:

- slot cost scales linearly with `max_lora_rank`
- slot cost scales linearly with `max_loras`

Under a MinT-style rule where active LoRAs should sit slightly below guaranteed
full-context concurrency, choose:

- `max_loras = N_long - 1`

## 5. Prompt-logprob headroom

Prompt-logprob scratch is not part of the KV reservation.
But it still uses real GPU memory at runtime.

For the path we inspected, the dominant lower-bound term is:

- `prompt_logprob_scratch ~= effective_chunk_tokens * vocab_size * 4`

This is why `max_num_batched_tokens` matters for long prompt logprob requests:

- smaller scheduled chunks reduce the `[tokens, vocab]` scratch term

On the current vLLM prompt-logprob path, there is also an internal chunk cap of
`1024` prompt tokens. For this path:

- `effective_chunk_tokens = min(max_num_batched_tokens, 1024)`

So the runtime peak is better modeled as:

- `runtime_peak = profiled_non_kv + kv_cache_in_use + prompt_logprob_scratch + runtime_overhead`

This also means lowering `gpu_memory_utilization` can help prompt-logprobs:

- it shrinks KV reservation
- which leaves more physical free memory for runtime scratch

## 6. Choosing the knobs

### `gpu_memory_utilization`

Choose from the need for runtime headroom, not only from KV math.

- Small dense models can usually tolerate a higher value.
- Large models, especially MoE, usually need more slack.

### `max_loras`

Choose from the number of distinct simultaneously active adapters you want to
support.

Under a MinT-like assumption of diverse active LoRAs:

- choose `max_loras` close to, but below, the conservative full-context
  concurrency target

### `max_cpu_loras`

This is a host-side cache size, not a GPU KV tradeoff knob.

- increase it when host RAM allows
- do not treat it as part of the GPU steady-state budget

### `max_num_seqs`

This is an admission cap for realistic mixed traffic.

It is not the same as the guaranteed number of fully accumulated long requests.
It may be larger than the conservative long-context concurrency target.

### `max_num_batched_tokens`

This is mainly a transient working-set knob.

Use it to bound:

- prefill working-set pressure
- prompt-logprob scratch

For large models, choose this from the largest chunk that still leaves enough
runtime headroom for long prompt logprobs.

## 7. Procedure for a new model or GPU shape

1. Fix the service target.
- Context length
- Whether long prompt logprobs must work
- Worst-case long-request concurrency
- Whether active LoRAs are expected to be diverse

2. Compute KV geometry.
- Derive `kv_bytes_per_token_per_gpu`
- Derive `kv_bytes_per_full_context_seq_per_gpu`

3. Estimate `profiled_non_kv`.
- Base weights per GPU
- LoRA slot/runtime capacity
- Other profiled executor overhead

If you cannot derive the full profiled term exactly from architecture alone,
introduce an explicit conservative upper bound `U_profile` for the remainder:

- `(weights_memory - theoretical_weight_shard)`
- `peak_activation_memory`
- `non_torch_increase`

Hard epistemic boundary:

- empirical calibration is allowed only inside `U_profile`
- do not use empirical data to modify KV geometry or slot geometry directly
- if runtime falsifies the current slot geometry, re-derive the tensor basis
  first and only then update `U_profile`

4. Solve the steady-state inequality.
- `requested_memory >= profiled_non_kv + N_long * kv_bytes_per_full_context_seq_per_gpu`

Equivalently, with explicit architecture terms:

- `requested_memory >= theoretical_weight_shard + lora_slot_cost + U_profile + N_long * kv_per_seq`

5. Add runtime scratch headroom.
- Especially prompt-logprob scratch

6. Choose:
- `gpu_memory_utilization`
- `max_loras`
- `max_cpu_loras`
- `max_num_batched_tokens`
- `max_num_seqs`

7. Validate empirically.
- Long sample
- Long prompt logprobs
- Multiple distinct LoRAs
- Concurrent traffic

## 8. Language discipline

When explaining these knobs:

- keep steady-state capacity separate from runtime scratch
- keep KV reservation separate from total physical VRAM use
- separate hard geometry from empirical safety margins
- label empirical conclusions as empirical
