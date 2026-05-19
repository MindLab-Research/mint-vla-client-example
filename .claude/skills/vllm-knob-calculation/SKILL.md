---
name: vllm-knob-calculation
description: |
  Back-of-envelope sizing workflow for vLLM serving knobs in mint-server.

  Use for: choosing or reviewing `gpu_memory_utilization`, `max_loras`,
  `max_cpu_loras`, `max_lora_rank`, `max_num_batched_tokens`, and the
  conservative long-context concurrency target for a new model or GPU shape.

  Triggers: "vllm knob calculation", "vllm sizing", "gpu_memory_utilization",
  "max_loras", "max_num_batched_tokens", "KV cache sizing", "32k concurrency"
---

# vLLM knob calculation

Read `references/sizing.md`.

Use this skill when the task is to size or explain serving knobs from first
principles rather than by blind trial-and-error.

The workflow is:

1. Define the service target.
- Context length to guarantee, for example `32768`.
- Whether long-prompt `compute_prompt_logprobs` must work.
- Whether active LoRAs are expected to be diverse.
- Conservative worst-case long-request concurrency target.

2. Compute the KV geometry per GPU.
- Derive `kv_bytes_per_token_per_gpu`.
- Derive `kv_bytes_per_full_context_seq_per_gpu`.
- Keep the math separate from empirical safety margins.

3. Estimate steady-state non-KV memory per GPU.
- Use vLLM's original accounting:
  - `weights_memory`
  - `peak_activation_memory`
  - `non_torch_increase`
- Remember that persistent LoRA slot tensors are part of `weights_memory`.
- If the full profiled term is not derivable exactly from architecture alone,
  introduce an explicit conservative upper bound `U_profile` for the remainder.
- Empirical calibration is allowed only inside `U_profile`.
- If runtime falsifies the slot cost itself, re-derive the active tensor basis
  before changing any knob.

4. Solve the steady-state budget.
- `total_gpu_mem * gpu_memory_utilization >= profiled_non_kv + long_seq_kv_budget`
- Use that to choose a conservative long-sequence concurrency target.

5. Add runtime scratch headroom.
- Prompt-logprob scratch is outside the KV reservation, but still consumes real
  VRAM at runtime.
- Use that to choose `max_num_batched_tokens` and, if needed, lower
  `gpu_memory_utilization`.

6. Choose `max_loras`.
- Under MinT-style diverse active LoRAs, choose:
  - `max_loras = N_long - 1`
  where `N_long` is the conservative guaranteed full-context concurrency target.

7. Choose `max_num_seqs`.
- Treat it as an admission cap for realistic mixed traffic, not as the same
  thing as guaranteed full-context concurrency.

8. Validate with real runs.
- Long sample.
- Long `compute_prompt_logprobs`.
- Multiple distinct LoRAs.
- Concurrent requests.

Hard rules:
- Do not size from checkpoint file size alone.
- Do not confuse KV reservation with total runtime memory pressure.
- Do not use "try a smaller setting" as the first move.
- Separate observation from inference when discussing empirical safety margins.
- If local patches bypass native vLLM startup profiling, state that the
  upstream sizing model is invalid until that behavior is removed.
- For `FusedMoEWithLoRA`, count all four persistent expert slot tensors
  (`w13_a`, `w13_b`, `w2_a`, `w2_b`); do not replace them with a shorthand
  "experts term" unless you have proved the tensor equivalence.
