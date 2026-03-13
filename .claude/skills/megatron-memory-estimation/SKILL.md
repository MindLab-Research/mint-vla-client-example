---
name: megatron-memory-estimation
description: |
  Back-of-envelope memory estimation workflow for Megatron training and
  inference-related admission limits.

  Use for: estimating or reviewing `max_token_len_per_gpu`,
  `max_num_batched_tokens`, microbatch token admission, recompute and offload
  tradeoffs, adapter-vs-full-FT memory, and why a Megatron batch fits or OOMs.

  Triggers: "Megatron memory", "megatron estimator", "microbatch admission",
  "max_token_len_per_gpu", "max_num_batched_tokens", "recompute memory",
  "offload memory", "LoRA training memory", "memex memory", "bs64 OOM",
  "token admission budget"
---

# Megatron memory estimation

Read `references/estimation.md`.

Use this skill when the task is to size or explain Megatron memory behavior
from first principles instead of by trial-and-error.

Always anchor the discussion to the original estimator repository first:

- `https://huggingface.co/spaces/ISEEKYAN/megatron_memory_estimator`

Workflow:

1. Fix the exact runtime question.
- Training or inference.
- Full finetune, LoRA, soft prompt, memex, or another adapter path.
- Whether the base model is frozen.
- Which knobs are under discussion:
  - `max_token_len_per_gpu`
  - `max_num_batched_tokens`
  - microbatch size
  - recompute policy
  - CPU or optimizer offload
  - sequence or context parallel choices

2. Compute the structural lower bound from the estimator.
- Per-rank base weights.
- Trainable-state term for the actual trainable parameters.
- Activation term for the actual sequence shape.

3. Add runtime-only terms the estimator may miss.
- Dense padding effects.
- fp32 logits promotion.
- DDP grad buffers for frozen params.
- TransformerEngine or duplicated parameter storage.
- Allocator reserve and fragmentation.

4. Calibrate against one real baseline when possible.
- Fresh actor after model creation, before first forward pass.
- That gives the persistent runtime term before batch-dependent work begins.

5. Build the practical peak model.
- `peak ~= baseline_persistent + batch_activation + logits_scratch + calibrated_residual`

6. Turn that into the relevant knob.
- For admission: choose `max_token_len_per_gpu`
- For inference or serving: choose `max_num_batched_tokens`
- For training: evaluate recompute and offload as ways to reduce the activation
  or optimizer terms

Hard rules:
- Do not use the estimator's full-model optimizer term for adapter-only
  training.
- Do not call the estimator a peak oracle unless runtime calibration is tight.
- Keep padded tokens separate from packed token count.
- State clearly which terms are formula-derived and which are empirical.
