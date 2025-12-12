# Tinker-Cookbook MoE Integration Test Report

**Model:** Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE, 8 GPUs)
**Server:** tinker-server on Volcano ML Platform
**Date:** 2025-12-11

---

## Test Summary

| Recipe | Status | Duration | Notes |
|--------|--------|----------|-------|
| Math RL (arithmetic) | PASS | 100 steps | Completed without errors |
| Chat SL (no_robots) | BLOCKED | - | Megatron forward lacks per-token logprobs |
| DPO (hhh) | BLOCKED | - | Megatron forward lacks per-token logprobs |
| Guess Number | PASS | 15+ steps | Reward: 0.42 -> 0.45 |
| Text Arena (TicTacToe) | PASS | 3+ steps | Reward: -0.38 -> -0.16 |
| Twenty Questions | - | - | Not tested |

---

## Detailed Results

### 1. Math RL (Arithmetic)

**Command:**
```bash
cd /home/yiwen/tinker_project/tinker-cookbook && \
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    renderer_name="qwen3_instruct" \
    group_size=4 groups_per_batch=8 learning_rate=1e-4 \
    max_tokens=16 eval_every=5 save_every=10
```

**Status:** PASS - Completed 100 steps

**Training Curve:**
```
Step | Reward | Accuracy
-----|--------|----------
10   | 0.45   | ~70%
50   | 0.52   | ~75%
100  | 0.55   | ~78%
```

**Notes:**
- Using smaller batch (groups_per_batch=8) for MoE memory constraints
- renderer_name="qwen3_instruct" bypasses model lookup
- Must kill vLLM before training to free GPU memory
- LoRA checkpoint saves work correctly (~0.16s)

---

### 2. Chat SL (NoRobots Dataset)

**Command:**
```bash
cd /home/yiwen/tinker_project/tinker-cookbook && \
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    renderer_name="qwen3_instruct" \
    dataset=no_robots \
    learning_rate=1e-4 batch_size=32 lora_rank=16 \
    eval_every=5 save_every=10
```

**Status:** BLOCKED - Megatron backend limitation

**Error:**
```
ValueError: zip() argument 2 is longer than argument 1
```

**Root Cause:**
Chat SL's NLL evaluator calls `forward` to compute per-token logprobs for evaluation. The Megatron distributed backend's `forward` method returns aggregate loss only, not per-token logprobs. Computing per-token logprobs requires gathering data across all tensor-parallel ranks, which is not implemented.

**Workaround Options:**
1. Use dense model backend (TrainingWorker) for SL tasks
2. Disable NLL evaluation in Chat SL recipe
3. Implement per-token logprob gathering in Megatron backend (complex)

---

### 3. DPO (HHH Dataset)

**Command:**
```bash
cd /home/yiwen/tinker_project/tinker-cookbook && \
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.preference.dpo.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    renderer_name="qwen3_instruct" \
    dataset=hhh \
    learning_rate=1e-5 dpo_beta=0.1 batch_size=32
```

**Status:** BLOCKED - Megatron backend limitation

**Error:**
```
IndexError: list index out of range
```

**Root Cause:**
DPO requires per-token logprobs from both policy and reference models. Same limitation as Chat SL - Megatron's `forward` returns aggregate loss, not per-token logprobs.

---

### 4. Guess Number

**Command:**
```bash
cd /home/yiwen/tinker_project/tinker-cookbook && \
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.multiplayer_rl.guess_number.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    renderer_name="qwen3_instruct" \
    group_size=4 batch_size=16 learning_rate=1e-4 \
    eval_every=5 save_every=10
```

**Status:** PASS

**Training Curve:**
```
Step | Reward | Notes
-----|--------|-------
2    | 0.42   | Initial
3    | 0.45   | Improving
15   | 0.47   | Training converging
```

**Notes:**
- Pure RL training (forward_backward + optim_step) works correctly
- No dependency on per-token logprobs
- LoRA checkpointing works

---

### 5. Text Arena (TicTacToe)

**Command:**
```bash
cd /home/yiwen/tinker_project/tinker-cookbook && \
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    renderer_name="qwen3_instruct" \
    batch_size=64 learning_rate=1e-4 \
    eval_every=5 save_every=10
```

**Status:** PASS

**Training Curve:**
```
Step | Reward | Notes
-----|--------|-------
0    | -0.38  | Initial (losing)
1    | -0.34  | Improving
2    | -0.16  | Significant improvement
```

---

## Issues Encountered

### 1. MegatronWorkerGroup missing `forward` method

**Symptom:** `'ActorHandle' object has no attribute 'forward'`

**Fix:** Added `forward` method to both `MegatronRankWorker` and `MegatronWorkerGroup` in `megatron_distributed.py`. Uses verl engine's `forward_backward_batch` with `forward_only=True`.

**Limitation:** Returns aggregate loss only, not per-token logprobs. Chat SL and DPO require per-token logprobs.

### 2. OOM with vLLM + Megatron

**Symptom:** `torch.OutOfMemoryError: CUDA out of memory`

**Fix:** Kill vLLM actor before Megatron training:
```bash
curl -X POST http://localhost:8000/api/v1/kill_vllm
```

---

## Performance Notes

- MoE model initialization: ~210s (first session), ~0.1s (reuse)
- forward_backward: ~8-9s per batch
- optim_step: ~4s
- LoRA extraction: ~0.16s
- vLLM inference: ~2s

---

## Environment

```
Server: volcano SSH server (no GPU)
Ray cluster: 24 GPUs (3x 8-GPU workers)
Model: Qwen3-30B-A3B (128 experts, top-2 routing)
Parallelism: TP=4, EP=2, world_size=8
vLLM: 0.12.0 (MoE LoRA support)
```

---

## Conclusion

**RL training works:** Math RL, Guess Number, and Text Arena all pass. The Megatron distributed backend correctly handles `forward_backward`, `optim_step`, and LoRA checkpoint saving.

**SL/DPO training blocked:** Chat SL and DPO require per-token logprobs from the `forward` method, which the Megatron backend doesn't provide. These recipes need either:
- Dense model backend (single-GPU TrainingWorker)
- Modified recipes that skip NLL evaluation
- Megatron backend enhancement to gather per-token logprobs across TP ranks
