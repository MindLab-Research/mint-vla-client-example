# Mint Server Roadmap

## Overview

**Mint** is a multi-tenant training/inference server compatible with Tinker APIs.

- Separated inference and training on different GPU pools
- LoRA-based training with sub-second adapter transfer
- Dense models via PEFT, MoE models via verl Megatron backend
- Multi-session actor sharing with LRU eviction

---

## Current State (Completed)

| Feature | Status |
|---------|--------|
| Dense model training (Qwen2.5-7B) | Verified |
| MoE model training (Qwen3-30B-A3B) | Verified |
| Multi-session actor sharing | Verified |
| Unified rank support (max-rank padding) | Verified |
| LRU-based actor eviction | Verified (unified ResourcePool with cross-actor LRU) |
| LoRA hot-swap to vLLM | Verified (0.16s extraction + 2.2s inference) |

---

## Roadmap

### 1. Model Lineup

#### T0 (Week 1-2) - Foundation

| Model | Type | Architecture | Train GPUs | Infer GPUs | Notes |
|-------|------|--------------|------------|------------|-------|
| Qwen/Qwen3-0.6B | Hybrid | Dense | 1 | 1 | **Verified** (90.8% loss reduction) |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | Instruction | MoE | 8 (TP4,EP2) | 4 (TP4) | **Verified** |
| moonshotai/Moonlight-16B-A3B-Instruct | Instruction | DeepSeekV3 MoE | 16 (TP2,EP8) | 2 (TP2) | **Verified** (MLA via value padding) |
| moonshotai/Kimi-K2-Thinking | Reasoning | DeepSeekV3 MoE | 64+ (H100+) | 32+ | MLA works (value padding), memory blocked |

#### T1 (Week 2-3) - Scale Up

| Model | Type | Architecture | Train GPUs | Infer GPUs | Notes |
|-------|------|--------------|------------|------------|-------|
| Qwen/Qwen3-4B-Instruct-2507 | Instruction | Dense | 1 | 1 | |
| Qwen/Qwen3-8B | Hybrid | Dense | 2 | 1 | |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | Instruction | MoE | 32 (TP8,EP4) | 16 | Multi-node |
| deepseek-ai/DeepSeek-V3.1 | Hybrid | MoE | 64+ | 32+ | Different MoE architecture |

#### T2 (Week 4+) - Multimodal & Embodied

| Model | Type | Architecture | Train GPUs | Infer GPUs | Notes |
|-------|------|--------------|------------|------------|-------|
| Qwen/Qwen3-VL-30B-A3B-Instruct | Vision | MoE | TBD | TBD | Vision encoder integration |
| Qwen/Qwen3-VL-235B-A22B-Instruct | Vision | MoE | TBD | TBD | |
| physical-intelligence/pi0 | VLA | PaliGemma+Expert | TBD | TBD | Flow matching, 50Hz actions |
| physical-intelligence/pi0.5 | VLA | PaliGemma+Expert | TBD | TBD | Open-world generalization |
| physical-intelligence/pi0-fast | VLA | PaliGemma+Expert | TBD | TBD | FAST action tokenizer |

#### Testing Strategy

Each tier involves:
1. Functional testing with supported models (SFT, RL, DPO)
2. Scientific experiments (base models, LoRA ranks, learning rates)
3. Testing with unsupported but compatible models (e.g., Llama dense if Qwen dense works)

#### Technical Challenges

| Model | Challenge | Mitigation |
|-------|-----------|------------|
| Kimi-K2 | 64 GPU memory requirement, Block-FP8 | MLA solved via value padding, need 64+ GPUs |
| DeepSeek-V3.1 | Different MoE routing | Architecture analysis needed |
| VL models | Vision encoder, multimodal inputs | New modality support |
| pi0/pi0.5 (VLA) | See VLA investigation below | Tentative - may require new backend |

#### K2 Support Details

**Model Specifications (Kimi-K2)**

| Spec | Value |
|------|-------|
| Total params | 1.04 trillion |
| Active params | 32B per token |
| Experts | 384 total, 8+1 per token |
| Hidden dim | 7168 |
| Context window | 128K tokens |
| Architecture | DeepSeek-V3 style MoE with MLA attention |
| Quantization | Block-FP8 (not per-tensor FP8) |

**References**: [HuggingFace](https://huggingface.co/moonshotai/Kimi-K2-Instruct), [Technical Report](https://arxiv.org/pdf/2507.20534)

**Framework Landscape**

Three relevant frameworks operate at different abstraction levels:

| Framework | Purpose | Megatron Access | Distributed | Data Format |
|-----------|---------|-----------------|-------------|-------------|
| **Verl** | PPO/RLHF workflows | `MegatronEngine` wrapper | Ray actors | `DataProto`, `TensorDict` |
| **MS-Swift** | Research training | Direct `megatron.training` | torch.distributed | HF datasets |
| **Tinker-Server** | Multi-tenant API | Via verl wrapper + direct calls | Ray actors | Custom `Datum` |

**What tinker-server uses from verl**:
- `MegatronEngineWithLMHead` - engine wrapper
- `HFModelConfig`, `McoreEngineConfig` - config dataclasses
- `vLLMHttpServerBase` - Ray actor base class
- Utility functions: `copy_to_local`, `get_adapter_state_dict`

**What tinker-server does NOT use**:
- `MegatronWorker` (PPO-specific actor/critic orchestration)
- `DataProto` (PPO-specific data format)
- Dispatch decorators (`@register`, `Dispatch`)

**k2-workspace reference implementation** (ms-swift based):
- Location: `../k2-workspace/workspace/ms-swift/examples/megatron/grpo/kimi-k2/`
- Config: `moe_colocate_lora.sh` - 64 GPU setup (8 nodes × 8 GPUs)
- Parallelism: TP=8, EP=64, PP=1, CP=1

**GPU Requirements**

| Configuration | GPUs | Notes |
|---------------|------|-------|
| Full (reference) | 64× H100 80GB | TP=8, EP=64, as in k2-workspace |
| Minimum (with FP8) | 16× H100 80GB | TP=8, EP=16, requires FP8 + offload |
| Minimum (INT4 inference only) | 8× H100 80GB | Inference only, no training |

**16-GPU Configuration** (for debugging/development):
```bash
COMMON_TP=8 COMMON_EP=16 COMMON_PP=1 COMMON_CP=1 INFER_TP=16
--fp8-param-gather                    # FP8 weights (H100 required)
--recompute-granularity full          # Activation checkpointing
--optimizer-cpu-offload               # Optimizer to CPU
--train_type lora --lora_rank 8
--micro_batch_size 1 --max_model_len 4096
```

Expected: 5-10× slower than 64-GPU baseline due to expert memory pressure and offloading.

**Required Code Changes**

| File | Change | Priority |
|------|--------|----------|
| `megatron_training.py:903` | Add `r"Kimi-K2"` to `moe_patterns` | Critical |
| `model_registry.py` | Add K2 config: `ModelConfig(True, 8, 8)` | Critical |
| `verl_training.py:941` | Use model registry instead of hardcoded TP=4, EP=2 | Critical |
| `megatron_distributed.py:267` | Add FP8 dtype support to `McoreEngineConfig` | Critical |
| `verl_inference.py:817` | Add `quantization="fp8"` to vLLM kwargs | High |
| `megatron_training.py:204` | Extract auxiliary loss from MoE router | Medium |

**Implementation Phases**

| Phase | Tasks | Effort |
|-------|-------|--------|
| 1. Detection | Add K2 to `is_moe_model()`, populate registry | 1-2h |
| 2. Parallelism | Dynamic TP/EP from registry, remove hardcoding | 2-3h |
| 3. FP8 | Add dtype detection and config plumbing | 3-4h |
| 4. Validation | Test standalone vLLM + ms-swift before integration | 4-8h |

**Pre-integration validation** (run before code changes):
```bash
# Test 1: vLLM inference with FP8
vllm serve moonshotai/Kimi-K2-Instruct \
    --tensor-parallel-size 8 --quantization fp8

# Test 2: ms-swift training (16 GPUs)
cd ../k2-workspace/workspace/ms-swift/examples/megatron/grpo/kimi-k2
COMMON_TP=8 COMMON_EP=16 INFER_TP=16 bash moe_colocate_lora.sh 2 0 127.0.0.1
```

If standalone tests fail, the issue is upstream (vLLM/Megatron K2 support), not tinker-server.

#### VLA Models Investigation (Tentative)

**What are VLA models?** Vision-Language-Action models for robot control. Output continuous action trajectories instead of text tokens.

**Architecture (pi0):**
- Base: PaliGemma 3B VLM backbone
- Action expert: +300M params (initialized from scratch)
- Total: ~3.3B parameters
- Output: Continuous action vectors at 50Hz via flow matching

**Key differences from standard VLM fine-tuning:**

| Aspect | Standard VLM | VLA (pi0) |
|--------|--------------|-----------|
| Output | Discrete text tokens | Continuous action vectors |
| Training objective | Cross-entropy | Flow matching |
| Inference rate | Variable | Fixed 50Hz real-time |
| Additional inputs | Image + text | Image + text + robot state |
| Expert module | None | 300M action expert |

**Implementation requirements:**

1. **Flow matching support** - Different from cross-entropy loss, generates smooth trajectories
2. **Action expert module** - Additional trainable module beyond VLM backbone
3. **Robot state inputs** - Proprioceptive data (joint positions, velocities)
4. **Continuous output** - Not discrete token prediction
5. **Real-time inference** - 50Hz control loop requirements

**Framework status:**
- Native: JAX with FSDP
- PyTorch: Recently added (DDP, multi-node via torchrun)
- LoRA fine-tuning: Supported (>22.5 GB VRAM)
- Full fine-tuning: >70 GB VRAM (A100/H100)

**Open questions:**
1. Can verl/Megatron support PaliGemma architecture?
2. How to integrate flow matching into existing training pipeline?
3. Is there demand for VLA fine-tuning via Tinker-style API?
4. Alternative: Direct integration with openpi repo?

**References:**
- [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- [pi0 Paper](https://www.physicalintelligence.company/download/pi0.pdf)
- [pi0.5 Paper](https://arxiv.org/abs/2504.16054)
- [HuggingFace Blog](https://huggingface.co/blog/pi0)

---

### 2. Resource Orchestration - COMPLETE

#### Implementation (2025-12-16)

Unified `ResourcePool` singleton with global GPU tracking and cross-actor LRU eviction.

```
ResourcePool (tinker_server/backend/resource_pool.py)
├── _entries: dict[actor_name, ResourceEntry]
├── register(actor_name, actor_type, num_gpus, ...)
├── ensure_gpus_available(needed_gpus) → evicts LRU idle actors
├── evict_for_gpus(needed_gpus) → int (GPUs freed)
└── list_actors() → status for monitoring
```

**Actor Types Tracked**:
| Type | Actor Name | GPU Usage |
|------|------------|-----------|
| MEGATRON | `persistent_megatron_worker_group_v2` | 8 (TP4,EP2) |
| DENSE | `dense_trainer_*` | 1 |
| VLLM | `tinker_vllm_server` | 1-4 |

**Monitoring**: `GET /api/v1/resource_pool`

#### GPU Requirements Reference

| Model | Inference | Training | Total |
|-------|-----------|----------|-------|
| Qwen3-0.6B | 1 | 1 | 2 |
| Qwen2.5-7B | 1 | 1 | 2 |
| Qwen3-8B | 1 | 2 | 3 |
| Qwen3-30B-A3B | 4 (TP4) | 8 (TP4,EP2) | 12 |
| Qwen3-235B-A22B | 16 (TP8) | 32 (TP8,EP4) | 48 |
| Kimi-K2 | 32+ | 64+ | 96+ |
| DeepSeek-V3.1 | 32+ | 64+ | 96+ |

#### Tasks

| Task | Priority | Complexity |
|------|----------|------------|
| Create unified `ResourceManager` | High | Medium |
| Implement cross-pool eviction | High | Medium |
| Add GPU requirement registry per model | Medium | Low |
| Future: multi-worker scaling for high load | Low | High |

---

### 3. Procedure Standardization

#### Branch Strategy

- `develop`: All development work
- `main`: Production-ready code, requires gate pass

#### Merge Gate (Claude Skill)

A skill that guides the agent through manual testing before merge:

| Step | Test | Pass Criteria |
|------|------|---------------|
| 1 | Bring up clean dev cluster | Cluster healthy, GPUs available |
| 2 | Functional test (dense) | SFT/RL/DPO with Qwen3-0.6B, loss decreases |
| 3 | Functional test (MoE) | SFT/RL/DPO with Qwen3-30B-A3B, loss decreases |
| 4 | API alignment test | tinker_test.ipynb passes |
| 5 | Stress test | 10 concurrent sessions, no crashes/deadlocks |

#### Tinker Comparison Skill

Compare Mint against official Tinker API:

- **Caching**: Store Tinker results in `results/tinker_baseline/` to avoid repeated API calls
- **Metrics**: Loss correlation (r > 0.99), final loss diff (< 1%), wall time ratio
- **API key**: `TINKER_API_KEY` in `.env` (gitignored)

#### Environment Separation

| Property | Dev | Prod |
|----------|-----|------|
| Port | 8000 | 18000 |
| API key required | No | Yes |
| SSH host | `volcano` | `mint-prod` |
| Unison profile | `volcano-tinker` | `volcano-tinker-auth` |

#### Deployment

After merge to main: automated deployment to prod cluster (nightly).

---

### 4. Cleanup

#### Project Rename: tinker-server → mint

| Category | Count | Action |
|----------|-------|--------|
| Directory | 1 | `tinker_server/` → `mint/` |
| Imports | ~50 files | `from tinker_server` → `from mint` |
| Config vars | ~10 | `TINKER_MODEL_PATH` → `MINT_MODEL_PATH` |
| Skills | 4 | Update paths |

Keep `TINKER_BASE_URL`, `TINKER_API_KEY` for client-side Tinker API compatibility.

#### Test Script Consolidation

Current: 39 scripts with significant overlap.

| Keep | Merge | Purpose |
|------|-------|---------|
| `test_sft_loop.py` | - | Dense SFT baseline |
| `test_moe_training.py` | 8 `test_moe_*.py` | MoE training |
| `test_concurrent_sessions.py` | stress scripts | Load testing |
| `test_phase6_isolation.py` | - | Session isolation |
| `test_time_sliced_isolation.py` | New | Correctness verification |
| `test_tinker_api_alignment.py` | New | API compatibility gate |

#### Common Test Utilities

Extract duplicated code into `tests/utils.py`:
- `poll_future()`
- `create_session()`
- `TinkerTestClient` class

#### Code Smell Watch List

- Duplicated `poll_future()` across scripts
- Hardcoded model names
- Similar helper functions

---

### 5. Agent Awareness Improvements

#### Problem

Common debugging loop wastes time:
1. Restart server (actors keep old code)
2. Test fails with stale behavior
3. Realize actors need killing
4. Kill actors, wait for GPU release
5. Retry

#### Solution

Add pre-flight checks to skills:

```bash
# Quick status command
ssh volcano 'python3 -c "
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
print(f\"GPUs: {r.get('GPU', 0):.0f} / {t.get('GPU', 0):.0f}\")
for name in ['persistent_megatron_worker_group_v2', 'tinker_vllm_server']:
    try:
        ray.get_actor(name, namespace='tinker')
        print(f'{name}: ALIVE')
    except ValueError:
        print(f'{name}: not running')
"'
```

#### Decision Matrix

| Code Changed | Actors Running | Action |
|--------------|----------------|--------|
| `megatron_*.py` | Megatron alive | Kill Megatron → restart server |
| `vllm_*.py` | vLLM alive | Kill vLLM → restart server |
| Routes only | Any | Restart server only |
| Any | 0 GPUs available | Kill idle actors → free GPUs → proceed |

---

### 6. Stateless Trainer Architecture - COMPLETE

#### Implementation (2025-12-16)

`SessionStateManager` persists state per-iteration for stateless trainers:

```
/tmp/mint_sessions/{session_id}_checkpoint/
├── adapter_model.safetensors  # LoRA weights
├── optimizer.pt               # Adam state (exp_avg, exp_avg_sq)
└── training_meta.json         # step count, learning_rate
```

**Key Methods**:
- `_ensure_session_loaded(session_id)`: Load if different from current
- `_save_session_state(session_id)`: Save after optim_step
- Overhead: ~0ms same session, ~100-200ms on switch

**Verification**: Interleaved sessions test passes (Session A → B → A continues from correct loss)

**Files Modified**:
- `verl_training.py`: Added `SessionStateManager`, modified `TrainingWorker`
- `scripts/test_interleaved_sessions.py`: Integration test

---

### 7. Tinker API Alignment

Detailed verification that Mint matches official Tinker SDK behavior.

#### Reference Test Cases (from tinker_test.ipynb)

| Test | Description | Expected Behavior |
|------|-------------|-------------------|
| 1. Service Client | `ServiceClient.get_server_capabilities()` | List supported models |
| 2. Training Client | `create_lora_training_client(base_model)` | Returns client with tokenizer |
| 3. Data Preparation | `types.Datum`, `types.ModelInput.from_ints()` | Token/weight format |
| 4. Forward-Backward | `forward_backward(data, "cross_entropy")` | Returns logprobs per token |
| 5. Optim Step | `optim_step(AdamParams(learning_rate=1e-4))` | Updates weights |
| 6. Loss Computation | Client-side: `-dot(logprobs, weights) / sum(weights)` | Matches server loss |
| 7. Sampling | `save_weights_and_get_sampling_client()` | Hot-reload LoRA |
| 8. Sample Generation | `sample(prompt, params, num_samples)` | Returns sequences |
| 9. Prompt Logprobs | `include_prompt_logprobs=True` | Returns per-token logprobs |
| 10. Top-k Logprobs | `topk_prompt_logprobs=5` | Returns top-k per position |
| 11. Save for Sampler | `save_weights_for_sampler(name)` | Returns path |
| 12. Save State | `save_state(name)` | Resume checkpoint path |
| 13. Load State | `load_state(path)` | Restores training state |

#### API Mapping: Tinker SDK → Mint

| Tinker SDK | Mint Endpoint | Notes |
|------------|---------------|-------|
| `get_server_capabilities()` | `/api/v1/healthz` | Extend for model list |
| `create_lora_training_client()` | `/api/v1/create_model` | Same |
| `forward_backward()` | `/api/v1/forward_backward` | Returns logprobs |
| `optim_step()` | `/api/v1/optim_step` | Same |
| `save_weights_and_get_sampling_client()` | `/api/v1/save_weights` | Hot-reload |
| `sample()` | `/api/v1/asample` | Same |
| `save_weights_for_sampler()` | `/api/v1/save_weights` | With name |
| `save_state()` | `/api/v1/save_state` | **NEW** |
| `load_state()` | `/api/v1/load_state` | **NEW** |

#### Test Cases

**Test 6.1: Pig Latin SFT (Dense)**

Replicate notebook's Pig Latin translation with Qwen2.5-7B.

| Metric | Target |
|--------|--------|
| Update 1 loss | ~2.45 |
| Update 6 loss | ~0.58 |
| Loss reduction | >75% |
| Iteration time | <2s |

**Test 6.2: Pig Latin SFT (MoE)**

Same task with Qwen3-30B-A3B.

| Metric | Target |
|--------|--------|
| Update 1 loss | ~2.5-3.0 |
| Update 10 loss | <1.0 |
| Loss reduction | >60% |
| Iteration time | <8s |

**Test 6.3: Forward-Backward Logprobs**

Verify client-computed loss matches server `metrics['loss:mean']`. Pass: diff < 0.01.

**Test 6.4: Prompt Logprobs**

Verify `include_prompt_logprobs=True` returns per-token logprobs. First token = 0.0.

**Test 6.5: Checkpoint Round-Trip**

Train 5 iter → save → train 5 more → load → verify loss matches step 6 from first run. Pass: diff < 0.05.

**Test 6.6: Hot-Reload Sampling**

Train on Pig Latin → `save_weights_and_get_sampling_client()` → sample → output differs from base model.

#### Data Format Comparison

| Aspect | Tinker SDK | Mint | Compatible? |
|--------|------------|------|-------------|
| Per-token weights | `weights` | `weights` or `loss_mask` | **Yes** - both accepted |
| Model input | `chunks` with `EncodedTextChunk` | `chunks` format | Yes |
| Tensor format | `TensorData{data, shape, dtype}` | `{data, shape, dtype}` | Yes |
| Loss functions | `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro` | Partial | `cispo`, `dro` pending |

#### Implementation Tasks

| Task | Priority | Status |
|------|----------|--------|
| Accept both `loss_mask` and `weights` field names | **Critical** | **DONE** |
| Verify `cross_entropy`, `importance_sampling`, `ppo` loss functions | High | **DONE** |
| Add `/api/v1/save_state` endpoint | High | **DONE** |
| Add `/api/v1/load_state` endpoint | High | **DONE** |
| Accept both `X-API-Key` and `Authorization: Bearer` auth | High | **DONE** |
| Accept both `sampling_session_id` and `model_id` in sample requests | High | **DONE** |
| Create merge-gate test suite | High | **DONE** (19/20 pass) |
| Verify logprobs format matches `TensorData` spec | High | **DONE** |
| Verify `include_prompt_logprobs` | Medium | **DONE** |
| Add `cispo`, `dro` loss functions | Medium | Pending |
| Add `forward_backward_custom` for arbitrary loss functions | Medium | Pending |
| Verify LoRA config options: `train_unembed`, `train_mlp`, `train_attn` | Medium | Pending |
| Verify `topk_prompt_logprobs` | Low | Pending |

---

### 8. DPO Support

#### Problem Statement

DPO (Direct Preference Optimization) requires comparing chosen vs rejected responses using reference model logprobs. Current implementation lacks reference model support.

#### DPO Loss Function

```
L_DPO = -log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
```

Where:
- `y_w`: chosen (winning) response
- `y_l`: rejected (losing) response
- `π`: policy model (being trained)
- `π_ref`: reference model (frozen)
- `β`: temperature parameter (default: 0.1)

#### Required Data Format (Tinker API)

```python
{
    "model_input": {"chunks": [{"tokens": [...], "type": "encoded_text"}]},
    "loss_fn_inputs": {
        "target_tokens": {"data": [...], "shape": [...], "dtype": "int64"},
        "loss_mask": {"data": [...], "shape": [...], "dtype": "float32"},
        "ref_logprobs": {"data": [...], "shape": [...], "dtype": "float32"},  # NEW
        "is_chosen": true/false,  # NEW
    }
}
```

#### Implementation Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: External ref logprobs** | Client computes ref_logprobs before training | No extra GPU, simple implementation | Requires separate inference pass |
| **B: Dual model loading** | Load ref model alongside policy model | One-shot training | 2x GPU memory |
| **C: Cached ref logprobs** | Pre-compute ref logprobs on session create | Good throughput | Storage overhead, staleness |

**Recommended: Option A** - Client-side ref logprobs computation matches tinker-cookbook pattern.

#### Implementation Tasks

| Task | Priority | Status |
|------|----------|--------|
| Add `dpo` loss function to TrainingWorker.forward_backward | High | Pending |
| Add `dpo` loss function to MegatronRankWorker | High | Pending |
| Add `ref_logprobs` and `is_chosen` to datum parsing | High | Pending |
| Create DPO merge-gate test | High | Pending |
| Document DPO workflow in API reference | Medium | Pending |

#### Example Workflow

```python
# 1. Get ref logprobs using forward-only pass (no gradients)
ref_result = training_client.forward(data=[chosen_datum, rejected_datum])
ref_logprobs_chosen = ref_result["loss_fn_outputs"][0]["logprobs"]
ref_logprobs_rejected = ref_result["loss_fn_outputs"][1]["logprobs"]

# 2. Add ref_logprobs to training data
chosen_datum["loss_fn_inputs"]["ref_logprobs"] = {"data": ref_logprobs_chosen, ...}
chosen_datum["loss_fn_inputs"]["is_chosen"] = True
rejected_datum["loss_fn_inputs"]["ref_logprobs"] = {"data": ref_logprobs_rejected, ...}
rejected_datum["loss_fn_inputs"]["is_chosen"] = False

# 3. Train with DPO loss
result = training_client.forward_backward(data=[chosen_datum, rejected_datum], loss_fn="dpo")
training_client.optim_step(adam_params)
```

---

## Known Issues

### LRU Eviction Not Wired Up - RESOLVED (2025-12-16)

**Problem**: `_evict_for_gpus()` method existed in `MegatronActorPool` and `DenseTrainerPool` but was never called.

**Solution**: Created unified `ResourcePool` singleton (`tinker_server/backend/resource_pool.py`) tracking ALL GPU-using actors with cross-actor LRU eviction.

**Implementation**:
- `ResourcePool.register()` called when creating vLLM, Megatron, and Dense actors
- `ResourcePool.ensure_gpus_available()` called before actor creation - evicts LRU idle actors if needed
- MIN_ACTOR_AGE = 300s prevents eviction thrashing
- `/api/v1/resource_pool` endpoint for monitoring

**Files modified**:
- `resource_pool.py` (new)
- `megatron_distributed.py` (added registration and ensure_gpus_available)
- `verl_training.py` (added registration)
- `multi_lora_engine.py` (added registration)
- `routes/service.py` (added monitoring endpoint)

**Limitation**: MoE requires STRICT_PACK (8 GPUs on single node). Evicting actors on different nodes doesn't help MoE creation.

---

### train_step API Breaks Tinker Compatibility

**Problem**: With MoE + `param_offload=True`, gradients are zeroed between `forward_backward` and `optim_step` calls.

**Current workaround**: `/api/v1/train_step` combines both in single context.

**Mitigation**: For LoRA, `grad_offload=False` keeps gradients in GPU memory.

**TODO**: Verify that LoRA + `grad_offload=False` allows standard Tinker API.

### Time-Sliced Isolation Not Yet Verified

Previous tests prove sessions have different weights (A ≠ B), but don't prove training is correct (A_interleaved == A_exclusive).

**TODO**: Implement `test_time_sliced_isolation.py` to verify stronger correctness condition.

---

### vLLM MoE Expert LoRA Inference - INVESTIGATION UPDATE (2025-12-17)

**Problem**: vLLM 0.12.0 has `FusedMoEWithLoRA` class but cannot load MoE expert LoRA weights for inference. Training with full MLP+attention LoRA works via Megatron, but inference is limited to attention-only LoRA.

**Impact**: MoE models (Qwen3-30B-A3B) train with full LoRA (attention + expert MLP) but inference only uses attention LoRA. This reduces LoRA effectiveness during inference while training quality remains unaffected.

#### vLLM 0.13.0rc2 Investigation (2025-12-17)

**CRITICAL FINDING: Known Bug - LoRA Loading Broken for Qwen3 MoE**

This is a **confirmed bug** in vLLM V1 engine with Qwen3 MoE models:
- GitHub Issue: [vllm-ascend #3377](https://github.com/vllm-project/vllm-ascend/issues/3377) - "Qwen3-30B-A3B cannot use enable-lora"
- Status: **OPEN** (as of 2025-12-17)
- Root cause: V1 engine `set_active_loras` fails during inference when LoRA adapter is requested
- Impact: **All LoRA loading broken** (not just expert LoRA - even attention-only LoRA fails)

**V0 Engine No Longer Available:**
- vLLM 0.13.0rc2 has V1 engine ONLY - V0 was removed
- Setting `VLLM_USE_V1=0` has no effect
- Cannot fall back to V0 engine as a workaround

**What was tested:**

| Test | Result |
|------|--------|
| Model init with `enable_lora=True` | PASSED |
| "MoE model detected. Using fused MoE LoRA implementation." | CONFIRMED (all TP workers) |
| Baseline generation (no LoRA) | PASSED |
| Create synthetic LoRA adapter (attention-only) | PASSED |
| Load attention-only LoRA adapter | **FAILED** (WorkerProc exception) |
| V0 engine fallback (`VLLM_USE_V1=0`) | **NOT AVAILABLE** (V0 removed) |
| Megatron → vLLM expert weight export | BLOCKED (basic LoRA fails first) |

**Error Location:**
```
gpu_model_runner.py:3005 → execute_model → _prepare_inputs → set_active_loras
  → lora_model_runner_mixin.py:70 → make_lora_inputs → _set_active_loras
  → WorkerProc exception on all TP workers
```

**Source Code Analysis:**

1. **`get_supported_lora_modules()` in `vllm/lora/utils.py`** includes both:
   - `LinearBase` subclasses (attention: qkv_proj, o_proj)
   - `FusedMoE` instances (expert layers)

2. **`FusedMoEWithLoRA` class exists** with full implementation:
   - Expert-specific weight shapes: `(num_experts, rank, hidden_size)`
   - Per-expert LoRA application via `add_lora_fused_moe()` kernel

3. **V1 Engine LoRA Pipeline (where it fails):**
   - `set_active_loras()` in `lora_model_runner_mixin.py:70`
   - Creates `LoRAMapping` from `make_lora_inputs()`
   - Calls `lora_manager.set_active_adapters()` → `_adapter_manager.set_adapter_mapping()`
   - Fails at `punica_wrapper.update_metadata()` for MoE models

**Conclusion:**
- vLLM 0.13.0rc2 infrastructure exists but is **broken** for Qwen3 MoE + LoRA
- This blocks both attention LoRA AND expert LoRA
- Must wait for upstream fix or downgrade to vLLM < 0.13.0

**NOT POSSIBLE UNTIL BUG FIXED:**
- Any LoRA inference on Qwen3 MoE models
- Megatron → vLLM expert LoRA weight export testing

#### Qwen1.5-MoE-A2.7B-Chat Testing (2025-12-18)

**BREAKTHROUGH: MoE Expert LoRA Works on Qwen2MoeForCausalLM**

Testing with the smaller Qwen1.5-MoE-A2.7B-Chat model (Qwen2MoeForCausalLM architecture) reveals that vLLM 0.13.0rc2 MoE LoRA **does work** for Qwen2 MoE models - the bug is **Qwen3-specific**.

| Model | Architecture | Result |
|-------|--------------|--------|
| Qwen3-30B-A3B-Instruct-2507 | Qwen3MoeForCausalLM | **WORKS** (see correction below) |
| Qwen1.5-MoE-A2.7B-Chat | Qwen2MoeForCausalLM | **WORKS** |

#### Qwen3-30B-A3B Testing - CORRECTION (2025-12-18)

**PREVIOUS CLAIM WAS INCORRECT**: The earlier finding that "LoRA loading is broken for Qwen3 MoE" was wrong. The cited issue (vllm-ascend #3377) is from the wrong repository (vllm-ascend, not vllm).

**ACTUAL ROOT CAUSE**: Using incorrect target modules. Qwen3MoeForCausalLM uses separate `q_proj`, `k_proj`, `v_proj` - NOT fused `qkv_proj`.

**Test Results on Ray Cluster (TP=4):**

| Test | Target Modules | Result |
|------|----------------|--------|
| Attention LoRA with `qkv_proj` | `["qkv_proj"]` | **FAILED** - module not supported |
| Attention LoRA with separate q/k/v | `["q_proj", "k_proj", "v_proj"]` | **PASSED** |
| Expert LoRA (all 128 experts) | `["experts.N.gate_proj", "experts.N.up_proj", "experts.N.down_proj"]` | **PASSED** |

**Error when using incorrect modules:**
```
ValueError: While loading /tmp/test_qwen3_30b_lora, expected target modules in
{'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate', 'experts.{N}.gate_proj', ...}
but received ['model.layers.0.self_attn.qkv_proj', ...]
```
Location: `vllm/lora/lora_model.py:168` in `check_unexpected_modules()`

**Expert LoRA Test Details:**
- Model config: 48 layers, 128 experts, hidden_size=2048, moe_intermediate_size=768
- Tensors created: 36,864 (48 layers × 128 experts × 3 projections × 2 A/B)
- vLLM version: 0.13.0rc2.dev207+g811cdf519
- Log: `"MoE model detected. Using fused MoE LoRA implementation."`

**Key Requirements for Qwen3-30B-A3B LoRA:**

1. **Use separate attention projections**:
   ```python
   "target_modules": ["q_proj", "k_proj", "v_proj"]  # NOT qkv_proj
   ```

2. **Expert LoRA requires ALL experts**:
   ```python
   target_modules = []
   for e in range(128):  # All 128 experts
       target_modules.extend([
           f"experts.{e}.gate_proj",
           f"experts.{e}.up_proj",
           f"experts.{e}.down_proj",
       ])
   ```

3. **Weight naming convention (PEFT format)**:
   ```python
   # Attention
   f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"
   f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight"

   # Expert
   f"base_model.model.model.layers.{layer}.mlp.experts.{expert}.gate_proj.lora_A.weight"
   f"base_model.model.model.layers.{layer}.mlp.experts.{expert}.gate_proj.lora_B.weight"
   ```

**Test Scripts:**
- Attention LoRA: `scripts/test_qwen3_30b_lora_correct.py`
- Expert LoRA: `scripts/test_qwen3_30b_expert_lora.py`
- Megatron → vLLM Integration: `scripts/test_megatron_qwen3_moe_export.py`

#### Megatron → vLLM MoE LoRA Integration Test (2025-12-18)

**Full pipeline test:** Megatron weight export → PEFT conversion → vLLM load

| Metric | Value |
|--------|-------|
| Megatron tensors created | 37,152 |
| PEFT conversion success | 37,152 (100%) |
| Conversion failures | 0 |
| vLLM initialization | PASS |
| Baseline generation | PASS |
| LoRA adapter load | PASS |

**Tensor breakdown:**
- Attention: 48 layers × 3 projections (q/k/v) × 2 (A/B) = 288
- Experts: 48 layers × 128 experts × 3 projections × 2 (A/B) = 36,864
- Total: 37,152

**Name conversion** (Megatron/HF → PEFT):
```python
# HF-style input (from verl bridge)
"model.layers.0.self_attn.q_proj.lora_a.weight"
"model.layers.0.mlp.experts.0.gate_proj.lora_a.weight"

# PEFT output
"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
"base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight"
```

**CONCLUSION: vLLM 0.13.0rc2 fully supports Qwen3-30B-A3B LoRA (both attention and expert layers).** The previous failure was user error (wrong target modules), not a vLLM bug.

**Test Results on Qwen1.5-MoE (vLLM 0.13.0rc2, single A800 80GB):**

| Test | Result |
|------|--------|
| Attention LoRA (q_proj, k_proj, v_proj) | **PASS** |
| Expert LoRA (all 60 experts × 3 layers) | **PASS** |

**Key Requirements for MoE Expert LoRA:**

1. **Attention projections must be separate** (not fused):
   - Use `q_proj`, `k_proj`, `v_proj` instead of fused `qkv_proj`
   - vLLM module whitelist doesn't include `qkv_proj`

2. **ALL experts must have LoRA weights** (no partial):
   - `pack_moe()` in `lora_weights.py:168` asserts `len(loras) % 3 == 0`
   - Each expert needs: gate_proj (w1), down_proj (w2), up_proj (w3)
   - TODO comment at line 175: "Consider the case where some experts don't have LoRA added"

3. **LoRA tensor naming format:**
   ```
   base_model.model.model.layers.{layer}.mlp.experts.{expert}.{gate|up|down}_proj.lora_{A|B}.weight
   ```

4. **Tensor shapes:**
   - gate_proj/up_proj (w1/w3): lora_A=[rank, hidden_size], lora_B=[moe_intermediate_size, rank]
   - down_proj (w2): lora_A=[rank, moe_intermediate_size], lora_B=[hidden_size, rank]

**Test Script:** `scripts/test_qwen15_moe_expert_lora.py`

**Implications for Qwen3-30B-A3B:**
- The MoE LoRA infrastructure works - only Qwen3MoeForCausalLM has the bug
- When GitHub #3377 is fixed, the same adapter format should work
- For now, can test Megatron → vLLM weight export using Qwen1.5-MoE as a proxy

#### Megatron → vLLM Weight Export Testing (2025-12-18)

Validated the full Megatron → vLLM LoRA export pipeline using Qwen1.5-MoE as a proxy for Qwen3-30B-A3B.

**Test Results:**

| Test | Status |
|------|--------|
| Name conversion (Megatron → PEFT) | **PASS** (5/5 patterns) |
| Simulated Megatron export → vLLM load | **PASS** |

**Conversion Patterns Tested:**
```
Megatron (input)                                              → PEFT (output)
decoder.layers.0.self_attention.linear_qkv.adapter.linear_in  → layers.0.self_attn.q_proj.lora_A
decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.adapter.linear_in → layers.0.mlp.experts.0.gate_proj.lora_A
decoder.layers.0.mlp.experts.local_experts.59.linear_fc2.adapter.linear_out → layers.0.mlp.experts.59.down_proj.lora_B
```

**Tensor Counts:**
- Attention LoRA: 24 layers × 3 modules (q/k/v) × 2 (A+B) = 144 tensors
- Expert LoRA: 24 layers × 60 experts × 3 modules (gate/up/down) × 2 (A+B) = 8,640 tensors
- Total: 8,784 tensors successfully converted and loaded

**Key Finding:**
vLLM's `FusedMoEWithLoRA` class correctly handles PEFT-format expert LoRA weights. The format conversion is:
- `lora_a`/`lora_b` → `lora_A`/`lora_B` (case normalization)
- `model.layers.N...` → `base_model.model.model.layers.N...` (PEFT prefix)

**Test Script:** `scripts/test_megatron_to_vllm_export.py`

#### Technical Analysis

**Location**: `/root/tinker_project/vllm/vllm/lora/`

**Blocker 1: Expert Parallelism Assertion** (`layers/fused_moe.py:48`)
```python
assert not self.base_layer.use_ep, (
    "EP support for Fused MoE LoRA is not implemented yet."
)
```
- Blocks when Expert Parallelism (EP) is enabled
- **Workaround**: Use TP=4 only (no EP) - currently implemented

**Blocker 2: Module Validation** (`models.py:188-213`)
```python
def check_unexpected_modules(modules: dict):
    for lora_module in modules.keys():
        # ...
        if ".experts" in module_name:
            expert_suffix = module_name[expert_idx + 1:]
            if expert_suffix not in expected_lora_modules:
                unexpected_modules.append(module_name)
```
- Rejects expert LoRA weights because suffix format doesn't match
- Expert weights have paths like `model.layers.0.mlp.experts.0.gate_proj`
- Suffix `experts.0.gate_proj` not in `expected_lora_modules` (which has `q_proj`, `k_proj`, etc.)

**Blocker 3: LoRA Weight Loading Format**
- `FusedMoEWithLoRA.set_lora()` expects 3 weight pairs: `(w1_lora_a, w2_lora_a, w3_lora_a), (w1_lora_b, w2_lora_b, w3_lora_b)`
- PEFT exports per-expert weights with different naming convention
- Need adapter to convert PEFT format to vLLM's expected tensor layout

#### vLLM 0.12.0 MoE LoRA Status

| Feature | Status | Notes |
|---------|--------|-------|
| FusedMoEWithLoRA class | EXISTS | Full implementation present |
| Module validation | BLOCKS | Rejects expert weight names |
| EP assertion | BLOCKS | Disabled for EP=0 |
| LoRA weight format | INCOMPATIBLE | PEFT → vLLM conversion needed |
| Attention-only LoRA | WORKS | Current workaround |

#### Patching Approach

**Option A: Minimal Patch (Recommended)**

1. **Patch `models.py` module validation** (~20 lines)
   - Modify `check_unexpected_modules` to recognize expert LoRA patterns
   - Allow `experts.{N}.gate_proj`, `experts.{N}.up_proj`, `experts.{N}.down_proj`

2. **Add weight format adapter** (~50 lines)
   - Convert PEFT expert LoRA format to vLLM's `(w1, w2, w3)` tensor layout
   - Handle per-expert indexing

**Option B: Full Patch**

In addition to Option A:
3. **Remove EP assertion** for TP-only mode
4. **Add expert weight routing** for multi-tenant scenarios

#### Implementation Plan

| Phase | Task | Effort | Risk |
|-------|------|--------|------|
| 1 | Fork vLLM 0.12.0, create `mint-vllm` branch | Low | Low |
| 2 | Patch module validation in `models.py` | Medium | Low |
| 3 | Add PEFT → vLLM weight format adapter | Medium | Medium |
| 4 | Test with Qwen3-30B-A3B expert LoRA | High | Medium |
| 5 | Upstream PR or maintain fork | - | - |

#### Current Workaround

Filter out MLP modules in `megatron_distributed.py:get_lora_state_dict()`:
- Training: Full MLP + attention LoRA via Megatron
- Inference: Attention-only LoRA via vLLM
- Loss: ~10-20% effectiveness reduction for MoE inference (based on Tinker docs)

**UPDATE (2025-12-17):** vLLM 0.13.0rc2 LoRA loading is completely broken for Qwen3 MoE models. The workaround above only applies to vLLM < 0.13.0. For 0.13.0rc2, **no LoRA inference is possible** until the upstream bug is fixed.

#### Options for Proceeding

| Option | Effort | Risk | Notes |
|--------|--------|------|-------|
| 1. Wait for upstream fix | None | Low | Monitor [vllm-ascend #3377](https://github.com/vllm-project/vllm-ascend/issues/3377) |
| 2. Downgrade to vLLM 0.12.x | Low | Medium | May lose other 0.13 features; need to verify 0.12.x works |
| 3. Debug V1 engine locally | High | High | Fix `set_active_loras` failure; complex multi-GPU debugging |
| 4. Use non-LoRA inference | Low | High | Train with LoRA, merge weights for inference (loses LoRA flexibility) |

**Recommended:** Option 1 (wait) with Option 4 (merge weights) as fallback. The bug is tracked upstream and affects many users - expect fix soon.

#### References

- [vLLM Forum: MoE LoRA on expert layers](https://discuss.vllm.ai/t/do-the-current-moe-models-support-setting-lora-adapters-on-expert-layers/1726)
- [vLLM Issue #18120: Qwen 3 MoE LoRA](https://github.com/vllm-project/vllm/issues/18120)
- [vLLM PR #20932: Adds warning but no fix](https://github.com/vllm-project/vllm/pull/20932)
- [TRL Issue #4584: vLLM upgrade for FusedMoE LoRA](https://github.com/huggingface/trl/issues/4584)

---

## Architecture Reference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Mint API Server                               │
│                     (Tinker-Compatible REST API)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ResourceManager                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Global GPU tracking, cross-pool LRU eviction                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
├──────────────────────────────┬──────────────────────────────────────────┤
│      Inference Pool          │           Training Pool                  │
│  ┌────────────────────────┐  │  ┌────────────────────────────────────┐  │
│  │  vLLM Engine           │  │  │  DenseTrainerPool                  │  │
│  │  - Multi-LoRA serving  │  │  │    (PEFT, <14B models)             │  │
│  │  - Hot-swap adapters   │  │  │                                    │  │
│  └────────────────────────┘  │  │  MegatronActorPool                 │  │
│             ▲                │  │    (verl Megatron, MoE models)     │  │
│             │                │  └────────────────────────────────────┘  │
│             │    LoRA Transfer (Ray ObjectRef, ~0.7s)                   │
│             └───────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Future Research

### Multi-LoRA Batched Training

**Problem**: Current architecture time-shares a single Megatron engine between different LoRA clients. Each `forward_backward` call trains one LoRA at a time. With N concurrent users, total time = N × single-user time.

**Opportunity**: Batch multiple LoRAs in a single forward-backward pass. When GPUs are undersaturated (small per-user batches), this could yield 2-3× speedup.

**How it works (conceptually)**:

```python
# Current: sequential
for lora in [A, B, C]:
    forward_backward(batch[lora], lora)  # 3 passes

# Batched: single pass with per-sequence adapter routing
batched_forward_backward(
    combined_batch,
    adapters=[A, B, C],
    adapter_indices=[0,0,1,1,2,2]  # which sequence uses which LoRA
)
```

**Current landscape**:

| System | Multi-LoRA Training | Production-ready |
|--------|---------------------|------------------|
| [mLoRA](https://github.com/TUDB-Labs/mLoRA) | Yes (BatchLoRA kernels) | No - research code |
| verl / Megatron | No | Yes |
| vLLM (inference only) | Yes (BGMV kernels) | Yes |

**Two approaches**:

| Approach | Description | Speedup | Complexity |
|----------|-------------|---------|------------|
| mLoRA-style (PyTorch) | Batch base model ops, separate LoRA ops | ~80-90% of optimal | Medium |
| Full BGMV/SGMV | Custom CUDA kernels for batched LoRA | ~100% of optimal | Very High |

mLoRA batches the expensive base model computation (`X @ W`) but runs N separate LoRA ops (`x @ A @ B`). Since LoRA params are <1% of base model, this captures most of the benefit without custom CUDA.

**Kernel backward pass status**:

| Kernel | Forward | Backward (training) |
|--------|---------|---------------------|
| Punica BGMV/SGMV | ✓ | ✗ (not implemented) |
| vLLM BGMV | ✓ | ✗ (inference only) |
| mLoRA BatchLoRA | ✓ (PyTorch) | ✓ (autograd) |

No public BGMV/SGMV backward implementations exist. All are inference-only.

**Recommended path** (mLoRA-style):

1. Modify `forward_backward_batch` to accept multiple adapter contexts
2. Stack inputs from multiple sessions before forward pass
3. Replace single-adapter LoRA modules with multi-adapter versions
4. Route gradients to correct adapter's optimizer
5. Manage per-adapter optimizer states

No custom CUDA kernels needed. Main challenge is integration with Megatron's tensor parallelism.

**When to prioritize**: Only worthwhile with 3+ concurrent users with small batches. Single-user or large-batch scenarios see minimal benefit.

**References**:
- [mLoRA Paper (VLDB 2024)](https://arxiv.org/abs/2312.02515) - BatchLoRA operator design (PyTorch-level)
- [Punica Paper](https://arxiv.org/abs/2310.18547) - SGMV kernels for multi-LoRA inference (no backward)
- [S-LoRA Blog](https://lmsys.org/blog/2023-11-15-slora/) - Serving thousands of adapters
- [Punica BGMV source](https://github.com/punica-ai/punica/blob/master/csrc/bgmv/bgmv_impl.cuh) - Forward-only CUDA kernel

---

## References

- [verl Megatron Backend](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html)
- [verl Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
- [Kimi-K2-Instruct on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- [Tinker SDK test notebook](~/Downloads/tinker_test.ipynb)
- [Tinker Official Agent Reference](./tinker_official_reference.txt) - Full API docs and type definitions
