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
| moonshotai/Kimi-K2-Thinking | Reasoning | MoE | 64+ | 32+ | Block-FP8, infra team has working impl |

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
| Kimi-K2 | Block-FP8 quantization, 1T params | Infra team has working impl, migrate to Mint |
| DeepSeek-V3.1 | Different MoE routing | Architecture analysis needed |
| VL models | Vision encoder, multimodal inputs | New modality support |
| pi0/pi0.5 (VLA) | See VLA investigation below | Tentative - may require new backend |

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

### vLLM MoE Expert LoRA Inference - INCOMPLETE (2025-12-17)

**Problem**: vLLM 0.12.0 has `FusedMoEWithLoRA` class but cannot load MoE expert LoRA weights for inference. Training with full MLP+attention LoRA works via Megatron, but inference is limited to attention-only LoRA.

**Impact**: MoE models (Qwen3-30B-A3B) train with full LoRA (attention + expert MLP) but inference only uses attention LoRA. This reduces LoRA effectiveness during inference while training quality remains unaffected.

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

## References

- [verl Megatron Backend](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html)
- [verl Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
- [Kimi-K2-Instruct on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
- [Tinker SDK test notebook](~/Downloads/tinker_test.ipynb)
- [Tinker Official Agent Reference](./tinker_official_reference.txt) - Full API docs and type definitions
