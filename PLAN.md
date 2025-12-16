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
| LRU-based actor eviction | Implemented |
| LoRA hot-swap to vLLM | Verified (0.16s extraction + 2.2s inference) |

---

## Roadmap

### 1. Model Lineup

#### T0 (Week 1-2) - Foundation

| Model | Type | Architecture | Train GPUs | Infer GPUs | Notes |
|-------|------|--------------|------------|------------|-------|
| Qwen/Qwen3-0.6B | Hybrid | Dense | 1 | 1 | |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | Instruction | MoE | 8 (TP4,EP2) | 4 (TP4) | |
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

### 2. Resource Orchestration

#### Current State

- Per-backend pools: `MegatronActorPool`, `DenseTrainerPool`
- LRU eviction within each pool
- No cross-pool awareness

#### Target State

Unified `ResourceManager` with global GPU tracking and cross-pool eviction.

```
ResourceManager
├── total_gpus: int (cluster capacity)
├── allocated: dict[actor_id, num_gpus]
├── pools: [MegatronActorPool, DenseTrainerPool, InferencePool]
└── allocate(model, num_gpus) → evicts across ALL pools if needed
```

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
for name in ['persistent_megatron_worker_group', 'persistent_vllm_actor']:
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

### 6. Tinker API Alignment

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
| Per-token weights | `weights` | `loss_mask` | **NO** - field name mismatch |
| Model input | `chunks` with `EncodedTextChunk` | `chunks` format | Yes |
| Tensor format | `TensorData{data, shape, dtype}` | `{data, shape, dtype}` | Yes |
| Loss functions | `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro` | ? | Verify |

**Action required**: Rename `loss_mask` → `weights` for Tinker API compatibility.

#### Implementation Tasks

| Task | Priority | Status |
|------|----------|--------|
| Rename `loss_mask` → `weights` in forward_backward | **Critical** | Pending |
| Verify all loss functions: `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro` | High | Pending |
| Add `/api/v1/save_state` | High | Pending |
| Add `/api/v1/load_state` | High | Pending |
| Add `forward_backward_custom` for arbitrary loss functions | Medium | Pending |
| Verify LoRA config options: `train_unembed`, `train_mlp`, `train_attn` | Medium | Pending |
| Verify logprobs format matches `TensorData` spec | High | Pending |
| Create `test_tinker_api_alignment.py` | High | Pending |
| Verify `include_prompt_logprobs` and `topk_prompt_logprobs` | Medium | Pending |

---

## Known Issues

### train_step API Breaks Tinker Compatibility

**Problem**: With MoE + `param_offload=True`, gradients are zeroed between `forward_backward` and `optim_step` calls.

**Current workaround**: `/api/v1/train_step` combines both in single context.

**Mitigation**: For LoRA, `grad_offload=False` keeps gradients in GPU memory.

**TODO**: Verify that LoRA + `grad_offload=False` allows standard Tinker API.

### Time-Sliced Isolation Not Yet Verified

Previous tests prove sessions have different weights (A ≠ B), but don't prove training is correct (A_interleaved == A_exclusive).

**TODO**: Implement `test_time_sliced_isolation.py` to verify stronger correctness condition.

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
