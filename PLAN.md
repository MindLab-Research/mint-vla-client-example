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

#### T2 (Week 4+) - Multimodal

| Model | Type | Architecture | Train GPUs | Infer GPUs | Notes |
|-------|------|--------------|------------|------------|-------|
| Qwen/Qwen3-VL-30B-A3B-Instruct | Vision | MoE | TBD | TBD | Vision encoder integration |
| Qwen/Qwen3-VL-235B-A22B-Instruct | Vision | MoE | TBD | TBD | |

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

## References

- [verl Megatron Backend](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html)
- [verl Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
- [Kimi-K2-Instruct on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
