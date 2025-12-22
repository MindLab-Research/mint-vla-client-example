# MinT Server Roadmap

## Overview

**MinT** is a multi-tenant training/inference server compatible with Tinker APIs.

- Separated inference and training on different GPU pools
- LoRA-based training with sub-second adapter transfer
- Dense models via PEFT, MoE models via verl Megatron backend
- Multi-session actor sharing with LRU eviction

---

## Current State (Completed)

| Feature | Status |
|---------|--------|
| Dense model training (Qwen2.5-7B) | Verified |
| Dense model RL (importance_sampling) | Verified |
| MoE model training (Qwen3-30B-A3B) | Verified (94.9% loss reduction) |
| MoE model RL (importance_sampling) | Verified (87.5% accuracy) |
| Moonlight (DeepseekV3 MLA) training | Verified (93.8% loss reduction) |
| Moonlight LoRA transfer | Verified (102s transfer time) |
| Multi-session actor sharing | Verified |
| Unified rank support (max-rank padding) | Verified |
| LRU-based actor eviction | Verified (unified ResourcePool with cross-actor LRU) |
| LoRA hot-swap to vLLM | Verified (0.16s extraction + 2.2s inference) |
| Stress tests (concurrent, eviction, rapid) | Verified (4/4 pass) |

---

## Roadmap

### 1. Model Lineup

#### T0 - Foundation (Near Complete)

| Model | Type | Train GPUs | Infer GPUs | Status |
|-------|------|------------|------------|--------|
| Qwen/Qwen3-0.6B | Dense | 1 | 1 | **Verified** (90.8% loss reduction) |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | MoE | 8 (TP4,EP2) | 4 (TP4) | **Verified** (SFT, RL, LoRA transfer) |
| moonshotai/Moonlight-16B-A3B-Instruct | MoE (MLA) | 8 (TP2,EP4) | 4 (TP4) | **Verified** (SFT, RL) |
| moonshotai/Kimi-K2-Thinking | MoE (MLA) | 64+ (H100+) | 32+ | **Pending** - MLA solved, needs cluster testing |

**Remaining**: K2 thinking model requires 64+ GPU cluster testing.

#### T1 - Scale Up

| Model | Type | Train GPUs | Infer GPUs | Status |
|-------|------|------------|------------|--------|
| Qwen/Qwen3-4B-Instruct-2507 | Dense | 1 | 1 | Ready (dense models verified) |
| Qwen/Qwen3-8B | Dense | 2 | 1 | Ready (dense models verified) |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | MoE | 32 (TP8,EP4) | 16 | Pending (after K2) |
| deepseek-ai/DeepSeek-V3.1 | MoE | 64+ | 32+ | Pending (after K2) |

**Note**: Dense models (4B, 8B) should work without issues given verified dense support. The 235B and DeepSeek-V3.1 testing deferred until K2 validation complete.

#### T2 - Multimodal & Embodied (Pending)

| Model | Type | Architecture | Notes |
|-------|------|--------------|-------|
| Qwen/Qwen3-VL-30B-A3B-Instruct | Vision | MoE | Vision encoder integration |
| Qwen/Qwen3-VL-235B-A22B-Instruct | Vision | MoE | Multi-node |
| physical-intelligence/pi0 | VLA | PaliGemma+Expert | Flow matching, 50Hz actions |
| physical-intelligence/pi0.5 | VLA | PaliGemma+Expert | Open-world generalization |

---

### 2. Resource Orchestration

**Status**: Framework implemented. Critical limitation on MoE multi-tenancy.

#### Critical: MoE max_loras=1 Defeats Multi-Tenancy

**This undermines the core value proposition of the project.**

Currently `multi_lora_engine.py:897` hardcodes `max_loras=1` for MoE models:

```python
model_max_loras = 1 if config.is_moe else self.max_loras
```

**Why it exists**: vLLM pre-allocates LoRA buffers for all experts. With 128 experts and default `max_loras=64`, memory explodes: `max_loras × num_experts × lora_rank × hidden_size` per layer.

**Impact**: Only ONE user can run inference on MoE models at a time. Other users must wait for the single LoRA slot. This reduces the multi-tenant server to single-tenant for MoE workloads.

**Required investigation**:
- Measure actual memory usage per LoRA slot for Qwen3-30B-A3B (128 experts)
- Determine maximum `max_loras` that fits in 4×A800 80GB (inference config)
- Test concurrent inference with 2, 4, 8 LoRA adapters

#### Implemented

- Unified `ResourcePool` singleton with global GPU tracking (`resource_pool.py`)
- LRU-based cross-actor eviction with MIN_ACTOR_AGE protection
- Per-model TP/EP/DP configuration via `model_registry.py`
- Actor registration and cleanup on server startup

#### Remaining Tasks

| Task | Priority | Description |
|------|----------|-------------|
| **MoE max_loras** | **Critical** | **Increase from 1. Current setting defeats multi-tenancy.** |
| Parallelism validation | High | Determine correct TP, EP, DP for each model through intensive testing. Current values may be arbitrary. |
| Max LoRA rank | High | Per-model maximum LoRA rank that fits in memory. |
| Orphaned CUDA memory | Medium | GPU memory not freed when actors crash inside Docker containers. Ray reports GPUs available, but OOM on actor start. Investigate container-level GPU cleanup and volcano platform specifics. |

---

### 3. Procedural Standardization & Automation

#### Completed

| Skill | Purpose | Location |
|-------|---------|----------|
| merge-gate | Pre-merge testing (20/20 tests) | `.claude/skills/merge-gate/` |
| mint-dev | Dev environment operations | `.claude/skills/mint-dev/` |
| mint-prod | Prod environment operations | `.claude/skills/mint-prod/` |
| volcano-cluster | Ray cluster lifecycle | `.claude/skills/volcano-cluster/` |

#### Automated Skills (Pending)

| Skill | Purpose | Trigger |
|-------|---------|---------|
| **CI/CD Deployment** | Review changes on main branch, deploy new version automatically. | Push to `main` |
| **Server Monitor** | Continuously watch server logs in background, detect anomalies, alert on issues. | Background daemon |
| **Client Test Agent** | Run examples from `../mint-quickstart` against production server. Create GitHub issues for confirmed bugs. **Read-only access to server logs.** Cannot kill server or modify implementation. | Periodic or on-demand |

#### Internal Documentation (Pending)

Architectural design documents for internal reference:

| Document | Content |
|----------|---------|
| Training backend architecture | Megatron vs PEFT selection, param/grad offloading, session state management |
| Inference backend architecture | vLLM actor lifecycle, multi-LoRA hot-swap, TP/EP constraints |
| Resource orchestration | ResourcePool design, LRU eviction policy, actor registration flow |
| LoRA weight transfer | Megatron → PEFT → vLLM conversion pipeline, naming conventions |
| API compatibility layer | Tinker API mapping, `train_step` vs separate calls, data format translation |

---

### 4. Client-Side Utilities

**Status**: Requires testing and naming standardization.

#### Repositories

| Path | Purpose |
|------|---------|
| `mint-doc/` | NextJS documentation site (API reference, guides) |
| `../mint-quickstart` | Quickstart notebooks and examples |
| `../mindlab-toolkit` | Client-side utilities |

#### Tasks

| Task | Priority |
|------|----------|
| Test all examples in `mint-quickstart` | High |
| Search for "tinker" occurrences across client repos | High |
| Standardize to "MinT" in all client-facing content | High |
| Acknowledge Tinker compatibility explicitly | Medium |

#### Application Layer Support

**Current deployment**: GPU cluster → Azure reverse proxy → clients. No dedicated application server.

Missing application server responsibilities (out of scope for this project):
- Multi-tenant authentication
- Usage metering and billing
- Load balancing across GPU clusters

**Required support from this project**:

| Task | Priority | Description |
|------|----------|-------------|
| Interface separation | High | Distinguish client-facing endpoints (training, inference) from internal endpoints (kill actor, resource pool status, health checks). Internal endpoints should not be exposed through Azure gateway. |
| Token metering | High | Measure token usage per request (input tokens, output tokens, training tokens). Define interface to return usage metrics to application server for billing. |

**Naming Policy**: Official name is "MinT" for all client-facing content. Tinker API compatibility is a feature to be documented, not hidden.

---

### 5. Cleanup & Refactoring (Pending)

Heavy logic accumulation requires cleanup.

| Category | Count | Action |
|----------|-------|--------|
| Directory | 1 | `tinker_server/` → `mint/` |
| Imports | ~50 files | `from tinker_server` → `from mint` |
| Config vars | ~10 | `TINKER_MODEL_PATH` → `MINT_MODEL_PATH` |
| Test scripts | 39 | Consolidate overlapping scripts |
| Common utilities | - | Extract duplicated `poll_future()`, `create_session()` to `tests/utils.py` |

Keep `TINKER_BASE_URL`, `TINKER_API_KEY` for client-side Tinker API compatibility.

---

### 6. Future Research

#### Multi-LoRA Batched Training

Current architecture time-shares Megatron between LoRA clients. With N concurrent users, total time = N × single-user time.

**Opportunity**: Batch multiple LoRAs in single forward-backward pass when GPUs undersaturated.

| System | Multi-LoRA Training | Production-ready |
|--------|---------------------|------------------|
| mLoRA | Yes (BatchLoRA kernels) | No - research code |
| verl / Megatron | No | Yes |
| vLLM (inference only) | Yes (BGMV kernels) | Yes |

**Recommended path**: mLoRA-style batching (PyTorch-level, no custom CUDA). Main challenge is Megatron tensor parallelism integration.

**Priority**: Low - only worthwhile with 3+ concurrent users with small batches.

---

## Critical Issue: MoE Training Breaks Tinker API Compatibility

**This is a fundamental incompatibility with no workaround.**

### Problem

MoE models require `param_offload=True` to fit in GPU memory. The verl/Megatron `train_mode()` context manager zeros gradients on entry. Standard Tinker API calls `forward_backward()` then `optim_step()` as separate requests - each enters `train_mode()` independently. Result: gradients computed in `forward_backward()` are zeroed before `optim_step()` runs.

```
Tinker API (BROKEN for MoE):
  forward_backward() → enter train_mode → compute grads → exit train_mode → grads offloaded
  optim_step()       → enter train_mode → GRADS ZEROED → optimizer sees zeros → no learning
```

### Why No Workaround Exists

| Attempted Fix | Why It Fails |
|---------------|--------------|
| `grad_offload=False` | Only prevents offloading. `train_mode()` still zeros on entry. |
| Keep `train_mode()` open | HTTP request/response model requires context exit between calls. |
| Disable `param_offload` | OOM - MoE models don't fit without offloading. |

### Current State

MinT provides `/api/v1/train_step` endpoint that combines both operations in single `train_mode()` context. All MoE training uses this endpoint.

**Impact on Tinker compatibility**:
- Dense models: Full Tinker API compatibility (no `param_offload` needed)
- MoE models: Must use `train_step` instead of separate `forward_backward` + `optim_step`

Existing Tinker client code for MoE models will silently fail to learn (zero gradients, no error).

### Resolution Options

| Option | Effort | Notes |
|--------|--------|-------|
| Upstream fix to verl | High | Modify `train_mode()` to not zero gradients unconditionally |
| Document as limitation | Low | MoE requires `train_step`, not standard API |
| Wrapper client | Medium | Client-side adapter that batches calls into `train_step` |

---

## Other Known Issues

### vLLM MoE Expert LoRA

**Status**: Resolved for Qwen3-30B-A3B.

Key requirements:
- Use separate attention projections (`q_proj`, `k_proj`, `v_proj`) not fused `qkv_proj`
- Expert LoRA requires ALL 128 experts to have weights
- PEFT naming format: `base_model.model.model.layers.{layer}.mlp.experts.{expert}.{gate|up|down}_proj.lora_{A|B}.weight`

---

## Architecture Reference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MinT API Server                               │
│                     (Tinker-Compatible REST API)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  ResourcePool                                                           │
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
- [Tinker Official Agent Reference](./tinker_official_reference.txt)
