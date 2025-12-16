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

### 6. Stateless Trainer Architecture

#### Problem Statement

**Current behavior**: Trainers hold session state (weights + optimizer) in memory. Multiple sessions cannot safely share a trainer, and actor eviction loses unsaved state.

```
Session A: create → [weights in memory] → train → train → [END]
Session B: create → [INHERITS A's weights - BUG!] → train → ...
```

**Desired behavior**: Trainers are stateless compute resources. Session state lives in persistent storage and is loaded/saved per operation.

```
Session A: create → load(A) → train → save(A) → load(A) → train → save(A) → [END]
Session B: create → load(B) → train → save(B) → ...
           (can interleave with A on same trainer)
```

#### Architecture

**Session State Storage**:
```
/tmp/mint_sessions/
├── session_A_checkpoint/
│   ├── adapter_model.safetensors  # LoRA weights
│   ├── optimizer.pt               # Adam state (exp_avg, exp_avg_sq)
│   └── training_meta.json         # step count, learning_rate
├── session_B_checkpoint/
│   └── ...
```

**Per-Iteration Flow**:
```
forward_backward(session_id, data):
    1. Load session state: load_session_state(session_id)
       - LoRA weights → model
       - Optimizer state → optimizer
       - Learning rate → optimizer.param_groups
    2. Forward pass: model(data)
    3. Backward pass: loss.backward()
    4. Return logprobs (gradients accumulated, not applied)

optim_step(session_id):
    1. Verify session state loaded (or load if not)
    2. Clip gradients
    3. optimizer.step()
    4. optimizer.zero_grad()
    5. Save session state: save_session_state(session_id)
       - LoRA weights → safetensors
       - Optimizer state → optimizer.pt
       - Increment step count
```

**Key Invariant**: After optim_step returns, session state is persisted. Trainer can be reused for any session.

#### Implementation Plan

**Phase 1: Session State Manager**

New class `SessionStateManager` handles checkpoint I/O:

```python
class SessionStateManager:
    def __init__(self, base_path: str = "/tmp/mint_sessions"):
        self.base_path = base_path

    def get_session_path(self, session_id: str) -> str:
        return os.path.join(self.base_path, f"{session_id}_checkpoint")

    def save_state(self, session_id: str, model, optimizer, step: int, lr: float):
        """Save LoRA weights + optimizer state + metadata."""
        ...

    def load_state(self, session_id: str, model, optimizer) -> dict:
        """Load state into model/optimizer. Returns metadata."""
        ...

    def session_exists(self, session_id: str) -> bool:
        """Check if session has saved state."""
        ...

    def delete_session(self, session_id: str):
        """Clean up session storage."""
        ...
```

**Phase 2: Modify TrainingWorker**

```python
class TrainingWorker:
    def __init__(self, ...):
        self.state_manager = SessionStateManager()
        self._current_session_id = None  # Track which session is loaded

    def _ensure_session_loaded(self, session_id: str):
        """Load session state if not already loaded."""
        if self._current_session_id != session_id:
            if self.state_manager.session_exists(session_id):
                self.state_manager.load_state(session_id, self.model, self.optimizer)
            else:
                # New session: reinitialize weights
                self.reinit_lora_weights()
            self._current_session_id = session_id

    def forward_backward(self, session_id: str, data_items, ...):
        self._ensure_session_loaded(session_id)
        # ... existing forward/backward logic ...

    def optim_step(self, session_id: str, learning_rate: float):
        self._ensure_session_loaded(session_id)
        # ... existing optim logic ...
        # Save state after update
        self.state_manager.save_state(
            session_id, self.model, self.optimizer,
            self._step_count, learning_rate
        )
```

**Phase 3: API Changes**

Add `session_id` parameter to all training operations (may already exist as `model_id`):

| Endpoint | Current | New |
|----------|---------|-----|
| `/forward_backward` | `model_id` routes to session | Same, state loaded per-call |
| `/optim_step` | `model_id` | Same, state saved after |
| `/train_step` | `model_id` | Same, load before + save after |

**Phase 4: Megatron Backend**

Apply same pattern to `MegatronRankWorker`:
- Distributed checkpoint save/load (all ranks coordinate)
- Use existing `save_adapter_state` / `load_adapter_state` methods
- Add per-iteration save after `optim_step`

#### Performance Considerations

| Operation | Estimated Time | Mitigation |
|-----------|---------------|------------|
| Load LoRA weights (32-rank) | 50-100ms | SSD storage, memory-mapped files |
| Load optimizer state | 100-200ms | Lazy load (skip if same session) |
| Save LoRA weights | 50-100ms | Async write (return before fsync) |
| Save optimizer state | 100-200ms | Async write |

**Optimization**: Track `_current_session_id` to skip load if same session as previous call. Most training loops call forward_backward → optim_step → forward_backward on same session, so load happens once at start.

**Total overhead per iteration**: ~0ms (same session) to ~300ms (session switch)

#### Testing Strategy

1. **Unit test**: SessionStateManager save/load roundtrip
2. **Integration test**: Two sessions interleaved on same trainer
   - Session A: 3 iterations, final loss L_A
   - Session B: 3 iterations, final loss L_B
   - Session A: 3 more iterations, should continue from L_A
3. **Stress test**: 10 sessions round-robin on single trainer
4. **Recovery test**: Kill trainer mid-training, recreate, verify session continues correctly

#### Migration Path

1. Implement SessionStateManager (no behavior change yet)
2. Add `_ensure_session_loaded` to TrainingWorker with feature flag
3. Enable by default, monitor for regressions
4. Apply to MegatronRankWorker
5. Remove in-memory-only code paths

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
