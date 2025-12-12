# Mint Server: Qwen3 MoE Support Plan

## Project Overview

**Mint** is a multi-tenant training/inference server compatible with Tinker APIs. It supports:
- Separated inference and training on different GPU pools
- LoRA-based training with quick adapter transfer between trainer and inferencer
- Dense models (<14B) via existing single-GPU TrainingWorker
- MoE models (30B+) via verl Megatron backend

---

## Current Status: Dense Models (Complete)

All Tinker API endpoints implemented and verified with Qwen2.5-7B-Instruct:

| Paradigm | Loss Function | Status |
|----------|---------------|--------|
| SFT | `cross_entropy` | Verified |
| Policy Gradient RL | `importance_sampling` | Verified |
| PPO | `ppo` | Verified |
| Custom Losses (DPO) | via `weights` | Verified |

Architecture: Single-GPU Ray actors with PEFT LoRA.

---

## Phase 1: MoE Inference Support (Complete)

**Goal:** Serve Qwen3 MoE models via vLLM with tensor parallelism.

| Task | Status |
|------|--------|
| Add `tensor_parallel_size` config | Done |
| Update vLLM server args | Done |
| Test Qwen3-30B-A3B inference | Done |

---

## Phase 2: MoE Training via verl Megatron

**Decision:** Use verl's Megatron backend instead of custom FSDP.

### Rationale

| Approach | Pros | Cons |
|----------|------|------|
| Custom FSDP | Full control | High maintenance, doesn't scale well for MoE |
| verl Megatron | Battle-tested LoRA+MoE, Expert Parallelism, maintained | Dependency on verl |

**Key factors:**
- verl has working Qwen3-30B-A3B LoRA training (`run_qwen3moe-30b_megatron_lora.sh`)
- Megatron Expert Parallelism (EP) is superior for MoE vs FSDP's all-gather
- verl handles checkpoint conversion, offloading, and multi-node automatically

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Mint API Server                               │
│                     (Tinker-Compatible REST API)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Session Manager                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ tenant_id → { base_model, lora_ref, inference_session_id }      │    │
│  │ LoRA Registry: { adapter_id → Ray ObjectRef }                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
├──────────────────────────────┬──────────────────────────────────────────┤
│      Inference Pool          │           Training Pool                  │
│                              │                                          │
│  ┌────────────────────────┐  │  ┌────────────────────────────────────┐  │
│  │  vLLM Engine (TP=4)    │  │  │  Dense (<14B): TrainingWorker      │  │
│  │  - MoE via TP          │  │  │    (existing single-GPU path)      │  │
│  │  - Multi-LoRA serving  │  │  │                                    │  │
│  │  - Hot-swap adapters   │  │  │  MoE (30B+): VerlMegatronAdapter   │  │
│  │                        │  │  │    - verl Megatron workers         │  │
│  │  GPU 0-3               │  │  │    - EP + TP + LoRA                │  │
│  └────────────────────────┘  │  │    GPU 4-7                         │  │
│             ▲                │  └────────────────────────────────────┘  │
│             │                │                  │                       │
│             │    LoRA Transfer (Ray ObjectRef)  │                       │
│             └───────────────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Status |
|------|------|--------|
| Create `MegatronTrainingWorker` class | `megatron_training.py` | Done |
| Implement Tinker → TensorDict data conversion | `megatron_training.py` | Done |
| Implement SFT + PPO loss functions | `megatron_training.py` | Done |
| Implement LoRA extraction via bridge | `megatron_training.py` | Done |
| Route MoE models to Megatron worker | `verl_training.py` | Done |
| Test with Qwen3-30B-A3B | - | Done |

*Full MoE training flow verified: create_model (213s) → forward_backward (8.5s) → optim_step (4.3s) with 8 GPUs (TP=4, EP=2).

### Implementation: VerlMegatronAdapter

```python
# tinker_server/backend/verl_megatron_adapter.py

"""verl Megatron adapter for MoE training.

Wraps verl's Megatron backend to provide Tinker-compatible API.
"""

from dataclasses import dataclass
from typing import Any
import ray
from omegaconf import OmegaConf


@dataclass
class VerlMegatronConfig:
    """Configuration translated from Tinker API to verl format."""
    model_path: str
    lora_rank: int = 16
    lora_alpha: int = 32
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 2
    expert_parallel_size: int = 4
    context_parallel_size: int = 2
    param_offload: bool = True
    optimizer_offload: bool = True
    grad_offload: bool = True


class VerlMegatronAdapter:
    """Adapter between Tinker API and verl Megatron trainer.

    This class wraps verl's Megatron workers to provide:
    - create_session(): Initialize verl workers with Megatron config
    - forward_backward(): Compute loss and gradients
    - optim_step(): Update LoRA weights
    - get_lora_checkpoint(): Extract LoRA state dict for inference
    """

    def __init__(self, model_id: str, base_model: str, lora_config: dict):
        self.model_id = model_id
        self.base_model = base_model
        self.lora_config = lora_config
        self.verl_config = self._build_verl_config()
        self.worker_group = None

    def _build_verl_config(self) -> VerlMegatronConfig:
        """Translate Tinker lora_config to verl Megatron config."""
        return VerlMegatronConfig(
            model_path=self.base_model,
            lora_rank=self.lora_config.get("rank", 16),
            lora_alpha=self.lora_config.get("alpha", 32),
            # Default parallelism for Qwen3-30B-A3B on 8 GPUs
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            expert_parallel_size=4,
            context_parallel_size=2,
        )

    async def initialize(self) -> dict:
        """Initialize verl Megatron worker group.

        This spawns verl's ActorRolloutRefWorker with Megatron backend.
        """
        # Import verl components
        from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        # Create resource pool for training GPUs
        # ... worker initialization ...

        return {"model_id": self.model_id, "status": "ready", "backend": "megatron"}

    async def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str,
        loss_fn_inputs: dict | None = None,
    ) -> dict:
        """Compute forward pass and gradients via verl Megatron.

        Translates Tinker data format to verl DataProto.
        """
        from verl import DataProto

        # Convert data_items to DataProto format
        # ... data conversion ...

        # Call verl's update_actor method
        # result = await self.worker_group.update_actor(data_proto)

        return {"loss": 0.0, "num_tokens": 0}  # Placeholder

    async def optim_step(self, adam_params: dict) -> dict:
        """Optimizer step (handled internally by verl)."""
        # verl handles optimizer step in update_actor
        return {"status": "ok"}

    async def get_lora_checkpoint(self) -> dict:
        """Extract LoRA weights from Megatron model.

        Uses verl's checkpoint manager to gather distributed weights.
        """
        # Use verl's checkpoint extraction
        # ... checkpoint logic ...

        return {}  # LoRA state dict

    def shutdown(self):
        """Clean up verl workers."""
        if self.worker_group:
            # ... cleanup ...
            pass
```

### verl Config Reference

From `run_qwen3moe-30b_megatron_lora.sh`:

```bash
# Model + LoRA config
actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Instruct-2507
actor_rollout_ref.model.lora.rank=16
actor_rollout_ref.model.lora.alpha=32

# Parallelism (TP=2, PP=2, EP=4, CP=2 for 8 GPUs)
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
actor_rollout_ref.actor.megatron.expert_model_parallel_size=4
actor_rollout_ref.actor.megatron.context_parallel_size=2

# Offloading (for memory efficiency)
actor_rollout_ref.actor.megatron.param_offload=True
actor_rollout_ref.actor.megatron.optimizer_offload=True
actor_rollout_ref.actor.megatron.grad_offload=True
```

---

## Phase 3: LoRA Transfer Pipeline (Complete)

**Goal:** Sub-second LoRA adapter transfer between training and inference.

### Transfer Flow

```python
# Training side: after update
lora_state = await trainer.get_lora_checkpoint()  # ~100MB
lora_ref = ray.put(lora_state)                    # Store in object store
session_manager.update_lora(tenant_id, lora_ref)

# Inference side: on sample request
lora_ref = session_manager.get_lora(tenant_id)
lora_state = ray.get(lora_ref)                    # ~0.1s for 100MB
vllm_engine.load_lora_adapter(lora_state)         # ~0.5s hot-swap
```

Actual latency: **0.16s extraction + 2.2s inference** (verified with Qwen3-30B-A3B)

### Tasks

| Task | Status |
|------|--------|
| Implement LoRA extraction from Megatron | Done |
| Validate checkpoint format compatibility | Done |
| Benchmark transfer latency | Done |

### Implementation Notes

LoRA extraction from distributed Megatron requires handling nested module lists (pipeline parallelism):
- `megatron_distributed.py:375-430`: `flatten_modules()` recursively extracts `nn.Module` objects
- Names converted from mbridge format to PEFT format: `layers.0.self_attn.q_proj.lora_A.weight` → `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`
- Integration test: `scripts/test_moe_lora_transfer.py`

---

## Phase 4: Multi-Tenant Support (Complete)

**Goal:** Multiple tenants share base model, each with isolated LoRA adapters.

### Design

- **Inference:** vLLM multi-LoRA serving (concurrent adapters)
- **Training:** Per-tenant LoRA training jobs, time-sliced on training pool
- **Isolation:** Separate session IDs, adapter storage, checkpoint paths

### Tasks

| Task | Status |
|------|--------|
| Multi-tenant session management | Done |
| vLLM multi-LoRA configuration | Done |
| Tenant isolation validation | Done |

### Implementation Notes

Multi-tenant support is built into the existing infrastructure:
- `max_loras=64` GPU slots, `max_cpu_loras=1024` overflow cache
- Per-session `lora_int_id` provides weight isolation
- `LoRARegistry` tracks session → adapter mapping with LRU eviction
- Concurrent inference verified with `scripts/test_multi_tenant.py`

**Not implemented (optional for production):**
- Tenant-level authentication/API keys
- Per-tenant resource quotas
- Cross-tenant access control

---

## Phase 5: Integration Testing (Complete)

### Test Matrix

| Model | GPUs | Paradigm | Status |
|-------|------|----------|--------|
| Qwen3-30B-A3B | 8 | SFT | [x] |
| Qwen3-30B-A3B | 8 | RL (GRPO) | [x] |
| Qwen3-30B-A3B | 8 | LoRA hot-swap | [x] |
| Qwen2.5-7B | 1 | Multi-tenant | [x] |

### Test Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_moe_lora_transfer.py` | SFT + LoRA extraction + vLLM inference |
| `scripts/test_moe_rl.py` | importance_sampling + PPO losses |
| `scripts/test_multi_tenant.py` | Concurrent multi-tenant inference |

### Validation Commands

```bash
# Start server with MoE model
TINKER_MODEL_PATH=/path/to/Qwen3-30B-A3B \
TENSOR_PARALLEL_SIZE=4 \
python scripts/run_server.py

# Create training session (server auto-detects Megatron requirement)
curl -X POST http://localhost:8000/api/v1/create_model \
  -d '{"base_model": "Qwen/Qwen3-30B-A3B", "lora_config": {"rank": 16}}'

# Run training (same API as dense models)
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name="Qwen/Qwen3-30B-A3B" \
    lora_rank=16
```

---

## Implementation Order

| Phase | Description | Status | Dependencies |
|-------|-------------|--------|--------------|
| 1 | MoE Inference (vLLM TP) | Done | - |
| 2 | verl Megatron Adapter | Done | verl installed |
| 3 | LoRA Transfer Pipeline | Done | Phase 2 |
| 4 | Multi-Tenant Support | Done | Phase 3 |
| 5 | Integration Testing | Done | All phases |

**All phases complete.** MoE training and inference fully operational.

---

## Backward Compatibility

- **Dense models (<14B):** Use existing `TrainingWorker` (unchanged)
- **MoE models (30B+):** Route to `VerlMegatronAdapter`

**No API changes.** Server auto-detects based on model name via `model_registry.py`.

---

## Dependencies

- **verl** (pinned version): Megatron backend, checkpoint management
- **Megatron-Core**: Required by verl for Megatron models
- **Megatron-Bridge** >= 0.2.0: Required for MoE LoRA with EP != TP
- **vLLM** >= 0.9.0: MoE inference with tensor parallelism

---

## Phase 6: Multi-Session Megatron Actor Sharing

**Goal:** Multiple training sessions share a single Megatron actor to avoid ~80s restart cost. Each session maintains isolated LoRA weights and optimizer state.

### Current Limitation

The Megatron actor is a singleton (`PERSISTENT_MEGATRON_ACTOR_NAME`) with parameter lock-in at creation time:

```
Session A creates actor: base_model=X, lora_rank=32, lr=1e-4
Session A ends (actor persists)
Session B starts: wants lora_rank=64, lr=1e-5
  → get_or_create returns existing actor
  → Session B's params IGNORED
  → Session B inherits Session A's final weights and hyperparameters
```

This is incorrect for independent sessions.

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MegatronActorPool                                    │
│                                                                              │
│  Key: (base_model, lora_rank) → ActorHandle                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Actor "qwen3-30b-r32"                                                 │   │
│  │   - base_model: Qwen/Qwen3-30B-A3B                                    │   │
│  │   - lora_rank: 32                                                     │   │
│  │   - current_session: session_A (or None if idle)                      │   │
│  │   - lock: asyncio.Lock                                                │   │
│  │                                                                        │   │
│  │   State Storage (per-session, on PFS):                                │   │
│  │   /checkpoints/{session_id}/                                          │   │
│  │     ├── adapter_checkpoint/mp_rank_XX_adapter.pt  (LoRA weights)      │   │
│  │     ├── optimizer/                                 (optimizer state)  │   │
│  │     └── training_meta.json                         (step, lr, etc)    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Actor "qwen3-30b-r64"                                                 │   │
│  │   - base_model: Qwen/Qwen3-30B-A3B                                    │   │
│  │   - lora_rank: 64                                                     │   │
│  │   - ...                                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

Session Flow:
┌─────────────┐     acquire_actor()     ┌─────────────────────┐
│  Session A  │ ───────────────────────►│  Lock acquired      │
│             │                         │  load_session_state │
│             │     forward_backward    │  (adapter + optim)  │
│             │ ───────────────────────►│                     │
│             │     optim_step          │                     │
│             │ ───────────────────────►│                     │
│             │     release_actor()     │  save_session_state │
│             │ ◄───────────────────────│  Lock released      │
└─────────────┘                         └─────────────────────┘
                                                  │
                                                  ▼
┌─────────────┐     acquire_actor()     ┌─────────────────────┐
│  Session B  │ ───────────────────────►│  Lock acquired      │
│  (waiting)  │                         │  load_session_state │
│             │                         │  (Session B's state)│
└─────────────┘                         └─────────────────────┘
```

### Investigation Findings

#### 1. State Components to Swap

| Component | Size (30B MoE, rank=32) | Storage Location | Swap Method |
|-----------|------------------------|------------------|-------------|
| LoRA adapter weights | ~100MB | PFS | `load_adapter_checkpoint()` |
| Optimizer state (Adam) | ~200MB | PFS | `set_optimizer_state_dict()` |
| LR scheduler state | <1KB | PFS | JSON serialize |
| RNG state | <1KB | PFS | `load_rng_states()` |
| Step counter | <1KB | Memory | Session metadata |

#### 2. verl API for State Loading

**Adapter weights** - `verl/utils/megatron_peft_utils.py`:
```python
# Save (existing)
save_adapter_checkpoint(model, checkpoint_path, rank)
# Load (existing)
load_adapter_checkpoint(model, checkpoint_path, strict=True)
```

**Optimizer state** - `verl/utils/checkpoint/megatron_checkpoint_manager.py`:
```python
# Save: generate_state_dict() includes optimizer via sharded_state_dict()
state_dict = manager.generate_state_dict(
    generate_model=False,
    generate_optimizer=True,
    generate_extra=True,
)

# Load: checkpoint_manager.load_checkpoint() handles optimizer
# WARNING: requires coordinated loading across all ranks (NCCL collectives)
```

#### 3. Locking Mechanism

Ray actors are single-threaded - one call at a time. But we need:
- **Explicit session lock** to prevent interleaving mid-batch
- **Async waiting** for sessions queued behind active session

Options:
1. **Actor-level asyncio.Lock** - simple, but requires all callers to acquire
2. **Wrapper class with context manager** - cleaner API
3. **Ray actor method ordering** - relies on Ray's FIFO guarantee

Recommendation: Option 2 - wrapper class that acquires lock + swaps state atomically.

#### 4. Actor Pool Keying

Key: `(base_model, lora_rank)` tuple

Rationale:
- `lora_rank` determines LoRA layer dimensions - cannot change after initialization
- `base_model` determines model architecture
- `learning_rate` can be changed per-session via optimizer state reload

Pool behavior:
- First session with (model, rank) creates actor
- Subsequent sessions reuse existing actor
- Actors persist until explicit kill or resource pressure

### Implementation Tasks

| # | Task | File | Complexity |
|---|------|------|------------|
| 1 | Add `load_adapter_state()` to `MegatronRankWorker` | `megatron_distributed.py` | Medium |
| 2 | Add `save_optimizer_state()` / `load_optimizer_state()` | `megatron_distributed.py` | High |
| 3 | Add `reset_optimizer()` for new sessions | `megatron_distributed.py` | Medium |
| 4 | Create `MegatronActorPool` class | `megatron_distributed.py` | Medium |
| 5 | Add session locking to `MegatronWorkerGroup` | `megatron_distributed.py` | Medium |
| 6 | Add `swap_session()` atomic operation | `megatron_distributed.py` | High |
| 7 | Update `VerlTrainingEngine` to use pool | `verl_training.py` | Medium |
| 8 | Add session state persistence paths | `training_session_manager.py` | Low |
| 9 | Test multi-session sequential access | `tests/` | Medium |
| 10 | Test multi-session concurrent queueing | `tests/` | Medium |

### Detailed Design

#### Task 1: `load_adapter_state()`

```python
# MegatronRankWorker
def load_adapter_state(self, checkpoint_path: str) -> None:
    """Load LoRA adapter weights from checkpoint.

    ALL ranks must call - uses NCCL collectives internally.
    """
    from verl.utils.megatron_peft_utils import load_adapter_checkpoint
    load_adapter_checkpoint(
        model=self.engine.module,
        checkpoint_path=checkpoint_path,
        strict=True,
    )
    logger.info(f"[Rank {self.rank}] Loaded adapter from {checkpoint_path}")
```

#### Task 2: Optimizer State Save/Load

Challenge: Megatron distributed optimizer uses `sharded_state_dict()` which requires coordinated save/load across all ranks.

```python
# MegatronRankWorker
def save_optimizer_state(self, save_path: str) -> None:
    """Save optimizer state to checkpoint path."""
    # Use checkpoint manager's generate_state_dict with optimizer only
    state_dict = self.engine.checkpoint_mananager.generate_state_dict(
        generate_model=False,
        generate_optimizer=True,
        generate_extra=True,  # includes LR scheduler, RNG
    )
    # Save via distributed checkpointing
    save_dist_checkpointing(state_dict, save_path, async_save=False)
    torch.distributed.barrier()

def load_optimizer_state(self, load_path: str) -> None:
    """Load optimizer state from checkpoint path."""
    # Load via checkpoint manager (handles NCCL coordination)
    self.engine.checkpoint_mananager.load_checkpoint(
        local_path=load_path,
        hdfs_path=None,
        del_local_after_load=False,
    )
```

#### Task 3: Reset Optimizer for New Sessions

```python
# MegatronRankWorker
def reset_optimizer(self, learning_rate: float) -> None:
    """Reset optimizer state for a fresh session.

    Clears momentum/variance, resets step counter.
    Sets new learning rate.
    """
    # Option A: Reinitialize optimizer (simple but may have overhead)
    self.engine.optimizer_config.lr = learning_rate
    self.engine.optimizer = self.engine._build_optimizer()

    # Option B: Zero out states (faster but more fragile)
    for group in self.engine.optimizer.param_groups:
        group['lr'] = learning_rate
    for state in self.engine.optimizer.state.values():
        if 'exp_avg' in state:
            state['exp_avg'].zero_()
        if 'exp_avg_sq' in state:
            state['exp_avg_sq'].zero_()
        if 'step' in state:
            state['step'] = 0
```

#### Task 4: MegatronActorPool

```python
# megatron_distributed.py

class MegatronActorPool:
    """Pool of Megatron actors keyed by (base_model, lora_rank)."""

    def __init__(self):
        self._actors: dict[tuple[str, int], MegatronActorEntry] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,  # Only used if creating new
        distributed_config: DistributedConfig,
    ) -> "MegatronActorEntry":
        key = (base_model, lora_rank)
        with self._lock:
            if key not in self._actors:
                actor = self._create_actor(base_model, lora_rank, learning_rate, distributed_config)
                self._actors[key] = MegatronActorEntry(actor=actor, base_model=base_model, lora_rank=lora_rank)
            return self._actors[key]

@dataclass
class MegatronActorEntry:
    actor: ray.actor.ActorHandle
    base_model: str
    lora_rank: int
    current_session: str | None = None
    session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

#### Task 6: Atomic Session Swap

```python
# MegatronWorkerGroup
async def swap_session(
    self,
    old_session_id: str | None,
    new_session_id: str,
    new_checkpoint_path: str | None,
    new_learning_rate: float,
) -> None:
    """Atomically swap from old session to new session.

    1. Save old session state (if any)
    2. Load new session state (or reset if new)
    3. Update current_session marker
    """
    async with self._session_lock:
        # Save old session state
        if old_session_id:
            old_path = self._get_session_checkpoint_path(old_session_id)
            await self._save_session_state(old_path)

        # Load new session state
        if new_checkpoint_path and os.path.exists(new_checkpoint_path):
            await self._load_session_state(new_checkpoint_path)
        else:
            # New session - reset to fresh state
            await self._reset_state(new_learning_rate)

        self._current_session = new_session_id
```

### Estimated Swap Latency

| Operation | Time (estimated) | Notes |
|-----------|-----------------|-------|
| Save adapter (100MB) | ~0.5s | PFS write |
| Save optimizer (200MB) | ~1s | Distributed checkpoint |
| Load adapter (100MB) | ~0.5s | PFS read + param copy |
| Load optimizer (200MB) | ~1s | Distributed checkpoint |
| **Total swap** | **~3s** | Much better than 80s restart |

### Open Questions

1. **Memory pressure:** With optimizer offload enabled, loading optimizer state may trigger CPU→GPU transfer. Need to measure actual latency.

2. **Concurrent session limit:** Should we cap concurrent queued sessions? If 10 sessions queue on same actor, tail latency could be high.

3. **Warm vs cold start:** For new sessions, should we:
   - (A) Reset optimizer to zeros (fast, no prior momentum)
   - (B) Copy from a "template" checkpoint (has warmup momentum)

4. **Checkpoint cleanup:** When should session checkpoints be garbage collected?

### Phase 6 Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M1 | Adapter hot-swap | Load different LoRA weights without restart |
| M2 | Optimizer state swap | Load/save optimizer state per-session |
| M3 | Session locking | Concurrent sessions queue without corruption |
| M4 | Full integration | cookbook tests pass with session reuse |

---

## Phase 7: Unified Rank Support via Max-Rank Padding

Key trainers by `base_model` only (not `base_model + lora_rank`). Initialize with max supported rank, pad/truncate adapters at load/save time.

**Mechanism:**

Rank-64 trainer can train rank-32 adapter:
- Load: zero-pad lora_A rows and lora_B columns from 32 to 64
- Save: truncate back to actual rank
- Scaling: store `actual_rank` per session, adjust `alpha/r` factor

```python
def load_adapter_padded(state_dict, trainer_rank, actual_rank):
    for name, tensor in state_dict.items():
        if 'lora_A' in name:  # (actual_rank, hidden) -> (trainer_rank, hidden)
            padded = torch.zeros(trainer_rank, tensor.shape[1])
            padded[:actual_rank] = tensor
            state_dict[name] = padded
        elif 'lora_B' in name:  # (hidden, actual_rank) -> (hidden, trainer_rank)
            padded = torch.zeros(tensor.shape[0], trainer_rank)
            padded[:, :actual_rank] = tensor
            state_dict[name] = padded
    return state_dict
```

**Scaling correction:**

LoRA output: `lora_B @ lora_A @ x * (alpha / rank)`

If trainer uses fixed `alpha = 2 * trainer_rank`, scale output by `trainer_rank / actual_rank` for sessions with smaller rank.

**Trade-off:**
- Pro: One trainer per base_model instead of per (base_model, rank)
- Con: Wasted FLOPs on zero dimensions, optimizer state for unused params

**Changes:**
- Pool key: `base_model` only
- Session metadata: `actual_rank`, `alpha`
- Load/save: pad/truncate helpers
- Forward: scaling adjustment based on actual_rank

---

## Phase 8: Backport Multi-Session Sharing to Dense Models

Same paradigm as Phase 6 for 7B dense models. Currently each session spawns new `TrainingWorker` (~30s init).

Changes:
- Add `save/load_session_state()` to `TrainingWorker`
- Create `DenseTrainerPool` with `base_model` keying (using Phase 7 max-rank approach)
- Unified interface with `MegatronActorPool`

Swap latency: ~1s (50MB LoRA + 100MB optimizer) vs 30s restart.

---

## Phase 9: Adaptive Resource Management

LRU-based actor pool with dynamic creation/eviction.

```
request_actor(model, type):
  1. Actor exists -> reuse
  2. Resources available -> create
  3. Resources exhausted -> LRU evict idle actors until enough GPUs freed
  4. Cannot free enough -> error
```

Safeguards:
- Never evict actors with active sessions
- Save session state before eviction
- Minimum actor lifetime before eligible for eviction

---

## Phase 10: Model Lineup

### Tinker Official Support (from tinker-cookbook/model_info.py)

**Qwen:**
| Model | Architecture | Active Params |
|-------|--------------|---------------|
| Qwen/Qwen2.5-{0.5,1.5,3,7,14,32,72}B[-Instruct] | Dense | Full |
| Qwen/Qwen3-{0.6,1.7,4,8,14,32}B[-Base] | Dense | Full |
| Qwen/Qwen3-30B-A3B[-Instruct-2507] | MoE 64E | 3B |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | MoE 128E | 22B |

**DeepSeek:**
| Model | Architecture | Active Params |
|-------|--------------|---------------|
| deepseek-ai/DeepSeek-V3.1[-Base] | MoE | 37B (671B total) |

**Kimi (Moonshot AI):**
| Model | Architecture | Active Params | Notes |
|-------|--------------|---------------|-------|
| moonshotai/Kimi-K2-Instruct | MoE | 32B (1T total) | Block-FP8 format |
| moonshotai/Kimi-K2-Thinking | MoE | 32B (1T total) | Reasoning model |
| moonshotai/Kimi-Dev-72B | Dense | 72B | Based on Qwen2.5-72B |

### Our Support Plan

**Supported:**
| Model | Backend | Train GPUs | Infer GPUs | Status |
|-------|---------|------------|------------|--------|
| Qwen/Qwen2.5-7B-Instruct | PEFT | 1 | 1 | Verified |
| Qwen/Qwen3-30B-A3B | Megatron | 8 (TP4,EP2) | 4 (TP4) | Verified |

**To Test:**
| Model | Backend | Train GPUs | Infer GPUs | Blockers |
|-------|---------|------------|------------|----------|
| Qwen/Qwen2.5-14B-Instruct | PEFT | 2 (TP2) | 2 | None |
| Qwen/Qwen3-32B | PEFT/Megatron | 4 (TP4) | 4 | Dense, may fit single-node |
| Qwen/Qwen3-235B-A22B | Megatron | 32 (TP8,EP4) | 16 | Multi-node |
| deepseek-ai/DeepSeek-V3.1 | Megatron | 64+ | 32+ | Needs architecture support |
| moonshotai/Kimi-K2-Instruct | Megatron | 64+ | 32+ | Block-FP8 format, 1T params |

**Not Planned:**
- Llama models (lower priority per user request)
- Vision-language models

### Architecture Requirements

Dense models: PEFT LoRA, TP for >14B.
MoE models: Megatron with TP + EP. Formula: `total_gpus = TP * EP`.

---

## Phase 11: Comprehensive Testing

Compare our implementation against Tinker official on identical model/data/hyperparameters.

**Correctness:**
- Loss curve correlation r > 0.99
- Final loss difference < 1%
- Checkpoint round-trip produces identical weights

**Performance:**
- Throughput >= 90% of Tinker
- Memory <= 110% of Tinker
- Session swap p99 < 5s

**Stress:**
- 10K iterations without OOM/NaN
- 100 sequential session swaps
- 10 concurrent sessions on 1 actor

---

## References

- [verl megatron_peft_utils](https://github.com/volcengine/verl/blob/main/verl/utils/megatron_peft_utils.py)
- [verl MegatronCheckpointManager](https://github.com/volcengine/verl/blob/main/verl/utils/checkpoint/megatron_checkpoint_manager.py)
- [Kimi-K2-Instruct on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2-Instruct)
