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
| Test with Qwen3-30B-A3B | - | [ ] |

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

## Phase 3: LoRA Transfer Pipeline

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

Expected latency: **0.6-0.7s** (validated in Phase 1 for dense models)

### Tasks

| Task | Status |
|------|--------|
| Implement LoRA extraction from Megatron | [ ] |
| Validate checkpoint format compatibility | [ ] |
| Benchmark transfer latency | [ ] |

---

## Phase 4: Multi-Tenant Support

**Goal:** Multiple tenants share base model, each with isolated LoRA adapters.

### Design

- **Inference:** vLLM multi-LoRA serving (concurrent adapters)
- **Training:** Per-tenant LoRA training jobs, time-sliced on training pool
- **Isolation:** Separate session IDs, adapter storage, checkpoint paths

### Tasks

| Task | Status |
|------|--------|
| Multi-tenant session management | [ ] |
| vLLM multi-LoRA configuration | [ ] |
| Tenant isolation validation | [ ] |

---

## Phase 5: Integration Testing

### Test Matrix

| Model | GPUs | Paradigm | Status |
|-------|------|----------|--------|
| Qwen3-30B-A3B | 8 | SFT | [ ] |
| Qwen3-30B-A3B | 8 | RL (GRPO) | [ ] |
| Qwen3-30B-A3B | 8 | LoRA hot-swap | [ ] |

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

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | MoE Inference (vLLM TP) | Done | - |
| 2 | verl Megatron Adapter | 2 weeks | verl installed |
| 3 | LoRA Transfer Pipeline | 1 week | Phase 2 |
| 4 | Multi-Tenant Support | 1 week | Phase 3 |
| 5 | Integration Testing | 1 week | All phases |

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

## References

- [verl Megatron Workers](https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py)
- [verl Qwen3-30B LoRA Script](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3moe-30b_megatron_lora.sh)
- [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [vLLM Multi-LoRA](https://docs.vllm.ai/en/latest/models/lora.html)
