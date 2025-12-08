# Tinker-Server: Qwen3 MoE Support Plan

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

## Milestone: Qwen3 MoE Support

### Target Models

| Model | Total Params | Active Params | Min GPUs | Status |
|-------|--------------|---------------|----------|--------|
| Qwen3-30B-A3B | 30B | 3B | 4x A100-80GB | Target |
| Qwen3-235B-A22B | 235B | 22B | 8x A100-80GB | Stretch |

### Architecture Requirements

MoE models require multi-GPU training with FSDP. Current single-GPU actor pattern insufficient.

**Current (Dense):**
```
1 Training Session = 1 Ray Actor = 1 Process = 1 GPU
```

**Required (MoE):**
```
1 Training Session = N Ray Actors = N Processes = N GPUs (FSDP)
```

---

## Phase 1: Inference Support

**Goal:** Serve Qwen3 MoE models for sampling and logprob computation.

**Dependencies:** vLLM >= 0.9.0

### Tasks

| Task | File | Status |
|------|------|--------|
| Add `tensor_parallel_size` config | `config.py` | [ ] |
| Add `expert_parallel_size` config | `config.py` | [ ] |
| Update vLLM server args | `verl_inference.py` | [ ] |
| Test Qwen3-30B-A3B inference | - | [ ] |

### Implementation

```python
# config.py
@dataclass
class ServerConfig:
    # Existing
    tensor_parallel_size: int = 1

    # New for MoE
    expert_parallel_size: int = 1  # vLLM expert parallelism
    enable_expert_parallel: bool = False
```

```python
# verl_inference.py - ExtendedVLLMHttpServer.__init__
if self.config.enable_expert_parallel:
    args.enable_expert_parallel = True
    args.expert_parallel_size = self.config.expert_parallel_size
```

### Validation

```bash
# Start server with Qwen3-30B-A3B
TINKER_MODEL_PATH=/path/to/Qwen3-30B-A3B \
TENSOR_PARALLEL_SIZE=4 \
python scripts/run_server.py

# Test sampling
curl -X POST http://localhost:8000/api/v1/asample ...
```

---

## Phase 2: FSDP Training Infrastructure

**Goal:** Replace single-GPU `TrainingWorker` with multi-GPU FSDP training via Ray placement groups.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FSDPTrainingSession                       │
│  (Manages placement group + coordinates N workers)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│   │ FSDPWorker   │  │ FSDPWorker   │  │ FSDPWorker   │ ...  │
│   │ rank=0       │  │ rank=1       │  │ rank=2       │      │
│   │ GPU 0        │  │ GPU 1        │  │ GPU 2        │      │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│          │                 │                 │               │
│          └─────────────────┼─────────────────┘               │
│                            │                                 │
│                    NCCL Collectives                          │
│                 (torch.distributed)                          │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

| Task | File | Status |
|------|------|--------|
| Create `ModelConfig` registry | `model_registry.py` | [ ] |
| Create `FSDPWorker` Ray actor | `fsdp_training.py` | [ ] |
| Create `FSDPTrainingSession` coordinator | `fsdp_training.py` | [ ] |
| Integrate with `TrainingSessionManager` | `training_session_manager.py` | [ ] |
| Update `POST /create_model` to auto-detect FSDP | `routes/training.py` | [ ] |
| Update `POST /forward_backward` for FSDP | `routes/training.py` | [ ] |
| FSDP checkpoint save/load | `fsdp_training.py` | [ ] |

### New File: `tinker_server/backend/fsdp_training.py`

```python
"""FSDP Training with Ray Placement Groups.

Multi-GPU training via coordinated Ray actors with torch.distributed.
"""

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import LoraConfig, get_peft_model


@ray.remote
def get_master_addr_port() -> tuple[str, str]:
    """Get master address and port for torch.distributed."""
    import socket
    addr = ray.util.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    return addr, str(port)


@ray.remote(num_gpus=1)
class FSDPWorker:
    """Single worker in an FSDP training group.

    Each worker holds a shard of the model and participates in
    collective operations via torch.distributed.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        base_model: str,
        lora_rank: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.base_model = base_model
        self.lora_rank = lora_rank

        # Set environment for torch.distributed
        import os
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        self.initialized = False

    def init_distributed(self):
        """Initialize torch.distributed and FSDP model."""
        if self.initialized:
            return

        # Initialize process group
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(0)  # Each worker sees only its GPU

        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )

        # Load with meta device for efficient FSDP init
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # Apply LoRA before FSDP wrapping
        peft_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            target_modules="all-linear",  # Includes MoE expert layers
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

        # Wrap with FSDP
        from torch.distributed.fsdp import ShardingStrategy
        from verl.utils.fsdp_utils import get_fsdp_wrap_policy

        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=get_fsdp_wrap_policy(model),
            device_id=torch.cuda.current_device(),
        )

        # Optimizer for LoRA params only
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        self.initialized = True
        return {"rank": self.rank, "status": "ready"}

    def forward_backward(self, data_items: list[dict], loss_fn: str) -> dict:
        """Forward + backward pass with gradient sync via FSDP."""
        self.model.train()

        # Process batch (same logic as current TrainingWorker)
        # ... loss computation ...

        loss.backward()  # FSDP handles gradient sync

        # Only rank 0 returns metrics
        if self.rank == 0:
            return {"loss": loss.item(), "metrics": {...}}
        return {}

    def optim_step(self, adam_params: dict) -> dict:
        """Optimizer step (synchronized across workers)."""
        # Update learning rate
        for pg in self.optimizer.param_groups:
            pg["lr"] = adam_params.get("learning_rate", pg["lr"])

        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"status": "ok"}

    def get_state_dict(self) -> dict:
        """Get full state dict (gathered from all shards)."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            if self.rank == 0:
                return self.model.state_dict()
        return {}


class FSDPTrainingSession:
    """Coordinator for multi-GPU FSDP training session.

    Manages a placement group of FSDPWorkers that form one training session.
    """

    def __init__(
        self,
        session_id: str,
        base_model: str,
        lora_rank: int,
        num_gpus: int = 4,
    ):
        self.session_id = session_id
        self.num_gpus = num_gpus
        self.workers: list[ray.actor.ActorHandle] = []

        # Create placement group
        self.pg = placement_group(
            bundles=[{"GPU": 1, "CPU": 4} for _ in range(num_gpus)],
            strategy="STRICT_PACK",  # Same node for NVLink
            name=f"fsdp_training_{session_id}",
        )
        ray.get(self.pg.ready())

        # Get master addr/port
        master_addr, master_port = ray.get(get_master_addr_port.remote())

        # Spawn workers
        for rank in range(num_gpus):
            worker = FSDPWorker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg,
                    placement_group_bundle_index=rank,
                )
            ).remote(
                rank=rank,
                world_size=num_gpus,
                master_addr=master_addr,
                master_port=master_port,
                base_model=base_model,
                lora_rank=lora_rank,
            )
            self.workers.append(worker)

        # Initialize all workers
        ray.get([w.init_distributed.remote() for w in self.workers])

    async def forward_backward(self, data_items: list[dict], loss_fn: str) -> dict:
        """Broadcast data to all workers, return metrics from rank 0."""
        results = await asyncio.gather(*[
            w.forward_backward.remote(data_items, loss_fn)
            for w in self.workers
        ])
        # Rank 0 has the metrics
        return results[0]

    async def optim_step(self, adam_params: dict) -> dict:
        """Synchronized optimizer step across all workers."""
        results = await asyncio.gather(*[
            w.optim_step.remote(adam_params)
            for w in self.workers
        ])
        return results[0]

    async def save_checkpoint(self, path: str) -> str:
        """Save FSDP checkpoint (rank 0 gathers full state)."""
        state_dict = await self.workers[0].get_state_dict.remote()
        torch.save(state_dict, path)
        return path

    def shutdown(self):
        """Clean up placement group and workers."""
        for worker in self.workers:
            ray.kill(worker)
        ray.util.remove_placement_group(self.pg)
```

### API Changes

**No API changes.** User specifies model name, server auto-detects GPU requirements.

```python
# routes/training.py
from ..backend.model_registry import get_model_config

def create_model(request: CreateModelRequest):
    model_config = get_model_config(request.base_model)

    if model_config.requires_fsdp:
        # MoE or large model -> FSDP path
        session = FSDPTrainingSession(
            session_id=model_id,
            base_model=request.base_model,
            lora_rank=request.lora_config.rank,
            num_gpus=model_config.min_gpus,
        )
    else:
        # Dense model that fits on single GPU -> legacy path
        worker = TrainingWorker.remote(...)
```

### New File: `tinker_server/backend/model_registry.py`

```python
"""Model configuration registry.

Maps model names to hardware requirements.
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Hardware requirements for a model."""
    requires_fsdp: bool  # True for MoE or models > single GPU memory
    min_gpus: int        # Minimum GPUs needed
    is_moe: bool         # Whether model uses MoE architecture
    total_params: str    # e.g., "30B", "235B"
    active_params: str   # For MoE: activated params per token

# Model registry
MODEL_CONFIGS = {
    # Dense models (single GPU)
    "Qwen/Qwen2.5-0.5B-Instruct": ModelConfig(False, 1, False, "0.5B", "0.5B"),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelConfig(False, 1, False, "1.5B", "1.5B"),
    "Qwen/Qwen2.5-3B-Instruct": ModelConfig(False, 1, False, "3B", "3B"),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(False, 1, False, "7B", "7B"),
    "Qwen/Qwen2.5-14B-Instruct": ModelConfig(False, 2, False, "14B", "14B"),
    "Qwen/Qwen2.5-32B-Instruct": ModelConfig(True, 2, False, "32B", "32B"),
    "Qwen/Qwen2.5-72B-Instruct": ModelConfig(True, 4, False, "72B", "72B"),

    # Qwen3 MoE models
    "Qwen/Qwen3-30B-A3B": ModelConfig(True, 4, True, "30B", "3B"),
    "Qwen/Qwen3-235B-A22B": ModelConfig(True, 8, True, "235B", "22B"),
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get hardware config for model, with fallback heuristics."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Fallback: estimate from model name
    # e.g., "Qwen/Qwen3-30B-A3B" -> MoE, "Llama-3-70B" -> large dense
    if "-A" in model_name.split("/")[-1]:
        # MoE pattern: "30B-A3B" means 30B total, 3B active
        return ModelConfig(True, 4, True, "unknown", "unknown")

    # Default: single GPU dense
    return ModelConfig(False, 1, False, "unknown", "unknown")
```

---

## Phase 3: LoRA on MoE

**Goal:** Correct LoRA target modules for Qwen3 MoE architecture.

### Qwen3 MoE Module Structure

```
Qwen3MoeDecoderLayer
├── self_attn
│   ├── q_proj, k_proj, v_proj, o_proj  # Standard attention
├── mlp (Qwen3MoeSparseMoeBlock)
│   ├── gate                             # Router (don't LoRA)
│   ├── experts                          # nn.ModuleList
│       ├── [i].gate_proj                # Expert FFN
│       ├── [i].up_proj
│       └── [i].down_proj
```

### LoRA Configuration

```python
# For Qwen3 MoE models
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        # Attention (standard)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Expert FFN (MoE-specific)
        "gate_proj", "up_proj", "down_proj",
    ],
    # Alternative: "all-linear" if PEFT handles MoE correctly
    modules_to_save=[],  # Don't save router weights
    lora_dropout=0.0,
    bias="none",
)
```

### Tasks

| Task | Status |
|------|--------|
| Verify `target_modules="all-linear"` works with Qwen3 MoE | [ ] |
| Test explicit module list if needed | [ ] |
| Validate LoRA checkpoint format for MoE | [ ] |

---

## Phase 4: Inference-Training Weight Sync

**Goal:** Transfer LoRA weights from FSDP training to vLLM inference.

### Challenge

FSDP shards weights across workers. Need to gather full state before syncing to inference.

### Implementation

```python
# FSDPTrainingSession.save_weights_for_sampler()
async def save_weights_for_sampler(self) -> tuple[dict, dict]:
    """Gather LoRA weights from FSDP shards for inference."""
    # Rank 0 gathers full state dict
    full_state = await self.workers[0].get_state_dict.remote()

    # Extract LoRA-only weights
    lora_state = {
        k: v for k, v in full_state.items()
        if "lora_" in k
    }

    # Get PEFT config
    peft_config = await self.workers[0].get_lora_config.remote()

    return lora_state, peft_config
```

---

## Phase 5: Integration Testing

### Test Matrix

| Model | GPUs | Paradigm | Status |
|-------|------|----------|--------|
| Qwen3-30B-A3B | 4 | SFT | [ ] |
| Qwen3-30B-A3B | 4 | RL (PPO) | [ ] |
| Qwen3-30B-A3B | 4 | DPO | [ ] |
| Qwen3-235B-A22B | 8 | SFT | [ ] |

### Validation Commands

```bash
# Start server with MoE model
TINKER_MODEL_PATH=/path/to/Qwen3-30B-A3B \
TENSOR_PARALLEL_SIZE=4 \
python scripts/run_server.py

# Create training session (server auto-detects FSDP requirement from model name)
curl -X POST http://localhost:8000/api/v1/create_model \
  -d '{"base_model": "Qwen/Qwen3-30B-A3B", "lora_config": {"rank": 32}}'

# Run training (same API as dense models)
python -m tinker_cookbook.recipes.chat_sl.train \
    model_name="Qwen/Qwen3-30B-A3B" \
    lora_rank=32
```

---

## Implementation Order

| Phase | Description | Dependencies |
|-------|-------------|--------------|
| 1 | Inference (vLLM config) | vLLM >= 0.9.0 |
| 2 | FSDP Training Infrastructure | Phase 1 |
| 3 | LoRA on MoE | Phase 2 |
| 4 | Weight Sync | Phase 2, 3 |
| 5 | Integration Testing | All phases |

---

## Backward Compatibility

Existing single-GPU training continues to work:

- Dense models (Qwen2.5-7B, etc.): Uses current `TrainingWorker` path
- MoE/large models (Qwen3-30B-A3B, etc.): Uses new `FSDPTrainingSession` path

**No API changes.** Server auto-detects based on model name via `model_registry.py`.

---

## Open Questions

1. **Expert Parallelism in Training:** Should we support EP in addition to FSDP for very large MoE models?

2. **Checkpoint Format:** Use verl's `FSDPCheckpointManager` or custom format?

3. **Mixed Precision:** bf16 vs fp16 for MoE expert computations?

4. **Router Training:** Should LoRA apply to router weights or freeze them?

---

## References

- [verl FSDP Workers](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py)
- [verl FSDPEngine](https://github.com/volcengine/verl/blob/main/verl/workers/engine/fsdp/transformer_impl.py)
- [HuggingFace Qwen3MoE](https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe)
- [PEFT LoRA for MoE](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [vLLM Expert Parallel](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)
- [Ray Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)
