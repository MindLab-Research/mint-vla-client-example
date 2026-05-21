# Ray placement groups (Megatron, multi-node vLLM, dense trainer pool)

Ray placement groups are used in Mint to co-schedule multi-actor GPU workloads as a unit and to prevent Ray from fragmenting GPU allocations across nodes in ways that break large-model initialization.

## What placement groups provide (in this codebase)

- Reserve `world_size` GPU "slots" together so the distributed job starts only when all ranks can run.
- Pin each rank worker to an explicit bundle index so rank 0..N-1 map to predictable GPU bundles.
- Prefer colocation via `strategy="PACK"` while still allowing multi-node scheduling when `world_size` exceeds a single node.

Mint uses placement groups for:
- MoE Megatron training (MegatronWorkerGroup rank workers).
- Multi-node vLLM inference (MultiNodeInferenceEngine: controller + captured worker actors).
- Dense training pool actors (DenseTrainerPool: isolates 1-GPU trainers and makes cleanup deterministic).

## Where they are created

`mint_server/backend/megatron_distributed.py` (MoE training):
- `MegatronWorkerGroup._initialize()` creates a placement group:
  - bundles: `[{\"GPU\": 1, \"CPU\": 1}] * world_size`
  - strategy: `"PACK"`
  - blocks on `placement_group.ready()`
- Rank workers are scheduled into bundle indices:
  - master address/port helper scheduled in bundle 0
  - each `MegatronRankWorker` scheduled with `placement_group_bundle_index=rank`

The key property is that Ray will not start the worker group unless it can reserve all bundles in the placement group.

`mint_server/backend/multinode_inference.py` (multi-node vLLM):
- Creates a detached placement group named `{actor_name}_pg` with `total_required_gpus = worker_gpus` GPU bundles plus one CPU-only controller bundle.
- Uses `strategy="PACK"` to keep 1-GPU workers from consuming 1 GPU on every node.
- Schedules the controller actor into the controller bundle and enables `placement_group_capture_child_tasks=True` so vLLM worker actors land in the same group.
- Capacity validation ignores the same named placement group in the Mint namespace
  before reusing it. A failed vLLM EngineCore may leave the actor's placement
  group reserved; retrying the same desired topology should reuse that reservation
  rather than reporting that the actor is blocked by its own placement group.

`mint_server/backend/verl_training.py` (DenseTrainerPool):
- Creates a detached placement group named `{actor_name}_pg` for each pooled `TrainingWorker` actor.

## Why PACK (not STRICT_PACK)

The intent is "single node if possible, multi-node if required".

For large configurations (for example, `world_size=16` on 8-GPU nodes), `STRICT_PACK` would block forever because it requires all bundles on one node. `PACK` still prefers colocation but permits spilling across nodes.

## Cleanup and failure mode

`MegatronWorkerGroup.shutdown()` calls `ray.util.remove_placement_group(self.placement_group)`.

If the worker group process is terminated without running that shutdown path (eviction, crash), placement groups can remain in Ray state and continue to reserve GPU resources. Symptoms:
- new Megatron worker groups fail to schedule despite idle GPUs
- multi-node vLLM init fails because expected bundles cannot be reserved

When this happens, clean up placement groups at the Ray cluster level (see `volcano-cluster` skill procedures).
