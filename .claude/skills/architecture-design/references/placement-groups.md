# Ray placement groups (Megatron worker groups)

Ray placement groups are used in Mint to co-schedule distributed Megatron training workers as a unit.

## What placement groups provide (in this codebase)

- Reserve `world_size` GPU "slots" together so the distributed job starts only when all ranks can run.
- Pin each rank worker to an explicit bundle index so rank 0..N-1 map to predictable GPU bundles.
- Prefer colocation via `strategy="PACK"` while still allowing multi-node scheduling when `world_size` exceeds a single node.

Mint uses placement groups only for MoE Megatron training. Inference (vLLM actors) does not create placement groups, but it reads placement group state to avoid placing inference on nodes already used by Megatron training.

## Where they are created

`tinker_server/backend/megatron_distributed.py`:
- `MegatronWorkerGroup._initialize()` creates a placement group:
  - bundles: `[{\"GPU\": 1, \"CPU\": 1}] * world_size`
  - strategy: `"PACK"`
  - blocks on `placement_group.ready()`
- Rank workers are scheduled into bundle indices:
  - master address/port helper scheduled in bundle 0
  - each `MegatronRankWorker` scheduled with `placement_group_bundle_index=rank`

The key property is that Ray will not start the worker group unless it can reserve all bundles in the placement group.

## Why PACK (not STRICT_PACK)

The intent is "single node if possible, multi-node if required".

For large configurations (for example, `world_size=16` on 8-GPU nodes), `STRICT_PACK` would block forever because it requires all bundles on one node. `PACK` still prefers colocation but permits spilling across nodes.

## Interactions with inference placement

`tinker_server/backend/multi_lora_engine.py` and `tinker_server/backend/multinode_inference.py` compute per-node "available GPUs" as:

`available = total - gpus_used_by_placement_groups - gpus_used_by_resource_pool_actors`

This is a workaround for the fact that "GPU slot assigned by Ray" is not the same thing as "CUDA memory available for another large model". Mint treats placement groups as hard occupancy signals when picking nodes for vLLM.

## Cleanup and failure mode

`MegatronWorkerGroup.shutdown()` calls `ray.util.remove_placement_group(self.placement_group)`.

If the worker group process is terminated without running that shutdown path (eviction, crash), placement groups can remain in Ray state and continue to reserve GPU resources. Symptoms:
- new Megatron worker groups fail to schedule despite idle GPUs
- vLLM placement avoids nodes due to "PG-used GPU" accounting

When this happens, clean up placement groups at the Ray cluster level (see `volcano-cluster` skill procedures).
