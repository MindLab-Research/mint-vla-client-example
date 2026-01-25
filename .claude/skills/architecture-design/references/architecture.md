# tinker-server architecture reference (index)

tinker-server ("MinT") is a FastAPI service that implements a Tinker-compatible REST API and brokers training/inference to Ray GPU actors.

This reference is split by topic for faster lookup. Start here, then open the relevant topic file.

- Overview (design decisions): `overview.md`
- System boundary and code map: `system.md`
- Identifiers and state ownership: `state.md`
- Async futures (Tinker polling protocol): `async-futures.md`
- Inference architecture (vLLM, multi-LoRA): `inference.md`
- Training architecture (dense vs Megatron): `training.md`
- Training multi-tenancy (state swap): `training-multitenancy.md`
- Ray placement groups (Megatron, multi-node vLLM, dense pool): `placement-groups.md`
- Weights and checkpoints: `weights-checkpoints.md`
- Auto eviction and GPU allocation: `eviction.md`
- Authentication and model access: `auth-access.md`
- Design constraints and change checklist: `constraints-checklist.md`
