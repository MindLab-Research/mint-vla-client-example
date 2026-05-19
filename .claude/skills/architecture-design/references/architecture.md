# mint-server architecture reference (index)

mint-server is the MinT FastAPI service that implements a Tinker-compatible REST API and brokers training/inference to Ray GPU actors.

This reference is split by topic for faster lookup. Start here, then open the relevant topic file.

- Overview (design decisions): `overview.md`
- System boundary and code map: `system.md`
- HTTP API boundary and internal routes: `internal-api.md`
- Identifiers and state ownership: `state.md`
- Async futures (Tinker polling protocol): `async-futures.md`
- Inference architecture (vLLM, multi-LoRA): `inference.md`
- Training architecture (dense vs Megatron): `training.md`
- Training multi-tenancy (state swap): `training-multitenancy.md`
- Ray placement groups (Megatron, multi-node vLLM, dense pool): `placement-groups.md`
- Weights and checkpoints: `weights-checkpoints.md`
- Auto eviction and GPU allocation: `eviction.md`
- Authentication and model access: `auth-access.md`
- Dependency architecture (runtime env root, image boundary, host bootstrap): `dependency-architecture.md`
- Runtime configuration and ConfigActor: `runtime-config.md`
- Topology-aware supervision and node metrics DaemonSet: `topology-daemon-metrics.md`
- Usage billing storage (JSONL -> async PostgreSQL): `usage-billing-storage.md`
- Design constraints and change checklist: `constraints-checklist.md`
- VLA user-facing API guide: `vla_tinker_api_guide.md`
- VLA implementation plan: `vla_implementation_plan.md`
