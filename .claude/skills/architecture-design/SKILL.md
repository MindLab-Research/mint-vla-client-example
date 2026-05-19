---
name: architecture-design
description: |
  Architectural design reference for mint-server (MinT).

  Use for: understanding or changing system boundaries (FastAPI routes, Ray actor lifecycles,
  training vs inference backends, weight transfer, and GPU scheduling via ModelActorSupervisor/ModelWorkScheduler).

  Triggers: "architecture", "system design", "refactor", "add endpoint", "Ray actor",
  "training/inference", "weights transfer", "session lifecycle", "model actor registry"
---

# Architecture design (mint-server)

Engineering docs/specs live under `.claude/skills/architecture-design/references/` (not `docs/`).

Read `references/architecture.md` first (index). For design intent, read `references/overview.md`. Then open the relevant topic file:
- Overview (design decisions): `references/overview.md`
- System boundary and code map: `references/system.md`
- Identifiers and state ownership: `references/state.md`
- Async futures (Tinker polling protocol): `references/async-futures.md`
- Inference architecture: `references/inference.md`
- Training architecture: `references/training.md`
- Training multi-tenancy (state swap): `references/training-multitenancy.md`
- Ray placement groups (Megatron, multi-node vLLM, dense pool): `references/placement-groups.md`
- Weights and checkpoints: `references/weights-checkpoints.md`
- Auto eviction and GPU allocation: `references/eviction.md`
- Authentication and model access: `references/auth-access.md`
- Design constraints and change checklist: `references/constraints-checklist.md`

When a change crosses a boundary, write down (in your own scratch notes) the answers to:
- Which identifier(s) are involved (`session_id`, `model_id`, `sampling_session_id`, `request_id`)?
- Where will the source-of-truth state live after the change (server memory, Ray actor, filesystem)?
- What happens on API server restart (which state is lost, which actors survive)?
- Does `ModelActorSupervisor` reconciliation or the API control-plane client boundary need updating?

For API semantics (types, loss functions, polling behavior), consult the `tinker-official-reference` skill section files under `.claude/skills/tinker-official-reference/references/upstream/` and keep `mint_server/models/types.py` aligned.
