# Historical VLA Support Plan for Mint

This file is historical design input, not the normative contract.

Current normative sources are:

- `docs/mint-openpi-vla-target.md`
- `docs/README.md`
- `docs/sub-targets/*.md`
- `docs/plans/2026-03-30-pr422-long-term-remediation.md`
- the verified stage-local scripts under `src/mindlab-toolkit/examples/`

This file remains useful only as a record of early design pressures and boundary questions. It should not be used as the merge-readiness checklist or as the current implementation plan.

## What survived from the historical plan

Several early decisions were directionally right and still survive in the current repo:

- keep `Datum` as the user-facing training unit
- keep `import mint` as the user entrypoint instead of pushing users into upstream `openpi`
- keep action inference separate from text token sampling
- treat `pi0-fast` and `pi0.5` as different model families with different loss semantics
- keep `pi0.5` RL out of the first-stage contract
- avoid promising vLLM-style multi-LoRA serving economics for OpenPI checkpoints

Those ideas now live in the target doc and stage docs instead of in this file.

## What changed after this plan

The original plan predated the final PR422 convergence work. The current repo contract differs in five important ways.

### 1. Public training examples now use `train_step(...)`

This file originally reasoned in terms of split-step public training such as:

- `forward_backward(...)`
- `optim_step(...)`

That is no longer the public teaching path.

Current public guidance is:

- `TrainingClient.train_step(...)` is the default public OpenPI training unit
- split `forward_backward` / `optim_step` remains internal Mint training semantics and low-level test surface
- shared deployment fairness is defined at the complete training-step boundary

### 2. Public action inference now uses MintX routes

The current canonical action boundary is:

- `POST /api/v1/mint/action_sessions`
- `POST /api/v1/mint/action_sessions/{action_session_id}/act`
- `DELETE /api/v1/mint/action_sessions/{action_session_id}`

The old `/api/v1/create_action_session`, `/api/v1/act`, and `/api/v1/action_sessions/{id}` routes are not the long-term contract.

### 3. Default action execution now stays inside Mint control-plane semantics

The current repo no longer treats action inference as a side path outside Mint scheduling.

Default action requests now:

- create a future
- go through `ModelWorkScheduler` when async model-runtime scheduling is needed
- execute on Mint-managed Ray actors
- surface placement and lifecycle through `ModelActorSupervisor`, `ModelWorkScheduler`, and `ModelActorSupervisorInventory`

### 4. Runtime declaration is now repo-owned

This historical plan was written before the runtime contract was pulled into Mint's canonical `runtime_env` metadata.

The current rule is:

- `src/mint/pyproject.toml` owns `tool.tinker.runtime_env`
- the OpenPI source contract includes both `src/openpi/src` and `src/openpi/packages/openpi-client/src`
- host requirements explicitly include the OpenPI worker stack
- `src/mint/scripts/build_runtime_env.py --inspect --env-root ...` is the canonical probe for manifest, layout, and host-python import checks

Do not infer runtime correctness from private workspace paths or from `unison` code sync.

### 5. Shared runtime rollout and model exposure are separate decisions

The shared candidate runtime artifact and the shared-service model allowlist are not the same decision.

Keep these distinct:

- shared runtime candidate: `/vePFS-Mindverse/share/code/mint-runtime-py31213-openpi-candidate-20260331-203300`
- rollback baseline: `/vePFS-Mindverse/share/code/mint-runtime-py31213`
- shared-service allowlist: deployment-level `MINT_SUPPORTED_MODELS`
- repo fallback default list: built-in `allowed` list in `list_supported_models()`

`pi0.5` should not be treated as repo-default exposure just because the candidate runtime root is valid.

## Current reading path

If you need the actual current contract, read in this order:

1. `docs/README.md`
2. `docs/mint-openpi-vla-target.md`
3. `docs/sub-targets/st-07-openpi-ray-gpu-actor.md`
4. `docs/sub-targets/st-08-openpi-shared-ray-actor-pool.md`
5. `docs/sub-targets/st-09-mintx-action-boundary-runtime-contract.md`
6. `docs/plans/2026-04-01-shared-runtime-candidate-rollout-verification.md`
7. `docs/plans/2026-04-01-mint-dev-shared-runtime-rollout-readiness.md`

Then use the verified example scripts as the executable surface:

- `src/mindlab-toolkit/examples/st06_mint_vla_minimal_closure.py`
- `src/mindlab-toolkit/examples/st07_openpi_ray_single_gpu_actor_acceptance.py`
- `src/mindlab-toolkit/examples/st08_openpi_shared_ray_actor_pool_acceptance.py`
- `src/mindlab-toolkit/examples/st09a_mintx_action_boundary_acceptance.py`
- `src/mindlab-toolkit/examples/st09b_openpi_action_ray_runtime_acceptance.py`

## What not to infer from this file

Do not use this file to infer any of the following:

- that split-step public training is still recommended
- that legacy `/api/v1/*` action routes are still canonical
- that action inference still runs outside Mint queue/capacity semantics
- that private workspace paths are an acceptable runtime contract
- that `pi0.5` RL or repo-default exposure is part of the current first-stage commitment

## Bottom line

Keep this file only as historical background.

For current development work, follow the target doc, stage docs, rollout/readiness docs, and the verified example scripts.
