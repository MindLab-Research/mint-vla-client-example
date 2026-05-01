---
name: merge-gate
description: |
  Dev-server validation slate for tinker-server before release or risky merges.

  Use for: explicit scenario selection against a real dev server, GPU-aware regression coverage,
  integrated training/control-plane validation, and iterative fix-and-rerun workflows.

  Triggers: "merge gate", "pre-merge", "validate for release", "run selected gate items"

  This skill is not a deterministic PASS/FAIL gate. It runs explicitly selected scenario items,
  gathers evidence, classifies failures, and co-iterates the server and the scenario runners.

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# merge-gate

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

## Intent

This skill validates `develop` against a real dev server by running selected scenario items.

The unit of work is a **scenario item**, not a pytest file and not a fixed phase.
Each merge-gate session must explicitly select which scenario items to run.

The purpose is to reveal server behavior under realistic training and control-plane workflows:

- training algorithms
- checkpoint and resume continuity
- session interleaving and loss spikes
- create-model queueing and admission control
- actor lifecycle and idle cleanup
- memory-retention and max-context transition failures
- opt-in architecture-specific coverage

This skill does **not** define a rigid merge blocker. It defines a disciplined validation loop.

## Hard Rules

- Dev only. Use `mint-dev` conventions and target the dev server, not prod.
- Selection is required every time. If the user does not name scenario items, stop and ask which items to run. Do not invent defaults.
- No unit tests. Every selected item must hit a real dev server.
- Pytest is allowed only as an integration harness. The scenario still has to exercise the live server.
- Every catalog item in this skill must be runnable today. Do not list placeholders in the active catalog.
- Active merge-gate scenarios must be listed and curated in this `SKILL.md`, not only implied by scattered scripts or tests elsewhere in the repo.
- For any selected item that produces a convergence curve or trajectory, you must generate the curve artifact, inspect it visually, and present that visual assessment to the user.
- Do not judge convergence with fixed numeric thresholds like "final loss < 0.8x initial loss" or similar ratio gates. Those thresholds are not valid evidence of healthy learning behavior.
- No deterministic PASS/FAIL gate language. Report observed behavior and remaining risk.
- Distinguish:
  - `server_issue`: the scenario is valid and the server/runtime failed.
  - `test_issue`: the scenario runner, oracle, or expectations are wrong or stale.
  - `skill_issue`: the catalog metadata, GPU accounting, sequencing, or workflow text is wrong.
- GPU-based deselection is automatic and explicit. If a selected item needs more free GPUs than currently available, mark it `deselected_for_gpu` and report the required vs observed budget.
- No Feishu posting in this workflow.
- Do not talk about "all tests passed therefore merge allowed". That is not the contract.

## Non-goals

- No local-only correctness checks as merge-gate coverage.
- No pure unit-test suites.
- No hidden profile system. Selection is by explicit item id every session.
- No silent fallback from a missing scenario to a weaker substitute.

## Environment

- Primary target: dev server via `http://localhost:8000` through the `mint-dev` tunnel.
- Use `mint-dev` for server operations.
- Use `volcano-cluster` when GPU capacity or worker topology must change.
- Never manually sync code; Unison is the sync mechanism.

## Artifacts

Write artifacts under:

- `results/merge-gate/<timestamp>/`

Recommended structure:

- `manifest.json`
- `session_notes.md`
- `items/<item_id>/stdout.log`
- `items/<item_id>/stderr.log`
- `items/<item_id>/summary.json`
- `items/<item_id>/classification.json`
- `items/<item_id>/artifacts/`

`manifest.json` should capture:

- selected item ids
- deselected item ids and reasons
- observed free GPU budget before each item
- server revision / git sha
- server target
- rerun history

## Single Source Of Truth

The canonical active merge-gate catalog is the human-readable catalog in this `SKILL.md`.

Rules:

- A scenario is not part of active merge-gate coverage unless it appears in the runnable catalog below.
- If a useful repro script exists elsewhere in the repo, that is fine, but it still must be curated into this `SKILL.md` before it counts as merge-gate coverage.
- The human-readable catalog is the operator-facing source of truth. Keep it current and explicit.

## Selection Contract

Every merge-gate session must specify scenario item ids explicitly.

Allowed forms:

- "run merge gate: `rl_sanity_dense_0p6b`, `resume_training_moe_30b`, `admission_control_backpressure`"
- "select only `sdpo_moe_30b` and `transition_sequence_235b_max_context`"
- "run all currently selected items except Moonlight"

If the user says only "run merge gate" with no items:

- do not pick a default slate
- do not infer a profile
- ask for explicit item ids

Recommended operator behavior:

- If sanity items are selected, run them first.
- If sanity items are not selected, state that clearly in the run manifest.

## GPU Accounting

`gpu_required` means the minimum free GPU budget required before starting the scenario.

Interpretation:

- It is a conservative lower bound for the selected dev topology.
- It is not a promise that the scenario will fit under all fragmentation or placement states.
- If the free budget is below `gpu_required`, the item is not run.

Minimum preflight evidence:

- dev server `healthz`
- current actor inventory
- current free GPU count

When possible, use both:

- server-facing actor visibility
- Ray free and total GPU visibility on the dev cluster

If these disagree, report the disagreement instead of inventing a single truth.

## Execution Loop

For each selected item:

1. Re-check free GPUs and actor state.
2. If GPU budget is insufficient, mark the item `deselected_for_gpu` and continue.
3. Run the item's configured runner against the real dev server.
4. Save raw artifacts.
5. Summarize the observed behavior in plain terms.
6. On failure, classify the failure as `server_issue`, `test_issue`, or `skill_issue`.
7. If a fix is applied, rerun:
   - the affected item
   - and any selected sanity items that cover the same surface

Do not escalate a single failed scenario into "merge blocked" without saying what was actually observed.

## Taxonomy

Suggested scenario categories:

- `sanity`
  - short RL-loop probes that tell you whether the dev server is fundamentally alive
- `training_algorithm`
  - DPO, SDPO, RL, SFT
- `checkpoint_resume`
  - save/load/resume trajectory fidelity
- `concurrency_isolation`
  - interleaving, loss spikes, same-session ordering, multi-session correctness
- `server_control_plane`
  - admission control, queueing, actor lifecycle, create-model contention
- `heavyweight_capacity`
  - max-context, large-model, or expensive cluster-stress scenarios
- `opt_in_architecture`
  - Moonlight or other non-core architecture coverage

Categories are organizational only. None of them are "secondary" by definition.

## Runnable Scenario Catalog

Every item below is runnable now and tied either to a related issue/PR or to a current live integration surface.

VLA rows below target the active Mint API server surface. Use `MINT_BASE_URL` (default `http://localhost:8000`) and `MINT_API_KEY` for auth-enabled servers. Use a dedicated VLA server only when you need isolation from other workloads, not as a mandatory requirement.

### Sanity

| id | provenance | model | gpu_required | runner | reveals |
|---|---|---|---:|---|---|
| `rl_sanity_dense_0p6b` | sanity-check workflow | `Qwen/Qwen3-0.6B` | 2 | `MINT_TEST_EXPERIMENT_ROOT=results/merge-gate/<ts>/rl_sanity_dense_0p6b python .claude/skills/sanity-check/mint_rl_test_long.py --model Qwen/Qwen3-0.6B --num-rl-steps=1 --batch-size=2 --group-size=4 --max-tokens=128` | Basic dense train, sample, save, and future retrieval path on the real dev server. |
| `rl_sanity_moe_30b` | sanity-check workflow | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 8 | `MINT_TEST_EXPERIMENT_ROOT=results/merge-gate/<ts>/rl_sanity_moe_30b python .claude/skills/sanity-check/mint_rl_test_long.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --num-rl-steps=1 --batch-size=2 --group-size=4 --max-tokens=128` | Basic Megatron plus vLLM health on the real dev server. |

### Training Algorithm

| id | provenance | model | gpu_required | runner | reveals |
|---|---|---|---:|---|---|
| `sft_dense_0p6b` | existing merge-gate dense SFT coverage | `Qwen/Qwen3-0.6B` | 1 | `python -m pytest .claude/skills/merge-gate/tests/test_dense_sft.py::TestDenseSFT::test_pig_latin_training -v -s` | Low-cost dense supervised training behavior. |
| `sft_moe_30b` | existing merge-gate MoE SFT coverage | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | `python -m pytest .claude/skills/merge-gate/tests/test_moe_sft.py::TestMoESFT::test_moe_sft_training -v -s` | Low-cost MoE supervised training path without relying on the RL loop. |
| `dpo_dense_0p6b` | current DPO path on dev | `Qwen/Qwen3-0.6B` | 2 | `python -m pytest .claude/skills/merge-gate/tests/test_dense_dpo.py::TestDenseDPO::test_dpo_training -v -s` | Dense DPO loss path and preference-margin behavior on the live server. |
| `dpo_moe_30b` | current MoE DPO path | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 8 | `python scripts/tools/verify_convergence_matrix.py --models Qwen/Qwen3-30B-A3B-Instruct-2507 --loss-fns dpo --seeds 1 --steps 3 --output-dir results/merge-gate/<ts>/dpo_moe_30b` | MoE DPO path on the real server using the cookbook preference-pair dataset. |
| `sdpo_dense_0p6b` | PR 375 | `Qwen/Qwen3-0.6B` | 2 | `python scripts/tools/mintx_sdpo_train.py --base-url http://localhost:8000 --api-key dummy --model Qwen/Qwen3-0.6B --output-dir results/merge-gate/<ts>/sdpo_dense_0p6b --steps 4 --train-batch-size 4 --eval-size 16 --train-size 64 --probe-size 8 --max-tokens 32` | MintX reverse-KL and checkpoint interpolation path on a small dense model. |
| `sdpo_moe_30b` | PR 375 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 8 | `python scripts/tools/mintx_sdpo_train.py --base-url http://localhost:8000 --api-key dummy --model Qwen/Qwen3-30B-A3B-Instruct-2507 --output-dir results/merge-gate/<ts>/sdpo_moe_30b --steps 4 --train-batch-size 4 --eval-size 16 --train-size 64 --probe-size 8 --max-tokens 32` | MintX reverse-KL path on the 30B MoE training stack. |
| `vla_sft_pi0_fast_libero` | PR 422 current-server SFT smoke | `openpi/pi0-fast-libero-low-mem-finetune` | 1 | `python scripts/wip/openpi_vla_smoke.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-dummy}" --model openpi/pi0-fast-libero-low-mem-finetune --skip-action --output-json results/merge-gate/<ts>/vla_sft_pi0_fast_libero.json` | VLA train-step path for FAST model on the active server endpoint. |
| `vla_sft_pi05_libero` | PR 422 current-server SFT smoke | `openpi/pi05-libero-low-mem-finetune` | 1 | `python scripts/wip/openpi_vla_smoke.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-dummy}" --model openpi/pi05-libero-low-mem-finetune --skip-action --output-json results/merge-gate/<ts>/vla_sft_pi05_libero.json` | VLA train-step path for pi0.5 model on the active server endpoint. |
| `vla_rl_pi0_fast_libero` | PR 422 current-server RL probe | `openpi/pi0-fast-libero-low-mem-finetune` | 2 | `python scripts/wip/openpi_libero_fast_group_rl.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --task-index 16 --steps 6 --groups-per-step 4 --group-size 4 --learning-rate 5e-6 --temperature 0.1 --resample-temperature-step 0.025 --max-group-resample-attempts 2 --min-accepted-groups-per-step 2 --eval-temperature 0.0 --training-state-path mint://rl-19b254e0a215_0/weights/group-rl-state-baseline --train-item-indices 0,4,7,10,2,6,3 --eval-item-indices 1,5,8,9 --output-dir results/merge-gate/<ts>/vla_rl_pi0_fast_libero` | FAST RL loop coverage on the active server endpoint. |

### Checkpoint and Resume

| id | provenance | model | gpu_required | runner | reveals |
|---|---|---|---:|---|---|
| `resume_training_dense_0p6b` | PR 315 contract generalized to dense | `Qwen/Qwen3-0.6B` | 1 | `TINKER_MODEL=Qwen/Qwen3-0.6B python scripts/tools/reproduce_issue_315_resume_training.py` | Exercises three resume surfaces explicitly: live-session `load_state(..., optimizer=true)`, fresh-session weights-only rollback to an older checkpoint, `create_model_from_state(..., load_optimizer=true)`, and a fresh `create_model` followed by `load_state(..., optimizer=true)`, then checks post-resume training continuity. |
| `resume_training_moe_30b` | PR 315, issue 283, and issue 404 follow-up | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | `TINKER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 python scripts/tools/reproduce_issue_315_resume_training.py` | Exercises the 30B Megatron resume family explicitly: live-session `load_state(..., optimizer=true)`, fresh-session weights-only rollback to an older checkpoint while the actor is on a newer checkpoint, `create_model_from_state(..., load_optimizer=true)`, and a fresh `create_model` plus `load_state(..., optimizer=true)`. This is the current merge-gate coverage for the illegal-CUDA same-actor resume family. |
| `vla_resume_pi0_fast_libero` | PR 422 current-server resume smoke | `openpi/pi0-fast-libero-low-mem-finetune` | 1 | `python scripts/wip/openpi_vla_resume_smoke.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-}" --model openpi/pi0-fast-libero-low-mem-finetune --output-json results/merge-gate/<ts>/vla_resume_pi0_fast_libero.json` | Save-resume continuity on the active server endpoint without external OpenPI Python package dependencies. |
| `vla_resume_pi05_libero` | PR 422 current-server resume smoke | `openpi/pi05-libero-low-mem-finetune` | 1 | `python scripts/wip/openpi_vla_resume_smoke.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-}" --model openpi/pi05-libero-low-mem-finetune --output-json results/merge-gate/<ts>/vla_resume_pi05_libero.json` | Save-resume continuity for pi0.5 model on the active server endpoint without external OpenPI Python package dependencies. |

### Concurrency and Isolation

| id | provenance | model | gpu_required | runner | reveals |
|---|---|---|---:|---|---|
| `session_switch_continuity_dense_0p6b` | issue 44 | `Qwen/Qwen3-0.6B` | 1 | `python scripts/tools/issue44.py concurrent --model Qwen/Qwen3-0.6B --steps 5 --batch-size 4 --learning-rate 1e-4` | Interleaved A/B dense training behaves like independent runs instead of corrupting session state. |
| `interleaved_loss_spike_dense_0p6b` | issues 193 and 194 | `Qwen/Qwen3-0.6B` | 1 | `python scripts/tools/reproduce_issue_193_194_high_load.py --model Qwen/Qwen3-0.6B --steps 8 --batch-size 8 --background-models 2 --output-dir results/merge-gate/<ts>/issue193_194_dense` | Same-session `forward_backward -> optim_step` ordering under load does not invert or trigger a loss spike. |
| `interleaved_loss_spike_moe_30b` | PR 350 built on issues 193 and 194 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | `python scripts/tools/reproduce_issue_193_194_high_load.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --steps 8 --batch-size 8 --background-models 2 --output-dir results/merge-gate/<ts>/issue193_194_moe` | Shared 30B Megatron trainer under session-switch load does not reproduce the old correlated loss-spike signature. |
| `transition_sequence_moe_30b` | PR 370 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 8 | `TINKER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 python scripts/tools/reproduce_issue_370_transition_sequence.py` | PR 370 transition sequence `A forward_backward -> B forward -> A optim_step` completes on a shared Megatron actor without retention-induced failure. |
| `transition_sequence_235b_max_context` | PR 370 heavy variant | `Qwen/Qwen3-235B-A22B-Instruct-2507` | 32 | `TINKER_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507 python scripts/tools/reproduce_issue_370_transition_sequence.py` | 235B max-context transition sequence `A forward_backward -> B forward -> A optim_step` does not OOM or fail on the shared Megatron path. |
| `vla_sampling_isolation_dual_fast` | current-server dual-train isolation probe | `openpi/pi0-fast-libero-low-mem-finetune` | 1 | `python scripts/wip/openpi_vla_dual_train_isolation.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-}" --model openpi/pi0-fast-libero-low-mem-finetune --steps-per-model 3 --output-json results/merge-gate/<ts>/vla_sampling_isolation_dual_fast.json` | Concurrent dual-session VLA train-step isolation path on the active server endpoint. |
| `vla_sampling_switch_pi0_fast` | current-server repeat-session churn probe | `openpi/pi0-fast-libero-low-mem-finetune` | 1 | `python scripts/wip/openpi_vla_dual_train_isolation.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-}" --model openpi/pi0-fast-libero-low-mem-finetune --steps-per-model 2 --output-json results/merge-gate/<ts>/vla_sampling_switch_pi0_fast.run1.json && python scripts/wip/openpi_vla_dual_train_isolation.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-}" --model openpi/pi0-fast-libero-low-mem-finetune --steps-per-model 2 --output-json results/merge-gate/<ts>/vla_sampling_switch_pi0_fast.run2.json` | Repeated dual-session churn probe on the same endpoint to detect cross-session regressions. |
| `vla_pressure_shared_openpi` | PR 422 mixed-client pressure run | `mixed OpenPI` | 2 | `python scripts/wip/openpi_vla_pressure_train_threads.py --base-url "${MINT_BASE_URL:-http://localhost:8000}" --api-key "${MINT_API_KEY:-}" --workers 6 --steps-per-worker 2 --output-json results/merge-gate/<ts>/vla_pressure_shared_openpi.json` | Concurrent VLA train-step pressure against the active server endpoint. |

### Server Control Plane

| id | provenance | model | gpu_required | runner | reveals |
|---|---|---|---:|---|---|
| `admission_control_backpressure` | current admission-control regression coverage | `N/A` | 0 | `python -m pytest .claude/skills/merge-gate/tests/test_admission_control.py::TestAdmissionControl::test_flood_rejected_429_no_capacity_leak_and_retry_works -v -s` | Oversized or flooded requests are rejected with stable capacity accounting and recovery after load drops. |
| `trainer_request_queueing_dense` | issue 230 style timeout avoidance | `Qwen/Qwen3-0.6B` | 1 | `python -m pytest .claude/skills/merge-gate/tests/test_admission_control.py::TestTrainerQueuing::test_competing_create_model_waits_dense_trainer -v -s` | Competing dense `create_model` requests queue instead of timing out and killing the actor. |
| `trainer_request_queueing_moe` | issue 230 plus scheduler and healthz hardening | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | `python -m pytest .claude/skills/merge-gate/tests/test_admission_control.py::TestTrainerQueuing::test_competing_create_model_waits_megatron_trainer -v -s` | Competing Megatron `create_model` requests queue instead of timing out and killing the actor. |
| `session_idle_cleanup_dense_0p6b` | issue 356 | `Qwen/Qwen3-0.6B` | 1 | `TINKER_MODEL=Qwen/Qwen3-0.6B python scripts/tools/reproduce_issue_356.py` | Idle training sessions are automatically cleaned up instead of persisting forever. Server precondition: dev server started with a short `MINT_TRAINING_INACTIVITY_TIMEOUT` such as `60`. |
| `rapid_session_creation` | existing stress coverage | `Qwen/Qwen3-0.6B` | 1 | `python -m pytest .claude/skills/merge-gate/tests/test_stress.py::TestStress::test_rapid_session_creation -v -s` | Fast create-model churn does not corrupt session management. |
| `actor_eviction_recycle` | existing eviction sentry | `mixed` | 16 | `python -m pytest .claude/skills/merge-gate/tests/test_stress.py::TestStress::test_mixed_model_lru_eviction -v -s` | Actor replacement and eviction remain observable and correct under pressure. |

### Opt-in Architecture

| id | provenance | model | gpu_required | runner | reveals |
|---|---|---|---:|---|---|
| `moonlight_sft` | existing Moonlight coverage | `moonshotai/Moonlight-16B-A3B-Instruct` | 8 | `python -m pytest .claude/skills/merge-gate/tests/test_moonlight_sft.py::TestMoonlight::test_moonlight_sft_training -v -s` | Moonlight supervised training path. Opt-in only. |
| `moonlight_lora_transfer` | existing Moonlight coverage | `moonshotai/Moonlight-16B-A3B-Instruct` | 8 | `python -m pytest .claude/skills/merge-gate/tests/test_moonlight_sft.py::TestMoonlight::test_moonlight_lora_transfer -v -s` | Moonlight train -> export -> sample path. Opt-in only. |
| `moonlight_rl` | existing Moonlight coverage | `moonshotai/Moonlight-16B-A3B-Instruct` | 8 | `python -m pytest .claude/skills/merge-gate/tests/test_moonlight_sft.py::TestMoonlight::test_moonlight_rl_smoke -v -s` | Moonlight RL smoke path. Opt-in only. |

## Catalog Discipline

Only items in the catalog above count as active merge-gate scenarios.

If a future idea has no runner yet:

- it is not part of the active catalog
- do not present it as selectable
- implement a runner first, then add it to the catalog

If a future idea already has a repro script elsewhere in the repo:

- do not treat that script alone as merge-gate coverage
- add a curated entry to the runnable catalog in this `SKILL.md`

## Preflight Procedure

Before any selected item runs:

1. Confirm dev targeting.
2. Confirm server health.
3. Confirm Unison is healthy.
4. Inspect current actor inventory.
5. Inspect current free and total GPUs.
6. Build a session manifest:
   - `selected`
   - `deselected_for_gpu`
   - `not_selected`

Suggested preflight evidence:

```bash
curl -s http://localhost:8000/api/v1/healthz
curl -s http://localhost:8000/api/v1/actors
```

If deeper GPU accounting is needed, use the `mint-dev` workflow and the cluster Python that matches the dev Ray runtime.

## Runner Guidance

The skill speaks in scenario ids, but execution may use either:

- a dedicated Python script under `scripts/tools/`
- a pytest integration target under `.claude/skills/merge-gate/tests/`
- a vendored real-loop script such as `.claude/skills/sanity-check/mint_rl_test_long.py`

Pytest-backed items are acceptable only because they hit the real dev server.

When a scenario runner exists in more than one form:

- prefer the most realistic real-loop runner
- prefer dedicated repro scripts over generic matrix scripts
- prefer generic scripts only when they are still concrete, runnable, and scenario-specific enough to reveal the intended behavior

For convergence-style items, the runner must leave behind enough information to inspect the learning trajectory, such as:

- plotted loss or reward curves
- per-step metrics CSV or JSON
- saved images referenced in the session report

If a convergence-style runner does not currently emit a curve, fix the runner before treating it as merge-gate evidence.

## Convergence Assessment

For any item whose main evidence is a training trajectory, such as SFT, DPO, SDPO, RL, resume continuity, or loss-spike regression:

- plot the curve
- inspect it visually
- describe the curve to the user in plain language

Minimum acceptable visual discussion:

- overall shape
- whether there are abrupt spikes, resets, plateaus, or divergence
- whether the post-resume or post-interleave segment looks continuous with the pre-event segment

Unacceptable judgments:

- "final loss is below some fixed fraction of initial loss"
- "the average is lower so it passed"
- "the threshold says this is good"

The correct question is whether the curve shape matches the intended behavior of the scenario.

## Classification Discipline

Every failure should end with one primary classification:

- `server_issue`
  - examples:
    - actor dies
    - queueing policy regresses
    - checkpoint resume diverges
    - max-context transition OOMs unexpectedly
- `test_issue`
  - examples:
    - stale dataset
    - wrong expectation
    - flaky oracle
    - runner assumes an old API contract
- `skill_issue`
  - examples:
    - wrong GPU requirement metadata
    - wrong sequencing
    - missing preflight invariant
    - stale catalog entry or wrong scenario provenance

Do not collapse these into a single "merge gate failed" statement.

## Reporting

At the end of a session, report:

- selected items
- deselected items and reasons
- items not selected
- observed free GPU budget
- per-item outcome
- per-item classification if failed
- what changed in the server or runner during the session
- what was rerun
- what remains unresolved

The report should read like an evidence summary, not a pass/fail verdict.

Minimum acceptable shape:

```markdown
- selected: rl_sanity_dense_0p6b, resume_training_moe_30b, admission_control_backpressure
- deselected_for_gpu: transition_sequence_235b_max_context (required=32, observed_free=16)
- rl_sanity_dense_0p6b: observed stable RL loop on dev; classification=ok
- resume_training_moe_30b: resumed loss spiked above pre-save baseline; classification=server_issue
- admission_control_backpressure: runner expected outdated internal payload field; classification=test_issue
- changes during session: patched server queue accounting; reran rl_sanity_dense_0p6b plus admission_control_backpressure
```

For convergence-style items, the report must also include a curve summary, for example:

```markdown
- resume_training_moe_30b: curve inspected visually. Post-resume loss stayed on the same descending trajectory as the pre-save segment; no reset-to-fresh-session spike was observed. Artifact: `results/merge-gate/<ts>/resume_training_moe_30b/...`
```

## Release Boundary

This skill is about validation only.

Tag creation, nightly publishing, and release PR mechanics are separate actions and should happen only when explicitly requested.

Do not imply that running merge-gate automatically creates a release artifact.

If the user explicitly asks for release mechanics after merge-gate work, keep the following workflow in this skill:

- resolve the real previous nightly tag
- review the actual code/content changes since that nightly
- summarize changes into a human-readable changelog
- create an annotated nightly tag named `nightly_YYYYMMDD`
- push the tag
- open a `develop -> main` PR with the summarized release body

Hard rules for that release flow:

- Do not create a tag or PR without explicit user confirmation for that step.
- Resolve the real previous nightly tag from the actual available tags. Never guess from one naming convention.
- Always check both nightly naming conventions: `nightly-*` and `nightly_*`.
- Prefer the newest real nightly by tag date / remote truth, not whatever one local grep happens to return first.
- Read the actual diff content before writing the tag message or PR body.
- Do not write filler like "included recent upstream changes", "various fixes", "multiple improvements", or any other umbrella phrase that avoids naming the substance.
- A raw commit list is not acceptable, and a hand-wavy summary is not acceptable.
- Every top-level bullet in the tag message and PR body must map to concrete inspected changes in the diff range.
- If a bullet cannot be backed by specific inspected changes, delete it.
- If merge-gate was skipped or only partially run, state that explicitly in the tag message and PR body.
- If validation found unresolved issues, do not hide them in the release summary.

Useful commands for the explicit release step:

```bash
git tag -l 'nightly-*' --sort=-creatordate
git tag -l 'nightly_*' --sort=-creatordate
git ls-remote --tags origin 'nightly*'
git log "$PREV_TAG"..HEAD --oneline --no-merges
git diff --stat "$PREV_TAG"..HEAD
```

Required release-summary method:

1. Resolve `PREV_TAG` from the real latest nightly tag after checking both naming schemes and remote tags.
2. Inspect all of:
   - `git log --oneline --no-merges "$PREV_TAG"..HEAD`
   - `git log --merges --oneline "$PREV_TAG"..HEAD`
   - `git diff --stat "$PREV_TAG"..HEAD`
   - targeted file diffs for every major changed area
3. Identify the real changed areas from the diff itself.
   Example categories:
   - training/runtime correctness
   - serving/routing
   - observability
   - developer/runbook/ops surface
4. Write the summary from those real changed areas.
   Each bullet must say what changed in plain language and why it matters.
5. If the diff is large, read more and write more.
   Diff size never justifies vagueness.
