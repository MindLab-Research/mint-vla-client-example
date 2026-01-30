---
name: alpha-user-prod
description: |
  Alpha-user exploration against MinT production (https://mint.macaron.im).

  Generates and runs new demos from the MinT/Tinker SDK contract (not limited to cookbook examples),
  maintains an internal roadmap + demo inventory (self-contained), triages failures (task vs client vs server vs missing feature),
  and files GitHub issues when evidence supports it.

  Triggers: "alpha user", "alpha-user", "prod battle test", "production battle-test", "continuous prod testing"
---

# alpha-user-prod

You behave like a curious, technically competent alpha user of the MinT SDK on production.

This is not a fixed test suite. It is an exploration loop: invent plausible research workflows from the SDK contract, run them, learn from outcomes, and evolve the roadmap.

## First-principles emphasis (do not become cookbook-bound)

The cookbook is a reference implementation, not the definition of what MinT can do.
The SDK documentation and exposed APIs are the contract.

Rules:
- Do not restrict ideas to what already exists in the cookbook.
- Start from a researcher/builder goal, then derive the required MinT API composition.
- Prefer novel compositions of existing primitives (training + sampling + saving/loading + async futures + evaluation hooks).
- If a workflow is strongly motivated but blocked by missing API surface, write a feature request instead of forcing it into an existing example shape.

## Hard constraints (non-negotiable)

- **Production is read-only.**
  - Do not use privileged/admin endpoints (kill/restart/actor management).
  - Read-only access is allowed:
    - You may SSH into the prod API host to tail logs (read-only) to speed up triage: `ssh mint-prod-volcano "tail -200 /tmp/tinker_server_auth.log"`.
    - Do not run supervisor restarts, kill endpoints, or any actor-management operations.
- **No secrets leakage.** Never print or dump `MINT_API_KEY`/`TINKER_API_KEY` (or any `*_KEY/*_TOKEN/*_SECRET/*_PASSWORD`).
- **No requirement substitution.**
  - Do not "prove" prod health by switching to dev.
  - Do not replace MinT runs with HuggingFace local inference/training as a substitute for MinT behavior.

## Workspace (must be maintained by this skill)

- Roadmap: `.claude/skills/alpha-user-prod/ROADMAP.md`
- Demo inventory (directory): `.claude/skills/alpha-user-prod/inventory/`
- Seed references: `.claude/skills/alpha-user-prod/SEEDS.md`
- Cookbook submodule: `.claude/skills/alpha-user-prod/tinker-cookbook/`

Artifacts:
- Store run artifacts under `results/alpha_user_prod/` (gitignored by repo root).
- Store throwaway demo scripts under `scripts/wip/alpha_user_prod/` (gitignored by repo root).

Production targeting:
- Base URL: `https://mint.macaron.im` (set `MINT_BASE_URL` explicitly for every run)
- Auth: `MINT_API_KEY` (required)

## Session protocol (run every time this skill triggers)

### 0) Load the roadmap, then choose one concrete goal

1. Read `.claude/skills/alpha-user-prod/ROADMAP.md`.
2. Pick exactly one "Current session task" (or create one if none exists).
3. State the chosen task as a single sentence with a concrete SDK primitive you will exercise.

Roadmap hygiene rules (prevent context loss):
- Keep tasks small and falsifiable (one workflow, one primary unknown).
- Every completed/blocked task must include a pointer to its artifact bundle path.
- If you discover new work, append it as a backlog item immediately (do not rely on memory).

### 1) Reconstruct the capability map (from SDK + server capabilities)

Goal: decide what is plausible to attempt today.

- Read the Tinker/MinT SDK reference (this repo packages it under the `tinker-official-reference` skill).
- Discover production-supported models via the SDK capability endpoint (preferred), then record the list in artifacts:
  - Prefer: `service_client.get_server_capabilities().supported_models`.
  - If unavailable: call the public models/capabilities endpoint exposed by the server API.

Never assume a model is supported without checking.

Default policy: only test the models that production advertises as supported.
Exception: if you have a strong user-story reason to support an additional model, write a feature request issue (do not silently test an unsupported model and mis-triage the failure).

### 2) Invent an alpha-user workflow (not a toy "test")

Invent a plausible user story that a researcher or builder would try next, given the capability map.

First-principles rule for ideation:
- Brainstorm candidate workflows from the SDK primitives first.
- Learn SDK call shapes from the `tinker-official-reference` skill (official contract).
- Consult the cookbook only after ideation for implementation idioms and common pitfalls.

Workflow emphasis:
- Researcher: explore algorithms, objectives, data shapes, and measurement.
- Builder: explore product-like agent/application scenarios that can be trained or refined with MinT.

Examples of workflow shapes (not a fixed menu):
- Researcher:
  - "I want to implement RL with a different advantage shaping and compare stability across losses."
  - "I want a preference-learning loop (collect pairs, train, evaluate), and to see what the SDK makes observable."
  - "I want to run small hyperparameter sweeps and inspect which metrics are available and trustworthy."
- Builder:
  - "I want to build an agent loop (tool-call format, structured outputs), then train a small adapter to reduce failure cases."
  - "I want long-context behavior in an application scenario (retrieval/synthesis) and to study drift under repeated updates."

Avoid "hard-coded oracle" thinking. ML outcomes are noisy. Use expectations + controls + invariants instead.

### 3) Define expectations and controls (required)

For every demo, write down three things before running:

1. **Contract invariants (hard):** what must hold if the system is functioning (futures resolve; values are finite; checkpoint paths reload; request ordering is respected; no cross-session leakage).
2. **Behavioral expectation (soft):** what qualitative trend or stability you expect from the workflow (not a numeric target).
3. **Controls (required):** reruns, no-op/baseline comparisons, and at least one perturbation meant to disambiguate "task/demo" vs "system".

A "soft expectation" failing is not a server bug by itself. Only treat it as a bug signal if controls indicate an impossible/systemic failure mode.

### 4) Implement the demo as a user would

- Write a runnable script under `scripts/wip/alpha_user_prod/<demo_id>.py`.
- Run it against production:
  - Base URL: `https://mint.macaron.im`
  - Auth: via `MINT_API_KEY` (or `TINKER_API_KEY` depending on the SDK alias used by the script).
- Prefer `asyncio` + async SDK methods for concurrency and for exercising futures ordering semantics.
- Emulate one or more "alpha users" as separate personas (independent sessions/clients) and interleave their requests.

### 5) Collect evidence and triage outcome

For each demo run, save an artifact bundle under `results/alpha_user_prod/<timestamp>/<demo_id>/` containing:
- `intent.md`: the user story, algorithm/scenario, invariants, expectations, and controls (written before execution).
- `run.jsonl`: request/response timeline (IDs, timings, key parameters, errors).
- `triage.md`: observed facts vs inferences, and which bucket the outcome belongs to:
  - demo/task issue
  - client-side implementation issue
  - server bug suspected
  - missing feature (SDK limitation / server capability gap)

### 6) Persist the demo intent + status in the skill inventory (required)

After the run:
- Create or update `.claude/skills/alpha-user-prod/inventory/demos/<demo_id>/`:
  - `INTENT.md` must be stable and reusable (not just a run diary).
  - `STATUS.md` must point to the latest artifact bundle and the next concrete action.
- Update `.claude/skills/alpha-user-prod/ROADMAP.md`:
  - Add the demo to "What has been tried" with a pointer to its inventory directory.
  - Mark the session task as complete/blocked.
  - Add follow-ups (especially disambiguating controls) if the result was ambiguous.
  - Add new ideas if the run revealed a new capability gap or a new promising workflow direction.

### 7) Auto-summarize and clean up the roadmap/inventory (required when they grow)

This skill is long-lived. Prevent context collapse by maintaining a compact structure:
- If `ROADMAP.md` grows large:
  - Merge repetitive tasks into one canonical task with variants listed.
  - Move old "What has been tried" entries into a short "Archived" section, keeping only demo_id pointers and 1-line outcomes.
- If `inventory/demos/` grows large:
  - Create thematic subdirectories (e.g. `training/`, `rl/`, `preference/`, `agents/`, `vlm/`).
  - Merge near-duplicates into one canonical demo and keep variants under `variants/`.
  - Keep each `STATUS.md` short: last verdict, last artifact pointer, next action.

## When to file issues (bug vs feature)

### Server bug issue (high bar)

File a server bug issue only when you have:
- A minimal reproduction demo script
- Reproducible failure signature across reruns
- Evidence pointing to contract/invariant violation or systemic failure (not "model quality")

When that bar is met: invoke the `issue-reporter` skill and file against the server repo with:
- Minimal repro command
- Artifact bundle path
- Observed vs inferred, explicitly separated

### Feature request issue

If you believe a workflow is strongly justified by the SDK's stated goals, but is blocked by the current SDK or server API:
- Write a feature request issue (also via `issue-reporter`).
- Include:
  - user story
  - why it is blocked today (concrete missing API surface)
  - a minimal proposed API shape (types and one example call)
  - expected failure modes / safety constraints

## Ideation via research (optional)

If you need new workflow ideas beyond your own:
- Use the `deep-research` skill to identify common LLM research/builder workflows.
- Convert each into a MinT-shaped user story (what primitives it needs), then add them to the roadmap backlog.
