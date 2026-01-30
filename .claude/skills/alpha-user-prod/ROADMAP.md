# alpha-user-prod roadmap

Rule: every time the `alpha-user-prod` skill triggers, read this file first, pick one concrete task, then update this file at end of session.

## Current session task (pick exactly one)

- [ ] Run a prod capability scan via SDK (`get_server_capabilities`), record supported models, then choose one supported model and run a minimal sample-only workflow with 3 concurrent personas to validate request ordering and isolation.
- [ ] Create a new alpha-user demo from first principles (not cookbook-derived): define a realistic researcher/builder user story, implement it, run it twice, and write triage.
- [ ] Perform a missing-feature hunt: identify a plausible workflow that the SDK cannot express cleanly, then draft a feature request with a minimal API sketch.

## What has been tried (update every session)

- (empty) Add entries that point to `.claude/skills/alpha-user-prod/inventory/demos/<demo_id>/` plus the latest artifact bundle path.

## Backlog: researcher and builder workflows (add to this list as you learn)

Researcher-oriented:
- Algorithm surface exploration:
  - Implement one alternative RL variant via available loss fns and data shaping; document what the SDK makes easy vs hard.
- Preference learning and ranking:
  - Try a realistic preference data loop (collect pairs, train, evaluate) and identify missing primitives if blocked.
- Evaluation while training:
  - Periodic evaluation sampling while training continues; treat eval as part of the research loop, not a reliability check.
- Hyperparameter sweeps:
  - Run small sweeps (LR, rank, batch shape) and record what signals are observable via the SDK.

Builder-oriented:
- Agent application scenario:
  - Build a small agent loop that relies on tool-call formatting or structured output; train or refine it with MinT.
- Long-context application:
  - Build a retrieval or multi-document synthesis scenario; study failure modes under repeated training/sampling cycles.
- Multi-modal (only if supported by prod capabilities):
  - Attempt a VLM-shaped workflow; if blocked, propose the missing API surface as a feature request.

## Seeds (bootstrapping references)

See `.claude/skills/alpha-user-prod/SEEDS.md`.

## Triage rules (keep short; details live in SKILL.md)

- Do not label "server bug" based on "bad model quality".
- File a bug only for invariant/contract violations or systemic failures reproducible across reruns.
- For "missing feature", write a feature request with a user story and a minimal API sketch.
