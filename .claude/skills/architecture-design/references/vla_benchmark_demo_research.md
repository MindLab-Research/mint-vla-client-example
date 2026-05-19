# VLA Benchmark And Demo Research

Date: 2026-04-04

This historical memo is the detailed benchmark and demo follow-up for PR 422 VLA
work as of 2026-04-04.

## Recommendation summary

Recommended order:

1. `LIBERO-plus`
2. `DROID`
3. `ALOHA`
4. `CALVIN`
5. `RLBench`

Why this order:

- `LIBERO-plus` is the cheapest next robustness step because it stays close to the current LIBERO/OpenPI path while adding perturbation stress.
- `DROID` is the best next benchmark for real-world distribution shift that still fits the current OpenPI stack.
- `ALOHA` is the strongest visible demo track.
- `CALVIN` is the next serious long-horizon language-conditioned simulator benchmark.
- `RLBench` is broad and useful, but heavier than the others and less aligned with the current near-term stack.

## Benchmark ladder

### 1. LIBERO-plus

Primary source:

- `https://github.com/sylvestf/LIBERO-plus`

Why it matters:

- It is the fastest way to tell whether current performance is real robustness or just fit to base LIBERO.
- It preserves the same broad task family and simulator assumptions, so comparisons against current LIBERO runs are meaningful.

Task modality:

- language-conditioned tabletop manipulation in simulation
- perturbation-oriented evaluation on top of the LIBERO family

Requirements:

- existing LIBERO stack
- LIBERO-plus assets/config
- same OpenPI/MinT evaluation path with modest integration work

Engineering difficulty:

- low to medium

Recommended success metric:

- average success by perturbation family
- delta versus base LIBERO success

### 2. DROID

Primary sources:

- `https://droid-dataset.github.io/`
- `https://github.com/droid-dataset/droid_policy_learning`
- `https://github.com/Physical-Intelligence/openpi`

Why it matters:

- It moves evaluation from curated simulator data toward real-world variation.
- OpenPI already has DROID support, so this is not a speculative benchmark choice.

Task modality:

- real-robot tabletop manipulation across many scenes and households

Requirements:

- DROID dataset / RLDS pipeline
- OpenPI DROID-compatible data path
- real robot stack if the goal includes live evaluation rather than offline or replay-based validation

Engineering difficulty:

- medium to high

Recommended success metric:

- held-out scene success rate
- first-attempt success
- robustness across scene/object variation

### 3. ALOHA

Primary sources:

- `https://tonyzhaozh.github.io/aloha/`
- `https://github.com/Physical-Intelligence/openpi/tree/main/examples/aloha_real`

Why it matters:

- It is the most compelling visible demo family in the current OpenPI ecosystem.
- It is easier to communicate than a pure benchmark table.

Task modality:

- bimanual real or simulated manipulation

Requirements:

- ALOHA hardware for real demos, or simulator path for iteration
- OpenPI example stack

Engineering difficulty:

- medium to high

Recommended success metric:

- per-task success rate
- retries per task
- cross-task generalization under the same policy family

### 4. CALVIN

Primary source:

- `https://github.com/mees/calvin`

Why it matters:

- It tests sustained multi-step language-conditioned control rather than isolated short-horizon tasks.

Task modality:

- long-horizon language-conditioned manipulation in simulation

Requirements:

- CALVIN environment
- CALVIN dataset and eval protocol

Engineering difficulty:

- high

Recommended success metric:

- sequence success rate
- average completed instruction-chain length

### 5. RLBench

Primary source:

- `https://github.com/stepjam/RLBench`

Why it matters:

- Broad task coverage
- useful once the current stack is stable enough to justify a heavier simulator integration

Task modality:

- diverse simulated robotic manipulation benchmark

Requirements:

- CoppeliaSim
- PyRep
- headless graphics and simulator integration work

Engineering difficulty:

- high to very high

Recommended success metric:

- task-family success
- generalization across task breadth

## Demo / POC candidates

### ALOHA multi-task reel

Candidate story:

- one model family performing several visually distinct manipulation tasks back to back

Why it is compelling:

- easy for non-specialists to understand
- directly visible task transfer story

Needed stack:

- OpenPI ALOHA real or sim example code
- camera and robot stack if using physical hardware

Best metric to show:

- task success rate
- retries per task

### DROID clutter generalization demo

Candidate story:

- the same policy handling cluttered tabletop scenes across different layouts and objects

Why it is compelling:

- it demonstrates real-world messiness instead of clean benchmark scenes

Needed stack:

- DROID evaluation/data pipeline
- OpenPI DROID path

Best metric to show:

- held-out scene success
- first-attempt success

### LIBERO-plus robustness wall

Candidate story:

- side-by-side comparison across perturbation categories instead of only clean baseline scenes

Why it is compelling:

- failures cluster visibly by perturbation type
- easy to compare against current base LIBERO runs

Needed stack:

- LIBERO-plus benchmark assets and evaluation path

Best metric to show:

- success by perturbation family
- delta versus base LIBERO

### CALVIN long-horizon chain demo

Candidate story:

- one uninterrupted episode completing a chain of instructions

Why it is compelling:

- shows continuity and long-horizon planning rather than single-step competence

Needed stack:

- CALVIN environment and evaluation path

Best metric to show:

- chain completion length
- sequence success rate

## Practical next-step recommendation

The next benchmark after the current LIBERO work should be `LIBERO-plus`, not a jump directly to a heavier simulator or a real-robot stack.

The next real-world benchmark should be `DROID`.

The next visible demo should be `ALOHA`.

`CALVIN` should be the next harder simulator benchmark once short-horizon behavior is stable.

`RLBench` should stay later in the queue unless breadth becomes the primary objective.
