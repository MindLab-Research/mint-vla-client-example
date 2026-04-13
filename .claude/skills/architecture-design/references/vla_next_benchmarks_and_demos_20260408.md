# VLA next benchmarks and demos

This note summarizes the most realistic next benchmarks and demos for the current MinT VLA stack, based on the OpenPI and LIBERO-oriented path that already runs in this repository.

## Source-grounded constraints

- OpenPI already treats LIBERO as the worked training, serving, and evaluation path for `pi0`, `pi0-fast`, and `pi0.5`, including policy-server evaluation flows. Source: https://github.com/Physical-Intelligence/openpi
- LIBERO is organized into four transfer-oriented suites over 130 tasks, so it is already a benchmark family rather than a single task list. Sources: https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/overview.html and https://github.com/Lifelong-Robot-Learning/LIBERO
- LeRobot already exposes benchmark wrappers for LIBERO, Meta-World, and IsaacLab Arena. That makes Meta-World a much more realistic next simulator target than starting from an unintegrated benchmark stack. Sources: https://huggingface.co/docs/lerobot/main/adding_benchmarks and https://huggingface.co/docs/lerobot/metaworld
- LIBERO-plus is explicitly a robustness-oriented extension of LIBERO, not a different benchmark family, which makes it the lowest-friction robustness expansion after the current LIBERO traces. Source: https://github.com/sylvestf/LIBERO-plus
- CALVIN is an authoritative long-horizon language-conditioned manipulation benchmark, but its value comes from sequential instruction evaluation rather than single-step action serving. Source: https://github.com/mees/calvin
- RLBench is an established simulator benchmark with wide task coverage, but its setup cost is materially higher because it requires CoppeliaSim and headless display handling. Source: https://github.com/stepjam/RLBench
- DROID and BridgeData V2 are both realistic real-world data sources for offline or no-robot VLA demos. Sources: https://droid-dataset.github.io/ and https://bridgedata-v2.github.io/

## Benchmarks we should run next

### 1. LIBERO suite expansion

This is the first benchmark that should be executed next.

Why it fits:
- It stays inside the current OpenPI and MinT VLA contract.
- It upgrades current evidence from a few task-specific traces to suite-level benchmark evidence.
- It minimizes adapter work because the current stack is already LIBERO-shaped.

What to run:
- Expand beyond the currently exercised tasks into the official LIBERO suite structure:
  - LIBERO-Spatial
  - LIBERO-Object
  - LIBERO-Goal
  - LIBERO-100

Why this should come first:
- It gives the fastest path from current bring-up artifacts to persuasive benchmark evidence.

### 2. LIBERO-plus robustness sweep

This is the second benchmark that should be executed next.

Why it fits:
- It is close to drop-in relative to LIBERO.
- It tests robustness rather than only nominal task success.
- It is a better next-step stressor than jumping immediately to a completely different simulator stack.

Why it is valuable:
- A MinT-served VLA policy that only works on nominal LIBERO scenes is much weaker evidence than one that survives LIBERO-plus perturbations.

## Next-tier benchmarks after LIBERO and LIBERO-plus

### Meta-World via LeRobot

Why it is attractive:
- LeRobot already has a benchmark wrapper for it.
- It is the cleanest next simulator expansion after LIBERO.

What blocks it:
- Action-space and interface adaptation are real work.
- It is not just another dataset swap.

### CALVIN

Why it is attractive:
- It tests long-horizon compositional control, which is a stronger story than single-instruction tabletop tasks.

What blocks it:
- The current MinT VLA path needs a better sequential-evaluation story first.

### RLBench

Why it is lower priority:
- High simulator and display-management friction.
- It is not the fastest path to the next persuasive benchmark result.

## Demo and POC priorities

### 1. MinT-hosted LIBERO policy-server demo

This is the most realistic immediate demo.

What it should show:
- train a policy through MinT VLA
- serve it through the MinT/OpenPI path
- evaluate it on LIBERO with rollout videos and success tables

Why it matters:
- It directly demonstrates the current architecture instead of only producing offline curves.

### 2. LIBERO-plus robustness demo

What it should show:
- the same MinT-served checkpoint under perturbations
- side-by-side nominal versus perturbed rollouts

Why it matters:
- It says something concrete about brittleness, not just basic task execution.

### 3. DROID no-robot serving demo

What it should show:
- MinT serving a DROID-style checkpoint or DROID-style observation stream without hardware

Why it matters:
- It broadens the story beyond LIBERO without requiring physical robots.

### 4. BridgeData V2 conversion smoke demo

What it should show:
- a small BridgeData V2 slice converted into the current training path
- short SFT and inference smoke evidence

Why it matters:
- It tests whether MinT VLA can ingest a broader real-world tabletop data source.

### 5. Meta-World small adapter demo

What it should show:
- a small task subset through a MinT-to-Meta-World action adapter

Why it matters:
- It is the most realistic second-simulator demo, but it is not the fastest one.

## Recommended execution order

1. LIBERO suite expansion
2. LIBERO-plus robustness sweep
3. MinT-hosted LIBERO policy-server demo
4. DROID no-robot serving demo
5. Meta-World small adapter pilot
6. CALVIN after the stack has a better sequential-evaluation story

## Current implication

The correct current status is:
- enough evidence exists to choose the next benchmark and demo work
- the next benchmark should stay inside the LIBERO family first
- the next persuasive demo should be a MinT-hosted LIBERO evaluation demo
- broader simulator expansion should come after that, not before
