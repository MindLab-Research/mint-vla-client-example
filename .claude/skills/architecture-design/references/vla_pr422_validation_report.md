# PR 422 VLA Validation Report

Date: 2026-04-03

## Scope

This report covers the end-to-end validation requested in `PROMPT.md` for PR 422 on `mint-dev`, using:

- dedicated code root `/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402`
- dedicated env `/vePFS-Mindverse/share/code/root/mint-runtime-py31213-openpi-pr422-20260402`
- dedicated Ray namespace `tinker_root_vla_pr422_20260402t`
- assigned worker `192.168.38.176` only

The work includes runtime bringup, SFT and RL experiments, resume testing, interleaving and pressure testing, merge-gate updates, and a final answer to the six requested verification areas.

## Branch Changes

Validation and follow-up fixes landed on `origin/vla-openpi-merge-develop` in these commits:

- `151d0cc` `test: add VLA merge-gate runners`
- `6a82cab` `fix: retry VLA backpressure on action sessions`
- `e23e6c4` `test: cover VLA action-session backpressure`
- `c6c54c5` `fix: extend VLA action-session wait budget`

These sit on top of the earlier PR-422 stabilization commits:

- `67fd539` `fix: stabilize openpi fast action runtime`
- `d6ca331` `fix: harden vla control plane and action worker`
- `dd3b43e` `fix: optimize openpi sampler export path`

## Results By Prompt Task

### 1. Dedicated troubleshooting env

Completed.

- OpenPI worker/runtime overlay started successfully from the dedicated env.
- OpenPI actors were pinned to `192.168.38.176`.
- No production/default py31213 env changes were used.
- GPU work ran on the worker, not on the API driver.

### 2. pi0-fast SFT

Completed.

Artifact:

- `results/sft_pi0fast_task16_full_k/summary.json`

Observed:

- task 16 `turn on the stove`
- loss `0.7759 -> 0.2169`
- minimum observed loss `0.2098`

Conclusion:

- `pi0-fast` SFT converged on a real LIBERO task.

### 3. pi0-fast RL

Completed.

Artifact:

- `results/rl_pi0fast_task16_full_t/summary.json`

Observed:

- task 16 `turn on the stove`
- full action-session, sampler-export, act, forward_backward, and PPO-style update path completed
- reward and loss curves were emitted

Conclusion:

- real FAST RL path works end to end on the assigned worker.

### 4. pi0.5 SFT

Completed.

Artifact:

- `results/sft_pi05_task10_full_k/summary.json`

Observed:

- task 10 `put the bowl on the plate`
- loss `0.1188 -> 0.0631`
- minimum observed loss `0.0607`

Conclusion:

- `pi0.5` SFT converged on a real LIBERO task.

### 5. Concurrency state isolation test

Completed.

Workload:

- `2` `pi0-fast` SFT clients
- `2` `pi0-fast` RL clients
- `3` `pi0.5` SFT clients

Observed:

- one shared FAST OpenPI trainer actor
- one shared pi0.5 OpenPI trainer actor
- separate FAST action actors for sampling
- no duplicate trainer actors
- no evidence of cross-tenant trainer-state contamination

Conclusion:

- interleaved training and sampling behaved as intended on the shared-trainer architecture.

### 6. Concurrency pressure test

Completed.

Important note:

- the naive `30`-process client launch was not a valid server pressure test because the API-host clients were OOM-killed before reaching MinT
- the valid pressure result is from the threaded logical-client harness `scripts/wip/openpi_vla_pressure_threads.py`

Final artifact:

- `results/pressure_threads_timeout3600_remote_p/batch_summary.json`

Final observed result:

- `count=30`
- `ok=30`
- `failed=0`

Final actor state after completion:

- exactly two idle shared OpenPI trainer actors remained
- no lingering action actors
- no duplicate trainers

Conclusion:

- under the validated server-side pressure harness, large logical-client concurrency completed successfully without duplicate trainer creation or server OOM.

### 7. Resume training test

Completed.

Observed:

- `pi0-fast` crossed the save/resume boundary without an immediate loss spike
- `pi0.5` crossed the save/resume boundary without an immediate loss spike

Conclusion:

- both resume paths are healthy for the tested boundary case.

### 8. Merge-gate update

Completed.

Added to `.claude/skills/merge-gate/SKILL.md`:

- `vla_sft_pi0_fast_libero`
- `vla_sft_pi05_libero`
- `vla_rl_pi0_fast_libero`
- `vla_resume_pi0_fast_libero`
- `vla_resume_pi05_libero`
- `vla_pressure_shared_openpi`

### 9. Research on next step

Completed.

Recommended benchmark ladder:

- LIBERO first
- then VLA-Arena OpenPI or LeRobot variant
- then CALVIN and RLBench
- use OXE, BridgeData V2, and DROID as transfer-data validation rather than primary benchmarks

## Answers To The Six Verification Areas

### 1. Does the openpi dependency install correctly to worker runtime overlay?

Yes.

The dedicated env and runtime overlay loaded OpenPI and its worker stack on `192.168.38.176`.

### 2. Do all GPU workloads happen on worker, not API driver?

Yes.

Training and action workloads ran on the worker. The API driver handled HTTP and control-plane logic only.

### 3. Are the request handling paths consistent with LLMs and do they reuse queues and futures?

Yes.

The validated VLA routes run through the MintX surface and reuse queueing and future retrieval rather than bypassing the control plane.

### 4. Do pi0-fast SFT, pi0-fast RL, and pi0.5 SFT all work as expected?

Yes.

All three completed on real LIBERO tasks with concrete artifacts and metrics.

### 5. Does multi-tenant interleaved training and sampling work as expected?

Yes, within the validated harness.

- shared trainers were reused correctly
- FAST action actors scaled separately
- trainer duplication was not observed
- final pressure run completed `30/30` logical clients successfully

The important scope note is that the final `30/30` result is for the threaded logical-client harness, not the invalid naive `30`-process client launch.

### 6. Are API docs and client examples clear and consistent?

Yes, after the branch updates.

- MintX docs were synchronized earlier
- repo-local VLA runners now serve as concise concrete examples
- merge-gate catalog entries point at those runners directly

## Final State

At the end of validation:

- PR branch head is `c6c54c5`
- `tests/test_mint_routes.py` passed locally
- `mint-dev` returned to two idle shared OpenPI trainer actors on `192.168.38.176`
- no lingering action actors remained
