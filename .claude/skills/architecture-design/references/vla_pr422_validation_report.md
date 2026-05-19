# VLA validation report

Date: 2026-04-08

## Environment used

- Ray head: `192.168.39.23`
- Assigned worker: `192.168.39.28`
- Dedicated code root: `/vePFS-Mindverse/share/code/root/mint-server-pr422-vla-20260402`
- Dedicated runtime env: `/vePFS-Mindverse/share/code/root/mint-runtime-py31213-openpi-pr422-20260402`
- Dedicated namespaces used during the clean latest runs include `mint_root_vla_pr422_20260408d` and `mint_root_vla_pr422_20260408f`

## Major fixes landed during validation

- Action temperature is now forwarded into action sampling in `mint_server/routes/action_sampling.py:20-30`.
- Action-session recovery across queued act paths is now worker-module-aware in `mint_server/backend/action_session_manager.py:96-122` and `mint_server/backend/action_session_manager.py:453-485`.
- Action-session state roots are now namespace-scoped in `mint_server/backend/openpi_ray_runtime.py:41`.
- FAST action decoding now fails loudly on malformed outputs in `mint_server/backend/openpi_fast_action_worker.py:239-338`.
- The grouped RL harness no longer aborts on a same-state group with zero within-group reward variance; zero-variance groups now contribute zero centered reward instead.
- Shared FAST action actor recreate drift on the tested deterministic paths was removed on the XLA-flags setup wired through `scripts/wip/openpi_vla_start_server.sh` and `mint_server/backend/openpi_ray_runtime.py`.

## Prompt task status

### 1. Dedicated troubleshooting env on `mint-dev`

Passed.

What is established:
- the dedicated vePFS code root, dedicated runtime env, dedicated Ray namespace, and assigned worker only were used for the current validated runs
- the VLA actors start and run on that dedicated path
- worker-local checkpoint/session-state disk exhaustion is not supported by the evidence gathered on the dedicated path

### 2. pi0-fast SFT experiment with a curve

Passed on one meaningful LIBERO task.

Artifacts:
- `results/sft_pi0fast_task16_full_k/summary.json`
- `results/sft_pi0fast_task16_full_k/loss_curve.png`

Observed:
- task 16: `turn on the stove`
- 12 steps
- loss `0.7759211196 -> 0.2169210192`
- minimum observed loss `0.2098036820`

### 3. pi0-fast RL experiment with a meaningful curve and metrics

Partial only.

Artifacts:
- `results/rl_pi0fast_grouped_samestate_batchstd_xla_lr1e5_t005_g4_6step_20260408d/run.log`
- `results/rl_pi0fast_grouped_samestate_batchstd_xla_lr1e5_t005_g4_6step_20260408d/metrics.jsonl`

Observed training reward trace:
- step 0 `-0.0023415950`
- step 1 `-0.0021082656`
- step 2 `-0.0021240816`
- step 3 `-0.0016613363`
- step 4 `-0.0016613363`
- step 5 `-0.0017247119`
- step 6 `-0.0018833488`

Observed PPO diagnostics at step 6:
- `loss = -7.554060882992214e-09`
- `loss_abs_mean = 0.8179361205548048`
- `ratio_mean = 1.0`
- `clipfrac_mean = 0.0`
- `post_update_ratio_mean = 0.9996794573962688`
- `post_update_clipfrac_mean = 0.0`

Current conclusion:
- the pi0-fast PPO path is no longer obviously mathematically broken
- the current 6-step train curve is still too short and too non-monotonic to count as a meaningful RL result

### 4. pi0.5 SFT experiment with a curve

Passed on one meaningful LIBERO task.

Artifacts:
- `results/sft_pi05_task10_full_k/summary.json`
- `results/sft_pi05_task10_full_k/loss_curve.png`

Observed:
- task 10: `put the bowl on the plate`
- 12 steps
- loss `0.1187784318 -> 0.0630739303`
- minimum observed loss `0.0606572889`

### 5. Concurrency state-isolation test

Partial only.

Positive evidence gathered:
- narrow mixed valid/invalid sampling isolation:
  - `results/dual_sampling_isolation_probe_20260406b.json`
- pi0-fast concurrent-task SFT traces:
  - `results/iso_fast_sft_task16_t/summary.json`
  - `results/iso_fast_sft_task17_t/summary.json`
- pi0-fast concurrent-task RL traces:
  - `results/iso_fast_rl_task18_t/summary.json`
  - `results/iso_fast_rl_task20_t/summary.json`
- pi0.5 concurrent-task SFT traces:
  - `results/iso_pi05_sft_task10_t/summary.json`
  - `results/iso_pi05_sft_task11_t/summary.json`
  - `results/iso_pi05_sft_task12_t/summary.json`

Current conclusion:
- concurrent clients on different tasks were run and produced distinct traces
- this is positive evidence, but it is still not a decisive proof that the full mixed-client matrix is contamination-free

### 6. Concurrency pressure test with 10 pi0-fast SFT, 10 pi0-fast RL, and 10 pi0.5 SFT clients

Partial only, not passed.

Artifact:
- `results/pressure_threads_final_503retry_k/batch_summary.json`

Observed:
- 30 logical clients launched
- 26 succeeded, 4 failed
- fast SFT: `10/10`
- pi0.5 SFT: `10/10`
- fast RL: `6/10`
- all 4 failures were fast-RL action-session creation failures
- failure logs:
  - `results/pressure_threads_final_503retry_k/pressure2_fast_rl_11_w.run.log`
  - `results/pressure_threads_final_503retry_k/pressure2_fast_rl_12_w.run.log`
  - `results/pressure_threads_final_503retry_k/pressure2_fast_rl_14_w.run.log`
  - `results/pressure_threads_final_503retry_k/pressure2_fast_rl_19_w.run.log`

Current conclusion:
- the pressure run exists and is useful evidence, but it does not pass because 4 of the 10 fast-RL clients still fail during action-session creation

### 7. Resume training test for pi0-fast and pi0.5

Passed on the tested tasks, with a caveat on the pi0-fast run cleanliness.

pi0-fast resume artifacts:
- `results/resume_pi0fast_task16_k/summary.json`
- `results/resume_pi0fast_task16_k/loss_curve.png`

Observed:
- presave loss `0.3178560336`
- first resumed loss `0.2805477182`
- final loss `0.2697689892`
- no immediate spike at the resume boundary
- caveat: the run log later shows a connection-refused failure after the useful continuity evidence was already produced

pi0.5 resume artifacts:
- `results/resume_pi05_task10_k/summary.json`
- `results/resume_pi05_task10_k/loss_curve.png`

Observed:
- presave loss `0.1067636572`
- first resumed loss `0.0972024137`
- final loss `0.0821148874`
- no immediate spike at the resume boundary

### 8. Merge-gate update

Passed.

What changed:
- `.claude/skills/merge-gate/SKILL.md` now includes runnable VLA catalog entries for:
  - pi0-fast SFT
  - pi0.5 SFT
  - pi0-fast short RL trace
  - pi0-fast resume
  - pi0.5 resume
  - dual sampling isolation
  - shared-action actor checkpoint switching
  - mixed OpenPI pressure run

### 9. Benchmark and demo research

Passed.

Artifact:
- `.claude/skills/architecture-design/references/vla_next_benchmarks_and_demos_20260408.md`

Current recommendation order:
1. LIBERO suite expansion
2. LIBERO-plus robustness sweep
3. MinT-hosted LIBERO policy-server demo
4. DROID no-robot serving demo
5. Meta-World adapter pilot
6. CALVIN after the stack has a stronger sequential-evaluation path

### 10. Final report

This file is the current repository-tracked validation report.

## Answers to the six verification areas

### 1. Does the OpenPI dependency install correctly to the worker runtime overlay?

Passed on the dedicated path used here.

### 2. Do all GPU workloads happen on the worker, not the API driver?

Passed on the tested VLA path.

### 3. Are request handling and queue/future reuse consistent with the existing MinT pattern?

Partial.

What is established:
- the path uses queues and futures
- real correctness bugs in temperature forwarding and detached action-session recovery were fixed

Current conclusion:
- the architecture is aligned, but the original implementation needed nontrivial fixes

### 4. Do pi0-fast SFT, pi0-fast RL, and pi0.5 SFT work as expected?

Partial.

Current conclusion:
- pi0-fast SFT: yes on the tested task 16 trace
- pi0.5 SFT: yes on the tested task 10 trace
- pi0-fast RL: still not at "works as expected"

### 5. Does multi-tenant training and sampling work as expected, without contamination, with correct actor sharing and concurrency handling?

Partial.

What is established:
- the earlier cross-namespace contamination bug was real and is fixed
- narrow mixed valid/invalid sampling isolation works
- tested shared-action actor checkpoint switching works on the validated deterministic pair
- lighter concurrent pi0-fast sampling can reuse one shared action actor

What is not established:
- broad mixed-client contamination freedom
- 30-client pressure success for all fast-RL clients
- general long-lived churn correctness across all checkpoint families

### 6. Are API contract docs clear and consistent, with concise client examples?

Not established in this pass.

## Bottom line

The current repository state supports these concrete claims:
- the dedicated OpenPI VLA path on the assigned worker is usable
- pi0-fast SFT and pi0.5 SFT each have real downward curves on tested LIBERO tasks
- tested save-resume continuity exists for pi0-fast and pi0.5 on the chosen tasks
- the pi0-fast PPO path is now numerically sane on the tested 6-step run
- narrow shared-action actor sampling correctness is established on a tested path

The current repository state does not yet support these stronger claims:
- meaningful pi0-fast RL
- decisive mixed-client contamination freedom
- fully passing 30-client pressure behavior
- completed benchmark/demo execution beyond the current LIBERO-oriented traces
