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

- `9671cfb` `fix: make VLA startup deterministic`

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
- Fresh cold-start from zero is now deterministic under the documented runbook on the assigned worker and fresh namespace, and is captured in `.claude/skills/architecture-design/references/vla_deterministic_startup_runbook.md`.
- The fixed startup procedure is also captured as a server-side script in `scripts/wip/openpi_vla_start_server.sh`.
- Root cause of the startup regression: detached control-plane actors were hard-pinned to `node:__internal_head__`, and nested detached actors created from other actors did not inherit the control-plane pin because `actor_runtime_env_vars()` was not forwarding the relevant env vars.

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

Completed for RL-path validation with a grouped imitation-reward PPO run, plus additional rollout-grounded probes.

Primary RL artifact used for validation:

- `/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402/results/rl_pi0fast_grouped_object16_v2/summary.json`
- `/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402/results/rl_pi0fast_grouped_object16_v2/reward_curve.png`
- `/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402/results/rl_pi0fast_grouped_object16_v2/loss_curve.png`

Observed for the grouped imitation-reward run:

- task 16 `turn on the stove`
- steps `4`
- reward `-0.3205 -> -0.3151 -> -0.3104 -> -0.2951`
- loss stayed at `0.0` on these four PPO updates
- `create_model`, `save_weights_for_sampler`, `create_action_session`, `act`, `forward_logprobs`, and PPO `train_step` all completed on the repaired startup path

Additional rollout-grounded probes also exist:

- `/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402/results/rl_pi0fast_real_eval_task0_r/summary.json`
- `/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402/results/rl_pi0fast_rollout_shaped_object0_v5/summary.json`

Observed for the rollout-grounded path:

- real simulator rollout does run end to end
- sparse success on the tested task remained poor (`0/3`)
- rollout-grounded shaped reward works as a diagnostic path but did not yet become the final reported RL curve

Conclusion:

- the repaired server now supports a valid PPO-style RL update path with a non-degenerate reward curve
- the fastest useful RL evidence in this session is the grouped imitation-reward run
- the rollout-grounded path still exists, but it remains slower and less mature than the grouped validation path

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

### Architectural caveat

The current OpenPI sampling side is still not MinT-clean multi-tenant sampling. Full note: `.claude/skills/architecture-design/references/vla_sampling_architecture_gap.md`.

- Training is shared-actor multi-tenant.
- Sampling is still checkpoint-per-session and actor-per-action-session.
- That means sampling isolation is achieved by separate action actors and full sampler checkpoints, not by a shared sampler substrate multiplexing tenants in memory.
- This is exactly why pressure on the sampling side shows up as action-actor/GPU pressure.

So the current VLA implementation can work operationally, but the sampling architecture is still a real mismatch with the intended MinT design.

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

Detailed memo: `.claude/skills/architecture-design/references/vla_benchmark_demo_research.md`.

Recommended benchmark ladder:

- LIBERO first
- then LIBERO-plus
- then DROID
- then ALOHA as the main demo track
- then CALVIN
- then RLBench

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

Partially.

- `pi0-fast` SFT works
- `pi0.5` SFT works
- `pi0-fast` action-sampling plus PPO control path works
- the meaningful simulator RL baseline also runs, but the tested policy baseline is weak (`0/3` success on the tested LIBERO task)

So the RL stack works, but the original MSE-reward harness should not be treated as a meaningful learning result.

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

- PR branch head is `24dfc88`, on top of the earlier `9671cfb` startup fix and `8ab2078` VLA exploration fix
- deterministic startup from zero was re-verified under the scripted runbook on port `18125`
- focused local verification passed after the final changes: `75 passed` on startup/runtime/worker slices
- grouped RL artifact completed at `/vePFS-Mindverse/share/code/root/tinker-server-pr422-vla-20260402/results/rl_pi0fast_grouped_object16_v2/summary.json`
- current `mint-dev` actor state returned to one idle shared OpenPI fast trainer on `192.168.38.176` after grouped RL cleanup
