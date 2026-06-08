---
name: sanity-check
description: |
  Production sanity-check runner for MinT/mint-server.

  Objective: run non-trivial RL training loops against MinT production for the 5 production base models
  (0.6B, 4B Instruct, 4B Thinking, 30B, 235B), collect timing evidence, perform only minimal ops remediation when justified,
  and send exactly one final Feishu report.

  Triggers: "sanity check", "sanity-check", "prod sanity", "production sanity"

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# sanity-check

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the wrapper or procedure is missing an important operation, update the wrapper or skill first. Do not improvise a parallel flow.

Hard rules:
- Production only. The base URL for this workflow is exactly `https://mint.macaron.xin`.
- Refuse to run if the effective base URL is anything else, including localhost, SSH tunnels, dev ports, `https://mint.macaron.im`, or an internal host/port.
- No requirement substitution: do not use inference-only, reduced health checks, or ad-hoc smoke tests as PASS evidence.
- Ops only during a sanity-check run: do not make non-trivial product code changes as part of remediation.
- Minimize downtime: prefer killing one specific actor for one specific model over restarting the API process; do not restart Ray/head/worker nodes as first response.
- Do not use an ad-hoc "hung after N seconds" cutoff. Under load, 235B can queue for many minutes. Treat only the configured timeout or explicit service/server error as failure.
- No START notifications. Send exactly one final Feishu report at the end, including preflight failures.
- Never print secrets, process environments, or full `.secrets.env` contents.

## Canonical Runner

Run from `/root/code/mint`:

```bash
./scripts/wip/check.sh --all-models --timeout-s=7200
```

Use the canonical wrapper unless the wrapper itself fails before it can enforce the contract. Do not hand-roll the old shell/Python snippets in the agent response.

Wrapper contract:
- Loads `.secrets.env` without printing secrets.
- Forces `MINT_BASE_URL` to `https://mint.macaron.xin`.
- Requires `MINT_API_KEY`.
- Requires `MINT_TEST_CHECKPOINT_OWNER_ID` to be a 24-character production owner ObjectId.
- Runs the real RL loop through `.claude/skills/sanity-check/mint_rl_test_long.py`; it must not use inference-only mode.
- A model is `PASS` only if the runner completes all requested RL steps. Setup, final save, eval sampling, or `save_state` success after a skipped/failed RL step is still `FAIL`.
- Runs the full model matrix sequentially in the required order.
- Writes artifacts under `/root/run_results/mint/<timestamp>/`.
- Captures per-model stdout/stderr and timing artifacts.
- Recursively discovers nested `timing_events.jsonl`, `timing_summary.json`, and `timing_summary.md`.
- Writes `summary.json`, `summary.md`, and `final_feishu_report.md`.
- Sends exactly one final Feishu report for `--all-models`, including preflight failures.

## Model Matrix

The production matrix is exactly:

1. `Qwen/Qwen3-0.6B`
2. `Qwen/Qwen3-4B-Instruct-2507`
3. `Qwen/Qwen3-4B-Thinking-2507`
4. `Qwen/Qwen3-30B-A3B-Instruct-2507`
5. `Qwen/Qwen3-235B-A22B-Instruct-2507`

Do not skip, reorder, parallelize, or replace models for the scheduled production sanity-check.

## Artifacts

The wrapper owns artifact layout. Preserve the whole run directory after every run.

Evidence that must exist for a completed wrapper attempt:
- `summary.json`
- `summary.md`
- `final_feishu_report.md`
- per-model `stdout.log`
- per-model `stderr.log`
- timing files when the runner progressed far enough to emit them

The final external report must not include local paths, internal base URLs/ports, full transcripts, raw log dumps, process environments, or secrets.

## Skill Routing

This skill owns the scheduled production sanity-check flow: pass/fail decision, artifact preservation, final report, and rerun coordination.

Use other skills only for their bounded responsibilities:
- `mint-prod`: production API process, production checkout, production logs, production config, and production server restart only when evidence shows the API process itself is unhealthy.
- `mint-ops`: internal actor inventory, model-specific actor kill, scheduler/admission/deep health, and Ray diagnostics through Mint control-plane APIs.
- `telemetry-query`: request IDs, trace IDs, error text, endpoint failures, Victoria/Grafana signals, metrics, and narrow time-window evidence.
- `volcano-cluster`: GPU worker lifecycle, Volcano job/node state, placement-group cleanup, and worker node recovery. Do not run local `ray` or `volc` commands.
- `issue-reporter`: GitHub issue creation after evidence supports an implementation or production defect.

Do not use `mint-dev` for production sanity-check remediation.

## Failure Handling

If the wrapper reports any failure:

1. Preserve all artifacts from the current attempt.
2. Classify the failure as one of:
   - `client env/auth`: base URL, API key, auth, or checkpoint owner id.
   - `client workflow`: wrapper/script compatibility problems, SDK URI mismatch, local validation errors, or report-generation bugs.
   - `server health/control-plane`: HTTP health preflight 503 or a public control-plane dependency reported unhealthy before model execution.
   - `capacity/scheduling`: placement pending, actor not registered, queue not consumed, GPU/PG held by stale actor.
   - `server exception`: traceback, 5xx, explicit request failure, actor crash, `ActorDiedError`, `EngineDeadError`, CUDA OOM.
   - `timing degradation`: no hard failure, but a stage is materially slower than expected.
3. Inspect `summary.json`, `summary.md`, `stdout.log`, `stderr.log`, request IDs, and timing summaries before doing ops.
   - If the failure surface is `rl_step_not_completed`, inspect the preceding failed stage/request first, commonly `save_weights_for_sampler`, `create_sampling_client`, `sample`, `forward_backward`, or `optim_step`.
4. Use `telemetry-query`, `mint-prod`, `mint-ops`, and `volcano-cluster` as needed to gather narrow evidence. Do not assume `235B` is routed to a specific upstream; prove current routing from live config, capabilities, actor inventory, or logs.
5. Remediate only when evidence supports it, and use the smallest blast radius. Prefer a model-specific actor action through `mint-ops`; restart the API process only when the API process is unhealthy. Do not restart Ray/head/worker nodes in this workflow.
6. Record any remediation in `INCIDENT.md` under the run root: timestamp, model, symptom, request IDs, evidence, exact ops action, and observed result. Do not include secrets.
7. After remediation, rerun the canonical wrapper over the full five-model matrix from the beginning.
8. If the same failure persists after one justified remediation pass, stop doing ops and create an issue through `issue-reporter` when evidence supports an implementation or production defect.

## Final Report

The canonical wrapper owns the report format. The report must include:
- Overall `PASS`, `PASS_WITH_DEGRADATION`, or `FAIL` and passed-model count.
- One line per model with status, slowest stage or slowest completed stage, `max`, and `wall`.
- For passed models, generated-token throughput when timing artifacts contain it. Report sample and eval-sample end-to-end throughput separately (`sample_e2e_tok_s`, `eval_e2e_tok_s`) so it is not confused with vLLM decode-only throughput. Models that pass but exceed the configured timing-degradation threshold should be reported as `DEGRADED`.
- For failed models: failing surface/class, ops attempted, and whether a GitHub issue was created.
- A clear next action.

The wrapper sends the final report through `.claude/skills/sanity-check/feishu_notify.py`.

If the wrapper exits before sending a final report, send one manually with the same compact structure and mark the wrapper failure explicitly. If Feishu posting fails, the sanity-check workflow is failed; do not silently ignore it.
