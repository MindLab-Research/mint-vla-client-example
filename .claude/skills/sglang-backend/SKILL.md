---
name: sglang-backend
description: |
  SGLang sampling backend development and validation workflow for mint-server.

  Use for: validating a dev or issue-scoped Mint server configured with
  `serving_backend="sglang"`, checking SGLang overlay/runtime configuration,
  running SGLang-targeted RL sanity through `scripts/tools/sglang_train_check.py`,
  preserving evidence, and deciding which SGLang backend changes are relevant.

  Triggers: "sglang backend", "SGLang sanity", "sglang sanity",
  "sglang_train_check", "latest sglang env", "SGLang worker", "non-235B SGLang"

  Procedure contract: read this SKILL.md end-to-end before acting.
---

# SGLang Backend

This skill is for MinT SGLang backend bringup and dev/issue-scoped validation.
It is not the production sanity-check workflow.

Use related skills for bounded responsibilities:
- Use `mint-dev` for dev server topology, Ray Client constraints, and issue-scoped server lifecycle.
- Use `mint-ops` for internal actor inventory, actor kill, scheduler diagnostics, and deep health.
- Use `volcano-cluster` for dev worker lifecycle and GPU worker tasks.
- Use `sanity-check` only for scheduled production sanity on `https://mint.macaron.xin`.

## Hard Rules

- Do not run production sanity through this skill.
- Do not use `scripts/wip/check.sh` for SGLang feature validation. Use `scripts/tools/sglang_train_check.py`.
- Do not treat inference-only smoke tests, health checks, or actor readiness as PASS evidence for RL sanity.
- Do not dump full process environments, `.env` files, secret files, or `/proc/<pid>/environ` verbatim. Whitelist non-secret SGLang/config keys.
- Do not force models that are not advertised by the target server. If `4B-Thinking` is absent from live capabilities, record that and do not substitute it.
- Keep `235B` SGLang support/configuration unless the user explicitly asks to remove it. If there are not enough GPUs, verify support/config and skip runtime sanity for `235B`.
- If code changes affect the running server, restart the issue-scoped server before validating behavior. Python servers do not hot-reload.

## Scope Boundary

This skill validates a target server that is already configured for SGLang, or
helps bring up such a server in a dev/issue namespace. It should prove:

- the target API is reachable;
- the server advertises the intended models;
- SGLang runtime env points to the intended overlay/interpreter;
- the requested models are routed to SGLang;
- real RL loop sanity completes for the runnable SGLang model set;
- artifacts are preserved and interpreted consistently.

It does not own:

- production full-matrix sanity or Feishu reporting;
- generic Ray/Volcano lifecycle;
- broad architecture refactors unrelated to SGLang sampling;
- trial-and-error shrinking of model settings without explaining the failure.

## Preflight

Identify the target API URL and process before running sanity. For issue-scoped
local servers, use the explicit port supplied by the user or by the server log.

Minimum live checks:

```bash
curl -sS "$BASE_URL/api/v1/healthz"
curl -sS "$BASE_URL/api/v1/get_server_capabilities"
```

If the server process is local to the workstation, find the listener and start
time without dumping secrets:

```bash
ss -ltnp | rg ":${PORT}|State"
ps -p "$PID" -o pid,lstart,cmd --no-headers
```

Whitelist only these environment keys when verifying runtime config:

- `MINT_SGLANG_PY_EXECUTABLE`
- `MINT_SGLANG_PYTHONPATH`
- `MINT_SGLANG_MODEL_PLACEMENT_JSON`
- `MINT_SUPPORTED_MODELS`
- `MINT_MODEL_CONFIG_OVERRIDES_JSON`

Confirm SGLang version from the configured interpreter:

```bash
"$MINT_SGLANG_PY_EXECUTABLE" -c 'import sglang; print(sglang.__version__)'
```

For worker-specific tasks, prove placement from
`MINT_SGLANG_MODEL_PLACEMENT_JSON` or internal actor/scheduler state. Do not
assume SSH access to the GPU worker.

## Runnable Model Set

Use live capabilities as the source of truth:

1. Read `supported_models` from `/api/v1/get_server_capabilities`.
2. Intersect with the user-requested SGLang validation scope.
3. Exclude `Qwen/Qwen3-235B-A22B-Instruct-2507` unless the user explicitly asks to run it and enough GPUs are available.
4. Record any supported production model that is absent from the target server, for example `Qwen/Qwen3-4B-Thinking-2507`.

Do not call the result "all production models" unless the full production
matrix was actually advertised and run.

## RL Sanity Runner

Run from the mint-server checkout:

```bash
python scripts/tools/sglang_train_check.py \
  --base-url "$BASE_URL" \
  --skip-preflight \
  --run-name "$RUN_NAME" \
  <models...>
```

Use `--skip-preflight` only when you already performed explicit health and
capability checks. Use `--dry-run` first when changing the model set or runner
arguments.

Common issue-scoped defaults:

```bash
python scripts/tools/sglang_train_check.py \
  --base-url http://127.0.0.1:18141 \
  --run-name codex-worker3-non235b-$(date -u +%Y%m%d-%H%M%S) \
  0.6b 4b-instruct 30b
```

The runner writes artifacts under `/root/run_results/mint-sglang/<run-name>/`
by default and marks `target_backend=sglang` in its outputs.

For Qwen3 MoE models the runner defaults to `--no-train-unembed`; override only
when the test specifically requires unembedding LoRA coverage.

## Evidence To Preserve

A completed SGLang sanity attempt should have:

- `summary.json`
- `summary.md`
- `final_sglang_report.md`
- per-model `stdout.log`
- per-model `stderr.log`
- timing files when the RL runner progressed far enough to emit them

The final internal report should include:

- target URL and backend marker;
- SGLang interpreter path and version;
- live advertised models;
- which models were run and why any model was skipped;
- per-model status, wall time, slowest stage, generated tokens, and sample/eval throughput when present;
- `PASS`, `PASS_WITH_DEGRADATION`, or `FAIL`;
- any ops action taken, or "none".

Timing degradation is not a functional failure if every requested RL loop
completed with exit code 0. Report it explicitly as `PASS_WITH_DEGRADATION`.

## Failure Classification

Inspect artifacts before doing ops. Classify failures narrowly:

- `client workflow`: runner bug, SDK compatibility, bad flags, URI resolution.
- `target config`: model not advertised, backend not SGLang, wrong overlay, wrong placement JSON.
- `capacity/scheduling`: placement pending, GPU unavailable, actor not registered, queue not consumed.
- `server exception`: traceback, 5xx, actor crash, `ActorDiedError`, `EngineDeadError`, CUDA OOM.
- `timing degradation`: no hard failure, but wall/slowest stage exceeds thresholds.

If the failure surface is `rl_step_not_completed`, inspect the preceding failed
stage/request first: commonly `save_weights_for_sampler`,
`create_sampling_client`, `sample`, `forward_backward`, or `optim_step`.

Use the smallest justified remediation:

- SGLang actor/session issue: use `mint-ops` to inspect/kill only the affected actor.
- Dev worker lifecycle or GPU allocation issue: use `volcano-cluster`.
- Issue-scoped API code/config changed: restart only the issue-scoped server you own.

Do not restart Ray/head/worker nodes as the first response.

## Code And Config Areas

Relevant SGLang backend changes usually live in:

- `mint_server/backend/sampling_backend.py`
- `mint_server/backend/sglang_actor.py`
- `mint_server/backend/sglang_capabilities.py`
- `mint_server/backend/sglang_engine.py`
- `mint_server/backend/model_actor_launchers.py`
- `mint_server/backend/model_actor_supervisor.py`
- `mint_server/backend/model_actor_inventory.py`
- `mint_server/backend/model_registry.py`
- `mint_server/routes/sampling.py`
- `mint_server/routes/service.py`
- `mint_server/backend/session_manager.py`
- `mint_server/runtime_config.py`
- `mint_server/config.py`

Training/export compatibility for SGLang LoRA sanity may also require:

- `mint_server/backend/bumblebee_distributed.py`
- `mint_server/backend/megatron_distributed.py`
- `mint_server/backend/verl_training.py`

Keep `235B` SGLang model-registry support unless the user explicitly removes
the feature. Runtime validation can skip `235B` when capacity is insufficient.

## Cleanup Discipline

Keep code that is necessary for:

- selecting SGLang as a sampling backend;
- launching SGLang actors with the intended runtime env and placement;
- loading/unloading LoRA adapters through SGLang sessions;
- SGLang sample/logprob/top-k routing;
- Tinker SDK compatibility such as `/api/v1/client/config` when required by the runner;
- LoRA target-module export compatibility needed by SGLang sanity.

Revert or avoid unrelated/overdesigned code such as:

- broad global actor scans when a deterministic actor or placement-group name is sufficient;
- generic placement-group cleanup helpers not required by the failure;
- start-helper environment preservation unrelated to SGLang;
- offline log reclassification or summary rewriting when fresh runner artifacts are available;
- fallback LoRA export paths that silently produce a format SGLang cannot load.

When cleaning, never revert user changes blindly. Use `git status`, inspect
diffs, and keep unrelated work untouched.

## Validation Commands

For code-level changes, run the focused tests first:

```bash
.venv/bin/python -m py_compile scripts/tools/sglang_train_check.py
.venv/bin/python -m pytest tests/sanity_check/test_sglang_train_check.py -q
.venv/bin/python -m pytest \
  tests/test_sglang_sampling_backend_formalization.py \
  tests/test_sglang_base_sampling_backend.py \
  tests/test_service_client_config.py \
  tests/test_bumblebee_backend_selection.py \
  -q
git diff --check
```

If the live server was already validated, do not rerun expensive model sanity
just because comments or skill docs changed.
