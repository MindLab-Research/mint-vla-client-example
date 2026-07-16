---
name: mint-vla-openpi-finetune
description: |
  LoRA fine-tune an OpenPI pi0.5 VLA model from a Lance dataset path, over a Ray-free mint-server.

  Use for: given a Lance dataset (image/wrist_image/state/actions/prompt schema),
  drive the full create_model -> train_step*N -> save_weights_for_sampler ->
  [inference check] -> [MSE evaluation] -> [merged-Lance inference dump] ->
  cleanup chain, asking the user for training knobs (steps, batch size,
  checkpoint name, LoRA rank, optional evaluation steps) instead of guessing
  them.

  Triggers: "vla finetune", "openpi finetune", "pi0.5 lora", "微调pi0.5",
  "lance数据集微调", "openpi lora微调", "mse评估", "推理写回lance"

  Scope: pi0.5 (flow_action policy_family) only, not pi0-fast. Only handles
  Lance datasets already in the image/wrist_image/state/actions/prompt schema
  (MANO/raw-capture -> rendered-image conversion is out of scope; see
  references/pipeline_reference.md section 6.1).

  Procedure contract: read this SKILL.md end-to-end before acting. Read
  references/pipeline_reference.md BEFORE touching the driver script or
  debugging a failure -- it documents real bugs already hit and fixed
  (LoRA rank constraint, model_id extraction, no-Ray local-Ray side effect,
  OOM root cause) so you don't re-discover them the slow way.
---

# mint-vla-openpi-finetune

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Read `references/pipeline_reference.md` before modifying the driver script
  or debugging any failure. It documents real, previously-hit bugs and server
  constraints discovered while building this skill. Do not re-derive them
  from scratch.
- If the procedure is missing something important, update the skill. Do not
  improvise around the gap.

## Overview

This skill fine-tunes the OpenPI pi0.5 VLA model (LoRA) from a Lance dataset,
against a Ray-free mint-server, and reports the result. The driver script is
`scripts/tools/openpi_vla_lora_finetune.py`, which reuses the verified
dataset/transform/HTTP plumbing from `scripts/wip/openpi_vla_smoke_lance.py`.

**Hard invariant**: the Lance dataset's per-frame `state`/`actions` vector
length must exactly equal the target model's `action_dim` in
`mint_server/backend/core/model_registry.py` (currently `32` for
`openpi/pi05-libero-low-mem-finetune`). Zero-padding or masking a mismatch is
a documented dead end -- see `ActionHeadSummary.md` in the repo root. The
driver script enforces this with a hard stop (`validate_action_dim`) before
any expensive work; do not try to work around it by padding data.

**Scope**: only `openpi/pi05-libero-low-mem-finetune` (or any future
`training_backend="openpi_pi05"` model) is supported. `pi0-fast` models use a
different policy family (`ar_action_tokens`) and a different driver
(`openpi_libero_fast_rl.py` family) -- not covered by this skill.

## Inputs required from the user

Before invoking the driver, ask the user (one batched `AskUserQuestion` round,
not one question per turn) for anything not already stated in the
conversation. Do not silently default these -- state the suggested default
and let the user confirm or override:

| Question | Suggested default | Source of default |
|---|---|---|
| Which Lance dataset path? | (required, no default) | user must provide |
| Training steps | 400 | `PI05lance_local_norray.sh`'s `MINT_PI05_STEPS` default |
| Batch size | 2 | same script's `MINT_PI05_BATCH` default |
| Checkpoint save name | `<none>` -> auto-generated `vla_lora_sampler_<hex8>` | driver's own fallback if omitted |
| Run inference smoke check after save? | yes | cheap correctness signal (single `act()` call, not eval) |
| Base model | `openpi/pi05-libero-low-mem-finetune` | only openpi_pi05-backend model currently in `MODEL_CONFIGS` |
| LoRA rank | 16 | the only value the server currently accepts (see below) |
| Run quantitative MSE evaluation after save? | no (opt-in) | extra network round-trips over `--eval-mse-indices`; skip for quick iteration |
| Write predictions back into a merged Lance dataset? | no (opt-in) | infers over the FULL dataset, can be slow; only ask if the user wants a file to inspect/replay |

**On LoRA rank**: mention the default (16) and that, as of this writing, the
server hard-rejects any other value plus any `train_attn`/`train_mlp`/
`train_unembed` != True (see `references/pipeline_reference.md` section 2.1).
It is fine to let the user request a different value -- the driver no longer
blocks it locally, it just warns and sends whatever was asked for; the
server's own rejection (with its exact error text) is the source of truth on
whether the constraint still holds. Do not claim it will fail as a certainty
if the user specifically wants to test whether the constraint has been
relaxed -- say it currently fails, per the last verification, and let them
try.

If the user already stated steps/batch-size/checkpoint-name earlier in the
conversation, skip re-asking for just those fields.

## Procedure

1. **Resolve and validate `--base-model`.** Confirm it exists in
   `MODEL_CONFIGS` with `training_backend == "openpi_pi05"`. If the user
   didn't specify one, use the default.

2. **Dry-run first.** Before starting any server or touching GPUs, run:

   ```bash
   GRB=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/gpu_rl
   EXTRA_PYDEPS=/vePFS-Mindverse/user/intern/wenxi/mint_env/extra-pydeps
   export PYTHONPATH="<repo_root>:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
   "${GRB}/host-venv/bin/python" scripts/tools/openpi_vla_lora_finetune.py \
     --lance-dataset <path> --dry-run
   ```

   This runs `probe_lance_dataset()` (readability probe, with automatic
   version-fallback suggestion on failure) and `validate_action_dim()` (hard
   stop on dimension mismatch) with zero network calls. If this fails, stop
   and report the exact error to the user -- do not try to route around an
   action_dim mismatch or an unreadable dataset version by modifying data.

3. **Ask remaining questions** (see table above) if not already answered.

4. **Decide fresh-vs-reuse server.** Default: start a fresh Ray-free server
   for this run via `scripts/vla/PI05lance_local_norray.sh`'s server-startup
   pattern (see below) on an unused port -- do not hardcode port 30510 if
   another server might already be using it; probe first or ask the user.
   Only reuse an already-running server if the user explicitly says one is
   up on a known port, and in that case verify its `MINT_SUPPORTED_MODELS`
   includes the requested `--base-model` before proceeding.

5. **Start the server** (fresh case). Reuse the exact environment variables
   from `scripts/vla/PI05lance_local_norray.sh` (do not invent new ones):

   ```bash
   export MINT_PORT=<port> MINT_HOST=0.0.0.0
   export MINT_UVICORN_WORKERS=1 MINT_SKIP_SUPERVISOR=1 MINT_ALLOW_NO_RAY=1 MINT_USAGE_BACKEND=disabled
   export MINT_RAY_NAMESPACE=<unique namespace>
   export MINT_SUPPORTED_MODELS="<base_model>"
   export OPENPI_DATA_HOME=/vePFS-Mindverse/share/models/openpi
   export HF_HOME=/vePFS-Mindverse/share/huggingface
   export MINT_OPENPI_PI05_CHECKPOINT_BASE_DIR=/vePFS-Mindverse/share/mint/dev/data/wenxi/openpi-pi05-checkpoints
   export MINT_OPENPI_PI05_ASSETS_BASE_DIR=/vePFS-Mindverse/share/code/conley/openpi/assets
   export MINT_OPENPI_PI05_WEIGHTS_PATH=/vePFS-Mindverse/share/models/openpi/pi05_base/params
   export MINT_OPENPI_PI05_ACTION_DIRECT_RUNTIME=1
   export MINT_RUNTIME_CHECKPOINT_DIR=/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints
   export XLA_FLAGS="--xla_gpu_enable_command_buffer="
   export CUDA_VISIBLE_DEVICES=3,4,5,6   # pin to idle cards on shared GPU box; override if needed
   export PYTHONPATH="<repo_root>:${EXTRA_PYDEPS}:${GRB}/site-packages:${GRB}/src/openpi/src:${GRB}/src/openpi/packages/openpi-client/src"
   nohup "${GRB}/host-venv/bin/python" -u <repo_root>/scripts/wip/_run_local_openpi_server.py > <server_log> 2>&1 &
   ```

   Wait for `GET /api/v1/healthz` to return `200` or `503` (both mean ready;
   `503`/"unhealthy" is the **expected** degraded-state marker in Ray-free
   mode, not a failure -- see `references/pipeline_reference.md` section 3).

6. **Run the driver:**

   ```bash
   MINT_BASE_URL="http://localhost:<port>" MINT_API_KEY=tml-dummy TINKER_API_KEY=tml-dummy JAX_PLATFORMS=cpu \
   "${GRB}/host-venv/bin/python" -u scripts/tools/openpi_vla_lora_finetune.py \
     --base-url "http://localhost:<port>" \
     --lance-dataset <path> \
     --base-model <base_model> \
     --steps <steps> --batch-size <batch_size> \
     --lora-rank <rank> \
     --save-checkpoint-name <name-or-omit> \
     [--skip-inference-check] \
     [--eval-mse --eval-mse-indices <comma-separated-indices>] \
     [--infer-to-lance --infer-to-lance-output <path>] \
     --output-json <path>
   ```

   `--lora-rank` defaults to 16 -- omit it unless the user explicitly asked
   for a different value. If a non-default value is passed and the server
   rejects it, the driver surfaces the server's own error text (do not treat
   this as a driver bug; it means the server-side constraint documented in
   `references/pipeline_reference.md` section 2.1 is still in effect).

   `--eval-mse` and `--infer-to-lance` are independent opt-in steps that only
   run if `--skip-save` is not set (both need a saved checkpoint's
   `model_path`). `--infer-to-lance` runs inference over every frame in the
   dataset (not a sample) -- for a large dataset this can take a while;
   warn the user before running it on more than a few hundred frames.

7. **Report results** to the user: final loss vs first-step loss, the
   `save_weights_for_sampler` checkpoint path (`mint://...` URI and/or
   filesystem path from the response), model_id, and the inference check
   result if run. If `--eval-mse` was run, report `aggregate.overall_mse` and
   `aggregate.mse_vs_baseline_ratio` (< 1 means the model beats an all-zero
   prediction baseline; close to or above 1 on a low-step run is not
   necessarily a problem -- it may just mean too few steps, not the
   action_dim/zero-padding failure mode). If `--infer-to-lance` was run,
   report the output path and `num_frames_written`. If loss trend is flat or
   rising, mention it may indicate an action_dim / zero-padding issue and
   point at `ActionHeadSummary.md`.

8. **Cleanup**: the driver's own `finally` block already calls
   `_delete_model` on exit -- this only removes server-side session state,
   **not** the checkpoint files `save_weights_for_sampler` already wrote to
   disk (those persist independently). If you started a fresh server for
   this run, stop it after reporting results (`kill <server_pid>`) unless the
   user asked to keep it running for further iteration.

## Known failure modes

See `references/troubleshooting.md` for the full list with symptoms and
fixes. Summary:

- **action_dim mismatch** -> hard stop by design, do not pad/mask around it.
- **Unreadable Lance dataset version** -> retry with
  `--lance-dataset-version <N>` (the driver's `probe_lance_dataset` suggests
  a working version number).
- **`create_model` 400 with a LoRA-related message** -> the server is
  currently rejecting a non-default `--lora-rank` or a `--lora-train-*` flag
  set to False (this is a *server-side* constraint the driver only warns
  about, not a local block -- see `RECURSIVE.md`'s improvement-target list
  if this constraint has since been relaxed and the driver's warning text
  needs updating). The error text the driver surfaces is the server's own
  `detail` field, not a guess.
- **`train_step` 503 right after a successful `create_model`** -> almost
  certainly a `model_id` extraction bug (the server appends a suffix to your
  `session_id`); read `references/pipeline_reference.md` section 2.2.
- **OOM partway through training** -> check whether the worker-side prompt
  padding fix is still in place before touching `XLA_FLAGS`/batch size; see
  `references/pipeline_reference.md` section 5.
- **`--eval-mse` aggregate shows `mse_vs_baseline_ratio` >= 1 on a short run**
  -> not necessarily a bug; a handful of steps may not be enough to beat the
  all-zero baseline. Only treat this as a real signal after a run long
  enough to expect convergence, and cross-check against the loss curve.

## Reference

- `RECURSIVE.md` -- continuously-updated inventory of every script and piece
  of logic in this repo related to this skill, plus known-limitation /
  improvement-target tracking (e.g. the LoRA rank server constraint). Update
  this file whenever a new script is added or an existing one's behavior
  changes -- it is the map for anyone extending this skill later.
- `references/pipeline_reference.md` -- full script map, server-side
  constraints, and root-caused historical bugs (read before debugging).
- `references/api_contracts.md` -- exact request/response field shapes for
  `create_model` / `vla/train_step` / `save_weights_for_sampler`.
- `references/troubleshooting.md` -- symptom -> cause -> fix table.
- `ActionHeadSummary.md` (repo root) -- the 10-experiment study establishing
  the action_dim invariant. Do not duplicate its content here; link to it.
