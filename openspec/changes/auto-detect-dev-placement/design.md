# Design

## Overview

Keep head/worker discovery at the dev launcher boundary for this change. The
core supervisor and placement controller already own runtime placement
semantics; this change only removes the manual pre-launch step that writes
worker IPs into an env file.

## Launcher Flow

1. Resolve `MINT_CODE_ROOT`, namespace, port, and dev defaults as today.
2. Source optional deployment env.
3. Source `MINT_DEV_RUN_ENV` if provided.
4. If no explicit placement variables are now set, read the current Ray head IP
   from the head-address file.
5. Run `scripts/tools/gen_dev_placement.py --head-ip <head> --models-from-env
   --output <run-local-env>`.
6. Source the generated env and continue with the existing preflight/bootstrap.

The generated env path lives under `MINT_TMP_ROOT` so it is visible and
inspectable in the dev runtime area, not hidden in local `/tmp`.

## Placement Generator

`gen_dev_placement.py` becomes both a manual tool and launcher helper:

- It can still accept repeated `--model` arguments.
- It adds `--models-from-env` to read `MINT_PERSISTENT_MODELS` first and fall
  back to `MINT_SUPPORTED_MODELS`.
- It excludes Ray head nodes and requires alive GPU workers.
- It writes all canonical placement variables:
  `MINT_MODEL_PLACEMENT_JSON`, `MINT_DENSE_MODEL_PLACEMENT_JSON`,
  `MINT_VLLM_MODEL_PLACEMENT_JSON`, and `MINT_MEGATRON_MODEL_PLACEMENT_JSON`.

This remains conservative: it only automates the worker-IP selection that was
previously manual. Explicit env remains authoritative for special cases.

## Error Handling

If the dashboard cannot be reached or no alive GPU workers are found, the
generator exits non-zero with the head IP and dashboard endpoint in the error
message. The launcher then stops before creating server actors.

## Testing

Add unit tests for dashboard parsing, model discovery from env, env file output,
and launcher text guardrails. Run shell syntax checks for the launcher.
