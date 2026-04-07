# Issue432 mixed-load harness

Script: `scripts/tools/bench_issue432_mixed_queue.py`

Purpose:
- keep one local scheduled training domain pending with `training.forward_backward`
- keep the legacy FIFO bucket under sustained pressure with `/internal/work_queue/noop`
- preserve raw queue/debug artifacts first, then derive a summary for Gate 5

## Preconditions
- API server reachable at `MINT_BASE_URL` or `--base-url`
- local server must support:
  - `POST /api/v1/create_model`
  - `POST /api/v1/forward_backward`
  - `POST /internal/work_queue/noop`
  - `GET /internal/admission_stats`
  - `GET /internal/work_queue/debug_state`
  - `GET /internal/metrics`
- training path must have Ray available. Current blocker on the local server used during development:
  - `POST /api/v1/create_model` returns `503`
  - body: `{"detail":"Ray unavailable: CapacityManager requires Ray"}`

## Why `/internal/work_queue/noop` is legacy on current HEAD
Code-path verification on current HEAD:
- `tinker_server/routes/internal.py:516` defines `/work_queue/noop`
- `tinker_server/routes/internal.py:539` marks only `{"op": "internal.noop"}`
- `tinker_server/routes/internal.py:540` to `tinker_server/routes/internal.py:546` enqueues with `extra={"ts": ...}` only
- `tinker_server/backend/api_work_queue.py:512` to `tinker_server/backend/api_work_queue.py:520` accepts scheduled work only if `WorkClassification.from_queue_extra(extra)` returns `queue_kind == "scheduled"` and both domain and session key are present
- `tinker_server/backend/work_classification.py:69` to `tinker_server/backend/work_classification.py:76` makes `queue_kind == "scheduled"` only when scheduler is enabled and `scheduler_domain`, `scheduler_session_key`, and `scheduler_capacity_owner` all exist
- `/internal/work_queue/noop` provides none of those scheduler fields, so it stays legacy-classified

## Environment variables
- `MINT_BASE_URL`: server base URL, default `http://localhost:8000`
- `MINT_API_KEY`: optional auth header for protected servers
- `MINT_ISSUE432_MODEL`: base model for scheduled training domain, default `Qwen/Qwen3-0.6B`
- `MINT_ISSUE432_LORA_RANK`: default `8`
- `MINT_ISSUE432_TRAINING_SESSIONS`: default `2`
- `MINT_ISSUE432_TRAINING_STEPS`: default `6`
- `MINT_ISSUE432_BATCH_SIZE`: default `2`
- `MINT_ISSUE432_SEQ_LEN`: default `256`
- `MINT_ISSUE432_LEGACY_TARGET_OUTSTANDING`: default `128`
- `MINT_ISSUE432_LEGACY_POLL_BATCH`: default `32`
- `MINT_ISSUE432_LEGACY_WARMUP_S`: default `1.0`
- `MINT_ISSUE432_SNAPSHOT_INTERVAL_S`: default `0.25`
- `MINT_ISSUE432_TIMEOUT_S`: default `1800`
- `MINT_ISSUE432_POLL_INTERVAL_S`: default `0.1`
- `MINT_ISSUE432_LABEL`: run label
- `MINT_ISSUE432_OUTPUT_DIR`: summary dir, default `cover/issue432`
- `MINT_ISSUE432_RAW_ROOT`: raw artifact root, default `/tmp/issue432-mixed`

## Example run
```bash
python scripts/tools/bench_issue432_mixed_queue.py \
  --label issue432-local \
  --training-sessions 2 \
  --training-steps 6 \
  --batch-size 2 \
  --seq-len 256 \
  --legacy-target-outstanding 128 \
  --legacy-poll-batch 32 \
  --legacy-warmup-s 1.0 \
  --snapshot-interval-s 0.25 \
  --timeout-s 1800
```

## Expected artifacts
Summary outputs in `cover/issue432/`:
- `<label>-summary.json`
- `<label>-summary.md`

Raw artifacts in `/tmp/issue432-mixed/<label>/`:
- `meta.json`
- `events.jsonl`
- `legacy.jsonl`
- `samples.jsonl`
- `training.results.json`
- `before.admission.json`
- `after.admission.json`
- `before.debug.json`
- `after.debug.json`
- `before.metrics.prom`
- `after.metrics.prom`

## Bring-up checklist for a real Gate 5 run
1. Start a local server where Ray-backed training is available, not only healthz.
2. Verify `POST /api/v1/create_model` works before running the harness.
3. Verify `GET /internal/admission_stats`, `GET /internal/work_queue/debug_state`, and `GET /internal/metrics` all respond.
4. If auth is enabled, export `MINT_API_KEY`.
5. Set scheduler knobs explicitly on the server if the run needs fixed guards, for example:
   - `MINT_SCHEDULER_ENABLE=1`
   - `MINT_LOCAL_DOMAIN_MAX_SERVICE_GAP_S=<bound under test>`
   - `MINT_MAX_LOCAL_LEGACY_STREAK_WITH_DOMAIN_PENDING=<bound under test>`
   - `MINT_API_WORK_QUEUE_DEBUG_MAX=<enough history for the run>`
6. Verify the chosen model supports local training bring-up on that server.
7. Run the harness with a unique `--label`.
8. Inspect `<label>-summary.json` plus raw artifacts before making claims about fairness.

## Mismatch between proxy load and the original issue
- The original incident involved heavy legacy `sampling.asample` traffic.
- This harness uses `internal.noop` as the legacy source to isolate the global arbiter from model-specific executor costs.
- Therefore this harness proves or disproves local cross-bucket arbitration behavior, not full production sampling throughput behavior.
- It is a Gate 5 local deterministic proxy, not final shared-cluster closure.
