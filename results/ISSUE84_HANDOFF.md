# Issue 84: bounded futures and overload behavior (implementation plus dev validation)

Date: 2026-02-22

This file records what was implemented for GitHub issue 84 and what was observed in a dev validation run.

## Problem recap (pre-fix evidence)

Isolated dev run on `mint-dev` port `18084` (2026-02-22), pushing many async requests without calling `/api/v1/retrieve_future`:
- API server RSS increased from about 1206 MB to about 3911 MB over about 590 s (about 4.6 MB/s).
- FutureStore counters grew large (`pending` and `results`), and API RSS growth was consistent with unbounded retention of pending request bodies and/or completed results.

## What was implemented (repo state in this worktree)

Goals:
- Keep API process heap bounded under large async backlog.
- Store completed results outside the API heap and release them on retrieve and TTL.
- Under resource pressure, reject new async requests with HTTP 429 (not OOM).
- After pressure drops, retries can succeed.

Key components:
- `tinker_server/backend/capacity_manager.py`: detached Ray capacity actor with two reservations:
  - Request queue bytes (serialized request JSON held in the work queue).
  - Result bytes (reserved against Ray object store free bytes from `ray.available_resources()["object_store_memory"]`).
- `tinker_server/backend/api_work_queue.py`: detached Ray work queue storing request JSON bodies and dispatching work from local worker loops (replaces FastAPI BackgroundTasks backlog).
- `tinker_server/backend/result_size_estimator.py`: conservative result size estimator used for object store reservation.
- `tinker_server/backend/future_store.py`: results stored as Ray objects plus metadata (no large DONE payloads in actor heap); explicit tombstones (`EXPIRED`, `RETRIEVED`) and `reap()` for TTL-driven release.
- `tinker_server/routes/futures.py`: retrieve returns explicit tombstones and performs cleanup plus reservation release on terminal outcomes.
- `tinker_server/routes/internal.py`: `GET /internal/admission_stats` for budgets, reservations, rejects, and queue depth.

Bug fixed during dev validation:
- `tinker_server/backend/future_store.py`: `FutureStore.get_result` no longer does a second `ray.get` (the extra `ray.get` caused HTTP 500 on `/api/v1/retrieve_future` on this Ray cluster).

Follow-up hardening (post-review):
- `tinker_server/backend/future_store.py`: adds a QUEUED-only timeout (`future_store_queue_ttl_s`) so futures cannot remain `PENDING` forever if work queue workers stop making progress. Timed out request_ids transition to `FAILED` with error `"queue timeout"` and are reaped by the server reaper loop, releasing reservations.
- `tinker_server/backend/result_size_estimator.py` + `tinker_server/routes/training.py`: `forward` and `forward_backward` now reserve object store bytes based on total target tokens (logprobs payload size), rather than the fixed `estimate_small_result_bytes()` (256 KiB).
- `tinker_server/config_file.py`: config file schema now supports `future_store.queue_ttl_s` and `future_store.tombstone_ttl_s` (previously env-only).

## Memory budgets (dev cluster, observed values)

Results (object store) budget signal on `mint-dev`:
- Ray address: `192.168.47.138:6379`
- Namespace used in validation: `tinker_issue84_e2e_yiwen`
- Observed at `2026-02-22 12:41:11` UTC:
  - `ray.available_resources()["object_store_memory"] = 203311660236` bytes

Requests (queue) budget in the validation run:
- `TINKER_CAPACITY_QUEUE_BYTES_BUDGET=8388608` (8 MiB)

Budgets and current reservations are visible via:
- `GET /internal/admission_stats`

## End-to-end pressure test (dev, isolated)

Isolation:
- Host: `mint-dev`
- Port: `18084`
- Ray namespace: `tinker_issue84_e2e_yiwen`
- Base model kept warm: `Qwen/Qwen3-0.6B`
- Server log: `/tmp/tinker_server_issue84_e2e.log`

Start the isolated server (run on `mint-dev`):
- `cd /vePFS-Mindverse/share/code/yiwen/tinker-server-issue-84-wt`
- `export TINKER_PORT=18084 TINKER_RAY_NAMESPACE=tinker_issue84_e2e_yiwen RAY_ADDRESS=192.168.47.138:6379`
- `export MINT_PERSISTENT_MODELS=Qwen/Qwen3-0.6B TINKER_CAPACITY_QUEUE_BYTES_BUDGET=8388608 TINKER_API_WORK_QUEUE_NUM_WORKERS=1`
- `nohup python scripts/run_server.py >> /tmp/tinker_server_issue84_e2e.log 2>&1 &`

Local tunnel:
- `ssh -f -N -L 18084:localhost:18084 mint-dev`

Pressure client (run locally):
- `TINKER_BASE_URL=http://localhost:18084 ISSUE84_MAX_TOKENS=1 ISSUE84_TOKEN_ID=1000 ISSUE84_CONCURRENCY=128 ISSUE84_MAX_TOTAL=20000 python scripts/wip/issue84_e2e_pressure.py`

Observed behavior (2026-02-22):
- Server stayed up: `GET /api/v1/healthz` continued returning 200.
- Under load, admissions returned HTTP 429 with explicit reason and budgets:
  - `code=tinker_overloaded`
  - `reason=queue_bytes_budget_exceeded`
  - `queue_bytes_budget=8388608`
  - `queue_bytes_reserved=8381736`
- After submissions stopped and the queue reservation fell, retries succeeded:
  - `retry_ok=128 retry_429=0`
- Draining a sample of futures succeeded:
  - `drain_terminal_status_counts={'200': 32}`

Observed behavior (2026-02-22, second run with smaller queue budget to force 429 quickly):
- Server budgets (via `GET /internal/admission_stats`):
  - `queue_bytes_budget=262144`
  - `object_store_free_bytes=203306287074` (bytes)
- Under sustained load, HTTP 429 was observed with explicit reason:
  - `reason=queue_bytes_budget_exceeded`
  - `queue_bytes_reserved` stabilized near `262063` bytes
- After queue relief, retries succeeded:
  - `retry_ok=256 retry_429=0` (in cycles 1, 2, 3)
- Draining futures remained healthy:
  - `drain_terminal_status_counts={'200': 32}` (in cycles 1, 2, 3)

Additional load evidence (2026-02-22):
- Sustained overload for 5 cycles (60 s per cycle) continued to return 429s while staying responsive:
  - Example cycle end snapshot included `queue_bytes_reserved` near budget and `rejects_total` increasing monotonically.
  - Retries continued to succeed after relief in every cycle.
- API RSS stayed stable during a 140 s window while the server was overloaded and returning many 429s:
  - `results/issue84_api_rss5.tsv` shows `rss_kb_min=1306932`, `rss_kb_max=1307460` (`delta_kb=528`).
  - `results/issue84_http_monitor5.tsv` shows `healthz_status=200` for all sampled points during the same window.

## Detached actor hygiene (required for code changes)

Detached Ray actors keep running old code across API server restarts. For this isolated namespace, the critical detached actors are:
- `tinker_future_store`
- `tinker_capacity_manager`
- `tinker_api_work_queue`

Reset SOP for the namespace:
- Stop the isolated server process.
- Kill the detached actors in `tinker_issue84_e2e_yiwen`.
- Restart the isolated server.
