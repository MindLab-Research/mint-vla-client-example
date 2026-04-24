---
name: prod-log-monitor
description: |
  Continuous, read-only monitoring of MinT production logs and basic API responsiveness.

  Watches:
  - API server logs on prod hosts
  - Ray/worker logs via Volcano/DLC log streaming (best-effort)
  - Healthz and status endpoint latency

  On anomaly, collects evidence and files a GitHub issue via the issue-reporter workflow.

  Triggers: "prod log monitor", "production log monitor", "monitor prod logs", "watch prod logs"

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# prod-log-monitor

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

Hard rule: production is READ-ONLY. This skill must not restart servers, kill actors, or mutate state.

## Scope

Monitor production deployments:
- Volcano router/master: `mint-prod` (port 18000, auth required)
- Aliyun upstream (if used): manage logs via `aliyun-cluster` skill (read-only)

## Inputs

Required environment variables (set on your workstation):
- `MINT_BASE_URL` (example: `https://mint.macaron.xin` (China), `https://mint.macaron.im` (international), or `http://localhost:18000` via SSH tunnel)
- `MINT_API_KEY` (do not print)

Optional:
- `MINT_LOG_MONITOR_INTERVAL_S` (default 30)
- `MINT_LOG_MONITOR_TAIL_LINES` (default 200)

## What to check (minimum)

1. HTTP health
- `GET /api/v1/healthz` (latency + payload)
- `GET /api/v1/actors` (read-only visibility; do not use kill endpoints)

2. Server logs (Volcano)
- `ssh mint-prod "tail -${MINT_LOG_MONITOR_TAIL_LINES:-200} /tmp/tinker_server_auth.log"`

3. Error signatures (grep over last tail window)
- `Traceback`
- `ERROR`
- `ActorDiedError`
- `RayTaskError`
- `RayChannelTimeoutError`
- `EngineDeadError`
- `OutOfMemoryError` / `CUDA out of memory`
- `Unknown request_id`

## Procedure

1. Confirm production targeting
- Refuse to run if `MINT_BASE_URL` is unset.
- Refuse to run if the base URL points to dev (`localhost:8000`) unless user explicitly requested dev monitoring.

2. Sample endpoints (read-only)
- Measure `healthz` response time and status code.
- If `actors` endpoint exists and is unauthenticated, call it read-only.

3. Tail logs (read-only)
- Tail `/tmp/tinker_server_auth.log` from `mint-prod`.
- If the deployment uses gateway routing to an Aliyun upstream, use `aliyun-cluster` to fetch worker logs (read-only).

4. Triage any anomaly
Classify into one:
- client misuse (wrong URL/auth; malformed request)
- transient infrastructure (Ray disconnect, worker eviction, DLC pod restart)
- server bug (contract/invariant violation, deterministic repro, stack trace)
- missing feature / limitation (expected gap)

Use `tinker-official-reference` as contract-of-record when deciding if behavior is correct.

5. File an issue when evidence supports it
Invoke `issue-reporter` and include:
- timestamps (UTC and local)
- base URL (redact secrets)
- request IDs (if available)
- minimal reproduction command or script path
- log excerpts (no secrets)

## Non-goals

- No production remediation (no restarts, no actor kills).
- No "fix by switching to dev".
