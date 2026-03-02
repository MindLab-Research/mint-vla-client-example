---
name: sanity-check
description: |
  Production sanity-check runner for MinT/tinker-server.

  Objective: run non-trivial RL training loops (not just healthz) against production for 4 base models
  (0.6B, 4B, 30B, 235B), collect evidence on failure, do minimal ops remediation when justified,
  and report a final incident report to a Feishu bot when issues are encountered.

  Triggers: "sanity check", "sanity-check", "prod sanity", "production sanity"
---

# sanity-check

Hard rules:
- Production only. Refuse to run if base URL points to dev (port 8000).
- No requirement substitution: do not use inference-only mode; run the RL loop.
- Ops only: no non-trivial code changes in this workflow.
- Minimize downtime: prefer killing a single actor over restarting the whole server.
- Never swallow an incident: always leave artifacts under `results/` and, when appropriate, file a GitHub issue.
- No START notifications. Send exactly one final Feishu report at the end, even if all results are OK.
- Use the official production base URL `https://mint.macaron.im` (do not use SSH tunnels / `localhost`).
- Do not use an ad-hoc "hung after 240s" cutoff. Under load, 235B sampling can queue for many minutes. Treat only the configured per-request timeout (`--timeout-s` / `MINT_TEST_TIMEOUT_S`) or explicit server-side errors as failure signals.

## Inputs

Required local file (repo root):
- `.secrets.env` (do not print); must provide `TINKER_BASE_URL` and `TINKER_API_KEY` (or `MINT_BASE_URL`/`MINT_API_KEY`).

Required access:
- `ssh mint-prod-volcano` (Volcano router/master)
- `ssh mint-prod-aliyun` (Aliyun upstream)

## What this skill runs

Main test script (vendored copy):
- `.claude/skills/sanity-check/mint_rl_test_long.py`

Feishu notifier:
- `.claude/skills/sanity-check/feishu_notify.py`

Artifacts (gitignored):
- `results/sanity-check/<timestamp>/`

## Procedure

### 0) Preflight (must)

1) Load credentials (local):
```bash
set -a && source .secrets.env && set +a
```

2) Confirm production targeting:
- Set `TINKER_BASE_URL=https://mint.macaron.im`.
- Refuse to run if `TINKER_BASE_URL` is anything else (including `localhost:*`).

3) (Optional) Quick read-only probes (not sufficient alone):
```bash
curl -sS "$TINKER_BASE_URL/api/v1/healthz"
curl -sS -H "X-API-Key: $TINKER_API_KEY" "$TINKER_BASE_URL/api/v1/actors"
```

### 1) Run the 4 RL tests (must)

Run sequentially (default models: 0.6B, 4B, 30B, 235B). Do this as an agent-run workflow,
capturing stdout/stderr per model into `results/sanity-check/<timestamp>/`.

Example per-model command:
```bash
python .claude/skills/sanity-check/mint_rl_test_long.py \
  --model <MODEL_NAME> \
  --num-rl-steps=1 \
  --batch-size=2 \
  --group-size=4 \
  --max-tokens=128
```

Do not add `--inference-only`.

### 2) If any test fails: evidence first, then remediation

1) Check prod logs:
- Volcano (0.6B, 4B, 30B):
  - `ssh mint-prod-volcano "tail -400 /tmp/tinker_server_auth.log"`
- Aliyun (235B):
  - `ssh mint-prod-aliyun "tail -400 /tmp/tinker_server_auth.log"`

2) If 235B is "slow but pending" (no exception, just long `retrieve_future` / pending sampling):
- Do not interrupt or kill vLLM solely due to elapsed time.
- Capture request_id(s) from local debug output, and rely on the configured per-request timeout for termination.

3) Determine failure class:
- Client-side (bad base URL/auth) -> fix env and rerun; still report as incident.
- Capacity / scheduling (placement group pending, GPUs held by detached actors) -> kill the smallest relevant actor(s).
- Server crash / exception -> collect traceback + request_id and file an issue (see below).

After the workflow stabilizes (either fixed+rerun passes, or persistent failure), send a final Feishu report.

### 3) Minimal ops remediation (allowed, when justified)

Prefer the smallest blast radius:

1) Kill a specific actor type for the specific model (admin only when auth enabled):
```bash
curl -sS -X POST \
  -H "X-API-Key: $TINKER_API_KEY" \
  -H "Content-Type: application/json" \
  "$TINKER_BASE_URL/api/v1/actors/kill" \
  -d '{"actor_type":"vllm","model_name":"Qwen/Qwen3-4B-Instruct-2507"}'
```

Valid `actor_type`: `vllm`, `megatron`, `dense`, `all`.

Notes:
- Detached actors persist across API server restarts; killing the API server alone may not free GPUs.
- Megatron and multi-node vLLM use Ray placement groups; if you kill a Megatron actor and GPUs stay reserved,
  treat it as a placement-group leak and follow the `volcano-cluster` cleanup SOP (do not guess).

2) Restart API server only if the server process itself is unhealthy:
- Volcano:
  - `ssh mint-prod-volcano 'supervisorctl restart tinker-server-auth'`
- Aliyun:
  - Follow `.claude/skills/aliyun-cluster/SKILL.md` "Start tinker-server on mint-prod-aliyun".
  - Do not restart the local Ray head while actors exist.

3) Log every remediation:
- Write a short incident note into `results/sanity-check/<timestamp>/INCIDENT.md` including:
  timestamp, model, failure symptom, request_id if available, what ops action was taken, and what was observed after.

### 4) Re-test (must)

After any remediation (actor kill, server restart, redeploy), rerun ALL four tests:
```bash
python .claude/skills/sanity-check/mint_rl_test_long.py --model Qwen/Qwen3-0.6B --num-rl-steps=1 --batch-size=2 --group-size=4 --max-tokens=128
python .claude/skills/sanity-check/mint_rl_test_long.py --model Qwen/Qwen3-4B-Instruct-2507 --num-rl-steps=1 --batch-size=2 --group-size=4 --max-tokens=128
python .claude/skills/sanity-check/mint_rl_test_long.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 --num-rl-steps=1 --batch-size=2 --group-size=4 --max-tokens=128
python .claude/skills/sanity-check/mint_rl_test_long.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 --num-rl-steps=1 --batch-size=2 --group-size=4 --max-tokens=128
```

If a failure persists after remediation, stop doing ops and switch to evidence + issue filing.

### 5) If it looks like an implementation issue: file an issue

Invoke `issue-reporter` and include:
- Exact command that failed (and the captured local stdout/stderr path)
- Model name and whether it was Volcano or Aliyun routed
- Timestamps
- Request IDs (from the test output, if present)
- Minimal relevant server log excerpt (no secrets)

## Feishu reporting

Do not send START notifications. Send exactly one final report at the end (both PASS and incident runs).

Required report style (message + evidence):
- Must include: model outcomes (OK/FAIL), the failure surface (what component/operation failed), what ops was attempted and whether it changed anything, and the GitHub issue link if filed.
- Must not include: local file paths, internal timestamps, base URLs/ports, log line numbers, or command transcripts.

Send via:
```bash
python .claude/skills/sanity-check/feishu_notify.py \
  --title "MinT sanity-check report" \
  --markdown "<agent-written report markdown>"
```

Override via env var:
- `FEISHU_WEBHOOK_URL` (default is the requested hook, hardcoded in `.claude/skills/sanity-check/feishu_notify.py`)

If Feishu posting fails, treat it as a failure of this workflow (do not ignore it).
