---
name: sanity-check
description: |
  Production sanity-check runner for MinT/mint-server.

  Objective: run non-trivial RL training loops against MinT production for the 4 production base models
  (0.6B, 4B, 30B, 235B), collect timing evidence, perform only minimal ops remediation when justified,
  and send exactly one final Feishu report.

  Triggers: "sanity check", "sanity-check", "prod sanity", "production sanity"

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# sanity-check

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill first. Do not improvise around the gap.

Hard rules:
- Production only. The base URL for this workflow is exactly `https://mint.macaron.xin`.
- Refuse to run if the effective base URL is anything else, including localhost, SSH tunnels, dev ports, `https://mint.macaron.im`, or an internal host/port.
- No requirement substitution: do not use `--inference-only`; run the RL loop with a training client.
- Ops only: no non-trivial code changes in this workflow.
- Minimize downtime: prefer killing one specific actor for one specific model over restarting the API process; do not restart Ray/head/worker nodes as first response.
- Do not use an ad-hoc "hung after N seconds" cutoff. Under load, 235B can queue for many minutes. Treat only the configured timeout or explicit service/server error as failure.
- No START notifications. Send exactly one final Feishu report at the end, even if all models pass.
- Never print secrets, process environments, or full `.secrets.env` contents.

## Inputs

Required working directory:
- `/root/code/mint`.

Required local file:
- `/root/code/mint/.secrets.env`.
- Source it without printing it. It must provide an API key through `MINT_API_KEY` or Tinker SDK compatibility `TINKER_API_KEY`.

Required production owner:
- `MINT_TEST_CHECKPOINT_OWNER_ID` must be set to the production owner ObjectId.
- It must be a 24-character hex ObjectId. Refuse to run if missing or invalid.

Required access:
- `ssh mint-prod-volcano` for router/master logs and production ops.
- `ssh mint-prod-aliyun` only if live routing/config proves the failing model is routed there.

Scripts:
- Canonical wrapper: `scripts/wip/check.sh`
- Wrapper implementation: `scripts/wip/train_check.py`
- Main runner: `.claude/skills/sanity-check/mint_rl_test_long.py`
- Feishu notifier: `.claude/skills/sanity-check/feishu_notify.py`

Artifact root:
- `/root/run_results/mint/<timestamp>/`, where `<timestamp>` is `YYYYMMDD-HHMMSS`.
- `/root/run_results/mint` is expected to point at `/vePFS-Mindverse/user/nolanho/run_results/mint`.

Per-model artifacts:
- Create one directory per model under the run root.
- Capture stdout/stderr separately for each model.
- Set `MINT_TEST_EXPERIMENT_ROOT` to that model directory.
- `mint_rl_test_long.py` creates an additional timestamped subdirectory under `MINT_TEST_EXPERIMENT_ROOT`; timing files are inside that subdirectory. Locate them with `find "$MODEL_DIR" -name timing_summary.json` rather than assuming they sit directly in the model directory.

Timing evidence to preserve:
- `timing_events.jsonl`
- `timing_summary.json`
- `timing_summary.md`

## Procedure

### 0) Preflight

Run from `/root/code/mint`:

```bash
cd /root/code/mint
set -a
. ./.secrets.env
set +a
export MINT_BASE_URL=https://mint.macaron.xin
export TINKER_BASE_URL=https://mint.macaron.xin
```

Validate target and required owner without printing secrets:

```bash
python - <<'PY'
import os, re
base = os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
owner = os.environ.get("MINT_TEST_CHECKPOINT_OWNER_ID", "")
if base != "https://mint.macaron.xin":
    raise SystemExit(f"refusing non-production sanity base URL: {base!r}")
if not re.fullmatch(r"[0-9a-fA-F]{24}", owner):
    raise SystemExit("MINT_TEST_CHECKPOINT_OWNER_ID must be a 24-character production owner ObjectId")
if not (os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY")):
    raise SystemExit("missing production API key")
print("sanity preflight ok: production URL and owner id are set; API key is present (redacted)")
PY
```

Create the run directory:

```bash
TS=$(date +%Y%m%d-%H%M%S)
RUN_ROOT=/root/run_results/mint/$TS
mkdir -p "$RUN_ROOT"
printf '%s\n' "$TS" > "$RUN_ROOT/timestamp.txt"
```

Optional read-only probes are allowed but are not the sanity check:

```bash
curl -sS "$MINT_BASE_URL/api/v1/healthz"
curl -sS -H "X-API-Key: $MINT_API_KEY" "$MINT_BASE_URL/api/v1/actors"
```

Do not use the optional probe result as PASS evidence.

### 1) Run the four RL loops

Preferred command:

```bash
./scripts/wip/check.sh --all-models --timeout-s=7200
```

The wrapper is part of this skill contract. It must:
- refuse any base URL other than `https://mint.macaron.xin`,
- load `.secrets.env` without printing secrets,
- require `MINT_TEST_CHECKPOINT_OWNER_ID`,
- run `--all-models` sequentially in the required order,
- write artifacts under `/root/run_results/mint/<timestamp>/`,
- discover nested timing files recursively,
- write `summary.json`, `summary.md`, and `final_feishu_report.md`,
- send exactly one final Feishu report for `--all-models`.

Use the manual pattern below only if the wrapper itself is broken.

Run exactly these four models in order:

1. `Qwen/Qwen3-0.6B`
2. `Qwen/Qwen3-4B-Instruct-2507`
3. `Qwen/Qwen3-30B-A3B-Instruct-2507`
4. `Qwen/Qwen3-235B-A22B-Instruct-2507`

Use this pattern for each model. Keep each model as a separate process and preserve its exit code.

```bash
MODEL='Qwen/Qwen3-0.6B'
SLUG=$(MODEL="$MODEL" python - <<'PY'
import os, re
print(re.sub(r'[^A-Za-z0-9_.-]+', '_', os.environ['MODEL']).strip('_'))
PY
)
MODEL_DIR="$RUN_ROOT/$SLUG"
mkdir -p "$MODEL_DIR"
START_EPOCH=$(date +%s)
(
  set -a
  . ./.secrets.env
  set +a
  export MINT_BASE_URL=https://mint.macaron.xin
  export TINKER_BASE_URL=https://mint.macaron.xin
  export MINT_TEST_EXPERIMENT_ROOT="$MODEL_DIR"
  export MINT_TEST_CHECKPOINT_OWNER_ID="$MINT_TEST_CHECKPOINT_OWNER_ID"
  python .claude/skills/sanity-check/mint_rl_test_long.py \
    --model "$MODEL" \
    --num-rl-steps=1 \
    --batch-size=2 \
    --group-size=4 \
    --max-tokens=128
) >"$MODEL_DIR/stdout.log" 2>"$MODEL_DIR/stderr.log"
STATUS=$?
END_EPOCH=$(date +%s)
MODEL="$MODEL" STATUS="$STATUS" WALL="$((END_EPOCH-START_EPOCH))" python - <<'PY' > "$MODEL_DIR/result.json"
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "exit_code": int(os.environ["STATUS"]),
    "wall_clock_s": int(os.environ["WALL"]),
}))
PY
find "$MODEL_DIR" -name timing_summary.json -o -name timing_summary.md -o -name timing_events.jsonl | sort > "$MODEL_DIR/artifacts.txt"
```

Important runner constraints:
- Do not pass `--inference-only`.
- Do not omit `MINT_TEST_CHECKPOINT_OWNER_ID`.
- Do not lower `MINT_TEST_TIMEOUT_S` just to make the run finish faster.
- For 235B, wait for the configured timeout or an explicit error. Pending sampling alone is not failure.

### 2) Summarize timing after each model

For each model, parse the newest `timing_summary.json` under that model directory and write a compact summary that can be used in the final Feishu report:

```bash
python - "$MODEL_DIR" <<'PY'
import json, pathlib, sys
model_dir = pathlib.Path(sys.argv[1])
result_path = model_dir / "result.json"
result = json.loads(result_path.read_text()) if result_path.exists() else {"exit_code": None, "wall_clock_s": None}
summaries = sorted(model_dir.glob("**/timing_summary.json"), key=lambda p: p.stat().st_mtime)
summary = json.loads(summaries[-1].read_text()) if summaries else {"stages": [], "wall_clock_s": result.get("wall_clock_s")}
stages = summary.get("stages") or []
slowest = max(stages, key=lambda s: float(s.get("max_s", 0.0)), default={"stage": "none", "max_s": 0.0})
out = {
    "model": result.get("model") or summary.get("base_model"),
    "ok": result.get("exit_code") == 0,
    "exit_code": result.get("exit_code"),
    "slowest_stage": slowest.get("stage"),
    "max_s": slowest.get("max_s"),
    "wall_clock_s": summary.get("wall_clock_s", result.get("wall_clock_s")),
}
(model_dir / "report_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, sort_keys=True))
PY
```

### 3) Failure handling: evidence first

If any model process exits non-zero:

1. Preserve local artifacts already written under `/root/run_results/mint/<timestamp>/`.
2. Record the failing model, exit code, failing stage if timing captured one, request IDs visible in stdout, and the slowest completed stage.
3. Check the production router/master logs first using the current production log path:

```bash
ssh mint-prod-volcano 'tail -400 /vePFS-Mindverse/share/mint/prod/logs/mint_server_auth.log'
```

4. Do not assume `235B -> Aliyun`. Before checking upstream logs, prove routing from current evidence:
   - live capabilities / actor inventory,
   - production topology/config in `/vePFS-Mindverse/share/mint/prod/config/`,
   - gateway config if present,
   - or logs that clearly show upstream forwarding.
5. If evidence proves a remote upstream is used, inspect only that target's logs via its own skill/SOP.

Failure classes:
- `client env/auth`: base URL, API key, platform auth, or owner id issue.
- `capacity/scheduling`: placement pending, actor not registered, queue not consumed, GPU/PG held by stale actor.
- `server exception`: traceback, 5xx, explicit request failure, actor crash.
- `timing degradation`: no hard failure, but one stage is materially slower than expected.

For 235B pending:
- Do not kill vLLM or Megatron solely because elapsed time is large.
- Capture request IDs from stdout and wait for the configured timeout or an explicit server error.

### 4) Minimal remediation, only when justified

Use the smallest blast radius.

Use model-specific actor kill only when evidence shows a specific actor is stale or crashed:

```bash
curl -sS -X POST \
  -H "X-API-Key: $MINT_API_KEY" \
  -H "Content-Type: application/json" \
  "$MINT_BASE_URL/internal/actors/kill" \
  -d '{"actor_type":"vllm","model_name":"Qwen/Qwen3-4B-Instruct-2507"}'
```

Valid actor types for this workflow: `vllm`, `megatron`, `dense`, `all`.

Rules:
- Prefer `vllm` for sampling/runtime crashes and `megatron` for training/runtime crashes.
- Do not kill `all` unless the evidence shows systemic actor corruption and a narrower kill cannot work.
- Restart the API server only when the server process itself is unhealthy. Detached actors survive API restart, so API restart is not actor remediation.
- Do not restart Ray head/worker nodes in this workflow.
- If a placement group leak is suspected after actor kill, follow `volcano-cluster` cleanup SOP; do not guess commands.

Write every remediation to:
- `$RUN_ROOT/INCIDENT.md`

Include timestamp, model, symptom, request IDs, evidence, exact ops action, and observed result. Do not include secrets.

### 5) Re-test after remediation

After any remediation, rerun all four models from the beginning in the same required order. Create a new attempt subdirectory under the same timestamp root, for example:

```bash
RUN_ROOT=/root/run_results/mint/$TS/attempt-2
mkdir -p "$RUN_ROOT"
```

If any failure persists after one justified remediation pass, stop doing ops and switch to evidence + issue filing.

### 6) GitHub issue when evidence supports an implementation issue

Invoke `issue-reporter` and include:
- Model name.
- Whether the model was local or remotely routed, with concrete target only if proven.
- Failing operation/stage and request IDs.
- Timing summary: slowest stage, `max_s`, `wall_clock_s`.
- Minimal relevant server log excerpt with no secrets.
- Remediation attempted and result.

Do not file an issue for a local client environment mistake that was fixed and rerun cleanly unless it exposed a product defect.

## Final Feishu report

Send exactly one final Feishu report at the end for both PASS and FAIL workflows.

The report must include one line per model with:
- `OK` or `FAIL`
- slowest stage
- `max_s`
- `wall_clock_s`

For failed models, also include:
- failure surface/class,
- failing stage or slowest completed stage if the failure happened after the last recorded stage,
- ops attempted,
- whether a GitHub issue was created.

Do not include:
- local file paths,
- internal base URLs/ports,
- full command transcripts,
- secrets,
- raw log dumps.

Required timing shape:

```markdown
- Qwen/Qwen3-0.6B: OK. Timing: slowest stage=`sample` max_s=`38.4` wall_clock_s=`91.2`.
- Qwen/Qwen3-235B-A22B-Instruct-2507: FAIL in `create_model`. Timing: slowest completed stage=`save_weights_for_sampler` max_s=`412.7` wall_clock_s=`645.9`. Ops: killed model-specific vLLM actor; rerun still failed. Issue: #123.
```

If the report omits timing lines, the Feishu step is not complete. Rewrite the report before sending.

Send via:

```bash
python .claude/skills/sanity-check/feishu_notify.py \
  --title "MinT sanity-check report" \
  --markdown "<agent-written report markdown>"
```

`FEISHU_WEBHOOK_URL` may override the default webhook. If posting fails, this workflow is failed; do not silently ignore it.
