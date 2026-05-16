---
name: auto-bugfix
description: |
  Orchestrate a nonstop loop fixing GitHub issues labeled `assign-to-bot`.

  SOP:
  1) List issues labeled `assign-to-bot`
  2) Sort by priority
  3) For each issue: branch, isolate dev namespace, reproduce, fix, re-run reproduction,
     update architecture docs if needed, commit/push as mindlab-bot, open PR to `develop`,
     spawn an independent reviewer subagent, then merge to `develop` if review passes.

  Refers to:
  - `bugfix` skill for reproduction discipline
  - `mint-dev` skill for dev environment constraints (SSH host `mint-dev`; default port 8000)
  - `architecture-design` skill for architecture docs alignment

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.

  Triggers: "auto-bugfix", "bot queue", "assign-to-bot"
---

# auto-bugfix (orchestrator)

This skill runs a loop: drain the `assign-to-bot` issue queue by producing reviewed PRs merged into `develop`.

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

Responsibility split:
- Orchestrator (this SKILL.md): queue management, branch management, bot-identity commit/push, PR creation, spawning reviewer, merge orchestration.
- Bugfixer subagent: all reproduction, troubleshooting, debugging, code edits, and dev-server operations needed to make reproduction pass.
- Reviewer subagent: independent review and a PR comment; no code changes.

GitHub accountability requirement:
- The PR is the accountability surface for auto-bugfix. The orchestrator MUST explicitly post GitHub comments/recommendations before merge.
- Minimum required GitHub artifacts on every auto-bugfix PR:
  - A PR comment from the orchestrator summarizing the problem, root cause, fix, exact runtime evidence, remaining risks, and merge recommendation.
  - A reviewer PR comment/review that states findings or explicitly says no findings and gives a merge recommendation.
- Do not treat the PR description alone as sufficient. The recommendation must be posted as an explicit GitHub comment/review on the PR before merge.

SOP step ownership (remove ambiguity):
- Orchestrator runs: 3a-3c, 3h-3k.
- Bugfixer runs: 3d-3g.
- Reviewer runs: 3j (and blocks merge if evidence or explanation is missing).

Hard rules:
- Production is read-only. Use the `mint-prod` skill (ssh host `mint-prod-volcano`) if production reads are required.
- Never substitute requirements. If reproduction fails, fix the real failure.
- Restart the issue-scoped dev server after code changes (Python server does not hot-reload).
- Do not stop/replace the default dev server. Auto-bugfix runs on an issue-specific port and issue-specific server root.
- Never create/get/kill Ray actors outside the active `TINKER_RAY_NAMESPACE` unless the user explicitly requests cross-namespace action.
- Do not modify git config (no `git config ...`). Commit identity must be set per command.
- Orchestrator, bugfixer, and reviewer MUST read the entire issue thread (body + all comments) before coding/reviewing.
  If the issue references another issue/PR for context, read that too before acting.
- Static-only reproductions/tests are forbidden. Do not close an issue based on source inspection (grep/AST/string checks).
- Mock-only or stub-only "runtime" is not an acceptable substitute for an integrated repro when an integrated repro is feasible.
  - Examples of partial evidence: calling a FastAPI route handler directly, stubbing `ray`/`fastapi`/`peft` so imports work, asserting only on
    computed resource numbers / GPU counts without starting the affected actor/engine and successfully completing a request through it.
  - For server/runtime bugs: the reproduction must hit the issue-scoped dev server over HTTP and exercise the real dependency stack (FastAPI + Ray).
- "Runtime" means the real runtime. If the production code path uses Ray, a "runtime test" that does not connect to real Ray is not a runtime test.
  - "Runtime without Ray" is a non-test for the server: do not accept it for Ray-backed endpoints, actor lifecycle, scheduling, vLLM, or Megatron paths.
- Reproduction scripts must be discriminative:
  - A repro that can PASS for unrelated reasons is invalid (example: returning PASS on any exception).
  - Encode acceptance criteria as assertions about the expected behavior, not a generic gate like "200 vs error" or "did not crash".
    - Assert response structure and at least one semantic invariant (required keys, list lengths, monotonic counters, content checks, etc).
    - Treat unexpected errors as FAIL and surface the error text in the repro output.
  - Reviewer must do an adversarial read of `reproduce_issue_<N>.py`:
    - For every PASS branch in the script, state what property it implies and why it matches the acceptance criteria.
    - Try to construct a false positive. If you can, recommendation must be iterate.
- Assume an integrated repro IS feasible by default. Use the issue-scoped server on `mint-dev` + SSH tunnel; do not accept "not run (server not available)" as closure evidence.
- For any change that touches scheduling / placement groups / GPU allocation / engine initialization:
  - A "repro" that only asserts on computed resource numbers / GPU counts is partial evidence.
  - Require an integrated repro that (1) creates the affected session/engine in Ray and (2) completes at least one request using it (e.g. create_sampling_session + asample + retrieve_future).
  - If the issue is scale-dependent (example: fails only at 16 GPUs / TP=16), the integrated repro must run at that target scale. Smaller-scale "it works at N<target" is partial evidence and is not merge/close evidence.
- Treat issue-thread comprehension as testable: the bugfixer + reviewer issue digests must include explicit acceptance criteria extracted from the issue/comments (not just a generic summary).
- Every issue MUST have runtime evidence:
  - Integrated reproduction:
    - Required: execute `scripts/tools/reproduce_issue_<N>.py` against the issue-scoped dev server over HTTP.
    - Required: FAIL on old code and PASS on new code, using the same command line.
    - The repro must exercise the production path of the bug (not just a helper function). If the bug is observable via an HTTP endpoint, the repro must call that endpoint.
    - If the production path uses Ray, the repro must trigger real Ray execution and must not bypass/stub Ray.
      - For actor lifecycle / resource scheduling / placement-group bugs: the repro must both (1) create the affected session/engine in Ray and (2) complete at least one request that uses it (e.g. create_sampling_session + asample + retrieve_future).
    - Only exception: if the issue is truly local-only (no server/runtime surface), run an executed repro or unit test locally and explicitly explain why an integrated repro cannot exercise it. Mock-only/stub-only tests are still forbidden under this exception.
    - Temporary lack of cluster resources (e.g. "no 16-GPU slot available right now") is not a justification to merge/close with a partial test. Treat it as blocking and wait/coordinate until the integrated repro can run.
  - Integrated smoke:
    - Always run `python scripts/tools/smoke.py service` against the issue-scoped dev server after the fix (proves the server still boots and basic
      HTTP flows work). This does not validate Ray.
    - If the issue or change touches Ray-backed paths (sampling/inference, training, scheduling, actor lifecycle), require at least one integrated check
      that triggers real Ray execution:
      - Sampling/inference: run `python scripts/tools/smoke.py service --create-sampling-session` (engine/actor init in Ray).
      - If the change touches inference/vLLM, do not accept "create_sampling_session succeeded" as sufficient evidence. Execute at least one end-to-end
        sample through the running system (submit `/api/v1/asample`, then poll `/api/v1/retrieve_future` until ready) and assert on the returned payload
        (prefer: in `reproduce_issue_<N>.py`).
      - Actor lifecycle / scheduling / placement-group changes: the issue-specific repro must also complete at least one request that uses the created
        session/engine (e.g. create_sampling_session + asample + retrieve_future).
      - Training/checkpointing: the issue-specific repro MUST exercise that path end-to-end.
  - Unit tests: `pytest -q` run on the PR branch (in addition to integrated checks; unit tests do not replace integrated checks).
  - Evidence recording: record the exact commands AND the observed PASS/FAIL output in the PR (description or comment). "I inspected the code" is invalid evidence.
- If you cannot run the reproduction or tests, do not close the issue and do not merge the PR. Post a blocking note explaining why it cannot be executed.
- Before merge, explicitly post the final recommendation to the PR:
  - `recommend merge` if the issue-specific evidence is complete and any remaining failures are proven pre-existing or unrelated.
  - `do not merge` if issue-specific evidence is incomplete or a remaining failure may be introduced by the branch.
- Retroactive enforcement: if a closed `assign-to-bot` issue was closed based on static-only or partial evidence (no integrated runtime repro), reopen it and requeue it for a real integrated repro + smoke + pytest.

Files:
- Bugfixer subagent prompt: `.claude/skills/auto-bugfix/prompts/bugfixer.md`
- Reviewer subagent prompt: `.claude/skills/auto-bugfix/prompts/reviewer.md`

---

## 1) List issues tagged `assign-to-bot`

Requires `gh auth status` to show an active session.

Queue (raw):
```bash
gh issue list --label assign-to-bot --state open --limit 200
```

Queue (machine readable):
```bash
gh issue list --label assign-to-bot --state open --limit 200 \
  --json number,title,url,labels,updatedAt
```

---

## 2) Sort by priority

This repo has priority labels:
- `prio:P0` (highest)
- `prio:P1`
- `prio:P2`

Sort key:
1) priority rank (P0, P1, P2, none)
2) `updatedAt` descending (recent activity first)
3) issue number ascending

```bash
gh issue list --label assign-to-bot --state open --limit 200 \
  --json number,title,url,labels,updatedAt \
| jq -r '
    def prio_rank:
      ([.labels[].name] as $ls
       | if ($ls|index("prio:P0")) then 0
         elif ($ls|index("prio:P1")) then 1
         elif ($ls|index("prio:P2")) then 2
         else 3 end);
    sort_by([prio_rank, -(.updatedAt|fromdateiso8601), .number])
    | .[]
    | "\(.number)\t\(.title)\t\(.url)\t\([.labels[].name] | join(\",\"))"
  '
```

---

## 3) Per-issue SOP (repeat until queue empty)

### 3a) Create branch based on latest `origin/develop`

Branch naming (stable per issue): `bot/issue-<NUMBER>`.

```bash
ISSUE=123
BRANCH="bot/issue-$ISSUE"

test -z "$(git status --porcelain)" || { echo "error: working tree is dirty"; exit 1; }

git fetch origin

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git switch "$BRANCH" || git switch -c "$BRANCH" --track "origin/$BRANCH"
  git pull --ff-only origin "$BRANCH" || { echo "error: local branch diverged from origin/$BRANCH"; exit 1; }
  git merge --no-edit origin/develop
else
  git switch -c "$BRANCH" origin/develop
fi
```

### 3b) Set up dev env using issue-specific namespace and PFS path

Use `mint-dev` for shared dev constraints and definitions. Do not reuse its start/stop/log commands because this skill must not stop or replace the default dev server.

Namespace goal: isolate Ray actor state per issue while requiring post-issue cleanup so detached actors do not proliferate.
`mint-dev` documents `TINKER_RAY_NAMESPACE` for this.
Hard rules:
- Do not kill or manipulate actors in other namespaces to "free GPUs" unless the user explicitly requests it.
- Use issue-specific `TINKER_RAY_NAMESPACE` (do not reuse across issues).
- Kill all actors in the issue namespace at the end of the issue (and before starting the issue-scoped server if rerunning) so detached actors do not accumulate.

Example issue-specific namespace + issue-specific PFS path:
```bash
export ISSUE=123
export TINKER_RAY_NAMESPACE="tinker_${USER}_issue_${ISSUE}"
export PFS_TINKER_PATH="/vePFS-Mindverse/share/code/$USER/tinker-server-issue-$ISSUE"
export UNISON_PROFILE="volcano-tinker-$USER-issue-$ISSUE"
export TINKER_PORT="$((10000 + ISSUE % 5000))"
```

Namespace cleanup (run before starting the issue-scoped server, and after finishing the issue):
```bash
# Pass `TINKER_RAY_NAMESPACE` explicitly (ssh does not forward local env by default).
ssh mint-dev "TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' python3 -c \"
import os
import ray
ray.init(address=\"auto\", ignore_reinit_error=True)
ns = os.environ[\"TINKER_RAY_NAMESPACE\"]
actors = ray.util.list_named_actors(all_namespaces=True)
killed = 0
for a in actors:
    if a.get(\"namespace\") != ns:
        continue
    try:
        ray.kill(ray.get_actor(a[\"name\"], namespace=ns))
        killed += 1
    except Exception as e:
        print(f\"kill_failed name={a.get('name')!r} namespace={ns!r} err={e!r}\")
print(f\"killed={killed} namespace={ns}\")
\""
```

Issue-specific code sync (do not manually sync; use unison daemon mode):
```bash
export LOCAL_ROOT="$(git rev-parse --show-toplevel)"

mkdir -p ~/.unison
python - <<'PY'
import os
from pathlib import Path

issue = os.environ["ISSUE"]
user = os.environ["USER"]
local_root = os.environ["LOCAL_ROOT"]
unison_profile = os.environ["UNISON_PROFILE"]

template = Path(".claude/skills/mint-dev/configs/volcano-tinker.prf").read_text()
lines = template.splitlines(True)
roots = [i for i, ln in enumerate(lines) if ln.startswith("root = ")]
if len(roots) != 2:
    raise SystemExit(f"expected 2 root lines, got {len(roots)}")

lines[roots[0]] = f"root = {local_root}\n"
lines[roots[1]] = f"root = ssh://mint-dev//vePFS-Mindverse/share/code/{user}/tinker-server-issue-{issue}\n"
out = "".join(lines).replace("__PFS_USER__", user)

dst = Path.home() / ".unison" / f"{unison_profile}.prf"
dst.write_text(out)
print(dst)
PY

test -n "$UNISON_PROFILE" || { echo "error: UNISON_PROFILE is empty"; exit 1; }
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/unison@.service <<'EOF'
[Unit]
Description=Unison (%i) watch

[Service]
Type=simple
ExecStart=/usr/bin/unison %i -repeat watch -ui text
Restart=always
RestartSec=2
StandardOutput=append:/tmp/unison-%i.log
StandardError=append:/tmp/unison-%i.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
loginctl enable-linger "$USER" || true
systemctl --user enable --now "unison@$UNISON_PROFILE.service"
systemctl --user status "unison@$UNISON_PROFILE.service" --no-pager
tail -n 200 "/tmp/unison-$UNISON_PROFILE.log"
```

Issue-specific server root on mint-dev:
```bash
ssh mint-dev "mkdir -p $PFS_TINKER_PATH && ln -sfn $PFS_TINKER_PATH /root/tinker_project/tinker-server-issue-$ISSUE"
```

Start an issue-scoped dev server (does not touch the default dev server on port 8000):
```bash
ssh mint-dev "cd /root/tinker_project/tinker-server-issue-$ISSUE && nohup bash -c \
  \"PYTHONPATH=/root/tinker_project/tinker-server-issue-$ISSUE:\\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=$PFS_TINKER_PATH \
   TINKER_RAY_NAMESPACE=$TINKER_RAY_NAMESPACE \
   MINT_RAY_NAMESPACE=$TINKER_RAY_NAMESPACE \
   TINKER_PORT=$TINKER_PORT \
   TINKER_USAGE_LOG_DIR=/tmp/tinker_usage_issue_$ISSUE \
   python scripts/run_server.py\" >> /tmp/tinker_server_issue_$ISSUE.log 2>&1 & echo \$! > /tmp/tinker_server_issue_$ISSUE.pid"
```

Issue-scoped health check (via local SSH tunnel):
```bash
ssh -f -N -L ${TINKER_PORT}:localhost:${TINKER_PORT} mint-dev
curl http://localhost:$TINKER_PORT/api/v1/healthz
```

Issue-scoped logs:
```bash
ssh mint-dev "tail -50 /tmp/tinker_server_issue_$ISSUE.log"
```

Issue-scoped stop:
```bash
ssh mint-dev "test -f /tmp/tinker_server_issue_$ISSUE.pid && xargs -r kill < /tmp/tinker_server_issue_$ISSUE.pid || true"
```

### 3c) Delegate the fix to a bugfixer subagent

Spawn a subagent using `.claude/skills/auto-bugfix/prompts/bugfixer.md`.

Inputs to bugfixer:
- issue number and URL
- branch name
- chosen `TINKER_RAY_NAMESPACE`
- chosen `PFS_TINKER_PATH`

Bugfixer deliverable back to orchestrator:
- a short issue digest (3-8 bullets) summarizing the problem + constraints, citing specific issue comment(s) (URL or quoted detail). If the issue has 2+ comments, cite at least 2 comments.
- a reproduction script at `scripts/tools/reproduce_issue_<NUMBER>.py`
- evidence that reproduction fails before the fix and passes after the fix (integrated dev server; no stubs)
- the exact reproduction command used (env vars + invocation)
- the exact dev-server start/stop/log commands used (if relevant to troubleshooting)
- any required updates to `.claude/skills/architecture-design/**`

### 3d) Bugfixer: Reproduce issue

Use the `bugfix` workflow:
- Read the entire issue thread (body + all comments) before writing the repro.
- Create an environment-agnostic reproduction script: `scripts/tools/reproduce_issue_<NUMBER>.py` that exercises the system (no source inspection, no stubs).
  - If the bug is observable via an HTTP endpoint, the repro must call that endpoint against the running issue-scoped server (not internal helpers).
  - If the production path uses Ray, the repro must run against a server connected to Ray and must not bypass/stub Ray.
- Run it against the issue-scoped dev server (`TINKER_BASE_URL=http://localhost:$TINKER_PORT`, `TINKER_API_KEY=dummy`) and capture output.
- Require: FAIL on old code, PASS on new code (same command line).
- After the fix, run a baseline integrated smoke check:
  - `TINKER_BASE_URL=http://localhost:$TINKER_PORT TINKER_API_KEY=dummy python scripts/tools/smoke.py service`
- Post the reproduction command(s) and observed output in the PR (so review does not rely on trust).

### 3e) Bugfixer: Fix issue

Implement the minimal root-cause fix.

After code changes:
1) verify code synced to mint-dev (unison)
2) if the change touches code that can be imported/executed inside detached actors, run the 3b "Namespace cleanup" snippet to kill the issue namespace actors so the next run loads new code:
   - vLLM: `tinker_server/backend/verl_inference.py`, `tinker_server/backend/multi_lora_engine.py`, `tinker_server/backend/multinode_inference.py`, `tinker_server/backend/vllm_*.py`
   - Megatron: `tinker_server/backend/megatron_distributed.py`, `tinker_server/backend/megatron_training.py`, `tinker_server/backend/verl_patches.py`
   - Dense training pool: `tinker_server/backend/verl_training.py`
   - Detached stores/schedulers: `tinker_server/backend/task_state_store.py`, `tinker_server/backend/model_work_scheduler.py`, `tinker_server/backend/model_runtime_actor.py`, `tinker_server/backend/training_session_store.py`, `tinker_server/backend/gateway_session_store.py`
   - Shared (kills required for all GPU actor types): `tinker_server/config.py`, `tinker_server/ray_utils.py`, `tinker_server/backend/ray_kill.py`, `tinker_server/backend/model_registry.py`
   - If uncertain: run namespace cleanup (issue namespace makes this safe).
3) restart the issue-scoped dev server (stop/start using the issue-scoped commands in 3b)
4) confirm health endpoint

### 3f) Bugfixer: Re-run reproduction script

Re-run the exact reproduction script from 3d.

### 3g) Bugfixer: Update architecture docs if outdated

If behavior or operator workflow changed, update:
- `.claude/skills/architecture-design/SKILL.md`
- any affected `.claude/skills/architecture-design/references/*.md`

### 3h) Orchestrator: Commit (mindlab-bot identity) and push

```bash
ISSUE=123
TITLE="$(gh issue view $ISSUE --json title -q .title)"

git add -A
GIT_AUTHOR_NAME='mindlab-bot' GIT_AUTHOR_EMAIL='contact@mindlab.ltd' \
GIT_COMMITTER_NAME='mindlab-bot' GIT_COMMITTER_EMAIL='contact@mindlab.ltd' \
  git commit -m "Fix #$ISSUE: $TITLE"
git push -u origin HEAD
```

### 3i) Orchestrator: Create PR (base=`develop`)

Prefer including `Fixes #<NUMBER>` in the PR body so the linkage is explicit.

PR body requirements (for accountability and reviewability):
- Must explain the fix at idea level: what failed, why it failed, what changed conceptually.
- Must not be only "Fixes #123" or "fixed #123".
- Must not be a diff walkthrough ("changed A, changed B, changed C").
- Use code snippets sparingly (only for a central invariant); keep snippets short.
- Must include executable evidence: the exact repro command(s) and observed FAIL before / PASS after, plus smoke and `pytest -q`.

Recommended PR body outline (Markdown):
- Problem (1 paragraph)
- Root cause (1 paragraph)
- Fix (1-2 paragraphs, mention invariants and failure modes handled)
- Evidence (bullets: commands and PASS/FAIL lines)
- `Fixes #<NUMBER>` (last line)

```bash
ISSUE=123
BRANCH="bot/issue-$ISSUE"
TITLE="$(gh issue view $ISSUE --json title -q .title)"

BODY="$(mktemp /tmp/pr-body.issue-$ISSUE.XXXXXX.md)"
cat > "$BODY" <<'MD'
Problem:
<1 paragraph>

Root cause:
<1 paragraph>

Fix:
<1-2 paragraphs>

Evidence:
- <command> -> <FAIL/PASS line(s)>

Fixes #ISSUE_NUMBER
MD

sed -i "s/#ISSUE_NUMBER/#$ISSUE/g" "$BODY"
PR_URL="$(gh pr create --base develop --head "$BRANCH" --title "Fix #$ISSUE: $TITLE" --body-file "$BODY")"
echo "$PR_URL"
```

If a PR already exists for the branch:
```bash
PR_URL="$(gh pr list --state open --head "$BRANCH" --json url -q '.[0].url')"
echo "$PR_URL"
```

### 3j) Spawn reviewer subagent (independent, critical)

Spawn a subagent using `.claude/skills/auto-bugfix/prompts/reviewer.md`.
Input to reviewer:
- PR URL
- issue URL
- reproduction script path
- any known edge cases

Review output contract:
- a detailed review report posted as a PR comment (via `gh pr comment`)
- a short issue digest (3-8 bullets) citing specific issue comment(s) (URL or quoted detail). If the issue has 2+ comments, cite at least 2 comments.
- a merge recommendation (`recommendation: merge` or `recommendation: iterate`)
- if iterate: blocking issues (with file paths and line numbers)
- commands actually run (repro + tests) and observed results; static-only repro is a blocking issue
- treat missing runtime execution (cannot run repro/tests) as blocking: recommendation must be iterate

### 3k) Merge into `develop` if review passed (iterate otherwise)

If reviewer recommendation is `iterate`:
- fix the blocking items on the same branch
- re-run reproduction script
- push
- re-run reviewer subagent

If reviewer recommendation is `merge`:
```bash
gh pr merge "$PR_URL" --merge --delete-branch --auto \
  || gh pr merge "$PR_URL" --merge --delete-branch
```

Then re-list the issue queue and continue until empty.
