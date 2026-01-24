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
  - `mint-dev` skill for dev environment constraints (volcano host; default port 8000)
  - `architecture-design` skill for architecture docs alignment

  Triggers: "auto-bugfix", "bot queue", "assign-to-bot"
---

# auto-bugfix (orchestrator)

This skill runs a loop: drain the `assign-to-bot` issue queue by producing reviewed PRs merged into `develop`.

Responsibility split:
- Orchestrator (this SKILL.md): queue management, branch management, bot-identity commit/push, PR creation, spawning reviewer, merge orchestration.
- Bugfixer subagent: all reproduction, troubleshooting, debugging, code edits, and dev-server operations needed to make reproduction pass.
- Reviewer subagent: independent review and a PR comment; no code changes.

Hard rules:
- Production is read-only. Use `mint-prod` skill if production reads are required.
- Never substitute requirements. If reproduction fails, fix the real failure.
- Restart the issue-scoped dev server after code changes (Python server does not hot-reload).
- Do not stop/replace the default dev server. Auto-bugfix runs on an issue-specific port and issue-specific server root.
- Orchestrator, bugfixer, and reviewer MUST read the entire issue thread (body + all comments) before coding/reviewing.
  If the issue references another issue/PR for context, read that too before acting.
- Static-only reproductions/tests are forbidden. Do not close an issue based on source inspection (grep/AST/string checks).
- Mock-only or stub-only "runtime" is not an acceptable substitute for an integrated repro when an integrated repro is feasible.
  - Examples of partial evidence: calling a FastAPI route handler directly, stubbing `ray`/`fastapi`/`peft` so imports work, asserting only on
    computed resource numbers / GPU counts without starting the affected actor/engine and successfully completing a request through it.
  - For server/runtime bugs: the reproduction must hit the issue-scoped dev server over HTTP and exercise the real dependency stack (FastAPI + Ray).
- "Runtime" means the real runtime. If the production code path uses Ray, a "runtime test" that does not connect to real Ray is not a runtime test.
  - "Runtime without Ray" is a non-test for the server: do not accept it for Ray-backed endpoints, actor lifecycle, scheduling, vLLM, or Megatron paths.
- Assume an integrated repro IS feasible by default. Use the issue-scoped server on volcano + SSH tunnel; do not accept "not run (server not available)" as closure evidence.
- For any change that touches scheduling / placement groups / GPU allocation / engine initialization:
  - A "repro" that only asserts on computed resource numbers / GPU counts is partial evidence.
  - Require an integrated repro that (1) creates the affected session/engine in Ray and (2) completes at least one request using it (e.g. create_sampling_session + asample + retrieve_future).
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
      HTTP flows work).
    - Treat this as the baseline Ray connectivity sanity check (create_session must succeed).
    - If the change touches sampling/inference/vLLM:
      - Also run `python scripts/tools/smoke.py service --create-sampling-session`.
      - Do not accept "create_sampling_session succeeded" as sufficient evidence. Execute at least one end-to-end sample through the running system
        (submit `/api/v1/asample`, then poll `/api/v1/retrieve_future` until ready) and assert on the returned payload (prefer: in `reproduce_issue_<N>.py`).
    - If the change touches training/checkpointing: run a minimal end-to-end call that exercises that path (issue-specific repro is preferred).
  - Unit tests: `pytest -q` run on the PR branch (in addition to integrated checks; unit tests do not replace integrated checks).
  - Evidence recording: record the exact commands AND the observed PASS/FAIL output in the PR (description or comment). "I inspected the code" is invalid evidence.
- If you cannot run the reproduction or tests, do not close the issue and do not merge the PR. Post a blocking note explaining why it cannot be executed.

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

Namespace goal: isolate Ray actor state per issue (avoid collisions across concurrent dev runs).
`mint-dev` documents `TINKER_RAY_NAMESPACE` for this.

Example namespace + issue-specific PFS path:
```bash
export ISSUE=123
export TINKER_RAY_NAMESPACE="tinker_${USER}_issue_${ISSUE}"
export PFS_TINKER_PATH="/vePFS-Mindverse/share/code/$USER/tinker-server-issue-$ISSUE"
export UNISON_PROFILE="volcano-tinker-$USER-issue-$ISSUE"
export TINKER_PORT="$((10000 + ISSUE % 5000))"
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
lines[roots[1]] = f"root = ssh://volcano//vePFS-Mindverse/share/code/{user}/tinker-server-issue-{issue}\n"
out = "".join(lines).replace("__PFS_USER__", user)

dst = Path.home() / ".unison" / f"{unison_profile}.prf"
dst.write_text(out)
print(dst)
PY

test -n "$UNISON_PROFILE" || { echo "error: UNISON_PROFILE is empty"; exit 1; }
pkill -f "[u]nison.*$UNISON_PROFILE" 2>/dev/null || true
nohup unison "$UNISON_PROFILE" -repeat watch > "/tmp/unison-$UNISON_PROFILE.log" 2>&1 &
pgrep -af "unison.*$UNISON_PROFILE"
```

Issue-specific server root on volcano:
```bash
ssh volcano "mkdir -p $PFS_TINKER_PATH && ln -sfn $PFS_TINKER_PATH /root/tinker_project/tinker-server-issue-$ISSUE"
```

Start an issue-scoped dev server (does not touch the default dev server on port 8000):
```bash
ssh volcano "cd /root/tinker_project/tinker-server-issue-$ISSUE && nohup bash -c \
  \"PYTHONPATH=/root/tinker_project/tinker-server-issue-$ISSUE:\\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=$PFS_TINKER_PATH \
   TINKER_RAY_NAMESPACE=$TINKER_RAY_NAMESPACE \
   TINKER_PORT=$TINKER_PORT \
   TINKER_USAGE_LOG_DIR=/tmp/tinker_usage_issue_$ISSUE \
   python scripts/run_server.py\" >> /tmp/tinker_server_issue_$ISSUE.log 2>&1 & echo \$! > /tmp/tinker_server_issue_$ISSUE.pid"
```

Issue-scoped health check (via local SSH tunnel):
```bash
ssh -f -N -L ${TINKER_PORT}:localhost:${TINKER_PORT} volcano
curl http://localhost:$TINKER_PORT/api/v1/healthz
```

Issue-scoped logs:
```bash
ssh volcano "tail -50 /tmp/tinker_server_issue_$ISSUE.log"
```

Issue-scoped stop:
```bash
ssh volcano "test -f /tmp/tinker_server_issue_$ISSUE.pid && xargs -r kill < /tmp/tinker_server_issue_$ISSUE.pid || true"
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

### 3d) Reproduce issue

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

### 3e) Fix issue

Implement the minimal root-cause fix.

After code changes:
1) verify code synced to volcano (unison)
2) restart the issue-scoped dev server (stop/start using the issue-scoped commands in 3b)
3) confirm health endpoint

### 3f) Re-run reproduction script

Re-run the exact reproduction script from 3d.

### 3g) Update architecture docs if outdated

If behavior or operator workflow changed, update:
- `.claude/skills/architecture-design/SKILL.md`
- any affected `.claude/skills/architecture-design/references/*.md`

### 3h) Commit (mindlab-bot identity) and push

```bash
ISSUE=123
TITLE="$(gh issue view $ISSUE --json title -q .title)"

git add -A
git -c user.name='mindlab-bot' -c user.email='contact@mindlab.ltd' \
  commit -m "Fix #$ISSUE: $TITLE"
git push -u origin HEAD
```

### 3i) Create PR (base=`develop`)

Prefer including `Fixes #<NUMBER>` in the PR body so the linkage is explicit.

```bash
ISSUE=123
BRANCH="bot/issue-$ISSUE"
TITLE="$(gh issue view $ISSUE --json title -q .title)"

PR_URL="$(gh pr create --base develop --head "$BRANCH" \
  --title "Fix #$ISSUE: $TITLE" \
  --body "Fixes #$ISSUE")"
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
