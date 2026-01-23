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
  - `mint-dev` skill for dev server operations (volcano:8000)
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
- Restart dev server after code changes (Python server does not hot-reload). See `mint-dev`.

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

git fetch origin

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git switch "$BRANCH" || git switch -c "$BRANCH" --track "origin/$BRANCH"
  git merge --no-edit origin/develop
else
  git switch -c "$BRANCH" origin/develop
fi
```

### 3b) Set up dev env using issue-specific namespace and PFS path

Use `mint-dev` for the exact SSH/log/start/stop commands.

Namespace goal: isolate Ray actor state per issue (avoid collisions across concurrent dev runs).
`mint-dev` documents `TINKER_RAY_NAMESPACE` for this.

Example namespace + issue-specific PFS path:
```bash
ISSUE=123
export TINKER_RAY_NAMESPACE="tinker_${USER}_issue_${ISSUE}"
export PFS_TINKER_PATH="/vePFS-Mindverse/share/code/$USER/tinker-server-issue-$ISSUE"
UNISON_PROFILE="volcano-tinker-$USER-issue-$ISSUE"
```

Issue-specific code sync (do not manually sync; use unison daemon mode):
```bash
LOCAL_ROOT="$(git rev-parse --show-toplevel)"

mkdir -p ~/.unison
sed \
  -e "s|root = /home/yiwen/tinker_project/tinker-server|root = $LOCAL_ROOT|" \
  -e "s|/vePFS-Mindverse/share/code/__PFS_USER__/tinker-server|/vePFS-Mindverse/share/code/__PFS_USER__/tinker-server-issue-$ISSUE|" \
  -e "s/__PFS_USER__/$USER/g" \
  .claude/skills/mint-dev/configs/volcano-tinker.prf \
  > ~/.unison/$UNISON_PROFILE.prf

nohup unison "$UNISON_PROFILE" -repeat watch > "/tmp/unison-$UNISON_PROFILE.log" 2>&1 &
pgrep -af "unison.*$UNISON_PROFILE"
```

Issue-specific server root on volcano (avoid touching `/root/tinker_project/tinker-server`):
```bash
ssh volcano "mkdir -p $PFS_TINKER_PATH && ln -sfn $PFS_TINKER_PATH /root/tinker_project/tinker-server-issue-$ISSUE"
```

Start dev server with both `TINKER_RAY_NAMESPACE` and issue-specific `PFS_TINKER_PATH`:
```bash
ssh volcano "cd /root/tinker_project/tinker-server-issue-$ISSUE && nohup bash -c \
  \"PYTHONPATH=/root/tinker_project/tinker-server-issue-$ISSUE:\\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=$PFS_TINKER_PATH \
   TINKER_RAY_NAMESPACE=$TINKER_RAY_NAMESPACE \
   python scripts/run_server.py\" > /tmp/tinker_server_issue_$ISSUE.log 2>&1 &"
```

Issue-specific logs:
```bash
ssh volcano "tail -50 /tmp/tinker_server_issue_$ISSUE.log"
```

### 3c) Delegate the fix to a bugfixer subagent

Spawn a subagent using `.claude/skills/auto-bugfix/prompts/bugfixer.md`.

Inputs to bugfixer:
- issue number and URL
- branch name
- chosen `TINKER_RAY_NAMESPACE`
- chosen `PFS_TINKER_PATH`

Bugfixer deliverable back to orchestrator:
- a reproduction script at `scripts/tools/reproduce_issue_<NUMBER>.py`
- evidence that reproduction fails before the fix and passes after the fix (dev)
- the exact reproduction command used (env vars + invocation)
- the exact dev-server start/stop/log commands used (if relevant to troubleshooting)
- any required updates to `.claude/skills/architecture-design/**`

### 3d) Reproduce issue

Use the `bugfix` workflow:
- Create an environment-agnostic reproduction script: `scripts/tools/reproduce_issue_<NUMBER>.py`
- Run it against dev (`TINKER_BASE_URL=http://localhost:8000`, `TINKER_API_KEY=dummy`)

### 3e) Fix issue

Implement the minimal root-cause fix.

After code changes:
1) verify code synced to volcano (unison)
2) restart dev server (see `mint-dev`)
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
- a merge recommendation (`recommendation: merge` or `recommendation: iterate`)
- if iterate: blocking issues (with file paths and line numbers)

### 3k) Merge into `develop` if review passed (iterate otherwise)

If reviewer recommendation is `iterate`:
- fix the blocking items on the same branch
- re-run reproduction script
- push
- re-run reviewer subagent

If reviewer recommendation is `merge`:
```bash
gh pr merge "$PR_URL" --merge --delete-branch --auto
```

Then re-list the issue queue and continue until empty.
