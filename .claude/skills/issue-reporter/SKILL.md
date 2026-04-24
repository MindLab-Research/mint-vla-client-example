---
name: issue-reporter
description: |
  Translate vague user reports into precise GitHub issues.

  Use for: bug reports, user complaints, screenshots of errors, vague descriptions.

  Triggers: "report issue", "user reported", "create issue", "file bug"

  **CRITICAL: Treat production as READ-ONLY unless the user explicitly requests production operations.**
  For any production operations (restart/kill actors/logs on prod hosts), invoke `mint-prod`.

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Issue Reporter

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

## 1. Reproduce

### Gather what the user provides

Screenshots, partial errors, vague descriptions. Extract: what action failed, what error appeared, which model/endpoint if mentioned.

### Write reproduction script

```python
# scripts/reproduce_user_report.py
import os
BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
```

### Run against production (HTTP-only)

Do not run any production-side kill/restart commands while triaging a report.

### Check logs (read-only)

```bash
ssh mint-prod-volcano "tail -200 /tmp/tinker_server_auth.log"
```

## 2. Analyze

Read source code to trace the failure. Reference specific files and line numbers.

**Epistemic discipline:**
- State what you observed: "Log shows X at timestamp Y"
- State what you infer: "This suggests Z, but not confirmed"
- Do not claim "root cause" unless you have definitive evidence
- "Cannot reproduce" is a valid finding - state it clearly

## 3. Check duplicates

```bash
gh issue list --repo MindLab-Research/tinker-server --state open --search "keyword"
gh issue list --repo MindLab-Research/tinker-server --state closed --search "keyword"
```

If duplicate exists, comment on existing issue instead of creating new one.

## 4. Create issue

### Query labels first

```bash
gh label list --repo MindLab-Research/tinker-server
```

Use only labels that exist. Do not fabricate.

### Write issue content

State the problem precisely. Include:
- Exact error message
- Reproduction steps or script
- Code references (file:line)
- What you observed vs what you infer

Omit anything that doesn't help fix the bug. No boilerplate headers. No padding.

### Hard self-containment requirement

The issue body must be understandable to a repo reader with no access to your local session.

Hard rules:
- Do not mention PR numbers, prompt files, rollout ids, local conversations, or "in this PR" / "the user asked" style context.
- Do not refer to local-only paths, local images, scratch notes, or private workstation state.
- Do not refer to untracked `scripts/wip/*` files as if they are durable repo interfaces. If a script is needed as part of the issue context, either promote it into tracked repo state first or rewrite the issue so it stands on code references and observed behavior without depending on that script.
- If you cite artifacts, prefer tracked repo paths or durable result paths that other engineers on the shared environment can inspect. Do not write things like "local curve copy" or "see my local image".
- The reader should be able to understand the bug from the issue body plus the repository itself, without needing your personal environment or prior turns.

### Create

```bash
gh issue create --repo MindLab-Research/tinker-server \
  --title "Concise technical description" \
  --body "..." \
  --label "label1,label2"
```

## Constraints

- Production is read-only
- Do not claim certainty without evidence
- Do not fabricate labels, errors, or code references
- Do not create duplicate issues
- Do not substitute a simpler problem for the reported one
- If unable to reproduce, say so - do not invent explanations
- Issue bodies must be self-contained and must not depend on private local context
