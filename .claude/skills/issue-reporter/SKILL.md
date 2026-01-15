---
name: issue-reporter
description: |
  Translate vague user reports into precise GitHub issues.

  Use for: bug reports, user complaints, screenshots of errors, vague descriptions.

  Triggers: "report issue", "user reported", "create issue", "file bug"

  **CRITICAL: Production is READ-ONLY. Analyze and reproduce, never modify.**
---

# Issue Reporter

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

### Run against production

```bash
TINKER_BASE_URL=https://mint.macaron.im TINKER_API_KEY=<admin_key> python scripts/reproduce_user_report.py
```

Admin key exposes server-side errors. Regular keys show only "Operation failed."

### Check logs (read-only)

```bash
ssh mint-prod "tail -200 /tmp/tinker_server_auth.log"
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

Quote the original user report at the end.

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
