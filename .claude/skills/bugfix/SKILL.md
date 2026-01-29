---
name: bugfix
description: |
  Issue reproduction and bugfix workflow.

  Use for: reproducing issues, testing fixes, validating bugfixes before merge.

  Triggers: "fix issue", "reproduce bug", "test bugfix", "issue #X"

  **CRITICAL: Production server is READ-ONLY. Never restart, modify, or touch prod server.**
---

# Bugfix Workflow

> **ABSOLUTE RULES**
>
> 1. **PRODUCTION IS READ-ONLY**: You may only READ logs. NEVER restart, kill, or modify prod server.
> 2. **NEVER SUBSTITUTE REQUIREMENTS**: If an issue is complex, solve it. Don't simplify the problem.
> 3. **VERIFY WITH REPRODUCTION SCRIPT**: A fix is not complete until the reproduction script passes.

---

## Phase 1: Understand the Issue

### 1.1 Create Environment-Agnostic Reproduction Script

Write a script that can run against either production or development:

```python
# scripts/tools/reproduce_issue_<NUMBER>.py
import os

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# ... reproduction logic using BASE_URL and API_KEY
```

**Script requirements:**
- Uses environment variables for URL and API key
- Clearly prints success/failure
- Defines issue scope (what fails, what works)

### 1.2 Reproduce on Production (Optional)

If the bug was reported from production, verify it exists:

```bash
# Use admin API key to see full server-side errors
TINKER_BASE_URL=https://mint.macaron.im \
TINKER_API_KEY=<admin_key> \
python scripts/tools/reproduce_issue_<NUMBER>.py
```

> **Admin API Key Benefit**: When using the admin API key, server-side error details
> are included in the response. Regular users only see generic error messages.
> This makes reproduction more targeted.

### 1.3 Check Production Logs (READ-ONLY)

```bash
# SSH to prod server to READ logs only
ssh mint-prod-volcano "tail -100 /tmp/tinker_server_auth.log"
ssh mint-prod-volcano "grep -i 'error\|exception' /tmp/tinker_server_auth.log | tail -30"
```

> **WARNING: DO NOT run any commands that modify state on mint-prod.**
> No `pkill`, no `kill`, no starting processes, no modifying files.

### 1.4 Document Issue Scope

Before proceeding, clearly document:
- What specifically fails (error message, behavior)
- What works (related functionality that's fine)
- Root cause hypothesis
- Files likely to need changes

---

## Phase 2: Fix and Test on Development

### 2.1 Ensure Dev Environment is Ready

```bash
# Verify unison is syncing
pgrep -af "unison.*volcano-tinker"

# Check dev server is running
curl http://localhost:8000/api/v1/healthz
```

If dev server is not running, use the `mint-dev` skill to start it.

### 2.2 Reproduce on Development

```bash
# Default environment uses dev server
python scripts/tools/reproduce_issue_<NUMBER>.py
```

Or explicitly:
```bash
TINKER_BASE_URL=http://localhost:8000 \
TINKER_API_KEY=dummy \
python scripts/tools/reproduce_issue_<NUMBER>.py
```

### 2.3 Implement Fix

1. Identify root cause from reproduction
2. Make code changes
3. **Restart dev server** (code changes require server restart):
   ```bash
   ssh mint-dev 'pkill -f "tinker-server.*run_server"'
   # Then start server using mint-dev skill
   ```

### 2.4 Verify Fix

```bash
# Run reproduction script - MUST now pass
python scripts/tools/reproduce_issue_<NUMBER>.py
```

### 2.5 Check for Regressions

Run related tests to ensure fix doesn't break existing functionality.

---

## Phase 3: Validate

### 3.1 Success Criteria

A fix is ONLY complete when:
1. Reproduction script passes (no errors)
2. Original functionality still works (no regressions)
3. Edge cases are handled

### 3.2 Never Substitute Requirements

**FORBIDDEN responses:**
- "The issue is complex, let's take a simpler approach"
- "Instead of fixing X, we can work around it by Y"
- "This would require significant changes, so let's just..."

**REQUIRED approach:**
- Fully understand the issue
- Implement the correct fix
- Test until it works
- If truly blocked, explain the specific technical blocker and ask for guidance

### 3.3 Sync Architecture Docs (If Needed)

After the reproduction script passes on development, check whether the bugfix changes any documented system behavior or operator workflow.

If it does, update the architecture docs accordingly:
- `.claude/skills/architecture-design/SKILL.md`
- Any referenced `.claude/skills/architecture-design/references/*.md` pages that describe the affected component

Examples that usually require doc updates:
- New/changed endpoints, request/response shapes, or auth behavior
- Ray actor lifecycle changes (naming, namespaces, eviction rules, persistence)
- Changes to training/inference session lifecycle or weight transfer semantics

---

## Quick Reference

| Task | Command |
|------|---------|
| Reproduce on prod | `TINKER_BASE_URL=https://mint.macaron.im TINKER_API_KEY=<admin_key> python scripts/tools/reproduce_issue_X.py` |
| Reproduce on dev | `python scripts/tools/reproduce_issue_X.py` |
| Prod logs (READ-ONLY) | `ssh mint-prod-volcano "tail -100 /tmp/tinker_server_auth.log"` |
| Dev logs | `ssh mint-dev "tail -100 /tmp/tinker_server.log"` |
| Health check | `curl http://localhost:8000/api/v1/healthz` |

---

## API Key Behavior

| API Key Type | Error Details | Use Case |
|--------------|---------------|----------|
| Admin key | Full server-side errors exposed | Debugging, reproduction |
| Regular key (`sk-*`) | Generic "Operation failed" message | Production users |

Admin keys are identified by `user_id == "admin"` in the auth system. When reproducing issues, always use an admin key to see the actual error.
