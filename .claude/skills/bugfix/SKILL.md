---
name: bugfix
description: |
  Issue reproduction and bugfix workflow.

  Use for: reproducing issues, testing fixes, validating bugfixes before merge.

  Triggers: "fix issue", "reproduce bug", "test bugfix", "issue #X"

  **CRITICAL: Treat production as READ-ONLY unless the user explicitly requests production operations.**
  For any production operations (restart/kill actors/logs on prod hosts), invoke `mint-prod`.

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Bugfix Workflow

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

> **ABSOLUTE RULES**
>
> 1. **PRODUCTION IS READ-ONLY BY DEFAULT**: Do not restart/kill/modify production unless the user explicitly requests it. Use `mint-prod` for any production operations.
> 2. **NEVER SUBSTITUTE REQUIREMENTS**: If an issue is complex, solve it. Don't simplify the problem.
> 3. **VERIFY WITH REPRODUCTION SCRIPT**: A fix is not complete until the reproduction script passes.

---

## Phase 1: Understand the Issue

### 1.1 Create Environment-Agnostic Reproduction Script

Write a script that can run against either production or development:

```python
# scripts/tools/reproduce_issue_<NUMBER>.py
import os

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")

# ... reproduction logic using BASE_URL and API_KEY
```

**Script requirements:**
- Uses environment variables for URL and API key
- Clearly prints success/failure
- Defines issue scope (what fails, what works)

### 1.2 Reproduce on Production (Optional, HTTP-only)

If the bug was reported from production, verify it exists:

Do not run any production-side kill/restart commands during reproduction. Use HTTP only.

### 1.3 Check Production Logs (READ-ONLY)

```bash
# SSH to prod server to READ logs only
ssh mint-prod-volcano "tail -100 /tmp/mint_server_auth.log"
ssh mint-prod-volcano "grep -i 'error\|exception' /tmp/mint_server_auth.log | tail -30"
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
# Check dev server is running
curl http://localhost:8000/api/v1/healthz
```

If dev server is not running, use the `mint-dev` skill to start it. Dev code is
updated through the git checkout at `/share/mint/dev/mint-server`, not file sync.

### 2.1.1 Path-Based Checkpoint Repros On Dev

If reproduction requires `model_path` or `state_path` pointing at an absolute checkpoint directory:

- Use an admin API key on the dev server. Absolute paths are rejected for non-admin requests.
- If you need a private dev server, follow the `mint-dev` skill's isolated debug server rules:
  - Python attach uses `ray://<head_ip>:10001`, not raw `:6379`
  - fresh `MINT_RAY_NAMESPACE`
  - fresh `MINT_STARTUP_LEASE_ACTOR_NAME`
  - `MINT_UVICORN_WORKERS=1`
- Do not spend time on model-level debugging until that private server can pass:
  1. `/api/v1/healthz`
  2. `create_sampling_session`
  3. one `/api/v1/asample`

### 2.2 Reproduce on Development

```bash
# Default environment uses dev server
python scripts/tools/reproduce_issue_<NUMBER>.py
```

Or explicitly:
```bash
MINT_BASE_URL=http://localhost:8000 \
MINT_API_KEY=dummy \
python scripts/tools/reproduce_issue_<NUMBER>.py
```

### 2.3 Implement Fix

1. Identify root cause from reproduction
2. Make code changes
3. **Restart dev server** (code changes require server restart):
   ```bash
   # Use mint-dev start/restart commands so shared config and runtime are used.
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
| Reproduce on prod (China) | `MINT_BASE_URL=https://mint.macaron.xin MINT_API_KEY=<admin_key> python scripts/tools/reproduce_issue_X.py` |
| Reproduce on prod (international) | `MINT_BASE_URL=https://mint.macaron.im MINT_API_KEY=<admin_key> python scripts/tools/reproduce_issue_X.py` |
| Reproduce on dev | `python scripts/tools/reproduce_issue_X.py` |
| Prod logs (READ-ONLY) | `ssh mint-prod-volcano "tail -100 /tmp/mint_server_auth.log"` |
| Dev logs | `ssh mint-dev "tail -100 /tmp/mint_server.log"` |
| Health check | `curl http://localhost:8000/api/v1/healthz` |

---

## API Key Behavior

| API Key Type | Error Details | Use Case |
|--------------|---------------|----------|
| Admin key | Full server-side errors exposed | Debugging, reproduction |
| Regular key (`sk-*`) | Generic "Operation failed" message | Production users |

Admin keys are identified by `user_id == "admin"` in the auth system. When reproducing issues, always use an admin key to see the actual error.
