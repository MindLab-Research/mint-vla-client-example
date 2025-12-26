---
name: bugfix
description: |
  Issue reproduction and bugfix workflow.

  Use for: reproducing issues, testing fixes, validating bugfixes before merge.

  Triggers: "fix issue", "reproduce bug", "test bugfix", "issue #X"

  **CRITICAL: Production server is READ-ONLY. Never restart, modify, or touch prod server.**
  **CRITICAL: Dev server symlink is READ-ONLY. Never modify shared dev configuration.**
---

# Bugfix Workflow

> **ABSOLUTE RULES - VIOLATION IS UNACCEPTABLE**
>
> 1. **PRODUCTION IS READ-ONLY**: You may only READ logs. NEVER restart, kill, or modify prod server.
> 2. **DEV SYMLINK IS READ-ONLY**: NEVER touch `/root/tinker_project/tinker-server` symlink on volcano.
> 3. **NEVER SUBSTITUTE REQUIREMENTS**: If an issue is complex, solve it. Don't simplify the problem.
> 4. **VERIFY WITH REPRODUCTION SCRIPT**: A fix is not complete until the reproduction script passes.

---

## Phase 1: Reproduce Issue on Production

### 1.1 Create Reproduction Script

Write a script that reliably reproduces the issue using the production API.

```bash
# Location: scripts/reproduce_issue_<NUMBER>.py
# Example for issue #7:
scripts/reproduce_issue_7.py
```

**Script requirements:**
- Uses production API: `https://mint-alpha.macaron.im`
- Uses production API key (from issue instructions)
- Clearly prints success/failure
- Defines issue scope (what fails, what works)

### 1.2 Run Reproduction Script Locally

```bash
# Run from LOCAL machine (has internet for tokenizers)
python scripts/reproduce_issue_<NUMBER>.py
```

### 1.3 Check Production Logs (READ-ONLY)

```bash
# SSH to prod server to READ logs only
ssh mint-prod "tail -100 /tmp/tinker_server_auth.log"
ssh mint-prod "grep -i 'error\|exception' /tmp/tinker_server_auth.log | tail -30"
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

## Phase 2: Set Up Bugfix Environment

### 2.1 Directory Structure

Local and remote directories MUST have `bugfix` suffix:

| Location | Path |
|----------|------|
| Local | `/home/yiwen/tinker_project/tinker-server-bugfix` |
| Remote (PFS) | `/vePFS-Mindverse/share/code/tinker-server-bugfix` |
| Remote symlink | `/root/tinker_project/tinker-server-bugfix` (NEW, separate from dev) |

### 2.2 Create Remote Directory and Symlink

```bash
# Create PFS directory (if not exists)
ssh volcano "mkdir -p /vePFS-Mindverse/share/code/tinker-server-bugfix"

# Create SEPARATE symlink for bugfix (NOT the shared dev symlink)
ssh volcano "ln -sf /vePFS-Mindverse/share/code/tinker-server-bugfix /root/tinker_project/tinker-server-bugfix"

# Verify
ssh volcano "ls -la /root/tinker_project/ | grep tinker"
```

**Expected output shows TWO symlinks:**
```
tinker-server -> /vePFS-Mindverse/share/code/tinker-server          # DEV - DO NOT TOUCH
tinker-server-bugfix -> /vePFS-Mindverse/share/code/tinker-server-bugfix  # BUGFIX - yours
```

### 2.3 Set Up SSH Tunnel

```bash
# Bugfix server uses port 8001 (different from dev port 8000)
ssh -f -N -L 8001:localhost:8001 volcano

# Verify tunnel
lsof -i :8001
```

### 2.4 Set Up Unison Profile

Create `~/.unison/volcano-tinker-bugfix.prf`:

```
root = /home/yiwen/tinker_project/tinker-server-bugfix
root = ssh://volcano//vePFS-Mindverse/share/code/tinker-server-bugfix

auto = true
batch = true
prefer = newer
times = true

ignore = Name {*.pyc}
ignore = Name {__pycache__}
ignore = Name {*.swp}
ignore = Name {*.swo}
ignore = Name {.DS_Store}
ignore = Name {.venv}
ignore = Name {.mypy_cache}
ignore = Name {.pytest_cache}
ignore = Name {*.egg-info}

servercmd = /usr/local/bin/unison
```

Start unison daemon:
```bash
# Check if already running
pgrep -af "unison.*volcano-tinker-bugfix"

# Start daemon
nohup unison volcano-tinker-bugfix -repeat watch > /tmp/unison-bugfix.log 2>&1 &

# Verify sync
tail -20 /tmp/unison-bugfix.log
```

---

## Phase 3: Test on Dev Cluster

### 3.1 Resource Sharing Notice

The bugfix server shares the mint-dev Ray cluster with feature development due to resource constraints.

**Rules:**
- Bugfix uses port **8001**, dev uses port **8000**
- Bugfix log: `/tmp/tinker_server_bugfix.log`
- Dev log: `/tmp/tinker_server.log`
- Code paths are separate (different symlinks)
- **Do NOT run both servers concurrently** - if dev server is running, notify user

### 3.2 Check for Concurrent Server

Before starting bugfix server:

```bash
# Check if dev server is running
ssh volcano "ps aux | grep 'run_server' | grep -v grep"

# If output shows a running server, STOP and notify user:
# "Dev server is currently running on port 8000. Cannot start bugfix server concurrently.
#  Please coordinate with dev team or wait for dev server to stop."
```

### 3.3 Start Bugfix Server

```bash
ssh volcano 'cd /root/tinker_project/tinker-server-bugfix && nohup bash -c \
  "PYTHONPATH=/root/tinker_project/tinker-server-bugfix:\$PYTHONPATH \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   TINKER_PORT=8001 \
   python scripts/run_server.py" > /tmp/tinker_server_bugfix.log 2>&1 &'
```

### 3.4 Health Check

```bash
# Via tunnel (port 8001)
curl http://localhost:8001/api/v1/healthz
```

### 3.5 Run Reproduction Script Against Bugfix Server

Modify reproduction script to use bugfix server:

```python
# Change from:
API_URL = "https://mint-alpha.macaron.im"
# To:
API_URL = "http://localhost:8001"
API_KEY = "dummy"  # Auth disabled on dev cluster
```

Or use environment variables:
```bash
TINKER_BASE_URL=http://localhost:8001 TINKER_API_KEY=dummy python scripts/reproduce_issue_<NUMBER>.py
```

### 3.6 Check Bugfix Server Logs

```bash
ssh volcano "tail -50 /tmp/tinker_server_bugfix.log"
ssh volcano "grep -i 'error\|exception' /tmp/tinker_server_bugfix.log | tail -20"
```

### 3.7 Stop Bugfix Server

```bash
ssh volcano 'pkill -f "tinker-server-bugfix.*run_server"'
```

---

## Phase 4: Validate Fix

### 4.1 Success Criteria

A fix is ONLY complete when:
1. Reproduction script passes (no errors)
2. Original functionality still works (no regressions)
3. Edge cases are handled

### 4.2 Never Substitute Requirements

**FORBIDDEN responses:**
- "The issue is complex, let's take a simpler approach"
- "Instead of fixing X, we can work around it by Y"
- "This would require significant changes, so let's just..."

**REQUIRED approach:**
- Fully understand the issue
- Implement the correct fix
- Test until it works
- If truly blocked, explain the specific technical blocker and ask for guidance

---

## Quick Reference

| Task | Command |
|------|---------|
| SSH tunnel (bugfix) | `ssh -f -N -L 8001:localhost:8001 volcano` |
| Start unison | `unison volcano-tinker-bugfix -repeat watch` |
| Check dev server | `ssh volcano "ps aux \| grep run_server \| grep -v grep"` |
| Start bugfix server | See section 3.3 |
| Health check | `curl http://localhost:8001/api/v1/healthz` |
| Bugfix logs | `ssh volcano "tail -50 /tmp/tinker_server_bugfix.log"` |
| Stop bugfix server | `ssh volcano 'pkill -f "tinker-server-bugfix.*run_server"'` |
| Prod logs (READ-ONLY) | `ssh mint-prod "tail -100 /tmp/tinker_server_auth.log"` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8001 in use | Check for stale bugfix server, kill it |
| Unison not syncing | Check `tail /tmp/unison-bugfix.log` |
| Dev server running | Notify user, wait for coordination |
| Fix doesn't work | Re-check reproduction, don't substitute requirements |
| Prod logs stale | Verify correct log file path on mint-prod |
