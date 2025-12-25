# tinker-server

## Skills Reference

| Skill | Purpose |
|-------|---------|
| `mint-dev` | Development server on `volcano` (port 8000, no auth) |
| `mint-prod` | Production server on `mint-prod` (port 18000, auth required) |
| `volcano-cluster` | Ray cluster lifecycle (create/teardown worker tasks) |
| `merge-gate` | Pre-merge testing |

**Use `mint-prod` for production operations. Use `mint-dev` for development.**

## Code Synchronization

Code sync handled by background `unison` process. **NEVER** manually sync.

## Remote Commands

**NEVER** run `ray` or `volc` commands locally. Invoke the appropriate skill instead.

## Quick Start (Development)

```bash
# Requires SSH tunnel: ssh -f -N -L 8000:localhost:8000 volcano
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/test_client.py
```

For production (port 18000), use the `mint-prod` skill.

## Tinker API Reference

When needing details about the official Tinker API (types, methods, loss functions, data formats), read `tinker_official_reference.txt` in the project root.

## Architecture

```
                        Development                          Production
                        ───────────                          ──────────
Local Machine           volcano:8000                         mint-prod:18000
─────────────           ────────────                         ──────────────
tinker-cookbook ──HTTP──> tinker-server ──Ray──> GPU Workers (MegatronWorker, vLLM)
                    ↑                                  ↑
              SSH tunnel                         SSH tunnel
         localhost:8000 → volcano:8000      localhost:18000 → mint-prod:18000
```

| Environment | SSH Host | Port | Auth | Log File |
|-------------|----------|------|------|----------|
| Development | `volcano` | 8000 | No | `/tmp/tinker_server.log` |
| Production | `mint-prod` | 18000 | Yes (`X-API-Key`) | `/tmp/tinker_server_auth.log` |

**Finding the running server process:**
```bash
ssh <host> 'ps aux | grep run_server | grep -v grep'
ssh <host> 'ls -la /proc/<PID>/fd/1'  # Shows actual log file location
```

**Key points:**
- **tinker-cookbook**: Runs LOCALLY on your workstation. Requires Python 3.11+ (for `chz` package).
- **tinker-server**: Runs on API server (volcano for dev, mint-prod for prod). Receives HTTP requests, dispatches to Ray workers.
- **GPU Workers**: Run on Ray cluster nodes. Execute training (Megatron) and inference (vLLM).
- **Data transfer**: Weights via Ray object store, not file paths.

### Inference Modes

- **Named Sessions:** `create_sampling_session(model_path=...)` - dedicated engine (~60s init)
- **Ephemeral Sessions:** `save_weights_and_get_sampling_client()` - hot-reload (~0.7s after first)

### vLLM Actor

Detached Ray actor surviving server restarts. First start ~80s, subsequent ~2s.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/healthz` | GET | Health check |
| `/api/v1/vllm_status` | GET | vLLM actor status |
| `/api/v1/kill_vllm` | POST | Kill vLLM actor |
| `/api/v1/create_session` | POST | Create session |
| `/api/v1/create_sampling_session` | POST | Create sampling session |
| `/api/v1/asample` | POST | Submit async sample |
| `/api/v1/retrieve_future` | POST | Poll result (408=pending, 200=ready) |

## Cookbook Tests (Development)

tinker-cookbook runs on your LOCAL machine against the **development** server. Ensure SSH tunnel to volcano is active.

```bash
# Run from LOCAL machine (requires Python 3.11+, chz package)
cd /home/yiwen/tinker_project/tinker-cookbook

# SSH tunnel (if not already running)
ssh -f -N -L 8000:localhost:8000 volcano

# Arithmetic RL (~5 min)
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
python -m tinker_cookbook.recipes.math_rl.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    renderer_name="qwen3_instruct" \
    group_size=4 groups_per_batch=100 learning_rate=1e-4
```

Note: `renderer_name="qwen3_instruct"` bypasses model lookup.

For production testing, use port 18000 and set valid `TINKER_API_KEY`.

---

## Hall of Shame

Agent mistakes that wasted time. Read this before starting any task.

| Date | Mistake | What Should Have Been Done |
|------|---------|---------------------------|
| 2024-12-18 | Ran merge gate tests when user explicitly asked for K2 work | Focus on the requested task. Do not divert to "system health checks" unless specifically asked. |
| 2024-12-18 | Repeatedly concluded "K2 is blocked" without checking reference implementation in k2_workspace | Always check existing working implementations before declaring something impossible. |
| 2024-12-18 | Assumed K2 inference needs 1TB VRAM based on web search instead of checking actual configs | Trust local reference code over generic web articles. |
| 2024-12-23 | Spent 15+ turns fumbling with SSH, log locations, process killing instead of following mint-dev skill | Read the skill docs first. SSH via `volcano` alias, logs at `/tmp/tinker_server.log`, restart commands are documented. Don't waste context on solved problems. |
| 2024-12-23 | VANDALISM: Copied files over unstaged work in develop worktree, then ran `git checkout --` destroying 4 hours of session state management code permanently. Two catastrophic errors in sequence: (1) overwrote files without backup, (2) destroyed recovery chance by reverting to HEAD. | NEVER touch another worktree's files without explicit backup. When user catches a mistake, STOP IMMEDIATELY - do not attempt to "fix" with more commands. Unstaged work is unrecoverable after `git checkout --`. |
| 2024-12-25 | Added "fallback" to `export_weights()` when `export_hf_weights()` unavailable, even though `export_weights()` merges LoRA into base weights - completely unusable for the requirement (separate LoRA matrices for vLLM). Fallback would silently produce wrong output. | NEVER fall back to an API that cannot satisfy the requirement. If the correct API is unavailable, FAIL LOUDLY with a clear error. A fallback that produces garbage is worse than no fallback. |
